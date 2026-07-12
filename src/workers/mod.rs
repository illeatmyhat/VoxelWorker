//! The generic background worker behind the async display pipelines.
//!
//! Three heavy rebuilds — the wholesale cuboid-mesh build (geometry), the diameter /
//! widest-run measurement, and the wholesale brick-pipeline build — each run the SAME
//! machine: a dedicated thread that blocks on a request channel, drains a burst of
//! queued requests down to the newest, builds one result, and ships it back. Doing any
//! of these inline on the event-loop thread freezes the UI, so each moves onto a
//! [`Worker`]: the shell keeps drawing the CURRENT (stale) artifact until the freshly
//! built one arrives (stale-while-rebuilding), then swaps it in.
//!
//! This module owns that shared plumbing ONCE; each domain module supplies only its
//! request/result types and a build closure. The supersede/interlock contracts that
//! decide WHICH rebuild is dispatched, and whether an arriving result is accepted, live
//! with the domains (their `route_*` decisions + [`crate::display::routing::GenerationTracker`]).
//!
//! The domain workers themselves live in the submodules: [`brick`], [`diameter`],
//! [`export`], [`geometry`], and the one-shot streaming [`scan`]. The [`export`] worker
//! diverges from the supersede contract below — a `.vox` is a user-chosen file, so the
//! shell serialises exports rather than draining to the latest (see its module doc).
//!
//! ## Supersede / generation (drain-to-latest)
//! A build carries whatever supersede key its domain chose (a monotonic generation). If
//! the user edits again mid-build, the shell dispatches a newer request; the worker
//! **drains its queue to the latest** — it never backlogs, building only the newest
//! pending request — and the shell **discards any received result whose key a later
//! dispatch superseded**. This module implements the drain half ([`drain_to_latest`]);
//! the accept/discard half is the domain's tracker.

use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;

pub mod brick;
pub mod diameter;
pub mod export;
pub mod geometry;
pub mod scan;

/// A background worker running one domain's rebuild on a dedicated thread. The shell
/// [`dispatch`](Self::dispatch)es `Request`s and polls [`try_recv_result`](Self::try_recv_result)
/// each frame; the build closure — captured at [`spawn`](Self::spawn) — turns each request
/// into a full `Response`.
///
/// The type parameters are named to the ROLE, not the payload: `Request` is what crosses
/// TO the worker, `Response` what crosses back (deliberately not `Result`, which would
/// shadow [`std::result::Result`]). The closure returns the WHOLE `Response`, so any
/// panic containment a domain needs (mapping a panicked build to a `None`-tagged result
/// rather than a thread exit) lives INSIDE that closure — this generic makes no such
/// policy, so a worker whose build genuinely cannot panic (the diameter measure) carries
/// none.
pub struct Worker<Request: Send + 'static, Response: Send + 'static> {
    request_sender: Sender<Request>,
    result_receiver: Receiver<Response>,
    /// Kept so the worker thread's lifetime is tied to the handle; the channel close on
    /// drop signals the loop to exit (its `recv` errors and it returns).
    _worker: JoinHandle<()>,
}

impl<Request: Send + 'static, Response: Send + 'static> Worker<Request, Response> {
    /// Spawn the worker on a dedicated thread named `thread_name`, running `build` for
    /// each request.
    ///
    /// The loop is the shared contract: block on the request channel, **drain to the
    /// latest** pending request (so a burst of edits collapses to one build of the newest
    /// — never a backlog), build it, and send the result back. It exits when the request
    /// channel closes (the shell, hence this `Worker`, dropped) or when the shell has
    /// dropped the result receiver (its `send` errors). `build` takes the request by value
    /// and returns the full `Response`, so a domain that must survive a build panic wraps
    /// its own body in a catch (see [`build_catching`]).
    pub fn spawn(
        thread_name: &str,
        mut build: impl FnMut(Request) -> Response + Send + 'static,
    ) -> Self {
        let (request_sender, request_receiver) = std::sync::mpsc::channel::<Request>();
        let (result_sender, result_receiver) = std::sync::mpsc::channel::<Response>();
        let worker = std::thread::Builder::new()
            .name(thread_name.to_string())
            .spawn(move || {
                // Block for the next request; when it closes (shell dropped) the loop ends.
                while let Ok(first) = request_receiver.recv() {
                    // Drain-to-latest: if the user edited again while we were idle-waiting,
                    // collapse the queued requests to the NEWEST so we never build a
                    // superseded generation.
                    let request = drain_to_latest(first, &request_receiver);
                    if result_sender.send(build(request)).is_err() {
                        // The shell is gone; stop.
                        return;
                    }
                }
            })
            .unwrap_or_else(|_| panic!("failed to spawn {thread_name}"));
        Self {
            request_sender,
            result_receiver,
            _worker: worker,
        }
    }

    /// Dispatch a request to the worker (non-blocking). The worker drains to the latest,
    /// so sending a newer request while one is in flight supersedes it. A send error
    /// (worker gone) is ignored — the shell keeps its stale artifact, never blocks.
    pub fn dispatch(&self, request: Request) {
        let _ = self.request_sender.send(request);
    }

    /// Poll the result channel WITHOUT blocking: return the latest finished result if one
    /// has arrived, else `None`. Called each frame in the event loop. Drains to the newest
    /// available (an in-flight supersede can leave more than one queued) so the shell never
    /// integrates a stale build when a newer one is ready.
    pub fn try_recv_result(&self) -> Option<Response> {
        let mut latest = None;
        while let Ok(result) = self.result_receiver.try_recv() {
            latest = Some(result);
        }
        latest
    }
}

/// Run a build closure under `catch_unwind` (issue #60 M1): return `Some(build)` on success,
/// or `None` (after logging to stderr) if it PANICKED. Factored out so the panic-survival
/// contract is unit-testable without a GPU — a worker loop's liveness escape hatch. Without
/// it a build panic (GPU OOM, an internal assert, a bad dimension) would exit the thread,
/// every future `dispatch` would keep succeeding, and `try_recv_result` would return `None`
/// FOREVER — all large rebuilds silently dropped, no crash/log/feedback. The geometry and
/// brick closures wrap their build in this; the diameter measure cannot panic and does not.
/// Generic over the built value so a test can inject a panicking closure with a trivial type.
pub(crate) fn build_catching<T>(generation: u64, build: impl FnOnce() -> T) -> Option<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(build)) {
        Ok(value) => Some(value),
        Err(_) => {
            eprintln!(
                "voxel-worker rebuild PANICKED building generation {generation} — the worker \
                 survived (caught); this rebuild is dropped and the shell keeps its current \
                 artifact. The next edit will re-dispatch."
            );
            None
        }
    }
}

/// Collapse any additional queued requests into the newest one (drain-to-latest), starting
/// from `first`. Non-blocking after `first` — takes whatever is already queued. The worker
/// never backlogs: it builds only the latest pending request. Generic over the request type
/// so every worker loop shares the ONE contract.
fn drain_to_latest<Request>(first: Request, request_receiver: &Receiver<Request>) -> Request {
    let mut latest = first;
    while let Ok(newer) = request_receiver.try_recv() {
        latest = newer;
    }
    latest
}

#[cfg(test)]
mod tests {
    use super::*;

    /// M1 — the worker's liveness escape hatch: a build that PANICS is caught and mapped to
    /// `None` (not a thread exit that would wedge the worker forever), and a SUBSEQUENT normal
    /// build still succeeds. This is the pure core of the run-loop's panic survival — the
    /// integration test drives it through the real thread, this proves the contract without a
    /// GPU. We swap in a silent panic hook so the caught panic doesn't spam test output.
    #[test]
    fn build_catching_survives_a_panic_and_still_builds_next() {
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        // A panicking build → None (caught), the "thread would have died" case.
        let panicked: Option<u32> = build_catching(1, || panic!("simulated GPU OOM in build"));
        assert!(
            panicked.is_none(),
            "a panicking build is caught and yields None — the worker does NOT die"
        );

        // The very next build still runs — the catch did not poison anything.
        let normal: Option<u32> = build_catching(2, || 42);
        assert_eq!(
            normal,
            Some(42),
            "after a caught panic, a subsequent normal build still completes (no wedge)"
        );

        std::panic::set_hook(previous_hook);
    }
}
