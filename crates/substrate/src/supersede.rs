//! Monotonic-generation, latest-wins supersede — a work-coalescing concurrency
//! primitive.
//!
//! One producer streams *requests* to a background *worker* that turns each into a
//! *response*; when requests arrive faster than they can be served, only the **newest**
//! matters and every superseded one is dropped, on both sides of the channel:
//!
//! * **Drain side (the worker):** [`CoalescingWorker`] runs one build on a dedicated
//!   thread. When it wakes it [`drain_to_latest`]s its request queue — collapsing a burst
//!   of queued requests down to the single newest — so it never backlogs and never spends
//!   work on a request a later one already superseded.
//! * **Accept side (the dispatcher):** [`GenerationTracker`] mints a strictly increasing
//!   *generation* for every dispatch and, when a result comes back, `accepts` it only if
//!   its generation is still the newest dispatched. A result from a request that a later
//!   dispatch superseded is discarded, so a stale result never overwrites a fresher state.
//!
//! Together these give the *stale-while-rebuilding* guarantee: the consumer keeps using
//! its current (possibly stale) value until a **strictly newer** result is ready, then
//! swaps it in — never a swap backwards. The two halves are independent — the drain keeps
//! the worker from wasting effort; the generation gate keeps a lost race (an in-flight
//! build finishing *after* a newer dispatch) from installing stale output — and correct
//! supersede needs both.
//!
//! ## Literature
//!
//! This pattern has no single canonical name: it is the confluence of *work coalescing*
//! (collapse a burst of pending work to its latest representative — the "conflation" of
//! event/market-data queues), *stale-while-revalidate* (serve the current value while a
//! fresh one is computed), and a *monotonic version counter* used as a lost-update guard.
//! The monotonic-counter reasoning — a strictly increasing generation totally orders the
//! dispatches, so "newest wins" is well-defined and a superseded generation can never be
//! mistaken for current — is the standard versioned-state argument; see Herlihy & Shavit,
//! *The Art of Multiprocessor Programming* (2nd ed., 2021), on monotonic counters and
//! sequence/version numbers for coordinating concurrent readers and writers.
//!
//! Everything in THIS module is `std`-only (`std::thread`, `std::sync::mpsc`,
//! `std::panic::catch_unwind`) — no third-party crate. (The substrate crate as a whole has one
//! dependency, `rayon`, used by [`crate::spatial::min_mip_pyramid`]'s parallel sort; the supersede
//! primitive itself pulls in none of it.) The
//! worker body's panic containment ([`catch_unwind_or_log`]) is the loop's liveness escape
//! hatch, kept here because it is part of the same protocol: a build that panics must not
//! silently kill the worker thread and wedge every future request.

use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;

/// A background worker that serves one stream of requests on a dedicated thread, coalescing
/// bursts to the latest (the *drain* half of the supersede protocol).
///
/// The caller [`dispatch`](Self::dispatch)es `Request`s and polls
/// [`try_recv_result`](Self::try_recv_result); the build closure — captured at
/// [`spawn`](Self::spawn) — turns each request into a whole `Response`.
///
/// The type parameters are named to the ROLE, not the payload: `Request` is what crosses
/// TO the worker, `Response` what crosses back (deliberately not `Result`, which would
/// shadow [`std::result::Result`]). The closure returns the WHOLE `Response`, so any panic
/// containment a caller needs (mapping a panicked build to a `None`-tagged result rather
/// than a thread exit) lives INSIDE that closure — this primitive makes no such policy, so
/// a worker whose build genuinely cannot panic carries none. See [`catch_unwind_or_log`]
/// for the ready-made containment helper.
pub struct CoalescingWorker<Request: Send + 'static, Response: Send + 'static> {
    request_sender: Sender<Request>,
    result_receiver: Receiver<Response>,
    /// Kept so the worker thread's lifetime is tied to the handle; the channel close on
    /// drop signals the loop to exit (its `recv` errors and it returns).
    _worker: JoinHandle<()>,
}

impl<Request: Send + 'static, Response: Send + 'static> CoalescingWorker<Request, Response> {
    /// Spawn the worker on a dedicated thread named `thread_name`, running `build` for
    /// each request.
    ///
    /// The loop is the shared contract: block on the request channel, **drain to the
    /// latest** pending request (so a burst of dispatches collapses to one build of the
    /// newest — never a backlog), build it, and send the result back. It exits when the
    /// request channel closes (the caller, hence this worker, dropped) or when the caller
    /// has dropped the result receiver (its `send` errors). `build` takes the request by
    /// value and returns the full `Response`, so a caller that must survive a build panic
    /// wraps its own body in a catch (see [`catch_unwind_or_log`]).
    pub fn spawn(
        thread_name: &str,
        mut build: impl FnMut(Request) -> Response + Send + 'static,
    ) -> Self {
        let (request_sender, request_receiver) = std::sync::mpsc::channel::<Request>();
        let (result_sender, result_receiver) = std::sync::mpsc::channel::<Response>();
        let worker = std::thread::Builder::new()
            .name(thread_name.to_string())
            .spawn(move || {
                // Block for the next request; when it closes (caller dropped) the loop ends.
                while let Ok(first) = request_receiver.recv() {
                    // Drain-to-latest: if newer requests were queued while we were
                    // idle-waiting, collapse them to the NEWEST so we never build a
                    // superseded generation.
                    let request = drain_to_latest(first, &request_receiver);
                    if result_sender.send(build(request)).is_err() {
                        // The caller is gone; stop.
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
    /// (worker gone) is ignored — the caller keeps its stale value, never blocks.
    pub fn dispatch(&self, request: Request) {
        let _ = self.request_sender.send(request);
    }

    /// Poll the result channel WITHOUT blocking: return the latest finished result if one
    /// has arrived, else `None`. Called each frame in an event loop. Drains to the newest
    /// available (an in-flight supersede can leave more than one queued) so the caller never
    /// integrates a stale build when a newer one is ready.
    pub fn try_recv_result(&self) -> Option<Response> {
        let mut latest = None;
        while let Ok(result) = self.result_receiver.try_recv() {
            latest = Some(result);
        }
        latest
    }
}

/// Run a build closure under `catch_unwind`: return `Some(build())` on success, or `None`
/// (after logging to stderr) if it PANICKED — the worker loop's liveness escape hatch.
///
/// Without it a build panic would unwind out of the loop and exit the thread; every future
/// `dispatch` would keep succeeding into a dead channel and `try_recv_result` would return
/// `None` FOREVER — all work silently dropped, no crash, no feedback. Factored out so the
/// panic-survival contract is unit-testable in isolation, and generic over the built value
/// so a test can inject a panicking closure with a trivial type. `generation` is the
/// supersede key of the build, carried only into the diagnostic so a dropped build is
/// identifiable in a log.
pub fn catch_unwind_or_log<T>(generation: u64, build: impl FnOnce() -> T) -> Option<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(build)) {
        Ok(value) => Some(value),
        Err(_) => {
            eprintln!(
                "supersede worker: build for generation {generation} PANICKED — caught; the \
                 worker survived, this result is dropped, and the caller keeps its current \
                 value. The next request re-dispatches."
            );
            None
        }
    }
}

/// Collapse any additional queued requests into the newest one (drain-to-latest), starting
/// from `first`. Non-blocking after `first` — takes whatever is already queued. The worker
/// never backlogs: it builds only the latest pending request. Generic over the request type
/// so every worker loop shares the ONE contract.
pub fn drain_to_latest<Request>(first: Request, request_receiver: &Receiver<Request>) -> Request {
    let mut latest = first;
    while let Ok(newer) = request_receiver.try_recv() {
        latest = newer;
    }
    latest
}

/// The monotonic-generation bookkeeping behind supersede — the *accept* half of the
/// protocol, factored out of any live shell so the accept/discard decision is unit-testable.
///
/// The dispatcher holds one of these. On each async dispatch it calls
/// [`next_generation`](Self::next_generation) to stamp the request; when a result arrives it
/// calls [`accepts`](Self::accepts) to decide whether to swap it in. A result is accepted
/// only when its generation is the NEWEST dispatched — an older generation (a build that a
/// later dispatch superseded) is discarded, so a stale result is never swapped in over a
/// fresher state.
#[derive(Debug, Default, Clone, Copy)]
pub struct GenerationTracker {
    /// The generation of the most recent request dispatched to the worker. `0` before any
    /// dispatch (nothing outstanding, so nothing is accepted).
    latest_dispatched: u64,
}

impl GenerationTracker {
    /// A fresh tracker (no request dispatched yet).
    pub fn new() -> Self {
        Self {
            latest_dispatched: 0,
        }
    }

    /// Mint the next generation for a dispatch and record it as the newest outstanding.
    /// Generations are strictly increasing from 1, so a later dispatch always outranks an
    /// earlier one still in flight.
    pub fn next_generation(&mut self) -> u64 {
        self.latest_dispatched += 1;
        self.latest_dispatched
    }

    /// Whether a result of `generation` should be accepted (swapped in) or discarded as
    /// stale. Accepted iff it matches the newest dispatched generation — a result from a
    /// superseded (older) request is discarded. A result arriving before any dispatch (or
    /// after the counter moved past it) is never accepted.
    pub fn accepts(&self, generation: u64) -> bool {
        generation != 0 && generation == self.latest_dispatched
    }

    /// The newest dispatched generation (diagnostic / test support).
    pub fn latest_dispatched(&self) -> u64 {
        self.latest_dispatched
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The worker body's liveness escape hatch: a build that PANICS is caught and mapped to
    /// `None` (not a thread exit that would wedge the worker forever), and a SUBSEQUENT
    /// normal build still succeeds. The pure core of the run-loop's panic survival. A silent
    /// panic hook keeps the caught panic from spamming test output.
    #[test]
    fn catch_unwind_or_log_survives_panic_and_runs_next() {
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        // A panicking build → None (caught), the "thread would have died" case.
        let panicked: Option<u32> = catch_unwind_or_log(1, || panic!("simulated build panic"));
        assert!(
            panicked.is_none(),
            "a panicking build is caught and yields None — the worker does NOT die"
        );

        // The very next build still runs — the catch did not poison anything.
        let normal: Option<u32> = catch_unwind_or_log(2, || 42);
        assert_eq!(
            normal,
            Some(42),
            "after a caught panic, a subsequent normal build still completes (no wedge)"
        );

        std::panic::set_hook(previous_hook);
    }

    /// [`drain_to_latest`] collapses a burst of already-queued requests to the newest one,
    /// consuming the intervening requests — the drain half of work coalescing.
    #[test]
    fn drain_to_latest_collapses_a_burst() {
        let (sender, receiver) = std::sync::mpsc::channel::<u32>();
        // The worker pulled `1` off the channel (its `first`); `2..=5` were queued behind it.
        for queued in 2..=5 {
            sender.send(queued).unwrap();
        }
        let latest = drain_to_latest(1, &receiver);
        assert_eq!(latest, 5, "the burst collapses to the newest queued request");
        assert!(
            receiver.try_recv().is_err(),
            "every intervening request is consumed — no backlog is left"
        );

        // With nothing queued behind it, `first` is returned unchanged.
        assert_eq!(drain_to_latest(7, &receiver), 7);
    }

    /// A fresh tracker accepts nothing — no dispatch is outstanding, so any result (which
    /// could only be a phantom) is discarded.
    #[test]
    fn fresh_generation_tracker_accepts_nothing() {
        let tracker = GenerationTracker::new();
        assert!(!tracker.accepts(0), "generation 0 is never a valid result");
        assert!(!tracker.accepts(1), "no dispatch yet → accept nothing");
    }

    /// The newest dispatched generation is accepted; every earlier one is discarded as
    /// stale. This is the supersede invariant: a mid-build dispatch mints a newer
    /// generation, so the older in-flight result must NOT swap in over the fresher state.
    #[test]
    fn newest_generation_wins_stale_superseded() {
        let mut tracker = GenerationTracker::new();
        let first = tracker.next_generation();
        assert_eq!(first, 1);
        // A result for the first request is accepted while it is the newest.
        assert!(tracker.accepts(first));

        // A newer request is dispatched mid-build.
        let second = tracker.next_generation();
        assert_eq!(second, 2, "generations strictly increase");
        // Now the FIRST (in-flight) result is stale and must be discarded…
        assert!(!tracker.accepts(first), "superseded generation is discarded");
        // …and only the newest is accepted.
        assert!(tracker.accepts(second));
    }

    /// Several supersedes in a row: only the final generation is accepted; every
    /// intermediate one is stale (the drain-to-latest + newest-wins contract).
    #[test]
    fn only_final_generation_accepted_after_burst() {
        let mut tracker = GenerationTracker::new();
        let mut last = 0;
        for _ in 0..5 {
            last = tracker.next_generation();
        }
        assert_eq!(last, 5);
        for stale in 1..last {
            assert!(!tracker.accepts(stale), "generation {stale} is superseded");
        }
        assert!(tracker.accepts(last), "only the newest generation wins");
        assert_eq!(tracker.latest_dispatched(), 5);
    }
}
