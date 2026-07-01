//! Async wholesale geometry rebuild worker (issue #60, ADR 0003 §7).
//!
//! The live app's WHOLESALE geometry rebuild — two-layer classify + the per-chunk
//! cuboid mesh's CPU build + GPU buffer upload — is ~3s for a large object on an
//! initial-create / resize / density / recentre edit. Doing it inline blocks the main
//! thread and freezes the UI. This module moves that build onto a background worker so
//! the UI never stalls: the main thread keeps rendering the CURRENT mesh
//! (stale-while-rebuilding) until the worker's freshly-built [`CuboidMeshRenderer`]
//! arrives, then swaps it in.
//!
//! ## What crosses the channel (why this is sound in wgpu 29)
//! wgpu 29's `Device`/`Queue` — and every GPU handle a [`CuboidMeshRenderer`] holds
//! (`RenderPipeline`, `Buffer`, `BindGroup`, `Sampler`, `BindGroupLayout`) — are
//! `Send + Sync + Clone` (Arc-backed). So the worker **clones `device`/`queue` and
//! builds the WHOLE renderer off-thread**, GPU buffers included, and the finished
//! renderer crosses the channel intact. Only surface acquire/present stays on the main
//! thread (never touched here). The mesh build calls the SAME
//! [`CuboidMeshRenderer::new_from_two_layer_chunks`] the synchronous path calls, so the
//! output is byte-identical (the build-equivalence net — see the tests).
//!
//! ## Division of labour (what the main thread still does synchronously)
//! The two-layer resolve/classify (`AppCore::rebuild`) runs on the main thread — it
//! mutates the resident cache (`&mut AppCore`, the sole document-adjacent writer) and
//! is comparatively cheap; it produces the OWNED `two_layer_chunks` (`Send`). Only the
//! **mesh CPU build + GPU upload** (the heavy `CuboidMeshRenderer` construction) is
//! dispatched here. Fog stays demand-driven on the main thread (#56–#59), unchanged.
//!
//! ## Supersede / generation (drain-to-latest)
//! Every request carries a monotonic [`generation`](GeometryRebuildRequest::generation).
//! If the user edits again mid-build, the shell sends a newer request; the worker
//! **drains its queue to the latest** (never backlogs — it builds only the newest
//! pending request) and the shell **discards any received result whose generation is
//! stale** (an older generation than the newest request it has dispatched). The
//! accept/discard decision is factored into [`GenerationTracker`] so it is unit-testable
//! without a live window.

use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;

use crate::cuboid_mesh::CuboidMeshRenderer;
use crate::two_layer_store::TwoLayerChunk;

/// The covering-chunk count above which a WHOLESALE geometry rebuild is dispatched to
/// the background worker instead of built inline (issue #60).
///
/// A rebuild covering at most this many chunks is cheap enough to build synchronously on
/// the main thread without a perceptible hitch (the small-object common case), so it
/// avoids the worker hop + its one-frame swap latency. Only a rebuild whose covering set
/// EXCEEDS this — a large object's initial create / resize / density / recentre, the ~3s
/// case the issue targets — goes async. Chosen conservatively: at the default density a
/// chunk is `4×4×4` blocks, so 128 chunks is a large multi-hundred-block object, well
/// past the point where an inline build stalls a frame. (Incremental dirty-chunk edits —
/// the #54/#55 fast path — stay inline REGARDLESS of this threshold; only WHOLESALE
/// rebuilds consult it.)
pub const ASYNC_REBUILD_CHUNK_THRESHOLD: usize = 128;

/// A request to build a wholesale cuboid mesh on the worker (issue #60). Carries the
/// OWNED two-layer chunks the resolve produced plus the frame parameters
/// [`CuboidMeshRenderer::new_from_two_layer_chunks`] needs — all `Send` plain data.
pub struct GeometryRebuildRequest {
    /// Monotonic generation stamp (supersede key). A result is accepted only when its
    /// generation matches the newest request the shell has dispatched (see
    /// [`GenerationTracker`]).
    pub generation: u64,
    /// The two-layer covering chunks the resolve produced (owned; `Send`). Meshed via
    /// coarse one-box + microblock cuboids + seam-flag culling — the sole runtime path.
    pub two_layer_chunks: Vec<([i32; 3], TwoLayerChunk)>,
    /// The whole composite grid's voxel dims (the band-clip layer mapping).
    pub grid_dimensions: [u32; 3],
    /// The composite recentre (floating origin, voxels; ADR 0008) the mesh lands in.
    pub recentre_voxels: [i64; 3],
    /// The document density (voxels per block) the chunks were resolved at.
    pub density: u32,
}

/// A finished wholesale mesh built by the worker (issue #60): the whole
/// [`CuboidMeshRenderer`] (GPU buffers included) tagged with the request generation it
/// was built for, so the shell can discard a stale result and swap in a fresh one.
pub struct GeometryRebuildResult {
    /// The generation of the [`GeometryRebuildRequest`] this result was built for.
    pub generation: u64,
    /// The freshly built renderer (crosses the channel whole — wgpu 29 handles are
    /// `Send`). Swapped into `WindowedState::cuboid_mesh_renderer` when accepted.
    pub renderer: CuboidMeshRenderer,
}

/// The background geometry worker (issue #60): owns the cloned `device`/`queue` and a
/// build loop. The shell sends [`GeometryRebuildRequest`]s and polls
/// [`try_recv_result`](Self::try_recv_result) each frame.
pub struct GeometryWorker {
    request_sender: Sender<GeometryRebuildRequest>,
    result_receiver: Receiver<GeometryRebuildResult>,
    /// Kept so the worker thread's lifetime is tied to the handle; the channel close on
    /// drop signals the loop to exit (its `recv` errors and it returns).
    _worker: JoinHandle<()>,
}

impl GeometryWorker {
    /// Spawn the worker with cloned GPU handles (issue #60). `device`/`queue` are cloned
    /// (wgpu 29 Arc-backed) so the worker can create the mesh's GPU buffers off the main
    /// thread; `color_format` is the render target format the pipelines are built for.
    ///
    /// The loop **drains its request channel to the latest** before building (so a burst
    /// of edits collapses to one build of the newest request — never a backlog), builds
    /// via the SAME [`CuboidMeshRenderer::new_from_two_layer_chunks`] the sync path uses,
    /// and sends the result back. It exits when the request channel closes (the shell,
    /// hence the `GeometryWorker`, dropped).
    pub fn spawn(
        device: wgpu::Device,
        queue: wgpu::Queue,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let (request_sender, request_receiver) = std::sync::mpsc::channel::<GeometryRebuildRequest>();
        let (result_sender, result_receiver) = std::sync::mpsc::channel::<GeometryRebuildResult>();
        let worker = std::thread::Builder::new()
            .name("voxel-worker geometry rebuild".to_string())
            .spawn(move || {
                run_geometry_worker(
                    &device,
                    &queue,
                    color_format,
                    &request_receiver,
                    &result_sender,
                );
            })
            .expect("failed to spawn geometry worker");
        Self {
            request_sender,
            result_receiver,
            _worker: worker,
        }
    }

    /// Dispatch a wholesale rebuild request to the worker (issue #60). Non-blocking; the
    /// worker drains to the latest, so sending a newer request while one is in flight
    /// supersedes it. A send error (worker gone) is ignored — the shell falls back to the
    /// stale mesh, never blocks.
    pub fn dispatch(&self, request: GeometryRebuildRequest) {
        let _ = self.request_sender.send(request);
    }

    /// Poll the worker's result channel WITHOUT blocking (issue #60): return the latest
    /// finished result if one has arrived, else `None`. Called each frame in the event
    /// loop; the shell then checks the result's generation against its tracker and, if
    /// fresh, swaps in the renderer + requests a redraw.
    ///
    /// Drains to the newest available result (an in-flight supersede can leave more than
    /// one queued) so the shell never integrates a stale build when a newer one is ready.
    pub fn try_recv_result(&self) -> Option<GeometryRebuildResult> {
        let mut latest = None;
        while let Ok(result) = self.result_receiver.try_recv() {
            latest = Some(result);
        }
        latest
    }
}

/// The worker loop (issue #60): block on the request channel, drain to the newest
/// pending request, build the mesh, send the result. Exits when the channel closes.
fn run_geometry_worker(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    color_format: wgpu::TextureFormat,
    request_receiver: &Receiver<GeometryRebuildRequest>,
    result_sender: &Sender<GeometryRebuildResult>,
) {
    // Block for the next request; when it closes (shell dropped) the loop ends.
    while let Ok(first) = request_receiver.recv() {
        // Drain-to-latest: if the user edited again while we were idle-waiting, collapse
        // the queued requests to the NEWEST so we never build a superseded generation.
        let request = drain_to_latest(first, request_receiver);
        let renderer = build_geometry(device, queue, color_format, &request);
        if result_sender
            .send(GeometryRebuildResult {
                generation: request.generation,
                renderer,
            })
            .is_err()
        {
            // The shell is gone; stop.
            return;
        }
    }
}

/// Collapse any additional queued requests into the newest one (drain-to-latest, issue
/// #60), starting from `first`. Non-blocking after `first` — takes whatever is already
/// queued. The worker never backlogs: it builds only the latest pending request.
fn drain_to_latest(
    first: GeometryRebuildRequest,
    request_receiver: &Receiver<GeometryRebuildRequest>,
) -> GeometryRebuildRequest {
    let mut latest = first;
    while let Ok(newer) = request_receiver.try_recv() {
        latest = newer;
    }
    latest
}

/// Build the wholesale cuboid mesh for a request (issue #60) — the SAME call the
/// synchronous path makes, so the built renderer is byte-identical (the build-equivalence
/// net asserts this). Factored out so the worker loop and the build-equivalence test share
/// one build entry.
pub fn build_geometry(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    color_format: wgpu::TextureFormat,
    request: &GeometryRebuildRequest,
) -> CuboidMeshRenderer {
    CuboidMeshRenderer::new_from_two_layer_chunks(
        device,
        queue,
        color_format,
        &request.two_layer_chunks,
        request.grid_dimensions,
        request.recentre_voxels,
        request.density,
    )
}

/// The monotonic generation bookkeeping behind supersede (issue #60) — factored out of
/// the live shell so the accept/discard decision is unit-testable without a window.
///
/// The shell holds one of these. On each WHOLESALE async dispatch it calls
/// [`next_generation`](Self::next_generation) to stamp the request; when a result arrives
/// it calls [`accepts`](Self::accepts) to decide whether to swap it in. A result is
/// accepted only when its generation is the NEWEST dispatched — an older generation (a
/// build that a later edit superseded) is discarded, so the stale mesh is never swapped
/// in over a fresher scene.
#[derive(Debug, Default, Clone, Copy)]
pub struct GenerationTracker {
    /// The generation of the most recent request dispatched to the worker. `0` before any
    /// dispatch (no async build is outstanding, so nothing is accepted).
    latest_dispatched: u64,
}

impl GenerationTracker {
    /// A fresh tracker (no async rebuild dispatched yet).
    pub fn new() -> Self {
        Self { latest_dispatched: 0 }
    }

    /// Mint the next generation for a wholesale async dispatch and record it as the newest
    /// outstanding (issue #60). Generations are strictly increasing from 1, so a later
    /// edit's request always outranks an earlier one still in flight.
    pub fn next_generation(&mut self) -> u64 {
        self.latest_dispatched += 1;
        self.latest_dispatched
    }

    /// Whether a result of `generation` should be accepted (swapped in) or discarded as
    /// stale (issue #60). Accepted iff it matches the newest dispatched generation — a
    /// result from a superseded (older) request is discarded. A result arriving before any
    /// dispatch (or after the counter moved past it) is never accepted.
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

    /// A fresh tracker accepts nothing — no async rebuild is outstanding, so any result
    /// (which could only be a phantom) is discarded.
    #[test]
    fn fresh_tracker_accepts_nothing() {
        let tracker = GenerationTracker::new();
        assert!(!tracker.accepts(0), "generation 0 is never a valid result");
        assert!(!tracker.accepts(1), "no dispatch yet → accept nothing");
    }

    /// The newest dispatched generation is accepted; every earlier one is discarded as
    /// stale. This is the supersede invariant: a mid-build edit dispatches a newer
    /// generation, so the older in-flight result must NOT swap in over the fresher scene.
    #[test]
    fn newest_wins_stale_discarded() {
        let mut tracker = GenerationTracker::new();
        let first = tracker.next_generation();
        assert_eq!(first, 1);
        // A result for the first request is accepted while it is the newest.
        assert!(tracker.accepts(first));

        // The user edits again mid-build → a newer request is dispatched.
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
