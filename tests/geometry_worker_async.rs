//! Headless coverage of the ASYNC geometry-rebuild worker loop (issue #60).
//!
//! The worker's load-bearing guarantees are LIVE-APP-only in `WindowedState` — the
//! synchronous `shot` goldens can't exercise a background thread + channel + supersede.
//! These tests drive the REAL [`GeometryWorker`] (a spawned thread, real mpsc channels,
//! the real `GenerationTracker`) headlessly on an offscreen wgpu device — no window, no
//! surface — and assert the guarantees a human reviewer would otherwise verify by hand:
//!
//! 1. **Non-blocking dispatch**: `dispatch` of a large (>threshold) request returns
//!    PROMPTLY (does not block for the full build); a poll immediately after is "not
//!    ready", and only after the worker finishes does a poll return the result — with the
//!    correct generation.
//! 2. **Supersede / newest-wins under REAL threading**: a burst of increasing-generation
//!    requests collapses to exactly ONE accepted result (the newest); every stale result
//!    is discarded — driving the actual worker + channel + `GenerationTracker`, exactly as
//!    `WindowedState::poll_geometry_worker` does.
//! 4. **Bad / empty request**: an empty scene (zero covering chunks) does not hang the
//!    worker; it returns a valid (empty) renderer tagged with the request generation.
//!
//! (Build-equivalence — a worker-built renderer matching a synchronous build — is test #3,
//! already covered by `worker_build_matches_sync_build_for_large_scene` in `gpu_parity`;
//! not duplicated here.)
//!
//! The accept/discard decision the shell makes on each poll is the PUBLIC
//! `GenerationTracker::accepts` (the shell's `poll_geometry_worker` is a thin wrapper over
//! it), so these tests reproduce the shell's exact decision without touching the window —
//! no testability refactor was needed.
//!
//! Run: `cargo test --features gpu --test geometry_worker_async`
#![cfg(feature = "gpu")]

use std::time::{Duration, Instant};

use voxel_worker::{
    GenerationTracker, GeometryRebuildRequest, GeometryRebuildResult, GeometryWorker, GpuContext,
    MaterialChoice, Scene, TwoLayerStore, ASYNC_REBUILD_CHUNK_THRESHOLD, COLOR_TARGET_FORMAT,
};
use voxel_worker::{GeometryParams, ShapeKind};

/// The bounded ceiling any poll-loop waits for the worker before failing LOUDLY. A hang is
/// a bug, so a timeout is a hard failure — never an unbounded wait. Generous (the large
/// fixture builds in well under a second on CI hardware) so a slow machine doesn't flake.
const WORKER_TIMEOUT: Duration = Duration::from_secs(30);

/// How long a NON-blocking `dispatch` is allowed to take before we call it "blocking". The
/// full build is many milliseconds; dispatch is a single channel `send`, so this is a wide
/// margin that still catches a regression that made dispatch wait on the build.
const DISPATCH_NONBLOCK_CEILING: Duration = Duration::from_millis(250);

/// Resolve a from-geometry box scene into the owned two-layer covering chunks + frame
/// params a real wholesale rebuild dispatches — exactly as `WindowedState` does
/// (`build_covering_chunks` + `recentre_voxels_for_resolve` + `placed_region_dimensions`).
/// `blocks_per_axis` sizes the covering set so a test can land above or below the async
/// threshold deterministically.
fn build_request(generation: u64, blocks_per_axis: u32, vpb: u32) -> GeometryRebuildRequest {
    let geometry = GeometryParams {
        shape: ShapeKind::Box,
        size_voxels: [blocks_per_axis * vpb; 3],
        size_measurements: None,
        voxels_per_block: vpb,
        wall_blocks: 1,
    };
    let scene = Scene::from_geometry(geometry, MaterialChoice::default());
    let two_layer_chunks = TwoLayerStore::enabled().build_covering_chunks(&scene, vpb, 0);
    let recentre_voxels = scene.recentre_voxels_for_resolve(vpb);
    let grid_dimensions = scene.placed_region_dimensions(vpb);
    GeometryRebuildRequest {
        generation,
        two_layer_chunks,
        grid_dimensions,
        recentre_voxels,
        density: vpb,
    }
}

/// A LARGE request whose covering set exceeds `ASYNC_REBUILD_CHUNK_THRESHOLD` — the case
/// the live shell actually dispatches to the worker. Asserts the precondition so the test
/// is representative (not silently building a trivially-small scene).
fn large_request(generation: u64) -> GeometryRebuildRequest {
    let vpb = 16u32;
    // 24³ blocks → 6×6×6 = 216 covering chunks at d16, comfortably > 128 threshold.
    let request = build_request(generation, 24, vpb);
    assert!(
        request.two_layer_chunks.len() > ASYNC_REBUILD_CHUNK_THRESHOLD,
        "fixture must exceed the async threshold to be representative: {} chunks (need > {})",
        request.two_layer_chunks.len(),
        ASYNC_REBUILD_CHUNK_THRESHOLD
    );
    request
}

/// Poll the worker with a BOUNDED wait, yielding between polls (mirrors the event loop's
/// per-frame `try_recv_result`). Returns the first result, or fails loudly on timeout — a
/// hang is a bug, so we never wait unbounded.
fn poll_until_result(worker: &GeometryWorker, context: &str) -> GeometryRebuildResult {
    let deadline = Instant::now() + WORKER_TIMEOUT;
    loop {
        if let Some(result) = worker.try_recv_result() {
            return result;
        }
        if Instant::now() >= deadline {
            panic!(
                "{context}: worker produced no result within {WORKER_TIMEOUT:?} — the loop \
                 hung (a bug), never an acceptable unbounded wait"
            );
        }
        std::thread::yield_now();
        std::thread::sleep(Duration::from_millis(1));
    }
}

// ===========================================================================
// Test 1 — non-blocking dispatch
// ===========================================================================

/// Dispatching a LARGE rebuild returns PROMPTLY (the build runs on the worker thread, not
/// inline). Immediately after dispatch a poll is "not ready"; only after the worker
/// finishes does a poll return the result — tagged with the dispatched generation. This is
/// the "the UI never freezes" guarantee: `dispatch` must not block for the ~3s build.
#[test]
fn dispatch_is_non_blocking_and_result_arrives_with_correct_generation() {
    let gpu = pollster::block_on(GpuContext::new(None));
    let worker = GeometryWorker::spawn(gpu.device.clone(), gpu.queue.clone(), COLOR_TARGET_FORMAT);

    let generation = 1u64;
    let request = large_request(generation);

    // (a) dispatch returns promptly — it is a single channel send, NOT the build.
    let started = Instant::now();
    worker.dispatch(request);
    let dispatch_elapsed = started.elapsed();
    assert!(
        dispatch_elapsed < DISPATCH_NONBLOCK_CEILING,
        "dispatch blocked for {dispatch_elapsed:?} (ceiling {DISPATCH_NONBLOCK_CEILING:?}) — \
         it must NOT wait on the build; the UI would freeze"
    );

    // (b) the result is not instantaneous — the worker still has to build it. A poll right
    // after dispatch overwhelmingly returns "not ready" (the build is many ms). This is a
    // best-effort observation of asynchrony: if the worker were somehow instant we don't
    // fail (that would be a strictly BETTER outcome), but the poll-loop below still proves
    // the result genuinely arrives from the thread.
    let immediate = worker.try_recv_result();
    assert!(
        immediate.is_none(),
        "a poll immediately after dispatching a large build unexpectedly had a result — the \
         build should still be running on the worker thread"
    );

    // (c) after the worker finishes, a bounded poll returns the result with OUR generation.
    let result = poll_until_result(&worker, "non-blocking dispatch");
    assert_eq!(
        result.generation, generation,
        "the arrived result must carry the dispatched generation"
    );
    // Sanity: the box actually meshed (the worker built real geometry, not an empty stub).
    assert!(
        result.renderer.face_count() > 0,
        "the large box must mesh to a non-empty face set (the worker built real geometry)"
    );

    // Nothing else is queued — a single dispatch yields exactly one result.
    assert!(
        worker.try_recv_result().is_none(),
        "only one result for one dispatch"
    );
}

// ===========================================================================
// Test 2 — supersede / newest-wins under REAL threading
// ===========================================================================

/// A burst of increasing-generation dispatches drives the REAL worker + channel +
/// `GenerationTracker`. The shell (here, this test — same decision as
/// `WindowedState::poll_geometry_worker`) accepts a result ONLY when its generation is the
/// newest dispatched; every stale result is discarded. Assert the newest generation is
/// ultimately accepted and no accepted result is ever stale — the "a mid-build edit is
/// never clobbered by an older in-flight build" invariant, exercised through the threaded
/// path (not the pure tracker, which `geometry_worker`'s unit tests already cover).
#[test]
fn burst_supersede_accepts_only_newest_generation_under_real_threading() {
    let gpu = pollster::block_on(GpuContext::new(None));
    let worker = GeometryWorker::spawn(gpu.device.clone(), gpu.queue.clone(), COLOR_TARGET_FORMAT);

    // The shell's generation bookkeeping — the SAME type the live app holds. We stamp each
    // dispatch with `next_generation` and accept via `accepts`, exactly as the shell does.
    let mut tracker = GenerationTracker::new();

    // Dispatch a burst of large requests in quick succession (increasing generation), the
    // "user edits again while a build is in flight" scenario. The worker drains-to-latest,
    // so it need not build every one; the contract is only that the NEWEST is accepted and
    // no stale result is.
    const BURST: u64 = 6;
    let mut newest = 0u64;
    for _ in 0..BURST {
        newest = tracker.next_generation();
        worker.dispatch(large_request(newest));
    }
    assert_eq!(newest, BURST, "generations strictly increase from 1");

    // Drive the shell's poll+accept decision under a bounded deadline until the NEWEST
    // generation's result has been accepted. Along the way, assert the shell NEVER accepts
    // a stale (superseded) result — the load-bearing supersede guarantee.
    let deadline = Instant::now() + WORKER_TIMEOUT;
    let mut accepted_newest = false;
    let mut results_seen = 0u64;
    while !accepted_newest {
        if let Some(result) = worker.try_recv_result() {
            results_seen += 1;
            let would_accept = tracker.accepts(result.generation);
            if result.generation == newest {
                assert!(
                    would_accept,
                    "the newest generation ({newest}) must be accepted — it is the latest \
                     dispatched"
                );
                assert!(
                    result.renderer.face_count() > 0,
                    "the accepted newest result must be a real (non-empty) build"
                );
                accepted_newest = true;
            } else {
                assert!(
                    !would_accept,
                    "a stale result (generation {}, newest is {newest}) must be DISCARDED, \
                     never accepted — that would clobber the fresher scene",
                    result.generation
                );
            }
        }
        if !accepted_newest && Instant::now() >= deadline {
            panic!(
                "supersede burst: the newest generation ({newest}) never arrived within \
                 {WORKER_TIMEOUT:?} (saw {results_seen} result(s)) — the worker loop hung"
            );
        }
        if !accepted_newest {
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    assert!(
        accepted_newest,
        "the newest generation must ultimately be accepted"
    );
    // The tracker still ranks only the newest as acceptable afterwards (no later dispatch).
    assert!(tracker.accepts(newest));
    for stale in 1..newest {
        assert!(
            !tracker.accepts(stale),
            "every superseded generation ({stale}) stays discarded"
        );
    }
}

// ===========================================================================
// Test 4 — bad / empty request does not hang the worker
// ===========================================================================

/// An empty scene (zero covering chunks) must NOT hang the worker: it drains, builds an
/// empty renderer, and returns a result tagged with the request generation. The worker
/// then stays alive and services a subsequent NORMAL request — proving a degenerate input
/// neither wedges the loop nor poisons the channel.
#[test]
fn empty_request_does_not_hang_worker_and_it_survives_for_the_next() {
    let gpu = pollster::block_on(GpuContext::new(None));
    let worker = GeometryWorker::spawn(gpu.device.clone(), gpu.queue.clone(), COLOR_TARGET_FORMAT);

    // A request with NO covering chunks — the degenerate "empty scene / zero chunks" case.
    let empty = GeometryRebuildRequest {
        generation: 1,
        two_layer_chunks: Vec::new(),
        grid_dimensions: [0, 0, 0],
        recentre_voxels: [0, 0, 0],
        density: 16,
    };
    worker.dispatch(empty);
    let result = poll_until_result(&worker, "empty request");
    assert_eq!(result.generation, 1, "the empty build carries its generation");
    assert_eq!(
        result.renderer.face_count(),
        0,
        "an empty scene meshes to zero faces (no geometry), but still returns a result"
    );

    // The worker survived the degenerate request — a normal follow-up still builds.
    let follow_up = large_request(2);
    worker.dispatch(follow_up);
    let result = poll_until_result(&worker, "post-empty follow-up");
    assert_eq!(
        result.generation, 2,
        "the worker services a normal request after an empty one (it did not wedge)"
    );
    assert!(
        result.renderer.face_count() > 0,
        "the follow-up box meshed — the worker loop is still healthy"
    );
}
