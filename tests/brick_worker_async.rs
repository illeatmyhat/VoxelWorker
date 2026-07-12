//! Headless coverage of the ASYNC brick-pipeline worker loop (perf follow-up to epic
//! #64, on the issue #60 stale-while-rebuilding pattern).
//!
//! The worker's load-bearing guarantees are LIVE-APP-only in `WindowedState` — a window
//! poll loop over a background thread + channel + supersede — so, like
//! `geometry_worker_async.rs`, these tests drive the REAL [`BrickWorker`] (a spawned
//! thread, real mpsc channels, the real [`GenerationTracker`]) headlessly and assert:
//!
//! 1. **Non-blocking dispatch**: dispatching a large (>threshold) wholesale brick build
//!    returns PROMPTLY; a poll immediately after is "not ready"; the result arrives with
//!    the dispatched generation and artifacts BYTE-IDENTICAL to a synchronous build.
//! 2. **Supersede / newest-wins under REAL threading**: a burst of increasing-generation
//!    requests collapses to exactly one accepted result (the newest); every stale result
//!    is discarded — the same accept/discard the shell's `poll_brick_worker` makes.
//! 3. **Empty request**: a zero-chunk request does not hang the worker (it yields
//!    [`BrickRebuildOutcome::Empty`]) and the worker survives to service the next
//!    normal request.
//!
//! The brick pipeline is pure CPU (no GPU handles cross the channel), so this file runs
//! on BOTH feature sets — no `gpu` gate, no offscreen device.
//!
//! Run: `cargo test --test brick_worker_async`

use std::time::{Duration, Instant};

use voxel_worker::{
    build_brick_field, BrickRebuildOutcome, BrickRebuildRequest, BrickRebuildResult,
    BrickWorker, GenerationTracker, GeometryParams, MaterialChoice, Scene, ShapeKind,
    TwoLayerStore, ASYNC_REBUILD_CHUNK_THRESHOLD,
};

/// The bounded ceiling any poll-loop waits for the worker before failing LOUDLY. A hang
/// is a bug, so a timeout is a hard failure — never an unbounded wait.
const WORKER_TIMEOUT: Duration = Duration::from_secs(60);

/// How long a NON-blocking `dispatch` is allowed to take before we call it "blocking".
/// The full build is many milliseconds; dispatch is a single channel `send`.
const DISPATCH_NONBLOCK_CEILING: Duration = Duration::from_millis(250);

/// Build a wholesale brick request for a from-geometry box of `blocks³` at `vpb` —
/// exactly the covering set + scalars the live shell dispatches.
fn build_request(generation: u64, blocks: u32, vpb: u32) -> BrickRebuildRequest {
    let scene = Scene::from_geometry(
        GeometryParams {
            shape: ShapeKind::Box,
            size_voxels: [blocks * vpb; 3],
            size_measurements: None,
            voxels_per_block: vpb,
            wall_blocks: 1,
        },
        MaterialChoice::Stone,
    );
    let two_layer_chunks = TwoLayerStore::enabled().build_covering_chunks(&scene, vpb, 0);
    BrickRebuildRequest {
        generation,
        two_layer_chunks,
        density: vpb,
        recentre_voxels: [7, -3, 11],
        build_display_artifacts: true,
    }
}

/// A LARGE request whose covering set exceeds `ASYNC_REBUILD_CHUNK_THRESHOLD` — the case
/// the live shell actually dispatches to the worker.
fn large_request(generation: u64) -> BrickRebuildRequest {
    // 24³ blocks → 6³ = 216 covering chunks, comfortably > the 128 threshold.
    let request = build_request(generation, 24, 4);
    assert!(
        request.two_layer_chunks.len() > ASYNC_REBUILD_CHUNK_THRESHOLD,
        "fixture must exceed the async threshold to be representative: {} chunks (need > {})",
        request.two_layer_chunks.len(),
        ASYNC_REBUILD_CHUNK_THRESHOLD
    );
    request
}

/// Poll the worker with a BOUNDED wait, yielding between polls (mirrors the event loop's
/// per-frame `try_recv_result`). Fails loudly on timeout — a hang is a bug.
fn poll_until_result(worker: &BrickWorker, context: &str) -> BrickRebuildResult {
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
// Test 1 — non-blocking dispatch + artifacts byte-identical to a sync build
// ===========================================================================

/// Dispatching a LARGE wholesale brick rebuild returns PROMPTLY (the ~2s-class build runs
/// on the worker thread, not inline — the "the UI never freezes" guarantee), and the
/// arrived result carries the dispatched generation + recentre and artifacts equal to the
/// synchronous calls the pre-async shell made (the build-equivalence net).
#[test]
fn dispatch_is_non_blocking_and_result_matches_sync_build() {
    let worker = BrickWorker::spawn();
    let generation = 1u64;
    let request = large_request(generation);
    // The ground truth the worker's artifacts must equal — the SAME calls the
    // synchronous path made, over the same covering set.
    let chunks = request.two_layer_chunks.clone();
    let density = request.density;
    let recentre = request.recentre_voxels;

    let started = Instant::now();
    worker.dispatch(request);
    let dispatch_elapsed = started.elapsed();
    assert!(
        dispatch_elapsed < DISPATCH_NONBLOCK_CEILING,
        "dispatch blocked for {dispatch_elapsed:?} (ceiling {DISPATCH_NONBLOCK_CEILING:?}) — \
         it must NOT wait on the build; the UI would freeze"
    );
    // Best-effort asynchrony observation (a build this size is many ms; an instant
    // worker would be strictly better, but the poll below still proves arrival).
    assert!(
        worker.try_recv_result().is_none(),
        "a poll immediately after dispatching a large build unexpectedly had a result"
    );

    let result = poll_until_result(&worker, "non-blocking dispatch");
    assert_eq!(result.generation, generation);
    assert_eq!(
        result.recentre_voxels, recentre,
        "the recentre travels with the build (ADR 0008 — never re-derived at install)"
    );
    let outcome = result.outcome.expect("a normal build returns Some outcome");
    let BrickRebuildOutcome::Display(install) = outcome else {
        panic!("a single-material box is representable — expected Display");
    };
    // One anchor assert that the THREADED result is the shared `build_brick_rebuild`
    // output — the full artifact-by-artifact byte equivalence (pyramid, GPU records,
    // mirror round-trip) is the unit test `display_outcome_equals_synchronous_build`
    // in `brick_worker.rs`, over the same entry point; not duplicated here.
    assert_eq!(
        install.build,
        build_brick_field(&chunks, density),
        "the worker-built field matches a synchronous build"
    );
    assert!(
        worker.try_recv_result().is_none(),
        "only one result for one dispatch"
    );
}

// ===========================================================================
// Test 2 — supersede / newest-wins under REAL threading
// ===========================================================================

/// A burst of increasing-generation dispatches drives the REAL worker + channel +
/// `GenerationTracker`: the shell accepts a result ONLY when its generation is the newest
/// dispatched — the same decision `poll_brick_worker` makes — so a mid-build edit is
/// never clobbered by an older in-flight brick build.
#[test]
fn burst_supersede_accepts_only_newest_generation_under_real_threading() {
    let worker = BrickWorker::spawn();
    let mut tracker = GenerationTracker::new();

    const BURST: u64 = 6;
    let mut newest = 0u64;
    for _ in 0..BURST {
        newest = tracker.next_generation();
        worker.dispatch(large_request(newest));
    }
    assert_eq!(newest, BURST, "generations strictly increase from 1");

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
                    "the newest generation ({newest}) must be accepted"
                );
                assert!(
                    matches!(result.outcome, Some(BrickRebuildOutcome::Display(_))),
                    "the accepted newest result is a real (representable) build"
                );
                accepted_newest = true;
            } else {
                assert!(
                    !would_accept,
                    "a stale result (generation {}, newest is {newest}) must be DISCARDED — \
                     installing it would clobber the fresher scene",
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
    for stale in 1..newest {
        assert!(
            !tracker.accepts(stale),
            "every superseded generation ({stale}) stays discarded"
        );
    }
}

// ===========================================================================
// Test 3 — empty request does not hang the worker
// ===========================================================================

/// A zero-chunk request must NOT hang the worker: it yields `Empty` tagged with the
/// request generation, and the worker survives to service a subsequent normal request.
#[test]
fn empty_request_does_not_hang_worker_and_it_survives_for_the_next() {
    let worker = BrickWorker::spawn();
    worker.dispatch(BrickRebuildRequest {
        generation: 1,
        two_layer_chunks: Vec::new(),
        density: 16,
        recentre_voxels: [0; 3],
        build_display_artifacts: true,
    });
    let result = poll_until_result(&worker, "empty request");
    assert_eq!(result.generation, 1);
    assert!(
        matches!(result.outcome, Some(BrickRebuildOutcome::Empty)),
        "an empty covering set yields Empty (the shell clears mirror + field)"
    );

    // The worker survived the degenerate request — a normal follow-up still builds.
    worker.dispatch(large_request(2));
    let result = poll_until_result(&worker, "post-empty follow-up");
    assert_eq!(result.generation, 2, "the worker did not wedge");
    assert!(
        matches!(result.outcome, Some(BrickRebuildOutcome::Display(_))),
        "the follow-up built a real field — the worker loop is still healthy"
    );
}
