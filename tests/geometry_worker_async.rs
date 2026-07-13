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
    build_brick_field, route_geometry_rebuild, spawn_geometry_worker, BrickFieldBuild,
    CuboidMeshRenderer, EditShape, GenerationTracker, GeometryRebuildRequest, GpuContext,
    IncrementalBrickField, LayerBand, MaterialChoice, RebuildRoute, RecentreVoxels, TwoLayerStore,
    ASYNC_REBUILD_CHUNK_THRESHOLD, COLOR_TARGET_FORMAT,
};

mod common;

/// The bounded ceiling any poll-loop waits for the worker before failing LOUDLY. A hang is
/// a bug, so a timeout is a hard failure — never an unbounded wait. Generous (the large
/// fixture builds in well under a second on CI hardware) so a slow machine doesn't flake.
const WORKER_TIMEOUT: Duration = Duration::from_secs(30);

/// Resolve a from-geometry box scene into the owned two-layer covering chunks + frame
/// params a real wholesale rebuild dispatches — exactly as `WindowedState` does
/// (`build_covering_chunks` + `recentre_voxels_for_resolve` + `placed_region_dimensions`).
/// `blocks_per_axis` sizes the covering set so a test can land above or below the async
/// threshold deterministically.
fn build_request(generation: u64, blocks_per_axis: u32, vpb: u32) -> GeometryRebuildRequest {
    let scene = common::box_scene(blocks_per_axis, vpb, MaterialChoice::default());
    let two_layer_chunks = TwoLayerStore::enabled().build_covering_chunks(&scene, vpb, 0);
    let recentre_voxels = scene.recentre_voxels_for_resolve(vpb);
    let grid_dimensions = scene.placed_region_dimensions(vpb);
    GeometryRebuildRequest {
        generation,
        two_layer_chunks,
        grid_dimensions,
        recentre_voxels,
        density: vpb,
        band: LayerBand::FULL,
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
    let worker = spawn_geometry_worker(gpu.device.clone(), gpu.queue.clone(), COLOR_TARGET_FORMAT);

    let generation = 1u64;
    let request = large_request(generation);

    // (a) dispatch returns promptly — it is a single channel send, NOT the build.
    let started = Instant::now();
    worker.dispatch(request);
    let dispatch_elapsed = started.elapsed();
    let ceiling = common::DISPATCH_NONBLOCK_CEILING;
    assert!(
        dispatch_elapsed < ceiling,
        "dispatch blocked for {dispatch_elapsed:?} (ceiling {ceiling:?}) — \
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
    let result = common::poll_until_result(&worker, WORKER_TIMEOUT, "non-blocking dispatch");
    assert_eq!(
        result.generation, generation,
        "the arrived result must carry the dispatched generation"
    );
    // Sanity: the box actually meshed (the worker built real geometry, not an empty stub).
    let renderer = result
        .renderer
        .expect("a normal build returns a renderer (not a panicked None)");
    assert!(
        renderer.face_count() > 0,
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
/// path (not the pure tracker, which `workers::geometry`'s unit tests already cover).
#[test]
fn burst_supersede_accepts_only_newest_generation_under_real_threading() {
    let gpu = pollster::block_on(GpuContext::new(None));
    let worker = spawn_geometry_worker(gpu.device.clone(), gpu.queue.clone(), COLOR_TARGET_FORMAT);

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
                let renderer = result
                    .renderer
                    .as_ref()
                    .expect("a normal build returns a renderer (not a panicked None)");
                assert!(
                    renderer.face_count() > 0,
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
    let worker = spawn_geometry_worker(gpu.device.clone(), gpu.queue.clone(), COLOR_TARGET_FORMAT);

    // A request with NO covering chunks — the degenerate "empty scene / zero chunks" case.
    let empty = GeometryRebuildRequest {
        generation: 1,
        two_layer_chunks: Vec::new(),
        grid_dimensions: [0, 0, 0],
        recentre_voxels: RecentreVoxels::new([0, 0, 0]),
        density: 16,
        band: LayerBand::FULL,
    };
    worker.dispatch(empty);
    let result = common::poll_until_result(&worker, WORKER_TIMEOUT, "empty request");
    assert_eq!(result.generation, 1, "the empty build carries its generation");
    assert_eq!(
        result
            .renderer
            .expect("an empty scene still returns a renderer (not a panicked None)")
            .face_count(),
        0,
        "an empty scene meshes to zero faces (no geometry), but still returns a result"
    );

    // The worker survived the degenerate request — a normal follow-up still builds.
    let follow_up = large_request(2);
    worker.dispatch(follow_up);
    let result = common::poll_until_result(&worker, WORKER_TIMEOUT, "post-empty follow-up");
    assert_eq!(
        result.generation, 2,
        "the worker services a normal request after an empty one (it did not wedge)"
    );
    assert!(
        result
            .renderer
            .expect("the follow-up returns a renderer")
            .face_count()
            > 0,
        "the follow-up box meshed — the worker loop is still healthy"
    );
}

// ===========================================================================
// C1 — the outstanding-build interlock: no Frankenstein mesh
// ===========================================================================

/// Synchronously build a full renderer for a scene's covering set — the ground truth a
/// non-Frankenstein install must equal (a full rebuild of the LATEST scene).
fn sync_full_build(gpu: &GpuContext, request: &GeometryRebuildRequest) -> CuboidMeshRenderer {
    CuboidMeshRenderer::new_from_two_layer_chunks(
        &gpu.device,
        &gpu.queue,
        COLOR_TARGET_FORMAT,
        &request.two_layer_chunks,
        request.grid_dimensions,
        request.recentre_voxels,
        request.density,
    )
}

/// C1 regression (integration): reproduce the exact stale-patch sequence and assert the
/// finally-installed renderer equals a FULL rebuild of the LATEST scene — no Frankenstein.
///
/// The bug: a large edit dispatches an async wholesale build (gen 1, scene S1); the INSTALLED
/// renderer is still S0 while the worker builds S1. Before S1 arrives the user makes another
/// edit whose resolve returns `incremental_dirty_chunks = Some(..)`. The OLD code inline-
/// patched the STALE S0 renderer (keeping every non-dirty S0 chunk) and bumped the generation
/// → the gen-1 S1 result was discarded → chunks that differed S0→S1 but weren't in the new
/// dirty set stayed at S0 forever (old geometry + one fresh patch).
///
/// The fix: while an async build is OUTSTANDING, `route_geometry_rebuild` routes EVERY edit —
/// even an incremental one — to a fresh WHOLESALE-async dispatch from the CURRENT full
/// covering set. So the install is a full wholesale of the latest scene, never a patch of a
/// stale one. This test drives the SAME decision + the REAL worker + the REAL tracker the
/// shell uses (`WindowedState::rebuild_geometry` / `poll_geometry_worker`); only the window-
/// coupled swap is modelled by a local `installed` renderer (see the honesty note at the
/// bottom).
#[test]
fn c1_outstanding_edit_reroutes_wholesale_no_frankenstein() {
    let gpu = pollster::block_on(GpuContext::new(None));
    let worker = spawn_geometry_worker(gpu.device.clone(), gpu.queue.clone(), COLOR_TARGET_FORMAT);

    // The shell's state we model: the installed renderer (S0), the generation tracker, and
    // the C1 outstanding flag — exactly the fields `WindowedState` holds.
    let mut tracker = GenerationTracker::new();
    let mut async_outstanding = false;

    // Three DISTINCT large scenes with DIFFERENT geometry (different sizes → different face
    // sets), so "installed == latest" is a meaningful (not vacuous) assertion.
    let s0 = build_request(0, 24, 16); // installed baseline
    let s1 = build_request(0, 28, 16); // the first async edit dispatches this
    let s2 = build_request(0, 32, 16); // the SECOND edit's LATEST scene (the resident cache)
    for (name, req) in [("s0", &s0), ("s1", &s1), ("s2", &s2)] {
        assert!(
            req.two_layer_chunks.len() > ASYNC_REBUILD_CHUNK_THRESHOLD,
            "{name} must exceed the async threshold to be representative"
        );
    }
    let s2_truth = sync_full_build(&gpu, &s2).face_count();
    let s0_face = sync_full_build(&gpu, &s0).face_count();
    assert_ne!(
        s0_face, s2_truth,
        "the fixtures must differ so a Frankenstein (S0-derived) install would be DETECTABLE"
    );

    // The installed renderer starts as S0.
    let mut installed = sync_full_build(&gpu, &s0);
    assert_eq!(installed.face_count(), s0_face);

    // --- Edit 1: a large wholesale edit → S1 dispatched async (the #60 case). ---
    let route = route_geometry_rebuild(
        async_outstanding,
        EditShape::Wholesale {
            chunk_count: s1.two_layer_chunks.len(),
        },
        ASYNC_REBUILD_CHUNK_THRESHOLD,
    );
    assert_eq!(route, RebuildRoute::WholesaleAsync);
    let gen1 = tracker.next_generation();
    async_outstanding = true;
    let mut s1_dispatch = build_request(gen1, 28, 16);
    s1_dispatch.generation = gen1;
    worker.dispatch(s1_dispatch);

    // --- Edit 2: BEFORE S1's result is polled, a small (incremental-shaped) edit to the
    // LATEST scene S2. This is the exact C1 trigger. The resident cache is already S2. ---
    let route = route_geometry_rebuild(
        async_outstanding, // still true — S1 has NOT been installed
        EditShape::Incremental,
        ASYNC_REBUILD_CHUNK_THRESHOLD,
    );
    assert_eq!(
        route,
        RebuildRoute::WholesaleAsync,
        "C1 interlock: an incremental edit while a build is outstanding must re-dispatch \
         wholesale (from the CURRENT resident cache), NOT inline-patch the stale S0 renderer"
    );
    let gen2 = tracker.next_generation();
    // Still outstanding (a wholesale-async re-dispatch keeps it true).
    async_outstanding = true;
    // The re-dispatch sends the CURRENT FULL covering set — S2, the latest resident cache.
    let mut s2_dispatch = build_request(gen2, 32, 16);
    s2_dispatch.generation = gen2;
    worker.dispatch(s2_dispatch);

    // --- Drive the shell's poll+accept loop until the newest (gen2 = S2) result installs.
    // Along the way, a stale gen1 (S1) result — if the worker built it before draining — is
    // DISCARDED (the tracker rejects it), never installed over the fresher S2. ---
    let deadline = Instant::now() + WORKER_TIMEOUT;
    let mut installed_newest = false;
    while !installed_newest {
        if let Some(result) = worker.try_recv_result() {
            if tracker.accepts(result.generation) {
                // The shell's `poll_geometry_worker`: accept → install + clear outstanding.
                installed = result
                    .renderer
                    .expect("a normal build returns a renderer (not a panicked None)");
                async_outstanding = false;
                assert_eq!(result.generation, gen2, "only the newest (S2) is accepted");
                installed_newest = true;
            } else {
                // A superseded (gen1 / S1) result — must be discarded, never installed.
                assert_ne!(
                    result.generation, gen2,
                    "the newest must be accepted, not discarded"
                );
            }
        }
        if !installed_newest && Instant::now() >= deadline {
            panic!("C1: the newest (S2) result never arrived — the worker loop hung");
        }
        if !installed_newest {
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    // THE C1 ASSERTION: the finally-installed renderer is a FULL rebuild of the LATEST scene
    // (S2) — NOT a patch of the stale S0 (which would have a different face set). The old
    // inline-patch bug would have left an S0-derived Frankenstein here.
    assert!(!async_outstanding, "installing the newest clears the outstanding flag");
    assert_eq!(
        installed.face_count(),
        s2_truth,
        "C1: the installed renderer must equal a full rebuild of the LATEST scene (no \
         Frankenstein). Got {} faces, expected S2's {} (S0 was {})",
        installed.face_count(),
        s2_truth,
        s0_face
    );
}

// ===========================================================================
// C1 (brick analogue, ADR 0011 G3) — the brick field follows the same
// stale-while-rebuilding discipline: no incremental patch while async outstanding
// ===========================================================================

/// A wholesale brick field for a from-geometry box of `blocks³` at `vpb` — a distinct
/// scene per size so "resident == latest" is a meaningful (not vacuous) assertion.
fn brick_build(blocks: u32, vpb: u32) -> BrickFieldBuild {
    let two_layer_chunks = common::box_covering_chunks(blocks, vpb, MaterialChoice::Stone);
    build_brick_field(&two_layer_chunks, vpb)
}

/// C1 for the G3 brick sink: while an async WHOLESALE mesh build is OUTSTANDING, an
/// incremental-shaped edit must NOT incrementally PATCH the brick field — it rebuilds the
/// field WHOLESALE from the CURRENT scene, exactly as `route_geometry_rebuild` sends the
/// mesh to a fresh wholesale-async dispatch. This proves the brick sink can never install a
/// patch derived from a state the mesh path treats as stale (the C1 lesson, ported to the
/// atlas). The finally-resident field equals a from-scratch build of the LATEST scene (S2),
/// never S1/S0.
///
/// The decision is driven by the SAME pure `route_geometry_rebuild` + the SAME brick
/// `patch_in_place = matches!(route, InlineIncremental) && field.is_some()` predicate the
/// shell (`WindowedState::rebuild_geometry`) applies; only the window-coupled swap itself is
/// modelled locally (see the honesty note at the bottom of this file).
#[test]
fn c1_brick_field_rebuilds_wholesale_while_outstanding_no_stale_patch() {
    let vpb = 4u32;
    let s0 = brick_build(6, vpb);
    let s1 = brick_build(8, vpb);
    let s2 = brick_build(10, vpb);
    assert_ne!(
        s0.brick_records.len(),
        s2.brick_records.len(),
        "the fixtures must differ so a stale (S0/S1-derived) field would be DETECTABLE"
    );

    // The shell's persistent brick state (starts == S0) + the C1 outstanding flag.
    let (mut field, _) = IncrementalBrickField::from_wholesale(s0.clone());
    assert_eq!(field.to_build(), s0, "the field starts as the installed baseline S0");
    let mut async_outstanding = false;

    // The brick sink's decision, mirroring the shell exactly: patch in place ONLY when the
    // route is InlineIncremental and a field is resident.
    let brick_patches_in_place = |route: RebuildRoute| matches!(route, RebuildRoute::InlineIncremental);

    // --- Edit 1: a LARGE wholesale edit → S1, dispatched async (the #60 case). ---
    let route1 = route_geometry_rebuild(
        async_outstanding,
        EditShape::Wholesale {
            chunk_count: ASYNC_REBUILD_CHUNK_THRESHOLD + 1,
        },
        ASYNC_REBUILD_CHUNK_THRESHOLD,
    );
    assert_eq!(route1, RebuildRoute::WholesaleAsync);
    async_outstanding = true;
    assert!(
        !brick_patches_in_place(route1),
        "a wholesale edit rebuilds the brick field wholesale"
    );
    field = IncrementalBrickField::from_wholesale(s1.clone()).0;
    assert_eq!(
        field.to_build(),
        s1,
        "after edit 1 (wholesale) the resident field is a full build of S1"
    );

    // --- Edit 2: an INCREMENTAL-shaped edit → S2 BEFORE S1's async result installs. ---
    let route2 = route_geometry_rebuild(
        async_outstanding, // still true — S1 has NOT installed
        EditShape::Incremental,
        ASYNC_REBUILD_CHUNK_THRESHOLD,
    );
    assert_eq!(
        route2,
        RebuildRoute::WholesaleAsync,
        "C1: an incremental edit while a build is outstanding routes wholesale"
    );
    assert!(
        !brick_patches_in_place(route2),
        "C1: the brick field must NOT incrementally patch while an async build is outstanding \
         (that would install a patch of a state the mesh path treats as stale)"
    );
    // The interlock: rebuild the brick field WHOLESALE from the CURRENT scene (S2).
    field = IncrementalBrickField::from_wholesale(s2.clone()).0;

    // THE ASSERTION: the finally-resident brick field is a from-scratch build of the LATEST
    // scene (S2) — never a stale S0/S1. `to_build` round-trips a wholesale seed byte-exactly.
    let resident = field.to_build();
    assert_eq!(
        resident, s2,
        "C1: the resident brick field must equal a wholesale build of the LATEST scene (S2)"
    );
    assert_ne!(resident, s1, "not the mid-flight scene S1");
    assert_ne!(resident, s0, "not the stale baseline S0");
}

// HONESTY NOTE (C1 headless coverage): the ROUTING decision (the actual fix — the outstanding
// interlock that keeps an incremental edit from patching a stale renderer) is driven exactly
// as the shell drives it (`route_geometry_rebuild` + the real `GenerationTracker` + the real
// threaded `GeometryWorker`). What is NOT driven headlessly is the window-coupled SWAP itself
// (`WindowedState`'s `cuboid_mesh_renderer` field + `request_redraw`): that lives inside a
// live winit event loop with a surface, which these offscreen tests cannot spin up. So this
// test models the install with a local `installed` renderer and asserts the same invariant
// the shell's swap must preserve — installed face-set == a full rebuild of the LATEST scene.
// The pure `route_geometry_rebuild` unit tests in `workers/geometry.rs` cover the decision
// table exhaustively; this integration test proves the decision + worker + tracker compose
// into a non-Frankenstein install.
