//! Headless coverage of the `DisplayOrchestrator` state machine (map item 2).
//!
//! The orchestrator owns the two display pipelines (the cuboid fallback mesh + the ADR 0011
//! brick raymarch), both async rebuild workers, and all the per-edit display bookkeeping that
//! decides WHICH pipeline draws. That state machine used to live diffusely on the winit shell
//! and could only be reasoned about by hand (the last change to it shipped three transition
//! bugs a multi-agent review caught). Extracted window-free, it is now drivable on an offscreen
//! wgpu device — no window, no surface — so these tests assert the transitions directly through
//! the public API (`first_build` / `rebuild` / the polls / `ensure_display_mesh_current` and the
//! accessor-observable state: brick engagement, renderer presence, cuboid face count, poll
//! return values).
//!
//! Run: `cargo test --features gpu --test display_orchestrator`
#![cfg(feature = "gpu")]

use std::sync::Arc;
use std::time::{Duration, Instant};

use voxel_worker::{
    DisplayOrchestrator, DisplayRefreshContext, GpuContext, LayerBand, MaterialChoice,
    TwoLayerChunk, TwoLayerResidentCache, COLOR_TARGET_FORMAT,
};

mod common;

/// The bounded ceiling any poll-loop waits for a worker before failing LOUDLY — a hang is a
/// bug, never an acceptable unbounded wait (mirrors `geometry_worker_async`'s discipline).
const WORKER_TIMEOUT: Duration = Duration::from_secs(30);

/// A small brick-representable box: a single-material solid whose covering set is well BELOW
/// the async threshold, so `first_build` installs the brick display INLINE (no worker hop).
const SMALL_BLOCKS: u32 = 2;
/// A large brick-representable box: a single-material solid whose covering set EXCEEDS the async
/// threshold (24³ blocks at d16 → 216 covering chunks > 128), so the brick build goes ASYNC.
const LARGE_BLOCKS: u32 = 24;
const VPB: u32 = 16;

/// The owned scene fixtures a `first_build` / `rebuild` / refresh-context needs.
struct Fixture {
    scene: voxel_worker::Scene,
    chunks: Vec<([i32; 3], Arc<TwoLayerChunk>)>,
    region_dimensions: [u32; 3],
    recentre_voxels: [i64; 3],
}

impl Fixture {
    fn new(blocks: u32) -> Self {
        let scene = common::box_scene(blocks, VPB, MaterialChoice::Stone);
        let chunks = common::box_covering_chunks(blocks, VPB, MaterialChoice::Stone);
        let region_dimensions = scene.placed_region_dimensions(VPB);
        let recentre_voxels = scene.recentre_voxels_for_resolve(VPB);
        Self { scene, chunks, region_dimensions, recentre_voxels }
    }

    /// Build the orchestrator from this fixture's startup covering set.
    fn first_build(&self, gpu: &GpuContext, debug_face_orientation: bool) -> DisplayOrchestrator {
        DisplayOrchestrator::first_build(
            gpu.device.clone(),
            gpu.queue.clone(),
            COLOR_TARGET_FORMAT,
            &self.chunks,
            self.region_dimensions,
            self.recentre_voxels,
            VPB,
            debug_face_orientation,
        )
    }

    /// A refresh context borrowing the given cache (the shell hands one of these to the polls
    /// and `ensure_display_mesh_current`).
    fn context<'a>(
        &'a self,
        cache: &'a mut TwoLayerResidentCache,
        debug_face_orientation: bool,
    ) -> DisplayRefreshContext<'a> {
        DisplayRefreshContext {
            scene: &self.scene,
            two_layer_cache: cache,
            density: VPB,
            region_dimensions: self.region_dimensions,
            recentre_voxels: self.recentre_voxels,
            band: LayerBand::FULL,
            debug_face_orientation,
        }
    }
}

// ===========================================================================
// Case 1 — small representable box: brick engaged (inline), mesh skipped
// ===========================================================================

/// `first_build` on a small representable box engages the brick display INLINE and SKIPS the
/// fallback mesh (built empty). Observable: a live brick renderer is resident, `brick_display_
/// engaged(false)` is true, and the cuboid mesh has zero faces (the ~333ms build was skipped).
#[test]
fn first_build_small_engages_brick_and_skips_mesh() {
    let gpu = pollster::block_on(GpuContext::new(None));
    let fixture = Fixture::new(SMALL_BLOCKS);
    let orchestrator = fixture.first_build(&gpu, false);

    assert!(
        orchestrator.brick_raymarch_renderer().is_some(),
        "a small representable box installs the brick raymarch inline at startup"
    );
    assert!(
        orchestrator.brick_display_engaged(false),
        "the installed brick field is the live display (no debug-face mode)"
    );
    assert_eq!(
        orchestrator.cuboid_mesh_renderer().face_count(),
        0,
        "the fallback mesh is SKIPPED (built empty) while the brick display is engaged"
    );
    // Debug-face mode disengages the brick display even with a live field resident.
    assert!(
        !orchestrator.brick_display_engaged(true),
        "debug-face orientation disengages the brick display (it needs the mesh's face colours)"
    );
}

// ===========================================================================
// Case 2 — large box: async dispatch, engagement predicted, install on poll
// ===========================================================================

/// `first_build` on a large box (covering set > threshold) DISPATCHES the brick build async: no
/// renderer is resident yet and the mesh is empty (engagement is PREDICTED for the skip). Polling
/// the brick worker until the result lands installs the display field — `poll_brick_worker`
/// returns `needs_redraw = true`, a live renderer becomes resident, and the display engages.
#[test]
fn first_build_large_dispatches_async_then_installs_on_poll() {
    let gpu = pollster::block_on(GpuContext::new(None));
    let fixture = Fixture::new(LARGE_BLOCKS);
    assert!(
        fixture.chunks.len() > voxel_worker::ASYNC_REBUILD_CHUNK_THRESHOLD,
        "the large fixture must exceed the async threshold to be representative ({} chunks)",
        fixture.chunks.len()
    );
    let mut orchestrator = fixture.first_build(&gpu, false);

    // Before the async field lands: no renderer, not engaged, mesh empty (predicted skip).
    assert!(
        orchestrator.brick_raymarch_renderer().is_none(),
        "the async brick build has not landed — no renderer is resident yet"
    );
    assert!(!orchestrator.brick_display_engaged(false));
    assert_eq!(
        orchestrator.cuboid_mesh_renderer().face_count(),
        0,
        "the mesh is skipped on the PREDICTED engagement while the field builds async"
    );

    // Poll until the worker's result installs — the shell's per-frame brick poll.
    let mut cache = TwoLayerResidentCache::enabled();
    let deadline = Instant::now() + WORKER_TIMEOUT;
    let mut installed = false;
    while !installed {
        let context = fixture.context(&mut cache, false);
        if orchestrator.poll_brick_worker(context) {
            installed = true;
        } else if Instant::now() >= deadline {
            panic!("the async brick field never installed within {WORKER_TIMEOUT:?}");
        } else {
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    assert!(
        orchestrator.brick_raymarch_renderer().is_some(),
        "the landed async build installed a live brick renderer"
    );
    assert!(
        orchestrator.brick_display_engaged(false),
        "the installed field is now the live display"
    );
    // Nothing else is queued — a second poll is a no-op (no redraw).
    let context = fixture.context(&mut cache, false);
    assert!(
        !orchestrator.poll_brick_worker(context),
        "a single dispatch yields a single install"
    );
}

// ===========================================================================
// Case 4 — a rebuild while the brick is engaged keeps the mesh skipped
// ===========================================================================

/// A `rebuild` whose scene stays brick-representable re-engages the brick display, so the mesh
/// route is `Skip` — the fallback mesh is NOT built and stays empty (the per-edit ~333ms win).
#[test]
fn rebuild_keeps_brick_engaged_and_skips_mesh() {
    let gpu = pollster::block_on(GpuContext::new(None));
    let fixture = Fixture::new(SMALL_BLOCKS);
    let mut orchestrator = fixture.first_build(&gpu, false);
    assert!(orchestrator.brick_display_engaged(false));
    assert_eq!(orchestrator.cuboid_mesh_renderer().face_count(), 0);

    // A small wholesale rebuild of the same representable scene: brick re-installs inline, the
    // mesh route is Skip (brick engaged), so the cuboid mesh stays empty.
    orchestrator.rebuild(
        fixture.chunks.clone(),
        None,
        true,
        fixture.region_dimensions,
        fixture.recentre_voxels,
        VPB,
        LayerBand::FULL,
        false,
    );

    assert!(
        orchestrator.brick_display_engaged(false),
        "the rebuild re-engaged the brick display"
    );
    assert_eq!(
        orchestrator.cuboid_mesh_renderer().face_count(),
        0,
        "the fallback mesh stays SKIPPED (empty) across the rebuild while the brick is engaged"
    );
}

// ===========================================================================
// Case 5 — ensure_display_mesh_current: no-op while engaged, builds on
// debug-face; waits while a brick build is outstanding
// ===========================================================================

/// `ensure_display_mesh_current` is a NO-OP while the brick display is engaged (the stale mesh
/// stays hidden + empty). Turning on debug-face mode disengages the brick, so — nothing
/// outstanding — the seam rebuilds the stale fallback mesh INLINE (now non-empty).
#[test]
fn ensure_display_mesh_current_noop_while_engaged_then_builds_on_debug_face() {
    let gpu = pollster::block_on(GpuContext::new(None));
    let fixture = Fixture::new(SMALL_BLOCKS);
    let mut orchestrator = fixture.first_build(&gpu, false);
    assert_eq!(orchestrator.cuboid_mesh_renderer().face_count(), 0);

    let mut cache = TwoLayerResidentCache::enabled();

    // Engaged + no debug-face → no-op: the mesh stays skipped (empty).
    let context = fixture.context(&mut cache, false);
    orchestrator.ensure_display_mesh_current(context);
    assert_eq!(
        orchestrator.cuboid_mesh_renderer().face_count(),
        0,
        "ensure is a no-op while the brick display is engaged"
    );

    // Debug-face on → brick disengaged, nothing outstanding → rebuild the stale mesh inline.
    let context = fixture.context(&mut cache, true);
    orchestrator.ensure_display_mesh_current(context);
    assert!(
        orchestrator.cuboid_mesh_renderer().face_count() > 0,
        "debug-face disengages the brick, so the stale fallback mesh is rebuilt (non-empty)"
    );
    assert!(
        !orchestrator.brick_display_engaged(true),
        "the display is the mesh now (debug-face mode)"
    );
}

/// While a brick build is OUTSTANDING (a large first-build, not yet landed) and debug-face is
/// off, `ensure_display_mesh_current` WAITS — it does not synchronously build the fallback mesh
/// (that would be the multi-second frame-one freeze the async pipeline exists to remove). The
/// mesh stays empty until the brick arrival decides the display.
#[test]
fn ensure_display_mesh_current_waits_while_brick_outstanding() {
    let gpu = pollster::block_on(GpuContext::new(None));
    let fixture = Fixture::new(LARGE_BLOCKS);
    let mut orchestrator = fixture.first_build(&gpu, false);
    // Large first-build dispatched the brick async — no renderer yet, mesh predicted-skipped.
    assert!(orchestrator.brick_raymarch_renderer().is_none());
    assert_eq!(orchestrator.cuboid_mesh_renderer().face_count(), 0);

    let mut cache = TwoLayerResidentCache::enabled();
    let context = fixture.context(&mut cache, false);
    orchestrator.ensure_display_mesh_current(context);
    assert_eq!(
        orchestrator.cuboid_mesh_renderer().face_count(),
        0,
        "ensure waits for the in-flight brick build rather than synchronously meshing"
    );
}

// ===========================================================================
// poll_geometry_worker — the async mesh install returns needs_redraw
// ===========================================================================

/// When the fallback mesh is the display but stale AND large, `ensure_display_mesh_current`
/// dispatches its rebuild to the geometry worker; `poll_geometry_worker` installs the result
/// (returning `needs_redraw = true`) and the cuboid mesh becomes non-empty. This drives the
/// window-free poll contract the shell relies on (the shell requests the redraw on `true`).
#[test]
fn poll_geometry_worker_installs_async_fallback_mesh() {
    let gpu = pollster::block_on(GpuContext::new(None));
    let fixture = Fixture::new(LARGE_BLOCKS);
    let mut orchestrator = fixture.first_build(&gpu, false);

    // Force the fallback mesh to be the display (debug-face) while it is stale + large: the seam
    // dispatches the rebuild to the geometry worker (nothing built inline this frame).
    let mut cache = TwoLayerResidentCache::enabled();
    let context = fixture.context(&mut cache, true);
    orchestrator.ensure_display_mesh_current(context);
    assert_eq!(
        orchestrator.cuboid_mesh_renderer().face_count(),
        0,
        "a large stale fallback mesh dispatches async — nothing is built inline this frame"
    );

    // Poll the geometry worker until the async mesh installs.
    let deadline = Instant::now() + WORKER_TIMEOUT;
    let mut installed = false;
    while !installed {
        if orchestrator.poll_geometry_worker() {
            installed = true;
        } else if Instant::now() >= deadline {
            panic!("the async fallback mesh never installed within {WORKER_TIMEOUT:?}");
        } else {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    assert!(
        orchestrator.cuboid_mesh_renderer().face_count() > 0,
        "the installed async fallback mesh is non-empty (real geometry)"
    );
}
