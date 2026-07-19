//! The display-state machine — the window-free owner of the two display pipelines
//! (the cuboid fallback mesh and the brick raymarch) and every piece of display state
//! that decides which of them draws each frame.
//!
//! This is the *actor* on the pure decisions in [`super::routing`]: it holds the two
//! renderers, the two async workers, their generation trackers + outstanding flags, the
//! `mesh_stale` / brick-handover bookkeeping, and the install seams that keep those in
//! lock-step. The winit shell keeps input, surface, egui, and camera, and reaches this
//! machine only at its (few) integration points — the first build, an edit's rebuild, the
//! per-frame polls + display-mesh refresh, and the accessors the draw path binds against.
//!
//! Because it is constructible without a window (it takes cloned wgpu handles, not a
//! surface), the state machine — not just its pure fragments — is unit-testable. See
//! `docs/architecture/03-display.md` for the display model this realises and
//! `docs/architecture/04-work.md` for the stale-while-rebuilding worker discipline.

use std::sync::Arc;

// Intra-crate: the pure per-edit routing policy and the async workers this state machine drives.
use crate::engagement::routing::{
    route_brick_rebuild, route_mesh_build, BrickRebuildAction, EditShape, GenerationTracker,
    MeshBuildRoute, RebuildRoute, ASYNC_REBUILD_CHUNK_THRESHOLD,
};
use crate::workers::brick::{
    spawn_brick_worker, BrickRebuildOutcome, BrickRebuildRequest, BrickWorker,
};
use crate::workers::geometry::{spawn_geometry_worker, GeometryRebuildRequest, GeometryWorker};
// Down-crate: the display GPU sinks this orchestrator owns + drives, and the evaluation /
// document / voxel_core values that cross the worker channels.
use display::brick::{
    build_brick_field_with_tiles, BrickFieldUpdate, BrickRaymarchRenderer, IncrementalBrickField,
    SculptedAtlasPayload,
};
use display::mesh::CuboidMeshRenderer;
use display::renderer::{LayerBand, RegionClip};
use document::scene::Scene;
use evaluation::two_layer_store::{TwoLayerChunk, TwoLayerResidentCache};
use voxel_core::voxel::RecentreVoxels;
// Consumed by the GPU display-install paths (alongside the CPU brick mirror the
// orchestrator maintains for every chunkable scene).
use crate::engagement::routing::{
    brick_display_handover, brick_patch_in_place, BrickDisplayHandover,
};
use display::brick::{pack_gpu_records, ClipmapPyramid};

/// The per-refresh context the shell hands the orchestrator whenever a display-mesh
/// rebuild might be needed off the main edit path (the per-frame polls and the
/// `ensure_display_mesh_current` seam). It bundles the borrows the orchestrator needs to
/// re-mesh the stale fallback from the RESIDENT two-layer cache (scene unchanged — an
/// O(chunks) `Arc` handout, not a from-scratch re-resolve) without owning any of the
/// shell's scene / panel / camera state. See `docs/architecture/03-display.md`.
pub struct DisplayRefreshContext<'a> {
    /// The current scene (for the resident-cache chunk handout — the scene is unchanged
    /// on these paths, so the cache returns the last resolve's set as `Arc` bumps).
    pub scene: &'a Scene,
    /// The resident two-layer cache (`= &mut app_core.two_layer_cache`) — the warm store
    /// the stale-mesh rebuild draws its covering chunks from.
    pub two_layer_cache: &'a mut TwoLayerResidentCache,
    /// The active density (voxels per block).
    pub density: u32,
    /// The last rebuild's region dimensions (voxels) — the mesh's frame parameters.
    pub region_dimensions: [u32; 3],
    /// The last rebuild's composite recentre (floating origin, voxels), carried as the
    /// frame value [`RecentreVoxels`].
    pub recentre_voxels: RecentreVoxels,
    /// The effective layer-clip band the render path will apply this frame (so a stale-mesh
    /// rebuild builds already clipped to it — no swap-frame re-mesh).
    pub band: LayerBand,
    /// The onion-fog region the band is confined to this frame (ADR 0018 Decision 5), or
    /// `None` for a scene-wide band / no clip.
    pub region: Option<RegionClip>,
    /// Whether debug-face orientation mode is active (the mesh-only display flag that drops
    /// brick engagement).
    pub debug_face_orientation: bool,
}

/// The display-state machine extracted from the winit shell. Owns both display renderers,
/// both async rebuild workers, and all the per-edit display bookkeeping that decides which
/// pipeline draws — constructible without a window (see the module doc).
pub struct DisplayOrchestrator {
    /// A clone of the wgpu device (wgpu 29 `Device` is `Send + Sync + Clone`, `Arc`-backed),
    /// so the orchestrator builds/patches its GPU renderers without borrowing the shell's
    /// `GpuContext`. See `docs/architecture/03-display.md`.
    device: wgpu::Device,
    /// A clone of the wgpu queue (same `Arc`-backed clone contract as `device`).
    queue: wgpu::Queue,
    /// The colour target format the renderers build against (the shared sRGB surface format,
    /// so the live window and the headless capture stay pixel-identical).
    color_format: wgpu::TextureFormat,
    /// The cuboid mesh renderer — the sole voxel render path (part of #20; the legacy
    /// instanced mesher was removed). Rebuilt from the resolve cache's per-chunk
    /// accessor on every geometry change in `rebuild_geometry`.
    cuboid_mesh_renderer: CuboidMeshRenderer,
    /// Brick-display perf follow-up to epic #64: whether `cuboid_mesh_renderer` currently
    /// holds a STALE (skipped / empty) mesh because the ADR 0011 brick raymarch is the live
    /// display and the fallback mesh was not worth the ~333ms serial build. While `true` the
    /// mesh must NOT be drawn (it isn't — the brick pass replaces it) and must NOT be
    /// inline-patched by an incremental edit (its buffers don't reflect the latest resolve);
    /// the next edit that needs the mesh — or [`Self::ensure_display_mesh_current`] on a
    /// debug-face / loaded-material transition — rebuilds it WHOLESALE. Composed into the C1
    /// interlock via [`route_mesh_build`].
    mesh_stale: bool,
    /// F1 (brick-display perf follow-up to epic #64): a DEFERRED brick-display handover is
    /// pending. When an edit drops brick representability while the replacement cuboid mesh
    /// builds ASYNC, the (now stale) brick field is KEPT drawing so the model never blanks for
    /// the seconds the worker takes — `clear_brick_field` is deferred to the mesh-install seam
    /// ([`Self::complete_brick_display_handover`], run from [`Self::finish_mesh_install`]).
    /// `true` only while such a handover is outstanding.
    brick_display_pending_clear: bool,
    /// ADR 0011 G1: the brick raymarch display sink. Created on first engagement
    /// (any non-empty chunkable scene) and kept — per-edit
    /// work is `install_brick_field` (records + atlas swap, no pipeline rebuild).
    /// When it holds a field and no mesh-only mode is active (debug-faces, a loaded
    /// VS material), the frame's voxel model draws from the brick atlas INSTEAD of
    /// the cuboid mesh; the mesh keeps rebuilding as the fallback + A/B reference
    /// (ADR 0011 Decision 6). `None` until the first engagement.
    brick_raymarch_renderer: Option<BrickRaymarchRenderer>,
    /// ADR 0011 G3: the PERSISTENT incremental brick field mirroring the boundary set —
    /// the CPU truth an incremental edit patches (dirty chunks re-evaluated, only their
    /// slots written) instead of rebuilding the whole field. `Some` for any chunkable
    /// scene; reset from a wholesale `build_brick_field` on a wholesale edit,
    /// patched in place on an incremental edit, and dropped when the scene leaves the
    /// gate / empties. `to_build()` always equals the resident atlas (ADR 0011 G3 gate).
    incremental_brick_field: Option<IncrementalBrickField>,
    /// Issue #60 (ADR 0003 §7): the background geometry-rebuild worker. A WHOLESALE
    /// rebuild whose covering-chunk count exceeds [`ASYNC_REBUILD_CHUNK_THRESHOLD`] —
    /// the ~3s large-object build — is dispatched here (cloned `device`/`queue`) instead
    /// of built inline, so the UI never freezes. The main thread keeps rendering the
    /// CURRENT `cuboid_mesh_renderer` (stale-while-rebuilding) until the worker's
    /// freshly-built renderer arrives, then swaps it in. Small / incremental edits stay
    /// synchronous.
    geometry_worker: GeometryWorker,
    /// Issue #60: the monotonic generation bookkeeping behind supersede. Each async
    /// dispatch stamps a fresh generation; a received result is swapped in only when its
    /// generation is still the newest dispatched (an edit mid-build supersedes the older
    /// in-flight build, whose result is then discarded — see [`GenerationTracker`]).
    geometry_generation: GenerationTracker,
    /// Issue #60 C1: whether an async WHOLESALE build is OUTSTANDING — dispatched but not
    /// yet accepted/installed. While `true` the currently-installed `cuboid_mesh_renderer`
    /// does NOT reflect the latest resolve (it is still S0 while the worker builds S1), so an
    /// incremental edit must NOT inline-patch it (that strands every chunk that differs
    /// S0→S1 but isn't in the new dirty set — the Frankenstein mesh). The rebuild is routed
    /// to a fresh wholesale-async dispatch instead (see [`route_geometry_rebuild`]). Cleared
    /// when `poll_geometry_worker` accepts + installs a result.
    geometry_async_outstanding: bool,
    /// The async wholesale brick-pipeline worker (perf follow-up to epic #64, issue #60
    /// pattern): a WHOLESALE brick rebuild whose covering-chunk count exceeds
    /// [`ASYNC_REBUILD_CHUNK_THRESHOLD`] — the ~2s record-build + pyramid + classify on a
    /// giant scene — is dispatched here (pure CPU, no GPU handles) instead of built inline.
    /// The main thread keeps drawing the CURRENT display (the stale brick field, or the
    /// mesh) until the artifacts arrive (`poll_brick_worker`), then installs them — only
    /// the `install_brick_field` upload stays on the main thread.
    brick_worker: BrickWorker,
    /// Supersede bookkeeping for `brick_worker` — the same [`GenerationTracker`] contract
    /// as the geometry and diameter workers: a result is accepted only when its generation
    /// is still the newest dispatched.
    brick_generation: GenerationTracker,
    /// Whether an async wholesale BRICK build is OUTSTANDING (dispatched, not yet
    /// accepted/installed). While `true` the resident `incremental_brick_field` mirror and
    /// the renderer's live field are STALE (S0 while the worker builds S1), so an
    /// incremental edit must NOT patch them — [`route_brick_rebuild`] sends every edit to
    /// a fresh wholesale dispatch instead (the brick analogue of the C1 interlock).
    /// Cleared when `poll_brick_worker` accepts a result, or when an inline wholesale
    /// build supersedes the in-flight one.
    brick_async_outstanding: bool,
}

impl DisplayOrchestrator {
    /// The STARTUP first-build (was the display block of `WindowedState::new`): decide brick
    /// engagement, spawn both workers, build the (possibly skipped-empty) cuboid mesh, and
    /// seed all display state. Takes cloned wgpu handles + the startup covering set + frame
    /// params — no window, no scene ownership. See `docs/architecture/03-display.md`.
    ///
    /// Perf follow-up to epic #64: the brick decision is made BEFORE the fallback cuboid mesh
    /// so that, when the brick display engages, the ~333ms serial mesh build (and its memory)
    /// is SKIPPED at startup — the persisted 8000×800×800 scene installs the brick sink and
    /// never meshes.
    #[allow(clippy::too_many_arguments)] // the startup frame parameters are irreducibly plural.
    pub fn first_build(
        device: wgpu::Device,
        queue: wgpu::Queue,
        color_format: wgpu::TextureFormat,
        two_layer_chunks: &[([i32; 3], Arc<TwoLayerChunk>)],
        region_dimensions: [u32; 3],
        recentre_voxels: RecentreVoxels,
        density: u32,
        debug_face_orientation: bool,
    ) -> Self {
        // Engage the brick raymarch from the FIRST frame for any non-empty chunkable scene.
        // The representability gate is deleted (material atlas): per-record
        // ids + overlay carry per-block detail and a mixed brick's per-voxel cell keys ride in
        // the side atlas, so mixed-material and overlay-disagreeing scenes engage too. Later
        // edits refresh it in `rebuild_geometry`.
        let mut brick_raymarch_renderer: Option<BrickRaymarchRenderer> = None;
        // ADR 0011 G3: the persistent incremental field seeded from the startup wholesale
        // build (kept in lock-step with `brick_raymarch_renderer`).
        let mut incremental_brick_field: Option<IncrementalBrickField> = None;
        // Perf follow-up to epic #64 (issue #60 pattern): the async brick-pipeline worker,
        // spawned BEFORE the startup brick decision so a giant persisted scene dispatches
        // its first wholesale build here instead of freezing the pre-first-frame startup
        // for the ~2s record build + pyramid + classify.
        let brick_worker = spawn_brick_worker();
        let mut brick_generation = GenerationTracker::new();
        let mut brick_async_outstanding = false;
        // The startup covering set is non-empty iff the scene is chunkable with geometry
        // (`resident_two_layer_chunks` yields nothing for a non-chunkable / VoxelBody-only scene),
        // so `!is_empty()` matches the old `has_chunkable_extent && !is_empty()` gate.
        if !two_layer_chunks.is_empty() {
            if two_layer_chunks.len() > ASYNC_REBUILD_CHUNK_THRESHOLD {
                // A giant persisted scene: dispatch the wholesale brick build ASYNC so the
                // window shows immediately — the model pops in when the field lands
                // (`poll_brick_worker`). The mesh-skip decision below PREDICTS engagement
                // from this dispatch; every non-empty scene engages the brick path on a gpu
                // build now (the representability gate is deleted), so the prediction only ever
                // corrects on an `Empty` arrival (which hands the display to the mesh).
                Self::dispatch_wholesale_brick_rebuild(
                    &brick_worker,
                    &mut brick_generation,
                    &mut brick_async_outstanding,
                    two_layer_chunks.to_vec(),
                    density,
                    recentre_voxels,
                );
                println!(
                    "brick raymarch: startup field building async ({} covering chunks)",
                    two_layer_chunks.len()
                );
            } else {
                // Small startup scene: build + install inline (no pop-in), routed through the
                // SAME shared entry the async worker uses (finding #6) — `build_brick_rebuild`
                // does the record build → pyramid + GPU record pack + cell-key pack in one place,
                // so the startup and worker paths can never drift step-for-step. Only the GPU
                // upload (which the off-thread worker can't do) stays here. An empty small scene
                // installs nothing and lets the cuboid mesh take over, exactly as before.
                let startup_request = BrickRebuildRequest {
                    // Not dispatched to the worker and never superseded, so the generation and
                    // recentre the request carries are unread by `build_brick_rebuild` (the
                    // install below uses the local `recentre_voxels`). A zero generation keeps
                    // `brick_generation` untouched — the inline path never consumed one.
                    generation: 0,
                    two_layer_chunks: two_layer_chunks.to_vec(),
                    density,
                    recentre_voxels,
                    build_display_artifacts: true,
                };
                if let BrickRebuildOutcome::Display(install) =
                    crate::workers::brick::build_brick_rebuild(&startup_request)
                {
                    let crate::workers::brick::BrickDisplayInstall {
                        atlas,
                        cell_key_atlas,
                        gpu_records,
                        pyramid,
                        mirror,
                    } = *install;
                    let mut renderer = BrickRaymarchRenderer::new(&device, &queue, color_format);
                    // The mirror is the single owner (item 9): install reads its records + the
                    // upload payload the build moved out alongside it. The cell-key side atlas
                    // rides too, so a mixed scene shades per-voxel (empty for a single-material one).
                    renderer.install_brick_field_with_cell_keys(
                        &device,
                        &queue,
                        mirror.records(),
                        &atlas,
                        &cell_key_atlas,
                        &gpu_records,
                        &pyramid,
                        recentre_voxels,
                    );
                    println!(
                        "brick raymarch: startup field installed ({} records, {} sculpted)",
                        mirror.records().len(),
                        mirror.sculpted_brick_count(),
                    );
                    incremental_brick_field = Some(mirror);
                    brick_raymarch_renderer = Some(renderer);
                }
            }
        }
        // ADR 0010 E5: the cuboid mesh is the fallback voxel render path AND it meshes THROUGH
        // the two-layer store (coarse one-box + microblock cuboids + seam-flag culling) — the
        // SAME path `rebuild_geometry` takes on every later edit, so the startup frame it draws
        // is pixel-identical to the two-layer runtime path. `build_covering_chunks` returns
        // empty for a VoxelBody-only scene (the windowed startup default is always chunkable).
        //
        // Brick-display perf follow-up to epic #64: when the brick raymarch engaged above and no
        // mesh-only mode is active (a config may persist `debug_face_orientation`; a material is
        // never loaded at startup), the mesh is NOT drawn — so SKIP its build entirely and mark
        // it stale. `ensure_display_mesh_current` (or an edit that drops brick engagement) builds
        // the real mesh the moment it is next needed. The empty renderer still carries the
        // pipeline / material bind-group layout / sampler the loaded-material path binds against.
        // Shared engagement predicate (the ONE brick-display gate — same terms as the rebuild,
        // `ensure_display_mesh_current`, and the per-frame draw gate).
        let brick_engaged_at_startup = Self::brick_display_engaged_predicate(
            // A field installed inline, OR one building async (stale-while-rebuilding:
            // the dispatch above predicts engagement — corrected on arrival only if the
            // scene turns out empty).
            brick_raymarch_renderer.is_some() || brick_async_outstanding,
            debug_face_orientation,
        );
        let mesh_stale = brick_engaged_at_startup;
        let cuboid_mesh_renderer = CuboidMeshRenderer::new_from_two_layer_chunks(
            &device,
            &queue,
            color_format,
            if brick_engaged_at_startup {
                // Cheap empty renderer: no chunk meshing, just the shared GPU pipeline objects.
                &[]
            } else {
                two_layer_chunks
            },
            region_dimensions,
            recentre_voxels,
            density,
        );

        // Issue #60 (ADR 0003 §7): spawn the background geometry-rebuild worker with
        // cloned GPU handles (wgpu 29 `Device`/`Queue` are `Send + Sync + Clone`, so the
        // worker builds the mesh's GPU buffers off the main thread). A large wholesale
        // rebuild dispatches here; the shell keeps rendering the current mesh until the
        // worker's result arrives, then swaps it in.
        let geometry_worker = spawn_geometry_worker(device.clone(), queue.clone(), color_format);
        let geometry_generation = GenerationTracker::new();

        Self {
            device,
            queue,
            color_format,
            cuboid_mesh_renderer,
            mesh_stale,
            brick_display_pending_clear: false,
            brick_raymarch_renderer,
            incremental_brick_field,
            geometry_worker,
            geometry_generation,
            geometry_async_outstanding: false,
            brick_worker,
            brick_generation,
            brick_async_outstanding,
        }
    }

    /// Is the ADR 0011 brick raymarch the live voxel display (so the fallback cuboid mesh is
    /// NOT drawn)? [pure] The SINGLE engagement predicate behind every gate — startup, the
    /// rebuild skip, [`Self::ensure_display_mesh_current`], and the per-frame draw gate — so
    /// they can never drift term-for-term. Engaged iff a live brick field is resident AND
    /// debug-face orientation is off.
    ///
    /// ADR 0011 G2 — a loaded VS material NO LONGER disengages the brick display: the block
    /// texture is a pure function of the lattice (the owner's determinism rule), so the
    /// raymarch shades solid hits per-face from the block's 6-layer D2Array by the SAME rule
    /// the merged mesh uses (`face_layer` + per-face UV + `fract`), with zero per-brick data.
    /// With the representability gate deleted (material atlas), the mesh is the fallback ONLY for
    /// debug-face mode (it needs the mesh's per-vertex face colours) and for machines/scenes with
    /// no live brick field — every non-empty scene otherwise engages the brick display.
    fn brick_display_engaged_predicate(
        has_live_brick_field: bool,
        debug_face_orientation: bool,
    ) -> bool {
        has_live_brick_field && !debug_face_orientation
    }

    /// The per-frame brick-display engagement gate against the LIVE renderer/panel state — the
    /// shared read used by BOTH the draw path and [`Self::ensure_display_mesh_current`], which
    /// MUST agree term-for-term. Delegates to [`Self::brick_display_engaged_predicate`]. `false`
    /// whenever no brick renderer/field is live.
    pub fn brick_display_engaged(&self, debug_face_orientation: bool) -> bool {
        Self::brick_display_engaged_predicate(
            self.brick_raymarch_renderer
                .as_ref()
                .is_some_and(|renderer| renderer.has_brick_field()),
            debug_face_orientation,
        )
    }

    /// F1: complete a DEFERRED brick-display handover. When an edit dropped brick
    /// representability while a large replacement mesh built ASYNC, the stale brick field was
    /// kept drawing (`brick_display_pending_clear`) so the model never blanked. Once the fresh
    /// mesh installs (this seam, via [`Self::finish_mesh_install`]) the stale field is cleared
    /// so the mesh takes the frame. A no-op when no handover is pending.
    fn complete_brick_display_handover(&mut self) {
        if self.brick_display_pending_clear {
            if let Some(renderer) = &mut self.brick_raymarch_renderer {
                renderer.clear_brick_field();
            }
            self.brick_display_pending_clear = false;
        }
    }

    /// Apply a pure [`BrickDisplayHandover`] decision (F1) to the resident brick display state —
    /// the ONE mutation the handover sites share: the rebuild's F1 reconcile, and the brick
    /// worker's `Empty` arrival. `KeepAsDisplay`/`ClearNow` cancel any
    /// pending deferred clear (and `ClearNow` also drops the live field this frame); `DeferUntilInstall`
    /// arms the deferred clear so the stale brick keeps drawing until the replacement mesh lands
    /// ([`Self::complete_brick_display_handover`]). `BrickDisplayHandover` and the field clear are
    /// the brick display's concern; a no-op when no field is live.
    fn apply_brick_display_handover(&mut self, decision: BrickDisplayHandover) {
        match decision {
            BrickDisplayHandover::KeepAsDisplay => self.brick_display_pending_clear = false,
            BrickDisplayHandover::ClearNow => {
                if let Some(renderer) = &mut self.brick_raymarch_renderer {
                    renderer.clear_brick_field();
                }
                self.brick_display_pending_clear = false;
            }
            BrickDisplayHandover::DeferUntilInstall => self.brick_display_pending_clear = true,
        }
    }

    /// The mesh-install seam (issue #60 supersede + brick-display perf follow-up). EVERY path
    /// that makes a freshly built/patched cuboid mesh the CURRENT display funnels through here:
    /// bump the generation (so a superseded in-flight worker result is discarded on arrival),
    /// drop the async-outstanding flag, clear `mesh_stale` (the mesh now reflects the latest
    /// resolve), and complete any deferred brick-display handover (F1). This is the SOLE writer
    /// that clears `mesh_stale`; the only other `mesh_stale` writer is the Skip arm's set-true.
    fn finish_mesh_install(&mut self) {
        self.geometry_generation.next_generation();
        self.geometry_async_outstanding = false;
        self.mesh_stale = false;
        self.complete_brick_display_handover();
    }

    /// The brick-install seam — the brick analogue of [`Self::finish_mesh_install`].
    /// EVERY path that makes the resident brick state (mirror + field) reflect the
    /// latest resolve funnels through here: bump the generation so a superseded
    /// in-flight worker result is discarded on arrival (this is what makes the
    /// inline-wholesale-while-outstanding route sound — see `route_brick_rebuild`'s
    /// divergence note), and drop the outstanding flag so the patch fast-path resumes.
    fn finish_brick_install(&mut self) {
        self.brick_generation.next_generation();
        self.brick_async_outstanding = false;
    }

    /// Dispatch a WHOLESALE brick rebuild to the async worker: mint the next generation,
    /// mark the build outstanding (the interlock — every edit routes wholesale until the
    /// result installs), and send the `Arc`-shared covering set. An associated function over
    /// the individual fields (not `&mut self`) so BOTH dispatch sites within the orchestrator
    /// share it: [`Self::first_build`], where the fields are still locals ahead of `Self`
    /// construction, and [`Self::rebuild`]'s WholesaleAsync arm.
    fn dispatch_wholesale_brick_rebuild(
        brick_worker: &BrickWorker,
        brick_generation: &mut GenerationTracker,
        brick_async_outstanding: &mut bool,
        two_layer_chunks: Vec<([i32; 3], Arc<TwoLayerChunk>)>,
        density: u32,
        recentre_voxels: RecentreVoxels,
    ) {
        let generation = brick_generation.next_generation();
        *brick_async_outstanding = true;
        brick_worker.dispatch(BrickRebuildRequest {
            generation,
            two_layer_chunks,
            density,
            recentre_voxels,
            // The display artifacts (classify + pyramid + GPU record pack) are consumed
            // by the raymarch display, matching the synchronous path.
            build_display_artifacts: true,
        });
    }

    /// Dispatch a WHOLESALE cuboid-mesh rebuild to the async geometry worker: mint the next
    /// generation, mark the build OUTSTANDING (the C1 interlock — every edit routes wholesale
    /// until the result installs), and send the owned covering set + frame params. The mesh
    /// analogue of [`Self::dispatch_wholesale_brick_rebuild`], shared by [`Self::rebuild`]'s
    /// WholesaleAsync arm and [`Self::rebuild_stale_display_mesh`]. The generation is minted
    /// BEFORE the outstanding flag is set BEFORE the dispatch — the C1 interlock depends on
    /// that exact ordering.
    fn dispatch_wholesale_mesh_rebuild(
        &mut self,
        two_layer_chunks: Vec<([i32; 3], Arc<TwoLayerChunk>)>,
        grid_dimensions: [u32; 3],
        recentre_voxels: RecentreVoxels,
        density: u32,
        band: LayerBand,
        region: Option<RegionClip>,
    ) {
        let generation = self.geometry_generation.next_generation();
        self.geometry_async_outstanding = true;
        self.geometry_worker.dispatch(GeometryRebuildRequest {
            generation,
            two_layer_chunks,
            grid_dimensions,
            recentre_voxels,
            density,
            band,
            region,
        });
    }

    /// Refresh the display artifacts for a rebuild the shell has already resolved: the brick
    /// sink (mirror + display field) and the fallback cuboid mesh, plus the F1 brick-display
    /// handover reconcile. The shell has already run `AppCore::rebuild`, captured the frame
    /// params, and computed the effective `band`; it hands the owned covering set + the edit's
    /// dirty-chunk hint here. See `docs/architecture/03-display.md` for the two-pipeline model
    /// and `docs/architecture/04-work.md` for the interlock this preserves.
    #[allow(clippy::too_many_arguments)] // one rebuild's frame parameters are irreducibly plural.
    pub fn rebuild(
        &mut self,
        two_layer_chunks: Vec<([i32; 3], Arc<TwoLayerChunk>)>,
        incremental_dirty_chunks: Option<Vec<[i32; 3]>>,
        chunkable: bool,
        grid_dimensions: [u32; 3],
        recentre_voxels: RecentreVoxels,
        density: u32,
        band: LayerBand,
        region: Option<RegionClip>,
        debug_face_orientation: bool,
    ) {
        // The brick mirror (`incremental_brick_field`, plain CPU) is maintained for ANY
        // chunkable scene. The DISPLAY raymarch keeps its stricter conditions inside the block.
        // ADR 0011 G2 — a loaded VS material no longer gates the display (it textures the
        // raymarch per-face); ADR 0012 retired the onion fog occupancy consumer of the mirror.
        let brick_gate = chunkable;

        // Issue #60 C1: classify the edit ONCE (shared by the brick sink and the mesh path).
        // While an async wholesale build is OUTSTANDING every edit routes to a fresh
        // wholesale-async dispatch (never inline-patch a stale artifact); only with nothing
        // outstanding do the inline fast-paths resume.
        let edit_shape = match &incremental_dirty_chunks {
            Some(_) => EditShape::Incremental,
            None => EditShape::Wholesale {
                chunk_count: two_layer_chunks.len(),
            },
        };
        // The BRICK pipeline's patch/wholesale-inline/wholesale-async decision (below). It
        // carries its OWN outstanding interlock (`brick_async_outstanding`), independent of
        // the mesh's: the brick mirror + field are patched inline, so only an in-flight
        // BRICK build makes them stale — a mesh build in flight does not (previously the
        // mesh's flag forced a full ~2s synchronous brick rebuild on every edit made while
        // a large mesh built; the split removes that hitch). The mesh's own route
        // (`route_mesh_build`, after the brick block) folds in mesh staleness + the
        // brick-display-engaged skip as before.
        let brick_route = route_brick_rebuild(
            self.brick_async_outstanding,
            matches!(edit_shape, EditShape::Incremental),
            self.incremental_brick_field.is_some(),
            two_layer_chunks.len(),
            ASYNC_REBUILD_CHUNK_THRESHOLD,
        );

        // ADR 0011 G1/G3/G5: refresh the brick field from THIS rebuild's resident chunk set
        // (the same boundary set the mesher consumes), before the mesh route can move
        // `two_layer_chunks`. `PatchInline` (an incremental edit, a resident mirror, no
        // brick build outstanding) PATCHES the field (G3): only the dirty chunks are
        // re-evaluated and only their atlas slots written. A small wholesale rebuilds
        // inline; a LARGE wholesale — or any edit while an async brick build is
        // outstanding (the interlock: never patch a stale artifact) — dispatches to the
        // async brick worker, and the current display keeps drawing until the artifacts
        // land (`poll_brick_worker`). A non-chunkable / empty scene clears the field to
        // the mesh fallback.
        //
        // The CPU brick MIRROR (`build`, `incremental_brick_field`) runs for ANY chunkable scene
        // (`brick_gate`). The DISPLAY raymarch (installed in section (B) below) keeps its stricter
        // `!loaded_material` + representability conditions; a mixed-material or textured scene
        // meshes its display. The block YIELDS whether the brick DISPLAY was installed this
        // rebuild — the mesh-skip decision below reads it.
        let brick_display_installed = {
            let mut brick_display_installed = false;
            if brick_gate && matches!(brick_route, BrickRebuildAction::WholesaleAsync) {
                // Large wholesale (or a large edit while a brick build is outstanding — the
                // interlock): dispatch the record build + pyramid + classify to the async
                // worker (the ~2s main-thread hitch this removes). Stale-while-rebuilding:
                // the resident mirror + the renderer's live field are left UNTOUCHED (the
                // route keeps every mid-flight edit off the patch path), and the CURRENT
                // display keeps drawing until `poll_brick_worker` installs the artifacts.
                Self::dispatch_wholesale_brick_rebuild(
                    &self.brick_worker,
                    &mut self.brick_generation,
                    &mut self.brick_async_outstanding,
                    // O(chunks) Arc bumps — the mesh route below may MOVE the vec.
                    two_layer_chunks.clone(),
                    density,
                    recentre_voxels,
                );
                // The mesh-skip decision reads "is the brick the display this rebuild"; while
                // the build is in flight that is a PREDICTION, and we predict ENGAGED: the
                // landing field installs as the display (the common case — building the giant
                // fallback mesh meanwhile would be pure double work, and it must NOT be built
                // behind the brick), unless the arrival is `Empty`, which hands the display to
                // the mesh and kicks its rebuild. A first build shows nothing until the field lands
                // (~seconds, window responsive) — the same pop-in as the async startup.
                // NOTE: deliberately NOT predicted from `has_brick_field()` — a live field
                // can be a stale F1 pending-clear placeholder, and treating that as "the
                // display" would Skip-cancel the replacement mesh the handover waits on.
                {
                    brick_display_installed = true;
                }
            } else if brick_gate {
                profiling::scope!("brick_field_build");
                // (A) Maintain the CPU brick MIRROR (ADR 0011 G5), built for any chunkable
                // scene regardless of display representability. Patch iff the brick route is
                // PatchInline (an incremental edit + a resident mirror + no brick build
                // outstanding — `route_brick_rebuild` folds all three in). `update` (the GPU
                // atlas-slot descriptor) is consumed by the display install in (B).
                let patch_mirror = matches!(brick_route, BrickRebuildAction::PatchInline);
                // `update` is the GPU atlas-slot descriptor (patch path); `wholesale_atlas` is
                // the upload payload MOVED out of a wholesale build (item 9 — one copy of the
                // field). The patch path reads records/geometry/dirty bytes from the resident
                // mirror directly (no `to_build()`), so it produces no payload here.
                let (update, wholesale_atlas): (
                    Option<BrickFieldUpdate>,
                    Option<SculptedAtlasPayload>,
                ) = if patch_mirror {
                    let dirty = incremental_dirty_chunks
                        .as_ref()
                        .expect("PatchInline ⇒ incremental_dirty_chunks is Some");
                    let field = self
                        .incremental_brick_field
                        .as_mut()
                        .expect("patch_mirror ⇒ Some");
                    debug_assert_eq!(
                        field.brick_edge_voxels(),
                        density,
                        "an incremental edit never changes density (it routes wholesale)"
                    );
                    let update = field.apply_dirty_update(&two_layer_chunks, dirty);
                    (Some(update), None)
                } else {
                    // Wholesale (re)build; RESET the mirror BY MOVE (records move into the
                    // mirror, atlas bytes into the upload payload) so the next incremental edit
                    // patches from a known-good full field.
                    let (build, slot_tiles) =
                        build_brick_field_with_tiles(&two_layer_chunks, density);
                    if build.brick_records.is_empty() {
                        self.incremental_brick_field = None;
                        (None, None)
                    } else {
                        let (mirror, atlas) =
                            IncrementalBrickField::from_wholesale_with_tiles(build, slot_tiles);
                        self.incremental_brick_field = Some(mirror);
                        (None, Some(atlas))
                    }
                };

                // The field is empty iff no mirror survived (a wholesale emptied it, or an
                // incremental patch removed the last record) — read from the single owner.
                let records_empty = self
                    .incremental_brick_field
                    .as_ref()
                    .is_none_or(|field| field.records().is_empty());
                if records_empty {
                    // The edit emptied the field — no display brick.
                    self.incremental_brick_field = None;
                } else {
                    // (B) DISPLAY: install/patch the GPU raymarch renderer.
                    // The representability gate is DELETED (material atlas):
                    // EVERY non-empty scene engages the brick path, including mixed-material and
                    // overlay-disagreeing ones — the texel carries the overlay bit and records
                    // carry material + overlay per-record, and a mixed brick's per-voxel cell keys
                    // ride in the side atlas. ADR 0011 G2 — a loaded VS material does not skip the
                    // install either (the raymarch textures per-face from the block's D2Array).
                    {
                            let pyramid = ClipmapPyramid::from_chunks(&two_layer_chunks);
                            // The single-owner mirror is the truth for records + atlas geometry;
                            // the renderer seams read straight from it (item 9).
                            let mirror = self
                                .incremental_brick_field
                                .as_ref()
                                .expect("records_empty false ⇒ a resident mirror");
                            // ADR 0011 interior elision: the record set is SURFACE-ONLY by
                            // construction (`build_brick_field` fuses the occlusion decision
                            // into emission — a fully-occluded interior block never becomes a
                            // record, so nothing here needs a second mask pass). For a large
                            // solid the per-edit record upload is ∝surface, not ∝volume.
                            // Interiors live in the two-layer chunks the clip-map derives from.
                            let gpu_records = pack_gpu_records(mirror.records(), |_| false);
                            // Patch in place iff we produced an incremental update AND the
                            // renderer actually HOLDS A LIVE, CURRENT FIELD; otherwise (wholesale,
                            // or the display re-engaging from a mesh fallback) install fresh. The
                            // staleness rules live in `brick_patch_in_place` (pure, unit-tested):
                            // F2 — a cleared/present-but-empty field must re-install, never patch;
                            // and a PENDING deferred clear (`brick_display_pending_clear`) marks
                            // the live field a stale F1 placeholder that must also re-install.
                            let renderer_holds_live_field = self
                                .brick_raymarch_renderer
                                .as_ref()
                                .is_some_and(|renderer| renderer.has_brick_field());
                            if brick_patch_in_place(
                                update.is_some(),
                                renderer_holds_live_field,
                                self.brick_display_pending_clear,
                            ) {
                                let update = update
                                    .as_ref()
                                    .expect("brick_patch_in_place true ⇒ an update was produced");
                                if update.atlas_grew {
                                    println!(
                                        "brick: atlas grew — full re-pack ({} sculpted slots)",
                                        mirror.sculpted_brick_count()
                                    );
                                }
                                let renderer = self
                                    .brick_raymarch_renderer
                                    .as_mut()
                                    .expect("brick_patch_in_place true ⇒ a live field is resident");
                                // `patch_brick_field` patches the cell-key side atlas from the
                                // mirror too (its own dirty-slot list), so mixed bricks stay current.
                                renderer.patch_brick_field(
                                    &self.device,
                                    &self.queue,
                                    mirror,
                                    update,
                                    &gpu_records,
                                    &pyramid,
                                    recentre_voxels,
                                );
                            } else {
                                // Wholesale install: the upload payload was moved out of the
                                // build; a re-engaging incremental edit (no wholesale payload)
                                // re-packs it once from the mirror (the legitimate resize pack).
                                // The cell-key side atlas is re-packed from the mirror the same way,
                                // so a mixed scene's per-voxel tiles upload with the occupancy atlas.
                                let atlas = wholesale_atlas
                                    .unwrap_or_else(|| mirror.pack_atlas_payload());
                                let cell_key_atlas = mirror.pack_cell_key_atlas_payload();
                                let renderer =
                                    self.brick_raymarch_renderer.get_or_insert_with(|| {
                                        BrickRaymarchRenderer::new(
                                            &self.device,
                                            &self.queue,
                                            self.color_format,
                                        )
                                    });
                                renderer.install_brick_field_with_cell_keys(
                                    &self.device,
                                    &self.queue,
                                    mirror.records(),
                                    &atlas,
                                    &cell_key_atlas,
                                    &gpu_records,
                                    &pyramid,
                                    recentre_voxels,
                                );
                            }
                            brick_display_installed = true;
                    } // end display-install block
                    // `build` (this rebuild's boundary set) is consumed only by the display
                    // install above; ADR 0012 retired the fog occupancy consumer.
                }
                // Inline install seam: the resident mirror/field now reflect THIS resolve;
                // discard any superseded in-flight async brick result on arrival.
                self.finish_brick_install();
            } else {
                // Non-chunkable (a VoxelBody-only field): no brick mirror, no display brick. Any
                // in-flight async brick result was built for a scene shape that no longer
                // applies — the seam's generation bump discards it on arrival.
                self.finish_brick_install();
                self.incremental_brick_field = None;
            }
            // NOTE: the gpu raymarch display is NOT cleared here anymore. When it did not install
            // the display must hand back to the mesh, but clearing NOW (before the replacement
            // mesh is current) blanked the model for the seconds a large async rebuild takes.
            // The handover is reconciled AFTER the mesh route is decided below (F1) — cleared
            // immediately when the mesh is current this frame, DEFERRED to the install seam when
            // the replacement builds async and the stale brick can keep drawing.
            brick_display_installed
        };

        // Brick-display perf follow-up to epic #64: the fallback cuboid mesh is DRAWN only when
        // the brick raymarch is not engaged. Engagement mirrors the per-frame gate
        // (`brick_raymarch_engaged`): a field installed this rebuild AND no debug-face mode.
        // ADR 0011 G2 — a loaded VS material now KEEPS the brick display (it textures the
        // raymarch per-face), so it no longer forces the mesh; with the representability gate
        // deleted only an EMPTY scene (no field installed ⇒ `brick_display_installed` false)
        // meshes. When engaged the mesh
        // is redundant → SKIP the build and mark it stale; the C1 interlock composes via
        // `route_mesh_build` (a stale mesh, like an outstanding async build, is never inline-
        // patched — it rebuilds wholesale when next needed).
        let brick_display_engaged = Self::brick_display_engaged_predicate(
            brick_display_installed,
            debug_face_orientation,
        );
        let mesh_route = route_mesh_build(
            brick_display_engaged,
            self.mesh_stale,
            self.geometry_async_outstanding,
            edit_shape,
            ASYNC_REBUILD_CHUNK_THRESHOLD,
        );
        // F1: does the mesh become CURRENT this frame (an inline build/patch), vs building async
        // (WholesaleAsync) or being skipped (brick still the display)? The brick-handover
        // reconcile below reads it to decide whether to clear the stale brick now or defer.
        let mesh_became_current = matches!(
            mesh_route,
            MeshBuildRoute::Build(RebuildRoute::InlineIncremental)
                | MeshBuildRoute::Build(RebuildRoute::WholesaleInline)
        );

        match mesh_route {
            MeshBuildRoute::Skip => {
                // The brick raymarch is the display — skip the ~333ms mesh build. Mark the mesh
                // stale so the next edit that needs it rebuilds wholesale. Bump the generation
                // and drop any outstanding async so a stale in-flight mesh result is discarded on
                // arrival (`poll_geometry_worker`) instead of being swapped in behind the brick.
                self.geometry_generation.next_generation();
                self.geometry_async_outstanding = false;
                self.mesh_stale = true;
            }
            MeshBuildRoute::Build(RebuildRoute::InlineIncremental) => {
                // Issue #54/#55 fast path: an incremental dirty-chunk re-mesh is already a
                // few chunks — build it inline (no worker hop, no added latency). Reached ONLY
                // when nothing is outstanding, so the installed renderer reflects the latest
                // resolve and patching it in place is sound.
                //
                // `finish_mesh_install` bumps the generation so any (phantom) in-flight result is
                // discarded on arrival — the tracker rejects a non-newest generation.
                let dirty = incremental_dirty_chunks
                    .expect("InlineIncremental is only routed for an incremental edit");
                profiling::scope!("cuboid_incremental_two_layer");
                self.cuboid_mesh_renderer.incremental_rebuild_from_two_layer_chunks(
                    &self.device,
                    &two_layer_chunks,
                    grid_dimensions,
                    recentre_voxels,
                    density,
                    &dirty,
                );
                // Reached only with `mesh_stale == false` (a stale mesh forces wholesale via
                // `route_mesh_build`), so the in-place patch is sound. The install seam clears
                // `mesh_stale` + bumps the generation (cleanup b: the sole non-Skip stale writer).
                self.finish_mesh_install();
            }
            MeshBuildRoute::Build(RebuildRoute::WholesaleAsync) => {
                // Issue #60: dispatch a WHOLESALE rebuild to the worker so the UI never
                // freezes (the ~3s classify ran above on the main thread; the heavy mesh CPU
                // build + GPU upload is what goes async). Stamp a fresh generation, send the
                // owned FULL covering set (the `AppCore` resident cache is always current on
                // the main thread, so a full wholesale is correct even when the edit itself
                // was incremental — the C1 interlock), and keep the CURRENT renderer drawing
                // (stale-while-rebuilding). Mark the async build OUTSTANDING so the NEXT edit
                // also routes here instead of inline-patching the still-stale renderer. The
                // result is polled + swapped in the event loop (`poll_geometry_worker`).
                self.dispatch_wholesale_mesh_rebuild(
                    two_layer_chunks,
                    grid_dimensions,
                    recentre_voxels,
                    density,
                    band,
                    region,
                );
                // The worker owns the (re)build now; the outstanding flag carries the C1
                // interlock. `mesh_stale` is intentionally NOT cleared here (cleanup b: only
                // `finish_mesh_install` clears it, and only the Skip arm sets it true) — leaving
                // it set while the async build is outstanding is harmless (the outstanding flag
                // already forces wholesale routing), and `poll_geometry_worker` clears it via
                // `finish_mesh_install` the moment the result installs.
            }
            MeshBuildRoute::Build(RebuildRoute::WholesaleInline) => {
                // A small wholesale rebuild (at/below the threshold), nothing outstanding:
                // build inline — cheap enough not to hitch a frame, and it avoids the worker's
                // one-frame swap latency. Bump the generation so any phantom in-flight result
                // is discarded on arrival. Build at the active band so the mesh matches the
                // render path immediately (no swap-frame re-mesh — same M2 reasoning). The
                // install seam bumps the generation + clears `mesh_stale`.
                self.cuboid_mesh_renderer = CuboidMeshRenderer::new_from_two_layer_chunks_banded(
                    &self.device,
                    &self.queue,
                    self.color_format,
                    &two_layer_chunks,
                    grid_dimensions,
                    recentre_voxels,
                    density,
                    band,
                    region,
                );
                self.finish_mesh_install();
            }
        }

        // F1 brick-display handover reconcile (a no-op while `brick_raymarch_renderer` is
        // `None`). When brick did NOT install this rebuild, the display must hand back to the
        // cuboid mesh. Clear the stale brick field NOW when the replacement mesh is already
        // current this frame (inline), OR the brick can't/needn't draw (a mesh-only mode is
        // active), OR no live field remains. But when a stale field is still live AND the
        // replacement builds ASYNC AND the brick would still draw (no debug-face / loaded
        // material), KEEP it drawing — clearing now blanks the model for the seconds the worker
        // takes. `finish_mesh_install` (via `complete_brick_display_handover`) clears it when the
        // fresh mesh lands. When brick DID install it is the display: cancel any pending handover.
        {
            let has_live_brick_field = self
                .brick_raymarch_renderer
                .as_ref()
                .is_some_and(|renderer| renderer.has_brick_field());
            // ADR 0011 G2 — a loaded VS material now keeps drawing as textured bricks, so it no
            // longer forces a handover to the mesh (mirrors the flipped engagement predicate).
            let brick_would_draw_if_kept = !debug_face_orientation;
            self.apply_brick_display_handover(brick_display_handover(
                brick_display_installed,
                mesh_became_current,
                brick_would_draw_if_kept,
                has_live_brick_field,
            ));
        }
    }

    /// Issue #60 (ADR 0003 §7): poll the geometry worker for a finished wholesale
    /// rebuild and, if it is NOT stale, swap it in. Returns whether a fresh renderer was
    /// installed (the shell requests the redraw). Called each frame in the event loop.
    /// Non-blocking — the app never waits on the worker.
    ///
    /// Stale-while-rebuilding: until a fresh result arrives, the current
    /// `cuboid_mesh_renderer` keeps drawing. On arrival, the [`GenerationTracker`] decides
    /// whether the result is still the newest dispatched (accept + swap) or was superseded
    /// by a later edit (discard). The worker drains-to-latest, so at most the newest built
    /// renderer is here; the tracker guards against a build that a mid-flight edit
    /// (wholesale OR incremental — both bump the generation) already superseded.
    pub fn poll_geometry_worker(&mut self) -> bool {
        let Some(result) = self.geometry_worker.try_recv_result() else {
            return false;
        };
        if !self.geometry_generation.accepts(result.generation) {
            // A later edit superseded this build — discard it (the stale mesh, or the newer
            // inline/incremental result, is already what's showing). The superseding edit set
            // its own outstanding state (a re-dispatched wholesale keeps it `true`; an inline
            // edit reached only when nothing was outstanding leaves it `false`), so we do NOT
            // touch `geometry_async_outstanding` here.
            return false;
        }
        // Issue #60 M1: a `None` renderer means the worker's build PANICKED (it logged to
        // stderr and stayed alive). Keep the current (stale) mesh and leave the outstanding
        // flag SET so the next edit re-dispatches a fresh wholesale — never silently wedge.
        let Some(renderer) = result.renderer else {
            return false;
        };
        // Fresh: swap the freshly-built renderer in (GPU buffers already uploaded on the
        // worker). The install seam drops the async-outstanding flag (the inline fast-paths
        // resume — issue #60 C1), clears `mesh_stale` (a freshly built worker mesh reflects
        // the latest resolve), and completes any deferred brick-display handover (F1 — clear
        // the stale brick kept drawing during the rebuild, now the fresh mesh takes the frame).
        self.cuboid_mesh_renderer = renderer;
        self.finish_mesh_install();
        true
    }

    /// Poll the async brick-pipeline worker (perf follow-up to epic #64, issue #60
    /// pattern) and, if a finished wholesale build is still the newest dispatched, install
    /// its artifacts: the CPU mirror always; the display field (a milliseconds install upload —
    /// the multi-second CPU build already happened on the worker) for every non-empty scene (the
    /// representability gate is deleted — mixed scenes install their cell-key side atlas too).
    /// Returns whether anything was installed (the shell requests the redraw). A superseded result
    /// is discarded via the [`GenerationTracker`]; a panicked build (`outcome == None`) keeps the
    /// stale field and LEAVES the outstanding flag set so the next edit re-dispatches.
    ///
    /// `_context` is unused now that only `Empty` (which hands off via the per-frame
    /// `ensure_display_mesh_current` seam) can fail to install a brick — the parameter is kept on
    /// the signature for symmetry with the other pollers and a future non-Empty handover.
    pub fn poll_brick_worker(&mut self, _context: DisplayRefreshContext) -> bool {
        let Some(result) = self.brick_worker.try_recv_result() else {
            return false;
        };
        if !self.brick_generation.accepts(result.generation) {
            // A later edit superseded this build — discard it. The superseding edit set its
            // own outstanding state; do not touch it here (mirrors `poll_geometry_worker`).
            return false;
        }
        let Some(outcome) = result.outcome else {
            // The worker's build PANICKED (logged on the worker, which stayed alive). Keep
            // the current (stale) field and leave the outstanding flag SET so the next edit
            // re-dispatches a fresh wholesale — never silently wedge.
            return false;
        };
        // The accepted result is about to become the resident brick state — the install
        // seam (generation bump is a no-op here: nothing newer is in flight, by `accepts`).
        self.finish_brick_install();
        match outcome {
            BrickRebuildOutcome::Empty => {
                // The scene emptied: drop the mirror and clear any live display field — the
                // mesh (trivially cheap for an empty scene) takes over via the per-frame
                // `ensure_display_mesh_current` seam. The field-clear + pending-clear reset is
                // exactly the `ClearNow` handover action (the mirror drop is the extra step
                // this arm adds).
                self.incremental_brick_field = None;
                self.apply_brick_display_handover(BrickDisplayHandover::ClearNow);
            }
            BrickRebuildOutcome::MirrorOnly { mirror } => {
                // The worker built no display artifacts: install the CPU mirror only.
                self.incremental_brick_field = Some(mirror);
            }
            BrickRebuildOutcome::Display(install) => {
                {
                    let crate::workers::brick::BrickDisplayInstall {
                        atlas,
                        cell_key_atlas,
                        gpu_records,
                        pyramid,
                        mirror,
                    } = *install;
                    // The mirror is the single owner; install reads its records + the upload
                    // payload the worker moved out of the build alongside it (item 9). The
                    // cell-key side atlas rides too (empty unless the scene has a mixed brick).
                    let mirror = self.incremental_brick_field.insert(mirror);
                    // Wholesale semantics: always a fresh INSTALL (never a patch) — the
                    // worker built the complete field, and a cleared/stale resident field
                    // must not be patched (the F2 gate's lesson).
                    let renderer = self.brick_raymarch_renderer.get_or_insert_with(|| {
                        BrickRaymarchRenderer::new(&self.device, &self.queue, self.color_format)
                    });
                    renderer.install_brick_field_with_cell_keys(
                        &self.device,
                        &self.queue,
                        mirror.records(),
                        &atlas,
                        &cell_key_atlas,
                        &gpu_records,
                        &pyramid,
                        result.recentre_voxels,
                    );
                    println!(
                        "brick: async wholesale field installed ({} records, {} sculpted)",
                        mirror.records().len(),
                        mirror.sculpted_brick_count(),
                    );
                    // The brick is (again) the display: cancel any pending deferred clear.
                    self.brick_display_pending_clear = false;
                }
            }
        }
        true
    }

    /// Rebuild the fallback cuboid mesh IF it is stale and about to become the display
    /// (brick-display perf follow-up to epic #64). The mesh is skipped while the ADR 0011 brick
    /// raymarch is engaged; a debug-face toggle or a loaded-material change are pure per-frame
    /// display flags that can drop that engagement WITHOUT a `scene_changed` rebuild, so the
    /// skipped mesh would otherwise be drawn stale/empty. This closes that gap: called every
    /// frame before the voxel draw, it is a no-op unless the mesh is stale AND the brick will
    /// not draw.
    ///
    /// F3: the rebuild REUSES the resident two-layer cache (the scene is unchanged, so this is an
    /// O(chunks) `Arc`-refcount handout — NOT a stateless from-scratch `build_covering_chunks`
    /// re-resolve, which FROZE the main thread for seconds on a large scene the instant a
    /// material tile / debug-face toggled). A small covering set builds inline; a large one is
    /// dispatched to the async geometry worker so the UI never freezes (the current display —
    /// the mesh, or the deferred brick raymarch of F1 — keeps drawing until the fresh mesh lands).
    pub fn ensure_display_mesh_current(&mut self, context: DisplayRefreshContext) {
        if !self.mesh_stale {
            return;
        }
        // Will the brick raymarch draw this frame? The shared per-frame gate (term-identical to
        // the draw path). If engaged, the (stale) mesh stays hidden — leave it stale, skip.
        if self.brick_display_engaged(context.debug_face_orientation) {
            return;
        }
        // A wholesale brick build is IN FLIGHT and, when it lands, will either install as
        // the display (the mesh stays skipped) or, on an `Empty` arrival, hand the display
        // off to the mesh via this per-frame seam. Do NOT
        // resolve the covering set + build the fallback mesh synchronously meanwhile — on
        // a giant scene that is the multi-second frame-one freeze the async pipeline
        // exists to remove, and the mesh it builds would sit unseen behind the landing
        // brick display. Debug-face mode still needs the mesh NOW (the brick will not
        // draw when it lands), so it proceeds. (A panicked build leaves the flag set with
        // no arrival — the display then stays stale until the next edit re-dispatches,
        // the same documented policy as the geometry worker's panic path.)
        if self.brick_async_outstanding && !context.debug_face_orientation {
            return;
        }
        self.rebuild_stale_display_mesh(context);
    }

    /// Rebuild the stale fallback mesh from the RESIDENT two-layer cache — the body of
    /// [`Self::ensure_display_mesh_current`] WITHOUT its brick-engagement gate, factored out so
    /// the deferred-handover path (where the stale brick field is DELIBERATELY kept drawing under
    /// F1 so the model never blanks) can rebuild the mesh without the engagement gate — which
    /// would see the live field and skip — applying. A no-op when the mesh is already current or a
    /// build is in flight.
    fn rebuild_stale_display_mesh(&mut self, context: DisplayRefreshContext) {
        if !self.mesh_stale {
            return;
        }
        // A wholesale build is already in flight for this stale mesh — wait for it (its
        // `poll_geometry_worker` install clears `mesh_stale` via `finish_mesh_install`). Don't
        // re-dispatch every frame while it builds (that would flood the worker).
        if self.geometry_async_outstanding {
            return;
        }
        // The mesh is about to be the display but is stale — rebuild it wholesale from the
        // RESIDENT two-layer cache (scene unchanged ⇒ the same set the last resolve produced,
        // handed out as O(chunks) Arc bumps). Route like any wholesale edit: small inline, large
        // async. The frame parameters come from the last rebuild's stored recentre + region.
        let density = context.density;
        let chunks = context
            .two_layer_cache
            .resident_two_layer_chunks(context.scene, density, 0);
        let grid_dimensions = context.region_dimensions;
        let recentre = context.recentre_voxels;
        let band = context.band;
        let region = context.region;
        if chunks.len() > ASYNC_REBUILD_CHUNK_THRESHOLD {
            // Large: dispatch async so the toggle never freezes. Keep drawing the current display
            // until the result installs (`poll_geometry_worker` → `finish_mesh_install` clears
            // `mesh_stale`). `mesh_stale` stays set meanwhile; the outstanding guard above keeps
            // us from re-dispatching each frame.
            self.dispatch_wholesale_mesh_rebuild(
                chunks,
                grid_dimensions,
                recentre,
                density,
                band,
                region,
            );
            println!(
                "mesh: fallback rebuild dispatched async (brick display disengaged — \
                 debug-face / material)"
            );
        } else {
            // Small: build inline (cheap), then install (bumps the generation + clears stale).
            self.cuboid_mesh_renderer = CuboidMeshRenderer::new_from_two_layer_chunks_banded(
                &self.device,
                &self.queue,
                self.color_format,
                &chunks,
                grid_dimensions,
                recentre,
                density,
                band,
                region,
            );
            self.finish_mesh_install();
            println!(
                "mesh: rebuilt fallback inline (brick display disengaged — debug-face / material)"
            );
        }
    }

    /// The cuboid mesh renderer (the draw path binds its `FrameOverlays` slot, the
    /// bind-group layout, and the sampler through here).
    pub fn cuboid_mesh_renderer(&self) -> &CuboidMeshRenderer {
        &self.cuboid_mesh_renderer
    }

    /// The cuboid mesh renderer, mutably (the render path's `update_uniforms`).
    pub fn cuboid_mesh_renderer_mut(&mut self) -> &mut CuboidMeshRenderer {
        &mut self.cuboid_mesh_renderer
    }

    /// The brick raymarch renderer, if one is resident (the draw path's `FrameOverlays` slot).
    pub fn brick_raymarch_renderer(&self) -> Option<&BrickRaymarchRenderer> {
        self.brick_raymarch_renderer.as_ref()
    }

    /// The brick raymarch renderer, mutably (the render path's uniform uploads).
    pub fn brick_raymarch_renderer_mut(&mut self) -> Option<&mut BrickRaymarchRenderer> {
        self.brick_raymarch_renderer.as_mut()
    }
}
