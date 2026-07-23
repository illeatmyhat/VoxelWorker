//! Headless orchestrator owning store + camera — the AppCore keystone.
//!
//! ADR 0003 (foundation rework). `AppCore` is the headless half of the app: it
//! owns the [`TwoLayerResidentCache`] (boundary-aware residency + per-chunk resolve;
//! ADR 0010 E5 — the SOLE runtime display path) and the [`OrbitCamera`], and exposes
//! the headless scene queries both binaries drive. The windowed
//! shell (`WindowedState`) and `bin/shot` keep the GPU renderers + winit/egui
//! plumbing and delegate to `AppCore` for the headless work; in **A3** `shot`
//! re-points here, at which point the golden net tests the real app instead of a
//! parallel render copy.
//!
//! **Ownership boundary (A2d).** `AppCore` owns the store + camera but BORROWS
//! the scene (`&Scene`) — the scene still lives in `PanelState` until Phase B/C
//! moves it here. The scene-query associated functions below therefore take
//! `&Scene` as a parameter; they become `&self` methods once `AppCore` owns the
//! scene. Resolve state + the borrow-sensitive `AppCore::rebuild` land in
//! **A2e**; `render` reads all headless data from here in **A2f**.

use std::sync::Arc;

use camera::OrbitCamera;
use document::command::CommandStack;
use voxel_core::spatial_index::LeafSpatialIndex;
use evaluation::two_layer_store::{TwoLayerChunk, TwoLayerResidentCache};
use voxel_core::voxel::RecentreVoxels;

mod intent;
mod picking;
pub use picking::{PickFrame, VoxelPick};
mod placement;
pub use placement::PlacementOutcome;
mod queries;
pub use queries::MeshClip;
mod rebuild;
mod replay;
mod selected_operand;

pub use replay::{default_replay_seed_scene, replay_intent_script};
pub use selected_operand::SelectedOperandGhost;

#[cfg(test)]
mod replay_tests;
#[cfg(test)]
mod undo_tests;
#[cfg(test)]
mod sketch_group_tests;
#[cfg(test)]
mod intent_dispatch_tests;

/// The headless orchestrator: owns the per-chunk resolve `Store` and the
/// [`OrbitCamera`], and answers the headless scene queries the shell renders from.
pub struct AppCore {
    /// The **boundary-aware two-layer** resolve cache (ADR 0010 E5 — the SOLE runtime
    /// display path; the dense `Store` is retired to a test oracle). The resolve
    /// mechanism behind the shell's geometry rebuild: it classifies each covering
    /// chunk's blocks air / coarse-solid / boundary via the one evaluator, keeps the
    /// two-layer chunks resident, and re-derives only the chunks an edit's world-AABB
    /// intersects (chunk-granular incremental, #54).
    pub two_layer_cache: TwoLayerResidentCache,
    /// The orbit camera (orbit angles + distance + projection). The windowed shell
    /// drives it from input; `shot` sets it from CLI flags.
    pub camera: OrbitCamera,
    /// The leaf spatial index (issue #27 S3) the LAST [`rebuild`](Self::rebuild)
    /// resolved from, kept so the next rebuild can diff against it to compute the
    /// edit's dirty world-AABB. `None` before the first rebuild (which clears
    /// wholesale).
    previous_leaf_index: Option<LeafSpatialIndex>,
    /// The composite recentre (floating origin, in voxels) the LAST rebuild resolved
    /// at (issue #20 S6c-2c): the resolve bookkeeping that records whether the
    /// floating origin shifted. `None` before the first rebuild.
    previous_recentre_voxels: Option<[i64; 3]>,
    /// The density the LAST rebuild resolved at (issue #40). A density change re-keys
    /// every chunk (chunk extent = `CHUNK_BLOCKS × density`), so even when the recentre
    /// happens to land at `[0,0,0]` at both densities the per-chunk buffers are in a
    /// different frame and the incremental cuboid path is unsafe — this gates it off.
    /// `None` before the first rebuild.
    previous_density: Option<u32>,
    /// The linear inverse-command stack behind undo/redo (ADR 0003 Phase C C2). Every
    /// non-selection-only `apply_intent` pushes a [`Command`] here; `undo`/`redo`
    /// shuttle commands between its two Vecs. Empty until the first undoable edit.
    command_stack: CommandStack,
}

/// The headless resolve output of a geometry [`rebuild`](AppCore::rebuild) (A2e;
/// ADR 0010 E5). Holds ONLY the **two-layer** covering chunks (owned) the shell meshes
/// through
/// [`CuboidMeshRenderer::new_from_two_layer_chunks`](display::mesh::CuboidMeshRenderer::new_from_two_layer_chunks),
/// plus the region dimensions + recentre the display frame is sized from.
///
/// **ADR 0011 G5 — the dense grid is gone.** A rebuild NO LONGER assembles a whole-region
/// `VoxelGrid`. The display meshes from `two_layer_chunks` and the brick sink packs from the
/// same set — neither needs a dense occupancy array. The only surviving dense resolves are the
/// compile-gated `oracle`-feature resolvers the parity tests cross-check against
/// (`Store::resolve_region` / `resolve_region_two_layer`), never a production path. So this
/// output is purely sparse + scalar metadata.
pub struct RebuildOutput {
    /// The region's voxel dimensions, read from the SCENE (see
    /// [`AppCore::region_dimensions_for`]) — what the camera auto-frame, gizmo,
    /// lattice, floor grid and layer scrubber are sized from.
    pub region_dimensions: [u32; 3],
    /// The **two-layer** covering chunks (`(absolute_chunk_coord, Arc<TwoLayerChunk>)`),
    /// `Arc`-shared out of the resident cache so they outlive the cache borrow WITHOUT a
    /// deep copy. The shell meshes them through
    /// [`CuboidMeshRenderer::new_from_two_layer_chunks`](display::mesh::CuboidMeshRenderer::new_from_two_layer_chunks)
    /// (coarse one-box + microblock cuboids + seam-flag culling) — the sole runtime
    /// display mesh path (ADR 0010 E5) — and the brick sink packs its records from the same
    /// set (ADR 0011 G3). Empty for a VoxelBody-only scene (no covering range).
    ///
    /// **Why `Arc`, not owned chunks.** Every rebuild used to deep-clone EVERY resident
    /// chunk into an owned `Vec` here (O(all-blocks) per edit) purely so the set could
    /// outlive the cache borrow / be moved into the async mesh request. Since the brick
    /// display's mesh route is `Skip`, the owned set is consumed only by borrowing readers
    /// on the primary path, so that deep clone was pure waste; sharing an `Arc` per chunk
    /// makes it an O(chunks) refcount bump and composes with the brick readers directly.
    pub two_layer_chunks: Vec<([i32; 3], Arc<TwoLayerChunk>)>,
    /// The composite recentre (floating origin, voxels; ADR 0008) the two-layer mesh
    /// lands its geometry in — the SAME frame the brick sink packs its records in. Carried
    /// as [`RecentreVoxels`] so the frame value travels compile-checked through the async
    /// display flow, unwrapped only at the point of positional arithmetic (a chunk rebase,
    /// a leaf stamp) and the GPU uniform packing.
    pub recentre_voxels: RecentreVoxels,
    /// **The chunk-granular incremental GPU-buffer re-mesh hint (issue #55).** `Some(dirty)`
    /// when this rebuild LOCALISED — the edit's dirty world-AABB evicted exactly the `dirty`
    /// chunks (from [`TwoLayerResidentCache::invalidate_aabb`]) and the density did NOT change
    /// — so the shell can re-mesh + re-upload ONLY `dirty ∪ 26-neighbourhood(dirty) ∩ resident`
    /// via [`CuboidMeshRenderer::incremental_rebuild_from_two_layer_chunks`], keeping every
    /// other chunk's GPU buffers in place. `None` when the edit could NOT localise — the first
    /// build (no previous index, wholesale [`clear`](TwoLayerResidentCache::clear)), a density
    /// change (re-keys every chunk's voxel extent), or a region-spanning VoxelBody edit (no
    /// localisable box) — in which case the shell re-meshes WHOLESALE via
    /// [`CuboidMeshRenderer::new_from_two_layer_chunks`]. This is the same split the resident
    /// cache itself uses (`invalidate_aabb` vs `clear`), surfaced to the GPU-buffer layer.
    ///
    /// [`TwoLayerResidentCache::invalidate_aabb`]: evaluation::two_layer_store::TwoLayerResidentCache::invalidate_aabb
    /// [`CuboidMeshRenderer::incremental_rebuild_from_two_layer_chunks`]: display::mesh::CuboidMeshRenderer::incremental_rebuild_from_two_layer_chunks
    /// [`CuboidMeshRenderer::new_from_two_layer_chunks`]: display::mesh::CuboidMeshRenderer::new_from_two_layer_chunks
    pub incremental_dirty_chunks: Option<Vec<[i32; 3]>>,
    /// How far the floating-origin recentre SHIFTED this rebuild, in render-frame
    /// voxels (`new_recentre − previous_recentre`; `[0, 0, 0]` on the first build).
    /// The composite is re-centred on the world origin every rebuild, so when its
    /// extent (or the density, since the recentre is in voxels) changes the whole
    /// resolved world slides by this amount under a fixed camera. The windowed shell
    /// subtracts this from `camera.target` so the view stays locked on the same WORLD
    /// point across an edit — making the recentre visually inert (the camera moves
    /// only on EXPLICIT Fit/Home/Focus/orbit actions). The `shot` path ignores it
    /// (its camera is set per-capture from CLI flags), so goldens are unaffected.
    pub recentre_shift_voxels: [i64; 3],
}

/// Outcome of [`AppCore::rebuild`]: either the resolve output, or a rejection when
/// the density's PER-CHUNK voxel bound is exceeded. AppCore never writes panel
/// state, so the shell surfaces the cap warning from the returned figure.
pub enum RebuildOutcome {
    /// The resolve succeeded; the cache holds the freshly resolved covering chunks.
    Built(RebuildOutput),
    /// The density's single-chunk voxel capacity exceeds the bound; the cache was
    /// left untouched. `chunk_voxels_millions` is the offending count (millions).
    DensityRejected { chunk_voxels_millions: f32 },
}

impl AppCore {
    /// Assemble the headless core from a camera (ADR 0010 E5). The two-layer resolve
    /// cache is constructed here (ENABLED — the sole runtime display path); the caller
    /// supplies only the camera (restored orbit/projection).
    pub fn new(camera: OrbitCamera) -> Self {
        Self {
            two_layer_cache: TwoLayerResidentCache::enabled(),
            camera,
            previous_leaf_index: None,
            previous_recentre_voxels: None,
            previous_density: None,
            command_stack: CommandStack::new(),
        }
    }

    /// An `AppCore` whose two-layer resolve cache is PRE-WARMED with the startup covering
    /// set (async-brick startup follow-up to epic #64). The windowed shell builds its
    /// startup chunks THROUGH this cache so a pre-first-edit display seam — the fallback
    /// mesh rebuild after an async brick build lands `Empty` — hands out the
    /// RESIDENT chunks as O(chunks) `Arc` bumps instead of synchronously re-resolving the
    /// whole covering set on the main thread (the multi-second frame-one freeze). Edit-time
    /// semantics are identical to [`Self::new`]: the first [`rebuild`](Self::rebuild) still
    /// clears the cache (`previous_leaf_index` starts `None`), so no stale chunk can survive
    /// the first edit.
    pub fn with_warm_two_layer_cache(
        camera: OrbitCamera,
        two_layer_cache: TwoLayerResidentCache,
    ) -> Self {
        Self {
            two_layer_cache,
            ..Self::new(camera)
        }
    }

    /// The number of commands on the undo stack (ADR 0003 Phase C C2 test support).
    #[cfg(test)]
    pub(crate) fn undo_depth(&self) -> usize {
        self.command_stack.undo.len()
    }

    /// The number of commands on the redo stack (ADR 0003 Phase C C2 test support).
    #[cfg(test)]
    pub(crate) fn redo_depth(&self) -> usize {
        self.command_stack.redo.len()
    }
}
