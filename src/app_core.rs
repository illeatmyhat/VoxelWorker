//! Headless orchestrator owning store + camera — the AppCore keystone.
//!
//! ADR 0003 (foundation rework). `AppCore` is the headless half of the app: it
//! owns the [`Store`] (residency + per-chunk resolve) and the [`OrbitCamera`],
//! and exposes the headless scene queries both binaries drive. The windowed
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

use crate::camera::OrbitCamera;
use crate::core_geom::CHUNK_BLOCKS;
use crate::panel::LayerRange;
use crate::renderer::OnionFogParams;
use crate::scene::Scene;
use crate::spatial_index::LeafSpatialIndex;
use crate::store::Store;
use crate::voxel::{chunk_extent_exceeds_bound, VoxelGrid};

/// The headless orchestrator: owns the per-chunk resolve [`Store`] and the
/// [`OrbitCamera`], and answers the headless scene queries the shell renders from.
pub struct AppCore {
    /// Per-chunk resolve cache (issue #27 S2): the resolve mechanism behind the
    /// shell's geometry rebuild and the diameter readout. Lazily resolves each
    /// covering chunk and keeps it resident for reuse.
    pub store: Store,
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
}

/// The headless resolve output of a geometry [`rebuild`](AppCore::rebuild) (A2e).
/// Holds the assembled region grid (owned) plus the per-chunk render accessor,
/// which BORROWS the store — so the shell must consume both (build the cuboid mesh
/// + upload the fog occupancy) BEFORE the next `&mut AppCore` call.
pub struct RebuildOutput<'store> {
    /// The assembled monolithic region grid (recentred): feeds the fog upload and
    /// the shell's diameter re-measure.
    pub grid: VoxelGrid,
    /// The region's voxel dimensions, read from the SCENE (see
    /// [`AppCore::region_dimensions_for`]) — what the camera auto-frame, gizmo,
    /// lattice, floor grid and layer scrubber are sized from.
    pub region_dimensions: [u32; 3],
    /// The per-covering-chunk render accessor
    /// (`(absolute_chunk_coord, &rebased_grid)`), borrowing the store. Drop it
    /// before the next `&mut AppCore`.
    pub render_chunks: Vec<([i32; 3], &'store VoxelGrid)>,
}

/// Outcome of [`AppCore::rebuild`]: either the resolve output, or a rejection when
/// the density's PER-CHUNK voxel bound is exceeded. AppCore never writes panel
/// state, so the shell surfaces the cap warning from the returned figure.
pub enum RebuildOutcome<'store> {
    /// The resolve succeeded; the store holds the freshly resolved covering chunks.
    Built(RebuildOutput<'store>),
    /// The density's single-chunk voxel capacity exceeds the bound; the store was
    /// left untouched. `chunk_voxels_millions` is the offending count (millions).
    DensityRejected { chunk_voxels_millions: f32 },
}

impl AppCore {
    /// Assemble the headless core from an already-constructed store + camera. The
    /// shell builds both (the store seeds the startup diameter readout, the camera
    /// restores persisted orbit/projection) and hands them over here.
    pub fn new(store: Store, camera: OrbitCamera) -> Self {
        Self {
            store,
            camera,
            previous_leaf_index: None,
            previous_recentre_voxels: None,
        }
    }

    /// **The headless geometry rebuild (A2e).** Route the resolve through the
    /// per-chunk store with issue #27 S3 TARGETED invalidation: build the new
    /// scene's leaf spatial index, diff it against the last rebuild's to get the
    /// edit's dirty world-AABB, and evict ONLY the chunks that AABB touches (every
    /// other cached chunk stays resident). Fall back to a wholesale `clear()` when a
    /// precise AABB can't be computed — the first rebuild (no previous index), a
    /// density change, or a region-spanning Part edit (no localisable box, see
    /// `LeafSpatialIndex::edit_aabb_since`). The reassembled grid is byte-identical
    /// either way (the same chunks are re-resolved; untouched chunks are reused).
    ///
    /// Returns the assembled grid + region dimensions + the per-chunk render
    /// accessor, which BORROWS the store. The returned [`RebuildOutcome`] therefore
    /// borrows `self`, so the shell must consume it (build the cuboid mesh, upload
    /// the fog occupancy) BEFORE the next `&mut AppCore` call. A density whose
    /// single-chunk voxel capacity exceeds the bound is rejected WITHOUT touching
    /// the store, returning the offending count so the shell can surface the cap
    /// warning (AppCore never writes panel state).
    pub fn rebuild<'a>(&'a mut self, scene: &Scene, density: u32) -> RebuildOutcome<'a> {
        // Issue #27 S2: the resolve is chunked + lazy, so the voxel bound is a
        // PER-CHUNK bound, not a whole-scene total. Only a pathological density
        // (one chunk's voxel capacity alone exceeds the bound) is rejected.
        if chunk_extent_exceeds_bound(density) {
            let chunk_extent = (CHUNK_BLOCKS * density.max(1)) as u64;
            let chunk_voxels = chunk_extent * chunk_extent * chunk_extent;
            return RebuildOutcome::DensityRejected {
                chunk_voxels_millions: chunk_voxels as f32 / 1_000_000.0,
            };
        }

        // S3 targeted invalidation. The cuboid renderer rebuilds every covering
        // chunk wholesale, so it needs no dirty set of its own — but the store's
        // invalidation side effects ARE still required: `invalidate_aabb` evicts the
        // edit's dirty chunks (so `resident_render_chunks` re-resolves them), and
        // `clear()` handles the first build / density change / region-spanning edit
        // where there is no localisable AABB.
        let new_leaf_index = scene.build_leaf_spatial_index(density);
        let new_recentre = scene.recentre_voxels_for_resolve(density);
        match self.previous_leaf_index.as_ref() {
            Some(previous) => match new_leaf_index.edit_aabb_since(previous) {
                Some(edit_aabb) => {
                    self.store.invalidate_aabb(&edit_aabb, density);
                }
                None => self.store.clear(),
            },
            None => self.store.clear(),
        }
        self.previous_recentre_voxels = Some(new_recentre);
        self.previous_leaf_index = Some(new_leaf_index);

        // Resolve the assembled grid (owned), then gather the per-chunk render
        // accessor LAST — it borrows the store, so every `&mut store` call above
        // must already be done. The grid drops straight into the fog upload; the
        // accessor feeds the cuboid mesher, then the shell drops it.
        let grid = self.store.resolve_region(scene, density, 0);
        let region_dimensions = Self::region_dimensions_for(scene, density, &grid);
        let render_chunks = self.store.resident_render_chunks(scene, density, 0);
        RebuildOutcome::Built(RebuildOutput {
            grid,
            region_dimensions,
            render_chunks,
        })
    }

    /// Resolve the whole [`Scene`] into a fresh grid (ADR 0001 step 2). Every
    /// visible node composites (union) into one region sized to the per-axis max of
    /// the nodes' extents, at full resolution (`lod 0`). `voxels_per_block` is the
    /// global app density (the inspector mirror's density). For a one-node scene
    /// this is identical to the step-1 behaviour.
    ///
    /// An associated function for now (it borrows the scene; A2d ownership boundary)
    /// — it becomes a `&self` method once `AppCore` owns the scene in Phase B/C.
    pub fn resolve_scene(scene: &Scene, voxels_per_block: u32) -> VoxelGrid {
        let region = scene.full_extent_blocks(voxels_per_block);
        scene.resolve_region(region, voxels_per_block, 0)
    }

    /// The region dimensions (in voxels) the camera auto-frame, origin gizmo, block
    /// lattice, fine floor grid and layer scrubber are sized from — read from the
    /// SCENE, not by reaching into the assembled `VoxelGrid` (issue #20 S6c-1, prep
    /// for the per-chunk renderer of S6c step 4). This is a behaviour-preserving
    /// substitution: for a chunkable scene (every Tool scene, including the startup
    /// default) the assembled grid is literally sized to
    /// [`Scene::placed_region_dimensions`] — so this returns BYTE-IDENTICAL
    /// dimensions (proven in
    /// `scene::tests::placed_region_dimensions_equals_assembled_grid`).
    ///
    /// A **Part-only** scene (e.g. a lone debug-cloud field) has no composite
    /// extent, so `placed_region_dimensions` would be `[0, 0, 0]`; that scene is
    /// resolved through the explicit-region path instead, so we fall back to the
    /// assembled grid's own dimensions — which (being the grid the consumers used
    /// before) is trivially identical to the old behaviour for that case.
    pub fn region_dimensions_for(scene: &Scene, density: u32, grid: &VoxelGrid) -> [u32; 3] {
        if scene.has_chunkable_extent(density) {
            scene.placed_region_dimensions(density)
        } else {
            grid.dimensions
        }
    }

    /// The camera's view-projection matrix for the given viewport aspect ratio —
    /// the recentred-frame matrix every overlay + the voxel pass draw with. A
    /// `&self` getter (it reads the owned camera) so the shell and `shot` source the
    /// frame matrix identically.
    pub fn view_projection(&self, aspect_ratio: f32) -> glam::Mat4 {
        self.camera.view_projection(aspect_ratio)
    }

    /// Where the transform gizmo (issue #29 S2) should sit: the SELECTED node's
    /// recentred pivot + its extent (in voxels), or `None` when nothing is selected
    /// (or the selection has no extent). An associated function for now (it borrows
    /// the scene; A2d ownership boundary) — becomes `&self` once `AppCore` owns the
    /// scene in Phase B/C.
    pub fn gizmo_placement(scene: &Scene, density: u32) -> Option<([f32; 3], [f32; 3])> {
        scene.active_gizmo_placement(density)
    }

    /// Build the onion-skin fog parameters (issue #12) from the camera-derived
    /// view-projection, grid, and layer-range scrubber. World-Y of layer `j` spans
    /// `[j - grid_y/2, j+1 - grid_y/2]` (voxel centres at `j + 0.5 - grid_y/2`). The
    /// solid band is layers `[lower, upper]`; the onion band extends `onion_depth`
    /// layers on each side.
    pub fn onion_fog_params(
        view_projection: glam::Mat4,
        grid_dimensions: [u32; 3],
        layer_range: LayerRange,
    ) -> OnionFogParams {
        let grid_y = grid_dimensions[1] as f32;
        let half_y = grid_y / 2.0;
        let depth = layer_range.onion_depth.clamp(1, 8) as f32;
        let lower = layer_range.lower as f32;
        let upper = layer_range.upper.min(grid_dimensions[1].saturating_sub(1)) as f32;
        OnionFogParams {
            inverse_view_projection: view_projection.inverse(),
            semi_axes: [
                grid_dimensions[0] as f32 / 2.0,
                grid_dimensions[1] as f32 / 2.0,
                grid_dimensions[2] as f32 / 2.0,
            ],
            // Onion band world-Y: `depth` layers below the band's bottom edge to
            // `depth` layers above its top edge.
            onion_y_min: (lower - depth) - half_y,
            onion_y_max: (upper + 1.0 + depth) - half_y,
            // Solid band world-Y (excluded from the fog).
            band_y_min: lower - half_y,
            band_y_max: (upper + 1.0) - half_y,
        }
    }
}
