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
use crate::panel::LayerRange;
use crate::renderer::OnionFogParams;
use crate::scene::Scene;
use crate::store::Store;
use crate::voxel::VoxelGrid;

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
}

impl AppCore {
    /// Assemble the headless core from an already-constructed store + camera. The
    /// shell builds both (the store seeds the startup diameter readout, the camera
    /// restores persisted orbit/projection) and hands them over here.
    pub fn new(store: Store, camera: OrbitCamera) -> Self {
        Self { store, camera }
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
