//! Headless render-data queries — region dims, view-projection, gizmo placement and
//! onion-skin params: the data the windowed shell + `shot` render from ([`AppCore`]).

use ui::panel::{LayerRange, ViewMode};
use display::renderer::{LayerBand, OnionFogParams, RegionClip, RegionRole};
use document::scene::{NodeId, Scene};

use super::AppCore;

/// The mesh/brick layer clip for a frame, region-scoped per ADR 0018 Decision 5. Bundles
/// the effective [`LayerBand`] (scene-absolute layers), the optional [`RegionClip`] the
/// band is confined to (the selected object's placed AABB, recentred voxels — `None` for a
/// scene-wide band / no clip), and the layer-track domain the UI scrubber spans.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshClip {
    /// The band the mesh/brick path clips to (scene-absolute Z-layer indices).
    pub band: LayerBand,
    /// The region the band is confined to (`role = ConfineBand`), or `None`.
    pub region: Option<RegionClip>,
    /// The layer-track length for the UI scrubber: the selected object's Z extent in
    /// Onion-fog mode with a selection, else the whole-scene `grid_z`.
    pub track_len: u32,
}

impl AppCore {
    /// Resolve the whole [`Scene`] into a fresh grid (ADR 0001 step 2). Every
    /// visible node composites (union) into one region sized to the per-axis max of
    /// the nodes' extents, at full resolution (`lod 0`). `voxels_per_block` is the
    /// global app density (the inspector mirror's density).
    ///
    /// ADR 0010 E5: this streams the whole-region grid from the **two-layer evaluator**
    /// (coarse fast-fill + boundary per-voxel), NOT the retired dense
    /// `Scene::resolve_region` — bit-identical (the E2 round-trip parity gate). A VoxelBody-only
    /// scene (no covering range) resolves to an empty grid, exactly as the dense store did.
    ///
    /// The startup region door — the SINGLE place the windowed shell seeds its first-frame
    /// display frame from (`WindowedState::new`). ADR 0011 G5: with the dense grid retired
    /// this constructs NO `VoxelGrid` at all — it returns only the region dimensions + the
    /// resolve recentre (the camera auto-frame, layer scrubber and fog frame consume these),
    /// exactly what the per-edit [`AppCore::rebuild`] yields. This is what closes the startup
    /// OOM on BOTH binaries: a persisted 8000×800×800 scene once resolved a dense
    /// ~5.1-billion-cell grid (~28.5 GB RSS → OOM hang before the first print), and the non-gpu
    /// binary streamed the same region; now neither materialises any occupancy at startup.
    pub fn startup_region(scene: &Scene, density: u32) -> ([u32; 3], [i64; 3]) {
        (
            scene.placed_region_dimensions(density),
            scene.recentre_voxels_for_resolve(density).voxels(),
        )
    }

    /// The region dimensions (in voxels) the camera auto-frame, origin gizmo, block
    /// lattice, fine floor grid and layer scrubber are sized from — read purely from the
    /// SCENE (issue #20 S6c-1). ADR 0011 G5: with the dense grid retired there is no
    /// assembled `VoxelGrid` to reach into, so this is just
    /// [`Scene::placed_region_dimensions`]. For a chunkable scene (every Tool scene,
    /// including the startup default) that is the composite extent (proven byte-identical to
    /// the old assembled grid in `scene::tests::placed_region_dimensions_equals_assembled_grid`);
    /// a **VoxelBody-only** scene (a lone debug-cloud field) has no composite extent, so this is
    /// `[0, 0, 0]` — exactly the empty grid's dimensions the old VoxelBody-only fallback returned.
    pub fn region_dimensions_for(scene: &Scene, density: u32) -> [u32; 3] {
        scene.placed_region_dimensions(density)
    }

    /// The camera's view-projection matrix for the given viewport aspect ratio —
    /// the recentred-frame matrix every overlay + the voxel pass draw with. A
    /// `&self` getter (it reads the owned camera) so the shell and `shot` source the
    /// frame matrix identically.
    ///
    /// `region_dimensions` is the resolved grid extent (voxels). The recentre
    /// centres the composite on the render-frame origin (Fit/Home both target
    /// `Vec3::ZERO`), so the scene's bounding sphere is `centre = ORIGIN`,
    /// `radius = ½·diagonal` (with a small margin for the integer-recentre's
    /// sub-voxel asymmetry and a floor for tiny scenes). The camera derives its
    /// near/far from that sphere so no part of the scene is ever depth-clipped.
    pub fn view_projection(&self, aspect_ratio: f32, region_dimensions: [u32; 3]) -> glam::Mat4 {
        let diagonal = glam::Vec3::new(
            region_dimensions[0] as f32,
            region_dimensions[1] as f32,
            region_dimensions[2] as f32,
        )
        .length();
        let scene_radius = (0.5 * diagonal * 1.15).max(1.0);
        self.camera
            .view_projection(aspect_ratio, glam::Vec3::ZERO, scene_radius)
    }

    /// Where the transform gizmo (issue #29 S2) should sit: the SELECTED node's
    /// recentred pivot + its extent (in voxels), or `None` when nothing is selected
    /// (or the selection has no extent). An associated function for now (it borrows
    /// the scene; A2d ownership boundary) — becomes `&self` once `AppCore` owns the
    /// scene in Phase B/C.
    pub fn gizmo_placement(scene: &Scene, density: u32) -> Option<([f32; 3], [f32; 3])> {
        scene.active_gizmo_placement(density)
    }

    /// The recentred `(pivot_voxels, extent_voxels)` for an ARBITRARY node id (not
    /// the active selection) — the camera "Focus" view action frames that node. A
    /// thin wrapper over [`Scene::gizmo_placement_for_id`]; `None` when the id no
    /// longer resolves or the node has no extent (Focus is then a no-op).
    pub fn gizmo_placement_for_id(
        scene: &Scene,
        node_id: NodeId,
        density: u32,
    ) -> Option<([f32; 3], [f32; 3])> {
        scene.gizmo_placement_for_id(node_id, density)
    }

    /// The region-scoped layer clip for a frame (ADR 0018 Decisions 4–5) — the SINGLE
    /// place both the windowed shell and `shot` derive the mesh/brick band + region from,
    /// so the two never drift. The band clips ONLY in **Onion-fog mode with a selection**;
    /// Normal / Show-booleans (and Onion-fog with nothing selected, or a debug-face render)
    /// render the whole scene finished (band FULL, no region — the pre-ADR-0018 scene-wide
    /// band clip is retired).
    ///
    /// In Onion-fog the scrubber's `lower`/`upper` are **object-relative** layer indices
    /// over the selected object's Z extent (Decision 5: the track spans the object, not the
    /// scene); this offsets them by the object's base layer into scene-absolute band indices
    /// and derives the recentred-voxel region the band is confined to. Selecting the ROOT
    /// part gives the whole-scene region (the pre-0018 behaviour recovered).
    pub fn mesh_clip(
        scene: &Scene,
        density: u32,
        view_mode: ViewMode,
        layer_range: LayerRange,
        scene_grid_z: u32,
        debug_face_orientation: bool,
    ) -> MeshClip {
        let finished = MeshClip {
            band: LayerBand::FULL,
            region: None,
            track_len: scene_grid_z,
        };
        // Debug-face mode + any non-onion mode render the whole model finished.
        if debug_face_orientation || view_mode != ViewMode::OnionFog {
            return finished;
        }
        // Onion-fog needs a selected object to scope the clip to. No selection / hidden /
        // empty subtree ⇒ finished (no implicit whole-scene clip — ADR 0018 Decision 2/5).
        let Some((rmin, rmax)) = scene.selected_region_extent_recentred_voxels(density) else {
            return finished;
        };
        // The mesher maps a recentred voxel-Z `v` to absolute layer `v + floor(dim_z/2)`.
        let half_z = (scene_grid_z / 2) as i64;
        // The object's bottom layer in scene-absolute layer indices.
        let obj_base_layer = rmin[2] + half_z;
        let track_len = (rmax[2] - rmin[2]).max(0) as u32;
        // Object-relative scrubber handles, clamped into the object's track.
        let lower_obj = layer_range.lower.min(track_len);
        let upper_obj = layer_range.upper.min(track_len);
        let onion_depth = if layer_range.onion_skin {
            layer_range.onion_depth.clamp(1, 8)
        } else {
            0
        };
        // A full-object band with no ghost is a no-op clip ⇒ render finished (and skip the
        // needless per-block densify the region path would do).
        if lower_obj == 0 && upper_obj >= track_len && onion_depth == 0 {
            return finished;
        }
        // Scene-absolute band (`upper` is the last visible layer; the region confines the
        // off-by-one at the object top and both are clamped into the grid).
        let band_min = (obj_base_layer + lower_obj as i64).clamp(0, scene_grid_z as i64) as u32;
        let band_max = (obj_base_layer + upper_obj as i64)
            .clamp(0, scene_grid_z.saturating_sub(1) as i64) as u32;
        let region = RegionClip {
            min: rmin,
            max: rmax,
            role: RegionRole::ConfineBand,
        };
        MeshClip {
            band: LayerBand {
                band_min,
                band_max,
                onion_depth,
            },
            region: Some(region),
            track_len,
        }
    }

    /// Build the onion-skin frame parameters (issue #12) from the camera-derived
    /// view-projection, grid, and layer-range scrubber — the recentred-Z spans the display
    /// paths' ghost pass derives its onion slabs from (ADR 0012; the volumetric fog that once
    /// consumed these is retired). Z-up: layers are Z-slices, so
    /// the band is a Z-range. Corner-anchoring: the grid's low corner in the recentred
    /// frame is `−floor(dim/2)`, so layer `k` has its voxel centre at
    /// `k + 0.5 − floor(grid_z/2)` and spans world-Z `[k − floor(grid_z/2),
    /// k+1 − floor(grid_z/2)]`. The solid band is layers `[lower, upper]`; the onion
    /// band extends `onion_depth` layers on each side.
    pub fn onion_fog_params(
        view_projection: glam::Mat4,
        grid_dimensions: [u32; 3],
        layer_range: LayerRange,
    ) -> OnionFogParams {
        // FLOORED half (`(dim/2) as f32`) throughout, for a frame CONSISTENT with the
        // corner-anchored voxels: the grid's low corner in the recentred frame is
        // `−floor(dim/2)`, so the layer→world-Z conversion AND the ellipsoid `semi_axes`
        // (which bounds the voxel volume `[−floor(dim/2), −floor(dim/2)+dim)`) must both
        // use the floored half. (`dim/2.0` would put the ghost-fog ellipsoid ½ voxel off
        // the voxels at an ODD dim; even-density goldens are unaffected either way.)
        let half_x = (grid_dimensions[0] / 2) as f32;
        let half_y = (grid_dimensions[1] / 2) as f32;
        let half_z = (grid_dimensions[2] / 2) as f32;
        let depth = layer_range.onion_depth.clamp(1, 8) as f32;
        let lower = layer_range.lower as f32;
        // Z-up: the layer band is along Z (index 2).
        let upper = layer_range.upper.min(grid_dimensions[2].saturating_sub(1)) as f32;
        OnionFogParams {
            inverse_view_projection: view_projection.inverse(),
            semi_axes: [half_x, half_y, half_z],
            // Onion band world-Z: `depth` layers below the band's bottom edge to
            // `depth` layers above its top edge.
            onion_z_min: (lower - depth) - half_z,
            onion_z_max: (upper + 1.0 + depth) - half_z,
            // Solid band world-Z (excluded from the fog).
            band_z_min: lower - half_z,
            band_z_max: (upper + 1.0) - half_z,
        }
    }
}
