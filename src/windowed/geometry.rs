//! The shell's per-edit GPU geometry rebuild: it delegates the headless resolve to
//! [`AppCore::rebuild`] and consumes the output into the display orchestrator + camera
//! recenter compensation + layer-band rescale.

use super::*;

impl WindowedState {
    /// Re-resolve the grid + GPU geometry for the current scene. Camera UX change:
    /// this NEVER moves the camera — edits keep the orbit target + distance fixed.
    /// Explicit framing (startup fit, Home/Fit, Focus) is handled by their own paths.
    pub(super) fn rebuild_geometry(&mut self) {
        profiling::scope!("rebuild_geometry");
        let density = self.panel_state.geometry.voxels_per_block;

        // Delegate the headless resolve (targeted invalidation + assemble) to
        // `AppCore::rebuild`, then consume its output here in the shell: build the
        // GPU cuboid mesh (the camera is NOT touched). A density whose
        // single-chunk voxel capacity exceeds the bound is rejected with the store
        // untouched, so we surface the cap warning and bail.
        let chunkable = self.panel_state.scene.has_chunkable_extent(density);
        let RebuildOutput {
            region_dimensions,
            two_layer_chunks,
            recenter_voxels,
            recenter_shift_voxels,
            incremental_dirty_chunks,
        } = match self.app_core.rebuild(&self.panel_state.scene, density) {
            RebuildOutcome::DensityRejected {
                chunk_voxels_millions,
            } => {
                self.panel_state.voxel_cap_warning_millions = Some(chunk_voxels_millions);
                return;
            }
            RebuildOutcome::Built(output) => {
                self.panel_state.voxel_cap_warning_millions = None;
                output
            }
        };

        // Read the OLD grid_z before reassigning `self.region_dimensions`, for the layer-band
        // rescale below (Z-up: layers are Z-slices, index 2).
        let previous_grid_z = self.region_dimensions[2];
        let grid_dimensions = region_dimensions;
        // The effective layer-clip band the render path will apply this frame. The async
        // worker builds the mesh already clipped to this band so the swap frame's
        // `rebuild_for_band` is a no-op, sparing a full main-thread re-mesh.
        let clip = self.current_mesh_clip(grid_dimensions[2]);
        // Keep this rebuild's resident chunks (and the band they were drawn at) so the
        // per-frame placement resolve has a `PickFrame` to march. The clone is Arc refcount
        // bumps; the originals move into the display orchestrator below.
        self.resident_chunks = two_layer_chunks.clone();
        self.last_pick_band = clip.band;
        // Delegate the display-artifact rebuild (the brick sink + the fallback cuboid mesh +
        // the F1 brick-display handover reconcile) to the orchestrator. The shell keeps the
        // camera recenter-shift compensation, the layer-band rescale, and the region /
        // measurement bookkeeping below. The async mesh build region-scopes via
        // `clip.region`.
        self.display.rebuild(
            two_layer_chunks,
            incremental_dirty_chunks,
            chunkable,
            grid_dimensions,
            recenter_voxels,
            density,
            clip.band,
            clip.region,
            self.panel_state.debug_face_orientation,
        );

        // Camera UX invariant: an edit must NEVER re-frame the view. The composite is
        // re-centered on the world origin every rebuild, so any extent change (add /
        // delete / offset) — and any density change, since the recenter is in voxels —
        // shifts the floating origin by `recenter_shift_voxels`. The camera target is
        // pinned in that same recentered render frame (voxels), so without compensation
        // the whole world would slide under the fixed camera (the "jump to center /
        // fit everything" the user reported). Subtract the shift so the target tracks
        // the SAME world point as the origin floats — net zero view motion. The shift
        // is `[0,0,0]` on the first build, and the explicit Fit/Home/Focus actions
        // OVERWRITE the target afterwards (they run on their own paths, not here), so
        // they keep re-framing exactly as before; orbit/pan/zoom are untouched.
        if recenter_shift_voxels != [0; 3] {
            self.app_core.camera.target -= glam::Vec3::new(
                recenter_shift_voxels[0] as f32,
                recenter_shift_voxels[1] as f32,
                recenter_shift_voxels[2] as f32,
            );
        }
        // Clamp/rescale the layer band to the new grid_z (re-snapping to block
        // multiples when snapping is on) so the render path draws the rescaled band this frame
        // (the render path runs after this rebuild returns and reads the rescaled band).
        // Z-up: index 2. `previous_grid_z` was captured before `grid` was reassigned.
        self.panel_state.layer_range.rescale_to_grid_z(
            previous_grid_z,
            region_dimensions[2],
            density,
        );

        // Onion skin is a per-frame ghost pass on the display paths (no occupancy
        // build/upload here) — a band scrub is a pure uniform update.

        // The transform gizmo is sized + positioned from the SELECTED node in the per-frame
        // render path (it must track selection changes, which don't trigger a geometry
        // rebuild), not here. The per-object block lattice + floor grid is likewise
        // (re)batched per frame from the grid-enabled nodes — a per-node toggle needs no
        // scene re-resolve.

        self.region_dimensions = region_dimensions;
        self.recenter_voxels = recenter_voxels;
        self.measured_band = (u32::MAX, u32::MAX); // force a re-measure next frame.
    }
}
