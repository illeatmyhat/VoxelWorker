//! The shell's `.vox` export dispatch: the save dialog + palette-colour assembly stay on the
//! main thread, but the multi-second [`TwoLayerStore`] build + streaming resolve + serialise +
//! write move to the background [`VoxExportWorker`]. The `.vox` palette helpers live here beside
//! their only caller. Split out of `windowed/mod.rs` (ADR 0016).

use super::*;

/// The per-`block_id` `.vox` palette over the three procedural materials (ADR 0003
/// §3a): slot `material_id` carries that material's average colour, so a multi-material
/// scene exports each block in its own colour.
fn vox_export_procedural_palette() -> interchange::vox_export::BlockPaletteColors {
    use voxel_core::core_geom::MaterialChoice;
    let mut palette = [[0u8; 4]; MaterialChoice::MATERIAL_COUNT];
    for (slot, color) in palette.iter_mut().enumerate() {
        *color = procedural_material_average_color(MaterialChoice::from_material_id(slot as u16));
    }
    palette
}

/// Default `.vox` filename from the shape + voxel dims (e.g. `cylinder_80x16x80.vox`).
fn default_vox_filename(shape: &SdfShape, voxels_per_block: u32) -> String {
    let [grid_x, grid_y, grid_z] = shape.grid_dimensions(voxels_per_block);
    let kind = format!("{:?}", shape.kind).to_lowercase();
    format!("{kind}_{grid_x}x{grid_y}x{grid_z}.vox")
}

impl WindowedState {
    /// Open the `.vox` save dialog and DISPATCH the export to the background worker
    /// (slow-paths item 2 — the build + write used to run inline here and freeze the UI
    /// for the whole multi-second export). The default filename encodes the shape + voxel
    /// dims (e.g. `cylinder_80x16x80.vox`). The palette colour is the active material's
    /// representative colour (a loaded block's average, or the procedural one), computed
    /// here on the main thread exactly as before.
    ///
    /// The dialog (a native modal, not the slow part) stays on this thread; everything
    /// after it — [`TwoLayerStore`] build, streaming resolve, serialise, write — moves to
    /// the [`VoxExportWorker`]. The button is disabled while `export_outstanding`, so this
    /// can't be re-entered mid-export (the worker carries no supersede generation — an
    /// export is a user-chosen file — so the shell serialises instead; see
    /// `workers::export`). The completion/failure readout lands in `poll_vox_export_worker`.
    pub(super) fn export_vox(&mut self) {
        // Single-flight invariant (depth-correct guard): only ONE export may be in flight.
        // The export button is disabled while `export_outstanding`, but guard the dispatch
        // seam too — a second queued export would be silently drain-to-latest-dropped by
        // the worker (an export is a user-chosen file, never superseded; see
        // `workers::export`). Bail before even opening the save dialog.
        if self.export_outstanding {
            return;
        }
        let density = self.panel_state.geometry.voxels_per_block;
        let shape = SdfShape::from_geometry(self.panel_state.geometry.clone());
        // ADR 0010 E4: the old `exceeds_voxel_cap` guard (the dense whole-region 6M
        // ceiling) is GONE on the export path — the streaming export never materialises
        // a dense interior, so an 800×800-revolve-class solid exports successfully. A
        // pathological per-CHUNK density is still bounded by the resolver itself.

        let representative = match &self.loaded_material {
            Some(loaded) => loaded.average_color,
            None => procedural_material_average_color(self.panel_state.material),
        };
        // ADR 0003 §3a: map each categorical `block_id` to its colour. The palette is
        // the three procedural materials' colours; the ACTIVE material's slot takes the
        // representative (a loaded VS block's average, when applied), so a single-active-
        // material scene exports byte-identically to the old single-colour `.vox`.
        let mut palette_colors = vox_export_procedural_palette();
        palette_colors[self.panel_state.material.material_id() as usize] = representative;

        let default_name = default_vox_filename(&shape, density);
        let Some(path) = rfd::FileDialog::new()
            .set_file_name(default_name)
            .add_filter("MagicaVoxel", &["vox"])
            .save_file()
        else {
            return;
        };

        // Size the progress denominator (covering chunks) + a large-export warning WITHOUT
        // resolving any occupancy — the worker's per-chunk counter counts up to exactly
        // this total (the streaming build ingests one covering chunk at a time; ADR 0010
        // E4). `0` for a Part-only / empty scene (still exports a valid empty `.vox`).
        let total_chunks = self.panel_state.scene.covering_chunk_count(density);
        let region_dimensions = self.panel_state.scene.placed_region_dimensions(density);
        // Large-export warning (non-blocking text, NOT a modal): the user's 8000³ scene is
        // ~1.95M covering chunks; a small model is hundreds. Above the threshold, warn that
        // the dispatched export may take a while and produce a large file.
        const LARGE_EXPORT_CHUNK_THRESHOLD: u64 = 100_000;
        self.export_status = (total_chunks > LARGE_EXPORT_CHUNK_THRESHOLD).then(|| {
            let [width, height, depth] = region_dimensions;
            format!(
                "Large export dispatched: {width}×{height}×{depth} voxels — this may take a \
                 while and produce a large file"
            )
        });

        // Clone the scene out of the document and hand the whole build to the worker. The
        // shell keeps a clone of the progress counter to read each frame; `export_outstanding`
        // disables the button so a second export can't be queued (drain-to-latest would drop
        // it — an export must never be silently superseded; see `workers::export`).
        let progress_chunks = Arc::new(std::sync::atomic::AtomicU64::new(0));
        self.export_progress = Some((Arc::clone(&progress_chunks), total_chunks));
        self.export_outstanding = true;
        self.vox_export_worker.dispatch(VoxExportRequest {
            scene: self.panel_state.scene.clone(),
            density,
            palette_colors,
            path,
            progress_chunks,
        });
    }
}
