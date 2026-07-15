//! The shell's palette-interaction seams: applying a clicked VS block variant (resolving its
//! per-face textures into a bound 6-layer material) and reacting to this frame's palette
//! `PanelResponse` (apply / connect-folder / revert-to-procedural / export). Split out of
//! `windowed/mod.rs` (ADR 0016).

use super::*;

impl WindowedState {
    /// Apply palette interactions from this frame's [`PanelResponse`] (M6):
    /// applying a block loads + binds its texture; "Connect folder…" opens the OS
    /// picker and starts a custom scan; selecting a procedural material clears the
    /// applied block.
    pub(super) fn handle_palette_response(&mut self, response: &crate::PanelResponse) {
        if response.selected_procedural_material {
            self.loaded_material = None;
            self.panel_state.applied_block_label = None;
        }
        if let Some(tile_index) = response.clicked_palette_tile {
            if let Some(variant_path) = self.palette.pick_variant(tile_index) {
                self.apply_block_variant(&variant_path, tile_index);
            }
        }
        if response.clicked_connect_folder {
            if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                // Reset the palette + any in-flight scan state, then start a fresh
                // scan of the picked folder.
                self.palette.clear();
                self.pending_groups.clear();
                self.scan_total = None;
                self.scan_source_name = None;
                self.palette.ui.status = "Scanning folder…".to_string();
                // Re-point the M7 face resolver at the same folder.
                self.face_resolver = FaceResolver::custom_folder(folder.clone());
                self.scan_handle = Some(spawn_custom_folder_scan(folder));
            }
        }
        if response.clicked_export_vox {
            self.export_vox();
        }
    }

    /// Resolve `variant_path`'s per-face textures (M7) and bind the 6-layer
    /// material. Uniform blocks resolve to the same PNG on all faces (the M6
    /// path); per-face blocks (e.g. a log: end-grain top, bark sides) bind each
    /// face's own PNG.
    fn apply_block_variant(&mut self, variant_path: &std::path::Path, tile_index: usize) {
        let Some(tile) = self.palette.ui.tiles.get(tile_index) else {
            return;
        };
        let label = tile.label.clone();
        let Some(group) = self.palette.group(tile_index) else {
            return;
        };
        let faces = self.face_resolver.resolve(group, variant_path);
        self.loaded_material = Some(LoadedMaterial::from_faces(
            &self.gpu.device,
            &self.gpu.queue,
            self.display.cuboid_mesh_renderer().material_bind_group_layout(),
            self.display.cuboid_mesh_renderer().material_sampler(),
            &faces,
            label.clone(),
        ));
        self.panel_state.applied_block_label = Some(label);
    }
}
