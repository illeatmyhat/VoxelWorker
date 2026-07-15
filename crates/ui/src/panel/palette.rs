//! The bottom palette dock (M6) and the shared shape-chip table.

use super::PanelResponse;
use crate::palette::BlockPalette;
use voxel_core::voxel::ShapeKind;

/// The palette dock (M6): a status line, a "Connect folder…" button, and a
/// scrollable grid of cube-thumbnail tiles. Clicking a tile applies a
/// pseudo-random variant; the dock sits along the bottom of the window.
pub(super) fn build_palette_dock(
    root_ui: &mut egui::Ui,
    palette: &BlockPalette,
    response: &mut PanelResponse,
) {
    egui::Panel::bottom("voxel_worker_palette")
        .resizable(false)
        .default_size(190.0)
        .show_inside(root_ui, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.strong("Blocks");
                ui.add_space(8.0);
                if ui.button("Connect folder…").clicked() {
                    response.clicked_connect_folder = true;
                }
                ui.add_space(8.0);
                ui.label(egui::RichText::new(&palette.status).small().weak());
            });
            ui.separator();

            // Each tile: the 96px cube thumbnail + "Label ·N" beneath it.
            const TILE_IMAGE: f32 = 72.0;
            egui::ScrollArea::horizontal()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        for (index, tile) in palette.tiles.iter().enumerate() {
                            let caption = if tile.variant_count > 1 {
                                format!("{} ·{}", tile.label, tile.variant_count)
                            } else {
                                tile.label.clone()
                            };
                            let clicked = ui
                                .vertical(|ui| {
                                    ui.set_width(TILE_IMAGE + 8.0);
                                    let image = egui::Image::new((
                                        tile.thumbnail_id,
                                        egui::vec2(TILE_IMAGE, TILE_IMAGE),
                                    ))
                                    .sense(egui::Sense::click());
                                    let hit = ui.add(image).on_hover_text(&caption).clicked();
                                    ui.label(
                                        egui::RichText::new(caption).small().weak(),
                                    );
                                    hit
                                })
                                .inner;
                            if clicked {
                                response.clicked_palette_tile = Some(index);
                            }
                        }
                    });
                });
        });
}

/// The shape chips, in panel order.
pub(super) const SHAPE_CHIPS: &[(ShapeKind, &str)] = &[
    (ShapeKind::Cylinder, "Cylinder"),
    (ShapeKind::Tube, "Tube"),
    (ShapeKind::Sphere, "Sphere"),
    (ShapeKind::Torus, "Torus"),
    (ShapeKind::Box, "Box"),
];
