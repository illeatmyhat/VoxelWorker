//! The shared egui side panel.
//!
//! Exactly one implementation, used by both the windowed app and the headless
//! screenshot harness (Hard requirement #3), so the captured frame is identical
//! to the live one.
//!
//! Milestone 3 makes the panel functional: shape chips, size/density/wall
//! sliders, the camera projection toggle, and an inert material selector. The
//! parameters are split by *consumer* (Milestone 3 hard requirement #3):
//!
//!   * [`GeometryParams`] (shape, size, density, wall) drive a **rebuild-dirty**
//!     flag. Changing them re-resolves the voxel grid.
//!   * Display/camera params (projection, material selection) live in
//!     [`PanelState`] directly and never trigger a voxel rebuild.
//!
//! This split is what enforces the regression guards: selecting a shape only
//! sets [`GeometryParams::shape`] (never the size or the camera), and changing
//! density only sets [`GeometryParams::voxels_per_block`] (never the block size).

use crate::block_palette::BlockPalette;
use crate::camera::ProjectionMode;
use crate::voxel::{ShapeKind, SliceImage};

/// Geometry parameters — the *only* params that trigger a voxel rebuild.
///
/// Sizes are in **whole blocks**; `voxels_per_block` is fineness only and never
/// changes the object's block size (DATA.md "the density bug").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeometryParams {
    /// Selected primitive.
    pub shape: ShapeKind,
    /// Bounding-box size in whole blocks (X, Y, Z).
    pub size_blocks: [u32; 3],
    /// Voxels per block (chisel fineness). Default 16.
    pub voxels_per_block: u32,
    /// Tube wall thickness in whole blocks (used by [`ShapeKind::Tube`] only).
    pub wall_blocks: u32,
}

impl Default for GeometryParams {
    fn default() -> Self {
        Self {
            shape: ShapeKind::Cylinder,
            size_blocks: [5, 1, 5],
            voxels_per_block: 16,
            wall_blocks: 1,
        }
    }
}

/// Procedural material choice. Selects which procedural texture (Stone/Wood/
/// Plain) binds in the M4 texture-slice shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MaterialChoice {
    #[default]
    Stone,
    Wood,
    Plain,
}

/// Mutable UI state passed to [`build_panel`].
///
/// Holds the geometry params (rebuild-driving) and the display/camera params
/// (no rebuild). The binaries own one of these and feed it to the panel each
/// frame; [`PanelResponse`] tells them what changed.
#[derive(Debug, Clone, Default)]
pub struct PanelState {
    /// Rebuild-driving geometry params.
    pub geometry: GeometryParams,
    /// Camera projection (display-only: no rebuild).
    pub projection_mode: ProjectionMode,
    /// Material selection (display-only: selects the M4 procedural texture).
    pub material: MaterialChoice,
    /// Whether the voxel/block grid overlay is drawn (M4 Display toggle).
    pub show_grid_overlay: bool,
    /// Whether the corner view cube is drawn (M5 Display toggle, ON by default).
    pub show_view_cube: bool,
    /// Whether the origin gizmo is drawn (M5 Display toggle, OFF by default).
    pub show_origin_gizmo: bool,
    /// When `Some`, the 3D rebuild was skipped because the grid exceeds the
    /// voxel cap; the panel shows a warning. Set by the caller after it decides
    /// whether to rebuild. Value is the would-be voxel count (in millions).
    pub voxel_cap_warning_millions: Option<f32>,
    /// When `Some`, a loaded VS block (M6) is the active material; the value is
    /// its label, shown under the Material selector. `None` = a procedural
    /// material is active.
    pub applied_block_label: Option<String>,
}

impl PanelState {
    /// Sensible defaults for the windowed app: like [`Default`] but with the view
    /// cube enabled (prototype `showCube: true`).
    pub fn with_view_cube_default() -> Self {
        Self {
            show_view_cube: true,
            ..Self::default()
        }
    }
}

/// What changed during a [`build_panel`] call, so the caller can react.
#[derive(Debug, Clone, Copy, Default)]
pub struct PanelResponse {
    /// A geometry param changed → re-resolve the grid + rebuild instances.
    pub geometry_changed: bool,
    /// Size or density specifically changed → also auto-frame the camera.
    /// (Shape change re-resolves but must NOT move the camera — guard #1.)
    pub size_or_density_changed: bool,
    /// A palette tile was clicked this frame → apply a pseudo-random variant of
    /// this tile index as the active loaded material (M6).
    pub clicked_palette_tile: Option<usize>,
    /// The "Connect folder…" button was clicked → open the OS folder picker and
    /// scan the chosen folder via `CustomFolderSource` (M6).
    pub clicked_connect_folder: bool,
    /// A built-in procedural material (Stone/Wood/Plain) was selected this frame →
    /// clear any applied loaded block and revert to the procedural material (M6).
    pub selected_procedural_material: bool,
}

/// Build the right-hand side panel into the root [`egui::Ui`] of the frame.
///
/// `slice` is the freshly-built 2D mid-Y slice map (M5); it is shown as a
/// nearest-filtered egui image. Returns a [`PanelResponse`] describing what the
/// user changed this frame.
pub fn build_panel(
    root_ui: &mut egui::Ui,
    state: &mut PanelState,
    slice: &SliceImage,
    palette: &BlockPalette,
) -> PanelResponse {
    let mut response = PanelResponse::default();

    // The palette dock lives along the bottom (prototype layout); it is its own
    // bottom panel so the right-hand controls keep their width.
    build_palette_dock(root_ui, palette, &mut response);

    egui::Panel::right("voxel_worker_controls")
        .resizable(false)
        .default_size(300.0)
        .show_inside(root_ui, |ui| {
            ui.add_space(8.0);
            ui.heading("VoxelWorker");
            ui.label("Vintage Story chiseling planner");
            ui.add_space(6.0);
            ui.separator();

            build_shape_section(ui, state, &mut response);
            build_size_section(ui, state, &mut response);
            build_density_section(ui, state, &mut response);
            build_camera_section(ui, state);
            build_material_section(ui, state, &mut response);
            build_display_section(ui, state);
            build_slice_section(ui, state, slice);

            if let Some(millions) = state.voxel_cap_warning_millions {
                ui.add_space(8.0);
                ui.separator();
                ui.colored_label(
                    egui::Color32::from_rgb(0xd9, 0x60, 0x3f),
                    format!("3D paused — {millions:.1}M voxels; lower size/density"),
                );
            }
        });

    response
}

/// The palette dock (M6): a status line, a "Connect folder…" button, and a
/// scrollable grid of cube-thumbnail tiles. Clicking a tile applies a
/// pseudo-random variant; the dock sits along the bottom of the window.
fn build_palette_dock(
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

/// Shape chips. Selecting a shape sets [`GeometryParams::shape`] ONLY — it never
/// touches the size or the camera (Milestone 3 guard #1).
fn build_shape_section(ui: &mut egui::Ui, state: &mut PanelState, response: &mut PanelResponse) {
    ui.add_space(8.0);
    ui.strong("Shape");
    ui.horizontal_wrapped(|ui| {
        for (kind, label) in SHAPE_CHIPS {
            let is_selected = state.geometry.shape == *kind;
            if ui.selectable_label(is_selected, *label).clicked() && !is_selected {
                state.geometry.shape = *kind;
                response.geometry_changed = true;
                // Deliberately NOT setting size_or_density_changed: a shape
                // switch re-resolves at the same size and must not auto-frame.
            }
        }
    });
    ui.separator();
}

/// Size sliders (whole blocks). Each shows the resulting voxel extent as a hint.
fn build_size_section(ui: &mut egui::Ui, state: &mut PanelState, response: &mut PanelResponse) {
    ui.add_space(8.0);
    ui.strong("Size (blocks)");

    let density = state.geometry.voxels_per_block;
    for (axis_index, axis_label) in ["X", "Y", "Z"].iter().enumerate() {
        let mut value = state.geometry.size_blocks[axis_index];
        let slider = egui::Slider::new(&mut value, 1..=16).text(*axis_label);
        if ui.add(slider).changed() {
            state.geometry.size_blocks[axis_index] = value;
            response.geometry_changed = true;
            response.size_or_density_changed = true;
        }
        let voxel_extent = value * density;
        ui.label(
            egui::RichText::new(format!("{value} blocks · {voxel_extent} vx"))
                .small()
                .weak(),
        );
    }

    // Conditional wall row — Tube only.
    if state.geometry.shape == ShapeKind::Tube {
        ui.add_space(4.0);
        let mut wall = state.geometry.wall_blocks;
        let slider = egui::Slider::new(&mut wall, 1..=8).text("Wall");
        if ui.add(slider).changed() {
            state.geometry.wall_blocks = wall;
            response.geometry_changed = true;
            response.size_or_density_changed = true;
        }
        ui.label(
            egui::RichText::new(format!("{wall} block wall"))
                .small()
                .weak(),
        );
    }
    ui.separator();
}

/// Density slider. Changes fineness ONLY — never the block size (guard #2).
fn build_density_section(ui: &mut egui::Ui, state: &mut PanelState, response: &mut PanelResponse) {
    ui.add_space(8.0);
    ui.strong("Density");
    let mut density = state.geometry.voxels_per_block;
    let slider = egui::Slider::new(&mut density, 2..=32).text("vx/block");
    if ui.add(slider).changed() {
        state.geometry.voxels_per_block = density;
        response.geometry_changed = true;
        response.size_or_density_changed = true;
    }
    ui.separator();
}

/// Camera projection toggle (display-only: no rebuild).
fn build_camera_section(ui: &mut egui::Ui, state: &mut PanelState) {
    ui.add_space(8.0);
    ui.strong("Camera → Projection");
    ui.horizontal(|ui| {
        ui.selectable_value(
            &mut state.projection_mode,
            ProjectionMode::Perspective,
            "Perspective",
        );
        ui.selectable_value(
            &mut state.projection_mode,
            ProjectionMode::Orthographic,
            "Orthographic",
        );
    });
    ui.separator();
}

/// Material selector: selects which procedural texture binds (M4). Selecting any
/// procedural material clears an applied loaded VS block (M6) and reverts to it.
fn build_material_section(ui: &mut egui::Ui, state: &mut PanelState, response: &mut PanelResponse) {
    ui.add_space(8.0);
    ui.strong("Material");
    ui.horizontal(|ui| {
        for (choice, label) in [
            (MaterialChoice::Stone, "Stone"),
            (MaterialChoice::Wood, "Wood"),
            (MaterialChoice::Plain, "Plain"),
        ] {
            if ui.selectable_value(&mut state.material, choice, label).clicked() {
                response.selected_procedural_material = true;
            }
        }
    });
    if let Some(applied) = &state.applied_block_label {
        ui.label(
            egui::RichText::new(format!("Applied: {applied}"))
                .small()
                .weak(),
        );
    }
    ui.separator();
}

/// Display section. M4 added the voxel-grid overlay; M5 wires the view cube and
/// the origin gizmo. (Lattice/floor are deferred — see the M5 report.)
fn build_display_section(ui: &mut egui::Ui, state: &mut PanelState) {
    ui.add_space(8.0);
    ui.strong("Display");
    ui.checkbox(&mut state.show_grid_overlay, "Voxel grid overlay");
    ui.checkbox(&mut state.show_view_cube, "View cube");
    ui.checkbox(&mut state.show_origin_gizmo, "Origin gizmo");
    ui.label(
        egui::RichText::new("Lattice · floor — deferred.")
            .small()
            .weak(),
    );
    ui.separator();
}

/// The 2D slice section (M5): the mid-Y layer of the resolved grid, shown as a
/// nearest-filtered egui image scaled to the panel width, plus the measured
/// widest run for round shapes ("Ø N vx · N.NN bl").
fn build_slice_section(ui: &mut egui::Ui, state: &mut PanelState, slice: &SliceImage) {
    ui.add_space(8.0);
    ui.strong("Slice (mid-layer)");

    let [width, height] = slice.size;
    if width == 0 || height == 0 || slice.rgba.len() != (width * height * 4) as usize {
        ui.label(
            egui::RichText::new("no slice")
                .small()
                .weak(),
        );
        ui.separator();
        return;
    }

    // Rebuild the egui texture from the CPU image every frame the grid changes;
    // `load_texture` re-uploads under the same name so the handle stays stable.
    let color_image = egui::ColorImage::from_rgba_unmultiplied(
        [width as usize, height as usize],
        &slice.rgba,
    );
    let texture = ui.ctx().load_texture(
        "voxel_worker_slice",
        color_image,
        egui::TextureOptions::NEAREST,
    );

    // Scale to fit the panel width, keeping the slice's aspect ratio; crisp
    // (nearest) so individual voxels stay square.
    let available_width = ui.available_width();
    let aspect = height as f32 / width as f32;
    let display_size = egui::vec2(available_width, available_width * aspect);
    ui.add(egui::Image::new(&texture).fit_to_exact_size(display_size));

    // Measured diameter readout for round shapes (prototype `updateStats`).
    let round = matches!(
        state.geometry.shape,
        ShapeKind::Cylinder | ShapeKind::Tube | ShapeKind::Sphere | ShapeKind::Torus
    );
    if round {
        let density = state.geometry.voxels_per_block.max(1) as f32;
        let blocks = slice.widest_run_voxels as f32 / density;
        ui.label(
            egui::RichText::new(format!(
                "Ø {} vx · {:.2} bl",
                slice.widest_run_voxels, blocks
            ))
            .small()
            .weak(),
        );
    }
    ui.separator();
}

/// The shape chips, in panel order.
const SHAPE_CHIPS: &[(ShapeKind, &str)] = &[
    (ShapeKind::Cylinder, "Cylinder"),
    (ShapeKind::Tube, "Tube"),
    (ShapeKind::Sphere, "Sphere"),
    (ShapeKind::Torus, "Torus"),
    (ShapeKind::Box, "Box"),
];
