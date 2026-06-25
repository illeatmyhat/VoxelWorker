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
use crate::voxel::ShapeKind;

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
    /// Debug producer override: when set, the grid is filled by the debug cloud
    /// field (`DebugCloudField`) instead of the parametric `shape` SDF — several
    /// distinct billowy blobs for exercising the renderer / onion fog. `shape`,
    /// `size_blocks` and `voxels_per_block` still set the grid dimensions.
    pub debug_clouds: bool,
}

impl Default for GeometryParams {
    fn default() -> Self {
        Self {
            shape: ShapeKind::Cylinder,
            size_blocks: [5, 1, 5],
            voxels_per_block: 16,
            wall_blocks: 1,
            debug_clouds: false,
        }
    }
}

/// Procedural material choice. Selects which procedural texture (Stone/Wood/
/// Plain) binds in the M4 texture-slice shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum MaterialChoice {
    #[default]
    Stone,
    Wood,
    Plain,
}

/// Layer-range scrubber state (issue #12).
///
/// The layer-range scrubber subsumes the old 2D mid-Y slice map. Layers run along
/// **Y** (height). `lower`/`upper` are voxel Y-layer indices selected on a track
/// `0..grid_y`; the visible band is layers `[lower, upper]` INCLUSIVE on both ends
/// (so `lower == upper` shows a single layer). Default = the full range.
///
/// When `snap_to_blocks` is on, the handles snap to multiples of
/// `voxels_per_block` (plus the endpoints `0` and `grid_y`); a narrowed
/// single-layer band viewed from the top is the chisel stencil. `onion_skin`
/// ghosts up to `onion_depth` layers on each side of the band (3D screen-door).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerRange {
    /// Lower handle: the first visible layer index (`0..=grid_y`).
    pub lower: u32,
    /// Upper handle: the last visible layer index (`lower..=grid_y`).
    pub upper: u32,
    /// Snap the handles to block boundaries (multiples of `voxels_per_block`).
    pub snap_to_blocks: bool,
    /// Show ghosted neighbour layers around the band (3D onion skin).
    pub onion_skin: bool,
    /// How many layers on each side of the band to ghost (1..=8).
    pub onion_depth: u32,
}

impl Default for LayerRange {
    fn default() -> Self {
        // Full range over the default cylinder grid_y (1 block × 16 density = 16).
        // The real bounds are clamped/rescaled to the live grid on first rebuild
        // and whenever grid_y changes (see `LayerRange::rescale_to_grid_y`).
        Self {
            lower: 0,
            upper: 16,
            snap_to_blocks: true,
            onion_skin: false,
            onion_depth: 2,
        }
    }
}

impl LayerRange {
    /// Snap a layer index to the nearest block boundary, keeping the endpoints
    /// `0` and `grid_y` exact (they are always valid snap points even when
    /// `grid_y` is not a clean multiple of the density, which it always is here).
    pub fn snap_value(value: u32, voxels_per_block: u32, grid_y: u32) -> u32 {
        let step = voxels_per_block.max(1);
        if value >= grid_y {
            return grid_y;
        }
        let snapped = ((value + step / 2) / step) * step;
        snapped.min(grid_y)
    }

    /// Clamp/rescale the bounds to a (possibly new) `grid_y`. Called on every
    /// geometry rebuild: when `grid_y` shrinks the handles are clamped in; the
    /// default full-range state widens to the new top. Re-snaps to block
    /// multiples when snapping is on so the band keeps landing on boundaries.
    pub fn rescale_to_grid_y(&mut self, previous_grid_y: u32, grid_y: u32, voxels_per_block: u32) {
        // A band that spanned the whole previous grid stays "full" on the new one.
        let was_full = self.lower == 0 && self.upper >= previous_grid_y;
        if was_full || previous_grid_y == 0 {
            self.lower = 0;
            self.upper = grid_y;
        } else {
            self.lower = self.lower.min(grid_y);
            self.upper = self.upper.min(grid_y);
        }
        if self.snap_to_blocks {
            self.lower = Self::snap_value(self.lower, voxels_per_block, grid_y);
            self.upper = Self::snap_value(self.upper, voxels_per_block, grid_y);
        }
        if self.lower > self.upper {
            std::mem::swap(&mut self.lower, &mut self.upper);
        }
        self.onion_depth = self.onion_depth.clamp(1, 8);
    }

    /// Whether this band covers the whole grid (so the 3D render is unclipped).
    pub fn is_full_range(&self, grid_y: u32) -> bool {
        self.lower == 0 && self.upper >= grid_y
    }
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
    /// Whether the block lattice (box lattice at block boundaries) is drawn (M8
    /// Display toggle, ON by default — matches the prototype `showLattice`).
    pub show_block_lattice: bool,
    /// Whether the fine floor grid (bottom-plane grid) is drawn (M8 Display
    /// toggle, OFF by default — matches the prototype `showFloor`).
    pub show_floor_grid: bool,
    /// Whether the voxel cubes render in face-orientation debug mode (colour by
    /// outward face normal + a back-facing marker, cull off). Display toggle, OFF
    /// by default; the standard way to verify face winding/culling.
    pub debug_face_orientation: bool,
    /// When `Some`, the 3D rebuild was skipped because the grid exceeds the
    /// voxel cap; the panel shows a warning. Set by the caller after it decides
    /// whether to rebuild. Value is the would-be voxel count (in millions).
    pub voxel_cap_warning_millions: Option<f32>,
    /// When `Some`, a loaded VS block (M6) is the active material; the value is
    /// its label, shown under the Material selector. `None` = a procedural
    /// material is active.
    pub applied_block_label: Option<String>,
    /// Layer-range scrubber state (issue #12): the visible band along Y plus the
    /// snap/onion controls. Bounds are clamped/rescaled to the grid on rebuild.
    pub layer_range: LayerRange,
}

impl PanelState {
    /// Sensible defaults for the windowed app: like [`Default`] but with the view
    /// cube enabled (prototype `showCube: true`).
    pub fn with_view_cube_default() -> Self {
        Self {
            show_view_cube: true,
            // Block lattice defaults ON (prototype `showLattice: true`); the fine
            // floor grid defaults OFF (`showFloor: false`).
            show_block_lattice: true,
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
    /// The "Export .vox" button was clicked this frame → open the OS save dialog
    /// and write the resolved grid as a MagicaVoxel `.vox` file (M8).
    pub clicked_export_vox: bool,
}

/// Build the right-hand side panel into the root [`egui::Ui`] of the frame.
///
/// `grid_y` is the current grid height in voxels (the layer-scrubber track spans
/// `0..grid_y`); `measured_diameter` is the widest occupied voxel run in the
/// active band (`grid.widest_run_in_band`), shown as a small stat line. Returns a
/// [`PanelResponse`] describing what the user changed this frame.
pub fn build_panel(
    root_ui: &mut egui::Ui,
    state: &mut PanelState,
    grid_y: u32,
    measured_diameter: u32,
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
            build_export_section(ui, &mut response);
            build_layers_section(ui, state, grid_y, measured_diameter);

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
/// touches the size or the camera (Milestone 3 guard #1). The trailing "Clouds"
/// chip is a debug producer: it swaps the SDF for the [`DebugCloudField`] at the
/// same grid size, and is mutually exclusive with the SDF chips.
fn build_shape_section(ui: &mut egui::Ui, state: &mut PanelState, response: &mut PanelResponse) {
    ui.add_space(8.0);
    ui.strong("Shape");
    ui.horizontal_wrapped(|ui| {
        for (kind, label) in SHAPE_CHIPS {
            // An SDF chip is selected only when the debug producer is off.
            let is_selected = !state.geometry.debug_clouds && state.geometry.shape == *kind;
            if ui.selectable_label(is_selected, *label).clicked() && !is_selected {
                state.geometry.shape = *kind;
                state.geometry.debug_clouds = false;
                response.geometry_changed = true;
                // Deliberately NOT setting size_or_density_changed: a shape
                // switch re-resolves at the same size and must not auto-frame.
            }
        }

        // Debug cloud field chip (separate visual group).
        ui.separator();
        let clouds_selected = state.geometry.debug_clouds;
        if ui
            .selectable_label(clouds_selected, "Clouds")
            .on_hover_text("Debug: distinct billowy blobs (fBm noise) instead of an SDF shape")
            .clicked()
            && !clouds_selected
        {
            state.geometry.debug_clouds = true;
            response.geometry_changed = true;
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

/// Display section. M4 added the voxel-grid overlay; M5 wired the view cube and
/// the origin gizmo; M8 wires the block lattice and fine floor grid (#10).
fn build_display_section(ui: &mut egui::Ui, state: &mut PanelState) {
    ui.add_space(8.0);
    ui.strong("Display");
    ui.checkbox(&mut state.show_grid_overlay, "Voxel grid on faces");
    ui.checkbox(&mut state.show_block_lattice, "Block lattice");
    ui.checkbox(&mut state.show_floor_grid, "Fine floor grid");
    ui.checkbox(&mut state.show_view_cube, "View cube");
    ui.checkbox(&mut state.show_origin_gizmo, "Origin gizmo");
    ui.checkbox(&mut state.debug_face_orientation, "Debug: face orientation");
    ui.separator();
}

/// Export section (M8): a single "Export .vox" button. The click is reported via
/// [`PanelResponse::clicked_export_vox`]; the caller opens the OS save dialog and
/// writes the resolved grid (so the panel stays free of file-system concerns).
fn build_export_section(ui: &mut egui::Ui, response: &mut PanelResponse) {
    ui.add_space(8.0);
    ui.strong("Export");
    if ui
        .button("Export .vox")
        .on_hover_text("Write the resolved voxels as a MagicaVoxel .vox file")
        .clicked()
    {
        response.clicked_export_vox = true;
    }
    ui.separator();
}

/// The Layers section (issue #12): the layer-range scrubber that subsumes the old
/// 2D mid-Y slice map. A video-clip-style track over `0..grid_y` with two trim
/// handles (lower/upper), the selected band highlighted, block-boundary ticks,
/// the layers/blocks readout, the snap + onion controls, and the measured-
/// diameter stat line (widest occupied voxel run in the active band).
fn build_layers_section(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    grid_y: u32,
    measured_diameter: u32,
) {
    ui.add_space(8.0);
    ui.strong("Layers");

    let voxels_per_block = state.geometry.voxels_per_block.max(1);
    // The scrubber edits `state.layer_range` in place; the bounds are kept valid
    // (clamped to grid_y, lower <= upper, snapped if requested) by the widget.
    layer_scrubber(ui, &mut state.layer_range, grid_y, voxels_per_block);

    let range = state.layer_range;
    // Readout: "layers L–U of N · blocks b0–b1".
    let block_lower = range.lower / voxels_per_block;
    let block_upper = range.upper.saturating_sub(1).max(range.lower) / voxels_per_block;
    ui.label(
        egui::RichText::new(format!(
            "layers {}–{} of {grid_y} · blocks {block_lower}–{block_upper}",
            range.lower, range.upper
        ))
        .small()
        .weak(),
    );

    ui.add_space(4.0);
    ui.checkbox(&mut state.layer_range.snap_to_blocks, "Snap to blocks");
    if state.layer_range.snap_to_blocks {
        // Re-snap the current handles immediately so toggling snap on tidies them.
        state.layer_range.lower =
            LayerRange::snap_value(state.layer_range.lower, voxels_per_block, grid_y);
        state.layer_range.upper =
            LayerRange::snap_value(state.layer_range.upper, voxels_per_block, grid_y);
        if state.layer_range.lower > state.layer_range.upper {
            std::mem::swap(&mut state.layer_range.lower, &mut state.layer_range.upper);
        }
    }
    ui.checkbox(&mut state.layer_range.onion_skin, "Onion skin");
    if state.layer_range.onion_skin {
        let mut depth = state.layer_range.onion_depth.clamp(1, 8);
        if ui
            .add(egui::Slider::new(&mut depth, 1..=8).text("onion depth"))
            .changed()
        {
            state.layer_range.onion_depth = depth;
        }
    }

    // Measured-diameter stat line: the widest occupied voxel run in the active
    // band (the chisel-diameter readout the old 2D slice carried).
    let blocks = measured_diameter as f32 / voxels_per_block as f32;
    ui.label(
        egui::RichText::new(format!("Ø {measured_diameter} vx · {blocks:.2} bl"))
            .small()
            .weak(),
    );
    ui.separator();
}

/// Custom range-scrubber widget (issue #12). Paints a track spanning `0..grid_y`
/// with block-boundary ticks, the selected band highlighted, and two draggable
/// trim handles (lower/upper). Drag is handled via `ui.interact` + the pointer:
/// the nearer handle to the press grabs, then follows the pointer (snapped to
/// block boundaries when `snap_to_blocks` is on). Keeps `lower <= upper` by
/// swapping when the handles cross. Edits `range` in place.
fn layer_scrubber(
    ui: &mut egui::Ui,
    range: &mut LayerRange,
    grid_y: u32,
    voxels_per_block: u32,
) {
    let grid_y = grid_y.max(1);
    let track_height = 26.0;
    let handle_half_width = 5.0;
    let desired = egui::vec2(ui.available_width(), track_height + 14.0);
    let (rect, response) = ui.allocate_exact_size(desired, egui::Sense::click_and_drag());

    // The track is inset so the handles have room at both ends.
    let track_left = rect.left() + handle_half_width + 2.0;
    let track_right = rect.right() - handle_half_width - 2.0;
    let track_width = (track_right - track_left).max(1.0);
    let track_top = rect.top() + 4.0;
    let track_bottom = track_top + track_height;
    let track_rect = egui::Rect::from_min_max(
        egui::pos2(track_left, track_top),
        egui::pos2(track_right, track_bottom),
    );

    // Map a layer index <-> an x pixel on the track.
    let layer_to_x = |layer: u32| -> f32 {
        track_left + (layer as f32 / grid_y as f32) * track_width
    };
    let x_to_layer = |x: f32| -> u32 {
        let t = ((x - track_left) / track_width).clamp(0.0, 1.0);
        (t * grid_y as f32).round() as u32
    };

    let painter = ui.painter_at(rect);
    let visuals = ui.visuals();

    // Track background.
    painter.rect_filled(track_rect, 3.0, egui::Color32::from_rgb(0x1b, 0x17, 0x12));

    // Block-boundary tick marks every `voxels_per_block` layers (the snap points).
    let mut boundary = 0u32;
    while boundary <= grid_y {
        let x = layer_to_x(boundary);
        painter.line_segment(
            [egui::pos2(x, track_top), egui::pos2(x, track_bottom)],
            egui::Stroke::new(1.0, egui::Color32::from_rgb(0x3a, 0x5f, 0x57)),
        );
        if boundary == grid_y {
            break;
        }
        boundary = (boundary + voxels_per_block).min(grid_y);
        if boundary == grid_y {
            // Draw the final endpoint tick then stop.
            let x = layer_to_x(grid_y);
            painter.line_segment(
                [egui::pos2(x, track_top), egui::pos2(x, track_bottom)],
                egui::Stroke::new(1.0, egui::Color32::from_rgb(0x3a, 0x5f, 0x57)),
            );
            break;
        }
    }

    // Selected band highlight between the handles.
    let lower_x = layer_to_x(range.lower);
    let upper_x = layer_to_x(range.upper);
    let band_rect = egui::Rect::from_min_max(
        egui::pos2(lower_x.min(upper_x), track_top),
        egui::pos2(lower_x.max(upper_x), track_bottom),
    );
    painter.rect_filled(band_rect, 0.0, egui::Color32::from_rgba_unmultiplied(0x5f, 0xb8, 0xa4, 70));

    // Drag handling: on press, grab whichever handle is nearer the pointer; while
    // dragging, that handle follows the pointer.
    if response.drag_started() || (response.clicked() && response.hover_pos().is_some()) {
        if let Some(pos) = response.interact_pointer_pos() {
            let dist_lower = (pos.x - lower_x).abs();
            let dist_upper = (pos.x - upper_x).abs();
            // Stash which handle is active in egui temp memory keyed by widget id.
            let active_upper = dist_upper < dist_lower;
            ui.memory_mut(|m| m.data.insert_temp(response.id, active_upper));
        }
    }
    if response.dragged() || response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let active_upper = ui
                .memory(|m| m.data.get_temp::<bool>(response.id))
                .unwrap_or_else(|| {
                    (pos.x - upper_x).abs() < (pos.x - lower_x).abs()
                });
            let mut value = x_to_layer(pos.x);
            if range.snap_to_blocks {
                value = LayerRange::snap_value(value, voxels_per_block, grid_y);
            }
            if active_upper {
                range.upper = value;
            } else {
                range.lower = value;
            }
            if range.lower > range.upper {
                std::mem::swap(&mut range.lower, &mut range.upper);
                // The active handle effectively swapped sides; update the memory so
                // continued dragging keeps tracking the same pointer.
                ui.memory_mut(|m| m.data.insert_temp(response.id, !active_upper));
            }
        }
    }

    // Draw the two handles last so they sit on top of the band.
    let handle_color = visuals.widgets.active.fg_stroke.color;
    for layer in [range.lower, range.upper] {
        let x = layer_to_x(layer);
        let handle_rect = egui::Rect::from_min_max(
            egui::pos2(x - handle_half_width, track_top - 3.0),
            egui::pos2(x + handle_half_width, track_bottom + 3.0),
        );
        painter.rect_filled(handle_rect, 2.0, handle_color);
        painter.rect_stroke(
            handle_rect,
            2.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(0x10, 0x0c, 0x08)),
            egui::StrokeKind::Inside,
        );
    }
}

/// The shape chips, in panel order.
const SHAPE_CHIPS: &[(ShapeKind, &str)] = &[
    (ShapeKind::Cylinder, "Cylinder"),
    (ShapeKind::Tube, "Tube"),
    (ShapeKind::Sphere, "Sphere"),
    (ShapeKind::Torus, "Torus"),
    (ShapeKind::Box, "Box"),
];
