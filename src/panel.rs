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

/// Procedural material choice. Stored only; it has NO visual effect in M3 — the
/// cubes stay flat-shaded. It will drive the M4 texture-slice shader.
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
#[derive(Debug, Clone, Copy, Default)]
pub struct PanelState {
    /// Rebuild-driving geometry params.
    pub geometry: GeometryParams,
    /// Camera projection (display-only: no rebuild).
    pub projection_mode: ProjectionMode,
    /// Material selection (display-only, inert in M3: no rebuild, no visual).
    pub material: MaterialChoice,
    /// When `Some`, the 3D rebuild was skipped because the grid exceeds the
    /// voxel cap; the panel shows a warning. Set by the caller after it decides
    /// whether to rebuild. Value is the would-be voxel count (in millions).
    pub voxel_cap_warning_millions: Option<f32>,
}

/// What changed during a [`build_panel`] call, so the caller can react.
#[derive(Debug, Clone, Copy, Default)]
pub struct PanelResponse {
    /// A geometry param changed → re-resolve the grid + rebuild instances.
    pub geometry_changed: bool,
    /// Size or density specifically changed → also auto-frame the camera.
    /// (Shape change re-resolves but must NOT move the camera — guard #1.)
    pub size_or_density_changed: bool,
}

/// Build the right-hand side panel into the root [`egui::Ui`] of the frame.
///
/// Returns a [`PanelResponse`] describing what the user changed this frame.
pub fn build_panel(root_ui: &mut egui::Ui, state: &mut PanelState) -> PanelResponse {
    let mut response = PanelResponse::default();

    egui::Panel::right("voxel_worker_controls")
        .resizable(false)
        .default_size(280.0)
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
            build_material_section(ui, state);
            build_display_placeholder(ui);

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

/// Material selector (display-only, inert in M3: stores the choice, no visual).
fn build_material_section(ui: &mut egui::Ui, state: &mut PanelState) {
    ui.add_space(8.0);
    ui.strong("Material");
    ui.horizontal(|ui| {
        ui.selectable_value(&mut state.material, MaterialChoice::Stone, "Stone");
        ui.selectable_value(&mut state.material, MaterialChoice::Wood, "Wood");
        ui.selectable_value(&mut state.material, MaterialChoice::Plain, "Plain");
    });
    ui.label(
        egui::RichText::new("Not applied yet — drives the M4 texture shader.")
            .small()
            .weak(),
    );
    ui.separator();
}

/// Display section — intentionally a "coming soon" placeholder so there are no
/// dead controls (these toggles become functional in M4/M5).
fn build_display_placeholder(ui: &mut egui::Ui) {
    ui.add_space(8.0);
    ui.strong("Display");
    ui.label(
        egui::RichText::new("Voxel grid · lattice · floor · view cube · gizmo — coming in M4/M5.")
            .small()
            .weak(),
    );
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
