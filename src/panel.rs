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
use crate::scene::{DefId, Node, NodeContent, NodePath, Part, Scene};
use crate::voxel::{SdfShape, ShapeKind};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum MaterialChoice {
    #[default]
    Stone,
    Wood,
    Plain,
}

impl MaterialChoice {
    /// The number of distinct procedural materials (Stone/Wood/Plain). The
    /// renderer's per-voxel base-colour uniform array is sized to this, and a
    /// `material_id` is always `< MATERIAL_COUNT`.
    pub const MATERIAL_COUNT: usize = 3;

    /// The per-voxel `material_id` this choice stamps onto its voxels (ADR 0001
    /// step 3 "Materials"). Stable, dense (`0..MATERIAL_COUNT`), so it indexes both
    /// the renderer's base-colour uniform array and the procedural-texture table.
    /// Stone = 0, Wood = 1, Plain = 2.
    pub fn material_id(self) -> u16 {
        match self {
            MaterialChoice::Stone => 0,
            MaterialChoice::Wood => 1,
            MaterialChoice::Plain => 2,
        }
    }

    /// The inverse of [`material_id`](Self::material_id): the choice for a stamped
    /// id. Ids outside the known set fall back to [`Stone`](Self::Stone).
    pub fn from_material_id(id: u16) -> Self {
        match id {
            0 => MaterialChoice::Stone,
            1 => MaterialChoice::Wood,
            2 => MaterialChoice::Plain,
            _ => MaterialChoice::Stone,
        }
    }
}

/// Which render path draws the voxels (ADR 0002 E3, part of #18). `Cuboid` is now
/// the DEFAULT (Vintage-Story-style box decomposition, ~37× fewer primitives, full
/// feature + multi-material parity per ADR 0002 E3c); `Instanced` is the legacy
/// one-cube-per-voxel path, kept fully working behind `--mesher instanced` / the
/// panel toggle as a debug fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum MesherChoice {
    /// The legacy instanced-cube renderer (one cube per voxel), kept as a fallback.
    Instanced,
    /// The cuboid mesh path — now the default (ADR 0002 E3c-2).
    #[default]
    Cuboid,
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
    /// The scene (ADR 0001): the flat node list that is now the panel's source of
    /// truth. The node list section adds/selects/deletes nodes; the inspector
    /// edits the ACTIVE node. [`geometry`](Self::geometry) / [`material`](Self::material)
    /// are the inspector's working mirror of the active Tool node (synced both
    /// ways) so the renderer/export call sites that read voxel dims + density keep
    /// working unchanged.
    pub scene: Scene,
    /// Rebuild-driving geometry params — the inspector's editing surface, mirrored
    /// onto the active Tool node (and re-read from it when the selection changes).
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
    /// Which render path draws the voxels (ADR 0002 E3, part of #18). Default
    /// [`MesherChoice::Instanced`] (the unchanged instanced-cube renderer);
    /// [`MesherChoice::Cuboid`] selects the experimental cuboid mesh path. Not
    /// persisted yet (a session-only experimental toggle).
    pub mesher: MesherChoice,
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
        let mut state = Self {
            show_view_cube: true,
            // Block lattice defaults ON (prototype `showLattice: true`); the fine
            // floor grid defaults OFF (`showFloor: false`).
            show_block_lattice: true,
            ..Self::default()
        };
        state.seed_scene_from_geometry();
        state
    }

    /// Seed the scene with a single Tool node from the current geometry/material
    /// mirror (the back-compat path: a default or a config-loaded one-geometry
    /// state becomes a one-Tool-node scene). Does nothing if the scene already has
    /// nodes.
    pub fn seed_scene_from_geometry(&mut self) {
        if self.scene.nodes.is_empty() {
            self.scene = Scene::from_geometry(self.geometry, self.material);
        }
        // issue #29 (grid rework S1): every scene carries exactly one Origin Point.
        // Idempotent, so calling it on an already-seeded scene is a no-op.
        self.scene.ensure_origin_point();
    }

    /// Copy the active node's parameters into the inspector mirror
    /// ([`geometry`](Self::geometry) / [`material`](Self::material)) when it is a
    /// Tool, so the inspector edits the active selection. Called whenever the
    /// active node changes (selection or delete). A Part active node leaves the
    /// mirror untouched (its editor shows name + seed instead).
    pub fn sync_mirror_from_active(&mut self) {
        if let Some(node) = self.scene.active_node() {
            if let NodeContent::Tool { shape, material } = &node.content {
                self.geometry = GeometryParams {
                    shape: shape.kind,
                    size_blocks: shape.size_blocks,
                    voxels_per_block: shape.voxels_per_block,
                    wall_blocks: shape.wall_blocks,
                };
                self.material = *material;
            }
        }
    }

    /// Write the inspector mirror back onto the active node when it is a Tool
    /// (shape/size/wall/material edits target the active selection). Density is
    /// global, but it is stored on every Tool's shape so the resolve reads it; the
    /// mirror's `voxels_per_block` is propagated here. No-op for a Part active node
    /// or an empty scene.
    pub fn write_mirror_to_active(&mut self) {
        let geometry = self.geometry;
        let material = self.material;
        if let Some(node) = self.scene.active_node_mut() {
            if let NodeContent::Tool { shape, material: node_material } = &mut node.content {
                *shape = SdfShape::from_geometry(geometry);
                *node_material = material;
            }
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
    /// The scene's node list changed this frame (a node was added, deleted, or the
    /// active selection switched) → the caller re-resolves the scene and re-frames
    /// as it would for a geometry change. Distinct from
    /// [`geometry_changed`](Self::geometry_changed) (an inspector edit), though the
    /// caller treats both as "rebuild".
    pub scene_changed: bool,
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

            build_node_list_section(ui, state, &mut response);
            build_inspector_section(ui, state, &mut response);
            build_camera_section(ui, state);
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

/// A label for a node row, switching on its content kind. Falls back to the
/// content-kind name when the node's own name is empty.
fn node_row_label(node: &Node) -> String {
    let kind = match &node.content {
        NodeContent::Tool { shape, .. } => format!("{:?}", shape.kind),
        NodeContent::Part(Part::DebugClouds { .. }) => "Clouds".to_string(),
        NodeContent::Group(children) => format!("Group ({})", children.len()),
        NodeContent::Instance(_) => "Instance".to_string(),
    };
    if node.name.is_empty() {
        kind
    } else {
        format!("{} · {}", node.name, kind)
    }
}

/// The scene node list (ADR 0001 step 4): the assembly rendered as an INDENTED
/// TREE so [`Group`](NodeContent::Group) children are visible and selectable at
/// any depth (not just top-level nodes). Each row carries a visibility checkbox, a
/// selectable name (indented by depth), and a per-row delete ✕. Beneath the tree:
///
///   * **+ Add** — append a Tool (any shape) or a Clouds Part at top level.
///   * **Group** — wrap the active node in a new Group (then add children to it
///     via "+ Add child").
///   * **+ Add child** — when the active node is a Group, append a Tool/Part into
///     it.
///   * **Make definition** — turn the active Group/node into a reusable
///     [`AssemblyDef`] and replace it with an `Instance` of it.
///   * a **Definitions** list with an **Add instance** button per definition (the
///     village workflow: one stored body, many placements).
///
/// Selecting any node (at any depth) makes it the inspector's active node.
fn build_node_list_section(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    response: &mut PanelResponse,
) {
    ui.add_space(8.0);
    ui.strong("Scene");

    let mut select: Option<NodePath> = None;
    let mut delete: Option<NodePath> = None;
    let mut visibility_toggled = false;

    // Walk the tree depth-first; each row is indented by its depth.
    let rows = state.scene.tree_rows();
    for (path, depth) in &rows {
        let is_active = state.scene.active.as_ref() == Some(path);
        // Read the node by path; mutate visibility in place via a separate lookup
        // so the borrow of `nodes` does not span the whole row.
        let label = match state.scene.node_at_path(path) {
            Some(node) => node_row_label(node),
            None => continue,
        };
        ui.horizontal(|ui| {
            ui.add_space(*depth as f32 * 14.0);
            if let Some(node) = state.scene.node_at_path_mut(path) {
                if ui
                    .checkbox(&mut node.visible, "")
                    .on_hover_text("Visible")
                    .changed()
                {
                    visibility_toggled = true;
                }
            }
            if ui.selectable_label(is_active, label).clicked() {
                select = Some(path.clone());
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("✕").on_hover_text("Delete node").clicked() {
                    delete = Some(path.clone());
                }
            });
        });
    }

    if state.scene.nodes.is_empty() {
        ui.label(egui::RichText::new("(no nodes — add one below)").small().weak());
    }

    if visibility_toggled {
        response.scene_changed = true;
    }

    build_node_actions(ui, state, response);
    build_definitions_section(ui, state, response);

    // Apply the deferred selection / delete after the walk (can't mutate the
    // active path while borrowing through the tree).
    if let Some(path) = delete {
        state.scene.remove_node(&path);
        state.sync_mirror_from_active();
        response.scene_changed = true;
    } else if let Some(path) = select {
        if state.scene.active.as_ref() != Some(&path) {
            state.scene.active = Some(path);
            state.sync_mirror_from_active();
            response.scene_changed = true;
        }
    }

    ui.separator();
}

/// A fresh Tool node of the given shape, inheriting the current size/density so it
/// renders immediately.
fn new_tool_node(kind: ShapeKind, label: &str, state: &PanelState) -> Node {
    let shape = SdfShape {
        kind,
        size_blocks: state.geometry.size_blocks,
        voxels_per_block: state.geometry.voxels_per_block,
        wall_blocks: state.geometry.wall_blocks,
    };
    Node::new(label, NodeContent::Tool { shape, material: state.material })
}

/// Build the action buttons under the tree: **+ Add** (top-level), **+ Add child**
/// (into the active Group), **Group** (wrap the active node), and **Make
/// definition** (turn the active node into a reusable [`AssemblyDef`] + Instance).
fn build_node_actions(ui: &mut egui::Ui, state: &mut PanelState, response: &mut PanelResponse) {
    ui.add_space(4.0);

    // Whether the active node is a Group (gates "+ Add child").
    let active_is_group = matches!(
        state.scene.active_node().map(|node| &node.content),
        Some(NodeContent::Group(_))
    );
    let has_active = state.scene.active.is_some();

    ui.horizontal_wrapped(|ui| {
        // + Add — a top-level Tool or Clouds Part.
        ui.menu_button("+ Add", |ui| {
            for (kind, label) in SHAPE_CHIPS {
                if ui.button(*label).clicked() {
                    let node = new_tool_node(*kind, label, state);
                    state.scene.add_node(node);
                    state.sync_mirror_from_active();
                    response.scene_changed = true;
                    ui.close();
                }
            }
            ui.separator();
            if ui.button("Clouds (Part)").clicked() {
                state
                    .scene
                    .add_node(Node::new("Clouds", NodeContent::Part(Part::DebugClouds { seed: 0 })));
                response.scene_changed = true;
                ui.close();
            }
        });

        // + Add child — into the active Group (only shown when one is selected).
        if active_is_group {
            let group_path = state.scene.active.clone();
            ui.menu_button("+ Add child", |ui| {
                for (kind, label) in SHAPE_CHIPS {
                    if ui.button(*label).clicked() {
                        let node = new_tool_node(*kind, label, state);
                        if let Some(path) = &group_path {
                            state.scene.add_child_to_group(path, node);
                            state.sync_mirror_from_active();
                            response.scene_changed = true;
                        }
                        ui.close();
                    }
                }
                ui.separator();
                if ui.button("Clouds (Part)").clicked() {
                    if let Some(path) = &group_path {
                        state.scene.add_child_to_group(
                            path,
                            Node::new("Clouds", NodeContent::Part(Part::DebugClouds { seed: 0 })),
                        );
                        response.scene_changed = true;
                    }
                    ui.close();
                }
            });
        }

        // Group — wrap the active node in a new Group.
        if ui
            .add_enabled(has_active, egui::Button::new("Group"))
            .on_hover_text("Wrap the selected node in a new Group")
            .clicked()
        {
            state.scene.group_active();
            state.sync_mirror_from_active();
            response.scene_changed = true;
        }

        // Make definition — the active node becomes a reusable AssemblyDef and is
        // replaced by an Instance of it.
        if ui
            .add_enabled(has_active, egui::Button::new("Make definition"))
            .on_hover_text("Turn the selected Group/node into a reusable definition, placed by an Instance")
            .clicked()
        {
            let def_name = state
                .scene
                .active_node()
                .map(|node| {
                    if node.name.is_empty() {
                        "Definition".to_string()
                    } else {
                        node.name.clone()
                    }
                })
                .unwrap_or_else(|| "Definition".to_string());
            state.scene.make_definition_from_active(def_name);
            state.sync_mirror_from_active();
            response.scene_changed = true;
        }
    });
}

/// The **Definitions** list (ADR 0001 step 4): the reusable [`AssemblyDef`]s, each
/// with an **Add instance** button that places another `Instance` of it at a
/// nudged offset (the village workflow: one stored body placed at several offsets).
fn build_definitions_section(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    response: &mut PanelResponse,
) {
    if state.scene.definitions.is_empty() {
        return;
    }
    ui.add_space(6.0);
    ui.strong("Definitions");

    // Collect (id, label) first so the per-row button can mutate the scene without
    // borrowing `definitions` across the click.
    let defs: Vec<(DefId, String)> = state
        .scene
        .definitions
        .iter()
        .map(|def| {
            let label = if def.name.is_empty() {
                format!("Def {}", def.id.0)
            } else {
                def.name.clone()
            };
            (def.id, format!("{} ({} node)", label, def.children.len()))
        })
        .collect();

    let mut add_instance_of: Option<DefId> = None;
    for (id, label) in &defs {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(label).small());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .small_button("Add instance")
                    .on_hover_text("Place another instance of this definition")
                    .clicked()
                {
                    add_instance_of = Some(*id);
                }
            });
        });
    }

    if let Some(id) = add_instance_of {
        state.scene.add_instance(id);
        state.sync_mirror_from_active();
        response.scene_changed = true;
    }
}

/// The inspector: switches on the active node. A **Tool** shows the shape chips,
/// size sliders, density slider and material selector (editing the active Tool
/// node, mirrored through [`PanelState::write_mirror_to_active`]). A **Clouds
/// Part** shows its name + seed instead. With no active node, a hint.
fn build_inspector_section(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    response: &mut PanelResponse,
) {
    /// Which inspector to show for the active node.
    enum ActiveKind {
        Tool,
        Part,
        Group,
        Instance,
        None,
    }
    let kind = match state.scene.active_node().map(|node| &node.content) {
        Some(NodeContent::Tool { .. }) => ActiveKind::Tool,
        Some(NodeContent::Part(_)) => ActiveKind::Part,
        Some(NodeContent::Group(_)) => ActiveKind::Group,
        Some(NodeContent::Instance(_)) => ActiveKind::Instance,
        None => ActiveKind::None,
    };

    match kind {
        ActiveKind::Tool => {
            build_shape_section(ui, state, response);
            build_size_section(ui, state, response);
            build_density_section(ui, state, response);
            build_material_section(ui, state, response);
            // Any inspector edit this frame is mirrored back onto the active node.
            // A material pick (`selected_procedural_material`) updates the Tool's
            // material; a geometry edit updates its shape — both write the mirror.
            if response.geometry_changed || response.selected_procedural_material {
                state.write_mirror_to_active();
            }
            // Placement (ADR 0001 step 3) is on the node's transform, not the
            // geometry mirror, so it is edited AFTER the mirror write-back (which
            // only touches shape + material) and is common to all node kinds.
            build_offset_section(ui, state, response);
        }
        ActiveKind::Part => {
            build_part_inspector_section(ui, state, response);
            build_offset_section(ui, state, response);
        }
        ActiveKind::Group => {
            build_group_inspector_section(ui, state, "Group");
            build_offset_section(ui, state, response);
        }
        ActiveKind::Instance => {
            build_group_inspector_section(ui, state, "Instance");
            build_offset_section(ui, state, response);
        }
        ActiveKind::None => {
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("Select or add a node to edit it.")
                    .small()
                    .weak(),
            );
            ui.separator();
        }
    }
}

/// Inspector for a Group or Instance active node (ADR 0001 step 4): its name (and,
/// for an Instance, the definition it references). The offset is edited by the
/// shared [`build_offset_section`], so Group/Instance get at least name + offset.
fn build_group_inspector_section(ui: &mut egui::Ui, state: &mut PanelState, heading: &str) {
    ui.add_space(8.0);
    ui.strong(heading);
    // Capture the def label (immutable borrow) before taking the mutable node.
    let def_label = match state.scene.active_node().map(|n| &n.content) {
        Some(NodeContent::Instance(def_id)) => state
            .scene
            .def_by_id(*def_id)
            .map(|def| {
                if def.name.is_empty() {
                    format!("Def {}", def.id.0)
                } else {
                    def.name.clone()
                }
            })
            .or_else(|| Some(format!("Def {} (missing)", def_id.0))),
        _ => None,
    };
    if let Some(node) = state.scene.active_node_mut() {
        ui.horizontal(|ui| {
            ui.label("Name");
            ui.text_edit_singleline(&mut node.name);
        });
    }
    if let Some(label) = def_label {
        ui.label(
            egui::RichText::new(format!("references: {label}"))
                .small()
                .weak(),
        );
    }
    ui.separator();
}

/// Inspector for a Clouds Part active node: its name and seed (its one knob). A
/// seed change re-resolves the scene.
fn build_part_inspector_section(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    response: &mut PanelResponse,
) {
    ui.add_space(8.0);
    ui.strong("Clouds (Part)");
    let Some(node) = state.scene.active_node_mut() else {
        return;
    };
    ui.horizontal(|ui| {
        ui.label("Name");
        ui.text_edit_singleline(&mut node.name);
    });
    if let NodeContent::Part(Part::DebugClouds { seed }) = &mut node.content {
        let mut value = *seed;
        if ui
            .add(egui::Slider::new(&mut value, 0..=64).text("seed"))
            .changed()
        {
            *seed = value;
            response.scene_changed = true;
        }
    }
    ui.separator();
}

/// Offset (placement) section (ADR 0001 step 3): three integer drag boxes
/// (X/Y/Z, may be negative) writing the active node's
/// [`NodeTransform::offset_blocks`](crate::scene::NodeTransform::offset_blocks).
/// Common to Tools and Parts — placement is on the node's transform, not the
/// producer. Editing it re-resolves the composited scene (a node moving changes
/// the composite extent, so it auto-frames like a size change via
/// [`PanelResponse::scene_changed`]).
///
/// Offsets are in-memory only for now — persistence is ADR 0001 step 8 (the
/// config round-trip does not yet carry them). // step 8: serialize offsets.
fn build_offset_section(ui: &mut egui::Ui, state: &mut PanelState, response: &mut PanelResponse) {
    let Some(node) = state.scene.active_node_mut() else {
        return;
    };
    ui.add_space(8.0);
    ui.strong("Offset (blocks)");
    let mut changed = false;
    ui.horizontal(|ui| {
        for (axis_index, axis_label) in ["X", "Y", "Z"].iter().enumerate() {
            let mut value = node.transform.offset_blocks[axis_index];
            let drag = egui::DragValue::new(&mut value)
                .speed(0.1)
                .prefix(format!("{axis_label} "));
            if ui.add(drag).changed() {
                node.transform.offset_blocks[axis_index] = value;
                changed = true;
            }
        }
    });
    if changed {
        // A placement edit re-resolves + re-frames the composite (treated like a
        // scene change so the caller auto-frames the whole composited extent).
        response.scene_changed = true;
    }
    ui.separator();
}

/// Shape chips. Selecting a shape sets [`GeometryParams::shape`] ONLY — it never
/// touches the size or the camera (Milestone 3 guard #1). Shown only for a Tool
/// active node; the edit is mirrored onto that node by the inspector.
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
    // ADR 0002 E3c-2 (part of #18): the cuboid mesh path is now the DEFAULT. The
    // checkbox (checked ⇒ cuboid) is left enabled so the legacy instanced path can
    // be selected as a fallback by unchecking it.
    let mut use_cuboid = state.mesher == MesherChoice::Cuboid;
    if ui
        .checkbox(&mut use_cuboid, "Cuboid mesher (default; off = legacy instanced)")
        .changed()
    {
        state.mesher = if use_cuboid {
            MesherChoice::Cuboid
        } else {
            MesherChoice::Instanced
        };
    }
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
