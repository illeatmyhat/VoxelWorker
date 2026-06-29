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
use crate::core_geom::MaterialChoice;
use crate::intent::{Intent, NodeSpec};
use crate::scene::{DefId, Node, NodeContent, NodeId, Part, Scene};
use crate::units::{self, DisplayUnit, MeasurementError};
use crate::voxel::{GeometryParams, SdfShape, ShapeKind};

/// Layer-range scrubber state (issue #12).
///
/// The layer-range scrubber subsumes the old 2D mid-vertical slice map. Z-up: layers
/// run along **Z** (height). `lower`/`upper` are voxel Z-layer indices selected on a
/// track `0..grid_z`; the visible band is layers `[lower, upper]` INCLUSIVE on both
/// ends (so `lower == upper` shows a single layer). Default = the full range.
///
/// When `snap_to_blocks` is on, the handles snap to multiples of
/// `voxels_per_block` (plus the endpoints `0` and `grid_z`); a narrowed
/// single-layer band viewed from the top is the chisel stencil. `onion_skin`
/// ghosts up to `onion_depth` layers on each side of the band (3D screen-door).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerRange {
    /// Lower handle: the first visible layer index (`0..=grid_z`).
    pub lower: u32,
    /// Upper handle: the last visible layer index (`lower..=grid_z`).
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
        // Full range over the default cylinder grid_z (1 block × 16 density = 16).
        // The real bounds are clamped/rescaled to the live grid on first rebuild
        // and whenever grid_z changes (see `LayerRange::rescale_to_grid_z`).
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
    /// `0` and `grid_z` exact (they are always valid snap points even when
    /// `grid_z` is not a clean multiple of the density, which it always is here).
    pub fn snap_value(value: u32, voxels_per_block: u32, grid_z: u32) -> u32 {
        let step = voxels_per_block.max(1);
        if value >= grid_z {
            return grid_z;
        }
        let snapped = ((value + step / 2) / step) * step;
        snapped.min(grid_z)
    }

    /// Clamp/rescale the bounds to a (possibly new) `grid_z` (Z-up: layers are
    /// Z-slices). Called on every geometry rebuild: when `grid_z` shrinks the handles
    /// are clamped in; the default full-range state widens to the new top. Re-snaps to
    /// block multiples when snapping is on so the band keeps landing on boundaries.
    pub fn rescale_to_grid_z(&mut self, previous_grid_z: u32, grid_z: u32, voxels_per_block: u32) {
        // A band that spanned the whole previous grid stays "full" on the new one.
        let was_full = self.lower == 0 && self.upper >= previous_grid_z;
        if was_full || previous_grid_z == 0 {
            self.lower = 0;
            self.upper = grid_z;
        } else {
            self.lower = self.lower.min(grid_z);
            self.upper = self.upper.min(grid_z);
        }
        if self.snap_to_blocks {
            self.lower = Self::snap_value(self.lower, voxels_per_block, grid_z);
            self.upper = Self::snap_value(self.upper, voxels_per_block, grid_z);
        }
        if self.lower > self.upper {
            std::mem::swap(&mut self.lower, &mut self.upper);
        }
        self.onion_depth = self.onion_depth.clamp(1, 8);
    }

    /// Whether this band covers the whole grid (so the 3D render is unclipped).
    pub fn is_full_range(&self, grid_z: u32) -> bool {
        self.lower == 0 && self.upper >= grid_z
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
    /// Whether the corner view cube is drawn (M5 Display toggle, ON by default).
    pub show_view_cube: bool,
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
    /// Layer-range scrubber state (issue #12): the visible band along Z (Z-up: layers
    /// are Z-slices) plus the snap/onion controls. Bounds clamped/rescaled on rebuild.
    pub layer_range: LayerRange,
    /// Where **+ Add Point** drops a new Point (issue #29 S5), in whole world blocks.
    /// The caller refreshes it each frame from the camera target (rounded to blocks)
    /// so a new Point lands where the user is looking; it defaults to the world origin
    /// (`[0, 0, 0]`) when the caller does not set it (e.g. the headless harness).
    pub point_add_position_blocks: [i64; 3],
}

impl PanelState {
    /// Sensible defaults for the windowed app: like [`Default`] but with the view
    /// cube enabled (prototype `showCube: true`).
    pub fn with_view_cube_default() -> Self {
        let mut state = Self {
            show_view_cube: true,
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
        if self.scene.roots.is_empty() {
            self.scene = Scene::from_geometry(self.geometry, self.material);
        }
        // issue #29 (grid rework S1): every scene carries exactly one Origin Point.
        // Idempotent, so calling it on an already-seeded scene is a no-op.
        self.scene.ensure_origin_point();
        // ADR 0003 Phase B: mint a stable NodeId for every node (idempotent).
        self.scene.ensure_node_ids();
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
                    // Density is document-level (ADR 0003 §3f(0)): the slider's
                    // transient mirror value comes from the scene, not the shape.
                    voxels_per_block: self.scene.voxels_per_block,
                    wall_blocks: shape.wall_blocks,
                };
                self.material = *material;
            }
        }
    }
}

/// What changed during a [`build_panel`] call, so the caller can react.
///
/// **ADR 0003 Phase C, slice C4a.** The panel no longer mutates `state.scene`
/// directly; instead every document mutation this frame is DESCRIBED as an
/// [`Intent`] pushed onto [`intents`](Self::intents), which the loop applies through
/// [`AppCore::apply_intent`](crate::AppCore::apply_intent) and folds the returned
/// [`IntentEffect`](crate::IntentEffect)s into its rebuild / points / selection
/// decisions. The remaining fields are NON-scene side effects (palette / export /
/// folder picker) the panel still only flags, plus the
/// [`frame_after_apply`](Self::frame_after_apply) auto-frame hint (which is a panel
/// UX concern — a size-slider `SetShape` re-frames, a shape-chip `SetShape` does
/// not, even though both are the same intent KIND — so it cannot be derived from the
/// intent alone and stays on the response).
#[derive(Debug, Clone, Default)]
pub struct PanelResponse {
    /// The document mutations the user made this frame, in emission order (ADR 0003
    /// Phase C C4a). The loop applies each through `AppCore::apply_intent` and merges
    /// the effects; the panel itself performs NONE of them.
    pub intents: Vec<Intent>,
    /// The caller should auto-frame the camera after applying this frame's intents
    /// (the typed successor of the old `size_or_density_changed || scene_changed`
    /// auto-frame trigger). Set by the panel for every emitted intent EXCEPT a pure
    /// shape-chip switch and a material pick (guard #1: a shape switch re-resolves at
    /// the same size and must NOT move the camera). A panel-level signal because the
    /// same intent KIND (`SetShape`) auto-frames from a size slider but not from a
    /// shape chip.
    pub frame_after_apply: bool,
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
    /// The user picked **Focus** from a node row's right-click context menu this
    /// frame → the loop should frame that node (set the camera target to the node's
    /// world centre + fit the distance). A VIEW action, NOT a document `Intent` (it
    /// is not undoable), so it rides on the response rather than `intents`. `None`
    /// when no Focus was requested.
    pub focus_node: Option<NodeId>,
}

impl PanelResponse {
    /// Push a mutation the user described this frame (ADR 0003 Phase C C4a). The loop
    /// applies it through `AppCore::apply_intent`; the panel never mutates the scene.
    fn emit(&mut self, intent: Intent) {
        self.intents.push(intent);
    }

    /// Push a mutation AND request an auto-frame after this frame's intents apply (the
    /// old `scene_changed` / `size_or_density_changed` behaviour). Used for structural
    /// edits and size/density edits — everything that re-frames; a shape-chip switch
    /// and a material pick use [`emit`](Self::emit) instead so the camera stays put.
    fn emit_and_frame(&mut self, intent: Intent) {
        self.frame_after_apply = true;
        self.intents.push(intent);
    }
}

/// Build the right-hand side panel into the root [`egui::Ui`] of the frame.
///
/// `grid_z` is the current grid height in voxels (Z-up: layers are Z-slices, so the
/// layer-scrubber track spans `0..grid_z`); `measured_diameter` is the widest
/// occupied voxel run in the active band (`grid.widest_run_in_band`), shown as a
/// small stat line. Returns a [`PanelResponse`] describing what the user changed.
pub fn build_panel(
    root_ui: &mut egui::Ui,
    state: &mut PanelState,
    grid_z: u32,
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
            build_points_section(ui, state, &mut response);
            build_inspector_section(ui, state, &mut response);
            build_camera_section(ui, state);
            build_display_section(ui, state, &mut response);
            build_export_section(ui, &mut response);
            build_layers_section(ui, state, grid_z, measured_diameter);

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
        NodeContent::SketchTool { .. } => "Sketch".to_string(),
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

    let mut select: Option<NodeId> = None;
    let mut delete: Option<NodeId> = None;
    let mut set_visible: Option<(NodeId, bool)> = None;
    // The camera "Focus" view action (right-click a row → frame that node). Deferred
    // like the others and carried out on the `PanelResponse` (a VIEW action, not an
    // undoable document intent).
    let mut focus: Option<NodeId> = None;

    // Walk the tree depth-first; each row is indented by its depth.
    // ADR 0003 Phase B4: each row carries its node's stable NodeId, so the
    // select/delete/visibility ops (now NodeId-typed) are fed it directly — the
    // path is kept only for depth/indentation.
    let rows = state.scene.tree_rows();
    // Selection is keyed by NodeId; compare each row's id against the active id so
    // the highlight tracks the selected node by identity.
    let active_id = state.scene.active;
    for (_path, id, depth) in &rows {
        let is_active = active_id == Some(*id);
        // Read the node by id; mutate visibility via a deferred op (a separate
        // lookup) so the borrow of `nodes` does not span the whole row.
        let (label, visible) = match state.scene.node_by_id(*id) {
            Some(node) => (node_row_label(node), node.visible),
            None => continue,
        };
        ui.horizontal(|ui| {
            ui.add_space(*depth as f32 * 14.0);
            let mut visible = visible;
            if ui
                .checkbox(&mut visible, "")
                .on_hover_text("Visible")
                .changed()
            {
                set_visible = Some((*id, visible));
            }
            let row_response = ui.selectable_label(is_active, label);
            if row_response.clicked() {
                select = Some(*id);
            }
            // Right-click → Focus: a VIEW action that frames this node (camera target
            // = node centre, distance fitted to its AABB). Carried on the response,
            // not as an Intent (Focus is not undoable).
            row_response.context_menu(|ui| {
                if ui.button("Focus").clicked() {
                    focus = Some(*id);
                    ui.close();
                }
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("✕").on_hover_text("Delete node").clicked() {
                    delete = Some(*id);
                }
            });
        });
    }

    // ADR 0003 Phase C C4a: a visibility toggle is now described as a `SetVisible`
    // intent (the loop applies it), not a direct `set_node_visible`. It re-resolves +
    // auto-frames exactly as before (the old `scene_changed`).
    if let Some((id, visible)) = set_visible {
        response.emit_and_frame(Intent::SetVisible { target: id, visible });
    }

    // Carry a right-click Focus request to the loop (a view action; the loop sets the
    // camera target + distance from this node's AABB). Not an Intent — Focus is not
    // an undoable document mutation.
    if let Some(id) = focus {
        response.focus_node = Some(id);
    }

    if state.scene.roots.is_empty() {
        ui.label(egui::RichText::new("(no nodes — add one below)").small().weak());
    }

    build_node_actions(ui, state, response);
    build_definitions_section(ui, state, response);

    // Apply the deferred selection / delete after the walk. ADR 0003 Phase C C4a:
    // both are now intents (`RemoveNode` / `SelectNode`) the loop applies — the panel
    // no longer touches `scene.active` or calls `remove_node`/`sync_mirror_from_active`
    // here (the loop re-syncs the inspector mirror on the returned effect).
    if let Some(id) = delete {
        response.emit_and_frame(Intent::RemoveNode { target: id });
    } else if let Some(clicked_id) = select {
        // ADR 0003 Phase B4: a clicked row reports its node's stable NodeId; select
        // THAT, so the highlight and inspector follow the node through later
        // structural edits. Only emit when it actually changes the selection (the old
        // guard) — a `SelectNode` is selection-only (no re-resolve, no auto-frame).
        if state.scene.active != Some(clicked_id) {
            response.emit(Intent::SelectNode { target: Some(clicked_id) });
        }
    }

    ui.separator();
}

/// The **Points** section (issue #29 S5): the world reference grid's frames. Lists
/// every [`Point`] with a visibility checkbox (bound to `!hidden`) and a selectable
/// name; **+ Add Point** appends a Point at the camera target (falling back to the
/// origin); and — for the selected Point — XZ/XY/YZ plane checkboxes, per-axis
/// X/Y/Z checkboxes, a whole-block position editor (HIDDEN for the Origin), and a **Delete**
/// button (hidden for the Origin, which is undeletable). Mirrors the node list's
/// deferred-mutation pattern: selection/delete are applied AFTER the read walk.
fn build_points_section(ui: &mut egui::Ui, state: &mut PanelState, response: &mut PanelResponse) {
    // A scene with NO Points (the headless `shot` path builds scenes WITHOUT the
    // synthesized Origin — `ensure_origin_point` runs only on the windowed load/seed
    // path) renders nothing here, so the section adds zero height and the existing
    // goldens stay byte-identical. The windowed app always carries the Origin Point,
    // so the section always shows there.
    if state.scene.points.is_empty() {
        return;
    }
    ui.add_space(8.0);
    ui.strong("Points");

    let mut select: Option<usize> = None;
    let mut delete: Option<usize> = None;
    let mut toggle_hidden: Option<usize> = None;

    // The Point rows: a visibility checkbox (bound to `!hidden`) + a selectable name.
    for index in 0..state.scene.points.len() {
        let (name, hidden, is_active) = {
            let point = &state.scene.points[index];
            let name = if point.name.is_empty() {
                format!("Point {index}")
            } else {
                point.name.clone()
            };
            (name, point.hidden, state.scene.active_point == Some(index))
        };
        ui.horizontal(|ui| {
            // Visibility is `!hidden`; toggling it flips the Point's `hidden` flag.
            let mut visible = !hidden;
            if ui.checkbox(&mut visible, "").on_hover_text("Visible").changed() {
                toggle_hidden = Some(index);
            }
            if ui.selectable_label(is_active, name).clicked() {
                select = Some(index);
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // The Origin is undeletable — no ✕ button for it.
                if !state.scene.points[index].is_origin
                    && ui.small_button("✕").on_hover_text("Delete point").clicked()
                {
                    delete = Some(index);
                }
            });
        });
    }

    // + Add Point — a fresh Point at the camera target (whole blocks), else the
    // origin. ADR 0003 Phase C C4a: described as an `AddPoint` intent; the panel
    // names it after the soon-to-be index (matching the old `format!`), and emits a
    // trailing `SelectPoint` so the new Point becomes active (the old
    // `active_point = len - 1`, which `add_point` itself does not set).
    if ui
        .button("+ Add Point")
        .on_hover_text("Add a reference Point at the camera target")
        .clicked()
    {
        let new_index = state.scene.points.len();
        response.emit(Intent::AddPoint {
            position_blocks: state.point_add_position_blocks,
            name: format!("Point {new_index}"),
        });
        response.emit(Intent::SelectPoint { target: Some(new_index) });
    }

    // The selected Point's editor: plane/axis toggles, position (hidden for Origin),
    // and a delete button (hidden for the Origin). ADR 0003 Phase C C4a: each widget
    // binds to a LOCAL copy of the Point's fields (egui needs the `&mut`); a change
    // emits the matching `SetPoint*` intent instead of mutating the Point. The buffer
    // is read fresh from the scene each frame, so it always reflects the live value.
    if let Some(active) = state.scene.active_point {
        if let Some(point) = state.scene.points.get(active) {
            let point = point.clone();
            ui.add_space(4.0);
            ui.separator();

            // Plane toggles → `SetPointPlanes` (carrying all three current values).
            // Z-up: the GROUND plane is XY (normal +Z) = the `plane_xy` flag; the
            // FRONT plane is XZ (normal +Y) = `plane_xz`; the SIDE plane is YZ.
            let mut plane_xz = point.plane_xz;
            let mut plane_xy = point.plane_xy;
            let mut plane_yz = point.plane_yz;
            let mut planes_changed = false;
            planes_changed |= ui.checkbox(&mut plane_xy, "Ground plane (XY)").changed();
            planes_changed |= ui.checkbox(&mut plane_xz, "Front plane (XZ)").changed();
            planes_changed |= ui.checkbox(&mut plane_yz, "Side plane (YZ)").changed();
            if planes_changed {
                response.emit(Intent::SetPointPlanes {
                    index: active,
                    xz: plane_xz,
                    xy: plane_xy,
                    yz: plane_yz,
                });
            }

            // Per-axis toggles (issue #29 fix): X/Y/Z each toggle independently →
            // `SetPointAxes` (carrying all three).
            let mut axis_x = point.axis_x;
            let mut axis_y = point.axis_y;
            let mut axis_z = point.axis_z;
            let mut axes_changed = false;
            ui.horizontal(|ui| {
                ui.label("Axes");
                axes_changed |= ui.checkbox(&mut axis_x, "X").changed();
                axes_changed |= ui.checkbox(&mut axis_y, "Y").changed();
                axes_changed |= ui.checkbox(&mut axis_z, "Z").changed();
            });
            if axes_changed {
                response.emit(Intent::SetPointAxes {
                    index: active,
                    x: axis_x,
                    y: axis_y,
                    z: axis_z,
                });
            }

            // Position editor — only for a user Point (the Origin is pinned at world 0).
            if !point.is_origin {
                let mut position = point.position_blocks;
                let mut position_changed = false;
                ui.horizontal(|ui| {
                    ui.label("Pos (blocks)");
                    for axis_value in &mut position {
                        position_changed |= ui
                            .add(egui::DragValue::new(axis_value).speed(1.0))
                            .changed();
                    }
                });
                if position_changed {
                    response.emit(Intent::SetPointPosition {
                        index: active,
                        position_blocks: position,
                    });
                }
                if ui.button("Delete point").clicked() {
                    delete = Some(active);
                }
            } else {
                ui.label(
                    egui::RichText::new("Origin — pinned at world origin, undeletable")
                        .small()
                        .weak(),
                );
            }
        }
    }

    // Apply deferred mutations after the read/borrow walk. ADR 0003 Phase C C4a: each
    // is described as an intent the loop applies.
    if let Some(index) = toggle_hidden {
        // The visibility checkbox is bound to `!hidden`; a toggle flips it. Read the
        // current flag and emit the explicit `SetPointHidden` for the new value (the
        // intent path is explicit, unlike `toggle_point_hidden`'s flip).
        if let Some(point) = state.scene.points.get(index) {
            response.emit(Intent::SetPointHidden { index, hidden: !point.hidden });
        }
    }
    if let Some(index) = delete {
        // `RemovePoint` is a no-op on the Origin (the UI already hides its delete
        // affordances). To preserve the old `active_point` fix-up (which `remove_point`
        // does not do), emit a trailing `SelectPoint` re-deriving the selection.
        let was_origin = state.scene.points.get(index).map(|p| p.is_origin).unwrap_or(false);
        if !was_origin {
            response.emit(Intent::RemovePoint { index });
            // After removing index, the list shrinks by one: re-derive the selection
            // exactly as the old code did (clamp to the new last, or clear if empty).
            let remaining = state.scene.points.len().saturating_sub(1);
            let next = if remaining == 0 {
                None
            } else {
                Some(index.min(remaining - 1))
            };
            response.emit(Intent::SelectPoint { target: next });
        }
    } else if let Some(index) = select {
        if state.scene.active_point != Some(index) {
            response.emit(Intent::SelectPoint { target: Some(index) });
        }
    }

    ui.separator();
}

/// A [`NodeSpec`] for a fresh Tool node of the given shape, inheriting the current
/// size/density/wall + material so it renders immediately (ADR 0003 Phase C C4a). The
/// spec's `into_node` names the node after the shape kind — identical to the old
/// `new_tool_node` label, since the [`SHAPE_CHIPS`] labels ARE the kind's Debug names.
fn tool_node_spec(kind: ShapeKind, state: &PanelState) -> NodeSpec {
    NodeSpec::Tool {
        shape: SdfShape {
            kind,
            size_blocks: state.geometry.size_blocks,
            wall_blocks: state.geometry.wall_blocks,
        },
        material: state.material,
    }
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
        // + Add — a top-level Tool or Clouds Part. ADR 0003 Phase C C4a: described as
        // `AddNode` intents (`NodeSpec` carries the same shape+material/Clouds the old
        // `new_tool_node` / `Node::new` built). The new node becomes active inside the
        // add op, so the loop re-syncs the inspector mirror on the returned effect.
        ui.menu_button("+ Add", |ui| {
            for (kind, label) in SHAPE_CHIPS {
                if ui.button(*label).clicked() {
                    response.emit_and_frame(Intent::AddNode {
                        content: tool_node_spec(*kind, state),
                    });
                    ui.close();
                }
            }
            ui.separator();
            if ui.button("Clouds (Part)").clicked() {
                response.emit_and_frame(Intent::AddNode {
                    content: NodeSpec::CloudsPart,
                });
                ui.close();
            }
        });

        // + Add child — into the active Group (only shown when one is selected).
        if active_is_group {
            // ADR 0003 Phase B4: `AddChild` targets a NodeId; this block only shows
            // when a Group is active, so the active selection IS the group's id.
            let group_id = state.scene.active;
            ui.menu_button("+ Add child", |ui| {
                for (kind, label) in SHAPE_CHIPS {
                    if ui.button(*label).clicked() {
                        if let Some(group_id) = group_id {
                            response.emit_and_frame(Intent::AddChild {
                                group: group_id,
                                content: tool_node_spec(*kind, state),
                            });
                        }
                        ui.close();
                    }
                }
                ui.separator();
                if ui.button("Clouds (Part)").clicked() {
                    if let Some(group_id) = group_id {
                        response.emit_and_frame(Intent::AddChild {
                            group: group_id,
                            content: NodeSpec::CloudsPart,
                        });
                    }
                    ui.close();
                }
            });
        }

        // Group — wrap the active node in a new Group → `GroupNode { target: active }`.
        if ui
            .add_enabled(has_active, egui::Button::new("Group"))
            .on_hover_text("Wrap the selected node in a new Group")
            .clicked()
        {
            if let Some(target) = state.scene.active {
                response.emit_and_frame(Intent::GroupNode { target });
            }
        }

        // Make definition — the active node becomes a reusable AssemblyDef and is
        // replaced by an Instance of it → `MakeDefinition { target: active, name }`.
        if ui
            .add_enabled(has_active, egui::Button::new("Make definition"))
            .on_hover_text("Turn the selected Group/node into a reusable definition, placed by an Instance")
            .clicked()
        {
            if let Some(target) = state.scene.active {
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
                response.emit_and_frame(Intent::MakeDefinition { target, name: def_name });
            }
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

    // ADR 0003 Phase C C4a: described as an `AddInstance` intent. The placed Instance
    // becomes active inside the add op, so the loop re-syncs the mirror on the effect.
    if let Some(id) = add_instance_of {
        response.emit_and_frame(Intent::AddInstance { def: id });
    }
}

/// The inspector: switches on the active node. A **Tool** shows the shape chips,
/// size sliders, density slider and material selector (editing the active Tool node;
/// ADR 0003 Phase C C4a routes each edit to a `SetShape`/`SetDensity`/`SetMaterial`
/// intent the loop applies). A **Clouds Part** shows its name + seed instead. With no
/// active node, a hint.
fn build_inspector_section(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    response: &mut PanelResponse,
) {
    /// Which inspector to show for the active node.
    enum ActiveKind {
        Tool,
        Sketch,
        Part,
        Group,
        Instance,
        None,
    }
    let kind = match state.scene.active_node().map(|node| &node.content) {
        Some(NodeContent::Tool { .. }) => ActiveKind::Tool,
        // ADR 0003 §3i Slice 2a has NO sketch-editing UI yet; the inspector shows
        // only the shared placement + grids sections (a sketch is placed/toggled
        // like any node). Profile/extrude editing arrives with the 2b/2c UI.
        Some(NodeContent::SketchTool { .. }) => ActiveKind::Sketch,
        Some(NodeContent::Part(_)) => ActiveKind::Part,
        Some(NodeContent::Group(_)) => ActiveKind::Group,
        Some(NodeContent::Instance(_)) => ActiveKind::Instance,
        None => ActiveKind::None,
    };

    match kind {
        ActiveKind::Tool => {
            // ADR 0003 Phase C C4a: the inspector still binds the widgets to the
            // `geometry`/`material` mirror buffer (egui needs the `&mut`), but a change
            // now EMITS the matching intent instead of calling `write_mirror_to_active`.
            // The active id is known (this is the Tool arm). `SetShape` carries the FULL
            // updated buffer (`from_geometry`) onto the active node — covering both a
            // shape-chip switch (no auto-frame, guard #1) and a size/wall edit
            // (auto-frame). Density is GLOBAL → `SetDensity` (rewrites every Tool);
            // material → `SetMaterial`. The mirror is now ONLY the widget buffer.
            let active = state.scene.active;
            let shape_changed = build_shape_section(ui, state);
            let size_changed = build_size_section(ui, state);
            let density_changed = build_density_section(ui, state);
            let material_changed = build_material_section(ui, state, response);

            if let Some(target) = active {
                // A shape OR size/wall edit rewrites the active Tool's shape from the
                // buffer. Size/wall auto-frames; a pure shape switch does not.
                if shape_changed || size_changed {
                    let shape = SdfShape::from_geometry(state.geometry);
                    let intent = Intent::SetShape { target, shape };
                    if size_changed {
                        response.emit_and_frame(intent);
                    } else {
                        response.emit(intent);
                    }
                }
                // Density is a document-level attribute (ADR 0003 §3f(0)): the slider's
                // transient value drives the single `scene.voxels_per_block` via
                // SetDensity. Auto-frames like a size change.
                if density_changed {
                    response.emit_and_frame(Intent::SetDensity {
                        voxels_per_block: state.geometry.voxels_per_block,
                    });
                }
                // A material pick updates the active Tool's material (no auto-frame).
                if material_changed {
                    response.emit(Intent::SetMaterial { target, material: state.material });
                }
            }
            // Placement (ADR 0001 step 3) is on the node's transform, common to all
            // node kinds; it emits its own `SetOffset` intent.
            build_offset_section(ui, state, response);
            build_node_grids_section(ui, state, response);
        }
        ActiveKind::Sketch => {
            // No sketch-editing widgets in Slice 2a — just a label + the shared
            // placement / grids sections so a sketch node is selectable and placeable.
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("Sketch → Extrude (no editor yet)")
                    .small()
                    .weak(),
            );
            ui.separator();
            build_offset_section(ui, state, response);
            build_node_grids_section(ui, state, response);
        }
        ActiveKind::Part => {
            build_part_inspector_section(ui, state, response);
            build_offset_section(ui, state, response);
            build_node_grids_section(ui, state, response);
        }
        ActiveKind::Group => {
            build_group_inspector_section(ui, state, "Group", response);
            build_offset_section(ui, state, response);
            build_node_grids_section(ui, state, response);
        }
        ActiveKind::Instance => {
            build_group_inspector_section(ui, state, "Instance", response);
            build_offset_section(ui, state, response);
            build_node_grids_section(ui, state, response);
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
/// shared [`build_offset_section`], so Group/Instance get at least name + offset. ADR
/// 0003 Phase C C4a: the name widget binds to a LOCAL buffer; a change emits `SetName`
/// WITHOUT an auto-frame (the old rename mutated `node.name` with no response flag, so
/// the camera never moved on rename).
fn build_group_inspector_section(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    heading: &str,
    response: &mut PanelResponse,
) {
    ui.add_space(8.0);
    ui.strong(heading);
    // Capture the def label (immutable borrow) before taking the active node.
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
    if let (Some(target), Some(node)) = (state.scene.active, state.scene.active_node()) {
        let mut name = node.name.clone();
        ui.horizontal(|ui| {
            ui.label("Name");
            // A rename did NOT auto-frame in the old code (it mutated `node.name`
            // with no response flag), so emit WITHOUT a frame. The `SetName` effect is
            // `scene_changed`, so the loop re-resolves (an identical grid — the name is
            // not geometry) but the camera stays put, matching the old visible result.
            if ui.text_edit_singleline(&mut name).changed() {
                response.emit(Intent::SetName { target, name: name.clone() });
            }
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
/// seed change re-resolves the scene. ADR 0003 Phase C C4a: the name/seed widgets bind
/// to LOCAL buffers (read from the active node each frame); a change emits `SetName` /
/// `SetCloudSeed` instead of mutating the node. A seed edit auto-frames like the old
/// `scene_changed`.
fn build_part_inspector_section(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    response: &mut PanelResponse,
) {
    ui.add_space(8.0);
    ui.strong("Clouds (Part)");
    let Some(target) = state.scene.active else {
        return;
    };
    let Some(node) = state.scene.active_node() else {
        return;
    };
    let mut name = node.name.clone();
    let current_seed = match &node.content {
        NodeContent::Part(Part::DebugClouds { seed }) => Some(*seed),
        _ => None,
    };
    ui.horizontal(|ui| {
        ui.label("Name");
        if ui.text_edit_singleline(&mut name).changed() {
            response.emit(Intent::SetName { target, name: name.clone() });
        }
    });
    if let Some(seed) = current_seed {
        let mut value = seed;
        if ui
            .add(egui::Slider::new(&mut value, 0..=64).text("seed"))
            .changed()
        {
            response.emit_and_frame(Intent::SetCloudSeed { target, seed: value });
        }
    }
    ui.separator();
}

/// Offset (placement) section (ADR 0003 §3f(0)): three per-axis text fields
/// (X/Y/Z, signed) accepting blocks+voxels unit expressions (e.g. `"3 blocks 8
/// voxels"`, `"-1b 4v"`, `"3.5 blocks"`). Each field is seeded from the canonical
/// voxel offset formatted as blocks+voxels and, on commit (Enter or focus loss),
/// parsed via [`units::parse`] and validated to land on a whole voxel at the
/// document density; on success it emits a single `SetOffset` carrying the
/// per-axis [`Measurement`](crate::units::Measurement)s (the edited axis plus the
/// two unchanged retained ones). A parse / non-landing error is shown inline (red)
/// and NOTHING is emitted, so the canonical offset never moves on bad input.
///
/// Common to Tools and Parts — placement is on the node's transform, not the
/// producer. A committed edit re-resolves + re-frames the composite (a node moving
/// changes the composite extent), so it auto-frames the whole composited extent.
///
/// The in-progress text + last error live in egui temp memory (keyed per axis by a
/// stable `Id`) so a partial edit and its error survive across frames; an unfocused
/// field re-syncs to the canonical value, so undo / external moves reflect.
fn build_offset_section(ui: &mut egui::Ui, state: &mut PanelState, response: &mut PanelResponse) {
    let Some(target) = state.scene.active else {
        return;
    };
    let Some(node) = state.scene.active_node() else {
        return;
    };
    let density = state.scene.voxels_per_block;
    // The canonical voxel offset (resolve's source of truth) and the RETAINED
    // per-axis measurements (the two unedited axes ride along unchanged in any
    // emitted intent so a single-axis edit does not disturb the others).
    let offset_voxels = node.transform.offset_voxels;
    let retained_measurements = node.transform.offset_measurements();

    ui.add_space(8.0);
    ui.strong("Offset (blocks + voxels)");

    for (axis_index, axis_label) in ["X", "Y", "Z"].iter().enumerate() {
        // Per-axis stable ids for the in-progress text buffer and last error.
        let text_id = egui::Id::new(("offset_axis_text", target, axis_index));
        let error_id = egui::Id::new(("offset_axis_error", target, axis_index));
        // The canonical seed: the current voxel offset as a blocks+voxels string.
        let seed = units::format(offset_voxels[axis_index], density, DisplayUnit::BlocksAndVoxels);

        // The text edit binds to a LOCAL buffer restored from temp memory; an
        // UNFOCUSED field re-syncs to the canonical seed so undo / external moves
        // and density changes reflect, while a focused field keeps the user's
        // in-progress text untouched.
        let mut buffer = ui
            .memory(|memory| memory.data.get_temp::<String>(text_id))
            .unwrap_or_else(|| seed.clone());

        let widget = egui::TextEdit::singleline(&mut buffer)
            .desired_width(110.0)
            .hint_text("blocks + voxels");
        let widget_response = ui.horizontal(|ui| {
            ui.label(format!("{axis_label} "));
            ui.add(widget)
        });
        let edit_response = widget_response.inner;

        // Editing again clears any stale error from a prior failed commit, so the
        // red message tracks the LAST committed attempt, not in-progress typing.
        if edit_response.changed() {
            ui.memory_mut(|memory| memory.data.remove::<String>(error_id));
        }

        // `lost_focus()` fires on Enter AND on click-away, so it is the single
        // commit trigger; the typed `buffer` is still live here (the unfocused
        // re-sync happens AFTER, so a commit reads the user's text, not the seed).
        // Only attempt a parse when the text actually differs from the canonical
        // seed (a focus loss with no edit is a no-op).
        let committed = edit_response.lost_focus() && buffer.trim() != seed;
        if committed {
            match units::parse(&buffer) {
                Ok(measurement) => match measurement.to_voxels(density) {
                    Ok(landed_voxels) => {
                        // Replace only this axis; the other two keep their retained
                        // measurements so a single-axis edit is isolated.
                        let mut next = retained_measurements;
                        next[axis_index] = measurement;
                        ui.memory_mut(|memory| memory.data.remove::<String>(error_id));
                        response.emit_and_frame(Intent::SetOffset {
                            target,
                            offset_measurements: next,
                        });
                        // Settle the field on the canonical form of the applied value.
                        buffer =
                            units::format(landed_voxels, density, DisplayUnit::BlocksAndVoxels);
                    }
                    Err(error) => {
                        ui.memory_mut(|memory| {
                            memory.data.insert_temp(error_id, offset_error_text(&error))
                        });
                    }
                },
                Err(error) => {
                    ui.memory_mut(|memory| {
                        memory.data.insert_temp(error_id, error.to_string())
                    });
                }
            }
        } else if !edit_response.has_focus() {
            // Not being edited and not a commit. If a prior commit FAILED, an error
            // is stored — keep the user's (rejected) text on screen alongside the
            // error so they can see and fix it; do NOT silently revert. With no
            // error, mirror the canonical value so undo / external moves / density
            // changes reflect in the field.
            let has_error = ui.memory(|memory| memory.data.get_temp::<String>(error_id).is_some());
            if !has_error {
                buffer = seed.clone();
            }
        }

        // Persist the buffer for the next frame (the focused, in-progress text).
        ui.memory_mut(|memory| memory.data.insert_temp(text_id, buffer));

        // Inline error (red), cleared on the next successful commit.
        if let Some(message) = ui.memory(|memory| memory.data.get_temp::<String>(error_id)) {
            ui.colored_label(egui::Color32::from_rgb(220, 80, 80), message);
        }
    }

    ui.separator();
}

/// Render a [`MeasurementError`] for the inline offset error label. A non-landing
/// block fraction reports the nearest representable voxel counts so the user can
/// pick one instead of being silently rounded (ADR 0003 §3f(0)).
fn offset_error_text(error: &MeasurementError) -> String {
    match error {
        MeasurementError::BlockTermNotWholeVoxels {
            density,
            nearest_floor_voxels,
            nearest_ceil_voxels,
        } => format!(
            "doesn't land on a whole voxel at density {density}; nearest are {nearest_floor_voxels} or {nearest_ceil_voxels} voxels"
        ),
        MeasurementError::ZeroDensity => "density must be at least 1".to_string(),
    }
}

/// Per-node grid toggles (issue #29 S3/S4): the active node's own
/// `voxel_grid_on_faces` / `block_lattice` / `floor_grid` flags, each ANDed with
/// its scene-wide master (in the Display section) to decide whether that node draws
/// the grid.
///
/// The block-lattice / floor toggles only need a per-frame batch rebuild — those
/// lines are re-walked from the scene every frame — so they signal NO scene
/// re-resolve, keeping a grid flip cheap. The **voxel-grid-on-faces** toggle (S4) is
/// different: the on-face-grid flag bit is baked onto each voxel's `material_id` at
/// RESOLVE time (so it survives chunk bucketing and the cuboid box-decomposition
/// key), so flipping it MUST re-resolve the scene — it signals `scene_changed`.
fn build_node_grids_section(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    response: &mut PanelResponse,
) {
    let Some(target) = state.scene.active else {
        return;
    };
    let Some(node) = state.scene.active_node() else {
        return;
    };
    ui.add_space(8.0);
    ui.strong("Grids (this object)");
    // ADR 0003 Phase C C4a: the three checkboxes bind to a LOCAL copy of the node's
    // grids; a change emits ONE `SetNodeGrids` carrying all three. The on-face-grid
    // flag is baked at RESOLVE time, so flipping it must re-resolve AND it auto-framed
    // before (it set `scene_changed`); the lattice/floor flags are read by the
    // per-frame line batch, so they did NOT auto-frame. We therefore auto-frame ONLY
    // when `voxel_grid_on_faces` flips. (`SetNodeGrids`'s effect is `scene_changed`, so
    // a lattice/floor toggle now also re-resolves — an identical grid, no camera move,
    // so the visible result is unchanged; the cost is a redundant re-resolve.)
    let mut grids = node.grids;
    let mut voxel_grid_changed = false;
    let mut other_changed = false;
    voxel_grid_changed |= ui
        .checkbox(&mut grids.voxel_grid_on_faces, "Voxel grid on faces")
        .changed();
    other_changed |= ui.checkbox(&mut grids.block_lattice, "Block lattice").changed();
    other_changed |= ui.checkbox(&mut grids.floor_grid, "Floor grid").changed();
    if voxel_grid_changed {
        response.emit_and_frame(Intent::SetNodeGrids { target, grids });
    } else if other_changed {
        response.emit(Intent::SetNodeGrids { target, grids });
    }
    ui.separator();
}

/// Shape chips. Selecting a shape sets [`GeometryParams::shape`] ONLY — it never
/// touches the size or the camera (Milestone 3 guard #1). Shown only for a Tool
/// active node. ADR 0003 Phase C C4a: returns `true` when the buffer's shape changed
/// (the inspector then emits a `SetShape` WITHOUT an auto-frame).
fn build_shape_section(ui: &mut egui::Ui, state: &mut PanelState) -> bool {
    ui.add_space(8.0);
    ui.strong("Shape");
    let mut changed = false;
    ui.horizontal_wrapped(|ui| {
        for (kind, label) in SHAPE_CHIPS {
            let is_selected = state.geometry.shape == *kind;
            if ui.selectable_label(is_selected, *label).clicked() && !is_selected {
                state.geometry.shape = *kind;
                changed = true;
                // The caller emits a `SetShape` but no auto-frame: a shape switch
                // re-resolves at the same size and must not move the camera.
            }
        }
    });
    ui.separator();
    changed
}

/// Size sliders (whole blocks). Each shows the resulting voxel extent as a hint. ADR
/// 0003 Phase C C4a: returns `true` when the buffer's size/wall changed (the inspector
/// then emits a `SetShape` AND auto-frames, the old `size_or_density_changed`).
fn build_size_section(ui: &mut egui::Ui, state: &mut PanelState) -> bool {
    ui.add_space(8.0);
    ui.strong("Size (blocks)");

    let mut changed = false;
    let density = state.geometry.voxels_per_block;
    for (axis_index, axis_label) in ["X", "Y", "Z"].iter().enumerate() {
        let mut value = state.geometry.size_blocks[axis_index];
        let slider = egui::Slider::new(&mut value, 1..=16).text(*axis_label);
        if ui.add(slider).changed() {
            state.geometry.size_blocks[axis_index] = value;
            changed = true;
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
            changed = true;
        }
        ui.label(
            egui::RichText::new(format!("{wall} block wall"))
                .small()
                .weak(),
        );
    }
    ui.separator();
    changed
}

/// Density slider. Changes fineness ONLY — never the block size (guard #2). ADR 0003
/// Phase C C4a: returns `true` when the buffer's density changed (the inspector then
/// emits a global `SetDensity` AND auto-frames).
fn build_density_section(ui: &mut egui::Ui, state: &mut PanelState) -> bool {
    ui.add_space(8.0);
    ui.strong("Density");
    let mut density = state.geometry.voxels_per_block;
    let slider = egui::Slider::new(&mut density, 2..=32).text("vx/block");
    let changed = ui.add(slider).changed();
    if changed {
        state.geometry.voxels_per_block = density;
    }
    ui.separator();
    changed
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
/// procedural material clears an applied loaded VS block (M6) and reverts to it. ADR
/// 0003 Phase C C4a: returns `true` when the buffer's material changed (the inspector
/// then emits a `SetMaterial`). Still sets `selected_procedural_material` for the
/// caller's M6 palette side-effect (clearing the applied loaded block — NOT a scene
/// mutation, so it stays a response flag, not an intent).
fn build_material_section(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    response: &mut PanelResponse,
) -> bool {
    ui.add_space(8.0);
    ui.strong("Material");
    let mut changed = false;
    ui.horizontal(|ui| {
        for (choice, label) in [
            (MaterialChoice::Stone, "Stone"),
            (MaterialChoice::Wood, "Wood"),
            (MaterialChoice::Plain, "Plain"),
        ] {
            if ui.selectable_value(&mut state.material, choice, label).clicked() {
                response.selected_procedural_material = true;
                changed = true;
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
    changed
}

/// Display section. M4 added the voxel-grid overlay; M5 wired the view cube and
/// the origin gizmo; M8 wires the block lattice and fine floor grid (#10).
fn build_display_section(ui: &mut egui::Ui, state: &mut PanelState, response: &mut PanelResponse) {
    ui.add_space(8.0);
    ui.strong("Display");
    // ADR 0003 Phase C C4a: the three grid MASTERS are scene fields, so they bind to
    // LOCAL copies and a change emits ONE `SetGridMasters`. The masters are read live
    // by the per-frame line batch / mesh shader (no re-resolve), so `SetGridMasters`'s
    // effect is `none()` — no rebuild, no auto-frame — matching the old direct writes.
    // `show_view_cube` / `debug_face_orientation` are PanelState DISPLAY fields (not
    // scene mutations), so they keep mutating in place.
    let mut voxel = state.scene.master_voxel_grid;
    let mut lattice = state.scene.master_block_lattice;
    let mut floor = state.scene.master_floor_grid;
    let mut masters_changed = false;
    // Issue #29 S4: the on-face voxel grid is per-object; this is the scene-wide
    // MASTER, ANDed (in the mesh shaders) with each node's own flag.
    masters_changed |= ui
        .checkbox(&mut voxel, "Voxel grid on faces (master)")
        .changed();
    // Issue #29 S3: scene-wide MASTERS for the per-object lattice / floor grids.
    masters_changed |= ui.checkbox(&mut lattice, "Block lattice (master)").changed();
    masters_changed |= ui.checkbox(&mut floor, "Floor grid (master)").changed();
    if masters_changed {
        response.emit(Intent::SetGridMasters { voxel, lattice, floor });
    }
    ui.checkbox(&mut state.show_view_cube, "View cube");
    // Issue #29 S2: the transform gizmo is now selection-driven (drawn on the
    // active node), so it no longer has a Display toggle.
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
/// 2D mid-vertical slice map. Z-up: layers are Z-slices. A video-clip-style track
/// over `0..grid_z` with two trim handles (lower/upper), the selected band
/// highlighted, block-boundary ticks, the layers/blocks readout, the snap + onion
/// controls, and the measured-diameter stat line (widest occupied run in the band).
fn build_layers_section(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    grid_z: u32,
    measured_diameter: u32,
) {
    ui.add_space(8.0);
    ui.strong("Layers");

    let voxels_per_block = state.geometry.voxels_per_block.max(1);
    // The scrubber edits `state.layer_range` in place; the bounds are kept valid
    // (clamped to grid_z, lower <= upper, snapped if requested) by the widget.
    layer_scrubber(ui, &mut state.layer_range, grid_z, voxels_per_block);

    let range = state.layer_range;
    // Readout: "layers L–U of N · blocks b0–b1".
    let block_lower = range.lower / voxels_per_block;
    let block_upper = range.upper.saturating_sub(1).max(range.lower) / voxels_per_block;
    ui.label(
        egui::RichText::new(format!(
            "layers {}–{} of {grid_z} · blocks {block_lower}–{block_upper}",
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
            LayerRange::snap_value(state.layer_range.lower, voxels_per_block, grid_z);
        state.layer_range.upper =
            LayerRange::snap_value(state.layer_range.upper, voxels_per_block, grid_z);
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

/// Custom range-scrubber widget (issue #12). Z-up: layers are Z-slices, so it paints
/// a track spanning `0..grid_z` with block-boundary ticks, the selected band
/// highlighted, and two draggable trim handles (lower/upper). Drag is handled via
/// `ui.interact` + the pointer: the nearer handle to the press grabs, then follows
/// the pointer (snapped to block boundaries when `snap_to_blocks` is on). Keeps
/// `lower <= upper` by swapping when the handles cross. Edits `range` in place.
fn layer_scrubber(
    ui: &mut egui::Ui,
    range: &mut LayerRange,
    grid_z: u32,
    voxels_per_block: u32,
) {
    let grid_z = grid_z.max(1);
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
        track_left + (layer as f32 / grid_z as f32) * track_width
    };
    let x_to_layer = |x: f32| -> u32 {
        let t = ((x - track_left) / track_width).clamp(0.0, 1.0);
        (t * grid_z as f32).round() as u32
    };

    let painter = ui.painter_at(rect);
    let visuals = ui.visuals();

    // Track background.
    painter.rect_filled(track_rect, 3.0, egui::Color32::from_rgb(0x1b, 0x17, 0x12));

    // Block-boundary tick marks every `voxels_per_block` layers (the snap points).
    let mut boundary = 0u32;
    while boundary <= grid_z {
        let x = layer_to_x(boundary);
        painter.line_segment(
            [egui::pos2(x, track_top), egui::pos2(x, track_bottom)],
            egui::Stroke::new(1.0, egui::Color32::from_rgb(0x3a, 0x5f, 0x57)),
        );
        if boundary == grid_z {
            break;
        }
        boundary = (boundary + voxels_per_block).min(grid_z);
        if boundary == grid_z {
            // Draw the final endpoint tick then stop.
            let x = layer_to_x(grid_z);
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
                value = LayerRange::snap_value(value, voxels_per_block, grid_z);
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
