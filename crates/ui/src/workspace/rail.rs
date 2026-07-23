//! The pinned rail: the two sets that cannot grow.
//!
//! Shapes and tools are a finite verb set, so they are permanent furniture the user builds
//! muscle memory against — never summoned, never searched. Materials and saved parts grow
//! with the project and live in the drawer instead. That single rule decides membership, so
//! nothing appears in both places.
//!
//! Shape cells take the TILE glyph at 26 px, falling back to the rail mark where a noun has
//! no tile drawing. The two families are separate drawings of the same noun rather than one
//! asset scaled, so the fallback is designed rather than a gap.
//!
//! A cell whose verb the document cannot yet express is drawn RESERVED — dimmed and inert,
//! the treatment the design mock gives `sweep`. It is deliberately not hidden: the shape of
//! the finished set is information, and a verb that silently appears later reads as a bug.

use document::intent::Intent;
use document::scene::NodeContent;
use document::sketch::Operation;
use document::voxel::SdfShape;
use voxel_core::voxel::ShapeKind;

use super::{hairline, region_frame, Edge, RAIL_WIDTH};
use crate::icons::{large::LargeIcon, Icon};
use crate::panel::{PanelResponse, PanelState};
use crate::signal_theme;

/// A shape cell: full rail width less the hairline, tall enough for a 26 px tile plus air.
const CELL_HEIGHT: f32 = 37.0;
/// A tool cell — rail glyphs at 19 px, so the cell is shorter.
const TOOL_CELL_HEIGHT: f32 = 30.0;
/// The tile glyph's box inside a shape cell.
const TILE_GLYPH: f32 = 26.0;
/// The rail glyph's box inside a tool cell.
const TOOL_GLYPH: f32 = 19.0;
/// Opacity of a reserved cell — present, legible, plainly not yet clickable.
const RESERVED_DIM: f32 = 0.35;

/// The shape set, in the order the design sheet pins it: the authoring atom first, then the
/// lifts, then the primitives that are sugar over them.
///
/// `Some(kind)` is a shape the document can express today, and clicking it retargets the
/// selected Tool. `None` is a producer that has a glyph but no reachable intent from here
/// yet — sketch-family verbs are authored through the inspector, and `sweep` is reserved.
const SHAPES: &[(Icon, Option<ShapeKind>)] = &[
    (Icon::Sketch, None),
    (Icon::Extrude, None),
    (Icon::Revolve, None),
    (Icon::Sweep, None),
    (Icon::BoxSolid, Some(ShapeKind::Box)),
    (Icon::Sphere, Some(ShapeKind::Sphere)),
    (Icon::Cylinder, Some(ShapeKind::Cylinder)),
    (Icon::Tube, Some(ShapeKind::Tube)),
    (Icon::Torus, Some(ShapeKind::Torus)),
    (Icon::HalfSpace, None),
];

/// The tool set. Only selection exists as a mode today; the rest are drawn reserved so the
/// finished shape of the toolbelt is visible without pretending the verbs work.
const TOOLS: &[(Icon, bool)] = &[
    (Icon::AxesGizmo, true),
    (Icon::SculptAdd, false),
    (Icon::Carve, false),
    (Icon::Material, false),
    (Icon::Probe, false),
    (Icon::Measure, false),
];

/// The sketch-mode rail toolset (ADR 0028): the direct-manipulation vertex tools. Rendered in
/// place of `SHAPES`/`TOOLS` while a sketch is being edited. Slice 1 (#93) shows them as the
/// mode indicator; the tools themselves arm in later slices (#94 select/move, #95 add/delete).
const SKETCH_TOOLS: &[(Icon, &str)] = &[
    (Icon::SelectVertex, "Select / move vertex"),
    (Icon::Polyline, "Line / polyline"),
    (Icon::Rectangle, "Rectangle"),
    (Icon::DeleteVertex, "Delete vertex"),
];

/// The set-operation picker on the sketch rail (ADR 0028 §1: the operation is a property of
/// the SAME fused node, moved here from the deleted right panel). Extrude + Revolve ship;
/// Sweep is the reserved arm (drawn dimmed). The picker is wired in #97.
const SKETCH_OPS: &[(Icon, &str, bool)] = &[
    (Icon::Extrude, "Extrude (set operation)", false),
    (Icon::Revolve, "Revolve (set operation)", false),
    (Icon::Sweep, "Sweep — reserved", true),
];

/// Build the pinned rail column. In **sketch mode** (ADR 0028) it swaps to the sketch toolset;
/// otherwise it shows the normal Shape + Tool sets.
pub(super) fn build_rail(
    root_ui: &mut egui::Ui,
    state: &mut PanelState,
    response: &mut PanelResponse,
) {
    egui::Panel::left("workspace_rail")
        .resizable(false)
        .default_size(RAIL_WIDTH)
        .frame(region_frame())
        .show_inside(root_ui, |ui| {
            let column = ui.max_rect();
            hairline(ui.painter(), column, Edge::Right, signal_theme::BORDER);

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
                    if state.sketch_mode.is_some() {
                        build_sketch_rail(ui, state);
                    } else {
                        rail_heading(ui, "Shape");
                        for &(icon, kind) in SHAPES {
                            shape_cell(ui, icon, kind, state, response);
                        }
                        rail_heading(ui, "Tool");
                        for &(icon, enabled) in TOOLS {
                            tool_cell(ui, icon, enabled);
                        }
                    }
                });
        });
}

/// The swapped rail while a sketch is being edited (ADR 0028): the accent `SKETCH` head (the
/// whole-mode indicator), the vertex tools, then an `OP` separator and the set-operation
/// picker. Slice 1 draws the tools inert — `Select` reads active, the current operation reads
/// active, `Sweep` reads reserved; arming them is later slices (#94–#97).
fn build_sketch_rail(ui: &mut egui::Ui, state: &PanelState) {
    // The current operation of the edited node, to light the matching OP cell.
    let current_op = state
        .sketch_mode
        .and_then(|id| state.scene.node_by_id(id))
        .and_then(|node| match &node.content {
            NodeContent::SketchTool { producer, .. } => Some(producer.operation.clone()),
            _ => None,
        });
    let op_is_active = |icon: Icon| {
        matches!(
            (&current_op, icon),
            (Some(Operation::Extrude { .. }), Icon::Extrude)
                | (Some(Operation::Revolve { .. }), Icon::Revolve)
        )
    };

    rail_heading_active(ui, "Sketch");
    // The vertex tools. Select is the default active tool in slice 1; the rest are drawn
    // live-but-inert (their arming lands with the editing slices).
    for (index, &(icon, tip)) in SKETCH_TOOLS.iter().enumerate() {
        sketch_cell(ui, icon, tip, index == 0, false);
    }
    rail_heading(ui, "Op");
    for &(icon, tip, reserved) in SKETCH_OPS {
        sketch_cell(ui, icon, tip, op_is_active(icon), reserved);
    }
}

/// A rail section heading: UPPERCASE micro-label over a hairline.
fn rail_heading(ui: &mut egui::Ui, title: &str) {
    ui.add_space(9.0);
    let galley = signal_theme::letter_spaced(ui, title, signal_theme::TEXT_HINT, 8.0, 1.2);
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(RAIL_WIDTH, galley.size().y + 5.0),
        egui::Sense::hover(),
    );
    let at = egui::pos2(rect.center().x - galley.size().x * 0.5, rect.top());
    ui.painter().galley(at, galley, signal_theme::TEXT_HINT);
    hairline(ui.painter(), rect, Edge::Bottom, signal_theme::RULE);
}

/// The **active** rail heading — the accent-filled `SKETCH` label that is the whole mode
/// indicator (ADR 0028, C2 mock's `.railhead`): dark text on the accent fill, spanning the
/// rail. Distinct from [`rail_heading`]'s faint hairline label so entering the mode is
/// unmistakable at a glance.
fn rail_heading_active(ui: &mut egui::Ui, title: &str) {
    let galley = signal_theme::letter_spaced(ui, title, signal_theme::BG, 9.0, 1.6);
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(RAIL_WIDTH, galley.size().y + 15.0),
        egui::Sense::hover(),
    );
    ui.painter().rect_filled(rect, 0.0, signal_theme::ACCENT);
    let at = egui::pos2(rect.center().x - galley.size().x * 0.5, rect.center().y - galley.size().y * 0.5);
    ui.painter().galley(at, galley, signal_theme::BG);
}

/// One sketch-mode rail cell (ADR 0028): a tool or set-operation glyph at rail-glyph size,
/// with the active accent bar and the reserved dim treatment. Slice 1 is display-only — the
/// cells report hover + tooltip but arm nothing yet (later slices wire the clicks).
fn sketch_cell(ui: &mut egui::Ui, icon: Icon, tip: &str, active: bool, reserved: bool) {
    let (rect, cell) = ui.allocate_exact_size(
        egui::vec2(RAIL_WIDTH, TOOL_CELL_HEIGHT),
        egui::Sense::hover(),
    );
    let hovered = cell.hovered() && !reserved;
    paint_cell(ui, rect, active, hovered);
    let color = cell_ink(active, hovered, reserved);
    let glyph = egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(TOOL_GLYPH));
    icon.draw(ui.painter(), glyph, color);
    cell.on_hover_text(tip.to_string());
}

/// One shape cell. Clicking an expressible shape retargets the SELECTED Tool node.
///
/// The target is read once, up front, and carried into the intent — an edit must never
/// resolve its own target through the selection at apply time, or it silently retargets when
/// the selection moves. A pure shape switch emits WITHOUT an auto-frame: re-resolving at the
/// same size must not move the camera.
fn shape_cell(
    ui: &mut egui::Ui,
    icon: Icon,
    kind: Option<ShapeKind>,
    state: &mut PanelState,
    response: &mut PanelResponse,
) {
    let target = state.scene.active;
    let reserved = kind.is_none() || target.is_none();
    let active = kind.is_some_and(|k| k == state.geometry.shape) && !reserved;

    let sense = if reserved {
        egui::Sense::hover()
    } else {
        egui::Sense::click()
    };
    let (rect, cell) = ui.allocate_exact_size(egui::vec2(RAIL_WIDTH, CELL_HEIGHT), sense);
    paint_cell(ui, rect, active, cell.hovered() && !reserved);

    let color = cell_ink(active, cell.hovered() && !reserved, reserved);
    let glyph = egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(TILE_GLYPH));
    // The tile drawing where the noun has one; otherwise its rail twin, which is the
    // designed fallback rather than a missing asset.
    match LargeIcon::for_icon(icon) {
        Some(tile) => tile.draw(ui.painter(), glyph, color),
        None => icon.draw(ui.painter(), glyph.shrink(3.0), color),
    }

    let tip = if reserved && kind.is_none() {
        format!("{} — reserved", icon.name())
    } else if reserved {
        format!("{} — select a shape node first", icon.name())
    } else {
        icon.name().to_string()
    };
    let cell = cell.on_hover_text(tip);

    if let (true, Some(kind), Some(target)) = (cell.clicked(), kind, target) {
        state.geometry.shape = kind;
        let shape = SdfShape::from_geometry(state.geometry.clone());
        response.emit(Intent::SetShape { target, shape });
    }
}

/// One tool cell, at rail-glyph size.
fn tool_cell(ui: &mut egui::Ui, icon: Icon, enabled: bool) {
    let sense = if enabled {
        egui::Sense::click()
    } else {
        egui::Sense::hover()
    };
    let (rect, cell) = ui.allocate_exact_size(egui::vec2(RAIL_WIDTH, TOOL_CELL_HEIGHT), sense);
    let hovered = cell.hovered() && enabled;
    // Selection is the only live tool, so it is the one that reads active.
    paint_cell(ui, rect, enabled, hovered);

    let color = cell_ink(enabled, hovered, !enabled);
    let glyph = egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(TOOL_GLYPH));
    icon.draw(ui.painter(), glyph, color);

    let tip = if enabled {
        icon.name().to_string()
    } else {
        format!("{} — reserved", icon.name())
    };
    cell.on_hover_text(tip);
}

/// The cell's fill and its active bar. Active is a 2 px accent inset on the leading edge —
/// never a glow, and never a second hue.
fn paint_cell(ui: &egui::Ui, rect: egui::Rect, active: bool, hovered: bool) {
    let painter = ui.painter();
    if hovered {
        painter.rect_filled(rect, 0.0, signal_theme::ACTIVE_BG);
    } else if active {
        painter.rect_filled(rect, 0.0, signal_theme::HOVER_BG);
    }
    if active {
        let bar = egui::Rect::from_min_size(rect.left_top(), egui::vec2(2.0, rect.height()));
        painter.rect_filled(bar, 0.0, signal_theme::ACCENT);
    }
}

/// A cell glyph's ink: accent when active, lifted on hover, dimmed when reserved.
fn cell_ink(active: bool, hovered: bool, reserved: bool) -> egui::Color32 {
    if reserved {
        signal_theme::TEXT_MUTED.gamma_multiply(RESERVED_DIM)
    } else if active {
        signal_theme::ACCENT
    } else if hovered {
        signal_theme::TEXT_HOVER
    } else {
        signal_theme::TEXT_MUTED
    }
}
