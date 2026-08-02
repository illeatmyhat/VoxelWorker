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
//! A cell whose verb the document cannot express is drawn RESERVED — dimmed and inert. It is
//! deliberately not hidden: the shape of the finished set is information, and a verb that
//! silently appears later reads as a bug.

use document::intent::Intent;
use document::scene::NodeContent;
use document::sketch::{Operation, RevolveAxis, SketchSolid};
use voxel_core::voxel::ShapeKind;

use super::{hairline, region_frame, Edge, RAIL_WIDTH};
use crate::icons::{large::LargeIcon, Icon};
use crate::panel::{
    ArmedConstraint, ConstraintVerb, PanelResponse, PanelState, PositionSnap, SketchTool,
};
use crate::theme;

/// A shape cell: full rail width less the hairline, tall enough for the tile plus air.
const CELL_HEIGHT: f32 = 44.0;
/// A tool cell. The same height as a shape cell now that both carry a 32 px glyph — the two used
/// to differ only because the glyphs did.
const TOOL_CELL_HEIGHT: f32 = 44.0;
/// The tile glyph's box inside a shape cell.
const TILE_GLYPH: f32 = 32.0;
/// The rail glyph's box inside a tool cell (owner 2026-07-30 — 19 px read as small).
const TOOL_GLYPH: f32 = 32.0;
/// Opacity of a reserved cell — present, legible, plainly not yet clickable.
const RESERVED_DIM: f32 = 0.35;

/// The shape set, in catalog order: the authoring atom first, then the lifts, then the
/// primitives that are sugar over them.
///
/// `Some(kind)` is a shape the document can express, and clicking it ARMS live placement of that
/// primitive. `None` is a producer that has a glyph but no cursor-snap placement — sketch-family
/// verbs are authored through "+ Add" and the inspector, and `sweep` is reserved.
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

/// The sketch-mode rail toolset: the direct-manipulation vertex tools, each ARMING its
/// [`SketchTool`] on click. Rendered in place of `SHAPES`/`TOOLS` while a sketch is being
/// edited. Delete is NOT a tool: it acts on the selection via the Delete key or context menu.
const SKETCH_TOOLS: &[(Icon, &str, Option<SketchTool>)] = &[
    (
        Icon::SelectVertex,
        "Select / move vertex",
        Some(SketchTool::Select),
    ),
    (
        Icon::AddPoint,
        "Add point — split an edge",
        Some(SketchTool::AddPoint),
    ),
    (
        Icon::Line,
        "Line — click for straight segments; drag the live end for a tangent arc",
        Some(SketchTool::Line),
    ),
    (
        Icon::MidpointLine,
        "Midpoint Line — click midpoint, then endpoint",
        Some(SketchTool::MidpointLine),
    ),
    (
        Icon::Rectangle,
        "Rectangle — drag opposite corners",
        Some(SketchTool::Rectangle),
    ),
    (
        Icon::Rectangle3Point,
        "3-Point Rectangle — click base endpoints, then width",
        Some(SketchTool::Rectangle3Point),
    ),
    (
        Icon::RectangleCenterCorner,
        "Center Rectangle — drag from center to corner",
        Some(SketchTool::RectangleCenterCorner),
    ),
    (
        Icon::ThreePointArc,
        "Arc — click start, end, then a point it passes through",
        Some(SketchTool::ThreePointArc),
    ),
    (
        Icon::ArcCenterEndpoints,
        "Center Point Arc — click center, start, then end direction",
        Some(SketchTool::ArcCenterEndpoints),
    ),
    (
        Icon::ArcTangent,
        "Tangent Arc — click a line/arc endpoint, then the other endpoint",
        Some(SketchTool::ArcTangent),
    ),
    (
        Icon::CircleCenterDiameter,
        "Circle — click center, then perimeter",
        Some(SketchTool::CircleCenterDiameter),
    ),
    (
        Icon::Circle2Point,
        "2-Point Circle — click opposite diameter endpoints",
        Some(SketchTool::Circle2Point),
    ),
    (
        Icon::Circle3Point,
        "3-Point Circle — click three circumference points",
        Some(SketchTool::Circle3Point),
    ),
    (
        Icon::Circle2Tangent,
        "2-Tangent Circle — select two lines, then place the radius",
        Some(SketchTool::Circle2Tangent),
    ),
    (
        Icon::Circle3Tangent,
        "3-Tangent Circle — select three lines",
        Some(SketchTool::Circle3Tangent),
    ),
    (
        Icon::PolygonInscribed,
        "Inscribed Polygon — click center, then a vertex",
        Some(SketchTool::PolygonInscribed),
    ),
    (
        Icon::PolygonCircumscribed,
        "Circumscribed Polygon — click center, then an edge midpoint",
        Some(SketchTool::PolygonCircumscribed),
    ),
    (
        Icon::PolygonEdge,
        "Edge Polygon — click edge endpoints, then choose the body side",
        Some(SketchTool::PolygonEdge),
    ),
    (
        Icon::SlotCenterToCenter,
        "Center-to-Center Slot — click cap centers, then width",
        Some(SketchTool::SlotCenterToCenter),
    ),
    (
        Icon::SlotOverall,
        "Overall Slot — click overall endpoints, then width",
        Some(SketchTool::SlotOverall),
    ),
    (
        Icon::SlotCenterPoint,
        "Center Point Slot — click midpoint, cap center, then width",
        Some(SketchTool::SlotCenterPoint),
    ),
    (
        Icon::Slot3PointArc,
        "3-Point Arc Slot — click arc endpoints, through point, then width",
        Some(SketchTool::Slot3PointArc),
    ),
    (
        Icon::SlotCenterPointArc,
        "Center Point Arc Slot — click center, start, end direction, then width",
        Some(SketchTool::SlotCenterPointArc),
    ),
];

/// The sketch-mode position-snap picker (#96): how a vertex edit quantizes on the sketch
/// plane's own grid. One is always active; clicking another switches
/// [`PanelState::sketch_snap`] — pure editing state, no document write.
const SKETCH_SNAPS: &[(Icon, &str, PositionSnap)] = &[
    (
        Icon::SnapNone,
        "Position snap — none: the vertex lands exactly under the cursor, sub-voxel",
        PositionSnap::NoSnap,
    ),
    (
        Icon::SnapVoxel,
        "Position snap — voxel: whole-voxel grid crossings. The default",
        PositionSnap::Voxel,
    ),
    (
        Icon::SnapBlock,
        "Position snap — block: block boundaries, for clean inter-part mating",
        PositionSnap::Block,
    ),
];

/// The constraint verbs on the sketch rail. These ARM like the drawing tools do: the cell
/// lights, and the picks that follow fill the constraint's slots until it is complete, at which
/// point it applies and the cell goes dark again.
///
/// Only the verbs carrying a residual are here — an armable verb that asserts nothing is worse
/// than no cell. Ordered by how often a drawing reaches for them, not alphabetically and not by
/// arity:
/// Coincident and Horizontal/Vertical carry most of the work on a real profile, the angle pair
/// comes next, and the two that place one thing against another sit last.
const SKETCH_CONSTRAINTS: &[ConstraintVerb] = &[
    ConstraintVerb::Coincident,
    ConstraintVerb::HorizontalOrVertical,
    ConstraintVerb::Parallel,
    ConstraintVerb::Perpendicular,
    ConstraintVerb::Equal,
    ConstraintVerb::Collinear,
    ConstraintVerb::Concentric,
    ConstraintVerb::Symmetry,
    ConstraintVerb::Tangent,
    ConstraintVerb::Midpoint,
    ConstraintVerb::Fix,
];

/// Sketch modification verbs in catalog order. A live bit means the command has a complete
/// document operation and shell route; the remaining glyphs stay visible but inert until that
/// contract exists. Keeping the inventory here prevents implementation order from silently
/// becoming the eventual rail order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SketchModifierRoute {
    Reserved,
    ConstructionAction,
    Tool(SketchTool),
}

const SKETCH_MODIFIERS: &[(Icon, &str, SketchModifierRoute)] = &[
    (
        Icon::ConstructionToggle,
        "Construction — toggle selected geometry between profile and reference",
        SketchModifierRoute::ConstructionAction,
    ),
    (
        Icon::Fillet,
        "Fillet — click along either leg to round its two-line corner",
        SketchModifierRoute::Tool(SketchTool::Fillet),
    ),
    (
        Icon::ChamferEqual,
        "Equal Distance Chamfer — click one leg to bevel both equally",
        SketchModifierRoute::Tool(SketchTool::ChamferEqual),
    ),
    (
        Icon::ChamferDistanceAngle,
        "Distance and Angle Chamfer — choose a distance, then the angle on the other leg",
        SketchModifierRoute::Tool(SketchTool::ChamferDistanceAngle),
    ),
    (
        Icon::ChamferTwoDistance,
        "Two Distance Chamfer — choose one tangent point on each leg",
        SketchModifierRoute::Tool(SketchTool::ChamferTwoDistance),
    ),
    (
        Icon::Trim,
        "Trim — remove the clicked interval to its neighboring intersections",
        SketchModifierRoute::Tool(SketchTool::Trim),
    ),
    (
        Icon::Extend,
        "Extend — grow the nearest endpoint to the next intersection",
        SketchModifierRoute::Tool(SketchTool::Extend),
    ),
    (
        Icon::BreakCurve,
        "Break — split a curve at its intersections",
        SketchModifierRoute::Tool(SketchTool::BreakCurve),
    ),
    (
        Icon::OffsetCurve,
        "Offset — select a curve, then place its parallel or concentric copy",
        SketchModifierRoute::Tool(SketchTool::Offset),
    ),
    (
        Icon::MoveCopy,
        "Move / Copy — choose a base and destination; hold Shift to copy",
        SketchModifierRoute::Tool(SketchTool::MoveCopy),
    ),
    (
        Icon::SketchScale,
        "Scale — choose a center, then a uniform size",
        SketchModifierRoute::Tool(SketchTool::Scale),
    ),
    (
        Icon::BlendCurve,
        "Blend Curve — reserved",
        SketchModifierRoute::Reserved,
    ),
];

/// The set-operation picker on the sketch rail — the operation is a property of the same fused
/// node. Extrude and Revolve switch the edited node's operation on click; Sweep is the reserved
/// arm, drawn dimmed.
const SKETCH_OPS: &[(Icon, &str, bool)] = &[
    (Icon::Extrude, "Extrude (set operation)", false),
    (Icon::Revolve, "Revolve (set operation)", false),
    (Icon::Sweep, "Sweep — reserved", true),
];

/// Build the pinned rail column. In **sketch mode** it swaps to the sketch toolset;
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
            hairline(ui.painter(), column, Edge::Right, theme::BORDER);

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
                    if state.sketch_mode.is_some() {
                        build_sketch_rail(ui, state, response);
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

/// The swapped rail while a sketch is being edited: the accent `SKETCH` head (the whole-mode
/// indicator), the vertex tools, then an `OP` separator and the set-operation picker. The
/// armable vertex tools select [`PanelState::sketch_tool`] on click and light the active one;
/// the current operation reads active and clicking the other one SWITCHES the edited node's
/// operation; `Sweep` reads reserved.
fn build_sketch_rail(ui: &mut egui::Ui, state: &mut PanelState, response: &mut PanelResponse) {
    // The edited node's producer: lights the matching OP cell and seeds an operation switch.
    let target = state.sketch_mode;
    let producer = target
        .and_then(|id| state.scene.node_by_id(id))
        .and_then(|node| match &node.content {
            NodeContent::SketchTool { producer, .. } => Some(producer.clone()),
            _ => None,
        });
    let current_op = producer.as_ref().map(|producer| &producer.operation);
    let op_is_active = |icon: Icon| {
        matches!(
            (&current_op, icon),
            (Some(Operation::Extrude { .. }), Icon::Extrude)
                | (Some(Operation::Revolve { .. }), Icon::Revolve)
        )
    };

    rail_heading_active(ui, "Sketch");
    // The vertex tools. An armable tool (Some) lights when it is the current `sketch_tool` and
    // selects it on click; unavailable entries (None) remain visibly reserved.
    for &(icon, tip, tool) in SKETCH_TOOLS {
        match tool {
            Some(tool) => {
                let active = state.sketch_tool == tool;
                if sketch_tool_cell(ui, icon, tip, active) {
                    state.sketch_tool = tool;
                }
            }
            None => sketch_cell(ui, icon, tip, false, true),
        }
    }
    rail_heading(ui, "Modify");
    for &(icon, tip, route) in SKETCH_MODIFIERS {
        match route {
            SketchModifierRoute::Reserved => sketch_cell(ui, icon, tip, false, true),
            SketchModifierRoute::ConstructionAction => {
                if sketch_tool_cell(ui, icon, tip, false) {
                    response.toggle_sketch_construction = true;
                }
            }
            SketchModifierRoute::Tool(tool) => {
                let active = state.sketch_tool == tool;
                if sketch_tool_cell(ui, icon, tip, active) {
                    state.sketch_tool = tool;
                }
            }
        }
    }
    if matches!(
        state.sketch_tool,
        SketchTool::PolygonInscribed | SketchTool::PolygonCircumscribed | SketchTool::PolygonEdge
    ) {
        if !(3..=128).contains(&state.sketch_polygon_sides) {
            state.sketch_polygon_sides = 6;
        }
        ui.horizontal(|ui| {
            ui.label("Sides");
            ui.add(
                egui::DragValue::new(&mut state.sketch_polygon_sides)
                    .range(3..=128)
                    .speed(0.1),
            );
        });
    }
    rail_heading(ui, "Snap");
    for &(icon, tip, snap) in SKETCH_SNAPS {
        let active = state.sketch_snap == snap;
        if sketch_tool_cell(ui, icon, tip, active) {
            state.sketch_snap = snap;
        }
    }
    rail_heading(ui, "Constrain");
    for &verb in SKETCH_CONSTRAINTS {
        let icon = verb.icon();
        let armed = state
            .armed_constraint
            .as_ref()
            .is_some_and(|armed| armed.verb() == verb);
        if sketch_tool_cell(ui, icon, verb.tooltip(), armed) {
            // Pressing the ARMED cell cancels, the way clicking an armed shape cell disarms
            // placement: the same press that started a command is the obvious way to abandon it.
            state.armed_constraint = if armed {
                None
            } else {
                Some(ArmedConstraint::new(verb))
            };
            // The gesture's picks ARE the selection while it runs, so it starts from empty.
            // Inheriting whatever was picked before would fill slots the author never aimed at.
            state.selection.clear_sketch_entities();
            // The last refusal answered the last gesture. Leaving it on the viewport notice while
            // a new one starts would read as this tool having already said no.
            state.sketch_constraint_refusal = None;
        }
    }
    rail_heading(ui, "Op");
    for &(icon, tip, reserved) in SKETCH_OPS {
        let active = op_is_active(icon);
        if reserved {
            sketch_cell(ui, icon, tip, active, true);
            continue;
        }
        // A live op cell is armable like a tool cell; clicking the inactive one switches the
        // node's operation through the SAME `SetSketch` door the inspector's picker uses,
        // carrying the same switch defaults. Clicking the active one changes nothing.
        if sketch_tool_cell(ui, icon, tip, active) && !active {
            if let (Some(target), Some(producer), Some(context)) = (
                target,
                producer.as_ref(),
                document::sketch::evaluation_context_from_density(state.scene.voxels_per_block),
            ) {
                response.emit_and_frame(Intent::SetSketch {
                    target,
                    producer: producer_switched_to(producer, icon, context),
                });
            }
        }
    }
}

/// The edited producer with its operation switched to the clicked OP cell's kind, the profile
/// preserved. Switch defaults mirror the inspector's Operation picker: Extrude seeds its height
/// from the rectangle depth span (else 16); Revolve seeds a full 360° turn about the first
/// in-plane axis.
fn producer_switched_to(
    producer: &SketchSolid,
    icon: Icon,
    context: parametric::EvaluationContext,
) -> SketchSolid {
    let sketch = producer.sketch.clone();
    match icon {
        Icon::Extrude => {
            let height = producer
                .rectangle_in_plane_spans(context)
                .map(|spans| spans[1])
                .unwrap_or(16)
                .max(1);
            SketchSolid::extrude(sketch, height)
        }
        _ => SketchSolid::revolve(sketch, RevolveAxis::InPlane0, 360),
    }
}

/// A rail section heading: UPPERCASE micro-label over a hairline.
fn rail_heading(ui: &mut egui::Ui, title: &str) {
    ui.add_space(9.0);
    let galley = theme::letter_spaced(ui, title, theme::TEXT_HINT, 8.0, 1.2);
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(RAIL_WIDTH, galley.size().y + 5.0),
        egui::Sense::hover(),
    );
    let at = egui::pos2(rect.center().x - galley.size().x * 0.5, rect.top());
    ui.painter().galley(at, galley, theme::TEXT_HINT);
    hairline(ui.painter(), rect, Edge::Bottom, theme::RULE);
}

/// The **active** rail heading — the accent-filled `SKETCH` label that is the whole mode
/// indicator: dark text on the accent fill, spanning the rail. Distinct from
/// [`rail_heading`]'s faint hairline label so entering the mode is unmistakable at a glance.
fn rail_heading_active(ui: &mut egui::Ui, title: &str) {
    let galley = theme::letter_spaced(ui, title, theme::BG, 9.0, 1.6);
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(RAIL_WIDTH, galley.size().y + 15.0),
        egui::Sense::hover(),
    );
    ui.painter().rect_filled(rect, 0.0, theme::ACCENT);
    let at = egui::pos2(
        rect.center().x - galley.size().x * 0.5,
        rect.center().y - galley.size().y * 0.5,
    );
    ui.painter().galley(at, galley, theme::BG);
}

/// One **armable** sketch-tool rail cell: a clickable tool glyph that lights
/// when `active`. Returns `true` the frame it is clicked, so the caller arms the tool. The
/// active bar / hover fill are the shared rail treatment ([`paint_cell`]).
fn sketch_tool_cell(ui: &mut egui::Ui, icon: Icon, tip: &str, active: bool) -> bool {
    let (rect, cell) = ui.allocate_exact_size(
        egui::vec2(RAIL_WIDTH, TOOL_CELL_HEIGHT),
        egui::Sense::click(),
    );
    let hovered = cell.hovered();
    paint_cell(ui, rect, active, hovered);
    let color = cell_ink(active, hovered, false);
    let glyph = egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(TOOL_GLYPH));
    icon.draw(ui.painter(), glyph, color);
    let cell = cell.on_hover_text(tip.to_string());
    cell.clicked()
}

/// One **inert** sketch-mode rail cell: a glyph whose verb is reserved — drawn with the active
/// accent bar and the reserved dim treatment, reporting hover + tooltip but arming nothing. Live
/// op cells go through [`sketch_tool_cell`] instead.
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
    let tip = if reserved {
        format!("{tip} — reserved")
    } else {
        tip.to_string()
    };
    cell.on_hover_text(tip);
}

/// One shape cell. Clicking an expressible shape ARMS live placement of that primitive — the
/// same flow as the scene panel's "+ Add" chips: a ghost follows the cursor and a stationary
/// click drops a node, staying armed for repeats. Clicking the already-armed cell disarms. The
/// cell is unrelated to the selection and leaves the inspector's mirror alone — the armed spec
/// takes the kind's own default size at current density/wall/material.
fn shape_cell(
    ui: &mut egui::Ui,
    icon: Icon,
    kind: Option<ShapeKind>,
    state: &PanelState,
    response: &mut PanelResponse,
) {
    let reserved = kind.is_none();
    // Armed, not selected-node shape: the accent means "this is in your hand".
    let armed = kind.is_some() && state.armed_shape() == kind;

    let sense = if reserved {
        egui::Sense::hover()
    } else {
        egui::Sense::click()
    };
    let (rect, cell) = ui.allocate_exact_size(egui::vec2(RAIL_WIDTH, CELL_HEIGHT), sense);
    paint_cell(ui, rect, armed, cell.hovered() && !reserved);

    let color = cell_ink(armed, cell.hovered() && !reserved, reserved);
    let glyph = egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(TILE_GLYPH));
    // The tile drawing where the noun has one; otherwise its rail twin, which is the
    // designed fallback rather than a missing asset.
    match LargeIcon::for_icon(icon) {
        Some(tile) => tile.draw(ui.painter(), glyph, color),
        None => icon.draw(ui.painter(), glyph.shrink(3.0), color),
    }

    let tip = if reserved {
        format!("{} — reserved", icon.name())
    } else if armed {
        format!("{} — armed, click to put it down", icon.name())
    } else {
        format!("{} — click to place", icon.name())
    };
    let cell = cell.on_hover_text(tip);

    if let (true, Some(kind)) = (cell.clicked(), kind) {
        if armed {
            response.disarm_tool = true;
        } else {
            response.arm_tool = Some(crate::panel::tool_node_spec(kind, state));
        }
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
        painter.rect_filled(rect, 0.0, theme::ACTIVE_BG);
    } else if active {
        painter.rect_filled(rect, 0.0, theme::HOVER_BG);
    }
    if active {
        let bar = egui::Rect::from_min_size(rect.left_top(), egui::vec2(2.0, rect.height()));
        painter.rect_filled(bar, 0.0, theme::ACCENT);
    }
}

/// A cell glyph's ink: accent when active, lifted on hover, dimmed when reserved.
fn cell_ink(active: bool, hovered: bool, reserved: bool) -> egui::Color32 {
    if reserved {
        theme::TEXT_MUTED.gamma_multiply(RESERVED_DIM)
    } else if active {
        theme::ACCENT
    } else if hovered {
        theme::TEXT_HOVER
    } else {
        theme::TEXT_MUTED
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sketch_rail_exposes_line_with_the_complete_line_glyph() {
        assert!(SKETCH_TOOLS.iter().any(|&(icon, tip, tool)| {
            icon == Icon::Line && tip.starts_with("Line —") && tool == Some(SketchTool::Line)
        }));
    }

    #[test]
    fn sketch_rail_lists_the_complete_modifier_catalog_and_only_live_commands_arm() {
        let icons: Vec<_> = SKETCH_MODIFIERS.iter().map(|&(icon, _, _)| icon).collect();
        assert_eq!(
            icons,
            vec![
                Icon::ConstructionToggle,
                Icon::Fillet,
                Icon::ChamferEqual,
                Icon::ChamferDistanceAngle,
                Icon::ChamferTwoDistance,
                Icon::Trim,
                Icon::Extend,
                Icon::BreakCurve,
                Icon::OffsetCurve,
                Icon::MoveCopy,
                Icon::SketchScale,
                Icon::BlendCurve,
            ]
        );
        assert_eq!(
            SKETCH_MODIFIERS
                .iter()
                .filter_map(|&(icon, _, route)| {
                    (route != SketchModifierRoute::Reserved).then_some(icon)
                })
                .collect::<Vec<_>>(),
            vec![
                Icon::ConstructionToggle,
                Icon::Fillet,
                Icon::ChamferEqual,
                Icon::ChamferDistanceAngle,
                Icon::ChamferTwoDistance,
                Icon::Trim,
                Icon::Extend,
                Icon::BreakCurve,
                Icon::OffsetCurve,
                Icon::MoveCopy,
                Icon::SketchScale,
            ]
        );
    }

    #[test]
    fn sketch_rail_places_midpoint_line_immediately_after_line() {
        assert!(SKETCH_TOOLS.windows(2).any(|pair| {
            matches!(
                pair,
                [
                    (_, _, Some(SketchTool::Line)),
                    (
                        Icon::MidpointLine,
                        "Midpoint Line — click midpoint, then endpoint",
                        Some(SketchTool::MidpointLine)
                    )
                ]
            )
        }));
    }

    #[test]
    fn sketch_rail_places_center_and_tangent_arcs_after_three_point_arc() {
        assert!(SKETCH_TOOLS.windows(3).any(|items| {
            matches!(
                items,
                [
                    (_, _, Some(SketchTool::ThreePointArc)),
                    (
                        Icon::ArcCenterEndpoints,
                        "Center Point Arc — click center, start, then end direction",
                        Some(SketchTool::ArcCenterEndpoints)
                    ),
                    (
                        Icon::ArcTangent,
                        "Tangent Arc — click a line/arc endpoint, then the other endpoint",
                        Some(SketchTool::ArcTangent)
                    )
                ]
            )
        }));
    }

    #[test]
    fn sketch_rail_groups_all_circle_grammars() {
        assert!(SKETCH_TOOLS.windows(5).any(|items| {
            matches!(
                items,
                [
                    (_, _, Some(SketchTool::CircleCenterDiameter)),
                    (Icon::Circle2Point, _, Some(SketchTool::Circle2Point)),
                    (Icon::Circle3Point, _, Some(SketchTool::Circle3Point)),
                    (Icon::Circle2Tangent, _, Some(SketchTool::Circle2Tangent)),
                    (Icon::Circle3Tangent, _, Some(SketchTool::Circle3Tangent))
                ]
            )
        }));
    }

    #[test]
    fn sketch_rail_groups_all_polygon_grammars_after_circles() {
        assert!(SKETCH_TOOLS.windows(4).any(|items| {
            matches!(
                items,
                [
                    (_, _, Some(SketchTool::Circle3Tangent)),
                    (
                        Icon::PolygonInscribed,
                        _,
                        Some(SketchTool::PolygonInscribed)
                    ),
                    (
                        Icon::PolygonCircumscribed,
                        _,
                        Some(SketchTool::PolygonCircumscribed)
                    ),
                    (Icon::PolygonEdge, _, Some(SketchTool::PolygonEdge))
                ]
            )
        }));
    }

    #[test]
    fn sketch_rail_groups_all_five_slot_grammars() {
        assert!(SKETCH_TOOLS.windows(5).any(|items| {
            matches!(
                items,
                [
                    (
                        Icon::SlotCenterToCenter,
                        _,
                        Some(SketchTool::SlotCenterToCenter)
                    ),
                    (Icon::SlotOverall, _, Some(SketchTool::SlotOverall)),
                    (Icon::SlotCenterPoint, _, Some(SketchTool::SlotCenterPoint)),
                    (Icon::Slot3PointArc, _, Some(SketchTool::Slot3PointArc)),
                    (
                        Icon::SlotCenterPointArc,
                        _,
                        Some(SketchTool::SlotCenterPointArc)
                    )
                ]
            )
        }));
    }

    #[test]
    fn sketch_rail_groups_all_rectangle_grammars() {
        assert!(SKETCH_TOOLS.windows(3).any(|items| {
            matches!(
                items,
                [
                    (_, _, Some(SketchTool::Rectangle)),
                    (Icon::Rectangle3Point, _, Some(SketchTool::Rectangle3Point)),
                    (
                        Icon::RectangleCenterCorner,
                        _,
                        Some(SketchTool::RectangleCenterCorner)
                    )
                ]
            )
        }));
    }
}
