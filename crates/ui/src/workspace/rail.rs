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

/// The grid's glyph column: one tool tile, the 32 px glyph with air around it.
pub(super) const CELL_WIDTH: f32 = 44.0;

/// The scroll indicator's lane, down the LEFT edge of the column.
///
/// Left because the right edge is the chevron column, and egui draws its own bar hard against the
/// right — over the very tile that opens a family. egui has no side to put it on other than that
/// one ([`egui::containers::scroll_area`] pins the bar to `max_cross`), so the rail hides egui's
/// bar and paints its own here.
pub(super) const BAR_LANE: f32 = 10.0;

/// One grid row: both columns, the lane excluded.
const ROW_WIDTH: f32 = CELL_WIDTH + CHEVRON_WIDTH;
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

/// One family of rail verbs, drawn as `[face][>]` — a single cell whose chevron slides the rest
/// out sideways, into `[face][member][member][>]`.
///
/// Sideways rather than down: the rail is a vertical scroll column, so a list unfolding downward
/// pushes every section below it and moves the target the author is reaching for. To the right
/// there is only viewport, and nothing to push.
///
/// # The face is fixed
///
/// A group's face is its canonical member, always — never the last one used. Fusion's last-used
/// face exists to serve re-invoking the same variant in a UI where a tool DISARMS after each use;
/// here a tool stays armed until it is disarmed, so the arming model already delivers what
/// last-used is for. What would remain is only its cost: the same pixel meaning a different verb
/// depending on history, which is what muscle memory cannot absorb.
///
/// A family of one has no chevron and draws as an ordinary cell. That is not a special case in
/// the rendering — it falls out of having nothing else to show.
struct RailGroup<Route: 'static> {
    /// The mark this family always shows on the rail.
    face: Icon,
    /// The family's name: the face's tooltip, and the id the open row animates under.
    name: &'static str,
    /// The verbs in the family, canonical member first.
    members: &'static [(Icon, &'static str, Route)],
}

/// The sketch CREATE families, in Fusion's own order and with Fusion's own membership.
///
/// # Fusion's cut, warts included
///
/// The owner has spent this epic reverse-engineering Fusion entity by entity and asked for
/// "Fusion-like tool groups" in those words, so a taxonomy invented here would make them relearn
/// a layout they already know and invite a re-cut at every "in Fusion this is under Modify".
/// Fusion's cut is also the principled one a level up: create MINTS entities, modify REWRITES
/// what is there, constrain RELATES them. Every verb does exactly one of those, including the
/// ones not written yet, which is why the cut does not need re-cutting as the set grows.
///
/// Deviations from Fusion, each deliberate: Midpoint Line rides in the Line family (Fusion has no
/// such grammar), and the three chamfers ride in one Modify family for the same reason. Select
/// and the snap picker are not Create verbs and sit outside this table.
const SKETCH_CREATE: &[RailGroup<SketchTool>] = &[
    RailGroup {
        face: Icon::Line,
        name: "Line",
        members: &[
            (
                Icon::Line,
                "Line — click for straight segments; drag the live end for a tangent arc",
                SketchTool::Line,
            ),
            (
                Icon::MidpointLine,
                "Midpoint Line — click midpoint, then endpoint",
                SketchTool::MidpointLine,
            ),
        ],
    },
    RailGroup {
        face: Icon::Rectangle,
        name: "Rectangle",
        members: &[
            (
                Icon::Rectangle,
                "Rectangle — drag opposite corners",
                SketchTool::Rectangle,
            ),
            (
                Icon::Rectangle3Point,
                "3-Point Rectangle — click base endpoints, then width",
                SketchTool::Rectangle3Point,
            ),
            (
                Icon::RectangleCenterCorner,
                "Center Rectangle — drag from center to corner",
                SketchTool::RectangleCenterCorner,
            ),
        ],
    },
    RailGroup {
        face: Icon::CircleCenterDiameter,
        name: "Circle",
        members: &[
            (
                Icon::CircleCenterDiameter,
                "Circle — click center, then perimeter",
                SketchTool::CircleCenterDiameter,
            ),
            (
                Icon::Circle2Point,
                "2-Point Circle — click opposite diameter endpoints",
                SketchTool::Circle2Point,
            ),
            (
                Icon::Circle3Point,
                "3-Point Circle — click three circumference points",
                SketchTool::Circle3Point,
            ),
            (
                Icon::Circle2Tangent,
                "2-Tangent Circle — select two lines, then place the radius",
                SketchTool::Circle2Tangent,
            ),
            (
                Icon::Circle3Tangent,
                "3-Tangent Circle — select three lines",
                SketchTool::Circle3Tangent,
            ),
        ],
    },
    RailGroup {
        face: Icon::ThreePointArc,
        name: "Arc",
        members: &[
            (
                Icon::ThreePointArc,
                "Arc — click start, end, then a point it passes through",
                SketchTool::ThreePointArc,
            ),
            (
                Icon::ArcCenterEndpoints,
                "Center Point Arc — click center, start, then end direction",
                SketchTool::ArcCenterEndpoints,
            ),
            (
                Icon::ArcTangent,
                "Tangent Arc — click a line/arc endpoint, then the other endpoint",
                SketchTool::ArcTangent,
            ),
        ],
    },
    RailGroup {
        face: Icon::PolygonInscribed,
        name: "Polygon",
        members: &[
            (
                Icon::PolygonInscribed,
                "Inscribed Polygon — click center, then a vertex",
                SketchTool::PolygonInscribed,
            ),
            (
                Icon::PolygonCircumscribed,
                "Circumscribed Polygon — click center, then an edge midpoint",
                SketchTool::PolygonCircumscribed,
            ),
            (
                Icon::PolygonEdge,
                "Edge Polygon — click edge endpoints, then choose the body side",
                SketchTool::PolygonEdge,
            ),
        ],
    },
    RailGroup {
        face: Icon::EllipseSketch,
        name: "Ellipse",
        members: &[(
            Icon::EllipseSketch,
            "Ellipse — click center, major-axis endpoint, then width",
            SketchTool::Ellipse,
        )],
    },
    RailGroup {
        face: Icon::SlotCenterToCenter,
        name: "Slot",
        members: &[
            (
                Icon::SlotCenterToCenter,
                "Center-to-Center Slot — click cap centers, then width",
                SketchTool::SlotCenterToCenter,
            ),
            (
                Icon::SlotOverall,
                "Overall Slot — click overall endpoints, then width",
                SketchTool::SlotOverall,
            ),
            (
                Icon::SlotCenterPoint,
                "Center Point Slot — click midpoint, cap center, then width",
                SketchTool::SlotCenterPoint,
            ),
            (
                Icon::Slot3PointArc,
                "3-Point Arc Slot — click arc endpoints, through point, then width",
                SketchTool::Slot3PointArc,
            ),
            (
                Icon::SlotCenterPointArc,
                "Center Point Arc Slot — click center, start, end direction, then width",
                SketchTool::SlotCenterPointArc,
            ),
        ],
    },
    RailGroup {
        face: Icon::SplineFitPoint,
        name: "Spline",
        members: &[
            (
                Icon::SplineFitPoint,
                "Fit Point Spline — place fit points; Enter finishes, click start to close",
                SketchTool::FitPointSpline,
            ),
            (
                Icon::SplineControlPoint,
                "Control Point Spline — place controls; Enter finishes",
                SketchTool::ControlPointSpline,
            ),
        ],
    },
    RailGroup {
        face: Icon::Conic,
        name: "Conic Curve",
        members: &[(
            Icon::Conic,
            "Conic — click start, end, then the on-curve vertex",
            SketchTool::Conic,
        )],
    },
    RailGroup {
        face: Icon::AddPoint,
        name: "Point",
        members: &[(
            Icon::AddPoint,
            "Add point — split an edge",
            SketchTool::AddPoint,
        )],
    },
];

/// The one cell that is not a verb: the pointer. It arms like a tool and reads like a mode, and
/// it belongs above Create rather than in it — selecting mints nothing.
const SKETCH_SELECT: (Icon, &str, SketchTool) = (
    Icon::SelectVertex,
    "Select / move vertex",
    SketchTool::Select,
);

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
    ConstraintVerb::Dimension,
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
    ConstraintVerb::Quantize,
    ConstraintVerb::Curvature,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SketchOperatorRoute {
    CloseLoopAction,
    Tool(SketchTool),
}

const SKETCH_OPERATORS: &[(Icon, &str, SketchOperatorRoute)] = &[
    (
        Icon::Mirror,
        "Mirror — select curves, then click the mirror line",
        SketchOperatorRoute::Tool(SketchTool::Mirror),
    ),
    (
        Icon::RectangularPattern,
        "Rectangular Pattern — select curves, then define the spacing directions",
        SketchOperatorRoute::Tool(SketchTool::RectangularPattern),
    ),
    (
        Icon::CircularPattern,
        "Circular Pattern — select curves, then click the center point",
        SketchOperatorRoute::Tool(SketchTool::CircularPattern),
    ),
    (
        Icon::CloseLoop,
        "Close Loop — join the active Line chain back to its start",
        SketchOperatorRoute::CloseLoopAction,
    ),
    (
        Icon::FillRegion,
        "Fill Region — click a bounded face to include it",
        SketchOperatorRoute::Tool(SketchTool::FillRegion),
    ),
    (
        Icon::CarveRegion,
        "Carve Region — click a bounded face to subtract it",
        SketchOperatorRoute::Tool(SketchTool::CarveRegion),
    ),
];

/// The sketch MODIFY families, in Fusion's order: the corner treatments, then the verbs that cut
/// and grow what is drawn, then the ones that move it.
///
/// Deviations from Fusion, each deliberate: the three chamfers are one family (Fusion's sketch
/// has no chamfer at all), Construction is a cell here rather than a palette toggle, and Blend
/// Curve is reserved — drawn so the finished shape of the set is visible without pretending the
/// verb works.
const SKETCH_MODIFY: &[RailGroup<SketchModifierRoute>] = &[
    RailGroup {
        face: Icon::Fillet,
        name: "Fillet",
        members: &[(
            Icon::Fillet,
            "Fillet — click along either leg to round its two-line corner",
            SketchModifierRoute::Tool(SketchTool::Fillet),
        )],
    },
    RailGroup {
        face: Icon::ChamferEqual,
        name: "Chamfer",
        members: &[
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
        ],
    },
    RailGroup {
        face: Icon::Trim,
        name: "Trim",
        members: &[(
            Icon::Trim,
            "Trim — remove the clicked interval to its neighboring intersections",
            SketchModifierRoute::Tool(SketchTool::Trim),
        )],
    },
    RailGroup {
        face: Icon::Extend,
        name: "Extend",
        members: &[(
            Icon::Extend,
            "Extend — grow the nearest endpoint to the next intersection",
            SketchModifierRoute::Tool(SketchTool::Extend),
        )],
    },
    RailGroup {
        face: Icon::BreakCurve,
        name: "Break",
        members: &[(
            Icon::BreakCurve,
            "Break — split a curve at its intersections",
            SketchModifierRoute::Tool(SketchTool::BreakCurve),
        )],
    },
    RailGroup {
        face: Icon::SketchScale,
        name: "Scale",
        members: &[(
            Icon::SketchScale,
            "Scale — choose a center, then a uniform size",
            SketchModifierRoute::Tool(SketchTool::Scale),
        )],
    },
    RailGroup {
        face: Icon::OffsetCurve,
        name: "Offset",
        members: &[(
            Icon::OffsetCurve,
            "Offset — select a curve, then place its parallel or concentric copy",
            SketchModifierRoute::Tool(SketchTool::Offset),
        )],
    },
    RailGroup {
        face: Icon::MoveCopy,
        name: "Move / Copy",
        members: &[(
            Icon::MoveCopy,
            "Move / Copy — choose a base and destination; hold Shift to copy",
            SketchModifierRoute::Tool(SketchTool::MoveCopy),
        )],
    },
    RailGroup {
        face: Icon::ConstructionToggle,
        name: "Construction",
        members: &[(
            Icon::ConstructionToggle,
            "Construction — toggle selected geometry between profile and reference",
            SketchModifierRoute::ConstructionAction,
        )],
    },
    RailGroup {
        face: Icon::BlendCurve,
        name: "Blend Curve",
        members: &[(
            Icon::BlendCurve,
            "Blend Curve — reserved",
            SketchModifierRoute::Reserved,
        )],
    },
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

            let lane = egui::Rect::from_min_max(
                column.min,
                egui::pos2(column.left() + BAR_LANE, column.bottom()),
            );
            let grid = egui::Rect::from_min_max(egui::pos2(lane.right(), column.top()), column.max);

            let scrolled = ui
                .scope_builder(egui::UiBuilder::new().max_rect(grid), |ui| {
                    rail_scroll_area().show(ui, |ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
                        if state.sketch_mode.is_some() {
                            build_sketch_rail(ui, state, response);
                        } else {
                            // Only the sketch rail has families, and the mode can end without a
                            // pointer press — undo deleting the sketch node, or a load that finds
                            // the id stale. Clearing here rather than at each of those exits means
                            // an open row cannot survive to slide out unprompted on the next entry.
                            state.open_rail_group = None;
                            rail_heading(ui, "Shape");
                            for &(icon, kind) in SHAPES {
                                shape_cell(ui, icon, kind, state, response);
                            }
                            rail_heading(ui, "Tool");
                            for &(icon, enabled) in TOOLS {
                                tool_cell(ui, icon, enabled);
                            }
                        }
                    })
                })
                .inner;

            rail_scroll_lane(ui, lane, &scrolled);
        });
}

/// The rail's own scroll indicator, painted down the [`BAR_LANE`] and draggable.
///
/// Hand-drawn because egui pins its bar to the right edge of the scrolling area, which here is the
/// chevron column — the tile that opens a family. The bar took those clicks, and the owner read a
/// bar sitting over the tools as the design being wrong rather than the click being lost, which it
/// was both times. So egui's bar is hidden and this one lives in a lane of its own, where it
/// overlaps nothing and can be dragged again.
///
/// Nothing is drawn while everything fits: a full-length handle is a bar that says nothing.
fn rail_scroll_lane<Route>(
    ui: &egui::Ui,
    lane: egui::Rect,
    scrolled: &egui::scroll_area::ScrollAreaOutput<Route>,
) {
    let reach = scrolled.content_size.y - scrolled.inner_rect.height();
    if reach <= 0.0 {
        return;
    }
    let travel = lane.height();
    let handle_height = (travel * scrolled.inner_rect.height() / scrolled.content_size.y)
        .max(BAR_HANDLE_MIN_HEIGHT)
        .min(travel);
    let at = scrolled.state.offset.y / reach;
    let top = lane.top() + at.clamp(0.0, 1.0) * (travel - handle_height);
    let handle = egui::Rect::from_min_size(
        egui::pos2(lane.center().x - BAR_HANDLE_WIDTH * 0.5, top),
        egui::vec2(BAR_HANDLE_WIDTH, handle_height),
    );

    let response = ui.interact(lane, ui.id().with("rail_scroll_lane"), egui::Sense::drag());
    if response.dragged() {
        if let Some(pointer) = response.interact_pointer_pos() {
            // Drag the HANDLE, not the lane: the pointer names where the middle of the handle
            // should sit, so grabbing it does not jump it under the cursor.
            let free = travel - handle_height;
            let wanted = (pointer.y - lane.top() - handle_height * 0.5) / free.max(f32::EPSILON);
            let mut state = scrolled.state;
            state.offset.y = wanted.clamp(0.0, 1.0) * reach;
            state.store(ui.ctx(), scrolled.id);
        }
    }

    let ink = if response.dragged() || response.hovered() {
        theme::TEXT_HINT
    } else {
        theme::RULE
    };
    ui.painter()
        .rect_filled(handle, BAR_HANDLE_WIDTH * 0.5, ink);
}

/// The rail's scrolling column, with egui's own bar hidden — [`rail_scroll_lane`] draws the one
/// the rail uses, in a lane where it overlaps no tile.
///
/// Shared with the tests rather than spelled out twice, so what they click through is the shipping
/// configuration.
fn rail_scroll_area() -> egui::ScrollArea {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
}

/// The scroll handle's width inside its lane, and the shortest it is allowed to get — a handle
/// that shrinks with the content stops being a target long before the content stops growing.
const BAR_HANDLE_WIDTH: f32 = 4.0;
const BAR_HANDLE_MIN_HEIGHT: f32 = 24.0;

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
    let (icon, tip, tool) = SKETCH_SELECT;
    if sketch_tool_cell(ui, icon, tip, state.sketch_tool == tool) {
        state.sketch_tool = tool;
    }

    rail_heading(ui, "Create");
    for group in SKETCH_CREATE {
        // A drawing tool is one route, so the group's activeness and its click are the same
        // question asked of whichever member the author reached: is this the armed tool, and make
        // it so.
        if let Some(tool) = rail_group(ui, state, group, |state, tool| state.sketch_tool == *tool) {
            state.sketch_tool = tool;
        }
    }
    // Mirror and the patterns are CREATE verbs in Fusion — they mint entities rather than rewrite
    // the ones named — and so they are here rather than in a section of their own. Close Loop and
    // the two region roles have no Fusion analog; they mint too, so they keep this company.
    //
    // Flat rather than grouped, deliberately: Mirror and Close Loop are alone, and the two
    // pattern verbs and the two region roles have no canonical member between them. A family
    // whose face is arbitrary buys one saved row for a click on every use of the other half.
    for &(icon, tip, route) in SKETCH_OPERATORS {
        let active = match route {
            SketchOperatorRoute::CloseLoopAction => false,
            SketchOperatorRoute::Tool(tool) => state.sketch_tool == tool,
        };
        if sketch_tool_cell(ui, icon, tip, active) {
            match route {
                SketchOperatorRoute::CloseLoopAction => response.close_sketch_loop = true,
                SketchOperatorRoute::Tool(tool) => state.sketch_tool = tool,
            }
        }
    }

    rail_heading(ui, "Modify");
    for group in SKETCH_MODIFY {
        let chosen = rail_group(ui, state, group, |state, route| match route {
            SketchModifierRoute::Tool(tool) => state.sketch_tool == *tool,
            SketchModifierRoute::Reserved | SketchModifierRoute::ConstructionAction => false,
        });
        match chosen {
            Some(SketchModifierRoute::Tool(tool)) => state.sketch_tool = tool,
            Some(SketchModifierRoute::ConstructionAction) => {
                response.toggle_sketch_construction = true;
            }
            Some(SketchModifierRoute::Reserved) | None => {}
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
    if state.sketch_tool == SketchTool::RectangularPattern {
        for count in &mut state.sketch_pattern_counts {
            *count = (*count).clamp(1, 128);
        }
        ui.horizontal(|ui| {
            ui.label("Count");
            ui.add(
                egui::DragValue::new(&mut state.sketch_pattern_counts[0])
                    .range(1..=128)
                    .prefix("X "),
            );
            ui.add(
                egui::DragValue::new(&mut state.sketch_pattern_counts[1])
                    .range(1..=128)
                    .prefix("Y "),
            );
        });
    }
    if state.sketch_tool == SketchTool::CircularPattern {
        state.sketch_circular_pattern_count = state.sketch_circular_pattern_count.clamp(2, 128);
        ui.horizontal(|ui| {
            ui.label("Count");
            ui.add(egui::DragValue::new(&mut state.sketch_circular_pattern_count).range(2..=128));
        });
    }
    rail_heading(ui, "Snap");
    for &(icon, tip, snap) in SKETCH_SNAPS {
        let active = state.sketch_snap == snap;
        if sketch_tool_cell(ui, icon, tip, active) {
            state.sketch_snap = snap;
        }
    }
    rail_heading(ui, "Constraints");
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
            SketchSolid::extrude(*sketch, height)
        }
        _ => SketchSolid::revolve(*sketch, RevolveAxis::InPlane0, 360),
    }
}

/// A rail section heading: UPPERCASE micro-label over a hairline.
fn rail_heading(ui: &mut egui::Ui, title: &str) {
    ui.add_space(9.0);
    let galley = theme::letter_spaced(ui, title, theme::TEXT_HINT, 8.0, 1.2);
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(ROW_WIDTH, galley.size().y + 5.0),
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
        egui::vec2(ROW_WIDTH, galley.size().y + 15.0),
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
        egui::vec2(CELL_WIDTH, TOOL_CELL_HEIGHT),
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
        egui::vec2(CELL_WIDTH, TOOL_CELL_HEIGHT),
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

/// The grid's chevron column, to the RIGHT of the glyph column rather than laid over it. A
/// family's chevron is a tile of the grid in its own right: on top of the glyph it was both hard
/// to aim at and hard to read (owner, 2026-08-03).
pub(super) const CHEVRON_WIDTH: f32 = 18.0;

/// How long a family takes to slide open or shut.
const SLIDE_SECONDS: f32 = 0.12;

/// One tool **family** on the rail: `[face][>]` collapsed, sliding open to
/// `[face][member][member][>]` along the row. Returns the route the author chose this frame,
/// whether by clicking the face (the canonical member) or a member out of the open row.
///
/// `is_active` answers, for one member, whether that verb is the one currently armed — the caller
/// owns that question because a route means different things per section. A family whose member is
/// armed lights that member, and lights the face while closed, so the rail still says what is in
/// hand even though the face never changes to say it.
///
/// A family of one draws as an ordinary cell with no chevron and nothing to slide; that falls out
/// of the member count rather than being a special case.
fn rail_group<Route: Copy>(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    group: &'static RailGroup<Route>,
    is_active: impl Fn(&PanelState, &Route) -> bool,
) -> Option<Route> {
    let lit: Vec<bool> = group
        .members
        .iter()
        .map(|(_, _, route)| is_active(state, route))
        .collect();
    let armed = lit.iter().any(|lit| *lit);
    let family = group.members.len() > 1;
    let open = family && state.open_rail_group == Some(group.face);

    // A full grid row — glyph column plus chevron column — and always the collapsed one: the part
    // that slides out hangs over the viewport, so opening a family never moves the cells under it.
    let (rect, cell) = ui.allocate_exact_size(
        egui::vec2(ROW_WIDTH, TOOL_CELL_HEIGHT),
        egui::Sense::click(),
    );
    let openness = ui.ctx().animate_bool_with_time(
        egui::Id::new(("rail_family", group.name)),
        open,
        SLIDE_SECONDS,
    );

    if openness > 0.0 {
        // Mid-slide the whole row belongs to the floating strip, face included, so the cell
        // underneath paints nothing and the two can never disagree about what is drawn where.
        let picked = rail_slide_out(ui, group, &lit, rect, openness);
        // Only an OPEN family dismisses. A row still sliding SHUT has a rect too, and would read
        // the press picking from whichever family replaced it as a dismissal of itself. No test
        // separates the two — both rows animate over the same span, so the new row is only worth
        // pressing once the old one has finished — but the guard makes it true by construction
        // rather than by two durations happening to match.
        let dismissed = open
            && ui.input(|input| {
                input.pointer.any_pressed()
                    && input
                        .pointer
                        .interact_pos()
                        .is_some_and(|at| !slid_row(rect, group, openness).contains(at))
            });
        return match picked {
            Some(SlidePick::Member(route)) => {
                state.open_rail_group = None;
                Some(route)
            }
            Some(SlidePick::Chevron) => {
                state.open_rail_group = None;
                None
            }
            None => {
                if dismissed {
                    state.open_rail_group = None;
                }
                None
            }
        };
    }

    let (face, chevron) = split_off_chevron(rect, family);
    let on_chevron = hovering(&cell, chevron);
    paint_cell(ui, face, armed, cell.hovered() && !on_chevron);
    group.face.draw(
        ui.painter(),
        egui::Rect::from_center_size(face.center(), egui::Vec2::splat(TOOL_GLYPH)),
        cell_ink(armed, cell.hovered() && !on_chevron, false),
    );
    if family {
        paint_cell(ui, chevron, false, on_chevron);
        paint_chevron(ui, chevron, cell_ink(false, on_chevron, false), 0.0);
    }

    let (_, canonical_tip, canonical) = group.members[0];
    let tip = if family {
        format!(
            "{} — {canonical_tip}\nThe chevron slides out the other {}",
            group.name,
            group.members.len() - 1
        )
    } else {
        canonical_tip.to_string()
    };
    let cell = cell.on_hover_text(tip);

    if !cell.clicked() {
        return None;
    }
    let hit_chevron = cell
        .interact_pointer_pos()
        .is_some_and(|at| chevron.contains(at));
    if family && hit_chevron {
        state.open_rail_group = Some(group.face);
        return None;
    }
    Some(canonical)
}

/// Whether the pointer hovering `cell` is inside `zone`.
fn hovering(cell: &egui::Response, zone: egui::Rect) -> bool {
    cell.hover_pos().is_some_and(|at| zone.contains(at))
}

/// A grid row's two columns: the glyph tile, which arms, and the chevron tile, which slides.
///
/// The glyph tile is [`CELL_WIDTH`] whether or not the row has a chevron, so a family of one lines
/// up with its neighbours instead of stretching across both columns. A family of one has nothing
/// to slide out, so its chevron tile is empty.
fn split_off_chevron(row: egui::Rect, family: bool) -> (egui::Rect, egui::Rect) {
    let split = row.left() + CELL_WIDTH;
    let face = egui::Rect::from_min_max(row.min, egui::pos2(split, row.bottom()));
    if !family {
        return (face, egui::Rect::NOTHING);
    }
    (
        face,
        egui::Rect::from_min_max(egui::pos2(split, row.top()), row.max),
    )
}

/// What a click on the slid-open row resolved to.
enum SlidePick<Route> {
    /// A member's box — arm that verb and close the row.
    Member(Route),
    /// The chevron at the end of the row — close it, arming nothing.
    Chevron,
}

/// The full extent of the row at this point in the slide, collapsed footprint included.
///
/// Open, the row is one glyph tile per member plus the chevron tile — the chevron is COUNTED, not
/// laid over the last member (owner, 2026-08-03). Collapsed, it is the grid row it grew from.
fn slid_row<Route>(cell: egui::Rect, group: &RailGroup<Route>, openness: f32) -> egui::Rect {
    #[expect(
        clippy::cast_precision_loss,
        reason = "a family holds single digits of members"
    )]
    let extra = (group.members.len() - 1) as f32 * CELL_WIDTH;
    egui::Rect::from_min_size(
        cell.min,
        egui::vec2(cell.width() + extra * openness, cell.height()),
    )
}

/// The open family's members, sliding out to the RIGHT of the rail: `[face][member][member][>]`.
///
/// A floating [`egui::Area`] rather than an in-place expansion, and deliberately: the rail is a
/// vertical scroll column, so growing the cell downward would push every section below it and move
/// the target the author is reaching for. Sideways there is nothing to push. It also must not
/// allocate in the rail's `Ui` — floating chrome that does carves a dead band out of the viewport
/// behind it.
///
/// The row is clipped to its animated width, so members emerge from under the face rather than
/// appearing somewhere they were not.
fn rail_slide_out<Route: Copy>(
    ui: &egui::Ui,
    group: &'static RailGroup<Route>,
    lit: &[bool],
    cell: egui::Rect,
    openness: f32,
) -> Option<SlidePick<Route>> {
    let row = slid_row(cell, group, openness);
    let mut picked = None;
    egui::Area::new(egui::Id::new(("rail_family_row", group.name)))
        .order(egui::Order::Foreground)
        .fixed_pos(cell.left_top())
        .show(ui.ctx(), |ui| {
            ui.set_clip_rect(row);
            ui.painter().rect_filled(row, 0.0, theme::BG);

            for (index, &(icon, tip, route)) in group.members.iter().enumerate() {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "a family holds single digits of members"
                )]
                let offset = index as f32 * CELL_WIDTH;
                let box_rect = egui::Rect::from_min_size(
                    cell.left_top() + egui::vec2(offset, 0.0),
                    egui::vec2(CELL_WIDTH, cell.height()),
                );
                let id = egui::Id::new(("rail_family_member", group.name, index));
                if member_box(ui, box_rect, id, icon, tip, lit.get(index) == Some(&true)) {
                    picked = Some(SlidePick::Member(route));
                }
            }

            // The chevron rides the row's right edge, so it is always at the end of whatever is
            // currently showing — the same pixel that opened the family closes it.
            let chevron = egui::Rect::from_min_max(
                egui::pos2(row.right() - CHEVRON_WIDTH, row.top()),
                row.max,
            );
            let response = ui.interact(
                chevron,
                egui::Id::new(("rail_family_close", group.name)),
                egui::Sense::click(),
            );
            paint_cell(ui, chevron, false, response.hovered());
            paint_chevron(
                ui,
                chevron,
                cell_ink(false, response.hovered(), false),
                openness,
            );
            if response.clicked() {
                picked = Some(SlidePick::Chevron);
            }
        });
    picked
}

/// One member box in the slid-open row: the member's own glyph at cell size, lit when it is the
/// armed one. Glyph only, as on the rail itself — the row is read by shape at a glance, and the
/// name is one hover away.
///
/// The caller supplies the id. Keyed by the family and the position within it rather than by the
/// tooltip, because two rows genuinely coexist — opening one family while another is still sliding
/// shut leaves both Areas live — and two members sharing an id would misroute the click with
/// nothing but a debug-build warning to say so.
fn member_box(
    ui: &egui::Ui,
    rect: egui::Rect,
    id: egui::Id,
    icon: Icon,
    tip: &str,
    active: bool,
) -> bool {
    let response = ui.interact(rect, id, egui::Sense::click());
    let hovered = response.hovered();
    paint_cell(ui, rect, active, hovered);
    icon.draw(
        ui.painter(),
        egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(TOOL_GLYPH)),
        cell_ink(active, hovered, false),
    );
    response.on_hover_text(tip.to_string()).clicked()
}

/// The mark that says "this cell is a family": a chevron pointing the way the row will travel,
/// turning to point back as it opens.
fn paint_chevron(ui: &egui::Ui, zone: egui::Rect, ink: egui::Color32, openness: f32) {
    if !zone.is_positive() {
        return;
    }
    let center = zone.center();
    // +1 at rest points right ("more this way"), −1 fully open points back ("put it away").
    let facing = 1.0 - 2.0 * openness;
    let (reach, rise) = (3.0 * facing, 4.0);
    ui.painter().add(egui::Shape::line(
        vec![
            center - egui::vec2(reach, rise),
            center + egui::vec2(reach, 0.0),
            center - egui::vec2(reach, -rise),
        ],
        egui::Stroke::new(1.5_f32, ink),
    ));
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
    let (rect, cell) = ui.allocate_exact_size(egui::vec2(CELL_WIDTH, CELL_HEIGHT), sense);
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
    let (rect, cell) = ui.allocate_exact_size(egui::vec2(CELL_WIDTH, TOOL_CELL_HEIGHT), sense);
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

    /// Every family's face is its canonical member's own glyph. The face is fixed, so if it were
    /// some other mark the rail would show a verb the click does not perform.
    #[test]
    fn every_family_faces_its_canonical_member() {
        for group in SKETCH_CREATE {
            assert_eq!(group.face, group.members[0].0, "{}", group.name);
        }
        for group in SKETCH_MODIFY {
            assert_eq!(group.face, group.members[0].0, "{}", group.name);
        }
    }

    /// A two-member family to drive the cell through a real egui frame loop. Named with a face no
    /// shipping family uses, so it cannot collide with the rail's own family ids.
    const PROBE: RailGroup<u8> = RailGroup {
        face: Icon::Probe,
        name: "Probe",
        members: &[
            (Icon::Probe, "First — the canonical member", 1),
            (Icon::Measure, "Second — behind the chevron", 2),
        ],
    };

    /// A second family, to put two of them in one column the way the shipping tables do. Its own
    /// face and its own tips, so nothing about it can be answered by [`PROBE`]'s ids.
    const PROBE_BELOW: RailGroup<u8> = RailGroup {
        face: Icon::Material,
        name: "Material",
        members: &[
            (Icon::Material, "Third — the second family's face", 3),
            (Icon::Carve, "Fourth — behind the second chevron", 4),
        ],
    };

    /// Run one frame of [`PROBE`]'s cell at the ui origin, feeding `events`, and report both what
    /// the cell returned and where it drew.
    fn probe_frame(
        context: &egui::Context,
        state: &mut PanelState,
        events: Vec<egui::Event>,
    ) -> (Option<u8>, egui::Rect) {
        let mut chosen = None;
        let mut face = egui::Rect::NOTHING;
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 800.0),
            )),
            events,
            ..Default::default()
        };
        let _ = context.run_ui(raw_input, |ui| {
            let at = ui.available_rect_before_wrap().min;
            face = egui::Rect::from_min_size(at, egui::vec2(ROW_WIDTH, TOOL_CELL_HEIGHT));
            chosen = rail_group(ui, state, &PROBE, |_, _| false);
        });
        (chosen, face)
    }

    /// Press and release at `at`, over three frames: egui resolves interaction against the
    /// PREVIOUS frame's widget rects, so the pointer has to arrive before the press does.
    fn click_at(
        context: &egui::Context,
        state: &mut PanelState,
        at: egui::Pos2,
    ) -> (Option<u8>, egui::Rect) {
        let button = |pressed| egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        probe_frame(context, state, vec![egui::Event::PointerMoved(at)]);
        probe_frame(context, state, vec![button(true)]);
        probe_frame(context, state, vec![button(false)])
    }

    /// Run frames until the slide has finished, so a click lands on a member box that has stopped
    /// moving.
    fn settle(context: &egui::Context, state: &mut PanelState) {
        for _ in 0..30 {
            probe_frame(context, state, Vec::new());
        }
    }

    /// The chevron's zone: the right strip of the cell, at full height. The point of the shape —
    /// the corner tick it replaced was under the scroll bar and too small to hit.
    fn chevron_of(cell: egui::Rect) -> egui::Pos2 {
        egui::pos2(cell.right() - CHEVRON_WIDTH * 0.5, cell.center().y)
    }

    /// The whole point of the reshape: a verb that is NOT its family's face is still reachable, in
    /// two clicks — the chevron, then the box that slid out. Clicking the face could never return
    /// the second member, so this cannot pass by accident.
    #[test]
    fn the_chevron_slides_the_family_out_and_its_boxes_arm_the_hidden_member() {
        let context = egui::Context::default();
        let mut state = PanelState::default();
        let (_, cell) = probe_frame(&context, &mut state, Vec::new());

        let (chosen, _) = click_at(&context, &mut state, chevron_of(cell));
        assert_eq!(
            chosen, None,
            "the chevron slides the family out, it does not arm it"
        );
        assert_eq!(state.open_rail_group, Some(Icon::Probe));

        settle(&context, &mut state);
        // The second member's box sits one full cell to the RIGHT of the face — the direction the
        // row travels, and a place the collapsed cell never occupied.
        let second = egui::pos2(cell.left() + CELL_WIDTH * 1.5, cell.center().y);
        let (chosen, _) = click_at(&context, &mut state, second);
        assert_eq!(chosen, Some(2), "the box that slid out arms its member");
        assert_eq!(
            state.open_rail_group, None,
            "picking slides the family shut behind it"
        );
    }

    /// The chevron rides the row's right edge, so the pixel that opened the family is the pixel
    /// that closes it — and closing arms nothing.
    #[test]
    fn the_chevron_at_the_end_of_the_open_row_slides_it_shut() {
        let context = egui::Context::default();
        let mut state = PanelState::default();
        let (_, cell) = probe_frame(&context, &mut state, Vec::new());
        click_at(&context, &mut state, chevron_of(cell));
        settle(&context, &mut state);

        let row = slid_row(cell, &PROBE, 1.0);
        assert!(
            row.width() > cell.width(),
            "the open row has to be wider than the cell it grew from: {row:?}"
        );
        let (chosen, _) = click_at(&context, &mut state, chevron_of(row));
        assert_eq!(chosen, None, "closing the row arms nothing");
        assert_eq!(state.open_rail_group, None);
    }

    /// One frame of [`PROBE`] above [`PROBE_BELOW`], reporting what was chosen and where each cell
    /// drew.
    fn two_family_frame(
        context: &egui::Context,
        state: &mut PanelState,
        events: Vec<egui::Event>,
    ) -> (Option<u8>, [egui::Rect; 2]) {
        let mut chosen = None;
        let mut cells = [egui::Rect::NOTHING; 2];
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 800.0),
            )),
            events,
            ..Default::default()
        };
        let _ = context.run_ui(raw_input, |ui| {
            ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
            for (slot, group) in cells.iter_mut().zip([&PROBE, &PROBE_BELOW]) {
                *slot = egui::Rect::from_min_size(
                    ui.available_rect_before_wrap().min,
                    egui::vec2(ROW_WIDTH, TOOL_CELL_HEIGHT),
                );
                chosen = rail_group(ui, state, group, |_, _| false).or(chosen);
            }
        });
        (chosen, cells)
    }

    /// Press and release at `at` over the two-family column, three frames as [`click_at`] does.
    /// Deliberately no settling frames: what these tests are after is the window while a row is
    /// still animating.
    fn two_family_press(
        context: &egui::Context,
        state: &mut PanelState,
        at: egui::Pos2,
    ) -> Option<u8> {
        let button = |pressed| egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        two_family_frame(context, state, vec![egui::Event::PointerMoved(at)]);
        two_family_frame(context, state, vec![button(true)]);
        two_family_frame(context, state, vec![button(false)]).0
    }

    /// One family at a time. `open_rail_group` holds a single face, so this is true by
    /// construction — the test is that the second chevron is actually reached and does the writing,
    /// rather than the first family swallowing the press as its own dismissal and leaving nothing
    /// open at all.
    #[test]
    fn opening_a_family_closes_the_one_that_was_open() {
        let context = egui::Context::default();
        let mut state = PanelState::default();

        let (_, cells) = two_family_frame(&context, &mut state, Vec::new());
        two_family_press(&context, &mut state, chevron_of(cells[0]));
        assert_eq!(state.open_rail_group, Some(PROBE.face));
        for _ in 0..30 {
            two_family_frame(&context, &mut state, Vec::new());
        }

        two_family_press(&context, &mut state, chevron_of(cells[1]));
        assert_eq!(
            state.open_rail_group,
            Some(PROBE_BELOW.face),
            "the press that opens the second family also dismisses the first, and the two must              not cancel out"
        );
    }

    /// **The bug the owner reported.** The chevron rides the cell's right edge, which is exactly
    /// where a scrolling column draws its bar — and a bar that senses clicks takes them, leaving
    /// the family unreachable. Run through [`rail_scroll_area`] itself, in a column short enough to
    /// overflow so the bar is really there.
    #[test]
    fn the_scroll_bar_does_not_eat_the_chevron() {
        let context = egui::Context::default();
        let mut state = PanelState::default();

        // A column the probe cell overflows several times over, so the bar has a handle to draw.
        let column = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(ROW_WIDTH, 60.0));
        let mut cell = egui::Rect::NOTHING;
        let mut frame = |events: Vec<egui::Event>, cell: &mut egui::Rect| {
            let raw_input = egui::RawInput {
                screen_rect: Some(column),
                events,
                ..Default::default()
            };
            let _ = context.run_ui(raw_input, |ui| {
                rail_scroll_area().show(ui, |ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
                    *cell = egui::Rect::from_min_size(
                        ui.available_rect_before_wrap().min,
                        egui::vec2(ROW_WIDTH, TOOL_CELL_HEIGHT),
                    );
                    rail_group(ui, &mut state, &PROBE, |_, _| false);
                    ui.allocate_space(egui::vec2(ROW_WIDTH, 400.0));
                });
            });
        };

        frame(Vec::new(), &mut cell);
        let at = chevron_of(cell);
        let button = |pressed| egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        frame(vec![egui::Event::PointerMoved(at)], &mut cell);
        frame(vec![button(true)], &mut cell);
        frame(vec![button(false)], &mut cell);

        assert_eq!(
            state.open_rail_group,
            Some(Icon::Probe),
            "the scroll bar swallowed the chevron's click at {at:?}"
        );
    }

    /// The open row counts its chevron rather than laying it over the last member. Geometry
    /// directly, because the failure is a drawing one the owner saw before any click misrouted:
    /// clicking the CENTER of the last member still worked while its right edge was under the
    /// chevron, so no gesture test would have caught it.
    #[test]
    fn the_open_row_reserves_a_column_for_its_chevron() {
        let cell = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(ROW_WIDTH, 44.0));
        let row = slid_row(cell, &PROBE, 1.0);

        #[expect(
            clippy::cast_precision_loss,
            reason = "a family holds single digits of members"
        )]
        let members = PROBE.members.len() as f32;
        let last_member_right = cell.left() + members * CELL_WIDTH;
        assert!(
            (row.right() - CHEVRON_WIDTH - last_member_right).abs() < f32::EPSILON,
            "the chevron column has to start where the members stop: row {row:?}, members end at \
             {last_member_right}"
        );
    }

    /// Every row's glyph tile is the same width, family or not, so the rail reads as a grid rather
    /// than as rows that stretch to fill whatever the chevron does not take.
    #[test]
    fn the_glyph_column_is_one_width() {
        let row = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(ROW_WIDTH, 44.0));
        for family in [true, false] {
            let (face, chevron) = split_off_chevron(row, family);
            assert!(
                (face.width() - CELL_WIDTH).abs() < f32::EPSILON,
                "family={family} face {face:?}"
            );
            assert_eq!(chevron.is_positive(), family);
        }
    }

    /// The face cell arms the canonical member directly — a family costs no extra click for the
    /// verb the author reaches for most.
    #[test]
    fn the_face_arms_the_canonical_member_without_opening_anything() {
        let context = egui::Context::default();
        let mut state = PanelState::default();
        let (_, face) = probe_frame(&context, &mut state, Vec::new());
        let (chosen, _) = click_at(&context, &mut state, face.center());
        assert_eq!(chosen, Some(1));
        assert_eq!(state.open_rail_group, None);
    }

    /// The open family is keyed by face glyph, so two families sharing a face would slide out
    /// each other's row.
    #[test]
    fn no_two_families_share_a_face() {
        let mut faces: Vec<Icon> = SKETCH_CREATE.iter().map(|group| group.face).collect();
        faces.extend(SKETCH_MODIFY.iter().map(|group| group.face));
        let mut seen = Vec::new();
        for face in faces {
            assert!(!seen.contains(&face), "{face:?} faces two families");
            seen.push(face);
        }
    }

    /// Two members must not share a tooltip. The interact ids no longer key off the tip — that
    /// coupling is exactly the silent failure this pins the other end of — but a duplicate tip is
    /// still a bug on its own terms: the hover is all that names a member in a glyph-only row, so
    /// two boxes reading the same is two verbs the author cannot tell apart.
    #[test]
    fn no_two_members_share_a_tooltip() {
        let mut tips: Vec<&str> = SKETCH_CREATE
            .iter()
            .flat_map(|group| group.members.iter().map(|(_, tip, _)| *tip))
            .collect();
        tips.extend(
            SKETCH_MODIFY
                .iter()
                .flat_map(|group| group.members.iter().map(|(_, tip, _)| *tip)),
        );
        let mut seen: Vec<&str> = Vec::new();
        for tip in tips {
            assert!(!seen.contains(&tip), "two members read {tip:?}");
            seen.push(tip);
        }
    }

    /// Regrouping a flat list into families is exactly the edit that drops a verb without anyone
    /// noticing, so the rail's tools and [`SketchTool::ALL`] must be the same set.
    #[test]
    fn the_rail_reaches_every_sketch_tool() {
        let mut on_rail = vec![SKETCH_SELECT.2];
        on_rail.extend(
            SKETCH_CREATE
                .iter()
                .flat_map(|group| group.members.iter().map(|&(_, _, tool)| tool)),
        );
        on_rail.extend(SKETCH_MODIFY.iter().flat_map(|group| {
            group
                .members
                .iter()
                .filter_map(|&(_, _, route)| match route {
                    SketchModifierRoute::Tool(tool) => Some(tool),
                    SketchModifierRoute::Reserved | SketchModifierRoute::ConstructionAction => None,
                })
        }));
        on_rail.extend(
            SKETCH_OPERATORS
                .iter()
                .filter_map(|&(_, _, route)| match route {
                    SketchOperatorRoute::Tool(tool) => Some(tool),
                    SketchOperatorRoute::CloseLoopAction => None,
                }),
        );
        for tool in SketchTool::ALL {
            assert!(on_rail.contains(tool), "{tool:?} has no rail cell");
        }
        for tool in &on_rail {
            assert!(
                SketchTool::ALL.contains(tool),
                "{tool:?} is on the rail but missing from SketchTool::ALL"
            );
        }
        assert_eq!(on_rail.len(), SketchTool::ALL.len(), "a tool has two cells");
    }

    /// The Create families in Fusion's order, each with the grammars Fusion puts inside it.
    #[test]
    fn create_families_carry_fusion_membership_in_fusion_order() {
        let families: Vec<(&str, Vec<SketchTool>)> = SKETCH_CREATE
            .iter()
            .map(|group| {
                (
                    group.name,
                    group.members.iter().map(|&(_, _, tool)| tool).collect(),
                )
            })
            .collect();
        assert_eq!(
            families,
            vec![
                ("Line", vec![SketchTool::Line, SketchTool::MidpointLine]),
                (
                    "Rectangle",
                    vec![
                        SketchTool::Rectangle,
                        SketchTool::Rectangle3Point,
                        SketchTool::RectangleCenterCorner
                    ]
                ),
                (
                    "Circle",
                    vec![
                        SketchTool::CircleCenterDiameter,
                        SketchTool::Circle2Point,
                        SketchTool::Circle3Point,
                        SketchTool::Circle2Tangent,
                        SketchTool::Circle3Tangent
                    ]
                ),
                (
                    "Arc",
                    vec![
                        SketchTool::ThreePointArc,
                        SketchTool::ArcCenterEndpoints,
                        SketchTool::ArcTangent
                    ]
                ),
                (
                    "Polygon",
                    vec![
                        SketchTool::PolygonInscribed,
                        SketchTool::PolygonCircumscribed,
                        SketchTool::PolygonEdge
                    ]
                ),
                ("Ellipse", vec![SketchTool::Ellipse]),
                (
                    "Slot",
                    vec![
                        SketchTool::SlotCenterToCenter,
                        SketchTool::SlotOverall,
                        SketchTool::SlotCenterPoint,
                        SketchTool::Slot3PointArc,
                        SketchTool::SlotCenterPointArc
                    ]
                ),
                (
                    "Spline",
                    vec![SketchTool::FitPointSpline, SketchTool::ControlPointSpline]
                ),
                ("Conic Curve", vec![SketchTool::Conic]),
                ("Point", vec![SketchTool::AddPoint]),
            ]
        );
    }

    /// The Modify families, in Fusion's order: the corner treatments, then the cutters, then the
    /// movers, then this app's two additions.
    #[test]
    fn modify_families_run_in_fusion_order_and_keep_blend_curve_reserved() {
        let names: Vec<&str> = SKETCH_MODIFY.iter().map(|group| group.name).collect();
        assert_eq!(
            names,
            vec![
                "Fillet",
                "Chamfer",
                "Trim",
                "Extend",
                "Break",
                "Scale",
                "Offset",
                "Move / Copy",
                "Construction",
                "Blend Curve",
            ]
        );
        let chamfer = SKETCH_MODIFY.iter().find(|group| group.name == "Chamfer");
        assert_eq!(chamfer.map(|group| group.members.len()), Some(3));
        let reserved: Vec<&str> = SKETCH_MODIFY
            .iter()
            .filter(|group| {
                group
                    .members
                    .iter()
                    .all(|&(_, _, route)| route == SketchModifierRoute::Reserved)
            })
            .map(|group| group.name)
            .collect();
        assert_eq!(reserved, vec!["Blend Curve"]);
    }

    /// Select is not a Create verb and does not ride in a family — it is the one cell above them.
    #[test]
    fn select_stands_outside_the_families() {
        assert_eq!(SKETCH_SELECT.2, SketchTool::Select);
        assert!(SKETCH_CREATE.iter().all(|group| group
            .members
            .iter()
            .all(|&(_, _, tool)| tool != SketchTool::Select)));
    }

    #[test]
    fn sketch_rail_exposes_associative_patterns_face_roles_and_close_loop() {
        assert_eq!(
            SKETCH_OPERATORS,
            &[
                (
                    Icon::Mirror,
                    "Mirror — select curves, then click the mirror line",
                    SketchOperatorRoute::Tool(SketchTool::Mirror)
                ),
                (
                    Icon::RectangularPattern,
                    "Rectangular Pattern — select curves, then define the spacing directions",
                    SketchOperatorRoute::Tool(SketchTool::RectangularPattern),
                ),
                (
                    Icon::CircularPattern,
                    "Circular Pattern — select curves, then click the center point",
                    SketchOperatorRoute::Tool(SketchTool::CircularPattern),
                ),
                (
                    Icon::CloseLoop,
                    "Close Loop — join the active Line chain back to its start",
                    SketchOperatorRoute::CloseLoopAction,
                ),
                (
                    Icon::FillRegion,
                    "Fill Region — click a bounded face to include it",
                    SketchOperatorRoute::Tool(SketchTool::FillRegion),
                ),
                (
                    Icon::CarveRegion,
                    "Carve Region — click a bounded face to subtract it",
                    SketchOperatorRoute::Tool(SketchTool::CarveRegion),
                ),
            ]
        );
    }
}
