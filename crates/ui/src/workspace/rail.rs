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

/// One family of rail verbs, drawn as a single cell whose flyout holds the rest.
///
/// # The face is fixed
///
/// A group's face is its canonical member, always — never the last one used. Fusion's last-used
/// face exists to serve re-invoking the same variant in a UI where a tool DISARMS after each use;
/// here a tool stays armed until it is disarmed, so the arming model already delivers what
/// last-used is for. What would remain is only its cost: the same pixel meaning a different verb
/// depending on history, which is what muscle memory cannot absorb.
///
/// A family of one has no flyout and draws as an ordinary cell. That is not a special case in the
/// rendering — it falls out of having nothing else to show.
struct RailGroup<Route: 'static> {
    /// The mark this family always shows on the rail.
    face: Icon,
    /// The family's name: the face's tooltip, and the flyout's heading.
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

/// The corner tick's box: the bottom-right square of a family's face cell, the zone that opens
/// the flyout instead of arming the face's verb.
const FAMILY_TICK: f32 = 13.0;

/// One tool **family** on the rail: the fixed face cell, its corner tick, and the flyout the tick
/// opens. Returns the route the author chose this frame, whether by clicking the face (the
/// canonical member) or by picking out of the flyout.
///
/// `is_active` answers, for one member, whether that verb is the one currently armed — the caller
/// owns that question because a route means different things per section. A family whose member is
/// armed lights its face, so the rail still says what is in hand even though the face never
/// changes to say it.
///
/// A family of one draws as an ordinary cell with no tick and no flyout; that falls out of the
/// member count rather than being a special case.
fn rail_group<Route: Copy>(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    group: &'static RailGroup<Route>,
    is_active: impl Fn(&PanelState, &Route) -> bool,
) -> Option<Route> {
    let armed = group
        .members
        .iter()
        .any(|(_, _, route)| is_active(state, route));
    let family = group.members.len() > 1;
    let open = family && state.open_rail_group == Some(group.face);

    let (rect, cell) = ui.allocate_exact_size(
        egui::vec2(RAIL_WIDTH, TOOL_CELL_HEIGHT),
        egui::Sense::click(),
    );
    let hovered = cell.hovered();
    paint_cell(ui, rect, armed, hovered);
    let ink = cell_ink(armed, hovered, false);
    let glyph = egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(TOOL_GLYPH));
    group.face.draw(ui.painter(), glyph, ink);

    let tick = egui::Rect::from_min_max(
        rect.right_bottom() - egui::Vec2::splat(FAMILY_TICK),
        rect.right_bottom(),
    );
    if family {
        paint_family_tick(ui, tick, ink);
    }

    let (_, canonical_tip, canonical) = group.members[0];
    let tip = if family {
        format!(
            "{} — {canonical_tip}\nCorner tick for the other {}",
            group.name,
            group.members.len() - 1
        )
    } else {
        canonical_tip.to_string()
    };
    let cell = cell.on_hover_text(tip);

    let hit_tick = cell
        .interact_pointer_pos()
        .is_some_and(|at| tick.contains(at));
    let mut chosen = None;
    if cell.clicked() {
        if family && hit_tick {
            state.open_rail_group = if open { None } else { Some(group.face) };
        } else {
            state.open_rail_group = None;
            chosen = Some(canonical);
        }
    }

    if open {
        // Precomputed because the flyout's closure cannot hold `state` while the pick below
        // writes to it.
        let lit: Vec<bool> = group
            .members
            .iter()
            .map(|(_, _, route)| is_active(state, route))
            .collect();
        let (picked, panel) = rail_flyout(ui, group, &lit, rect);
        // Any press that missed both the face and the panel dismisses it, the way every other
        // menu in the app closes without a Cancel.
        let dismissed = ui.input(|input| {
            input.pointer.any_pressed()
                && input
                    .pointer
                    .interact_pos()
                    .is_some_and(|at| !panel.contains(at) && !rect.contains(at))
        });
        if picked.is_some() || dismissed {
            state.open_rail_group = None;
        }
        chosen = picked.or(chosen);
    }
    chosen
}

/// The open family's member list: a floating panel beside the face cell, one row per verb.
///
/// An [`egui::Area`] rather than an in-place expansion, and deliberately: the rail is a scroll
/// column, so growing a cell would push every section below it down and move the target the author
/// is reaching for. It also must not allocate in the rail's `Ui` — floating chrome that does
/// carves a dead band out of whatever is behind it.
fn rail_flyout<Route: Copy>(
    ui: &egui::Ui,
    group: &'static RailGroup<Route>,
    lit: &[bool],
    face: egui::Rect,
) -> (Option<Route>, egui::Rect) {
    let mut picked = None;
    let area = egui::Area::new(egui::Id::new(("rail_flyout", group.name)))
        .order(egui::Order::Foreground)
        .fixed_pos(face.right_top())
        .show(ui.ctx(), |ui| {
            ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
            egui::Frame::popup(ui.style())
                .fill(theme::DIALOG_BG)
                .stroke(egui::Stroke::new(1.0_f32, theme::DIALOG_BORDER))
                .show(ui, |ui| {
                    let heading = theme::letter_spaced(ui, group.name, theme::TEXT_HINT, 8.0, 1.2);
                    ui.label(heading);
                    for (index, &(icon, tip, route)) in group.members.iter().enumerate() {
                        if rail_flyout_row(ui, icon, tip, lit.get(index) == Some(&true)) {
                            picked = Some(route);
                        }
                    }
                });
        });
    (picked, area.response.rect)
}

/// One flyout row: the member's own glyph beside the verb's name, lit when that member is the
/// armed one. The name is the tooltip's first clause — the flyout is where the author goes to read
/// what a family contains, so the words have to be on screen rather than behind a hover.
fn rail_flyout_row(ui: &mut egui::Ui, icon: Icon, tip: &str, active: bool) -> bool {
    let name = tip.split(" — ").next().unwrap_or(tip);
    let galley = ui.painter().layout_no_wrap(
        name.to_string(),
        egui::TextStyle::Body.resolve(ui.style()),
        cell_ink(active, false, false),
    );
    let size = egui::vec2(
        TOOL_GLYPH + 10.0 + galley.size().x + 8.0,
        TOOL_CELL_HEIGHT * 0.6,
    );
    let (rect, row) = ui.allocate_exact_size(size, egui::Sense::click());
    let hovered = row.hovered();
    paint_cell(ui, rect, active, hovered);
    let ink = cell_ink(active, hovered, false);
    let glyph = egui::Rect::from_center_size(
        egui::pos2(rect.left() + 4.0 + TOOL_GLYPH * 0.5, rect.center().y),
        egui::Vec2::splat(TOOL_GLYPH * 0.75),
    );
    icon.draw(ui.painter(), glyph, ink);
    ui.painter().galley(
        egui::pos2(glyph.right() + 8.0, rect.center().y - galley.size().y * 0.5),
        galley,
        ink,
    );
    row.on_hover_text(tip.to_string()).clicked()
}

/// The mark that says "this cell is a family": a small filled corner wedge, in the cell's own ink
/// so it reads as part of the glyph rather than a second signal.
fn paint_family_tick(ui: &egui::Ui, tick: egui::Rect, ink: egui::Color32) {
    let wedge = vec![
        tick.right_bottom(),
        tick.right_bottom() - egui::vec2(FAMILY_TICK * 0.55, 0.0),
        tick.right_bottom() - egui::vec2(0.0, FAMILY_TICK * 0.55),
    ];
    ui.painter()
        .add(egui::Shape::convex_polygon(wedge, ink, egui::Stroke::NONE));
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
    /// shipping family uses, so it cannot collide with the rail's own flyout ids.
    const PROBE: RailGroup<u8> = RailGroup {
        face: Icon::Probe,
        name: "Probe",
        members: &[
            (Icon::Probe, "First — the canonical member", 1),
            (Icon::Measure, "Second — behind the flyout", 2),
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
            face = egui::Rect::from_min_size(at, egui::vec2(RAIL_WIDTH, TOOL_CELL_HEIGHT));
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

    /// The whole point of the reshape: a verb that is NOT its family's face is still reachable, in
    /// two clicks — the corner tick, then the row. Clicking the face could never return the second
    /// member, so this cannot pass by accident.
    #[test]
    fn the_corner_tick_opens_the_flyout_and_its_rows_arm_the_hidden_member() {
        let context = egui::Context::default();
        let mut state = PanelState::default();
        let (_, face) = probe_frame(&context, &mut state, Vec::new());

        let tick = face.right_bottom() - egui::Vec2::splat(FAMILY_TICK * 0.5);
        let (chosen, _) = click_at(&context, &mut state, tick);
        assert_eq!(
            chosen, None,
            "the tick opens the family, it does not arm it"
        );
        assert_eq!(state.open_rail_group, Some(Icon::Probe));

        // Draw the open flyout once so its area rect is on record, then click its LAST row — the
        // member the face cannot reach.
        probe_frame(&context, &mut state, Vec::new());
        let panel = context
            .memory(|memory| memory.area_rect(egui::Id::new(("rail_flyout", PROBE.name))))
            .unwrap_or(egui::Rect::NOTHING);
        assert!(panel.is_positive(), "the open flyout registers an area");
        let last_row = panel.right_bottom() - egui::vec2(8.0, 8.0);
        let (chosen, _) = click_at(&context, &mut state, last_row);
        assert_eq!(
            chosen,
            Some(2),
            "the flyout's last row arms the last member"
        );
        assert_eq!(
            state.open_rail_group, None,
            "picking closes the family behind it"
        );
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

    /// The open flyout is keyed by face glyph, so two families sharing a face would open each
    /// other's list.
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
