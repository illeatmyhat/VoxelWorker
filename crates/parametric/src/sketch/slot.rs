//! Continuous construction geometry for linear and circular-arc slots.

use std::f64::consts::PI;

use super::{
    center_arc_candidate, three_point_circle_candidate, ArcTurn, InternalContainment, LineSide,
    TangentBranch,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SlotEdgeCandidate {
    Line {
        from: [f64; 2],
        to: [f64; 2],
    },
    Arc {
        from: [f64; 2],
        to: [f64; 2],
        sweep_degrees: f64,
    },
}

/// Whether a slot's centerline runs straight or turns about a center.
///
/// This one discriminant also settles how the two rails are held to each other — a straight spine
/// gives rails that must stay parallel, a turning one gives concentric rails — so the two facts
/// cannot drift apart into disagreement. What it deliberately does NOT settle is the width: that
/// is the one freedom the shape keeps, and it is what an author changes by dragging a rail.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SlotTurn {
    Straight,
    About([f64; 2]),
}

/// The centerline a slot's rails are offset from, in the terms an author holds it by.
///
/// A slot IS a spine plus a width, but the boundary the tool emits has no spine in it — the
/// centerline is exactly the curve that does not get drawn. Carrying it here is what lets the
/// commit reify the author's own picks and tie them to the boundary, so that afterwards the
/// center translates the shape and the ends reshape it.
///
/// Every grammar reports the same three handles regardless of what it asked the author to click:
/// Overall Slot picks the extremes and Center Point Slot picks a midpoint, but both RESOLVE to a
/// spine, and it is the spine the drawing should remember.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlotSpine {
    /// Where the centerline starts — the center of the cap that closes that end.
    pub start: [f64; 2],
    /// Where it ends, likewise.
    pub end: [f64; 2],
    pub turn: SlotTurn,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SlotCandidate {
    /// Four boundary curves in connected traversal order. The two long rails are `[0]` and `[2]`;
    /// the caps between them are `[1]` and `[3]`, closing the `end` and `start` of the spine
    /// respectively.
    pub edges: [SlotEdgeCandidate; 4],
    /// The centerline these edges are offset from.
    pub spine: SlotSpine,
    /// The two points the author clicked, when the grammar asked for the slot's OVERALL extent.
    ///
    /// Every other grammar's picks already ARE spine points, and this is `None` for them. Overall
    /// is the one that asks for something else — the far ends of the finished shape — and those
    /// picks would otherwise be spent computing the spine and thrown away, leaving the author with
    /// no handle on the quantity they actually named.
    pub reach: Option<[[f64; 2]; 2]>,
    /// How the boundary meets itself at each of its four corners: junction `i` joins
    /// edge `i` to edge `i + 1` of [`edges`](Self::edges), wrapping, in that member order.
    ///
    /// A slot is smooth all the way round — that is what makes it a slot rather than four curves
    /// that happen to touch — so every junction is a tangency, and the branch is read off the
    /// geometry HERE rather than re-derived by whoever asserts it. A caller that had to work out
    /// which side of a straight edge its cap sits on, or which of two circles contains the other,
    /// would be deriving a second time what this function already knew exactly.
    pub junctions: [TangentBranch; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotCandidateError {
    NonFinite,
    CollapsedCenterline,
    CollapsedWidth,
    WidthExceedsArcRadius,
    InvalidArc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearSlotKind {
    CenterToCenter,
    Overall,
    CenterPoint,
}

/// Construct one of the three linear slot forms from its first two defining picks and a width
/// point on either straight edge.
///
/// # Errors
///
/// Refuses non-finite, collapsed, or (for Overall Slot) cap-overlapping geometry.
pub fn linear_slot_candidate(
    kind: LinearSlotKind,
    first: [f64; 2],
    second: [f64; 2],
    width_point: [f64; 2],
) -> Result<SlotCandidate, SlotCandidateError> {
    finite([first, second, width_point])?;
    let authored = [second[0] - first[0], second[1] - first[1]];
    let authored_length = authored[0].hypot(authored[1]);
    if authored_length <= f64::EPSILON {
        return Err(SlotCandidateError::CollapsedCenterline);
    }
    let direction = [authored[0] / authored_length, authored[1] / authored_length];
    let signed_width = direction[0].mul_add(
        width_point[1] - first[1],
        -direction[1] * (width_point[0] - first[0]),
    );
    let half_width = signed_width.abs();
    if half_width <= f64::EPSILON {
        return Err(SlotCandidateError::CollapsedWidth);
    }
    let side = signed_width.signum();
    let normal = [-direction[1] * side, direction[0] * side];
    let (start_center, end_center) = match kind {
        LinearSlotKind::CenterToCenter => (first, second),
        LinearSlotKind::Overall => {
            if authored_length <= 2.0 * half_width {
                return Err(SlotCandidateError::CollapsedCenterline);
            }
            (
                [
                    direction[0].mul_add(half_width, first[0]),
                    direction[1].mul_add(half_width, first[1]),
                ],
                [
                    (-direction[0]).mul_add(half_width, second[0]),
                    (-direction[1]).mul_add(half_width, second[1]),
                ],
            )
        }
        LinearSlotKind::CenterPoint => (
            [
                first[0].mul_add(2.0, -second[0]),
                first[1].mul_add(2.0, -second[1]),
            ],
            second,
        ),
    };
    let mut candidate = linear_boundary(start_center, end_center, normal, half_width);
    // Only Overall spent its picks on something other than the spine, so only Overall has a reach
    // to report. The other two grammars' picks survive as the spine points themselves.
    if kind == LinearSlotKind::Overall {
        candidate.reach = Some([first, second]);
    }
    Ok(candidate)
}

/// Construct a curved slot whose centerline is the circular arc through three points.
///
/// # Errors
///
/// Refuses non-finite or degenerate center arcs and widths that collapse/cross the inner edge.
pub fn three_point_arc_slot_candidate(
    start: [f64; 2],
    end: [f64; 2],
    through: [f64; 2],
    width_point: [f64; 2],
) -> Result<SlotCandidate, SlotCandidateError> {
    finite([width_point])?;
    let spine = three_point_arc_slot_spine(start, end, through)?;
    arc_boundary(
        spine.center,
        spine.start,
        spine.end,
        spine.radius,
        spine.sweep_radians,
        width_point,
    )
}

/// The CENTERLINE of an arc slot — the arc its two rails are offset from, before any width.
///
/// Split out so the width step's preview can draw the very arc the commit will build its rails
/// around, rather than a straight run through the picks that looks nothing like the result. The
/// candidate constructors below take their own spine from here, so there is one definition of it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArcSlotSpine {
    /// The arc's center.
    pub center: [f64; 2],
    /// Where the centerline starts.
    pub start: [f64; 2],
    /// Where it ends.
    pub end: [f64; 2],
    /// Its radius.
    pub radius: f64,
    /// Signed sweep; negative is clockwise.
    pub sweep_radians: f64,
}

/// The centerline arc through three points, in the three-point grammar's own reading of them.
///
/// # Errors
///
/// Refuses non-finite or degenerate input, exactly as the full candidate does.
pub fn three_point_arc_slot_spine(
    start: [f64; 2],
    end: [f64; 2],
    through: [f64; 2],
) -> Result<ArcSlotSpine, SlotCandidateError> {
    finite([start, end, through])?;
    let circle = three_point_circle_candidate(start, end, through)
        .map_err(|_| SlotCandidateError::InvalidArc)?;
    let sweep_radians = signed_sweep_through(circle.center, start, end, through)?;
    Ok(ArcSlotSpine {
        center: circle.center,
        start,
        end,
        radius: circle.radius,
        sweep_radians,
    })
}

/// The centerline arc from a center, a start point, and a direction to end in.
///
/// # Errors
///
/// Refuses non-finite or degenerate input, exactly as the full candidate does.
pub fn center_arc_slot_spine(
    center: [f64; 2],
    start: [f64; 2],
    end_direction: [f64; 2],
    turn: ArcTurn,
) -> Result<ArcSlotSpine, SlotCandidateError> {
    finite([center, start, end_direction])?;
    let centerline = center_arc_candidate(center, start, end_direction, turn)
        .map_err(|_| SlotCandidateError::InvalidArc)?;
    Ok(ArcSlotSpine {
        center,
        start,
        end: centerline.endpoint,
        radius: centerline.radius,
        sweep_radians: centerline.sweep_radians,
    })
}

/// Construct a curved slot from its arc center, start point, end direction, and width point.
///
/// # Errors
///
/// Refuses non-finite or degenerate center arcs and widths that collapse/cross the inner edge.
pub fn center_arc_slot_candidate(
    center: [f64; 2],
    start: [f64; 2],
    end_direction: [f64; 2],
    turn: ArcTurn,
    width_point: [f64; 2],
) -> Result<SlotCandidate, SlotCandidateError> {
    finite([width_point])?;
    let spine = center_arc_slot_spine(center, start, end_direction, turn)?;
    arc_boundary(
        spine.center,
        spine.start,
        spine.end,
        spine.radius,
        spine.sweep_radians,
        width_point,
    )
}

fn linear_boundary(
    start_center: [f64; 2],
    end_center: [f64; 2],
    normal: [f64; 2],
    radius: f64,
) -> SlotCandidate {
    let offset = [normal[0] * radius, normal[1] * radius];
    let points = [
        [start_center[0] + offset[0], start_center[1] + offset[1]],
        [end_center[0] + offset[0], end_center[1] + offset[1]],
        [end_center[0] - offset[0], end_center[1] - offset[1]],
        [start_center[0] - offset[0], start_center[1] - offset[1]],
    ];
    // Every cap sits on the spine, and every straight edge is offset one half-width off the spine
    // to the `normal` side, so all four junctions see their cap on the side OPPOSITE `normal` —
    // and the same named side for both edges, because the second edge runs the other way and its
    // offset flips with it. `offset_hand` is which hand `normal` is on relative to the spine's
    // direction; the caps are on the other one.
    let spine = [
        end_center[0] - start_center[0],
        end_center[1] - start_center[1],
    ];
    let offset_hand = (-normal[0]).mul_add(spine[1], normal[1] * spine[0]);
    let cap_side = if offset_hand > 0.0 {
        LineSide::Right
    } else {
        LineSide::Left
    };
    // And which way each cap turns, off the same reading. A cap runs from the far rail's corner to
    // the near one and has to bulge OUT past the end of the slot; mirror the offset and that same
    // half-turn comes back around the inside, biting a bite out of each end. A hardcoded sign was
    // right for one hand and wrong for the other, which is a thing no test that always picks its
    // width on the same side of the spine can see.
    let cap_sweep_degrees = if offset_hand > 0.0 { -180.0 } else { 180.0 };
    SlotCandidate {
        spine: SlotSpine {
            start: start_center,
            end: end_center,
            turn: SlotTurn::Straight,
        },
        reach: None,
        junctions: [TangentBranch::Line(cap_side); 4],
        edges: [
            SlotEdgeCandidate::Line {
                from: points[0],
                to: points[1],
            },
            SlotEdgeCandidate::Arc {
                from: points[1],
                to: points[2],
                sweep_degrees: cap_sweep_degrees,
            },
            SlotEdgeCandidate::Line {
                from: points[2],
                to: points[3],
            },
            SlotEdgeCandidate::Arc {
                from: points[3],
                to: points[0],
                sweep_degrees: cap_sweep_degrees,
            },
        ],
    }
}

fn arc_boundary(
    center: [f64; 2],
    start: [f64; 2],
    end: [f64; 2],
    radius: f64,
    sweep_radians: f64,
    width_point: [f64; 2],
) -> Result<SlotCandidate, SlotCandidateError> {
    let width_radius = (width_point[0] - center[0]).hypot(width_point[1] - center[1]);
    let half_width = (width_radius - radius).abs();
    if half_width <= f64::EPSILON {
        return Err(SlotCandidateError::CollapsedWidth);
    }
    if half_width >= radius {
        return Err(SlotCandidateError::WidthExceedsArcRadius);
    }
    let unit = |point: [f64; 2]| {
        [
            (point[0] - center[0]) / radius,
            (point[1] - center[1]) / radius,
        ]
    };
    let start_unit = unit(start);
    let end_unit = unit(end);
    let at_radius = |unit: [f64; 2], edge_radius: f64| {
        [
            unit[0].mul_add(edge_radius, center[0]),
            unit[1].mul_add(edge_radius, center[1]),
        ]
    };
    let points = [
        at_radius(start_unit, radius + half_width),
        at_radius(end_unit, radius + half_width),
        at_radius(end_unit, radius - half_width),
        at_radius(start_unit, radius - half_width),
    ];
    let sweep_degrees = sweep_radians.to_degrees();
    let cap_sweep = sweep_degrees.signum() * 180.0;
    Ok(SlotCandidate {
        spine: SlotSpine {
            start,
            end,
            turn: SlotTurn::About(center),
        },
        reach: None,
        // The two long arcs are concentric with the spine at radius +/- the half-width; each cap
        // is centered ON the spine, a full spine radius from that shared center. So the OUTER arc
        // and a cap have their centers (R + w) - w apart, which is internal tangency with the
        // outer one containing the cap; the INNER arc and a cap are (R - w) + w apart, which is
        // external. Containment is stated relative to the junction's own member order.
        junctions: [
            TangentBranch::Internal {
                contains: InternalContainment::First,
            },
            TangentBranch::External,
            TangentBranch::External,
            TangentBranch::Internal {
                contains: InternalContainment::Second,
            },
        ],
        edges: [
            SlotEdgeCandidate::Arc {
                from: points[0],
                to: points[1],
                sweep_degrees,
            },
            SlotEdgeCandidate::Arc {
                from: points[1],
                to: points[2],
                sweep_degrees: cap_sweep,
            },
            SlotEdgeCandidate::Arc {
                from: points[2],
                to: points[3],
                sweep_degrees: -sweep_degrees,
            },
            SlotEdgeCandidate::Arc {
                from: points[3],
                to: points[0],
                sweep_degrees: cap_sweep,
            },
        ],
    })
}

fn signed_sweep_through(
    center: [f64; 2],
    start: [f64; 2],
    end: [f64; 2],
    through: [f64; 2],
) -> Result<f64, SlotCandidateError> {
    let bearing = |point: [f64; 2]| (point[1] - center[1]).atan2(point[0] - center[0]);
    let start_angle = bearing(start);
    let end_angle = bearing(end);
    let through_angle = bearing(through);
    let ccw_sweep = (end_angle - start_angle).rem_euclid(2.0 * PI);
    let ccw_through = (through_angle - start_angle).rem_euclid(2.0 * PI);
    if ccw_sweep <= f64::EPSILON || 2.0_f64.mul_add(PI, -ccw_sweep) <= f64::EPSILON {
        return Err(SlotCandidateError::InvalidArc);
    }
    Ok(if ccw_through < ccw_sweep {
        ccw_sweep
    } else {
        2.0_f64.mul_add(-PI, ccw_sweep)
    })
}

fn finite<const N: usize>(points: [[f64; 2]; N]) -> Result<(), SlotCandidateError> {
    points
        .into_iter()
        .flatten()
        .all(f64::is_finite)
        .then_some(())
        .ok_or(SlotCandidateError::NonFinite)
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used
)]
mod tests {
    use super::*;

    #[test]
    fn three_linear_grammars_resolve_to_two_lines_and_two_caps() {
        for (kind, first, second, expected_span) in [
            (LinearSlotKind::CenterToCenter, [0.0, 0.0], [6.0, 0.0], 8.0),
            (LinearSlotKind::Overall, [0.0, 0.0], [6.0, 0.0], 6.0),
            (LinearSlotKind::CenterPoint, [0.0, 0.0], [3.0, 0.0], 8.0),
        ] {
            let slot = linear_slot_candidate(kind, first, second, [0.0, 1.0]).unwrap();
            let SlotEdgeCandidate::Line { from, .. } = slot.edges[0] else {
                panic!("straight edge")
            };
            let SlotEdgeCandidate::Arc { to, .. } = slot.edges[1] else {
                panic!("cap")
            };
            assert_eq!(to[0] - from[0] + 2.0, expected_span);
        }
    }

    #[test]
    fn both_arc_grammars_make_concentric_boundaries_and_semicircular_caps() {
        let through = three_point_arc_slot_candidate(
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0_f64.sqrt(), 2.0_f64.sqrt()],
            [3.0, 0.0],
        )
        .unwrap();
        let centered = center_arc_slot_candidate(
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            ArcTurn::CounterClockwise,
            [3.0, 0.0],
        )
        .unwrap();
        for (through_edge, centered_edge) in through.edges.iter().zip(centered.edges.iter()) {
            let (
                SlotEdgeCandidate::Arc {
                    from: through_from,
                    to: through_to,
                    sweep_degrees: through_sweep,
                },
                SlotEdgeCandidate::Arc {
                    from: centered_from,
                    to: centered_to,
                    sweep_degrees: centered_sweep,
                },
            ) = (through_edge, centered_edge)
            else {
                panic!("arc slot boundaries contain only arcs")
            };
            for (left, right) in through_from
                .iter()
                .chain(through_to)
                .zip(centered_from.iter().chain(centered_to))
            {
                assert!((left - right).abs() < 1e-12);
            }
            assert!((through_sweep - centered_sweep).abs() < 1e-12);
        }
        assert!(matches!(
            centered.edges[1],
            SlotEdgeCandidate::Arc {
                sweep_degrees: 180.0,
                ..
            }
        ));
    }
}
