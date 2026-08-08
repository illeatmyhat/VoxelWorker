//! `angle` — an angular dimension between two lines.
//!
//! An arc centered on the two lines' intersection, with the terminators TANGENT to the arc at each
//! leg. The tight case makes the same reversal [`axis_span`](super::axis_span()) makes — arc length below two
//! arrow lengths and the arrows swing outside the legs pointing in — which is what keeps the two
//! gizmos legible as one family rather than two conventions.
//!
//! **The vertex is never drawn.** When the two lines do not actually meet, the intersection is
//! virtual: thin extension lines carry each leg out to the arc, and the vertex stays absent
//! because it is not an entity the sketch contains. Drawing it would put a point on screen that no
//! pick can ever land on.

use egui::{Pos2, Vec2};

use super::{arrowhead, value_width, Anchor, Drawing, Label, Piece, Rank, ARROW_LENGTH, GAP};

/// How far past the arc an extension line runs when a leg falls short of it.
const OVERRUN: f32 = 8.0;

/// How far the leader runs out along the bisector when the value cannot sit on the arc.
const LEADER: f32 = 30.0;

/// How much of one arm's own geometry lies along the ray its corner is struck on, as the interval
/// it occupies measured out from the vertex.
///
/// An interval and not a length, because a line does not have to start at the vertex. Two lines
/// that cross without touching each begin some way out along their own ray, and the arc struck
/// between them can land nearer than either — so the dogleg that carries a leg to the arc runs
/// INWARD there, where for a line the arc overshoots it runs outward. One number cannot say both.
///
/// Both may be negative, which is the arm the corner points away from: the drawing extends it back
/// through the vertex, which is the same rule and not a case of its own.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Leg {
    /// Where the arm's own geometry starts, along the ray.
    pub nearest: f32,
    /// Where it stops.
    pub furthest: f32,
}

/// An angle at `vertex` between the bearings `from` and `to` (radians, y running down), its arc
/// struck at `radius`.
///
/// `legs` says where each arm's own geometry sits along the ray its side of the corner is struck
/// on, one entry per bearing — see [`Leg`]. Wherever that does not already cover the arc, this
/// draws the extension line that carries it there.
pub fn angle(
    vertex: Pos2,
    from: f32,
    to: f32,
    radius: f32,
    legs: [Leg; 2],
    value: &str,
    rank: Rank,
) -> Drawing {
    let text = rank.indication("", value);
    let width = value_width(&text);
    let at =
        |bearing: f32, distance: f32| vertex + Vec2::new(bearing.cos(), bearing.sin()) * distance;

    let mut pieces = Vec::new();
    // Extension lines, each drawn only across the gap ITS OWN arm leaves. Asked per leg because
    // the answer is per leg: the dogleg has to start where that line stops, and which end of it
    // that is depends on whether the arc overshoots the arm or falls short of it.
    let target = radius + OVERRUN;
    for (bearing, leg) in [from, to].into_iter().zip(legs) {
        let gap = if target > leg.furthest {
            Some((leg.furthest, target))
        } else if target < leg.nearest {
            // Coming the other way, so the overrun is on the INNER side: the line crosses the arc
            // and runs its eight past, exactly as it does when the arm stops short. Starting at
            // `target` instead would leave a hole the width of the overrun between the arc and
            // the line, with the arc's own terminator joined to nothing.
            Some(((radius - OVERRUN).max(0.0), leg.nearest))
        } else {
            // The arc lands on the arm itself, which is already drawn.
            None
        };
        if let Some((start, end)) = gap {
            pieces.push(Piece::Polyline(vec![at(bearing, start), at(bearing, end)]));
        }
    }

    let sweep = to - from;
    let direction = sweep.signum();
    let tangent = |bearing: f32| Vec2::new(-bearing.sin(), bearing.cos()) * direction;
    let (start, end) = (at(from, radius), at(to, radius));

    // Both tests, on the ARC LENGTH rather than the chord: an angle is dimensioned along its arc,
    // and using the chord would let a wide sweep read as too tight.
    let arc_length = radius * sweep.abs();
    let arrows_fit = arc_length >= 2.0 * ARROW_LENGTH + 2.0;
    let value_fits = arc_length >= 2.0 * ARROW_LENGTH + width + 2.0 * GAP;

    let step = ARROW_LENGTH / radius * direction;
    if arrows_fit {
        // The arc runs between the arrow bases, and each arrow points outward along the arc.
        pieces.push(Piece::Arc {
            center: vertex,
            radius,
            from: from + step,
            to: to - step,
        });
        pieces.push(arrowhead(start, -tangent(from)));
        pieces.push(arrowhead(end, tangent(to)));
    } else {
        // Too tight: the arc still spans the legs, and the arrows swing outside pointing in.
        pieces.push(Piece::Arc {
            center: vertex,
            radius,
            from,
            to,
        });
        pieces.push(Piece::Arc {
            center: vertex,
            radius,
            from: from - 2.0 * step,
            to: from,
        });
        pieces.push(Piece::Arc {
            center: vertex,
            radius,
            from: to,
            to: to + 2.0 * step,
        });
        pieces.push(arrowhead(start, tangent(from)));
        pieces.push(arrowhead(end, -tangent(to)));
    }

    let bisector = (from + to) / 2.0;
    let label = if value_fits && arrows_fit {
        Label {
            at: at(bisector, radius),
            // Tangent to the arc where the value sits, folded upright.
            radians: super::upright_radians(bisector + std::f32::consts::FRAC_PI_2),
            text,
            anchor: Anchor::Middle,
            lift: GAP,
        }
    } else {
        // The value leaves on a leader along the BISECTOR — the one direction that belongs to
        // neither leg, so a tight angle's value never looks attached to one of them.
        let jog = at(bisector, radius + LEADER);
        let side = if jog.x >= vertex.x { 1.0 } else { -1.0 };
        pieces.push(Piece::Polyline(vec![
            at(bisector, radius),
            jog,
            jog + Vec2::X * side * (width + 2.0 * GAP),
        ]));
        Label {
            at: jog + Vec2::X * side * GAP,
            text,
            radians: 0.0,
            anchor: if side > 0.0 {
                Anchor::Start
            } else {
                Anchor::End
            },
            lift: GAP,
        }
    };

    Drawing {
        pieces,
        labels: vec![label],
        rank,
    }
}
