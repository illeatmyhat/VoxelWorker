//! `axis_span` — a linear dimension between two points, along whatever direction it is measured in.
//!
//! The layout turns on TWO INDEPENDENT TESTS, and treating them as one is the mistake this file
//! exists to not make:
//!
//! | arrows fit | value fits | drawing |
//! |---|---|---|
//! | yes | yes | everything inside |
//! | yes | no | arrows stay in, the value moves out onto an extension |
//! | no | no | arrows flip outside pointing in, the value on the extension |
//!
//! The middle row is the one a single test loses. A span of 30 holds two 9-unit arrows with room
//! to spare, but the value has to clear BOTH ARROW BASES, not merely the span — so it leaves while
//! the arrows stay. There is no fourth row: a value that fits inside a span too short for its own
//! arrows cannot happen, because the arrow test is the weaker of the two.
//!
//! There is ONE entry point, because there is one drawing. An aligned span is the case where the
//! direction handed in is the run's own, and both extension lines come out the same length; a width
//! or a height hands in one of the plane's directions instead, and each end reaches the line by a
//! different amount with one of them possibly already sitting on it. A second entry point for the
//! aligned case would be the same code with a scalar offset in place of a point — which is exactly
//! the freedom the author loses when the value cannot be carried ALONG its own line.

use egui::{Pos2, Vec2};

use super::{
    arrowhead, value_width, Anchor, Drawing, Label, Piece, Rank, ARROW_LENGTH, GAP, OVERRUN,
};

/// How far apart two points stand ALONG ONE DIRECTION — a width or a height rather than a length.
///
/// `along` is that direction in screen pixels, and `through` a point the dimension line passes
/// through: where the author dropped the annotation. Each end reaches the line by its own
/// perpendicular, which is what distinguishes this drawing from an aligned one — for a diagonal run
/// the two extension lines have different lengths, and for an axis-aligned run one of them vanishes
/// because the point is already on the line.
pub fn axis_span(
    from: Pos2,
    to: Pos2,
    along: Vec2,
    through: Pos2,
    value: &str,
    rank: Rank,
) -> Drawing {
    let length = along.length();
    let unit = if length > f32::EPSILON {
        along / length
    } else {
        Vec2::X
    };
    let normal = Vec2::new(unit.y, -unit.x);
    // Where a feature point meets the dimension line, measured straight across it.
    let foot = |point: Pos2| point + normal * (through - point).dot(normal);

    let (near, far) = (foot(from), foot(to));
    let extensions = [(from, near), (to, far)]
        .into_iter()
        .filter_map(|(feature, meeting)| {
            let reach = meeting - feature;
            let distance = reach.length();
            // A point already on the dimension line needs no line drawn to it, and a stub of
            // length GAP + OVERRUN drawn anyway would read as a tick the drawing does not mean.
            (distance > GAP).then(|| {
                let toward = reach / distance;
                Piece::Polyline(vec![feature + toward * GAP, meeting + toward * OVERRUN])
            })
        })
        .collect();
    measured_between(near, far, extensions, Some(through), value, rank)
}

/// The dimension line itself, between the two points its extension lines reach — arrows, the two
/// fit tests, and the label that leaves when it does not fit.
///
/// `pieces` arrives holding the extension lines, because that is the one part the two callers
/// disagree about. `beside` is where the author dropped the annotation, which is what decides where
/// ALONG the line the value rides and which end it leaves by when it does not fit. `None` keeps the
/// value in the middle and sends it out past the far end, which is where a drawing with nothing
/// placed has always put it.
fn measured_between(
    near: Pos2,
    far: Pos2,
    mut pieces: Vec<Piece>,
    beside: Option<Pos2>,
    value: &str,
    rank: Rank,
) -> Drawing {
    let text = rank.indication("", value);
    let width = value_width(&text);

    let length = (far - near).length();
    let along = if length > f32::EPSILON {
        (far - near) / length
    } else {
        Vec2::X
    };

    // The two tests. Both are computed; neither is inferred from the other.
    let arrows_fit = length >= 2.0 * ARROW_LENGTH + 2.0;
    let value_fits = length >= 2.0 * ARROW_LENGTH + width + 2.0 * GAP;

    if arrows_fit {
        // The dimension line stops at the arrow BASES, so nothing pokes past a tip.
        pieces.push(Piece::Polyline(vec![
            near + along * ARROW_LENGTH,
            far - along * ARROW_LENGTH,
        ]));
        pieces.push(arrowhead(near, -along));
        pieces.push(arrowhead(far, along));
    } else {
        // Arrows outside pointing in. The line stays continuous between the extension lines, so
        // the dimension still reads as spanning them rather than as two detached ticks.
        pieces.push(Piece::Polyline(vec![near, far]));
        pieces.push(Piece::Polyline(vec![
            near - along * 2.0 * ARROW_LENGTH,
            near,
        ]));
        pieces.push(arrowhead(near, along));
        pieces.push(arrowhead(far, -along));
    }

    let bearing = super::upright_radians(along.y.atan2(along.x));
    // How far along the line the author put the value, measured from `near`.
    let put = beside.map_or(length / 2.0, |at| (at - near).dot(along));
    // The room a value needs to ride inline: half of itself, clear of the arrowhead beside it.
    // At the middle of a span the value fits in, this is exactly `value_fits` — which is why an
    // unplaced drawing lands on the same layout it always did.
    let room = width / 2.0 + ARROW_LENGTH + GAP;
    let label = if value_fits && arrows_fit && put >= room && put <= length - room {
        Label {
            at: near + along * put,
            text,
            radians: bearing,
            anchor: Anchor::Middle,
            lift: GAP,
        }
    } else {
        // It leaves by the end it was dropped past, on a leader that REACHES it — never a naked
        // line running off to a number floating somewhere past its end, and never a number left
        // behind at the line when the hand carried it further.
        let by_the_near_end = put < length / 2.0;
        let (end, direction, dropped) = if by_the_near_end {
            (near, -along, -put)
        } else {
            (far, along, put - length)
        };
        let out = dropped.max(width + 2.0 * GAP);
        pieces.push(Piece::Polyline(vec![end, end + direction * out]));
        // The value reads left to right whichever end it left by, so which of its two edges lands
        // on the leader depends on whether the reading direction agrees with the way out.
        let reading = Vec2::new(bearing.cos(), bearing.sin());
        Label {
            at: end + direction * (out - width - GAP),
            text,
            radians: bearing,
            anchor: if reading.dot(direction) >= 0.0 {
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
