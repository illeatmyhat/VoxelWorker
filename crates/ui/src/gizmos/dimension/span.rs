//! `span` — a linear dimension between two points, aligned with them or with one plane axis.
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
//! The two entry points differ only in where the dimension line ends up. An aligned span puts it
//! parallel to the run, so both extension lines are the same length; an axis span puts it along one
//! of the plane's own directions, so each end reaches the line by a different amount and one of
//! them may already be sitting on it. Everything after that — arrows, the two tests, the label —
//! is one piece of code, because two copies of it would drift.

use egui::{Pos2, Vec2};

use super::{
    arrowhead, value_width, Anchor, Drawing, Label, Piece, Rank, ARROW_LENGTH, GAP, OVERRUN,
};

/// A span between two feature points, its dimension line `offset` away along the normal.
///
/// `offset`'s sign picks the side. The sheet draws every span horizontal because it has no sketch
/// to point at; here `from` and `to` are wherever the two picked points projected to, and the
/// whole layout is done in the span's own frame so nothing is bound to the screen axes.
pub fn span(from: Pos2, to: Pos2, offset: f32, value: &str, rank: Rank) -> Drawing {
    let length = (to - from).length();
    let along = if length > f32::EPSILON {
        (to - from) / length
    } else {
        Vec2::X
    };
    let normal = Vec2::new(along.y, -along.x) * offset.signum();
    let reach = normal * offset.abs();

    let (near, far) = (from + reach, to + reach);
    let extensions = vec![
        // Gapped off the feature, run past the dimension line.
        Piece::Polyline(vec![from + normal * GAP, near + normal * OVERRUN]),
        Piece::Polyline(vec![to + normal * GAP, far + normal * OVERRUN]),
    ];
    measured_between(near, far, extensions, value, rank)
}

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
    measured_between(near, far, extensions, value, rank)
}

/// The dimension line itself, between the two points its extension lines reach — arrows, the two
/// fit tests, and the label that leaves when it does not fit.
///
/// `pieces` arrives holding the extension lines, because that is the one part the two callers
/// disagree about.
fn measured_between(
    near: Pos2,
    far: Pos2,
    mut pieces: Vec<Piece>,
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
    let label = if value_fits && arrows_fit {
        Label {
            at: near + (far - near) / 2.0,
            text,
            radians: bearing,
            anchor: Anchor::Middle,
            lift: GAP,
        }
    } else {
        // The value leaves on an extension whose length IS the text advance — never a naked line
        // running off to a number floating somewhere past its end.
        pieces.push(Piece::Polyline(vec![
            far,
            far + along * (width + 2.0 * GAP),
        ]));
        Label {
            at: far + along * GAP,
            text,
            radians: bearing,
            anchor: Anchor::Start,
            lift: GAP,
        }
    };

    Drawing {
        pieces,
        labels: vec![label],
        rank,
    }
}
