//! `span` — a linear dimension between two points.
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
    let text = rank.indication("", value);
    let width = value_width(&text);

    let length = (to - from).length();
    let along = if length > f32::EPSILON {
        (to - from) / length
    } else {
        Vec2::X
    };
    let normal = Vec2::new(along.y, -along.x) * offset.signum();
    let reach = normal * offset.abs();

    // The two tests. Both are computed; neither is inferred from the other.
    let arrows_fit = length >= 2.0 * ARROW_LENGTH + 2.0;
    let value_fits = length >= 2.0 * ARROW_LENGTH + width + 2.0 * GAP;

    let (near, far) = (from + reach, to + reach);
    let mut pieces = vec![
        // Extension lines: gapped off the feature, run past the dimension line.
        Piece::Polyline(vec![from + normal * GAP, near + normal * OVERRUN]),
        Piece::Polyline(vec![to + normal * GAP, far + normal * OVERRUN]),
    ];

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
