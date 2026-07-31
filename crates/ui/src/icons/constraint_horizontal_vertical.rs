//! `constraint-horizontal-vertical` — along an axis, whichever axis the line is already nearer.
//!
//! ONE tool, TWO constraints (Fusion's arrangement): the author says "line this up with an axis"
//! and the drawing decides which axis it meant. What gets asserted is a plain
//! [`Horizontal`](super::constraint_horizontal) or [`Vertical`](super::constraint_vertical), and
//! the badge left behind carries THAT mark rather than this one — this glyph belongs to the
//! question, those belong to the answer.
//!
//! So it is drawn as exactly those two marks superimposed: the same bars, the same end nodes, the
//! same coordinates, one atop the other. Nothing here is a third shape to learn. The nodes are a
//! shade smaller than the pair's own, which is the only concession the crossing asks for.

use super::{Ink, Mark};

/// The horizontal bar's ends — [`constraint_horizontal`](super::constraint_horizontal)'s own.
const LEFT: (f32, f32) = (3.5, 9.0);
const RIGHT: (f32, f32) = (14.5, 9.0);
/// The vertical bar's ends — [`constraint_vertical`](super::constraint_vertical)'s own.
const TOP: (f32, f32) = (9.0, 3.5);
const BOTTOM: (f32, f32) = (9.0, 14.5);

/// Slightly under the pair's 2.6, so four nodes on one crossing stay four nodes.
const NODE: f32 = 2.2;

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[LEFT, RIGHT],
        ink: Ink::CONSTRAINT,
    },
    Mark::Line {
        points: &[TOP, BOTTOM],
        ink: Ink::CONSTRAINT,
    },
    Mark::Node {
        center: LEFT,
        size: NODE,
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: RIGHT,
        size: NODE,
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: TOP,
        size: NODE,
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: BOTTOM,
        size: NODE,
        ink: Ink::SOLID,
    },
];
