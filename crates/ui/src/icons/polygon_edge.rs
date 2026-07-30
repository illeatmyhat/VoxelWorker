//! `polygon-edge` — a polygon built from ONE EDGE outwards.
//!
//! No construction circle at all, and the polygon left open along its base: the two accented nodes
//! are the edge's ends, and everything else follows from them. Dropping the circle is what says
//! this tool has no centre — it is the third polygon mark and the only one that does not start
//! from the middle.

use super::{Ink, Mark};

/// The authored edge's ends — the only two points this tool asks for.
const LEFT: (f32, f32) = (5.18, 13.76);
const RIGHT: (f32, f32) = (12.82, 13.76);

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[LEFT, (2.82, 6.49), (9.0, 2.0), (15.18, 6.49), RIGHT],
        ink: Ink::SOLID,
    },
    // The base, run past both ends: the edge is the input, not just a side.
    Mark::Line {
        points: &[(4.0, 13.76), (14.0, 13.76)],
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: LEFT,
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: RIGHT,
        size: 2.6,
        ink: Ink::ACCENT,
    },
];
