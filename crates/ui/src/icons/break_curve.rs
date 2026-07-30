//! `break-curve` — splits a curve in two at a point, changing nothing about its shape.
//!
//! Both curves are drawn whole and the accent is the new VERTEX, because that is the entire
//! result: nothing is added, nothing is removed, and one entity becomes two that happen to lie
//! exactly where the one did. A mark that drew a gap would be describing Trim.
//!
//! The square is the set's authored-vertex mark (ADR 0030 §5) — a disc would say the pick was
//! consumed rather than kept.

use super::{Ink, Mark};

/// The vertex the split mints, at the crossing of the two curves.
const SPLIT: (f32, f32) = (9.0, 9.0);

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(2.0, 9.0), (16.0, 9.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(9.0, 3.0), (9.0, 15.0)],
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: SPLIT,
        size: 2.6,
        ink: Ink::ACCENT,
    },
];
