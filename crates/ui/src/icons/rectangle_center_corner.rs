//! `rectangle-center-corner` — a rectangle grown symmetrically from its middle.
//!
//! Five nodes, but only two carry the accent: the centre and one corner, which are the two clicks.
//! The other three corners are drawn plain so the mark shows a whole rectangle while still saying
//! that three of its corners are consequences.

use super::{Ink, Mark};

/// The two clicks: the anchor, and the corner that sets both extents.
const CENTRE: (f32, f32) = (9.0, 9.0);
const CORNER: (f32, f32) = (15.0, 13.5);

pub(super) const DRAW: &[Mark] = &[
    Mark::Rect {
        a: (3.0, 4.5),
        b: CORNER,
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: (3.0, 4.5),
        size: 2.6,
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: (15.0, 4.5),
        size: 2.6,
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: (3.0, 13.5),
        size: 2.6,
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: CENTRE,
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: CORNER,
        size: 2.6,
        ink: Ink::ACCENT,
    },
];
