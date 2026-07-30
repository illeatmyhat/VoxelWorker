//! `chamfer-equal` — cuts a corner off, the same distance along each leg.
//!
//! [`fillet`](super::fillet)'s composition with a straight bridge: two legs stopping short, and
//! an accent spanning the gap. The three chamfers differ only in where the bridge lands, which is
//! exactly what the three ways of specifying one differ in.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(3.0, 2.0), (3.0, 7.5)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(8.5, 13.0), (15.5, 13.0)],
        ink: Ink::SOLID,
    },
    // Five along each leg from the corner at (3, 13): the bevel runs at 45°.
    Mark::Line {
        points: &[(3.0, 8.0), (8.0, 13.0)],
        ink: Ink::ACCENT,
    },
];
