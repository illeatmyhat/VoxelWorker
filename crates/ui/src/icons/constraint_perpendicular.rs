//! `constraint-perpendicular` — directions differ by 90°.
//!
//! A chevron, not an L. An L makes one arm the base and the other dependent, and perpendicularity
//! is symmetric — the two arms have to look like peers even though only one of them moves.
//!
//! The small inner chevron is the right-angle tick, drawn as a chevron for the same reason.

use super::{Ink, Mark};

/// The vertex the two arms meet at.
const CORNER: (f32, f32) = (9.0, 14.0);

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(3.0, 7.0), CORNER],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[CORNER, (15.0, 7.0)],
        ink: Ink::CONSTRAINT,
    },
    Mark::Line {
        points: &[(6.7, 11.85), (9.0, 9.7), (11.3, 11.85)],
        ink: Ink::SOLID,
    },
];
