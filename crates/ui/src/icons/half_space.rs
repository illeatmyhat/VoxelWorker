//! `half-space` — a plane in perspective with hatching falling away beneath it.
//!
//! The hatching is what says *half-space* rather than *plane*: the body is everything on one
//! side, and the plane itself is unbounded, so the glyph cannot be a closed shape.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The plane.
    Mark::Closed {
        points: &[(1.5, 11.0), (6.5, 6.0), (16.5, 6.0), (11.5, 11.0)],
        ink: Ink::SOLID,
    },
    // The side that is body.
    Mark::Line {
        points: &[(4.0, 12.5), (4.0, 15.5)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(8.0, 12.5), (8.0, 15.5)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(12.0, 12.5), (12.0, 15.5)],
        ink: Ink::SOLID,
    },
];
