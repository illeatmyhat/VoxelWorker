//! `measure` — a ruler with graduations dropping from its top edge.
//!
//! The ticks are uneven in length in the source and kept that way: a ruler with identical ticks
//! reads as a hatched bar, and the graduation is the point.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Rect {
        a: (1.5, 6.5),
        b: (16.5, 11.5),
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(5.25, 6.5), (5.25, 9.25)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(9.0, 6.5), (9.0, 9.25)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(12.75, 6.5), (12.75, 9.25)],
        ink: Ink::SOLID,
    },
];
