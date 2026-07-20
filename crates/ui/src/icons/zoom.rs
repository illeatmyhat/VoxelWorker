//! `zoom` — a lens with a plus: dolly in and out.
//!
//! It shares the lens-and-handle body with `search`, and the plus is the only thing that
//! separates them; the two never appear on the same rail, so the shared body is a saving
//! rather than a collision.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Circle {
        center: (7.5, 7.5),
        radius: 5.0,
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(11.2, 11.2), (15.8, 15.8)],
        ink: Ink::SOLID,
    },
    // The plus inside the lens.
    Mark::Line {
        points: &[(5.2, 7.5), (9.8, 7.5)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(7.5, 5.2), (7.5, 9.8)],
        ink: Ink::SOLID,
    },
];
