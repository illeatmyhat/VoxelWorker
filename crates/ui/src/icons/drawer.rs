//! `drawer` — a panel with a docked right edge and a caret pointing into it.
//!
//! The caret points outward, toward the edge the stack folds to, so the mark reads as the
//! action rather than as a static layout diagram.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Rect {
        a: (2.5, 2.5),
        b: (15.5, 15.5),
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(11.0, 2.5), (11.0, 15.5)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(12.5, 6.5), (14.5, 9.0), (12.5, 11.5)],
        ink: Ink::SOLID,
    },
];
