//! `pan` — a four-headed cross: slide the target in the ground plane.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The two axes.
    Mark::Line {
        points: &[(9.0, 2.0), (9.0, 16.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(2.0, 9.0), (16.0, 9.0)],
        ink: Ink::SOLID,
    },
    // Arrowheads, one per direction.
    Mark::Line {
        points: &[(7.4, 3.6), (9.0, 2.0), (10.6, 3.6)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(7.4, 14.4), (9.0, 16.0), (10.6, 14.4)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(3.6, 7.4), (2.0, 9.0), (3.6, 10.6)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(14.4, 7.4), (16.0, 9.0), (14.4, 10.6)],
        ink: Ink::SOLID,
    },
];
