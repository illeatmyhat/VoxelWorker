//! `sweep` — a profile carried along a curved path to a dashed destination.
//!
//! The far profile is dashed because sweep is the reserved third lift: the mark is honest that
//! the far end is not yet a body the app will build.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The path — the SVG cubic `C3 8 8.5 4.5 15 4.5`.
    Mark::Cubic {
        p0: (3.0, 15.0),
        p1: (3.0, 8.0),
        p2: (8.5, 4.5),
        p3: (15.0, 4.5),
        ink: Ink::SOLID,
    },
    // The profile at the start, and where it is headed.
    Mark::Rect {
        a: (1.2, 13.2),
        b: (4.8, 16.8),
        ink: Ink::SOLID,
    },
    Mark::Rect {
        a: (13.2, 2.7),
        b: (16.8, 6.3),
        ink: Ink::DASHED,
    },
];
