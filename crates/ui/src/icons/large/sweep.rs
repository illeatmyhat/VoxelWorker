//! `sweep` (tile) — a profile carried along a path to a dashed destination.
//!
//! The far profile is dashed because sweep is the reserved third lift: the mark stays
//! honest that the far end is not yet a body the app will build. Same construction as the
//! rail twin, with room for both profiles to be squares rather than ticks.

use crate::icons::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The path.
    Mark::Cubic {
        p0: (5.0, 21.0),
        p1: (5.0, 11.0),
        p2: (12.0, 6.0),
        p3: (21.0, 6.0),
        ink: Ink::SOLID,
    },
    // The profile at the start, and where it is headed.
    Mark::Rect {
        a: (2.0, 18.0),
        b: (8.0, 24.0),
        ink: Ink::SOLID,
    },
    Mark::Rect {
        a: (18.0, 3.0),
        b: (24.0, 9.0),
        ink: Ink::DASHED,
    },
];
