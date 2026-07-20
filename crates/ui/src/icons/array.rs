//! `array` — one solid cell followed by two dashed repeats.
//!
//! Only the first is solid: exactly one node was authored, and the repeats are placements the
//! decorator generates, which is also why the array is one entry in the fold and not three.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Rect {
        a: (1.5, 6.5),
        b: (5.5, 11.5),
        ink: Ink::SOLID,
    },
    Mark::Rect {
        a: (7.0, 6.5),
        b: (11.0, 11.5),
        ink: Ink::DASHED,
    },
    Mark::Rect {
        a: (12.5, 6.5),
        b: (16.5, 11.5),
        ink: Ink::DASHED,
    },
];
