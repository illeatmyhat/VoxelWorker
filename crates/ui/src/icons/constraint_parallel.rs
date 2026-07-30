//! `constraint-parallel` — equal direction.
//!
//! Two runs at the same slope. Authored on a 36 × 26 canvas because the mark IS wider than tall;
//! it is centred onto the square glyph grid rather than padded out to fill it, and nothing is
//! gained by stretching it.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(4.5, 13.0), (8.5, 5.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(9.5, 13.0), (13.5, 5.0)],
        ink: Ink::ACCENT,
    },
];
