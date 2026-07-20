//! `emboss-recess` — the emboss footprint with the surface driven down instead of up.
//!
//! It is the same mark mirrored about the surface line, which is exactly the relationship the
//! op has: one amount, signed.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(1.5, 7.0), (6.0, 7.0), (6.0, 12.0), (12.0, 12.0), (12.0, 7.0), (16.5, 7.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(6.0, 2.5), (6.0, 15.5)],
        ink: Ink::DASHED,
    },
    Mark::Line {
        points: &[(12.0, 2.5), (12.0, 15.5)],
        ink: Ink::DASHED,
    },
];
