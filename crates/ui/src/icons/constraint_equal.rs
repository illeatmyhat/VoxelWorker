//! `constraint-equal` — equal length, or equal radius.
//!
//! The constraint ink is the member that TAKES the other's size. A member already carrying a
//! dimension wins, which respects work already done; with no dimensioned member the first
//! selected wins. The same rule governs
//! [`constraint_coincident`](super::constraint_coincident).

use super::{Ink, Mark};

/// Both bars run the same span — that is the whole assertion.
const LEFT: f32 = 4.0;
const RIGHT: f32 = 14.0;

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(LEFT, 7.0), (RIGHT, 7.0)],
        ink: Ink::CONSTRAINT,
    },
    Mark::Line {
        points: &[(LEFT, 11.0), (RIGHT, 11.0)],
        ink: Ink::SOLID,
    },
];
