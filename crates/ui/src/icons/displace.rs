//! `displace` — a perturbed surface above its dashed flat reference.
//!
//! The datum below is what makes the zigzag mean displacement rather than terrain: the field
//! is a deviation from a reference, and the reference has to be visible to be deviated from.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(1.5, 10.0), (5.0, 6.5), (8.5, 10.0), (12.0, 6.5), (15.5, 10.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(1.5, 14.0), (16.5, 14.0)],
        ink: Ink::DASHED,
    },
];
