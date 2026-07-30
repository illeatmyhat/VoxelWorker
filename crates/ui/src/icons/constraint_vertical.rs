//! `constraint-vertical` — the exact 90° rotation of
//! [`constraint_horizontal`](super::constraint_horizontal).
//!
//! Authored on a 22 × 36 canvas to its twin's 36 × 22 and centred onto the square grid, so the two
//! really are one drawing turned. Padding either back to a square would break the correspondence
//! that makes them legible as a pair.

use super::{Ink, Mark};

/// The bar's ends, which are also its two constrained points.
const FROM: (f32, f32) = (9.0, 3.5);
const TO: (f32, f32) = (9.0, 14.5);

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[FROM, TO],
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: FROM,
        size: 2.6,
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: TO,
        size: 2.6,
        ink: Ink::SOLID,
    },
];
