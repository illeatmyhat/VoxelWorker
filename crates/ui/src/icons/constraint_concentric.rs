//! `constraint-concentric` — shared center, radii free.
//!
//! Two rings and deliberately NO center dot: at 16 px a dot inside the inner ring closes the gap
//! and the whole mark collapses into a disc. The relation is the shared center, and the mark says
//! it by the rings agreeing rather than by drawing the center at all.

use super::{Ink, Mark};

/// The center both rings share — the relation itself.
const CENTER: (f32, f32) = (9.0, 9.0);

pub(super) const DRAW: &[Mark] = &[
    Mark::Circle {
        center: CENTER,
        radius: 6.5,
        ink: Ink::SOLID,
    },
    Mark::Circle {
        center: CENTER,
        radius: 3.25,
        ink: Ink::CONSTRAINT,
    },
];
