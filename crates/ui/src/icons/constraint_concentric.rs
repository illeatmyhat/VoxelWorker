//! `constraint-concentric` — shared centre, radii free.
//!
//! Two rings and deliberately NO centre dot: at 16 px a dot inside the inner ring closes the gap
//! and the whole mark collapses into a disc. The relation is the shared centre, and the mark says
//! it by the rings agreeing rather than by drawing the centre at all.

use super::{Ink, Mark};

/// The centre both rings share — the relation itself.
const CENTRE: (f32, f32) = (9.0, 9.0);

pub(super) const DRAW: &[Mark] = &[
    Mark::Circle {
        center: CENTRE,
        radius: 6.5,
        ink: Ink::SOLID,
    },
    Mark::Circle {
        center: CENTRE,
        radius: 3.25,
        ink: Ink::ACCENT,
    },
];
