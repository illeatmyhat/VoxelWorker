//! `constraint-midpoint` — a point sits at the parametric middle of a curve.
//!
//! The carrier is DASHED so the mark cannot be read as
//! [`constraint_horizontal`](super::constraint_horizontal), which is a solid bar with its nodes at
//! the ends. Here the bar is the reference and the single centred node is what the constraint
//! drives — the two marks would otherwise be the same picture.

use super::{Ink, Mark};

/// The carrier's ends, and the middle the node is pinned to.
const FROM: (f32, f32) = (2.5, 9.0);
const TO: (f32, f32) = (15.5, 9.0);

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[FROM, TO],
        ink: Ink::DASHED,
    },
    Mark::Node {
        center: ((FROM.0 + TO.0) / 2.0, FROM.1),
        size: 2.6,
        ink: Ink::CONSTRAINT,
    },
];
