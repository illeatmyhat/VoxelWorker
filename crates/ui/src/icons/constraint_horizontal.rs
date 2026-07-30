//! `constraint-horizontal` — level in the SKETCH PLANE.
//!
//! Not level in the world: the constraint is plane-local, like every other thing a sketch snaps
//! to. The two end nodes are drawn because it applies to a PAIR OF POINTS, not only to a segment,
//! and a bare bar would say the narrower thing.
//!
//! [`constraint_vertical`](super::constraint_vertical) is this mark's exact quarter turn, so the
//! pair reads as a pair.

use super::{Ink, Mark};

/// The bar's ends, which are also its two constrained points.
const FROM: (f32, f32) = (3.5, 9.0);
const TO: (f32, f32) = (14.5, 9.0);

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
