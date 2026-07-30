//! `conic` — a conic through two ends, an apex, and a rho point.
//!
//! The straight legs to the apex and the curve inside them, with four nodes: both ends, the apex,
//! and the rho point on the curve. Drawing the legs is what makes the apex legible as a CONTROL
//! rather than as a point the curve reaches — the curve visibly misses it.
//!
//! Rho is the fourth click and sits between the apex and the chord's middle; at rho = 1/2 the
//! conic is a parabola, which is what is drawn.

use super::{Ink, Mark};

/// The two ends, the apex the legs meet at, and the rho point on the curve.
const LEFT: (f32, f32) = (2.5, 13.0);
const RIGHT: (f32, f32) = (15.5, 13.0);
const APEX: (f32, f32) = (9.0, 2.0);
const RHO: (f32, f32) = (9.0, 7.5);

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[LEFT, APEX, RIGHT],
        ink: Ink::SOLID,
    },
    // The quadratic through the apex, raised to a cubic — exactly, not fitted.
    Mark::Cubic {
        p0: LEFT,
        p1: (6.8333, 5.6667),
        p2: (11.1667, 5.6667),
        p3: RIGHT,
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: LEFT,
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: RIGHT,
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: APEX,
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: RHO,
        size: 2.6,
        ink: Ink::ACCENT,
    },
];
