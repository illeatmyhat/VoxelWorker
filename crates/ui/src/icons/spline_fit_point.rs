//! `spline-fit-point` — a spline through points you place ON it.
//!
//! The nodes sit on the curve. That is the whole distinction from
//! [`spline_control_point`](super::spline_control_point), whose nodes sit off it and pull, and it
//! is why the two glyphs draw the same curve: same result, different handles.
//!
//! No control polygon here — a fit-point spline has no polygon a user ever sees.

use super::{Ink, Mark};

/// The two ends, and the middle fit point — all three lie on the curve.
const START: (f32, f32) = (2.5, 12.0);
const END: (f32, f32) = (15.5, 4.0);

pub(super) const DRAW: &[Mark] = &[
    Mark::Cubic {
        p0: START,
        p1: (5.5, 2.0),
        p2: (9.5, 13.0),
        p3: END,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: START,
        size: 2.6,
        ink: Ink::SOLID,
    },
    // The curve's own midpoint at t = 1/2, so the node really is on it.
    Mark::Node {
        center: (7.875, 7.625),
        size: 2.6,
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: END,
        size: 2.6,
        ink: Ink::SOLID,
    },
];
