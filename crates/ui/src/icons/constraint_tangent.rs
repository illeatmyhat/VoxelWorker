//! `constraint-tangent` — touch with a common direction.
//!
//! Really tangent, not nearly: the line sits at y = 6.5 and the ring's top is 11 − 4.5 = 6.5. A
//! mark about a tangency that is off by half a unit is a mark that teaches the wrong tolerance.

use super::{Ink, Mark};

/// The ring, and the height its top therefore reaches.
const CENTER: (f32, f32) = (8.0, 11.0);
const RADIUS: f32 = 4.5;
const TOUCH: f32 = CENTER.1 - RADIUS;

pub(super) const DRAW: &[Mark] = &[
    Mark::Circle {
        center: CENTER,
        radius: RADIUS,
        ink: Ink::CONSTRAINT,
    },
    Mark::Line {
        points: &[(2.0, TOUCH), (16.0, TOUCH)],
        ink: Ink::SOLID,
    },
];
