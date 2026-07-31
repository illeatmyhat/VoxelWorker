//! `fillet` — rounds a corner where two curves meet.
//!
//! The two legs stop short of where they would have crossed, and the accent arc bridges the gap
//! tangentially. The gap is the mark's whole argument: a fillet does not *add* a round, it
//! *replaces* the corner, and legs drawn all the way to the vertex would say the opposite.
//!
//! Chamfer is the same composition with a straight bridge — the three chamfers and this one are
//! siblings on purpose, so the shelf reads as one operation with four ways to specify it.

use super::{Ink, Mark};

const PI: f32 = std::f32::consts::PI;

/// The round's radius, and the distance each leg stops short by.
const RADIUS: f32 = 5.0;

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(3.0, 2.0), (3.0, 7.5)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(8.5, 13.0), (15.5, 13.0)],
        ink: Ink::SOLID,
    },
    // Center is RADIUS in from both legs, so the arc meets each along its tangent.
    Mark::Arc {
        center: (8.0, 8.0),
        rx: RADIUS,
        ry: RADIUS,
        from: PI,
        to: PI / 2.0,
        ink: Ink::ACCENT,
    },
];
