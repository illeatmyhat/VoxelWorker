//! `constraint-fix` — position frozen in sketch space.
//!
//! A padlock, following Fusion. The alternatives both collide with things this app already has:
//! SolidWorks' anchor reads as a viewport gizmo, and Onshape's hatched ground symbol drawn over a
//! viewport that CONTAINS a ground plane is a real misread, not a theoretical one.
//!
//! Fix is one of the fundamental constraints and is authorable like any other — it pins the
//! remaining freedoms of a point rather than marking it as special.

use super::{Ink, Mark};

const PI: f32 = std::f32::consts::PI;

/// The shackle's radius, and the body's top edge.
const SHACKLE: f32 = 2.5;
const LID: f32 = 10.0;

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(6.5, LID), (6.5, 7.0)],
        ink: Ink::SOLID,
    },
    // The half turn over the top: PI to TAU is the upper arc with y running down.
    Mark::Arc {
        center: (9.0, 7.0),
        rx: SHACKLE,
        ry: SHACKLE,
        from: PI,
        to: 2.0 * PI,
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(11.5, 7.0), (11.5, LID)],
        ink: Ink::SOLID,
    },
    Mark::Rect {
        a: (5.5, LID),
        b: (12.5, 15.5),
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: (9.0, 12.75),
        size: 1.7,
        ink: Ink::SOLID,
    },
];
