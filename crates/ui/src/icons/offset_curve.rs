//! `offset-curve` — a parallel copy at a fixed distance.
//!
//! The source is line art and the copy is the accent, held three units off along its whole length.
//! The copy's corner is ROUNDED where the source's is square: that is what an outward offset
//! actually produces, and drawing it mitred would promise a shape the operation does not make.

use super::{Ink, Mark};

const PI: f32 = std::f32::consts::PI;

/// The offset distance, which is also the radius the outer corner rounds at.
const DISTANCE: f32 = 3.0;

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(4.0, 3.0), (4.0, 12.0), (15.0, 12.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(1.0, 3.0), (1.0, 12.0)],
        ink: Ink::ACCENT,
    },
    // The corner an offset really turns: an arc about the source's vertex.
    Mark::Arc {
        center: (4.0, 12.0),
        rx: DISTANCE,
        ry: DISTANCE,
        from: PI,
        to: PI / 2.0,
        ink: Ink::ACCENT,
    },
    Mark::Line {
        points: &[(4.0, 15.0), (15.0, 15.0)],
        ink: Ink::ACCENT,
    },
];
