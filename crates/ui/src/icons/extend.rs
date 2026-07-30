//! `extend` — grows a curve to its nearest boundary.
//!
//! Trim's inverse, and drawn as its mirror: the accent is on the stretch that appears rather than
//! the stretch that goes. The boundary is a full-height line so it reads as a wall the growth
//! stops against, and the arrow below states the direction a static mark otherwise could not.

use super::{Ink, Mark};

/// Where the existing curve ends and the new stretch begins.
const REACH: (f32, f32) = (6.0, 11.0);

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(1.5, 11.0), REACH],
        ink: Ink::SOLID,
    },
    // What the click adds.
    Mark::Line {
        points: &[REACH, (14.0, 11.0)],
        ink: Ink::ACCENT,
    },
    // The boundary it grows to.
    Mark::Line {
        points: &[(14.0, 3.0), (14.0, 15.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(4.0, 6.5), (11.5, 6.5)],
        ink: Ink::SOLID,
    },
    Mark::Closed {
        points: &[(13.0, 6.5), (9.75, 7.7), (9.75, 5.3)],
        ink: Ink::SOLID,
    },
];
