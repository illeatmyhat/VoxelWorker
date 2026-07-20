//! `extrude` (tile) — a profile with per-vertex depth ticks.
//!
//! The sketch is the subject and the ticks are the sweep. Depth is drawn per vertex rather
//! than as one ground rule, which is what distinguishes extruding a profile from setting a
//! datum under it.

use crate::icons::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The profile.
    Mark::Line {
        points: &[(6.0, 17.0), (10.0, 9.0), (16.0, 12.0), (20.0, 6.0)],
        ink: Ink::SOLID,
    },
    // Where each vertex is carried to.
    Mark::Line {
        points: &[(6.0, 17.0), (6.0, 21.0)],
        ink: Ink::faint(0.5),
    },
    Mark::Line {
        points: &[(20.0, 6.0), (20.0, 10.0)],
        ink: Ink::faint(0.5),
    },
    Mark::Line {
        points: &[(13.0, 12.0), (13.0, 18.0)],
        ink: Ink::faint(0.5),
    },
];
