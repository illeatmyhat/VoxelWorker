//! `axes-gizmo` — the Z-up triad: one long vertical arm, two short ground arms.
//!
//! The vertical arm is drawn longest on purpose. The world is Z-up, and a triad whose three
//! arms are equal leaves the reader to guess which one is vertical.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // +Z: vertical, and the dominant arm.
    Mark::Line {
        points: &[(9.0, 10.5), (9.0, 2.5)],
        ink: Ink::SOLID,
    },
    // The ground plane, XY.
    Mark::Line {
        points: &[(9.0, 10.5), (15.5, 14.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(9.0, 10.5), (2.5, 14.0)],
        ink: Ink::SOLID,
    },
    // The stub toward the viewer: front is −Y.
    Mark::Line {
        points: &[(9.0, 10.5), (9.0, 12.5)],
        ink: Ink::SOLID,
    },
];
