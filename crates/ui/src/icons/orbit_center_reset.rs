//! `orbit-center-reset` — the same pivot, with its ring wound back.
//!
//! Arms and dot are [`orbit_center_place`](super::orbit_center_place)'s, unchanged and to the
//! digit: the two rows act on one thing, so they share a silhouette and differ in exactly one
//! feature. That feature is the ring, opened and given a head — the universal revert sign, spent
//! on the one part of the mark that was already a circle.
//!
//! The opening falls on the east arm; the top is taken, since
//! [`orbit_constrained`](super::orbit_constrained) is a ring gapped at the top with a head in the
//! gap. The sweep is counter-clockwise — clockwise is refresh, a different promise.

use super::{Ink, Mark};

/// Where the ring stops, in radians clockwise from +x. The gap spans ±0.45 about east.
const GAP_HALF_WIDTH: f32 = 0.45;

pub(super) const DRAW: &[Mark] = &[
    Mark::Arc {
        center: (9.0, 9.0),
        rx: 4.25,
        ry: 4.25,
        from: GAP_HALF_WIDTH,
        to: std::f32::consts::TAU - GAP_HALF_WIDTH,
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(9.0, 2.5), (9.0, 6.25)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(9.0, 11.75), (9.0, 15.5)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(2.5, 9.0), (6.25, 9.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(11.75, 9.0), (15.5, 9.0)],
        ink: Ink::SOLID,
    },
    Mark::Disc {
        center: (9.0, 9.0),
        radius: 1.4,
        ink: Ink::SOLID,
    },
    // FILLED, like every other arrowhead in the set: a stroked chevron this size closes into a
    // blob at rail size. Tip on the arc's lower end, pointing the way the sweep travels.
    Mark::Fill {
        points: &[(12.83, 10.85), (12.91, 13.22), (10.93, 12.26)],
        opacity: 1.0,
    },
];
