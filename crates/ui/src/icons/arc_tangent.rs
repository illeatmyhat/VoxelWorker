//! `arc-tangent` — an arc leaving an existing curve along its direction.
//!
//! The straight run and the arc meet with no kink: the run ends at (9, 13) and the arc's centre is
//! directly above it, so the arc's tangent there is horizontal. A mark that only nearly did this
//! would be teaching that "tangent" means "close enough".
//!
//! Its sibling is [`line`](super::line), which drags into the same arc from a segment's end. The
//! difference is only which one you start from.

use super::{Ink, Mark};

const PI: f32 = std::f32::consts::PI;

/// The seam: the run's end, and the arc's start.
const SEAM: (f32, f32) = (9.0, 13.0);

/// Directly above the seam, which is what makes the join tangent.
const CENTRE: (f32, f32) = (9.0, 8.0);
const RADIUS: f32 = 5.0;

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(2.0, 13.0), SEAM],
        ink: Ink::SOLID,
    },
    Mark::Arc {
        center: CENTRE,
        rx: RADIUS,
        ry: RADIUS,
        from: PI / 2.0,
        to: 0.0,
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: SEAM,
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: (14.0, 8.0),
        size: 2.6,
        ink: Ink::ACCENT,
    },
];
