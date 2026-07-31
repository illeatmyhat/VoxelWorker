//! `add-point` — place a profile point on the plane.
//!
//! A **target reticle**: four inward ticks with a center gap converging on a node — "a point
//! lands here". General placement, NOT edge-splitting: Add Point drops a point anywhere on the
//! grid (free or snapped), which is one verb of the entity-based sketch model.
//!
//! The center is a [`Mark::Node`] rather than a hand-spelled `Fill` quad, so the square-vertex
//! law is stated by the type rather than by four corners, and it is ACCENT because on the
//! create shelf the accent is what the tool will produce.
//!
//! Distinct from `snap-voxel` (full through-lines, no gap) and from `select-vertex` (an arrow
//! over a profile).

use super::{Ink, Mark};

/// The placed point. Larger than a profile vertex, because here it IS the subject.
const POINT: f32 = 3.5;

pub(super) const DRAW: &[Mark] = &[
    // The four inward reticle ticks, with a center gap the point sits in.
    Mark::Line {
        points: &[(9.0, 2.5), (9.0, 5.5)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(9.0, 12.5), (9.0, 15.5)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(2.5, 9.0), (5.5, 9.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(12.5, 9.0), (15.5, 9.0)],
        ink: Ink::SOLID,
    },
    // The placed point.
    Mark::Node {
        center: (9.0, 9.0),
        size: POINT,
        ink: Ink::ACCENT,
    },
];
