//! `add-point` — place a profile point on the plane.
//!
//! A **target reticle**: four inward ticks with a center gap converging on a node — "a point
//! lands here". General placement, NOT edge-splitting: Add Point drops a point anywhere on the
//! grid (free or snapped), which is one verb of the entity-based sketch model (ADR 0028; owner
//! reframe 2026-07-23).
//!
//! Re-drawn from the sheet (owner, 2026-07-30). Three changes, each making it agree with the
//! rest of the create shelf: the center is a [`Mark::Node`] rather than a hand-spelled `Fill`
//! quad, which is the square-vertex law (ADR 0030 §5) said in the type rather than by four
//! corners; it is ACCENT, because on this shelf the accent is what the tool will produce; and
//! the reticle centers on 9,9 instead of 9,10, where the old drawing sat a half unit low
//! against every other mark in the set.
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
