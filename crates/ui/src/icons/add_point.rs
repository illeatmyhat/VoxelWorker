//! `add-point` — place a profile point on the plane.
//!
//! A **target reticle**: four inward ticks with a centre gap converging on a filled point — "a
//! point lands here". General placement, NOT edge-splitting: Add Point drops a point anywhere on
//! the grid (free or snapped), which is one verb of the entity-based sketch model (ADR 0028;
//! owner reframe 2026-07-23). The solid centre node = the committed point; distinct from
//! `snap-voxel` (full through-lines, no gap) and from `select-vertex` (an arrow carrying a node).
//! Chosen glyph = the "general verbs" sheet's add-point v3.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The four inward reticle ticks, with a centre gap (vertical on x=9, horizontal on y=10).
    Mark::Line {
        points: &[(9.0, 3.0), (9.0, 6.5)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(9.0, 13.5), (9.0, 17.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(2.5, 10.0), (6.0, 10.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(12.0, 10.0), (15.5, 10.0)],
        ink: Ink::SOLID,
    },
    // The placed point — a filled node at the reticle centre.
    Mark::Fill {
        points: &[(7.7, 8.7), (10.3, 8.7), (10.3, 11.3), (7.7, 11.3)],
        opacity: 1.0,
    },
];
