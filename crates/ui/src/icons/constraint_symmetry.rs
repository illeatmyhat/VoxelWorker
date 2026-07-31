//! `constraint-symmetry` — two entities mirrored about an axis.
//!
//! The axis is line art because it is the REFERENCE; both mirrored entities are driven, so both
//! carry the constraint ink. That is the one place in the set where the red appears twice, and it
//! is correct: symmetry has no privileged side.
//!
//! Near-identical to [`mirror`](super::mirror), and safely so — Mirror GENERATES entities, this
//! asserts a relation between entities that already exist. Symmetry binds the underlying curves,
//! not their endpoints.

use super::{Ink, Mark};

/// The mirror line, drawn dashed because it is a reference and not geometry.
const AXIS: f32 = 9.0;

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(AXIS, 2.0), (AXIS, 16.0)],
        ink: Ink::DASHED,
    },
    Mark::Line {
        points: &[(5.5, 5.5), (3.0, 9.0), (5.5, 12.5)],
        ink: Ink::CONSTRAINT,
    },
    Mark::Line {
        points: &[(12.5, 5.5), (15.0, 9.0), (12.5, 12.5)],
        ink: Ink::CONSTRAINT,
    },
];
