//! `mirror` — generates the reflection of a selection about an axis.
//!
//! Nearly [`constraint_symmetry`](super::constraint_symmetry)'s construction, and that is now
//! safe: Mirror GENERATES entities, Symmetry asserts a relation between entities that already
//! exist. On the sheet the difference is carried by blue against red; in the rail both resolve to
//! the accent, so what separates them here is the shelf they sit on and the wider chevrons.

use super::{Ink, Mark};

/// The mirror line — dashed, because it is a reference and not geometry the tool emits.
const AXIS: f32 = 9.0;

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(AXIS, 2.0), (AXIS, 16.0)],
        ink: Ink::DASHED,
    },
    Mark::Line {
        points: &[(6.0, 5.0), (2.5, 9.0), (6.0, 13.0)],
        ink: Ink::ACCENT,
    },
    Mark::Line {
        points: &[(12.0, 5.0), (15.5, 9.0), (12.0, 13.0)],
        ink: Ink::ACCENT,
    },
];
