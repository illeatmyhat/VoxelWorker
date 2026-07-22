//! `delete-vertex` — remove a profile point, the inverse of place.
//!
//! A single node struck through with an X. Deliberately distinct from `subtract` (two
//! overlapping bodies): this removes one vertex, it does not compose two solids. ADR 0028
//! slice 2.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The vertex being removed.
    Mark::Rect {
        a: (6.0, 6.0),
        b: (12.0, 12.0),
        ink: Ink::SOLID,
    },
    // The delete cross.
    Mark::Line {
        points: &[(6.0, 6.0), (12.0, 12.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(12.0, 6.0), (6.0, 12.0)],
        ink: Ink::SOLID,
    },
];
