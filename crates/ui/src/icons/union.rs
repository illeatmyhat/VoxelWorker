//! `union` — the merged outline of two overlapping squares, drawn as one body.
//!
//! There is no seam inside it. That is the whole statement: after the fold there is one
//! accumulated surface, not two bodies sharing a wall.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Closed {
        points: &[
            (2.5, 2.5),
            (10.5, 2.5),
            (10.5, 7.5),
            (15.5, 7.5),
            (15.5, 15.5),
            (7.5, 15.5),
            (7.5, 10.5),
            (2.5, 10.5),
        ],
        ink: Ink::SOLID,
    },
];
