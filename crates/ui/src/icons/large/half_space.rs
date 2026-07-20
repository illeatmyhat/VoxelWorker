//! `half-space` (tile) — a ground rule and an inclined plane leaving it.
//!
//! Neither line terminates inside the box, which is what says UNBOUNDED without drawing a
//! boundary. Two strokes, no fades — the sparsest mark in the tile set, and it needs no
//! more.

use crate::icons::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(3.0, 17.0), (23.0, 17.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(8.0, 17.0), (14.0, 9.0), (23.0, 9.0)],
        ink: Ink::SOLID,
    },
];
