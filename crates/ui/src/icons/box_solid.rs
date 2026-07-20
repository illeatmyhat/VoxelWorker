//! `box` — an isometric cube: the box primitive.
//!
//! The module is `box_solid` only because `box` is a reserved word in Rust; the glyph's name in
//! the set, and what a designer calls it, is `box`.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Closed {
        points: &[
            (9.0, 1.5),
            (16.0, 5.5),
            (16.0, 12.5),
            (9.0, 16.5),
            (2.0, 12.5),
            (2.0, 5.5),
        ],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(2.0, 5.5), (9.0, 9.5), (16.0, 5.5)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(9.0, 9.5), (9.0, 16.5)],
        ink: Ink::SOLID,
    },
];
