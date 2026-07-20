//! `part` — a container square with one child inside it.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Rect {
        a: (2.5, 2.5),
        b: (15.5, 15.5),
        ink: Ink::SOLID,
    },
    Mark::Rect {
        a: (7.0, 7.0),
        b: (11.0, 11.0),
        ink: Ink::SOLID,
    },
];
