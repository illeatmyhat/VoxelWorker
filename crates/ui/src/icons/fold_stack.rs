//! `fold-stack` — three stacked rows: the ordered fold of the active scope.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Rect {
        a: (2.0, 2.5),
        b: (16.0, 5.5),
        ink: Ink::SOLID,
    },
    Mark::Rect {
        a: (2.0, 7.5),
        b: (16.0, 10.5),
        ink: Ink::SOLID,
    },
    Mark::Rect {
        a: (2.0, 12.5),
        b: (16.0, 15.5),
        ink: Ink::SOLID,
    },
];
