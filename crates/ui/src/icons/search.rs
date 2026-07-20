//! `search` — a lens and handle: filter by name.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Circle {
        center: (7.5, 7.5),
        radius: 5.0,
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(11.2, 11.2), (15.8, 15.8)],
        ink: Ink::SOLID,
    },
];
