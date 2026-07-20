//! `cancel` — a cross: nothing is written to the document.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(3.5, 3.5), (14.5, 14.5)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(14.5, 3.5), (3.5, 14.5)],
        ink: Ink::SOLID,
    },
];
