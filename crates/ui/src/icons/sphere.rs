//! `sphere` — a circle with its equator, so the mark reads as a solid and not as a disc.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Circle {
        center: (9.0, 9.0),
        radius: 6.5,
        ink: Ink::SOLID,
    },
    Mark::Ellipse {
        center: (9.0, 9.0),
        rx: 6.5,
        ry: 2.4,
        ink: Ink::SOLID,
    },
];
