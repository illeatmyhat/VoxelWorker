//! `sphere` (tile) — a ball, its roundness carried by one receding equator.
//!
//! The equator rides at half weight so it reads as an interior contour. Drawn at equal
//! weight it becomes a second silhouette and the mark turns into a lens or an eye.

use crate::icons::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Circle {
        center: (13.0, 13.0),
        radius: 9.0,
        ink: Ink::SOLID,
    },
    Mark::Ellipse {
        center: (13.0, 13.0),
        rx: 9.0,
        ry: 3.4,
        ink: Ink::faint(0.5),
    },
];
