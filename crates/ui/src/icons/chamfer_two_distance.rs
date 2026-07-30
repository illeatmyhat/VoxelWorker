//! `chamfer-two-distance` — a distance along each leg, set independently.
//!
//! The third of the chamfer siblings. Its bevel sits between
//! [`chamfer_equal`](super::chamfer_equal)'s 45° and
//! [`chamfer_distance_angle`](super::chamfer_distance_angle)'s shallower run.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(3.0, 2.0), (3.0, 5.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(9.5, 13.0), (15.5, 13.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(3.0, 5.5), (9.0, 13.0)],
        ink: Ink::ACCENT,
    },
];
