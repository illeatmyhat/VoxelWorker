//! `slot-3-point-arc` — a curved slot through two ends and a point between them.
//!
//! [`slot_center_point_arc`](super::slot_center_point_arc)'s outline with the center and its two
//! dashed radii REMOVED, and a third node placed on the path instead. The center is solved for
//! here rather than clicked, so drawing it would name a point the tool never asks for.

use super::{Ink, Mark};

const PI: f32 = std::f32::consts::PI;

/// The center the arcs are struck from — never accented, because it is derived.
const CENTER: (f32, f32) = (9.0, 13.0);

/// The cap centers, following the sheet's resolved arc conversion (see
/// [`slot_center_point_arc`](super::slot_center_point_arc) on the ~8e-4 offsets).
const LEFT_CAP: (f32, f32) = (2.9386, 9.4986);
const RIGHT_CAP: (f32, f32) = (15.0614, 9.4986);

pub(super) const DRAW: &[Mark] = &[
    Mark::Arc {
        center: CENTER,
        rx: 9.0,
        ry: 9.0,
        from: -2.617999,
        to: -0.523594,
        ink: Ink::SOLID,
    },
    Mark::Arc {
        center: RIGHT_CAP,
        rx: 2.0,
        ry: 2.0,
        from: -0.522763,
        to: 2.617157,
        ink: Ink::SOLID,
    },
    Mark::Arc {
        center: CENTER,
        rx: 5.0,
        ry: 5.0,
        from: -PI / 6.0,
        to: -2.618003,
        ink: Ink::SOLID,
    },
    Mark::Arc {
        center: LEFT_CAP,
        rx: 2.0,
        ry: 2.0,
        from: 0.524435,
        to: 3.664355,
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: (2.9378, 9.5),
        size: 2.2,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: (15.0622, 9.5),
        size: 2.2,
        ink: Ink::ACCENT,
    },
    // The through-point, on the path itself rather than at a center.
    Mark::Node {
        center: (9.0, 6.0),
        size: 2.2,
        ink: Ink::ACCENT,
    },
];
