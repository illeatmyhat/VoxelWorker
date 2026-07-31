//! `slot-center-point-arc` — a curved slot swept about a center.
//!
//! Two concentric arcs closed by two caps, with the sweep's center accented and DASHED radii out
//! to each end. The radii are what say the slot follows a circle rather than an arbitrary curve —
//! they are references, not geometry, which is why they dash.
//!
//! Its sibling [`slot_3_point_arc`](super::slot_3_point_arc) drops them, and the center with them.

use super::{Ink, Mark};

const PI: f32 = std::f32::consts::PI;

/// The sweep's center — this tool's anchor, and the reason for the dashed radii.
const CENTER: (f32, f32) = (9.0, 11.0);

/// The two cap centers. Off the sixths by ~8e-4 radians: they follow the sheet's resolved arc
/// conversion rather than an idealised 30°, and the gate compares to 2e-3.
const LEFT_CAP: (f32, f32) = (2.9386, 7.4986);
const RIGHT_CAP: (f32, f32) = (15.0614, 7.4986);

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
    Mark::Line {
        points: &[CENTER, (2.9378, 7.5)],
        ink: Ink::DASHED,
    },
    Mark::Line {
        points: &[CENTER, (15.0622, 7.5)],
        ink: Ink::DASHED,
    },
    Mark::Node {
        center: CENTER,
        size: 2.2,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: (2.9378, 7.5),
        size: 2.2,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: (15.0622, 7.5),
        size: 2.2,
        ink: Ink::ACCENT,
    },
];
