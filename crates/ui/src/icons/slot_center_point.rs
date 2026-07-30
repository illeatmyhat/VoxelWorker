//! `slot-center-point` — a slot placed from its own middle outwards.
//!
//! The third of the straight slots, and the only one whose first click is not on an end: the
//! accented middle is the anchor, and the length grows symmetrically about it. Outline shared with
//! [`slot_center_to_center`](super::slot_center_to_center) and
//! [`slot_overall`](super::slot_overall).

use super::{Ink, Mark};

const PI: f32 = std::f32::consts::PI;

/// The slot's own middle — the point this tool anchors on.
const MIDDLE: (f32, f32) = (9.0, 6.0);
const CAP: f32 = 3.0;

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(5.5, 3.0), (12.5, 3.0)],
        ink: Ink::SOLID,
    },
    Mark::Arc {
        center: (12.5, 6.0),
        rx: CAP,
        ry: CAP,
        from: -PI / 2.0,
        to: PI / 2.0,
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(12.5, 9.0), (5.5, 9.0)],
        ink: Ink::SOLID,
    },
    Mark::Arc {
        center: (5.5, 6.0),
        rx: CAP,
        ry: CAP,
        from: PI / 2.0,
        to: 3.0 * PI / 2.0,
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: MIDDLE,
        size: 2.2,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: (12.5, 6.0),
        size: 2.2,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: (12.5, 3.0),
        size: 2.2,
        ink: Ink::ACCENT,
    },
];
