//! `slot-overall` — a slot given its FULL length, caps included.
//!
//! [`slot_center_to_center`](super::slot_center_to_center)'s outline with the two accented points
//! moved out to the extreme ends. That shift is the entire difference between the two tools, and
//! it is why both marks have to keep the same outline: move anything else and the comparison stops
//! being readable.

use super::{Ink, Mark};

const PI: f32 = std::f32::consts::PI;

/// The extremes the typed length spans — the cap centres pushed out by the cap radius.
const LEFT: (f32, f32) = (2.5, 6.0);
const RIGHT: (f32, f32) = (15.5, 6.0);
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
        center: LEFT,
        size: 2.2,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: RIGHT,
        size: 2.2,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: (12.5, 3.0),
        size: 2.2,
        ink: Ink::ACCENT,
    },
];
