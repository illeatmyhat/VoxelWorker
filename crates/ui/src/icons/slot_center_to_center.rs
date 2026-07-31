//! `slot-center-to-center` — a slot given the distance between its two arc CENTERS.
//!
//! The three straight-slot glyphs share one outline and differ only in which three points are
//! accented, because that is the only thing that differs about the tools. Here the two accented
//! points are the arc centers, so the typed length excludes the caps.

use super::{Ink, Mark};

const PI: f32 = std::f32::consts::PI;

/// The two cap centers, and the radius the caps turn at.
const LEFT: (f32, f32) = (5.5, 6.0);
const RIGHT: (f32, f32) = (12.5, 6.0);
const CAP: f32 = 3.0;

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(5.5, 3.0), (12.5, 3.0)],
        ink: Ink::SOLID,
    },
    Mark::Arc {
        center: RIGHT,
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
        center: LEFT,
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
    // The width click, on the flank — shared by all three straight slots.
    Mark::Node {
        center: (12.5, 3.0),
        size: 2.2,
        ink: Ink::ACCENT,
    },
];
