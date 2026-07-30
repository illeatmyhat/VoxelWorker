//! `rectangle-3-point` — a rectangle at an arbitrary angle.
//!
//! Drawn TILTED, which is the entire content of the mark: the first two clicks set one edge and
//! its direction, the third sets the depth, so unlike the axis-aligned rectangle this one is not
//! bound to the sketch axes. An upright box here would say nothing the two-point tool does not.
//!
//! Three accented corners, in click order; the fourth is derived and stays plain.

use super::{Ink, Mark};

/// The three clicks, in order: edge start, edge end, then depth.
const FIRST: (f32, f32) = (2.5, 12.0);
const SECOND: (f32, f32) = (11.5, 15.0);
const THIRD: (f32, f32) = (14.346, 6.462);

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[FIRST, SECOND, THIRD, (5.346, 3.462), FIRST],
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: FIRST,
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: SECOND,
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: THIRD,
        size: 2.6,
        ink: Ink::ACCENT,
    },
];
