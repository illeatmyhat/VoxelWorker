//! `circle-center-diameter` — click the center, drag the radius.
//!
//! The four circle tools are one ring differing only in WHICH points are accented, because which
//! points you click is the entire difference between them. Here it is the center and one point on
//! the ring, joined by the radius the drag traces.

use super::{Ink, Mark};

/// The ring every circle glyph in the family shares.
const CENTER: (f32, f32) = (9.0, 9.0);
const RADIUS: f32 = 6.0;

/// Where the radius line meets the ring — 45°, so it clears the grid's own axes.
const HANDLE: (f32, f32) = (13.2425, 13.2425);

pub(super) const DRAW: &[Mark] = &[
    Mark::Circle {
        center: CENTER,
        radius: RADIUS,
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[CENTER, HANDLE],
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: CENTER,
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: HANDLE,
        size: 2.6,
        ink: Ink::ACCENT,
    },
];
