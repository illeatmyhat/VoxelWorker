//! `polygon-circumscribed` — a polygon whose EDGES touch the circle.
//!
//! [`polygon_inscribed`](super::polygon_inscribed)'s twin, and deliberately the same pentagon on
//! the same construction circle. The accented handle has moved from a vertex to an edge midpoint,
//! which is the only difference between the two tools and so has to be the only difference
//! between the two marks.

use super::{Ink, Mark};

/// The construction circle, and the edge-midpoint handle that rides it.
const CENTRE: (f32, f32) = (9.0, 9.0);
const RADIUS: f32 = 5.0;
const HANDLE: (f32, f32) = (11.94, 4.955);

pub(super) const DRAW: &[Mark] = &[
    Mark::Circle {
        center: CENTRE,
        radius: RADIUS,
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[
            (9.0, 2.82),
            (14.88, 7.09),
            (12.635, 14.0),
            (5.365, 14.0),
            (3.12, 7.09),
            (9.0, 2.82),
        ],
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: CENTRE,
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: HANDLE,
        size: 2.6,
        ink: Ink::ACCENT,
    },
];
