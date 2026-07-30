//! `polygon-inscribed` — a polygon whose VERTICES touch the circle.
//!
//! The pair with [`polygon_circumscribed`](super::polygon_circumscribed) only reads if both draw
//! the same pentagon on the same construction circle and differ solely in which touches the
//! circle. Here the accented handle is a vertex, sitting on the ring.
//!
//! A pentagon rather than a hexagon: a hexagon's flats align with the grid and the inscribed and
//! circumscribed versions become hard to tell apart at 16 px.

use super::{Ink, Mark};

/// The construction circle, and the vertex handle that rides it.
const CENTRE: (f32, f32) = (9.0, 9.0);
const RADIUS: f32 = 6.5;
const HANDLE: (f32, f32) = (9.0, 2.5);

pub(super) const DRAW: &[Mark] = &[
    Mark::Circle {
        center: CENTRE,
        radius: RADIUS,
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[
            HANDLE,
            (15.18, 6.99),
            (12.82, 14.26),
            (5.18, 14.26),
            (2.82, 6.99),
            HANDLE,
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
