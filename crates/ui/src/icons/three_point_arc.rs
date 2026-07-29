//! `three-point-arc` — the arc tool: two endpoints and a point the curve passes through.
//!
//! The two endpoint nodes are drawn as the square vertices the other sketch marks use, so an
//! arc reads as the same kind of geometry a polyline is; the through-point is a DOT, because
//! it is consumed at creation and never becomes an entity (ADR 0030 §5, #102).

use super::{Ink, Mark};

const PI: f32 = std::f32::consts::PI;

pub(super) const DRAW: &[Mark] = &[
    // The curve, bulging over its chord.
    Mark::Arc {
        center: (9.0, 13.5),
        rx: 6.5,
        ry: 6.5,
        from: PI,
        to: 2.0 * PI,
        ink: Ink::SOLID,
    },
    // The two endpoint vertices.
    Mark::Rect {
        a: (1.4, 12.4),
        b: (3.6, 14.6),
        ink: Ink::SOLID,
    },
    Mark::Rect {
        a: (14.4, 12.4),
        b: (16.6, 14.6),
        ink: Ink::SOLID,
    },
    // The through-point: an input, not a vertex.
    Mark::Disc {
        center: (9.0, 7.0),
        radius: 1.3,
    },
];
