//! `spline-control-point` — a spline shaped by handles OFF the curve.
//!
//! Same cubic as [`spline_fit_point`](super::spline_fit_point), drawn with its control polygon
//! dashed and the two interior control points nodded. Neither of those nodes touches the curve,
//! which is the point: a control point pulls, it does not pass through.
//!
//! The polygon is dashed for the same reason every reference in this set is — it is not geometry
//! the tool emits.

use super::{Ink, Mark};

/// The four control points. The interior two are the handles a user drags.
const P0: (f32, f32) = (2.5, 12.0);
const P1: (f32, f32) = (5.5, 2.0);
const P2: (f32, f32) = (9.5, 13.0);
const P3: (f32, f32) = (15.5, 4.0);

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[P0, P1, P2, P3],
        ink: Ink::DASHED,
    },
    Mark::Cubic {
        p0: P0,
        p1: P1,
        p2: P2,
        p3: P3,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: P1,
        size: 2.6,
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: P2,
        size: 2.6,
        ink: Ink::SOLID,
    },
];
