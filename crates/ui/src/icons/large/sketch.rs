//! `sketch` (tile) — a curve with two authored endpoints, drawn as solid handles.
//!
//! The filled dots are the whole point: a sketch is a thing with grabbable control points,
//! and this is the only mark in either family that says so. The rail twin cannot — at 15 pt
//! a filled dot and a stroked ring are the same three pixels — so it settles for hollow
//! vertex squares instead. This is why the two sets are separate drawings.

use crate::icons::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Cubic {
        p0: (5.0, 18.0),
        p1: (8.0, 6.0),
        p2: (18.0, 6.0),
        p3: (21.0, 18.0),
        ink: Ink::SOLID,
    },
    Mark::Disc {
        center: (5.0, 18.0),
        radius: 1.8,
    },
    Mark::Disc {
        center: (21.0, 18.0),
        radius: 1.8,
    },
];
