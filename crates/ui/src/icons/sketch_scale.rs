//! `sketch-scale` — resizes a selection uniformly about a fixed corner.
//!
//! Two nested rectangles sharing their top-left corner, and a diagonal arrow leaving it. Sharing
//! the corner is the mark's claim: scaling has an ANCHOR, and two concentric boxes would say it
//! grows about its centre, which is a different tool.
//!
//! Uniform is said by the outer box being the inner one's shape, not merely a larger rectangle —
//! both are squares here so the ratio is visible at a glance.

use super::{Ink, Mark};

/// The corner both boxes share, and the point the scale is measured from.
const ANCHOR: (f32, f32) = (1.5, 14.5);

pub(super) const DRAW: &[Mark] = &[
    Mark::Rect {
        a: (1.5, 9.0),
        b: (7.0, ANCHOR.1),
        ink: Ink::SOLID,
    },
    Mark::Rect {
        a: (1.5, 4.5),
        b: (11.5, ANCHOR.1),
        ink: Ink::ACCENT,
    },
    Mark::Line {
        points: &[(12.5, 3.5), (14.0, 2.0)],
        ink: Ink::ACCENT,
    },
    Mark::Closed {
        points: &[(15.0, 1.0), (13.5504, 4.1466), (11.8534, 2.4496)],
        ink: Ink::ACCENT,
    },
];
