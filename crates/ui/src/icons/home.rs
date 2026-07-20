//! `home` — a house silhouette: a full-width roof over a body left open at the top.
//!
//! The body deliberately has no top edge; it tucks under the roof, which is what keeps the
//! mark reading as a house rather than as a triangle stacked on a box at 15 pt.
//!
//! Sits on the set's dominant 2.5–15.5 box — the square `fit`, `part`, `density`, `material`
//! and `outset` share — so it carries the same optical weight as the rest of the rail. It was
//! previously drawn on a 12 × 11 box, smaller than every other mark in its own Navigation
//! group, for no reason anyone had recorded. The body keeps the sheet's 3-to-4 ratio against
//! the roof's span.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // Roof: left eave → apex → right eave.
    Mark::Line {
        points: &[(2.5, 8.5), (9.0, 2.5), (15.5, 8.5)],
        ink: Ink::SOLID,
    },
    // Body: left wall down, floor, right wall up.
    Mark::Line {
        points: &[(4.25, 8.0), (4.25, 15.5), (13.75, 15.5), (13.75, 8.0)],
        ink: Ink::SOLID,
    },
];
