//! `blend-curve` — a curve joining two others tangentially.
//!
//! Two stubs and an S between them. The cubic's control points sit ON the stubs' own directions —
//! `p1` continues the left stub upward, `p2` continues the right one downward — so the join is a
//! real tangency rather than a curve that merely arrives at the right place.
//!
//! The S rather than a C is deliberate: a blend between two curves pointing the same way has an
//! inflection, and a C would be drawing [`fillet`](super::fillet) again.

use super::{Ink, Mark};

/// Where the blend meets the left stub, and the right one.
const LEFT: (f32, f32) = (2.0, 10.0);
const RIGHT: (f32, f32) = (16.0, 6.0);

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(2.0, 14.0), LEFT],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(16.0, 2.0), RIGHT],
        ink: Ink::SOLID,
    },
    // Controls held on each stub's line: vertical off both ends, so both joins are tangent.
    Mark::Cubic {
        p0: LEFT,
        p1: (2.0, 4.0),
        p2: (16.0, 12.0),
        p3: RIGHT,
        ink: Ink::ACCENT,
    },
];
