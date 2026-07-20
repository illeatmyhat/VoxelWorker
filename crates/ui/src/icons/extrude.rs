//! `extrude` — a dashed profile lifted to a solid one, with the travel arrow beside it.
//!
//! The dashed rectangle is the sketch and the solid one the swept result, so the mark shows
//! the transformation rather than just the outcome.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The resulting body.
    Mark::Rect {
        a: (2.5, 11.0),
        b: (9.5, 14.5),
        ink: Ink::SOLID,
    },
    // The profile it came from.
    Mark::Rect {
        a: (2.5, 3.5),
        b: (9.5, 7.0),
        ink: Ink::DASHED,
    },
    // The lift.
    Mark::Line {
        points: &[(13.5, 14.5), (13.5, 4.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(11.9, 5.6), (13.5, 4.0), (15.1, 5.6)],
        ink: Ink::SOLID,
    },
];
