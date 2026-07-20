//! `fit` — four corner brackets around a centre square: frame the subject.
//!
//! The brackets are open Ls rather than a closed frame, so the mark reads as *framing
//! something* rather than as a window or a crop region.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // Corner brackets, each two segments meeting at the corner.
    Mark::Line {
        points: &[(2.5, 6.0), (2.5, 2.5), (6.0, 2.5)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(12.0, 2.5), (15.5, 2.5), (15.5, 6.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(15.5, 12.0), (15.5, 15.5), (12.0, 15.5)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(6.0, 15.5), (2.5, 15.5), (2.5, 12.0)],
        ink: Ink::SOLID,
    },
    // The subject being framed.
    Mark::Rect {
        a: (6.5, 6.5),
        b: (11.5, 11.5),
        ink: Ink::SOLID,
    },
];
