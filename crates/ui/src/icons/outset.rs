//! `outset` — a solid body with the dilated envelope dashed around it, and two growth ticks.
//!
//! The ticks run outward from the body's corners to the envelope's. Without them the nested
//! squares are ambiguous — they could equally read as a part containing a child — and the
//! direction of the field is the whole content of the mark.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The authored body.
    Mark::Rect {
        a: (6.0, 6.0),
        b: (12.0, 12.0),
        ink: Ink::SOLID,
    },
    // The dilated envelope.
    Mark::Rect {
        a: (2.5, 2.5),
        b: (15.5, 15.5),
        ink: Ink::DASHED,
    },
    // Which way the field moves.
    Mark::Line {
        points: &[(12.0, 6.0), (15.5, 2.5)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(6.0, 12.0), (2.5, 15.5)],
        ink: Ink::SOLID,
    },
];
