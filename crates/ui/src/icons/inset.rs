//! `inset` — `outset` reversed: the authored body is the outer solid, the eroded result dashed.
//!
//! Which square is solid is the only difference, and it is the correct one: the solid square is
//! always the body the user authored, and the ticks always point the way the field moves.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Rect {
        a: (2.5, 2.5),
        b: (15.5, 15.5),
        ink: Ink::SOLID,
    },
    Mark::Rect {
        a: (6.0, 6.0),
        b: (12.0, 12.0),
        ink: Ink::DASHED,
    },
    Mark::Line {
        points: &[(15.5, 2.5), (12.0, 6.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(2.5, 15.5), (6.0, 12.0)],
        ink: Ink::SOLID,
    },
];
