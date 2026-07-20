//! `subtract` — a body with a bite taken out of it, the operand left dashed over the gap.
//!
//! The operand stays drawn because subtract is an occupancy-only mask: the thing that was
//! removed is a shape the user authored, not an absence.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // What survives the fold.
    Mark::Closed {
        points: &[
            (2.5, 2.5),
            (10.5, 2.5),
            (10.5, 7.5),
            (7.5, 7.5),
            (7.5, 10.5),
            (2.5, 10.5),
        ],
        ink: Ink::SOLID,
    },
    // The cutter.
    Mark::Rect {
        a: (7.5, 7.5),
        b: (15.5, 15.5),
        ink: Ink::DASHED,
    },
];
