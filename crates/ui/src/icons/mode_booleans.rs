//! `mode-booleans` — a solid body with a dashed operand overlapping it.
//!
//! Dashed is the set's mark for an operand, so the pairing states the mode exactly: the
//! operands stay visible as x-ray ghosts *alongside* the folded result, rather than being
//! consumed by it.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The folded result.
    Mark::Rect {
        a: (2.5, 2.5),
        b: (10.5, 10.5),
        ink: Ink::SOLID,
    },
    // The operand ghost.
    Mark::Rect {
        a: (7.5, 7.5),
        b: (15.5, 15.5),
        ink: Ink::DASHED,
    },
];
