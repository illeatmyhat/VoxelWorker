//! `fold-cursor` — the stack with a caret in the gutter and the row past it dashed.
//!
//! The dashed row is dropped from this evaluation, not deleted, and dashing rather than
//! omitting it is what carries that distinction.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Rect {
        a: (4.0, 2.5),
        b: (16.0, 5.5),
        ink: Ink::SOLID,
    },
    Mark::Rect {
        a: (4.0, 7.5),
        b: (16.0, 10.5),
        ink: Ink::SOLID,
    },
    // Past the cursor: still authored, not evaluated.
    Mark::Rect {
        a: (4.0, 12.5),
        b: (16.0, 15.5),
        ink: Ink::DASHED,
    },
    // The insert caret.
    Mark::Closed {
        points: &[(0.5, 9.5), (3.0, 11.5), (0.5, 13.5)],
        ink: Ink::SOLID,
    },
];
