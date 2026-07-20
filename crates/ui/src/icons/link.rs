//! `link` — a definition and an instance of it, tied.
//!
//! The asset drawer places LINKED instances, not copies: edit the definition and every
//! instance follows. That is the whole reason make-unique has to be a deliberate act, so
//! the mark has to show a reference relationship rather than a duplicate.
//!
//! The definition is solid and the instance dashed — the set's established convention
//! that dashed means "the referenced thing", the same sense it carries on `array` and on
//! every operand mark. The tie between them is what separates this from `array`: two
//! bodies in a reference relationship, not three repeats of one.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The definition: the body that is actually authored.
    Mark::Rect {
        a: (1.5, 6.5),
        b: (7.0, 11.5),
        ink: Ink::SOLID,
    },
    // An instance of it — a reference, carrying no body of its own.
    Mark::Rect {
        a: (11.0, 6.5),
        b: (16.5, 11.5),
        ink: Ink::DASHED,
    },
    // The link itself.
    Mark::Line {
        points: &[(7.0, 9.0), (11.0, 9.0)],
        ink: Ink::SOLID,
    },
];
