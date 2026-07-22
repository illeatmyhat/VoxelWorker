//! `close-loop` — complete the profile by joining back to the start vertex.
//!
//! Three committed sides solid, the closing run dashed (uncommitted until clicked), and the
//! start node emphasised — the "click here to close" affordance, drawn in the family's
//! dashed-means-uncommitted idiom. ADR 0028 slice 3.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The three committed sides.
    Mark::Line {
        points: &[(4.5, 14.5), (4.5, 4.5), (13.5, 4.5), (13.5, 14.5)],
        ink: Ink::SOLID,
    },
    // The closing run — dashed until the click commits it.
    Mark::Line {
        points: &[(4.5, 14.5), (13.5, 14.5)],
        ink: Ink::DASHED,
    },
    // The start vertex the loop closes onto.
    Mark::Rect {
        a: (3.2, 3.2),
        b: (5.8, 5.8),
        ink: Ink::SOLID,
    },
];
