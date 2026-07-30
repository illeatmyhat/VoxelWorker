//! `trim` — deletes a curve back to its nearest crossing.
//!
//! One curve crosses another; the stub on the far side of the crossing carries the accent. The
//! accent is on **what goes**, not on what stays, which is the opposite of most of the set and is
//! the only way a static mark can name a deletion.
//!
//! The crossing is a real intersection of the two drawn lines, not a point placed near them:
//! `(8.335, 10)` is where the diagonal meets `y = 10`.

use super::{Ink, Mark};

/// Where the two curves cross, and so where the trim stops.
const CROSSING: (f32, f32) = (8.335, 10.0);

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(5.0, 15.0), (13.0, 3.0)],
        ink: Ink::SOLID,
    },
    // What survives.
    Mark::Line {
        points: &[CROSSING, (16.0, 10.0)],
        ink: Ink::SOLID,
    },
    // What the click removes.
    Mark::Line {
        points: &[(2.0, 10.0), CROSSING],
        ink: Ink::ACCENT,
    },
];
