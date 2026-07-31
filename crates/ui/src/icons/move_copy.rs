//! `move-copy` — the free transform handle.
//!
//! Four arrows on two axes through one center. It draws the HANDLE rather than a moved shape,
//! because the tool works on whatever is selected and a glyph that showed a specific shape would
//! be naming a thing the tool has no opinion about.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(2.0, 8.0), (16.0, 8.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(9.0, 1.0), (9.0, 15.0)],
        ink: Ink::SOLID,
    },
    Mark::Closed {
        points: &[(16.5, 8.0), (13.25, 9.2), (13.25, 6.8)],
        ink: Ink::SOLID,
    },
    Mark::Closed {
        points: &[(1.5, 8.0), (4.75, 6.8), (4.75, 9.2)],
        ink: Ink::SOLID,
    },
    Mark::Closed {
        points: &[(9.0, 0.5), (10.2, 3.75), (7.8, 3.75)],
        ink: Ink::SOLID,
    },
    Mark::Closed {
        points: &[(9.0, 15.5), (7.8, 12.25), (10.2, 12.25)],
        ink: Ink::SOLID,
    },
];
