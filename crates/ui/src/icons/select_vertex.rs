//! `select-vertex` — the default sketch arrow, carrying a small node at its tip.
//!
//! The arrow says "pick"; the node at the tip says "a profile vertex", which is what
//! distinguishes it from the 3D move gizmo — this grabs a POINT on the plane, not a whole
//! node. ADR 0028 slice 1, the mode's default tool.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The pointer body.
    Mark::Closed {
        points: &[
            (3.5, 2.5),
            (3.5, 13.0),
            (6.5, 10.2),
            (8.9, 15.0),
            (10.7, 14.1),
            (8.3, 9.4),
            (12.5, 9.1),
        ],
        ink: Ink::SOLID,
    },
    // The vertex node at the tip.
    Mark::Rect {
        a: (12.2, 12.2),
        b: (14.6, 14.6),
        ink: Ink::SOLID,
    },
];
