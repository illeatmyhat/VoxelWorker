//! `polyline` — click to place connected profile points, the organic value prop.
//!
//! A run of segments with a square node at each joint: the profile is built vertex by vertex,
//! and the nodes sit on the drawn line to say "these are the points you placed". ADR 0028
//! slice 3.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The placed run.
    Mark::Line {
        points: &[(2.5, 14.0), (6.5, 5.5), (11.0, 10.0), (15.5, 3.0)],
        ink: Ink::SOLID,
    },
    // The joint nodes, one per placed vertex.
    Mark::Rect {
        a: (1.4, 12.9),
        b: (3.6, 15.1),
        ink: Ink::SOLID,
    },
    Mark::Rect {
        a: (5.4, 4.4),
        b: (7.6, 6.6),
        ink: Ink::SOLID,
    },
    Mark::Rect {
        a: (9.9, 8.9),
        b: (12.1, 11.1),
        ink: Ink::SOLID,
    },
    Mark::Rect {
        a: (14.4, 1.9),
        b: (16.6, 4.1),
        ink: Ink::SOLID,
    },
];
