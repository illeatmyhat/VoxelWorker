//! `sketch` — a closed quadrilateral profile with a handle square at each vertex.
//!
//! The profile is drawn irregular, not as a rectangle: the sketch is the authoring atom and
//! organic outlines are its point, while a rectangle would read as the box primitive that is
//! merely sugar over it.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The flattened polygon.
    Mark::Closed {
        points: &[(4.0, 12.5), (6.5, 4.5), (14.0, 6.5), (11.5, 14.5)],
        ink: Ink::SOLID,
    },
    // The authored vertices.
    Mark::Rect {
        a: (3.2, 11.7),
        b: (4.8, 13.3),
        ink: Ink::SOLID,
    },
    Mark::Rect {
        a: (5.7, 3.7),
        b: (7.3, 5.3),
        ink: Ink::SOLID,
    },
    Mark::Rect {
        a: (13.2, 5.7),
        b: (14.8, 7.3),
        ink: Ink::SOLID,
    },
    Mark::Rect {
        a: (10.7, 13.7),
        b: (12.3, 15.3),
        ink: Ink::SOLID,
    },
];
