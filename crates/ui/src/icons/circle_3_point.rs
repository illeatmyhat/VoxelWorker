//! `circle-3-point` — three points the ring passes through.
//!
//! Three accented nodes ON the ring and no chord at all. The absent chord is the distinction from
//! [`circle_2_point`](super::circle_2_point): draw one here and the mark would claim a diameter
//! relation that three arbitrary points do not have.
//!
//! The three sit at 90° apart rather than 120° so none of them lands on the ring's leftmost point,
//! where a node would read as a flat spot at 16 px.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Circle {
        center: (9.0, 9.0),
        radius: 6.0,
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: (9.0, 3.0),
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: (15.0, 9.0),
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: (9.0, 15.0),
        size: 2.6,
        ink: Ink::ACCENT,
    },
];
