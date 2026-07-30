//! `circle-3-tangent` — the incircle of three picked curves.
//!
//! Three tangencies use every freedom, so unlike
//! [`circle_2_tangent`](super::circle_2_tangent) there is no radius left to type — and the glyph
//! drops the radius line to say so. Losing that one stroke is the whole difference between the two
//! marks, which is why neither may gain a decoration the other lacks.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(1.0, 15.0), (17.0, 15.0)],
        ink: Ink::ACCENT,
    },
    Mark::Line {
        points: &[(2.0, 3.5), (2.0, 16.0)],
        ink: Ink::ACCENT,
    },
    Mark::Line {
        points: &[(1.2, 3.9), (16.8, 15.6)],
        ink: Ink::ACCENT,
    },
    Mark::Circle {
        center: (5.5, 11.5),
        radius: 3.5,
        ink: Ink::SOLID,
    },
];
