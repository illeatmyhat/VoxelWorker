//! `circular-pattern` — copies swept about an axis.
//!
//! Five elements at 72°, seed in line art and copies in the accent, matching
//! [`rectangular_pattern`](super::rectangular_pattern). The cross IS the axis — there is no path
//! circle, because the ring of elements already describes the path and drawing it twice only costs
//! legibility at 16 px.
//!
//! Discs rather than squares: these are instances, not authored vertices.

use super::{Ink, Mark};

/// The axis the copies turn about, and the radius they sit at.
const AXIS: (f32, f32) = (9.0, 9.0);
const RING: f32 = 6.5;

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(6.5, AXIS.1), (11.5, AXIS.1)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(AXIS.0, 6.5), (AXIS.0, 11.5)],
        ink: Ink::SOLID,
    },
    Mark::Disc {
        center: (AXIS.0, AXIS.1 - RING),
        radius: 1.3,
        ink: Ink::SOLID,
    },
    Mark::Disc {
        center: (15.18, 6.99),
        radius: 1.3,
        ink: Ink::ACCENT,
    },
    Mark::Disc {
        center: (12.82, 14.26),
        radius: 1.3,
        ink: Ink::ACCENT,
    },
    Mark::Disc {
        center: (5.18, 14.26),
        radius: 1.3,
        ink: Ink::ACCENT,
    },
    Mark::Disc {
        center: (2.82, 6.99),
        radius: 1.3,
        ink: Ink::ACCENT,
    },
];
