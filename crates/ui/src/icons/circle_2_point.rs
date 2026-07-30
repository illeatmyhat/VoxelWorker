//! `circle-2-point` — two points that become the ENDS OF A DIAMETER.
//!
//! The chord through both is drawn, and it passes through the centre: that is what says "these two
//! are a diameter" rather than "these two are on the ring", which would be a three-point circle
//! missing its third click. No centre node, because the centre is derived and never clicked.

use super::{Ink, Mark};

/// The two clicks, diametrically opposite on the shared ring.
const FIRST: (f32, f32) = (4.7573, 13.2426);
const SECOND: (f32, f32) = (13.2426, 4.7573);

pub(super) const DRAW: &[Mark] = &[
    Mark::Circle {
        center: (9.0, 9.0),
        radius: 6.0,
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[FIRST, SECOND],
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: FIRST,
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: SECOND,
        size: 2.6,
        ink: Ink::ACCENT,
    },
];
