//! `arc-center-endpoints` — centre first, then the two ends.
//!
//! A quarter turn with its centre drawn and one radius shown. Three accented nodes, because all
//! three are clicked — this is the tool where the centre IS a pick, which is what separates it
//! from [`three_point_arc`](super::three_point_arc), where the centre is solved for.

use super::{Ink, Mark};

const PI: f32 = std::f32::consts::PI;

/// The centre, which here is a click and not a derived point.
const CENTRE: (f32, f32) = (5.0, 14.0);
const RADIUS: f32 = 10.0;

pub(super) const DRAW: &[Mark] = &[
    // From straight up round to the right: a quarter turn, y running down.
    Mark::Arc {
        center: CENTRE,
        rx: RADIUS,
        ry: RADIUS,
        from: -PI / 2.0,
        to: 0.0,
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[CENTRE, (15.0, 14.0)],
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: (5.0, 4.0),
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: (15.0, 14.0),
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: CENTRE,
        size: 2.6,
        ink: Ink::ACCENT,
    },
];
