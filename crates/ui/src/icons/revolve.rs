//! `revolve` — a profile beside a dashed axis, with a sweep arrow curving around it.
//!
//! The axis is dashed because it is a datum and not an edge of the body; the arrow curves to
//! the far side so the mark measures ROUND where `extrude` measures square.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The axis of revolution.
    Mark::Line {
        points: &[(9.0, 1.5), (9.0, 16.5)],
        ink: Ink::DASHED,
    },
    // The profile.
    Mark::Rect {
        a: (3.0, 4.5),
        b: (6.5, 13.5),
        ink: Ink::SOLID,
    },
    // The sweep: the SVG arc from (11, 4.2) to (11, 13.8) at r 5.4, resolved to its centre.
    Mark::Arc {
        center: (8.526, 9.0),
        rx: 5.4,
        ry: 5.4,
        from: -1.096,
        to: 1.096,
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(9.4, 12.2), (11.0, 13.8), (9.4, 15.4)],
        ink: Ink::SOLID,
    },
];
