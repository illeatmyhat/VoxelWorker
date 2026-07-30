//! `midpoint-line` — a segment placed by its CENTRE and one end.
//!
//! The two tick marks are what separate it from [`line`](super::line): they say the mark is about
//! a measured middle, not about a run. Both the centre and the end carry the accent because both
//! are clicked; the far end is derived and so stays line art.

use super::{Ink, Mark};

/// The centre the tool is anchored on, and the end the drag defines.
const CENTRE: (f32, f32) = (9.0, 9.0);
const END: (f32, f32) = (15.0, 15.0);

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(3.0, 3.0), END],
        ink: Ink::SOLID,
    },
    // Ticks at the quarter points: equal halves, stated rather than implied.
    Mark::Line {
        points: &[(4.9394, 7.0606), (7.0606, 4.9394)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(10.9393, 13.0607), (13.0607, 10.9393)],
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: (3.0, 3.0),
        size: 2.6,
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: CENTRE,
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: END,
        size: 2.6,
        ink: Ink::ACCENT,
    },
];
