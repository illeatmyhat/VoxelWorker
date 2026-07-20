//! `cylinder` (tile) — a lathe body: solid top cap, faded bottom cap.
//!
//! Both caps are whole ellipses, not a front-only arc. The fade does the occlusion, which
//! is what says "a transparent construction seen through" rather than "a can".

use crate::icons::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Ellipse {
        center: (13.0, 7.0),
        rx: 8.0,
        ry: 3.2,
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(5.0, 7.0), (5.0, 19.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(21.0, 7.0), (21.0, 19.0)],
        ink: Ink::SOLID,
    },
    Mark::Ellipse {
        center: (13.0, 19.0),
        rx: 8.0,
        ry: 3.2,
        ink: Ink::faint(0.5),
    },
];
