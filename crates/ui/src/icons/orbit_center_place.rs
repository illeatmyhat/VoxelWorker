//! `orbit-center-place` — the pivot gizmo itself, reduced to the icon box.
//!
//! Transpiled from `chrome/d-signal/orbit-center-icons.html` on the design project. The row and the
//! marker it puts on screen are deliberately the SAME mark: the menu says what will appear, and
//! what appears is what the menu said. Nothing else in the set is a portrait of a gizmo, and
//! nothing else needs to be — this is the one command whose whole result is a mark.
//!
//! The proportions are [`orbit_center`](crate::gizmos::orbit_center)'s, rescaled onto the set's
//! dominant 2.5–15.5 box: arms 2.75 → 6.5 out from center, ring at 4.25 so the arms cross it
//! rather than stopping at it, dot 1.4. The crossing is the whole reading — a ring with arms
//! butted against it is a wheel, and a ring the arms pass through is a sight.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Circle {
        center: (9.0, 9.0),
        radius: 4.25,
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(9.0, 2.5), (9.0, 6.25)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(9.0, 11.75), (9.0, 15.5)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(2.5, 9.0), (6.25, 9.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(11.75, 9.0), (15.5, 9.0)],
        ink: Ink::SOLID,
    },
    Mark::Disc {
        center: (9.0, 9.0),
        radius: 1.4,
        ink: Ink::SOLID,
    },
];
