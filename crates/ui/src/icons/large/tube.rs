//! `tube` (tile) — a lathe body with a bore: the cylinder, opened.
//!
//! Deliberately built ON the `cylinder` tile — same cap radii, same wall x, same faded
//! bottom cap — because a tube IS a cylinder with a hole and the two marks should read as
//! neighbours rather than as unrelated drawings. Everything added here states the bore.
//!
//! Unlike `torus`, the concentric cap ring is safe: the walls and the faded bottom cap
//! already fix the reading as a solid of revolution before the eye reaches the inner curve,
//! so no iris forms. The bore's near walls drop only a short way before fading out — enough
//! to say "this goes down" without drawing a second full cylinder inside the first.

use crate::icons::{Ink, Mark};

/// Bore geometry seen through the top opening — present, but never competing with the body.
const BORE: Ink = Ink::faint(0.45);

pub(super) const DRAW: &[Mark] = &[
    // The top cap's outer rim (the `cylinder` tile's cap, unchanged).
    Mark::Ellipse {
        center: (13.0, 7.0),
        rx: 8.0,
        ry: 3.2,
        ink: Ink::SOLID,
    },
    // The bore's mouth.
    Mark::Ellipse {
        center: (13.0, 7.0),
        rx: 3.4,
        ry: 1.4,
        ink: Ink::SOLID,
    },
    // The outer walls.
    Mark::Line {
        points: &[(5.0, 7.0), (5.0, 19.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(21.0, 7.0), (21.0, 19.0)],
        ink: Ink::SOLID,
    },
    // The bore's own walls, dropping into the body and fading as they go out of sight.
    Mark::Line {
        points: &[(9.6, 7.0), (9.6, 11.4)],
        ink: BORE,
    },
    Mark::Line {
        points: &[(16.4, 7.0), (16.4, 11.4)],
        ink: BORE,
    },
    // The bottom cap, receding — the x-ray reading the `cylinder` tile established.
    Mark::Ellipse {
        center: (13.0, 19.0),
        rx: 8.0,
        ry: 3.2,
        ink: Ink::faint(0.5),
    },
];
