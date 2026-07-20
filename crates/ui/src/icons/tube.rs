//! `tube` — the `cylinder` mark, opened by a bore.
//!
//! Built on `cylinder` deliberately: same cap radii, same wall x, same near-half base sweep.
//! A tube is a cylinder with a hole, and the two rail marks sit next to each other in the
//! shape set, so the only thing that should differ between them is the thing that differs.
//!
//! The bore is a plain ellipse rather than the tile twin's faded inner walls. At 15 pt those
//! walls are two pixels of hairline inside another two pixels of hairline and close to mush;
//! the cap ring alone survives, and the walls plus the base sweep have already established
//! the solid of revolution, so no iris forms from the concentric pair.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The top cap's outer rim (the `cylinder` mark's cap, unchanged).
    Mark::Ellipse {
        center: (9.0, 4.6),
        rx: 5.5,
        ry: 2.2,
        ink: Ink::SOLID,
    },
    // The bore's mouth.
    Mark::Ellipse {
        center: (9.0, 4.6),
        rx: 2.5,
        ry: 1.0,
        ink: Ink::SOLID,
    },
    // The outer walls.
    Mark::Line {
        points: &[(3.5, 4.6), (3.5, 13.4)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(14.5, 4.6), (14.5, 13.4)],
        ink: Ink::SOLID,
    },
    // The near half of the base, sweeping under — the far half is hidden by the body.
    Mark::Arc {
        center: (9.0, 13.4),
        rx: 5.5,
        ry: 2.2,
        from: std::f32::consts::PI,
        to: 0.0,
        ink: Ink::SOLID,
    },
];
