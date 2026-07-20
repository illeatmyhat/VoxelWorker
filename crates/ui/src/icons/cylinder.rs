//! `cylinder` — a full top ellipse, two walls, and only the near half of the base.
//!
//! The base's far half is omitted rather than drawn: an opaque cylinder hides it, and drawing
//! it would turn the mark into a wireframe of something the display never shows.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Ellipse {
        center: (9.0, 4.6),
        rx: 5.5,
        ry: 2.2,
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(3.5, 4.6), (3.5, 13.4)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(14.5, 4.6), (14.5, 13.4)],
        ink: Ink::SOLID,
    },
    // The near half of the base, sweeping under.
    Mark::Arc {
        center: (9.0, 13.4),
        rx: 5.5,
        ry: 2.2,
        from: std::f32::consts::PI,
        to: 0.0,
        ink: Ink::SOLID,
    },
];
