//! `torus` — a ring solid, its bore displaced for a three-quarter view.
//!
//! The same construction as the tile twin, and for the same reason: a closed curve drawn
//! concentrically inside another reads as an EYE at every ratio, so the bore sits ABOVE the
//! silhouette's centre and its far half recedes. At rail size the risk is worse, not better —
//! a filled disc and a stroked ring are the same three pixels here, so the mark cannot lean
//! on that difference and has to carry the reading in its construction alone.
//!
//! The bore is kept generous relative to the body. A small bore at 15 pt closes to a dot and
//! the mark becomes a filled ellipse; a wide one stays legibly a ring.

use super::{Ink, Mark};

/// The bore's far rim, dropped back.
const FAR: Ink = Ink::faint(0.5);

pub(super) const DRAW: &[Mark] = &[
    // The ring's outer silhouette.
    Mark::Ellipse {
        center: (9.0, 9.6),
        rx: 7.0,
        ry: 4.0,
        ink: Ink::SOLID,
    },
    // The bore's near half, solid and set high.
    Mark::Arc {
        center: (9.0, 8.7),
        rx: 2.9,
        ry: 1.45,
        from: 0.0,
        to: std::f32::consts::PI,
        ink: Ink::SOLID,
    },
    // The bore's far half, seen through the ring.
    Mark::Arc {
        center: (9.0, 8.7),
        rx: 2.9,
        ry: 1.45,
        from: std::f32::consts::PI,
        to: std::f32::consts::TAU,
        ink: FAR,
    },
];
