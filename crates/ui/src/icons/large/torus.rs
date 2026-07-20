//! `torus` (tile) — a ring solid in three-quarter view, its bore's far rim receding.
//!
//! The whole design problem here is that a closed curve drawn concentrically inside another
//! closed curve reads as an EYE, at every ratio — the set already learned this the expensive
//! way on `orbit`, across eight variants tuned by arithmetic that were all irises. The fix is
//! never to adjust the radii; it is to change the construction so the offending silhouette
//! never forms. Two things do that here:
//!
//! * **The bore is displaced upward**, not concentric. A real ring seen from above-front
//!   projects its hole above the silhouette's centre, so the gap above the bore is about half
//!   the gap below it. That asymmetry is what an iris never has.
//! * **The bore's far half is faded** while its near half is solid, so the mark states a
//!   direction of view. An eye is symmetric front-on; this cannot be.
//!
//! The fade is the tile set's x-ray idiom (see `box` and `cylinder`): a transparent
//! construction seen through, rather than an opaque object.

use crate::icons::{Ink, Mark};

/// The bore's far rim, dropped back — the same 0.5 the tile set's other hidden edges use.
const FAR: Ink = Ink::faint(0.5);

pub(super) const DRAW: &[Mark] = &[
    // The ring's outer silhouette.
    Mark::Ellipse {
        center: (13.0, 13.4),
        rx: 10.5,
        ry: 6.0,
        ink: Ink::SOLID,
    },
    // The bore, near half — solid, and sitting ABOVE the silhouette's centre.
    Mark::Arc {
        center: (13.0, 12.0),
        rx: 4.3,
        ry: 2.1,
        from: 0.0,
        to: std::f32::consts::PI,
        ink: Ink::SOLID,
    },
    // The bore, far half — seen through the ring, so it recedes.
    Mark::Arc {
        center: (13.0, 12.0),
        rx: 4.3,
        ry: 2.1,
        from: std::f32::consts::PI,
        to: std::f32::consts::TAU,
        ink: FAR,
    },
    // The tube's far inner wall, glimpsed through the bore: a shallow arc between the bore
    // and the silhouette's top. Without it the ring reads flat, like a washer.
    Mark::Arc {
        center: (13.0, 10.2),
        rx: 7.2,
        ry: 3.0,
        from: std::f32::consts::PI,
        to: std::f32::consts::TAU,
        ink: FAR,
    },
];
