//! `sculpt` (tile) — a brush core inside the ring it reaches to.
//!
//! Solid core, dashed outer ring: the radius is a Measurement and the ring is where it
//! stops. The dash is felt as a boundary rather than drawn as one, which is the right claim
//! for a brush — and at tile size the dash rhythm reads as a boundary instead of as a
//! broken circle.
//!
//! The ring follows the SET's dash rhythm (2.2 on, 1.8 off) rather than the c-palette
//! original's airier 2/3. A second dash rhythm inside one family reads as an inconsistency,
//! not as a distinction — the ring's meaning is carried by being dashed at all.

use crate::icons::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The core the brush actually writes.
    Mark::Circle {
        center: (13.0, 13.0),
        radius: 6.0,
        ink: Ink::SOLID,
    },
    // How far it reaches.
    Mark::Ellipse {
        center: (13.0, 13.0),
        rx: 9.5,
        ry: 9.5,
        ink: Ink::faint_dashed(0.4),
    },
];
