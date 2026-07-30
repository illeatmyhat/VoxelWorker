//! `sketch-dimension` — a distance that DRIVES the geometry.
//!
//! A measured entity above, its two extension lines dropping, and the dimension line with arrows
//! between them. The accent is on the dimension line rather than on the geometry, because the
//! dimension is the thing being authored — the geometry is what it moves.
//!
//! Drawn as a full dimension apparatus and not as a bare arrow: this is a stored, solver-visible
//! Measurement (ADR 0029), not a readout, and [`measure`](super::measure) is the readout.

use super::{Ink, Mark};

/// The two points the dimension spans.
const LEFT: f32 = 2.5;
const RIGHT: f32 = 15.5;

/// Where the dimension line itself runs, clear of the geometry it drives.
const LINE: f32 = 4.5;

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(LEFT, 11.5), (RIGHT, 11.5)],
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: (LEFT, 11.5),
        size: 2.6,
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: (RIGHT, 11.5),
        size: 2.6,
        ink: Ink::SOLID,
    },
    // Extension lines, gapped off the geometry as a drawing would gap them.
    Mark::Line {
        points: &[(LEFT, 9.5), (LEFT, 3.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(RIGHT, 9.5), (RIGHT, 3.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(5.75, LINE), (12.25, LINE)],
        ink: Ink::ACCENT,
    },
    Mark::Closed {
        points: &[(LEFT, LINE), (5.75, 3.3), (5.75, 5.7)],
        ink: Ink::ACCENT,
    },
    Mark::Closed {
        points: &[(RIGHT, LINE), (12.25, 5.7), (12.25, 3.3)],
        ink: Ink::ACCENT,
    },
];
