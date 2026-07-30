//! `rectangular-pattern` — copies laid out along two directions.
//!
//! Line art is the SEED, the accent are the copies it generates, and the two floating bars are the
//! pitch it steps by — one per direction. Same line-art-is-the-reference rule the constraints use,
//! so one reading carries across both families.

use super::{Ink, Mark};

/// The seed's corner, the step, and the square each instance is drawn as.
const SEED: (f32, f32) = (4.5, 4.5);
const PITCH: f32 = 9.0;
const SIZE: f32 = 4.0;

pub(super) const DRAW: &[Mark] = &[
    // The two pitch bars, floating between the seed and its neighbours.
    Mark::Line {
        points: &[(7.25, SEED.1), (10.75, SEED.1)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(SEED.0, 7.25), (SEED.0, 10.75)],
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: SEED,
        size: SIZE,
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: (SEED.0 + PITCH, SEED.1),
        size: SIZE,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: (SEED.0, SEED.1 + PITCH),
        size: SIZE,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: (SEED.0 + PITCH, SEED.1 + PITCH),
        size: SIZE,
        ink: Ink::ACCENT,
    },
];
