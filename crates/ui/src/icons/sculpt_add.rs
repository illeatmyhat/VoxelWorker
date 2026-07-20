//! `sculpt-add` — a dashed brush disc with a plus at its centre.
//!
//! The disc is dashed because it is a brush *radius*, an authored Measurement, not the edge of
//! a body — the same reason a chisel or pencil was rejected for this mark.
//!
//! The brush radius is a circle dashed by drawing only part of each arc segment, since the
//! kit's dash helpers work on straight runs. The nine arcs are written out rather than looped:
//! a glyph is data, so the repetition is visible instead of generated. `carve` is the same
//! disc under a different fold and states it the same way.

use super::{Ink, Mark};

/// One ninth of the disc — the dash pitch.
const STEP: f32 = std::f32::consts::TAU / 9.0;

pub(super) const DRAW: &[Mark] = &[
    Mark::Arc {
        center: (9.0, 9.0),
        rx: 5.5,
        ry: 5.5,
        from: 0.0,
        to: STEP * 0.55,
        ink: Ink::SOLID,
    },
    Mark::Arc {
        center: (9.0, 9.0),
        rx: 5.5,
        ry: 5.5,
        from: STEP,
        to: STEP + STEP * 0.55,
        ink: Ink::SOLID,
    },
    Mark::Arc {
        center: (9.0, 9.0),
        rx: 5.5,
        ry: 5.5,
        from: STEP * 2.0,
        to: STEP * 2.0 + STEP * 0.55,
        ink: Ink::SOLID,
    },
    Mark::Arc {
        center: (9.0, 9.0),
        rx: 5.5,
        ry: 5.5,
        from: STEP * 3.0,
        to: STEP * 3.0 + STEP * 0.55,
        ink: Ink::SOLID,
    },
    Mark::Arc {
        center: (9.0, 9.0),
        rx: 5.5,
        ry: 5.5,
        from: STEP * 4.0,
        to: STEP * 4.0 + STEP * 0.55,
        ink: Ink::SOLID,
    },
    Mark::Arc {
        center: (9.0, 9.0),
        rx: 5.5,
        ry: 5.5,
        from: STEP * 5.0,
        to: STEP * 5.0 + STEP * 0.55,
        ink: Ink::SOLID,
    },
    Mark::Arc {
        center: (9.0, 9.0),
        rx: 5.5,
        ry: 5.5,
        from: STEP * 6.0,
        to: STEP * 6.0 + STEP * 0.55,
        ink: Ink::SOLID,
    },
    Mark::Arc {
        center: (9.0, 9.0),
        rx: 5.5,
        ry: 5.5,
        from: STEP * 7.0,
        to: STEP * 7.0 + STEP * 0.55,
        ink: Ink::SOLID,
    },
    Mark::Arc {
        center: (9.0, 9.0),
        rx: 5.5,
        ry: 5.5,
        from: STEP * 8.0,
        to: STEP * 8.0 + STEP * 0.55,
        ink: Ink::SOLID,
    },
    // The plus — the one pair of strokes that separates this from `carve`.
    Mark::Line {
        points: &[(9.0, 5.5), (9.0, 12.5)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(5.5, 9.0), (12.5, 9.0)],
        ink: Ink::SOLID,
    },
];
