//! `carve` — the same dashed brush disc as `sculpt-add`, with only the horizontal bar.
//!
//! Deliberately one stroke different from its additive twin: they are the same tool under a
//! different fold, and a reader should see them as a pair.
//!
//! The nine arcs are written out rather than looped. A glyph is data, so the repetition is
//! visible instead of generated — which is the point: there is no control flow here to get
//! wrong. `STEP * 0.55` is the on-fraction of each dash, so the gaps are the remaining 0.45.

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
    // The subtractive bar — the one stroke that separates this from `sculpt-add`.
    Mark::Line {
        points: &[(5.5, 9.0), (12.5, 9.0)],
        ink: Ink::SOLID,
    },
];
