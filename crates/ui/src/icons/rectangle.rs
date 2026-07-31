//! `rectangle` — drag a box into a four-point profile, the box-drag sugar inside the mode.
//!
//! The two corner nodes are the drag diagonal — the grab corner and the cursor — so the glyph
//! reads as "drag from here to here" rather than a static box. ADR 0028 slice 3.
//!
//! Re-drawn from the sheet (owner, 2026-07-30), and the re-draw INVERTS which part carries the
//! accent. The old glyph accented nothing and drew all three marks in line art; the sheet lights
//! the BOX and leaves the corners white, because on the create shelf the accent is what the tool
//! makes and the corners are what it asks you for. That reading is what makes the rectangle
//! family legible as a family: `rectangle`, `rectangle-3-point` and `rectangle-center-corner`
//! differ only in where the white nodes sit, over the same accented box.

use super::{Ink, Mark};

/// A drag corner. The sheet's 2.6 — a step above a profile vertex, because these are picks.
const CORNER: f32 = 2.6;

pub(super) const DRAW: &[Mark] = &[
    // The box the drag encloses — what the tool produces.
    Mark::Rect {
        a: (3.0, 2.5),
        b: (15.0, 13.5),
        ink: Ink::ACCENT,
    },
    // The drag diagonal's two corners — what the tool asks for.
    Mark::Node {
        center: (3.0, 2.5),
        size: CORNER,
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: (15.0, 13.5),
        size: CORNER,
        ink: Ink::SOLID,
    },
];
