//! `three-point-arc` — the arc tool: two endpoints and a point the curve passes through.
//!
//! Re-drawn from the sheet (owner, 2026-07-30). The old glyph drew an arc bulging over a chord
//! with its three inputs in line art; this one draws the arc that ACTUALLY passes through its
//! three marks. The center and radius below are the circumcircle of the three nodes, not a
//! convenient curve with points laid near it — which is the difference between a glyph that
//! depicts the tool and one that depicts its result.
//!
//! **This departs from ADR 0030 §5**, which reserves a disc for a pick consumed at creation and
//! a square for an authored vertex: the through-point is consumed, so the old drawing gave it a
//! disc. The sheet makes all three accented squares, on the create shelf's own rule that the
//! accent names what the tool will ask you for. Both readings are defensible and the sheet's is
//! the consistent one within the shelf — but it is an ADR departure and is flagged as one.

use super::{Ink, Mark};

/// The three picks, and the circumcircle they define.
///
/// Center and radius are resolved from `(2.5, 13)`, `(15.5, 13)`, `(9, 4)`: the perpendicular
/// bisector of the chord is `x = 9`, and the other bisector puts the center at `y = 97.625/9`.
/// The angles are the two endpoints' bearings about that center, `to` carried past a full turn
/// so the sweep runs the long way, over the top, through the third point.
const CENTER: (f32, f32) = (9.0, 10.8473);
const RADIUS: f32 = 6.8472;
const FROM: f32 = 2.821776;
const TO: f32 = 6.603002;
const PICK: f32 = 2.6;

pub(super) const DRAW: &[Mark] = &[
    Mark::Arc {
        center: CENTER,
        rx: RADIUS,
        ry: RADIUS,
        from: FROM,
        to: TO,
        ink: Ink::SOLID,
    },
    // The two endpoints...
    Mark::Node {
        center: (2.5, 13.0),
        size: PICK,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: (15.5, 13.0),
        size: PICK,
        ink: Ink::ACCENT,
    },
    // ...and the point the curve is made to pass through.
    Mark::Node {
        center: (9.0, 4.0),
        size: PICK,
        ink: Ink::ACCENT,
    },
];
