//! `three-point-arc` — the arc tool: two endpoints and a point the curve passes through.
//!
//! The arc drawn is the one that ACTUALLY passes through its three marks: the center and radius
//! below are the circumcircle of the three nodes, not a convenient curve with points laid near
//! it — the difference between a glyph that depicts the tool and one that depicts its result.
//!
//! All three picks are accented SQUARES, on the create shelf's rule that the accent names what
//! the tool will ask you for. The set's wider convention reserves a square for an authored
//! vertex and a disc for a pick consumed at creation, and the through-point is consumed — this
//! glyph takes the shelf's reading over that one deliberately.

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
