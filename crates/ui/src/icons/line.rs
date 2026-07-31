//! `line` — the Line tool: click for a segment, drag from the end for a tangent arc.
//!
//! The arc is not decoration. Line makes lines AND tangent arcs, so a glyph that drew only a
//! segment would name half the tool; the seam between them is the whole point, and it is a real
//! tangency — the arc's center sits perpendicular to the segment at the junction, so the two
//! meet with no kink at any size.
//!
//! The run enters at 45° and ends in a 240° curl — two thirds of a circle, but a sixth of the
//! drawing, so the mark reads as a long line with a hook rather than as an arc with a tail. The
//! 120° left open is load-bearing: closed, this would be the Circle mark.
//!
//! [`RADIUS`] has a floor set by the squares, not by taste.

use super::{Ink, Mark};

const PI: f32 = std::f32::consts::PI;

/// The seam: where the straight run hands over to the arc.
const SEAM: (f32, f32) = (11.8785, 4.2115);

/// The curl's radius. A 240° sweep leaves the seam and the end `1.732 * RADIUS` apart, so at
/// a node size of 2.6 anything under about 2.5 merges the two squares into one lozenge and the
/// glyph quietly loses a pick. Held at 3 for margin.
const RADIUS: f32 = 3.0;

pub(super) const DRAW: &[Mark] = &[
    // The straight run, rising at 45°.
    Mark::Line {
        points: &[(2.3, 13.79), SEAM],
        ink: Ink::SOLID,
    },
    // The tangent arc. Center is RADIUS perpendicular to the run at the seam, so the sweep
    // starts traveling in exactly the run's direction; -135° through to 105° is the 240°.
    Mark::Arc {
        center: (13.9998, 6.3328),
        rx: RADIUS,
        ry: RADIUS,
        from: -3.0 * PI / 4.0,
        to: 7.0 * PI / 12.0,
        ink: Ink::SOLID,
    },
    // The three picks: start, seam, end.
    Mark::Node {
        center: (2.3, 13.79),
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: SEAM,
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: (13.2234, 9.2306),
        size: 2.6,
        ink: Ink::ACCENT,
    },
];
