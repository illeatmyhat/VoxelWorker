//! `line` — the Line tool: click for a segment, drag from the end for a tangent arc.
//!
//! The arc is not decoration. Line makes lines AND tangent arcs, so a glyph that drew only a
//! segment would name half the tool; the seam between them is the whole point, and it is a real
//! tangency — the arc's centre sits perpendicular to the segment at the junction, so the two
//! meet with no kink at any size.
//!
//! The run enters at 45° and the arc then carries through 240° — two thirds of a circle. The
//! 120° it leaves open is load-bearing: closed, this would be the Circle mark, and the gap is
//! what keeps the two apart at 16 px. It is also well past Fusion's semicircle, so the shape
//! reads as a hook rather than as their candy cane.

use super::{Ink, Mark};

const PI: f32 = std::f32::consts::PI;

/// The seam: where the straight run hands over to the arc.
const SEAM: (f32, f32) = (5.6918, 3.8638);

pub(super) const DRAW: &[Mark] = &[
    // The straight run, rising at 45°.
    Mark::Line {
        points: &[(2.5098, 7.0458), SEAM],
        ink: Ink::SOLID,
    },
    // The tangent arc. Centre is r perpendicular to the run at the seam, so the sweep starts
    // travelling in exactly the run's direction; -135° through to 105° is the 240°.
    Mark::Arc {
        center: (10.288, 8.46),
        rx: 6.5,
        ry: 6.5,
        from: -3.0 * PI / 4.0,
        to: 7.0 * PI / 12.0,
        ink: Ink::SOLID,
    },
    // The three picks: start, seam, end.
    Mark::Node {
        center: (2.5098, 7.0458),
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: SEAM,
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: (8.6057, 14.7385),
        size: 2.6,
        ink: Ink::ACCENT,
    },
];
