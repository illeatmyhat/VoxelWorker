//! `line` — the Line tool: click for a segment, drag from the end for a tangent arc.
//!
//! The arc is not decoration. Line makes lines AND tangent arcs, so a glyph that drew only a
//! segment would name half the tool; the seam between them is the whole point, and it is a real
//! tangency — the arc's centre sits perpendicular to the segment at the junction, so the two
//! meet with no kink at any size.
//!
//! Drawn at 45° and bending through only 45° more, so the arc LEVELS OFF rather than hooking
//! back on itself. That asymmetry is what keeps it from reading as an arch, which is what the
//! conic and the three-point arc already are.

use super::{Ink, Mark};

const PI: f32 = std::f32::consts::PI;

/// The seam: where the straight run hands over to the arc.
const SEAM: (f32, f32) = (10.25, 6.75);

pub(super) const DRAW: &[Mark] = &[
    // The straight run, rising at 45°.
    Mark::Line {
        points: &[(3.75, 13.25), SEAM],
        ink: Ink::SOLID,
    },
    // The tangent arc. Centre is 6 units perpendicular to the run at the seam, so the sweep
    // starts travelling in exactly the run's direction.
    Mark::Arc {
        center: (14.4926, 10.9926),
        rx: 6.0,
        ry: 6.0,
        from: -3.0 * PI / 4.0,
        to: -PI / 2.0,
        ink: Ink::SOLID,
    },
    // The three picks: start, seam, end.
    Mark::Node {
        center: (3.75, 13.25),
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: SEAM,
        size: 2.6,
        ink: Ink::ACCENT,
    },
    Mark::Node {
        center: (14.4926, 4.9926),
        size: 2.6,
        ink: Ink::ACCENT,
    },
];
