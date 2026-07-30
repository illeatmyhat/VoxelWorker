//! `constraint-curvature` — curvature matches across a joint (G2).
//!
//! A COMB, which is what Alias, Rhino and Onshape all use. Three properties make it read as
//! curvature rather than as hatching:
//!
//! 1. the hairs are normal to the curve, on its convex side;
//! 2. their lengths are proportional to actual κ, so the comb peaks where the curve is tightest;
//! 3. the curve has no inflection, so the comb stays on one side.
//!
//! Drawn vertical, or over an inflection, a comb says the opposite of what it means.

use super::{Ink, Mark};

/// The joint the constraint acts at — the middle hair's root.
const JOINT: (f32, f32) = (7.75, 7.5);

pub(super) const DRAW: &[Mark] = &[
    Mark::Cubic {
        p0: (3.0, 15.0),
        p1: (5.0, 7.6667),
        p2: (9.3333, 5.0),
        p3: (16.0, 7.0),
        ink: Ink::ACCENT,
    },
    Mark::Line {
        points: &[(4.94, 10.375), (3.06, 9.185)],
        ink: Ink::SOLID,
    },
    // The longest hair, at the tightest point: κ is highest here.
    Mark::Line {
        points: &[JOINT, (5.915, 4.52)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(11.44, 6.375), (11.27, 3.62)],
        ink: Ink::SOLID,
    },
    Mark::Node {
        center: JOINT,
        size: 2.2,
        ink: Ink::SOLID,
    },
];
