//! `revolve` (tile) — a profile, an axis, and the arc that carries one onto the other.
//!
//! Shares the rail mark's construction (profile rect + axis + swept arc) rather than the
//! c-palette original's equator ellipse, so the two sizes read as the same verb. What the
//! tile adds is room for the profile to be a real closed body instead of a hint.

use crate::icons::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The axis, subordinate to what spins about it.
    Mark::Line {
        points: &[(13.0, 2.0), (13.0, 24.0)],
        ink: Ink::faint(0.5),
    },
    // The sweep.
    Mark::Cubic {
        p0: (13.0, 5.0),
        p1: (21.0, 8.0),
        p2: (21.0, 18.0),
        p3: (13.0, 21.0),
        ink: Ink::SOLID,
    },
    // The profile being revolved.
    Mark::Rect {
        a: (5.0, 7.0),
        b: (9.5, 19.0),
        ink: Ink::SOLID,
    },
];
