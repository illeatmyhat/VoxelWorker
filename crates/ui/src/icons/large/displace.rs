//! `displace` (tile) — a surface pushed by a field, over the datum it left.
//!
//! The datum is a faded rule under the curve: what the surface would be if nothing pushed
//! it. That is the reading the rail twin cannot hold, where a zigzag over a dashed rule
//! says "bumpy" rather than "moved".
//!
//! The curve is the c-palette original's two cubics, with its relative and smooth segments
//! (`c …` / `s …`) resolved to absolute control points — the second cubic's first control is
//! the reflection of the first's second control about the join, which is what keeps the
//! crest continuous rather than kinked.

use crate::icons::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The displaced surface: up over the crest, then down past it.
    Mark::Cubic {
        p0: (4.0, 18.0),
        p1: (8.0, 17.0),
        p2: (8.0, 11.0),
        p3: (12.0, 11.0),
        ink: Ink::SOLID,
    },
    Mark::Cubic {
        p0: (12.0, 11.0),
        p1: (16.0, 11.0),
        p2: (16.0, 17.0),
        p3: (22.0, 15.0),
        ink: Ink::SOLID,
    },
    // The undisturbed datum.
    Mark::Line {
        points: &[(4.0, 22.0), (22.0, 22.0)],
        ink: Ink::faint(0.4),
    },
];
