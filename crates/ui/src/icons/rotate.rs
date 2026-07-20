//! `rotate` — a near-closed ring broken at the top right, with a corner tick in the gap.
//!
//! The tick is a right angle rather than a curved arrowhead: rotation on the lattice comes in
//! quarter turns only, and a smooth arrow would promise arbitrary angles the lattice cannot
//! represent without resampling.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The SVG's large arc, resolved onto a centred r-6 ring (its endpoints sit a little off
    // that radius in the source; the ring is regularised so it does not wobble when scaled).
    Mark::Arc {
        center: (9.0, 9.0),
        rx: 6.0,
        ry: 6.0,
        from: -0.507,
        to: -6.038,
        ink: Ink::SOLID,
    },
    // The quarter-turn corner.
    Mark::Line {
        points: &[(13.5, 2.5), (13.5, 7.0), (9.0, 7.0)],
        ink: Ink::SOLID,
    },
];
