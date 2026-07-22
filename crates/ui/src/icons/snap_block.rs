//! `snap-block` — the vertex snaps to block boundaries, for clean inter-part mating.
//!
//! The node sits on the corner of one coarse cell: quantised to a multiple of density, the
//! coarsest of the three position snaps. The 2D reuse of ADR 0027's snap (ADR 0028 §5).

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The coarse block cell.
    Mark::Rect {
        a: (4.0, 4.0),
        b: (15.0, 15.0),
        ink: Ink::SOLID,
    },
    // The vertex, locked on the cell corner.
    Mark::Rect {
        a: (2.6, 2.6),
        b: (5.4, 5.4),
        ink: Ink::SOLID,
    },
];
