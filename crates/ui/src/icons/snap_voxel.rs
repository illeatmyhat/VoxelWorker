//! `snap-voxel` — the vertex snaps to the fine lattice crossing. The default.
//!
//! The node sits ON the crossing of the two in-plane axes: whole-voxel quantisation, the
//! sketch's default position snap. The 2D reuse of ADR 0027's snap (ADR 0028 §5).

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The in-plane axes.
    Mark::Line {
        points: &[(9.0, 3.0), (9.0, 15.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(3.0, 9.0), (15.0, 9.0)],
        ink: Ink::SOLID,
    },
    // The vertex, locked on the crossing.
    Mark::Rect {
        a: (7.4, 7.4),
        b: (10.6, 10.6),
        ink: Ink::SOLID,
    },
];
