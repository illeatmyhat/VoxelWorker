//! `density` — one block ruled into a 3×3 of voxels.
//!
//! The outer square never changes size between states of this mark: density is voxels per
//! block, fineness and never extent, and only the interior ruling would differ.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The block.
    Mark::Rect {
        a: (2.5, 2.5),
        b: (15.5, 15.5),
        ink: Ink::SOLID,
    },
    // Its voxels.
    Mark::Line {
        points: &[(6.83, 2.5), (6.83, 15.5)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(11.17, 2.5), (11.17, 15.5)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(2.5, 6.83), (15.5, 6.83)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(2.5, 11.17), (15.5, 11.17)],
        ink: Ink::SOLID,
    },
];
