//! `chamfer-distance-angle` — one distance and the angle the bevel leaves at.
//!
//! A sibling of [`chamfer_equal`](super::chamfer_equal), differing only in the bevel's slope: a
//! shallower run says the two legs were not cut back by the same amount.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(3.0, 2.0), (3.0, 5.5)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(11.0, 13.0), (15.5, 13.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(3.0, 6.0), (10.5, 13.0)],
        ink: Ink::ACCENT,
    },
];
