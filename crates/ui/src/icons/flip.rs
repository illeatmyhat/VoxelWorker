//! `flip` — two carets facing away from a dashed mirror axis.
//!
//! The axis is dashed for the same reason as in `revolve`: it is a datum the user picks, not an
//! edge of any body.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(9.0, 1.5), (9.0, 16.5)],
        ink: Ink::DASHED,
    },
    Mark::Line {
        points: &[(6.5, 5.0), (2.5, 9.0), (6.5, 13.0)],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(11.5, 5.0), (15.5, 9.0), (11.5, 13.0)],
        ink: Ink::SOLID,
    },
];
