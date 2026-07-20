//! `emboss` — a surface profile stepping up between two dashed footprint walls.
//!
//! This is the FOOTPRINT mark, and it is the primary on purpose. Emboss moves the accumulated
//! surface within the cutter's footprint; it does not add a body. The sheet also carried a
//! ridge take on the same op, which is not ported: a ridge lies the moment the amount goes
//! negative, whereas the footprint reads correctly in both directions (see `emboss_recess`).

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The accumulated surface, lifted where the footprint covers it.
    Mark::Line {
        points: &[(1.5, 12.0), (6.0, 12.0), (6.0, 7.0), (12.0, 7.0), (12.0, 12.0), (16.5, 12.0)],
        ink: Ink::SOLID,
    },
    // The footprint walls — the cutter's extent, not a body.
    Mark::Line {
        points: &[(6.0, 2.5), (6.0, 15.5)],
        ink: Ink::DASHED,
    },
    Mark::Line {
        points: &[(12.0, 2.5), (12.0, 15.5)],
        ink: Ink::DASHED,
    },
];
