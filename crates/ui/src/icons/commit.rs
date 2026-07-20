//! `commit` — a check: the edit lands in the fold.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[Mark::Line {
    points: &[(2.5, 9.5), (6.75, 13.75), (15.5, 5.0)],
    ink: Ink::SOLID,
}];
