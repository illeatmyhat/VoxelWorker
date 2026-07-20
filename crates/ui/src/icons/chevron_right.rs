//! `chevron-right` — disclosure, closed.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[Mark::Line {
    points: &[(6.5, 3.0), (12.5, 9.0), (6.5, 15.0)],
    ink: Ink::SOLID,
}];
