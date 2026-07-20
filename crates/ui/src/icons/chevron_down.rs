//! `chevron-down` — disclosure, open.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[Mark::Line {
    points: &[(3.0, 6.5), (9.0, 12.5), (15.0, 6.5)],
    ink: Ink::SOLID,
}];
