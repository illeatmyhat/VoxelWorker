//! `constraint-collinear` — two segments share one infinite carrier.
//!
//! The gap is what makes them TWO. Close it and the mark is a line rather than a relation, so the
//! 2.5-unit break is load-bearing and not a styling choice.
//!
//! Collinear is kept as its own constraint even though a coincidence acting on the segments'
//! infinite carriers would subsume it, so this mark has to look nothing like
//! [`constraint_coincident`](super::constraint_coincident).

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(3.0, 15.0), (7.75, 10.25)],
        ink: Ink::SOLID,
    },
    // Resumed 2.5 units along the same carrier — the break that makes them two.
    Mark::Line {
        points: &[(10.25, 7.75), (15.0, 3.0)],
        ink: Ink::CONSTRAINT,
    },
];
