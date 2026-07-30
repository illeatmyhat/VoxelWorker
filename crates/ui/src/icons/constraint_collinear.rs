//! `constraint-collinear` — two segments share one infinite carrier.
//!
//! The gap is what makes them TWO. Close it and the mark is a line rather than a relation, so the
//! 2.5-unit break is load-bearing and not a styling choice.
//!
//! Collinear only exists because the app is Fusion-shaped: Onshape deletes it, because its
//! Coincident acts on the infinite underlying geometry and coincident-on-two-lines already IS
//! collinear. Keeping it is right for transfer, and it is why this mark has to look nothing like
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
        ink: Ink::ACCENT,
    },
];
