//! `root-part` — a part resting on the ground rule.
//!
//! The rule runs wider than the part and is the only thing separating this from `part`: the
//! root is not a different kind of container, it is the one sitting on the scene's ground.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Rect {
        a: (3.5, 2.5),
        b: (14.5, 12.5),
        ink: Ink::SOLID,
    },
    Mark::Rect {
        a: (7.0, 6.5),
        b: (11.0, 10.5),
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(1.5, 15.5), (16.5, 15.5)],
        ink: Ink::SOLID,
    },
];
