//! `constraint-coincident` — two points become one point.
//!
//! Two runs meeting at a single node. The constraint ink is the one that MOVES onto the other: a
//! member already carrying a dimension wins, and with no dimensioned member the first selected
//! does. That rule is shared with [`constraint_equal`](super::constraint_equal), which is why
//! both marks put the red on the same side of the relation.
//!
//! The node is a square, this app's authored-vertex mark — a disc would say "a through-point",
//! which is a different thing.

use super::{Ink, Mark};

/// Where the two runs meet, and the only place the mark has a node.
const MEETING: (f32, f32) = (9.0, 12.0);

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(3.0, 4.5), MEETING],
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[(15.0, 4.5), MEETING],
        ink: Ink::CONSTRAINT,
    },
    Mark::Node {
        center: MEETING,
        size: 2.6,
        ink: Ink::CONSTRAINT,
    },
];
