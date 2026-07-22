//! `rectangle` — drag a box into a four-point profile, the box-drag sugar inside the mode.
//!
//! The two filled corner nodes are the drag diagonal — the grab corner and the cursor — so the
//! glyph reads as "drag from here to here" rather than a static box. ADR 0028 slice 3.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The box the drag encloses.
    Mark::Rect {
        a: (3.5, 5.0),
        b: (14.5, 13.5),
        ink: Ink::SOLID,
    },
    // The drag diagonal's two corner nodes.
    Mark::Rect {
        a: (2.4, 3.9),
        b: (4.6, 6.1),
        ink: Ink::SOLID,
    },
    Mark::Rect {
        a: (13.4, 12.4),
        b: (15.6, 14.6),
        ink: Ink::SOLID,
    },
];
