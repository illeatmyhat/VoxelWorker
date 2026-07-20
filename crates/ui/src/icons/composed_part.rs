//! `composed-part` — a container whose children have already merged into a single outline.
//!
//! This replaces the harvested sheet's `sealed-part` padlock, and the swap is a ruling. Every
//! part is a sealed scope, so a badge that says "sealed" distinguishes nothing — it labels the
//! universal. What a user can actually verify is the composition: a part folds into its parent
//! as ONE body, which is why an outset on the part dilates the whole rather than each child.
//! So the interior is drawn as the merged union outline — seamless, exactly like the `union`
//! glyph — sitting inside the part's boundary.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    // The part's boundary.
    Mark::Rect {
        a: (2.0, 2.0),
        b: (16.0, 16.0),
        ink: Ink::SOLID,
    },
    // Its children, already folded into one composed body — no interior seam.
    Mark::Closed {
        points: &[
            (5.0, 5.0),
            (9.9, 5.0),
            (9.9, 8.1),
            (13.0, 8.1),
            (13.0, 13.0),
            (8.1, 13.0),
            (8.1, 9.9),
            (5.0, 9.9),
        ],
        ink: Ink::SOLID,
    },
];
