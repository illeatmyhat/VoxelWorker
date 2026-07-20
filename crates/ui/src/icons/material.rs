//! `material` — a 2×2 of swatches, one diagonal solid and the other dashed.
//!
//! The set has no filled noise swatch: a material thumbnail is content generated per material
//! and belongs to the browser, so the glyph states *assignment* — some cells carry a material,
//! some do not — rather than trying to depict a material.

use super::{Ink, Mark};

pub(super) const DRAW: &[Mark] = &[
    Mark::Rect {
        a: (2.5, 2.5),
        b: (8.5, 8.5),
        ink: Ink::SOLID,
    },
    Mark::Rect {
        a: (9.5, 2.5),
        b: (15.5, 8.5),
        ink: Ink::DASHED,
    },
    Mark::Rect {
        a: (2.5, 9.5),
        b: (8.5, 15.5),
        ink: Ink::DASHED,
    },
    Mark::Rect {
        a: (9.5, 9.5),
        b: (15.5, 15.5),
        ink: Ink::SOLID,
    },
];
