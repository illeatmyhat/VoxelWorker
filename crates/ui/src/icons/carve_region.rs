//! `carve-region` — unpick a sketch region: the face inside the boundary becomes a hole.
//!
//! The additive twin of [`super::fill_region`], one mark apart: the inner face is dashed rather
//! than filled, the set's "authored, but not what you are looking at" (ADR 0030 §3, #100).

use super::{Ink, Mark};

/// The inner face — the region the author is carving away.
const FACE: &[(f32, f32)] = &[(6.5, 6.5), (11.5, 6.5), (11.5, 11.5), (6.5, 11.5)];

pub(super) const DRAW: &[Mark] = &[
    // The enclosing profile.
    Mark::Rect {
        a: (2.5, 2.5),
        b: (15.5, 15.5),
        ink: Ink::SOLID,
    },
    Mark::Closed {
        points: FACE,
        ink: Ink::DASHED,
    },
];
