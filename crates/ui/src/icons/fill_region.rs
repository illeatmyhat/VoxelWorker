//! `fill-region` — pick a sketch region back: the face inside the boundary becomes material.
//!
//! One mark apart from [`super::carve_region`], the way `sculpt-add` and `carve` are a pair: the
//! inner face is FILLED here and dashed there, so picked and unpicked read against each other
//! rather than each on its own.

use super::{Ink, Mark};

/// The inner face — the region the author is picking.
const FACE: &[(f32, f32)] = &[(6.5, 6.5), (11.5, 6.5), (11.5, 11.5), (6.5, 11.5)];

pub(super) const DRAW: &[Mark] = &[
    // The enclosing profile.
    Mark::Rect {
        a: (2.5, 2.5),
        b: (15.5, 15.5),
        ink: Ink::SOLID,
    },
    Mark::Fill {
        points: FACE,
        opacity: 0.55,
    },
    Mark::Closed {
        points: FACE,
        ink: Ink::SOLID,
    },
];
