//! `construction-toggle` — flip an entity between real and construction.
//!
//! A box with one diagonal drawn in the construction linetype. This is the only rail glyph that
//! uses [`Ink::CONSTRUCTION`], and it should stay that way: the ink QUOTES what construction
//! geometry already looks like in the viewport, so spending it anywhere else would cost the quote
//! its meaning.
//!
//! The box is solid alongside it because the toggle is about the difference — one glyph showing
//! both states is what a toggle is.

use super::{Ink, Mark};

/// The box's corners; the diagonal runs between them.
const A: (f32, f32) = (2.5, 2.0);
const B: (f32, f32) = (15.5, 12.0);

pub(super) const DRAW: &[Mark] = &[
    Mark::Rect {
        a: A,
        b: B,
        ink: Ink::SOLID,
    },
    Mark::Line {
        points: &[A, B],
        ink: Ink::CONSTRUCTION,
    },
    Mark::Node {
        center: (9.0, 7.0),
        size: 2.6,
        ink: Ink::SOLID,
    },
];
