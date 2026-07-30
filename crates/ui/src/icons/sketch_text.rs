//! `sketch-text` — text converted to profile geometry.
//!
//! One letterform on a baseline. An `A` because it is the one glyph whose outline is unmistakably
//! a shape rather than lettering — the mark has to say "this becomes a PROFILE", not "this places
//! a label", and a label is exactly what a text cursor or a rendered word would say.
//!
//! The letter carries the accent and the baseline is line art: the baseline is where you click,
//! the outline is what you get.

use super::{Ink, Mark};

/// The baseline the text sits on — the click, not the product.
const BASELINE: f32 = 14.0;

pub(super) const DRAW: &[Mark] = &[
    Mark::Line {
        points: &[(4.5, 12.0), (9.0, 2.5), (13.5, 12.0)],
        ink: Ink::ACCENT,
    },
    Mark::Line {
        points: &[(6.4, 8.0), (11.6, 8.0)],
        ink: Ink::ACCENT,
    },
    Mark::Line {
        points: &[(2.5, BASELINE), (15.5, BASELINE)],
        ink: Ink::SOLID,
    },
];
