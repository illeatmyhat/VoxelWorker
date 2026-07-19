//! `array` — one solid cell followed by two dashed repeats.
//!
//! Only the first is solid: exactly one node was authored, and the repeats are placements the
//! decorator generates, which is also why the array is one entry in the fold and not three.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    g.rect((1.5, 6.5), (5.5, 11.5));
    g.dashed_rect((7.0, 6.5), (11.0, 11.5));
    g.dashed_rect((12.5, 6.5), (16.5, 11.5));
}
