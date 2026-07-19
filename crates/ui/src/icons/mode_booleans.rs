//! `mode-booleans` — a solid body with a dashed operand overlapping it.
//!
//! Dashed is the set's mark for an operand, so the pairing states the mode exactly: the
//! operands stay visible as x-ray ghosts *alongside* the folded result, rather than being
//! consumed by it.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    // The folded result.
    g.rect((2.5, 2.5), (10.5, 10.5));
    // The operand ghost.
    g.dashed_rect((7.5, 7.5), (15.5, 15.5));
}
