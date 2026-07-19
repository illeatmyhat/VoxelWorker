//! `outset` — a solid body with the dilated envelope dashed around it, and two growth ticks.
//!
//! The ticks run outward from the body's corners to the envelope's. Without them the nested
//! squares are ambiguous — they could equally read as a part containing a child — and the
//! direction of the field is the whole content of the mark.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    // The authored body.
    g.rect((6.0, 6.0), (12.0, 12.0));
    // The dilated envelope.
    g.dashed_rect((2.5, 2.5), (15.5, 15.5));
    // Which way the field moves.
    g.line(&[(12.0, 6.0), (15.5, 2.5)]);
    g.line(&[(6.0, 12.0), (2.5, 15.5)]);
}
