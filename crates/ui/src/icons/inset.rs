//! `inset` — `outset` reversed: the authored body is the outer solid, the eroded result dashed.
//!
//! Which square is solid is the only difference, and it is the correct one: the solid square is
//! always the body the user authored, and the ticks always point the way the field moves.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    g.rect((2.5, 2.5), (15.5, 15.5));
    g.dashed_rect((6.0, 6.0), (12.0, 12.0));
    g.line(&[(15.5, 2.5), (12.0, 6.0)]);
    g.line(&[(2.5, 15.5), (6.0, 12.0)]);
}
