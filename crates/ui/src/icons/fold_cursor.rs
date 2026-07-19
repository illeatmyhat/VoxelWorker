//! `fold-cursor` — the stack with a caret in the gutter and the row past it dashed.
//!
//! The dashed row is dropped from this evaluation, not deleted, and dashing rather than
//! omitting it is what carries that distinction.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    g.rect((4.0, 2.5), (16.0, 5.5));
    g.rect((4.0, 7.5), (16.0, 10.5));
    // Past the cursor: still authored, not evaluated.
    g.dashed_rect((4.0, 12.5), (16.0, 15.5));
    // The insert caret.
    g.closed(&[(0.5, 9.5), (3.0, 11.5), (0.5, 13.5)]);
}
