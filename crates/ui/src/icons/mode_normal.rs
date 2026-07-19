//! `mode-normal` — a plain solid cube: the finished look, nothing added.
//!
//! It is deliberately the bare cube. Normal mode shows the resolved surface and nothing else,
//! so any extra mark on the glyph would promise a second thing the mode does not do.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    g.closed(&[
        (9.0, 1.5),
        (16.0, 5.5),
        (16.0, 12.5),
        (9.0, 16.5),
        (2.0, 12.5),
        (2.0, 5.5),
    ]);
    g.line(&[(2.0, 5.5), (9.0, 9.5), (16.0, 5.5)]);
    g.line(&[(9.0, 9.5), (9.0, 16.5)]);
}
