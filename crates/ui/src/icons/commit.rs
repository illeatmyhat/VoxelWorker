//! `commit` — a check: the edit lands in the fold.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    g.line(&[(2.5, 9.5), (6.75, 13.75), (15.5, 5.0)]);
}
