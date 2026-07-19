//! `chevron-right` — disclosure, closed.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    g.line(&[(6.5, 3.0), (12.5, 9.0), (6.5, 15.0)]);
}
