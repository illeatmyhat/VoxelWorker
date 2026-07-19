//! `chevron-down` — disclosure, open.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    g.line(&[(3.0, 6.5), (9.0, 12.5), (15.0, 6.5)]);
}
