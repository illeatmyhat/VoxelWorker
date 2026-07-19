//! `cancel` — a cross: nothing is written to the document.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    g.line(&[(3.5, 3.5), (14.5, 14.5)]);
    g.line(&[(14.5, 3.5), (3.5, 14.5)]);
}
