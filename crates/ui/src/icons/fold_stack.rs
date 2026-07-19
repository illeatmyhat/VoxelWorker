//! `fold-stack` — three stacked rows: the ordered fold of the active scope.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    g.rect((2.0, 2.5), (16.0, 5.5));
    g.rect((2.0, 7.5), (16.0, 10.5));
    g.rect((2.0, 12.5), (16.0, 15.5));
}
