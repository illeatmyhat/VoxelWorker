//! `part` — a container square with one child inside it.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    g.rect((2.5, 2.5), (15.5, 15.5));
    g.rect((7.0, 7.0), (11.0, 11.0));
}
