//! `search` — a lens and handle: filter by name.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    g.circle((7.5, 7.5), 5.0);
    g.line(&[(11.2, 11.2), (15.8, 15.8)]);
}
