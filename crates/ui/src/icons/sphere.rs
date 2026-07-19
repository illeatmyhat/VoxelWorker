//! `sphere` — a circle with its equator, so the mark reads as a solid and not as a disc.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    g.circle((9.0, 9.0), 6.5);
    g.ellipse((9.0, 9.0), 6.5, 2.4);
}
