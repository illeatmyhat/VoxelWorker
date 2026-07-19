//! `sphere` (tile) — a ball, its roundness carried by one receding equator.
//!
//! The equator rides at half weight so it reads as an interior contour. Drawn at equal
//! weight it becomes a second silhouette and the mark turns into a lens or an eye.

use crate::icons::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    g.circle((13.0, 13.0), 9.0);
    g.ellipse_with((13.0, 13.0), 9.0, 3.4, g.faint(0.5));
}
