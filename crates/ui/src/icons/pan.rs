//! `pan` — a four-headed cross: slide the target in the ground plane.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    // The two axes.
    g.line(&[(9.0, 2.0), (9.0, 16.0)]);
    g.line(&[(2.0, 9.0), (16.0, 9.0)]);
    // Arrowheads, one per direction.
    g.line(&[(7.4, 3.6), (9.0, 2.0), (10.6, 3.6)]);
    g.line(&[(7.4, 14.4), (9.0, 16.0), (10.6, 14.4)]);
    g.line(&[(3.6, 7.4), (2.0, 9.0), (3.6, 10.6)]);
    g.line(&[(14.4, 7.4), (16.0, 9.0), (14.4, 10.6)]);
}
