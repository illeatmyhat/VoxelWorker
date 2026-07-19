//! `home` — a house silhouette: a full-width roof over a body left open at the top.
//!
//! The body deliberately has no top edge; it tucks under the roof, which is what keeps the
//! mark reading as a house rather than as a triangle stacked on a box at 15 pt.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    // Roof: left eave → apex → right eave.
    g.line(&[(3.0, 8.5), (9.0, 3.5), (15.0, 8.5)]);
    // Body: left wall down, floor, right wall up.
    g.line(&[(5.0, 8.0), (5.0, 14.5), (13.0, 14.5), (13.0, 8.0)]);
}
