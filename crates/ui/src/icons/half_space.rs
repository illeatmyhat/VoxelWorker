//! `half-space` — a plane in perspective with hatching falling away beneath it.
//!
//! The hatching is what says *half-space* rather than *plane*: the body is everything on one
//! side, and the plane itself is unbounded, so the glyph cannot be a closed shape.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    // The plane.
    g.closed(&[(1.5, 11.0), (6.5, 6.0), (16.5, 6.0), (11.5, 11.0)]);
    // The side that is body.
    g.line(&[(4.0, 12.5), (4.0, 15.5)]);
    g.line(&[(8.0, 12.5), (8.0, 15.5)]);
    g.line(&[(12.0, 12.5), (12.0, 15.5)]);
}
