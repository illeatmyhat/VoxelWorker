//! `emboss-recess` — the emboss footprint with the surface driven down instead of up.
//!
//! It is the same mark mirrored about the surface line, which is exactly the relationship the
//! op has: one amount, signed.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    g.line(&[(1.5, 7.0), (6.0, 7.0), (6.0, 12.0), (12.0, 12.0), (12.0, 7.0), (16.5, 7.0)]);
    g.dashed_line((6.0, 2.5), (6.0, 15.5));
    g.dashed_line((12.0, 2.5), (12.0, 15.5));
}
