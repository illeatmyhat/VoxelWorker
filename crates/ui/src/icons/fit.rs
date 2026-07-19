//! `fit` — four corner brackets around a centre square: frame the subject.
//!
//! The brackets are open Ls rather than a closed frame, so the mark reads as *framing
//! something* rather than as a window or a crop region.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    // Corner brackets, each two segments meeting at the corner.
    g.line(&[(2.5, 6.0), (2.5, 2.5), (6.0, 2.5)]);
    g.line(&[(12.0, 2.5), (15.5, 2.5), (15.5, 6.0)]);
    g.line(&[(15.5, 12.0), (15.5, 15.5), (12.0, 15.5)]);
    g.line(&[(6.0, 15.5), (2.5, 15.5), (2.5, 12.0)]);
    // The subject being framed.
    g.rect((6.5, 6.5), (11.5, 11.5));
}
