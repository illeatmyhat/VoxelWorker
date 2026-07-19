//! `extrude` — a dashed profile lifted to a solid one, with the travel arrow beside it.
//!
//! The dashed rectangle is the sketch and the solid one the swept result, so the mark shows
//! the transformation rather than just the outcome.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    // The resulting body.
    g.rect((2.5, 11.0), (9.5, 14.5));
    // The profile it came from.
    g.dashed_rect((2.5, 3.5), (9.5, 7.0));
    // The lift.
    g.line(&[(13.5, 14.5), (13.5, 4.0)]);
    g.line(&[(11.9, 5.6), (13.5, 4.0), (15.1, 5.6)]);
}
