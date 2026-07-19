//! `flip` — two carets facing away from a dashed mirror axis.
//!
//! The axis is dashed for the same reason as in `revolve`: it is a datum the user picks, not an
//! edge of any body.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    g.dashed_line((9.0, 1.5), (9.0, 16.5));
    g.line(&[(6.5, 5.0), (2.5, 9.0), (6.5, 13.0)]);
    g.line(&[(11.5, 5.0), (15.5, 9.0), (11.5, 13.0)]);
}
