//! `drawer` — a panel with a docked right edge and a caret pointing into it.
//!
//! The caret points outward, toward the edge the stack folds to, so the mark reads as the
//! action rather than as a static layout diagram.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    g.rect((2.5, 2.5), (15.5, 15.5));
    g.line(&[(11.0, 2.5), (11.0, 15.5)]);
    g.line(&[(12.5, 6.5), (14.5, 9.0), (12.5, 11.5)]);
}
