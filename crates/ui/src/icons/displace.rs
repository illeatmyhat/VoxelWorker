//! `displace` — a perturbed surface above its dashed flat reference.
//!
//! The datum below is what makes the zigzag mean displacement rather than terrain: the field
//! is a deviation from a reference, and the reference has to be visible to be deviated from.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    g.line(&[(1.5, 10.0), (5.0, 6.5), (8.5, 10.0), (12.0, 6.5), (15.5, 10.0)]);
    g.dashed_line((1.5, 14.0), (16.5, 14.0));
}
