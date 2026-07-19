//! `extrude` (tile) — a profile with per-vertex depth ticks.
//!
//! The sketch is the subject and the ticks are the sweep. Depth is drawn per vertex rather
//! than as one ground rule, which is what distinguishes extruding a profile from setting a
//! datum under it.

use crate::icons::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    // The profile.
    g.line(&[(6.0, 17.0), (10.0, 9.0), (16.0, 12.0), (20.0, 6.0)]);
    // Where each vertex is carried to.
    let depth = g.faint(0.5);
    g.line_with(&[(6.0, 17.0), (6.0, 21.0)], depth);
    g.line_with(&[(20.0, 6.0), (20.0, 10.0)], depth);
    g.line_with(&[(13.0, 12.0), (13.0, 18.0)], depth);
}
