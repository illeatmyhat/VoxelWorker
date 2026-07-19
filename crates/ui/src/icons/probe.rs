//! `probe` — one voxel, a leader, and its authorship read back in fold order.
//!
//! The probe answers "why is this voxel like this?". The answer is not a single name: it
//! is the ordered list of everything that claimed the voxel, with the LOSERS shown struck
//! through rather than hidden — seeing what was overridden is the whole point, because a
//! voxel that looks wrong usually looks wrong because of something later in the fold.
//!
//! So the glyph is a cell, a leader out of it, and two answer rows: the winner solid, the
//! loser dashed. Dashing carries the "still authored, did not survive" sense it already
//! carries on `fold-cursor` and on every operand mark in the set.
//!
//! Distinct from `search` (a magnifier, which filters names) and from `measure` (a ruler,
//! which answers distances). This one answers provenance.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    // The interrogated cell.
    g.rect((2.0, 12.0), (6.5, 16.5));
    // The leader, out of the cell and up to the readout.
    g.line(&[(6.5, 12.0), (10.0, 8.5)]);
    // The winner: what this voxel actually is.
    g.line(&[(10.0, 4.0), (16.5, 4.0)]);
    // A loser, kept visible and struck through.
    g.dashed_line((10.0, 8.5), (14.5, 8.5));
}
