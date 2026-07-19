//! `density` — one block ruled into a 3×3 of voxels.
//!
//! The outer square never changes size between states of this mark: density is voxels per
//! block, fineness and never extent, and only the interior ruling would differ.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    // The block.
    g.rect((2.5, 2.5), (15.5, 15.5));
    // Its voxels.
    g.line(&[(6.83, 2.5), (6.83, 15.5)]);
    g.line(&[(11.17, 2.5), (11.17, 15.5)]);
    g.line(&[(2.5, 6.83), (15.5, 6.83)]);
    g.line(&[(2.5, 11.17), (15.5, 11.17)]);
}
