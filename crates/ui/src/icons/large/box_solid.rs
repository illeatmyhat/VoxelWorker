//! `box` (tile) — a complete iso cube with its three hidden back edges receding.
//!
//! The block IS a cube, so the tile draws one in projection rather than the rail's flat
//! silhouette. The fade on the back edges is x-ray, the same reading the app's operand
//! ghosts use — depth without a hidden-line pass.

use crate::icons::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    // The cube's visible silhouette.
    g.closed(&[(13.0, 3.0), (23.0, 8.0), (23.0, 18.0), (13.0, 23.0), (3.0, 18.0), (3.0, 8.0)]);
    // The three edges meeting at the far corner, dropped back.
    let behind = g.faint(0.5);
    g.line_with(&[(3.0, 8.0), (13.0, 13.0), (23.0, 8.0)], behind);
    g.line_with(&[(13.0, 13.0), (13.0, 23.0)], behind);
}
