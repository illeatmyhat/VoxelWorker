//! `mode-normal` — a lit solid: the finished look, as the build will be.
//!
//! The three visible faces are FILLED at descending weights (1 / .55 / .3) so the cube reads
//! as a lit body rather than a wireframe. That is the whole distinction the mode is making —
//! Normal shows the resolved surface, where its siblings show slabs (`mode-onion`) and
//! operand ghosts (`mode-booleans`).
//!
//! It was previously the bare outlined cube, which was **the same drawing as `box`** — a
//! primitive body and a viewer mode sharing one silhouette, found at 100% geometric overlap
//! by a measured pass over the set. Shading is the way out, and it is true rather than
//! decorative: the fill does structural work (solid versus wireframe), which is the only kind
//! of fill that survives rail size.

use super::IconPainter;

/// The top face, lit.
const TOP: [(f32, f32); 4] = [(9.0, 1.5), (16.0, 5.5), (9.0, 9.5), (2.0, 5.5)];
/// The near-right face, falling away.
const RIGHT: [(f32, f32); 4] = [(16.0, 5.5), (16.0, 12.5), (9.0, 16.5), (9.0, 9.5)];
/// The near-left face, furthest from the light.
const LEFT: [(f32, f32); 4] = [(2.0, 5.5), (9.0, 9.5), (9.0, 16.5), (2.0, 12.5)];

pub(super) fn draw(g: &IconPainter) {
    // Lit from above: the top face full, the sides falling away.
    g.fill(&TOP);
    g.fill_with(&RIGHT, g.faint(0.55).color);
    g.fill_with(&LEFT, g.faint(0.3).color);
    // The silhouette last, so the outer edge stays crisp against the fills.
    g.closed(&[
        (9.0, 1.5),
        (16.0, 5.5),
        (16.0, 12.5),
        (9.0, 16.5),
        (2.0, 12.5),
        (2.0, 5.5),
    ]);
}
