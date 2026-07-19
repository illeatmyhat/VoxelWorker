//! `view-cube` — an isometric cube with its near face ruled into zones.
//!
//! The two extra rules on the right face are what separate this from the plain `box` glyph:
//! the cube is not a body here, it is a control whose faces, edges and corners are all
//! pickable stations.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    // The silhouette.
    g.closed(&[
        (9.0, 1.5),
        (16.0, 5.5),
        (16.0, 12.5),
        (9.0, 16.5),
        (2.0, 12.5),
        (2.0, 5.5),
    ]);
    // The three visible faces.
    g.line(&[(2.0, 5.5), (9.0, 9.5), (16.0, 5.5)]);
    g.line(&[(9.0, 9.5), (9.0, 16.5)]);
    // Zone rules on the near-right face.
    g.line(&[(11.33, 8.17), (11.33, 14.83)]);
    g.line(&[(13.67, 6.83), (13.67, 13.5)]);
}
