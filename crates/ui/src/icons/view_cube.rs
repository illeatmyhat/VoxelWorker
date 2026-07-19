//! `view-cube` — an isometric cube with its origin corner lit.
//!
//! The cube is not a body here: it is a control whose faces, edges and corners are all
//! pickable camera stations. The lit cell is the **front-bottom-right** corner — the one
//! visible origin the axis-coloured edges share, by owner ruling — so the mark says the cube
//! has addressable regions, and where they are counted from.
//!
//! It previously carried two hairline zone rules on the near face, which at 15 pt closed up
//! and left it indistinct from the plain `box` cube; a measured pass found this glyph 82%
//! overlapping both `box` and `mode-normal`. A filled cell survives where rules do not.

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
    // The origin corner, lit: half of each edge of the near-right face taken from the bottom
    // vertex — big enough to hold at 15 pt, where a ruled cell would not.
    g.fill(&[(9.0, 16.5), (12.5, 14.5), (12.5, 11.0), (9.0, 13.0)]);
}
