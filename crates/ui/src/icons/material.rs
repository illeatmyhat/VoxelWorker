//! `material` — a 2×2 of swatches, one diagonal solid and the other dashed.
//!
//! The set has no filled noise swatch: a material thumbnail is content generated per material
//! and belongs to the browser, so the glyph states *assignment* — some cells carry a material,
//! some do not — rather than trying to depict a material.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    g.rect((2.5, 2.5), (8.5, 8.5));
    g.dashed_rect((9.5, 2.5), (15.5, 8.5));
    g.dashed_rect((2.5, 9.5), (8.5, 15.5));
    g.rect((9.5, 9.5), (15.5, 15.5));
}
