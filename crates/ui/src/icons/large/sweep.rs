//! `sweep` (tile) — a profile carried along a path to a dashed destination.
//!
//! The far profile is dashed because sweep is the reserved third lift: the mark stays
//! honest that the far end is not yet a body the app will build. Same construction as the
//! rail twin, with room for both profiles to be squares rather than ticks.

use crate::icons::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    // The path.
    g.cubic((5.0, 21.0), (5.0, 11.0), (12.0, 6.0), (21.0, 6.0));
    // The profile at the start, and where it is headed.
    g.rect((2.0, 18.0), (8.0, 24.0));
    g.dashed_rect((18.0, 3.0), (24.0, 9.0));
}
