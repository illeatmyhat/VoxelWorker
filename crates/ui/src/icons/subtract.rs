//! `subtract` — a body with a bite taken out of it, the operand left dashed over the gap.
//!
//! The operand stays drawn because subtract is an occupancy-only mask: the thing that was
//! removed is a shape the user authored, not an absence.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    // What survives the fold.
    g.closed(&[
        (2.5, 2.5),
        (10.5, 2.5),
        (10.5, 7.5),
        (7.5, 7.5),
        (7.5, 10.5),
        (2.5, 10.5),
    ]);
    // The cutter.
    g.dashed_rect((7.5, 7.5), (15.5, 15.5));
}
