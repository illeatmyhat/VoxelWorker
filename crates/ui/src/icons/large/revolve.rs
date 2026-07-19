//! `revolve` (tile) — a profile, an axis, and the arc that carries one onto the other.
//!
//! Shares the rail mark's construction (profile rect + axis + swept arc) rather than the
//! c-palette original's equator ellipse, so the two sizes read as the same verb. What the
//! tile adds is room for the profile to be a real closed body instead of a hint.

use crate::icons::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    // The axis, subordinate to what spins about it.
    g.line_with(&[(13.0, 2.0), (13.0, 24.0)], g.faint(0.5));
    // The sweep.
    g.cubic((13.0, 5.0), (21.0, 8.0), (21.0, 18.0), (13.0, 21.0));
    // The profile being revolved.
    g.rect((5.0, 7.0), (9.5, 19.0));
}
