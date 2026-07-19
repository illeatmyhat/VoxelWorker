//! `revolve` — a profile beside a dashed axis, with a sweep arrow curving around it.
//!
//! The axis is dashed because it is a datum and not an edge of the body; the arrow curves to
//! the far side so the mark measures ROUND where `extrude` measures square.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    // The axis of revolution.
    g.dashed_line((9.0, 1.5), (9.0, 16.5));
    // The profile.
    g.rect((3.0, 4.5), (6.5, 13.5));
    // The sweep: the SVG arc from (11, 4.2) to (11, 13.8) at r 5.4, resolved to its centre.
    g.arc((8.526, 9.0), 5.4, 5.4, -1.096, 1.096);
    g.line(&[(9.4, 12.2), (11.0, 13.8), (9.4, 15.4)]);
}
