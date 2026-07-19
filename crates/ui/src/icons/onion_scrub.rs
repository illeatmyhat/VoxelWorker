//! `onion-scrub` — a stack of layer rules with a handle riding the left edge.
//!
//! The handle is the subject: the layers are the given, and what the control does is move a
//! band through them.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    // The layers.
    g.line(&[(5.0, 3.5), (16.5, 3.5)]);
    g.line(&[(5.0, 7.0), (16.5, 7.0)]);
    g.line(&[(5.0, 10.5), (16.5, 10.5)]);
    g.line(&[(5.0, 14.0), (16.5, 14.0)]);
    // The scrub handle.
    g.rect((1.5, 9.0), (3.5, 12.0));
}
