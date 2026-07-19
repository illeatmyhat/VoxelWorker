//! `zoom` — a lens with a plus: dolly in and out.
//!
//! It shares the lens-and-handle body with `search`, and the plus is the only thing that
//! separates them; the two never appear on the same rail, so the shared body is a saving
//! rather than a collision.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    g.circle((7.5, 7.5), 5.0);
    g.line(&[(11.2, 11.2), (15.8, 15.8)]);
    // The plus inside the lens.
    g.line(&[(5.2, 7.5), (9.8, 7.5)]);
    g.line(&[(7.5, 5.2), (7.5, 9.8)]);
}
