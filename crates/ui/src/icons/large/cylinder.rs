//! `cylinder` (tile) — a lathe body: solid top cap, faded bottom cap.
//!
//! Both caps are whole ellipses, not a front-only arc. The fade does the occlusion, which
//! is what says "a transparent construction seen through" rather than "a can".

use crate::icons::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    g.ellipse((13.0, 7.0), 8.0, 3.2);
    g.line(&[(5.0, 7.0), (5.0, 19.0)]);
    g.line(&[(21.0, 7.0), (21.0, 19.0)]);
    g.ellipse_with((13.0, 19.0), 8.0, 3.2, g.faint(0.5));
}
