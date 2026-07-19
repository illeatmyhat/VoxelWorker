//! `axes-gizmo` — the Z-up triad: one long vertical arm, two short ground arms.
//!
//! The vertical arm is drawn longest on purpose. The world is Z-up, and a triad whose three
//! arms are equal leaves the reader to guess which one is vertical.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    // +Z: vertical, and the dominant arm.
    g.line(&[(9.0, 10.5), (9.0, 2.5)]);
    // The ground plane, XY.
    g.line(&[(9.0, 10.5), (15.5, 14.0)]);
    g.line(&[(9.0, 10.5), (2.5, 14.0)]);
    // The stub toward the viewer: front is −Y.
    g.line(&[(9.0, 10.5), (9.0, 12.5)]);
}
