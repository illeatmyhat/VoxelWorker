//! `root-part` — a part resting on the ground rule.
//!
//! The rule runs wider than the part and is the only thing separating this from `part`: the
//! root is not a different kind of container, it is the one sitting on the scene's ground.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    g.rect((3.5, 2.5), (14.5, 12.5));
    g.rect((7.0, 6.5), (11.0, 10.5));
    g.line(&[(1.5, 15.5), (16.5, 15.5)]);
}
