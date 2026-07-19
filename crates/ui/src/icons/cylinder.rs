//! `cylinder` — a full top ellipse, two walls, and only the near half of the base.
//!
//! The base's far half is omitted rather than drawn: an opaque cylinder hides it, and drawing
//! it would turn the mark into a wireframe of something the display never shows.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    g.ellipse((9.0, 4.6), 5.5, 2.2);
    g.line(&[(3.5, 4.6), (3.5, 13.4)]);
    g.line(&[(14.5, 4.6), (14.5, 13.4)]);
    // The near half of the base, sweeping under.
    g.arc((9.0, 13.4), 5.5, 2.2, std::f32::consts::PI, 0.0);
}
