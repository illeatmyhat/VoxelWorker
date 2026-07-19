//! `sketch` — a closed quadrilateral profile with a handle square at each vertex.
//!
//! The profile is drawn irregular, not as a rectangle: the sketch is the authoring atom and
//! organic outlines are its point, while a rectangle would read as the box primitive that is
//! merely sugar over it.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    // The flattened polygon.
    g.closed(&[(4.0, 12.5), (6.5, 4.5), (14.0, 6.5), (11.5, 14.5)]);
    // The authored vertices.
    g.rect((3.2, 11.7), (4.8, 13.3));
    g.rect((5.7, 3.7), (7.3, 5.3));
    g.rect((13.2, 5.7), (14.8, 7.3));
    g.rect((10.7, 13.7), (12.3, 15.3));
}
