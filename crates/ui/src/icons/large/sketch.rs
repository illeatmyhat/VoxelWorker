//! `sketch` (tile) — a curve with two authored endpoints, drawn as solid handles.
//!
//! The filled dots are the whole point: a sketch is a thing with grabbable control points,
//! and this is the only mark in either family that says so. The rail twin cannot — at 15 pt
//! a filled dot and a stroked ring are the same three pixels — so it settles for hollow
//! vertex squares instead. This is why the two sets are separate drawings.

use crate::icons::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    g.cubic((5.0, 18.0), (8.0, 6.0), (18.0, 6.0), (21.0, 18.0));
    g.filled_circle((5.0, 18.0), 1.8);
    g.filled_circle((21.0, 18.0), 1.8);
}
