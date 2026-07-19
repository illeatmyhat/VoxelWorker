//! `mode-onion` — lifted layer slices: two faint carets above a SOLID band, one faint below.
//!
//! Ported verbatim from the shipped chrome (`signal_chrome::draw_layers`), which is the glyph
//! the owner approved in the prototype round; the harvested sheet's redraw is deliberately not
//! used here. The filled band is the active layer and the carets are the layers it sits
//! between — the one place in the set where opacity carries meaning rather than depth, because
//! "the band you are standing on" is precisely a contrast statement.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    let faint = g.faint(0.45);
    let mid = g.faint(0.7);
    // Carets above: the layers already passed.
    g.line_with(&[(3.0, 5.0), (9.0, 2.5), (15.0, 5.0)], faint);
    g.line_with(&[(3.0, 8.0), (9.0, 5.5), (15.0, 8.0)], mid);
    // The active band — a thin hexagonal slab, filled.
    g.fill(&[
        (9.0, 8.8),
        (15.0, 11.2),
        (15.0, 11.4),
        (9.0, 13.9),
        (3.0, 11.4),
        (3.0, 11.2),
    ]);
    // The caret below: what is still to come.
    g.line_with(&[(3.0, 14.5), (9.0, 17.0), (15.0, 14.5)], faint);
}
