//! `sculpt-add` — a dashed brush disc with a plus at its centre.
//!
//! The disc is dashed because it is a brush *radius*, an authored Measurement, not the edge of
//! a body — the same reason a chisel or pencil was rejected for this mark.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    dashed_disc(g);
    g.line(&[(9.0, 5.5), (9.0, 12.5)]);
    g.line(&[(5.5, 9.0), (12.5, 9.0)]);
}

/// The brush radius: a circle dashed by drawing only part of each arc segment, since the kit's
/// dash helpers work on straight runs.
fn dashed_disc(g: &IconPainter) {
    let dashes = 9;
    let step = std::f32::consts::TAU / dashes as f32;
    for dash in 0..dashes {
        let start = dash as f32 * step;
        g.arc((9.0, 9.0), 5.5, 5.5, start, start + step * 0.55);
    }
}
