//! `carve` — the same dashed brush disc as `sculpt-add`, with only the horizontal bar.
//!
//! Deliberately one stroke different from its additive twin: they are the same tool under a
//! different fold, and a reader should see them as a pair.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    let dashes = 9;
    let step = std::f32::consts::TAU / dashes as f32;
    for dash in 0..dashes {
        let start = dash as f32 * step;
        g.arc((9.0, 9.0), 5.5, 5.5, start, start + step * 0.55);
    }
    g.line(&[(5.5, 9.0), (12.5, 9.0)]);
}
