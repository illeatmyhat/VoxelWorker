//! `orbit` — a body at rest inside a dashed, tilted orbit ring.
//!
//! The ring is dashed and tilted rather than a plain circle because the mark must say the
//! *camera* travels while the body stays put: a solid ring at 15 pt reads as a rotating part,
//! which is the opposite of what orbit does.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    // The body being orbited — it does not move.
    g.rect((6.5, 6.5), (11.5, 11.5));
    // The camera's path: an ellipse tilted −27°, dashed. The kit's ellipse is axis-aligned, so
    // the tilt is applied here, one dash at a time.
    let (cx, cy) = (9.0, 9.0);
    let (rx, ry) = (8.0, 3.4);
    let (sin, cos) = (-27.0f32).to_radians().sin_cos();
    let dashes = 10;
    let step = std::f32::consts::TAU / dashes as f32;
    for dash in 0..dashes {
        let start = dash as f32 * step;
        let points: Vec<(f32, f32)> = (0..=4)
            .map(|k| {
                let t = start + step * 0.55 * (k as f32 / 4.0);
                let (x, y) = (rx * t.cos(), ry * t.sin());
                (cx + x * cos - y * sin, cy + x * sin + y * cos)
            })
            .collect();
        g.line(&points);
    }
}
