//! `sweep` — a profile carried along a curved path to a dashed destination.
//!
//! The far profile is dashed because sweep is the reserved third lift: the mark is honest that
//! the far end is not yet a body the app will build.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    // The path — the SVG cubic `C3 8 8.5 4.5 15 4.5`, sampled as a polyline.
    let (p0, p1, p2, p3) = ((3.0, 15.0), (3.0, 8.0), (8.5, 4.5), (15.0, 4.5));
    let steps = 16;
    let points: Vec<(f32, f32)> = (0..=steps)
        .map(|i| {
            let t = i as f32 / steps as f32;
            let u = 1.0 - t;
            let (a, b, c, d) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
            (
                a * p0.0 + b * p1.0 + c * p2.0 + d * p3.0,
                a * p0.1 + b * p1.1 + c * p2.1 + d * p3.1,
            )
        })
        .collect();
    g.line(&points);
    // The profile at the start, and where it is headed.
    g.rect((1.2, 13.2), (4.8, 16.8));
    g.dashed_rect((13.2, 2.7), (16.8, 6.3));
}
