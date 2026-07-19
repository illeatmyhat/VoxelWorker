//! `orbit` — a planet with a moon on a tilted path: the camera travels, the body does not.
//!
//! The mark says the SUBJECT stays put while the eye moves around it, which is what a planet
//! and its moon say and what the previous square-in-a-dashed-ring did not.
//!
//! ## The planet is painted OVER the path, and that is the whole trick
//!
//! A closed ellipse drawn concentrically around a body reads as an **eye** — at every ratio,
//! however the radii are tuned. The way out is not tuning: it is occlusion. The planet is
//! filled and painted after the path, and because both carry the same ink it simply swallows
//! the part of the orbit behind it. What survives is the two outer wings, the shape the eye
//! already knows from Saturn, and there is no ring left encircling the body for the iris
//! reading to latch onto.
//!
//! Consequently the planet is LARGE — nearly twice the orbit's minor radius, deliberately, so
//! that the path is cut rather than framed.
//!
//! ## The far half recedes
//!
//! The half of the path behind the planet is drawn faint and the near half solid. Because the
//! planet then covers the middle of both, each surviving wing keeps ONE solid edge and one
//! receding edge — which is what tips the mark from a flat badge into something with a near
//! side and a far side. The moon does the rest: it is the single mark that says the path is
//! travelled.

use super::IconPainter;

/// How far the orbital plane leans, in degrees.
const TILT_DEGREES: f32 = -18.0;
/// Where the moon sits on the path, in radians — high and to the right, clear of both wings.
const MOON_ANGLE: f32 = -0.32;
/// How far the far side of the path recedes.
const FAR_SIDE_OPACITY: f32 = 0.45;

pub(super) fn draw(g: &IconPainter) {
    let (center_x, center_y) = (9.0_f32, 9.0_f32);
    let (radius_x, radius_y) = (7.5_f32, 2.3_f32);
    let (sin_tilt, cos_tilt) = TILT_DEGREES.to_radians().sin_cos();

    // A point at angle `t` on the tilted orbit.
    let on_orbit = |t: f32| {
        let (x, y) = (radius_x * t.cos(), radius_y * t.sin());
        (
            center_x + x * cos_tilt - y * sin_tilt,
            center_y + x * sin_tilt + y * cos_tilt,
        )
    };
    // Half the path, sampled from `from` to `to`.
    let half = |from: f32, to: f32| -> Vec<(f32, f32)> {
        (0..=32)
            .map(|step| on_orbit(from + (to - from) * step as f32 / 32.0))
            .collect()
    };

    // The path, first — everything after this occludes it. The far side recedes.
    let pi = std::f32::consts::PI;
    g.line_with(&half(pi, std::f32::consts::TAU), g.faint(FAR_SIDE_OPACITY));
    g.line(&half(0.0, pi));

    // The body being orbited. Filled and wide enough to cut the path down to its wings.
    g.filled_circle((center_x, center_y), 4.4);

    // The camera, out on the path.
    g.filled_circle(on_orbit(MOON_ANGLE), 1.4);
}
