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
//! the part of the orbit behind it. What survives is the two outer wings, which is the shape
//! the eye already knows from Saturn, and there is no ring left encircling the body for the
//! iris reading to latch onto.
//!
//! Consequently the planet is LARGE — larger than the orbit's minor radius, deliberately, so
//! that the path is cut rather than framed. The moon then does the rest: it is the one mark
//! that says the path is travelled.

use super::IconPainter;

/// How far the orbital plane leans, in degrees.
const TILT_DEGREES: f32 = -22.0;
/// Where the moon sits on the path, in radians — high and to the right, clear of both wings.
const MOON_ANGLE: f32 = -0.40;

pub(super) fn draw(g: &IconPainter) {
    let (center_x, center_y) = (9.0_f32, 9.0_f32);
    let (radius_x, radius_y) = (7.6_f32, 3.0_f32);
    let (sin_tilt, cos_tilt) = TILT_DEGREES.to_radians().sin_cos();

    // A point at angle `t` on the tilted orbit.
    let on_orbit = |t: f32| {
        let (x, y) = (radius_x * t.cos(), radius_y * t.sin());
        (
            center_x + x * cos_tilt - y * sin_tilt,
            center_y + x * sin_tilt + y * cos_tilt,
        )
    };

    // The path, first — everything after this occludes it.
    let path: Vec<(f32, f32)> = (0..=64)
        .map(|step| on_orbit(std::f32::consts::TAU * step as f32 / 64.0))
        .collect();
    g.line(&path);

    // The body being orbited. Filled and wide enough to cut the path down to its wings.
    g.filled_circle((center_x, center_y), 4.6);

    // The camera, out on the path.
    g.filled_circle(on_orbit(MOON_ANGLE), 1.5);
}
