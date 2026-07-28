//! `orbit-constrained` — the turntable: a rod standing in an orbit, and the orbit is an arrow.
//!
//! The LINE is the subject: the content of "constrained" is that ONE axis is privileged — world-up
//! stays up, the camera never rolls, the poles are places you cannot get past. It runs the full
//! height of the box and out past the path at both ends, because that overhang is the whole
//! difference between a rod THROUGH an orbit and a hoop with something stuck on it.
//!
//! The path is an ELLIPSE, not a circle, and it leans. A circle round a standing axis is a circle
//! only when you are looking down the axis, which is the one viewpoint from which the axis itself
//! is invisible; drawn flat-on it reads as a swirl on the page rather than a track in space. Wide
//! and shallow with a small tilt is a circle seen from a little above — the same view the mark's
//! own reader has of the scene.
//!
//! It is gapped at the top, where the far side passes behind the axis, and the arrowhead sits at
//! the gap's far lip pointing into it. Direction is what separates an orbit from a ring, so the
//! head is oversized and narrow-legged — at 15 pt the glyph box is about eighteen PIXELS, and a
//! short wide chevron there closes into a blob.
//!
//! Four other paths were drawn and rendered before this one, and each failed the same way: a front
//! half-ring is a bowl the axis stands in; a small hooked loop at the axis's end is a four-pixel
//! squiggle; a third-turn arc over the top of a hanging axis is a PICKAXE, which the eye reaches
//! well before it reaches "orbit"; and a perfect circle is the swirl above.
//!
//! [`orbit_free`](super::orbit_free) is deliberately a different construction, not this one plus a
//! stroke: a small cross with two open arrows half a turn apart. Line against cross, one arrow
//! against two.

use super::{IconPainter, TurningArrow};

/// Half the axis's length. It must exceed the orbit's vertical reach — the overhang is what makes
/// it a rod through the path rather than a stem under it.
const AXIS_HALF_LENGTH: f32 = 7.6;
/// The orbit's semi-axes: wide and shallow, a circle round a STANDING axis seen from a little
/// above. [`orbit_free`](super::orbit_free) uses the same pair, swapped for its second arrow.
const ORBIT_RADII: (f32, f32) = (5.9, 3.6);
/// The lean of the orbit's plane. Small — enough that the path is not sitting flat on the page,
/// not so much that it stops reading as level.
const ORBIT_TILT: f32 = -0.18;
/// The sweep, in the painter's angles (clockwise from +x, y growing downward): from just clockwise
/// of the top, round the near side, ending just anticlockwise of the top again. The gap left at the
/// top is the far side, which the axis stands in front of, and it is where the arrowhead points.
///
/// The angle is small because the ellipse is wide: a quarter of a radian either side of the top is
/// nearly three units of GAP on a path this flat, and a wider one stops reading as an occlusion and
/// starts reading as a broken curve. It is centred on where the TILTED path crosses the axis, which
/// is not the parameter's own top — tilt one and the two part company.
const ORBIT_SWEEP: (f32, f32) = (-1.22, 4.58);
/// The arrowhead: longer and narrower than the sheet's static arrows, because here it carries the
/// mark's whole meaning at 15 pt and a short wide one closes up into a blob.
const ARROW_HEAD: (f32, f32) = (2.9, 1.6);

pub(super) fn draw(g: &IconPainter) {
    let center = (9.0_f32, 9.0_f32);

    // The fixed world-up: the subject.
    g.line(&[
        (center.0, center.1 - AXIS_HALF_LENGTH),
        (center.0, center.1 + AXIS_HALF_LENGTH),
    ]);

    // The traveller: round the near side, ending at the gap's far lip and pointing into it.
    g.turning_arrow(TurningArrow {
        center,
        radii: ORBIT_RADII,
        sweep: ORBIT_SWEEP,
        tilt: ORBIT_TILT,
        head: ARROW_HEAD,
    });
}
