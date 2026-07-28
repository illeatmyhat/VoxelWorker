//! `orbit-constrained` — the turntable: a rod threaded through a ring, and the ring is an arrow.
//!
//! The LINE is the subject: the content of "constrained" is that ONE axis is privileged — world-up
//! stays up, the camera never rolls, the poles are places you cannot get past. It runs the full
//! height of the box and out past the ring at both ends, because that overhang is the whole
//! difference between a rod THROUGH a hoop and a hoop with something stuck on it.
//!
//! The ring is gapped at the top, where the axis passes and the far side would be hidden anyway,
//! and the arrowhead sits at the gap's left lip pointing into it. Direction is what separates an
//! orbit from a ring, so the head is oversized and narrow-legged — at 15 pt the glyph box is about
//! eighteen PIXELS, and a short wide chevron there closes into a blob.
//!
//! Four other paths were drawn and rendered before this one, and each failed the same way: a front
//! half-ring is a bowl the axis stands in; a small hooked loop at the axis's end is a four-pixel
//! squiggle; a shallow ellipse crossing the axis is a dagger; and a third-turn arc over the top of
//! a hanging axis is a PICKAXE — a curved head on a long handle, which the eye reaches well before
//! it reaches "orbit".
//!
//! [`orbit_free`](super::orbit_free) is deliberately a different construction, not this one plus a
//! stroke: a small cross with two open arrows half a turn apart. Line against cross, one arrow
//! against two.

use super::IconPainter;

/// Half the axis's length. It must exceed `ARC_RADIUS` — the overhang is what makes it a rod
/// through the ring rather than a stem under it.
const AXIS_HALF_LENGTH: f32 = 7.6;
/// The ring's radius about the glyph centre. [`orbit_free`](super::orbit_free) uses the same one,
/// so the pair sit on a common circle.
const ARC_RADIUS: f32 = 5.5;
/// The sweep, in the painter's angles (clockwise from +x, y growing downward): from just clockwise
/// of the top, all the way round, ending just anticlockwise of it. The ~0.45 rad left open at the
/// top is where the axis passes and where the arrowhead points.
const ARC_FROM: f32 = -1.12;
const ARC_TO: f32 = 4.26;
/// The arrowhead: longer and narrower than the sheet's static arrows, because here it carries the
/// mark's whole meaning at 15 pt and a short wide one closes up into a blob.
const ARROW_TRAIL: f32 = 2.9;
const ARROW_SPREAD: f32 = 1.6;

pub(super) fn draw(g: &IconPainter) {
    let center = (9.0_f32, 9.0_f32);

    // The fixed world-up: the subject.
    g.line(&[
        (center.0, center.1 - AXIS_HALF_LENGTH),
        (center.0, center.1 + AXIS_HALF_LENGTH),
    ]);

    // The traveller: all the way round, ending at the gap's left lip and pointing into it.
    g.arrowed_arc(
        center,
        ARC_RADIUS,
        ARC_RADIUS,
        ARC_FROM,
        ARC_TO,
        (ARROW_TRAIL, ARROW_SPREAD),
    );
}
