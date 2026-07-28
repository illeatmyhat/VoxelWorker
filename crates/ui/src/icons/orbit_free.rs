//! `orbit-free` — the trackball: a small cross, and an orbit round each of its arms.
//!
//! Two arrows on two different axes, which is the whole content of "free": there is no privileged
//! up, so no one orbit is the orbit. [`orbit_constrained`](super::orbit_constrained) is the same
//! drawing with the second orbit taken away and the cross straightened into a single long rod.
//!
//! Both paths are ellipses that lean, because a circle round an axis is a circle only when you are
//! looking straight down that axis — and here there are two axes, so a flat-on circle would have to
//! be wrong about at least one of them. The two share a lean, so they read as two tracks in one
//! space rather than two swirls that happen to overlap.
//!
//! ## The cross is small, and that is the whole point of it
//!
//! It was first drawn at a rod's length, and at that size it is two long strokes meeting in the
//! middle and it swamps the box. That is backwards. The cross is not the subject; it is the
//! registration mark the orbits are aligned on, and the orbits are the subject. Cut to about a
//! third of the box it reads as a centre, and the outer band — the only place an arrow can be big
//! enough to see — comes free.
//!
//! ## The cross is upright, and that is a ruling against the reference
//!
//! Fusion tips its cross a few degrees. Drawn that way at 15 pt it failed, and not marginally: an
//! axis-aligned stroke lands on the pixel grid, and a stroke a few degrees off it resolves as a
//! two-pixel grey smear, so both arms went soft at once. Tipping to a full 45° keeps them crisp but
//! hands the glyph the `cancel` silhouette. The lean the reference is asking for lives in the
//! orbits instead, where a curve is already off-grid everywhere and pays nothing for it.

use super::{IconPainter, TurningArrow};

/// Half an arm's length — about a third of the box. The cross registers the centre; it is not the
/// subject, and it must stay well inside the orbits.
const ARM_HALF_LENGTH: f32 = 1.9;
/// The orbit round the upright arm: wide and shallow, seen from a little above.
/// [`orbit_constrained`](super::orbit_constrained) draws the same one.
const UPRIGHT_ORBIT_RADII: (f32, f32) = (7.0, 4.3);
/// The orbit round the flat arm: the same ellipse stood on its end.
const FLAT_ORBIT_RADII: (f32, f32) = (4.3, 7.0);
/// The lean both orbits share.
const ORBIT_TILT: f32 = -0.18;
/// Each sweep leaves a gap where its own arm passes in front — the top for the upright orbit, the
/// right flank for the flat one — and the arrowhead ends at the far lip of that gap. The angle is
/// small because these ellipses are elongated: a quarter of a radian at the narrow end of one is
/// nearly three units of gap.
/// Each is centred on where the TILTED path crosses its arm, which is not the parameter's own top
/// or flank — tilt one and the two part company.
const UPRIGHT_ORBIT_SWEEP: (f32, f32) = (-1.26, 4.62);
const FLAT_ORBIT_SWEEP: (f32, f32) = (0.31, 6.20);
/// The arrowheads: longer and narrower than the sheet's static arrows, because at 15 pt a short
/// wide chevron with its own curve running past it closes into a blob.
const ARROW_HEAD: (f32, f32) = (2.6, 1.5);

pub(super) fn draw(g: &IconPainter) {
    let center = (9.0_f32, 9.0_f32);

    // The two axes: the registration mark, and the reason there is no world-up here.
    g.line(&[
        (center.0, center.1 - ARM_HALF_LENGTH),
        (center.0, center.1 + ARM_HALF_LENGTH),
    ]);
    g.line(&[
        (center.0 - ARM_HALF_LENGTH, center.1),
        (center.0 + ARM_HALF_LENGTH, center.1),
    ]);

    for (radii, sweep) in [
        (UPRIGHT_ORBIT_RADII, UPRIGHT_ORBIT_SWEEP),
        (FLAT_ORBIT_RADII, FLAT_ORBIT_SWEEP),
    ] {
        g.turning_arrow(TurningArrow {
            center,
            radii,
            sweep,
            tilt: ORBIT_TILT,
            head: ARROW_HEAD,
        });
    }
}
