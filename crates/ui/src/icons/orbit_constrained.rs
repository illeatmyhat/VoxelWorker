//! `orbit-constrained` — a ring turning about the one axis the camera keeps upright.
//!
//! Transpiled from the owner's SVG (`scratchpad/icons/owner/constrained-s22.svg`), which derives
//! this from the free mark by taking one ring away and standing it on its axis. The numbers below
//! are that file's, converted once: it opens its ellipse with `stroke-dasharray` against
//! `pathLength="360"`, which measures ARC LENGTH, not the parametric angle [`IconPainter::arc`]
//! takes — so the sweep here came from integrating the ellipse, not from reading the dash off.
//!
//! ## Depth is occlusion, not skew
//!
//! The ellipse is not tilted. Earlier drafts leaned it to suggest perspective and every one read as
//! a swirl; what actually says "a ring around a rod" is one stroke stopping where another passes in
//! front of it. That costs a gap and buys the third dimension.
//!
//! ## Only ONE cut is real
//!
//! The ring's opening already sits at the top, exactly where the axis would have stood in front of
//! it — so cutting the ring there removes nothing. What is left is the near crossing: at the bottom
//! the ring passes in FRONT, so the AXIS breaks and the ring stays whole.

use super::IconPainter;

const CENTER: (f32, f32) = (9.0, 9.0);
/// Wide and shallow: a circle about the vertical axis, seen from a little above.
const RING_RADII: (f32, f32) = (6.1875, 2.8125);
/// The opening, at the top, where the axis passes in front and the arrowhead lands.
const RING_SWEEP: (f32, f32) = (5.3197, 10.2950);
/// The arrowhead, FILLED — a stroked chevron this size closes into a blob at rail size.
const ARROW_HEAD: [(f32, f32); 3] = [(11.5312, 7.0312), (13.3312, 5.8500), (13.3312, 8.2125)];
/// The axis, in two runs; the gap is where the ring's near side crosses in front. It spans the
/// free mark's tall ring, so the pair fills one box, and overruns the ring at both ends — the
/// overhang is what reads as "through" rather than "attached to".
const AXIS_X: f32 = 9.0;
const AXIS_FAR: (f32, f32) = (2.8125, 10.3458);
const AXIS_NEAR: (f32, f32) = (13.2817, 15.1875);

pub(super) fn draw(g: &IconPainter) {
    g.line(&[(AXIS_X, AXIS_FAR.0), (AXIS_X, AXIS_FAR.1)]);
    g.line(&[(AXIS_X, AXIS_NEAR.0), (AXIS_X, AXIS_NEAR.1)]);
    g.arc(
        CENTER,
        RING_RADII.0,
        RING_RADII.1,
        RING_SWEEP.0,
        RING_SWEEP.1,
    );
    g.fill(&ARROW_HEAD);
}
