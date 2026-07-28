//! `orbit-free` — the trackball: two rings threaded through each other, and no axis at all.
//!
//! Transpiled from the owner's SVG (`scratchpad/icons/owner/free-s22.svg`). The same conversion
//! note applies as for [`orbit_constrained`](super::orbit_constrained): the source opens its
//! ellipses with `stroke-dasharray` against `pathLength="360"`, which measures arc length, so
//! these sweeps came from integrating the ellipses rather than from the dash numbers.
//!
//! ## The two rings interlock — each passes in front exactly once
//!
//! That is the whole content of "free": no ring is privileged, so neither can be wholly in front.
//! The source says it with a small circular clip that swaps which mask wins in one corner; here it
//! is the same thing said as sweeps, since a painter has no masks. The wide ring is cut TWICE (the
//! tall one crosses in front on both sides of the swap) and the tall ring once.
//!
//! ## The cross is small, and that is the point of it
//!
//! It is not the subject; it is the registration mark the rings are aligned on, and its arms stop
//! well inside them. At a rod's length it swamps the box and the mark becomes
//! [`orbit_constrained`](super::orbit_constrained) with an extra stroke.
//!
//! It is axis-aligned rather than tipped. Tipping a few degrees puts both arms off the pixel grid
//! at once and each resolves as a two-pixel grey smear; a full 45° stays crisp but hands the glyph
//! the `cancel` silhouette.

use super::IconPainter;

const CENTER: (f32, f32) = (9.0, 9.0);
/// The wide ring — the same ellipse [`orbit_constrained`](super::orbit_constrained) draws, which
/// is what makes the pair read as one family rather than two unrelated marks.
const WIDE_RADII: (f32, f32) = (6.1875, 2.8125);
/// Its two visible runs. The gap between them is where the tall ring crosses in front; the ends
/// are the ring's own opening, which carries the arrowhead.
const WIDE_SWEEPS: [(f32, f32); 2] = [(5.3197, 8.0170), (8.5863, 10.2648)];
/// The tall ring: the same ellipse stood on its end.
const TALL_RADII: (f32, f32) = (2.8125, 6.1875);
/// One run — cut only where the wide ring wins the swap, at the lower right.
const TALL_SWEEP: (f32, f32) = (0.7317, 5.5663);
/// The two arrowheads, FILLED, their bearings a right angle apart. Two headings that far apart is
/// what says these are two different turns rather than one long swirl.
const WIDE_HEAD: [(f32, f32); 3] = [(11.5312, 7.0312), (13.3312, 5.8500), (13.3312, 8.2125)];
const TALL_HEAD: [(f32, f32); 3] = [(10.9688, 6.4688), (9.7875, 4.6688), (12.1500, 4.6688)];
/// Half an arm of the centre cross.
const ARM_HALF_LENGTH: f32 = 1.6875;

pub(super) fn draw(g: &IconPainter) {
    for (from, to) in WIDE_SWEEPS {
        g.arc(CENTER, WIDE_RADII.0, WIDE_RADII.1, from, to);
    }
    g.fill(&WIDE_HEAD);

    g.arc(
        CENTER,
        TALL_RADII.0,
        TALL_RADII.1,
        TALL_SWEEP.0,
        TALL_SWEEP.1,
    );
    g.fill(&TALL_HEAD);

    g.line(&[
        (CENTER.0 - ARM_HALF_LENGTH, CENTER.1),
        (CENTER.0 + ARM_HALF_LENGTH, CENTER.1),
    ]);
    g.line(&[
        (CENTER.0, CENTER.1 - ARM_HALF_LENGTH),
        (CENTER.0, CENTER.1 + ARM_HALF_LENGTH),
    ]);
}
