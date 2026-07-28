//! `orbit-free` — the trackball: a small cross, and two curved arrows going round it.
//!
//! One arrow drawn twice, half a turn from itself, round a cross. The comparison the eye makes on
//! the split button's face is [`orbit_constrained`](super::orbit_constrained)'s single rod through
//! a closed ring against this open pinwheel — because a turntable has an up and a trackball does
//! not. Both sit on the same radius, so the pair reads as one family.
//!
//! ## The cross is small, and that is the whole point of it
//!
//! It was first drawn at the sibling's length, and at that size it is two long strokes meeting in
//! the middle and it swamps the box. That is backwards. The cross is not the subject here; it is
//! the registration mark the arrows are aligned on, and the arrows are the subject. Cut to about a
//! third of the box it reads as a centre, and the outer band — the only place an arrow can be big
//! enough to see — comes free.
//!
//! ## The cross is upright, and that is a ruling against the reference
//!
//! Fusion tips its cross a few degrees, and this mark was drawn that way first: a 16° tip, so that
//! NEITHER arm claims the vertical the way a constrained orbit's axis does. At 15 pt it failed, and
//! not marginally. An axis-aligned stroke lands on the pixel grid; a stroke a few degrees off it
//! resolves as a two-pixel grey smear, so both arms went soft at once. Tipping to a full 45° keeps
//! them crisp but hands the glyph the `cancel` silhouette, which is not a trade worth making on a
//! navigation button. So the arms stay on the cardinals, and the two arrows carry the meaning.
//!
//! ## Half a turn apart, not a quarter
//!
//! The reference puts one small hooked arrow round each arm, and that was drawn: a little loop
//! wrapping the vertical arm, another wrapping the horizontal. It does not survive, for a reason
//! that is structural rather than a matter of tuning. Two loops on arms ninety degrees apart are
//! themselves ninety degrees apart, and a loop needs nearly that much angular room to be legible,
//! so they meet. Shrinking them until they clear each other makes each one a four-pixel squiggle;
//! rendered, that is a scribble at every size it was tried at.
//!
//! Two big arrows on one radius, half a turn apart, have the opposite property: each is long enough
//! to read as a curve, and the two are as far from each other as anything in the box can be. The
//! mark is symmetric under a half turn, so neither arm looks like the main one — which is exactly
//! what free orbit means.
//!
//! The literal "one arrow per axis" is what is given up. What is kept is what reading the mark
//! depends on: two arrows, each visibly an arrow, going round a centre that has no up.

use super::IconPainter;

/// Half an arm's length — about a third of the box, and deliberately far short of
/// [`orbit_constrained`](super::orbit_constrained)'s axis. The cross registers the centre; it is
/// not the subject, and it must stay well inside the arrows' radius.
const ARM_HALF_LENGTH: f32 = 3.0;
/// The arrows' radius — [`orbit_constrained`](super::orbit_constrained)'s ring radius, so the two
/// marks sit on a common circle and read as one family.
const ARC_RADIUS: f32 = 5.5;
const ARC_FROM: f32 = -0.30;
const ARC_TO: f32 = -2.70;
const ARROW_TRAIL: f32 = 2.9;
const ARROW_SPREAD: f32 = 1.6;

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

    // One arrow, and its twin half a turn away.
    for half_turn in [0.0, std::f32::consts::PI] {
        g.arrowed_arc(
            center,
            ARC_RADIUS,
            ARC_RADIUS,
            ARC_FROM + half_turn,
            ARC_TO + half_turn,
            (ARROW_TRAIL, ARROW_SPREAD),
        );
    }
}
