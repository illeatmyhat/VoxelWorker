//! `radius` — a radial dimension on an arc or circle.
//!
//! **The arc point is derived from the anchor, never stored beside it.** The anchor is the only
//! input the user drags; the point where the leader meets the curve is
//! `center + radius·(cos a, sin a)` for `a = atan2(anchor − center)`, and the jog is the anchor
//! projected onto that same ray. A radius is only a radius because its leader points at the
//! center — deriving the arc point means no anchor position can break that, whereas storing the
//! two beside each other means every edit has to remember to keep them agreeing.
//!
//! That is the same discipline the spatial frames keep: carry the value, never re-derive one
//! that has to agree with another.

use egui::{Pos2, Vec2};

use super::{
    arrowhead, value_width, Anchor, Drawing, Label, Piece, Rank, Rim, ARROW_LENGTH, GAP, OVERRUN,
};

/// Half the center mark's arm length — the cross that says "this is a center".
const CENTER_ARM: f32 = 4.0;

/// The extension that carries a curve round to where a leader along `ray` meets its circle, as
/// `(from, to)` bearings — `None` when the curve already reaches there, and for a whole circle,
/// which reaches everywhere. Shared with [`diameter`](super::diameter()), which asks it twice.
pub(super) fn carry(rim: Option<Rim>, radius: f32, ray: Vec2) -> Option<(f32, f32)> {
    // The overrun is stated in pixels like every other one, so it has to become a turn before an
    // arc can run through it — a fixed angle would overshoot a large curve and vanish on a small.
    let overrun = OVERRUN / radius.max(f32::EPSILON);
    rim?.carry_to(ray.y.atan2(ray.x), overrun)
}

/// A radius on the circle at `center`, dimensioned toward `anchor`.
///
/// Whether the anchor falls inside or outside the curve chooses between the two drawings, and
/// nothing else does: the anchor is dragged freely and the gizmo answers for wherever it lands.
///
/// `rim` says how much of the circle the curve itself draws — `None` for a whole circle. An arc
/// dimensioned toward a bearing its own curve never reaches gets the extension that carries it
/// there, in BOTH drawings: the leader has to arrive at the geometry either way round.
pub fn radius(
    center: Pos2,
    radius: f32,
    anchor: Pos2,
    rim: Option<Rim>,
    value: &str,
    rank: Rank,
) -> Drawing {
    let text = rank.indication("R", value);
    let width = value_width(&text);

    let reach = anchor - center;
    let distance = reach.length();
    let ray = if distance > f32::EPSILON {
        reach / distance
    } else {
        Vec2::X
    };
    // The arc point, on the anchor's own ray. This is the derivation the module turns on.
    let touch = center + ray * radius;

    // A center CROSS in dimension ink, not a filled dot in geometry ink — a filled dot is what a
    // sketch point looks like, and the center of a dimensioned circle is not one.
    let mut pieces = vec![
        Piece::Polyline(vec![
            center - Vec2::X * CENTER_ARM,
            center + Vec2::X * CENTER_ARM,
        ]),
        Piece::Polyline(vec![
            center - Vec2::Y * CENTER_ARM,
            center + Vec2::Y * CENTER_ARM,
        ]),
    ];
    if let Some((from, to)) = carry(rim, radius, ray) {
        pieces.push(Piece::Arc {
            center,
            radius,
            from,
            to,
        });
    }

    let label = if distance < radius {
        // Leader runs center → arc with one arrow at the arc pointing OUT, and the text rides the
        // leader at its own angle rather than sitting horizontally across it.
        pieces.push(Piece::Polyline(vec![center, touch - ray * ARROW_LENGTH]));
        pieces.push(arrowhead(touch, ray));
        Label {
            at: center + ray * radius * 0.55,
            text,
            radians: super::upright_radians(ray.y.atan2(ray.x)),
            anchor: Anchor::Middle,
            lift: GAP,
        }
    } else {
        // Leader leaves the arc along the SAME ray, jogs where the anchor is, and lands under the
        // text. The arrow reverses to point back at the curve.
        let side = if anchor.x >= center.x { 1.0 } else { -1.0 };
        pieces.push(Piece::Polyline(vec![
            touch + ray * ARROW_LENGTH,
            anchor,
            anchor + Vec2::X * side * (width + 2.0 * GAP),
        ]));
        pieces.push(arrowhead(touch, -ray));
        Label {
            at: anchor + Vec2::X * side * GAP,
            text,
            radians: 0.0,
            anchor: if side > 0.0 {
                Anchor::Start
            } else {
                Anchor::End
            },
            lift: GAP,
        }
    };

    Drawing {
        pieces,
        labels: vec![label],
        rank,
    }
}
