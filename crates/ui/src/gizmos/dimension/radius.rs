//! `radius` — a radial dimension on an arc or circle.
//!
//! **The arc point is derived from the anchor, never stored beside it.** The anchor is the only
//! input the user drags; the point where the leader meets the curve is
//! `centre + radius·(cos a, sin a)` for `a = atan2(anchor − centre)`, and the jog is the anchor
//! projected onto that same ray. A radius is only a radius because its leader points at the
//! centre — deriving the arc point means no anchor position can break that, whereas storing the
//! two beside each other means every edit has to remember to keep them agreeing.
//!
//! That is the same discipline as ADR 0008 and the anchored-DDA law: carry the frame, never
//! re-derive a value that has to agree with another.

use egui::{Pos2, Vec2};

use super::{arrowhead, value_width, Anchor, Drawing, Label, Piece, Rank, ARROW_LENGTH, GAP};

/// Half the centre mark's arm length — the cross that says "this is a centre".
const CENTRE_ARM: f32 = 4.0;

/// A radius on the circle at `center`, dimensioned toward `anchor`.
///
/// Whether the anchor falls inside or outside the curve chooses between the two drawings, and
/// nothing else does: the anchor is dragged freely and the gizmo answers for wherever it lands.
pub fn radius(center: Pos2, radius: f32, anchor: Pos2, value: &str, rank: Rank) -> Drawing {
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

    // A centre CROSS in dimension ink, not a filled dot in geometry ink — a filled dot is what a
    // sketch point looks like, and the centre of a dimensioned circle is not one.
    let mut pieces = vec![
        Piece::Polyline(vec![
            center - Vec2::X * CENTRE_ARM,
            center + Vec2::X * CENTRE_ARM,
        ]),
        Piece::Polyline(vec![
            center - Vec2::Y * CENTRE_ARM,
            center + Vec2::Y * CENTRE_ARM,
        ]),
    ];

    let label = if distance < radius {
        // Leader runs centre → arc with one arrow at the arc pointing OUT, and the text rides the
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
