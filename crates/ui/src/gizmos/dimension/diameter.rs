//! `diameter` — a size dimension struck straight through a circle's center.
//!
//! **The same claim as [`radius`](super::radius()), drawn the other way.** A radius leaves the center
//! and stops at the rim; a diameter crosses from rim to rim and passes THROUGH the center, which is
//! the whole of what the drawing has to say differently. Which one an author sees is theirs to
//! choose — a hole is read across and a fillet is read out — so the two gizmos exist beside each
//! other rather than one being derived from the other at paint time.
//!
//! The ray is derived from the anchor exactly the way `radius` derives its touch point, and for the
//! same reason: no anchor position can put the line off the center if the center is what the line
//! is built from.
//!
//! **The prefix is `D`, not the drafting glyph.** `⌀` is an icon spelled as a character, and the
//! value type is monospace so a font without it would both draw tofu and measure the label wrong.

use egui::{Pos2, Vec2};

use super::{arrowhead, value_width, Anchor, Drawing, Label, Piece, Rank, Rim, ARROW_LENGTH, GAP};

/// Half the center mark's arm length, matching [`radius`](super::radius())'s cross.
const CENTER_ARM: f32 = 4.0;

/// A diameter across the circle at `center`, struck along the ray toward `anchor`.
///
/// The one test is whether the value clears both arrow bases on the through-line. It cannot be
/// inferred from whether the arrows themselves fit: a circle wide enough for two arrows is not
/// wide enough for two arrows and a number between them, and that is precisely the case a single
/// test loses — the same middle row [`span`](super::span()) documents.
///
/// `rim` is [`radius`](super::radius())'s, asked TWICE: a through-line meets the circle at both
/// ends, so an arc read across can fall short of its own drawing at either or both of them.
pub fn diameter(
    center: Pos2,
    radius: f32,
    anchor: Pos2,
    rim: Option<Rim>,
    value: &str,
    rank: Rank,
) -> Drawing {
    let text = rank.indication("D", value);
    let width = value_width(&text);

    let reach = anchor - center;
    let distance = reach.length();
    let ray = if distance > f32::EPSILON {
        reach / distance
    } else {
        Vec2::X
    };
    let (near, far) = (center - ray * radius, center + ray * radius);
    let across = 2.0 * radius;

    // A center CROSS in dimension ink — a filled dot is what a sketch point looks like, and the
    // center a diameter passes through is not one.
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
    for reaching in [ray, -ray] {
        if let Some((from, to)) = super::radius::carry(rim, radius, reaching) {
            pieces.push(Piece::Arc {
                center,
                radius,
                from,
                to,
            });
        }
    }

    let label = if across >= 2.0 * ARROW_LENGTH + width + 2.0 * GAP {
        // Everything inside: the line stops at the arrow bases so nothing pokes past a tip, and
        // the value rides the line at its own angle rather than sitting horizontally across it.
        pieces.push(Piece::Polyline(vec![
            near + ray * ARROW_LENGTH,
            far - ray * ARROW_LENGTH,
        ]));
        pieces.push(arrowhead(near, -ray));
        pieces.push(arrowhead(far, ray));
        Label {
            at: center,
            text,
            radians: super::upright_radians(ray.y.atan2(ray.x)),
            anchor: Anchor::Middle,
            lift: GAP,
        }
    } else {
        // Too tight to read across: the arrows flip outside pointing in, the through-line stays
        // continuous so it still reads as crossing the circle, and the value leaves on a leader
        // that carries on along the SAME ray to wherever the anchor was dragged.
        let side = if anchor.x >= center.x { 1.0 } else { -1.0 };
        // The leader ends at the anchor, or at the rim if the anchor was dragged back inside:
        // a leader that doubled back would point into the circle it had just left.
        let stop = if (anchor - far).dot(ray) > 0.0 {
            anchor
        } else {
            far
        };
        pieces.push(Piece::Polyline(vec![near - ray * 2.0 * ARROW_LENGTH, near]));
        pieces.push(Piece::Polyline(vec![
            near,
            stop,
            stop + Vec2::X * side * (width + 2.0 * GAP),
        ]));
        pieces.push(arrowhead(near, ray));
        pieces.push(arrowhead(far, -ray));
        Label {
            at: stop + Vec2::X * side * GAP,
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
