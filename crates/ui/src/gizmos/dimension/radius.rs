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
//!
//! **But the ray gives a DIRECTION, not a distance.** A circle drawn in a sketch plane is not a
//! circle on screen: the plane is projected, so unless it faces the camera the drawing is an
//! ellipse and a screen radius is right only in the one direction it happened to be measured. So
//! the point where the leader meets the curve is asked of [`Rim`], which knows where
//! the curve stands, and the radius handed in is a LENGTH the layout reasons about rather than a
//! distance anything steps out by.

use egui::{Pos2, Vec2};

use super::{
    arrowhead, value_width, Anchor, Drawing, Label, Piece, PlaneFrame, Rank, Rim, ARROW_LENGTH,
    GAP, OVERRUN,
};

/// Half the center mark's arm length — the cross that says "this is a center".
const CENTER_ARM: f32 = 4.0;

/// The extension that carries a curve round to where a leader along `ray` meets its circle, as
/// screen points — `None` when the curve already reaches there, and for a whole rim, which reaches
/// everywhere. Shared with [`diameter`](super::diameter()), which asks it twice.
///
/// Sampled along the rim rather than struck as a screen arc, so it lies ON the drawing at a plane
/// the camera is not square to instead of only where it was measured.
pub(super) fn carry(rim: Rim, radius: f32, ray: Vec2) -> Option<Piece> {
    // The overrun is stated in pixels like every other one, so it has to become a turn before an
    // arc can run through it — a fixed angle would overshoot a large curve and vanish on a small.
    let overrun = OVERRUN / radius.max(f32::EPSILON);
    let (from, to) = rim.carry_to(ray.y.atan2(ray.x), overrun)?;
    Some(Piece::Polyline(rim.between(from, to)))
}

/// A radius on the circle at `center`, dimensioned toward `anchor`.
///
/// Whether the anchor falls inside or outside the curve chooses between the two drawings, and
/// nothing else does: the anchor is dragged freely and the gizmo answers for wherever it lands.
///
/// `rim` says how much of the circle the curve draws and where it stands. An arc dimensioned
/// toward a bearing its own curve never reaches gets the extension that carries it there, in BOTH
/// drawings: the leader has to arrive at the geometry either way round.
///
/// `plane` lays the value out in the sketch rather than on the glass — the shoulder it lands on and
/// the direction it reads along are the PLANE's level, not the screen's.
pub fn radius(
    center: Pos2,
    radius: f32,
    anchor: Pos2,
    rim: Rim,
    plane: PlaneFrame,
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
    // The arc point, on the anchor's own ray. This is the derivation the module turns on — asked
    // of the rim, because the ray says which way the curve is and only the rim says how far.
    let bearing = ray.y.atan2(ray.x);
    let touch = rim.touch(bearing);
    // The arrow meets the curve SQUARE IN THE PLANE, which on a circle is along the radius — so
    // the ray the anchor was reached along IS the aim, and the leader that runs out along that
    // same ray stops dead on the arrow's base with no kink in it. Off the plane, the screen's
    // square of the ellipse would be a different direction and would put that kink back.
    let aim = rim.aim(bearing);
    // How far out the curve actually stands THERE, which is what decides inside from outside. The
    // nominal radius would flip the drawing early on one side of an ellipse and late on the other.
    let stands = (touch - center).length();

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
    if let Some(carried) = carry(rim, radius, ray) {
        pieces.push(carried);
    }

    let label = if distance < stands {
        // Leader runs center → arc with one arrow at the arc pointing OUT, and the text rides the
        // leader at its own angle rather than sitting horizontally across it.
        pieces.push(Piece::Polyline(vec![center, touch - aim * ARROW_LENGTH]));
        pieces.push(arrowhead(touch, aim));
        // The leader runs from the center to a point on the curve, so both of its ends are points
        // in the sketch and the line through them is already the image of a plane line — the value
        // rides it exactly.
        let seat = center + (touch - center) * 0.55;
        let reading = super::upright_direction(touch - center);
        Label {
            at: seat,
            text,
            along: reading,
            across: plane.square_to(reading, seat),
            anchor: Anchor::Middle,
            lift: GAP,
        }
    } else {
        // Leader leaves the arc along the SAME ray, jogs where the anchor is, and lands under the
        // text. The arrow reverses to point back at the curve.
        // The shoulder runs LEVEL IN THE PLANE, which on a tilted sketch is not level on screen.
        // Struck at the anchor, because a projection that divides carries the plane's own +X to a
        // different screen direction at every point.
        let reading = plane.reading_at(anchor);
        let side = if (anchor - center).dot(reading) >= 0.0 {
            1.0
        } else {
            -1.0
        };
        let away = reading * side;
        pieces.push(Piece::Polyline(vec![
            touch + aim * ARROW_LENGTH,
            anchor,
            anchor + away * (width + 2.0 * GAP),
        ]));
        pieces.push(arrowhead(touch, -aim));
        let seat = anchor + away * GAP;
        Label {
            at: seat,
            text,
            along: reading,
            across: plane.square_to(reading, seat),
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
