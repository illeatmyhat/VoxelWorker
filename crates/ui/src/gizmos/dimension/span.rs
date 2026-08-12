//! `axis_span` — a linear dimension between two points, along whatever direction it is measured in.
//!
//! The layout turns on TWO INDEPENDENT TESTS, and treating them as one is the mistake this file
//! exists to not make:
//!
//! | arrows fit | value fits | drawing |
//! |---|---|---|
//! | yes | yes | everything inside |
//! | yes | no | arrows stay in, the value moves out onto an extension |
//! | no | no | arrows flip outside pointing in, the value on the extension |
//!
//! The middle row is the one a single test loses. A span of 30 holds two 9-unit arrows with room
//! to spare, but the value has to clear BOTH ARROW BASES, not merely the span — so it leaves while
//! the arrows stay. There is no fourth row: a value that fits inside a span too short for its own
//! arrows cannot happen, because the arrow test is the weaker of the two.
//!
//! There is ONE entry point, because there is one drawing. An aligned span is the case where the
//! direction handed in is the run's own, and both extension lines come out the same length; a width
//! or a height hands in one of the plane's directions instead, and each end reaches the line by a
//! different amount with one of them possibly already sitting on it. A second entry point for the
//! aligned case would be the same code with a scalar offset in place of a point — which is exactly
//! the freedom the author loses when the value cannot be carried ALONG its own line.

use egui::{Pos2, Vec2};

use super::{
    arrowhead, value_width, Anchor, Drawing, Label, Piece, PlaneFrame, Rank, ARROW_LENGTH, GAP,
    OVERRUN,
};

/// How far apart two points stand ALONG ONE DIRECTION — a width or a height rather than a length.
///
/// `along` is that direction in screen pixels, and `through` a point the dimension line passes
/// through: where the author dropped the annotation. Each end reaches the line by its own
/// extension, which is what distinguishes this drawing from an aligned one — for a diagonal run
/// the two extension lines have different lengths, and for an axis-aligned run one of them vanishes
/// because the point is already on the line.
///
/// **`across` is the direction those extensions run, and the caller states it — ONE PER END.** It
/// is not `perp(along)`: a sketch plane the camera is not square to projects by a homography, and
/// the screen perpendicular of a projected direction is the image of some OTHER plane direction
/// entirely. At a three-quarter view the two are 31 degrees apart, and extension lines are the one
/// part of the drawing whose whole job is to read square — so a dimension laid out on the screen's
/// perpendicular reads as leaning out of the plane it annotates.
///
/// There are two because a projection that DIVIDES carries one plane direction to a different
/// screen direction at every point: square to the run at `from` and square to it at `to` are the
/// same direction in the plane and differ by 13 degrees on screen at the far end of a strongly
/// perspective view. Each end is asked at itself. A caller drawing on a flat page passes
/// `perp(along)` twice and gets exactly the drawing it always did.
///
/// `plane` is the same fact for everywhere ELSE the drawing needs it — the value's own baseline and
/// lift, which stand at a point the caller has not measured and cannot have. The two are kept apart
/// deliberately: `across` is measured at points with real plane coordinates and is exact there,
/// while the frame answers any point at all, and a drawing should take the sharper of the two where
/// it has it.
#[allow(clippy::too_many_arguments)]
pub fn axis_span(
    from: Pos2,
    to: Pos2,
    along: Vec2,
    across: [Vec2; 2],
    through: Pos2,
    plane: PlaneFrame,
    value: &str,
    rank: Rank,
) -> Drawing {
    let length = along.length();
    let unit = if length > f32::EPSILON {
        along / length
    } else {
        Vec2::X
    };
    // The screen's own square, which is what `across` reduces to on a plane facing the camera and
    // what the drawing falls back on when the caller hands in nothing usable.
    let screens_own = Vec2::new(unit.y, -unit.x);
    // Where a feature point meets the dimension line: the line struck through it along ITS OWN
    // `across`, met with the line struck through the anchor along `along`. Both are images of plane
    // lines, so the meeting point is the image of where they meet IN THE PLANE — which is the whole
    // reason the direction is handed in rather than turned out of `along` here.
    let foot = move |point: Pos2, across: Vec2| {
        let reach = across.length();
        let sideways = if reach > f32::EPSILON {
            across / reach
        } else {
            screens_own
        };
        // How far from parallel the two directions are. Seen edge-on they collapse together and no
        // meeting point exists; the caller declines before that, and the fallback here only has to
        // keep the drawing finite.
        let apart = unit.x.mul_add(sideways.y, -(unit.y * sideways.x));
        let toward = through - point;
        if apart.abs() <= f32::EPSILON {
            return point + screens_own * toward.dot(screens_own);
        }
        point + sideways * (toward.x.mul_add(unit.y, -(toward.y * unit.x)) / -apart)
    };

    let (near, far) = (foot(from, across[0]), foot(to, across[1]));
    let extensions = [(from, near), (to, far)]
        .into_iter()
        .filter_map(|(feature, meeting)| {
            let reach = meeting - feature;
            let distance = reach.length();
            // A point already on the dimension line needs no line drawn to it, and a stub of
            // length GAP + OVERRUN drawn anyway would read as a tick the drawing does not mean.
            (distance > GAP).then(|| {
                let toward = reach / distance;
                Piece::Polyline(vec![feature + toward * GAP, meeting + toward * OVERRUN])
            })
        })
        .collect();
    measured_between(near, far, extensions, through, plane, value, rank)
}

/// The dimension line itself, between the two points its extension lines reach — arrows, the two
/// fit tests, and the label that leaves when it does not fit.
///
/// `pieces` arrives holding the extension lines, because that is the one part the two callers
/// disagree about. `beside` is where the annotation stands, which is what decides where ALONG the
/// line the value rides and which end it leaves by when it does not fit — the author's own drop
/// where there is one, and otherwise the caller's default, which is the plane's middle and not the
/// chord's.
#[allow(clippy::too_many_arguments)]
fn measured_between(
    near: Pos2,
    far: Pos2,
    mut pieces: Vec<Piece>,
    beside: Pos2,
    plane: PlaneFrame,
    value: &str,
    rank: Rank,
) -> Drawing {
    let text = rank.indication("", value);
    let width = value_width(&text);

    let length = (far - near).length();
    let along = if length > f32::EPSILON {
        (far - near) / length
    } else {
        Vec2::X
    };

    // The two tests. Both are computed; neither is inferred from the other.
    let arrows_fit = length >= 2.0 * ARROW_LENGTH + 2.0;
    let value_fits = length >= 2.0 * ARROW_LENGTH + width + 2.0 * GAP;

    if arrows_fit {
        // The dimension line stops at the arrow BASES, so nothing pokes past a tip.
        pieces.push(Piece::Polyline(vec![
            near + along * ARROW_LENGTH,
            far - along * ARROW_LENGTH,
        ]));
        pieces.push(arrowhead(near, -along));
        pieces.push(arrowhead(far, along));
    } else {
        // Arrows outside pointing in. The line stays continuous between the extension lines, so
        // the dimension still reads as spanning them rather than as two detached ticks.
        pieces.push(Piece::Polyline(vec![near, far]));
        pieces.push(Piece::Polyline(vec![
            near - along * 2.0 * ARROW_LENGTH,
            near,
        ]));
        pieces.push(arrowhead(near, along));
        pieces.push(arrowhead(far, -along));
    }

    // The run's own direction, folded so the value reads from above. Already a plane direction:
    // both ends are points in the sketch, so the line through them is the image of a plane line.
    let reading = super::upright_direction(along);
    // How far along the line the value stands, measured from `near`. A screen distance read off a
    // screen point, so it round-trips exactly through `near + along * put` below — the station is
    // the anchor's own foot on the line, wherever the anchor came by it.
    let put = (beside - near).dot(along);
    // The room a value needs to ride inline: half of itself, clear of the arrowhead beside it.
    // At the middle of a span the value fits in, this is exactly `value_fits` — which is why an
    // unplaced drawing lands on the same layout it always did.
    let room = width / 2.0 + ARROW_LENGTH + GAP;
    let label = if value_fits && arrows_fit && put >= room && put <= length - room {
        let at = near + along * put;
        Label {
            at,
            text,
            along: reading,
            across: plane.square_to(reading, at),
            anchor: Anchor::Middle,
            lift: GAP,
        }
    } else {
        // It leaves by the end it was dropped past, on a leader that REACHES it — never a naked
        // line running off to a number floating somewhere past its end, and never a number left
        // behind at the line when the hand carried it further.
        let by_the_near_end = put < length / 2.0;
        let (end, direction, dropped) = if by_the_near_end {
            (near, -along, -put)
        } else {
            (far, along, put - length)
        };
        let out = dropped.max(width + 2.0 * GAP);
        pieces.push(Piece::Polyline(vec![end, end + direction * out]));
        // The value reads left to right whichever end it left by, so which of its two edges lands
        // on the leader depends on whether the reading direction agrees with the way out.
        let at = end + direction * (out - width - GAP);
        Label {
            at,
            text,
            along: reading,
            across: plane.square_to(reading, at),
            anchor: if reading.dot(direction) >= 0.0 {
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
