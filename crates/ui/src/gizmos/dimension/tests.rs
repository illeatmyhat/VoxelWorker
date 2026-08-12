//! The layout rules, gated without a painter.
//!
//! Everything asserted here is a decision the design sheet argues for and that a rewrite could
//! silently undo: the three span states, the radial derivation, the upright fold, and the
//! parenthesis wrapping the whole indication.

#![allow(
    clippy::duration_subsec,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::match_same_arms,
    clippy::panic,
    clippy::semicolon_if_nothing_returned,
    clippy::unwrap_used,
    clippy::while_float
)]

use egui::{Pos2, Vec2};

use super::*;

/// Count the arrowheads, and answer whether they point toward each other or away.
fn heads(drawing: &Drawing) -> Vec<[Pos2; 4]> {
    drawing
        .pieces
        .iter()
        .filter_map(|piece| match piece {
            Piece::Head(points) => Some(*points),
            Piece::Polyline(_) => None,
        })
        .collect()
}

/// A span whose dimension line runs parallel to its own run, `offset` away along the normal —
/// the shape most of these tests are about, spelled once so each call says what it tests.
fn span(from: Pos2, to: Pos2, offset: f32, value: &str, rank: Rank) -> Drawing {
    let run = to - from;
    let length = run.length();
    let along = if length > f32::EPSILON {
        run / length
    } else {
        Vec2::X
    };
    let normal = Vec2::new(along.y, -along.x);
    // Offset from the MIDDLE of the run, which is where a span with nothing placed
    // has always carried its value.
    let middle = from + run / 2.0;
    axis_span(
        from,
        to,
        along,
        [normal; 2],
        middle + normal * offset,
        value,
        rank,
    )
}

/// A head's nose: the midpoint of the flat its point was cut off at.
fn nose(head: [Pos2; 4]) -> Pos2 {
    head[0] + (head[1] - head[0]) / 2.0
}

/// A head's direction: nose minus base midpoint, normalized.
fn aim(head: [Pos2; 4]) -> Vec2 {
    let base = head[3] + (head[2] - head[3]) / 2.0;
    (nose(head) - base).normalized()
}

/// The point a head NAMES — where its own point would have been, a setback on from the nose it
/// was cut back to so that its halo could stop at the line rather than spike through it.
fn points_at(head: [Pos2; 4]) -> Pos2 {
    nose(head) + aim(head) * ARROW_SETBACK
}

/// Where a rim stands, on a flat page. These tests have no projection, so a circle really is a
/// circle and stands the same distance out at every bearing — which is exactly the assumption the
/// app cannot make, and the reason a rim is asked rather than stepped out to.
fn round(center: Pos2, radius: f32) -> impl Fn(f32) -> Pos2 {
    move |bearing| center + Vec2::angled(bearing) * radius
}

/// The arc an angle is drawn on, on a flat page. A circle really is a circle here, which is what a
/// sketch plane facing the camera projects to, and the one view that cannot tell an arc asked of
/// its rim apart from one struck at a screen radius.
fn struck(vertex: Pos2, radius: f32) -> impl Fn(f32) -> Pos2 {
    round(vertex, radius)
}

/// A rim that draws the whole of its own circle, so it falls short of nothing.
fn whole(at: &dyn Fn(f32) -> Pos2) -> Rim<'_> {
    Rim {
        from: 0.0,
        turn: std::f32::consts::TAU,
        at,
    }
}

#[test]
fn a_roomy_span_keeps_everything_inside() {
    let drawing = span(
        Pos2::new(44.0, 28.0),
        Pos2::new(216.0, 28.0),
        0.0,
        "172",
        Rank::Driving,
    );
    assert_eq!(drawing.labels.len(), 1);
    let label = &drawing.labels[0];
    assert_eq!(label.anchor, Anchor::Middle, "the value sits on the line");
    assert_eq!(label.at, Pos2::new(130.0, 28.0), "at the span's middle");

    let arrows = heads(&drawing);
    assert_eq!(arrows.len(), 2);
    // Tips at the extension lines, pointing outward at each other's origin.
    assert_eq!(points_at(arrows[0]), Pos2::new(44.0, 28.0));
    assert_eq!(points_at(arrows[1]), Pos2::new(216.0, 28.0));
    assert!(
        aim(arrows[0]).x < 0.0 && aim(arrows[1]).x > 0.0,
        "pointing out"
    );
}

/// The state a single fit test loses, and the reason there are two.
#[test]
fn a_span_can_hold_its_arrows_and_still_evict_its_value() {
    let drawing = span(
        Pos2::new(115.0, 28.0),
        Pos2::new(145.0, 28.0),
        0.0,
        "30",
        Rank::Driving,
    );
    // 30 units clears 2 * 9 + 2 = 20, so the arrows stay in and still point outward...
    let arrows = heads(&drawing);
    assert!(
        aim(arrows[0]).x < 0.0 && aim(arrows[1]).x > 0.0,
        "the arrows did not flip: the span holds them",
    );
    // ...but the value needs 2 * 9 + width("30") + 2 * 5 = 41.2, which 30 does not clear.
    let label = &drawing.labels[0];
    assert_eq!(
        label.anchor,
        Anchor::Start,
        "the value left on an extension"
    );
    assert!(label.at.x > 145.0, "and it left past the far end");
}

#[test]
fn a_span_too_short_for_its_arrows_flips_them_outward() {
    let drawing = span(
        Pos2::new(123.0, 28.0),
        Pos2::new(137.0, 28.0),
        0.0,
        "14",
        Rank::Driving,
    );
    let arrows = heads(&drawing);
    assert!(
        aim(arrows[0]).x > 0.0 && aim(arrows[1]).x < 0.0,
        "outside, pointing in",
    );
    assert_eq!(drawing.labels[0].anchor, Anchor::Start);
}

/// There is no fourth state: the value test is strictly the stronger of the two, so a span that
/// fails the arrow test cannot pass the value one at any width.
#[test]
fn the_value_test_is_never_the_looser_of_the_two() {
    for length in 1..200 {
        let length = length as f32;
        let drawing = span(Pos2::ZERO, Pos2::new(length, 0.0), 0.0, "8", Rank::Driving);
        let arrows = heads(&drawing);
        let arrows_fit = aim(arrows[0]).x < 0.0;
        let value_inside = drawing.labels[0].anchor == Anchor::Middle;
        assert!(
            arrows_fit || !value_inside,
            "at length {length} the value stayed in a span its own arrows do not fit",
        );
    }
}

#[test]
fn a_span_lays_out_in_its_own_frame_not_the_screens() {
    // The same span, turned a quarter turn: the value's anchor must ride round with it.
    let drawing = span(
        Pos2::new(50.0, 200.0),
        Pos2::new(50.0, 40.0),
        12.0,
        "160",
        Rank::Driving,
    );
    let label = &drawing.labels[0];
    assert_eq!(label.anchor, Anchor::Middle);
    assert!(
        (label.at.x - 38.0).abs() < 1e-4,
        "the dimension line sits 12 off along the normal, not along +x: {:?}",
        label.at,
    );
    assert!((label.at.y - 120.0).abs() < 1e-4, "{:?}", label.at);
}

/// **An extent's two extension lines are different lengths**, which is the whole reason it cannot
/// borrow the aligned span's drawing.
///
/// The run goes (20,100) to (100,60) and the dimension line is horizontal at y = 30, so the near
/// end reaches it by 70 and the far end by 30. An aligned span would have drawn both the same.
#[test]
fn an_extent_reaches_its_dimension_line_by_a_different_amount_at_each_end() {
    let drawing = axis_span(
        Pos2::new(20.0, 100.0),
        Pos2::new(100.0, 60.0),
        Vec2::X,
        [Vec2::Y; 2],
        Pos2::new(60.0, 30.0),
        "80",
        Rank::Driving,
    );
    let extensions: Vec<f32> = drawing
        .pieces
        .iter()
        .filter_map(|piece| match piece {
            Piece::Polyline(points) if points.len() == 2 => ((points[0].x - points[1].x).abs()
                < 1e-4)
                .then(|| (points[0].y - points[1].y).abs()),
            _ => None,
        })
        .collect();
    assert_eq!(
        extensions.len(),
        2,
        "one extension line per end: {extensions:?}"
    );
    // 70 and 30, each shortened by the GAP off the feature and lengthened by the OVERRUN past the
    // line — so the two differ by the 40 the run rises, whatever those constants are.
    let (long, short) = (
        extensions[0].max(extensions[1]),
        extensions[0].min(extensions[1]),
    );
    assert!(
        (long - short - 40.0).abs() < 1e-4,
        "the ends reach by {long} and {short}, which should differ by the rise"
    );
    // And the dimension line itself lies along the axis it measures, not along the run.
    let label = &drawing.labels[0];
    assert!((label.at.y - 30.0).abs() < 1e-4, "{:?}", label.at);
    assert!(
        label.radians.abs() < 1e-4,
        "a width reads level: {}",
        label.radians
    );
}

/// An end already standing on the dimension line grows no extension to it — a stub of pure GAP and
/// OVERRUN would read as a tick mark the drawing does not mean.
#[test]
fn an_end_on_the_dimension_line_grows_no_extension() {
    let drawing = axis_span(
        Pos2::new(20.0, 30.0),
        Pos2::new(100.0, 60.0),
        Vec2::X,
        [Vec2::Y; 2],
        Pos2::new(60.0, 30.0),
        "80",
        Rank::Driving,
    );
    let verticals = drawing
        .pieces
        .iter()
        .filter(|piece| match piece {
            Piece::Polyline(points) if points.len() == 2 => {
                (points[0].x - points[1].x).abs() < 1e-4
            }
            _ => false,
        })
        .count();
    assert_eq!(verticals, 1, "only the end that is off the line reaches it");
}

/// The module's central claim: the arc point is on the anchor's ray, whatever the anchor.
#[test]
fn the_radial_leader_cannot_be_made_non_radial() {
    let center = Pos2::new(92.0, 80.0);
    let radius_length = 21.0;
    let standing = round(center, radius_length);
    for anchor in [
        Pos2::new(156.0, 44.0),
        Pos2::new(20.0, 150.0),
        Pos2::new(92.0, 12.0),
        Pos2::new(95.0, 82.0),
    ] {
        let drawing = radius(
            center,
            radius_length,
            anchor,
            whole(&standing),
            "21",
            Rank::Driving,
        );
        let head = heads(&drawing)[0];
        let touch = points_at(head);
        // The arrow's tip is the arc point: it must be exactly `radius` from the center, and
        // exactly on the ray through the anchor.
        assert!(
            ((touch - center).length() - radius_length).abs() < 1e-3,
            "the leader met the curve at the wrong distance for anchor {anchor:?}",
        );
        let ray = (anchor - center).normalized();
        let along = (touch - center).normalized();
        assert!(
            (ray.x - along.x).abs() < 1e-3 && (ray.y - along.y).abs() < 1e-3,
            "the arc point left the anchor's ray for anchor {anchor:?}",
        );
    }
}

#[test]
fn an_anchor_inside_the_curve_points_the_arrow_outward() {
    let center = Pos2::new(130.0, 74.0);
    let (far_out, close_in) = (round(center, 46.0), round(center, 21.0));
    let inside = radius(
        center,
        46.0,
        Pos2::new(152.0, 56.0),
        whole(&far_out),
        "46",
        Rank::Driving,
    );
    let outward = aim(heads(&inside)[0]);
    let ray = (Pos2::new(152.0, 56.0) - center).normalized();
    assert!(
        outward.dot(ray) > 0.9,
        "inside: the arrow points out at the curve"
    );

    let outside = radius(
        center,
        21.0,
        Pos2::new(196.0, 24.0),
        whole(&close_in),
        "21",
        Rank::Driving,
    );
    let inward = aim(heads(&outside)[0]);
    let ray = (Pos2::new(196.0, 24.0) - center).normalized();
    assert!(inward.dot(ray) < -0.9, "outside: it reverses to point back");
}

/// **A leader has to arrive at the drawing, not at the circle the drawing lies on.**
///
/// An arc occupies part of its circle, and the ray the author drags the annotation along can meet
/// that circle anywhere. Where it meets a bearing the curve itself never reaches, the curve is
/// carried round to it — from the nearer of its two ends, whichever side that is on, and in both
/// the inside and the outside drawing, because the leader touches the rim either way.
#[test]
fn a_radius_carries_its_arc_round_to_a_leader_the_curve_does_not_reach() {
    let center = Pos2::new(100.0, 100.0);
    let quarter = std::f32::consts::FRAC_PI_2;
    // A quarter of the circle, from due east round to due south (screen y runs down).
    let standing = round(center, 40.0);
    let rim = Rim {
        from: 0.0,
        turn: quarter,
        at: &standing,
    };
    // The carried extension is the piece that runs ALONG the rim: it is sampled from the curve, so
    // every one of its points stands the rim's own distance out, which no other piece does.
    let extensions = |drawing: &Drawing| -> Vec<(f32, f32)> {
        drawing
            .pieces
            .iter()
            .filter_map(|piece| {
                let Piece::Polyline(points) = piece else {
                    return None;
                };
                let on_the_rim = points
                    .iter()
                    .all(|at| ((*at - center).length() - 40.0).abs() < 1e-2);
                if points.len() > 2 && on_the_rim {
                    let bearing = |at: Pos2| (at - center).angle();
                    Some((bearing(points[0]), bearing(points[points.len() - 1])))
                } else {
                    None
                }
            })
            .collect()
    };
    let struck = |at: Pos2| radius(center, 40.0, at, rim, "40", Rank::Driving);

    // Dropped over the curve's own quarter: nothing to carry.
    assert!(
        extensions(&struck(Pos2::new(150.0, 150.0))).is_empty(),
        "the curve already reaches there"
    );

    // Dropped just past the far end: the extension leaves that end and runs on.
    let past = extensions(&struck(Pos2::new(90.0, 160.0)));
    assert_eq!(past.len(), 1, "one extension, from one end");
    assert!(
        (past[0].0 - quarter).abs() < 1e-4,
        "it starts at the end the curve stops at: {:?}",
        past[0]
    );
    assert!(past[0].1 > past[0].0, "and carries on the same way round");

    // Dropped short of the near end: the same rule the other way — the extension runs BACKWARD
    // out of the start, because that is the end nearer what the leader is asking for.
    let short = extensions(&struck(Pos2::new(160.0, 90.0)));
    assert_eq!(short.len(), 1);
    assert!(
        (short[0].0).abs() < 1e-4,
        "out of the start: {:?}",
        short[0]
    );
    assert!(short[0].1 < short[0].0, "and the other way round");

    // The inside drawing is not exempt: the leader still ends on the rim.
    let inside = extensions(&radius(
        center,
        40.0,
        Pos2::new(94.0, 118.0),
        rim,
        "40",
        Rank::Driving,
    ));
    assert_eq!(
        inside.len(),
        1,
        "an anchor inside still needs the rim {inside:?}"
    );

    // A whole circle has no end to fall short of.
    assert!(extensions(&radius(
        center,
        40.0,
        Pos2::new(40.0, 40.0),
        whole(&standing),
        "40",
        Rank::Driving,
    ))
    .is_empty());
}

/// **A terminator's point is cut off where it meets what it points at.**
///
/// The halo is a stroke and a stroke MITRES: at a sharp arrowhead's 19-degree apex it ran
/// `(halo / 2) / sin(9.5°)` ≈ 7 past the point — a background spike as long as the arrow, laid
/// through the very line the arrow was terminating on, and longer for every bit of halo added.
/// Cutting the nose back by a setback leaves no corner sharp enough to spike, and puts the halo's
/// own leading edge where the point would have been.
#[test]
fn an_arrowhead_stops_at_the_line_it_points_at() {
    let at = Pos2::new(60.0, 40.0);
    for bearing in [0.0_f32, 1.1, 2.7, -2.0] {
        let direction = Vec2::angled(bearing);
        let Piece::Head(head) = arrowhead(at, direction) else {
            panic!("an arrowhead is a head")
        };
        // The ink stops a setback short, and still names the line it stopped at.
        for corner in head {
            assert!(
                (corner - at).dot(direction) <= -ARROW_SETBACK + 1e-3,
                "{corner:?} crossed the line the arrow terminates on"
            );
        }
        assert!((points_at(head) - at).length() < 1e-3, "it names its line");
        assert!(aim(head).dot(direction) > 0.999, "and still points at it");

        // No corner is sharp enough for the stroked halo to mitre into a spike. The reach a mitre
        // adds is `(halo / 2) / sin(corner / 2)`, which is the whole of what went wrong before.
        for index in 0..4 {
            let (here, before, after) = (head[index], head[(index + 3) % 4], head[(index + 1) % 4]);
            let corner = (before - here)
                .normalized()
                .dot((after - here).normalized())
                .acos();
            let reach = (HALO_WIDTH / 2.0) / (corner / 2.0).sin();
            assert!(
                reach <= HALO_WIDTH,
                "corner {index} mitres {reach} past itself"
            );
        }
    }
}

/// **A mark that lands on the curve is AIMED by the curve.** A rim on a plane the camera is not
/// square to draws an ellipse, and there the ray an annotation was dragged along is not the
/// direction the curve faces — they part by as much as the tilt. An arrowhead aimed along the ray
/// lies ACROSS its own drawing instead of meeting it, which is what a slanted radius looked like.
#[test]
fn an_arrowhead_meets_a_slanted_rim_square_to_it() {
    let center = Pos2::new(100.0, 100.0);
    let (wide, tall) = (120.0_f32, 30.0_f32);
    // The ellipse point ON the ray at a bearing — the convention a projected ring answers in.
    let squashed = move |bearing: f32| {
        let (sine, cosine) = bearing.sin_cos();
        center + Vec2::new(cosine, sine) / (cosine / wide).hypot(sine / tall)
    };
    let rim = whole(&squashed);

    // Along either axis the ray IS the way the curve faces, so nothing moves.
    for square_on in [0.0, std::f32::consts::FRAC_PI_2] {
        assert!(
            rim.aim(square_on).dot(Vec2::angled(square_on)) > 0.999,
            "on an axis the two directions agree"
        );
    }
    // Off them they part, and the aim is the one perpendicular to the drawing.
    let slant = std::f32::consts::FRAC_PI_4;
    let facing = rim.aim(slant);
    assert!(
        facing.dot(Vec2::angled(slant)) < 0.9,
        "the ray is not the way the curve faces here: {facing:?}"
    );
    let along = (squashed(slant + 0.005) - squashed(slant - 0.005)).normalized();
    assert!(
        facing.dot(along).abs() < 1e-2,
        "the aim left square to the drawing: {facing:?} against {along:?}"
    );

    // The drawing uses it: the arrow is square to the rim, and the leader ends at its base rather
    // than short of it along a ray that no longer points the same way.
    let anchor = center + (squashed(slant) - center) * 0.5;
    let drawing = radius(center, wide, anchor, rim, "120", Rank::Driving);
    let head = heads(&drawing)[0];
    assert!(
        aim(head).dot(facing) > 0.999,
        "the arrow left the direction the curve faces: {:?}",
        aim(head)
    );
    let base = head[3] + (head[2] - head[3]) / 2.0;
    let leader = drawing
        .pieces
        .iter()
        .find_map(|piece| match piece {
            Piece::Polyline(points) if points.first() == Some(&center) => points.last().copied(),
            _ => None,
        })
        .expect("a leader out of the center");
    assert!(
        (leader - base).length() < 1e-3,
        "the leader stopped somewhere other than the arrow's base"
    );

    // A diameter's two runs still MEET at the center, so it still reads as crossing it, even
    // though its two ends now stop at arrows that are no longer opposite each other.
    let across = diameter(center, wide, anchor, rim, "240", Rank::Driving);
    assert!(
        across.pieces.iter().any(|piece| matches!(
            piece,
            Piece::Polyline(points) if points.len() == 3 && (points[1] - center).length() < 1e-3
        )),
        "the through-line lost its center"
    );
}

/// A diameter meets the rim TWICE, so an arc read across can fall short at either end or both.
#[test]
fn a_diameter_carries_both_of_the_ends_it_lands_on() {
    let center = Pos2::new(100.0, 100.0);
    // Half the circle, the eastern side.
    let standing = round(center, 30.0);
    let rim = Rim {
        from: -std::f32::consts::FRAC_PI_2,
        turn: std::f32::consts::PI,
        at: &standing,
    };
    let carried = |drawing: &Drawing| {
        drawing
            .pieces
            .iter()
            .filter(|piece| match piece {
                Piece::Polyline(points) => {
                    points.len() > 2
                        && points
                            .iter()
                            .all(|at| ((*at - center).length() - 30.0).abs() < 1e-2)
                }
                Piece::Head(_) => false,
            })
            .count()
    };
    // Struck due east: one end is on the curve, the other is on the half it does not draw.
    assert_eq!(
        carried(&diameter(
            center,
            30.0,
            Pos2::new(160.0, 100.0),
            rim,
            "60",
            Rank::Driving
        )),
        1,
        "only the end that misses is carried",
    );
    // Struck due north-south: both ends sit on the curve's own two tips, so neither is carried.
    assert_eq!(
        carried(&diameter(
            center,
            30.0,
            Pos2::new(100.0, 40.0),
            rim,
            "60",
            Rank::Driving
        )),
        0,
    );
}

/// **A diameter crosses the center, and both its ends sit on the rim.** That is the entire
/// difference from a radius, so it is the thing a rewrite must not lose: two arrowheads, one on
/// each side, each exactly `radius` from the center and on the anchor's own ray.
#[test]
fn a_diameter_touches_the_rim_on_both_sides_of_the_center() {
    let center = Pos2::new(120.0, 90.0);
    let standing = round(center, 34.0);
    for anchor in [
        Pos2::new(184.0, 54.0),
        Pos2::new(48.0, 160.0),
        Pos2::new(120.0, 22.0),
    ] {
        let drawing = diameter(center, 34.0, anchor, whole(&standing), "68", Rank::Driving);
        let touches = heads(&drawing);
        assert_eq!(touches.len(), 2, "one head per side for anchor {anchor:?}");
        let ray = (anchor - center).normalized();
        let along: Vec<f32> = touches
            .iter()
            .map(|head| {
                assert!(
                    ((points_at(*head) - center).length() - 34.0).abs() < 1e-3,
                    "a tip left the rim for anchor {anchor:?}"
                );
                (points_at(*head) - center).dot(ray)
            })
            .collect();
        assert!(
            along[0] * along[1] < 0.0,
            "both tips landed on the same side of the center for anchor {anchor:?}"
        );
    }
}

/// **A diameter's fit test is the value's, not the arrows'** — the same middle row the span
/// documents. A 24-wide circle holds two 9-unit arrows and still cannot hold a number between
/// them, so the value leaves and the arrows reverse to point back in.
#[test]
fn a_diameter_too_tight_to_read_across_evicts_its_value() {
    let center = Pos2::new(90.0, 90.0);
    let anchor = Pos2::new(130.0, 50.0);
    let ray = (anchor - center).normalized();
    let (wide, narrow) = (round(center, 40.0), round(center, 12.0));

    let roomy = diameter(center, 40.0, anchor, whole(&wide), "80", Rank::Driving);
    let inward: Vec<f32> = heads(&roomy)
        .iter()
        .map(|head| aim(*head).dot(ray))
        .collect();
    assert!(
        inward[0] < -0.9 && inward[1] > 0.9,
        "roomy: the arrows point out at the rim, {inward:?}"
    );
    assert!(
        (roomy.labels[0].at - center).length() < 1e-3,
        "roomy: the value rides the line at the center"
    );

    let tight = diameter(center, 12.0, anchor, whole(&narrow), "24", Rank::Driving);
    let outward: Vec<f32> = heads(&tight)
        .iter()
        .map(|head| aim(*head).dot(ray))
        .collect();
    assert!(
        outward[0] > 0.9 && outward[1] < -0.9,
        "tight: the arrows flipped to point back in, {outward:?}"
    );
    assert!(
        (tight.labels[0].at - center).length() > 12.0,
        "tight: the value left the circle"
    );
    assert_eq!(tight.labels[0].radians, 0.0, "an evicted value reads level");
}

/// A leader never doubles back into the circle it left, however far inside the anchor is dragged.
#[test]
fn a_diameter_leader_dragged_inside_still_stops_at_the_rim() {
    let center = Pos2::new(60.0, 60.0);
    let standing = round(center, 10.0);
    let drawing = diameter(
        center,
        10.0,
        Pos2::new(63.0, 58.0),
        whole(&standing),
        "20",
        Rank::Driving,
    );
    let ray = (Pos2::new(63.0, 58.0) - center).normalized();
    assert!(
        (drawing.labels[0].at - center).dot(ray) >= 10.0 - 1e-3,
        "the value came to rest inside the rim"
    );
}

/// An arm that runs from the vertex out to `reach` — the ordinary case, where the two lines
/// actually meet at the corner being dimensioned.
fn from_the_vertex(reach: f32) -> Leg {
    Leg {
        nearest: 0.0,
        furthest: reach,
    }
}

#[test]
fn a_wide_angle_puts_its_value_on_the_arc() {
    let arc = struck(Pos2::new(70.0, 106.0), 54.0);
    let drawing = angle(
        Pos2::new(70.0, 106.0),
        -std::f32::consts::FRAC_PI_2,
        -0.5,
        54.0,
        whole(&arc),
        [from_the_vertex(80.0); 2],
        "62°",
        Rank::Driving,
    );
    assert_eq!(drawing.labels[0].anchor, Anchor::Middle);
    let on_arc = (drawing.labels[0].at - Pos2::new(70.0, 106.0)).length();
    assert!(
        (on_arc - 54.0).abs() < 1e-3,
        "the value rides the arc itself"
    );
}

#[test]
fn a_tight_angle_makes_the_same_reversal_the_span_does() {
    let vertex = Pos2::new(96.0, 108.0);
    let (from, to) = (-1.78, -1.36);
    let arc = struck(vertex, 34.0);
    let drawing = angle(
        vertex,
        from,
        to,
        34.0,
        whole(&arc),
        [from_the_vertex(60.0); 2],
        "24°",
        Rank::Driving,
    );
    // Arc length 34 * 0.42 = 14.3, under 2 * 9 + 2.
    let arrows = heads(&drawing);
    let outward_tangent = Vec2::new(-from.sin(), from.cos());
    assert!(
        aim(arrows[0]).dot(outward_tangent) > 0.9,
        "the first arrow swung outside pointing in",
    );
    assert_ne!(
        drawing.labels[0].anchor,
        Anchor::Middle,
        "the value left too"
    );
}

#[test]
fn a_leg_that_already_reaches_the_arc_grows_no_extension() {
    let vertex = Pos2::new(120.0, 116.0);
    let (from, to) = (-2.7, -0.7);
    let arc = struck(vertex, 46.0);
    let short = angle(
        vertex,
        from,
        to,
        46.0,
        whole(&arc),
        [from_the_vertex(40.0); 2],
        "115°",
        Rank::Driving,
    );
    let long = angle(
        vertex,
        from,
        to,
        46.0,
        whole(&arc),
        [from_the_vertex(90.0); 2],
        "115°",
        Rank::Driving,
    );
    let lines = |d: &Drawing| {
        d.pieces
            .iter()
            .filter(|p| matches!(p, Piece::Polyline(_)))
            .count()
    };
    assert_eq!(
        lines(&short) - lines(&long),
        2,
        "the virtual-intersection case carries each leg out; the reaching one does not",
    );
}

/// Two lines can cross at a point neither of them contains, and then an arc struck near that
/// crossing sits in a gap: the dogleg runs INWARD, from where the line starts back down to the arc.
#[test]
fn an_arm_the_arc_falls_short_of_is_carried_inward_to_meet_it() {
    let vertex = Pos2::new(100.0, 100.0);
    let (from, to) = (0.0, std::f32::consts::FRAC_PI_2);
    // The second arm starts 70 out and the arc is struck at 20, well inside it.
    let arc = struck(vertex, 20.0);
    let drawing = angle(
        vertex,
        from,
        to,
        20.0,
        whole(&arc),
        [
            from_the_vertex(60.0),
            Leg {
                nearest: 70.0,
                furthest: 120.0,
            },
        ],
        "90°",
        Rank::Driving,
    );
    let dogleg = drawing
        .pieces
        .iter()
        .find_map(|piece| match piece {
            Piece::Polyline(points) if points.len() == 2 => Some((points[0], points[1])),
            _ => None,
        })
        .expect("the far arm needs carrying to the arc");
    let out = |at: Pos2| (at - vertex).length();
    assert!(
        (out(dogleg.0) - 12.0).abs() < 1e-3 && (out(dogleg.1) - 70.0).abs() < 1e-3,
        "the dogleg crosses the arc and runs up to where the line starts: {dogleg:?}",
    );
}

#[test]
fn a_reference_dimension_parenthesises_the_whole_indication() {
    assert_eq!(Rank::Driving.indication("R", "21"), "R21");
    assert_eq!(
        Rank::Reference.indication("R", "21"),
        "(R21)",
        "ASME Y14.5 §5.9 wraps the prefix too — never R(21)",
    );
    assert_eq!(Rank::Reference.indication("", "62°"), "(62°)");
    assert_ne!(
        Rank::Driving.color(),
        Rank::Reference.color(),
        "and it is one rank quieter, so the two channels are independent",
    );
}

/// Total, not a special case: every bearing folds into a readable one.
#[test]
fn text_is_never_upside_down_from_any_quadrant() {
    let mut degrees = -720.0_f32;
    while degrees <= 720.0 {
        let folded = upright_radians(degrees.to_radians()).to_degrees();
        assert!(
            folded > -90.5 && folded <= 90.5,
            "{degrees}° folded to {folded}°, which reads upside-down",
        );
        // Folding may only turn the text by a half turn — never point it somewhere else.
        let turned = (degrees - folded).rem_euclid(180.0);
        assert!(
            !(0.5..=179.5).contains(&turned),
            "{degrees}° folded to {folded}°, which is not the same line",
        );
        degrees += 0.5;
    }
}

#[test]
fn every_piece_is_finite_at_a_degenerate_input() {
    // A zero-length span and a zero-radius anchor are both reachable by dragging, and a NaN here
    // would reach the painter as an invisible gizmo rather than a crash.
    let (ten, none) = (round(Pos2::ZERO, 10.0), round(Pos2::ZERO, 0.0));
    let drawings = [
        span(Pos2::ZERO, Pos2::ZERO, 10.0, "0", Rank::Driving),
        axis_span(
            Pos2::ZERO,
            Pos2::ZERO,
            Vec2::ZERO,
            [Vec2::ZERO; 2],
            Pos2::ZERO,
            "0",
            Rank::Driving,
        ),
        radius(
            Pos2::ZERO,
            10.0,
            Pos2::ZERO,
            whole(&ten),
            "10",
            Rank::Driving,
        ),
        diameter(
            Pos2::ZERO,
            0.0,
            Pos2::ZERO,
            whole(&none),
            "0",
            Rank::Driving,
        ),
        angle(
            Pos2::ZERO,
            1.0,
            1.0,
            20.0,
            whole(&none),
            [from_the_vertex(30.0); 2],
            "0°",
            Rank::Driving,
        ),
    ];
    for drawing in &drawings {
        for piece in &drawing.pieces {
            match piece {
                Piece::Polyline(points) => {
                    assert!(points.iter().all(|p| p.x.is_finite() && p.y.is_finite()))
                }
                Piece::Head(points) => {
                    assert!(points.iter().all(|p| p.x.is_finite() && p.y.is_finite()))
                }
            }
        }
        assert!(drawing.labels.iter().all(|l| l.radians.is_finite()));
    }
}
/// **A bearing a curve does not reach is answered with the end that is nearer.** This is what keeps
/// a dimension between two rims ON both of them: the annotation hangs off an end rather than
/// floating out past where anything is drawn, and the extension lines grow to say so.
#[test]
fn a_rim_answers_a_bearing_it_does_not_reach_with_its_nearer_end() {
    let quarter = std::f32::consts::FRAC_PI_2;
    let standing = round(Pos2::ZERO, 1.0);
    // A quarter of the circle, from due east round to due south (screen y runs down).
    let rim = Rim {
        from: 0.0,
        turn: quarter,
        at: &standing,
    };
    let near = |bearing: f32| rim.nearest_drawn(bearing);
    assert!(
        (near(quarter / 2.0) - quarter / 2.0).abs() < 1e-6,
        "the curve is drawn there, so the ask is its own answer"
    );
    assert!(
        (near(quarter * 1.2) - quarter).abs() < 1e-6,
        "just past the far end lands on the far end"
    );
    assert!(
        near(-0.2).abs() < 1e-6,
        "just short of the near end lands on the near end"
    );
    // Due west is a half turn from the start and a quarter from the end, so the END is nearer.
    assert!(
        (near(std::f32::consts::PI) - quarter).abs() < 1e-6,
        "the nearer of the two ends wins, not whichever comes first"
    );
    // A rim that turns the other way answers the mirror of all of it.
    let widdershins = Rim {
        from: 0.0,
        turn: -quarter,
        at: &standing,
    };
    assert!(
        (widdershins.nearest_drawn(-0.2) + 0.2).abs() < 1e-6,
        "inside its own sweep, which now runs the other way"
    );
    assert!(
        widdershins.nearest_drawn(0.2).abs() < 1e-6,
        "just past the start it now leaves behind, so the start is the answer"
    );
}
/// **Where the author drops the value is where it rides**, along its own dimension line as well as
/// across it. A dimension that could only be pushed sideways would be answering half the gesture.
#[test]
fn a_value_rides_where_it_was_dropped_and_leaves_by_the_end_it_was_carried_past() {
    let (near, far) = (Pos2::new(40.0, 100.0), Pos2::new(240.0, 100.0));
    // Dropped on the line itself, so the drawing is only deciding WHERE along it.
    let dropped = |x: f32| {
        axis_span(
            near,
            far,
            Vec2::X,
            [Vec2::Y; 2],
            Pos2::new(x, 100.0),
            "200",
            Rank::Driving,
        )
    };

    let middle = dropped(140.0);
    assert_eq!(middle.labels[0].anchor, Anchor::Middle);
    assert!((middle.labels[0].at.x - 140.0).abs() < 1e-4);

    // Pushed along, still inside: it goes with the hand rather than springing back.
    let along = dropped(200.0);
    assert_eq!(along.labels[0].anchor, Anchor::Middle);
    assert!(
        (along.labels[0].at.x - 200.0).abs() < 1e-4,
        "the value followed: {:?}",
        along.labels[0].at
    );

    // Carried past the far end: it leaves that way, and the leader REACHES it rather than
    // stopping short and leaving the number floating.
    let past = dropped(320.0);
    assert_eq!(past.labels[0].anchor, Anchor::Start);
    let leader = past
        .pieces
        .iter()
        .filter_map(|piece| match piece {
            Piece::Polyline(points) if points.len() == 2 => Some((points[0], points[1])),
            _ => None,
        })
        .find(|(start, _)| (start.x - far.x).abs() < 1e-4 && (start.y - far.y).abs() < 1e-4)
        .expect("a leader leaving the far end");
    assert!(
        (leader.1.x - 320.0).abs() < 1e-4,
        "the leader ran out to the hand: {:?}",
        leader.1
    );
    assert!(
        past.labels[0].at.x > far.x && past.labels[0].at.x < 320.0,
        "the value sits at the far end of that leader: {:?}",
        past.labels[0].at
    );

    // Carried past the NEAR end: it leaves that way instead. The old drawing only ever left by
    // the far one, which put the number on the opposite side from the hand that moved it.
    let before = dropped(-60.0);
    assert!(
        before.labels[0].at.x < near.x,
        "the value went the way it was carried: {:?}",
        before.labels[0].at
    );
    assert!(before.pieces.iter().any(|piece| matches!(
        piece,
        Piece::Polyline(points)
            if points.len() == 2
                && (points[0].x - near.x).abs() < 1e-4
                && (points[1].x + 60.0).abs() < 1e-4
    )));
}

/// A sketch plane seen from off to one side, as the door the shell hands these gizmos.
///
/// Orthographic on purpose. The map only has to be a HOMOGRAPHY for the bug to exist — the screen
/// perpendicular of a projected direction is the image of some other plane direction under any of
/// them, and the inverse stays exact either way, so unprojecting a drawn point back to the plane
/// launders nothing.
///
/// The shell's own door is `to_px` at `src/windowed/render.rs`: plane to world, times the view
/// projection, a perspective divide, then the viewport — which composes to exactly this, a
/// homography on the plane's two coordinates plus a `w > 0` cull. `depth` is what makes it one
/// rather than an affine map, and it is set BOTH ways in every test here: an orthographic fixture
/// is structurally unable to go red on anything the divide causes.
#[derive(Clone, Copy)]
struct Tilted {
    /// Which way the plane is turned under the camera, radians.
    azimuth: f32,
    /// How far the camera stands above the plane, radians. A quarter turn looks straight down and
    /// the projection becomes a similarity, which is the one view that cannot tell this bug apart.
    elevation: f32,
    /// How hard the map divides. Zero is orthographic; anything else and a plane direction no
    /// longer projects the same way at every point, which is the property an affine stand-in
    /// silently grants the code under test.
    depth: f32,
}

/// Where the drawing sits on screen, so the fixture's numbers read like a viewport's.
const ORIGIN: Pos2 = Pos2::new(300.0, 200.0);

/// Screen points per plane unit, before the foreshortening.
const SCALE: f32 = 3.0;

impl Tilted {
    /// Where a plane coordinate lands on screen.
    fn at(self, plane: [f32; 2]) -> Pos2 {
        let (turn, up) = (self.azimuth.sin_cos(), self.elevation.sin());
        let across = plane[1].mul_add(turn.1, -(plane[0] * turn.0));
        let into = plane[0].mul_add(turn.1, plane[1] * turn.0);
        let divide = self.depth.mul_add(into, 1.0);
        ORIGIN + Vec2::new(SCALE * across / divide, SCALE * up * into / divide)
    }

    /// Which way a step IN THE PLANE runs on screen, TAKEN AT A POINT.
    ///
    /// Position-dependent on purpose: once the map divides, a plane direction runs differently at
    /// every point, and a probe that forgot to say where would be answering for the orthographic
    /// map only. The step is taken at unit plane length, which is what the shell does and for the
    /// same reason — a step as long as the thing it describes can land behind the camera.
    fn direction(self, at: [f32; 2], step: [f32; 2]) -> Vec2 {
        let span = step[0].hypot(step[1]);
        let there = self.at([
            step[0].mul_add(1.0 / span, at[0]),
            step[1].mul_add(1.0 / span, at[1]),
        ]);
        (there - self.at(at)).normalized()
    }

    /// Back to the plane, exactly. The divide is recovered first, and then the 2x2 part, which is a
    /// reflection and so its own inverse.
    fn back(self, screen: Pos2) -> [f32; 2] {
        let (turn, up) = (self.azimuth.sin_cos(), self.elevation.sin());
        let (across, down) = ((screen.x - ORIGIN.x) / SCALE, (screen.y - ORIGIN.y) / SCALE);
        let divide = 1.0 / self.depth.mul_add(-down / up, 1.0);
        let (across, into) = (across * divide, down * divide / up);
        [
            into.mul_add(turn.1, -(across * turn.0)),
            across.mul_add(turn.1, into * turn.0),
        ]
    }
}

/// A three-quarter view: turned 45 degrees and looked at from 30 up. The lean this catches is 31
/// degrees there, and the table across elevations 15 to 60 never falls below 6.
///
/// Both the affine reading and a genuinely projective one, because they fail differently: the
/// divide is what makes a plane direction position-dependent, and every observable here is run
/// under both. The depth is chosen so the near and far ends of a 40-unit drawing differ by about
/// half again in scale — a strong perspective, not a token one.
fn three_quarters() -> [Tilted; 2] {
    [0.0, 0.008].map(|depth| Tilted {
        azimuth: std::f32::consts::FRAC_PI_4,
        elevation: std::f32::consts::FRAC_PI_6,
        depth,
    })
}

/// **A dimension's extension lines stand square to it IN THE PLANE, not on the screen.**
///
/// They are the one part of the drawing whose whole job is to read square, so when they are struck
/// on the screen's perpendicular instead of the plane's, a sketch looked at from an angle grows
/// dimensions that lean out of it. The owner's words: "the dimensions were not coplanar with the
/// sketch plane. They moved into the third dimension."
///
/// The first assertion is about the FIXTURE, not the code: seen square on, a plane's perpendicular
/// projects to the screen's own and the broken layout passes for free. A camera that cannot tell
/// the two apart makes every assertion below vacuous, so it has to be shown to tell them apart
/// before any of them is allowed to count.
#[test]
fn extension_lines_stand_square_in_the_plane_not_on_the_screen() {
    for view in three_quarters() {
        let (tail, head) = ([0.0_f32, 0.0], [40.0, 0.0]);
        let (run, square) = ([1.0_f32, 0.0], [0.0_f32, 1.0]);
        // Where the annotation was dropped, and so where the dimension line stands. `along` is the
        // run's direction THERE, not at the feature: under a dividing projection the plane's
        // parallels converge, and the line the drawing is laid out on is the one through the
        // anchor. The shell has no plane point for a screen anchor and reaches the same direction
        // through the vanishing point; here the anchor is a plane point, so it can be said outright.
        let dropped = [20.0_f32, 12.0];
        let along = view.direction(dropped, run);
        let across = [view.direction(tail, square), view.direction(head, square)];

        let screens_own = Vec2::new(along.y, -along.x);
        let apart = screens_own
            .x
            .mul_add(across[0].y, -(screens_own.y * across[0].x))
            .abs();
        assert!(
            apart > 0.4,
            "this view cannot tell the plane's square from the screen's ({apart}), so nothing \
             below would mean anything"
        );

        let drawing = axis_span(
            view.at(tail),
            view.at(head),
            along,
            across,
            view.at(dropped),
            "40",
            Rank::Driving,
        );
        // Both ends stand off the dimension line, so both grow an extension, and `axis_span`
        // collects them before anything else it draws.
        assert!(drawing.pieces.len() > 2, "{:?}", drawing.pieces);
        let mut feet = Vec::new();
        for piece in drawing.pieces.iter().take(2) {
            let Piece::Polyline(points) = piece else {
                panic!("the two extension lines come first: {piece:?}");
            };
            assert_eq!(points.len(), 2, "{points:?}");
            let (one, other) = (view.back(points[0]), view.back(points[1]));
            let reach = [other[0] - one[0], other[1] - one[1]];
            let length = reach[0].hypot(reach[1]);
            assert!(length > 1.0, "an extension line of nothing: {reach:?}");
            let leaning = reach[0].mul_add(run[0], reach[1] * run[1]) / length;
            assert!(
                leaning.abs() < 1.0e-3,
                "at depth {} an extension line leans {} degrees off square in the plane",
                view.depth,
                90.0 - leaning.abs().acos().to_degrees()
            );
            // An extension is drawn from GAP short of its feature point to OVERRUN past the
            // dimension line, so the meeting itself is one overrun back from the drawn end.
            feet.push(view.back(points[1] - (points[1] - points[0]).normalized() * OVERRUN));
        }

        // And the line those two feet stand on is the run's own, carried over to where the author
        // dropped the annotation. This is the claim the extension lines cannot make on their own:
        // two segments can each be square to the run and still meet a line that is not parallel to
        // it, which draws a dimension that reads as measuring some other direction.
        let between = [feet[1][0] - feet[0][0], feet[1][1] - feet[0][1]];
        let span = between[0].hypot(between[1]);
        assert!(span > 1.0, "a dimension line of nothing: {between:?}");
        let veering = between[0].mul_add(square[0], between[1] * square[1]) / span;
        assert!(
            veering.abs() < 1.0e-3,
            "at depth {} the dimension line veers {} degrees off the run it measures",
            view.depth,
            veering.abs().asin().to_degrees()
        );
    }
}

/// Where a circle drawn IN THE PLANE stands on screen, asked by screen bearing.
///
/// The shell's own door, spelled here in the small: sample the plane circle, project it point by
/// point, and strike the ray out of the projected center against the ring. That is the only way to
/// answer a bearing on a curve whose image is an ellipse, and it needs no inverse of the projection.
fn projected(view: Tilted, center: [f32; 2], radius: f32) -> impl Fn(f32) -> Pos2 {
    const STEPS: usize = 72;
    let middle = view.at(center);
    let ring: Vec<Pos2> = (0..STEPS)
        .map(|step| {
            #[allow(clippy::cast_precision_loss)]
            let turn = std::f32::consts::TAU * step as f32 / STEPS as f32;
            view.at([
                radius.mul_add(turn.cos(), center[0]),
                radius.mul_add(turn.sin(), center[1]),
            ])
        })
        .collect();
    move |bearing| {
        let out = Vec2::angled(bearing);
        let across = Vec2::new(out.y, -out.x);
        for index in 0..STEPS {
            let (here, next) = (ring[index], ring[(index + 1) % STEPS]);
            let (side, other) = ((here - middle).dot(across), (next - middle).dot(across));
            if (side <= 0.0) == (other <= 0.0) {
                continue;
            }
            let hit = here + (next - here) * (side / (side - other));
            if (hit - middle).dot(out) >= 0.0 {
                return hit;
            }
        }
        middle + out
    }
}

/// **An angle's arc is a circle IN THE PLANE, so on screen it is the ellipse the plane draws.**
///
/// This is the one mark in the family that genuinely leaves the plane when it is struck on the
/// screen instead of asked of its rim. A straight piece cannot: plane to screen is a homography, so
/// every screen line through the image of an on-plane point is the image of SOME plane line, and a
/// span's dimension line only ever reads the wrong metric. A circle has no such excuse — a plane
/// circle images as an ellipse, and a screen circle drawn over it is a different curve.
///
/// As with the extension lines, the fixture is made to prove its own discriminating power first:
/// the projected ring has to be visibly out of round, or an arc struck at a screen radius would
/// draw the same thing and the assertion below would hold for the broken drawing too.
#[test]
fn an_angles_arc_is_a_circle_in_the_plane_not_on_the_screen() {
    for view in three_quarters() {
        an_angles_arc_stands_in_the_plane(view);
    }
}

fn an_angles_arc_stands_in_the_plane(view: Tilted) {
    let corner = [0.0_f32, 0.0];
    let vertex = view.at(corner);
    let in_plane = 14.0_f32;
    let standing = projected(view, corner, in_plane);
    let rim = whole(&standing);

    #[allow(clippy::cast_precision_loss)]
    let reaches: Vec<f32> = (0..8)
        .map(|step| (rim.touch(step as f32 * std::f32::consts::FRAC_PI_4) - vertex).length())
        .collect();
    let out_of_round = reaches.iter().fold(0.0_f32, |most, at| most.max(*at))
        / reaches.iter().fold(f32::MAX, |least, at| least.min(*at));
    assert!(
        out_of_round > 1.5,
        "this view draws the plane's circle nearly round ({out_of_round}), so a screen arc would \
         pass for it: {reaches:?}"
    );
    #[allow(clippy::cast_precision_loss)]
    let nominal = reaches.iter().sum::<f32>() / reaches.len() as f32;

    let drawing = angle(
        vertex,
        -1.2,
        0.4,
        nominal,
        rim,
        [from_the_vertex(60.0); 2],
        "92",
        Rank::Driving,
    );

    // The arc is the one piece drawn as a run of many points; an extension line has two and the
    // leader three.
    let drawn: Vec<&Vec<Pos2>> = drawing
        .pieces
        .iter()
        .filter_map(|piece| match piece {
            Piece::Polyline(points) if points.len() > 3 => Some(points),
            _ => None,
        })
        .collect();
    assert_eq!(drawn.len(), 1, "one arc: {:?}", drawing.pieces);

    // The ring is sampled at 72 steps and a bearing is answered on the chord between two of them,
    // so the drawing sits up to one half-step's sag inside the circle it means — 0.013 of 14 here.
    // That is the curve the shell draws too, so it is the tolerance rather than an error.
    let sag = in_plane * (1.0 - (std::f32::consts::TAU / 144.0).cos());
    for point in drawn[0] {
        let back = view.back(*point);
        let reach = (back[0] - corner[0]).hypot(back[1] - corner[1]);
        assert!(
            (reach - in_plane).abs() < sag * 1.5,
            "at depth {} the arc stands {reach} from the vertex in the plane where it should \
             stand {in_plane}",
            view.depth
        );
    }
}
