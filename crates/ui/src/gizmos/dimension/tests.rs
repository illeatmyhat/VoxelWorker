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
fn heads(drawing: &Drawing) -> Vec<[Pos2; 3]> {
    drawing
        .pieces
        .iter()
        .filter_map(|piece| match piece {
            Piece::Head(points) => Some(*points),
            _ => None,
        })
        .collect()
}

/// A head's direction: tip minus base midpoint, normalized.
fn aim(head: [Pos2; 3]) -> Vec2 {
    let base = head[1] + (head[2] - head[1]) / 2.0;
    (head[0] - base).normalized()
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
    assert_eq!(arrows[0][0], Pos2::new(44.0, 28.0));
    assert_eq!(arrows[1][0], Pos2::new(216.0, 28.0));
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
    for anchor in [
        Pos2::new(156.0, 44.0),
        Pos2::new(20.0, 150.0),
        Pos2::new(92.0, 12.0),
        Pos2::new(95.0, 82.0),
    ] {
        let drawing = radius(center, radius_length, anchor, "21", Rank::Driving);
        let head = heads(&drawing)[0];
        let touch = head[0];
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
    let inside = radius(center, 46.0, Pos2::new(152.0, 56.0), "46", Rank::Driving);
    let outward = aim(heads(&inside)[0]);
    let ray = (Pos2::new(152.0, 56.0) - center).normalized();
    assert!(
        outward.dot(ray) > 0.9,
        "inside: the arrow points out at the curve"
    );

    let outside = radius(center, 21.0, Pos2::new(196.0, 24.0), "21", Rank::Driving);
    let inward = aim(heads(&outside)[0]);
    let ray = (Pos2::new(196.0, 24.0) - center).normalized();
    assert!(inward.dot(ray) < -0.9, "outside: it reverses to point back");
}

/// **A diameter crosses the center, and both its ends sit on the rim.** That is the entire
/// difference from a radius, so it is the thing a rewrite must not lose: two arrowheads, one on
/// each side, each exactly `radius` from the center and on the anchor's own ray.
#[test]
fn a_diameter_touches_the_rim_on_both_sides_of_the_center() {
    let center = Pos2::new(120.0, 90.0);
    for anchor in [
        Pos2::new(184.0, 54.0),
        Pos2::new(48.0, 160.0),
        Pos2::new(120.0, 22.0),
    ] {
        let drawing = diameter(center, 34.0, anchor, "68", Rank::Driving);
        let touches = heads(&drawing);
        assert_eq!(touches.len(), 2, "one head per side for anchor {anchor:?}");
        let ray = (anchor - center).normalized();
        let along: Vec<f32> = touches
            .iter()
            .map(|head| {
                assert!(
                    ((head[0] - center).length() - 34.0).abs() < 1e-3,
                    "a tip left the rim for anchor {anchor:?}"
                );
                (head[0] - center).dot(ray)
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

    let roomy = diameter(center, 40.0, anchor, "80", Rank::Driving);
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

    let tight = diameter(center, 12.0, anchor, "24", Rank::Driving);
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
    let drawing = diameter(center, 10.0, Pos2::new(63.0, 58.0), "20", Rank::Driving);
    let ray = (Pos2::new(63.0, 58.0) - center).normalized();
    assert!(
        (drawing.labels[0].at - center).dot(ray) >= 10.0 - 1e-3,
        "the value came to rest inside the rim"
    );
}

#[test]
fn a_wide_angle_puts_its_value_on_the_arc() {
    let drawing = angle(
        Pos2::new(70.0, 106.0),
        -std::f32::consts::FRAC_PI_2,
        -0.5,
        54.0,
        80.0,
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
    let drawing = angle(vertex, from, to, 34.0, 60.0, "24°", Rank::Driving);
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
    let short = angle(vertex, from, to, 46.0, 40.0, "115°", Rank::Driving);
    let long = angle(vertex, from, to, 46.0, 90.0, "115°", Rank::Driving);
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
    let drawings = [
        span(Pos2::ZERO, Pos2::ZERO, 10.0, "0", Rank::Driving),
        axis_span(
            Pos2::ZERO,
            Pos2::ZERO,
            Vec2::ZERO,
            Pos2::ZERO,
            "0",
            Rank::Driving,
        ),
        radius(Pos2::ZERO, 10.0, Pos2::ZERO, "10", Rank::Driving),
        diameter(Pos2::ZERO, 0.0, Pos2::ZERO, "0", Rank::Driving),
        angle(Pos2::ZERO, 1.0, 1.0, 20.0, 30.0, "0°", Rank::Driving),
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
                Piece::Arc {
                    center,
                    radius,
                    from,
                    to,
                } => assert!(
                    center.x.is_finite()
                        && radius.is_finite()
                        && from.is_finite()
                        && to.is_finite()
                ),
            }
        }
        assert!(drawing.labels.iter().all(|l| l.radians.is_finite()));
    }
}
