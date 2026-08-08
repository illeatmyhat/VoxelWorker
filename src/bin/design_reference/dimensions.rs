//! The dimension-gizmo specimens.
//!
//! Its own file rather than another shelf in [`sheet`](super::sheet), because a dimension is the
//! one gizmo whose drawing is a LAYOUT: each row here exists to show a state the layout chooses,
//! not a shape it has, so the row and the state it demonstrates have to stay written together.
//!
//! Every row draws the FEATURE with the ordinary sketch gizmos and the dimension over it. That
//! composition is the point: a dimension never draws the geometry it measures, so a sheet showing
//! one alone would be showing half of it.

#![allow(clippy::arithmetic_side_effects, clippy::too_many_lines)]

use egui::{Painter, Pos2, Stroke, Ui, Vec2};
use ui::gizmos::dimension::{self, Rank};
use ui::gizmos::{self};
use ui::theme::color_palette;

use crate::sheet::Sheet;

/// A span whose dimension line runs parallel to its own run, `offset` away along the normal. The
/// sheet draws this case on its own because it is the one an author reads as "the length of that",
/// and the gizmo reaches it by being handed the run's own direction.
fn aligned(from: Pos2, to: Pos2, offset: f32, value: &str, rank: Rank) -> dimension::Drawing {
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
    dimension::axis_span(from, to, along, middle + normal * offset, value, rank)
}

/// An arm running from the corner out to `reach`, which is the case where the two lines meet.
const fn whole_arm(reach: f32) -> dimension::Leg {
    dimension::Leg {
        nearest: 0.0,
        furthest: reach,
    }
}

/// Where a rim stands, on a flat page. The sheet has no sketch plane and no projection, so a
/// circle here really is a circle: it stands the same distance out at every bearing. The app hands
/// in a projected ring instead, which is the whole reason a rim is ASKED where it stands rather
/// than stepped out to along a screen radius.
fn round(center: Pos2, radius: f32) -> impl Fn(f32) -> Pos2 {
    move |bearing| center + Vec2::angled(bearing) * radius
}

/// A rim that draws the whole of its own circle, so it falls short of nothing.
fn whole(at: &dyn Fn(f32) -> Pos2) -> dimension::Rim<'_> {
    dimension::Rim {
        from: 0.0,
        turn: std::f32::consts::TAU,
        at,
    }
}

/// Draws the part of a circle an arc actually occupies, in geometry ink — the sheet has no sketch
/// to point at, so the specimen has to draw its own curve for the dimension to fall short of.
fn rim_curve(painter: &Painter, rim: dimension::Rim) {
    let steps = 24;
    let at = |step: i32| {
        #[allow(clippy::cast_precision_loss)]
        rim.touch(rim.turn.mul_add(step as f32 / steps as f32, rim.from))
    };
    for step in 0..steps {
        gizmos::segment(painter, at(step), at(step + 1));
    }
}

/// Draws the sketch line an angle's leg is read off, over the interval that line actually occupies.
fn arm(painter: &Painter, vertex: Pos2, bearing: f32, leg: dimension::Leg) {
    let along = Vec2::new(bearing.cos(), bearing.sin());
    gizmos::segment(
        painter,
        vertex + along * leg.nearest,
        vertex + along * leg.furthest,
    );
}

impl Sheet {
    /// Both ranks, the three span states, one extent of a diagonal, the two rim cases, the two
    /// angles.
    pub(crate) fn dimension_gizmos(&mut self, ui: &mut Ui) {
        self.specimen_row(
            ui,
            "span · everything inside · driving / reference",
            "A linear dimension: extension lines gapped off the feature, the dimension line \
             stopping at the arrow bases, the value riding it. Driving on the left, reference on \
             the right — parenthesised WHOLE per ASME Y14.5 §5.9, and one rank quieter, so the \
             two are told apart on a channel that survives grayscale and one that works in \
             peripheral vision.",
            |p, s| {
                let left = (
                    Pos2::new(s.left() + 12.0, s.top() + 28.0),
                    Pos2::new(s.left() + 92.0, s.top() + 28.0),
                );
                gizmos::segment(p, left.0, left.1);
                aligned(left.0, left.1, -30.0, "80", Rank::Driving).paint(p);

                let right = (
                    Pos2::new(s.left() + 112.0, s.top() + 28.0),
                    Pos2::new(s.left() + 194.0, s.top() + 28.0),
                );
                gizmos::segment(p, right.0, right.1);
                aligned(right.0, right.1, -30.0, "82", Rank::Reference).paint(p);
            },
        );
        self.specimen_row(
            ui,
            "span · the value leaves, then the arrows follow",
            "TWO independent fit tests, which is why there are three states and not two. Left: 30 \
             units holds both arrows with room to spare, but the value has to clear both arrow \
             BASES, so it leaves while they stay. Right: 14 units holds neither, and the arrows \
             swing outside pointing in. There is no fourth state — the arrow test is the weaker \
             of the two, so a value can never sit inside a span its own arrows do not fit.",
            |p, s| {
                let short = (
                    Pos2::new(s.left() + 20.0, s.top() + 30.0),
                    Pos2::new(s.left() + 50.0, s.top() + 30.0),
                );
                gizmos::segment(p, short.0, short.1);
                aligned(short.0, short.1, -28.0, "30", Rank::Driving).paint(p);

                let tiny = (
                    Pos2::new(s.left() + 126.0, s.top() + 30.0),
                    Pos2::new(s.left() + 140.0, s.top() + 30.0),
                );
                gizmos::segment(p, tiny.0, tiny.1);
                aligned(tiny.0, tiny.1, -28.0, "14", Rank::Driving).paint(p);
            },
        );
        self.specimen_row(
            ui,
            "span · one extent of a diagonal run",
            "The width and the height of the SAME run, which is the drawing the nine-region drop \
             rule exists to reach. Neither is the aligned span above: the dimension line lies \
             along one plane axis, so the two extension lines have DIFFERENT lengths and each end \
             reaches the line by its own perpendicular. Left states the width, dropped above; \
             right states the height of the same diagonal, dropped beside it.",
            |p, s| {
                let run = (
                    Pos2::new(s.left() + 20.0, s.bottom() - 20.0),
                    Pos2::new(s.left() + 84.0, s.top() + 34.0),
                );
                gizmos::segment(p, run.0, run.1);
                dimension::axis_span(
                    run.0,
                    run.1,
                    Vec2::X,
                    Pos2::new(s.left() + 52.0, s.top() + 16.0),
                    "64",
                    Rank::Driving,
                )
                .paint(p);

                let same = (
                    Pos2::new(s.left() + 130.0, s.bottom() - 20.0),
                    Pos2::new(s.left() + 194.0, s.top() + 34.0),
                );
                gizmos::segment(p, same.0, same.1);
                dimension::axis_span(
                    same.0,
                    same.1,
                    Vec2::Y,
                    Pos2::new(s.left() + 116.0, s.center().y),
                    "42",
                    Rank::Driving,
                )
                .paint(p);
            },
        );
        self.specimen_row(
            ui,
            "gap · a point off a line · two rails that never meet",
            "One measurement reached by two gestures. The dimension line runs ACROSS the line it \
             is measured against, so the two extension lines run parallel to that line and each \
             end reaches by its own perpendicular. Left: a point standing off a rail. Right: two \
             parallel rails, set past each other so neither perpendicular foot lands on the \
             other's drawn run — the extension carries on past the end, because the claim is \
             against the whole line and not the piece of it that is drawn.",
            |p, s| {
                let rail = (
                    Pos2::new(s.left() + 16.0, s.bottom() - 26.0),
                    Pos2::new(s.left() + 96.0, s.bottom() - 26.0),
                );
                gizmos::segment(p, rail.0, rail.1);
                let stood = Pos2::new(s.left() + 62.0, s.bottom() - 70.0);
                gizmos::vertex_handle(p, stood, 3.0, gizmos::HandleState::Idle, false);
                dimension::axis_span(
                    stood,
                    Pos2::new(stood.x, rail.0.y),
                    Vec2::Y,
                    Pos2::new(s.left() + 30.0, s.bottom() - 48.0),
                    "44",
                    Rank::Driving,
                )
                .paint(p);

                let along = Vec2::new(0.6, -0.8);
                let across = Vec2::new(along.y, -along.x);
                let lower = Pos2::new(s.left() + 118.0, s.bottom() - 26.0);
                gizmos::segment(p, lower, lower + along * 44.0);
                // Parallel, twenty-six across, and slid along so the two runs barely
                // overlap.
                let upper = lower - across * 26.0 + along * 34.0;
                gizmos::segment(p, upper, upper + along * 44.0);
                gizmos::vertex_handle(p, upper, 3.0, gizmos::HandleState::Idle, false);
                let foot = lower + along * (upper - lower).dot(along);
                dimension::axis_span(
                    upper,
                    foot,
                    across,
                    upper - across * 13.0 - along * 22.0,
                    "26",
                    Rank::Driving,
                )
                .paint(p);
            },
        );
        self.specimen_row(
            ui,
            "rim gap · two rims about one center",
            "The same measurement a gap across a line makes, read on a curve: the dimension line \
             runs ALONG a radius, so each extension line lies on the tangent at the rim it leaves. \
             Left: two whole circles, the annotation dropped at the bearing it is measured out along, \
             so neither extension has anywhere to go and the value is evicted the way any span too \
             narrow to hold it is. Right: two arcs, with the annotation pulled round past where the \
             outer one is drawn — the bearing lands on that arc's nearer end and both tangents grow \
             to reach the line.",
            |p, s| {
                let center = Pos2::new(s.left() + 46.0, s.center().y);
                for radius in [12.0_f32, 30.0] {
                    let standing = round(center, radius);
                    rim_curve(p, whole(&standing));
                }
                dimension::axis_span(
                    center + Vec2::X * 12.0,
                    center + Vec2::X * 30.0,
                    Vec2::X,
                    center + Vec2::X * 21.0,
                    "18",
                    Rank::Driving,
                )
                .paint(p);

                let hub = Pos2::new(s.left() + 150.0, s.center().y + 14.0);
                // The outer arc stops short of where the text was dropped; the inner one does not.
                let (inner, outer) = (round(hub, 12.0), round(hub, 30.0));
                let drawn = [
                    dimension::Rim {
                        from: -2.4,
                        turn: 2.4,
                        at: &inner,
                    },
                    dimension::Rim {
                        from: -2.0,
                        turn: 1.4,
                        at: &outer,
                    },
                ];
                for rim in drawn {
                    rim_curve(p, rim);
                }
                let dropped = -0.2;
                let bearing = drawn
                    .into_iter()
                    .fold(dropped, |bearing, rim| rim.nearest_drawn(bearing));
                let out = Vec2::angled(bearing);
                dimension::axis_span(
                    drawn[0].touch(bearing),
                    drawn[1].touch(bearing),
                    out,
                    hub + Vec2::angled(dropped) * 21.0,
                    "18",
                    Rank::Driving,
                )
                .paint(p);
            },
        );
        self.specimen_row(
            ui,
            "radius · anchor outside / inside the curve",
            "The arc point is DERIVED from the anchor, never stored beside it: the leader meets \
             the curve on the anchor's own ray, so no drag can make a radius stop pointing at its \
             center. Outside, the leader jogs at the anchor and the arrow reverses to point back \
             at the curve; inside, it runs center → arc with the text riding it. The center is a \
             cross in dimension ink, never a filled dot — a dot is what a sketch point looks like.",
            |p, s| {
                let stroke = Stroke::new(1.5_f32, color_palette::ACCENT);
                let outside = Pos2::new(s.left() + 46.0, s.center().y + 4.0);
                p.circle_stroke(outside, 20.0, stroke);
                let standing = round(outside, 20.0);
                dimension::radius(
                    outside,
                    20.0,
                    Pos2::new(s.left() + 74.0, s.top() + 26.0),
                    whole(&standing),
                    "20",
                    Rank::Driving,
                )
                .paint(p);

                let inside = Pos2::new(s.left() + 150.0, s.center().y + 4.0);
                p.circle_stroke(inside, 34.0, stroke);
                let standing = round(inside, 34.0);
                dimension::radius(
                    inside,
                    34.0,
                    Pos2::new(s.left() + 166.0, s.center().y - 8.0),
                    whole(&standing),
                    "34",
                    Rank::Driving,
                )
                .paint(p);
            },
        );
        self.specimen_row(
            ui,
            "diameter · reads across / evicted",
            "The same claim as the radius, struck THROUGH the center instead of out from it — a \
             hole is read across and a fillet is read out, and which one shows is the author's. \
             The one fit test is whether the value clears both arrow bases, which is not the same \
             question as whether the arrows fit: the small circle here holds its arrows and still \
             sends its number out on a leader. The prefix is D, because the drafting glyph is an \
             icon spelled as a character and the value type is monospace.",
            |p, s| {
                let stroke = Stroke::new(1.5_f32, color_palette::ACCENT);
                let wide = Pos2::new(s.left() + 56.0, s.center().y + 4.0);
                p.circle_stroke(wide, 30.0, stroke);
                let standing = round(wide, 30.0);
                dimension::diameter(
                    wide,
                    30.0,
                    Pos2::new(s.left() + 78.0, s.center().y - 18.0),
                    whole(&standing),
                    "30",
                    Rank::Driving,
                )
                .paint(p);

                let tight = Pos2::new(s.left() + 160.0, s.center().y + 4.0);
                p.circle_stroke(tight, 12.0, stroke);
                let standing = round(tight, 12.0);
                dimension::diameter(
                    tight,
                    12.0,
                    Pos2::new(s.left() + 190.0, s.center().y - 22.0),
                    whole(&standing),
                    "12",
                    Rank::Driving,
                )
                .paint(p);
            },
        );
        self.specimen_row(
            ui,
            "radius · a leader the arc does not reach",
            "An arc draws part of its circle and the annotation can be dragged anywhere round it, \
             so the leader can land on a bearing the curve never gets to. The curve is carried \
             round to meet it, out of whichever of its two ends is nearer — the same rule the \
             angle's doglegs follow, on a curve instead of a line. It applies to BOTH drawings: \
             left, the anchor outside; right, the anchor pulled inside, where the leader runs out \
             from the center and still has to arrive at the rim.",
            |p, s| {
                let outside = Pos2::new(s.left() + 42.0, s.center().y + 6.0);
                let standing = round(outside, 28.0);
                let short = dimension::Rim {
                    from: -2.5,
                    turn: 1.1,
                    at: &standing,
                };
                rim_curve(p, short);
                dimension::radius(
                    outside,
                    28.0,
                    Pos2::new(s.left() + 92.0, s.center().y + 26.0),
                    short,
                    "28",
                    Rank::Driving,
                )
                .paint(p);

                let inside = Pos2::new(s.left() + 150.0, s.center().y + 6.0);
                let standing = round(inside, 32.0);
                let half = dimension::Rim {
                    from: -0.5,
                    turn: -1.9,
                    at: &standing,
                };
                rim_curve(p, half);
                dimension::radius(
                    inside,
                    32.0,
                    Pos2::new(s.left() + 164.0, s.center().y + 20.0),
                    half,
                    "32",
                    Rank::Driving,
                )
                .paint(p);
            },
        );
        self.specimen_row(
            ui,
            "angle · wide / tight",
            "The terminators are TANGENT to the arc, and both fit tests run on the ARC LENGTH \
             rather than the chord — a wide sweep measured across its chord would read as too \
             tight for arrows it comfortably holds. The tight case makes the same reversal the \
             span does, which is what keeps the two legible as one family. The vertex is never \
             drawn: where the legs do not meet, the intersection is virtual and no pick can ever \
             land on it.",
            |p, s| {
                let wide = Pos2::new(s.left() + 26.0, s.bottom() - 18.0);
                let (from, to) = (-std::f32::consts::FRAC_PI_2, -0.5_f32);
                arm(p, wide, from, whole_arm(62.0));
                arm(p, wide, to, whole_arm(62.0));
                // 48, not 40: at 40 the arc is 42.8 long and the value is evicted onto a leader,
                // which is the tight drawing — this row is here to show the wide one.
                dimension::angle(
                    wide,
                    from,
                    to,
                    48.0,
                    [whole_arm(62.0); 2],
                    "62°",
                    Rank::Driving,
                )
                .paint(p);

                let tight = Pos2::new(s.left() + 128.0, s.bottom() - 6.0);
                let (from, to) = (-1.78_f32, -1.36_f32);
                arm(p, tight, from, whole_arm(56.0));
                arm(p, tight, to, whole_arm(56.0));
                dimension::angle(
                    tight,
                    from,
                    to,
                    34.0,
                    [whole_arm(56.0); 2],
                    "24°",
                    Rank::Driving,
                )
                .paint(p);
            },
        );
        self.specimen_row(
            ui,
            "angle · a corner neither line reaches",
            "Two lines that cross where neither of them runs. The corner is still a real angle and \
             still dimensionable, and the arc is struck wherever the author dropped the text — \
             here inside both lines, so each dogleg runs INWARD, from where its line starts back \
             down to the arc. That is the same rule as the wide case above, read the other way: a \
             dogleg spans whatever its line does not. Right: the arc struck outside one of them, \
             so one dogleg runs in and the other out.",
            |p, s| {
                let inside = Pos2::new(s.left() + 26.0, s.bottom() - 14.0);
                let (from, to) = (-1.35_f32, -0.30_f32);
                let (near, far) = (
                    dimension::Leg {
                        nearest: 34.0,
                        furthest: 66.0,
                    },
                    dimension::Leg {
                        nearest: 38.0,
                        furthest: 74.0,
                    },
                );
                arm(p, inside, from, near);
                arm(p, inside, to, far);
                dimension::angle(inside, from, to, 24.0, [near, far], "60°", Rank::Driving).paint(p);

                let across = Pos2::new(s.left() + 126.0, s.bottom() - 14.0);
                let (from, to) = (-1.60_f32, -0.30_f32);
                let (stops_short, runs_past) = (
                    dimension::Leg {
                        nearest: 0.0,
                        furthest: 26.0,
                    },
                    dimension::Leg {
                        nearest: 58.0,
                        furthest: 76.0,
                    },
                );
                arm(p, across, from, stops_short);
                arm(p, across, to, runs_past);
                dimension::angle(
                    across,
                    from,
                    to,
                    40.0,
                    [stops_short, runs_past],
                    "74°",
                    Rank::Driving,
                )
                .paint(p);
            },
        );
    }
}
