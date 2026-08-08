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

/// An arm running from the corner out to `reach`, which is the case where the two lines meet.
const fn whole_arm(reach: f32) -> dimension::Leg {
    dimension::Leg {
        nearest: 0.0,
        furthest: reach,
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
                dimension::span(left.0, left.1, -30.0, "80", Rank::Driving).paint(p);

                let right = (
                    Pos2::new(s.left() + 112.0, s.top() + 28.0),
                    Pos2::new(s.left() + 194.0, s.top() + 28.0),
                );
                gizmos::segment(p, right.0, right.1);
                dimension::span(right.0, right.1, -30.0, "82", Rank::Reference).paint(p);
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
                dimension::span(short.0, short.1, -28.0, "30", Rank::Driving).paint(p);

                let tiny = (
                    Pos2::new(s.left() + 126.0, s.top() + 30.0),
                    Pos2::new(s.left() + 140.0, s.top() + 30.0),
                );
                gizmos::segment(p, tiny.0, tiny.1);
                dimension::span(tiny.0, tiny.1, -28.0, "14", Rank::Driving).paint(p);
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
                dimension::radius(
                    outside,
                    20.0,
                    Pos2::new(s.left() + 74.0, s.top() + 26.0),
                    "20",
                    Rank::Driving,
                )
                .paint(p);

                let inside = Pos2::new(s.left() + 150.0, s.center().y + 4.0);
                p.circle_stroke(inside, 34.0, stroke);
                dimension::radius(
                    inside,
                    34.0,
                    Pos2::new(s.left() + 166.0, s.center().y - 8.0),
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
                dimension::diameter(
                    wide,
                    30.0,
                    Pos2::new(s.left() + 78.0, s.center().y - 18.0),
                    "30",
                    Rank::Driving,
                )
                .paint(p);

                let tight = Pos2::new(s.left() + 160.0, s.center().y + 4.0);
                p.circle_stroke(tight, 12.0, stroke);
                dimension::diameter(
                    tight,
                    12.0,
                    Pos2::new(s.left() + 190.0, s.center().y - 22.0),
                    "12",
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
