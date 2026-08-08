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

impl Sheet {
    /// The four rows: both ranks, the three span states, the two radius cases, the two angles.
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
                let leg = |p: &Painter, at: Pos2, bearing: f32, reach: f32| {
                    gizmos::segment(p, at, at + Vec2::new(bearing.cos(), bearing.sin()) * reach);
                };
                let wide = Pos2::new(s.left() + 26.0, s.bottom() - 18.0);
                let (from, to) = (-std::f32::consts::FRAC_PI_2, -0.5_f32);
                leg(p, wide, from, 62.0);
                leg(p, wide, to, 62.0);
                // 48, not 40: at 40 the arc is 42.8 long and the value is evicted onto a leader,
                // which is the tight drawing — this row is here to show the wide one.
                dimension::angle(wide, from, to, 48.0, 62.0, "62°", Rank::Driving).paint(p);

                let tight = Pos2::new(s.left() + 128.0, s.bottom() - 6.0);
                let (from, to) = (-1.78_f32, -1.36_f32);
                leg(p, tight, from, 56.0);
                leg(p, tight, to, 56.0);
                dimension::angle(tight, from, to, 34.0, 56.0, "24°", Rank::Driving).paint(p);
            },
        );
    }
}
