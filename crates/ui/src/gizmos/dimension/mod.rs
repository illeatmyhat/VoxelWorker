//! Dimension gizmos — spans, radii and angles drawn in the viewport at true scale.
//!
//! Unlike the rest of
//! [`gizmos`](crate::gizmos), a dimension is not one shape with a state: it is a small LAYOUT
//! problem whose answer changes with the size of the thing being dimensioned, so this module is
//! split in two. [`axis_span()`], [`radius()`] and [`angle()`] each answer a [`Drawing`] — pure geometry,
//! no painter — and [`Drawing::paint`] puts it on screen. That split is what lets the fit rules
//! be tested without a GPU, and the fit rules are where every mistake lives.
//!
//! ## The halo is the answer to the ground problem
//!
//! A dimension is drawn over whatever the viewport happens to show — the near-black background,
//! a pale sandstone block, a mid-tone green one. No single flat color survives all three; red
//! least of all. So the ink is the theme's foreground and every stroke is backed by a halo in the
//! theme's background, which is invisible over the viewport and load-bearing over a bright block.
//! It costs one extra pass, no color decision, and inverts with the theme for free.
//!
//! **Every halo is painted before any ink**, as two passes over the WHOLE gizmo — not
//! halo-then-ink per element, which is what lets an arrowhead's halo bite into the dimension line
//! it terminates. Layer order is: all halos, all ink, then values.
//!
//! **The value is the one exception**, deliberately: its halo is painted over the dimension line,
//! because a dimension line is supposed to break where the number sits.
//!
//! ## Ink
//!
//! The sheet asks for "the theme's high-contrast foreground" and, for a reference dimension, the
//! same hue family one rank quieter — explicitly NOT a disabled gray, because a reference
//! dimension is fully live and updates on every solve. Those are
//! [`TEXT_PRIMARY`](color_palette::TEXT_PRIMARY) and
//! [`TEXT_SECONDARY`](color_palette::TEXT_SECONDARY) as already registered; no dimension-specific
//! token is minted, because two near-duplicates of an existing step is exactly the drift the
//! palette registry exists to stop.
//!
//! ## What this module does NOT draw
//!
//! The geometry being dimensioned. The design sheet draws a segment or a circle beside each gizmo
//! because the sheet has no sketch to point at; in the app the sketch draws its own entities, and
//! a gizmo that drew them again would double every stroke. The one exception is the radius center
//! mark, which is dimension ink and belongs to the dimension.

use egui::{Color32, FontId, Painter, Pos2, Rect, Shape, Stroke, Vec2};

use crate::theme::color_palette;

#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::doc_link_code,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::needless_pass_by_ref_mut,
    clippy::option_if_let_else,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_lines,
    clippy::tuple_array_conversions,
    clippy::unreadable_literal,
    clippy::use_self,
    clippy::wildcard_imports
)]
mod angle;
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::doc_link_code,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::needless_pass_by_ref_mut,
    clippy::option_if_let_else,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_lines,
    clippy::tuple_array_conversions,
    clippy::unreadable_literal,
    clippy::use_self,
    clippy::wildcard_imports
)]
mod diameter;
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::doc_link_code,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::needless_pass_by_ref_mut,
    clippy::option_if_let_else,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_lines,
    clippy::tuple_array_conversions,
    clippy::unreadable_literal,
    clippy::use_self,
    clippy::wildcard_imports
)]
mod radius;
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::doc_link_code,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::needless_pass_by_ref_mut,
    clippy::option_if_let_else,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_lines,
    clippy::tuple_array_conversions,
    clippy::unreadable_literal,
    clippy::use_self,
    clippy::wildcard_imports
)]
mod span;

pub use angle::{angle, Leg};
pub use diameter::diameter;
pub use radius::radius;

pub use span::axis_span;

/// The chrome weight: dimension line, extension line and leader all share it, per ISO 128-20.
pub const LINE_WIDTH: f32 = 0.8;

/// Added to a stroke's width to make its halo.
pub const HALO_WIDTH: f32 = 2.4;

/// Arrowhead length, and half its base width — so the base is 3.0 and the ratio is 3:1.
pub const ARROW_LENGTH: f32 = 9.0;
const ARROW_HALF_WIDTH: f32 = 1.5;

/// How far a terminator's nose is cut back from the line it names — half the halo, so the halo's
/// own leading edge lands exactly where the point would have. See [`arrowhead`].
const ARROW_SETBACK: f32 = HALO_WIDTH / 2.0;

/// ISO 129-1: the gap left between the feature and the start of its extension line.
pub const GAP: f32 = 5.0;

/// ISO 129-1: how far an extension line runs past the dimension line it crosses.
const OVERRUN: f32 = 8.0;

/// How much of its own circle a curve actually draws, as screen bearings.
///
/// A whole circle draws all of it and has nothing to fall short of, so it passes `None`. An arc
/// draws part of it, and a radius or a diameter struck along the anchor's ray can land on the
/// circle at a bearing the curve itself never reaches — the leader would then point at nothing.
///
/// Where that happens the drawing is carried round the circle to meet it, from whichever end of the
/// curve is nearer. That is the same rule an angle's legs already follow, on a curve instead of a
/// line: **the extension spans whatever the geometry does not**.
#[derive(Clone, Copy)]
pub struct Rim<'a> {
    /// Where the curve starts, as a bearing from the center (radians, y running down).
    pub from: f32,
    /// How far it turns to reach its other end. Signed; a whole turn or more is a closed rim,
    /// which falls short of nothing.
    pub turn: f32,
    /// Where the curve's circle stands at a bearing.
    ///
    /// A circle drawn in a sketch plane is NOT a circle on screen: the plane is projected, so
    /// unless it faces the camera the drawing is an ellipse, and a screen radius is right only in
    /// the one direction it happened to be measured. Every mark that has to LAND on the curve —
    /// an arrowhead, an extension carried round to a leader — asks this instead of stepping out
    /// along a radius.
    pub at: &'a dyn Fn(f32) -> Pos2,
}

impl Rim<'_> {
    /// How far round from `self.from`, in the direction the curve turns, `bearing` lies.
    fn round_to(self, bearing: f32) -> f32 {
        ((bearing - self.from) * self.turn.signum()).rem_euclid(std::f32::consts::TAU)
    }

    /// How far this curve falls short of `bearing`, as `(the end that is nearer, the signed turn
    /// from it to the ask)` — `None` when the curve is drawn there and falls short of nothing.
    fn shortfall(self, bearing: f32) -> Option<(f32, f32)> {
        let round = self.round_to(bearing);
        if round <= self.turn.abs() {
            return None;
        }
        // Past the far end, or short of the near one — whichever is the shorter way to reach it.
        let direction = self.turn.signum();
        let past = round - self.turn.abs();
        let short = std::f32::consts::TAU - round;
        Some(if past <= short {
            (self.from + self.turn, direction * past)
        } else {
            (self.from, -direction * short)
        })
    }

    /// The arc that carries this curve round to `bearing`, as `(from, to)` bearings — `None` when
    /// the curve already reaches it. `overrun` is the extra turn past the meeting point, so the
    /// extension crosses what it is reaching for rather than stopping dead on it.
    fn carry_to(self, bearing: f32, overrun: f32) -> Option<(f32, f32)> {
        let (end, over) = self.shortfall(bearing)?;
        Some((end, end + over + over.signum() * overrun))
    }

    /// Where the curve stands at `bearing`, gapped `out` further from the center — the point a
    /// mark that has to LAND on the drawing uses in place of stepping out along a screen radius.
    #[must_use]
    pub fn touch(self, bearing: f32) -> Pos2 {
        (self.at)(bearing)
    }

    /// Which way the curve FACES at a bearing: its outward unit normal on screen.
    ///
    /// A mark that has to sit SQUARE to the drawing — an arrowhead, and whatever line runs into its
    /// base — is aimed by this rather than by the ray it was reached along. On a circle the two are
    /// the same direction; on the ellipse a tilted plane projects to they differ by as much as the
    /// tilt, and an arrow aimed along the ray lies across its own curve instead of meeting it.
    #[must_use]
    pub fn aim(self, bearing: f32) -> Vec2 {
        // A secant, not a derivative: the rim is sampled, so asking either side of the bearing
        // reads the drawing's own direction there rather than a curve it only approximates.
        const NUDGE: f32 = 1e-2;
        let along = self.touch(bearing + NUDGE) - self.touch(bearing - NUDGE);
        let radial = Vec2::angled(bearing);
        let out = Vec2::new(along.y, -along.x);
        if out.length() <= f32::EPSILON {
            return radial;
        }
        // Of the tangent's two perpendiculars, the one pointing away from the center — which the
        // bearing already names, because the rim answers a bearing with a point along that ray.
        let out = out.normalized();
        if out.dot(radial) >= 0.0 {
            out
        } else {
            -out
        }
    }

    /// The curve sampled from one bearing round to another, as screen points.
    pub(super) fn between(self, from: f32, to: f32) -> Vec<Pos2> {
        // One step per few degrees: fine enough that a projected rim reads as a curve, coarse
        // enough that a whole turn is a few dozen points rather than a few hundred.
        let steps = ((to - from).abs() / 0.12).ceil().max(1.0);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let count = steps as usize;
        (0..=count)
            .map(|step| {
                #[allow(clippy::cast_precision_loss)]
                let fraction = step as f32 / steps;
                self.touch((to - from).mul_add(fraction, from))
            })
            .collect()
    }

    /// The bearing on this curve nearest the one asked for: the ask itself where the curve is drawn
    /// there, and the nearer of its two ends where it is not.
    ///
    /// This is what keeps a dimension between two curves ON both of them. Where either falls short,
    /// the annotation hangs off an end rather than floating out past where anything is drawn, and
    /// the extension lines grow to say so.
    #[must_use]
    pub fn nearest_drawn(self, bearing: f32) -> f32 {
        self.shortfall(bearing).map_or(bearing, |(end, _)| end)
    }
}

/// The value's type size, and the monospace advance that follows from it.
///
/// Layout has to know how wide a value will be BEFORE anything is painted — the whole span rule
/// turns on whether the text clears both arrow bases — so the advance is a constant of the
/// monospace face rather than a measurement taken from a painter that layout does not have.
const VALUE_SIZE: f32 = 11.0;
const VALUE_ADVANCE: f32 = VALUE_SIZE * 0.6;

/// Whether a dimension DRIVES the geometry or merely reports it.
///
/// The two are told apart on two channels at once, which is deliberate: a reference dimension is
/// parenthesised whole — `(R21)`, never `R(21)` — and drawn one rank quieter. The parenthesis
/// survives grayscale; the weight works in peripheral vision.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rank {
    /// The solver holds it. Editing the number moves geometry.
    Driving,
    /// Derived and displayed. Fully live — it updates on every solve — but it drives nothing.
    Reference,
}

impl Rank {
    /// The ink this rank is drawn in.
    pub fn color(self) -> Color32 {
        match self {
            Rank::Driving => color_palette::TEXT_PRIMARY,
            Rank::Reference => color_palette::TEXT_SECONDARY,
        }
    }

    /// The whole indication, prefix included.
    ///
    /// ASME Y14.5 §5.9: the parenthesis wraps everything. It does not deny the measurement, it
    /// declares the indication derived — which is why an auxiliary dimension may never carry a
    /// tolerance.
    pub fn indication(self, prefix: &str, value: &str) -> String {
        match self {
            Rank::Driving => format!("{prefix}{value}"),
            Rank::Reference => format!("({prefix}{value})"),
        }
    }
}

/// How a value sits against its anchor point.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Anchor {
    /// Centered — a value sitting on the dimension line between the arrows.
    Middle,
    /// Left edge at the anchor — a value that has left on a leader running right.
    Start,
    /// Right edge at the anchor — the same leader running left.
    End,
}

/// A value, placed and angled.
#[derive(Clone, Debug, PartialEq)]
pub struct Label {
    /// Where the anchor sits.
    pub at: Pos2,
    /// The whole indication, already parenthesised if it is a reference.
    pub text: String,
    /// The baseline's bearing, folded upright by [`upright_radians`].
    pub radians: f32,
    /// Which edge of the text `at` refers to.
    pub anchor: Anchor,
    /// How far the text is lifted off the line it rides, along the baseline's normal.
    pub lift: f32,
}

/// One stroked or filled element of a dimension.
#[derive(Clone, Debug, PartialEq)]
pub enum Piece {
    /// A run of connected straight segments.
    ///
    /// Curves included. There is no arc piece: every curve a dimension draws is a circle IN THE
    /// SKETCH PLANE, which projects to an ellipse whenever the camera is not square to it, so a
    /// piece struck at a screen center and a screen radius could only ever draw the wrong one.
    /// Curves are sampled where they stand — [`Rim::between`] — and arrive here already flattened.
    Polyline(Vec<Pos2>),
    /// A filled arrowhead: the two nose corners, then the two base corners.
    Head([Pos2; 4]),
}

/// A laid-out dimension: everything to draw, and nothing about how.
#[derive(Clone, Debug, PartialEq)]
pub struct Drawing {
    /// Lines, arcs and arrowheads, in no particular order — the paint order is fixed by
    /// [`Drawing::paint`] and is not the caller's to choose.
    pub pieces: Vec<Piece>,
    /// The values. Painted last, each with its own halo, so each breaks the line it sits on.
    pub labels: Vec<Label>,
    /// Which ink the whole gizmo takes.
    pub rank: Rank,
}

impl Drawing {
    /// Paint the gizmo: all halos, then all ink, then the values.
    ///
    /// The order is the module's central rule and lives here rather than at any call site — a
    /// caller that painted these in a different order would reintroduce exactly the bug the
    /// bucketing exists to prevent.
    pub fn paint(&self, painter: &Painter) {
        let halo = Stroke::new(LINE_WIDTH + HALO_WIDTH, color_palette::BG);
        let ink = Stroke::new(LINE_WIDTH, self.rank.color());

        for pass in [halo, ink] {
            for piece in &self.pieces {
                match piece {
                    Piece::Polyline(points) => {
                        painter.add(Shape::line(points.clone(), pass));
                    }
                    // An arrowhead is filled, so its halo is the same shape stroked outward.
                    Piece::Head(points) => {
                        if pass.color == color_palette::BG {
                            painter.add(Shape::convex_polygon(
                                points.to_vec(),
                                color_palette::BG,
                                Stroke::new(HALO_WIDTH, color_palette::BG),
                            ));
                        } else {
                            painter.add(Shape::convex_polygon(
                                points.to_vec(),
                                pass.color,
                                Stroke::NONE,
                            ));
                        }
                    }
                }
            }
        }

        for label in &self.labels {
            self.paint_label(painter, label);
        }
    }

    /// A box around each value, in screen space, for a caller that has to make the number
    /// CLICKABLE.
    ///
    /// A dimension is the one relation with no badge — the number IS the mark — so the number is
    /// also the only thing a click can land on to select or edit it. The extent is estimated from
    /// the monospace advance rather than laid out, because the shell hit-tests before it has a
    /// painter; the type is monospace precisely so that estimate is exact in width.
    ///
    /// Axis-aligned around the ROTATED text, so an angled value stays clickable over its whole
    /// run rather than only where an unrotated box happened to cover it.
    #[must_use]
    pub fn label_boxes(&self) -> Vec<Rect> {
        self.labels
            .iter()
            .map(|label| {
                let size = Vec2::new(value_width(&label.text), VALUE_SIZE);
                Rect::from_points(&label_corners(label, size))
            })
            .collect()
    }

    fn paint_label(&self, painter: &Painter, label: &Label) {
        let color = self.rank.color();
        let galley =
            painter.layout_no_wrap(label.text.clone(), FontId::monospace(VALUE_SIZE), color);
        let size = galley.size();
        let at = label_corners(label, size)[0];

        // The value's halo IS painted over the dimension line: a line is supposed to break where
        // the number sits, which is the one place the two-pass rule is deliberately broken.
        painter.add(
            egui::epaint::TextShape::new(at, galley, color)
                .with_angle(label.radians)
                .with_underline(Stroke::NONE),
        );
    }
}

/// Fold a bearing into `(-90°, 90°]` so aligned text is never upside-down.
///
/// Total rather than a special case: callers hand this a bearing from any quadrant — a leader
/// angle, an arc tangent — and it stays readable without any of them checking first.
pub fn upright_radians(radians: f32) -> f32 {
    let turn = std::f32::consts::TAU;
    let mut folded = radians.rem_euclid(turn);
    if folded > std::f32::consts::FRAC_PI_2 && folded <= 3.0 * std::f32::consts::FRAC_PI_2 {
        folded -= std::f32::consts::PI;
    } else if folded > 3.0 * std::f32::consts::FRAC_PI_2 {
        folded -= turn;
    }
    folded
}

/// How wide a value will be once laid out — known before anything is painted.
/// The four corners of a value's box, top-left first, in the order the text is laid out.
///
/// egui draws a galley from its top-left and rotates about that point, so the offset that realises
/// the anchor and the lift has to be applied in the ROTATED frame. Written once because the paint
/// path and [`Drawing::label_boxes`] have to agree exactly — a hit target that missed the mark it
/// is standing for would be a click that does nothing on something the author can plainly see.
fn label_corners(label: &Label, size: Vec2) -> [Pos2; 4] {
    let along = Vec2::new(label.radians.cos(), label.radians.sin());
    let normal = Vec2::new(along.y, -along.x);
    let shift = match label.anchor {
        Anchor::Middle => -size.x / 2.0,
        Anchor::Start => 0.0,
        Anchor::End => -size.x,
    };
    let top_left = label.at + along * shift + normal * (label.lift + size.y);
    let down = -normal;
    [
        top_left,
        top_left + along * size.x,
        top_left + along * size.x + down * size.y,
        top_left + down * size.y,
    ]
}

pub(crate) fn value_width(text: &str) -> f32 {
    text.chars().count() as f32 * VALUE_ADVANCE
}

/// A filled arrowhead: nose at `at`, body back along `-direction`.
///
/// `direction` must be a unit vector; it names where the arrow POINTS, so a terminator that flips
/// outside is the same call with the direction negated rather than a second shape.
///
/// **The point is cut off where it meets what it points at.** A terminator lands ON a line — an
/// extension line, a rim, a sketch segment — and a sharp one cannot be painted there. The halo is
/// a stroke, a stroke MITRES, and at this shape's 19° apex it runs `(halo / 2) / sin(9.5°)` ≈ 7
/// past the point: a background spike the length of the arrow itself, laid straight through the
/// line the arrow was terminating on. It grows with the halo, so there is no width at which
/// contrast and a clean termination are both had.
///
/// Cutting the nose back by [`ARROW_SETBACK`] answers both at once. No corner is sharp enough to
/// mitre into a spike, and the halo's own leading edge now lands where the point would have: the
/// arrow's whole painted extent, ink and halo together, stops AT the line instead of crossing it.
/// The nose is four tenths of a point wide, so the terminator still reads as one.
pub(crate) fn arrowhead(at: Pos2, direction: Vec2) -> Piece {
    let across = Vec2::new(-direction.y, direction.x) * ARROW_HALF_WIDTH;
    let (nose, base) = (
        at - direction * ARROW_SETBACK,
        at - direction * ARROW_LENGTH,
    );
    let cut = ARROW_SETBACK / ARROW_LENGTH;
    Piece::Head([
        nose + across * cut,
        nose - across * cut,
        base - across,
        base + across,
    ])
}

#[cfg(test)]
mod tests;
