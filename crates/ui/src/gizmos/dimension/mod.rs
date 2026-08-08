//! Dimension gizmos — spans, radii and angles drawn in the viewport at true scale.
//!
//! Unlike the rest of
//! [`gizmos`](crate::gizmos), a dimension is not one shape with a state: it is a small LAYOUT
//! problem whose answer changes with the size of the thing being dimensioned, so this module is
//! split in two. [`span()`], [`radius()`] and [`angle()`] each answer a [`Drawing`] — pure geometry,
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

pub use angle::angle;
pub use diameter::diameter;
pub use radius::radius;
pub use span::span;

/// The chrome weight: dimension line, extension line and leader all share it, per ISO 128-20.
pub const LINE_WIDTH: f32 = 0.8;

/// Added to a stroke's width to make its halo.
pub const HALO_WIDTH: f32 = 2.4;

/// Arrowhead length, and half its base width — so the base is 3.0 and the ratio is 3:1.
pub const ARROW_LENGTH: f32 = 9.0;
const ARROW_HALF_WIDTH: f32 = 1.5;

/// ISO 129-1: the gap left between the feature and the start of its extension line.
pub const GAP: f32 = 5.0;

/// ISO 129-1: how far an extension line runs past the dimension line it crosses.
const OVERRUN: f32 = 8.0;

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
    Polyline(Vec<Pos2>),
    /// A circular arc, `from` to `to` in radians, y running down.
    Arc {
        center: Pos2,
        radius: f32,
        from: f32,
        to: f32,
    },
    /// A filled arrowhead, tip first.
    Head([Pos2; 3]),
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
                    Piece::Arc {
                        center,
                        radius,
                        from,
                        to,
                    } => {
                        painter.add(Shape::line(sample_arc(*center, *radius, *from, *to), pass));
                    }
                    // An arrowhead is filled, so its halo is the same triangle stroked outward.
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

/// Sample an arc into a polyline, finely enough that no facet shows at its on-screen size.
fn sample_arc(center: Pos2, radius: f32, from: f32, to: f32) -> Vec<Pos2> {
    let sweep = to - from;
    let steps = ((radius * sweep.abs() / 2.0).ceil() as usize).clamp(8, 96);
    (0..=steps)
        .map(|i| {
            let t = from + sweep * (i as f32 / steps as f32);
            center + Vec2::new(t.cos(), t.sin()) * radius
        })
        .collect()
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

/// A filled arrowhead: tip at `tip`, body back along `-direction`.
///
/// `direction` must be a unit vector; it names where the arrow POINTS, so a terminator that flips
/// outside is the same call with the direction negated rather than a second shape.
pub(crate) fn arrowhead(tip: Pos2, direction: Vec2) -> Piece {
    let base = tip - direction * ARROW_LENGTH;
    let across = Vec2::new(-direction.y, direction.x) * ARROW_HALF_WIDTH;
    Piece::Head([tip, base + across, base - across])
}

#[cfg(test)]
mod tests;
