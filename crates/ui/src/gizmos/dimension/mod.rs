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

/// How the sketch plane runs on screen, as the one 3x3 that says it exactly.
///
/// A sketch plane reaches the viewport by an affine into the world, a matrix into clip space and a
/// divide into pixels, and the composition of those on a PLANE is a homography — three by three, no
/// approximation anywhere in it. Carrying that matrix is what lets a dimension ask two questions no
/// pair of screen directions can answer: which way a plane direction runs AT A GIVEN PLACE, since a
/// projection that divides answers differently at every point, and HOW FAR a plane unit reaches
/// there, which is the foreshortening that makes text look like it is lying on the plane rather
/// than merely leaning.
///
/// The inverse is kept beside it, so a mark standing at a bare screen point — a leader's jog, a
/// value's standoff — is answered as exactly as one sitting on the geometry. There is no nearest-
/// on-plane-point approximation and no species distinction: every pixel in the viewport has a plane
/// coordinate under it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneFrame {
    /// Plane coordinates to screen points, homogeneous. The last row is the clip `w`, so its sign
    /// is the in-front-of-the-camera test and it is never rescaled.
    to_screen: [[f64; 3]; 3],
    /// Its inverse, computed once at construction.
    to_plane: [[f64; 3]; 3],
}

/// Multiply a homogeneous 3-vector by a 3x3.
fn carried(matrix: &[[f64; 3]; 3], point: [f64; 3]) -> [f64; 3] {
    let row = |index: usize| {
        matrix[index][0].mul_add(
            point[0],
            matrix[index][1].mul_add(point[1], matrix[index][2] * point[2]),
        )
    };
    [row(0), row(1), row(2)]
}

impl PlaneFrame {
    /// A plane facing the camera, where the plane's axes are the screen's own.
    ///
    /// What a drawing on a flat page has always been laid out in, and what every projected view
    /// approaches as it comes square. Every layout rule in this module reduces to the one it had
    /// before any of this existed when handed this frame — that is the parity the tests hold.
    #[must_use]
    pub fn facing() -> Self {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        Self {
            to_screen: identity,
            to_plane: identity,
        }
    }

    /// The frame for a given plane-to-screen homography, or `None` if it is singular.
    ///
    /// Rows map `(u, v, 1)` to a homogeneous screen point. The last row must be the projection's
    /// own `w` UNSCALED, because its sign is what [`PlaneFrame::at`] culls on.
    #[must_use]
    pub fn from_plane_to_screen(to_screen: [[f64; 3]; 3]) -> Option<Self> {
        let cofactor = |row: usize, column: usize| {
            let (r0, r1) = ((row + 1) % 3, (row + 2) % 3);
            let (c0, c1) = ((column + 1) % 3, (column + 2) % 3);
            to_screen[r0][c0].mul_add(to_screen[r1][c1], -(to_screen[r0][c1] * to_screen[r1][c0]))
        };
        let determinant = (0..3).fold(0.0, |sum, column| {
            to_screen[0][column].mul_add(cofactor(0, column), sum)
        });
        // Only a determinant that cannot be divided by declines. The test used to be
        // `abs() <= f64::EPSILON`, which is an absolute threshold on a quantity carrying units of
        // pixels squared per plane unit squared: it scales as one over the distance squared, so
        // the frame's own idea of whether it existed was a function of how far the author had
        // zoomed out. The honest test is whether the inverse comes out finite, which is the thing
        // the caller actually needs and is free of any scale.
        if !determinant.is_finite() || determinant == 0.0 {
            return None;
        }
        // The adjugate is the cofactor matrix TRANSPOSED, which is where the index swap comes from.
        let mut to_plane = [[0.0; 3]; 3];
        for row in 0..3 {
            for column in 0..3 {
                to_plane[row][column] = cofactor(column, row) / determinant;
            }
        }
        if to_plane
            .iter()
            .any(|row| row.iter().any(|entry| !entry.is_finite()))
        {
            return None;
        }
        Some(Self {
            to_screen,
            to_plane,
        })
    }

    /// Where a plane coordinate lands on screen — `None` behind the camera.
    #[must_use]
    pub fn at(&self, plane: [f64; 2]) -> Option<Pos2> {
        let [x, y, w] = carried(&self.to_screen, [plane[0], plane[1], 1.0]);
        (w > 0.0 && w.is_finite()).then(|| Pos2::new((x / w) as f32, (y / w) as f32))
    }

    /// Which plane coordinate lies under a screen point — every pixel has one.
    #[must_use]
    pub fn plane_of(&self, at: Pos2) -> Option<[f64; 2]> {
        let [u, v, w] = carried(&self.to_plane, [f64::from(at.x), f64::from(at.y), 1.0]);
        (w.abs() > f64::EPSILON && w.is_finite()).then(|| [u / w, v / w])
    }

    /// What the plane's two unit steps reach on screen at `at` — the projection's Jacobian there,
    /// in pixels per plane unit.
    ///
    /// These are the columns everything else in the frame is built out of. Their DIRECTIONS give
    /// the plane's two families of lines as the screen bends them; their LENGTHS give the
    /// foreshortening, which no construction from vanishing points can recover.
    #[must_use]
    pub fn axes_at(&self, at: Pos2) -> Option<[Vec2; 2]> {
        let plane = self.plane_of(at)?;
        let [x, y, w] = carried(&self.to_screen, [plane[0], plane[1], 1.0]);
        if w.abs() <= f64::EPSILON || !w.is_finite() {
            return None;
        }
        let (x, y) = (x / w, y / w);
        let column = |index: usize| {
            Vec2::new(
                ((self.to_screen[0][index] - x * self.to_screen[2][index]) / w) as f32,
                ((self.to_screen[1][index] - y * self.to_screen[2][index]) / w) as f32,
            )
        };
        let axes = [column(0), column(1)];
        // NAMED AND LEFT: the plane coordinate this was evaluated at comes back through the
        // inverse, which is itself ill-conditioned on a near-singular frame — under a projection
        // that divides, the columns are then read at the wrong place. The window where that bites
        // is the window where everything it could mis-draw has already collapsed, so fixing it
        // here would be fixing an unmeasured bug behind a measured one. Under an orthographic
        // projection the Jacobian is constant and the wrong place costs nothing.
        //
        // Finite, and nothing else. A SHORT column is not a failure to answer — it is the answer:
        // the plane reaches almost nowhere in that direction, which is what a plane drawn nearly
        // edge-on does. Declining on shortness meant declining on a quantity measured in pixels
        // per plane unit, so the frame stopped answering once the author zoomed out far enough,
        // and every caller's decline is the SCREEN's reading at full length.
        axes.iter().all(|axis| axis.is_finite()).then_some(axes)
    }

    /// How far OPEN the plane stands to the camera where it passes through `at`: one when its two
    /// unit steps image equally long and square to each other, zero when the plane has collapsed
    /// to a line and there is no second direction left.
    ///
    /// The inverse condition number of the Jacobian — its smaller singular value over its larger —
    /// which is the one reading of a projected plane's health that is DIMENSIONLESS. Every guard
    /// in this drawing that has ever been wrong about a degenerate plane was wrong because it
    /// measured something with units: a length in pixels per plane unit, or a determinant in the
    /// square of that, both of which shrink as the author zooms out and neither of which says
    /// anything about the plane. This does not move when the camera pulls back.
    ///
    /// Not the same as the sine between the two projected DIRECTIONS, and the difference is the
    /// eighth report: an axis can collapse in LENGTH while its direction still reads square, and
    /// at that point the direction being read is the direction of a vector too short to have one.
    ///
    /// Zero where the frame cannot answer at all.
    #[must_use]
    pub fn opening_at(&self, at: Pos2) -> f32 {
        let Some([larger, smaller]) = self.reaches_at(at) else {
            return 0.0;
        };
        if larger <= 0.0 {
            return 0.0;
        }
        let opening = smaller / larger;
        if opening.is_finite() {
            opening.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// The FURTHEST any unit step in the plane reaches on screen at `at`, in points per plane unit.
    ///
    /// The scale a mark that must hold a constant screen size is sized against: divide the size it
    /// wants in points by this and the answer is how big to draw it IN PLANE UNITS. Every other
    /// direction then images shorter, so the mark's largest extent on screen is exactly the size
    /// asked for and it can never overshoot — where normalizing on one named axis blows up the
    /// moment that axis is the collapsed one.
    ///
    /// A MAGNITUDE and not a direction, which is the one class of frame reading that stays honest
    /// as a plane goes edge-on: magnitudes collapse toward zero truthfully, while the direction of a
    /// vector with no length left is noise. Zero where the frame cannot answer at all.
    #[must_use]
    pub fn largest_reach_at(&self, at: Pos2) -> f32 {
        self.reaches_at(at).map_or(0.0, |[larger, _]| larger)
    }

    /// The Jacobian's two singular values at `at`, largest first — how far the plane's best and
    /// worst unit steps reach on screen, in points per plane unit.
    ///
    /// `sigma_1 * sigma_2` is `|det|` and `sigma_1^2 + sigma_2^2` is the squared Frobenius norm, so
    /// both fall out of one quadratic and neither needs an eigen decomposition.
    fn reaches_at(&self, at: Pos2) -> Option<[f32; 2]> {
        let [across, down] = self.axes_at(at)?;
        let determinant = across.x.mul_add(down.y, -(across.y * down.x)).abs();
        let frobenius = across.length_sq() + down.length_sq();
        let gap = frobenius.mul_add(frobenius, -(4.0 * determinant * determinant));
        let larger = ((frobenius + gap.max(0.0).sqrt()) / 2.0).max(0.0).sqrt();
        if !larger.is_finite() || larger <= 0.0 {
            return None;
        }
        let smaller = determinant / larger;
        smaller.is_finite().then_some([larger, smaller])
    }

    /// The plane's own +X where it passes through `at`, as a unit screen direction, folded upright.
    ///
    /// This is the level a shoulder runs along and a value reads along when it has left its
    /// geometry — the plane's level, not the screen's. Pinned to the plane's authored +X rather
    /// than to anything the camera decides, so it cannot swap axis mid-orbit.
    #[must_use]
    pub fn reading_at(&self, at: Pos2) -> Vec2 {
        self.axes_at(at)
            .map_or(Vec2::X, |[across, _]| upright_direction(across))
    }

    /// The image of the point a FRACTION of the way along the plane segment between two screen
    /// points.
    ///
    /// Not `from.lerp(to, fraction)`. A homography carries the segment to the segment, so both
    /// answers lie on the same screen line and no amount of looking at the line will separate
    /// them — but it does not carry the FRACTION, so a mark seated by a screen lerp slides toward
    /// the near end of a receding run. The image of the middle sits at `w_far / (w_near + w_far)`
    /// of the screen chord, and the two only agree when the ends are the same depth.
    ///
    /// That last clause is why this is a bug a drawing shows only SOMETIMES: the drift is a
    /// function of the run's BEARING, so one sketch seats some of its marks right and some of them
    /// wrong, and the wrong ones move as the camera turns. Measured on a 45 degree view of a plane
    /// two units across at 1280 by 800: nothing at all along the view-symmetric diagonal, 28 pixels
    /// across a 382-pixel span at an ordinary three-quarter, 78 pixels close in. A constraint badge
    /// is 32 pixels wide.
    ///
    /// Falls back to the screen lerp where the frame declines — a plane seen edge-on, or a station
    /// that lands behind the camera — which is the seat the drawing had before the frame existed.
    #[must_use]
    pub fn along(&self, from: Pos2, to: Pos2, fraction: f32) -> Pos2 {
        let screens_own = || from + (to - from) * fraction;
        let (Some(near), Some(far)) = (self.plane_of(from), self.plane_of(to)) else {
            return screens_own();
        };
        let fraction = f64::from(fraction);
        let station = [
            (far[0] - near[0]).mul_add(fraction, near[0]),
            (far[1] - near[1]).mul_add(fraction, near[1]),
        ];
        self.at(station)
            .filter(|seat| seat.is_finite())
            .unwrap_or_else(screens_own)
    }

    /// The plane direction SQUARE to `along` at `at`, scaled so `along` stays one unit long.
    ///
    /// Two things at once, and both are the complaint. **The direction** is not `perp(along)`: a
    /// homography carries the plane's right angle to some other screen angle — 31 degrees away at a
    /// three-quarter view — so a value lifted along the screen's perpendicular leans out of the
    /// sketch it annotates. **The length** is the foreshortening, and it is the half a pair of
    /// vanishing points cannot supply. Tilt the camera straight down over a plane with no compound
    /// turn and there is NO shear at all: the plane's axes still image square, one of them merely
    /// images short. A frame that normalized both axes would draw that view identically to a flat
    /// page, which is exactly the view an author takes to look at a sketch from an angle.
    ///
    /// Normalized on `along` rather than on area: a value keeps its type size along its own
    /// baseline at every view, and loses height as the plane recedes, which is what ink on
    /// receding paper does. Holding the AREA instead would inflate the baseline as the height
    /// shrank, and a number that swells as the camera turns is the thing the constant-size rule for
    /// sketch marks exists to prevent.
    ///
    /// Signed to agree with `perp(along)`, which is the same condition as keeping the frame
    /// right-handed — so this one comparison both keeps a drawing lifting its values the way it
    /// always did and stops a plane seen from BEHIND drawing every glyph mirror-image.
    #[must_use]
    pub fn square_to(&self, along: Vec2, at: Pos2) -> Vec2 {
        let screens_own = Vec2::new(along.y, -along.x);
        let Some([across, down]) = self.axes_at(at) else {
            return screens_own;
        };
        // `along` in the plane's own coordinates, made a unit step THERE, turned a quarter there,
        // and carried back — so what returns is the image of a plane right angle.
        let Some([reading, lifting]) = stepped_in_plane([across, down], along) else {
            return screens_own;
        };
        // Only a frame that cannot answer at all falls back to the screen's. A SHORT square is
        // the answer for a plane seen nearly edge-on, and handing back a unit-length screen
        // perpendicular there is the frame telling a figure to stand up out of the plane it is
        // lying in — which is the whole eighth report.
        let pace = (across * reading + down * lifting).length();
        if !pace.is_finite() || pace <= 0.0 {
            return screens_own;
        }
        let square = (across * -lifting + down * reading) / pace;
        if !square.is_finite() {
            return screens_own;
        }
        if square.dot(screens_own) >= 0.0 {
            square
        } else {
            -square
        }
    }

    /// The unit direction halfway between two screen directions IN THE PLANE — `None` where they
    /// double back in the plane and name no side to be between.
    ///
    /// Halfway on screen is not halfway in the plane, for the same reason square is not square: a
    /// homography does not preserve angles, so two projected plane lines are drawn at an angle
    /// the plane never held them at, and splitting THAT angle points somewhere the drawing does
    /// not. The two arms are carried into plane coordinates, made unit steps and added there, and
    /// the sum is carried back.
    ///
    /// Falls back to the screen's own bisector wherever the plane's axes do not span — the flat
    /// reading, which is right exactly where there is nothing to correct.
    #[must_use]
    pub fn bisector_of(&self, first: Vec2, second: Vec2, at: Pos2) -> Option<Vec2> {
        let screens_own = || {
            let sum = first.normalized() + second.normalized();
            (sum.length() > f32::EPSILON).then(|| sum.normalized())
        };
        let Some(axes) = self.axes_at(at) else {
            return screens_own();
        };
        let (Some(first_step), Some(second_step)) = (
            stepped_in_plane(axes, first),
            stepped_in_plane(axes, second),
        ) else {
            return screens_own();
        };
        let sum = [
            first_step[0] + second_step[0],
            first_step[1] + second_step[1],
        ];
        let length = sum[0].hypot(sum[1]);
        if length <= f32::EPSILON {
            // They double back IN THE PLANE, which is the honest no-answer: the screen's bisector
            // would invent a side out of the projection's own distortion.
            return None;
        }
        let carried = axes[0] * (sum[0] / length) + axes[1] * (sum[1] / length);
        if !carried.is_finite() || carried.length() <= f32::EPSILON {
            return screens_own();
        }
        Some(carried.normalized())
    }
}

/// `direction` in the plane's own coordinates at a place its two projected axes are known, as a
/// UNIT step there — `None` where the axes do not span, which is a plane seen edge-on.
///
/// The one decomposition both [`PlaneFrame::square_to`] and [`PlaneFrame::bisector_of`] turn on:
/// a question about the plane's own metric is answered by carrying the screen direction back into
/// the plane, doing the plane geometry THERE, and carrying the answer out again.
fn stepped_in_plane([across, down]: [Vec2; 2], direction: Vec2) -> Option<[f32; 2]> {
    // The adjugate's two rows, WITHOUT the divide by the determinant that would turn them into
    // the plane coordinates themselves. The step is normalized on the next line, so that divide
    // cancels out of the answer completely — everything except its SIGN, which is folded back in
    // at the end. Dividing first would blow up on a plane drawn as a line and then be undone, and
    // the guard that used to stop it blowing up was a threshold on pixels per plane unit: the
    // reason the whole reading was a function of how far the author had zoomed out.
    let spread = across.x.mul_add(down.y, -(across.y * down.x));
    let reading = direction.x.mul_add(down.y, -(direction.y * down.x));
    let lifting = across.x.mul_add(direction.y, -(across.y * direction.x));
    let step = reading.hypot(lifting);
    if !step.is_finite() || step <= 0.0 {
        return None;
    }
    // Near collapse the determinant is f32 noise, so this reads the NOISE's sign and the answer
    // can flip between frames. Self-limiting rather than a flicker to chase: what flips is a
    // vector whose length has collapsed with the plane, so the flip is ratio-scale invisible.
    let turn = if spread < 0.0 { -1.0 } else { 1.0 };
    Some([turn * reading / step, turn * lifting / step])
}

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

    /// Which way the curve RUNS at a bearing: the plane's tangent there, imaged, unit, turning
    /// the way the bearing does.
    ///
    /// A secant, not a derivative: the rim is sampled, so asking either side of the bearing reads
    /// the drawing's own direction there rather than a curve it only approximates. Both samples
    /// are images of plane points, so the chord between them is the image of a plane chord and
    /// tends to the image of the plane tangent.
    ///
    /// Deliberately NOT the screen perpendicular of [`aim`](Self::aim) — see there.
    #[must_use]
    pub fn tangent(self, bearing: f32) -> Vec2 {
        const NUDGE: f32 = 1e-2;
        let along = self.touch(bearing + NUDGE) - self.touch(bearing - NUDGE);
        if along.length() <= f32::EPSILON {
            // A rim collapsed to a point has no direction of its own, so the only answer left is
            // the bearing's own square — which is the right one wherever the drawing is a curve.
            let radial = Vec2::angled(bearing);
            return Vec2::new(-radial.y, radial.x);
        }
        along.normalized()
    }

    /// Which way the curve FACES at a bearing: the outward normal of the PLANE curve, imaged.
    ///
    /// A mark that has to sit SQUARE to the drawing — an arrowhead, and whatever line runs into
    /// its base or leaves it — is aimed by this. Square to a circle **in its own plane** is along
    /// the radius, so this is the ray, and the screen perpendicular of the projected curve is
    /// something else: the image of a different plane direction, off by as much as the tilt. At a
    /// 3:1 squash the two are 84 degrees apart, which is an arrowhead lying flat across its own
    /// rim rather than meeting it.
    ///
    /// **The precondition that makes it the ray is that the rim is struck at the image of the
    /// plane center**, which is what [`touch`](Self::touch) means: it answers a bearing with the
    /// point where the ray from that center meets the drawing. Center and touch are then the
    /// images of two plane points, so the segment between them is the image of the plane radius
    /// exactly, at any perspective. A rim struck at a center that is NOT a projected plane point
    /// would need the projected curve's own perpendicular, and that would have to be a third,
    /// screen-named answer rather than a bend in this one.
    ///
    /// Not perpendicular on screen to [`tangent`](Self::tangent), and that is the point: the
    /// plane's two square directions image at an angle that is not a right one.
    #[must_use]
    pub fn aim(self, bearing: f32) -> Vec2 {
        Vec2::angled(bearing)
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

/// How square a lift has to stand before the drawing stops reaching further out along it.
///
/// The lift is a travel along a PLANE direction, so at a slant it projects short and the value
/// creeps back onto the line it is supposed to clear. Reaching further along that same direction
/// buys the clearance back without leaving the plane, which is the one fix that does not trade the
/// attachment away for the legibility. The floor is the sine the shell already declines a dimension
/// below, so a drawing that reaches it was going to be refused anyway.
const A_LIFT_TOO_SLANTED_TO_CLEAR: f32 = 0.1;

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

/// A value, placed and laid out in the sketch plane.
///
/// Both directions are PLANE directions as the screen sees them, which is the difference between a
/// number that annotates the drawing and one that floats in front of it. They are not each other's
/// perpendicular: a projection carries the plane's right angle to some other screen angle, and
/// carrying both is what lets [`Drawing::paint`] shear the glyphs into that angle.
#[derive(Clone, Debug, PartialEq)]
pub struct Label {
    /// Where the anchor sits.
    pub at: Pos2,
    /// The whole indication, already parenthesised if it is a reference.
    pub text: String,
    /// The baseline direction on screen: unit, and folded upright so the text is never inverted.
    pub along: Vec2,
    /// The plane direction square to [`Label::along`] IN THE PLANE, on the lift side, scaled so a
    /// unit step along the baseline is one unit long.
    ///
    /// Two facts in one vector, and the drawing needs both. Its DIRECTION is not `perp(along)`,
    /// which is the screen's square and the image of some entirely different plane direction. Its
    /// LENGTH is the foreshortening — how short a plane step across the baseline projects compared
    /// with one along it — which is the whole of the tilt cue in a view that has no shear at all.
    /// [`PlaneFrame::square_to`] is the one way to strike this.
    pub across: Vec2,
    /// Which edge of the text `at` refers to.
    pub anchor: Anchor,
    /// How far the text is lifted off the line it rides, as a travel along [`Label::across`].
    pub lift: f32,
}

impl Label {
    /// The baseline's bearing, for a caller that wants the angle rather than the direction.
    #[must_use]
    pub fn radians(&self) -> f32 {
        self.along.y.atan2(self.along.x)
    }
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

    /// Paint one value as GEOMETRY rather than as a turned billboard.
    ///
    /// [`TextShape::with_angle`](egui::epaint::TextShape::with_angle) is a pure rotation, and a
    /// rotation cannot put text in a tilted plane: the plane's own right angle is not a right angle
    /// on screen, so rotated glyphs stand square to a baseline that nothing in the sketch is square
    /// to. The value reads as pasted on the glass. Laying it out in the plane needs a SHEAR, which
    /// is not in `TextShape` at any angle.
    ///
    /// It is in the galley, though, and for free. A laid-out galley already carries its rows as
    /// finished meshes in galley-local points, so this maps each vertex through the plane's own
    /// two directions and emits the mesh directly. Every glyph is one quad with an affine texture
    /// map, so an affine on the quad IS that affine on the glyph image — the shear is exact, and
    /// nothing here re-tessellates a font.
    ///
    /// The uv coordinates a row carries are in TEXELS, which the text path would have normalized
    /// on the way past; a mesh handed over ready-made goes through the tessellator untouched, so
    /// the normalizing happens here against the atlas's current size.
    ///
    /// The origin is snapped to a whole PHYSICAL pixel, which is what the text path does and what
    /// keeps a glyph's own pixel grid landing on the screen's. That is the one place the paint and
    /// [`Drawing::label_boxes`] part company, by at most half a pixel and never structurally: the
    /// box stays the unrounded truth, because a hit target has no reason to care and the snap
    /// depends on a device ratio the shell hit-tests without.
    fn paint_label(&self, painter: &Painter, label: &Label) {
        let color = self.rank.color();
        let galley =
            painter.layout_no_wrap(label.text.clone(), FontId::monospace(VALUE_SIZE), color);
        let (origin, across_page, down_page) = label_frame(label, galley.size());
        let ratio = painter.ctx().pixels_per_point().max(f32::EPSILON);
        let origin = Pos2::new(
            (origin.x * ratio).round() / ratio,
            (origin.y * ratio).round() / ratio,
        );

        let atlas = painter.ctx().fonts(|fonts| fonts.font_image_size());
        let (wide, tall) = (atlas[0].max(1) as f32, atlas[1].max(1) as f32);

        // The value's halo IS painted over the dimension line: a line is supposed to break where
        // the number sits, which is the one place the two-pass rule is deliberately broken.
        for row in &galley.rows {
            if row.visuals.mesh.is_empty() {
                continue;
            }
            let mut mesh = row.visuals.mesh.clone();
            for vertex in &mut mesh.vertices {
                let local = row.pos.to_vec2() + vertex.pos.to_vec2();
                vertex.pos = origin + across_page * local.x + down_page * local.y;
                vertex.uv = egui::pos2(vertex.uv.x / wide, vertex.uv.y / tall);
                if vertex.color == Color32::PLACEHOLDER {
                    vertex.color = color;
                }
            }
            painter.add(Shape::Mesh(mesh.into()));
        }
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

/// Fold a direction upright, so a value laid along it is never read from below.
///
/// The direction form of [`upright_radians`], for the callers that have a vector in hand and want
/// one back — which is all of them since a plane direction is struck rather than named as an angle.
#[must_use]
pub fn upright_direction(direction: Vec2) -> Vec2 {
    let folded = upright_radians(direction.y.atan2(direction.x));
    Vec2::new(folded.cos(), folded.sin())
}

/// How wide a value will be once laid out — known before anything is painted.
/// Where a value's text starts and which way its page runs: top-left, then across, then down.
///
/// The two returned directions span a PARALLELOGRAM, not a rectangle — that is the whole of the
/// coplanarity, and everything else about the value is unchanged. The glyph SIZE stays a screen
/// quantity: both directions come back unit, so a number keeps its type size as the camera turns
/// and only the angle between the two of them carries the tilt. Foreshortening the glyphs as well
/// would be one scale factor here, and it would swell the number at a slant, which is the visible
/// thing the constant-size rule for sketch marks exists to prevent.
///
/// Written once because the paint path and [`Drawing::label_boxes`] have to agree exactly — a hit
/// target that missed the mark it stands for would be a click that does nothing on something the
/// author can plainly see.
fn label_frame(label: &Label, size: Vec2) -> (Pos2, Vec2, Vec2) {
    let along = label.along;
    let normal = Vec2::new(along.y, -along.x);
    // Direction and foreshortening are read apart: the standoff is a CLEARANCE, measured in screen
    // points, while the text's height is a plane travel that is supposed to shorten with the plane.
    let squash = label.across.length();
    let across = if squash > f32::EPSILON {
        label.across / squash
    } else {
        normal
    };
    let shift = match label.anchor {
        Anchor::Middle => -size.x / 2.0,
        Anchor::Start => 0.0,
        Anchor::End => -size.x,
    };
    // The lift is a plane travel that has to clear a screen distance, so it reaches as far along
    // `across` as it takes for the part of it that stands off the line to come to `lift`.
    let reach = label.lift / across.dot(normal).max(A_LIFT_TOO_SLANTED_TO_CLEAR);
    let down_page = -across * squash.max(f32::EPSILON);
    let top_left = label.at + along * shift + across * reach - down_page * size.y;
    (top_left, along, down_page)
}

/// The four corners of a value's box, top-left first, in the order the text is laid out.
fn label_corners(label: &Label, size: Vec2) -> [Pos2; 4] {
    let (top_left, across_page, down_page) = label_frame(label, size);
    [
        top_left,
        top_left + across_page * size.x,
        top_left + across_page * size.x + down_page * size.y,
        top_left + down_page * size.y,
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
