//! `segment` / `dashed_segment` / `marked_segment` — a profile edge: committed (solid), closing
//! (dashed), or armed for deletion (warn-red with a `✕`).

use egui::{Painter, Pos2, Stroke, Vec2};

use super::{
    dashed, HandleState, HANDLE_ACCENT, HANDLE_HOVER, STROKE_GUIDE, STROKE_HANDLE, STROKE_SEGMENT,
};
use crate::theme::color_palette;

/// Half-length (points) of the arms of the warn `✕` stamped on a [`marked_segment`] — sized to
/// the vertex handle's own cross so a segment delete-hover and a vertex one read at one scale.
const MARK_CROSS_ARM: f32 = 4.0;

/// A **committed profile segment** — a real edge between two placed vertices. Solid accent; it is
/// an entity, not a preview.
pub fn segment(painter: &Painter, a: Pos2, b: Pos2) {
    painter.line_segment([a, b], Stroke::new(STROKE_SEGMENT, HANDLE_ACCENT));
}

/// The hovered-edge stroke weight — noticeably thicker than the committed [`STROKE_SEGMENT`], not
/// merely brighter, so "the pointer is over this edge" reads at a glance.
const STROKE_SEGMENT_HOVER: f32 = 2.75;
/// The picked-edge stroke weight — thickest of the three, so a selected edge is unmistakable next
/// to both an idle (thin accent) and a hovered (medium bright) one. Thickness is the primary cue
/// — color-only contrast is too weak to see what is selected.
const STROKE_SEGMENT_SELECTED: f32 = 4.0;

/// A committed profile segment drawn in an interaction [`HandleState`] — the edge analog of
/// [`vertex_handle`](super::vertex_handle()), so a point and a segment answer the pointer with one
/// vocabulary. `Idle` is the thin accent edge; `Hover` is a thicker brighter edge (the pointer is
/// over it and it is selectable); `Selected` is the thickest accent edge (picked); `Marked` is the
/// Delete-armed warn edge with a `✕`. `Snapped` is unused for edges and reads as `Idle`.
pub fn styled_segment(painter: &Painter, a: Pos2, b: Pos2, state: HandleState) {
    match state {
        HandleState::Hover => {
            painter.line_segment([a, b], Stroke::new(STROKE_SEGMENT_HOVER, HANDLE_HOVER));
        }
        HandleState::Selected => {
            painter.line_segment([a, b], Stroke::new(STROKE_SEGMENT_SELECTED, HANDLE_ACCENT));
        }
        HandleState::Marked => marked_segment(painter, a, b),
        HandleState::Idle | HandleState::Snapped => segment(painter, a, b),
    }
}

/// The stroke a profile edge takes for a given interaction state and linetype.
///
/// The two are ORTHOGONAL and compose: the linetype picks the ink and whether the line is dashed,
/// the state picks the weight. Flattening role into [`HandleState`] would need a
/// `ConstructionHover`/`ConstructionSelected`/… variant per combination, and there is no such
/// thing as a segment that is construction *instead of* hovered.
///
/// `Marked` is the one state that overrides the linetype's ink: an edge armed for deletion must
/// read destructive whether or not it is construction.
pub fn curve_stroke(state: HandleState, construction: bool) -> Stroke {
    let width = match state {
        HandleState::Hover => STROKE_SEGMENT_HOVER,
        HandleState::Selected => STROKE_SEGMENT_SELECTED,
        HandleState::Idle | HandleState::Snapped | HandleState::Marked => STROKE_SEGMENT,
    };
    let ink = match (state, construction) {
        (HandleState::Marked, _) => color_palette::WARN,
        (_, true) => color_palette::SKETCH_CONSTRUCTION,
        (HandleState::Hover, false) => HANDLE_HOVER,
        (_, false) => HANDLE_ACCENT,
    };
    Stroke::new(width, ink)
}

/// One chord of a profile curve, in its state and linetype.
///
/// Construction geometry is DASHED — the linetype's own definition, and the reason the ink is
/// warm rather than another accent step: dashed-plus-accent already means "uncommitted preview",
/// so the ink is what keeps the two apart on a frame that shows both.
pub fn roled_segment(painter: &Painter, a: Pos2, b: Pos2, state: HandleState, construction: bool) {
    let stroke = curve_stroke(state, construction);
    if construction {
        dashed(painter, a, b, stroke);
    } else {
        painter.line_segment([a, b], stroke);
    }
}

/// A whole profile CURVE, already flattened to chords, in its state and linetype.
///
/// Not a loop over [`roled_segment`]: each dash call restarts the rhythm on a full dash, and a
/// flattened chord is usually shorter than one dash, so chord-by-chord dashing draws a
/// construction curve SOLID. Dashing the polyline in one run is both the correct linetype and one
/// shape instead of one per chord.
pub fn roled_curve(painter: &Painter, chords: &[Pos2], state: HandleState, construction: bool) {
    if chords.len() < 2 {
        return;
    }
    let stroke = curve_stroke(state, construction);
    if construction {
        super::dashed_polyline(painter, chords, stroke);
    } else {
        painter.add(egui::Shape::line(chords.to_vec(), stroke));
    }
}

/// A profile segment **armed for deletion** — the Delete tool is hovering this edge (and no
/// vertex, which would take priority). The whole line goes warn-red with a warn `✕` at its
/// midpoint: the line analog of the vertex handle's [`Marked`](super::HandleState::Marked)
/// state, so a segment delete-hover carries the same destructive vocabulary as a vertex one
/// The whole line colors rather than taking an overlay, so the "this edge goes" cue survives at
/// any zoom.
pub fn marked_segment(painter: &Painter, a: Pos2, b: Pos2) {
    warn_segment(painter, a, b);
    warn_cross(painter, a + (b - a) * 0.5);
}

/// The warn-red line of a delete-armed edge, without the `✕`. A curve stamps the cross once at its
/// own midpoint (see [`crate::chrome::sketch_arc_curves`]), not once per chord.
pub fn warn_segment(painter: &Painter, a: Pos2, b: Pos2) {
    painter.line_segment([a, b], Stroke::new(STROKE_SEGMENT, color_palette::WARN));
}

/// The warn `✕` of a delete-armed edge, centered on `at`.
pub fn warn_cross(painter: &Painter, at: Pos2) {
    warn_cross_sized(painter, at, MARK_CROSS_ARM);
}

/// The warn `✕` at an arbitrary arm length — the delete mark's own glyph, reused wherever
/// something must read as "this cannot happen" rather than "this will be removed".
pub fn warn_cross_sized(painter: &Painter, at: Pos2, arm: f32) {
    let cross = Stroke::new(STROKE_HANDLE, color_palette::WARN);
    painter.line_segment([at + Vec2::splat(-arm), at + Vec2::splat(arm)], cross);
    painter.line_segment(
        [at + Vec2::new(arm, -arm), at + Vec2::new(-arm, arm)],
        cross,
    );
}

/// A **dashed closing run** — the uncommitted segment back to the start vertex, in the family's
/// dashed-means-uncommitted idiom. Becomes a solid [`segment`] once the click commits the loop.
pub fn dashed_segment(painter: &Painter, a: Pos2, b: Pos2) {
    dashed(painter, a, b, Stroke::new(STROKE_SEGMENT, HANDLE_ACCENT));
}

/// The dashed **preview polyline** — [`dashed_segment`]'s whole-run form, for a preview that is a
/// flattened curve rather than one straight run. See [`roled_curve`] for why this cannot be a loop.
pub fn dashed_preview_polyline(painter: &Painter, points: &[Pos2]) {
    if points.len() < 2 {
        return;
    }
    super::dashed_polyline(painter, points, Stroke::new(STROKE_SEGMENT, HANDLE_ACCENT));
}

/// The dashed **guide polyline** — the datum a preview is derived from rather than the shape being
/// authored: a polygon's base circle, a slot's spine.
///
/// Same cool dashed ink as [`dashed_preview_polyline`], at the lighter [`STROKE_GUIDE`] weight the
/// family already reserves for a datum. The weight is the whole distinction, deliberately: a
/// second ink here would collide with CONSTRUCTION, which is what warm dashes already mean.
pub fn dashed_guide_polyline(painter: &Painter, points: &[Pos2]) {
    if points.len() < 2 {
        return;
    }
    super::dashed_polyline(painter, points, Stroke::new(STROKE_GUIDE, HANDLE_ACCENT));
}
