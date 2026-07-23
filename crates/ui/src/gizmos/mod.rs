//! Sketch-mode on-canvas manipulators and cursor states, as reusable `egui` painters.
//!
//! These are the gizmos and pointer states of the sketch scope (ADR 0028) — the profile vertex
//! handle, the open/committed segments, the snap indicators, the close-loop ring, and the pieces
//! the four cursor states are built from. They are **not** [`icons`](crate::icons): a glyph in
//! that set is a single `currentColor` outline on the 18-unit grid, but a manipulator is
//! **two-tone** (a dark thumb with an accent border, filling accent when selected) and
//! **stateful**, so it cannot be one of that family. That is why the design authored them on a
//! separate sheet (`sketch-gizmos.html`) rather than in the icon sheet.
//!
//! **One gizmo per file**, under `gizmos/`, exactly as `icons/` keeps one glyph per file: the
//! file is the unit a designer edits, and `mod.rs` holds only the shared vocabulary (the palette,
//! the stroke weights, the dash rhythm, [`HandleState`], [`Axis`]) and the re-exports.
//!
//! ## These are a SCREEN-SPACE overlay, drawn at PROJECTED positions
//!
//! The sketch is authored on a plane **in 3D**, under the free orbit camera. These primitives are
//! the 2D overlay pass on top of that: the feature projects each profile vertex's world position →
//! a screen [`Pos2`](egui::Pos2) once, then calls these to draw the manipulators there. That is
//! not an approximation — it is how grabbable handles must work over a 3D plane. A handle billboards
//! (constant pixel size, camera-facing) so it stays clickable when the plane tilts edge-on, where a
//! foreshortened one would collapse to a sliver; a straight profile edge projects to a straight 2D
//! segment between its projected endpoints, so [`segment`] is exact in perspective. Curved
//! affordances ([`close_loop_ring`]) are billboards by intent — a fixed-radius ring around the
//! projected vertex, a UI affordance rather than plane geometry.
//!
//! The **working plane itself is NOT here**: it is 3D geometry that foreshortens with the camera,
//! drawn projected (or by the GPU grid renderers, `SceneGridRenderer` / `InfiniteGridRenderer`),
//! never as a flat screen-space rectangle. The `design_reference` catalogue draws a flat plane grid
//! as a stage backdrop only because the sheet has no camera; that flat grid is reference decoration,
//! not a reusable gizmo.
//!
//! One authoring, two consumers: the live sketch overlay and the `design_reference` catalogue,
//! with no second copy to drift. The **second channel is texture, not a second hue**
//! (`docs/design/colour-vocabulary.md`): dashed = uncommitted / a felt boundary, solid = a real
//! placed entity. Snapping IS the constraint vocabulary (ADR 0028 §5), so a snap indicator names
//! *why* a point locked — hence the axis-coloured guides and the label chips.

use egui::{Color32, Painter, Pos2, Rect, Shape, Stroke};

use crate::theme::color_palette;

mod axis_guide;
mod close_loop_ring;
mod crosshair;
mod diamond;
mod ghost_node;
mod label_chip;
mod open_segment;
mod segment;
mod snap_ticks;
mod vertex_handle;

pub use axis_guide::axis_guide;
pub use close_loop_ring::close_loop_ring;
pub use crosshair::crosshair;
pub use diamond::diamond;
pub use ghost_node::ghost_node;
pub use label_chip::label_chip;
pub use open_segment::open_segment;
pub use segment::{dashed_segment, marked_segment, segment, styled_segment};
pub use snap_ticks::snap_ticks;
pub use vertex_handle::{vertex_handle, HandleState};

/// The spatial axis a snap guide follows — its colour IS the constraint it stands in for (ADR
/// 0028 §5). X = warn-red, Y = green, Z = accent, from the shared token table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Axis {
    /// The X in-plane axis — [`color_palette::WARN`].
    X,
    /// The Y in-plane axis — [`color_palette::AXIS_Y`].
    Y,
    /// The Z axis — [`color_palette::ACCENT`].
    Z,
}

impl Axis {
    /// The axis's hue from the shared palette.
    pub fn color(self) -> Color32 {
        match self {
            Axis::X => color_palette::WARN,
            Axis::Y => color_palette::AXIS_Y,
            Axis::Z => color_palette::ACCENT,
        }
    }
}

/// The vertex handle's dark thumb fill — the shipped handle idiom (dark fill · accent border).
pub(crate) const HANDLE_FILL: Color32 = color_palette::BG;
/// The handle border / selected fill: the accent.
pub(crate) const HANDLE_ACCENT: Color32 = color_palette::ACCENT;
/// A hovered handle's / edge's fill+stroke — now the [`color_palette::HANDLE_HOVER`] Signal token, so it
/// appears in the design_reference palette by construction (owner 2026-07-23). Was a raw `#c7d3e0`
/// here, outside the token map.
pub(crate) const HANDLE_HOVER: Color32 = color_palette::HANDLE_HOVER;

/// The manipulator stroke (handles, rings) — the 1.25 pt family weight.
pub(crate) const STROKE_HANDLE: f32 = 1.25;
/// A committed / open segment is a real entity — drawn heavier than the guides.
pub(crate) const STROKE_SEGMENT: f32 = 1.5;
/// A datum: a snap guide, a tick-cross, the kept ghost — the lightest weight.
pub(crate) const STROKE_GUIDE: f32 = 1.0;
/// The dash rhythm, in egui points (the family's 2.2-on / 1.8-off, matching the icon set).
pub(crate) const DASH_ON: f32 = 2.2;
pub(crate) const DASH_OFF: f32 = 1.8;

/// Stroke a dashed straight segment in the family rhythm — the one dash helper the primitives
/// share (egui has no dashed [`Painter`] method).
pub(crate) fn dashed(painter: &Painter, a: Pos2, b: Pos2, stroke: Stroke) {
    painter.extend(Shape::dashed_line(&[a, b], stroke, DASH_ON, DASH_OFF));
}

/// Stroke a dashed rectangle — once per side, so each side begins on a full dash and the corners
/// stay square (the icon set's own rule for dashed rects).
pub(crate) fn dashed_rect(painter: &Painter, rect: Rect, stroke: Stroke) {
    let corners = [rect.left_top(), rect.right_top(), rect.right_bottom(), rect.left_bottom()];
    for i in 0..4 {
        dashed(painter, corners[i], corners[(i + 1) % 4], stroke);
    }
}
