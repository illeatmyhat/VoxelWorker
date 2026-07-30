//! The **Signal** icon set, painted with `egui` strokes rather than shipped as textures.
//!
//! Every glyph is authored on an **18-unit grid** at a **1.25 pt stroke**, square caps, miter
//! joins, zero rounding — the rules in `docs/design/viewport-chrome-signal.md` §Icon set. A
//! glyph is a `&'static [`[`Mark`]`]` — DATA, painted through [`IconPainter::at`] — so the same
//! source draws at a 15 pt rail button and at a 44 pt palette tile with no second asset and no
//! resampling. Colour is never baked in: the host passes it, which is what lets one glyph be
//! idle, hovered and accent without three copies.
//!
//! Glyphs were once `fn(&IconPainter)`, and the change is not cosmetic: executable glyphs can
//! fail to terminate, and one did — a float-accumulating dash walk hung the reference binary on
//! a white window. Data cannot loop, so the property becomes a fact about the type rather than
//! something anyone has to remember.
//!
//! **The three orbit glyphs are the exception**, and it is a principled one rather than
//! unfinished work. Their paths are TILTED ellipses the kit has no rotation for, and `orbit`'s
//! moon sits at `on_orbit(MOON_ANGLE)` — a trigonometric position. `f32::cos` is not a
//! `const fn`, so the only way to make them data is to freeze those positions as literals, which
//! would decouple the numbers from the `MOON_ANGLE` the module explains. A glyph whose constants
//! no longer match its own reasoning is worse than a glyph that traces. They carry no unbounded
//! loop: the sampling is a fixed `0..=32`. [`Icon::marks`] answers an empty slice for these
//! three and for nothing else, which `glyphs_are_data` gates.
//!
//! **One glyph per file**, under `icons/`. The file is the unit a designer edits, the module
//! name is the glyph name, and [`Icon`] is the only index — a glyph that is not in the enum
//! does not exist, so the set cannot grow a shadow member that the reference binary never
//! shows.
//!
//! The catalogue carries a one-line [`Icon::note`] per glyph in the project's own vocabulary
//! (ordered fold, composed body, Measurement, later-wins). That note is not decoration: it is
//! what the `design_reference` binary displays, and it is the difference between an icon sheet
//! and a set of shapes nobody can assign a meaning to.
//!
//! Two glyphs deliberately depart from the harvested sheet, and the reasons are rulings:
//!
//!   * there is no `sealed_part` — every part is a sealed scope, so the word carries no
//!     information in an interface (`docs/design/colour-vocabulary.md` reasons about the same
//!     scarcity for colour). The glyph that survives is [`Icon::ComposedPart`], which says the
//!     thing a user can verify: a part folds in as ONE body.
//!   * there is no `emboss_ridge`. Emboss moves an accumulated surface within a footprint, and
//!     a ridge glyph would lie the moment the amount goes negative, so the footprint mark is
//!     the primary and [`Icon::EmbossRecess`] is its negative-amount variant.

use crate::theme;
use egui::{Color32, Painter, Pos2, Rect, Shape, Stroke};

mod add_point;
mod array;
mod axes_gizmo;
mod blend_curve;
mod box_solid;
mod break_curve;
mod cancel;
mod carve;
mod carve_region;
mod chamfer_distance_angle;
mod chamfer_equal;
mod chamfer_two_distance;
mod chevron_down;
mod chevron_right;
mod close_loop;
mod commit;
mod composed_part;
mod cylinder;
mod density;
/// Generated fixture, read only by [`tests::glyphs_match_the_design_sheet`] — it carries no
/// glyph the app paints, so it stays out of the shipped binary.
#[cfg(test)]
mod design_reference;
mod displace;
mod drawer;
mod emboss;
mod emboss_recess;
mod extend;
mod extrude;
mod fill_region;
mod fillet;
mod fit;
mod flip;
mod fold_cursor;
mod fold_stack;
mod half_space;
mod home;
mod inset;
mod intersect;
pub mod large;
mod line;
mod link;
mod mark;
mod material;
mod measure;
mod mode_booleans;
mod mode_normal;
mod mode_onion;
mod move_copy;
mod offset_curve;
mod onion_scrub;
mod orbit;
mod orbit_center_place;
mod orbit_center_reset;
mod orbit_constrained;
mod orbit_free;
mod outset;
mod pan;
mod part;
mod polyline;
mod probe;
mod rectangle;
mod revolve;
mod root_part;
mod rotate;
mod sculpt_add;
mod search;
mod select_vertex;
mod sketch;
mod sketch_scale;
mod snap_block;
mod snap_none;
mod snap_voxel;
mod sphere;
mod subtract;
mod sweep;
mod three_point_arc;
mod torus;
mod trim;
mod tube;
mod union;
mod view_cube;
mod zoom;

pub use mark::{Ink, InkRole, Mark};

/// The authoring grid every RAIL glyph is traced on: 18 × 18 units. The large tile family
/// ([`large`]) is traced on its own coarser grid — see [`large::GRID`].
pub const GRID: f32 = 18.0;

/// The Signal glyph stroke, in design points.
pub const STROKE_WIDTH: f32 = 1.25;

/// Painting kit handed to a glyph: it owns the mapping from the authoring grid onto the
/// on-screen box, so an icon file never does arithmetic on `Rect`s and cannot drift off the
/// grid.
///
/// The grid is per-painter rather than a constant, because the two glyph families are drawn
/// at different resolutions: the rail set on 18 units at a 1.25 stroke, the tile set on 26 at
/// 1.1. They are separate DRAWINGS of the same nouns, not one asset scaled — so the kit has
/// to be told which grid it is on.
pub struct IconPainter<'a> {
    painter: &'a Painter,
    rect: Rect,
    stroke: Stroke,
    grid: f32,
    accent: Color32,
    construction: Color32,
}

impl<'a> IconPainter<'a> {
    /// Bind the kit to a box on screen for a RAIL glyph. `rect` is the glyph's full 18-unit
    /// square; callers that want padding shrink before calling.
    pub fn new(painter: &'a Painter, rect: Rect, color: Color32) -> Self {
        Self::new_on_grid(painter, rect, color, GRID, STROKE_WIDTH)
    }

    /// Bind the kit for a glyph traced on `grid` units at `stroke_width` design points.
    ///
    /// Only the two families' own modules should call this — [`new`](Self::new) for the rail
    /// set, [`large`] for the tile set. A caller picking its own grid would be authoring a
    /// third family by accident.
    pub fn new_on_grid(
        painter: &'a Painter,
        rect: Rect,
        color: Color32,
        grid: f32,
        stroke_width: f32,
    ) -> Self {
        // The stroke scales with the glyph so a 44 pt tile is not drawn with a hairline meant
        // for a 15 pt rail button — the large-format sheet's own finding, that a large mark
        // rides a proportionally lighter stroke, is why this is a ratio and not a constant.
        let scale = rect.width() / grid;
        Self {
            painter,
            rect,
            stroke: Stroke::new(stroke_width * scale.max(0.55), color),
            grid,
            // When the host has ALREADY painted the glyph in the accent — an armed rail button —
            // an accent mark drawn in the accent would vanish into the line art and the glyph
            // would lose exactly the distinction it was drawn to make. Stepping up to the hover
            // tone keeps the two inks apart in the one state where it matters most.
            accent: if color == theme::ACCENT {
                theme::HANDLE_HOVER
            } else {
                theme::ACCENT
            },
            construction: theme::SKETCH_CONSTRUCTION,
        }
    }

    /// The stroke for an ink role at an opacity — how [`Ink`] resolves itself.
    pub(super) fn inked(&self, role: mark::InkRole, opacity: f32) -> Stroke {
        let color = match role {
            mark::InkRole::LineArt => self.stroke.color,
            mark::InkRole::Accent => self.accent,
            mark::InkRole::Construction => self.construction,
        };
        Stroke::new(
            self.stroke.width,
            if opacity >= 1.0 {
                color
            } else {
                color.gamma_multiply(opacity)
            },
        )
    }

    /// Map a point on the authoring grid onto the glyph box.
    pub fn at(&self, x: f32, y: f32) -> Pos2 {
        Pos2::new(
            self.rect.left() + (x / self.grid) * self.rect.width(),
            self.rect.top() + (y / self.grid) * self.rect.height(),
        )
    }

    /// The glyph's stroke.
    pub fn stroke(&self) -> Stroke {
        self.stroke
    }

    /// The stroke dimmed to `factor` — the set's one legitimate use of opacity, for a receding
    /// edge or a datum line that must not compete with the subject.
    pub fn faint(&self, factor: f32) -> Stroke {
        Stroke::new(self.stroke.width, self.stroke.color.gamma_multiply(factor))
    }

    /// Trace a polyline through grid points.
    pub fn line(&self, points: &[(f32, f32)]) {
        self.line_with(points, self.stroke);
    }

    /// Trace a polyline in a given stroke.
    pub fn line_with(&self, points: &[(f32, f32)], stroke: Stroke) {
        let mapped: Vec<Pos2> = points.iter().map(|&(x, y)| self.at(x, y)).collect();
        for pair in mapped.windows(2) {
            self.painter.line_segment([pair[0], pair[1]], stroke);
        }
    }

    /// Trace a closed polygon outline.
    pub fn closed(&self, points: &[(f32, f32)]) {
        self.closed_with(points, self.stroke);
    }

    /// Trace a closed polygon outline in a given stroke.
    pub fn closed_with(&self, points: &[(f32, f32)], stroke: Stroke) {
        let mut looped: Vec<(f32, f32)> = points.to_vec();
        if let Some(&first) = points.first() {
            looped.push(first);
        }
        self.line_with(&looped, stroke);
    }

    /// Fill a convex polygon — used where a glyph states a *region* rather than an edge.
    pub fn fill(&self, points: &[(f32, f32)]) {
        self.fill_with(points, self.stroke.color);
    }

    /// Fill a convex polygon in a given colour.
    ///
    /// Pair with [`faint`](Self::faint)`(f).color` to shade a face back: several faces of one
    /// solid at descending weights read as a lit body, which is a thing no outline can say.
    pub fn fill_with(&self, points: &[(f32, f32)], color: Color32) {
        let mapped: Vec<Pos2> = points.iter().map(|&(x, y)| self.at(x, y)).collect();
        self.painter
            .add(Shape::convex_polygon(mapped, color, Stroke::NONE));
    }

    /// Stroke the axis-aligned rectangle spanned by two grid corners.
    pub fn rect(&self, a: (f32, f32), b: (f32, f32)) {
        self.rect_with(a, b, self.stroke);
    }

    /// Stroke a rectangle in a given stroke.
    pub fn rect_with(&self, a: (f32, f32), b: (f32, f32), stroke: Stroke) {
        self.closed_with(&[a, (b.0, a.1), b, (a.0, b.1)], stroke);
    }

    /// Stroke a dashed rectangle — the set's mark for "an operand", the thing a boolean reads.
    pub fn dashed_rect(&self, a: (f32, f32), b: (f32, f32)) {
        self.dashed_rect_with(a, b, self.stroke);
    }

    /// Stroke a dashed rectangle in a given stroke.
    ///
    /// Dashed once per SIDE rather than once around the outline, so every side begins on a full
    /// dash and the corners stay square.
    pub fn dashed_rect_with(&self, a: (f32, f32), b: (f32, f32), stroke: Stroke) {
        for side in [
            [a, (b.0, a.1)],
            [(b.0, a.1), b],
            [b, (a.0, b.1)],
            [(a.0, b.1), a],
        ] {
            self.dashed_polyline_with(&side, stroke);
        }
    }

    /// Stroke a dashed segment on the grid (2.2 on, 1.8 off, in grid units).
    pub fn dashed_line(&self, a: (f32, f32), b: (f32, f32)) {
        self.dash_path(&[self.at(a.0, a.1), self.at(b.0, b.1)], self.stroke);
    }

    /// Stroke a dashed ellipse — a boundary that is felt rather than drawn (a brush's
    /// falloff, a dilation envelope), sampled as a polyline and dashed by arc length so the
    /// rhythm stays even the whole way round.
    pub fn dashed_ellipse(&self, center: (f32, f32), rx: f32, ry: f32) {
        self.dashed_ellipse_with(center, rx, ry, self.stroke);
    }

    /// Stroke a dashed polyline through grid points.
    ///
    /// For paths the axis-aligned helpers cannot express — a TILTED ellipse, say, which the
    /// kit has no rotation for. The caller supplies the points; the dash rhythm is walked by
    /// arc length across the whole strip, so it stays even around a curve.
    pub fn dashed_polyline(&self, points: &[(f32, f32)]) {
        self.dashed_polyline_with(points, self.stroke);
    }

    /// Stroke a dashed polyline in a given stroke.
    pub fn dashed_polyline_with(&self, points: &[(f32, f32)], stroke: Stroke) {
        let mapped: Vec<Pos2> = points.iter().map(|&(x, y)| self.at(x, y)).collect();
        self.dash_path(&mapped, stroke);
    }

    /// Stroke a dashed ellipse in a given stroke.
    pub fn dashed_ellipse_with(&self, center: (f32, f32), rx: f32, ry: f32, stroke: Stroke) {
        let segments = ((self.rect.width() * 0.9) as usize).clamp(24, 96);
        let points: Vec<Pos2> = (0..=segments)
            .map(|i| {
                let t = std::f32::consts::TAU * (i as f32 / segments as f32);
                self.at(center.0 + rx * t.cos(), center.1 + ry * t.sin())
            })
            .collect();
        self.dash_path(&points, stroke);
    }

    /// Walk a polyline in screen space, painting the set's dash rhythm along it.
    ///
    /// The phase carries across the segments of ONE call, which is what keeps a dashed
    /// ellipse — sampled as dozens of short chords — from restarting its rhythm at every
    /// sample. [`dashed_rect`](Self::dashed_rect) deliberately calls this once per side, so
    /// each side still begins on a full dash and the corners stay square.
    ///
    /// The period is in the painter's own grid units, so it is the same visual rhythm
    /// whichever family is drawing (the rail's 18-unit grid or the tile set's 26).
    fn dash_path(&self, points: &[Pos2], stroke: Stroke) {
        let period = (2.2 + 1.8) / self.grid * self.rect.width();
        let on = 2.2 / self.grid * self.rect.width();
        // Below about half a pixel the rhythm cannot resolve on screen — a solid line is what
        // it would read as anyway — and refusing it keeps the dash count below bounded.
        if period < 0.5 {
            for pair in points.windows(2) {
                self.painter.line_segment([pair[0], pair[1]], stroke);
            }
            return;
        }
        // How far into the rhythm this segment begins, wrapped into one period so the
        // arithmetic stays exact however long the path is.
        let mut phase_at_segment_start = 0.0_f32;
        for pair in points.windows(2) {
            let span = pair[1] - pair[0];
            let length = span.length();
            if length <= f32::EPSILON {
                continue;
            }
            let direction = span / length;
            // Dashes are placed by INDEX, never by advancing a cursor. The cursor form landed
            // exactly on a dash boundary every step, so whenever rounding put it one ULP short
            // the next advance (~1e-7) was smaller than the cursor's own precision: the cursor
            // did not move and the walk span forever. A hang, not a cosmetic error — it hung
            // `design_reference` on a plain 18 pt dashed line.
            let dashes = ((length + phase_at_segment_start) / period).floor() as i64 + 1;
            for dash_index in 0..dashes {
                let dash_start = dash_index as f32 * period - phase_at_segment_start;
                let start = dash_start.max(0.0);
                let end = (dash_start + on).min(length);
                if end > start {
                    self.painter.line_segment(
                        [pair[0] + direction * start, pair[0] + direction * end],
                        stroke,
                    );
                }
            }
            phase_at_segment_start = (phase_at_segment_start + length) % period;
        }
    }

    /// Stroke a circle centred on a grid point.
    pub fn circle(&self, center: (f32, f32), radius: f32) {
        self.circle_with(center, radius, self.stroke);
    }

    /// A SOLID disc — a mark too small to be a ring.
    ///
    /// At 15 pt a two-pixel ring is mush where a two-pixel dot is crisp, so a disc is the
    /// right choice for anything genuinely tiny (the `orbit` moon). What does NOT survive at
    /// rail size is the *distinction* between a disc and a ring — which is why the tile
    /// `sketch` can say "grabbable handle" with a filled endpoint and its rail twin cannot.
    pub fn filled_circle(&self, center: (f32, f32), radius: f32) {
        let scaled = (radius / self.grid) * self.rect.width();
        self.painter
            .circle_filled(self.at(center.0, center.1), scaled, self.stroke.color);
    }

    /// Trace a cubic Bézier through its four control points, sampled as a polyline.
    ///
    /// The sample count follows the on-screen size for the same reason [`arc`](Self::arc)'s
    /// does: a 44 pt tile must not show facets and a 15 pt button must not pay for segments it
    /// cannot resolve.
    pub fn cubic(&self, p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), p3: (f32, f32)) {
        self.cubic_with(p0, p1, p2, p3, self.stroke);
    }

    /// Trace a cubic Bézier in a given stroke.
    pub fn cubic_with(
        &self,
        p0: (f32, f32),
        p1: (f32, f32),
        p2: (f32, f32),
        p3: (f32, f32),
        stroke: Stroke,
    ) {
        self.line_with(&self.cubic_points(p0, p1, p2, p3), stroke);
    }

    /// Sample a cubic Bézier into grid-space points, at the same size-adaptive count an arc
    /// uses.
    fn cubic_points(
        &self,
        p0: (f32, f32),
        p1: (f32, f32),
        p2: (f32, f32),
        p3: (f32, f32),
    ) -> Vec<(f32, f32)> {
        let segments = ((self.rect.width() * 0.9) as usize).clamp(12, 64);
        (0..=segments)
            .map(|i| {
                let t = i as f32 / segments as f32;
                let u = 1.0 - t;
                let (a, b, c, d) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
                (
                    a * p0.0 + b * p1.0 + c * p2.0 + d * p3.0,
                    a * p0.1 + b * p1.1 + c * p2.1 + d * p3.1,
                )
            })
            .collect()
    }

    /// Stroke a circle in a given stroke.
    pub fn circle_with(&self, center: (f32, f32), radius: f32, stroke: Stroke) {
        self.ellipse_with(center, radius, radius, stroke);
    }

    /// Stroke an axis-aligned ellipse in grid units — the set's roundness mark (a sphere's
    /// equator, a cylinder's cap), sampled as a polyline so it inherits the same stroke.
    pub fn ellipse(&self, center: (f32, f32), rx: f32, ry: f32) {
        self.ellipse_with(center, rx, ry, self.stroke);
    }

    /// Stroke an ellipse in a given stroke.
    pub fn ellipse_with(&self, center: (f32, f32), rx: f32, ry: f32, stroke: Stroke) {
        self.arc_with(center, rx, ry, 0.0, std::f32::consts::TAU, stroke);
    }

    /// Stroke an elliptical arc, angles in radians, clockwise from +x on the grid (y grows
    /// downward, matching the SVG the set was authored in).
    pub fn arc(&self, center: (f32, f32), rx: f32, ry: f32, from: f32, to: f32) {
        self.arc_with(center, rx, ry, from, to, self.stroke);
    }

    /// Stroke an arc in a given stroke.
    pub fn arc_with(
        &self,
        center: (f32, f32),
        rx: f32,
        ry: f32,
        from: f32,
        to: f32,
        stroke: Stroke,
    ) {
        self.line_with(&self.arc_points(center, rx, ry, from, to), stroke);
    }

    /// Sample an arc into grid-space points.
    ///
    /// Segment count follows the on-screen size, so a 44 pt tile does not show facets and a
    /// 15 pt button does not pay for 64 segments it cannot resolve.
    fn arc_points(
        &self,
        center: (f32, f32),
        rx: f32,
        ry: f32,
        from: f32,
        to: f32,
    ) -> Vec<(f32, f32)> {
        let segments = ((self.rect.width() * 0.9) as usize).clamp(12, 64);
        (0..=segments)
            .map(|i| {
                let t = from + (to - from) * (i as f32 / segments as f32);
                (center.0 + rx * t.cos(), center.1 + ry * t.sin())
            })
            .collect()
    }
}

/// Which shelf of the set a glyph belongs to. The grouping is the vocabulary's own, not a
/// visual one: a reader looking for "the mark for a subtract" looks under [`Group::Combine`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Group {
    /// Camera verbs — the rail beneath the view cube.
    Navigation,
    /// The three exclusive viewer modes, plus the layer scrub.
    ViewerModes,
    /// The `CombineOp` a node carries in the ordered fold.
    Combine,
    /// Field decorators: what measures rather than classifies.
    Fields,
    /// What makes a body — the sketch→volume atom and its sugar.
    Producers,
    /// Document structure: parts, the root, the fold itself.
    Structure,
    /// Authoring tools.
    Tools,
    /// The sketch-mode rail: the profile-vertex tools and their position snap (ADR 0028).
    Sketch,
    /// Sketch tools that ADD an entity to the sketch (ADR 0035).
    SketchCreate,
    /// Sketch tools that change entities already there — every one needs curve–curve
    /// intersection, which is why they land as one group and not scattered through the rail.
    SketchModify,
    /// Sketch tools that read existing entities and EMIT new ones: mirror and the two patterns.
    ///
    /// Neither create nor modify. A create tool asks the pointer for picks and a modify tool
    /// changes what it is given; these take a selection plus a rule and hand back more geometry,
    /// which is why the same glyph has to show both the source and the copies.
    SketchOperator,
    /// The constraint palette: what the solver is told, rather than what is drawn.
    SketchConstraint,
    /// Dimension gizmos — the authored quantities a constraint is driven by (ADR 0029).
    SketchDimension,
    /// Interface furniture.
    Chrome,
}

impl Group {
    /// The shelf's display name.
    pub fn title(self) -> &'static str {
        match self {
            Group::Navigation => "Viewport · navigation",
            Group::ViewerModes => "Viewer modes",
            Group::Combine => "Combine ops",
            Group::Fields => "Fields",
            Group::Producers => "Producers",
            Group::Structure => "Structure",
            Group::Tools => "Tools",
            Group::Sketch => "Sketch mode",
            Group::SketchCreate => "Sketch · create",
            Group::SketchModify => "Sketch · modify",
            Group::SketchOperator => "Sketch · operators",
            Group::SketchConstraint => "Sketch · constraints",
            Group::SketchDimension => "Sketch · dimensions",
            Group::Chrome => "Chrome",
        }
    }

    /// A faint subtitle: what the shelf is *for*, in the project's vocabulary.
    pub fn subtitle(self) -> &'static str {
        match self {
            Group::Navigation => "camera verbs on the rail under the view cube",
            Group::ViewerModes => "exclusive — one mode at a time; the lit one carries the accent",
            Group::Combine => "the CombineOp on a node in the ordered fold",
            Group::Fields => "decorators that measure: outset is a Measurement, and signed",
            Group::Producers => "sketch→volume is the atom; primitives are sugar over it",
            Group::Structure => "parts, the root part, and the fold itself",
            Group::Tools => "what the pointer is currently doing",
            Group::Sketch => "the sketch scope's rail: vertex tools + profile position snap (extrude/revolve reuse Producers)",
            Group::SketchCreate => "adds an entity: the picks carry the accent, so a glyph says what it will ask you for",
            Group::SketchModify => "changes what is already drawn — each one needs curve–curve intersection",
            Group::SketchOperator => "reads a selection and emits more of it: the glyph has to show source AND copies",
            Group::SketchConstraint =>"what the solver is told; a constraint produces no geometry of its own",
            Group::SketchDimension => "authored quantities (ADR 0029) — the parametric handles a solver drives",
            Group::Chrome => "furniture: disclosure, commit, drawer, search",
        }
    }
}

/// One glyph in the set.
///
/// The enum is the index: [`Icon::ALL`] is what the `design_reference` binary walks, so a glyph
/// added to `icons/` without a variant here is invisible and a variant without a file does not
/// compile.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Icon {
    // Navigation.
    Home,
    Fit,
    Orbit,
    OrbitConstrained,
    OrbitFree,
    OrbitCenterPlace,
    OrbitCenterReset,
    Pan,
    Zoom,
    AxesGizmo,
    ViewCube,
    // Viewer modes.
    ModeNormal,
    ModeOnion,
    ModeBooleans,
    OnionScrub,
    // Combine ops.
    Union,
    Subtract,
    Intersect,
    Emboss,
    EmbossRecess,
    // Fields.
    Outset,
    Inset,
    Displace,
    Array,
    // Producers.
    Sketch,
    Extrude,
    Revolve,
    Sweep,
    BoxSolid,
    Sphere,
    Cylinder,
    Tube,
    Torus,
    HalfSpace,
    // Structure.
    Part,
    ComposedPart,
    RootPart,
    FoldStack,
    FoldCursor,
    Link,
    // Tools.
    SculptAdd,
    Carve,
    Measure,
    Probe,
    Material,
    Rotate,
    Flip,
    Density,
    // Sketch mode.
    SelectVertex,
    AddPoint,
    Polyline,
    Rectangle,
    ThreePointArc,
    /// The Line tool: a segment, and the tangent arc it drags into.
    Line,
    // Sketch · modify. Fillet and the three chamfers are one composition with four bridges.
    Fillet,
    ChamferEqual,
    ChamferDistanceAngle,
    ChamferTwoDistance,
    Trim,
    Extend,
    BreakCurve,
    OffsetCurve,
    MoveCopy,
    SketchScale,
    BlendCurve,
    CloseLoop,
    FillRegion,
    CarveRegion,
    SnapNone,
    SnapVoxel,
    SnapBlock,
    // Chrome.
    ChevronRight,
    ChevronDown,
    Commit,
    Cancel,
    Drawer,
    Search,
}

impl Icon {
    /// Every glyph, in catalogue order.
    pub const ALL: &'static [Icon] = &[
        Icon::Home,
        Icon::Fit,
        Icon::Orbit,
        Icon::OrbitConstrained,
        Icon::OrbitFree,
        Icon::OrbitCenterPlace,
        Icon::OrbitCenterReset,
        Icon::Pan,
        Icon::Zoom,
        Icon::AxesGizmo,
        Icon::ViewCube,
        Icon::ModeNormal,
        Icon::ModeOnion,
        Icon::ModeBooleans,
        Icon::OnionScrub,
        Icon::Union,
        Icon::Subtract,
        Icon::Intersect,
        Icon::Emboss,
        Icon::EmbossRecess,
        Icon::Outset,
        Icon::Inset,
        Icon::Displace,
        Icon::Array,
        Icon::Sketch,
        Icon::Extrude,
        Icon::Revolve,
        Icon::Sweep,
        Icon::BoxSolid,
        Icon::Sphere,
        Icon::Cylinder,
        Icon::Tube,
        Icon::Torus,
        Icon::HalfSpace,
        Icon::Part,
        Icon::ComposedPart,
        Icon::RootPart,
        Icon::FoldStack,
        Icon::FoldCursor,
        Icon::Link,
        Icon::SculptAdd,
        Icon::Carve,
        Icon::Measure,
        Icon::Probe,
        Icon::Material,
        Icon::Rotate,
        Icon::Flip,
        Icon::Density,
        Icon::SelectVertex,
        Icon::AddPoint,
        Icon::Polyline,
        Icon::Rectangle,
        Icon::ThreePointArc,
        Icon::Line,
        Icon::Fillet,
        Icon::ChamferEqual,
        Icon::ChamferDistanceAngle,
        Icon::ChamferTwoDistance,
        Icon::Trim,
        Icon::Extend,
        Icon::BreakCurve,
        Icon::OffsetCurve,
        Icon::MoveCopy,
        Icon::SketchScale,
        Icon::BlendCurve,
        Icon::CloseLoop,
        Icon::FillRegion,
        Icon::CarveRegion,
        Icon::SnapNone,
        Icon::SnapVoxel,
        Icon::SnapBlock,
        Icon::ChevronRight,
        Icon::ChevronDown,
        Icon::Commit,
        Icon::Cancel,
        Icon::Drawer,
        Icon::Search,
    ];

    /// Paint the glyph into `rect` in `color`.
    pub fn draw(self, painter: &Painter, rect: Rect, color: Color32) {
        let g = IconPainter::new(painter, rect, color);
        match self {
            // The three orbit marks compose ROTATED ellipses at paint time, which `Mark` cannot
            // describe, so they stay imperative and answer an empty slice from `marks`.
            Icon::Orbit => orbit::draw(&g),
            Icon::OrbitConstrained => orbit_constrained::draw(&g),
            Icon::OrbitFree => orbit_free::draw(&g),
            other => g.marks(other.marks()),
        }
    }

    /// The glyph's marks, as data.
    ///
    /// Empty for the three imperative orbit marks and for nothing else — `glyphs_are_data`
    /// gates that, so an empty answer here is a test failure rather than a blank icon.
    ///
    /// Splitting this out of [`Icon::draw`] is what lets the design sheet's geometry be diffed
    /// against the glyph without painting anything: see `design_reference`.
    pub(super) fn marks(self) -> &'static [Mark] {
        match self {
            Icon::Orbit | Icon::OrbitConstrained | Icon::OrbitFree => &[],
            Icon::Home => home::DRAW,
            Icon::Fit => fit::DRAW,
            Icon::OrbitCenterPlace => orbit_center_place::DRAW,
            Icon::OrbitCenterReset => orbit_center_reset::DRAW,
            Icon::Pan => pan::DRAW,
            Icon::Zoom => zoom::DRAW,
            Icon::AxesGizmo => axes_gizmo::DRAW,
            Icon::ViewCube => view_cube::DRAW,
            Icon::ModeNormal => mode_normal::DRAW,
            Icon::ModeOnion => mode_onion::DRAW,
            Icon::ModeBooleans => mode_booleans::DRAW,
            Icon::OnionScrub => onion_scrub::DRAW,
            Icon::Union => union::DRAW,
            Icon::Subtract => subtract::DRAW,
            Icon::Intersect => intersect::DRAW,
            Icon::Emboss => emboss::DRAW,
            Icon::EmbossRecess => emboss_recess::DRAW,
            Icon::Outset => outset::DRAW,
            Icon::Inset => inset::DRAW,
            Icon::Displace => displace::DRAW,
            Icon::Array => array::DRAW,
            Icon::Sketch => sketch::DRAW,
            Icon::Extrude => extrude::DRAW,
            Icon::Revolve => revolve::DRAW,
            Icon::Sweep => sweep::DRAW,
            Icon::BoxSolid => box_solid::DRAW,
            Icon::Sphere => sphere::DRAW,
            Icon::Cylinder => cylinder::DRAW,
            Icon::Tube => tube::DRAW,
            Icon::Torus => torus::DRAW,
            Icon::HalfSpace => half_space::DRAW,
            Icon::Part => part::DRAW,
            Icon::ComposedPart => composed_part::DRAW,
            Icon::RootPart => root_part::DRAW,
            Icon::FoldStack => fold_stack::DRAW,
            Icon::FoldCursor => fold_cursor::DRAW,
            Icon::Link => link::DRAW,
            Icon::SculptAdd => sculpt_add::DRAW,
            Icon::Carve => carve::DRAW,
            Icon::Measure => measure::DRAW,
            Icon::Probe => probe::DRAW,
            Icon::Material => material::DRAW,
            Icon::Rotate => rotate::DRAW,
            Icon::Flip => flip::DRAW,
            Icon::Density => density::DRAW,
            Icon::SelectVertex => select_vertex::DRAW,
            Icon::AddPoint => add_point::DRAW,
            Icon::Polyline => polyline::DRAW,
            Icon::Rectangle => rectangle::DRAW,
            Icon::ThreePointArc => three_point_arc::DRAW,
            Icon::Line => line::DRAW,
            Icon::Fillet => fillet::DRAW,
            Icon::ChamferEqual => chamfer_equal::DRAW,
            Icon::ChamferDistanceAngle => chamfer_distance_angle::DRAW,
            Icon::ChamferTwoDistance => chamfer_two_distance::DRAW,
            Icon::Trim => trim::DRAW,
            Icon::Extend => extend::DRAW,
            Icon::BreakCurve => break_curve::DRAW,
            Icon::OffsetCurve => offset_curve::DRAW,
            Icon::MoveCopy => move_copy::DRAW,
            Icon::SketchScale => sketch_scale::DRAW,
            Icon::BlendCurve => blend_curve::DRAW,
            Icon::CloseLoop => close_loop::DRAW,
            Icon::FillRegion => fill_region::DRAW,
            Icon::CarveRegion => carve_region::DRAW,
            Icon::SnapNone => snap_none::DRAW,
            Icon::SnapVoxel => snap_voxel::DRAW,
            Icon::SnapBlock => snap_block::DRAW,
            Icon::ChevronRight => chevron_right::DRAW,
            Icon::ChevronDown => chevron_down::DRAW,
            Icon::Commit => commit::DRAW,
            Icon::Cancel => cancel::DRAW,
            Icon::Drawer => drawer::DRAW,
            Icon::Search => search::DRAW,
        }
    }

    /// The glyph's kebab-case name — what a designer calls it.
    pub fn name(self) -> &'static str {
        match self {
            Icon::Home => "home",
            Icon::Fit => "fit",
            Icon::Orbit => "orbit",
            Icon::OrbitConstrained => "orbit-constrained",
            Icon::OrbitFree => "orbit-free",
            Icon::OrbitCenterPlace => "orbit-center-place",
            Icon::OrbitCenterReset => "orbit-center-reset",
            Icon::Pan => "pan",
            Icon::Zoom => "zoom",
            Icon::AxesGizmo => "axes-gizmo",
            Icon::ViewCube => "view-cube",
            Icon::ModeNormal => "mode-normal",
            Icon::ModeOnion => "mode-onion",
            Icon::ModeBooleans => "mode-booleans",
            Icon::OnionScrub => "onion-scrub",
            Icon::Union => "union",
            Icon::Subtract => "subtract",
            Icon::Intersect => "intersect",
            Icon::Emboss => "emboss",
            Icon::EmbossRecess => "emboss-recess",
            Icon::Outset => "outset",
            Icon::Inset => "inset",
            Icon::Displace => "displace",
            Icon::Array => "array",
            Icon::Sketch => "sketch",
            Icon::Extrude => "extrude",
            Icon::Revolve => "revolve",
            Icon::Sweep => "sweep",
            Icon::BoxSolid => "box",
            Icon::Sphere => "sphere",
            Icon::Cylinder => "cylinder",
            Icon::Tube => "tube",
            Icon::Torus => "torus",
            Icon::HalfSpace => "half-space",
            Icon::Part => "part",
            Icon::ComposedPart => "composed-part",
            Icon::RootPart => "root-part",
            Icon::FoldStack => "fold-stack",
            Icon::FoldCursor => "fold-cursor",
            Icon::Link => "link",
            Icon::SculptAdd => "sculpt-add",
            Icon::Carve => "carve",
            Icon::Measure => "measure",
            Icon::Probe => "probe",
            Icon::Material => "material",
            Icon::Rotate => "rotate",
            Icon::Flip => "flip",
            Icon::Density => "density",
            Icon::SelectVertex => "select-vertex",
            Icon::AddPoint => "add-point",
            Icon::Polyline => "polyline",
            Icon::Rectangle => "rectangle",
            Icon::ThreePointArc => "three-point-arc",
            Icon::Line => "line",
            Icon::Fillet => "fillet",
            Icon::ChamferEqual => "chamfer-equal",
            Icon::ChamferDistanceAngle => "chamfer-distance-angle",
            Icon::ChamferTwoDistance => "chamfer-two-distance",
            Icon::Trim => "trim",
            Icon::Extend => "extend",
            Icon::BreakCurve => "break-curve",
            Icon::OffsetCurve => "offset-curve",
            Icon::MoveCopy => "move-copy",
            Icon::SketchScale => "sketch-scale",
            Icon::BlendCurve => "blend-curve",
            Icon::CloseLoop => "close-loop",
            Icon::FillRegion => "fill-region",
            Icon::CarveRegion => "carve-region",
            Icon::SnapNone => "snap-none",
            Icon::SnapVoxel => "snap-voxel",
            Icon::SnapBlock => "snap-block",
            Icon::ChevronRight => "chevron-right",
            Icon::ChevronDown => "chevron-down",
            Icon::Commit => "commit",
            Icon::Cancel => "cancel",
            Icon::Drawer => "drawer",
            Icon::Search => "search",
        }
    }

    /// Which shelf the glyph sits on.
    pub fn group(self) -> Group {
        match self {
            Icon::Home
            | Icon::Fit
            | Icon::Orbit
            | Icon::OrbitConstrained
            | Icon::OrbitFree
            | Icon::OrbitCenterPlace
            | Icon::OrbitCenterReset
            | Icon::Pan
            | Icon::Zoom
            | Icon::AxesGizmo
            | Icon::ViewCube => Group::Navigation,
            Icon::ModeNormal | Icon::ModeOnion | Icon::ModeBooleans | Icon::OnionScrub => {
                Group::ViewerModes
            }
            Icon::Union | Icon::Subtract | Icon::Intersect | Icon::Emboss | Icon::EmbossRecess => {
                Group::Combine
            }
            Icon::Outset | Icon::Inset | Icon::Displace | Icon::Array => Group::Fields,
            Icon::Sketch
            | Icon::Extrude
            | Icon::Revolve
            | Icon::Sweep
            | Icon::BoxSolid
            | Icon::Sphere
            | Icon::Cylinder
            | Icon::Tube
            | Icon::Torus
            | Icon::HalfSpace => Group::Producers,
            Icon::Part
            | Icon::ComposedPart
            | Icon::RootPart
            | Icon::FoldStack
            | Icon::FoldCursor
            | Icon::Link => Group::Structure,
            Icon::SculptAdd
            | Icon::Carve
            | Icon::Measure
            | Icon::Probe
            | Icon::Material
            | Icon::Rotate
            | Icon::Flip
            | Icon::Density => Group::Tools,
            Icon::SelectVertex
            | Icon::AddPoint
            | Icon::Polyline
            | Icon::Rectangle
            | Icon::ThreePointArc
            | Icon::CloseLoop
            | Icon::FillRegion
            | Icon::CarveRegion
            | Icon::SnapNone
            | Icon::SnapVoxel
            | Icon::SnapBlock => Group::Sketch,
            Icon::Line => Group::SketchCreate,
            Icon::Fillet
            | Icon::ChamferEqual
            | Icon::ChamferDistanceAngle
            | Icon::ChamferTwoDistance
            | Icon::Trim
            | Icon::Extend
            | Icon::BreakCurve
            | Icon::OffsetCurve
            | Icon::MoveCopy
            | Icon::SketchScale
            | Icon::BlendCurve => Group::SketchModify,
            Icon::ChevronRight
            | Icon::ChevronDown
            | Icon::Commit
            | Icon::Cancel
            | Icon::Drawer
            | Icon::Search => Group::Chrome,
        }
    }

    /// What the glyph means, in the project's vocabulary. Shown beside it in the reference
    /// binary — a mark whose meaning has to be guessed is a mark that will be misused.
    pub fn note(self) -> &'static str {
        match self {
            Icon::Home => "Snap the camera back to the authored home framing of the root part.",
            Icon::Fit => "Frame the selection — or the whole fold when nothing is selected.",
            Icon::Orbit => "Tumble the camera about the pan target; the body stays put.",
            Icon::OrbitConstrained => {
                "Constrained orbit: world-up stays up, so the camera never rolls — the turntable."
            }
            Icon::OrbitFree => {
                "Free orbit: the trackball. No privileged axis, so roll accumulates and the poles                  are ordinary points."
            }
            Icon::OrbitCenterPlace => {
                "Put the Shift+MMB pivot on the surface under the cursor. The mark IS the marker."
            }
            Icon::OrbitCenterReset => {
                "Send the Shift+MMB pivot back to the world origin, where it started."
            }
            Icon::Pan => "Slide the pan target across the ground plane; the orbit is unchanged.",
            Icon::Zoom => "Dolly in and out. The readout stays in blocks and voxels, never pixels.",
            Icon::AxesGizmo => "The Z-up triad: vertical is +Z, ground is XY, front is −Y.",
            Icon::ViewCube => "The 26-zone cube: faces, edges and corners are all camera stations.",
            Icon::ModeNormal => "Normal: the finished look, as the build will be.",
            Icon::ModeOnion => "Onion: ghost-shaded clip slabs lift the layers apart for scrubbing.",
            Icon::ModeBooleans => "Show booleans: operand bodies x-ray over the folded result.",
            Icon::OnionScrub => "Scrub the layer band — O(1) per step, the clip is uniform-driven.",
            Icon::Union => "Union: the later node adds its body, and carries material.",
            Icon::Subtract => "Subtract: an occupancy-only mask; survivors keep their material.",
            Icon::Intersect => "Intersect: only where both bodies agree survives the fold.",
            Icon::Emboss => {
                "Emboss: within the cutter's footprint the accumulated surface MOVES — \
                 it is not a new body."
            }
            Icon::EmbossRecess => "Emboss with a negative amount: the surface sinks into the body.",
            Icon::Outset => "Outset: dilate a body by a Measurement. Any node may carry one.",
            Icon::Inset => "The same field, signed: a negative outset shrinks, authoring a gap.",
            Icon::Displace => "Displace: a bounded field pushes the surface along its normal.",
            Icon::Array => "Array: one authored node, repeated placements, one entry in the fold.",
            Icon::Sketch => "The authoring atom: a profile, flattened to a polygon at 1/256 block.",
            Icon::Extrude => "Sketch → volume along an axis. An extruded polygon measures SQUARE.",
            Icon::Revolve => "Sketch → volume about an axis. A revolve measures ROUND.",
            Icon::Sweep => "The reserved third lift: a profile carried along a path.",
            Icon::BoxSolid => "Box primitive — sugar over a rectangular sketch, extruded.",
            Icon::Sphere => "Sphere primitive; at low density it reads as the stepped shell it is.",
            Icon::Cylinder => "Cylinder primitive — a circular sketch extruded along +Z.",
            Icon::Tube => "Tube primitive — the cylinder with a bore; an annular sketch, extruded.",
            Icon::Torus => {
                "Torus primitive — a circular sketch revolved about an offset axis. Drawn in \
                 three-quarter view with the bore set high, because a concentric ring reads as \
                 an eye at every ratio."
            }
            Icon::HalfSpace => {
                "Half-space: an unbounded plane. Plane + Subtract replaces a whole trim tool."
            }
            Icon::Part => "A part: the container an assembly is made of.",
            Icon::ComposedPart => {
                "A part folds into its parent as ONE composed body — which is why an outset on \
                 it dilates the whole, and the weakest member's metric wins."
            }
            Icon::RootPart => "The root part: the scene's own container, selectable like any other.",
            Icon::FoldStack => "The ordered fold of the active scope. Later wins.",
            Icon::FoldCursor => "The insert cursor: where the next node lands. Later nodes drop out.",
            Icon::Link => {
                "A linked instance: edit the definition and every instance follows. \
                 Make-unique is the deliberate way out."
            }
            Icon::SculptAdd => "Sculpt: a stroke whose radius is a Measurement, quantised to voxels.",
            Icon::Carve => "Sculpt, removing: the same stroke folded under Subtract.",
            Icon::Measure => "Measure: answer a distance in blocks and voxels, exactly.",
            Icon::Probe => {
                "Why is this voxel like this? The authorship of one cell, in fold order, \
                 with the losers struck through rather than hidden."
            }
            Icon::Material => "Assign a material. Later wins governs the interior.",
            Icon::Rotate => "Rotate by a quarter turn — the lattice admits 24 orientations, no more.",
            Icon::Flip => "Mirror on an axis; like rotation, exact on the lattice.",
            Icon::Density => "Density: voxels per block, bounded 1..=64. Fineness, never size.",
            Icon::SelectVertex => {
                "Sketch: the default arrow — pick and drag a profile vertex. The node at the tip \
                 says 'a point', not a whole body."
            }
            Icon::AddPoint => {
                "Sketch: click a profile edge to insert a vertex there, splitting the segment at \
                 the grid-snapped click."
            }
            Icon::Line => {
                "Sketch: click two points for a segment, or drag from an end for an arc tangent \
                 to it; the seam inherits the run's direction, so the join has no kink."
            }
            Icon::Fillet => {
                "Sketch: round a corner. The legs stop short and an arc bridges them tangentially — \
                 the corner is replaced, not decorated."
            }
            Icon::ChamferEqual => {
                "Sketch: cut a corner off, the same distance along each leg. A 45° bevel."
            }
            Icon::ChamferDistanceAngle => {
                "Sketch: chamfer given one distance and the angle the bevel leaves at."
            }
            Icon::ChamferTwoDistance => {
                "Sketch: chamfer given a distance along each leg, set independently."
            }
            Icon::Trim => {
                "Sketch: delete a curve back to its nearest crossing. The accent is on what GOES — \
                 the only way a still mark can name a deletion."
            }
            Icon::Extend => {
                "Sketch: grow a curve to its nearest boundary. Trim's inverse, and drawn as its \
                 mirror: the accent is the stretch that appears."
            }
            Icon::BreakCurve => {
                "Sketch: split a curve in two at a point. Nothing moves — one entity becomes two \
                 lying exactly where the one did, so the accent is the new vertex."
            }
            Icon::OffsetCurve => {
                "Sketch: a parallel copy at a fixed distance. The outer corner rounds, because \
                 that is what an offset produces."
            }
            Icon::MoveCopy => {
                "Sketch: free transform of the selection. The glyph draws the handle, not a shape, \
                 because the tool has no opinion about what it moves."
            }
            Icon::SketchScale => {
                "Sketch: resize the selection uniformly about a fixed corner. The shared corner \
                 IS the anchor."
            }
            Icon::BlendCurve => {
                "Sketch: a curve joining two others tangentially. The S rather than a C — a blend \
                 between curves pointing the same way has an inflection."
            }
            Icon::Polyline => "Sketch: click to place connected profile points — arbitrary organic outlines.",
            Icon::Rectangle => "Sketch: drag a box into a four-point profile — the box-drag sugar, inside the mode.",
            Icon::ThreePointArc => {
                "Sketch: click start, end, then a point the curve passes through; the arc stores \n                 endpoints plus the solved included angle."
            }
            Icon::CloseLoop => {
                "Sketch: join the open polyline back to its start vertex; the closing run is dashed \
                 until the click commits it."
            }
            Icon::FillRegion => {
                "Sketch: pick this region back — the face inside the boundary resolves as material                  again. Every face starts picked."
            }
            Icon::CarveRegion => {
                "Sketch: unpick this region — the face becomes a hole the profile CSG subtracts,                  keyed by its boundary edges' origins so a drag or a split keeps it."
            }
            Icon::SnapNone => {
                "Sketch position snap — none: the vertex rides sub-voxel under the cursor, the \
                 fraction on offset_local (ADR 0027 reuse)."
            }
            Icon::SnapVoxel => "Sketch position snap — voxel: the vertex locks to the fine lattice crossing. The default.",
            Icon::SnapBlock => "Sketch position snap — block: the vertex locks to block boundaries, for clean inter-part mating.",
            Icon::ChevronRight => "Disclosure, closed.",
            Icon::ChevronDown => "Disclosure, open.",
            Icon::Commit => "Accept: the edit lands in the fold as a node.",
            Icon::Cancel => "Cancel: nothing is written to the document.",
            Icon::Drawer => "The asset drawer — open sets browse; closed sets stay pinned.",
            Icon::Search => "Filter by name.",
        }
    }
}

#[cfg(test)]
mod tests;
