//! The sketch-mode overlay painters: the exit control + immersive border, the
//! add-point insert marker, the committed segment lines, and the profile vertex handles. Drawn at
//! shell-projected positions; the shell owns projection, hit-testing and the drag.

use egui::{Color32, Id, LayerId, Order, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};

use crate::gizmos;
use crate::panel::SketchExit;
use crate::theme;

/// The half-extent (egui points) of a sketch vertex handle's square thumb.
pub const SKETCH_HANDLE_HALF: f32 = 5.0;
/// Extra pixels around a handle's thumb that still count as a grab / chrome hit (the shell's press
/// hit-test uses the same `SKETCH_HANDLE_HALF + SKETCH_HANDLE_GRAB_PAD` radius).
pub const SKETCH_HANDLE_GRAB_PAD: f32 = 5.0;
/// How close (egui points) the cursor must come to an edge for the add-point tool to hover it.
pub const SKETCH_SEGMENT_GRAB_PAD: f32 = 7.0;
/// How far (egui points) a drag's snap may carry the drawing off the cursor.
///
/// Deliberately generous — the author asked for a generous limit, and the snap it bounds is one the
/// drawing is entitled to keep. Roughly three fifths of it holds the quantity exactly and the rest
/// is the falloff letting go, so this is the whole band, not the yank.
///
/// **What the number decides is a gesture LENGTH, and the same one at every zoom.** The cone is a
/// share of travel and travel is read from the cursor, so the ceiling engages once
/// `share × travel` passes it — with travel in these same screen points. At the shares the kernel
/// holds, that is a drag of **120 points before the ceiling touches a radius** and **360 before it
/// touches a span**; below that the gesture's own cone is the narrower of the two and this does
/// nothing at all, which is the intent.
///
/// Zoom cancels out of that comparison entirely, and it is measured:
/// `a_ceiling_in_screen_points_means_the_same_at_every_zoom` scales a drawing, its gesture and its
/// ceiling together fourfold and the answers agree to a part in a million. The one approximation is
/// on this side of the seam — under perspective on a tilted plane, units-per-pixel varies across
/// the screen and the shell measures it once, at the cursor.
pub const SKETCH_SNAP_REACH: f32 = 90.0;
/// The half-extent (egui points) of the add-point insert-preview diamond.
pub const SKETCH_INSERT_MARKER_HALF: f32 = 4.0;
/// Where the snapping mark's arms start and end (egui points from the pick).
///
/// Wider than the ring a snapped vertex handle wears, and deliberately: a handle's ring surrounds
/// a filled thumb that has already caught the eye, while this one has to catch it alone, in the
/// one place the author is not looking — under their own cursor. The gap has to clear the pointer
/// too, which is why the arms start out here rather than at the pick.
pub const SKETCH_SNAP_MARKER_INNER: f32 = SKETCH_HANDLE_HALF + 4.0;
/// The outer end of the snapping mark's arms. See [`SKETCH_SNAP_MARKER_INNER`].
pub const SKETCH_SNAP_MARKER_OUTER: f32 = SKETCH_HANDLE_HALF + 13.0;
/// The snapping mark's own stroke weight (egui points) — heavier than the guide weight the
/// handle's decoration takes, because this one is read against a lit curve rather than a thumb.
pub const SKETCH_SNAP_MARKER_STROKE: f32 = 2.0;
/// How far (egui points) the dark backing extends past the mark's stroke on each side.
pub const SKETCH_SNAP_MARKER_HALO: f32 = 2.5;
/// The side (egui points) of a constraint badge's glyph box. Constant on screen, like every
/// other sketch mark: a badge says *what is asserted*, and a claim does not get smaller with
/// distance.
pub const SKETCH_CONSTRAINT_BADGE: f32 = 32.0;
/// How far (egui points) a badge sits off the geometry it belongs to, and how far successive
/// badges on the same anchor step along that offset. Scaled with the box, so the arrangement
/// reads the same at any badge size.
pub const SKETCH_CONSTRAINT_BADGE_OFFSET: f32 = 30.0;

/// How far a picked badge's plate stands outside the glyph it is behind, total across both sides.
pub const SKETCH_CONSTRAINT_PLATE_MARGIN: f32 = 8.0;

/// The layer every sketch mark is painted on: over the scene, under the floating chrome, and
/// CLIPPED TO THE VIEWPORT.
///
/// A sketch mark is a projection of the drawing, so it goes wherever the drawing goes — including
/// off the side of the viewport when the camera is close or the geometry is wide. Nothing about
/// that projection knows where the panels are, so without the clip the marks paint straight across
/// the side panel and the palette dock, and the drawing appears to float on top of the
/// application. The clip is what makes "the sketch lives in the viewport" true of the marks rather
/// than merely true of the camera.
///
/// [`Order::Middle`] rather than `Foreground` is the second half of the same statement. Every
/// instrument goes through [`chrome_layer`](super::chrome_layer) onto `Foreground`, and two layers
/// sharing an order are emptied in hash-map order — so a mark on `Foreground` was ABOVE or BELOW
/// the rail by luck. `Middle` is this application's only occupant of its tier, which turns the
/// arrangement into something the enum decides.
///
/// It also means the marks are over the root ui, and that is a claim about what is IN the root ui.
/// It cost a release to learn: the DISPLAY stack was a non-allocating child that never named a
/// layer, so it inherited the root's background one and this tier promptly guaranteed the drawing
/// would cover it. See [`chrome_layer`](super::chrome_layer) — a floating instrument names its
/// tier, and the audit is the two doors rather than a search for the ones that mention an order.
///
/// [`crate::gizmos::orbit_reticle_overlay`] takes `Background` for what looks like the same
/// problem, and the PRINCIPLE is the same — pick the tier that makes the requirement a guarantee
/// instead of luck — but the requirement is the opposite one. The reticle is a whole-frame camera
/// mark that the chrome *should* cover, so it wants to be under everything egui draws. Sketch
/// marks are the drawing itself and must not go under the panels. Only the clip is common to both.
fn sketch_mark_painter(ui: &egui::Ui, viewport: Rect, id: &'static str) -> egui::Painter {
    ui.ctx()
        .layer_painter(LayerId::new(Order::Middle, Id::new(id)))
        .with_clip_rect(viewport)
}

/// The sketch-mode exit control + immersive border: a faint accent inset border framing the
/// viewport plus the floating `CANCEL` / `FINISH SKETCH` pair bottom-right; returns the clicked
/// arm. Registers the buttons as chrome so a click never leaks to the camera orbit.
pub fn sketch_exit_control(
    ui: &egui::Ui,
    viewport_rect: Rect,
    chrome_rects: &mut Vec<Rect>,
) -> Option<SketchExit> {
    let painter = ui
        .ctx()
        .layer_painter(super::chrome_layer("sketch_exit_control"));
    painter.rect_stroke(
        viewport_rect.shrink(1.0),
        0.0,
        Stroke::new(2.0_f32, theme::ACCENT_FAINT),
        StrokeKind::Inside,
    );

    let mono = egui::FontId::monospace(10.0);
    let pad = Vec2::new(14.0, 8.0);
    let gap = 9.0;
    let margin = 16.0;
    let bottom = viewport_rect.bottom() - margin;
    let mut right = viewport_rect.right() - margin;
    let mut clicked = None;

    for (exit, label, primary) in [
        (SketchExit::Finish, "FINISH SKETCH", true),
        (SketchExit::Cancel, "CANCEL", false),
    ] {
        // PLACEHOLDER ink so one color-independent layout serves measure + paint.
        let galley = painter.layout_no_wrap(label.to_string(), mono.clone(), Color32::PLACEHOLDER);
        let size = galley.size() + pad * 2.0;
        let rect = Rect::from_min_max(
            Pos2::new(right - size.x, bottom - size.y),
            Pos2::new(right, bottom),
        );
        let response = ui.interact(rect, Id::new(("sketch_exit", label)), Sense::click());
        let hovered = response.hovered();

        painter.rect_filled(rect, 0.0, if primary { theme::ACCENT } else { theme::BG });
        painter.rect_stroke(
            rect,
            0.0,
            Stroke::new(
                1.0_f32,
                if primary {
                    theme::ACCENT
                } else {
                    theme::BORDER
                },
            ),
            StrokeKind::Inside,
        );
        let ink = if primary {
            theme::BG
        } else if hovered {
            theme::HANDLE_HOVER
        } else {
            theme::TEXT_MUTED
        };
        painter.galley(rect.min + pad, galley, ink);

        if response.clicked() {
            clicked = Some(exit);
        }
        chrome_rects.push(rect);
        right = rect.left() - gap;
    }
    clicked
}

/// Draw the add-point insert-preview diamond at `center` (already-projected). Not chrome — a
/// passive preview, so a click passes through to the shell's insert. Clipped to `viewport` like
/// every sketch mark; see [`sketch_mark_painter`].
pub fn sketch_insert_marker(ui: &egui::Ui, viewport: Rect, center: Pos2) {
    let painter = sketch_mark_painter(ui, viewport, "sketch_insert_marker");
    gizmos::diamond(&painter, center, SKETCH_INSERT_MARKER_HALF);
}

/// Draw the **snapping mark** at `center` (already projected): the pick standing on a curve rather
/// than on the grid.
///
/// The tick-cross a dragged handle wears when it engages the lattice, on its own here because
/// there is nothing under it yet to decorate — a drawing tool's pick is not a point until the
/// click. Passive like the insert diamond, so the click passes straight through.
///
/// Drawn heavier than the handle's, and over a dark halo, because it has to survive what it stands
/// on: it appears exactly when a curve under it is lit, and an accent hairline on a lit accent
/// curve is invisible. The halo is what separates the mark from the curve; the weight is what
/// separates it from the guide lines it crosses. A ring at the center closes it, so the mark reads
/// as pointing at a place rather than as four loose strokes.
pub fn sketch_snap_marker(ui: &egui::Ui, viewport: Rect, center: Pos2) {
    let painter = sketch_mark_painter(ui, viewport, "sketch_snap_marker");
    let arms = |color, width| {
        gizmos::snap_ticks_weighted(
            &painter,
            center,
            SKETCH_SNAP_MARKER_INNER,
            SKETCH_SNAP_MARKER_OUTER,
            color,
            width,
        );
    };
    arms(
        theme::BG,
        SKETCH_SNAP_MARKER_STROKE + SKETCH_SNAP_MARKER_HALO,
    );
    painter.circle_stroke(
        center,
        SKETCH_SNAP_MARKER_INNER,
        Stroke::new(
            SKETCH_SNAP_MARKER_STROKE + SKETCH_SNAP_MARKER_HALO,
            theme::BG,
        ),
    );
    arms(theme::ACCENT, SKETCH_SNAP_MARKER_STROKE);
    painter.circle_stroke(
        center,
        SKETCH_SNAP_MARKER_INNER,
        Stroke::new(SKETCH_SNAP_MARKER_STROKE, theme::ACCENT),
    );
}

/// The half-extent (egui points) of the diamond standing on a point a gesture has already taken.
/// Matches [`SKETCH_INSERT_MARKER_HALF`] — both mean "a point that is not an entity yet".
pub const SKETCH_PREVIEW_POINT_HALF: f32 = 4.0;
/// The arm half-length (egui points) of the refusal cross at the cursor.
const SKETCH_REFUSAL_ARM: f32 = 6.0;

/// Which linetype a preview polyline takes.
///
/// Both are cool and dashed — the family's "uncommitted" read, and what keeps a preview distinct
/// from the warm dashes of CONSTRUCTION geometry. The WEIGHT separates them, on the vocabulary the
/// gizmo set already reserved: a real edge's weight for the shape being authored, and the lighter
/// datum weight for the thing it is being derived FROM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SketchPreviewLine {
    /// The geometry this click is about to author.
    Outline,
    /// The construction the outline is derived from — a polygon's base circle, a slot's spine.
    /// Never authored, and gone when the preview is.
    Guide,
}

/// One mark in a drawing tool's preview for THIS frame.
///
/// A tool mid-gesture has more to say than one polyline: the points it has already consumed, the
/// guide it is deriving from, and whether it is currently refusing. One anonymous point list could
/// say none of that, so a three-click tool showed nothing until it had enough for a whole shape.
///
/// Rebuilt from scratch every frame and never retained — a preview has no identity, so a flat list
/// is the honest shape for it.
#[derive(Debug, Clone, PartialEq)]
pub enum SketchPreviewMark {
    /// A point the gesture has already taken. The general multi-step affordance: a tool that has
    /// consumed a click must show what it consumed.
    Point {
        /// Already projected by the shell.
        at: Pos2,
    },
    /// A run of already-projected chords — the shape, or a guide it rests on.
    Polyline {
        /// Already projected by the shell, in order.
        chords: Vec<Pos2>,
        /// Which linetype this run takes.
        line: SketchPreviewLine,
        /// How much of the ink this run gets, from one to nothing.
        ///
        /// A mark that is only PARTLY in force should say so with its ink. A snap's circle is the
        /// case that wanted it: the hold fades continuously to nothing at the rim of its cone, and
        /// a ring drawn at full strength there announces a quantity that is no longer being kept.
        /// Everything that stands unconditionally passes one.
        strength: f32,
    },
    /// The tool cannot complete from where the cursor is. Drawn AT the cursor rather than in the
    /// notice corner: the condition is continuous and cursor-tracking, so it belongs where the
    /// author is already looking, and it lives and dies with the frame instead of needing a
    /// debounce the notice channel would.
    Refused {
        /// Already projected by the shell.
        at: Pos2,
    },
}

/// Draw a drawing tool's preview marks for this frame. Not chrome — a passive preview, so the
/// press/release passes through to the shell.
///
/// Ordered so the reading is bottom-up: guides under the outline they explain, the taken points
/// over both (they are the author's own input and must never be buried), and a refusal on top of
/// everything, because it is the reason nothing else is happening.
pub fn sketch_draw_preview(ui: &egui::Ui, viewport: Rect, marks: &[SketchPreviewMark]) {
    let painter = sketch_mark_painter(ui, viewport, "sketch_draw_preview");
    let polylines = |want: SketchPreviewLine| {
        for mark in marks {
            if let SketchPreviewMark::Polyline {
                chords,
                line,
                strength,
            } = mark
            {
                if *line == want {
                    // One run, not one per chord: a curve preview arrives already flattened, and
                    // restarting the dash rhythm on every short chord would draw it solid.
                    match want {
                        SketchPreviewLine::Outline => {
                            gizmos::dashed_preview_polyline(&painter, chords, *strength);
                        }
                        SketchPreviewLine::Guide => {
                            gizmos::dashed_guide_polyline(&painter, chords, *strength);
                        }
                    }
                }
            }
        }
    };
    polylines(SketchPreviewLine::Guide);
    polylines(SketchPreviewLine::Outline);
    for mark in marks {
        if let SketchPreviewMark::Point { at } = mark {
            gizmos::diamond(&painter, *at, SKETCH_PREVIEW_POINT_HALF);
        }
    }
    for mark in marks {
        if let SketchPreviewMark::Refused { at } = mark {
            gizmos::warn_cross_sized(&painter, *at, SKETCH_REFUSAL_ARM);
        }
    }
}

/// One constraint badge, as the overlay draws it and as the shell's hit-test reads it.
///
/// The id travels WITH the position rather than beside it in a parallel array, because the badge
/// is how a constraint gets picked: a click resolves to a `constraint`, and nothing about that
/// should depend on two vectors staying the same length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstraintBadge {
    /// Where the badge's hit square is centered, in egui points: the image of
    /// [`seat`](Self::seat).
    ///
    /// The hit square is the one part of a badge that stays square to the glass, because a pointer
    /// is a screen thing and a click has no plane coordinate to be at. Everything DRAWN — the glyph
    /// and the selection [`plate`](Self::plate) — is built in the plane.
    pub center: Pos2,
    /// The glyph — the same mark the rail cell that made this constraint carries.
    pub icon: crate::icons::Icon,
    /// The glyph's own center, IN SKETCH PLANE COORDINATES.
    pub seat: [f64; 2],
    /// The plane vector the glyph's +x side runs along, at full glyph width, in PLANE units.
    ///
    /// A plane vector and not a screen one, which is the whole of this type's second attempt at
    /// coplanarity. The first carried the projection's derivative sampled at the badge's center —
    /// a pair of screen directions — and built the mark on the glass from them, which is exact
    /// only for a small mark on a plane the camera is nearly square to. Every point of the glyph
    /// is now a point of the plane, and it reaches the screen the way a curve's vertex does.
    pub across: [f64; 2],
    /// Its +y side, in plane units. The glyph's grid runs y DOWNWARD, so this normally points the
    /// opposite way from the plane's own up.
    pub down: [f64; 2],
    /// The plane's projection, which is what turns the three fields above into pixels.
    ///
    /// The FORWARD map alone. A badge never asks which plane coordinate is under a pixel, and
    /// taking the type that cannot answer that is how it stays drawable at the one camera where the
    /// question has no answer: a plane seen exactly edge-on images to a line, so
    /// [`PlaneFrame`](crate::gizmos::dimension::PlaneFrame) does not exist there, and a badge that
    /// demanded one was substituting the flat page and standing up out of the drawing.
    pub plane: crate::gizmos::dimension::PlaneMap,
    /// The glyph's intended size on screen in points, which is what the PEN scales by.
    ///
    /// The path became the plane's; the pen stays the screen's. Stroke width, dash period and how
    /// many segments a curve is sampled into are all read off this, so a badge on a raked plane is
    /// a squashed drawing at an unsquashed line weight.
    pub span: f32,
    /// The constraint this badge stands for, so a click on it names an entity.
    pub constraint: document::sketch::EntityId,
    /// Whether that constraint is in the selection.
    pub picked: bool,
}

/// The plane's own right and up where it passes through `seat`, each as a signed plane AXIS,
/// chosen so a glyph built on them reads left-to-right and is never drawn mirrored.
///
/// Reads the SIGNS of two imaged directions and nothing else. A sign is the one thing still worth
/// reading on a plane seen edge-on: the magnitudes there are honestly near zero and the directions
/// are noise, so this flips between frames at exactly the cameras where what it flips is a sliver
/// a pixel wide. Self-limiting — and the alternative, pinning the basis to the plane's authored
/// axes, draws every glyph mirrored whenever the author looks at the sketch from behind, which is
/// a wrong drawing at an ordinary camera rather than a shimmer at an extreme one.
///
/// **Exactly edge-on the sign is a coin, so the coin is named rather than left to the arithmetic.**
/// One of the two directions is then exactly zero and has no side to be on; both comparisons are
/// written to fall to `+1` on a zero, so the answer is the plane's own axes unflipped. Reading
/// `signum` and taking whatever it does with a zero would be the same picture and an unrepeatable
/// one — a golden that renders differently on a different rounding, and a mark that can shimmer
/// between two mirror images while the author holds the camera still.
///
/// Public because the shell places badges by stepping out from an anchor along the plane's own
/// diagonal, which is these two summed, and a second copy of the fold could disagree with the one
/// the glyph is actually built on.
#[must_use]
pub fn a_planes_upright_axes(
    plane: crate::gizmos::dimension::PlaneMap,
    seat: [f64; 2],
) -> Option<([f64; 2], [f64; 2])> {
    let here = plane.at(seat)?;
    let image = |step: [f64; 2]| {
        plane
            .at([seat[0] + step[0], seat[1] + step[1]])
            .map(|there| there - here)
    };
    let (rightward, upward) = (image([1.0, 0.0])?, image([0.0, 1.0])?);
    // `>=`, so a plane whose +x images to nothing keeps its authored +x. See the doc above.
    let rights = if rightward.x >= 0.0 { 1.0 } else { -1.0 };
    // A glyph's grid runs y DOWNWARD, so its second side is the plane's up REVERSED, and the pair
    // reads unmirrored when that side turns clockwise from the first on a screen whose y runs down.
    let across = rightward * (rights as f32);
    // `<=`, for the same reason: a zero turn is a plane with no second direction left, and it
    // keeps its authored up.
    let turn = across.x.mul_add(upward.y, -(across.y * upward.x));
    let ups = if turn <= 0.0 { 1.0 } else { -1.0 };
    Some(([rights, 0.0], [0.0, ups]))
}

impl ConstraintBadge {
    /// A badge of the constant screen size, seated at a plane coordinate and lying in that plane.
    ///
    /// **The size is chosen in PLANE UNITS so that it images to [`SKETCH_CONSTRAINT_BADGE`]
    /// points**, which is the whole trick that lets a mark be built entirely in the plane — where
    /// it is coplanar by construction — and still not shrink as the author zooms out. The divisor
    /// is the plane's largest projected reach, so the glyph's largest extent on screen is exactly
    /// the badge size and no direction of it can overshoot.
    ///
    /// `None` where the seat is behind the camera or the plane images to a point, which is the same
    /// rule the drawing itself follows: geometry the camera cannot see carries no mark. A plane
    /// imaged to a LINE is not that case and seats a badge: a line still has a length to be 32
    /// points of, and the badge is a sliver lying along it.
    ///
    /// Everything here is asked of the forward map at the SEAT, never of a projected pixel. That is
    /// the difference between a mark that survives edge-on and one that does not — see
    /// [`plane`](Self::plane).
    #[must_use]
    pub fn seated(
        plane: crate::gizmos::dimension::PlaneMap,
        seat: [f64; 2],
        icon: crate::icons::Icon,
        constraint: document::sketch::EntityId,
        picked: bool,
    ) -> Option<Self> {
        let center = plane.at(seat)?;
        let reach = plane.largest_reach_at_plane(seat);
        if !(reach > 0.0) || !reach.is_finite() {
            return None;
        }
        let size = f64::from(SKETCH_CONSTRAINT_BADGE / reach);
        let (rights, ups) = a_planes_upright_axes(plane, seat)?;
        Some(Self {
            center,
            icon,
            seat,
            across: [rights[0] * size, rights[1] * size],
            // The glyph's grid runs y downward and the plane's second axis runs up.
            down: [-ups[0] * size, -ups[1] * size],
            plane,
            span: SKETCH_CONSTRAINT_BADGE,
            constraint,
            picked,
        })
    }

    /// Where a glyph grid coordinate lands on screen: into the plane first, then through the
    /// plane's own projection.
    ///
    /// The grid's center is the badge's [`seat`](Self::seat), so the glyph is built outward from
    /// the point the badge is about rather than from a corner — which is what makes
    /// [`center`](Self::center) the seat's image and lets the hit square be struck around it.
    ///
    /// A grid point that lands BEHIND the camera collapses onto the badge's center. It takes a
    /// plane whose image runs off the horizon within the width of one badge to reach that, and the
    /// honest drawing there is a degenerate mark rather than a vertex flung across the viewport.
    fn grid_to_screen(self, fit: f64) -> impl Fn(f32, f32) -> Pos2 {
        let grid = f64::from(crate::icons::GRID);
        move |x, y| {
            let across = f64::from(x) / grid - 0.5;
            let down = f64::from(y) / grid - 0.5;
            self.plane
                .at(self.in_plane(across * fit, down * fit))
                .unwrap_or(self.center)
        }
    }

    /// The selection plate's four corners, in ring order, as points of the PLANE.
    ///
    /// The glyph's own square grown by the plate margin — so on a raked plane the plate leans with
    /// the mark it is behind, and on a plane drawn edge-on it is a sliver like everything else.
    /// A corner behind the camera collapses onto the badge's center, the same rule the glyph's own
    /// grid follows.
    #[must_use]
    pub fn plate(self) -> [Pos2; 4] {
        let grow = f64::from((SKETCH_CONSTRAINT_BADGE + SKETCH_CONSTRAINT_PLATE_MARGIN) / 2.0)
            / f64::from(SKETCH_CONSTRAINT_BADGE);
        [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].map(|(across, down)| {
            self.plane
                .at(self.in_plane(across * grow, down * grow))
                .unwrap_or(self.center)
        })
    }

    /// A point of the glyph in plane coordinates, given its offsets along the two sides as
    /// fractions of the full glyph width — so the four corners are the four combinations of
    /// plus and minus a half.
    fn in_plane(self, across: f64, down: f64) -> [f64; 2] {
        [
            self.across[0].mul_add(across, self.down[0].mul_add(down, self.seat[0])),
            self.across[1].mul_add(across, self.down[1].mul_add(down, self.seat[1])),
        ]
    }

    /// How far the glyph must shrink to keep its ink inside the square that claims its click.
    ///
    /// **What is clickable is exactly what is drawn**, and the hit square is screen-axis-aligned
    /// and constant, so a glyph on a raked plane can reach past it: a corner of the projected
    /// parallelogram stands further out than a side does, and at a hard rake both sides can image
    /// near their full length in nearly the same screen direction. Shrinking by the worst corner's
    /// overhang fixes it with no threshold and no second look — square-on the overhang is below one
    /// and nothing moves.
    ///
    /// **Measured on the INK and not on the box**, through [`crate::icons::INK_REACH`]: no glyph on
    /// the constraint shelf fills its own box, so fitting the box would shrink every badge to clear
    /// room nothing occupies.
    ///
    /// All FOUR corners, where the parallelogram this replaced needed only two. A projection that
    /// divides does not carry a corner to the negative of its opposite, so the four are four
    /// separate questions.
    ///
    /// One pass. The correction is a scalar on a mark 32 points wide, over which the projection is
    /// near enough affine that a second pass would move the answer by less than the ink's own
    /// stroke width.
    fn inscribed(self) -> f64 {
        let half = f64::from(SKETCH_CONSTRAINT_BADGE) / 2.0;
        if half <= 0.0 {
            return 1.0;
        }
        let reach = [(0.5, 0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, -0.5)]
            .into_iter()
            .filter_map(|(across, down)| self.plane.at(self.in_plane(across, down)))
            .flat_map(|corner| {
                let away = corner - self.center;
                [f64::from(away.x.abs()), f64::from(away.y.abs())]
            })
            .fold(0.0_f64, f64::max);
        let overhang = reach * f64::from(crate::icons::INK_REACH) / half;
        if overhang > 1.0 {
            1.0 / overhang
        } else {
            1.0
        }
    }
}

/// One dimension gizmo: the laid-out drawing, and what the shell needs to treat it as a target.
///
/// A dimension is the one relation with NO badge — the number is the mark — so the number is also
/// what a click lands on. The constraint id travels with the drawing for the same reason it
/// travels with a [`ConstraintBadge`]: what is clickable is exactly what is drawn.
#[derive(Debug, Clone, PartialEq)]
pub struct DimensionGizmo {
    /// Everything to draw, already projected into egui points and laid out by
    /// [`crate::gizmos::dimension`].
    pub drawing: crate::gizmos::dimension::Drawing,
    /// The constraint this gizmo stands for, so a click on its number names an entity —
    /// `None` for the GHOST of a dimension being placed, which stands for nothing yet and so
    /// must not answer a pick with an id no drawing holds.
    pub constraint: Option<document::sketch::EntityId>,
    /// Whether that constraint is in the selection.
    pub picked: bool,
}

/// Draw the dimension gizmos: each authored quantity as its own measured mark.
///
/// Painted UNDER the constraint badges and over the geometry, in the same spirit: a dimension is a
/// claim about the drawing and must not be buried by it. The paint order within one gizmo is the
/// gizmo module's own and is not this function's to choose.
///
/// A PICKED dimension gets an accent plate behind its value, which is where a badge puts its
/// plate too. The number is the only part of a dimension an author can reliably aim at, so it is
/// the part that reports being held.
pub fn sketch_dimension_gizmos(ui: &egui::Ui, viewport: Rect, gizmos: &[DimensionGizmo]) {
    let painter = sketch_mark_painter(ui, viewport, "sketch_dimension_gizmos");
    for gizmo in gizmos {
        if gizmo.picked {
            for box_rect in gizmo.drawing.label_boxes() {
                let plate = box_rect.expand(3.0);
                painter.rect_filled(plate, 3.0, theme::ACCENT_FAINT);
                painter.rect_stroke(
                    plate,
                    3.0,
                    Stroke::new(1.0_f32, theme::ACCENT),
                    StrokeKind::Inside,
                );
            }
        }
        gizmo.drawing.paint(&painter);
    }
}

/// Draw the constraint badges: each asserted relation's own glyph, standing beside the geometry
/// it names. Positions are shell-projected, so a badge tracks its entity
/// through every camera move — the mark belongs to the entity graph, not to the screen.
///
/// It is the same glyph as the rail cell that made the constraint, in the constraint ink. That
/// correspondence is the whole mechanism: a solve moves geometry until the assertion holds, after
/// which the evidence is a line that merely *looks* level, and only the badge distinguishes
/// "asserted horizontal" from "drawn nearly horizontal".
///
/// A PICKED badge switches to the accent, giving up its role color for as long as it is held.
/// The role says what the mark is and the accent says you have hold of it, and on every other
/// surface in the workspace the second reading wins while it applies.
///
/// Not chrome — a press over a badge is resolved by the shell's own hit-test, which needs the
/// click rather than having egui swallow it.
pub fn sketch_constraint_badges(ui: &egui::Ui, viewport: Rect, badges: &[ConstraintBadge]) {
    paint_constraint_badges(
        &sketch_mark_painter(ui, viewport, "sketch_constraint_badges"),
        badges,
    );
}

/// The badges onto a painter that is already the right one.
///
/// Split out so the design sheet, which is handed a painter and never a `Ui`, draws the badge
/// through this exact code rather than through a second copy of it that could disagree with what
/// the app puts on screen.
pub fn paint_constraint_badges(painter: &egui::Painter, badges: &[ConstraintBadge]) {
    for badge in badges {
        let ink = if badge.picked {
            theme::ACCENT
        } else {
            theme::SKETCH_CONSTRAINT
        };
        if badge.picked {
            // **The plate lies in the plane too.** It was a screen-axis rounded rectangle struck
            // around the badge's center, which is a full-size upright square standing on a drawing
            // that at an edge-on camera has no width at all — the one part of a badge still built
            // on the glass after the glyph moved into the plane. A selection plate is part of the
            // figure, not chrome floating over it, so it is the glyph's own square grown by the
            // same margin and projected corner by corner. The rounding goes with the rectangle:
            // there is no radius to keep once the shape is a projected quad.
            let plate = badge.plate();
            painter.add(egui::Shape::convex_polygon(
                plate.to_vec(),
                theme::ACCENT_FAINT,
                Stroke::new(1.0_f32, theme::ACCENT),
            ));
        }
        let fit = badge.inscribed();
        badge.icon.draw_on_plane(
            painter,
            &badge.grid_to_screen(fit),
            badge.span * (fit as f32),
            ink,
        );
    }
}

/// Draw the directional marquee rubber band. `window` (drag
/// left→right, fully-enclosed semantic) = solid accent outline + the stronger fill; crossing
/// (right→left, any-intersection) = dashed outline + lighter fill — dashed already means
/// "looser" in the gizmo family, so the semantic is legible mid-drag. Not chrome — a passive
/// preview over an already-armed press.
pub fn sketch_marquee_band(ui: &egui::Ui, viewport: Rect, rect: Rect, window: bool) {
    let painter = sketch_mark_painter(ui, viewport, "sketch_marquee_band");
    let stroke = Stroke::new(1.0_f32, theme::ACCENT);
    if window {
        painter.rect_filled(rect, 0.0, theme::MARQUEE_WINDOW_FILL);
        painter.rect_stroke(rect, 0.0, stroke, StrokeKind::Inside);
    } else {
        painter.rect_filled(rect, 0.0, theme::MARQUEE_CROSSING_FILL);
        gizmos::dashed_rect(&painter, rect, stroke);
    }
}

/// One straight profile edge ready to paint: where it is, how the pointer sees it, and whether it
/// is construction. State and linetype are separate fields because they are separate questions —
/// see [`gizmos::curve_stroke`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SketchEdgeLine {
    pub a: Pos2,
    pub b: Pos2,
    pub state: gizmos::HandleState,
    pub construction: bool,
}

/// One curved profile entity ready to paint, as the chords the viewer already tessellated.
#[derive(Debug, Clone, PartialEq)]
pub struct SketchCurveLine {
    pub chords: Vec<Pos2>,
    pub state: gizmos::HandleState,
    pub ink: SketchCurveInk,
}

/// What a curved mark IS, which is what decides how it draws.
///
/// Three answers rather than a `construction` flag, because a tangent lever is neither: it is not
/// the shape (Real) and it is not reference geometry the shape is built against (Construction),
/// it is a manipulator that happens to be a line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SketchCurveInk {
    /// Profile geometry — part of the shape.
    Real,
    /// Reference geometry — locates, but is not part of the shape. Dashed, warm.
    Construction,
    /// A fit point's tangent lever. Solid teal, and nothing snaps or constrains to it.
    TangentLever,
    /// The part of a curve's support the author never drew, shown because a point is standing on
    /// it past the curve's own end. Dashed and quiet — an explanation, not geometry, and nothing
    /// snaps or constrains to it either.
    UndrawnReach,
}

/// What a sketch DOT is, which is what decides its ink at rest.
///
/// Three answers rather than two flags, because the three are mutually exclusive and a pair of
/// booleans would admit a fourth combination that means nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SketchVertexInk {
    /// A curve's drawn path runs through it. The dot is part of the drawing, so it takes the
    /// drawing's ink.
    OnInk,
    /// Nothing is drawn through it — a center, a control point, a free point. It is a handle FOR
    /// the drawing rather than the drawing, and steps back a value to say so.
    OffInk,
    /// One of a tangent lever's two arms — green, its own family, and never at rest in the sense
    /// the other two are.
    TangentArm,
}

/// One sketch point ready to paint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SketchVertexHandle {
    pub at: Pos2,
    pub state: gizmos::HandleState,
    pub ink: SketchVertexInk,
}

/// Draw the committed segment lines between their projected endpoints. Idle edges first, then the
/// single hovered/marked one on top so its brighter line (or warn line + ✕) is never clipped.
pub fn sketch_segment_lines(ui: &egui::Ui, viewport: Rect, lines: &[SketchEdgeLine]) {
    let painter = sketch_mark_painter(ui, viewport, "sketch_segment_lines");
    let draw = |line: &SketchEdgeLine| {
        gizmos::roled_segment(&painter, line.a, line.b, line.state, line.construction);
        if line.state == gizmos::HandleState::Marked {
            gizmos::warn_cross(&painter, line.a + (line.b - line.a) * 0.5);
        }
    };
    for line in lines {
        if line.state == gizmos::HandleState::Idle {
            draw(line);
        }
    }
    for line in lines {
        if line.state != gizmos::HandleState::Idle {
            draw(line);
        }
    }
}

/// Draw the committed arc curves as polylines through their projected chords. Same
/// idle-then-emphasised ordering and the same [`gizmos::HandleState`] vocabulary the segment lines
/// use, so an arc and a straight edge answer the pointer identically. A `Marked` arc stamps its
/// warn `✕` once, at the curve's midpoint, rather than once per chord.
pub fn sketch_arc_curves(ui: &egui::Ui, viewport: Rect, curves: &[SketchCurveLine]) {
    let painter = sketch_mark_painter(ui, viewport, "sketch_arc_curves");
    let draw = |curve: &SketchCurveLine| {
        match curve.ink {
            SketchCurveInk::TangentLever => {
                gizmos::tangent_lever(&painter, &curve.chords, curve.state);
            }
            // No state: an explanation is not a thing the pointer can be over.
            SketchCurveInk::UndrawnReach => gizmos::undrawn_reach(&painter, &curve.chords),
            ink => gizmos::roled_curve(
                &painter,
                &curve.chords,
                curve.state,
                ink == SketchCurveInk::Construction,
            ),
        }
        if curve.state == gizmos::HandleState::Marked {
            if let Some(mid) = curve.chords.get(curve.chords.len() / 2) {
                gizmos::warn_cross(&painter, *mid);
            }
        }
    };
    for curve in curves {
        if curve.state == gizmos::HandleState::Idle {
            draw(curve);
        }
    }
    for curve in curves {
        if curve.state != gizmos::HandleState::Idle {
            draw(curve);
        }
    }
}

/// Draw the profile vertex handles at their projected positions and register each grab rect as
/// chrome so a press drags the vertex instead of orbiting.
pub fn sketch_vertex_handles(
    ui: &egui::Ui,
    viewport: Rect,
    handles: &[SketchVertexHandle],
    chrome_rects: &mut Vec<Rect>,
) {
    let painter = sketch_mark_painter(ui, viewport, "sketch_vertex_handles");
    let grab = SKETCH_HANDLE_HALF + SKETCH_HANDLE_GRAB_PAD;
    for handle in handles {
        // The grab is registered as the part of the reach the viewport actually shows, and a reach
        // it shows nothing of is skipped outright. The rect in `chrome_rects` is the shell's
        // do-not-orbit gate, so anything registered past the clip is an armed grab with no pixels
        // under it — a hold the author cannot see, sitting over the side panel. Intersecting
        // rather than testing keeps that true at the viewport edge as well as outside it.
        let reach = Rect::from_center_size(handle.at, Vec2::splat(grab * 2.0)).intersect(viewport);
        if !reach.is_positive() {
            continue;
        }
        match handle.ink {
            SketchVertexInk::TangentArm => {
                gizmos::tangent_arm_handle(&painter, handle.at, SKETCH_HANDLE_HALF, handle.state);
            }
            ink => gizmos::vertex_handle(
                &painter,
                handle.at,
                SKETCH_HANDLE_HALF,
                handle.state,
                ink == SketchVertexInk::OnInk,
            ),
        }
        chrome_rects.push(reach);
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::{
        sketch_draw_preview, ConstraintBadge, SketchPreviewLine, SketchPreviewMark,
        SKETCH_CONSTRAINT_BADGE,
    };
    use egui::{pos2, Context, Pos2, RawInput, Rect, Vec2};

    /// A viewport the whole test screen fits inside, for the tests that are asking about ink and
    /// arrangement rather than about the clip. The clip is its own test.
    fn whole_screen() -> Rect {
        Rect::EVERYTHING
    }

    /// **Every sketch mark is clipped to the viewport, so a drawing that runs off the side of the
    /// screen does not paint across the panels.**
    ///
    /// Reported from the app: "the egui sketch renders on top of the UI when it should be
    /// constrained to the viewport." A sketch mark is a projection of the drawing and knows
    /// nothing about where the panels are, so a curve that leaves the viewport used to keep going
    /// straight over the side panel and the palette dock.
    ///
    /// What is asserted is the CLIP RECT each shape carries, not where the shape's own vertices
    /// are. Vertices outside the viewport are the normal case and the whole point — a segment
    /// leaving the frame still has to draw the part of itself that stays. The clip is what decides
    /// how much of it lands, so the clip is the thing under test.
    ///
    /// **Seen red**: with the `.with_clip_rect(viewport)` removed from
    /// [`super::sketch_mark_painter`], every mark reported `[0 0] - [400 300]`, the whole screen.
    #[test]
    fn every_sketch_mark_is_clipped_to_the_viewport() {
        let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(400.0, 300.0));
        // A viewport with room on all four sides, so a mark escaping in any direction shows up.
        let viewport = Rect::from_min_max(pos2(60.0, 40.0), pos2(300.0, 240.0));
        // Everything below deliberately runs out past the viewport on at least one side.
        let outside = pos2(380.0, 280.0);
        let inside = pos2(120.0, 90.0);

        let mut chrome = Vec::new();
        let output = Context::default().run_ui(
            RawInput {
                screen_rect: Some(screen),
                ..Default::default()
            },
            |ui| {
                super::sketch_segment_lines(
                    ui,
                    viewport,
                    &[super::SketchEdgeLine {
                        a: inside,
                        b: outside,
                        state: crate::gizmos::HandleState::Idle,
                        construction: false,
                    }],
                );
                super::sketch_arc_curves(
                    ui,
                    viewport,
                    &[super::SketchCurveLine {
                        chords: vec![inside, pos2(200.0, 20.0), outside],
                        state: crate::gizmos::HandleState::Idle,
                        ink: super::SketchCurveInk::Real,
                    }],
                );
                super::sketch_vertex_handles(
                    ui,
                    viewport,
                    &[super::SketchVertexHandle {
                        at: inside,
                        state: crate::gizmos::HandleState::Idle,
                        ink: super::SketchVertexInk::OnInk,
                    }],
                    &mut chrome,
                );
                super::sketch_insert_marker(ui, viewport, outside);
                super::sketch_snap_marker(ui, viewport, outside);
                super::sketch_marquee_band(ui, viewport, Rect::from_min_max(inside, outside), true);
                sketch_draw_preview(
                    ui,
                    viewport,
                    &[SketchPreviewMark::Polyline {
                        chords: vec![inside, outside],
                        line: SketchPreviewLine::Outline,
                        strength: 1.0,
                    }],
                );
            },
        );

        assert!(
            !output.shapes.is_empty(),
            "the marks painted nothing, so the clip is not what this measured"
        );
        for clipped in &output.shapes {
            assert!(
                viewport.contains_rect(clipped.clip_rect),
                "a sketch mark was clipped to {:?}, which reaches outside the viewport {viewport:?}",
                clipped.clip_rect
            );
        }
    }

    /// **A vertex handle the viewport does not reach registers no grab.**
    ///
    /// The clip stops such a handle being DRAWN, which is exactly what makes this worth asserting
    /// separately: a rect left in the chrome list is the shell's do-not-orbit gate, so it would be
    /// an invisible grab sitting over the side panel, belonging to a vertex the author cannot see.
    /// Being unable to see it is what makes it unfindable rather than merely wrong.
    ///
    /// **Seen red**: with the cull removed, both handles registered.
    #[test]
    fn a_handle_outside_the_viewport_registers_no_grab() {
        let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(400.0, 300.0));
        let viewport = Rect::from_min_max(pos2(60.0, 40.0), pos2(300.0, 240.0));
        let handle = |at| super::SketchVertexHandle {
            at,
            state: crate::gizmos::HandleState::Idle,
            ink: super::SketchVertexInk::OnInk,
        };
        let mut chrome = Vec::new();
        let _ = Context::default().run_ui(
            RawInput {
                screen_rect: Some(screen),
                ..Default::default()
            },
            |ui| {
                super::sketch_vertex_handles(
                    ui,
                    viewport,
                    &[handle(pos2(120.0, 90.0)), handle(pos2(370.0, 90.0))],
                    &mut chrome,
                );
            },
        );
        assert_eq!(
            chrome.len(),
            1,
            "only the handle in the viewport: {chrome:?}"
        );
        assert!(viewport.intersects(chrome[0]));
    }

    fn painted(marks: &[SketchPreviewMark]) -> usize {
        let context = Context::default();
        context
            .run_ui(
                RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::splat(32.0))),
                    ..Default::default()
                },
                |ui| sketch_draw_preview(ui, whole_screen(), marks),
            )
            .shapes
            .len()
    }

    /// The strongest stroke alpha any mark in the set painted.
    fn ink_alpha(marks: &[SketchPreviewMark]) -> u8 {
        fn strongest(shape: &egui::Shape) -> u8 {
            match shape {
                egui::Shape::LineSegment { stroke, .. } => stroke.color.a(),
                egui::Shape::Path(path) => match path.stroke.color {
                    egui::epaint::ColorMode::Solid(color) => color.a(),
                    egui::epaint::ColorMode::UV(_) => 0,
                },
                egui::Shape::Vec(inner) => inner.iter().map(strongest).max().unwrap_or(0),
                _ => 0,
            }
        }
        let context = Context::default();
        context
            .run_ui(
                RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::splat(32.0))),
                    ..Default::default()
                },
                |ui| sketch_draw_preview(ui, whole_screen(), marks),
            )
            .shapes
            .iter()
            .map(|clipped| strongest(&clipped.shape))
            .max()
            .unwrap_or(0)
    }

    /// A guide that is only PARTLY in force is drawn only partly.
    ///
    /// The snap's circle is the case: its hold fades continuously to nothing at the rim of the
    /// cone, and before this the ring appeared at full ink the moment the cone was entered — at its
    /// loudest exactly where the quantity had stopped being kept. Half the hold is now visibly less
    /// than the whole of it, and a hold of nothing paints nothing.
    #[test]
    fn a_guide_is_drawn_at_the_strength_it_is_held_at() {
        let guide = |strength| {
            ink_alpha(&[SketchPreviewMark::Polyline {
                chords: ring(),
                line: SketchPreviewLine::Guide,
                strength,
            }])
        };
        let (whole, half, none) = (guide(1.0), guide(0.5), guide(0.0));
        assert!(whole > 0 && half > 0, "a held guide has to be visible");
        assert!(
            half < whole,
            "half a hold painted {half} against {whole} for the whole of it"
        );
        assert_eq!(none, 0, "a hold of nothing painted {none}");
    }

    fn ring() -> Vec<Pos2> {
        vec![
            pos2(10.0, 5.0),
            pos2(5.0, 10.0),
            pos2(0.0, 5.0),
            pos2(5.0, 0.0),
            pos2(10.0, 5.0),
        ]
    }

    #[test]
    fn a_closed_preview_ring_paints_through_the_dashed_preview_layer() {
        assert!(
            painted(&[SketchPreviewMark::Polyline {
                chords: ring(),
                line: SketchPreviewLine::Outline,
                strength: 1.0,
            }]) > 0,
            "the closed preview produced foreground ink"
        );
    }

    /// Each kind of mark draws on its own — a tool mid-gesture with only points taken and no shape
    /// yet still shows something, which is the whole reason the channel is a list of marks.
    #[test]
    fn every_mark_kind_paints_on_its_own() {
        for mark in [
            SketchPreviewMark::Point { at: pos2(8.0, 8.0) },
            SketchPreviewMark::Refused { at: pos2(8.0, 8.0) },
            SketchPreviewMark::Polyline {
                chords: ring(),
                line: SketchPreviewLine::Guide,
                strength: 1.0,
            },
        ] {
            assert!(
                painted(std::slice::from_ref(&mark)) > 0,
                "{mark:?} produced no ink"
            );
        }
    }

    /// A guide and the outline it explains are DIFFERENT ink, so one cannot be read as the other.
    #[test]
    fn a_guide_and_an_outline_are_not_the_same_mark() {
        let as_outline = painted(&[SketchPreviewMark::Polyline {
            chords: ring(),
            line: SketchPreviewLine::Outline,
            strength: 1.0,
        }]);
        let as_guide = painted(&[SketchPreviewMark::Polyline {
            chords: ring(),
            line: SketchPreviewLine::Guide,
            strength: 1.0,
        }]);
        assert!(as_outline > 0 && as_guide > 0);
    }

    /// **The snapping mark is the handle's tick-cross, drawn to be seen on its own.**
    ///
    /// Same four arms on the same four bearings — the vocabulary an author has already learned
    /// from a dragged handle — but reaching further and backed, because this one appears under the
    /// cursor over a curve that is lit at the very color it is drawn in. Every arm is painted
    /// twice, dark then accent, and the halo is the wider of the two; without that the mark and
    /// the curve it stands on are the same stroke.
    #[test]
    fn the_snapping_mark_is_the_handle_ring_drawn_to_be_seen() {
        let strokes = |draw: &dyn Fn(&egui::Ui)| -> Vec<([Pos2; 2], f32)> {
            Context::default()
                .run_ui(
                    RawInput {
                        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::splat(64.0))),
                        ..Default::default()
                    },
                    |ui| draw(ui),
                )
                .shapes
                .into_iter()
                .filter_map(|clipped| match clipped.shape {
                    egui::Shape::LineSegment { points, stroke } => Some((points, stroke.width)),
                    _ => None,
                })
                .collect()
        };
        let center = pos2(32.0, 32.0);
        let mark = strokes(&|ui| super::sketch_snap_marker(ui, whole_screen(), center));
        let on_a_handle = strokes(&|ui| {
            crate::gizmos::vertex_handle(
                ui.painter(),
                center,
                super::SKETCH_HANDLE_HALF,
                crate::gizmos::HandleState::Snapped,
                true,
            );
        });
        assert_eq!(mark.len(), 8, "four arms, each haloed: {mark:?}");
        assert_eq!(on_a_handle.len(), 4);

        let bearing = |([from, to], _): &([Pos2; 2], f32)| {
            let span = *to - *from;
            (span.x.signum() as i32, span.y.signum() as i32)
        };
        let bearings = |set: &[([Pos2; 2], f32)]| {
            set.iter()
                .map(bearing)
                .collect::<std::collections::BTreeSet<_>>()
        };
        assert_eq!(
            bearings(&mark),
            bearings(&on_a_handle),
            "the same four arms, whatever they are drawn at"
        );

        let reach = |([from, to], _): &([Pos2; 2], f32)| {
            (*to - center).length().max((*from - center).length())
        };
        let furthest = |set: &[([Pos2; 2], f32)]| set.iter().map(reach).fold(0.0_f32, f32::max);
        assert!(
            furthest(&mark) > furthest(&on_a_handle),
            "the standalone mark reaches further: {} vs {}",
            furthest(&mark),
            furthest(&on_a_handle)
        );

        let widths: Vec<f32> = mark.iter().map(|(_, width)| *width).collect();
        let thinnest = widths.iter().copied().fold(f32::MAX, f32::min);
        let widest = widths.iter().copied().fold(0.0_f32, f32::max);
        assert!(widest > thinnest, "the halo is wider than the ink it backs");
        assert!(
            thinnest > on_a_handle[0].1,
            "and even the ink is heavier than the handle's decoration"
        );
    }

    /// A badge posed as the shell poses it, reading the plane at its own center.
    fn a_badge_on(frame: crate::gizmos::dimension::PlaneFrame, center: Pos2) -> ConstraintBadge {
        a_badge_of(crate::icons::Icon::ConstraintParallel, frame, center)
    }

    /// The same, for a test that cares WHICH glyph — the shelf's marks do not all reach as far.
    ///
    /// Posed by naming the SCREEN point the mark should sit on and asking the frame which plane
    /// coordinate is under it, because a specimen wants its panel laid out on a grid while a badge
    /// is made of plane coordinates.
    fn a_badge_of(
        icon: crate::icons::Icon,
        frame: crate::gizmos::dimension::PlaneFrame,
        center: Pos2,
    ) -> ConstraintBadge {
        let seat = frame
            .plane_of(center)
            .expect("a posed frame answers under its own center");
        ConstraintBadge::seated(frame.forward(), seat, icon, 1, false)
            .expect("a posed frame seats a badge")
    }

    /// The glyph's two sides AS DRAWN: the screen vectors between the corners the badge's own
    /// projection puts them at.
    ///
    /// Measured through [`ConstraintBadge::grid_to_screen`] and not off any pair of vectors the
    /// badge carries, because it carries none — the mark is a set of projected plane points and
    /// its shape on screen is an OUTCOME rather than a field. A test that read a stored frame
    /// would be asserting about a claim; this asserts about the drawing.
    fn sides_of(badge: ConstraintBadge) -> (Vec2, Vec2) {
        let grid = crate::icons::GRID;
        let at = badge.grid_to_screen(1.0);
        let origin = at(0.0, 0.0);
        (at(grid, 0.0) - origin, at(0.0, grid) - origin)
    }

    /// Raked hard enough that the shelf's widest glyph genuinely spills: without the fit its ink
    /// stands 2.7 points outside the square. The three-quarter plane does NOT — nothing escapes
    /// there — so a containment test posed on that one cannot fail and proves nothing.
    fn a_hard_raked_plane() -> crate::gizmos::dimension::PlaneFrame {
        crate::gizmos::dimension::PlaneFrame::from_plane_to_screen([
            [3.0, 2.4, 300.0],
            [0.3, 1.6, 200.0],
            [0.0006, 0.0002, 1.0],
        ])
        .expect("a frame that inverts")
    }

    /// A camera looking straight down at a plane with no compound turn: no shear at all, and the
    /// whole tilt is the axis RATIO. The view an author takes to look at a sketch from an angle,
    /// and the one a frame built from directions alone draws identically to a flat page.
    fn a_pitched_plane() -> crate::gizmos::dimension::PlaneFrame {
        crate::gizmos::dimension::PlaneFrame::from_plane_to_screen([
            [3.0, 0.0, 300.0],
            [0.0, 1.5, 200.0],
            [0.0, 0.0, 1.0],
        ])
        .expect("a frame that inverts")
    }

    /// Turned as well as pitched, and with a divide in it, so the frame answers a different
    /// direction at every point.
    fn a_three_quarter_plane() -> crate::gizmos::dimension::PlaneFrame {
        crate::gizmos::dimension::PlaneFrame::from_plane_to_screen([
            [3.0, 1.2, 300.0],
            [0.4, 2.4, 200.0],
            [0.0006, 0.0002, 1.0],
        ])
        .expect("a frame that inverts")
    }

    /// **Square on, a badge draws the box it has always drawn.** The whole set is authored against
    /// an upright box, so the frame that replaces it has to reduce to that one exactly — otherwise
    /// every glyph in the app moves for a change that was only ever about tilted planes.
    #[test]
    fn a_badge_square_on_is_the_box_it_replaced() {
        let center = pos2(120.0, 80.0);
        let badge = a_badge_on(crate::gizmos::dimension::PlaneFrame::facing(), center);
        let at = badge.grid_to_screen(1.0);
        let was = Rect::from_center_size(center, Vec2::splat(SKETCH_CONSTRAINT_BADGE));
        let (across, down) = sides_of(badge);
        assert_eq!(at(0.0, 0.0), was.left_top());
        assert_eq!(across, Vec2::new(was.width(), 0.0));
        assert_eq!(down, Vec2::new(0.0, was.height()));
    }

    /// **A badge lies in the plane: it squashes where the plane recedes and shears where it turns,
    /// and neither is something a rotation can do.**
    ///
    /// The pitched case is the one that decides it. There the plane's two axes still image at a
    /// right angle and only their LENGTHS differ, so a badge that merely turned to face along the
    /// plane would be drawn exactly as it is on a flat page — the commonest off-square view, and
    /// the one the report was about.
    #[test]
    fn a_badge_lies_in_the_plane_rather_than_turning_to_face_it() {
        let center = pos2(320.0, 240.0);

        let (across, down) = sides_of(a_badge_on(a_pitched_plane(), center));
        assert!(
            across.normalized().dot(down.normalized()).abs() < 1e-3,
            "a pure pitch has no shear in it, and this one grew {across:?} against {down:?}"
        );
        assert!(
            down.length() < across.length() * 0.75,
            "the plane recedes 2:1 and the glyph kept its height: {} against {}",
            down.length(),
            across.length()
        );

        let (across, down) = sides_of(a_badge_on(a_three_quarter_plane(), center));
        let leaning = across
            .normalized()
            .dot(down.normalized())
            .abs()
            .acos()
            .to_degrees();
        assert!(
            (leaning - 90.0).abs() > 10.0,
            "a turned plane shears its ink, and this glyph is still a rectangle at {leaning} deg"
        );
    }

    /// **A sheared glyph stays inside the square that makes it clickable.**
    ///
    /// A parallelogram reaches further from its center than the upright box it came from — about
    /// 125% of the half-width at a 31 degree shear — so without the inscribe fit the ink spills
    /// past the plate behind it and past the hit rect, and a badge stops being picked where it is
    /// drawn. Asserted on the PAINT rather than on the frame, because the frame is only a claim
    /// about where the marks will go.
    ///
    /// Posed on the hardest reachable rake with the shelf's WIDEST glyph, because neither alone
    /// is enough: at the three-quarter view nothing escapes, and the narrow glyphs never escape at
    /// any rake. A containment test that cannot fail is the failure.
    ///
    /// **Seen red**: with the fit forced to 1.0, this put ink 2.7 points outside the box.
    #[test]
    fn a_sheared_badge_stays_inside_the_square_that_claims_it() {
        let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(640.0, 480.0));
        let center = pos2(320.0, 240.0);
        let badge = a_badge_of(
            crate::icons::Icon::ConstraintQuantize,
            a_hard_raked_plane(),
            center,
        );

        // The plane has to be genuinely off-square, or an upright glyph passes for free.
        let (across, down) = sides_of(badge);
        assert!(
            across.normalized().dot(down.normalized()).abs() > 0.15,
            "the plane is not raked enough for this test to be able to fail"
        );

        let context = Context::default();
        let output = context.run_ui(
            RawInput {
                screen_rect: Some(screen),
                ..Default::default()
            },
            |ui| super::sketch_constraint_badges(ui, whole_screen(), &[badge]),
        );

        // The hit rect exactly, with a half point for the stroke's own width either side.
        let claimed = Rect::from_center_size(center, Vec2::splat(SKETCH_CONSTRAINT_BADGE))
            .expand(crate::icons::STROKE_WIDTH);
        let mut marks = 0_usize;
        for primitive in context.tessellate(output.shapes, 1.0) {
            let egui::epaint::Primitive::Mesh(mesh) = primitive.primitive else {
                continue;
            };
            for vertex in &mesh.vertices {
                assert!(
                    claimed.contains(vertex.pos),
                    "a badge mark stands at {:?}, outside the {claimed:?} that claims the click",
                    vertex.pos
                );
                marks += 1;
            }
        }
        assert!(
            marks >= 8,
            "the badge drew almost nothing: {marks} vertices"
        );
    }

    /// The strike the owner's own F9 dump produced: a `PlaneAxis::Z` sketch with the eye exactly in
    /// the plane's plane, where one axis reaches 0.611921 points per plane unit and the other
    /// reaches 1.97e-7 — a collapse of three million to one.
    ///
    /// Copied from `tests/sketch_plane_frame.rs`, which measures it off the real camera. Posed here
    /// as a matrix because this file is about what gets PAINTED and has no camera in it.
    fn the_plane_the_report_was_filed_on() -> crate::gizmos::dimension::PlaneFrame {
        crate::gizmos::dimension::PlaneFrame::from_plane_to_screen([
            [0.611_921, 0.0, 320.0],
            [0.0, 1.97e-7, 240.0],
            [0.0, 0.0, 1.0],
        ])
        .expect("a frame that inverts")
    }

    /// **Every mark a badge PAINTS on an edge-on plane lands on the line that plane draws.**
    ///
    /// The ninth report at the level of the ink, where the integration probe checks it at the level
    /// of the glyph's corners. The paint runs through the grid seam and the inscribe fit, and
    /// neither of those is the geometry the probe reads — a badge whose corners lie in the plane
    /// could still stroke its curves somewhere else, since ellipses and cubics are sampled into
    /// grid points and only then mapped.
    ///
    /// **The tolerance is the PEN, and it is derived rather than tuned.** A tessellated stroke is
    /// the OUTLINE of a thick path, so its vertices stand half a stroke width off the centerline,
    /// and egui feathers each edge by a further pixel. Both are screen quantities by the standing
    /// split — the path is the plane's, the pen is the screen's — so ink standing off the line by
    /// the pen's own reach is the drawing working, not failing.
    ///
    /// Posed on a STROKED glyph for that reason. `ConstraintQuantize` is a lattice of DISCS, and a
    /// disc's radius is pen too: its center lands on the line and its ink reaches 2.33 points off,
    /// measured, which is a bound about that glyph rather than about coplanarity. A test whose
    /// tolerance has to be widened per glyph is measuring the shelf, not the mark.
    #[test]
    fn an_edge_on_badge_paints_every_mark_onto_the_line_its_plane_draws() {
        let plane = the_plane_the_report_was_filed_on();
        let badge = a_badge_of(
            crate::icons::Icon::ConstraintParallel,
            plane,
            pos2(320.0, 240.0),
        );
        // Half the stroke the kit scales to this badge's size, plus egui's one-pixel feather, plus
        // a thousandth for the float. **The ink lands on that bound exactly** — measured 1.6666718
        // against a derived 1.6666667 — so every mark this badge paints on a collapsed plane is a
        // stroked line ON the line and there is no third thing in the drawing.
        let pen = crate::icons::STROKE_WIDTH * (SKETCH_CONSTRAINT_BADGE / crate::icons::GRID) / 2.0
            + 1.001;

        // Discriminability: this plane really has collapsed, so a mark that left it has somewhere
        // to go, and the badge really does still reach across the screen so there is ink to place.
        assert!(
            plane.opening_at(badge.center) < 1e-5,
            "the plane is not edge-on, so nothing here can fail"
        );
        let (across, _) = sides_of(badge);
        assert!(
            across.length() > 10.0,
            "the badge collapsed in BOTH directions and draws nothing: {across:?}"
        );

        // The line the plane draws, taken from the plane and not from the badge.
        let (near, far) = (
            plane.at([-500.0, 0.0]).expect("in front of the camera"),
            plane.at([500.0, 0.0]).expect("in front of the camera"),
        );
        let along = (far - near).normalized();

        let context = Context::default();
        let output = context.run_ui(
            RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(640.0, 480.0))),
                ..Default::default()
            },
            |ui| super::sketch_constraint_badges(ui, whole_screen(), &[badge]),
        );
        let mut marks = 0_usize;
        for primitive in context.tessellate(output.shapes, 1.0) {
            let egui::epaint::Primitive::Mesh(mesh) = primitive.primitive else {
                continue;
            };
            for vertex in &mesh.vertices {
                let away = vertex.pos - near;
                let off = (away - along * away.dot(along)).length();
                assert!(
                    off < pen,
                    "a painted mark stands {off} points off the line its own plane draws, past \
                     the {pen} its own pen can reach"
                );
                marks += 1;
            }
        }
        assert!(
            marks >= 8,
            "the badge drew almost nothing: {marks} vertices"
        );
    }
}
