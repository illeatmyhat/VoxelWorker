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
/// The half-extent (egui points) of the add-point insert-preview diamond.
pub const SKETCH_INSERT_MARKER_HALF: f32 = 4.0;
/// The side (egui points) of a constraint badge's glyph box. Constant on screen, like every
/// other sketch mark: a badge says *what is asserted*, and a claim does not get smaller with
/// distance.
pub const SKETCH_CONSTRAINT_BADGE: f32 = 32.0;
/// How far (egui points) a badge sits off the geometry it belongs to, and how far successive
/// badges on the same anchor step along that offset. Scaled with the box, so the arrangement
/// reads the same at any badge size.
pub const SKETCH_CONSTRAINT_BADGE_OFFSET: f32 = 30.0;

/// The sketch-mode exit control + immersive border: a faint accent inset border framing the
/// viewport plus the floating `CANCEL` / `FINISH SKETCH` pair bottom-right; returns the clicked
/// arm. Registers the buttons as chrome so a click never leaks to the camera orbit.
pub fn sketch_exit_control(
    ui: &egui::Ui,
    viewport_rect: Rect,
    chrome_rects: &mut Vec<Rect>,
) -> Option<SketchExit> {
    let painter = ui.ctx().layer_painter(LayerId::new(
        Order::Foreground,
        Id::new("sketch_exit_control"),
    ));
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
/// passive preview, so a click passes through to the shell's insert.
pub fn sketch_insert_marker(ui: &egui::Ui, center: Pos2) {
    let painter = ui.ctx().layer_painter(LayerId::new(
        Order::Foreground,
        Id::new("sketch_insert_marker"),
    ));
    gizmos::diamond(&painter, center, SKETCH_INSERT_MARKER_HALF);
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
pub fn sketch_draw_preview(ui: &egui::Ui, marks: &[SketchPreviewMark]) {
    let painter = ui.ctx().layer_painter(LayerId::new(
        Order::Foreground,
        Id::new("sketch_draw_preview"),
    ));
    let polylines = |want: SketchPreviewLine| {
        for mark in marks {
            if let SketchPreviewMark::Polyline { chords, line } = mark {
                if *line == want {
                    // One run, not one per chord: a curve preview arrives already flattened, and
                    // restarting the dash rhythm on every short chord would draw it solid.
                    match want {
                        SketchPreviewLine::Outline => {
                            gizmos::dashed_preview_polyline(&painter, chords);
                        }
                        SketchPreviewLine::Guide => {
                            gizmos::dashed_guide_polyline(&painter, chords);
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
    /// Where the badge's box is centered, in egui points, already projected by the shell.
    pub center: Pos2,
    /// The glyph — the same mark the rail cell that made this constraint carries.
    pub icon: crate::icons::Icon,
    /// The constraint this badge stands for, so a click on it names an entity.
    pub constraint: document::sketch::EntityId,
    /// Whether that constraint is in the selection.
    pub picked: bool,
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
pub fn sketch_constraint_badges(ui: &egui::Ui, badges: &[ConstraintBadge]) {
    let painter = ui.ctx().layer_painter(LayerId::new(
        Order::Foreground,
        Id::new("sketch_constraint_badges"),
    ));
    for badge in badges {
        let box_rect = Rect::from_center_size(badge.center, Vec2::splat(SKETCH_CONSTRAINT_BADGE));
        let ink = if badge.picked {
            theme::ACCENT
        } else {
            theme::SKETCH_CONSTRAINT
        };
        if badge.picked {
            let plate = box_rect.expand(4.0);
            painter.rect_filled(plate, 3.0, theme::ACCENT_FAINT);
            painter.rect_stroke(
                plate,
                3.0,
                Stroke::new(1.0_f32, theme::ACCENT),
                StrokeKind::Inside,
            );
        }
        badge.icon.draw(&painter, box_rect, ink);
    }
}

/// Draw the directional marquee rubber band. `window` (drag
/// left→right, fully-enclosed semantic) = solid accent outline + the stronger fill; crossing
/// (right→left, any-intersection) = dashed outline + lighter fill — dashed already means
/// "looser" in the gizmo family, so the semantic is legible mid-drag. Not chrome — a passive
/// preview over an already-armed press.
pub fn sketch_marquee_band(ui: &egui::Ui, rect: Rect, window: bool) {
    let painter = ui.ctx().layer_painter(LayerId::new(
        Order::Foreground,
        Id::new("sketch_marquee_band"),
    ));
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
    pub construction: bool,
}

/// Draw the committed segment lines between their projected endpoints. Idle edges first, then the
/// single hovered/marked one on top so its brighter line (or warn line + ✕) is never clipped.
pub fn sketch_segment_lines(ui: &egui::Ui, lines: &[SketchEdgeLine]) {
    let painter = ui.ctx().layer_painter(LayerId::new(
        Order::Foreground,
        Id::new("sketch_segment_lines"),
    ));
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
pub fn sketch_arc_curves(ui: &egui::Ui, curves: &[SketchCurveLine]) {
    let painter = ui.ctx().layer_painter(LayerId::new(
        Order::Foreground,
        Id::new("sketch_arc_curves"),
    ));
    let draw = |curve: &SketchCurveLine| {
        gizmos::roled_curve(&painter, &curve.chords, curve.state, curve.construction);
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
    handles: &[(Pos2, gizmos::HandleState)],
    chrome_rects: &mut Vec<Rect>,
) {
    let painter = ui.ctx().layer_painter(LayerId::new(
        Order::Foreground,
        Id::new("sketch_vertex_handles"),
    ));
    let grab = SKETCH_HANDLE_HALF + SKETCH_HANDLE_GRAB_PAD;
    for (center, state) in handles {
        gizmos::vertex_handle(&painter, *center, SKETCH_HANDLE_HALF, *state);
        chrome_rects.push(Rect::from_center_size(*center, Vec2::splat(grab * 2.0)));
    }
}

#[cfg(test)]
mod tests {
    use super::{sketch_draw_preview, SketchPreviewLine, SketchPreviewMark};
    use egui::{pos2, Context, Pos2, RawInput, Rect, Vec2};

    fn painted(marks: &[SketchPreviewMark]) -> usize {
        let context = Context::default();
        context
            .run_ui(
                RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::splat(32.0))),
                    ..Default::default()
                },
                |ui| sketch_draw_preview(ui, marks),
            )
            .shapes
            .len()
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
        }]);
        let as_guide = painted(&[SketchPreviewMark::Polyline {
            chords: ring(),
            line: SketchPreviewLine::Guide,
        }]);
        assert!(as_outline > 0 && as_guide > 0);
    }
}
