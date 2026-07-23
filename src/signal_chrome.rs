//! The Signal viewport chrome that lives in egui: the **icon rail** under the view
//! cube and the **status line** bottom-left (ADR 0018 Decision 8;
//! `docs/design/viewport-chrome-signal.md`).
//!
//! Both are drawn inside [`run_egui_frame`](crate::run_egui_frame) via a foreground
//! [`layer_painter`](egui::Context::layer_painter) at absolute coordinates, so they
//! render IDENTICALLY on the windowed surface and in the headless `shot` capture (the
//! one-panel-for-both rule). Immediate-mode painting (not an [`egui::Area`]) is what
//! makes them appear on `shot`'s SINGLE frame — a floating Area needs a prior frame to
//! settle its size. Unlike the S6 hovered-zone readout (a windowed-only overlay fed
//! `None` on the `shot` path), the rail and status line are PERSISTENT chrome and draw
//! on both paths.
//!
//! The icons are drawn as [`egui::Painter`] vector strokes on an 18-unit grid with a
//! 1.25 px stroke (the design doc's icon set), not as textures — the cube's own chrome
//! (rotate/roll arrows) stays a GPU glyph overlay, but the rail is pure egui so it
//! composes with the panel layout and tooltips.

use egui::{Color32, FontId, Id, LayerId, Order, Pos2, Rect, Sense, Stroke, StrokeKind, TextFormat, Vec2};
use ui::icons::Icon;
use ui::panel::SketchExit;

use crate::ViewMode;

// --- Signal tokens (docs/design/viewport-chrome-signal.md §Tokens) ---
/// Panel background `#0b0d0f`, OPAQUE (issue #91 item 6): the rail must read solid over a
/// textured voxel scene (matching the approved screenshots), so no scene bleeds through.
const RAIL_BG: Color32 = Color32::from_rgb(0x0b, 0x0d, 0x0f);
/// Hairline outer border `#2b3238`.
const BORDER: Color32 = Color32::from_rgb(0x2b, 0x32, 0x38);
/// Hairline inner rule / separator `#1c2126`.
const SEPARATOR: Color32 = Color32::from_rgb(0x1c, 0x21, 0x26);
/// Idle rail glyph `#78828c`.
const GLYPH_IDLE: Color32 = Color32::from_rgb(0x78, 0x82, 0x8c);
/// Hover rail glyph `#c7d3e0`.
const GLYPH_HOVER: Color32 = Color32::from_rgb(0xc7, 0xd3, 0xe0);
/// Hover rail-button fill `#161a1e`.
const HOVER_BG: Color32 = Color32::from_rgb(0x16, 0x1a, 0x1e);
/// The single accent — the ADR 0012 onion-haze hue `#9cb4d8` (lit mode glyph + bar).
const ACCENT: Color32 = Color32::from_rgb(0x9c, 0xb4, 0xd8);
/// Lit mode-button fill `#12161b`.
const LIT_BG: Color32 = Color32::from_rgb(0x12, 0x16, 0x1b);
/// Status-line faint text `#4d565f`.
const STATUS_FAINT: Color32 = Color32::from_rgb(0x4d, 0x56, 0x5f);
/// The immersive sketch-mode border tint — the accent at ~33% (C2 mock `#9cb4d84d`).
const ACCENT_FAINT: Color32 = Color32::from_rgba_premultiplied(0x2f, 0x37, 0x43, 0x4d);
/// Near-opaque panel fill for the floating exit buttons (mock `#0b0d0feb`).
const FLOAT_BG: Color32 = Color32::from_rgb(0x0b, 0x0d, 0x0f);

/// Rail column width (design points; §Chrome layout: 34 px).
const RAIL_WIDTH: f32 = 34.0;
/// Height of each icon-only rail button (design points; three 32 px cells).
const BUTTON_HEIGHT: f32 = 32.0;
/// Gap (points) between the cube's bottom edge and the rail's top.
const RAIL_GAP: f32 = 6.0;
/// The centred square a rail glyph is drawn into (design points). 18 pt = one unit per
/// grid unit of the rail set's 18-unit authoring grid, which makes `IconPainter`'s scale
/// exactly 1 and so lands the stroke on the design's 1.25 pt without restating it here.
const GLYPH_BOX: f32 = 18.0;

/// A rail button the user clicked this frame. The caller maps [`Home`](Self::Home) /
/// [`Fit`](Self::Fit) onto the SAME [`ChromeClickAction`](camera::ChromeClickAction)s
/// the retired cube badges dispatched (reusing the shell's `run_chrome_action`), and
/// [`CycleMode`](Self::CycleMode) onto [`ViewMode::next`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RailClick {
    /// Home — snap to the saved home view (the old Home badge's action).
    Home,
    /// Fit — frame the scene (the old Fit badge's action).
    Fit,
    /// Cycle the viewport mode Normal -> Onion fog -> Show booleans -> Normal.
    CycleMode,
}

/// The full rail height (three buttons), used to place the readout below the rail.
pub fn rail_height() -> f32 {
    3.0 * BUTTON_HEIGHT
}

/// The rail's top Y (points) given the cube's bottom edge — the readout stacks below.
pub fn rail_top(cube_bottom: f32) -> f32 {
    cube_bottom + RAIL_GAP
}

/// The rail's full rect (egui points) from the cube anchors — the shell's chrome
/// hit-rect for the camera gate (the same geometry [`icon_rail`] draws at).
pub fn rail_rect(cube_left: f32, cube_bottom: f32, cube_size: f32) -> Rect {
    let rail_left = cube_left + (cube_size - RAIL_WIDTH) * 0.5;
    Rect::from_min_size(
        Pos2::new(rail_left, rail_top(cube_bottom)),
        Vec2::new(RAIL_WIDTH, rail_height()),
    )
}

/// Draw the Signal **icon rail** directly under the view cube and return a click, if
/// any. `cube_left` / `cube_bottom` are the cube's top-left-derived screen anchors (in
/// egui points, the same the S6 readout uses); `cube_size` is the cube's on-screen edge
/// in points. The rail is centred under the cube. Home / Fit / viewport-mode-cycle,
/// icon-only with native tooltips; the mode button is "lit" (accent glyph + a 2 px
/// accent inset bar + a dark fill) whenever `view_mode` is non-Normal.
///
/// Painted through a foreground [`layer_painter`](egui::Context::layer_painter) at
/// absolute coordinates (NOT an `egui::Area`) so it draws correctly on the headless
/// `shot`'s single frame — a floating Area needs a prior frame to settle its size, but
/// this chrome must render identically in the very first captured frame. Interaction
/// (hover highlight + click) is via [`Ui::interact`](egui::Ui::interact) on the same
/// rects, which needs no settling.
pub fn icon_rail(
    ui: &egui::Ui,
    cube_left: f32,
    cube_bottom: f32,
    cube_size: f32,
    view_mode: ViewMode,
) -> Option<RailClick> {
    let rail_rect = rail_rect(cube_left, cube_bottom, cube_size);
    let painter = ui
        .ctx()
        .layer_painter(LayerId::new(Order::Foreground, Id::new("signal_icon_rail")));
    // The near-black rail body (drawn first; the border is stroked last, on top).
    painter.rect_filled(rail_rect, 0.0, RAIL_BG);

    let mut click = None;
    for index in 0..3usize {
        let button_rect = Rect::from_min_size(
            Pos2::new(rail_rect.left(), rail_rect.top() + index as f32 * BUTTON_HEIGHT),
            Vec2::new(RAIL_WIDTH, BUTTON_HEIGHT),
        );
        let response = ui.interact(button_rect, Id::new(("signal_rail_button", index)), Sense::click());
        let hovered = response.hovered();
        // Only the viewport-mode button (index 2) lights, and only off-Normal.
        let lit = index == 2 && view_mode != ViewMode::Normal;

        // Button fill: hover wins, else the lit dark cell, else the rail body.
        if hovered {
            painter.rect_filled(button_rect, 0.0, HOVER_BG);
        } else if lit {
            painter.rect_filled(button_rect, 0.0, LIT_BG);
        }
        // Hairline separator above every button but the first.
        if index > 0 {
            painter.line_segment(
                [
                    Pos2::new(rail_rect.left(), button_rect.top()),
                    Pos2::new(rail_rect.right(), button_rect.top()),
                ],
                Stroke::new(1.0_f32, SEPARATOR),
            );
        }
        // Lit mode: a 2 px accent inset bar on the leading (left) edge.
        if lit {
            let bar = Rect::from_min_size(button_rect.left_top(), Vec2::new(2.0, BUTTON_HEIGHT));
            painter.rect_filled(bar, 0.0, ACCENT);
        }

        let glyph_color = if lit {
            ACCENT
        } else if hovered {
            GLYPH_HOVER
        } else {
            GLYPH_IDLE
        };
        draw_glyph(&painter, button_rect, index, view_mode, glyph_color);

        let response = response.on_hover_text(match index {
            0 => "Home view",
            1 => "Fit scene",
            _ => "Viewport mode",
        });
        if response.clicked() {
            click = Some(match index {
                0 => RailClick::Home,
                1 => RailClick::Fit,
                _ => RailClick::CycleMode,
            });
        }
    }

    // Outer hairline border on top of the button fills.
    painter.rect_stroke(rail_rect, 0.0, Stroke::new(1.0_f32, BORDER), StrokeKind::Inside);

    click
}

/// Draw the Signal **status line** pinned bottom-left of the viewport:
/// `VIEWPORT <MODE> · SEL <node> · <dims> · <density> vx/blk` in faint mono, with the
/// mode name and selection in the accent and `·` separators in the border grey.
/// `viewport_rect` is the central 3D rect (egui points); `selection` is the active
/// node's name or `None` (-> `—`); `dims` the resolved grid extent (voxels); `density`
/// voxels-per-block.
///
/// Painted through a foreground [`layer_painter`](egui::Context::layer_painter) at an
/// absolute position (NOT an `egui::Area`) so it renders on the headless `shot`'s single
/// captured frame (see [`icon_rail`]).
pub fn status_line(
    ui: &egui::Ui,
    viewport_rect: Rect,
    view_mode: ViewMode,
    selection: Option<&str>,
    dims: [u32; 3],
    density: u32,
) {
    let mono = FontId::monospace(10.0);
    let format_with = |color: Color32| TextFormat {
        font_id: mono.clone(),
        color,
        ..Default::default()
    };
    let faint = format_with(STATUS_FAINT);
    let accent = format_with(ACCENT);
    let dot = format_with(BORDER);

    let mut job = egui::text::LayoutJob::default();
    job.append("VIEWPORT ", 0.0, faint.clone());
    job.append(view_mode.status_label(), 0.0, accent.clone());
    job.append("  ·  ", 0.0, dot.clone());
    job.append("SEL ", 0.0, faint.clone());
    job.append(selection.unwrap_or("—"), 0.0, accent);
    job.append("  ·  ", 0.0, dot.clone());
    job.append(&format!("{}×{}×{}", dims[0], dims[1], dims[2]), 0.0, faint.clone());
    job.append("  ·  ", 0.0, dot);
    job.append(&format!("{density} vx/blk"), 0.0, faint);

    let painter = ui
        .ctx()
        .layer_painter(LayerId::new(Order::Foreground, Id::new("signal_status_line")));
    let galley = painter.layout_job(job);
    // Bottom-left, a touch in from the viewport edges; up by the line height + a small
    // margin so the baseline sits clear of the viewport's bottom edge.
    let pos = Pos2::new(
        viewport_rect.left() + 10.0,
        viewport_rect.bottom() - galley.size().y - 6.0,
    );
    painter.galley(pos, galley, STATUS_FAINT);
}

/// The sketch-mode exit control + immersive border (ADR 0028; C2 mock). Draws a faint accent
/// inset border framing the viewport — the "you are editing a sketch" tint — plus the floating
/// `CANCEL` / `FINISH SKETCH` pair bottom-right, and returns the arm the user clicked. This and
/// the rail swap are the two mode signals the owner review kept (no banner). `Finish` commits
/// (from #94, closes the undo group as one main-history entry); `Cancel` discards.
///
/// Immediate-mode painted through a foreground [`layer_painter`](egui::Context::layer_painter)
/// at absolute coordinates (NOT an `egui::Area`), like [`icon_rail`], so it renders on the
/// headless `shot`'s single captured frame.
pub fn sketch_exit_control(
    ui: &egui::Ui,
    viewport_rect: Rect,
    chrome_rects: &mut Vec<Rect>,
) -> Option<SketchExit> {
    let painter = ui
        .ctx()
        .layer_painter(LayerId::new(Order::Foreground, Id::new("sketch_exit_control")));
    // The immersive accent inset border — the mode tint (mock: `inset 0 0 0 2px` accent-alpha).
    painter.rect_stroke(
        viewport_rect.shrink(1.0),
        0.0,
        Stroke::new(2.0_f32, ACCENT_FAINT),
        StrokeKind::Inside,
    );

    // The two buttons, laid right-to-left from the viewport's bottom-right corner. Finish is
    // the primary (accent fill, dark ink); Cancel is a bordered near-black box.
    let mono = FontId::monospace(10.0);
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
        // `PLACEHOLDER` colour so `painter.galley` fills the real ink at paint time; the size
        // is colour-independent, so one layout serves both measuring and painting.
        let galley = painter.layout_no_wrap(label.to_string(), mono.clone(), Color32::PLACEHOLDER);
        let size = galley.size() + pad * 2.0;
        let rect = Rect::from_min_max(
            Pos2::new(right - size.x, bottom - size.y),
            Pos2::new(right, bottom),
        );
        let response = ui.interact(rect, Id::new(("sketch_exit", label)), Sense::click());
        let hovered = response.hovered();

        painter.rect_filled(rect, 0.0, if primary { ACCENT } else { FLOAT_BG });
        painter.rect_stroke(
            rect,
            0.0,
            Stroke::new(1.0_f32, if primary { ACCENT } else { BORDER }),
            StrokeKind::Inside,
        );
        let ink = if primary {
            RAIL_BG
        } else if hovered {
            GLYPH_HOVER
        } else {
            GLYPH_IDLE
        };
        painter.galley(rect.min + pad, galley, ink);

        if response.clicked() {
            clicked = Some(exit);
        }
        // Register the button as chrome so a click on it never leaks to the camera orbit.
        chrome_rects.push(rect);
        right = rect.left() - gap;
    }
    clicked
}

/// The half-extent (egui points) of a sketch profile-vertex handle's square thumb — the
/// visible size the [`ui::gizmos::vertex_handle`] painter draws.
pub const SKETCH_HANDLE_HALF: f32 = 5.0;

/// The extra pixels around a handle's thumb that still count as a grab / chrome hit — a
/// forgiving target so a vertex is easy to pick up and so a click near it never leaks to
/// the camera orbit. The shell's press hit-test uses the SAME radius (`SKETCH_HANDLE_HALF
/// + SKETCH_HANDLE_GRAB_PAD`).
pub const SKETCH_HANDLE_GRAB_PAD: f32 = 5.0;

/// How close (egui points) the cursor must come to a profile edge for the add-point tool to
/// treat it as hovering that segment (#95). Wider than a vertex grab so an edge is an easy
/// target, but the shell prefers a vertex hit first, so the two never fight over the same click.
pub const SKETCH_SEGMENT_GRAB_PAD: f32 = 7.0;

/// The half-extent (egui points) of the add-point insert-preview diamond — the hollow marker
/// on the hovered edge showing where a click drops a vertex.
pub const SKETCH_INSERT_MARKER_HALF: f32 = 4.0;

/// Draw the add-point **insert-preview** marker (ADR 0028, #95): a hollow diamond at `center`
/// (already-projected egui points) on the hovered profile edge — "a vertex lands here". Mirrors
/// [`sketch_vertex_handles`]'s foreground-`layer_painter` idiom so it paints over the scene, and
/// is deliberately NOT registered as chrome: it is a passive preview, so a click passes through
/// to the shell's stationary-release insert rather than being swallowed here.
pub fn sketch_insert_marker(ui: &egui::Ui, center: Pos2) {
    let painter = ui
        .ctx()
        .layer_painter(LayerId::new(Order::Foreground, Id::new("sketch_insert_marker")));
    ui::gizmos::diamond(&painter, center, SKETCH_INSERT_MARKER_HALF);
}

/// Draw the sketch profile's **vertex handles** (ADR 0028, #94) at their already-projected
/// screen positions, each in the given [`HandleState`], and register each handle's grab
/// rect as chrome so a press on a handle drags the vertex instead of orbiting the camera.
///
/// Mirrors [`sketch_exit_control`]'s foreground-`layer_painter` idiom so the handles render
/// on the headless `shot` single frame as well as the live window. Pure drawing — the shell
/// owns the projection (world→screen), the hit-testing and the drag; this only paints what
/// the shell computed.
pub fn sketch_vertex_handles(
    ui: &egui::Ui,
    handles: &[(Pos2, ui::gizmos::HandleState)],
    chrome_rects: &mut Vec<Rect>,
) {
    let painter = ui
        .ctx()
        .layer_painter(LayerId::new(Order::Foreground, Id::new("sketch_vertex_handles")));
    let grab = SKETCH_HANDLE_HALF + SKETCH_HANDLE_GRAB_PAD;
    for (center, state) in handles {
        ui::gizmos::vertex_handle(&painter, *center, SKETCH_HANDLE_HALF, *state);
        chrome_rects.push(Rect::from_center_size(*center, Vec2::splat(grab * 2.0)));
    }
}

/// The glyph box inside a rail button: a CENTRED SQUARE, because the rail set is authored
/// on a square 18-unit grid and a non-square box would stretch every mark horizontally.
/// Side 18 pt keeps the vertical extent the inset used to give and makes
/// `IconPainter`'s scale exactly 1, so the stroke lands on the design's 1.25 pt.
fn glyph_box(button_rect: Rect) -> Rect {
    Rect::from_center_size(button_rect.center(), Vec2::splat(GLYPH_BOX))
}

/// Draw the glyph for rail button `index` (0 Home, 1 Fit, 2 viewport-mode) in `color`.
/// The mode glyph depends on `view_mode`: a solid cube (Normal), lifted layer slices
/// (Onion fog), or a solid-∩-dashed square pair (Show booleans).
///
/// The marks come from `ui::icons`, which is the ONE authoring of the rail set — the same
/// data the `design_reference` gallery paints. This file used to trace its own copies of
/// the five, and they had already drifted: `home` was regridded onto the 2.5–15.5 house box
/// and `mode-normal` was redrawn as a shaded cube, neither of which reached the shipped
/// rail. A glyph the design sheet shows and the app does not draw is worse than no sheet,
/// so the rail reads the set rather than mirroring it.
fn draw_glyph(painter: &egui::Painter, button_rect: Rect, index: usize, view_mode: ViewMode, color: Color32) {
    let icon = match index {
        0 => Icon::Home,
        1 => Icon::Fit,
        _ => match view_mode {
            ViewMode::Normal => Icon::ModeNormal,
            ViewMode::OnionFog => Icon::ModeOnion,
            ViewMode::ShowBooleans => Icon::ModeBooleans,
        },
    };
    icon.draw(painter, glyph_box(button_rect), color);
}

// The five glyph painters that used to live here (home, fit, cube, layers, booleans) are
// gone: `ui::icons` owns those drawings now. See `draw_glyph` above for why.
