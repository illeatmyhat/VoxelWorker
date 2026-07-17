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

use crate::ViewMode;

// --- Signal tokens (docs/design/viewport-chrome-signal.md §Tokens) ---
/// Panel background `#0b0d0f` at ~85 % over the viewport.
const RAIL_BG: Color32 = Color32::from_rgba_unmultiplied_const(0x0b, 0x0d, 0x0f, 217);
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

/// Rail column width (design points; §Chrome layout: 34 px).
const RAIL_WIDTH: f32 = 34.0;
/// Height of each icon-only rail button (design points; three 32 px cells).
const BUTTON_HEIGHT: f32 = 32.0;
/// Gap (points) between the cube's bottom edge and the rail's top.
const RAIL_GAP: f32 = 6.0;
/// Icon inset within a button so the glyph sits on an 18-unit grid inside 32 px.
const ICON_INSET: f32 = 7.0;
/// Signal glyph stroke width (design points; §Icon set: 1.25 px).
const STROKE_WIDTH: f32 = 1.25;

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
    let rail_left = cube_left + (cube_size - RAIL_WIDTH) * 0.5;
    let top = rail_top(cube_bottom);
    let rail_rect = Rect::from_min_size(Pos2::new(rail_left, top), Vec2::new(RAIL_WIDTH, rail_height()));
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
                Stroke::new(1.0, SEPARATOR),
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
    painter.rect_stroke(rail_rect, 0.0, Stroke::new(1.0, BORDER), StrokeKind::Inside);

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

/// Map an `(u, v)` in `[0, 1]²` onto the icon's inset box within `button_rect`.
fn icon_point(button_rect: Rect, u: f32, v: f32) -> Pos2 {
    let icon = button_rect.shrink(ICON_INSET);
    Pos2::new(icon.left() + u * icon.width(), icon.top() + v * icon.height())
}

/// Draw the glyph for rail button `index` (0 Home, 1 Fit, 2 viewport-mode) in `color`.
/// The mode glyph depends on `view_mode`: a solid cube (Normal), lifted layer slices
/// (Onion fog), or a solid-∩-dashed square pair (Show booleans).
fn draw_glyph(painter: &egui::Painter, button_rect: Rect, index: usize, view_mode: ViewMode, color: Color32) {
    let stroke = Stroke::new(STROKE_WIDTH, color);
    match index {
        0 => draw_home(painter, button_rect, stroke),
        1 => draw_fit(painter, button_rect, stroke),
        _ => match view_mode {
            ViewMode::Normal => draw_cube(painter, button_rect, stroke),
            ViewMode::OnionFog => draw_layers(painter, button_rect, stroke),
            ViewMode::ShowBooleans => draw_booleans(painter, button_rect, stroke),
        },
    }
}

/// A house silhouette: a roof triangle over a rectangular body.
fn draw_home(painter: &egui::Painter, rect: Rect, stroke: Stroke) {
    let p = |u: f32, v: f32| icon_point(rect, u, v);
    // Roof (left eave -> apex -> right eave).
    polyline(painter, &[p(0.10, 0.46), p(0.50, 0.06), p(0.90, 0.46)], stroke);
    // Body (walls + floor), tucked under the roofline.
    stroke_rect(painter, p(0.22, 0.46), p(0.78, 0.92), stroke);
}

/// A "fit to view" mark: four corner brackets framing a small centre square.
fn draw_fit(painter: &egui::Painter, rect: Rect, stroke: Stroke) {
    let p = |u: f32, v: f32| icon_point(rect, u, v);
    // Corner brackets (each an L: two segments meeting at the corner).
    polyline(painter, &[p(0.08, 0.32), p(0.08, 0.08), p(0.32, 0.08)], stroke); // TL
    polyline(painter, &[p(0.68, 0.08), p(0.92, 0.08), p(0.92, 0.32)], stroke); // TR
    polyline(painter, &[p(0.08, 0.68), p(0.08, 0.92), p(0.32, 0.92)], stroke); // BL
    polyline(painter, &[p(0.68, 0.92), p(0.92, 0.92), p(0.92, 0.68)], stroke); // BR
    // Centre square.
    stroke_rect(painter, p(0.38, 0.38), p(0.62, 0.62), stroke);
}

/// A solid cube (front square + a back square offset up-right + the four join edges).
fn draw_cube(painter: &egui::Painter, rect: Rect, stroke: Stroke) {
    let p = |u: f32, v: f32| icon_point(rect, u, v);
    // Front face.
    stroke_rect(painter, p(0.16, 0.34), p(0.64, 0.82), stroke);
    // Back face (offset up and to the right).
    stroke_rect(painter, p(0.36, 0.14), p(0.84, 0.62), stroke);
    // Connecting edges (front corners -> back corners).
    painter.line_segment([p(0.16, 0.34), p(0.36, 0.14)], stroke);
    painter.line_segment([p(0.64, 0.34), p(0.84, 0.14)], stroke);
    painter.line_segment([p(0.16, 0.82), p(0.36, 0.62)], stroke);
    painter.line_segment([p(0.64, 0.82), p(0.84, 0.62)], stroke);
}

/// Lifted layer slices: three stacked isometric rhombi (the onion-fog mode glyph).
fn draw_layers(painter: &egui::Painter, rect: Rect, stroke: Stroke) {
    let p = |u: f32, v: f32| icon_point(rect, u, v);
    for &cy in &[0.26f32, 0.50, 0.74] {
        // A flat diamond centred at (0.5, cy): left, top, right, bottom, back to left.
        polyline(
            painter,
            &[
                p(0.16, cy),
                p(0.50, cy - 0.14),
                p(0.84, cy),
                p(0.50, cy + 0.14),
                p(0.16, cy),
            ],
            stroke,
        );
    }
}

/// A solid square intersecting a dashed square (the show-booleans mode glyph).
fn draw_booleans(painter: &egui::Painter, rect: Rect, stroke: Stroke) {
    let p = |u: f32, v: f32| icon_point(rect, u, v);
    // Solid square (the kept body).
    stroke_rect(painter, p(0.12, 0.20), p(0.60, 0.68), stroke);
    // Dashed square (the boolean operand), overlapping the first.
    dashed_rect(painter, p(0.40, 0.34), p(0.88, 0.82), stroke);
}

/// Stroke the axis-aligned rectangle spanned by two opposite corners.
fn stroke_rect(painter: &egui::Painter, a: Pos2, b: Pos2, stroke: Stroke) {
    let rect = Rect::from_two_pos(a, b);
    painter.rect_stroke(rect, 0.0, stroke, StrokeKind::Middle);
}

/// Draw a connected polyline through `points`.
fn polyline(painter: &egui::Painter, points: &[Pos2], stroke: Stroke) {
    for pair in points.windows(2) {
        painter.line_segment([pair[0], pair[1]], stroke);
    }
}

/// Stroke a rectangle's outline as dashes (short segments with gaps) along each edge.
fn dashed_rect(painter: &egui::Painter, a: Pos2, b: Pos2, stroke: Stroke) {
    let rect = Rect::from_two_pos(a, b);
    let corners = [
        rect.left_top(),
        rect.right_top(),
        rect.right_bottom(),
        rect.left_bottom(),
    ];
    for i in 0..4 {
        dashed_line(painter, corners[i], corners[(i + 1) % 4], stroke);
    }
}

/// Draw a dashed segment from `a` to `b` (dash 2.2 pt, gap 1.8 pt).
fn dashed_line(painter: &egui::Painter, a: Pos2, b: Pos2, stroke: Stroke) {
    const DASH: f32 = 2.2;
    const GAP: f32 = 1.8;
    let delta = b - a;
    let length = delta.length();
    if length <= f32::EPSILON {
        return;
    }
    let direction = delta / length;
    let mut travelled = 0.0;
    while travelled < length {
        let start = a + direction * travelled;
        let end = a + direction * (travelled + DASH).min(length);
        painter.line_segment([start, end], stroke);
        travelled += DASH + GAP;
    }
}
