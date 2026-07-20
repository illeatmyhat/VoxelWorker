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
