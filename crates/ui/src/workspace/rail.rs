//! The pinned rail: the two sets that cannot grow.
//!
//! Shapes and tools are a finite verb set, so they are permanent furniture the user builds
//! muscle memory against — never summoned, never searched. Materials and saved parts grow
//! with the project and live in the drawer instead. That single rule decides membership, so
//! nothing appears in both places.
//!
//! Shape cells take the TILE glyph at 26 px, falling back to the rail mark where a noun has
//! no tile drawing. The two families are separate drawings of the same noun rather than one
//! asset scaled, so the fallback is designed rather than a gap.
//!
//! A cell whose verb the document cannot yet express is drawn RESERVED — dimmed and inert,
//! the treatment the design mock gives `sweep`. It is deliberately not hidden: the shape of
//! the finished set is information, and a verb that silently appears later reads as a bug.

use document::intent::Intent;
use document::voxel::SdfShape;
use voxel_core::voxel::ShapeKind;

use super::{hairline, region_frame, Edge, RAIL_WIDTH};
use crate::icons::{large::LargeIcon, Icon};
use crate::panel::{PanelResponse, PanelState};
use crate::signal_theme;

/// A shape cell: full rail width less the hairline, tall enough for a 26 px tile plus air.
const CELL_HEIGHT: f32 = 37.0;
/// A tool cell — rail glyphs at 19 px, so the cell is shorter.
const TOOL_CELL_HEIGHT: f32 = 30.0;
/// The tile glyph's box inside a shape cell.
const TILE_GLYPH: f32 = 26.0;
/// The rail glyph's box inside a tool cell.
const TOOL_GLYPH: f32 = 19.0;
/// Opacity of a reserved cell — present, legible, plainly not yet clickable.
const RESERVED_DIM: f32 = 0.35;

/// The shape set, in the order the design sheet pins it: the authoring atom first, then the
/// lifts, then the primitives that are sugar over them.
///
/// `Some(kind)` is a shape the document can express today, and clicking it retargets the
/// selected Tool. `None` is a producer that has a glyph but no reachable intent from here
/// yet — sketch-family verbs are authored through the inspector, and `sweep` is reserved.
const SHAPES: &[(Icon, Option<ShapeKind>)] = &[
    (Icon::Sketch, None),
    (Icon::Extrude, None),
    (Icon::Revolve, None),
    (Icon::Sweep, None),
    (Icon::BoxSolid, Some(ShapeKind::Box)),
    (Icon::Sphere, Some(ShapeKind::Sphere)),
    (Icon::Cylinder, Some(ShapeKind::Cylinder)),
    (Icon::Tube, Some(ShapeKind::Tube)),
    (Icon::Torus, Some(ShapeKind::Torus)),
    (Icon::HalfSpace, None),
];

/// The tool set. Only selection exists as a mode today; the rest are drawn reserved so the
/// finished shape of the toolbelt is visible without pretending the verbs work.
const TOOLS: &[(Icon, bool)] = &[
    (Icon::AxesGizmo, true),
    (Icon::SculptAdd, false),
    (Icon::Carve, false),
    (Icon::Material, false),
    (Icon::Probe, false),
    (Icon::Measure, false),
];

/// Build the pinned rail column.
pub(super) fn build_rail(
    root_ui: &mut egui::Ui,
    state: &mut PanelState,
    response: &mut PanelResponse,
) {
    egui::Panel::left("workspace_rail")
        .resizable(false)
        .default_size(RAIL_WIDTH)
        .frame(region_frame())
        .show_inside(root_ui, |ui| {
            let column = ui.max_rect();
            hairline(ui.painter(), column, Edge::Right, signal_theme::BORDER);

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
                    rail_heading(ui, "Shape");
                    for &(icon, kind) in SHAPES {
                        shape_cell(ui, icon, kind, state, response);
                    }
                    rail_heading(ui, "Tool");
                    for &(icon, enabled) in TOOLS {
                        tool_cell(ui, icon, enabled);
                    }
                });
        });
}

/// A rail section heading: UPPERCASE micro-label over a hairline.
fn rail_heading(ui: &mut egui::Ui, title: &str) {
    ui.add_space(9.0);
    let galley = signal_theme::letter_spaced(ui, title, signal_theme::TEXT_HINT, 8.0, 1.2);
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(RAIL_WIDTH, galley.size().y + 5.0),
        egui::Sense::hover(),
    );
    let at = egui::pos2(rect.center().x - galley.size().x * 0.5, rect.top());
    ui.painter().galley(at, galley, signal_theme::TEXT_HINT);
    hairline(ui.painter(), rect, Edge::Bottom, signal_theme::RULE);
}

/// One shape cell. Clicking an expressible shape retargets the SELECTED Tool node.
///
/// The target is read once, up front, and carried into the intent — an edit must never
/// resolve its own target through the selection at apply time, or it silently retargets when
/// the selection moves. A pure shape switch emits WITHOUT an auto-frame: re-resolving at the
/// same size must not move the camera.
fn shape_cell(
    ui: &mut egui::Ui,
    icon: Icon,
    kind: Option<ShapeKind>,
    state: &mut PanelState,
    response: &mut PanelResponse,
) {
    let target = state.scene.active;
    let reserved = kind.is_none() || target.is_none();
    let active = kind.is_some_and(|k| k == state.geometry.shape) && !reserved;

    let sense = if reserved {
        egui::Sense::hover()
    } else {
        egui::Sense::click()
    };
    let (rect, cell) = ui.allocate_exact_size(egui::vec2(RAIL_WIDTH, CELL_HEIGHT), sense);
    paint_cell(ui, rect, active, cell.hovered() && !reserved);

    let color = cell_ink(active, cell.hovered() && !reserved, reserved);
    let glyph = egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(TILE_GLYPH));
    // The tile drawing where the noun has one; otherwise its rail twin, which is the
    // designed fallback rather than a missing asset.
    match LargeIcon::for_icon(icon) {
        Some(tile) => tile.draw(ui.painter(), glyph, color),
        None => icon.draw(ui.painter(), glyph.shrink(3.0), color),
    }

    let tip = if reserved && kind.is_none() {
        format!("{} — reserved", icon.name())
    } else if reserved {
        format!("{} — select a shape node first", icon.name())
    } else {
        icon.name().to_string()
    };
    let cell = cell.on_hover_text(tip);

    if let (true, Some(kind), Some(target)) = (cell.clicked(), kind, target) {
        state.geometry.shape = kind;
        let shape = SdfShape::from_geometry(state.geometry.clone());
        response.emit(Intent::SetShape { target, shape });
    }
}

/// One tool cell, at rail-glyph size.
fn tool_cell(ui: &mut egui::Ui, icon: Icon, enabled: bool) {
    let sense = if enabled {
        egui::Sense::click()
    } else {
        egui::Sense::hover()
    };
    let (rect, cell) = ui.allocate_exact_size(egui::vec2(RAIL_WIDTH, TOOL_CELL_HEIGHT), sense);
    let hovered = cell.hovered() && enabled;
    // Selection is the only live tool, so it is the one that reads active.
    paint_cell(ui, rect, enabled, hovered);

    let color = cell_ink(enabled, hovered, !enabled);
    let glyph = egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(TOOL_GLYPH));
    icon.draw(ui.painter(), glyph, color);

    let tip = if enabled {
        icon.name().to_string()
    } else {
        format!("{} — reserved", icon.name())
    };
    cell.on_hover_text(tip);
}

/// The cell's fill and its active bar. Active is a 2 px accent inset on the leading edge —
/// never a glow, and never a second hue.
fn paint_cell(ui: &egui::Ui, rect: egui::Rect, active: bool, hovered: bool) {
    let painter = ui.painter();
    if hovered {
        painter.rect_filled(rect, 0.0, signal_theme::ACTIVE_BG);
    } else if active {
        painter.rect_filled(rect, 0.0, signal_theme::HOVER_BG);
    }
    if active {
        let bar = egui::Rect::from_min_size(rect.left_top(), egui::vec2(2.0, rect.height()));
        painter.rect_filled(bar, 0.0, signal_theme::ACCENT);
    }
}

/// A cell glyph's ink: accent when active, lifted on hover, dimmed when reserved.
fn cell_ink(active: bool, hovered: bool, reserved: bool) -> egui::Color32 {
    if reserved {
        signal_theme::TEXT_MUTED.gamma_multiply(RESERVED_DIM)
    } else if active {
        signal_theme::ACCENT
    } else if hovered {
        signal_theme::TEXT_HOVER
    } else {
        signal_theme::TEXT_MUTED
    }
}
