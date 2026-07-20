//! The browser: everything in the document, by name.
//!
//! The fold strip shows the active scope's own fold in ORDER; this column shows the same
//! document by NAME, which is the other question a user asks of it. Depth here is a real
//! indent, unlike the strip, where a part is one opaque card and depth is navigated rather
//! than drawn.
//!
//! Selecting a row emits `SelectNode`. Selection is view state — it re-resolves nothing and
//! its inverse is a no-op — but it still travels as an intent, because there is one door for
//! change and no second way to edit.

use document::intent::Intent;
use document::scene::{NodeContent, NodeId, Scene};

use super::{hairline, region_frame, Edge, BROWSER_WIDTH};
use crate::icons::Icon;
use crate::panel::{PanelResponse, PanelState};
use crate::signal_theme;

/// Row height, sized for a 13 px rail glyph beside 10 px monospace.
const ROW_HEIGHT: f32 = 22.0;
/// The rail glyph's box in a row — the browser tier of the size table.
const ROW_GLYPH: f32 = 13.0;
/// One indent step.
const INDENT: f32 = 12.0;

/// Build the browser column.
pub(super) fn build_browser(
    root_ui: &mut egui::Ui,
    state: &mut PanelState,
    response: &mut PanelResponse,
) {
    egui::Panel::left("workspace_browser")
        .resizable(false)
        .default_size(BROWSER_WIDTH)
        .frame(region_frame())
        .show_inside(root_ui, |ui| {
            let column = ui.max_rect();
            hairline(ui.painter(), column, Edge::Right, signal_theme::BORDER);

            ui.add_space(9.0);
            heading(ui, "Browser");
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
                    // `tree_rows` yields the root part first at depth 0, then its members —
                    // the root is a concrete node, so whole-scene actions are expressed by
                    // selecting a thing rather than by the absence of a selection.
                    for (_, id, depth) in state.scene.tree_rows() {
                        node_row(ui, &state.scene, id, depth, response);
                    }
                });
        });
}

/// A column heading.
fn heading(ui: &mut egui::Ui, title: &str) {
    let galley = signal_theme::letter_spaced(ui, title, signal_theme::TEXT_MUTED, 9.0, 2.0);
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(BROWSER_WIDTH, galley.size().y + 7.0),
        egui::Sense::hover(),
    );
    ui.painter()
        .galley(egui::pos2(rect.left() + 11.0, rect.top()), galley, signal_theme::TEXT_MUTED);
    hairline(ui.painter(), rect, Edge::Bottom, signal_theme::RULE);
}

/// One node row: glyph, name, and the selected treatment.
fn node_row(
    ui: &mut egui::Ui,
    scene: &Scene,
    id: NodeId,
    depth: usize,
    response: &mut PanelResponse,
) {
    let Some(node) = scene.node_by_id(id) else {
        return;
    };
    let selected = scene.active == Some(id);
    let (rect, row) = ui.allocate_exact_size(
        egui::vec2(BROWSER_WIDTH, ROW_HEIGHT),
        egui::Sense::click(),
    );
    let painter = ui.painter();

    if selected {
        painter.rect_filled(rect, 0.0, signal_theme::HOVER_BG);
        let bar = egui::Rect::from_min_size(rect.left_top(), egui::vec2(2.0, rect.height()));
        painter.rect_filled(bar, 0.0, signal_theme::ACCENT);
    } else if row.hovered() {
        painter.rect_filled(rect, 0.0, signal_theme::HOVER_BG);
    }

    let ink = if selected {
        signal_theme::ACCENT
    } else if row.hovered() {
        signal_theme::TEXT_HOVER
    } else {
        signal_theme::TEXT_SECONDARY
    };

    let left = rect.left() + 10.0 + depth as f32 * INDENT;
    let glyph = egui::Rect::from_center_size(
        egui::pos2(left + ROW_GLYPH * 0.5, rect.center().y),
        egui::Vec2::splat(ROW_GLYPH),
    );
    node_icon(node.content_kind_icon()).draw(painter, glyph, ink);

    let label = signal_theme::letter_spaced(ui, &node.name, ink, 10.0, 0.6);
    let at = egui::pos2(
        glyph.right() + 7.0,
        rect.center().y - label.size().y * 0.5,
    );
    ui.painter().galley(at, label, ink);

    if row.clicked() && !selected {
        response.emit(Intent::SelectNode { target: Some(id) });
    }
}

/// Identity mapping kept as a seam: the icon a node kind reads as.
fn node_icon(icon: Icon) -> Icon {
    icon
}

/// The glyph a node's content reads as, in the browser and on a fold card.
trait ContentIcon {
    fn content_kind_icon(&self) -> Icon;
}

impl ContentIcon for document::scene::Node {
    fn content_kind_icon(&self) -> Icon {
        match &self.content {
            NodeContent::Tool { shape, .. } => match shape.kind {
                voxel_core::voxel::ShapeKind::Box => Icon::BoxSolid,
                voxel_core::voxel::ShapeKind::Sphere => Icon::Sphere,
                voxel_core::voxel::ShapeKind::Cylinder => Icon::Cylinder,
                voxel_core::voxel::ShapeKind::Tube => Icon::Tube,
                voxel_core::voxel::ShapeKind::Torus => Icon::Torus,
            },
            NodeContent::SketchTool { .. } => Icon::Sketch,
            NodeContent::VoxelBody(_) => Icon::ComposedPart,
            // The root part and any authored part read as the same container noun; the
            // browser's own indent already says which is which.
            NodeContent::Group(_) => Icon::Part,
            // A linked instance is hatched elsewhere; here the link glyph carries it.
            NodeContent::Instance(_) => Icon::Link,
        }
    }
}
