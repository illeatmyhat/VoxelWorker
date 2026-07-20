//! The top bar: who you are, where you are, what you are looking at, and the numbers.
//!
//! Left to right — brand, the scope breadcrumb, the exclusive viewer segment; then, hard
//! against the right edge, the readouts. The viewer segment is the *document-independent*
//! half of the bar: mode is viewer state and never enters the scene or the undo stack, so it
//! mutates `PanelState` in place rather than emitting an intent.
//!
//! Everything here is painted at absolute positions inside the band rather than flowed
//! through nested egui layouts. A right-aligned group nested inside a horizontal layout
//! fights the parent for the same cursor and lands the readouts on top of the segment — it
//! did exactly that on the first cut. Explicit rects are also how the shipped viewport chrome
//! is drawn, so the two surfaces agree about what "hard against the edge" means.

use document::scene::ROOT_NODE_ID;

use super::{hairline, region_frame, Edge, TOP_BAR_HEIGHT};
use crate::panel::{PanelResponse, PanelState, ViewMode};
use crate::signal_theme;

/// The segmented viewer control's cell size.
const SEGMENT_HEIGHT: f32 = 22.0;
const SEGMENT_WIDTH: f32 = 104.0;
/// Padding in from either end of the band.
const EDGE_PAD: f32 = 13.0;
/// Gap between readout stacks.
const READOUT_GAP: f32 = 22.0;

/// Build the top bar band.
pub(super) fn build_top_bar(
    root_ui: &mut egui::Ui,
    state: &mut PanelState,
    _response: &mut PanelResponse,
) {
    egui::Panel::top("workspace_top_bar")
        .resizable(false)
        .default_size(TOP_BAR_HEIGHT)
        .frame(region_frame())
        .show_inside(root_ui, |ui| {
            // Claim the band's full height explicitly. This region PAINTS at absolute
            // positions rather than flowing widgets, so without an allocation egui sizes the
            // panel to its (empty) content and the bar collapses to a sliver —
            // `default_size` only seeds a RESIZABLE panel, which this is not.
            let (band, _) = ui.allocate_exact_size(
                egui::vec2(ui.available_width(), TOP_BAR_HEIGHT),
                egui::Sense::hover(),
            );
            hairline(ui.painter(), band, Edge::Bottom, signal_theme::BORDER);
            let middle = band.center().y;

            // --- left, packed in reading order ---
            let mut x = band.left() + EDGE_PAD;
            x += paint_left(ui, "VoxelWorker", signal_theme::TEXT_PRIMARY, 11.0, 3.0, x, middle);
            x += 14.0;
            rule(ui, x, band);
            x += 11.0;

            let root_name = state
                .scene
                .node_by_id(ROOT_NODE_ID)
                .map(|node| node.name.clone())
                .unwrap_or_else(|| "Part".to_string());
            x += paint_left(ui, &root_name, signal_theme::TEXT_PRIMARY, 10.0, 1.6, x, middle);
            x += 14.0;
            rule(ui, x, band);
            x += 12.0;

            x += paint_left(ui, "Viewer", signal_theme::TEXT_MUTED, 9.0, 1.4, x, middle);
            x += 9.0;
            viewer_segment(ui, state, x, middle);

            // --- right, laid out from the edge inwards ---
            readouts(ui, state, band, middle);
        });
}

/// Paint a letter-spaced run at `x`, vertically centred on `middle`. Returns its width so
/// the caller can advance.
fn paint_left(
    ui: &egui::Ui,
    text: &str,
    color: egui::Color32,
    size: f32,
    spacing: f32,
    x: f32,
    middle: f32,
) -> f32 {
    let galley = signal_theme::letter_spaced(ui, text, color, size, spacing);
    let width = galley.size().x;
    ui.painter().galley(
        egui::pos2(x, middle - galley.size().y * 0.5),
        galley,
        color,
    );
    width
}

/// The three exclusive viewer modes as one segmented control.
///
/// Exclusive is the point rather than a simplification: onion clip and boolean x-ray can
/// never co-render, which is what dissolved the "fog over the object" bug class. Exactly one
/// cell is accent-filled and carries dark text — the Signal active treatment, never a glow.
fn viewer_segment(ui: &mut egui::Ui, state: &mut PanelState, x: f32, middle: f32) {
    for (index, mode) in [ViewMode::Normal, ViewMode::OnionFog, ViewMode::ShowBooleans]
        .into_iter()
        .enumerate()
    {
        let rect = egui::Rect::from_min_size(
            egui::pos2(x + index as f32 * SEGMENT_WIDTH, middle - SEGMENT_HEIGHT * 0.5),
            egui::vec2(SEGMENT_WIDTH, SEGMENT_HEIGHT),
        );
        let hit = ui.interact(
            rect,
            egui::Id::new(("workspace_viewer_segment", index)),
            egui::Sense::click(),
        );
        let active = mode == state.view_mode;
        let painter = ui.painter();

        if active {
            painter.rect_filled(rect, 0.0, signal_theme::ACCENT);
        } else if hit.hovered() {
            painter.rect_filled(rect, 0.0, signal_theme::HOVER_BG);
        }
        painter.rect_stroke(
            rect,
            0.0,
            egui::Stroke::new(1.0_f32, signal_theme::BORDER),
            egui::StrokeKind::Inside,
        );

        let ink = if active {
            signal_theme::ACCENT_TEXT
        } else if hit.hovered() {
            signal_theme::TEXT_HOVER
        } else {
            signal_theme::TEXT_SECONDARY
        };
        let galley = signal_theme::letter_spaced(ui, mode.status_label(), ink, 9.0, 1.2);
        let at = rect.center() - galley.size() * 0.5;
        ui.painter().galley(at, galley, ink);

        if hit.clicked() {
            state.view_mode = mode;
        }
    }
}

/// The right-hand readouts, measured and placed from the right edge inwards so they never
/// collide with the left-packed run however wide the window is.
fn readouts(ui: &egui::Ui, state: &PanelState, band: egui::Rect, middle: f32) {
    let mut right = band.right() - EDGE_PAD;
    for (label, value) in [
        ("Nodes", format!("{}", state.scene.arena.len())),
        ("Density", format!("{}³ / block", state.scene.voxels_per_block)),
    ] {
        let l = signal_theme::letter_spaced(ui, label, signal_theme::TEXT_HINT, 8.5, 1.4);
        let v = signal_theme::letter_spaced(ui, &value, signal_theme::TEXT_SECONDARY, 9.5, 0.8);
        let width = l.size().x.max(v.size().x);
        let left = right - width;
        let block_height = l.size().y + v.size().y + 2.0;
        let top = middle - block_height * 0.5;
        ui.painter()
            .galley(egui::pos2(left, top), l, signal_theme::TEXT_HINT);
        ui.painter().galley(
            egui::pos2(left, top + block_height - v.size().y),
            v,
            signal_theme::TEXT_SECONDARY,
        );
        right = left - READOUT_GAP;
    }
}

/// A full-height hairline divider between top-bar groups.
fn rule(ui: &egui::Ui, x: f32, band: egui::Rect) {
    ui.painter().line_segment(
        [
            egui::pos2(x, band.top() + 7.0),
            egui::pos2(x, band.bottom() - 7.0),
        ],
        egui::Stroke::new(1.0_f32, signal_theme::RULE),
    );
}
