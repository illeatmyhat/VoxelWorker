//! The fold strip: the ordered fold, as a flat row of cards.
//!
//! A line, never a tree and never a graph. Composition is an ordered fold and later wins, so
//! reading position left-to-right IS reading the meaning. A part is ONE opaque card: its
//! members have no address out here, so nothing in this scope can name them, and depth is
//! navigated by activating the part rather than drawn as nesting.
//!
//! ## The insert cursor is deliberately absent
//!
//! The design synthesis puts a first-class **insert cursor** in the gaps between cards: new
//! nodes land AT it rather than at the end, and nodes past it are dropped from the evaluation
//! without being deleted. That is the whole meaning of an edit under later-wins, and it is
//! genuinely new state — per-scope, view-ish but evaluation-affecting, and named nowhere in
//! `docs/adr/` or `docs/architecture/`. It needs a decision before it exists in code, so this
//! strip currently shows the fold and its selection only. Adding a cursor silently here would
//! be deciding an architecture question inside a widget.

use document::intent::Intent;
use document::scene::{NodeId, Scene};

use super::{hairline, region_frame, Edge, FOLD_STRIP_HEIGHT};
use crate::panel::{PanelResponse, PanelState};
use crate::signal_theme;

/// One fold card.
const CARD_WIDTH: f32 = 150.0;
/// Card height, leaving room for the strip header above it.
const CARD_HEIGHT: f32 = 104.0;
/// The gap between cards — where the insert cursor will live once it is decided.
const CARD_GAP: f32 = 10.0;

/// Build the fold strip band.
pub(super) fn build_fold_strip(
    root_ui: &mut egui::Ui,
    state: &mut PanelState,
    response: &mut PanelResponse,
) {
    egui::Panel::bottom("workspace_fold_strip")
        .resizable(false)
        .default_size(FOLD_STRIP_HEIGHT)
        .frame(region_frame())
        .show_inside(root_ui, |ui| {
            let band = ui.max_rect();
            hairline(ui.painter(), band, Edge::Top, signal_theme::BORDER);

            ui.add_space(9.0);
            header(ui, state);
            ui.add_space(8.0);

            let roots = state.scene.roots.clone();
            egui::ScrollArea::horizontal()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(CARD_GAP, 0.0);
                        ui.add_space(11.0);
                        for (index, id) in roots.iter().enumerate() {
                            card(ui, &state.scene, *id, index, response);
                        }
                    });
                });
        });
}

/// The strip header: which scope's fold this is, and the law that governs it.
fn header(ui: &mut egui::Ui, state: &PanelState) {
    let count = state.scene.roots.len();
    ui.horizontal(|ui| {
        ui.add_space(11.0);
        let title = signal_theme::letter_spaced(
            ui,
            "Fold · root part",
            signal_theme::TEXT_MUTED,
            9.0,
            2.0,
        );
        let (r, _) = ui.allocate_exact_size(title.size(), egui::Sense::hover());
        ui.painter().galley(r.min, title, signal_theme::TEXT_MUTED);
        ui.add_space(10.0);
        let sub = signal_theme::letter_spaced(
            ui,
            &format!("{count} nodes · later wins · depth is navigated, not drawn"),
            signal_theme::TEXT_HINT,
            8.5,
            1.2,
        );
        let (r, _) = ui.allocate_exact_size(sub.size(), egui::Sense::hover());
        ui.painter().galley(r.min, sub, signal_theme::TEXT_HINT);
    });
}

/// One fold card: the operation it folds as, its name, and what it contributes.
fn card(
    ui: &mut egui::Ui,
    scene: &Scene,
    id: NodeId,
    index: usize,
    response: &mut PanelResponse,
) {
    let Some(node) = scene.node_by_id(id) else {
        return;
    };
    let selected = scene.active == Some(id);
    let (rect, hit) = ui.allocate_exact_size(
        egui::vec2(CARD_WIDTH, CARD_HEIGHT),
        egui::Sense::click(),
    );
    let painter = ui.painter();

    painter.rect_filled(rect, 0.0, signal_theme::BG);
    let edge = if selected {
        signal_theme::ACCENT
    } else if hit.hovered() {
        signal_theme::TEXT_FAINT
    } else {
        signal_theme::BORDER
    };
    painter.rect_stroke(rect, 0.0, egui::Stroke::new(1.0_f32, edge), egui::StrokeKind::Inside);

    let ink = if selected {
        signal_theme::TEXT_PRIMARY
    } else {
        signal_theme::TEXT_SECONDARY
    };

    // Row 1: the fold index and the operation, which is what position MEANS here.
    let op = signal_theme::letter_spaced(
        ui,
        &format!("#{}  {}", index + 1, node.operation_label()),
        if selected { signal_theme::ACCENT } else { signal_theme::TEXT_MUTED },
        8.5,
        1.4,
    );
    ui.painter().galley(
        egui::pos2(rect.left() + 10.0, rect.top() + 10.0),
        op,
        signal_theme::TEXT_MUTED,
    );

    // Row 2: the name.
    let name = signal_theme::letter_spaced(ui, &node.name, ink, 10.0, 0.6);
    ui.painter()
        .galley(egui::pos2(rect.left() + 10.0, rect.top() + 30.0), name, ink);

    if hit.clicked() && !selected {
        response.emit(Intent::SelectNode { target: Some(id) });
    }
}

/// How a node's combine operation reads on a card.
trait OperationLabel {
    fn operation_label(&self) -> &'static str;
}

impl OperationLabel for document::scene::Node {
    fn operation_label(&self) -> &'static str {
        use document::scene::CombineOp;
        match self.operation {
            CombineOp::Union => "UNION",
            CombineOp::Subtract => "SUBTRACT",
            CombineOp::Intersect => "INTERSECT",
            CombineOp::Emboss { .. } => "EMBOSS",
        }
    }
}
