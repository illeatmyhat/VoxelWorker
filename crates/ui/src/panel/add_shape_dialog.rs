//! The armed-tool **`Add <shape>` dialog** (owner ruling 2026-07-21): a small floating panel
//! pinned to the top-left of the viewport while a primitive is armed, carrying the two
//! session-durable placement snap settings ([`PositionSnap`], [`AngleSnap`]).
//!
//! Drawn with the same **absolute-rect immediate-mode child** idiom as the Signal stack
//! (`signal_stack.rs`) — `root_ui.new_child(..)` at a corner of the central rect, background
//! painted behind via an [`egui::Frame`], the painted `min_rect()` returned as a chrome
//! hit-rect — rather than an `egui::Area`, so it renders on the single-frame headless `shot`
//! capture and the shell's camera gate treats it as chrome (clicks on it don't orbit).

use egui::{CornerRadius, Margin, Pos2, Rect, Stroke, UiBuilder, Vec2};
use voxel_core::voxel::ShapeKind;

use super::state::{AngleSnap, PanelState, PlacementPivot, PositionSnap};

/// Margin from the central-viewport corner, matching the Signal stack's inset.
const DIALOG_MARGIN: f32 = 12.0;
/// Fixed dialog width (points) — wide enough for the three-position row's labels.
const DIALOG_WIDTH: f32 = 232.0;
const BG: egui::Color32 = crate::theme::DIALOG_BG;
const BORDER: egui::Color32 = crate::theme::DIALOG_BORDER;

/// Human label for the dialog title.
fn shape_label(kind: ShapeKind) -> &'static str {
    match kind {
        ShapeKind::Cylinder => "Cylinder",
        ShapeKind::Tube => "Tube",
        ShapeKind::Sphere => "Sphere",
        ShapeKind::Torus => "Torus",
        ShapeKind::Box => "Box",
    }
}

/// Draw the `Add <shape>` dialog for the currently armed `kind`, editing `state.placement_snap`
/// in place. Returns the painted rect (egui points) so the shell can register it as chrome.
pub fn build_add_shape_dialog(
    root_ui: &mut egui::Ui,
    state: &mut PanelState,
    central_rect: Rect,
    kind: ShapeKind,
) -> Rect {
    let left = central_rect.left() + DIALOG_MARGIN;
    let top = central_rect.top() + DIALOG_MARGIN;
    let max_rect = Rect::from_min_size(
        Pos2::new(left, top),
        Vec2::new(DIALOG_WIDTH, (central_rect.height() - 2.0 * DIALOG_MARGIN).max(0.0)),
    );
    let mut dialog_ui = root_ui.new_child(UiBuilder::new().max_rect(max_rect));

    egui::Frame::new()
        .fill(BG)
        .stroke(Stroke::new(1.0_f32, BORDER))
        .corner_radius(CornerRadius::same(3))
        .inner_margin(Margin::same(10))
        .show(&mut dialog_ui, |ui| {
            ui.set_width(DIALOG_WIDTH - 20.0);
            ui.strong(format!("Add {}", shape_label(kind)));
            ui.add_space(8.0);

            ui.label("Position");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut state.placement_snap.position, PositionSnap::NoSnap, "No snap");
                ui.selectable_value(&mut state.placement_snap.position, PositionSnap::Block, "Block");
                ui.selectable_value(&mut state.placement_snap.position, PositionSnap::Voxel, "Voxel");
            });
            ui.add_space(6.0);

            ui.label("Angle");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut state.placement_snap.angle, AngleSnap::Continuous, "Continuous");
                ui.selectable_value(&mut state.placement_snap.angle, AngleSnap::Deg15, "15°");
            });
            ui.add_space(6.0);

            ui.label("Pivot");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut state.placement_snap.pivot, PlacementPivot::Base, "Base");
                ui.selectable_value(&mut state.placement_snap.pivot, PlacementPivot::VolumetricCenter, "Center");
            });
        });

    dialog_ui.min_rect()
}
