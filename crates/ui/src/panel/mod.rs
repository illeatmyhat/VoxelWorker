//! The shared egui side panel.
//!
//! Exactly one implementation, used by both the windowed app and the headless
//! screenshot harness, so the captured frame is identical to the live one.
//!
//! The inspector's [`GeometryParams`](document::voxel::GeometryParams) (shape, size,
//! density, wall) are split from display/camera params (projection, material selection) by
//! *consumer*:
//!
//!   * [`GeometryParams`](document::voxel::GeometryParams) drives a **rebuild-dirty** flag.
//!     Changing it re-resolves the voxel grid.
//!   * Display/camera params live in [`PanelState`] directly and never trigger a voxel
//!     rebuild. (`projection_mode` is a `PanelState` display field on this no-rebuild side
//!     of the split even though its toggle is drawn by the floating Signal display stack.)
//!
//! This split is what enforces the regression guards: selecting a shape only
//! sets [`GeometryParams::shape`](document::voxel::GeometryParams::shape) (never
//! the size or the camera), and changing density only sets
//! [`GeometryParams::voxels_per_block`](document::voxel::GeometryParams::voxels_per_block)
//! (never the block size).
//!
//! The panel is one logical unit split across submodules by section identity:
//! [`state`] holds the mutable state + response types; the `build_*` section
//! builders live in [`nodes`], [`points`], [`inspector`], [`controls`],
//! [`layers`], and [`palette`]; [`build_panel`] (here) is the top-level
//! assembler that lays them out. Every public item is re-exported here, so
//! `ui::panel::…` paths resolve whichever submodule owns it.

#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::doc_link_code,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::needless_pass_by_ref_mut,
    clippy::option_if_let_else,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_lines,
    clippy::tuple_array_conversions,
    clippy::unreadable_literal,
    clippy::use_self,
    clippy::wildcard_imports
)]
mod add_shape_dialog;
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::doc_link_code,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::needless_pass_by_ref_mut,
    clippy::option_if_let_else,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_lines,
    clippy::tuple_array_conversions,
    clippy::unreadable_literal,
    clippy::use_self,
    clippy::wildcard_imports
)]
mod controls;
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::doc_link_code,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::needless_pass_by_ref_mut,
    clippy::option_if_let_else,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_lines,
    clippy::tuple_array_conversions,
    clippy::unreadable_literal,
    clippy::use_self,
    clippy::wildcard_imports
)]
mod inspector;
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::doc_link_code,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::needless_pass_by_ref_mut,
    clippy::option_if_let_else,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_lines,
    clippy::tuple_array_conversions,
    clippy::unreadable_literal,
    clippy::use_self,
    clippy::wildcard_imports
)]
mod layers;
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::doc_link_code,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::needless_pass_by_ref_mut,
    clippy::option_if_let_else,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_lines,
    clippy::tuple_array_conversions,
    clippy::unreadable_literal,
    clippy::use_self,
    clippy::wildcard_imports
)]
mod nodes;
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::doc_link_code,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::needless_pass_by_ref_mut,
    clippy::option_if_let_else,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_lines,
    clippy::tuple_array_conversions,
    clippy::unreadable_literal,
    clippy::use_self,
    clippy::wildcard_imports
)]
mod palette;
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::doc_link_code,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::needless_pass_by_ref_mut,
    clippy::option_if_let_else,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_lines,
    clippy::tuple_array_conversions,
    clippy::unreadable_literal,
    clippy::use_self,
    clippy::wildcard_imports
)]
mod points;
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::doc_link_code,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::needless_pass_by_ref_mut,
    clippy::option_if_let_else,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_lines,
    clippy::tuple_array_conversions,
    clippy::unreadable_literal,
    clippy::use_self,
    clippy::wildcard_imports
)]
mod selection;
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::doc_link_code,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::needless_pass_by_ref_mut,
    clippy::option_if_let_else,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_lines,
    clippy::tuple_array_conversions,
    clippy::unreadable_literal,
    clippy::use_self,
    clippy::wildcard_imports
)]
mod signal_stack;
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::doc_link_code,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::needless_pass_by_ref_mut,
    clippy::option_if_let_else,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_lines,
    clippy::tuple_array_conversions,
    clippy::unreadable_literal,
    clippy::use_self,
    clippy::wildcard_imports
)]
mod sketch_constraint;
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::doc_link_code,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::needless_pass_by_ref_mut,
    clippy::option_if_let_else,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_lines,
    clippy::tuple_array_conversions,
    clippy::unreadable_literal,
    clippy::use_self,
    clippy::wildcard_imports
)]
mod state;

pub use add_shape_dialog::build_add_shape_dialog;
pub(crate) use nodes::tool_node_spec;
pub use selection::{Selection, SelectionRequest, SelectionTarget};
pub use signal_stack::{build_signal_stack, cube_right_inset_points};
pub use sketch_constraint::{
    constraint_icon, ArmedConstraint, ConstraintVerb, Offer, PickRequirement, SketchEntity,
    SlotKind,
};
pub use state::{
    AngleSnap, ArmedTool, CurveUnderPointer, ExportPanelState, LayerRange, ModeCommand,
    OrbitCenterRequest, OrbitMode, PanelResponse, PanelState, PlacementGhost, PlacementPivot,
    PlacementSnap, PositionSnap, SignalStackState, SketchExit, SketchTool, ViewMode,
};

use crate::palette::BlockPalette;
use crate::theme;

/// A floating instrument's own [`egui::Ui`]: a NON-ALLOCATING child of the root at an absolute
/// rect, on its own [`egui::Order::Foreground`] layer.
///
/// Both halves are load-bearing and each was learned the hard way.
///
/// Non-allocating, because [`egui::Ui::scope_builder`] advances the PARENT cursor past the child,
/// and `Context::run_ui` records the root's remaining `available_rect_before_wrap` as the "not
/// over egui" input region — so a scoped instrument carves a full-width band out of the viewport's
/// input and the shell's orbit, pan and zoom go dead across it.
///
/// On a named layer, because a child that does not name one paints in the root ui's, which is
/// [`egui::LayerId::background()`]. That is the trap: every other piece of chrome says
/// `Order::Foreground` out loud, so a search for the instruments finds the ones that opted in and
/// silently omits the ones that inherited — and an instrument in the background tier is one the
/// sketch overlay is entitled to paint straight over. **A floating instrument states its tier;
/// inheriting one is how it ends up under the drawing.**
pub(crate) fn floating_instrument(
    root_ui: &mut egui::Ui,
    max_rect: egui::Rect,
    id: &'static str,
) -> egui::Ui {
    root_ui.new_child(
        egui::UiBuilder::new()
            .max_rect(max_rect)
            .layer_id(crate::chrome::chrome_layer(id)),
    )
}

/// Build the right-hand side panel into the root [`egui::Ui`] of the frame.
///
/// The sidebar hosts the scene tree, points, inspector and export; the display-related
/// sections (VIEWPORT / ONION FOG / GRIDS) live in the floating Signal display stack
/// ([`build_signal_stack`]), which the shell renders separately with the layer-track length +
/// measured diameter. Returns a [`PanelResponse`] describing what the user changed.
/// The sidebar's section stack, independent of which column hosts it.
///
/// Factored out so the [`workspace`](crate::workspace) inspector column hosts the same
/// sections rather than a duplicate set that drifts apart from these.
pub(crate) fn build_sidebar_sections(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    export: ExportPanelState,
    response: &mut PanelResponse,
) {
    nodes::build_node_list_section(ui, state, response);
    points::build_points_section(ui, state, response);
    inspector::build_inspector_section(ui, state, response);
    controls::build_export_section(ui, response, export);
}

pub fn build_panel(
    root_ui: &mut egui::Ui,
    state: &mut PanelState,
    export: ExportPanelState,
    palette: &BlockPalette,
) -> PanelResponse {
    let mut response = PanelResponse::default();

    // The palette dock lives along the bottom, as its own bottom panel, so the right-hand
    // controls keep their width.
    palette::build_palette_dock(root_ui, palette, &mut response);

    egui::Panel::right("voxel_worker_controls")
        .resizable(false)
        .default_size(300.0)
        .show_inside(root_ui, |ui| {
            // The panel outgrows short windows; scroll (wheel or drag) instead
            // of clipping the lower sections.
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    // The title block: primary-tier wordmark over a faint subtitle.
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new("VoxelWorker")
                            .monospace()
                            .size(15.0)
                            .strong()
                            .color(theme::TEXT_PRIMARY),
                    );
                    ui.label(
                        egui::RichText::new("Vintage Story chiseling planner")
                            .monospace()
                            .size(9.5)
                            .color(theme::TEXT_FAINT),
                    );
                    ui.add_space(6.0);
                    ui.separator();

                    // The display-related sections (VIEWPORT / ONION FOG / GRIDS) live in the
                    // floating Signal display stack (`panel::signal_stack`, rendered by
                    // `run_egui_frame`). The sidebar keeps the scene tree, points, inspector
                    // and export.
                    build_sidebar_sections(ui, state, export, &mut response);

                    if let Some(millions) = state.voxel_cap_warning_millions {
                        ui.add_space(8.0);
                        ui.separator();
                        ui.colored_label(
                            theme::WARN,
                            format!("3D paused — {millions:.1}M voxels; lower size/density"),
                        );
                    }
                    if state.coordinate_limit_warning {
                        ui.add_space(8.0);
                        ui.separator();
                        ui.colored_label(
                            theme::WARN,
                            "position exceeds the ±1,000,000-block coordinate limit",
                        );
                    }
                });
        });

    response
}
