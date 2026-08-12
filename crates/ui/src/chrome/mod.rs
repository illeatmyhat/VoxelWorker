//! The Signal viewport chrome painters: the icon [`rail`], the floating [`notice`], and the
//! sketch [`sketch_overlay`]. Pure egui painting at shell-computed positions — the shell owns the
//! projection, hit-testing and interaction routing; these only draw and report clicks.
//!
//! The viewport's bottom-left corner belongs to the [`notice`], which is only there when it has
//! something to say. Nothing persistent stands over the drawing: the viewer mode is on the top
//! bar, and the extent and the density are inspector facts.

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
mod notice;
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
mod rail;
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
mod sketch_overlay;
mod view_cube;

/// The layer every INSTRUMENT draws on — one door, so the tier is auditable.
///
/// Chrome is over the drawing and the drawing is over the scene, and egui gives that to us only
/// through the `Order` tier: within one order, the layers this application makes are not areas, so
/// `GraphicLayers::drain` empties them in HASH-MAP ITERATION ORDER, out of a branch egui's own
/// comment calls a safety net it does not expect to reach. Nothing within a tier is ordered. The
/// tiers are carrying the whole arrangement.
///
/// Which is why there are exactly two doors to a tier — this one, and
/// [`sketch_mark_painter`](sketch_overlay) for the marks below it — and the audit is
/// `LayerId::new` outside them. Enumerating the chrome by grepping for `Order::` was tried and it
/// is not the same thing: it finds everything that NAMES an order and is blind to anything that
/// inherited one. The DISPLAY stack inherited the root ui's background layer that way and spent a
/// release under the sketch.
///
/// [`crate::gizmos::orbit_reticle_overlay`] is the one deliberate exception, and it documents
/// itself: it is a camera mark the chrome is supposed to cover.
pub(crate) fn chrome_layer(id: &'static str) -> egui::LayerId {
    egui::LayerId::new(egui::Order::Foreground, egui::Id::new(id))
}

/// The tier a MENU is raised on — the third door, above the instruments.
///
/// A menu has to cover the instrument that opened it, and on the chrome tier it would not. Within
/// one order `GraphicLayers::drain` empties the AREAS first and every other layer after them, so a
/// menu — which is an `Area` — comes out UNDER a bare instrument layer sharing its order. That is
/// backwards for the two pairs that matter (the cube's context menu over the cube, the rail's
/// orbit-type menu over the rail), and it is not luck that can be left alone: it is a guarantee
/// pointing the wrong way.
///
/// So menus take the next tier up. There is nothing between `Foreground` and `Tooltip` to take
/// instead — the enum has five values and this application already occupies three of them.
/// Sharing the tier with egui's own tooltips is harmless: both are areas, and areas within a tier
/// ARE ordered, by which one the pointer last touched.
pub const MENU_ORDER: egui::Order = egui::Order::Tooltip;

pub use notice::viewport_notice;
pub use rail::{icon_rail, orbit_type_button_rect, rail_height, rail_rect, rail_top, RailClick};
pub use sketch_overlay::{
    sketch_arc_curves, sketch_constraint_badges, sketch_dimension_gizmos, sketch_draw_preview,
    sketch_exit_control, sketch_insert_marker, sketch_marquee_band, sketch_segment_lines,
    sketch_snap_marker, sketch_vertex_handles, ConstraintBadge, DimensionGizmo, SketchCurveInk,
    SketchCurveLine, SketchEdgeLine, SketchPreviewLine, SketchPreviewMark, SketchVertexHandle,
    SketchVertexInk, SKETCH_CONSTRAINT_BADGE, SKETCH_CONSTRAINT_BADGE_OFFSET,
    SKETCH_HANDLE_GRAB_PAD, SKETCH_HANDLE_HALF, SKETCH_INSERT_MARKER_HALF,
    SKETCH_PREVIEW_POINT_HALF, SKETCH_SEGMENT_GRAB_PAD, SKETCH_SNAP_MARKER_INNER,
    SKETCH_SNAP_MARKER_OUTER, SKETCH_SNAP_REACH,
};
pub use view_cube::view_cube_image;
