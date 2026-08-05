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

pub use notice::viewport_notice;
pub use rail::{icon_rail, orbit_type_button_rect, rail_height, rail_rect, rail_top, RailClick};
pub use sketch_overlay::{
    sketch_arc_curves, sketch_constraint_badges, sketch_draw_preview, sketch_exit_control,
    sketch_insert_marker, sketch_marquee_band, sketch_segment_lines, sketch_vertex_handles,
    ConstraintBadge, SketchCurveInk, SketchCurveLine, SketchEdgeLine, SketchPreviewLine,
    SketchPreviewMark, SketchVertexHandle, SketchVertexInk, SKETCH_CONSTRAINT_BADGE,
    SKETCH_CONSTRAINT_BADGE_OFFSET, SKETCH_HANDLE_GRAB_PAD, SKETCH_HANDLE_HALF,
    SKETCH_INSERT_MARKER_HALF, SKETCH_PREVIEW_POINT_HALF, SKETCH_SEGMENT_GRAB_PAD,
};
