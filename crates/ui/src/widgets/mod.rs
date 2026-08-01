//! The shared widget vocabulary: the controls a panel is ASSEMBLED from.
//!
//! A widget here knows nothing about the scene, the selection, or which section drew it
//! — it takes plain values and reports what the user did with them. That is what makes
//! it reusable across information architectures: the same field serves an inspector
//! bound to one selected node and a card bound to a fold entry, because it was never
//! told which one it was in.
//!
//! Sections belong in [`crate::panel`]; only the reusable pieces belong here.

#[allow(clippy::arithmetic_side_effects, clippy::as_conversions, clippy::cast_lossless, clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::derive_partial_eq_without_eq, clippy::doc_link_code, clippy::doc_markdown, clippy::indexing_slicing, clippy::items_after_statements, clippy::manual_midpoint, clippy::map_unwrap_or, clippy::missing_const_for_fn, clippy::must_use_candidate, clippy::needless_pass_by_ref_mut, clippy::option_if_let_else, clippy::redundant_clone, clippy::redundant_closure_for_method_calls, clippy::return_self_not_must_use, clippy::similar_names, clippy::struct_excessive_bools, clippy::suboptimal_flops, clippy::too_long_first_doc_paragraph, clippy::too_many_lines, clippy::tuple_array_conversions, clippy::unreadable_literal, clippy::use_self, clippy::wildcard_imports)]
pub mod measurement_field;

pub use measurement_field::{measurement_error_text, MeasurementCommit, MeasurementField};
