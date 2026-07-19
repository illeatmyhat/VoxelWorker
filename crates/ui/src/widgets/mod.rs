//! The shared widget vocabulary: the controls a panel is ASSEMBLED from.
//!
//! A widget here knows nothing about the scene, the selection, or which section drew it
//! — it takes plain values and reports what the user did with them. That is what makes
//! it reusable across information architectures: the same field serves an inspector
//! bound to one selected node and a card bound to a fold entry, because it was never
//! told which one it was in.
//!
//! Sections belong in [`crate::panel`]; only the reusable pieces belong here.

pub mod measurement_field;

pub use measurement_field::{measurement_error_text, MeasurementCommit, MeasurementField};
