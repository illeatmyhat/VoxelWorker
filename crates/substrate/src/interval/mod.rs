//! Interval and exact arithmetic: field-interval analysis under CSG lattice ops,
//! the sorted disjoint-interval set, and the exact rational. Each module carries
//! its own literature citations.

pub mod disjoint_interval_set;
pub mod field_interval;
pub mod rational;

pub use disjoint_interval_set::DisjointIntervalSet;
pub use field_interval::{union_field_intervals, FieldClassification, FieldInterval};
pub use rational::Rational;
