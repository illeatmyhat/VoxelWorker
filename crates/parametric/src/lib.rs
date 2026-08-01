//! # parametric — authored quantities, dimensions, and expressions
//!
//! Everything the author *types* and the document *retains as an expression* rather than
//! as a number. It is the substrate the constraint solver drives and the parameters panel
//! edits.
//!
//! ## The three layers
//!
//! - **[`units`]** — the blocks/voxels measurement core. [`units::Measurement`] is a length
//!   as an authored expression (an exact-rational block term plus an integer voxel term),
//!   and [`units::AngleMeasurement`] is an angle in exact degrees. These are the
//!   **statically typed** faces: a radius field takes a `Measurement`, an angle dimension
//!   takes an `AngleMeasurement`, and adding one to the other does not compile.
//! - **[`dimension`]** — the exponent algebra that says *why* it does not compile.
//! - **[`quantity`]** — [`quantity::Quantity`], the **dynamically typed** value the
//!   expression evaluator works in, because `wall / gap` has a dimension only at eval time.
//!
//! The seam between the last two is the whole design: evaluate an expression, get a
//! `Quantity`, check its dimension against the destination field's static type, then store
//! or report. There is one umbrella quantity type — the `Quantity` — with the static
//! wrappers above it.
//!
//! ## Exactness is a storage property, not a solver property
//!
//! Nothing here touches `f64` except at an explicit evaluation door
//! ([`units::Measurement::to_voxels`], [`units::AngleMeasurement::to_degrees_f64`]).
//! Authored values are exact rationals so a persisted document is float-free end to end and
//! a density re-target re-evaluates losslessly.
//!
//! The continuous sketch solver lives here. It owns only resolved planar geometry and
//! relation semantics; document ids, persistence, density, and voxel evaluation remain above it.
//!
//! ## Why it is its own crate
//!
//! `substrate` may not name a block or a voxel, and `voxel_core` is the crate of
//! plain resolved *values* — a quantity that carries an unevaluated expression, a symbol
//! table and a type system is not that. So this sits between them:
//! `substrate ← parametric ← voxel_core ← the rest of the app`.

pub mod curve_parameter;
pub mod dimension;
pub mod evaluation;
pub mod expression;
pub mod quantity;
pub mod sketch;
pub mod units;

pub use curve_parameter::{ArcSweep, CircleRadius, CurveParameter, ResolvedLength};
pub use dimension::Dimension;
pub use evaluation::EvaluationContext;
pub use expression::{Expression, SymbolTable};
pub use quantity::Quantity;
pub use units::{AngleMeasurement, ExactRational, Measurement};
