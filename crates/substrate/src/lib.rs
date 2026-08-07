#![allow(clippy::too_long_first_doc_paragraph)]

//! # substrate — the pure computer-science / mathematics library
//!
//! This crate holds the load-bearing data structures whose identity is purely
//! *algorithmic*, split out of the domain so they can be identified, read, and
//! reasoned about (including their performance) in isolation. It is not intended
//! for release; it is intended for reading. The application crate depends on
//! substrate; substrate depends on no domain code, and that direction is
//! compile-enforced by the crate boundary.
//!
//! See the **Substrate** section of `docs/architecture/data-structures.md` for
//! the same rules stated over the whole structure set.
//!
//! ## The boundary law
//!
//! A component belongs in this crate if and only if it is describable *entirely*
//! in textbook computer-science / mathematics vocabulary — a bounding-volume
//! hierarchy, an axis-aligned box, a bit-packed occupancy cube, interval
//! arithmetic, a min-mip pyramid, a slot allocator, a space-filling key codec, a
//! rational, a supersede protocol — and is parameterized only by plain numbers
//! and generics, **never by domain types**. Anything that must name a scene, a
//! producer, a chunk, or a brick-as-block is a domain adapter and stays in the
//! application crate at its own seam.
//!
//! ## Naming rule
//!
//! Each component lives in its own module, and the well-known name from the
//! scientific literature *is* the type's name (`MedianSplitBvh`, `LatticeAabb`,
//! `BitCube`, `DisjointIntervalSet`, `ExactRational`, …). The explanation of the
//! structure and the citations to the canonical literature — together with a note
//! on how this implementation's variant deviates — live in the component's own
//! definition, not here. Domain vocabulary survives only at the adapter seams in
//! the application crate.
//!
//! ## Benches
//!
//! Criterion microbenches (`crates/substrate/benches/`) exist for the *hot*
//! components only, and are run on demand — never part of the commit gates.
//!
//! ## Components
//!
//! Each is a self-contained module with its own literature citations and oracles,
//! grouped into category modules:
//!
//! - [`spatial`] — [`LatticeAabb`](spatial::LatticeAabb) and its closed f32 twin
//!   [`RealAabb`](spatial::RealAabb), the [`Bvh`](spatial::Bvh), the
//!   [`lattice_key`](spatial::lattice_key) packing codec, the [`Ray`](spatial::Ray) primitive
//!   with its slab-method box test, and the [`SparseMinMipPyramid`](spatial::SparseMinMipPyramid)
//!   occupancy fold.
//! - [`geom2d`] — planar computational geometry, split across two float widths on purpose
//!   (see its module docs before touching either half). The exact `f64` **predicates**
//!   [`orient2d`](geom2d::orient2d), [`segments_intersect`](geom2d::segments_intersect),
//!   [`segment_intersects_rect`](geom2d::segment_intersects_rect) and
//!   [`rectangle_inside_polygon`](geom2d::rectangle_inside_polygon); the `f32`
//!   **measurements** [`Metric`](geom2d::Metric),
//!   [`distance_point_to_segment`](geom2d::distance_point_to_segment),
//!   [`signed_distance_to_polygon`](geom2d::signed_distance_to_polygon) and
//!   [`point_in_polygon`](geom2d::point_in_polygon), which a WGSL preview mirrors.
//! - [`curve_intersection`] — where two planar curves meet:
//!   [`PlanarCurve`](curve_intersection::PlanarCurve) and the
//!   [`CurveCrossing`](curve_intersection::CurveCrossing)s it reports, located by PARAMETER on
//!   both curves so an arrangement can cut at them. Exact-`f64` throughout, because a missed
//!   crossing changes the topology of the answer rather than its precision.
//! - [`noise`] — a procedural-generation kit: the [`SmallRng`](noise::SmallRng) LCG and
//!   [`PerlinNoise`](noise::PerlinNoise) gradient noise with fBm.
//! - [`nonlinear_least_squares`] — Powell's Dog Leg over a rank-revealing linear solve:
//!   [`solve`](nonlinear_least_squares::solve) drives a
//!   [`ResidualSystem`](nonlinear_least_squares::ResidualSystem) to its nearest solution, and the
//!   [`SolveReport`](nonlinear_least_squares::SolveReport) carries the Jacobian's rank as degrees
//!   of freedom and redundant residuals — the numerical core a constraint solver runs on.
//! - [`complete_orthogonal_decomposition`] — LAPACK's `xGELSY`:
//!   [`minimum_norm_least_squares`](complete_orthogonal_decomposition::minimum_norm_least_squares)
//!   answers `Ax ≈ b` for any shape and any rank, by pivoted Householder QR and a trapezoidal
//!   reduction, never forming `AᵀA` and so never squaring its conditioning. Picking the shortest of
//!   a rank-deficient system's equally good answers is a stated GAUGE CHOICE — `docs/adr/`
//!   ADR 0047 records what leaving it to a damped factorisation cost.
//! - [`interval`] — [`FieldInterval`](interval::FieldInterval),
//!   [`DisjointIntervalSet`](interval::DisjointIntervalSet), and [`Rational`](interval::Rational).
//! - [`occupancy`] — the bit/atlas kit: [`BitCube`](occupancy::BitCube) and its payload sibling
//!   [`ValueCube`](occupancy::ValueCube), [`SlotFreeList`](occupancy::SlotFreeList),
//!   [`CubeTilePacking`](occupancy::CubeTilePacking), the
//!   [`ShelfBinPack`](occupancy::ShelfBinPack) rectangle packer, and the
//!   [`SortedKeyBitmaskMap`](occupancy::SortedKeyBitmaskMap).
//! - [`solids`] — the [`CellClassification`](solids::CellClassification) black/white/gray CSG cell
//!   classifier, the [`GreedyCuboidDecomposition`](solids::GreedyCuboidDecomposition) into
//!   [`Cuboid`](solids::Cuboid)s, and the [`CulledBoxMeshing`](solids::CulledBoxMeshing)
//!   exposed-face determination.
//! - crate root — the [`supersede`] protocol ([`CoalescingWorker`], [`GenerationTracker`], and
//!   their [`drain_to_latest`] / [`catch_unwind_or_log`] helpers) and the [`srgb`]
//!   transfer-function codec, which belong to no family.
//!
//! See the extraction map referenced above for each component's provenance and the domain
//! adapter that wraps it.

// Components are grouped into category modules so the taxonomy is visible at the
// call site (`substrate::spatial::LatticeAabb`, `substrate::occupancy::BitCube`);
// each category module re-exports its own types. `supersede` and `srgb` belong to
// no family and stay at the crate root.
pub mod complete_orthogonal_decomposition;
pub mod curve_intersection;
pub mod geom2d;
pub mod interval;
pub mod noise;
pub mod nonlinear_least_squares;
pub mod occupancy;
pub mod rational_bezier;
pub mod solids;
pub mod spatial;
pub mod srgb;
pub mod supersede;
pub mod winding;

pub use srgb::{srgb_component_to_linear, srgb_hex_to_linear};
pub use supersede::{catch_unwind_or_log, drain_to_latest, CoalescingWorker, GenerationTracker};
