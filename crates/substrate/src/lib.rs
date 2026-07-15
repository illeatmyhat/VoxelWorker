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
//! the timeless statement of the same rules; the dated component inventory and
//! slice order live in `docs/design/substrate-extraction-map.md`.
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
//! - [`interval`] — [`FieldInterval`](interval::FieldInterval),
//!   [`DisjointIntervalSet`](interval::DisjointIntervalSet), and [`Rational`](interval::Rational).
//! - [`occupancy`] — the bit/atlas kit: [`BitCube`](occupancy::BitCube) and its payload sibling
//!   [`ValueCube`](occupancy::ValueCube), [`SlotFreeList`](occupancy::SlotFreeList),
//!   [`CubeTilePacking`](occupancy::CubeTilePacking), the
//!   [`ShelfBinPack`](occupancy::ShelfBinPack) rectangle packer, and the
//!   [`SortedKeyBitmaskMap`](occupancy::SortedKeyBitmaskMap).
//! - [`solids`] — the [`CellClassification`](solids::CellClassification) black/white/grey CSG cell
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
pub mod interval;
pub mod occupancy;
pub mod solids;
pub mod spatial;
pub mod srgb;
pub mod supersede;

pub use srgb::{srgb_component_to_linear, srgb_hex_to_linear};
pub use supersede::{catch_unwind_or_log, drain_to_latest, CoalescingWorker, GenerationTracker};
