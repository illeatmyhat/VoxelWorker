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
//! scientific literature *is* the type's name (`MedianSplitBvh`, `IntegerAabb`,
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
//! ---
//!
//! Components arrive one extraction slice at a time, each carrying its own oracles,
//! per the extraction map referenced above. Extracted so far: slice S1 (spatial) —
//! [`Aabb`], [`Bvh`], and the [`lattice_key`] packing codec; slice S2 (intervals +
//! rational) — [`FieldInterval`], [`DisjointIntervalSet`], and [`Rational`]; slice S3
//! (decomposition) — [`GreedyCuboidDecomposition`] over a [`CellGrid`] into [`Cuboid`]s;
//! slice S4 (concurrency) — the [`supersede`] protocol: [`CoalescingWorker`],
//! [`GenerationTracker`], and their [`drain_to_latest`] / [`catch_unwind_or_log`] helpers;
//! slice S5 (bit/atlas kit) — [`BitCube`], [`SlotFreeList`], and [`CubeTilePacking`]; slice S7
//! (first kernel-only tier-3 extraction) — the [`SparseMinMipPyramid`] fold (the pure core of the
//! domain's clip-map builders, whose chunk traversal stays in the app crate); slice S8 (second
//! kernel-only tier-3 extraction) — the [`SortedKeyBitmaskMap`] sorted parallel-array map (the
//! storage shape of the domain's block-occupancy masks, whose `from_chunks` builder stays domain).

pub mod aabb;
pub mod bit_cube;
pub mod bitmask_map;
pub mod bvh;
pub mod cube_packing;
pub mod disjoint_interval_set;
pub mod field_interval;
pub mod free_list;
pub mod greedy_cuboid_decomposition;
pub mod lattice_key;
pub mod min_mip_pyramid;
pub mod rational;
pub mod supersede;

pub use aabb::Aabb;
pub use bit_cube::BitCube;
pub use bitmask_map::{mask_bit_is_set, set_mask_bit, SortedKeyBitmaskMap};
pub use bvh::Bvh;
pub use cube_packing::CubeTilePacking;
pub use disjoint_interval_set::DisjointIntervalSet;
pub use field_interval::{union_field_intervals, FieldClassification, FieldInterval};
pub use free_list::SlotFreeList;
pub use greedy_cuboid_decomposition::{CellGrid, Cuboid, GreedyCuboidDecomposition};
pub use min_mip_pyramid::{MinMipLevel, SparseMinMipPyramid};
pub use rational::Rational;
pub use supersede::{catch_unwind_or_log, drain_to_latest, CoalescingWorker, GenerationTracker};
