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
//! ## Components
//!
//! Each is a self-contained module with its own literature citations and oracles:
//! [`Aabb`], [`Bvh`], and the [`lattice_key`] packing codec (spatial); [`FieldInterval`],
//! [`DisjointIntervalSet`], and [`Rational`] (interval + rational arithmetic);
//! [`GreedyCuboidDecomposition`] over a [`CellGrid`] into [`Cuboid`]s (box decomposition);
//! the [`supersede`] protocol — [`CoalescingWorker`], [`GenerationTracker`], and their
//! [`drain_to_latest`] / [`catch_unwind_or_log`] helpers (concurrency); [`BitCube`],
//! [`SlotFreeList`], [`CubeTilePacking`], and the [`ShelfBinPack`] rectangle packer (bit/atlas
//! kit); the [`SparseMinMipPyramid`] occupancy fold; the [`SortedKeyBitmaskMap`] sorted
//! parallel-array map; the [`CellClassification`] black/white/grey CSG cell classifier; the
//! [`CulledBoxMeshing`] exposed-face determination; the [`Ray`] primitive with its slab-method
//! ray–box test; and the [`srgb`] transfer-function codec. See the extraction map referenced
//! above for each component's provenance and the domain adapter that wraps it.

pub mod aabb;
pub mod bit_cube;
pub mod bitmask_map;
pub mod bvh;
pub mod cell_classification;
pub mod cube_packing;
pub mod culled_box_meshing;
pub mod disjoint_interval_set;
pub mod field_interval;
pub mod free_list;
pub mod greedy_cuboid_decomposition;
pub mod lattice_key;
pub mod min_mip_pyramid;
pub mod ray;
pub mod rational;
pub mod shelf_bin_pack;
pub mod srgb;
pub mod supersede;

pub use aabb::Aabb;
pub use bit_cube::BitCube;
pub use bitmask_map::{mask_bit_is_set, set_mask_bit, SortedKeyBitmaskMap};
pub use bvh::Bvh;
pub use cell_classification::{CellClassification, CellCombineOp, CellContribution};
pub use cube_packing::CubeTilePacking;
pub use culled_box_meshing::CulledBoxMeshing;
pub use disjoint_interval_set::DisjointIntervalSet;
pub use field_interval::{union_field_intervals, FieldClassification, FieldInterval};
pub use free_list::SlotFreeList;
pub use greedy_cuboid_decomposition::{CellGrid, Cuboid, GreedyCuboidDecomposition};
pub use min_mip_pyramid::{MinMipLevel, SparseMinMipPyramid};
pub use ray::{Ray, RayBoxIntersection};
pub use rational::Rational;
pub use shelf_bin_pack::{
    NormalizedTileRect, PackedTilePlacement, ShelfBinPack, TileImage, TileSize,
};
pub use srgb::{srgb_component_to_linear, srgb_hex_to_linear};
pub use supersede::{catch_unwind_or_log, drain_to_latest, CoalescingWorker, GenerationTracker};
