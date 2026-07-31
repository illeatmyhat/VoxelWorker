//! The boundary-aware **two-layer chunk** representation and the block classifier,
//! proven bit-exact against the dense [`crate::store::Store`] path.
//!
//! The **one evaluator** classifies each BLOCK of a covering chunk **air / coarse-solid
//! / boundary** via the interval bound (`VoxelProducer::cell_field_interval`), then
//! materialises only the boundary
//! blocks per-voxel. A solid interior carries block ids with **NO voxel data** — the
//! whole point of the port (an 800×800-revolve-class solid stops densifying its interior).
//!
//! ## The pieces
//!
//! * [`BlockClassification`] / [`classify_chunk_block`] —
//!   the conservative classifier: compose every leaf's field interval over a block cell by
//!   CSG interval arithmetic (v1 only has [`document::scene::CombineOp::Union`]) and take
//!   the 3-way verdict through the substrate black/white/gray
//!   [`substrate::solids::CellClassification`] kernel. An unboundable producer
//!   (`cell_field_interval == None`) surfaces as "cannot classify" and forces the block
//!   BOUNDARY.
//! * [`TwoLayerChunk`] — the per-chunk store: a coarse per-block [`BlockId`](voxel_core::core_geom::BlockId) grid
//!   (coarse-solid blocks carry their id, no voxels) + a SPARSE map of boundary blocks to
//!   their decomposed [`VoxelBox`](crate::cuboid::VoxelBox) geometry + per-face [`SeamSolidity`] flags.
//! * [`TwoLayerStore::build_chunk`] — runs the evaluator for one chunk behind the
//!   capability flag; [`TwoLayerChunk::expand_occupancy_into`] streams it back to full
//!   occupancy (coarse fast-fill + boundary per-voxel) for the parity gate.
//!
//! This is the SOLE runtime display path. The mesher consumes the layers; export and the
//! diameter query stream cacheless from the evaluator ([`stream_vox_occupancy`] /
//! [`streamed_widest_run_in_band`]); the live display cache is the
//! [`TwoLayerResidentCache`]. The dense `Store::resolve_region` and
//! `resolve_region_two_layer` reach no RUNTIME path and exist only as the test parity +
//! golden reference oracles, compile-gated behind the `oracle` feature.
//!
//! ## Frame — the voxel-frame invariant
//!
//! The coarse grid is **chunk-local integer**: a coarse cell is addressed by its
//! chunk-local block index `[0, CHUNK_BLOCKS)`, and the absolute origin lives in the chunk
//! key ([`crate::store::ChunkCacheKey::chunk_coord`]). The boundary blocks' [`VoxelBox`](crate::cuboid::VoxelBox)es
//! are in **chunk-local voxel** indices `[0, chunk_extent_voxels)`. The expansion stamps
//! voxels into the SAME (recentered / floating-origin-rebased) frame
//! [`Scene::resolve_chunk_rebased`](document::scene::Scene::resolve_chunk_rebased)
//! produces, so the round-trip is occupancy-identical to the dense path.
//!
//! ## Modules
//!
//! * [`chunk`] — the two-layer chunk value ([`TwoLayerChunk`] + [`MicroblockGeometry`] +
//!   [`SeamSolidity`] + [`BlockClassification`]) and its occupancy expansion.
//! * [`classify`] — the interval-bound block classifier + boundary-block per-voxel resolve
//!   + seam-solidity computation.
//! * [`builder`] — the [`TwoLayerStore`] and the per-chunk covering / broadphase /
//!   candidate-leaf helpers and the chunk build.
//! * [`resident_cache`] — the [`TwoLayerResidentCache`] incremental resident set.
//! * [`stream`] — the whole-region + cacheless streams (the dense oracle, export/measure).

mod builder;
mod chunk;
mod classify;
mod field_probe;
mod resident_cache;
mod stream;
#[cfg(test)]
mod tests;

// Internal cross-submodule sharing: every submodule reads its siblings' helpers through
// this glob. It re-exports at `pub(crate)`; the external API is raised to `pub` below.
pub(crate) use builder::*;
pub(crate) use chunk::*;
pub(crate) use classify::*;

// The public boundary-residency surface.
pub use builder::TwoLayerStore;
pub use chunk::{BlockClassification, MicroblockGeometry, SeamSolidity, TwoLayerChunk};
pub use classify::seat_center_at;
pub use field_probe::composed_field_at;
pub use resident_cache::TwoLayerResidentCache;
pub use stream::{stream_vox_occupancy, streamed_widest_run_in_band};

// The dense whole-region resolve oracle is compile-gated out of production builds (see the
// proof chapter's "Oracles" section, `docs/architecture/05-proof.md`).
#[cfg(any(test, feature = "oracle"))]
pub use stream::resolve_region_two_layer;
// A dense test-oracle expander over an already-resident chunk set. It ships
// only in TEST builds; the app crate's tests reach it across the boundary via the
// `test-support` feature (its dev-dependency turns it on), exactly like `oracle`.
#[cfg(any(test, feature = "test-support"))]
pub use stream::expand_resident_chunks_into_grid;
