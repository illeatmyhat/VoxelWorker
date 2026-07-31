//! The store — a per-chunk resolve cache keyed by
//! `(chunk_coord, lod)` that resolves a chunk **on demand** (lazily) and stores the
//! result, so a second request for the same chunk is a map lookup instead of a
//! re-resolve.
//!
//! ## What the store IS (and what it is NOT)
//!
//! * **Lazy per-chunk resolve + cache.** [`Store::chunk`] returns the cached per-chunk
//!   [`VoxelGrid`](voxel_core::voxel::VoxelGrid) (in **absolute** composite voxel
//!   coordinates, exactly as `Scene::resolve_chunk` produces), resolving + storing it on
//!   a miss.
//! * **Per-chunk voxel bound.** The whole-region `MAX_GRID_VOXELS` guard is a *per-chunk*
//!   bound (a single chunk can't exceed it), so a scene whose TOTAL voxel count is far
//!   beyond the 6M whole-region ceiling resolves fine as long as every chunk is small.
//!   See [`voxel_core::voxel::MAX_CHUNK_VOXELS`].
//! * **Identical render output.** The dense whole-region `Store::resolve_region` oracle
//!   rebuilds the SAME recentered monolithic grid the mesher/exporter consume — but
//!   assembled from cached chunks. It is compile-gated behind the `oracle` feature.
//!
//! **Smart invalidation** sits on this seam: [`Store::invalidate_aabb`] evicts exactly
//! the chunks an edit's world-AABB intersects, at whole-chunk dirty granularity. The edit
//! AABB is computed by
//! [`LeafSpatialIndex::edit_aabb_since`](voxel_core::spatial_index::LeafSpatialIndex::edit_aabb_since)
//! (diffing the scene's leaf spatial index before vs after the edit); [`Store::clear`]
//! remains the fallback for edits that can't be localized (a density change or a
//! region-spanning VoxelBody edit).
//!
//! ## Modules
//!
//! * [`key`] — the [`ChunkCacheKey`] `(chunk_coord, lod)` addressing.
//! * [`cache`] — the [`Store`] itself: residency, resolve, invalidation, out-of-core spill.
//! * [`rebuild_plan`] — the pure GPU-free incremental rebuild planner
//!   ([`incremental_rebuild_plan`] → [`IncrementalRebuildPlan`]).

mod cache;
mod key;
mod rebuild_plan;
#[cfg(test)]
mod tests;

pub use cache::Store;
pub use key::ChunkCacheKey;
pub use rebuild_plan::{incremental_rebuild_plan, IncrementalRebuildPlan};

/// Back-compat alias for the pre-A2b name. Call sites refer to the store as
/// `ChunkResolveCache`; it is the same type as [`Store`].
pub type ChunkResolveCache = Store;
