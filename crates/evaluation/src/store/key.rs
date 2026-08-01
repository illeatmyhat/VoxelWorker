#![allow(clippy::missing_const_for_fn, clippy::must_use_candidate)]

//! The resolve cache's key: a `(chunk_coord, lod)` pair addressing one resolved chunk.

/// The cache key: a chunk coordinate (in `CHUNK_BLOCKS`-cell space) plus its
/// level-of-detail. `lod` is the LOD seam: it is always `0` and is carried so
/// adding a down-sampling LOD level is a behavioral change, not a key-shape change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkCacheKey {
    /// The chunk's integer cell coordinate (see [`Scene::resolve_chunk`](document::scene::Scene::resolve_chunk)).
    pub chunk_coord: [i32; 3],
    /// Level of detail (always `0`).
    pub lod: u32,
}

impl ChunkCacheKey {
    /// A key for `chunk_coord` at the given `lod`.
    pub fn new(chunk_coord: [i32; 3], lod: u32) -> Self {
        Self { chunk_coord, lod }
    }
}
