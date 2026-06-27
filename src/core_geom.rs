//! ADR 0003 bottom layer: dependency-free geometry primitives + the streaming
//! quantum; depends on nothing in the crate.

/// Edge length of a render chunk, in BLOCKS (ADR 0002 Decision 3, part of #19).
/// A chunk therefore spans `CHUNK_BLOCKS * voxels_per_block` voxels per axis
/// (e.g. 4 blocks × density 16 = 64 voxels/axis). Chosen as a small whole-block
/// multiple so a chunk stays a phase-aligned, frustum-cullable unit while the
/// draw-call count stays sane. The resolved grid's occupied voxels are bucketed
/// into these chunks at rebuild time; each frame only the chunks whose world
/// AABB intersects the camera frustum are drawn.
pub const CHUNK_BLOCKS: u32 = 4;
