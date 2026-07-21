//! The TwoLayerStore builder + the per-chunk covering/broadphase/candidate-leaf helpers and the chunk build.

use std::sync::Arc;

use rayon::prelude::*;

use voxel_core::core_geom::CHUNK_BLOCKS;
use document::scene::{LeafProducer, Scene};
use voxel_core::spatial_index::{EditBroadphaseBvh, VoxelAabb};

#[allow(unused_imports)]
use super::*;

/// The capability that builds the [`TwoLayerChunk`] display cache from the one evaluator
/// (ADR 0010 Decision 3 / 6). Every live caller constructs it via [`enabled`](Self::enabled)
/// — E3's mesher, the export/diameter workers, and `shot` — because ADR 0010 E5 landed the
/// two-layer path as the SOLE runtime display path; the dense [`crate::store::Store`] path
/// is retired to a test-and-golden oracle (see the module docs' "Status" section). The
/// `Default`-constructed, disabled instance survives for the tests that pin the
/// off-behaviour ([`build_chunk`](Self::build_chunk) returning `None`).
///
/// It is a thin, stateless builder (no resident cache of its own — every call
/// re-classifies a chunk from the scene; `TwoLayerResidentCache` is the incremental
/// resident cache built on top); a chunk is built on demand from the scene.
#[derive(Debug, Clone, Copy, Default)]
pub struct TwoLayerStore {
    /// The capability flag (ADR 0010 Decision 6). `false` (the default) means the
    /// two-layer path is OFF and [`build_chunk`](Self::build_chunk) returns `None`, so a
    /// caller falls back to the dense path; `true` engages the boundary-aware build.
    enabled: bool,
}

impl TwoLayerStore {
    /// A store with the two-layer capability ENABLED. The default ([`Default`]) is
    /// DISABLED, matching the ADR's "OFF by default, dense fallback" coexistence.
    pub fn enabled() -> Self {
        Self { enabled: true }
    }

    /// Whether the two-layer capability is engaged.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Build the [`TwoLayerChunk`] for EVERY covering chunk of `scene` (ADR 0010 E3): the
    /// `(absolute_chunk_coord, chunk)` list the two-layer mesher
    /// (`CuboidMeshRenderer::new_from_two_layer_chunks`, up in the display layer) consumes,
    /// visited in the SAME z,y,x order the dense store assembles. Returns an empty list when
    /// the capability is OFF or the scene has no covering chunk range (a VoxelBody-only scene —
    /// the caller falls back to the dense path). This keeps the `pub(crate)` chunk-range
    /// logic inside the crate while exposing the covering-chunk build to the `shot` binary.
    pub fn build_covering_chunks(
        &self,
        scene: &Scene,
        voxels_per_block: u32,
        lod: u32,
    ) -> Vec<([i32; 3], Arc<TwoLayerChunk>)> {
        if !self.enabled {
            return Vec::new();
        }
        let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(voxels_per_block) else {
            return Vec::new();
        };
        debug_assert_eq!(lod, 0, "E2 only builds full resolution (lod 0)");
        // #63: HOIST the leaf list out of the per-chunk build — compute it ONCE (re-walking
        // the node tree + cloning every producer per covering chunk was the O(objects²) sink)
        // and share the read-only slice across the parallel build. Then the EDIT BROADPHASE
        // (#66, ADR 0011 Decision 4b): a stateless per-build BVH over the leaf world-AABBs;
        // each chunk queries its own box and is classified against only the overlapping
        // candidates, keeping the build ~O(chunks × (log leaves + candidates)).
        let leaves = scene.leaf_producers(voxels_per_block);
        let broadphase = leaf_edit_broadphase(&leaves, voxels_per_block);
        // Each covering chunk is built independently from the (read-only, `Sync`) leaf slice
        // + broadphase, so the wholesale build is embarrassingly parallel (#57). Enumerate
        // the coords in the SAME z,y,x order the dense store assembles (X fastest), then map
        // each coord → its chunk with rayon. A parallel `.collect()` PRESERVES ordering, so
        // the output Vec is byte-identical to the serial nested loop regardless of thread
        // count.
        let coords = enumerate_covering_chunk_coords(min_chunk, max_chunk);
        coords
            .into_par_iter()
            .map(|coord| {
                let candidates =
                    chunk_candidate_leaves(&broadphase, &leaves, coord, voxels_per_block);
                let chunk =
                    build_two_layer_chunk_from_leaves(coord, &candidates, voxels_per_block);
                // `Arc`-wrap the freshly built chunk so this owned covering set can be
                // handed to the mesh / brick / fog readers, retained in the mesh
                // renderer, AND moved into the async `GeometryRebuildRequest` with only
                // O(1) refcount bumps — never a deep `TwoLayerChunk` copy (the per-edit
                // clone this cleanup killed; ADR 0011 G3 record/atlas territory).
                (coord, Arc::new(chunk))
            })
            .collect()
    }

    /// Build the [`TwoLayerChunk`] for `chunk_coord` from the scene's one evaluator, or
    /// `None` when the capability is OFF (the caller then uses the dense path). The
    /// returned chunk is in chunk-local frame; [`TwoLayerChunk::expand_occupancy_into`]
    /// rebases it to match the dense store.
    ///
    /// `lod` is the parked LOD seam (always `0`), kept for call-site symmetry with the
    /// dense store.
    pub fn build_chunk(
        &self,
        chunk_coord: [i32; 3],
        scene: &Scene,
        voxels_per_block: u32,
        lod: u32,
    ) -> Option<TwoLayerChunk> {
        debug_assert_eq!(lod, 0, "E2 only builds full resolution (lod 0)");
        if !self.enabled {
            return None;
        }
        Some(build_two_layer_chunk(chunk_coord, scene, voxels_per_block))
    }
}


/// Enumerate every covering chunk coord in the inclusive `[min_chunk, max_chunk]` range,
/// in the SAME z,y,x order (X fastest, then Y, then Z) the dense store assembles them. This
/// materialises the coords into a `Vec` so the wholesale build (#57) can `into_par_iter()`
/// them and `.collect()` back into an identically-ordered result.
pub(crate) fn enumerate_covering_chunk_coords(min_chunk: [i32; 3], max_chunk: [i32; 3]) -> Vec<[i32; 3]> {
    let mut coords = Vec::new();
    for chunk_z in min_chunk[2]..=max_chunk[2] {
        for chunk_y in min_chunk[1]..=max_chunk[1] {
            for chunk_x in min_chunk[0]..=max_chunk[0] {
                coords.push([chunk_x, chunk_y, chunk_z]);
            }
        }
    }
    coords
}

/// The leaf's world-AABB in absolute voxels, corner-anchored — the SAME box
/// [`classify_chunk_block`] / [`resolve_boundary_block`] test each block against. A
/// region-spanning VoxelBody (the cloud field) reports its composite-region `full_dimensions`, so
/// its box correctly spans every chunk it fills.
///
/// ADR 0027: an oriented leaf's world extent must enclose its **continuously rotated** grid, not
/// just the discrete lattice turn — so this delegates to [`leaf_world_box`](super::classify::leaf_world_box),
/// the ONE rotation-aware extent the classifier folds through. (It formerly used the lattice
/// `orientation.turn_extent`, which is blind to the ADR 0027 quaternion: a leaf with an identity
/// lattice orientation but a non-identity continuous rotation — a tube seated on a curved surface —
/// reserved an UPRIGHT box, so the edit broadphase dropped it from every chunk its tilted body
/// occupied beyond that box, truncating the tube. The two must agree box-for-box, and now do.)
pub(crate) fn leaf_world_aabb(leaf: &LeafProducer, voxels_per_block: u32) -> VoxelAabb {
    super::classify::leaf_world_box(leaf, voxels_per_block)
}

/// **The edit broadphase over a scene's leaves (#66, ADR 0011 Decision 4b).** Build the
/// stateless per-build [`EditBroadphaseBvh`] over every leaf's world AABB, indexed by the
/// leaf's position in `leaves` (document order). Rebuilt from scratch on every wholesale
/// build / edit — never persisted across edits (no invalidation obligation, the C1 lesson).
pub(crate) fn leaf_edit_broadphase(leaves: &[LeafProducer], voxels_per_block: u32) -> EditBroadphaseBvh {
    let leaf_aabbs: Vec<VoxelAabb> = leaves
        .iter()
        .map(|leaf| leaf_world_aabb(leaf, voxels_per_block))
        .collect();
    EditBroadphaseBvh::build(&leaf_aabbs)
}

/// The half-open absolute-voxel box of the chunk at `chunk_coord` — the query box the edit
/// broadphase answers per covering chunk.
pub(crate) fn chunk_world_voxel_aabb(chunk_coord: [i32; 3], voxels_per_block: u32) -> VoxelAabb {
    let chunk_extent_voxels = (CHUNK_BLOCKS * voxels_per_block.max(1)) as i64;
    let min = [
        chunk_coord[0] as i64 * chunk_extent_voxels,
        chunk_coord[1] as i64 * chunk_extent_voxels,
        chunk_coord[2] as i64 * chunk_extent_voxels,
    ];
    VoxelAabb::new(
        min,
        [
            min[0] + chunk_extent_voxels,
            min[1] + chunk_extent_voxels,
            min[2] + chunk_extent_voxels,
        ],
    )
}

/// Resolve one chunk's broadphase candidates to borrows of the shared leaf slice (no clone
/// — the classifier reads them read-only). The BVH returns indices sorted ascending, i.e. a
/// document-order subsequence of `leaves` (a filter, never a reorder), so later-wins Union
/// material resolution is unchanged.
///
/// EXACTNESS: a Union/Subtract leaf whose AABB does not overlap the chunk cannot affect any
/// block in it, so a chunk classified against only its overlapping candidates is
/// byte-identical to one classified against all leaves (the per-block AABB tests inside the
/// classifier already narrow further per block — the broadphase just hands them a smaller
/// exact-superset set). An INTERSECT-influence leaf (ADR 0017 #75,
/// [`LeafProducer::masks_beyond_bounds`]) breaks that argument — its mask kills cells
/// anywhere OUTSIDE its box — so every such leaf is kept in EVERY chunk's candidate set
/// regardless of overlap (merged back in at its document-order position; the per-block
/// filters downstream apply the same keep rule).
pub(crate) fn chunk_candidate_leaves<'leaf_slice>(
    broadphase: &EditBroadphaseBvh,
    leaves: &'leaf_slice [LeafProducer],
    chunk_coord: [i32; 3],
    voxels_per_block: u32,
) -> Vec<&'leaf_slice LeafProducer> {
    let mut include = vec![false; leaves.len()];
    for leaf_index in
        broadphase.overlapping_input_indices(&chunk_world_voxel_aabb(chunk_coord, voxels_per_block))
    {
        include[leaf_index] = true;
    }
    for (leaf_index, leaf) in leaves.iter().enumerate() {
        if leaf.masks_beyond_bounds() {
            include[leaf_index] = true;
        }
    }
    leaves
        .iter()
        .enumerate()
        .filter(|(leaf_index, _)| include[*leaf_index])
        .map(|(_, leaf)| leaf)
        .collect()
}

/// Build one chunk's two-layer representation by classifying every block and resolving the
/// boundary blocks per-voxel (the evaluator → display-cache step, ADR 0010 Decision 3).
///
/// Stateless single-chunk entry (tests, incremental single-chunk rebuild): it computes the
/// scene's leaf list itself, then delegates to [`build_two_layer_chunk_from_leaves`]. The
/// BULK paths ([`TwoLayerStore::build_covering_chunks`],
/// [`TwoLayerResidentCache::resident_two_layer_chunks`]) hoist `leaf_producers` out of the
/// per-chunk loop (#63) and pass a pre-filtered candidate slice into
/// [`build_two_layer_chunk_from_leaves`] directly — so the O(chunks) tree-walk + producer
/// clone never happens per chunk there.
pub(crate) fn build_two_layer_chunk(
    chunk_coord: [i32; 3],
    scene: &Scene,
    voxels_per_block: u32,
) -> TwoLayerChunk {
    let leaves = scene.leaf_producers(voxels_per_block);
    let candidates: Vec<&LeafProducer> = leaves.iter().collect();
    build_two_layer_chunk_from_leaves(chunk_coord, &candidates, voxels_per_block)
}

/// Build one chunk's two-layer representation from a pre-computed leaf candidate slice — the
/// hoisted core of [`build_two_layer_chunk`] (#63).
///
/// `leaves` MUST be a document-order subsequence of `scene.leaf_producers(voxels_per_block)`
/// (a filter, never a reorder) that INCLUDES every leaf whose world AABB overlaps this chunk
/// AND every Intersect-influence leaf regardless of overlap (ADR 0017 #75,
/// [`LeafProducer::masks_beyond_bounds`] — a mask affects cells outside its own box). The
/// edit broadphase ([`chunk_candidate_leaves`]) guarantees exactly that: a Union/Subtract
/// leaf whose AABB does NOT overlap the chunk cannot affect ANY block in it (the per-block
/// AABB tests inside [`classify_chunk_block`] / [`resolve_boundary_block`] would skip it
/// regardless), so passing only these candidates yields IDENTICAL coarse / microblock / seam
/// output while preserving later-wins Union material resolution (document order kept).
pub(crate) fn build_two_layer_chunk_from_leaves(
    chunk_coord: [i32; 3],
    leaves: &[&LeafProducer],
    voxels_per_block: u32,
) -> TwoLayerChunk {
    let density = voxels_per_block.max(1);
    let chunk_extent_voxels = (CHUNK_BLOCKS * density) as i64;
    let chunk_min_voxels = [
        chunk_coord[0] as i64 * chunk_extent_voxels,
        chunk_coord[1] as i64 * chunk_extent_voxels,
        chunk_coord[2] as i64 * chunk_extent_voxels,
    ];

    // CHUNK-GRANULAR INTERVAL FAST PATH (ADR 0010 Decision 2): decide the whole chunk from
    // ONE composed interval at the chunk cell. A solid interior chunk is 1 interval call
    // instead of 64 per-block calls — the O(volume) → O(surface) win for large solids.
    // Only a verdict that PROVABLY implies the identical per-block outcome short-circuits;
    // any ambiguity falls back to the byte-identical per-block sweep below.
    let chunk_abs = VoxelAabb::new(
        chunk_min_voxels,
        [
            chunk_min_voxels[0] + chunk_extent_voxels,
            chunk_min_voxels[1] + chunk_extent_voxels,
            chunk_min_voxels[2] + chunk_extent_voxels,
        ],
    );
    match classify_whole_chunk(leaves, chunk_abs, voxels_per_block) {
        WholeChunkVerdict::AllAir => return TwoLayerChunk::empty(density),
        WholeChunkVerdict::AllCoarse { block_id, overlay } => {
            let mut chunk = TwoLayerChunk::empty(density);
            for flat in 0..chunk.coarse.len() {
                chunk.coarse[flat] = Some(block_id);
                chunk.coarse_overlay[flat] = overlay;
            }
            return chunk;
        }
        WholeChunkVerdict::PerBlock => {}
    }

    build_two_layer_chunk_per_block(chunk_min_voxels, leaves, density, voxels_per_block)
}

/// The per-block classify sweep — the always-correct fallback of
/// [`build_two_layer_chunk_from_leaves`] (and the parity oracle the fast-path test compares
/// against). Classifies every one of the chunk's `CHUNK_BLOCKS³` blocks independently and
/// resolves boundary blocks per-voxel. `chunk_min_voxels` is the chunk's low corner in
/// absolute voxels.
pub(crate) fn build_two_layer_chunk_per_block(
    chunk_min_voxels: [i64; 3],
    leaves: &[&LeafProducer],
    density: u32,
    voxels_per_block: u32,
) -> TwoLayerChunk {
    let block_extent = density as i64;

    let mut chunk = TwoLayerChunk::empty(density);

    for block_z in 0..CHUNK_BLOCKS {
        for block_y in 0..CHUNK_BLOCKS {
            for block_x in 0..CHUNK_BLOCKS {
                let block = [block_x, block_y, block_z];
                // The block's half-open box in the SCENE's absolute voxel frame.
                let block_min = [
                    chunk_min_voxels[0] + block_x as i64 * block_extent,
                    chunk_min_voxels[1] + block_y as i64 * block_extent,
                    chunk_min_voxels[2] + block_z as i64 * block_extent,
                ];
                let block_abs = VoxelAabb::new(
                    block_min,
                    [
                        block_min[0] + block_extent,
                        block_min[1] + block_extent,
                        block_min[2] + block_extent,
                    ],
                );

                match classify_chunk_block(leaves, block_abs, voxels_per_block) {
                    BlockClassification::Air => {}
                    BlockClassification::CoarseSolid(block_id) => {
                        let flat = coarse_flat_index(block);
                        chunk.coarse[flat] = Some(block_id);
                        // A coarse-solid block is owned by EXACTLY ONE leaf (the classifier
                        // forces multi-leaf overlaps to boundary), so its on-face-grid
                        // overlay (ADR 0003 §3c) is that single leaf's `grid_overlay`.
                        chunk.coarse_overlay[flat] =
                            single_overlapping_leaf_overlay(leaves, block_abs, voxels_per_block);
                    }
                    BlockClassification::Boundary => {
                        let geometry =
                            resolve_boundary_block(leaves, block_min, density, voxels_per_block);
                        // A boundary verdict the per-voxel pass found EMPTY contributes
                        // nothing (still exact — the interval was merely conservative).
                        if !geometry.cuboids.is_empty() {
                            chunk.microblocks.insert(block, geometry);
                        }
                    }
                }
            }
        }
    }

    chunk
}

