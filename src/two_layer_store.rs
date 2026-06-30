//! ADR 0010 E2 — the boundary-aware **two-layer chunk** representation + the block
//! classifier, an **OFF-by-default** capability proven bit-exact against the dense
//! [`crate::store::Store`] path (CONTEXT.md "Boundary residency").
//!
//! This is the heart of the ADR 0009 → ADR 0010 port: the **one evaluator** classifies
//! each BLOCK of a covering chunk **air / coarse-solid / boundary** via the E1 interval
//! bound ([`VoxelProducer::cell_field_interval`]), then materialises only the boundary
//! blocks per-voxel. A solid interior carries block ids with **NO voxel data** — the
//! whole point of the port (an 800×800-revolve-class solid stops densifying its interior).
//!
//! ## What this slice IS
//!
//! * [`BlockClassification`] / [`classify_chunk_block`] — the conservative classifier:
//!   compose every leaf's field interval over a block cell by CSG interval arithmetic
//!   (v1 only has [`crate::scene::CombineOp::Union`], so the composition is
//!   [`crate::voxel::union_field_intervals`]), then [`FieldInterval::classify`]. An
//!   unboundable producer (`cell_field_interval == None`) forces the block BOUNDARY.
//! * [`TwoLayerChunk`] — the per-chunk store: a coarse per-block [`BlockId`] grid
//!   (coarse-solid blocks carry their id, no voxels) + a SPARSE map of boundary blocks
//!   to their decomposed [`VoxelBox`] geometry + per-face [`SeamSolidity`] flags.
//! * [`TwoLayerStore::build_chunk`] — runs the evaluator for one chunk behind the
//!   capability flag; [`TwoLayerChunk::expand_occupancy_into`] streams it back to full
//!   occupancy (coarse fast-fill + boundary per-voxel) for the parity gate + as a
//!   transition shim.
//!
//! ## What this slice is NOT (deferred — ADR 0010 E3/E4/E5)
//!
//! * **The mesher does not consume the layers yet** (E3 / #50): the seam-solidity flags
//!   are populated + unit-tested here, but the cuboid mesher still runs on the dense path.
//! * **Export / the diameter query are not repointed** (E4 / #51): they still read the
//!   dense [`Store::resolve_region`](crate::store::Store::resolve_region).
//! * **The dense `resolve_region` is NOT deleted** (E5): it stays the live default. This
//!   capability is OFF by default and coexists exactly as the GPU fog coexists with the
//!   CPU fallback.
//!
//! ## Frame (ADR 0008 — the voxel-frame invariant)
//!
//! The coarse grid is **chunk-local integer**: a coarse cell is addressed by its
//! chunk-local block index `[0, CHUNK_BLOCKS)`, and the absolute origin lives in the
//! chunk key ([`crate::store::ChunkCacheKey::chunk_coord`]). The boundary blocks'
//! [`VoxelBox`]es are in **chunk-local voxel** indices `[0, chunk_extent_voxels)`. The
//! expansion stamps voxels into the SAME (recentred / floating-origin-rebased) frame
//! [`Scene::resolve_chunk_rebased`](crate::scene::Scene::resolve_chunk_rebased) produces,
//! so the round-trip is occupancy-identical to the dense path.

use std::collections::BTreeMap;

use crate::core_geom::{BlockAttrs, BlockId, CHUNK_BLOCKS};
use crate::cuboid::{decompose_into_boxes, VoxelBox, VoxelRegion};
use crate::scene::{LeafProducer, Scene};
use crate::spatial_index::VoxelAabb;
use crate::voxel::{
    union_field_intervals, FieldClassification, Voxel, VoxelGrid, SURFACE_ISOLEVEL,
};

/// The coarse verdict for a single BLOCK of a chunk (ADR 0010 Decision 2). Distinct from
/// [`FieldClassification`] (the per-producer interval verdict) because the BLOCK verdict
/// is the COMPOSED result over the whole op-stack plus the sculpt-touched override:
/// any block a sculpt delta touches is forced [`BlockClassification::Boundary`], and an
/// unboundable op (`cell_field_interval == None`) collapses the block to boundary too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockClassification {
    /// Every voxel in the block is empty — the block holds nothing (no coarse id, no
    /// microblock entry).
    Air,
    /// Every voxel in the block is occupied by a single material — the block lives in
    /// the COARSE layer as one [`BlockId`], with NO per-voxel data (interior elision).
    CoarseSolid(BlockId),
    /// The block straddles the surface (or an op could not bound it, or a sculpt delta
    /// touched it): it lives in the MICROBLOCK layer, resolved per-voxel and decomposed
    /// to cuboids. Always the SAFE verdict.
    Boundary,
}

/// Per-face solidity flags for a boundary block — the coarse/microblock analogue of the
/// dense-fog **apron** (CONTEXT.md "Seam solidity"; VS `sideAlmostSolid`). Each face flag
/// is `true` iff that whole face of the block is solid (every voxel of the `density²`
/// face cells is occupied), so E3's mesher can cull a seam face against a fully-solid
/// neighbour face without expanding the neighbour's voxels.
///
/// Faces are indexed by `(axis, side)`: axis 0/1/2 = X/Y/Z (Z-up), side
/// 0 = the LOW face (`coord == 0`), side 1 = the HIGH face (`coord == density - 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SeamSolidity {
    /// `[axis][side]`: `solid[axis][0]` is the low face, `solid[axis][1]` the high face.
    pub solid: [[bool; 2]; 3],
}

impl SeamSolidity {
    /// Whether the face on `axis` (0/1/2 = X/Y/Z), `side` (0 = low, 1 = high), is solid.
    pub fn face_is_solid(&self, axis: usize, side: usize) -> bool {
        self.solid[axis][side]
    }
}

/// The geometry of one boundary block (CONTEXT.md "Microblock layer"): its sub-block
/// voxels already decomposed to cuboids, plus its per-face seam-solidity flags.
///
/// The [`VoxelBox`]es are in **block-local voxel** indices `[0, density)` per axis (the
/// frame [`decompose_into_boxes`] yields over a `density³` [`VoxelRegion`]); the owning
/// block's chunk-local position is the map key in [`TwoLayerChunk::microblocks`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroblockGeometry {
    /// The block's solid voxels, greedy-decomposed to single-material cuboids
    /// (block-local voxel indices `[0, density)`). Empty when the block resolved to no
    /// voxels (a boundary verdict the per-voxel pass found empty — still exact).
    ///
    /// Each cuboid's `material_id` is the cuboid mesher's **render-cell key** (ADR 0003
    /// §3c): the clean categorical `block_id` in the low bits, the transient on-face-grid
    /// overlay marker in [`crate::cuboid_mesh::MESH_GRID_OVERLAY_BIT`]. The decomposition
    /// therefore splits a box across differing overlay states exactly like the dense
    /// mesher, and E3's mesher reads the box's clean id + overlay back out of this key
    /// without the render flag ever entering the categorical cell. Consumers that want the
    /// clean id (E2 occupancy expansion) mask the overlay bit off.
    pub cuboids: Vec<VoxelBox>,
    /// Per-face solidity flags (the seam apron analogue) for this block.
    pub seam_solidity: SeamSolidity,
}

/// The boundary-aware two-layer representation of ONE chunk (ADR 0010 Decision 1).
///
/// * `coarse` — a per-BLOCK [`Option<BlockId>`] grid over the chunk's `CHUNK_BLOCKS³`
///   blocks (chunk-local integer, ADR 0008). `Some(id)` is a coarse-solid block (id, no
///   voxels); `None` is air OR a boundary block (boundary geometry lives in `microblocks`).
/// * `microblocks` — a SPARSE map of boundary blocks (keyed by chunk-local block index)
///   to their [`MicroblockGeometry`]. Only surface blocks appear.
///
/// A coarse-solid block stores ZERO voxels (interior elision); only boundary blocks carry
/// per-voxel geometry, and even those are stored as cuboids, never a dense `density³` grid.
#[derive(Debug, Clone, Default)]
pub struct TwoLayerChunk {
    /// The chunk's voxels-per-block density (chunk extent in voxels is
    /// `CHUNK_BLOCKS * density`). Carried so the expansion / interior count can size the
    /// per-block fill without re-deriving it.
    pub voxels_per_block: u32,
    /// Per-block coarse layer, row-major over the chunk's `CHUNK_BLOCKS³` blocks
    /// (X fastest, then Y, then Z): `Some(id)` = coarse-solid, `None` = air or boundary.
    pub coarse: Vec<Option<BlockId>>,
    /// Per-block on-face-grid overlay marker (ADR 0003 §3c), parallel to `coarse`: `true`
    /// iff the coarse-solid block's single owning leaf had `grid_overlay` set. Meaningful
    /// only where `coarse[i].is_some()`; the E3 mesher reads it to flag a coarse block's
    /// one-box draw. A RENDER hint only — never part of the categorical id / occupancy.
    pub coarse_overlay: Vec<bool>,
    /// Sparse boundary-block geometry, keyed by chunk-local block index `[bx, by, bz]`
    /// (each component `< CHUNK_BLOCKS`).
    pub microblocks: BTreeMap<[u32; 3], MicroblockGeometry>,
}

/// Chunk-local block index → flat row-major index over `CHUNK_BLOCKS³` (X fastest).
#[inline]
fn coarse_flat_index(block: [u32; 3]) -> usize {
    let n = CHUNK_BLOCKS as usize;
    (block[2] as usize * n + block[1] as usize) * n + block[0] as usize
}

impl TwoLayerChunk {
    /// An all-air chunk at `voxels_per_block` (no coarse ids, no microblocks).
    fn empty(voxels_per_block: u32) -> Self {
        let block_count = (CHUNK_BLOCKS as usize).pow(3);
        Self {
            voxels_per_block,
            coarse: vec![None; block_count],
            coarse_overlay: vec![false; block_count],
            microblocks: BTreeMap::new(),
        }
    }

    /// The coarse-layer id at chunk-local block index `block` (`Some` only for a
    /// coarse-solid block; `None` for air or boundary).
    pub fn coarse_block(&self, block: [u32; 3]) -> Option<BlockId> {
        self.coarse[coarse_flat_index(block)]
    }

    /// The on-face-grid overlay marker (ADR 0003 §3c) of the coarse-solid block at
    /// `block` — `true` iff that block's owning leaf had `grid_overlay` set. Only
    /// meaningful when [`coarse_block`](Self::coarse_block) is `Some`.
    pub fn coarse_block_overlay(&self, block: [u32; 3]) -> bool {
        self.coarse_overlay[coarse_flat_index(block)]
    }

    /// The TOTAL voxel count this chunk STORES — the sum of every boundary block's
    /// decomposed cuboid voxels. A coarse-solid block contributes ZERO (interior
    /// elision); this is the measured "surface-only" residency the ADR demands (an
    /// 800×800 revolve's interior holds no voxels here, unlike the dense path).
    pub fn stored_voxel_count(&self) -> u64 {
        self.microblocks
            .values()
            .flat_map(|geometry| geometry.cuboids.iter())
            .map(VoxelBox::voxel_count)
            .sum()
    }

    /// Stream this chunk back to full occupancy into `output` (ADR 0010 Decision 3 /
    /// parity gate (a)): a coarse-solid block is a fast `density³` fill at its block id;
    /// a boundary block expands its cuboids per-voxel. Each voxel is stamped at its
    /// CHUNK-LOCAL voxel index (the chunk's own `[0, chunk_extent_voxels)` frame) +
    /// `index_offset`, so the caller can rebase the whole chunk into the recentred /
    /// floating-origin frame in one integer add (mirroring
    /// [`Scene::resolve_chunk_rebased`](crate::scene::Scene::resolve_chunk_rebased)).
    ///
    /// `index_offset` is added (in i64, before the i32 downcast) to every emitted voxel
    /// index: pass `chunk_min_voxels − floating_origin_voxels` to land the chunk in the
    /// exact frame the dense store assembles, or `[0,0,0]` for chunk-local indices.
    fn expand_occupancy_into(&self, output: &mut Vec<Voxel>, index_offset: [i64; 3]) {
        let density = self.voxels_per_block.max(1);
        let block_extent = density as i64;

        for block_z in 0..CHUNK_BLOCKS {
            for block_y in 0..CHUNK_BLOCKS {
                for block_x in 0..CHUNK_BLOCKS {
                    let block = [block_x, block_y, block_z];
                    let block_low_voxels = [
                        block_x as i64 * block_extent,
                        block_y as i64 * block_extent,
                        block_z as i64 * block_extent,
                    ];
                    if let Some(block_id) = self.coarse_block(block) {
                        // Coarse-solid: fast d³ fill at the single block id.
                        Self::fill_solid_block(
                            output,
                            block_low_voxels,
                            density,
                            block_id,
                            index_offset,
                        );
                    } else if let Some(geometry) = self.microblocks.get(&block) {
                        // Boundary: expand each cuboid per-voxel at its material id.
                        Self::expand_boundary_block(
                            output,
                            block_low_voxels,
                            geometry,
                            index_offset,
                        );
                    }
                    // else: air block, nothing to emit.
                }
            }
        }
    }

    /// Fast-fill a coarse-solid block: every `density³` voxel at `block_id`.
    fn fill_solid_block(
        output: &mut Vec<Voxel>,
        block_low_voxels: [i64; 3],
        density: u32,
        block_id: BlockId,
        index_offset: [i64; 3],
    ) {
        for voxel_z in 0..density {
            for voxel_y in 0..density {
                for voxel_x in 0..density {
                    let chunk_local = [
                        block_low_voxels[0] + voxel_x as i64,
                        block_low_voxels[1] + voxel_y as i64,
                        block_low_voxels[2] + voxel_z as i64,
                    ];
                    output.push(stamped_voxel(
                        chunk_local,
                        [voxel_x as u8, voxel_y as u8, voxel_z as u8],
                        block_id,
                        index_offset,
                    ));
                }
            }
        }
    }

    /// Expand a boundary block's cuboids back to per-voxel occupancy at their material.
    fn expand_boundary_block(
        output: &mut Vec<Voxel>,
        block_low_voxels: [i64; 3],
        geometry: &MicroblockGeometry,
        index_offset: [i64; 3],
    ) {
        for cuboid in &geometry.cuboids {
            // The cuboid's `material_id` is the render-cell key (block_id | overlay<<15);
            // occupancy is the CLEAN categorical id, so mask the overlay bit off (ADR
            // 0003 §3c — the overlay never enters the occupancy / categorical cell).
            let block_id = BlockId(crate::cuboid_mesh::clean_block_id(cuboid.material_id));
            for voxel_z in cuboid.min[2]..=cuboid.max[2] {
                for voxel_y in cuboid.min[1]..=cuboid.max[1] {
                    for voxel_x in cuboid.min[0]..=cuboid.max[0] {
                        let chunk_local = [
                            block_low_voxels[0] + voxel_x as i64,
                            block_low_voxels[1] + voxel_y as i64,
                            block_low_voxels[2] + voxel_z as i64,
                        ];
                        output.push(stamped_voxel(
                            chunk_local,
                            [voxel_x as u8, voxel_y as u8, voxel_z as u8],
                            block_id,
                            index_offset,
                        ));
                    }
                }
            }
        }
    }
}

/// Build one [`Voxel`] at chunk-local voxel index `chunk_local + index_offset` (i64 add
/// before the i32 downcast, ADR 0008), with `block_local_coord` and `block_id`. The
/// `grid_overlay` render marker is `false` (E3 wires the two-layer mesher; this slice
/// never feeds the renderer, so the parity gate compares the categorical id only).
#[inline]
fn stamped_voxel(
    chunk_local: [i64; 3],
    block_local_coord: [u8; 3],
    block_id: BlockId,
    index_offset: [i64; 3],
) -> Voxel {
    Voxel {
        local_index: [
            (chunk_local[0] + index_offset[0]) as i32,
            (chunk_local[1] + index_offset[1]) as i32,
            (chunk_local[2] + index_offset[2]) as i32,
        ],
        block_local_coord,
        block_id,
        attrs: BlockAttrs::DEFAULT,
        grid_overlay: false,
    }
}

/// Classify ONE block (the absolute-voxel box `block_abs_voxels`) against the scene's
/// leaves (ADR 0010 Decision 2), composing each boundable leaf's field interval by CSG
/// interval arithmetic and overriding to BOUNDARY for any unboundable leaf.
///
/// `block_abs_voxels` is the block's half-open `[min, max)` box in the SCENE's ABSOLUTE
/// voxel frame (the frame [`Scene::resolve_chunk`](crate::scene::Scene::resolve_chunk)
/// clips against — recentre is applied later as a pure index offset, so classification is
/// frame-independent). Each leaf's interval is taken in its OWN local voxel-index frame by
/// subtracting the leaf's `world_offset_voxels` (the same map
/// [`stamp_producer_into_chunk`](crate::scene) uses for its resolve window).
///
/// Returns:
/// * [`BlockClassification::Air`] iff EVERY overlapping leaf provably misses the block
///   (and no leaf is unboundable) — the conservative interval guarantees brute force
///   finds zero voxels.
/// * [`BlockClassification::CoarseSolid`] iff a single leaf provably fills the WHOLE block
///   solid (a Union's nearer surface wins) — every voxel occupied, one material.
/// * [`BlockClassification::Boundary`] otherwise (straddling, multi-leaf overlap that
///   can't be proven uniformly solid, or any unboundable leaf) — the always-safe verdict.
///
/// The classifier is CONSERVATIVE: a block it calls AIR or COARSE-SOLID is occupancy-
/// IDENTICAL to brute force; a block it calls BOUNDARY is resolved per-voxel and is exact
/// regardless. This is what makes the round-trip bit-identical to the dense path.
pub(crate) fn classify_chunk_block(
    leaves: &[LeafProducer],
    block_abs_voxels: VoxelAabb,
    voxels_per_block: u32,
) -> BlockClassification {
    // Gather the leaves whose own grid AABB overlaps this block (others contribute
    // nothing). A leaf's absolute span is `[off, off + grid)` (corner-anchored).
    let mut overlapping: Vec<&LeafProducer> = Vec::new();
    for leaf in leaves {
        let grid_dimensions = leaf.producer.full_dimensions(voxels_per_block);
        let leaf_box = VoxelAabb::new(
            leaf.world_offset_voxels,
            [
                leaf.world_offset_voxels[0] + grid_dimensions[0] as i64,
                leaf.world_offset_voxels[1] + grid_dimensions[1] as i64,
                leaf.world_offset_voxels[2] + grid_dimensions[2] as i64,
            ],
        );
        if leaf_box.intersects(&block_abs_voxels) {
            overlapping.push(leaf);
        }
    }

    if overlapping.is_empty() {
        // No leaf touches the block ⇒ provably empty.
        return BlockClassification::Air;
    }

    // v1 composes leaves by Union (CombineOp::Union, later-wins material on overlap).
    // Compose the conservative field intervals by `union_field_intervals` (min-of-fields):
    // any unboundable operand collapses the union to `None` ⇒ BOUNDARY.
    let composed = union_field_intervals(overlapping.iter().map(|leaf| {
        // Map the absolute block box into THIS leaf's local voxel-index frame `[0, full)`
        // by subtracting its world offset — the exact frame `cell_field_interval` expects
        // (ADR 0008: the frame is carried, never re-derived).
        let cell_local = VoxelAabb::new(
            [
                block_abs_voxels.min[0] - leaf.world_offset_voxels[0],
                block_abs_voxels.min[1] - leaf.world_offset_voxels[1],
                block_abs_voxels.min[2] - leaf.world_offset_voxels[2],
            ],
            [
                block_abs_voxels.max[0] - leaf.world_offset_voxels[0],
                block_abs_voxels.max[1] - leaf.world_offset_voxels[1],
                block_abs_voxels.max[2] - leaf.world_offset_voxels[2],
            ],
        );
        leaf.producer.cell_field_interval(cell_local, voxels_per_block)
    }));

    let Some(interval) = composed else {
        // An unboundable leaf in the union ⇒ resolve the block per-voxel (BOUNDARY).
        return BlockClassification::Boundary;
    };

    match interval.classify(SURFACE_ISOLEVEL) {
        FieldClassification::Air => BlockClassification::Air,
        FieldClassification::Boundary => BlockClassification::Boundary,
        FieldClassification::CoarseSolid => {
            // A composed-solid verdict is only SAFELY coarse when EXACTLY ONE leaf
            // overlaps the block: with two leaves the Union's per-voxel MATERIAL is
            // "later wins on overlap", which the composed interval cannot resolve (it
            // proves geometric solidity, not which id each voxel takes). A multi-leaf
            // solid block is therefore forced BOUNDARY so the per-voxel pass assigns the
            // correct (later-wins) material — still exact, just unelided. (A single-leaf
            // solid block is uniform-material by construction: a Tool is single-material.)
            match (overlapping.len(), overlapping[0].material) {
                // Single single-material leaf provably filling the block ⇒ elide to coarse.
                (1, Some(block_id)) => BlockClassification::CoarseSolid(block_id),
                // Multi-leaf overlap (Union material is per-voxel later-wins, not coarsely
                // decidable) OR a single leaf with no single-material id (a Part's
                // per-voxel materials) ⇒ resolve per-voxel. Still exact, just unelided.
                _ => BlockClassification::Boundary,
            }
        }
    }
}

/// The on-face-grid overlay (ADR 0003 §3c) of the SINGLE leaf overlapping `block_abs_voxels`.
///
/// A coarse-solid block is owned by exactly one leaf (the classifier forces any multi-leaf
/// overlap to boundary), so its render overlay is unambiguous: this returns the first
/// overlapping leaf's `grid_overlay`, or `false` if none overlaps (an unreachable case for a
/// coarse-solid verdict — guarded defensively). The overlap test mirrors
/// [`classify_chunk_block`]'s, so the same single leaf is found.
fn single_overlapping_leaf_overlay(
    leaves: &[LeafProducer],
    block_abs_voxels: VoxelAabb,
    voxels_per_block: u32,
) -> bool {
    for leaf in leaves {
        let grid_dimensions = leaf.producer.full_dimensions(voxels_per_block);
        let leaf_box = VoxelAabb::new(
            leaf.world_offset_voxels,
            [
                leaf.world_offset_voxels[0] + grid_dimensions[0] as i64,
                leaf.world_offset_voxels[1] + grid_dimensions[1] as i64,
                leaf.world_offset_voxels[2] + grid_dimensions[2] as i64,
            ],
        );
        if leaf_box.intersects(&block_abs_voxels) {
            return leaf.grid_overlay;
        }
    }
    false
}

/// The OFF-by-default capability that builds the [`TwoLayerChunk`] display cache from the
/// one evaluator (ADR 0010 Decision 3 / 6). When the capability is OFF the live store
/// stays on the dense [`crate::store::Store`] path; this type is only constructed when a
/// caller opts in (the parity gate, and — later — E3's mesher).
///
/// It is a thin, stateless builder (no resident cache of its own in this slice — the
/// dense `Store` remains the live cache); a chunk is built on demand from the scene.
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
    /// ([`crate::cuboid_mesh::CuboidMeshRenderer::new_from_two_layer_chunks`]) consumes,
    /// visited in the SAME z,y,x order the dense store assembles. Returns an empty list when
    /// the capability is OFF or the scene has no covering chunk range (a Part-only scene —
    /// the caller falls back to the dense path). This keeps the `pub(crate)` chunk-range
    /// logic inside the crate while exposing the covering-chunk build to the `shot` binary.
    pub fn build_covering_chunks(
        &self,
        scene: &Scene,
        voxels_per_block: u32,
        lod: u32,
    ) -> Vec<([i32; 3], TwoLayerChunk)> {
        if !self.enabled {
            return Vec::new();
        }
        let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(voxels_per_block) else {
            return Vec::new();
        };
        let mut chunks = Vec::new();
        for chunk_z in min_chunk[2]..=max_chunk[2] {
            for chunk_y in min_chunk[1]..=max_chunk[1] {
                for chunk_x in min_chunk[0]..=max_chunk[0] {
                    let coord = [chunk_x, chunk_y, chunk_z];
                    let chunk = self
                        .build_chunk(coord, scene, voxels_per_block, lod)
                        .expect("the two-layer capability is enabled");
                    chunks.push((coord, chunk));
                }
            }
        }
        chunks
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

/// Build one chunk's two-layer representation by classifying every block and resolving the
/// boundary blocks per-voxel (the evaluator → display-cache step, ADR 0010 Decision 3).
fn build_two_layer_chunk(
    chunk_coord: [i32; 3],
    scene: &Scene,
    voxels_per_block: u32,
) -> TwoLayerChunk {
    let density = voxels_per_block.max(1);
    let chunk_extent_voxels = (CHUNK_BLOCKS * density) as i64;
    let block_extent = density as i64;
    let leaves = scene.leaf_producers(voxels_per_block);

    let chunk_min_voxels = [
        chunk_coord[0] as i64 * chunk_extent_voxels,
        chunk_coord[1] as i64 * chunk_extent_voxels,
        chunk_coord[2] as i64 * chunk_extent_voxels,
    ];

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

                match classify_chunk_block(&leaves, block_abs, voxels_per_block) {
                    BlockClassification::Air => {}
                    BlockClassification::CoarseSolid(block_id) => {
                        let flat = coarse_flat_index(block);
                        chunk.coarse[flat] = Some(block_id);
                        // A coarse-solid block is owned by EXACTLY ONE leaf (the classifier
                        // forces multi-leaf overlaps to boundary), so its on-face-grid
                        // overlay (ADR 0003 §3c) is that single leaf's `grid_overlay`.
                        chunk.coarse_overlay[flat] =
                            single_overlapping_leaf_overlay(&leaves, block_abs, voxels_per_block);
                    }
                    BlockClassification::Boundary => {
                        let geometry =
                            resolve_boundary_block(&leaves, block_min, density, voxels_per_block);
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

/// Resolve a boundary block per-voxel into a dense `density³` [`VoxelRegion`] (the
/// material at each occupied voxel), decompose it to cuboids, and compute its per-face
/// seam-solidity flags. `block_min_abs` is the block's low corner in absolute voxels.
///
/// Per-voxel resolution reuses each overlapping leaf's [`VoxelProducer::resolve_into`]
/// over the block window, composed by the SAME Union semantics the dense path uses
/// (document order, later-wins on overlap) — so the materialised block is bit-identical
/// to the dense store's voxels for that block.
fn resolve_boundary_block(
    leaves: &[LeafProducer],
    block_min_abs: [i64; 3],
    density: u32,
    voxels_per_block: u32,
) -> MicroblockGeometry {
    let extent = [density, density, density];
    let mut region = VoxelRegion::new_empty(extent);

    // Compose leaves in DOCUMENT ORDER (the order `leaf_producers` yields them, which is
    // `for_each_leaf`'s walk order), later-wins on overlap — exactly the dense Union.
    for leaf in leaves {
        let grid_dimensions = leaf.producer.full_dimensions(voxels_per_block);
        let leaf_box = VoxelAabb::new(
            leaf.world_offset_voxels,
            [
                leaf.world_offset_voxels[0] + grid_dimensions[0] as i64,
                leaf.world_offset_voxels[1] + grid_dimensions[1] as i64,
                leaf.world_offset_voxels[2] + grid_dimensions[2] as i64,
            ],
        );
        let block_abs = VoxelAabb::new(
            block_min_abs,
            [
                block_min_abs[0] + density as i64,
                block_min_abs[1] + density as i64,
                block_min_abs[2] + density as i64,
            ],
        );
        if !leaf_box.intersects(&block_abs) {
            continue;
        }

        // Resolve JUST this block's window in the leaf's local voxel-index frame.
        let window_local = VoxelAabb::new(
            [
                block_min_abs[0] - leaf.world_offset_voxels[0],
                block_min_abs[1] - leaf.world_offset_voxels[1],
                block_min_abs[2] - leaf.world_offset_voxels[2],
            ],
            [
                block_min_abs[0] + density as i64 - leaf.world_offset_voxels[0],
                block_min_abs[1] + density as i64 - leaf.world_offset_voxels[1],
                block_min_abs[2] + density as i64 - leaf.world_offset_voxels[2],
            ],
        );
        let mut local = VoxelGrid::default();
        leaf.producer
            .resolve_into(&mut local, voxels_per_block, window_local);

        // Stamp each emitted voxel into the block-local region at its material (a Tool
        // overrides every voxel's id; a Part keeps its own per-voxel id). The voxel's
        // local index is in the LEAF's frame, so shift back to block-local by adding the
        // leaf offset and subtracting the block's absolute low corner.
        for voxel in &local.occupied {
            let block_local = [
                voxel.local_index[0] as i64 + leaf.world_offset_voxels[0] - block_min_abs[0],
                voxel.local_index[1] as i64 + leaf.world_offset_voxels[1] - block_min_abs[1],
                voxel.local_index[2] as i64 + leaf.world_offset_voxels[2] - block_min_abs[2],
            ];
            if block_local.iter().any(|&c| c < 0 || c >= density as i64) {
                continue; // Outside this block (the window clamps, but guard anyway).
            }
            let block_id = match leaf.material {
                Some(id) => id.0,
                None => voxel.block_id.0,
            };
            // Stamp the cuboid mesher's RENDER-CELL key (ADR 0003 §3c): the clean
            // categorical id in the low bits, this leaf's on-face-grid overlay in the
            // dedicated bit. So `decompose_into_boxes` splits a box across differing
            // overlay states exactly like the dense mesher, and the E3 mesher reads the
            // box's clean id + overlay back out — without the render flag ever entering
            // the categorical cell (the E2 occupancy expansion masks the bit off).
            let render_key = crate::cuboid_mesh::compose_cell_key(block_id, leaf.grid_overlay);
            // Later document-order leaf wins on overlap: a plain overwrite reproduces
            // the dense Union (the walk visits in document order, last write persists).
            region.set(
                block_local[0] as u32,
                block_local[1] as u32,
                block_local[2] as u32,
                Some(render_key),
            );
        }
    }

    let cuboids = decompose_into_boxes(&region);
    let seam_solidity = compute_seam_solidity(&region);
    MicroblockGeometry {
        cuboids,
        seam_solidity,
    }
}

/// Compute the per-face seam-solidity flags for a resolved boundary block: a face is solid
/// iff EVERY voxel cell on that face of the `density³` region is occupied.
fn compute_seam_solidity(region: &VoxelRegion) -> SeamSolidity {
    let [width, height, depth] = region.extent;
    let mut solid = [[true; 2]; 3];
    // Degenerate (zero-extent) region: no face can be solid.
    if width == 0 || height == 0 || depth == 0 {
        return SeamSolidity {
            solid: [[false; 2]; 3],
        };
    }

    // X faces (axis 0): low x == 0, high x == width - 1.
    for &(side, x) in &[(0usize, 0u32), (1usize, width - 1)] {
        let mut face_solid = true;
        'scan: for z in 0..depth {
            for y in 0..height {
                if region.material_at(x, y, z).is_none() {
                    face_solid = false;
                    break 'scan;
                }
            }
        }
        solid[0][side] = face_solid;
    }
    // Y faces (axis 1): low y == 0, high y == height - 1.
    for &(side, y) in &[(0usize, 0u32), (1usize, height - 1)] {
        let mut face_solid = true;
        'scan: for z in 0..depth {
            for x in 0..width {
                if region.material_at(x, y, z).is_none() {
                    face_solid = false;
                    break 'scan;
                }
            }
        }
        solid[1][side] = face_solid;
    }
    // Z faces (axis 2): low z == 0, high z == depth - 1.
    for &(side, z) in &[(0usize, 0u32), (1usize, depth - 1)] {
        let mut face_solid = true;
        'scan: for y in 0..height {
            for x in 0..width {
                if region.material_at(x, y, z).is_none() {
                    face_solid = false;
                    break 'scan;
                }
            }
        }
        solid[2][side] = face_solid;
    }

    SeamSolidity { solid }
}

/// Stream a whole scene's two-layer chunks back to one recentred [`VoxelGrid`], in the
/// EXACT frame the dense [`Store::resolve_region`](crate::store::Store::resolve_region)
/// assembles — the parity gate's "two-layer round-trip" (ADR 0010 parity (a)). Builds
/// each covering chunk via the capability, expands it (coarse fast-fill + boundary
/// per-voxel), and rebases by `chunk_min_voxels − recentre` so the occupied SET matches
/// the dense path bit-for-bit (position + block id).
///
/// Returns `None` when the capability is OFF (the caller stays on the dense path) or the
/// scene has no covering chunk range (a Part-only scene — handled by the dense path).
pub fn resolve_region_two_layer(
    store: &TwoLayerStore,
    scene: &Scene,
    voxels_per_block: u32,
    lod: u32,
) -> Option<VoxelGrid> {
    if !store.is_enabled() {
        return None;
    }
    debug_assert_eq!(lod, 0, "E2 only resolves full resolution (lod 0)");

    let region_dimensions = scene.placed_region_dimensions(voxels_per_block);
    let mut output = VoxelGrid::new(region_dimensions);
    // ADR 0008: carry the recentre so consumers decode world→index without re-deriving it
    // — the same value the dense `resolve_region` stamps (the parity gate compares it).
    let recentre_voxels = scene.recentre_voxels_for_resolve(voxels_per_block);
    output.recentre_voxels = recentre_voxels;

    let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(voxels_per_block) else {
        return Some(output); // No composite extent (Part-only): empty region.
    };

    let chunk_extent_voxels = (CHUNK_BLOCKS * voxels_per_block.max(1)) as i64;
    // Visit chunks in the SAME z,y,x order the dense store assembles them in, so the
    // emitted voxel order matches too (the multiset compare is order-independent, but
    // keeping the order identical makes a future ordered compare cheap).
    for chunk_z in min_chunk[2]..=max_chunk[2] {
        for chunk_y in min_chunk[1]..=max_chunk[1] {
            for chunk_x in min_chunk[0]..=max_chunk[0] {
                let chunk_coord = [chunk_x, chunk_y, chunk_z];
                let chunk = store
                    .build_chunk(chunk_coord, scene, voxels_per_block, lod)
                    .expect("capability is enabled");
                // Rebase chunk-local indices into the recentred frame: a voxel at
                // chunk-local index `l` has absolute index `chunk_min + l`, and the
                // recentred frame subtracts `recentre` — so the offset is
                // `chunk_min − recentre` (i64, before the i32 downcast). This is exactly
                // `resolve_chunk_rebased`'s rebase with floating origin = recentre.
                let index_offset = [
                    chunk_x as i64 * chunk_extent_voxels - recentre_voxels[0],
                    chunk_y as i64 * chunk_extent_voxels - recentre_voxels[1],
                    chunk_z as i64 * chunk_extent_voxels - recentre_voxels[2],
                ];
                chunk.expand_occupancy_into(&mut output.occupied, index_offset);
            }
        }
    }

    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_geom::MaterialChoice;
    use crate::scene::{DefId, Node, NodeContent, NodeTransform};
    use crate::voxel::{GeometryParams, SdfShape, ShapeKind};

    /// Canonicalise an occupied set into the **resolved occupancy SET**: a map from each
    /// bit-exact voxel position to the block id of the LAST (document-order) writer at that
    /// position. This is the ADR 0010 parity-gate canonical form (the resolved occupancy
    /// SET keyed by position+block_id) — it differs from the dense store's
    /// `cache_region_matches_monolithic_*` MULTISET only at positions where leaves overlap.
    ///
    /// The two-layer store is a one-id-per-cell representation (a boundary block resolves
    /// to a dense region where the later leaf overwrites the earlier — Union "later wins"),
    /// so it never carries the dense path's DUPLICATE Vec entries at a shared position. The
    /// dense `Scene::resolve_region` emits leaves in document order, so the LAST entry at a
    /// position is the winner there too — taking the last writer on BOTH sides compares the
    /// true resolved occupancy. For every non-overlapping scene (all the SDF-shape /
    /// flat-odd cases) each position has exactly one writer, so this is byte-identical to
    /// the dense multiset; only genuinely-overlapping leaves (cloud-over-box) differ, and
    /// there the resolved-set is the correct comparison.
    ///
    /// Keying on the raw `f32` bits (`to_bits`) asserts the BYTES a consumer reads are
    /// identical, not merely the rounded voxel set.
    fn resolved_occupancy_set(
        grid: &VoxelGrid,
    ) -> std::collections::BTreeMap<[u32; 3], u16> {
        let mut set = std::collections::BTreeMap::new();
        for voxel in &grid.occupied {
            let position = voxel.world_position();
            let key = [
                position[0].to_bits(),
                position[1].to_bits(),
                position[2].to_bits(),
            ];
            // Last document-order writer wins (Union later-wins material).
            set.insert(key, voxel.color_index());
        }
        set
    }

    fn shape_scene(kind: ShapeKind, voxels_per_block: u32) -> Scene {
        Scene::from_geometry(
            GeometryParams {
                shape: kind,
                size_voxels: [
                    5 * voxels_per_block,
                    5 * voxels_per_block,
                    5 * voxels_per_block,
                ],
                size_measurements: None,
                voxels_per_block,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        )
    }

    /// THE GATE (parity (a)): the two-layer round-trip occupancy (coarse fast-fill +
    /// boundary per-voxel) is BIT-IDENTICAL (position + block id) to the dense
    /// `Scene::resolve_region`, for the gated scene. Mirrors
    /// `store.rs::cache_region_matches_monolithic_*`. Returns the chunk + cell counts the
    /// build classified (so the harness can report coverage).
    fn assert_two_layer_round_trip_matches_dense(
        scene: &Scene,
        voxels_per_block: u32,
        label: &str,
    ) -> (usize, u64) {
        let dense = scene.resolve_region(
            scene.full_extent_blocks(voxels_per_block),
            voxels_per_block,
            0,
        );
        let store = TwoLayerStore::enabled();
        let assembled = resolve_region_two_layer(&store, scene, voxels_per_block, 0)
            .expect("the capability is enabled");

        assert_eq!(
            assembled.dimensions, dense.dimensions,
            "[{label}] two-layer round-trip dimensions must match dense resolve_region"
        );
        assert_eq!(
            assembled.recentre_voxels, dense.recentre_voxels,
            "[{label}] two-layer round-trip must carry the SAME recentre as dense"
        );
        let dense_set = resolved_occupancy_set(&dense);
        let assembled_set = resolved_occupancy_set(&assembled);
        assert_eq!(
            assembled_set.len(),
            dense_set.len(),
            "[{label}] two-layer resolved occupancy count must match dense (the dense Vec \
             may hold duplicate entries at overlap positions; the resolved SET must agree)"
        );
        assert_eq!(
            assembled_set, dense_set,
            "[{label}] two-layer round-trip resolved occupancy SET (position + block id, \
             last-writer-wins) must be BIT-IDENTICAL to the dense resolve_region"
        );

        // Coverage accounting: count chunks + blocks the build classified.
        let (min_chunk, max_chunk) = scene
            .covering_chunk_range(voxels_per_block)
            .unwrap_or(([0; 3], [-1; 3]));
        let chunks = if max_chunk[0] < min_chunk[0] {
            0
        } else {
            ((max_chunk[0] - min_chunk[0] + 1)
                * (max_chunk[1] - min_chunk[1] + 1)
                * (max_chunk[2] - min_chunk[2] + 1)) as usize
        };
        let cells = chunks as u64 * (CHUNK_BLOCKS as u64).pow(3);
        (chunks, cells)
    }

    #[test]
    fn round_trip_matches_dense_for_all_shapes() {
        let mut total_chunks = 0usize;
        let mut total_cells = 0u64;
        for kind in [
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Tube,
            ShapeKind::Torus,
            ShapeKind::Box,
        ] {
            let scene = shape_scene(kind, 16);
            let (chunks, cells) =
                assert_two_layer_round_trip_matches_dense(&scene, 16, &format!("{kind:?}"));
            total_chunks += chunks;
            total_cells += cells;
        }
        eprintln!(
            "two-layer parity (all shapes): {total_chunks} chunks, {total_cells} block cells"
        );
    }

    /// FLAT / odd-sized shapes — the S0 covering-range regression case (a 1-block axis
    /// straddles two chunks). The classifier must cover the producer-true voxel extent
    /// and round-trip bit-identically, just as the dense net pins.
    #[test]
    fn round_trip_matches_dense_for_flat_and_odd_shapes() {
        let mut total_cells = 0u64;
        for kind in [ShapeKind::Cylinder, ShapeKind::Sphere, ShapeKind::Torus] {
            for size in [[5u32, 1, 5], [3, 1, 3], [5, 3, 5], [1, 1, 1]] {
                let scene = Scene::from_geometry(
                    GeometryParams {
                        shape: kind,
                        size_voxels: [size[0] * 16, size[1] * 16, size[2] * 16],
                        size_measurements: None,
                        voxels_per_block: 16,
                        wall_blocks: 1,
                    },
                    MaterialChoice::Stone,
                );
                let (_chunks, cells) = assert_two_layer_round_trip_matches_dense(
                    &scene,
                    16,
                    &format!("{kind:?} {size:?}"),
                );
                total_cells += cells;
            }
        }
        eprintln!("two-layer parity (flat/odd): {total_cells} block cells");
    }

    fn make_tool(kind: ShapeKind, offset: [i64; 3], material: MaterialChoice, density: u32) -> Node {
        let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, density);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset, density);
        node
    }

    #[test]
    fn round_trip_matches_dense_for_demo_scene() {
        let density = 16;
        let scene = Scene::from_nodes(vec![
            make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone, density),
            make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood, density),
            make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain, density),
        ]);
        assert_two_layer_round_trip_matches_dense(&scene, density, "demo-scene");
    }

    #[test]
    fn round_trip_matches_dense_for_demo_village() {
        let density = 16;
        let house_def_id = DefId(1);
        let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, size, 1, density);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = NodeTransform::from_blocks(offset, density);
            node
        };
        let instance = |name: &str, offset: [i64; 3]| {
            let mut node = Node::new(name, NodeContent::Instance(house_def_id));
            node.transform = NodeTransform::from_blocks(offset, density);
            node
        };
        let mut scene = Scene::from_nodes(vec![
            instance("House 1", [0, 0, 0]),
            instance("House 2", [6, 0, 0]),
            instance("House 3", [12, 0, 0]),
            instance("House 4", [18, 0, 0]),
        ]);
        scene.add_definition(
            house_def_id,
            "House".to_string(),
            vec![
                tool(ShapeKind::Box, [2, 2, 2], [0, 0, 0], MaterialChoice::Stone),
                tool(ShapeKind::Cylinder, [1, 2, 1], [0, 2, 0], MaterialChoice::Wood),
            ],
        );
        assert_two_layer_round_trip_matches_dense(&scene, density, "demo-village");
    }

    /// A sketch-revolve solid (the 800×800-revolve CLASS that stressed the dense cap): a
    /// sketch always classifies BOUNDARY (its polygon fill is not a coarse box), so this
    /// pins the per-voxel boundary path round-trips bit-identically to the dense store.
    #[test]
    fn round_trip_matches_dense_for_sketch_revolve() {
        use crate::sketch::{PlaneAxis, RevolveAxis, Sketch, SketchSolid};
        let density = 16;
        let profile = Sketch::rectangle(PlaneAxis::Z, 24, 16);
        let producer = SketchSolid::revolve(profile, RevolveAxis::InPlane0, 360);
        let node = Node::new(
            "Revolve",
            NodeContent::SketchTool {
                producer,
                material: MaterialChoice::Stone,
            },
        );
        let scene = Scene::from_nodes(vec![node]);
        assert_two_layer_round_trip_matches_dense(&scene, density, "sketch-revolve");
    }

    /// A region-spanning UNBOUNDABLE producer (the fBm cloud field) forces every covering
    /// block BOUNDARY (its `cell_field_interval` is `None`) and STILL round-trips
    /// bit-identically — the "unboundable ops fall back, still exact" acceptance criterion.
    /// (Mixed with a Tool so the scene has a composite chunk extent.)
    #[test]
    fn round_trip_matches_dense_with_unboundable_cloud() {
        use crate::scene::Part;
        let density = 16;
        let mut cloud = Node::new("Clouds", NodeContent::Part(Part::DebugClouds { seed: 7 }));
        cloud.transform = NodeTransform::from_blocks([0, 0, 0], density);
        let scene = Scene::from_nodes(vec![
            make_tool(ShapeKind::Box, [0, 0, 0], MaterialChoice::Stone, density),
            cloud,
        ]);
        assert_two_layer_round_trip_matches_dense(&scene, density, "tool+cloud");
    }

    /// INTERIOR ELISION (the whole point): a LARGE solid box stores ZERO interior voxels
    /// under the two-layer path — only its surface shell lives in the microblock layer,
    /// while its interior is coarse block ids. The dense path would densify all
    /// ~`(size·d)³` interior voxels (and a revolve-class size blows the 6M cap); the
    /// two-layer stored count is surface-only.
    #[test]
    fn large_solid_box_stores_zero_interior_voxels() {
        let density = 16;
        // 50×50×50 BLOCKS @ d16 = 800×800×800 voxels — the revolve-class size the ADR
        // calls out. Dense interior would be 800³ ≈ 5.1e8 voxels (far past the 6M cap);
        // the two-layer interior holds NONE.
        let blocks = 50u32;
        let shape = SdfShape::from_blocks(ShapeKind::Box, [blocks, blocks, blocks], 1, density);
        let node = Node::new(
            "BigBox",
            NodeContent::Tool {
                shape,
                material: MaterialChoice::Stone,
            },
        );
        let scene = Scene::from_nodes(vec![node]);
        let store = TwoLayerStore::enabled();

        let (min_chunk, max_chunk) = scene
            .covering_chunk_range(density)
            .expect("a placed box has a covering chunk range");

        let mut total_stored = 0u64;
        let mut interior_chunks = 0u64;
        let mut total_chunks = 0u64;
        // An interior chunk (no block of it touches a face of the box) is entirely
        // coarse-solid, so it must store ZERO voxels.
        for chunk_z in min_chunk[2]..=max_chunk[2] {
            for chunk_y in min_chunk[1]..=max_chunk[1] {
                for chunk_x in min_chunk[0]..=max_chunk[0] {
                    let chunk = store
                        .build_chunk([chunk_x, chunk_y, chunk_z], &scene, density, 0)
                        .unwrap();
                    let stored = chunk.stored_voxel_count();
                    total_stored += stored;
                    total_chunks += 1;
                    let is_interior_chunk = chunk_x > min_chunk[0]
                        && chunk_x < max_chunk[0]
                        && chunk_y > min_chunk[1]
                        && chunk_y < max_chunk[1]
                        && chunk_z > min_chunk[2]
                        && chunk_z < max_chunk[2];
                    if is_interior_chunk {
                        interior_chunks += 1;
                        assert_eq!(
                            stored, 0,
                            "interior chunk ({chunk_x},{chunk_y},{chunk_z}) of a solid box \
                             must store ZERO voxels (interior elision), got {stored}"
                        );
                    }
                }
            }
        }
        let dense_interior_voxels = (blocks as u64 * density as u64).pow(3);
        assert!(
            interior_chunks > 0,
            "the box must be large enough to have fully-interior chunks"
        );
        // The stored voxels are the 1-block-thick SURFACE SHELL only (each surface block
        // is d³ = 4096 voxels at d16, so a 50²-face shell is legitimately ~12% of the
        // volume) — a fraction of the dense interior, and every FULLY-interior chunk
        // (asserted above) holds ZERO. The dense path would densify the whole volume and
        // blow the 6M cap; the two-layer path never builds the interior.
        assert!(
            total_stored < dense_interior_voxels / 4,
            "two-layer stored voxels ({total_stored}) must be well below the dense interior \
             volume ({dense_interior_voxels}) — surface-shell-only residency"
        );
        eprintln!(
            "interior elision: {total_chunks} chunks ({interior_chunks} fully interior); \
             two-layer stored {total_stored} voxels vs dense interior {dense_interior_voxels}"
        );
    }

    /// A fully-interior block of a solid box classifies COARSE-SOLID (no voxels); a block
    /// straddling the box face classifies BOUNDARY; a block well outside classifies AIR.
    #[test]
    fn classifier_sorts_air_coarse_and_boundary() {
        let density = 8u32;
        // A 5×5×5-block box at the origin → voxel extent [0, 40) per axis.
        let shape = SdfShape::from_blocks(ShapeKind::Box, [5, 5, 5], 1, density);
        let node = Node::new(
            "Box",
            NodeContent::Tool {
                shape,
                material: MaterialChoice::Wood,
            },
        );
        let scene = Scene::from_nodes(vec![node]);
        let leaves = scene.leaf_producers(density);
        let block = density as i64;

        // A deep-interior block ([16,24) on each axis, well inside [0,40)) is coarse-solid.
        let interior = VoxelAabb::new([16, 16, 16], [16 + block, 16 + block, 16 + block]);
        assert_eq!(
            classify_chunk_block(&leaves, interior, density),
            BlockClassification::CoarseSolid(MaterialChoice::Wood.block_id()),
            "a deep-interior block of a solid box must be coarse-solid at its material"
        );

        // A block straddling the +X face (the box ends at voxel 40; block [40−4,40+4)).
        let straddle = VoxelAabb::new([36, 16, 16], [36 + block, 16 + block, 16 + block]);
        assert_eq!(
            classify_chunk_block(&leaves, straddle, density),
            BlockClassification::Boundary,
            "a block straddling the box surface must be boundary"
        );

        // A block far outside the box ([200, 208) on X) is air.
        let outside = VoxelAabb::new([200, 16, 16], [200 + block, 16 + block, 16 + block]);
        assert_eq!(
            classify_chunk_block(&leaves, outside, density),
            BlockClassification::Air,
            "a block well outside every leaf must be air"
        );
    }

    /// Sculpt-touched / multi-leaf-overlap conservatism: a block where TWO Tools overlap
    /// is forced BOUNDARY (the Union's per-voxel later-wins material is not coarsely
    /// decidable), even if geometrically solid — still exact after per-voxel.
    #[test]
    fn overlapping_leaves_force_boundary() {
        let density = 8u32;
        // Two boxes overlapping at the origin region, different materials.
        let scene = Scene::from_nodes(vec![
            make_tool_density(ShapeKind::Box, [0, 0, 0], MaterialChoice::Stone, density, 5),
            make_tool_density(ShapeKind::Box, [1, 0, 0], MaterialChoice::Wood, density, 5),
        ]);
        let leaves = scene.leaf_producers(density);
        let block = density as i64;
        // A block in the overlap region of both boxes.
        let overlap = VoxelAabb::new([16, 16, 16], [16 + block, 16 + block, 16 + block]);
        assert_eq!(
            classify_chunk_block(&leaves, overlap, density),
            BlockClassification::Boundary,
            "a block two leaves both fill must be boundary (per-voxel material resolution)"
        );
    }

    fn make_tool_density(
        kind: ShapeKind,
        offset: [i64; 3],
        material: MaterialChoice,
        density: u32,
        size_blocks: u32,
    ) -> Node {
        let shape = SdfShape::from_blocks(kind, [size_blocks; 3], 1, density);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset, density);
        node
    }

    /// SEAM-SOLIDITY flags: a boundary block's per-face flag matches its ACTUAL face
    /// occupancy. We resolve a boundary block of a solid box and assert the face that lies
    /// INSIDE the box is solid while the face that pokes OUT of the box is not.
    #[test]
    fn seam_solidity_flags_match_face_occupancy() {
        let density = 8u32;
        // A solid box [0,40) per axis. Take the block straddling the +X face: block
        // [32,40) on X (the last fully-inside block column is [32,40); the face at X=39
        // is the box's last solid layer, and X=40+ is air). To get a STRADDLING block on
        // a different axis we instead take a block at the +X edge whose low-X face is
        // solid (inside the box) and whose geometry is the surface shell.
        let shape = SdfShape::from_blocks(ShapeKind::Box, [5, 5, 5], 1, density);
        let scene = Scene::from_nodes(vec![Node::new(
            "Box",
            NodeContent::Tool {
                shape,
                material: MaterialChoice::Stone,
            },
        )]);
        let leaves = scene.leaf_producers(density);

        // The block [32,40) on X, interior on Y/Z ([16,24)). The box fills X∈[0,40), so
        // this whole block is solid — BUT it touches the +X face of the box, so its
        // classification depends on the conservative bound. Resolve it directly to read
        // its seam flags regardless of the coarse verdict.
        let block_min = [32i64, 16, 16];
        let geometry = resolve_boundary_block(&leaves, block_min, density, density);
        // The low-X face (X=32, the 0th local layer) is deep inside the box ⇒ fully solid.
        assert!(
            geometry.seam_solidity.face_is_solid(0, 0),
            "the low-X face of an interior-touching block must be solid"
        );
        // Every face of this all-solid block is solid (the box fully covers [32,40)³ here,
        // since Y,Z ∈ [16,24) ⊂ [0,40) and X ∈ [32,40) ⊂ [0,40)).
        for axis in 0..3 {
            for side in 0..2 {
                assert!(
                    geometry.seam_solidity.face_is_solid(axis, side),
                    "a fully-solid block must report every face solid (axis {axis}, side {side})"
                );
            }
        }

        // A block straddling the +X surface (X∈[36,44), so X∈[40,44) is OUTSIDE the box):
        // its low-X face (X=36, inside) is solid; its high-X face (X=43, outside) is NOT.
        let straddle_min = [36i64, 16, 16];
        let straddle = resolve_boundary_block(&leaves, straddle_min, density, density);
        assert!(
            straddle.seam_solidity.face_is_solid(0, 0),
            "the inside (low-X) face of a +X-straddling block must be solid"
        );
        assert!(
            !straddle.seam_solidity.face_is_solid(0, 1),
            "the outside (high-X) face of a +X-straddling block must NOT be solid"
        );
    }

    /// The capability is OFF by default: `build_chunk` / `resolve_region_two_layer` return
    /// `None` so the caller falls back to the dense path (the coexistence contract).
    #[test]
    fn capability_off_by_default_returns_none() {
        let scene = shape_scene(ShapeKind::Sphere, 16);
        let store = TwoLayerStore::default();
        assert!(!store.is_enabled());
        assert!(store.build_chunk([0, 0, 0], &scene, 16, 0).is_none());
        assert!(resolve_region_two_layer(&store, &scene, 16, 0).is_none());
    }
}
