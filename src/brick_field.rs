//! ADR 0011 G0 — the **brick-field BUILD**: pack ADR 0010's two-layer boundary set into
//! a sorted [`BrickRecord`] array (keyed by a packed world-block key) + an R8 3D
//! texture atlas of sculpted-brick occupancy. **Wired to NOTHING** — no renderer reads
//! this yet; it is the standalone build + parity harness (the analogue of ADR 0007's
//! atlas-mechanic-proven step and ADR 0010's E1), gated by `tests/gpu_parity.rs`
//! before any live raymarch (G1) consumes it.
//!
//! The mapping is ADR 0011 Decision 2, one-to-one onto the [`TwoLayerChunk`] partition:
//!
//! * **air block** → no record (the ray skips it via the clip-map, later).
//! * **coarse-solid block** → one [`BrickPayload::CoarseSolid`] record: a solid
//!   block-cube at its [`BlockId`], **no atlas slot, no per-voxel data** — the
//!   interior-elision win carried onto the GPU.
//! * **boundary block** (a `microblocks` entry) → one [`BrickPayload::Sculpted`] record
//!   whose atlas slot holds the block's voxel occupancy, rasterized from its cuboids.
//!
//! **The brick granule is ONE BLOCK** (ADR 0011 Decision 1): the brick edge is
//! `voxels_per_block` — block-denominated, correct at ANY density; nothing here may
//! hard-code 16. Per-face [`SeamSolidity`] flags carry across unchanged (they are the
//! brick-field's apron analogue).
//!
//! ## Frame (ADR 0008)
//!
//! A brick key is the block's **absolute world-block coordinate**
//! (`chunk_coord * CHUNK_BLOCKS + chunk_local_block_index`) — the same world-fixed
//! integer lattice the chunk keys live on; recentre/floating-origin never enters the
//! key (a recentre shift leaves every record valid, exactly like the chunk cache).
//!
//! ## Exactness (the ADR 0011 parity gate, clause (a))
//!
//! Packing is pure integer work: a sculpted brick's atlas bytes must be BYTE-IDENTICAL
//! to the CPU two-layer boundary set's occupancy for that block, and a coarse-solid
//! block must emit exactly one coarse record and consume no atlas slot. The
//! `--features gpu` parity tests assert this through the full texture round-trip.

use crate::core_geom::{BlockId, CHUNK_BLOCKS};
use crate::cuboid_mesh::clean_block_id;
use crate::two_layer_store::{SeamSolidity, TwoLayerChunk};

/// Signed world-block coordinates are biased into this many bits per axis inside the
/// packed key: ±2^20 (~1M) blocks per axis, far beyond the anisotropic 10k+-block
/// target. Three 21-bit lanes fill bits 0..63 (z high), so the packed key's integer
/// order IS lexicographic (z, y, x) block order — sortable on the CPU and binary-
/// searchable as a `(hi, lo)` u32 pair in WGSL (no u64 there).
const WORLD_BLOCK_KEY_BITS_PER_AXIS: u32 = 21;
const WORLD_BLOCK_KEY_BIAS: i64 = 1 << (WORLD_BLOCK_KEY_BITS_PER_AXIS - 1);

/// Pack an absolute world-block coordinate into the sorted-record key (z-major
/// lexicographic order). Panics if a coordinate falls outside the ±2^20 biased lane —
/// a scene that large is out of every current target's range, and a silent wrap would
/// alias two blocks onto one brick.
pub fn pack_world_block_key(world_block: [i64; 3]) -> u64 {
    let mut packed = 0u64;
    // z fills the highest lane so integer order == (z, y, x) lexicographic order.
    for (lane, &coordinate) in [world_block[2], world_block[1], world_block[0]]
        .iter()
        .enumerate()
    {
        let biased = coordinate + WORLD_BLOCK_KEY_BIAS;
        assert!(
            (0..(1i64 << WORLD_BLOCK_KEY_BITS_PER_AXIS)).contains(&biased),
            "world-block coordinate {coordinate} exceeds the packed-key lane (±2^20 blocks)"
        );
        packed |= (biased as u64) << ((2 - lane) as u32 * WORLD_BLOCK_KEY_BITS_PER_AXIS);
    }
    packed
}

/// Unpack a [`pack_world_block_key`] key back to its world-block coordinate (the
/// parity harness's mismatch-location readout; the shader never needs it).
pub fn unpack_world_block_key(key: u64) -> [i64; 3] {
    let lane_mask = (1u64 << WORLD_BLOCK_KEY_BITS_PER_AXIS) - 1;
    let unpack_lane = |lane: u32| -> i64 {
        ((key >> (lane * WORLD_BLOCK_KEY_BITS_PER_AXIS)) & lane_mask) as i64
            - WORLD_BLOCK_KEY_BIAS
    };
    [unpack_lane(0), unpack_lane(1), unpack_lane(2)]
}

// ============================================================================
// Clip-map occupancy pyramid (ADR 0011 Decision 4a / slice G2+G4) — THREE
// WORLD-FIXED coarse "any-brick-inside" levels above the brick set, a min-mip of
// the record keys on an 8× cell progression (8 → 64 → 512 blocks/cell). The
// hierarchical DDA (brick_raymarch.wgsl) jumps a ray to the exit of the coarsest
// EMPTY level covering its position — one stride through empty space — descending
// to per-block brick work only where a level reports occupancy. This is the port
// of ADR 0009's measured 160→10240 (~64×) scattered-ceiling lift; G4 adds the
// third level (512-block cells) so a wide scatter skips whole 512-block voids in
// one stride instead of eight L2 strides, closing most of the raw scattered
// ceiling gap vs the rasterized mesh (frustum/Z cull it gets for free).
//
// Why stop at three: the packed key is 21 bits/axis (±2^20 blocks), so a fourth
// level (4096 blocks/cell) has at most ~512 cells of span to skip; on realistic
// 10k-block-span scenes L3 already caps the empty-void skip at a handful of
// strides and an L4 stride only replaces ~8 already-cheap L3 strides — measured
// not to pay (see `clipmap_scattered_scene_skips_empty_space`'s +L4 column).
// ============================================================================

/// Level 1 (fine) clip-map cell edge, in BLOCKS — the benchmark's proven config
/// (ADR 0011 Decision 4a). Block-denominated (density-agnostic by construction),
/// never a hard-coded voxel count.
pub const CLIPMAP_LEVEL_1_BLOCKS_PER_CELL: u32 = 8;
/// Level 2 (middle) clip-map cell edge, in BLOCKS (the benchmark's L2) — 8× L1.
pub const CLIPMAP_LEVEL_2_BLOCKS_PER_CELL: u32 = 64;
/// Level 3 (coarse) clip-map cell edge, in BLOCKS (G4) — 8× L2, checked first by
/// the hierarchical DDA so a wide empty void skips in one 512-block stride.
pub const CLIPMAP_LEVEL_3_BLOCKS_PER_CELL: u32 = 512;

/// One clip-map occupancy level: cells of `blocks_per_cell` blocks per axis, each
/// a packed cell key (the SAME 21-bit z-major packing as a brick record's block
/// key, applied to the CELL coordinate = `floor_div(absolute_block,
/// blocks_per_cell)`). `cell_keys` is sorted strictly ascending + unique — the
/// order the in-shader binary search relies on, exactly like the record array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClipmapLevel {
    /// Cell edge in blocks (8 for L1, 64 for L2). Block-denominated.
    pub blocks_per_cell: u32,
    /// The occupied cells' packed keys, sorted ascending + deduplicated — a
    /// SUPERSET of the true occupied cells by construction (every record's cell is
    /// present), so the hierarchical DDA only ever skips provably-empty space.
    pub cell_keys: Vec<u64>,
}

impl ClipmapLevel {
    /// An empty level (no occupied cells) — the "pyramid off" form the renderer
    /// installs to A/B the hierarchical skip (`record_count == 0` ⇒ the shader
    /// never skips, so the march is the flat G1 block-DDA).
    pub fn empty(blocks_per_cell: u32) -> Self {
        ClipmapLevel {
            blocks_per_cell: blocks_per_cell.max(1),
            cell_keys: Vec::new(),
        }
    }

    /// Fold a record set's block keys into this level's occupied-cell set: every
    /// record's block maps to exactly one cell; the deduplicated, sorted set is
    /// the min-mip. Pure function of the record keys (ADR 0011 4a).
    pub fn from_records(records: &[BrickRecord], blocks_per_cell: u32) -> Self {
        let blocks_per_cell = blocks_per_cell.max(1);
        let cell_size = blocks_per_cell as i64;
        let mut cell_keys: Vec<u64> = records
            .iter()
            .map(|record| {
                let block = unpack_world_block_key(record.packed_world_block_key);
                let cell = [
                    block[0].div_euclid(cell_size),
                    block[1].div_euclid(cell_size),
                    block[2].div_euclid(cell_size),
                ];
                pack_world_block_key(cell)
            })
            .collect();
        cell_keys.sort_unstable();
        cell_keys.dedup();
        ClipmapLevel {
            blocks_per_cell,
            cell_keys,
        }
    }
}

/// The three-level clip-map pyramid (L1 = 8-block cells, L2 = 64-block cells, L3
/// = 512-block cells; ADR 0011 Decision 4a + G4). A derived, rebuildable min-mip
/// of the brick records — never truth (ADR 0006/0009 4c). The DDA descends the
/// levels coarsest-first (L3 → L2 → L1) via [`Self::levels_coarse_to_fine`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClipmapPyramid {
    /// Fine level (8-block cells).
    pub level_1: ClipmapLevel,
    /// Middle level (64-block cells).
    pub level_2: ClipmapLevel,
    /// Coarse level (512-block cells) — checked first by the hierarchical DDA.
    pub level_3: ClipmapLevel,
}

impl ClipmapPyramid {
    /// Build all levels from a brick-field's sorted records (a pure function of
    /// the record keys — the sink derives it next to the record set).
    pub fn from_records(records: &[BrickRecord]) -> Self {
        ClipmapPyramid {
            level_1: ClipmapLevel::from_records(records, CLIPMAP_LEVEL_1_BLOCKS_PER_CELL),
            level_2: ClipmapLevel::from_records(records, CLIPMAP_LEVEL_2_BLOCKS_PER_CELL),
            level_3: ClipmapLevel::from_records(records, CLIPMAP_LEVEL_3_BLOCKS_PER_CELL),
        }
    }

    /// The "pyramid off" form — every level empty, so the shader's hierarchical
    /// skip never fires (the flat G1 block-DDA). Used by the pyramid-on == off
    /// parity assertion and the perf probe's baseline.
    pub fn empty() -> Self {
        ClipmapPyramid {
            level_1: ClipmapLevel::empty(CLIPMAP_LEVEL_1_BLOCKS_PER_CELL),
            level_2: ClipmapLevel::empty(CLIPMAP_LEVEL_2_BLOCKS_PER_CELL),
            level_3: ClipmapLevel::empty(CLIPMAP_LEVEL_3_BLOCKS_PER_CELL),
        }
    }

    /// The levels ordered COARSEST → FINEST (L3, L2, L1) — the order the
    /// hierarchical DDA descends (skip by the coarsest empty level covering the
    /// ray's block). The CPU march mirror ([`crate::brick_raymarch::cpu_march_brick_field`])
    /// and the perf probe iterate this slice.
    pub fn levels_coarse_to_fine(&self) -> [&ClipmapLevel; 3] {
        [&self.level_3, &self.level_2, &self.level_1]
    }
}

/// Split a level's sorted u64 cell keys into the `(hi, lo)` u32 pairs the WGSL
/// binary search consumes (no u64 in WGSL) — the pyramid analogue of
/// `pack_gpu_records`' key split.
pub fn pack_clipmap_level_keys(level: &ClipmapLevel) -> Vec<[u32; 2]> {
    level
        .cell_keys
        .iter()
        .map(|&key| [(key >> 32) as u32, key as u32])
        .collect()
}

/// What a brick holds — ADR 0011 Decision 2's two record kinds. The enum makes
/// "a coarse record consumes no atlas slot" structural, not a convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrickPayload {
    /// **Kind 0** — an analytic coarse brick: the whole block is solid at `block_id`,
    /// stored as this one record with no per-voxel data (interior elision on the GPU;
    /// also the residency-miss fallback form the G1 contract renders).
    CoarseSolid { block_id: BlockId },
    /// **Kind 1** — a sculpted brick: the block's voxel occupancy lives in atlas slot
    /// `atlas_slot` (an `edge³` R8 tile, edge = `voxels_per_block`).
    Sculpted { atlas_slot: u32 },
}

impl BrickPayload {
    /// The GPU-side record-kind discriminant (0 = coarse, 1 = sculpted). Pinned here —
    /// like `shape_kind_discriminant` — so a future enum reorder can't silently desync
    /// the G1 shader.
    pub fn kind_discriminant(&self) -> u32 {
        match self {
            BrickPayload::CoarseSolid { .. } => 0,
            BrickPayload::Sculpted { .. } => 1,
        }
    }
}

/// One resident brick: a non-air block of the two-layer boundary set, keyed for the
/// sorted-array binary search the G1 raymarch resolves residency with (ADR 0011 4b).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrickRecord {
    /// [`pack_world_block_key`] of the block's absolute world-block coordinate.
    pub packed_world_block_key: u64,
    /// The block's clean render-cell material colour index (`0..MATERIAL_COUNT`) — the
    /// `block_id`'s colour index for a coarse block, the (single) microblock material for
    /// a boundary block. The R8 atlas is occupancy-only (ADR 0011 G2), so this is the
    /// per-BLOCK material the raymarch shades with, packed into the GPU record's `kind`
    /// high bits by [`pack_gpu_records`]. A block that MIXES materials across its
    /// microblocks is not brick-representable (it never engages the sink), so this holds
    /// the first microblock's material there — unused, never shaded.
    pub material_id: u16,
    /// Coarse (kind 0) or sculpted (kind 1) — see [`BrickPayload`].
    pub payload: BrickPayload,
    /// Per-face seam-solidity flags, carried UNCHANGED from the boundary set for a
    /// sculpted brick. A coarse-solid block is solid through, so every face flag is
    /// `true` by construction (the block-DDA culls against it identically either way).
    pub seam_solidity: SeamSolidity,
}

/// The built brick field: the sorted record array + the sculpted-brick occupancy atlas
/// bytes in the ADR 0007 tile-cube layout (`bricks_per_axis³` slots of `edge³` texels,
/// linear slot index → 3D tile coord exactly as `upload_grid_per_chunk` packs fog
/// tiles). [`upload_brick_atlas`] lands the bytes in an R8 3D texture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrickFieldBuild {
    /// Every non-air block's record, sorted strictly ascending by
    /// `packed_world_block_key` (unique — a block is coarse XOR boundary).
    pub brick_records: Vec<BrickRecord>,
    /// `atlas_dim_voxels³` occupancy bytes (0 empty / 255 occupied), slot-packed;
    /// tile slots past the last sculpted brick stay all-zero.
    pub sculpted_atlas_bytes: Vec<u8>,
    /// The brick edge in voxels — `voxels_per_block`, the ONE-BLOCK granule
    /// (ADR 0011 Decision 1). Block-denominated: never a hard-coded voxel count.
    pub brick_edge_voxels: u32,
    /// Sculpted-brick tile slots per atlas axis (`ceil(cbrt(sculpted_count))`).
    pub bricks_per_axis: u32,
    /// `bricks_per_axis * brick_edge_voxels` — the atlas texture dimension per axis
    /// (0 when the build has no sculpted brick).
    pub atlas_dim_voxels: u32,
}

/// The occupancy byte a solid voxel packs to — the fog atlas's 0/255 R8 convention.
const SCULPTED_BRICK_OCCUPIED: u8 = 255;

impl BrickFieldBuild {
    /// Resolve the record for an absolute world-block coordinate by binary search over
    /// the sorted array — the CPU mirror of the in-shader residency lookup (ADR 0011
    /// 4b), and the parity harness's per-block accessor. `None` = air.
    pub fn find_record(&self, world_block: [i64; 3]) -> Option<&BrickRecord> {
        let key = pack_world_block_key(world_block);
        self.brick_records
            .binary_search_by_key(&key, |record| record.packed_world_block_key)
            .ok()
            .map(|index| &self.brick_records[index])
    }

    /// How many records are sculpted bricks (== atlas slots in use; slots are assigned
    /// densely `0..count`).
    pub fn sculpted_brick_count(&self) -> usize {
        self.brick_records
            .iter()
            .filter(|record| matches!(record.payload, BrickPayload::Sculpted { .. }))
            .count()
    }

    /// The low-corner texel of `atlas_slot`'s tile in the atlas cube (linear slot →
    /// 3D tile coord, x-fastest — the `upload_grid_per_chunk` tile layout).
    fn atlas_slot_origin_texels(&self, atlas_slot: u32) -> [usize; 3] {
        let tiles = self.bricks_per_axis.max(1);
        let tile = [
            atlas_slot % tiles,
            (atlas_slot / tiles) % tiles,
            atlas_slot / (tiles * tiles),
        ];
        let edge = self.brick_edge_voxels as usize;
        [
            tile[0] as usize * edge,
            tile[1] as usize * edge,
            tile[2] as usize * edge,
        ]
    }

    /// Copy one sculpted brick's `edge³` occupancy bytes out of the atlas (block-local
    /// x-fastest order — the order the boundary set's per-block occupancy expands in).
    pub fn sculpted_brick_occupancy(&self, atlas_slot: u32) -> Vec<u8> {
        let edge = self.brick_edge_voxels as usize;
        let atlas_dim = self.atlas_dim_voxels as usize;
        let origin = self.atlas_slot_origin_texels(atlas_slot);
        let mut brick_bytes = vec![0u8; edge * edge * edge];
        for local_z in 0..edge {
            for local_y in 0..edge {
                for local_x in 0..edge {
                    let atlas_index = ((origin[2] + local_z) * atlas_dim + origin[1] + local_y)
                        * atlas_dim
                        + origin[0]
                        + local_x;
                    brick_bytes[(local_z * edge + local_y) * edge + local_x] =
                        self.sculpted_atlas_bytes[atlas_index];
                }
            }
        }
        brick_bytes
    }
}

/// Build the brick field from a scene's two-layer boundary set (the
/// `build_covering_chunks` / resident-cache output): walk every chunk's block
/// partition, emit one record per non-air block, rasterize each boundary block's
/// cuboids into its atlas slot, and sort the records by packed world-block key.
///
/// `voxels_per_block` is the document density every chunk was built at (each chunk
/// carries it; a mismatch is a caller bug, asserted in debug).
pub fn build_brick_field(
    two_layer_chunks: &[([i32; 3], TwoLayerChunk)],
    voxels_per_block: u32,
) -> BrickFieldBuild {
    let brick_edge_voxels = voxels_per_block.max(1);
    let mut brick_records: Vec<BrickRecord> = Vec::new();
    // One `edge³` byte tile per sculpted brick, in slot order; scattered into the
    // atlas cube once the final count fixes the tile geometry.
    let mut sculpted_brick_tiles: Vec<Vec<u8>> = Vec::new();

    for (chunk_coord, chunk) in two_layer_chunks {
        debug_assert_eq!(
            chunk.voxels_per_block, brick_edge_voxels,
            "every chunk of one build shares the document density"
        );
        for block_z in 0..CHUNK_BLOCKS {
            for block_y in 0..CHUNK_BLOCKS {
                for block_x in 0..CHUNK_BLOCKS {
                    let block = [block_x, block_y, block_z];
                    let world_block = [
                        chunk_coord[0] as i64 * CHUNK_BLOCKS as i64 + block_x as i64,
                        chunk_coord[1] as i64 * CHUNK_BLOCKS as i64 + block_y as i64,
                        chunk_coord[2] as i64 * CHUNK_BLOCKS as i64 + block_z as i64,
                    ];
                    // Classify the block once (shared with the G3 incremental update so
                    // both paths emit identical records); the wholesale build assigns
                    // sculpted slots densely in record order.
                    match classify_block_brick(chunk, block, world_block, brick_edge_voxels) {
                        BlockBrick::Air => {}
                        BlockBrick::Coarse(record) => brick_records.push(record),
                        BlockBrick::Sculpted {
                            material_id,
                            seam_solidity,
                            tile,
                        } => {
                            let atlas_slot = sculpted_brick_tiles.len() as u32;
                            sculpted_brick_tiles.push(tile);
                            brick_records.push(BrickRecord {
                                packed_world_block_key: pack_world_block_key(world_block),
                                material_id,
                                payload: BrickPayload::Sculpted { atlas_slot },
                                seam_solidity,
                            });
                        }
                    }
                }
            }
        }
    }

    brick_records.sort_unstable_by_key(|record| record.packed_world_block_key);
    debug_assert!(
        brick_records
            .windows(2)
            .all(|pair| pair[0].packed_world_block_key < pair[1].packed_world_block_key),
        "brick keys must be unique (each world block appears in exactly one chunk)"
    );

    // Tile geometry mirrors `upload_grid_per_chunk`: a cubic-ish slot grid bounded by
    // the SCULPTED count (coarse records consume none of it), then scatter each tile.
    let (bricks_per_axis, atlas_dim_voxels, sculpted_atlas_bytes) =
        pack_sculpted_atlas(&sculpted_brick_tiles, brick_edge_voxels);

    BrickFieldBuild {
        brick_records,
        sculpted_atlas_bytes,
        brick_edge_voxels,
        bricks_per_axis,
        atlas_dim_voxels,
    }
}

/// Whether a world-block coordinate fits the packed-key lanes ([`pack_world_block_key`]
/// asserts otherwise). Used to skip an out-of-range NEIGHBOUR probe without panicking —
/// a neighbour one block past a valid block only exceeds the lane at the ±2^20 extreme,
/// which no target scene reaches, but a display-path pass must never panic on it.
fn world_block_in_key_range(world_block: [i64; 3]) -> bool {
    world_block.iter().all(|&coordinate| {
        let biased = coordinate + WORLD_BLOCK_KEY_BIAS;
        (0..(1i64 << WORLD_BLOCK_KEY_BITS_PER_AXIS)).contains(&biased)
    })
}

/// **Interior-elision mask for the DISPLAY record buffer (ADR 0011 — the brick sink's
/// analogue of the mesh's interior-face culling and the sketch producer's coarse-solid
/// elision).** Returns a per-record `keep` flag over the full, sorted `records`: `true`
/// for a block a ray could reach, `false` for one whose SIX face-neighbours are each
/// present AND solid on the shared face — a fully-occluded interior block.
///
/// Such a block is never a ray's first hit: the block-DDA
/// ([`cpu_march_brick_field`](crate::brick_raymarch::cpu_march_brick_field)) returns at
/// the FIRST block carrying a record, and a ray reaching an interior block must first pass
/// through the solid neighbour surrounding it (which keeps its record). So eliding the
/// `false` records from the shader's record buffer is **hit-identical** — proven in
/// `tests/gpu_parity.rs::brick_surface_elision_hit_set_unchanged`. The clip-map, atlas and
/// fog keep the FULL set; only the per-edit record buffer the shader binary-searches
/// shrinks (∝ surface, not volume, for a large solid).
///
/// **Conservative direction:** a neighbour that is ABSENT (air) or only PARTIALLY solid on
/// the shared face keeps the block. The mask is thus always a superset of the truly-visible
/// blocks — it can never drop a block a ray can see, only ever an unreachable interior one.
pub fn surface_record_mask(records: &[BrickRecord]) -> Vec<bool> {
    use std::collections::HashMap;
    let index: HashMap<u64, usize> = records
        .iter()
        .enumerate()
        .map(|(record_index, record)| (record.packed_world_block_key, record_index))
        .collect();
    // Is the neighbour of `block` across `(axis, delta)` present AND solid on the face it
    // shares with `block` — i.e. its `facing_side` (the opposite side to `block`'s)?
    let neighbour_face_solid = |block: [i64; 3], axis: usize, delta: i64, facing_side: usize| {
        let mut neighbour = block;
        neighbour[axis] += delta;
        world_block_in_key_range(neighbour)
            && index
                .get(&pack_world_block_key(neighbour))
                .is_some_and(|&i| records[i].seam_solidity.face_is_solid(axis, facing_side))
    };
    records
        .iter()
        .map(|record| {
            let block = unpack_world_block_key(record.packed_world_block_key);
            // Occluded ⇔ each axis is capped on BOTH sides: the +1 neighbour's LOW face
            // covers this block's HIGH face, and the −1 neighbour's HIGH face covers its LOW.
            let occluded = (0..3).all(|axis| {
                neighbour_face_solid(block, axis, 1, 0) && neighbour_face_solid(block, axis, -1, 1)
            });
            !occluded
        })
        .collect()
}

/// Rasterize one boundary block's cuboids into an `edge³` occupancy tile (0/255,
/// block-local x-fastest). Occupancy only: the cuboid `material_id` render-cell key
/// (id + overlay bit) never enters the R8 payload — any voxel a cuboid covers is 255.
fn rasterize_brick_occupancy(
    geometry: &crate::two_layer_store::MicroblockGeometry,
    brick_edge_voxels: u32,
) -> Vec<u8> {
    let edge = brick_edge_voxels as usize;
    let mut brick_bytes = vec![0u8; edge * edge * edge];
    for cuboid in &geometry.cuboids {
        for voxel_z in cuboid.min[2]..=cuboid.max[2] {
            for voxel_y in cuboid.min[1]..=cuboid.max[1] {
                let row = (voxel_z as usize * edge + voxel_y as usize) * edge;
                brick_bytes[row + cuboid.min[0] as usize..=row + cuboid.max[0] as usize]
                    .fill(SCULPTED_BRICK_OCCUPIED);
            }
        }
    }
    brick_bytes
}

/// One block's brick contribution, INDEPENDENT of atlas-slot assignment — the shared
/// classifier both the wholesale [`build_brick_field`] and the G3 incremental update
/// ([`IncrementalBrickField::apply_dirty_update`]) run, so a block classifies to the
/// exact same record kind + material + occupancy either way (only the slot NUMBER
/// differs: wholesale packs `0..count` in record order, incremental allocates from a
/// free-list). Keeping ONE classifier is what makes "incremental == wholesale byte-exact"
/// structural rather than a convention two code paths must independently uphold.
enum BlockBrick {
    /// Air — no record (ADR 0011 Decision 2).
    Air,
    /// A coarse-solid block: the whole record (no atlas slot).
    Coarse(BrickRecord),
    /// A boundary block: the record MINUS its atlas slot (the caller's allocator assigns
    /// it) plus the occupancy tile to land in that slot.
    Sculpted {
        material_id: u16,
        seam_solidity: SeamSolidity,
        tile: Vec<u8>,
    },
}

/// Classify one block of a [`TwoLayerChunk`] into its [`BlockBrick`] — the coarse XOR
/// boundary XOR air partition (ADR 0011 Decision 2). `world_block` is the block's
/// absolute world-block coordinate (its packed key).
fn classify_block_brick(
    chunk: &TwoLayerChunk,
    block: [u32; 3],
    world_block: [i64; 3],
    brick_edge_voxels: u32,
) -> BlockBrick {
    if let Some(block_id) = chunk.coarse_block(block) {
        // Coarse XOR boundary is the classifier's invariant; a block in both layers
        // would double-emit its key.
        debug_assert!(
            !chunk.microblocks.contains_key(&block),
            "a block must be coarse XOR boundary"
        );
        BlockBrick::Coarse(BrickRecord {
            packed_world_block_key: pack_world_block_key(world_block),
            material_id: block_id.color_index(),
            payload: BrickPayload::CoarseSolid { block_id },
            // Fully solid through ⇒ every face is solid.
            seam_solidity: SeamSolidity {
                solid: [[true; 2]; 3],
            },
        })
    } else if let Some(geometry) = chunk.microblocks.get(&block) {
        // The block's material is the clean render-cell id of its microblocks; a
        // representable block is single-material, so the first cuboid's id is the
        // block's (a mixed block never engages the sink).
        let material_id = geometry
            .cuboids
            .first()
            .map(|cuboid| clean_block_id(cuboid.material_id))
            .unwrap_or(0);
        BlockBrick::Sculpted {
            material_id,
            seam_solidity: geometry.seam_solidity,
            tile: rasterize_brick_occupancy(geometry, brick_edge_voxels),
        }
    } else {
        BlockBrick::Air
    }
}

/// Scatter a slot-indexed set of `edge³` occupancy tiles into the ADR 0007 tile-cube
/// atlas layout: a cubic-ish `bricks_per_axis³` slot grid (bounded by the slot count,
/// linear slot → 3D tile x-fastest), returning `(bricks_per_axis, atlas_dim_voxels,
/// bytes)`. Shared by the wholesale build and [`IncrementalBrickField::to_build`] so the
/// two produce byte-identical layouts for the same tile vector. A slot with a FREED
/// (dead) tile is scattered as-is — its bytes are unreachable from any live record, so
/// they may be garbage (the free-slot discipline).
fn pack_sculpted_atlas(slot_tiles: &[Vec<u8>], brick_edge_voxels: u32) -> (u32, u32, Vec<u8>) {
    let edge = brick_edge_voxels as usize;
    let slot_count = slot_tiles.len();
    let (bricks_per_axis, atlas_dim_voxels) = if slot_count == 0 {
        (0, 0)
    } else {
        let tiles = ((slot_count as f64).cbrt().ceil() as u32).max(1);
        (tiles, tiles * brick_edge_voxels)
    };
    let atlas_dim = atlas_dim_voxels as usize;
    let mut bytes = vec![0u8; atlas_dim * atlas_dim * atlas_dim];
    for (slot, tile) in slot_tiles.iter().enumerate() {
        debug_assert_eq!(tile.len(), edge * edge * edge, "every slot tile is edge³");
        let tiles = bricks_per_axis;
        let s = slot as u32;
        let origin = [
            (s % tiles) as usize * edge,
            ((s / tiles) % tiles) as usize * edge,
            (s / (tiles * tiles)) as usize * edge,
        ];
        for local_z in 0..edge {
            for local_y in 0..edge {
                let source_row = (local_z * edge + local_y) * edge;
                let atlas_row = ((origin[2] + local_z) * atlas_dim + origin[1] + local_y)
                    * atlas_dim
                    + origin[0];
                bytes[atlas_row..atlas_row + edge]
                    .copy_from_slice(&tile[source_row..source_row + edge]);
            }
        }
    }
    (bricks_per_axis, atlas_dim_voxels, bytes)
}

/// The absolute CHUNK coordinate that owns an absolute world block (`floor_div` by
/// [`CHUNK_BLOCKS`]) — the partition the resident cache dirties on, so a record can be
/// tested for membership in an edit's dirty-chunk set.
fn chunk_coord_of_world_block(world_block: [i64; 3]) -> [i32; 3] {
    let blocks = CHUNK_BLOCKS as i64;
    [
        world_block[0].div_euclid(blocks) as i32,
        world_block[1].div_euclid(blocks) as i32,
        world_block[2].div_euclid(blocks) as i32,
    ]
}

/// What an [`IncrementalBrickField::apply_dirty_update`] touched — the per-edit "dirty
/// region" made observable so the GPU sink patches ONLY these atlas slots (never the
/// untouched ones) and the parity net can assert the cost is proportional to the edit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BrickFieldUpdate {
    /// Atlas slots (re)written this edit — newly allocated or overwritten sculpted
    /// bricks. When `atlas_grew` is false these are the ONLY slots the GPU patch writes.
    pub written_slots: Vec<u32>,
    /// Slots FREED this edit (their block became air/coarse or its chunk was removed);
    /// their tiles are now dead until reallocated. Free bytes are never uploaded.
    pub freed_slots: Vec<u32>,
    /// Whether the atlas tile geometry GREW (`bricks_per_axis` increased) — then every
    /// slot's 3D position moved, so the sink MUST re-pack + re-upload the whole atlas
    /// (the one legitimate wholesale re-pack, ADR 0011 pitfalls / ADR 0007 resize
    /// precedent). False ⇒ untouched slots keep their texels.
    pub atlas_grew: bool,
}

/// The PERSISTENT incremental brick field (ADR 0011 slice G3). Maintains the sorted
/// [`BrickRecord`] array + a slot-allocated atlas ACROSS edits so a per-edit update
/// re-evaluates only the DIRTY chunks' blocks and patches only their slots — the
/// "per-edit cost proportional to the dirty region, not the scene" win ADR 0009 promised.
///
/// Slots are managed by a **free-list** (allocate on a new sculpted brick, free when a
/// brick becomes air/coarse or its chunk is dirtied away), so slot numbers are STABLE
/// across edits and differ from the wholesale build's dense `0..count`. The invariant the
/// parity gate proves: after any edit, every LIVE record's slot bytes equal a from-scratch
/// [`build_brick_field`] of the same scene (free slots may hold garbage — they are
/// unreachable). The pyramid is REBUILT (not patched) from the merged record keys per
/// edit (a cheap pure function; incremental pyramid patching is deferred to G4).
#[derive(Debug, Clone)]
pub struct IncrementalBrickField {
    /// The brick edge in voxels (`voxels_per_block`, the ONE-BLOCK granule) — fixed for
    /// the field's life (a density change resets the field via a wholesale rebuild).
    brick_edge_voxels: u32,
    /// Records sorted strictly ascending by packed world-block key — the same order and
    /// content [`build_brick_field`] emits, only the sculpted records' slot NUMBERS differ.
    records: Vec<BrickRecord>,
    /// Per-slot occupancy tiles (`edge³` bytes each), indexed by atlas slot. A FREED
    /// slot's entry is retained (kept `edge³` so the atlas packer never trips) but is
    /// unreferenced — dead bytes until the slot is reallocated.
    slot_tiles: Vec<Vec<u8>>,
    /// Reusable slot indices freed by removed / transitioned sculpted bricks — the
    /// free-list. A new sculpted brick pops from here before growing `slot_tiles`.
    free_slots: Vec<u32>,
}

impl IncrementalBrickField {
    /// Seed the incremental field from a wholesale [`build_brick_field`] (the reset a
    /// scene load / density change / gate re-engagement performs). Slots are the build's
    /// dense `0..sculpted_count`; the free-list starts empty.
    pub fn from_wholesale(build: &BrickFieldBuild) -> Self {
        let sculpted_count = build.sculpted_brick_count();
        let slot_tiles: Vec<Vec<u8>> = (0..sculpted_count as u32)
            .map(|slot| build.sculpted_brick_occupancy(slot))
            .collect();
        Self {
            brick_edge_voxels: build.brick_edge_voxels,
            records: build.brick_records.clone(),
            slot_tiles,
            free_slots: Vec::new(),
        }
    }

    /// The brick edge (voxels_per_block) the field is bound to.
    pub fn brick_edge_voxels(&self) -> u32 {
        self.brick_edge_voxels
    }

    /// The live record count (coarse + sculpted).
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// The atlas slot high-water mark (live + freed slots) — the tile count the atlas is
    /// sized to address. `>= ` the live sculpted count (holes from freed slots).
    pub fn slot_high_water(&self) -> usize {
        self.slot_tiles.len()
    }

    /// Re-evaluate ONLY the blocks of the dirty chunks and merge them into the field.
    ///
    /// * `fresh_chunks` — the FULL current covering set (dirty chunks freshly resolved,
    ///   clean chunks reused verbatim). Only the dirty chunks are read.
    /// * `dirty_chunks` — the chunk coords the edit invalidated
    ///   ([`TwoLayerResidentCache::invalidate_aabb`](crate::two_layer_store::TwoLayerResidentCache::invalidate_aabb)
    ///   evicted). Every changed block lives in one of these (a block's record is
    ///   intrinsic — seam flags included — so no neighbour dilation is needed, unlike the
    ///   mesh's cross-chunk seam culling).
    ///
    /// Removes every previous record whose block is in a dirty chunk (freeing its
    /// sculpted slot), rebuilds those chunks' records fresh (allocating slots from the
    /// free-list), and re-sorts. Returns the [`BrickFieldUpdate`] describing exactly which
    /// slots were touched (the GPU patch's work-list).
    pub fn apply_dirty_update(
        &mut self,
        fresh_chunks: &[([i32; 3], TwoLayerChunk)],
        dirty_chunks: &[[i32; 3]],
    ) -> BrickFieldUpdate {
        let edge = self.brick_edge_voxels;
        let dirty: std::collections::BTreeSet<[i32; 3]> = dirty_chunks.iter().copied().collect();
        let previous_bricks_per_axis = sculpted_atlas_bricks_per_axis(self.slot_tiles.len());

        // 1. Drop every previous record whose block is in a dirty chunk, freeing its slot.
        let mut freed_slots = Vec::new();
        self.records.retain(|record| {
            let chunk =
                chunk_coord_of_world_block(unpack_world_block_key(record.packed_world_block_key));
            if dirty.contains(&chunk) {
                if let BrickPayload::Sculpted { atlas_slot } = record.payload {
                    freed_slots.push(atlas_slot);
                }
                false
            } else {
                true
            }
        });
        // Freed slots return to the pool (ascending pop order keeps allocation
        // deterministic — a nicety for test readability, not correctness).
        self.free_slots.extend(freed_slots.iter().copied());
        self.free_slots.sort_unstable();
        self.free_slots.dedup();

        // 2. Rebuild every dirty chunk's records from its FRESH data.
        let mut written_slots = Vec::new();
        for (chunk_coord, chunk) in fresh_chunks {
            if !dirty.contains(chunk_coord) {
                continue;
            }
            for block_z in 0..CHUNK_BLOCKS {
                for block_y in 0..CHUNK_BLOCKS {
                    for block_x in 0..CHUNK_BLOCKS {
                        let block = [block_x, block_y, block_z];
                        let world_block = [
                            chunk_coord[0] as i64 * CHUNK_BLOCKS as i64 + block_x as i64,
                            chunk_coord[1] as i64 * CHUNK_BLOCKS as i64 + block_y as i64,
                            chunk_coord[2] as i64 * CHUNK_BLOCKS as i64 + block_z as i64,
                        ];
                        match classify_block_brick(chunk, block, world_block, edge) {
                            BlockBrick::Air => {}
                            BlockBrick::Coarse(record) => self.records.push(record),
                            BlockBrick::Sculpted {
                                material_id,
                                seam_solidity,
                                tile,
                            } => {
                                let slot = self.allocate_slot(tile);
                                written_slots.push(slot);
                                self.records.push(BrickRecord {
                                    packed_world_block_key: pack_world_block_key(world_block),
                                    material_id,
                                    payload: BrickPayload::Sculpted { atlas_slot: slot },
                                    seam_solidity,
                                });
                            }
                        }
                    }
                }
            }
        }

        // 3. Re-sort (O(n log n) over records — trivially small next to atlas work).
        self.records
            .sort_unstable_by_key(|record| record.packed_world_block_key);
        debug_assert!(
            self.records
                .windows(2)
                .all(|pair| pair[0].packed_world_block_key < pair[1].packed_world_block_key),
            "brick keys must stay unique + sorted after an incremental merge"
        );

        let atlas_grew =
            sculpted_atlas_bricks_per_axis(self.slot_tiles.len()) != previous_bricks_per_axis;
        BrickFieldUpdate {
            written_slots,
            freed_slots,
            atlas_grew,
        }
    }

    /// Allocate a slot for a fresh sculpted tile: reuse a freed slot if one is available
    /// (keeping the high-water mark — and thus the atlas — from growing needlessly),
    /// else append a new slot.
    fn allocate_slot(&mut self, tile: Vec<u8>) -> u32 {
        match self.free_slots.pop() {
            Some(slot) => {
                self.slot_tiles[slot as usize] = tile;
                slot
            }
            None => {
                let slot = self.slot_tiles.len() as u32;
                self.slot_tiles.push(tile);
                slot
            }
        }
    }

    /// Materialise the current field as a [`BrickFieldBuild`] (records + packed atlas) —
    /// the form the GPU install / full re-upload and the parity net consume. The atlas is
    /// sized to the slot high-water mark (live + freed holes), so a live record's slot
    /// bytes are always in range.
    pub fn to_build(&self) -> BrickFieldBuild {
        let (bricks_per_axis, atlas_dim_voxels, sculpted_atlas_bytes) =
            pack_sculpted_atlas(&self.slot_tiles, self.brick_edge_voxels);
        BrickFieldBuild {
            brick_records: self.records.clone(),
            sculpted_atlas_bytes,
            brick_edge_voxels: self.brick_edge_voxels,
            bricks_per_axis,
            atlas_dim_voxels,
        }
    }
}

/// The `bricks_per_axis` a slot-tile count packs to (`ceil(cbrt(count))`, 0 for empty) —
/// the atlas tile-grid edge, shared by the packer and the grow test.
fn sculpted_atlas_bricks_per_axis(slot_count: usize) -> u32 {
    if slot_count == 0 {
        0
    } else {
        ((slot_count as f64).cbrt().ceil() as u32).max(1)
    }
}

/// Land the sculpted-brick atlas bytes in an R8Unorm 3D texture — the shipped fog-atlas
/// upload mechanic (`upload_grid_per_chunk`'s `write_texture`, no row padding needed).
/// `COPY_SRC` is set so the parity net can read the texture back; a build with no
/// sculpted brick returns a 1³ placeholder (nothing samples it — every record is
/// coarse/air).
pub fn upload_brick_atlas(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    build: &BrickFieldBuild,
) -> wgpu::Texture {
    let atlas_dim = build.atlas_dim_voxels.max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("brick-field sculpted atlas"),
        size: wgpu::Extent3d {
            width: atlas_dim,
            height: atlas_dim,
            depth_or_array_layers: atlas_dim,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    if build.atlas_dim_voxels > 0 {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &build.sculpted_atlas_bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(atlas_dim),
                rows_per_image: Some(atlas_dim),
            },
            wgpu::Extent3d {
                width: atlas_dim,
                height: atlas_dim,
                depth_or_array_layers: atlas_dim,
            },
        );
    }
    texture
}

/// Read an `atlas_dim³` R8 atlas texture back to row-unpadded bytes — the parity net's
/// A/B readback ONLY (mirrors `dispatch_atlas`; per ADR 0006 §4 nothing ever reads a
/// texture back as truth on a live path).
pub fn read_back_brick_atlas(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    atlas_dim: u32,
) -> Vec<u8> {
    if atlas_dim == 0 {
        return Vec::new();
    }
    // `copy_texture_to_buffer` rows must be 256-aligned (unlike `write_texture`).
    const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;
    let padded_row = atlas_dim.div_ceil(COPY_BYTES_PER_ROW_ALIGNMENT) * COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes = padded_row as u64 * atlas_dim as u64 * atlas_dim as u64;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("brick-field atlas readback"),
        size: padded_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(atlas_dim),
            },
        },
        wgpu::Extent3d {
            width: atlas_dim,
            height: atlas_dim,
            depth_or_array_layers: atlas_dim,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll failed");
    receiver
        .recv()
        .expect("map_async channel dropped")
        .expect("buffer map failed");

    let mapped = slice.get_mapped_range();
    let atlas_dim_usize = atlas_dim as usize;
    let padded_row_usize = padded_row as usize;
    let mut atlas_bytes = vec![0u8; atlas_dim_usize.pow(3)];
    for atlas_z in 0..atlas_dim_usize {
        for atlas_y in 0..atlas_dim_usize {
            let source = (atlas_z * atlas_dim_usize + atlas_y) * padded_row_usize;
            let destination = (atlas_z * atlas_dim_usize + atlas_y) * atlas_dim_usize;
            atlas_bytes[destination..destination + atlas_dim_usize]
                .copy_from_slice(&mapped[source..source + atlas_dim_usize]);
        }
    }
    drop(mapped);
    readback.unmap();
    atlas_bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_geom::MaterialChoice;
    use crate::scene::Scene;
    use crate::two_layer_store::TwoLayerStore;
    use crate::voxel::{GeometryParams, ShapeKind, Voxel};

    #[test]
    fn world_block_key_round_trips_and_orders_z_major() {
        let coordinates = [
            [0i64, 0, 0],
            [-1, -2, -3],
            [17, -300, 4096],
            [-(1 << 19), (1 << 19), 0],
        ];
        for &world_block in &coordinates {
            assert_eq!(
                unpack_world_block_key(pack_world_block_key(world_block)),
                world_block
            );
        }
        // Integer key order is (z, y, x) lexicographic — the sort the shader's
        // binary search relies on.
        assert!(pack_world_block_key([5, 0, 0]) < pack_world_block_key([0, 1, 0]));
        assert!(pack_world_block_key([0, 5, 0]) < pack_world_block_key([0, 0, 1]));
        assert!(pack_world_block_key([-1, 0, 0]) < pack_world_block_key([0, 0, 0]));
    }

    /// A gated scene's brick set maps the two-layer partition one-to-one: coarse-solid
    /// → one kind-0 record (id carried, no slot), boundary → one kind-1 record (dense
    /// unique slots, seam flags carried unchanged), air → nothing; records sorted
    /// strictly ascending. This is the CPU half of the ADR 0011 gate clause (a); the
    /// `--features gpu` parity test re-asserts the bytes through the texture round-trip.
    #[test]
    fn brick_records_map_two_layer_partition_one_to_one() {
        // d4 deliberately (ADR 0011 Decision 1): the brick edge must follow the
        // density, not the number 16; odd voxel extents give partial boundary blocks.
        let voxels_per_block = 4;
        let scene = Scene::from_geometry(
            GeometryParams {
                shape: ShapeKind::Sphere,
                size_voxels: [33, 33, 33],
                size_measurements: None,
                voxels_per_block,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        );
        let two_layer_chunks =
            TwoLayerStore::enabled().build_covering_chunks(&scene, voxels_per_block, 0);
        let build = build_brick_field(&two_layer_chunks, voxels_per_block);

        assert_eq!(build.brick_edge_voxels, voxels_per_block);
        assert!(
            build
                .brick_records
                .windows(2)
                .all(|pair| pair[0].packed_world_block_key < pair[1].packed_world_block_key),
            "records must be sorted strictly ascending (unique keys)"
        );

        let mut expected_coarse = 0usize;
        let mut expected_sculpted = 0usize;
        let mut seen_slots = std::collections::BTreeSet::new();
        for (chunk_coord, chunk) in &two_layer_chunks {
            for block_z in 0..CHUNK_BLOCKS {
                for block_y in 0..CHUNK_BLOCKS {
                    for block_x in 0..CHUNK_BLOCKS {
                        let block = [block_x, block_y, block_z];
                        let world_block = [
                            chunk_coord[0] as i64 * CHUNK_BLOCKS as i64 + block_x as i64,
                            chunk_coord[1] as i64 * CHUNK_BLOCKS as i64 + block_y as i64,
                            chunk_coord[2] as i64 * CHUNK_BLOCKS as i64 + block_z as i64,
                        ];
                        let record = build.find_record(world_block);
                        if let Some(block_id) = chunk.coarse_block(block) {
                            expected_coarse += 1;
                            let record = record.expect("coarse-solid block must have a record");
                            assert_eq!(record.payload.kind_discriminant(), 0);
                            assert_eq!(
                                record.payload,
                                BrickPayload::CoarseSolid { block_id },
                                "coarse record carries the block id, no atlas slot"
                            );
                            assert_eq!(record.seam_solidity.solid, [[true; 2]; 3]);
                        } else if let Some(geometry) = chunk.microblocks.get(&block) {
                            expected_sculpted += 1;
                            let record = record.expect("boundary block must have a record");
                            assert_eq!(record.payload.kind_discriminant(), 1);
                            let BrickPayload::Sculpted { atlas_slot } = record.payload else {
                                panic!("boundary block must be a sculpted record");
                            };
                            assert!(
                                seen_slots.insert(atlas_slot),
                                "atlas slot {atlas_slot} assigned twice"
                            );
                            assert_eq!(
                                record.seam_solidity, geometry.seam_solidity,
                                "seam-solidity flags must carry across unchanged"
                            );
                        } else {
                            assert!(record.is_none(), "air block must emit nothing");
                        }
                    }
                }
            }
        }
        assert_eq!(build.brick_records.len(), expected_coarse + expected_sculpted);
        assert_eq!(build.sculpted_brick_count(), expected_sculpted);
        // Slots are dense 0..count — the atlas holds exactly the sculpted bricks.
        assert_eq!(
            seen_slots.iter().copied().collect::<Vec<_>>(),
            (0..expected_sculpted as u32).collect::<Vec<_>>()
        );
        // The scene must actually exercise both kinds, else the mapping is untested.
        assert!(expected_coarse > 0, "fixture must contain coarse-solid blocks");
        assert!(expected_sculpted > 0, "fixture must contain boundary blocks");
    }

    /// **Interior elision, CPU tier (ADR 0011 — the display record buffer).**
    /// [`surface_record_mask`] over a SOLID box keeps exactly the surface blocks (a block
    /// with ≥1 absent/air neighbour) and drops the strictly-interior ones (all six
    /// neighbours present + solid). Checked against an independent neighbour-count oracle;
    /// the `--features gpu` [`brick_surface_elision_hit_set_unchanged`] proves the elided
    /// buffer renders the same hit set.
    #[test]
    fn surface_record_mask_drops_fully_occluded_interior_of_a_solid_box() {
        let voxels_per_block = 4;
        // A solid box (ShapeKind::Box ignores wall_blocks — that is Tube-only), 6 blocks
        // per axis, so there is a genuine 4×4×4 fully-occluded interior to elide.
        let scene = Scene::from_geometry(
            GeometryParams {
                shape: ShapeKind::Box,
                size_voxels: [6 * voxels_per_block, 6 * voxels_per_block, 6 * voxels_per_block],
                size_measurements: None,
                voxels_per_block,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        );
        let two_layer_chunks =
            TwoLayerStore::enabled().build_covering_chunks(&scene, voxels_per_block, 0);
        let build = build_brick_field(&two_layer_chunks, voxels_per_block);
        assert!(!build.brick_records.is_empty(), "fixture must build records");
        // Every block of a solid box is coarse-solid (all faces solid).
        assert!(
            build
                .brick_records
                .iter()
                .all(|r| r.seam_solidity.solid == [[true; 2]; 3]),
            "a solid box classifies every block coarse-solid"
        );

        let mask = surface_record_mask(&build.brick_records);
        assert_eq!(mask.len(), build.brick_records.len());

        // Independent oracle: with all blocks coarse-solid, a block is INTERIOR iff all six
        // of its neighbours are present in the record set.
        let keys: std::collections::HashSet<u64> =
            build.brick_records.iter().map(|r| r.packed_world_block_key).collect();
        let mut expected_surface = 0usize;
        for (record, &keep) in build.brick_records.iter().zip(&mask) {
            let block = unpack_world_block_key(record.packed_world_block_key);
            let all_neighbours_present = [
                [1i64, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
            ]
            .iter()
            .all(|d| {
                let nb = [block[0] + d[0], block[1] + d[1], block[2] + d[2]];
                world_block_in_key_range(nb) && keys.contains(&pack_world_block_key(nb))
            });
            let expected_keep = !all_neighbours_present;
            assert_eq!(keep, expected_keep, "block {block:?} elision disagrees with the oracle");
            if keep {
                expected_surface += 1;
            }
        }
        let kept = mask.iter().filter(|&&k| k).count();
        assert_eq!(kept, expected_surface);
        // A solid box has a genuine interior to elide AND a surface to keep — the split is
        // non-trivial in both directions (else the elision would be vacuous or wrong).
        let dropped = mask.len() - kept;
        assert!(dropped > 0, "a solid box must have fully-occluded interior blocks to elide");
        assert!(kept > 0, "the surface blocks must be kept");
        assert!(kept < mask.len(), "not every block can be surface for a solid box");
    }

    /// The clip-map pyramid is CONSERVATIVE (ADR 0011 parity gate, coarse tier):
    /// each level's occupied-cell set is a SUPERSET of the true occupied cells
    /// (every record's cell present), sorted strictly ascending + unique, at ANY
    /// density (block-denominated cells — nothing hard-codes 16). A scattered
    /// multi-object scene so the levels actually span more than one cell.
    #[test]
    fn clipmap_pyramid_is_conservative_and_sorted() {
        use crate::{Node, NodeContent, NodeTransform};
        for &voxels_per_block in &[16u32, 4] {
            // A dozen small shapes far apart — the scattered scene the LOD targets.
            let mut nodes = Vec::new();
            for i in 0..12i64 {
                let shape = crate::voxel::SdfShape::from_blocks(
                    ShapeKind::Sphere,
                    [3, 3, 3],
                    1,
                    voxels_per_block,
                );
                let mut node = Node::new(
                    format!("s{i}"),
                    NodeContent::Tool {
                        shape,
                        material: MaterialChoice::Stone,
                    },
                );
                // Spread them ~16 blocks apart on a lattice so cells are scattered.
                node.transform = NodeTransform::from_blocks(
                    [(i % 4) * 16, (i / 4) * 16, (i % 3) * 20],
                    voxels_per_block,
                );
                nodes.push(node);
            }
            let scene = Scene::from_nodes(nodes);
            let two_layer_chunks =
                TwoLayerStore::enabled().build_covering_chunks(&scene, voxels_per_block, 0);
            let build = build_brick_field(&two_layer_chunks, voxels_per_block);
            assert!(!build.brick_records.is_empty());
            let pyramid = ClipmapPyramid::from_records(&build.brick_records);

            for (level, blocks_per_cell) in [
                (&pyramid.level_1, CLIPMAP_LEVEL_1_BLOCKS_PER_CELL),
                (&pyramid.level_2, CLIPMAP_LEVEL_2_BLOCKS_PER_CELL),
                (&pyramid.level_3, CLIPMAP_LEVEL_3_BLOCKS_PER_CELL),
            ] {
                assert_eq!(level.blocks_per_cell, blocks_per_cell);
                assert!(
                    level.cell_keys.windows(2).all(|pair| pair[0] < pair[1]),
                    "level {blocks_per_cell} keys must be sorted strictly ascending + unique"
                );
                // Truth: the cell of every record must be present (superset ⇒ the
                // DDA never strides past a real surface).
                let level_set: std::collections::BTreeSet<u64> =
                    level.cell_keys.iter().copied().collect();
                let cell_size = blocks_per_cell as i64;
                let mut true_cells = std::collections::BTreeSet::new();
                for record in &build.brick_records {
                    let b = unpack_world_block_key(record.packed_world_block_key);
                    let cell = [
                        b[0].div_euclid(cell_size),
                        b[1].div_euclid(cell_size),
                        b[2].div_euclid(cell_size),
                    ];
                    true_cells.insert(pack_world_block_key(cell));
                }
                assert!(
                    true_cells.is_subset(&level_set),
                    "level {blocks_per_cell} must cover every occupied cell (conservative)"
                );
                // The min-mip carries no cell the records don't (exactness of the
                // derivation — a spurious occupied cell would only cost perf, but
                // proves the fold has no stray keys).
                assert_eq!(level_set, true_cells);
                assert!(!level.cell_keys.is_empty());
            }
            // Each coarser level must not be finer than the one below (monotone
            // cell counts as the cell size grows 8× per level).
            assert!(pyramid.level_2.cell_keys.len() <= pyramid.level_1.cell_keys.len());
            assert!(pyramid.level_3.cell_keys.len() <= pyramid.level_2.cell_keys.len());
        }
    }

    /// CPU byte-exactness at a non-16 density: every sculpted brick's atlas bytes equal
    /// the block occupancy the SHIPPED expansion (`expand_occupancy_into`, itself
    /// proven bit-exact vs the dense oracle) reports — rasterization from cuboids and
    /// expansion are independent paths over the same boundary set.
    #[test]
    fn sculpted_brick_bytes_match_expanded_occupancy_at_non_16_density() {
        let voxels_per_block = 4;
        let scene = Scene::from_geometry(
            GeometryParams {
                shape: ShapeKind::Torus,
                size_voxels: [49, 13, 49],
                size_measurements: None,
                voxels_per_block,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        );
        let two_layer_chunks =
            TwoLayerStore::enabled().build_covering_chunks(&scene, voxels_per_block, 0);
        let build = build_brick_field(&two_layer_chunks, voxels_per_block);

        let edge = voxels_per_block as usize;
        let mut compared_bricks = 0usize;
        for (chunk_coord, chunk) in &two_layer_chunks {
            // Chunk-local occupancy bitmap via the shipped expansion (offset zero).
            let mut expanded: Vec<Voxel> = Vec::new();
            chunk.expand_occupancy_into(&mut expanded, [0, 0, 0]);
            let chunk_extent = (CHUNK_BLOCKS * voxels_per_block) as usize;
            let mut chunk_occupancy = vec![0u8; chunk_extent.pow(3)];
            for voxel in &expanded {
                let [x, y, z] = voxel.local_index;
                chunk_occupancy
                    [(z as usize * chunk_extent + y as usize) * chunk_extent + x as usize] =
                    SCULPTED_BRICK_OCCUPIED;
            }

            for block in chunk.microblocks.keys() {
                let world_block = [
                    chunk_coord[0] as i64 * CHUNK_BLOCKS as i64 + block[0] as i64,
                    chunk_coord[1] as i64 * CHUNK_BLOCKS as i64 + block[1] as i64,
                    chunk_coord[2] as i64 * CHUNK_BLOCKS as i64 + block[2] as i64,
                ];
                let record = build.find_record(world_block).expect("sculpted record");
                let BrickPayload::Sculpted { atlas_slot } = record.payload else {
                    panic!("boundary block must be sculpted");
                };
                let brick_bytes = build.sculpted_brick_occupancy(atlas_slot);
                let mut expected = vec![0u8; edge.pow(3)];
                for local_z in 0..edge {
                    for local_y in 0..edge {
                        for local_x in 0..edge {
                            let chunk_voxel = [
                                block[0] as usize * edge + local_x,
                                block[1] as usize * edge + local_y,
                                block[2] as usize * edge + local_z,
                            ];
                            expected[(local_z * edge + local_y) * edge + local_x] =
                                chunk_occupancy[(chunk_voxel[2] * chunk_extent
                                    + chunk_voxel[1])
                                    * chunk_extent
                                    + chunk_voxel[0]];
                        }
                    }
                }
                assert_eq!(
                    brick_bytes, expected,
                    "brick bytes must equal the expanded block occupancy at {world_block:?}"
                );
                compared_bricks += 1;
            }
        }
        assert!(compared_bricks > 0, "fixture must contain sculpted bricks");
    }
}

/// ADR 0011 slice G3 — the incremental dirty-brick atlas update net. The load-bearing
/// assertion: an [`IncrementalBrickField`] patched edit-by-edit (only dirty chunks
/// re-evaluated, slots free-listed) is byte-exact vs a from-scratch [`build_brick_field`]
/// of the SAME scene, after EVERY step, across explicit block-kind transitions
/// (air↔sculpted↔coarse) and add / move / recolour / delete edits.
#[cfg(test)]
mod incremental_tests {
    use super::*;
    use crate::core_geom::MaterialChoice;
    use crate::scene::{Node, NodeContent, NodeTransform, Scene};
    use crate::two_layer_store::{TwoLayerChunk, TwoLayerResidentCache};
    use crate::voxel::{ShapeKind, SdfShape};

    /// The owned covering set the shell feeds `apply_dirty_update` / `build_brick_field`
    /// (the resident cache borrows, so clone out — exactly as `AppCore::rebuild` does).
    fn covering_owned(
        cache: &mut TwoLayerResidentCache,
        scene: &Scene,
        density: u32,
    ) -> Vec<([i32; 3], TwoLayerChunk)> {
        cache
            .resident_two_layer_chunks(scene, density, 0)
            .into_iter()
            .map(|(coord, chunk)| (coord, chunk.clone()))
            .collect()
    }

    /// A tool node (single material, so the scene stays brick-representable) of `blocks³`
    /// at a block offset — the small edited object.
    fn tool(kind: ShapeKind, offset_blocks: [i64; 3], material: MaterialChoice, density: u32) -> Node {
        let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, density);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset_blocks, density);
        node
    }

    /// The set of atlas slots the live sculpted records reference, plus a check that no
    /// two live records share a slot (a "ghost brick" would show as a duplicate).
    fn live_slots(build: &BrickFieldBuild) -> std::collections::BTreeSet<u32> {
        let mut slots = std::collections::BTreeSet::new();
        for record in &build.brick_records {
            if let BrickPayload::Sculpted { atlas_slot } = record.payload {
                assert!(
                    slots.insert(atlas_slot),
                    "live slot {atlas_slot} referenced twice (ghost brick)"
                );
            }
        }
        slots
    }

    /// Assert the incremental field materialisation is byte-exact vs the wholesale build
    /// of the same scene: SAME record keys, kinds, materials, seam flags; each sculpted
    /// record's atlas bytes equal (slot NUMBERS differ — the free-list vs dense `0..count`
    /// — so compare the occupancy, not the slot). Free slots may hold garbage: they are
    /// asserted unreachable from live records (the `live_slots` uniqueness check).
    fn assert_incremental_matches_wholesale(
        incremental: &BrickFieldBuild,
        wholesale: &BrickFieldBuild,
        label: &str,
    ) {
        assert_eq!(
            incremental.brick_edge_voxels, wholesale.brick_edge_voxels,
            "[{label}] brick edge must match"
        );
        assert_eq!(
            incremental.brick_records.len(),
            wholesale.brick_records.len(),
            "[{label}] record count must match wholesale"
        );
        let _ = live_slots(incremental); // no ghost bricks (live slots unique)
        for whole_record in &wholesale.brick_records {
            let block = unpack_world_block_key(whole_record.packed_world_block_key);
            let inc_record = incremental
                .find_record(block)
                .unwrap_or_else(|| panic!("[{label}] incremental missing record at {block:?}"));
            assert_eq!(
                inc_record.packed_world_block_key, whole_record.packed_world_block_key,
                "[{label}] key mismatch at {block:?}"
            );
            assert_eq!(
                inc_record.material_id, whole_record.material_id,
                "[{label}] material mismatch at {block:?}"
            );
            assert_eq!(
                inc_record.seam_solidity, whole_record.seam_solidity,
                "[{label}] seam-solidity mismatch at {block:?}"
            );
            assert_eq!(
                inc_record.payload.kind_discriminant(),
                whole_record.payload.kind_discriminant(),
                "[{label}] kind mismatch at {block:?}"
            );
            match (inc_record.payload, whole_record.payload) {
                (
                    BrickPayload::CoarseSolid { block_id: a },
                    BrickPayload::CoarseSolid { block_id: b },
                ) => assert_eq!(a, b, "[{label}] coarse block id mismatch at {block:?}"),
                (
                    BrickPayload::Sculpted { atlas_slot: inc_slot },
                    BrickPayload::Sculpted { atlas_slot: whole_slot },
                ) => {
                    // Slot NUMBERS differ (free-list vs dense) — compare the bytes.
                    assert_eq!(
                        incremental.sculpted_brick_occupancy(inc_slot),
                        wholesale.sculpted_brick_occupancy(whole_slot),
                        "[{label}] sculpted occupancy bytes mismatch at {block:?}"
                    );
                }
                _ => panic!("[{label}] payload kind disagreement at {block:?}"),
            }
        }
    }

    /// THE parity gate for G3 (issue #69 acceptance): drive a scripted sequence of edits
    /// — recolour, move, shape-swap, delete, re-add — applying each INCREMENTALLY, and
    /// after every step assert the incremental field equals a from-scratch wholesale build
    /// of the same scene. Two fixed anchor tools at the extremes pin the covering set so an
    /// incremental edit never grows it (the app's reframe guard — a growth routes wholesale).
    /// A non-16 density exercises the block-denominated granule.
    #[test]
    fn incremental_dirty_update_equals_wholesale_after_every_step() {
        let density = 4u32;
        let material = MaterialChoice::Stone;
        // Two anchors far apart fix the covering chunk range; the middle tool is edited.
        let anchor_lo = tool(ShapeKind::Box, [-14, 0, 0], material, density);
        let anchor_hi = tool(ShapeKind::Box, [14, 0, 0], material, density);
        let scene_with = |middle: Option<Node>| {
            let mut nodes = vec![anchor_lo.clone(), anchor_hi.clone()];
            if let Some(m) = middle {
                nodes.push(m);
            }
            Scene::from_nodes(nodes)
        };

        // The scripted edits (each keeps the anchors, edits the middle) — chosen to force
        // block-kind transitions: add (air→sculpted/coarse), move (sculpted↔air↔coarse),
        // recolour (sculpted/coarse material change), shape-swap (occupancy change), delete.
        let scenes = [
            ("initial", scene_with(None)),
            ("add-sphere", scene_with(Some(tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Wood, density)))),
            ("recolour", scene_with(Some(tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Plain, density)))),
            ("move", scene_with(Some(tool(ShapeKind::Sphere, [2, 1, 0], MaterialChoice::Plain, density)))),
            ("shape-swap", scene_with(Some(tool(ShapeKind::Box, [2, 1, 0], MaterialChoice::Plain, density)))),
            ("delete", scene_with(None)),
            ("re-add", scene_with(Some(tool(ShapeKind::Torus, [0, 0, 0], MaterialChoice::Wood, density)))),
        ];

        let mut cache = TwoLayerResidentCache::enabled();
        cache.clear();
        let scene0 = &scenes[0].1;
        let mut previous_index = scene0.build_leaf_spatial_index(density);
        let fresh0 = covering_owned(&mut cache, scene0, density);
        let build0 = build_brick_field(&fresh0, density);
        let mut field = IncrementalBrickField::from_wholesale(&build0);
        let mut covering: std::collections::BTreeSet<[i32; 3]> =
            fresh0.iter().map(|(coord, _)| *coord).collect();
        assert_incremental_matches_wholesale(&field.to_build(), &build0, scenes[0].0);

        let mut incremental_steps = 0usize;
        for (label, scene) in &scenes[1..] {
            let new_index = scene.build_leaf_spatial_index(density);
            let edit_aabb = new_index.edit_aabb_since(&previous_index);
            // Mirror `AppCore::rebuild`: localisable edit → invalidate its chunks; a `None`
            // (wholesale) edit clears. Build the fresh covering set AFTER invalidation.
            let dirty = match &edit_aabb {
                Some(aabb) => cache.invalidate_aabb(aabb, density),
                None => {
                    cache.clear();
                    Vec::new()
                }
            };
            let fresh = covering_owned(&mut cache, scene, density);
            let new_covering: std::collections::BTreeSet<[i32; 3]> =
                fresh.iter().map(|(coord, _)| *coord).collect();

            // Incremental applies only when localisable AND the covering set is invariant
            // (the app routes a growth/reframe wholesale). Otherwise reset from wholesale.
            if edit_aabb.is_some() && new_covering == covering {
                field.apply_dirty_update(&fresh, &dirty);
                incremental_steps += 1;
            } else {
                let build = build_brick_field(&fresh, density);
                field = IncrementalBrickField::from_wholesale(&build);
            }
            covering = new_covering;

            let wholesale = build_brick_field(&fresh, density);
            assert_incremental_matches_wholesale(&field.to_build(), &wholesale, label);
            previous_index = new_index;
        }
        assert!(
            incremental_steps >= 4,
            "the script must exercise the INCREMENTAL path on most steps (was {incremental_steps})"
        );
    }

    /// Untouched-slot discipline (issue #69 acceptance): an edit confined to ONE chunk
    /// writes only that chunk's blocks' slots (+ frees), never the whole scene's — the
    /// "per-edit cost ∝ dirty region" claim made testable. A recolour keeps occupancy
    /// identical, so exactly the dirty chunk's sculpted blocks are freed + rewritten.
    #[test]
    fn one_chunk_edit_writes_only_that_chunks_slots() {
        let density = 4u32;
        // Anchors fix the covering set; a compact middle tool occupies its own chunks.
        let anchor_lo = tool(ShapeKind::Box, [-14, 0, 0], MaterialChoice::Stone, density);
        let anchor_hi = tool(ShapeKind::Box, [14, 0, 0], MaterialChoice::Stone, density);
        let scene_a = Scene::from_nodes(vec![
            anchor_lo.clone(),
            anchor_hi.clone(),
            tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Wood, density),
        ]);
        let scene_b = Scene::from_nodes(vec![
            anchor_lo,
            anchor_hi,
            // Same shape/placement, DIFFERENT material — a pure recolour (occupancy fixed).
            tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Plain, density),
        ]);

        let mut cache = TwoLayerResidentCache::enabled();
        cache.clear();
        let index_a = scene_a.build_leaf_spatial_index(density);
        let fresh_a = covering_owned(&mut cache, &scene_a, density);
        let build_a = build_brick_field(&fresh_a, density);
        let mut field = IncrementalBrickField::from_wholesale(&build_a);
        let total_sculpted = build_a.sculpted_brick_count();

        let index_b = scene_b.build_leaf_spatial_index(density);
        let edit_aabb = index_b
            .edit_aabb_since(&index_a)
            .expect("a recolour is a localisable edit");
        let dirty = cache.invalidate_aabb(&edit_aabb, density);
        let fresh_b = covering_owned(&mut cache, &scene_b, density);

        // Count the sculpted blocks living in the dirty chunks (the recolour re-writes
        // exactly these — occupancy is unchanged, only the record material differs).
        let dirty_set: std::collections::BTreeSet<[i32; 3]> = dirty.iter().copied().collect();
        let expected_written: usize = fresh_b
            .iter()
            .filter(|(coord, _)| dirty_set.contains(coord))
            .map(|(_, chunk)| chunk.microblocks.len())
            .sum();

        let update = field.apply_dirty_update(&fresh_b, &dirty);

        assert!(
            !dirty.is_empty() && dirty.len() < covering_owned(&mut cache, &scene_b, density).len(),
            "the edit must dirty SOME but not ALL chunks (dirtied {} of the covering set)",
            dirty.len()
        );
        assert_eq!(
            update.written_slots.len(),
            expected_written,
            "an edit must write exactly the dirty chunks' sculpted slots, no more"
        );
        assert!(
            update.written_slots.len() < total_sculpted,
            "a one-region edit must write FEWER than every scene slot ({} of {})",
            update.written_slots.len(),
            total_sculpted
        );
        // A pure recolour keeps occupancy, so freed == rewritten (slots recycled in place)
        // and the atlas does not grow.
        assert_eq!(update.freed_slots.len(), expected_written, "recolour frees what it rewrites");
        assert!(!update.atlas_grew, "a recolour does not grow the atlas");
        // And the result is still byte-exact vs wholesale.
        let wholesale = build_brick_field(&fresh_b, density);
        assert_incremental_matches_wholesale(&field.to_build(), &wholesale, "one-chunk-recolour");
    }

    /// Perf probe (issue #69, `#[ignore]`d — run in release): a ~1–2k-block scene, a
    /// one-region recolour, incremental patch vs a full `build_brick_field`. The headless
    /// stand-in for the Tracy live latency measurement; numbers go in the commit message.
    /// Run: `cargo test --release incremental_vs_wholesale_perf_probe -- --ignored --nocapture`.
    #[test]
    #[ignore = "perf probe — run in release with --nocapture"]
    fn incremental_vs_wholesale_perf_probe() {
        use std::time::Instant;
        let density = 8u32;
        let anchor_lo = tool(ShapeKind::Box, [-20, 0, 0], MaterialChoice::Stone, density);
        let anchor_hi = tool(ShapeKind::Box, [20, 0, 0], MaterialChoice::Stone, density);
        let scene_a = Scene::from_nodes(vec![
            anchor_lo.clone(),
            anchor_hi.clone(),
            tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Wood, density),
        ]);
        let scene_b = Scene::from_nodes(vec![
            anchor_lo,
            anchor_hi,
            tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Plain, density),
        ]);
        let mut cache = TwoLayerResidentCache::enabled();
        cache.clear();
        let index_a = scene_a.build_leaf_spatial_index(density);
        let fresh_a = covering_owned(&mut cache, &scene_a, density);
        let build_a = build_brick_field(&fresh_a, density);
        let mut field = IncrementalBrickField::from_wholesale(&build_a);

        let index_b = scene_b.build_leaf_spatial_index(density);
        let edit_aabb = index_b.edit_aabb_since(&index_a).expect("localisable");
        let dirty = cache.invalidate_aabb(&edit_aabb, density);
        let fresh_b = covering_owned(&mut cache, &scene_b, density);

        let started = Instant::now();
        let update = field.apply_dirty_update(&fresh_b, &dirty);
        let _incremental_build = field.to_build();
        let incremental = started.elapsed();

        let started = Instant::now();
        let _ = build_brick_field(&fresh_b, density);
        let wholesale = started.elapsed();

        println!(
            "G3 perf probe: scene {} records, edit dirtied {} chunk(s) / {} slots — \
             incremental {:?} vs wholesale {:?} ({:.1}× )",
            build_a.brick_records.len(),
            dirty.len(),
            update.written_slots.len(),
            incremental,
            wholesale,
            wholesale.as_secs_f64() / incremental.as_secs_f64().max(1e-9),
        );
        assert!(update.written_slots.len() < build_a.sculpted_brick_count());
    }
}
