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
// Clip-map occupancy pyramid (ADR 0011 Decision 4a / slice G2) — two WORLD-FIXED
// coarse "any-brick-inside" levels above the brick set, a min-mip of the record
// keys. The hierarchical DDA (brick_raymarch.wgsl) jumps a ray to the exit of the
// coarsest EMPTY level covering its position — one stride through empty space —
// descending to per-block brick work only where a level reports occupancy. This
// is the port of ADR 0009's measured 160→10240 (~64×) scattered-ceiling lift.
// ============================================================================

/// Level 1 (fine) clip-map cell edge, in BLOCKS — the benchmark's proven config
/// (ADR 0011 Decision 4a). Block-denominated (density-agnostic by construction),
/// never a hard-coded voxel count.
pub const CLIPMAP_LEVEL_1_BLOCKS_PER_CELL: u32 = 8;
/// Level 2 (coarse) clip-map cell edge, in BLOCKS (the benchmark's L2).
pub const CLIPMAP_LEVEL_2_BLOCKS_PER_CELL: u32 = 64;

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

/// The two-level clip-map pyramid (L1 = 8-block cells, L2 = 64-block cells; ADR
/// 0011 Decision 4a, 2 levels first). A derived, rebuildable min-mip of the brick
/// records — never truth (ADR 0006/0009 4c).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClipmapPyramid {
    /// Fine level (8-block cells).
    pub level_1: ClipmapLevel,
    /// Coarse level (64-block cells) — checked first by the hierarchical DDA.
    pub level_2: ClipmapLevel,
}

impl ClipmapPyramid {
    /// Build both levels from a brick-field's sorted records (a pure function of
    /// the record keys — the sink derives it next to the record set).
    pub fn from_records(records: &[BrickRecord]) -> Self {
        ClipmapPyramid {
            level_1: ClipmapLevel::from_records(records, CLIPMAP_LEVEL_1_BLOCKS_PER_CELL),
            level_2: ClipmapLevel::from_records(records, CLIPMAP_LEVEL_2_BLOCKS_PER_CELL),
        }
    }

    /// The "pyramid off" form — both levels empty, so the shader's hierarchical
    /// skip never fires (the flat G1 block-DDA). Used by the pyramid-on == off
    /// parity assertion and the perf probe's baseline.
    pub fn empty() -> Self {
        ClipmapPyramid {
            level_1: ClipmapLevel::empty(CLIPMAP_LEVEL_1_BLOCKS_PER_CELL),
            level_2: ClipmapLevel::empty(CLIPMAP_LEVEL_2_BLOCKS_PER_CELL),
        }
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
                    if let Some(block_id) = chunk.coarse_block(block) {
                        // Coarse XOR boundary is the classifier's invariant; a block in
                        // both layers would double-emit its key.
                        debug_assert!(
                            !chunk.microblocks.contains_key(&block),
                            "a block must be coarse XOR boundary"
                        );
                        brick_records.push(BrickRecord {
                            packed_world_block_key: pack_world_block_key(world_block),
                            payload: BrickPayload::CoarseSolid { block_id },
                            // Fully solid through ⇒ every face is solid.
                            seam_solidity: SeamSolidity {
                                solid: [[true; 2]; 3],
                            },
                        });
                    } else if let Some(geometry) = chunk.microblocks.get(&block) {
                        let atlas_slot = sculpted_brick_tiles.len() as u32;
                        sculpted_brick_tiles
                            .push(rasterize_brick_occupancy(geometry, brick_edge_voxels));
                        brick_records.push(BrickRecord {
                            packed_world_block_key: pack_world_block_key(world_block),
                            payload: BrickPayload::Sculpted { atlas_slot },
                            seam_solidity: geometry.seam_solidity,
                        });
                    }
                    // else: air block — nothing (ADR 0011 Decision 2).
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
    let sculpted_count = sculpted_brick_tiles.len();
    let (bricks_per_axis, atlas_dim_voxels) = if sculpted_count == 0 {
        (0, 0)
    } else {
        let tiles = ((sculpted_count as f64).cbrt().ceil() as u32).max(1);
        (tiles, tiles * brick_edge_voxels)
    };
    let atlas_dim = atlas_dim_voxels as usize;
    let mut sculpted_atlas_bytes = vec![0u8; atlas_dim * atlas_dim * atlas_dim];
    let edge = brick_edge_voxels as usize;
    for (atlas_slot, brick_bytes) in sculpted_brick_tiles.iter().enumerate() {
        let tiles = bricks_per_axis;
        let slot = atlas_slot as u32;
        let origin = [
            (slot % tiles) as usize * edge,
            ((slot / tiles) % tiles) as usize * edge,
            (slot / (tiles * tiles)) as usize * edge,
        ];
        for local_z in 0..edge {
            for local_y in 0..edge {
                let source_row = (local_z * edge + local_y) * edge;
                let atlas_row = ((origin[2] + local_z) * atlas_dim + origin[1] + local_y)
                    * atlas_dim
                    + origin[0];
                sculpted_atlas_bytes[atlas_row..atlas_row + edge]
                    .copy_from_slice(&brick_bytes[source_row..source_row + edge]);
            }
        }
    }

    BrickFieldBuild {
        brick_records,
        sculpted_atlas_bytes,
        brick_edge_voxels,
        bricks_per_axis,
        atlas_dim_voxels,
    }
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
            // The coarse level must not be finer than L1 (fewer-or-equal cells).
            assert!(pyramid.level_2.cell_keys.len() <= pyramid.level_1.cell_keys.len());
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
