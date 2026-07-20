use super::*;

/// What a brick holds — the record kinds of the brick partition. The enum makes "a coarse
/// record consumes no atlas slot" and "only a MIXED block owns a cell-key tile" structural,
/// not conventions.
///
/// A sculpted block is **uniform** when every one of its microblock cuboids carries the same
/// cell key (block-palette id + on-face-grid overlay bit) — then the key lives once, on the
/// record ([`BrickRecord::material_id`] + [`BrickRecord::overlay`]). It is **mixed** when the
/// cuboids disagree; then its per-voxel keys live in a cell-key tile of the separately-pooled
/// material side atlas, and the record's own material/overlay are don't-care. See
/// docs/architecture/03-display.md (the brick-field atlas).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrickPayload {
    /// **Kind 0** — an analytic coarse brick: the whole block is solid at `block_id`,
    /// stored as this one record with no per-voxel data (interior elision on the GPU;
    /// also the residency-miss fallback form the G1 contract renders).
    CoarseSolid { block_id: BlockId },
    /// **Kind 1** — a sculpted brick whose voxels all share ONE cell key: the block's voxel
    /// occupancy lives in atlas slot `atlas_slot` (an `edge³` R8 tile, edge =
    /// `voxels_per_block`); its material + overlay live on the record.
    Sculpted { atlas_slot: u32 },
    /// **Kind 1, mixed** — a sculpted brick whose microblocks disagree on their cell key:
    /// occupancy in `atlas_slot` exactly as [`Sculpted`](Self::Sculpted), PLUS a per-voxel
    /// cell-key tile in `cell_key_slot` of the material side atlas (an independent pool with
    /// its own free-list — a cell-key slot number is unrelated to an occupancy slot number).
    SculptedMixed {
        atlas_slot: u32,
        cell_key_slot: u32,
    },
}

impl BrickPayload {
    /// The GPU-side record-kind discriminant: **0** = coarse, **1** = sculpted-uniform,
    /// **2** = sculpted-MIXED. Pinned here — like `shape_kind_discriminant` — so a future enum
    /// reorder can't silently desync the shader: `pack_gpu_records` packs THIS value into the
    /// GPU record's `kind` bits and the WGSL decodes it there.
    ///
    /// Kinds 1 and 2 TRAVERSE identically (both descend into an occupancy atlas slot); they
    /// differ only in where the hit's SHADE comes from — the record's own material + overlay
    /// (1), or the per-voxel cell-key texel of the material side atlas (2).
    pub fn kind_discriminant(&self) -> u32 {
        match self {
            BrickPayload::CoarseSolid { .. } => 0,
            BrickPayload::Sculpted { .. } => 1,
            BrickPayload::SculptedMixed { .. } => 2,
        }
    }

    /// The occupancy atlas slot of a sculpted brick (uniform or mixed); `None` for a coarse
    /// record (which consumes no slot). The ONE reader of "does this record own an occupancy
    /// tile", so a new sculpted arm can never be missed by a slot-bookkeeping site.
    pub fn occupancy_atlas_slot(&self) -> Option<u32> {
        match *self {
            BrickPayload::CoarseSolid { .. } => None,
            BrickPayload::Sculpted { atlas_slot }
            | BrickPayload::SculptedMixed { atlas_slot, .. } => Some(atlas_slot),
        }
    }

    /// The material side-atlas slot holding this brick's per-voxel cell-key tile — `Some`
    /// only for a MIXED sculpted brick (a uniform or coarse block carries its one cell key on
    /// the record and owns no tile).
    pub fn cell_key_slot(&self) -> Option<u32> {
        match *self {
            BrickPayload::SculptedMixed { cell_key_slot, .. } => Some(cell_key_slot),
            _ => None,
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
    /// `block_id`'s colour index for a coarse block, the single microblock material for a
    /// UNIFORM boundary block. The occupancy atlas is occupancy-only, so this is the
    /// per-BLOCK material the raymarch shades with, packed into the GPU record's `kind`
    /// high bits by `pack_gpu_records`. **Don't-care for a
    /// [`SculptedMixed`](BrickPayload::SculptedMixed) block** — its per-voxel keys are the
    /// truth (this holds the first cuboid's clean id there, never read as the block's).
    pub material_id: u16,
    /// The block's on-face-grid overlay bit — the other half of its cell key (the render-cell
    /// key is `material_id | overlay`, see [`voxel_core::core_geom::CellKey`]). Carried
    /// per-RECORD so a scene whose blocks disagree on the overlay is still one brick field.
    /// Meaningful for coarse + UNIFORM sculpted blocks; don't-care for a
    /// [`SculptedMixed`](BrickPayload::SculptedMixed) block (its tile's per-voxel keys carry
    /// the overlay bit themselves).
    pub overlay: bool,
    /// Coarse (kind 0), sculpted-uniform or sculpted-mixed (kind 1) — see [`BrickPayload`].
    pub payload: BrickPayload,
    /// Per-face seam-solidity flags, carried UNCHANGED from the boundary set for a
    /// sculpted brick. A coarse-solid block is solid through, so every face flag is
    /// `true` by construction (the block-DDA culls against it identically either way).
    pub seam_solidity: SeamSolidity,
}

/// The built brick field: the sorted record array + the sculpted-brick occupancy atlas
/// bytes in the ADR 0007 tile-cube layout (`bricks_per_axis³` slots of `edge³` texels,
/// linear slot index → 3D tile coord, x-fastest). [`upload_brick_atlas`] lands the bytes
/// in an R8 3D texture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrickFieldBuild {
    /// Every non-air block's record, sorted strictly ascending by
    /// `packed_world_block_key` (unique — a block is coarse XOR boundary).
    pub brick_records: Vec<BrickRecord>,
    /// `atlas_dim_voxels³` occupancy bytes (0 empty / 255 occupied), slot-packed;
    /// tile slots past the last sculpted brick stay all-zero.
    pub sculpted_atlas_bytes: Vec<u8>,
    /// The MIXED bricks' per-voxel cell-key tiles, indexed by the `cell_key_slot` their
    /// records carry (dense `0..mixed_count` in this build's traversal order). EMPTY for a
    /// scene whose every sculpted block is uniform — the sparse-side-atlas contract: only a
    /// block that mixes cell keys pays per-voxel material cost.
    ///
    /// Unlike the occupancy atlas, these are **not** packed to a byte blob here: packing is
    /// deferred to the install/patch seam (`cell_key_atlas_payload` / `pack_cell_key_atlas_payload`),
    /// which is also where the GPU side atlas is (re)built, so a build with no mixed brick pays
    /// zero packing cost and the tiles travel as tiles (the single-owner tile law — moved into
    /// [`IncrementalBrickField`], never cloned per edit).
    pub cell_key_tiles: Vec<BrickCellKeyTile>,
    /// The brick edge in voxels — `voxels_per_block`, the ONE-BLOCK granule
    /// (ADR 0011 Decision 1). Block-denominated: never a hard-coded voxel count.
    pub brick_edge_voxels: u32,
    /// Sculpted-brick tile slots per atlas axis (`ceil(cbrt(sculpted_count))`).
    pub bricks_per_axis: u32,
    /// `bricks_per_axis * brick_edge_voxels` — the atlas texture dimension per axis
    /// (0 when the build has no sculpted brick).
    pub atlas_dim_voxels: u32,
}

/// The GPU upload payload for the sculpted-brick atlas — the ONE place the flat R8 byte
/// blob still lives after item 9's single-owner rework (see `docs/architecture/`, the
/// brick-field display chapter). A wholesale build hands this to
/// [`BrickRaymarchRenderer::install_brick_field`](crate::brick::BrickRaymarchRenderer::install_brick_field)
/// by MOVE ([`IncrementalBrickField::from_wholesale`]); the incremental patch path never
/// materialises one except on the legitimate atlas-grow re-pack
/// ([`IncrementalBrickField::pack_atlas_payload`]). Carries the atlas GEOMETRY alongside
/// the bytes so the install seam sets its frame scalars without a `BrickFieldBuild`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SculptedAtlasPayload {
    /// `geometry.atlas_dim_voxels³` occupancy bytes (0 empty / 255 occupied), slot-packed —
    /// the bytes [`upload_brick_atlas`] lands in the R8 3D texture.
    pub bytes: Vec<u8>,
    /// The atlas tile geometry (tile-grid edge, texture dimension, brick edge) — shared with
    /// the incremental owner's [`IncrementalBrickField::atlas_geometry`] so the two never
    /// drift on the tile-cube math.
    pub geometry: SculptedAtlasGeometry,
    /// Live sculpted-brick count (the wholesale install's `last_atlas_slots_written`).
    pub sculpted_slot_count: u32,
}

/// The sculpted atlas's tile geometry — `bricks_per_axis` / `atlas_dim_voxels` / brick edge,
/// factored so the incremental owner and the packer never drift on the tile-cube math.
/// ([`IncrementalBrickField::atlas_geometry`].)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SculptedAtlasGeometry {
    /// Sculpted-brick tile slots per atlas axis (`ceil(cbrt(slot_high_water))`).
    pub bricks_per_axis: u32,
    /// The atlas texture dimension per axis (`bricks_per_axis * brick_edge_voxels`).
    pub atlas_dim_voxels: u32,
    /// The brick edge in voxels (`voxels_per_block`).
    pub brick_edge_voxels: u32,
}

/// The GPU upload payload for the **material side atlas**: the MIXED bricks' per-voxel
/// cell-key tiles packed into one 16-bit-texel cube, landed by
/// [`upload_brick_cell_key_atlas`] in an R16Uint 3D texture. The sibling of
/// [`SculptedAtlasPayload`] — a SECOND, independently pooled atlas (its own slot numbering,
/// its own free-list, its own tile-grid edge), sparse by construction: only a block whose
/// microblocks disagree on their cell key owns a slot here, so a scene of uniform blocks packs
/// ZERO bytes. See docs/architecture/03-display.md (the brick-field atlas).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SculptedCellKeyAtlasPayload {
    /// `2 · geometry.atlas_dim_voxels³` bytes: one **little-endian u16 cell key per voxel**
    /// (clean block-palette id + the on-face-grid overlay bit, verbatim — no indirection, no
    /// per-brick palette). An air voxel's texel is a documented don't-care (occupancy gates
    /// every read). EMPTY when no brick is mixed.
    pub bytes: Vec<u8>,
    /// The side atlas's OWN tile geometry — derived from the cell-key slot count, never from
    /// the occupancy pool's.
    pub geometry: SculptedCellKeyAtlasGeometry,
    /// Live cell-key slot count (== the mixed-brick count of a wholesale build).
    pub cell_key_slot_count: u32,
}

/// The material side atlas's tile geometry — the twin of [`SculptedAtlasGeometry`] computed
/// from the CELL-KEY slot count (the pools size independently: a scene of 10k sculpted bricks
/// with 3 mixed ones has a 22-tile occupancy grid and a 2-tile material grid).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SculptedCellKeyAtlasGeometry {
    /// Cell-key tile slots per atlas axis (`ceil(cbrt(cell_key_slot_high_water))`).
    pub bricks_per_axis: u32,
    /// The side atlas's texture dimension per axis in voxels (`bricks_per_axis *
    /// brick_edge_voxels`); 0 when no brick is mixed.
    pub atlas_dim_voxels: u32,
    /// The brick edge in voxels (`voxels_per_block`) — the same granule the occupancy atlas
    /// tiles at (one cell-key texel per occupancy voxel, cell-for-cell).
    pub brick_edge_voxels: u32,
}

impl SculptedCellKeyAtlasPayload {
    /// The side atlas of a field with NO mixed brick: zero bytes, zero slots, zero dimension —
    /// what every scene the representability gate admits today packs to.
    pub fn empty(brick_edge_voxels: u32) -> Self {
        Self {
            bytes: Vec::new(),
            geometry: SculptedCellKeyAtlasGeometry {
                bricks_per_axis: 0,
                atlas_dim_voxels: 0,
                brick_edge_voxels: brick_edge_voxels.max(1),
            },
            cell_key_slot_count: 0,
        }
    }
}

/// Pack a slot-indexed set of cell-key tiles into the side atlas's little-endian u16 texel cube
/// — substrate's [`CubeTilePacking::pack_u16_value_cubes`], the payload sibling of the
/// occupancy scatter, at its own (cell-key) slot count. Shared by the wholesale build and
/// [`IncrementalBrickField::pack_cell_key_atlas_payload`] so the two are byte-identical for the
/// same tile vector. A FREED (dead) slot's tile is scattered as-is: unreachable from any live
/// record, so its texels may be garbage.
pub(crate) fn pack_cell_key_atlas(
    slot_tiles: &[BrickCellKeyTile],
    brick_edge_voxels: u32,
) -> (u32, u32, Vec<u8>) {
    CubeTilePacking::pack_u16_value_cubes(slot_tiles, brick_edge_voxels)
}

/// The occupancy byte a solid voxel packs to — the sculpted atlas's 0/255 R8 convention. Injected
/// into [`BrickOccupancyTile::expand_to_bytes`] / [`CubeTilePacking::pack_bit_cubes`] at the
/// atlas seam (substrate names no such byte — a set bit reads as whatever the caller passes).
pub(crate) const SCULPTED_BRICK_OCCUPIED: u8 = 255;

impl BrickFieldBuild {
    /// Materialise this build's sculpted atlas as an upload [`SculptedAtlasPayload`] — the
    /// wholesale-build → install adapter for the callers that keep the `BrickFieldBuild`
    /// around (the `shot` golden tool and the parity tests). CLONES the atlas bytes; the
    /// live worker/orchestrator paths move them instead via
    /// [`IncrementalBrickField::from_wholesale`].
    pub fn atlas_payload(&self) -> SculptedAtlasPayload {
        SculptedAtlasPayload {
            bytes: self.sculpted_atlas_bytes.clone(),
            geometry: SculptedAtlasGeometry {
                bricks_per_axis: self.bricks_per_axis,
                atlas_dim_voxels: self.atlas_dim_voxels,
                brick_edge_voxels: self.brick_edge_voxels,
            },
            sculpted_slot_count: self.sculpted_brick_count() as u32,
        }
    }

    /// Materialise this build's MATERIAL SIDE ATLAS as an upload
    /// [`SculptedCellKeyAtlasPayload`] — the second pool's install adapter, packed from the
    /// mixed bricks' cell-key tiles at their own dense slot numbering. A build with no mixed
    /// brick yields the empty payload (zero bytes: the sparse-side-atlas contract).
    pub fn cell_key_atlas_payload(&self) -> SculptedCellKeyAtlasPayload {
        let (bricks_per_axis, atlas_dim_voxels, bytes) =
            pack_cell_key_atlas(&self.cell_key_tiles, self.brick_edge_voxels);
        SculptedCellKeyAtlasPayload {
            bytes,
            geometry: SculptedCellKeyAtlasGeometry {
                bricks_per_axis,
                atlas_dim_voxels,
                brick_edge_voxels: self.brick_edge_voxels,
            },
            cell_key_slot_count: self.mixed_brick_count() as u32,
        }
    }

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

    /// How many records are sculpted bricks — uniform AND mixed (== occupancy atlas slots in
    /// use; slots are assigned densely `0..count`).
    pub fn sculpted_brick_count(&self) -> usize {
        self.brick_records
            .iter()
            .filter(|record| record.payload.occupancy_atlas_slot().is_some())
            .count()
    }

    /// How many records are MIXED sculpted bricks (== cell-key tiles, i.e. material
    /// side-atlas slots in use; densely `0..count` in a wholesale build).
    pub fn mixed_brick_count(&self) -> usize {
        self.brick_records
            .iter()
            .filter(|record| record.payload.cell_key_slot().is_some())
            .count()
    }

    /// The low-corner texel of `atlas_slot`'s tile in the atlas cube (linear slot →
    /// 3D tile coord, x-fastest — the ADR 0007 tile-cube layout).
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
