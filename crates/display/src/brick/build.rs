use super::*;

/// Build the **surface-only** brick field from a scene's two-layer boundary set (the
/// `build_covering_chunks` / resident-cache output): walk every chunk's block partition,
/// emit one record per SURFACE non-air block — a fully-occluded coarse interior block
/// emits nothing (ADR 0011 interior elision, fused into the build via
/// [`BrickOcclusionOracle`]; interiors stay queryable through the chunks) — rasterize each
/// boundary block's cuboids into its atlas slot, and sort the records by packed
/// world-block key.
///
/// **O(surface), not O(volume) (the 8000³-freeze fix):** an all-interior chunk (fully
/// solid, fully-solid face-neighbours) is skipped whole without visiting its blocks, so a
/// 125M-block solid emits ~1.5M records and the build touches only the ~1-chunk-thick
/// boundary shell. Every consumer downstream (sort, GPU pack, incremental mirror clone)
/// inherits the ∝-surface cost. The interior-INCLUSIVE build survives as
/// [`build_brick_field_all_blocks`], the parity oracle.
///
/// `voxels_per_block` is the document density every chunk was built at (each chunk
/// carries it; a mismatch is a caller bug, asserted in debug).
///
/// **Why the classify pass stays SERIAL (measured).** The per-block classify + slot
/// assignment is coarse-dominated and memory-bound, so a rayon per-chunk split measured NO
/// net win: the parallel classify gain was cancelled by the extra ordered-merge pass needed
/// to keep the sculpted atlas-slot numbering byte-identical (slots are assigned in
/// traversal order — ADR 0011 G3's incremental-atlas contract — so a parallel build must
/// re-derive that exact order, adding an O(records) merge). Only the final key sort and the
/// oracle's chunk classification are worth parallelising. The record ORDER + sculpted slot
/// numbering are produced by the same serial traversal as the oracle build — sculpted slots
/// bit-for-bit identical (the sculpted set is never elided).
pub fn build_brick_field(
    two_layer_chunks: &[([i32; 3], Arc<TwoLayerChunk>)],
    voxels_per_block: u32,
) -> BrickFieldBuild {
    // The build-only entry (lib re-export; the golden `shot` tool, parity/perf tests, the
    // non-tile-carrying orchestrator/startup paths). Drops the rasterised tiles; the live
    // worker/orchestrator wholesale path calls `build_brick_field_with_tiles` to keep and
    // MOVE them into the mirror (skipping the from-atlas-bytes re-derive).
    build_brick_field_with_tiles(two_layer_chunks, voxels_per_block).0
}

/// Like [`build_brick_field`] but ALSO returns the per-sculpted-slot occupancy tiles it
/// rasterised (dense slot order — the `atlas_slot` numbering baked into the records), so a
/// wholesale reset can MOVE them straight into the incremental mirror
/// ([`IncrementalBrickField::from_wholesale_with_tiles`]) instead of re-gathering + re-bit-
/// packing them out of the flat atlas bytes the packer just produced. The `BrickFieldBuild`
/// is byte-identical to [`build_brick_field`]'s (same records, same packed atlas bytes) —
/// this only hands back the intermediate the plain entry discards.
pub fn build_brick_field_with_tiles(
    two_layer_chunks: &[([i32; 3], Arc<TwoLayerChunk>)],
    voxels_per_block: u32,
) -> (BrickFieldBuild, Vec<BrickOccupancyTile>) {
    let brick_edge_voxels = voxels_per_block.max(1);
    let oracle = BrickOcclusionOracle::new(two_layer_chunks);
    let mut brick_records: Vec<BrickRecord> = Vec::new();
    // One bit-packed `edge²`-word tile per sculpted brick, in slot order; unpacked into
    // the atlas cube once the final count fixes the tile geometry.
    let mut sculpted_brick_tiles: Vec<BrickOccupancyTile> = Vec::new();
    // One `edge³` cell-key tile per MIXED sculpted brick, in cell-key-slot order — an
    // INDEPENDENT dense numbering (a mixed brick's two slots are unrelated numbers).
    let mut cell_key_tiles: Vec<BrickCellKeyTile> = Vec::new();

    for (chunk_coord, chunk) in two_layer_chunks {
        debug_assert_eq!(
            chunk.voxels_per_block, brick_edge_voxels,
            "every chunk of one build shares the document density"
        );
        // Interior-chunk fast path (∝ surface, the 8000³-freeze fix): a fully-solid chunk
        // ringed by fully-solid face-neighbours is all-occluded — emit NOTHING for it
        // without visiting a single block. An interior chunk has no microblocks by
        // definition, so no sculpted record is skipped here.
        if oracle.chunk_is_all_interior(*chunk_coord) {
            continue;
        }
        let occlusion = oracle.context_for_chunk(*chunk_coord, chunk.as_ref());
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
                        // A coarse-solid block emits ONLY when a ray could reach it: the
                        // fused interior elision (never emitted ⇒ never sorted, packed,
                        // uploaded). Interiors stay queryable through the chunks.
                        BlockBrick::Coarse(record) => {
                            if !occlusion.coarse_block_occluded(block) {
                                brick_records.push(record);
                            }
                        }
                        // A boundary (sculpted) block is surface by definition here: its
                        // record — and thus its occupancy atlas tile — is NEVER elided,
                        // so sculpted slot numbering matches the interior-inclusive
                        // oracle build tile-for-tile.
                        BlockBrick::Sculpted {
                            material_id,
                            overlay,
                            seam_solidity,
                            tile,
                            cell_keys,
                        } => {
                            let atlas_slot = sculpted_brick_tiles.len() as u32;
                            sculpted_brick_tiles.push(tile);
                            brick_records.push(BrickRecord {
                                packed_world_block_key: pack_world_block_key(world_block),
                                material_id,
                                overlay,
                                payload: sculpted_payload_dense(
                                    atlas_slot,
                                    cell_keys,
                                    &mut cell_key_tiles,
                                ),
                                seam_solidity,
                            });
                        }
                    }
                }
            }
        }
    }

    // The keys are UNIQUE (each world block appears in exactly one chunk — asserted below),
    // so a parallel unstable sort yields the byte-identical order a serial sort would, at any
    // thread count. (A filtered emission of the serial traversal stays traversal-ordered, so
    // this is the same sort the interior-inclusive build performs — the shader binary search
    // and the G3 patch protocol see a sorted, unique array either way.)
    brick_records.par_sort_unstable_by_key(|record| record.packed_world_block_key);
    debug_assert!(
        brick_records
            .windows(2)
            .all(|pair| pair[0].packed_world_block_key < pair[1].packed_world_block_key),
        "brick keys must be unique (each world block appears in exactly one chunk)"
    );

    let (bricks_per_axis, atlas_dim_voxels, sculpted_atlas_bytes) =
        pack_sculpted_atlas(&sculpted_brick_tiles, brick_edge_voxels);

    let build = BrickFieldBuild {
        brick_records,
        sculpted_atlas_bytes,
        cell_key_tiles,
        brick_edge_voxels,
        bricks_per_axis,
        atlas_dim_voxels,
    };
    (build, sculpted_brick_tiles)
}

/// Assign a sculpted block's payload in a WHOLESALE build's dense numbering: the occupancy
/// slot is the caller's (already pushed), and a MIXED block's cell-key tile appends to
/// `cell_key_tiles` at the next dense material slot. The two pools are independent — a mixed
/// brick's `atlas_slot` and `cell_key_slot` are unrelated numbers. Shared by the surface-only
/// build and the interior-inclusive oracle build so both number the pools identically.
pub(crate) fn sculpted_payload_dense(
    atlas_slot: u32,
    cell_keys: Option<BrickCellKeyTile>,
    cell_key_tiles: &mut Vec<BrickCellKeyTile>,
) -> BrickPayload {
    match cell_keys {
        None => BrickPayload::Sculpted { atlas_slot },
        Some(tile) => {
            let cell_key_slot = cell_key_tiles.len() as u32;
            cell_key_tiles.push(tile);
            BrickPayload::SculptedMixed {
                atlas_slot,
                cell_key_slot,
            }
        }
    }
}

/// The interior-INCLUSIVE brick-field build: one record per NON-AIR block (coarse-solid or
/// boundary), the pre-elision reference. **Oracle only** — the live sink uses the surface-only
/// [`build_brick_field`]; this stays as the parity oracle for the interior-elision gates
/// (`brick_surface_elision_hit_set_unchanged`, `clipmap_from_chunks_equals_from_full_records`)
/// and any consumer that genuinely needs every block (none on the live path).
pub fn build_brick_field_all_blocks(
    two_layer_chunks: &[([i32; 3], Arc<TwoLayerChunk>)],
    voxels_per_block: u32,
) -> BrickFieldBuild {
    let brick_edge_voxels = voxels_per_block.max(1);
    let mut brick_records: Vec<BrickRecord> = Vec::new();
    // One bit-packed `edge²`-word tile per sculpted brick, in slot order; unpacked into
    // the atlas cube once the final count fixes the tile geometry.
    let mut sculpted_brick_tiles: Vec<BrickOccupancyTile> = Vec::new();
    // One `edge³` cell-key tile per MIXED sculpted brick, in cell-key-slot order.
    let mut cell_key_tiles: Vec<BrickCellKeyTile> = Vec::new();

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
                            overlay,
                            seam_solidity,
                            tile,
                            cell_keys,
                        } => {
                            let atlas_slot = sculpted_brick_tiles.len() as u32;
                            sculpted_brick_tiles.push(tile);
                            brick_records.push(BrickRecord {
                                packed_world_block_key: pack_world_block_key(world_block),
                                material_id,
                                overlay,
                                payload: sculpted_payload_dense(
                                    atlas_slot,
                                    cell_keys,
                                    &mut cell_key_tiles,
                                ),
                                seam_solidity,
                            });
                        }
                    }
                }
            }
        }
    }

    // The keys are UNIQUE (each world block appears in exactly one chunk — asserted below),
    // so a parallel unstable sort yields the byte-identical order a serial sort would, at any
    // thread count. This is the one part of the build that measurably parallelises.
    brick_records.par_sort_unstable_by_key(|record| record.packed_world_block_key);
    debug_assert!(
        brick_records
            .windows(2)
            .all(|pair| pair[0].packed_world_block_key < pair[1].packed_world_block_key),
        "brick keys must be unique (each world block appears in exactly one chunk)"
    );

    // Tile geometry follows the ADR 0007 tile-cube layout: a cubic-ish slot grid bounded by
    // the SCULPTED count (coarse records consume none of it), then scatter each tile.
    let (bricks_per_axis, atlas_dim_voxels, sculpted_atlas_bytes) =
        pack_sculpted_atlas(&sculpted_brick_tiles, brick_edge_voxels);

    BrickFieldBuild {
        brick_records,
        sculpted_atlas_bytes,
        cell_key_tiles,
        brick_edge_voxels,
        bricks_per_axis,
        atlas_dim_voxels,
    }
}

/// The six face-neighbour chunk offsets — the reach of a block's occlusion verdict at the
/// chunk granularity (a block's six face-neighbours land in its own chunk or one of these).
pub(crate) const FACE_NEIGHBOUR_CHUNK_OFFSETS: [[i32; 3]; 6] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
];

/// **The occlusion oracle over a two-layer covering set (ADR 0011 interior elision — the
/// brick sink's analogue of the mesh's interior-face culling).** Decides which coarse-solid
/// blocks are FULLY OCCLUDED — all six face-neighbours present AND solid on the shared face
/// — so [`build_brick_field`] / [`IncrementalBrickField::apply_dirty_update`] can fuse the
/// interior elision INTO record emission (the record set is surface-only by construction;
/// no post-hoc mask pass over an O(volume) record array).
///
/// A fully-occluded block is never a ray's first hit: the block-DDA
/// ([`cpu_march_brick_field`](crate::brick::cpu_march_brick_field)) returns at the
/// FIRST block carrying a record, and a ray reaching an occluded block must first pass
/// through the solid neighbour surrounding it (which keeps its record). So never emitting it
/// is **hit-identical** — proven against the interior-inclusive oracle build in
/// `tests/gpu_parity.rs::brick_surface_elision_hit_set_unchanged`.
///
/// **Chunk-level fast path ([`Self::chunk_is_all_interior`]):** a chunk that is itself fully
/// coarse-solid AND whose six FACE-neighbour chunks are all fully coarse-solid has every one
/// of its blocks occluded (each block's six neighbours land in this chunk or a full
/// neighbour, all solid) — the builder skips the whole chunk with one set lookup, visiting
/// none of its blocks. Only the ~1-chunk-thick boundary shell (and any chunk carrying
/// microblocks) does per-block work, so the build is ∝ surface, not volume.
///
/// **Conservative direction:** a neighbour that is ABSENT (air) or only PARTIALLY solid on
/// the shared face keeps the block. The emitted set is thus always a superset of the
/// truly-visible blocks — elision can never drop a block a ray can see.
pub(crate) struct BrickOcclusionOracle<'a> {
    /// Every covering chunk by absolute chunk coord (the neighbour-resolution index).
    chunk_by_coord: std::collections::HashMap<[i32; 3], &'a TwoLayerChunk>,
    /// Chunks that are fully coarse-solid AND ringed by fully coarse-solid face-neighbours —
    /// every block of these is provably occluded (the bulk fast path).
    interior_chunk: std::collections::HashSet<[i32; 3]>,
}

impl<'a> BrickOcclusionOracle<'a> {
    /// Classify the chunk set once (parallel — the full-solidity scan is a pure per-chunk
    /// fold, and set membership is order-free).
    pub(crate) fn new(chunks: &'a [([i32; 3], Arc<TwoLayerChunk>)]) -> Self {
        let chunk_by_coord: std::collections::HashMap<[i32; 3], &TwoLayerChunk> = chunks
            .iter()
            .map(|(coord, chunk)| (*coord, chunk.as_ref()))
            .collect();
        // A chunk is "full-solid" iff every one of its CHUNK_BLOCKS³ blocks is coarse-solid
        // and it carries no microblocks — then every block of it is solid on every face.
        let full_solid: std::collections::HashSet<[i32; 3]> = chunks
            .par_iter()
            .filter(|(_, chunk)| {
                chunk.microblocks.is_empty() && chunk.coarse.iter().all(Option::is_some)
            })
            .map(|(coord, _)| *coord)
            .collect();
        let interior_chunk: std::collections::HashSet<[i32; 3]> = full_solid
            .par_iter()
            .filter(|coord| {
                FACE_NEIGHBOUR_CHUNK_OFFSETS.iter().all(|d| {
                    full_solid.contains(&[coord[0] + d[0], coord[1] + d[1], coord[2] + d[2]])
                })
            })
            .copied()
            .collect();
        Self {
            chunk_by_coord,
            interior_chunk,
        }
    }

    /// Whether every block of `chunk_coord` is provably occluded (the bulk fast path): the
    /// chunk and its six face-neighbours are all fully coarse-solid. The builder emits
    /// nothing for such a chunk without visiting a single block.
    pub(crate) fn chunk_is_all_interior(&self, chunk_coord: [i32; 3]) -> bool {
        self.interior_chunk.contains(&chunk_coord)
    }

    /// The per-chunk occlusion context: this chunk plus its six face-neighbour chunk refs,
    /// hoisted ONCE per chunk so the per-block six-neighbour test needs no hashing.
    pub(crate) fn context_for_chunk(
        &self,
        chunk_coord: [i32; 3],
        chunk: &'a TwoLayerChunk,
    ) -> ChunkOcclusionContext<'a> {
        // [axis][side]: side 0 = the low-face neighbour (coord − 1), side 1 = high (+1).
        let mut face_neighbours: [[Option<&TwoLayerChunk>; 2]; 3] = [[None; 2]; 3];
        for (axis, sides) in face_neighbours.iter_mut().enumerate() {
            for (side, slot) in sides.iter_mut().enumerate() {
                let mut coord = chunk_coord;
                coord[axis] += if side == 0 { -1 } else { 1 };
                *slot = self.chunk_by_coord.get(&coord).copied();
            }
        }
        ChunkOcclusionContext {
            chunk,
            face_neighbours,
        }
    }
}

/// One chunk's occlusion window: the chunk itself + its six face-neighbour chunks (resolved
/// once — see [`BrickOcclusionOracle::context_for_chunk`]). Answers the per-block
/// six-neighbour occlusion test in O(1) chunk resolution (a block's neighbours land in this
/// chunk or a face-adjacent one, never farther).
pub(crate) struct ChunkOcclusionContext<'a> {
    chunk: &'a TwoLayerChunk,
    /// `[axis][side]`: side 0 = the low-face neighbour chunk, 1 = high. `None` = absent (air).
    face_neighbours: [[Option<&'a TwoLayerChunk>; 2]; 3],
}

impl ChunkOcclusionContext<'_> {
    /// Whether the coarse-solid block at chunk-local `block` is FULLY OCCLUDED: each axis
    /// capped on BOTH sides — the +1 neighbour's LOW face covers this block's HIGH face, and
    /// the −1 neighbour's HIGH face covers its LOW. Occluded ⇒ no record is emitted.
    pub(crate) fn coarse_block_occluded(&self, block: [u32; 3]) -> bool {
        (0..3).all(|axis| {
            self.neighbour_face_solid(block, axis, 1) && self.neighbour_face_solid(block, axis, -1)
        })
    }

    /// Is the neighbour of chunk-local `block` across `(axis, delta)` present AND solid on
    /// the face it shares with `block`? A coarse-solid neighbour is solid on every face; a
    /// boundary neighbour consults its per-face seam flag; an air block / absent chunk is
    /// not solid (the conservative direction). Semantics identical to resolving through the
    /// absolute-coordinate chunk map — only the chunk lookup is hoisted.
    fn neighbour_face_solid(&self, block: [u32; 3], axis: usize, delta: i64) -> bool {
        // The face the NEIGHBOUR shares with `block`: stepping +1 lands on the neighbour's
        // LOW face (side 0); stepping −1 on its HIGH face (side 1).
        let facing_side = if delta > 0 { 0 } else { 1 };
        let stepped = block[axis] as i64 + delta;
        let mut local = block;
        let neighbour_chunk = if (0..CHUNK_BLOCKS as i64).contains(&stepped) {
            local[axis] = stepped as u32;
            Some(self.chunk)
        } else {
            local[axis] = stepped.rem_euclid(CHUNK_BLOCKS as i64) as u32;
            self.face_neighbours[axis][if delta > 0 { 1 } else { 0 }]
        };
        let Some(chunk) = neighbour_chunk else {
            return false;
        };
        if chunk.coarse_block(local).is_some() {
            true
        } else if let Some(geometry) = chunk.microblocks.get(&local) {
            geometry.seam_solidity.face_is_solid(axis, facing_side)
        } else {
            false
        }
    }
}

// The occupancy tile itself is substrate's `BitCube` (aliased to `BrickOccupancyTile` at the
// top-of-module seam): edge-≤64 word-packed 3D bitset, one voxel per bit. `empty`, `set_x_run`,
// `expand_to_bytes(byte)`, `from_bytes`, `is_set`, `popcount`, `edge()` live there. The atlas
// seam injects `SCULPTED_BRICK_OCCUPIED` as the "set-bit byte"; substrate names no such byte.

/// The single cell key shared by every cuboid of a boundary block, or `None` when they
/// disagree — the **uniform vs MIXED** classification, made at emission (the one place that
/// decides whether a block owns a cell-key tile). An empty block (no cuboids) is trivially
/// uniform at the fallback key `0`.
pub(crate) fn uniform_cell_key(geometry: &evaluation::two_layer_store::MicroblockGeometry) -> Option<u16> {
    let mut cuboids = geometry.cuboids.iter();
    let first = match cuboids.next() {
        Some(cuboid) => cuboid.material_id(),
        None => return Some(AIR_CELL_KEY_DONT_CARE),
    };
    cuboids
        .all(|cuboid| cuboid.material_id() == first)
        .then_some(first)
}

/// Rasterize one boundary block's cuboids into its `edge³` occupancy tile (block-local
/// x-fastest) and — for a MIXED block only — its per-voxel cell-key tile, in ONE walk over
/// the cuboids (the occupancy bit and the cell key of a voxel are written by the same X-run;
/// the tiles share their row layout, so a second pass would re-derive the same indices).
///
/// A cuboid's `material_id` IS its render-cell key (clean block id + overlay bit); the
/// occupancy tile never sees it (any voxel a cuboid covers is occupied), the cell-key tile
/// stores it verbatim. Air voxels of the cell-key tile keep [`AIR_CELL_KEY_DONT_CARE`] —
/// occupancy gates every read. `mixed` is the [`uniform_cell_key`] verdict: a uniform block
/// gets no tile (its one key rides on the record).
pub(crate) fn rasterize_brick_tiles(
    geometry: &evaluation::two_layer_store::MicroblockGeometry,
    brick_edge_voxels: u32,
    mixed: bool,
) -> (BrickOccupancyTile, Option<BrickCellKeyTile>) {
    let mut occupancy = BrickOccupancyTile::empty(brick_edge_voxels);
    let mut cell_keys = mixed
        .then(|| BrickCellKeyTile::new_filled(brick_edge_voxels, AIR_CELL_KEY_DONT_CARE));
    for cuboid in &geometry.cuboids {
        let cell_key = cuboid.material_id();
        for voxel_z in cuboid.min[2]..=cuboid.max[2] {
            for voxel_y in cuboid.min[1]..=cuboid.max[1] {
                occupancy.set_x_run(voxel_y, voxel_z, cuboid.min[0], cuboid.max[0]);
                if let Some(tile) = cell_keys.as_mut() {
                    tile.fill_x_run(voxel_y, voxel_z, cuboid.min[0], cuboid.max[0], cell_key);
                }
            }
        }
    }
    (occupancy, cell_keys)
}

/// One block's brick contribution, INDEPENDENT of atlas-slot assignment — the shared
/// classifier both the wholesale [`build_brick_field`] and the G3 incremental update
/// ([`IncrementalBrickField::apply_dirty_update`]) run, so a block classifies to the
/// exact same record kind + material + occupancy either way (only the slot NUMBER
/// differs: wholesale packs `0..count` in record order, incremental allocates from a
/// free-list). Keeping ONE classifier is what makes "incremental == wholesale byte-exact"
/// structural rather than a convention two code paths must independently uphold.
pub(crate) enum BlockBrick {
    /// Air — no record (ADR 0011 Decision 2).
    Air,
    /// A coarse-solid block: the whole record (no atlas slot).
    Coarse(BrickRecord),
    /// A boundary block: the record MINUS its slots (the caller's allocators assign them),
    /// the occupancy tile to land in its atlas slot, and — iff the block is MIXED — the
    /// per-voxel cell-key tile to land in its (independently pooled) material slot. A uniform
    /// block yields `cell_keys: None` and its one cell key as `material_id` + `overlay`.
    Sculpted {
        material_id: u16,
        overlay: bool,
        seam_solidity: SeamSolidity,
        tile: BrickOccupancyTile,
        cell_keys: Option<BrickCellKeyTile>,
    },
}

/// Classify one block of a [`TwoLayerChunk`] into its [`BlockBrick`] — the coarse XOR
/// boundary XOR air partition (ADR 0011 Decision 2). `world_block` is the block's
/// absolute world-block coordinate (its packed key).
pub(crate) fn classify_block_brick(
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
            // A coarse block's cell key is its id + the chunk's per-block overlay marker.
            overlay: chunk.coarse_block_overlay(block),
            payload: BrickPayload::CoarseSolid { block_id },
            // Fully solid through ⇒ every face is solid.
            seam_solidity: SeamSolidity {
                solid: [[true; 2]; 3],
            },
        })
    } else if let Some(geometry) = chunk.microblocks.get(&block) {
        // Uniform vs MIXED, decided here and nowhere else: all cuboids sharing ONE cell key
        // ⇒ that key rides on the record (no cell-key tile); disagreeing cuboids ⇒ a
        // per-voxel cell-key tile, and the record's material/overlay become don't-care (kept
        // as the first cuboid's, exactly as before, so a uniform scene's records are
        // byte-identical to the pre-material-atlas ones).
        let uniform = uniform_cell_key(geometry);
        let record_cell_key = uniform.unwrap_or_else(|| {
            geometry
                .cuboids
                .first()
                .map(|cuboid| cuboid.material_id())
                .unwrap_or(AIR_CELL_KEY_DONT_CARE)
        });
        let (tile, cell_keys) =
            rasterize_brick_tiles(geometry, brick_edge_voxels, uniform.is_none());
        BlockBrick::Sculpted {
            material_id: CellKey::from_raw(record_cell_key).block_id(),
            overlay: CellKey::from_raw(record_cell_key).has_overlay(),
            seam_solidity: geometry.seam_solidity,
            tile,
            cell_keys,
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
///
/// The tile-cube geometry + the per-row expand ARE substrate's [`CubeTilePacking`]; this seam
/// only injects [`SCULPTED_BRICK_OCCUPIED`] as the "set-bit byte" so everything GPU-facing keeps
/// consuming `sculpted_atlas_bytes` unchanged. Returns `(bricks_per_axis, atlas_dim_voxels,
/// bytes)`.
pub(crate) fn pack_sculpted_atlas(
    slot_tiles: &[BrickOccupancyTile],
    brick_edge_voxels: u32,
) -> (u32, u32, Vec<u8>) {
    CubeTilePacking::pack_bit_cubes(slot_tiles, brick_edge_voxels, SCULPTED_BRICK_OCCUPIED)
}

/// The absolute CHUNK coordinate that owns an absolute world block (`floor_div` by
/// [`CHUNK_BLOCKS`]) — the partition the resident cache dirties on, so a record can be
/// tested for membership in an edit's dirty-chunk set.
pub(crate) fn chunk_coord_of_world_block(world_block: [i64; 3]) -> [i32; 3] {
    let blocks = CHUNK_BLOCKS as i64;
    [
        world_block[0].div_euclid(blocks) as i32,
        world_block[1].div_euclid(blocks) as i32,
        world_block[2].div_euclid(blocks) as i32,
    ]
}
