use super::*;

/// A CPU march hit containing the hit voxel and entered-face normal.
///
/// Coordinates use the evaluator's absolute frame. The exact ±1 normal drives the
/// loaded-material `face_layer` rule used by the color-parity test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuMarchHit {
    pub absolute_voxel: [i32; 3],
    pub face_normal: [i32; 3],
}

/// The pixel-center camera ray in the shifted march frame — mirrors `camera_ray`:
/// the CAMERA-RELATIVE unproject yields eye-relative points, and `eye_sv` (the
/// pre-combined eye + half-extent + shift) carries the sv frame's one large term.
pub(crate) fn cpu_camera_ray(
    frame: &BrickMarchFrame,
    pixel: glam::Vec2,
) -> (glam::Vec3, glam::Vec3) {
    let ndc_x = (pixel.x - frame.viewport[0]) / frame.viewport[2] * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel.y - frame.viewport[1]) / frame.viewport[3] * 2.0;
    let near_h = frame.ray_inverse_unprojection * glam::Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let far_h = frame.ray_inverse_unprojection * glam::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    let near_eye_relative = near_h.truncate() / near_h.w;
    let far_eye_relative = far_h.truncate() / far_h.w;
    let direction = (far_eye_relative - near_eye_relative).normalize();
    (frame.eye_sv + near_eye_relative, direction)
}

/// Is a sculpted brick's block-local voxel occupied in the build's atlas bytes?
pub(crate) fn cpu_sculpted_voxel_occupied(
    build: &BrickFieldBuild,
    atlas_slot: u32,
    brick_local: [i32; 3],
) -> bool {
    let tiles = build.bricks_per_axis.max(1);
    let edge = build.brick_edge_voxels.max(1) as usize;
    let atlas_dim = build.atlas_dim_voxels as usize;
    let tile = [
        (atlas_slot % tiles) as usize,
        ((atlas_slot / tiles) % tiles) as usize,
        (atlas_slot / (tiles * tiles)) as usize,
    ];
    let coord = [
        tile[0] * edge + brick_local[0] as usize,
        tile[1] * edge + brick_local[1] as usize,
        tile[2] * edge + brick_local[2] as usize,
    ];
    build.sculpted_atlas_bytes[(coord[2] * atlas_dim + coord[1]) * atlas_dim + coord[0]] > 127
}

/// Binary-search the packed GPU records for a split key — mirrors the shader.
pub(crate) fn cpu_find_brick_record(
    records: &[BrickGpuRecord],
    key_hi: u32,
    key_lo: u32,
) -> Option<usize> {
    let key = ((key_hi as u64) << 32) | key_lo as u64;
    records
        .binary_search_by_key(&key, |record| {
            ((record.key_hi as u64) << 32) | record.key_lo as u64
        })
        .ok()
}

/// The split (hi, lo) key of an absolute block — mirrors the shader's packing.
pub(crate) fn cpu_pack_key_split(absolute_block: [i32; 3]) -> (u32, u32) {
    const BIAS: i32 = 1 << 20;
    let biased_x = (absolute_block[0] + BIAS) as u32;
    let biased_y = (absolute_block[1] + BIAS) as u32;
    let biased_z = (absolute_block[2] + BIAS) as u32;
    (
        (biased_z << 10) | (biased_y >> 11),
        ((biased_y & 0x7ff) << 21) | biased_x,
    )
}

/// Is the clip-map cell containing `absolute_block` occupied — or the level OFF
/// (empty ⇒ no hierarchical skip, the flat DDA)? Mirrors the shader's
/// `clipmap_cell_occupied`: floor-div the absolute block into the cell lattice,
/// pack the cell key, binary-search the sorted level.
pub(crate) fn cpu_clipmap_cell_occupied(level: &ClipmapLevel, absolute_block: glam::IVec3) -> bool {
    // Domain policy: a level with NO keys is "off" — never skip, so report every cell occupied
    // (the flat DDA). This "empty ⇒ occupied" reading is the domain's, not the kernel's; the
    // pure fold+binary-search below is substrate's `sorted_cell_keys_contain`.
    if level.cell_keys.is_empty() {
        return true;
    }
    substrate::spatial::min_mip_pyramid::sorted_cell_keys_contain(
        &level.cell_keys,
        [
            absolute_block.x as i64,
            absolute_block.y as i64,
            absolute_block.z as i64,
        ],
        level.blocks_per_cell,
    )
}

/// March one pixel-center ray through the brick field on the CPU.
///
/// This is a step-for-step f32 mirror of WGSL `march_brick_field`, including operation order,
/// tie-breaks, clamped boxes, residency misses, and hierarchical clip-map skips. Empty pyramid
/// levels select the flat block-DDA baseline.
pub fn cpu_march_brick_field(
    frame: &BrickMarchFrame,
    records: &[BrickGpuRecord],
    build: &BrickFieldBuild,
    pyramid: &ClipmapPyramid,
    pixel: glam::Vec2,
) -> Option<CpuMarchHit> {
    cpu_march_brick_field_counted(frame, records, build, pyramid, pixel).0
}

/// Run [`cpu_march_brick_field`] and count its block-DDA iterations.
///
/// Each iteration is one hierarchical jump or one per-block step; the count is the
/// empty-space-skip metric used by the scattered-scene performance probe.
pub fn cpu_march_brick_field_counted(
    frame: &BrickMarchFrame,
    records: &[BrickGpuRecord],
    build: &BrickFieldBuild,
    pyramid: &ClipmapPyramid,
    pixel: glam::Vec2,
) -> (Option<CpuMarchHit>, u32) {
    cpu_march_levels_counted(
        frame,
        records,
        build,
        &pyramid.levels_coarse_to_fine(),
        pixel,
    )
}

/// Run the core hierarchical-DDA CPU march over arbitrary clip-map levels.
///
/// Levels are ordered coarsest to finest, matching the shader's descent. Empty levels are
/// skipped, and the result contains the absolute hit voxel plus the block-DDA iteration count.
pub fn cpu_march_levels_counted(
    frame: &BrickMarchFrame,
    records: &[BrickGpuRecord],
    build: &BrickFieldBuild,
    levels_coarse_to_fine: &[&ClipmapLevel],
    pixel: glam::Vec2,
) -> (Option<CpuMarchHit>, u32) {
    // The pure hierarchical march lives in `raycast::march_brick_hierarchy` (the WGSL's
    // GPU-mirror specification). This function is the domain ADAPTER (the frame is carried,
    // never re-derived; see docs/architecture/03-display.md): it derives the ray from the frame,
    // packs the frame's plain numerics into the kernel's params, and builds the three
    // injected occupancy closures from the records/atlas/clip-map. The kernel's `MarchHit`
    // maps 1:1 onto `CpuMarchHit`.
    let (origin, direction) = cpu_camera_ray(frame, pixel);
    let params = raycast::HierarchicalMarchParams {
        traversal_lo: frame.traversal_lo,
        traversal_hi: frame.traversal_hi,
        brick_edge_voxels: frame.brick_edge_voxels,
        block_bias: glam::IVec3::from_array(frame.block_bias),
        voxel_bias: frame.voxel_bias,
        band_voxel_sv: frame.band_voxel_sv,
        level_blocks_per_cell: levels_coarse_to_fine
            .iter()
            .map(|level| level.blocks_per_cell as i32)
            .collect(),
    };
    let (hit, steps) = raycast::march_brick_hierarchy(
        substrate::spatial::Ray::new(origin, direction),
        &params,
        // Level-occupancy: the domain's "empty level ⇒ occupied (skip disabled)" policy
        // over substrate's sorted cell-key search.
        |level_index, absolute_block| {
            cpu_clipmap_cell_occupied(levels_coarse_to_fine[level_index], absolute_block)
        },
        // Per-block classification: the record binary search + the WGSL kind decode. A
        // sculpted block carries a closure over its atlas slot for the inner voxel DDA.
        |absolute_block| {
            let (key_hi, key_lo) = cpu_pack_key_split(absolute_block);
            match cpu_find_brick_record(records, key_hi, key_lo) {
                None => raycast::BlockContents::Empty,
                Some(record_index) => {
                    let record = records[record_index];
                    if record_is_coarse_form(&record) {
                        raycast::BlockContents::CoarseSolid
                    } else {
                        let atlas_slot = record.atlas_slot;
                        raycast::BlockContents::Sculpted(move |brick_local| {
                            cpu_sculpted_voxel_occupied(build, atlas_slot, brick_local)
                        })
                    }
                }
            }
        },
    );
    (
        hit.map(|hit| CpuMarchHit {
            absolute_voxel: hit.absolute_voxel,
            face_normal: hit.face_normal,
        }),
        steps,
    )
}

/// March a pixel-center ray over the evaluator's occupancy.
///
/// This plain voxel-level DDA uses the same frame and band without bricks or records. It is the
/// independent parity oracle for the brick march's hit-voxel set.
pub fn cpu_march_exact_occupancy(
    frame: &BrickMarchFrame,
    occupied: &dyn Fn([i64; 3]) -> bool,
    pixel: glam::Vec2,
) -> Option<CpuMarchHit> {
    // Domain adapter over `raycast::march_exact_occupancy` (the flat reference kernel):
    // derive the ray from the shifted frame, pass the band + biases, and forward the
    // absolute-voxel occupancy predicate unchanged. See docs/architecture/03-display.md.
    let (origin, direction) = cpu_camera_ray(frame, pixel);
    let params = raycast::ExactMarchParams {
        traversal_lo: frame.traversal_lo,
        traversal_hi: frame.traversal_hi,
        band_voxel_sv: frame.band_voxel_sv,
        voxel_bias: frame.voxel_bias,
    };
    raycast::march_exact_occupancy(
        substrate::spatial::Ray::new(origin, direction),
        &params,
        occupied,
    )
    .map(|hit| CpuMarchHit {
        absolute_voxel: hit.absolute_voxel,
        face_normal: hit.face_normal,
    })
}

/// Resolve the material used to shade a brick hit.
///
/// Mixed bricks read the same cell-key tile and hit voxel as the shader; coarse and
/// sculpted-uniform bricks use their per-record material. The GPU parity test compares this
/// result with [`BrickRaymarchRenderer::render_material_identity_image`].
///
/// The material is a DOMAIN fact (a cell key, a palette id, an overlay bit); the `raycast` kernel
/// stays material-free, so this resolves off the returned [`CpuMarchHit::absolute_voxel`] — the
/// hit voxel and the carried march frame's `brick_edge_voxels` recover the block and the
/// brick-local voxel exactly (`voxel_bias` is a multiple of the brick edge, so absolute-voxel
/// `div`/`rem` edge give the absolute block and brick-local coordinate the record search + tile
/// sample need).
pub fn cpu_brick_hit_material(
    records: &[BrickGpuRecord],
    build: &BrickFieldBuild,
    brick_edge_voxels: i32,
    hit: CpuMarchHit,
) -> u32 {
    let edge = brick_edge_voxels.max(1);
    let absolute_block = [
        hit.absolute_voxel[0].div_euclid(edge),
        hit.absolute_voxel[1].div_euclid(edge),
        hit.absolute_voxel[2].div_euclid(edge),
    ];
    let brick_local = [
        hit.absolute_voxel[0].rem_euclid(edge) as u32,
        hit.absolute_voxel[1].rem_euclid(edge) as u32,
        hit.absolute_voxel[2].rem_euclid(edge) as u32,
    ];
    let (key_hi, key_lo) = cpu_pack_key_split(absolute_block);
    match cpu_find_brick_record(records, key_hi, key_lo) {
        None => 0,
        Some(index) => {
            let record = records[index];
            if record_kind_discriminant(record.kind) == 2
                && record.cell_key_slot != NON_RESIDENT_ATLAS_SLOT
            {
                // The mixed brick's per-voxel cell key, masked to its clean block id — the CPU
                // twin of the shader's `mixed_voxel_material` (same tile, same voxel, same mask).
                let cell_key = build.cell_key_tiles[record.cell_key_slot as usize].get(
                    brick_local[0],
                    brick_local[1],
                    brick_local[2],
                );
                CellKey::from_raw(cell_key).block_id() as u32
            } else {
                record_material_id(record.kind)
            }
        }
    }
}
