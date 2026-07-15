use super::*;

/// A CPU march hit: the hit voxel in ABSOLUTE voxel coordinates (the exact
/// evaluator's frame), plus the entered face's outward normal as an exact ±1 axis
/// vector (`[i32; 3]`, so `Eq` still derives). The normal drives the loaded-material
/// shading rule (`face_layer`) the colour-parity test cross-checks (ADR 0011 G2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuMarchHit {
    pub absolute_voxel: [i32; 3],
    pub face_normal: [i32; 3],
}

/// The pixel-centre camera ray in the shifted march frame — mirrors `camera_ray`.
pub(crate) fn cpu_camera_ray(frame: &BrickMarchFrame, pixel: glam::Vec2) -> (glam::Vec3, glam::Vec3) {
    let ndc_x = (pixel.x - frame.viewport[0]) / frame.viewport[2] * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel.y - frame.viewport[1]) / frame.viewport[3] * 2.0;
    let near_h = frame.inverse_view_projection * glam::Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let far_h = frame.inverse_view_projection * glam::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    let near_world = near_h.truncate() / near_h.w;
    let far_world = far_h.truncate() / far_h.w;
    let direction = (far_world - near_world).normalize();
    let shift = glam::Vec3::new(
        frame.lattice_shift[0] as f32,
        frame.lattice_shift[1] as f32,
        frame.lattice_shift[2] as f32,
    );
    (near_world + frame.grid_half_extent + shift, direction)
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
pub(crate) fn cpu_find_brick_record(records: &[BrickGpuRecord], key_hi: u32, key_lo: u32) -> Option<usize> {
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
/// (empty ⇒ no hierarchical skip, the flat G1 DDA)? Mirrors the shader's
/// `clipmap_cell_occupied`: floor-div the absolute block into the cell lattice,
/// pack the cell key, binary-search the sorted level.
pub(crate) fn cpu_clipmap_cell_occupied(level: &ClipmapLevel, absolute_block: glam::IVec3) -> bool {
    // Domain policy: a level with NO keys is "off" — never skip, so report every cell occupied
    // (the flat G1 DDA). This "empty ⇒ occupied" reading is the domain's, not the kernel's; the
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

/// March one pixel-centre ray through the brick field on the CPU — a step-for-step
/// f32 mirror of the WGSL `march_brick_field` (same op order, same tie-breaks, same
/// clamped boxes, residency-miss branch, and G2 hierarchical clip-map skip),
/// returning the hit voxel in absolute coordinates. The parity net asserts the GPU
/// hit-identity image equals this. `pyramid` with empty levels is the "pyramid off"
/// form (the flat block-DDA) — the A/B baseline the pyramid-on == off parity uses.
pub fn cpu_march_brick_field(
    frame: &BrickMarchFrame,
    records: &[BrickGpuRecord],
    build: &BrickFieldBuild,
    pyramid: &ClipmapPyramid,
    pixel: glam::Vec2,
) -> Option<CpuMarchHit> {
    cpu_march_brick_field_counted(frame, records, build, pyramid, pixel).0
}

/// [`cpu_march_brick_field`] plus the number of block-DDA loop iterations the ray
/// took (each iteration is one hierarchical jump OR one per-block step) — the
/// empty-space-skip metric the scattered-scene perf probe reports pyramid on vs off.
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

/// The core hierarchical-DDA CPU march, generalized over an arbitrary set of
/// clip-map levels ordered COARSEST → FINEST (the shader's else-if descent, as a
/// loop). `cpu_march_brick_field_counted` passes the production pyramid's three
/// levels; the perf probe passes custom level sets (L2-only, +L3, +L4) to measure
/// each configuration's block-steps/ray honestly. An empty level (off) is skipped
/// over. Returns the hit voxel (absolute) plus the block-DDA iteration count.
pub fn cpu_march_levels_counted(
    frame: &BrickMarchFrame,
    records: &[BrickGpuRecord],
    build: &BrickFieldBuild,
    levels_coarse_to_fine: &[&ClipmapLevel],
    pixel: glam::Vec2,
) -> (Option<CpuMarchHit>, u32) {
    // The pure hierarchical march lives in `raycast::march_brick_hierarchy` (the WGSL's
    // GPU-mirror specification). This function is the domain ADAPTER (ADR 0008 carried
    // frame, docs/architecture/03-display.md): it derives the ray from the shifted frame,
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

/// March one pixel-centre ray over the EXACT evaluator's occupancy — a plain
/// voxel-level DDA (no bricks, no records) inside the same frame/band, querying
/// `occupied(absolute_voxel)`. This is the parity net's INDEPENDENT content
/// oracle: the brick march's hit-voxel set must equal this march's hit-voxel set
/// (ADR 0011 parity gate clause (b)).
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
    raycast::march_exact_occupancy(substrate::spatial::Ray::new(origin, direction), &params, |absolute| {
        occupied(absolute)
    })
    .map(|hit| CpuMarchHit {
        absolute_voxel: hit.absolute_voxel,
        face_normal: hit.face_normal,
    })
}

/// The MATERIAL a brick hit shades from — the CPU-march reference for ADR 0013's per-voxel
/// mixed shading (`docs/architecture/03-display.md`, the brick-field atlas). For a MIXED brick
/// (kind 2 with a resident cell-key slot) it samples the SAME cell-key tile at the SAME hit
/// voxel the shader's `mixed_voxel_material` reads and returns its clean block id; for a coarse
/// or sculpted-UNIFORM block it returns the per-record material. `tests/gpu_parity.rs` asserts
/// [`BrickRaymarchRenderer::render_material_identity_image`] equals this at every agreeing pixel.
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
