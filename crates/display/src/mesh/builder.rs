use super::*;

/// A built CPU mesh of a WHOLE grid's exposed cuboid faces (one flat vertex/index
/// list). This is the structural REFERENCE for the per-chunk apron mesher — the
/// parity test asserts the per-chunk-with-apron exposed-face SET equals this — and
/// the CPU adapter the older `build_cuboid_mesh*` tests exercise. The live GPU path
/// uses [`build_chunk_meshes_with_apron`] + per-chunk buffers, not this struct.
#[derive(Debug, Default, Clone)]
pub struct CuboidMesh {
    pub(crate) vertices: Vec<CuboidVertex>,
    /// Triangle indices for boxes WITHOUT the on-face-grid overlay (ADR 0003 §3c). The
    /// overlay-on boxes index into the same `vertices` via `indices_overlay`.
    pub(crate) indices: Vec<u32>,
    /// Triangle indices for the overlay-ON boxes (the split that replaced the per-vertex
    /// overlay flag, ADR 0010 E3). Empty whenever no box carried the overlay marker.
    pub(crate) indices_overlay: Vec<u32>,
    /// Number of boxes the grid decomposed into (diagnostic).
    pub(crate) box_count: u32,
}

impl CuboidMesh {
    /// Total number of triangles in the mesh (both overlay runs).
    pub fn triangle_count(&self) -> u32 {
        ((self.indices.len() + self.indices_overlay.len()) / 3) as u32
    }

    /// Total number of exposed quad faces (two triangles each, both overlay runs).
    pub fn face_count(&self) -> u32 {
        ((self.indices.len() + self.indices_overlay.len()) / 6) as u32
    }

    /// Number of vertices.
    pub fn vertex_count(&self) -> u32 {
        self.vertices.len() as u32
    }

    /// Number of indices (both overlay runs).
    pub fn index_count(&self) -> u32 {
        (self.indices.len() + self.indices_overlay.len()) as u32
    }

    /// Number of cuboid boxes the grid decomposed into.
    pub fn box_count(&self) -> u32 {
        self.box_count
    }
}

/// Build the exposed-face mesh for a whole [`VoxelGrid`], partitioned into the
/// same render chunks the instanced path uses (so the chunk world-AABBs frustum-
/// cull identically).
///
/// Exposed-face culling: the grid is decomposed into single-material boxes, then
/// for each box face we emit a quad only when the voxel cell on the far side of
/// that face is air (or outside the grid). This culls faces internal to the same
/// box AND faces against an adjacent solid voxel/box — the silhouette is the
/// outer surface of the solid set.
pub fn build_cuboid_mesh(grid: &VoxelGrid, voxels_per_block: u32) -> CuboidMesh {
    build_cuboid_mesh_banded(grid, voxels_per_block, LayerBand::FULL)
}

/// Build the exposed-face mesh CLIPPED to a layer-range band (issue #12 parity).
///
/// Z-up: layers are Z-slices. The cuboid path masks the densified region to the
/// band's absolute Z-layer range `[band.band_min, band.band_max]` (INCLUSIVE) BEFORE
/// decomposition. Masking (not a fragment discard) is required so the band's
/// top/bottom voxels expose real CAP faces: a single tall merged column has only one
/// +Z face — at the model's true top — so discarding its out-of-band fragments would
/// leave the displayed slab open-topped. Masking makes the cells just outside the
/// band air, so the greedy mesher caps the slab exactly like a per-voxel top/bottom.
///
/// `LayerBand::FULL` (band_max = u32::MAX) masks nothing — the full model is built,
/// byte-identical to the unbanded path.
pub fn build_cuboid_mesh_banded(
    grid: &VoxelGrid,
    _voxels_per_block: u32,
    band: LayerBand,
) -> CuboidMesh {
    let [grid_x, grid_y, grid_z] = grid.dimensions;
    if grid_x == 0 || grid_y == 0 || grid_z == 0 || grid.occupied.is_empty() {
        return CuboidMesh::default();
    }

    // Densify the WHOLE grid into a region anchored on the ACTUAL occupied voxel
    // cloud rather than assuming it is perfectly centred at `dimensions/2`. The
    // scene resolve path (`Scene::resolve_region`) can recentre a composite by a
    // non-zero offset (an odd block size shifts the cloud off the geometric
    // centre), so densifying with the project-wide `round(world + dimensions/2 -
    // 0.5)` convention anchored at index 0 mapped the shifted cloud partly OUT of
    // `[0, dimensions)` and silently dropped voxels — the cuboid cylinder lost
    // ~55% of its voxels this way and rendered a wedge. The instanced path is
    // immune because it draws raw `world_position`s; `region_from_voxel_cloud`
    // makes the cuboid path likewise shift-invariant, and returns the world offset
    // that places the mesh exactly where the instanced voxels sit.
    let (mut region, world_offset) = region_from_voxel_cloud(grid);

    // --- Layer-range band clip (issue #12 parity) ---
    // Z-up: layers are Z-slices. Mask region cells whose ABSOLUTE Z-layer falls
    // outside `[band_min, band_max]` to air, so the greedy mesher below produces real
    // cap faces at the band edges. The clip keys by the absolute layer
    // `floor(world_position.z + half_z)`; a region-local Z index `lz` maps to that
    // absolute layer by a constant `base_layer = floor(min_world.z + half_z)`
    // (= `floor(world_offset.z + 0.5 + half_z)`), so absolute layer = `base_layer +
    // lz`. We invert the band into region-local Z and clear everything outside it.
    if band.band_min > 0 || band.band_max != u32::MAX {
        // Corner-anchoring: FLOORED half so the absolute layer matches
        // `floor(world.z + floor(dim/2))` for any dim parity.
        let half_z = (grid_z / 2) as f32;
        let base_layer = (world_offset[2] + 0.5 + half_z).floor() as i64;
        // Region-local Z range that maps into [band_min, band_max] (inclusive).
        let local_lo = band.band_min as i64 - base_layer;
        let local_hi = band.band_max as i64 - base_layer;
        let [rx, ry, rz] = region.extent;
        for lz in 0..rz {
            let in_band = (lz as i64) >= local_lo && (lz as i64) <= local_hi;
            if in_band {
                continue;
            }
            for ly in 0..ry {
                for lx in 0..rx {
                    region.set(lx, ly, lz, None);
                }
            }
        }
    }

    let boxes = decompose_into_boxes(&region);

    // `world_offset` maps a REGION-LOCAL voxel index to its world min-corner plane at
    // the EXACT location the instanced path draws that voxel, i.e.
    // `min(world_position) - 0.5`. Adding it to a local index `l` gives the box's
    // world corner, so the reference mesh sits pixel-for-pixel on the instanced
    // voxels even when the scene recentred the cloud off the geometric centre.
    //
    // This WHOLE-GRID builder is the per-chunk mesher's structural REFERENCE (the
    // parity test asserts the per-chunk-with-apron exposed-face SET equals this), so
    // it emits one flat vertex/index list with no chunk partition (the per-chunk GPU
    // buffers come from [`build_chunk_meshes_with_apron`]).
    let mut vertices: Vec<CuboidVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut indices_overlay: Vec<u32> = Vec::new();
    let mut aabb = Aabb::empty();
    for voxel_box in &boxes {
        // ADR 0003 §3c: route each box's faces to the overlay-off or overlay-on index run
        // by its decomposition key's overlay bit (a box never spans both states).
        let index_sink = if box_has_overlay(voxel_box) {
            &mut indices_overlay
        } else {
            &mut indices
        };
        emit_box_faces(voxel_box, &region, world_offset, &mut vertices, index_sink, &mut aabb);
    }

    CuboidMesh {
        vertices,
        indices,
        indices_overlay,
        box_count: boxes.len() as u32,
    }
}

/// Densify a whole [`VoxelGrid`]'s occupied set into a [`VoxelRegion`] anchored on
/// the cloud's ACTUAL minimum voxel, returning the region plus the world-space
/// min-corner plane of region-local index `(0,0,0)`.
///
/// Unlike [`region_from_voxel_grid`] — which uses the project-wide
/// `round(world + dimensions/2 - 0.5)` index convention anchored at index 0 — this
/// anchors region-local index 0 at the cloud's own minimum voxel
/// (`round(world - min_world_center)`). That makes it **shift-invariant**: a
/// composite recentred off `dimensions/2` (e.g. an odd block size, via
/// `Scene::resolve_region`) still densifies into the region with no voxel falling
/// out of bounds — the previous "anchor at 0" densification silently dropped the
/// voxels whose shifted convention index went negative or past `dimensions` (the
/// cuboid cylinder lost ~55% of its voxels and rendered a wedge).
///
/// The returned `world_offset` is `min(world_position) - 0.5` per axis: adding it
/// to a region-local index reproduces the EXACT world position the instanced path
/// draws that voxel at, so the cuboid mesh overlays the instanced one pixel-for-
/// pixel. For a perfectly centred grid the indices and offset collapse to the old
/// behaviour (`world_offset = [-w/2, -h/2, -d/2]`).
///
/// Two distinct voxels can only collide on the same region index if they already
/// shared a world position (the grid is a set of distinct cells), so densification
/// is lossless. The region extent is the cloud's per-axis index span, never larger
/// than `grid.dimensions`.
pub(crate) fn region_from_voxel_cloud(grid: &VoxelGrid) -> (VoxelRegion, [f32; 3]) {
    if grid.occupied.is_empty() {
        return (VoxelRegion::new_empty([0, 0, 0]), [0.0; 3]);
    }

    // Pass 1: the cloud's minimum voxel centre per axis (the anchor).
    let mut min_world = [f32::INFINITY; 3];
    for voxel in &grid.occupied {
        let position = voxel.world_position();
        for (axis, min_axis) in min_world.iter_mut().enumerate() {
            *min_axis = min_axis.min(position[axis]);
        }
    }

    // Region index of a voxel = round(world_center - min_world_center) (≥ 0).
    let region_index = |world: [f32; 3]| -> [i64; 3] {
        [
            (world[0] - min_world[0]).round() as i64,
            (world[1] - min_world[1]).round() as i64,
            (world[2] - min_world[2]).round() as i64,
        ]
    };

    // Pass 2: the max index → region extent.
    let mut max_index = [0i64; 3];
    for voxel in &grid.occupied {
        let index = region_index(voxel.world_position());
        for axis in 0..3 {
            max_index[axis] = max_index[axis].max(index[axis]);
        }
    }
    let extent = [
        (max_index[0] + 1) as u32,
        (max_index[1] + 1) as u32,
        (max_index[2] + 1) as u32,
    ];

    // Pass 3: stamp the cuboid mesher's region-cell key (block_id + transient overlay
    // bit, ADR 0003 §3c) into the dense region.
    let mut region = VoxelRegion::new_empty(extent);
    for voxel in &grid.occupied {
        let [lx, ly, lz] = region_index(voxel.world_position());
        region.set(lx as u32, ly as u32, lz as u32, Some(voxel.cell_key().raw()));
    }

    // World min-corner plane of region-local index 0 = its centre minus 0.5.
    let world_offset = [
        min_world[0] - 0.5,
        min_world[1] - 0.5,
        min_world[2] - 0.5,
    ];
    (region, world_offset)
}

/// A built CPU mesh of ONE render chunk's exposed cuboid faces (issue #20 S6c-2d):
/// the chunk's absolute coord, its vertex/index buffers, and its world AABB for
/// frustum culling. Produced by [`build_chunk_meshes_with_apron`] and uploaded to
/// one [`CuboidChunkBuffers`] per chunk.
#[derive(Debug, Clone)]
pub struct CuboidChunkMesh {
    /// Absolute chunk coord (the coord `resident_render_chunks` reports).
    pub coord: [i32; 3],
    /// The chunk's exposed-face vertices.
    pub(crate) vertices: Vec<CuboidVertex>,
    /// Triangle indices for the overlay-OFF boxes into `vertices` (ADR 0003 §3c).
    pub(crate) indices: Vec<u32>,
    /// Triangle indices for the overlay-ON boxes into `vertices` (the split that replaced
    /// the per-vertex overlay flag, ADR 0010 E3). Empty when no box carried the marker.
    pub(crate) indices_overlay: Vec<u32>,
    /// World-space AABB of the chunk's emitted geometry (frustum cull key).
    pub(crate) aabb: Aabb,
    /// Boxes the chunk's interior decomposed into (diagnostic).
    pub(crate) box_count: u32,
}

impl CuboidChunkMesh {
    /// Total exposed quad faces (two triangles each, both overlay runs).
    pub fn face_count(&self) -> u32 {
        ((self.indices.len() + self.indices_overlay.len()) / 6) as u32
    }
    /// Total triangles (both overlay runs).
    pub fn triangle_count(&self) -> u32 {
        ((self.indices.len() + self.indices_overlay.len()) / 3) as u32
    }
    /// Boxes the chunk's interior decomposed into.
    pub fn box_count(&self) -> u32 {
        self.box_count
    }
}

/// Global absolute-voxel-index occupancy + anchor for a set of per-chunk grids.
///
/// `world_offset` is the world min-corner plane of absolute index `(0,0,0)` —
/// `min(world_position) - 0.5` over EVERY voxel in EVERY chunk grid (the same cloud
/// anchor [`region_from_voxel_cloud`] computes for the whole grid). `occupied` is a
/// DENSE row-major region (X fastest) of the union cloud, indexed DIRECTLY by the
/// absolute global index `round(world - min_world)` (which is `>= 0` per axis since
/// `min_world` is the per-axis minimum). `extent` is the union's per-axis index span.
///
/// A DENSE region (issue #20 perf) replaces the former `HashMap<[i64;3], u16>`: the
/// apron build then copies a contiguous sub-window per chunk instead of doing a hash
/// lookup per apron cell — the apron fill (per-cell `HashMap::get`) was the dominant
/// rebuild cost. Building it dense is O(voxels) with no hashing, and the per-chunk
/// window copy is row-major `memcpy`. The OUTPUT (occupancy queried) is identical.
pub(crate) struct GlobalOccupancy {
    world_offset: [f32; 3],
    extent: [u32; 3],
    occupied: Vec<Option<u16>>,
}

/// Build the global occupancy + cloud anchor over all per-chunk grids (issue #20
/// S6c-2d). The anchor is the union cloud's minimum voxel centre, identical to the
/// whole-region path's [`region_from_voxel_cloud`] anchor (the union of the chunk
/// grids IS the assembled whole grid, voxel-for-voxel, by the S6c-2a seam).
pub(crate) fn global_occupancy_from_chunks(chunk_grids: &[([i32; 3], &VoxelGrid)]) -> GlobalOccupancy {
    let mut min_world = [f32::INFINITY; 3];
    let mut max_world = [f32::NEG_INFINITY; 3];
    let mut any = false;
    for (_coord, grid) in chunk_grids {
        for voxel in &grid.occupied {
            any = true;
            let position = voxel.world_position();
            for axis in 0..3 {
                min_world[axis] = min_world[axis].min(position[axis]);
                max_world[axis] = max_world[axis].max(position[axis]);
            }
        }
    }
    if !any {
        return GlobalOccupancy {
            world_offset: [0.0; 3],
            extent: [0, 0, 0],
            occupied: Vec::new(),
        };
    }
    // Max absolute index per axis = round(max_world - min_world); extent = max + 1.
    let extent = [
        ((max_world[0] - min_world[0]).round() as i64 + 1) as u32,
        ((max_world[1] - min_world[1]).round() as i64 + 1) as u32,
        ((max_world[2] - min_world[2]).round() as i64 + 1) as u32,
    ];
    let [w, h, d] = extent;
    let mut occupied = vec![None; w as usize * h as usize * d as usize];
    for (_coord, grid) in chunk_grids {
        for voxel in &grid.occupied {
            let position = voxel.world_position();
            let x = (position[0] - min_world[0]).round() as u32;
            let y = (position[1] - min_world[1]).round() as u32;
            let z = (position[2] - min_world[2]).round() as u32;
            let flat = (z as usize * h as usize + y as usize) * w as usize + x as usize;
            occupied[flat] = Some(voxel.cell_key().raw());
        }
    }
    GlobalOccupancy {
        world_offset: [min_world[0] - 0.5, min_world[1] - 0.5, min_world[2] - 0.5],
        extent,
        occupied,
    }
}

/// Apron-aware per-chunk cuboid meshing (issue #20 S6c-2d) — the DEFAULT render
/// path, meshed one chunk at a time instead of densifying + greedy-decomposing the
/// WHOLE region.
///
/// For each `(coord, &grid)` chunk:
/// 1. Densify the chunk's OWN voxels into an interior region anchored on the global
///    cloud (so emitted world positions are byte-identical to the whole-region
///    mesher → pixel parity).
/// 2. Build a co-located APRON region of the same extent whose every cell — interior
///    AND the 1-voxel border — is filled from the GLOBAL occupancy. The apron is
///    used ONLY for [`face_is_exposed`] (no apron geometry is emitted), so a seam
///    face between two solid chunks is correctly culled and the chunk's exposed-face
///    SET equals the whole-region mesher's.
/// 3. Apply the layer-range band clip to the interior region per chunk (absolute
///    layers; the band edge synthesises real cap faces inside the chunk).
/// 4. `decompose_into_boxes` on the INTERIOR region (apron cells are air for
///    decomposition, so no box ever spans into the apron), then `emit_box_faces`
///    with exposure tested against the APRON region.
///
/// `grid_dimensions` is the whole composite grid's voxel dims; Z-up: only the Z half
/// is used (to map an absolute layer to the global region-local Z for the band clip,
/// since layers are Z-slices). Chunks that mesh to zero faces are omitted.
/// The apron-aware incremental rebuild plan for the cuboid mesher (issue #40).
pub struct CuboidRebuildPlan {
    /// Chunk coords to re-mesh + re-upload (occupied, and either changed or a
    /// neighbour-of-changed).
    pub rebuild: Vec<[i32; 3]>,
    /// Resident chunk coords to drop (no longer occupied — vacated or emptied).
    pub evict: Vec<[i32; 3]>,
}

/// Decide which chunks an edit forces the cuboid mesher to re-mesh, ACCOUNTING FOR THE
/// 1-VOXEL APRON: a chunk's boundary faces are culled against its neighbours
/// ([`build_chunk_meshes_with_apron`]), so a neighbour's occupancy change can alter
/// this chunk's mesh. This is the load-bearing difference from the instanced-era
/// [`evaluation::store::incremental_rebuild_plan`] (one-instance-per-voxel, no
/// inter-chunk dependency): here the dirty set is DILATED by the 26-neighbourhood.
///
/// - `resident` — the chunk coords whose state the renderer currently holds (its
///   `source_chunk_grids` coords, NOT just the buffered ones, so fully-occluded
///   occupied chunks stay stable instead of re-meshing every edit).
/// - `evicted_dirty` — the resolve cache's evicted coords for this edit (chunks whose
///   OWN occupancy may have changed; from [`evaluation::store::Store::invalidate_aabb`]).
/// - `occupied` — the post-edit covering coords that resolve to a NON-EMPTY grid.
///
/// `seed` = changed-occupancy chunks = `evicted_dirty` ∪ newly-appeared
/// (`occupied \ resident`). `rebuild` = `seed` dilated by the 26-neighbourhood ∩
/// `occupied` (only non-empty chunks are meshed; a neighbour that went empty drops out
/// here and re-exposes its occupied neighbours' faces, which ARE in `rebuild`).
/// `evict` = `resident \ occupied`. Applying this plan — re-mesh `rebuild`, drop
/// `evict`, keep the rest — yields a per-chunk buffer set byte-identical to a wholesale
/// rebuild (proven by the CPU parity test).
pub fn cuboid_incremental_plan(
    resident: &[[i32; 3]],
    evicted_dirty: &[[i32; 3]],
    occupied: &[[i32; 3]],
) -> CuboidRebuildPlan {
    use std::collections::HashSet;
    let resident_set: HashSet<[i32; 3]> = resident.iter().copied().collect();
    let occupied_set: HashSet<[i32; 3]> = occupied.iter().copied().collect();

    // seed = chunks whose occupancy changed: evicted (own may have changed) ∪
    // newly-appeared (occupied this rebuild but the renderer didn't know them before).
    let mut seed: HashSet<[i32; 3]> = evicted_dirty.iter().copied().collect();
    for coord in occupied {
        if !resident_set.contains(coord) {
            seed.insert(*coord);
        }
    }

    // Dilate the seed by the 26-neighbourhood (the apron footprint) and keep only
    // occupied coords — those are the chunks whose mesh can have changed.
    let mut rebuild_set: HashSet<[i32; 3]> = HashSet::new();
    for coord in &seed {
        for delta_z in -1..=1 {
            for delta_y in -1..=1 {
                for delta_x in -1..=1 {
                    let neighbour = [coord[0] + delta_x, coord[1] + delta_y, coord[2] + delta_z];
                    if occupied_set.contains(&neighbour) {
                        rebuild_set.insert(neighbour);
                    }
                }
            }
        }
    }
    let mut rebuild: Vec<[i32; 3]> = rebuild_set.into_iter().collect();
    rebuild.sort_unstable();

    // evict = resident coords that are no longer occupied (a removed/shrunk node
    // vacated them, or an edit turned them empty).
    let mut evict: Vec<[i32; 3]> = resident
        .iter()
        .copied()
        .filter(|coord| !occupied_set.contains(coord))
        .collect();
    evict.sort_unstable();
    evict.dedup();

    CuboidRebuildPlan { rebuild, evict }
}

pub(crate) fn build_chunk_meshes_with_apron(
    chunk_grids: &[([i32; 3], &VoxelGrid)],
    grid_dimensions: [u32; 3],
    band: LayerBand,
    region: Option<RegionClip>,
) -> Vec<CuboidChunkMesh> {
    build_chunk_meshes_with_apron_filtered(chunk_grids, None, grid_dimensions, band, region)
}

/// Like [`build_chunk_meshes_with_apron`] but meshes ONLY the chunks in `only`
/// (when `Some`). The global occupancy — hence every meshed chunk's apron — is still
/// computed from the FULL `chunk_grids` set, so a subset build is byte-identical to
/// the same chunks within a wholesale build. `None` meshes every chunk (the wholesale
/// path). This is the seam the INCREMENTAL rebuild uses: it passes the full resident
/// set for correct aprons but re-meshes only the dirty-dilated subset.
pub(crate) fn build_chunk_meshes_with_apron_filtered(
    chunk_grids: &[([i32; 3], &VoxelGrid)],
    only: Option<&std::collections::HashSet<[i32; 3]>>,
    grid_dimensions: [u32; 3],
    band: LayerBand,
    region: Option<RegionClip>,
) -> Vec<CuboidChunkMesh> {
    let global = global_occupancy_from_chunks(chunk_grids);
    if global.occupied.is_empty() {
        return Vec::new();
    }
    let world_offset = global.world_offset;

    // Z-up: the band clip works in GLOBAL absolute-index Z (layers are Z-slices). A
    // voxel's global index is `round(world - min_world)`; the absolute layer is
    // `floor(world.z + half_z)`. With `world.z = global_index_z + min_world.z` and
    // `min_world.z = world_offset.z + 0.5`, absolute layer = `global_index_z +
    // base_layer`, `base_layer = floor(world_offset.z + 0.5 + half_z)`. So a global
    // index Z is in-band iff `base_layer + gz ∈ [band_min, band_max]`.
    let band_active = band.band_min > 0 || band.band_max != u32::MAX;
    // Corner-anchoring: FLOORED half (matches `floor(world.z + floor(dim/2))`).
    let half_z = (grid_dimensions[2] / 2) as f32;
    let base_layer = (world_offset[2] + 0.5 + half_z).floor() as i64;
    let global_z_in_band = |gz: i64| -> bool {
        if !band_active {
            return true;
        }
        let layer = base_layer + gz;
        layer >= band.band_min as i64 && layer <= band.band_max as i64
    };

    // ADR 0018 Decision 5 — region-scoped clip. A global index `gi` sits at recentred
    // voxel coord `gi + world_offset` (both this dense path and the two-layer path emit
    // that voxel at the same recentred world position; `world_offset` is integer-valued —
    // `min_world` is a half-integer voxel centre — so the round is exact). The band test
    // in the recentred frame is `layer = recentred_z + floor(dim_z/2) ∈ [band_min,
    // band_max]`, equal to `global_z_in_band(gz)`.
    let half_z_i = (grid_dimensions[2] / 2) as i64;
    let world_offset_int = [
        world_offset[0].round() as i64,
        world_offset[1].round() as i64,
        world_offset[2].round() as i64,
    ];
    let z_in_band_recentred = |recentred_z: i64| -> bool {
        if !band_active {
            return true;
        }
        let layer = recentred_z + half_z_i;
        layer >= band.band_min as i64 && layer <= band.band_max as i64
    };
    // Whether the voxel at global index `gi` is meshed (band + optional region clip).
    let voxel_shown = |gi: [i64; 3]| -> bool {
        let recentred = [
            gi[0] + world_offset_int[0],
            gi[1] + world_offset_int[1],
            gi[2] + world_offset_int[2],
        ];
        voxel_meshed(recentred, &z_in_band_recentred, region)
    };

    let mut meshes = Vec::new();
    for (coord, grid) in chunk_grids {
        if grid.occupied.is_empty() {
            continue;
        }
        // Incremental subset: skip chunks not in the rebuild set. The apron still sees
        // every chunk (global occupancy above is over the FULL set), so a skipped
        // neighbour's occupancy correctly culls the meshed chunk's seam faces.
        if let Some(only) = only {
            if !only.contains(coord) {
                continue;
            }
        }
        // The chunk's own voxels as global absolute indices (band-clipped).
        let mut chunk_indices: Vec<([i64; 3], u16)> = Vec::with_capacity(grid.occupied.len());
        let mut gmin = [i64::MAX; 3];
        let mut gmax = [i64::MIN; 3];
        for voxel in &grid.occupied {
            let position = voxel.world_position();
            let index = [
                (position[0] - (world_offset[0] + 0.5)).round() as i64,
                (position[1] - (world_offset[1] + 0.5)).round() as i64,
                (position[2] - (world_offset[2] + 0.5)).round() as i64,
            ];
            if !voxel_shown(index) {
                continue;
            }
            for axis in 0..3 {
                gmin[axis] = gmin[axis].min(index[axis]);
                gmax[axis] = gmax[axis].max(index[axis]);
            }
            chunk_indices.push((index, voxel.cell_key().raw()));
        }
        if chunk_indices.is_empty() {
            continue; // every voxel clipped away by the band
        }

        // Region-local origin = chunk min minus one apron cell; extent spans the
        // chunk's voxels plus a 1-cell apron on every side.
        let origin = [gmin[0] - 1, gmin[1] - 1, gmin[2] - 1];
        let extent = [
            (gmax[0] - gmin[0] + 3) as u32,
            (gmax[1] - gmin[1] + 3) as u32,
            (gmax[2] - gmin[2] + 3) as u32,
        ];

        // Interior region: ONLY this chunk's own voxels (apron stays air, so the
        // decomposition never grows a box into the apron).
        let mut interior = VoxelRegion::new_empty(extent);
        for (index, material) in &chunk_indices {
            let lx = (index[0] - origin[0]) as u32;
            let ly = (index[1] - origin[1]) as u32;
            let lz = (index[2] - origin[2]) as u32;
            interior.set(lx, ly, lz, Some(*material));
        }

        // Apron region: same frame; every cell (interior + border) read from the
        // GLOBAL occupancy, BAND-CLIPPED exactly as the interior — so a seam
        // neighbour that the band masked out reads as air and the cap face is
        // synthesised, identical to whole-region meshing under the same band.
        //
        // The global occupancy is a DENSE row-major region (issue #20 perf), so a
        // chunk's apron window `[origin, origin+extent)` is a contiguous run per X
        // row: copy each in-bounds, in-band row with `copy_from_slice` instead of a
        // per-cell hash lookup (the former per-cell `HashMap::get` dominated the
        // rebuild). Rows outside the global extent or out of band stay air. The
        // queried occupancy — hence the meshed output — is identical.
        let mut apron = VoxelRegion::new_empty(extent);
        let [gw, gh, gd] = global.extent;
        let [aw, ah, _ad] = extent;
        if region.is_none() {
            // Band-only (or FULL) path: a whole out-of-band Z plane reads as air (cap face
            // at the band edge), and each in-band row is copied with `copy_from_slice`.
            for lz in 0..extent[2] {
                let gz = origin[2] + lz as i64;
                if gz < 0 || gz >= gd as i64 || !global_z_in_band(gz) {
                    continue;
                }
                for ly in 0..extent[1] {
                    let gy = origin[1] + ly as i64;
                    if gy < 0 || gy >= gh as i64 {
                        continue;
                    }
                    // The apron row spans global X in `[origin.x, origin.x + aw)`; clip
                    // it to the global region's `[0, gw)` and copy the overlap directly.
                    let row_gx0 = origin[0].max(0);
                    let row_gx1 = (origin[0] + aw as i64).min(gw as i64);
                    if row_gx1 <= row_gx0 {
                        continue;
                    }
                    let src_base =
                        (gz as usize * gh as usize + gy as usize) * gw as usize + row_gx0 as usize;
                    let len = (row_gx1 - row_gx0) as usize;
                    let dst_lx = (row_gx0 - origin[0]) as u32;
                    let dst_base =
                        (lz as usize * ah as usize + ly as usize) * aw as usize + dst_lx as usize;
                    apron.cells[dst_base..dst_base + len]
                        .copy_from_slice(&global.occupied[src_base..src_base + len]);
                }
            }
        } else {
            // ADR 0018 Decision 5 — region-scoped clip: the mask is XY-dependent, so fill
            // the apron PER CELL (the region clip is an onion-mode-only path; the dense
            // mesher here is the shot/oracle reference, not the live incremental path).
            for lz in 0..extent[2] {
                let gz = origin[2] + lz as i64;
                if gz < 0 || gz >= gd as i64 {
                    continue;
                }
                for ly in 0..extent[1] {
                    let gy = origin[1] + ly as i64;
                    if gy < 0 || gy >= gh as i64 {
                        continue;
                    }
                    for lx in 0..extent[0] {
                        let gx = origin[0] + lx as i64;
                        if gx < 0 || gx >= gw as i64 {
                            continue;
                        }
                        if !voxel_shown([gx, gy, gz]) {
                            continue;
                        }
                        let src = (gz as usize * gh as usize + gy as usize) * gw as usize
                            + gx as usize;
                        let dst = (lz as usize * ah as usize + ly as usize) * aw as usize
                            + lx as usize;
                        apron.cells[dst] = global.occupied[src];
                    }
                }
            }
        }

        // The world offset that maps this region's local index 0 to world space:
        // global index 0 sits at `world_offset`, and local 0 = global `origin`, so
        // the region's local offset is `world_offset + origin`.
        let region_offset = [
            world_offset[0] + origin[0] as f32,
            world_offset[1] + origin[1] as f32,
            world_offset[2] + origin[2] as f32,
        ];

        let boxes = decompose_into_boxes(&interior);
        let mut vertices: Vec<CuboidVertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut indices_overlay: Vec<u32> = Vec::new();
        let mut aabb = Aabb::empty();
        for voxel_box in &boxes {
            // Decompose on the interior region but test exposure against the apron.
            // ADR 0003 §3c: route to the overlay-off / overlay-on index run by the box key.
            let index_sink = if box_has_overlay(voxel_box) {
                &mut indices_overlay
            } else {
                &mut indices
            };
            emit_box_faces(voxel_box, &apron, region_offset, &mut vertices, index_sink, &mut aabb);
        }
        if indices.is_empty() && indices_overlay.is_empty() {
            continue;
        }
        meshes.push(CuboidChunkMesh {
            coord: *coord,
            vertices,
            indices,
            indices_overlay,
            aabb,
            box_count: boxes.len() as u32,
        });
    }
    meshes
}
