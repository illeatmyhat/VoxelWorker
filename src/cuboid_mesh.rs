//! Cuboid mesh render path (ADR 0002 E3b-1, part of #18) — BEHIND A FLAG.
//!
//! The instanced renderer (`crate::renderer::VoxelRenderer`) draws one cube
//! per occupied voxel. This module is the FIRST step of replacing that with a
//! Vintage-Story-style **cuboid mesher**: it decomposes the resolved grid into a
//! small set of single-material axis-aligned boxes ([`crate::cuboid`]) and builds
//! a triangle mesh of each box's **exposed faces only** (faces internal to the
//! solid set are culled). Each face vertex carries the box's `material_id` and a
//! face normal; the shader (`shaders/cuboid.wgsl`) flat-shades it with the same
//! normal-based lighting + per-material base-colour modulation the instanced
//! path uses.
//!
//! SCOPE (E3b-1): SHAPE parity + per-box material colour + basic lighting.
//! SCOPE (E3b-2, this sub-step): adds the per-voxel TEXTURE SLICE (block texture
//! tiled once per voxel across a merged box face, via a voxel-unit UV + a Repeat
//! sampler, replicating the instanced per-face UV direction so even non-symmetric
//! textures land texel-exact), the per-face D2Array layer selection from the face
//! normal, and the position-based per-voxel/per-block GRID OVERLAY — all matching
//! the instanced path. STILL NO layer-range clip, NO debug-faces (later E3 sub-
//! steps). The instanced path stays the DEFAULT and is untouched; this path is
//! selected only when the `cuboid` mesher flag is on.
//!
//! ## Geometry / coordinate mapping
//! A voxel at region-local index `(x, y, z)` occupies the world-space cell
//! `[i - half, i+1 - half]` per axis, where `i` is the ABSOLUTE voxel index and
//! `half = dimensions / 2`. This matches the instanced path, where a voxel cube
//! is centred at `world_position = i + 0.5 - half` and spans centre ± 0.5. Since
//! we decompose the whole grid with `origin = [0,0,0]`, the region-local index IS
//! the absolute index, so a box spanning voxels `min..=max` becomes the world AABB
//! `[min - half, (max+1) - half]`.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;
use wgpu::util::DeviceExt;

use crate::core_geom::{MaterialChoice, CHUNK_BLOCKS};
use crate::cuboid::{decompose_into_boxes, VoxelBox, VoxelBoxMaterial, VoxelRegion};
use substrate::solids::CulledBoxMeshing;
use camera::frustum::Frustum;
use substrate::spatial::RealAabb as Aabb;
use crate::renderer::{LayerBand, DEPTH_FORMAT, MSAA_SAMPLE_COUNT};
use crate::texture_atlas::MaterialAtlas;
use crate::core_geom::CellKey;
use crate::two_layer_store::{MicroblockGeometry, SeamSolidity, TwoLayerChunk};
use crate::voxel::{RecentreVoxels, VoxelGrid};

/// Compose the cuboid mesher's region-cell key for one resolved voxel (ADR 0003 §3c):
/// the clean categorical colour index in the low bits, the transient on-face-grid marker
/// in the high bit. A thin `Voxel`-reading wrapper over [`CellKey::compose`] so callers
/// that already hold a `Voxel` (a higher layer than `core_geom`) build the SAME key. The
/// overlay bit lives ONLY in this render-side key — never in the persistent `Voxel`
/// payload, the chunk-storage codec, or the `.vox` export.
pub fn mesh_cell_key(voxel: &crate::voxel::Voxel) -> u16 {
    CellKey::compose(voxel.color_index(), voxel.grid_overlay).raw()
}

/// One mesh vertex of a cuboid face: world position, the face's outward normal, and the
/// box's `block_id` (the clean colour index, constant across the face).
///
/// ADR 0010 E3 / ADR 0003 §3c: the on-face-grid overlay flag is **no longer a vertex
/// attribute**. A chunk mesh is SPLIT into an overlay-off and an overlay-on index run over
/// this one shared vertex list (a box never spans both — the overlay bit is part of the
/// decomposition key), and the draw selects the per-draw overlay-active uniform per run. So
/// the render flag is entirely out of the per-vertex format while the per-object behaviour
/// (the `voxel_grid_flag_bit_is_per_object` invariant) is preserved by the split.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CuboidVertex {
    position: [f32; 3],
    normal: [f32; 3],
    material_id: u32,
}

/// The six cube-face directions, each with its outward normal and the four
/// corner offsets (in voxel units, relative to the box's min corner, scaled by
/// the box's extent) wound COUNTER-CLOCKWISE when viewed from OUTSIDE — so
/// `front_face: Ccw` + `cull_mode: Back` keeps the outward faces (matching the
/// instanced cube's winding convention in `renderer::unit_cube_geometry`).
///
/// Each corner is `[x, y, z]` in {0,1}: 0 = the box's min-corner plane on that
/// axis, 1 = its max-corner plane. The mesh builder maps 0→`min` and
/// 1→`max+1` (inclusive box → exclusive far plane) to get the world corner.
struct FaceTemplate {
    /// `+1`/`-1` direction along the axis this face faces; used both for the
    /// outward normal and to find the neighbour cell to test for exposure.
    neighbor_delta: [i32; 3],
    normal: [f32; 3],
    /// Four corners as {0,1} per axis, CCW from outside.
    corners: [[u32; 3]; 4],
}

const FACE_TEMPLATES: [FaceTemplate; 6] = [
    // +X
    FaceTemplate {
        neighbor_delta: [1, 0, 0],
        normal: [1.0, 0.0, 0.0],
        corners: [[1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0]],
    },
    // -X
    FaceTemplate {
        neighbor_delta: [-1, 0, 0],
        normal: [-1.0, 0.0, 0.0],
        corners: [[0, 1, 1], [0, 1, 0], [0, 0, 0], [0, 0, 1]],
    },
    // +Y
    FaceTemplate {
        neighbor_delta: [0, 1, 0],
        normal: [0.0, 1.0, 0.0],
        corners: [[0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]],
    },
    // -Y
    FaceTemplate {
        neighbor_delta: [0, -1, 0],
        normal: [0.0, -1.0, 0.0],
        corners: [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
    },
    // +Z
    FaceTemplate {
        neighbor_delta: [0, 0, 1],
        normal: [0.0, 0.0, 1.0],
        corners: [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    },
    // -Z
    FaceTemplate {
        neighbor_delta: [0, 0, -1],
        normal: [0.0, 0.0, -1.0],
        corners: [[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]],
    },
];

/// A built CPU mesh of a WHOLE grid's exposed cuboid faces (one flat vertex/index
/// list). This is the structural REFERENCE for the per-chunk apron mesher — the
/// parity test asserts the per-chunk-with-apron exposed-face SET equals this — and
/// the CPU adapter the older `build_cuboid_mesh*` tests exercise. The live GPU path
/// uses [`build_chunk_meshes_with_apron`] + per-chunk buffers, not this struct.
#[derive(Debug, Default, Clone)]
pub struct CuboidMesh {
    vertices: Vec<CuboidVertex>,
    /// Triangle indices for boxes WITHOUT the on-face-grid overlay (ADR 0003 §3c). The
    /// overlay-on boxes index into the same `vertices` via `indices_overlay`.
    indices: Vec<u32>,
    /// Triangle indices for the overlay-ON boxes (the split that replaced the per-vertex
    /// overlay flag, ADR 0010 E3). Empty whenever no box carried the overlay marker.
    indices_overlay: Vec<u32>,
    /// Number of boxes the grid decomposed into (diagnostic).
    box_count: u32,
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
fn region_from_voxel_cloud(grid: &VoxelGrid) -> (VoxelRegion, [f32; 3]) {
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
        region.set(lx as u32, ly as u32, lz as u32, Some(mesh_cell_key(voxel)));
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
    vertices: Vec<CuboidVertex>,
    /// Triangle indices for the overlay-OFF boxes into `vertices` (ADR 0003 §3c).
    indices: Vec<u32>,
    /// Triangle indices for the overlay-ON boxes into `vertices` (the split that replaced
    /// the per-vertex overlay flag, ADR 0010 E3). Empty when no box carried the marker.
    indices_overlay: Vec<u32>,
    /// World-space AABB of the chunk's emitted geometry (frustum cull key).
    aabb: Aabb,
    /// Boxes the chunk's interior decomposed into (diagnostic).
    box_count: u32,
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
struct GlobalOccupancy {
    world_offset: [f32; 3],
    extent: [u32; 3],
    occupied: Vec<Option<u16>>,
}

/// Build the global occupancy + cloud anchor over all per-chunk grids (issue #20
/// S6c-2d). The anchor is the union cloud's minimum voxel centre, identical to the
/// whole-region path's [`region_from_voxel_cloud`] anchor (the union of the chunk
/// grids IS the assembled whole grid, voxel-for-voxel, by the S6c-2a seam).
fn global_occupancy_from_chunks(chunk_grids: &[([i32; 3], &VoxelGrid)]) -> GlobalOccupancy {
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
            occupied[flat] = Some(mesh_cell_key(voxel));
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
/// [`crate::store::incremental_rebuild_plan`] (one-instance-per-voxel, no
/// inter-chunk dependency): here the dirty set is DILATED by the 26-neighbourhood.
///
/// - `resident` — the chunk coords whose state the renderer currently holds (its
///   `source_chunk_grids` coords, NOT just the buffered ones, so fully-occluded
///   occupied chunks stay stable instead of re-meshing every edit).
/// - `evicted_dirty` — the resolve cache's evicted coords for this edit (chunks whose
///   OWN occupancy may have changed; from [`crate::store::Store::invalidate_aabb`]).
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

fn build_chunk_meshes_with_apron(
    chunk_grids: &[([i32; 3], &VoxelGrid)],
    grid_dimensions: [u32; 3],
    band: LayerBand,
) -> Vec<CuboidChunkMesh> {
    build_chunk_meshes_with_apron_filtered(chunk_grids, None, grid_dimensions, band)
}

/// Like [`build_chunk_meshes_with_apron`] but meshes ONLY the chunks in `only`
/// (when `Some`). The global occupancy — hence every meshed chunk's apron — is still
/// computed from the FULL `chunk_grids` set, so a subset build is byte-identical to
/// the same chunks within a wholesale build. `None` meshes every chunk (the wholesale
/// path). This is the seam the INCREMENTAL rebuild uses: it passes the full resident
/// set for correct aprons but re-meshes only the dirty-dilated subset.
fn build_chunk_meshes_with_apron_filtered(
    chunk_grids: &[([i32; 3], &VoxelGrid)],
    only: Option<&std::collections::HashSet<[i32; 3]>>,
    grid_dimensions: [u32; 3],
    band: LayerBand,
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
            if !global_z_in_band(index[2]) {
                continue;
            }
            for axis in 0..3 {
                gmin[axis] = gmin[axis].min(index[axis]);
                gmax[axis] = gmax[axis].max(index[axis]);
            }
            chunk_indices.push((index, mesh_cell_key(voxel)));
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
        for lz in 0..extent[2] {
            let gz = origin[2] + lz as i64;
            // Z-up: the band clip is along Z (layers are Z-slices), so an out-of-band
            // Z plane reads as air — synthesising the cap face at the band edge.
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

// ===========================================================================
// ADR 0010 E3 — the TWO-LAYER mesher (one-box coarse + cuboid microblock +
// seam-flag culling). Builds a chunk's mesh from its [`TwoLayerChunk`] instead of
// a dense `VoxelGrid`, and PROVES (the E3 parity gate) the exposed-face set is
// identical to the dense [`build_chunk_meshes_with_apron`].
// ===========================================================================

/// Whether the WHOLE shared face of one block is solid, for seam-flag culling (ADR 0010
/// Decision 4). A face that is fully solid backs every cell on the neighbour's matching
/// face, so the neighbour's face there is occluded and culled. A coarse-solid block is
/// solid on all 6 faces; an air block on none; a boundary block per its [`SeamSolidity`].
/// `None` = the block is air / does not exist (no covering chunk) ⇒ never solid.
#[derive(Debug, Clone, Copy)]
enum BlockFaceSolidity {
    /// Every face fully solid (a coarse-solid block, or a fully-interior boundary block).
    AllSolid,
    /// Per-face solidity (a boundary block's stored seam flags).
    PerFace(SeamSolidity),
    /// Air / outside any covering chunk — no face is solid.
    None,
}

impl BlockFaceSolidity {
    /// Whether this block's face on `axis` (0/1/2), `side` (0 low / 1 high) is fully solid.
    fn face_is_solid(&self, axis: usize, side: usize) -> bool {
        match self {
            BlockFaceSolidity::AllSolid => true,
            BlockFaceSolidity::PerFace(seam) => seam.face_is_solid(axis, side),
            BlockFaceSolidity::None => false,
        }
    }
}

/// The PER-CELL occupancy of a neighbour block's face abutting a boundary block (ADR 0010 E3).
/// A coarse-solid neighbour is `Solid` (the seam-flag fast path — no densification); an air /
/// missing neighbour is `Air`; a boundary neighbour carries its face layer's `density²`
/// occupancy bitmap. This is the exact neighbour info the dense apron carried, restricted to
/// the SURFACE blocks so coarse interiors are never densified.
enum NeighbourFace {
    /// The whole face is solid (a coarse-solid neighbour).
    Solid,
    /// The whole face is air (no covering chunk, or an air block).
    Air,
    /// Per-cell occupancy, indexed `cells[in_plane_b * density + in_plane_a]` over the two
    /// axes other than the face axis (ascending order — see [`in_plane_axes`]).
    Cells(Vec<bool>),
}

/// The two axes IN the plane of a face whose normal is along `axis` (0/1/2 = X/Y/Z), in
/// ascending order: axis 0 → (1, 2), axis 1 → (0, 2), axis 2 → (0, 1). The canonical
/// in-plane (a, b) ordering both the neighbour-face bitmap and the apron fill index by.
#[inline]
fn in_plane_axes(axis: usize) -> (usize, usize) {
    match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

/// Build the per-chunk exposed-face meshes from the two-layer chunks (ADR 0010 E3). A
/// coarse-solid block emits ONE box (no per-voxel decompose of the solid interior); a
/// boundary block emits its stored microblock cuboids; inter-block / inter-chunk seam
/// faces are culled via the per-face seam-solidity flags (the coarse-vs-microblock apron
/// analogue) rather than a densified neighbour apron.
///
/// `chunks` is `(absolute_chunk_coord, TwoLayerChunk)` per covering chunk;
/// `grid_dimensions` is the whole composite voxel dims — only the Z half is read, to map a
/// recentred-frame voxel index to its ABSOLUTE layer for the band clip (Z-up: layers are
/// Z-slices); `recentre_voxels` is the resolve's carried recentre (ADR 0008) so the emitted
/// vertices land in the SAME world frame the dense path assembles (its global cloud-min
/// anchor cancels to exactly this recentred index — proven in the E3 parity test).
/// `voxels_per_block` is the chunk density.
///
/// `band` (ADR 0010 #53): a layer-range (Z-slice) clip. `LayerBand::FULL` (the default) keeps
/// the E3-proven FAST paths byte-for-byte — a coarse-solid block is ONE box, a boundary block
/// its stored cuboids. An ACTIVE band (the layer scrubber) clips each block to the band's
/// recentred voxel-Z range: a coarse block the band CUTS through emits the clipped one-box (the
/// block ∩ band), a boundary block clips each cuboid; blocks fully outside the band are skipped.
/// Cut-plane faces are VISIBLE — a band edge reads the out-of-band neighbour cell as AIR, so the
/// clip synthesises a real cap face there, mirroring the dense [`build_chunk_meshes_with_apron`]
/// banded behaviour exactly (it masks the apron + interior so a merged column caps at the edge).
fn build_two_layer_chunk_meshes(
    chunks: &[([i32; 3], Arc<TwoLayerChunk>)],
    grid_dimensions: [u32; 3],
    recentre: RecentreVoxels,
    voxels_per_block: u32,
    band: LayerBand,
) -> Vec<CuboidChunkMesh> {
    build_two_layer_chunk_meshes_filtered(
        chunks,
        None,
        grid_dimensions,
        recentre,
        voxels_per_block,
        band,
    )
}

/// Like [`build_two_layer_chunk_meshes`] but meshes ONLY the chunks in `only` (when
/// `Some`), the two-layer analogue of [`build_chunk_meshes_with_apron_filtered`] (issue
/// #55). Seam-flag culling reads every chunk's neighbours from the FULL `chunks` set (the
/// `chunk_by_coord` lookup below is over ALL chunks), so a subset build is byte-identical to
/// the same chunks within a wholesale build — a skipped neighbour's coarse / microblock face
/// solidity still culls the meshed chunk's seam faces. `None` meshes every chunk (the
/// wholesale path). This is the seam the two-layer INCREMENTAL rebuild
/// ([`CuboidMeshRenderer::incremental_rebuild_from_two_layer_chunks`]) uses: it passes the
/// full resident set for correct seam culling but re-meshes only the dirty-dilated subset.
fn build_two_layer_chunk_meshes_filtered(
    chunks: &[([i32; 3], Arc<TwoLayerChunk>)],
    only: Option<&std::collections::HashSet<[i32; 3]>>,
    grid_dimensions: [u32; 3],
    recentre: RecentreVoxels,
    voxels_per_block: u32,
    band: LayerBand,
) -> Vec<CuboidChunkMesh> {
    // Unwrap the carried frame at the per-chunk rebase arithmetic below (`chunk_min_recentred`).
    let recentre_voxels = recentre.voxels();
    let density = voxels_per_block.max(1);
    let block_extent = density as i64;

    // Z-up band clip (ADR 0010 #53): the band is in ABSOLUTE layer indices. A voxel at
    // recentred-frame min-corner `v` (the frame this mesher emits in) sits at world.z = v +
    // 0.5, so its absolute layer = floor(world.z + half_z) = v + half_z (integer-valued for
    // an integer `v`, `half_z`). Inverting the band into the recentred frame: a recentred
    // voxel-Z `v` is in-band iff `band_min - half_z <= v <= band_max - half_z`. FLOORED half
    // (matches the dense path's `floor(world.z + floor(dim/2))` for any dim parity).
    let band_active = band.band_min > 0 || band.band_max != u32::MAX;
    let half_z = (grid_dimensions[2] / 2) as i64;
    let band_lo_recentred = band.band_min as i64 - half_z;
    let band_hi_recentred = (band.band_max as i64).saturating_sub(half_z);
    // Whether a recentred-frame voxel-Z index is inside the band.
    let z_in_band = |recentred_z: i64| -> bool {
        if !band_active {
            return true;
        }
        recentred_z >= band_lo_recentred && recentred_z <= band_hi_recentred
    };
    let chunk_extent_voxels = (CHUNK_BLOCKS * density) as i64;

    // A lookup of every covering chunk by coord so a block can consult its neighbour's
    // coarse / microblock face solidity across a block OR chunk seam.
    let chunk_by_coord: std::collections::HashMap<[i32; 3], &TwoLayerChunk> =
        chunks.iter().map(|(coord, chunk)| (*coord, chunk.as_ref())).collect();

    // The block-face solidity of the block at ABSOLUTE block coord `abs_block` (across all
    // chunks): resolve which chunk + chunk-local block it is, then read its layer.
    let face_solidity_at = |abs_block: [i64; 3]| -> BlockFaceSolidity {
        let chunk_blocks = CHUNK_BLOCKS as i64;
        let chunk_coord = [
            abs_block[0].div_euclid(chunk_blocks) as i32,
            abs_block[1].div_euclid(chunk_blocks) as i32,
            abs_block[2].div_euclid(chunk_blocks) as i32,
        ];
        let Some(chunk) = chunk_by_coord.get(&chunk_coord) else {
            return BlockFaceSolidity::None;
        };
        let local = [
            abs_block[0].rem_euclid(chunk_blocks) as u32,
            abs_block[1].rem_euclid(chunk_blocks) as u32,
            abs_block[2].rem_euclid(chunk_blocks) as u32,
        ];
        if chunk.coarse_block(local).is_some() {
            BlockFaceSolidity::AllSolid
        } else if let Some(geometry) = chunk.microblocks.get(&local) {
            BlockFaceSolidity::PerFace(geometry.seam_solidity)
        } else {
            BlockFaceSolidity::None
        }
    };

    // The PER-CELL occupancy of the block at `abs_block`'s face on `(axis, side)` — the
    // 1-voxel layer that abuts a neighbouring block across that face. A coarse-solid block
    // is fully solid (the seam-flag fast path — no densification); an air block fully air; a
    // boundary block expands ITS cuboids' face layer per cell. This is the exact neighbour
    // info the dense apron carried — but only for the SURFACE (boundary) blocks, so coarse
    // interiors are still never densified. The returned bitmap is indexed
    // `cell[in_plane_b * density + in_plane_a]`, with `(in_plane_a, in_plane_b)` = the two
    // axes other than `axis` in ascending order — the SAME order the apron fill walks.
    let face_cells_at = |abs_block: [i64; 3], axis: usize, side: usize| -> NeighbourFace {
        let chunk_blocks = CHUNK_BLOCKS as i64;
        let chunk_coord = [
            abs_block[0].div_euclid(chunk_blocks) as i32,
            abs_block[1].div_euclid(chunk_blocks) as i32,
            abs_block[2].div_euclid(chunk_blocks) as i32,
        ];
        let Some(chunk) = chunk_by_coord.get(&chunk_coord) else {
            return NeighbourFace::Air;
        };
        let local = [
            abs_block[0].rem_euclid(chunk_blocks) as u32,
            abs_block[1].rem_euclid(chunk_blocks) as u32,
            abs_block[2].rem_euclid(chunk_blocks) as u32,
        ];
        if chunk.coarse_block(local).is_some() {
            return NeighbourFace::Solid;
        }
        let Some(geometry) = chunk.microblocks.get(&local) else {
            return NeighbourFace::Air;
        };
        // Expand the boundary block's cuboids' face layer (the plane `coord == 0` for the low
        // face, `coord == density-1` for the high face on `axis`) into a density² bitmap.
        let (axis_a, axis_b) = in_plane_axes(axis);
        let plane = if side == 0 { 0u32 } else { density - 1 };
        let mut cells = vec![false; (density * density) as usize];
        for cuboid in &geometry.cuboids {
            // Does this cuboid touch the requested plane on `axis`?
            if (cuboid.min[axis]..=cuboid.max[axis]).contains(&plane) {
                for a in cuboid.min[axis_a]..=cuboid.max[axis_a] {
                    for b in cuboid.min[axis_b]..=cuboid.max[axis_b] {
                        cells[(b * density + a) as usize] = true;
                    }
                }
            }
        }
        NeighbourFace::Cells(cells)
    };

    // Each chunk meshes INDEPENDENTLY: the per-chunk body below writes only its own local
    // `vertices` / `indices` / `indices_overlay` / `aabb` / `box_count`, reading only shared-
    // IMMUTABLE state (the `chunk_by_coord` map, the `face_solidity_at` / `face_cells_at` /
    // `z_in_band` closures, the `only` filter, the band bounds). So the chunk list is meshed in
    // parallel with rayon. A parallel `.collect()` PRESERVES the source order (issue #57
    // convention), so the output Vec — hence GPU buffer order and the goldens — is byte-identical
    // to the former serial loop.
    let meshes: Vec<CuboidChunkMesh> = chunks
        .par_iter()
        .filter_map(|(chunk_coord, chunk)| {
        // Incremental subset (issue #55): skip chunks not in the rebuild set. Seam culling
        // still consults every chunk (the `chunk_by_coord` lookup above is over the FULL set),
        // so a skipped neighbour's face solidity correctly culls the meshed chunk's seam faces.
        if let Some(only) = only {
            if !only.contains(chunk_coord) {
                return None;
            }
        }
        // The chunk's low voxel corner in the RECENTRED frame (ADR 0008): a chunk-local
        // voxel index `lv` lands at world min-corner `chunk_min - recentre + lv`. Emitting
        // box corners there matches the dense path's `global_index + (min_world - 0.5)`
        // exactly (its cloud-min anchor cancels — see the parity test).
        let chunk_min_recentred = [
            chunk_coord[0] as i64 * chunk_extent_voxels - recentre_voxels[0],
            chunk_coord[1] as i64 * chunk_extent_voxels - recentre_voxels[1],
            chunk_coord[2] as i64 * chunk_extent_voxels - recentre_voxels[2],
        ];
        // Each block's absolute block coord low = chunk_coord * CHUNK_BLOCKS + local block.
        let chunk_block_base = [
            chunk_coord[0] as i64 * CHUNK_BLOCKS as i64,
            chunk_coord[1] as i64 * CHUNK_BLOCKS as i64,
            chunk_coord[2] as i64 * CHUNK_BLOCKS as i64,
        ];

        let mut vertices: Vec<CuboidVertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut indices_overlay: Vec<u32> = Vec::new();
        let mut aabb = Aabb::empty();
        let mut box_count = 0u32;

        for block_z in 0..CHUNK_BLOCKS {
            for block_y in 0..CHUNK_BLOCKS {
                for block_x in 0..CHUNK_BLOCKS {
                    let block = [block_x, block_y, block_z];
                    let abs_block = [
                        chunk_block_base[0] + block_x as i64,
                        chunk_block_base[1] + block_y as i64,
                        chunk_block_base[2] + block_z as i64,
                    ];
                    // The block's low voxel corner in the recentred frame.
                    let block_low_recentred = [
                        chunk_min_recentred[0] + block_x as i64 * block_extent,
                        chunk_min_recentred[1] + block_y as i64 * block_extent,
                        chunk_min_recentred[2] + block_z as i64 * block_extent,
                    ];

                    // ADR 0010 #53: under an ACTIVE band, route every (coarse OR boundary)
                    // block through the band-aware apron mesher — it densifies only the
                    // block (never the whole solid interior), masks out-of-band Z to air on
                    // BOTH interior and apron (so a band-edge cut synthesises a real cap
                    // face), and skips blocks fully outside the band. FULL-band keeps the
                    // E3-proven FAST paths byte-for-byte below.
                    if band_active {
                        let block_lo_z = block_low_recentred[2];
                        let block_hi_z = block_lo_z + block_extent - 1;
                        // Skip blocks the band does not touch at all (every voxel-Z out of band).
                        if block_hi_z < band_lo_recentred || block_lo_z > band_hi_recentred {
                            continue;
                        }
                        box_count += emit_block_banded(
                            density,
                            block_low_recentred,
                            abs_block,
                            &chunk_by_coord,
                            &z_in_band,
                            &mut vertices,
                            &mut indices,
                            &mut indices_overlay,
                            &mut aabb,
                        );
                    } else if let Some(block_id) = chunk.coarse_block(block) {
                        // COARSE-SOLID → ONE box spanning the block (no per-voxel decompose).
                        let overlay = chunk.coarse_block_overlay(block);
                        emit_coarse_block_box(
                            block_id,
                            overlay,
                            density,
                            block_low_recentred,
                            abs_block,
                            &face_solidity_at,
                            &mut vertices,
                            &mut indices,
                            &mut indices_overlay,
                            &mut aabb,
                        );
                        box_count += 1;
                    } else if let Some(geometry) = chunk.microblocks.get(&block) {
                        // BOUNDARY → its stored microblock cuboids, exposure tested against a
                        // block-local apron filled PER CELL from the NEIGHBOUR blocks' face
                        // occupancy (coarse → whole-face solid via the seam flag; boundary →
                        // its own cuboids' face layer) — matching the dense apron exactly.
                        emit_boundary_block_cuboids(
                            geometry,
                            density,
                            block_low_recentred,
                            abs_block,
                            &face_cells_at,
                            &mut vertices,
                            &mut indices,
                            &mut indices_overlay,
                            &mut aabb,
                        );
                        box_count += geometry.cuboids.len() as u32;
                    }
                    // else: air block, nothing to emit.
                }
            }
        }

        if indices.is_empty() && indices_overlay.is_empty() {
            return None;
        }
        Some(CuboidChunkMesh {
            coord: *chunk_coord,
            vertices,
            indices,
            indices_overlay,
            aabb,
            box_count,
        })
        })
        .collect();
    meshes
}

/// Emit a COARSE-SOLID block as ONE box (ADR 0010 Decision 4): the whole `density³` block
/// at `block_id`, culling each of its 6 block faces when the neighbour block's matching
/// face is fully solid (seam-flag culling — no densified apron, no per-voxel decompose of
/// the solid interior). `block_low_recentred` is the block's low voxel corner in the
/// recentred frame; `abs_block` its absolute block coord (to look up neighbours).
#[allow(clippy::too_many_arguments)]
fn emit_coarse_block_box(
    block_id: crate::core_geom::BlockId,
    overlay: bool,
    density: u32,
    block_low_recentred: [i64; 3],
    abs_block: [i64; 3],
    face_solidity_at: &dyn Fn([i64; 3]) -> BlockFaceSolidity,
    vertices: &mut Vec<CuboidVertex>,
    indices: &mut Vec<u32>,
    indices_overlay: &mut Vec<u32>,
    aabb: &mut Aabb,
) {
    let material_id = CellKey::compose(block_id.0, overlay).raw();
    // The box spans the block: world min corner = block_low_recentred, far plane = + density.
    let lo = [
        block_low_recentred[0] as f32,
        block_low_recentred[1] as f32,
        block_low_recentred[2] as f32,
    ];
    let hi = [
        (block_low_recentred[0] + density as i64) as f32,
        (block_low_recentred[1] + density as i64) as f32,
        (block_low_recentred[2] + density as i64) as f32,
    ];
    aabb.expand(glam::Vec3::new(lo[0], lo[1], lo[2]));
    aabb.expand(glam::Vec3::new(hi[0], hi[1], hi[2]));

    let sink = if overlay { indices_overlay } else { indices };
    let clean_material = CellKey::from_raw(material_id).block_id() as u32;
    for face in &FACE_TEMPLATES {
        // The face's axis + side, and the neighbour block across it.
        let (axis, side) = face_axis_side(face.neighbor_delta);
        let neighbour = [
            abs_block[0] + face.neighbor_delta[0] as i64,
            abs_block[1] + face.neighbor_delta[1] as i64,
            abs_block[2] + face.neighbor_delta[2] as i64,
        ];
        // The neighbour's MATCHING face is on the same axis, OPPOSITE side. If it is fully
        // solid, every cell behind this face is backed ⇒ cull. Otherwise emit the whole
        // block face (the merged-box over-draw rule — any partly-exposed face is emitted,
        // and a fully-occluded over-draw is back-face-culled / depth-buried).
        let neighbour_face_solid = face_solidity_at(neighbour).face_is_solid(axis, 1 - side);
        if neighbour_face_solid {
            continue;
        }
        let base = vertices.len() as u32;
        for corner in &face.corners {
            let world = [
                if corner[0] == 0 { lo[0] } else { hi[0] },
                if corner[1] == 0 { lo[1] } else { hi[1] },
                if corner[2] == 0 { lo[2] } else { hi[2] },
            ];
            vertices.push(CuboidVertex {
                position: world,
                normal: face.normal,
                material_id: clean_material,
            });
        }
        sink.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// Emit a BOUNDARY block's stored microblock cuboids (ADR 0010 Decision 4), exposure tested
/// against a `(density+2)³` apron region whose interior is the block's own voxels (re-expanded
/// from the cuboids) and whose 1-voxel border is filled PER CELL from each NEIGHBOUR block's
/// face occupancy (coarse → whole-face solid via the seam flag, NO densification of the coarse
/// interior; boundary → its own cuboids' face layer; air → empty). This reproduces the dense
/// apron EXACTLY at the block seam, so it reuses [`emit_box_faces`] / [`face_is_exposed`]
/// unchanged and culls every boundary face the dense mesher culls — no over-draw at a partial
/// boundary-to-boundary seam (which would otherwise render as a spurious surface).
#[allow(clippy::too_many_arguments)]
fn emit_boundary_block_cuboids(
    geometry: &MicroblockGeometry,
    density: u32,
    block_low_recentred: [i64; 3],
    abs_block: [i64; 3],
    face_cells_at: &dyn Fn([i64; 3], usize, usize) -> NeighbourFace,
    vertices: &mut Vec<CuboidVertex>,
    indices: &mut Vec<u32>,
    indices_overlay: &mut Vec<u32>,
    aabb: &mut Aabb,
) {
    // Apron frame: a (density+2)³ region with the block's voxels at local index +1, so the
    // 1-voxel border is the apron. `face_is_exposed` then tests the neighbour cell exactly.
    let apron_extent = [density + 2, density + 2, density + 2];
    let mut apron = VoxelRegion::new_empty(apron_extent);

    // Interior: the block's own voxels (the cuboids' render keys), shifted +1.
    for cuboid in &geometry.cuboids {
        for vz in cuboid.min[2]..=cuboid.max[2] {
            for vy in cuboid.min[1]..=cuboid.max[1] {
                for vx in cuboid.min[0]..=cuboid.max[0] {
                    apron.set(vx + 1, vy + 1, vz + 1, Some(cuboid.material_id()));
                }
            }
        }
    }

    // Apron border: each of the 6 outer planes is filled PER CELL from the neighbour block's
    // matching (opposite-side) face. A constant non-zero key marks "solid" (the apron is only
    // read for occupancy by `face_is_exposed`).
    const APRON_SOLID: u16 = 1;
    let d = density;
    for (axis, side, delta) in [
        (0usize, 0usize, [-1i64, 0, 0]),
        (0, 1, [1, 0, 0]),
        (1, 0, [0, -1, 0]),
        (1, 1, [0, 1, 0]),
        (2, 0, [0, 0, -1]),
        (2, 1, [0, 0, 1]),
    ] {
        let neighbour = [
            abs_block[0] + delta[0],
            abs_block[1] + delta[1],
            abs_block[2] + delta[2],
        ];
        // The neighbour's MATCHING face is on the same axis, OPPOSITE side.
        let neighbour_face = face_cells_at(neighbour, axis, 1 - side);
        if matches!(neighbour_face, NeighbourFace::Air) {
            continue; // fully air ⇒ nothing to cull against on this plane
        }
        let plane = if side == 0 { 0u32 } else { d + 1 };
        let (axis_a, axis_b) = in_plane_axes(axis);
        for ai in 0..d {
            for bi in 0..d {
                let solid = match &neighbour_face {
                    NeighbourFace::Solid => true,
                    NeighbourFace::Cells(cells) => cells[(bi * d + ai) as usize],
                    NeighbourFace::Air => false,
                };
                if !solid {
                    continue;
                }
                // Apron-local cell: the block's in-plane index `ai/bi` sits at apron +1; the
                // out-of-plane coord is the border `plane`.
                let mut coord = [0u32; 3];
                coord[axis] = plane;
                coord[axis_a] = ai + 1;
                coord[axis_b] = bi + 1;
                apron.set(coord[0], coord[1], coord[2], Some(APRON_SOLID));
            }
        }
    }

    // Region offset maps apron-local index 0 to the recentred frame: the block's low voxel
    // is apron-local +1, so apron-local 0 sits at `block_low_recentred - 1`.
    let region_offset = [
        (block_low_recentred[0] - 1) as f32,
        (block_low_recentred[1] - 1) as f32,
        (block_low_recentred[2] - 1) as f32,
    ];
    for cuboid in &geometry.cuboids {
        // The cuboid in apron-local frame (+1 shift).
        let shifted = VoxelBox {
            min: [cuboid.min[0] + 1, cuboid.min[1] + 1, cuboid.min[2] + 1],
            max: [cuboid.max[0] + 1, cuboid.max[1] + 1, cuboid.max[2] + 1],
            label: cuboid.material_id(),
        };
        let sink = if box_has_overlay(&shifted) {
            &mut *indices_overlay
        } else {
            &mut *indices
        };
        emit_box_faces(&shifted, &apron, region_offset, vertices, sink, aabb);
    }
}

/// Stamp the block at chunk-local-or-neighbour block index `abs_block`'s per-voxel occupancy
/// into `region` at the apron-local offset `dst_lo` (so a neighbour block lands at the apron
/// border), CLIPPED to the band via `z_in_band` (ADR 0010 #53). A coarse-solid block fills
/// every `density³` cell at its render key; a boundary block stamps each cuboid; an air /
/// missing block stamps nothing. `block_low_recentred_z` is the block's low voxel-Z in the
/// recentred frame, so a block-local voxel-Z `vz` maps to recentred Z
/// `block_low_recentred_z + vz` for the band test — masking out-of-band voxels to air on BOTH
/// the meshed interior and the neighbour apron, exactly as the dense banded path masks apron.
///
/// Writes only cells whose apron-local index lands inside `region.extent` (a neighbour block
/// contributes only its 1-voxel abutting border layer). Returns nothing; the caller sizes the
/// apron and supplies `dst_lo`.
#[allow(clippy::too_many_arguments)]
fn stamp_block_into_region_banded(
    chunk_by_coord: &std::collections::HashMap<[i32; 3], &TwoLayerChunk>,
    abs_block: [i64; 3],
    density: u32,
    block_low_recentred_z: i64,
    dst_lo: [i64; 3],
    z_in_band: &dyn Fn(i64) -> bool,
    region: &mut VoxelRegion,
) {
    let chunk_blocks = CHUNK_BLOCKS as i64;
    let chunk_coord = [
        abs_block[0].div_euclid(chunk_blocks) as i32,
        abs_block[1].div_euclid(chunk_blocks) as i32,
        abs_block[2].div_euclid(chunk_blocks) as i32,
    ];
    let Some(chunk) = chunk_by_coord.get(&chunk_coord) else {
        return; // no covering chunk → air
    };
    let local = [
        abs_block[0].rem_euclid(chunk_blocks) as u32,
        abs_block[1].rem_euclid(chunk_blocks) as u32,
        abs_block[2].rem_euclid(chunk_blocks) as u32,
    ];
    let [ex, ey, ez] = region.extent;

    // Stamp one block-local voxel `(vx, vy, vz)` of render key `key` into the region, band-
    // masked on Z and bounds-checked against the apron extent.
    let stamp = |vx: u32, vy: u32, vz: u32, key: u16, region: &mut VoxelRegion| {
        if !z_in_band(block_low_recentred_z + vz as i64) {
            return;
        }
        let lx = dst_lo[0] + vx as i64;
        let ly = dst_lo[1] + vy as i64;
        let lz = dst_lo[2] + vz as i64;
        if lx < 0 || ly < 0 || lz < 0 || lx >= ex as i64 || ly >= ey as i64 || lz >= ez as i64 {
            return;
        }
        region.set(lx as u32, ly as u32, lz as u32, Some(key));
    };

    if let Some(block_id) = chunk.coarse_block(local) {
        let key = CellKey::compose(block_id.0, chunk.coarse_block_overlay(local)).raw();
        for vz in 0..density {
            for vy in 0..density {
                for vx in 0..density {
                    stamp(vx, vy, vz, key, region);
                }
            }
        }
    } else if let Some(geometry) = chunk.microblocks.get(&local) {
        for cuboid in &geometry.cuboids {
            for vz in cuboid.min[2]..=cuboid.max[2] {
                for vy in cuboid.min[1]..=cuboid.max[1] {
                    for vx in cuboid.min[0]..=cuboid.max[0] {
                        stamp(vx, vy, vz, cuboid.material_id(), region);
                    }
                }
            }
        }
    }
    // else: air block, nothing to stamp.
}

/// Mesh ONE block (coarse OR boundary) under an ACTIVE layer band (ADR 0010 #53). Builds a
/// `(density+2)³` apron region whose INTERIOR is the block's own band-clipped voxels and whose
/// 1-voxel border is each neighbour block's abutting band-clipped face — then decomposes the
/// interior and emits via [`emit_box_faces`]/[`face_is_exposed`], so a band-edge cut (the
/// out-of-band neighbour cell reads as AIR) synthesises a real cap face, and a non-cut seam
/// against a solid neighbour is still culled. This is the dense banded apron restricted to one
/// block: it densifies ONLY the band-cut block (never the whole solid interior). Returns the
/// number of boxes the interior decomposed into (the diagnostic box count).
#[allow(clippy::too_many_arguments)]
fn emit_block_banded(
    density: u32,
    block_low_recentred: [i64; 3],
    abs_block: [i64; 3],
    chunk_by_coord: &std::collections::HashMap<[i32; 3], &TwoLayerChunk>,
    z_in_band: &dyn Fn(i64) -> bool,
    vertices: &mut Vec<CuboidVertex>,
    indices: &mut Vec<u32>,
    indices_overlay: &mut Vec<u32>,
    aabb: &mut Aabb,
) -> u32 {
    // Apron frame: a (density+2)³ region with the block's voxels at local index +1, so the
    // 1-voxel border is the apron (identical to `emit_boundary_block_cuboids`).
    let apron_extent = [density + 2, density + 2, density + 2];
    let mut interior = VoxelRegion::new_empty(apron_extent);

    // Interior = THIS block's own voxels at local +1, band-clipped on Z.
    stamp_block_into_region_banded(
        chunk_by_coord,
        abs_block,
        density,
        block_low_recentred[2],
        [1, 1, 1],
        z_in_band,
        &mut interior,
    );

    // The interior decomposition + the apron border share one region: decompose reads only the
    // interior (+1 shift keeps the border air for the decompose), and `face_is_exposed` reads
    // the SAME region's border. We therefore clone the interior-only region for decomposition
    // BEFORE filling the apron border (so a box never grows into the border), then fill the
    // border into the exposure region.
    let decompose_region = interior.clone();
    let mut apron = interior; // reuse as the exposure region; add the neighbour border below.

    // Apron border: each of the 6 neighbour blocks' abutting face, band-clipped. A neighbour
    // block landed at the apron border via `dst_lo` = its block offset relative to this block
    // (−1 block on the low side, +1 on the high side, scaled to the apron's +1 interior
    // origin). Only the single border layer of each neighbour falls inside the apron extent.
    for (delta, dst_lo) in [
        ([-1i64, 0, 0], [1 - density as i64, 1, 1]),
        ([1, 0, 0], [1 + density as i64, 1, 1]),
        ([0, -1, 0], [1, 1 - density as i64, 1]),
        ([0, 1, 0], [1, 1 + density as i64, 1]),
        ([0, 0, -1], [1, 1, 1 - density as i64]),
        ([0, 0, 1], [1, 1, 1 + density as i64]),
    ] {
        let neighbour = [
            abs_block[0] + delta[0],
            abs_block[1] + delta[1],
            abs_block[2] + delta[2],
        ];
        let neighbour_low_z = block_low_recentred[2] + delta[2] * density as i64;
        stamp_block_into_region_banded(
            chunk_by_coord,
            neighbour,
            density,
            neighbour_low_z,
            dst_lo,
            z_in_band,
            &mut apron,
        );
    }

    // Region offset maps apron-local index 0 to the recentred frame: the block's low voxel is
    // apron-local +1, so apron-local 0 sits at `block_low_recentred - 1`.
    let region_offset = [
        (block_low_recentred[0] - 1) as f32,
        (block_low_recentred[1] - 1) as f32,
        (block_low_recentred[2] - 1) as f32,
    ];
    let boxes = decompose_into_boxes(&decompose_region);
    for voxel_box in &boxes {
        let sink = if box_has_overlay(voxel_box) {
            &mut *indices_overlay
        } else {
            &mut *indices
        };
        emit_box_faces(voxel_box, &apron, region_offset, vertices, sink, aabb);
    }
    boxes.len() as u32
}

/// The `(axis, side)` a face-template's `neighbor_delta` points along: axis 0/1/2 = X/Y/Z,
/// side 0 = low (delta −1), side 1 = high (delta +1).
#[inline]
fn face_axis_side(delta: [i32; 3]) -> (usize, usize) {
    for (axis, &d) in delta.iter().enumerate() {
        if d > 0 {
            return (axis, 1);
        }
        if d < 0 {
            return (axis, 0);
        }
    }
    (0, 0)
}

/// Emit the exposed faces of one box into the shared vertex/index buffers,
/// expanding `aabb` to contain the box. A face is exposed when the voxel cell
/// immediately beyond it (per axis, across the box's full extent on the other two
/// axes) is air — at minimum this culls box-internal faces; here it also culls
/// faces fully covered by adjacent solid voxels.
fn emit_box_faces(
    voxel_box: &VoxelBox,
    region: &VoxelRegion,
    world_offset: [f32; 3],
    vertices: &mut Vec<CuboidVertex>,
    indices: &mut Vec<u32>,
    aabb: &mut Aabb,
) {
    let [min_x, min_y, min_z] = voxel_box.min;
    let [max_x, max_y, max_z] = voxel_box.max;
    // Inclusive box → the far plane is at max + 1.
    let lo = [min_x as f32, min_y as f32, min_z as f32];
    let hi = [
        (max_x + 1) as f32,
        (max_y + 1) as f32,
        (max_z + 1) as f32,
    ];

    // Expand the chunk AABB to this box's world extent (local index + offset).
    aabb.expand(glam::Vec3::new(lo[0] + world_offset[0], lo[1] + world_offset[1], lo[2] + world_offset[2]));
    aabb.expand(glam::Vec3::new(hi[0] + world_offset[0], hi[1] + world_offset[1], hi[2] + world_offset[2]));

    // The clean colour index (ADR 0003 §3c): the box's on-face-grid flag is NOT a vertex
    // attribute — the caller routed this box to the overlay-on or overlay-off index run by
    // its key bit, and the draw sets the per-draw overlay-active uniform per run. So strip
    // the overlay bit here and write only the categorical id into the vertex.
    let material_id = CellKey::from_raw(voxel_box.material_id()).block_id() as u32;

    for face in &FACE_TEMPLATES {
        if !face_is_exposed(voxel_box, region, face.neighbor_delta) {
            continue;
        }
        let base = vertices.len() as u32;
        for corner in &face.corners {
            // 0 → min plane (lo), 1 → max+1 plane (hi); shift into world space.
            let world = [
                (if corner[0] == 0 { lo[0] } else { hi[0] }) + world_offset[0],
                (if corner[1] == 0 { lo[1] } else { hi[1] }) + world_offset[1],
                (if corner[2] == 0 { lo[2] } else { hi[2] }) + world_offset[2],
            ];
            vertices.push(CuboidVertex {
                position: world,
                normal: face.normal,
                material_id,
            });
        }
        // Two CCW triangles per quad (matching the instanced winding scheme).
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// Whether a decomposed box carries the on-face-grid overlay marker in its region-cell
/// key (ADR 0003 §3c). Routes the box to the overlay-on index run.
#[inline]
fn box_has_overlay(voxel_box: &VoxelBox) -> bool {
    CellKey::from_raw(voxel_box.material_id()).has_overlay()
}

/// Is the given face of the box exposed against the dense apron `region`? Thin domain
/// adapter over the substrate [`CulledBoxMeshing`] culling kernel (slice S10): it supplies
/// the neighbour-solidity oracle by reading this mesher's [`VoxelRegion`] occupancy — a cell
/// is solid iff it is in bounds and carries a render key. Negative or out-of-extent cells
/// answer air (exposed), reproducing the dense apron's border-is-air convention exactly.
///
/// The kernel keeps ONE quad per box face (not per voxel): if a merged face is partially
/// exposed, the whole quad is emitted (over-draw of at most the box's own face, never a
/// hole). See [`CulledBoxMeshing::face_is_exposed`] and `docs/architecture/03-display.md`.
fn face_is_exposed(voxel_box: &VoxelBox, region: &VoxelRegion, delta: [i32; 3]) -> bool {
    CulledBoxMeshing::face_is_exposed(voxel_box, delta, |[nx, ny, nz]| {
        nx >= 0
            && ny >= 0
            && nz >= 0
            && region.cell_at(nx as u32, ny as u32, nz as u32).is_some()
    })
}

/// std140-safe uniform block for the cuboid pass (ADR 0002 E3b-2). Carries the
/// camera matrix, the grid half-extent and density (driving the per-voxel texture
/// slice and the position-based grid overlay), the grid-overlay parameters, and
/// the per-material base colours (reused from the instanced step-3b modulation).
/// Every `vec3` is followed by a scalar so it never straddles a 16-byte boundary;
/// the four grid-line scalars then fill the slot before the `vec4` array (which
/// must be 16-aligned). Field order matches the WGSL `CuboidUniforms` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CuboidUniforms {
    view_projection: [[f32; 4]; 4],
    grid_half_extent: [f32; 3],
    voxels_per_block: f32,
    voxel_line_color: [f32; 3],
    grid_overlay_enabled: f32,
    block_line_color: [f32; 3],
    material_modulation_enabled: f32,
    voxel_line_half_width: f32,
    block_line_half_width: f32,
    voxel_line_alpha: f32,
    block_line_alpha: f32,
    // Layer-range band clip (issue #12 parity) + debug-faces flag. The two band
    // bounds plus the debug flag plus a pad fill one 16-byte slot, so the colour
    // array below stays 16-aligned (matching the WGSL `CuboidUniforms`).
    band_min: f32,
    band_max: f32,
    debug_face_mode: f32,
    /// ADR 0012 (H1): the onion GHOST flag (0 = normal solid render, 1 = flat
    /// translucent ghost tint). Occupies the former `_band_pad` slot; `0.0` for the
    /// solid draw keeps the solid uniform bytes identical (non-onion goldens byte-green).
    ghost_mode: f32,
    material_base_colors: [[f32; 4]; MaterialChoice::MATERIAL_COUNT],
    /// Per-material atlas sub-rect (ADR 0002 E3c-1 / O8), indexed by `material_id`:
    /// `[inset_min_u, inset_min_v, inset_size_u, inset_size_v]`. The shader maps the
    /// per-voxel slice's `fract`-tiled UV into this window of the single atlas, so a
    /// chunk of mixed materials is ONE mesh = ONE draw (no per-material texture
    /// bind). Each `vec4` is naturally 16-aligned.
    material_atlas_rects: [[f32; 4]; MaterialChoice::MATERIAL_COUNT],
    /// ADR 0012 (H1): the onion ghost tint (linear RGB + src alpha), read only when
    /// `ghost_mode > 0.5`. Appended so the solid draw's uniform layout is unchanged.
    ghost_tint: [f32; 4],
}

/// Convert a packed [`MaterialAtlas`]'s per-material sub-rects into the uniform
/// array layout `[inset_min_u, inset_min_v, inset_size_u, inset_size_v]` the shader
/// indexes by `material_id`. Materials without a packed sub-rect (should not happen
/// for the procedural set) fall back to the WHOLE atlas (`[0,0,1,1]`), so a missing
/// id degrades to "sample the atlas" rather than panicking.
pub(crate) fn atlas_rects_from(atlas: &MaterialAtlas) -> [[f32; 4]; MaterialChoice::MATERIAL_COUNT] {
    let mut rects = [[0.0, 0.0, 1.0, 1.0]; MaterialChoice::MATERIAL_COUNT];
    for (slot, sub_rect) in rects.iter_mut().zip(atlas.sub_rects.iter()) {
        let [size_u, size_v] = sub_rect.inset_size();
        *slot = [sub_rect.inset_min_u, sub_rect.inset_min_v, size_u, size_v];
    }
    rects
}

/// The per-draw on-face-grid overlay-active bind-group layout (group 2, ADR 0003 §3c / ADR
/// 0010 E3): one `u32` uniform read with a DYNAMIC OFFSET, so the overlay-off and overlay-on
/// draws of a chunk select `0` / `1` from a two-entry buffer without a per-vertex flag.
fn overlay_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cuboid overlay-active bind group layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<u32>() as u64),
            },
            count: None,
        }],
    })
}

/// Build the two-entry per-draw overlay-active uniform buffer + its dynamic-offset bind
/// group (ADR 0003 §3c). Entry 0 = `0` (overlay off), entry 1 (at the device's
/// `min_uniform_buffer_offset_alignment`) = `1` (overlay on). Returns the bind group and
/// the stride to pass as the dynamic offset for the overlay-on draw.
fn build_overlay_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> (wgpu::BindGroup, u32) {
    let stride = device
        .limits()
        .min_uniform_buffer_offset_alignment
        .max(std::mem::size_of::<u32>() as u32);
    // Two `u32` entries, each at a `stride`-aligned offset (the rest is padding).
    let mut bytes = vec![0u8; (stride as usize) + std::mem::size_of::<u32>()];
    bytes[0..4].copy_from_slice(&0u32.to_ne_bytes()); // entry 0: overlay OFF
    bytes[stride as usize..stride as usize + 4].copy_from_slice(&1u32.to_ne_bytes()); // entry 1: overlay ON
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cuboid overlay-active uniform"),
        contents: &bytes,
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cuboid overlay-active bind group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: std::num::NonZeroU64::new(std::mem::size_of::<u32>() as u64),
            }),
        }],
    });
    (bind_group, stride)
}

/// The cuboid atlas bind-group layout: a single 2D texture (binding 0) + sampler
/// (binding 1). One atlas for ALL materials replaces the former per-material
/// D2Array binds (ADR 0002 O8).
pub(crate) fn build_atlas_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cuboid atlas bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

/// Upload a packed [`MaterialAtlas`] image as a single RGBA8 sRGB 2D texture
/// (Nearest, no mipmaps), matching the instanced path's sRGB decode so lighting +
/// overlay run in linear space and the sRGB target re-encodes on write.
pub(crate) fn upload_atlas_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    atlas: &MaterialAtlas,
) -> wgpu::Texture {
    let size = wgpu::Extent3d {
        width: atlas.width.max(1),
        height: atlas.height.max(1),
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cuboid material atlas"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &atlas.pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * atlas.width.max(1)),
            rows_per_image: Some(atlas.height.max(1)),
        },
        size,
    );
    texture
}

/// One render chunk's GPU buffers for the cuboid path (issue #20 S6c-2d): its own
/// vertex + index buffer, the index count, and the world AABB for frustum culling.
/// Mirrors the instanced [`crate::renderer::InstancedChunkBuffers`]. A chunk that
/// meshes to zero faces is never stored (no buffer allocated).
struct CuboidChunkBuffers {
    vertex_buffer: wgpu::Buffer,
    /// One index buffer holding the overlay-OFF run followed by the overlay-ON run (ADR
    /// 0003 §3c). `index_count` is the overlay-off run length (drawn with the per-draw
    /// overlay-active uniform = 0); `index_count_overlay` is the overlay-on run, drawn at
    /// byte offset `index_count * 4` with the uniform = 1. Splitting by overlay state into
    /// two draws keeps the render flag out of the vertex format while preserving the
    /// per-object overlay behaviour.
    index_buffer: wgpu::Buffer,
    index_count: u32,
    index_count_overlay: u32,
    aabb: Aabb,
    /// Boxes this chunk decomposed into (diagnostic). Retained per chunk so the
    /// renderer's `total_box_count` can be recomputed exactly after an INCREMENTAL
    /// rebuild touches only a subset of chunks (an incremental update can't sum from
    /// the freshly-built meshes alone — the untouched chunks' buffers carry it).
    box_count: u32,
}

/// All GPU resources for drawing the cuboid mesh (DEFAULT render path; per-chunk
/// buffers since issue #20 S6c-2d).
pub struct CuboidMeshRenderer {
    pipeline: wgpu::RenderPipeline,
    /// Face-orientation debug pipeline: identical to `pipeline` except
    /// `cull_mode: None`, so a back face that is the nearest surface (a winding
    /// bug) still draws and is flagged by the shader's `front_facing` marker.
    /// Selected in `draw` when `debug_face_mode` is on — mirroring the instanced
    /// path's cull-off debug pipeline.
    debug_pipeline: wgpu::RenderPipeline,
    /// Loaded-VS-block pipelines (part of #20): same vertex layout + uniform group,
    /// but group(1) is a 6-layer D2Array (the block's per-face textures) instead of
    /// the procedural atlas, and the shader (`cuboid_loaded.wgsl`) selects the face
    /// layer FROM THE FACE NORMAL — exactly like the instanced loaded path. Selected
    /// in `draw` when a loaded material's bind group is supplied (else the procedural
    /// atlas pipelines above run, unchanged). The debug variant is cull-off.
    loaded_pipeline: wgpu::RenderPipeline,
    loaded_debug_pipeline: wgpu::RenderPipeline,
    /// Whether the last `update_uniforms` requested debug-faces mode (selects the
    /// cull-off pipeline in `draw`, matching the uploaded `debug_face_mode` flag).
    debug_face_mode: bool,
    /// Per-chunk GPU buffers (issue #20 S6c-2d), keyed by absolute chunk coord (the
    /// coord `resident_render_chunks` reports). Replaces the single monolithic
    /// vertex/index buffer + `CuboidMesh.chunks` index ranges: each chunk owns its
    /// own buffers, meshed from its own per-chunk grid + a 1-voxel neighbour apron.
    chunk_buffers: std::collections::HashMap<[i32; 3], CuboidChunkBuffers>,
    /// Chunk coords (keys into `chunk_buffers`) that survived the last frustum cull;
    /// computed in `update_uniforms`, consumed in `draw`. Sorted for a deterministic
    /// draw order (cross-chunk order is pixel-irrelevant: opaque + depth-tested).
    visible_chunks: Vec<[i32; 3]>,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    /// Per-draw on-face-grid overlay-active bind group (group 2, ADR 0003 §3c / ADR 0010
    /// E3): a single tiny `u32` uniform read with a DYNAMIC OFFSET. The backing buffer
    /// holds the value `0` at offset 0 and `1` at offset `overlay_dynamic_stride`, so the
    /// overlay-off draw binds offset 0 and the overlay-on draw binds the stride — the
    /// per-draw uniform that replaced the per-vertex overlay flag (one bool per draw, §3c).
    overlay_bind_group: wgpu::BindGroup,
    /// The dynamic-offset stride between the two overlay-active uniform entries (the
    /// device's `min_uniform_buffer_offset_alignment`, rounded up from the `u32` value).
    overlay_dynamic_stride: u32,
    /// ONE atlas bind group (ADR 0002 E3c-1 / O8): all material textures packed
    /// into a single 2D atlas texture + sampler. Replaces the former per-material
    /// D2Array binds — a chunk of mixed materials is now one mesh = one draw, with
    /// the shader mapping each face's `material_id` to its atlas sub-rect (carried
    /// in the uniforms). Clamp-to-edge sampler: the shader tiles the per-voxel slice
    /// itself via `fract` mapped into the sub-rect (a Repeat sampler would wrap into
    /// a neighbouring material's cell).
    atlas_bind_group: wgpu::BindGroup,
    /// The packed atlas's per-material sub-rects (inset sampling window), uploaded
    /// in the per-frame uniforms so the shader maps `material_id` → atlas window.
    atlas_rects: [[f32; 4]; MaterialChoice::MATERIAL_COUNT],
    /// Which procedural material the per-frame modulation was bound to.
    /// `update_uniforms` records it (drives the per-box base-colour modulation only;
    /// the atlas is bound once regardless of material).
    bound_material: MaterialChoice,
    /// The per-chunk grids the mesh was last built from (OWNED copies), retained so
    /// the mesh can be re-built CLIPPED to a new layer-range band (issue #12 parity)
    /// without the caller re-supplying them. The cuboid band clip masks each chunk's
    /// region before decomposition (real cap faces), so a band change re-meshes; we
    /// cache the last band and rebuild only when it differs.
    source_chunk_grids: Vec<([i32; 3], VoxelGrid)>,
    /// The two-layer chunks the mesh was last built from (ADR 0010 #53), retained so a band
    /// reclip (the layer scrubber) can re-mesh DIRECTLY from the two-layer store — no dense
    /// source grids. Empty on the dense path; populated only by [`new_from_two_layer_chunks`].
    /// `recentre`/`density` are the frame + density the two-layer mesher needs to re-emit in
    /// the SAME world frame on every band change.
    source_two_layer_chunks: Vec<([i32; 3], Arc<crate::two_layer_store::TwoLayerChunk>)>,
    source_two_layer_recentre: RecentreVoxels,
    source_two_layer_density: u32,
    /// The whole composite grid's voxel dims (the band clip maps an absolute layer to
    /// the global region-local Z; only the Z half is used).
    source_grid_dimensions: [u32; 3],
    /// Total boxes across all chunks the last build produced (diagnostic).
    total_box_count: u32,
    current_band: LayerBand,
    /// The loaded-VS-block material bind-group layout (a 6-layer D2Array + sampler,
    /// from [`crate::renderer::build_face_material_layout`]). Retained so a
    /// runtime-loaded block (M6/M7) can build a bind group of the SAME shape via
    /// [`Self::material_bind_group_layout`] and be drawn by the loaded pipeline.
    loaded_material_layout: wgpu::BindGroupLayout,
    /// The shared material sampler (nearest, clamp-to-edge) reused by loaded
    /// materials so they slice/filter exactly like the procedural atlas. Exposed via
    /// [`Self::material_sampler`].
    loaded_material_sampler: wgpu::Sampler,
    // --- ADR 0012 (H1): the onion GHOST pass ---
    /// The ghost pipeline: the SAME procedural `cuboid.wgsl` vertex/fragment (its
    /// `ghost_mode` branch flat-tints), but alpha-blended over the solid with the depth
    /// test ON (`Less`) and depth WRITE OFF, so solid geometry occludes the ghost and
    /// the ghost occludes nothing. Used for BOTH procedural and loaded-material scenes
    /// (the ghost never textures — flat tint even over `cuboid_loaded`).
    ghost_pipeline: wgpu::RenderPipeline,
    /// The ghost draw's uniform buffer (`ghost_mode = 1` + tint), separate from the
    /// solid `uniform_buffer` so the same frame carries both states.
    ghost_uniform_buffer: wgpu::Buffer,
    ghost_uniform_bind_group: wgpu::BindGroup,
    /// The GHOST geometry: two thin per-slab meshes clipped to the onion slabs below /
    /// above the band (`[band_min − depth, band_min)` and `(band_max, band_max + depth]`,
    /// ADR 0012). Built via the SAME banded mesher the solid uses (so the two paths — and
    /// the dense vs two-layer builds — ghost identically), just at the slab bands. Empty
    /// when onion is off. Kept as two maps because a tall chunk can straddle both slabs.
    ghost_lower_buffers: std::collections::HashMap<[i32; 3], CuboidChunkBuffers>,
    ghost_upper_buffers: std::collections::HashMap<[i32; 3], CuboidChunkBuffers>,
    /// The band the ghost slabs were last built for (`None` = never built / cleared), so
    /// a same-band frame skips the slab re-mesh and a band change (or the first frame
    /// after an async swap that built only the solid) rebuilds them.
    ghost_built_band: Option<LayerBand>,
}

impl CuboidMeshRenderer {
    /// Build the cuboid renderer from a WHOLE grid (the wrapper kept for `shot.rs`
    /// and tests that have a monolithic grid). Buckets the grid into per-chunk
    /// sub-grids by `floor(world_position / chunk_extent)` — the SAME key the
    /// instanced `crate::renderer::VoxelRenderer::rebuild_instances` wrapper uses —
    /// then meshes per chunk with an apron via [`Self::new_from_chunks`]. So a build
    /// from the whole grid is byte-identical to a build from the resolve cache's
    /// per-chunk accessor.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        grid: &VoxelGrid,
        voxels_per_block: u32,
    ) -> Self {
        let buckets = bucket_grid_into_chunk_grids(grid, voxels_per_block);
        let chunk_refs: Vec<([i32; 3], &VoxelGrid)> =
            buckets.iter().map(|(coord, g)| (*coord, g)).collect();
        Self::new_from_chunks(
            device,
            queue,
            color_format,
            &chunk_refs,
            grid.dimensions,
        )
    }

    /// Build the cuboid renderer DIRECTLY from the resolve cache's per-chunk grids
    /// (issue #20 S6c-2d). `chunk_grids` is `resident_render_chunks`'s output
    /// (`(absolute_chunk_coord, &rebased_grid)` per covering chunk); `grid_dimensions`
    /// is the whole composite grid's voxel dims (the band-clip layer mapping). Meshes
    /// every chunk with a 1-voxel neighbour apron (see [`build_chunk_meshes_with_apron`])
    /// and stores one [`CuboidChunkBuffers`] per non-empty chunk.
    pub fn new_from_chunks(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        chunk_grids: &[([i32; 3], &VoxelGrid)],
        grid_dimensions: [u32; 3],
    ) -> Self {
        profiling::scope!("cuboid_mesh_build");
        let source_chunk_grids: Vec<([i32; 3], VoxelGrid)> = chunk_grids
            .iter()
            .map(|(coord, grid)| (*coord, (*grid).clone()))
            .collect();
        let chunk_meshes =
            build_chunk_meshes_with_apron(chunk_grids, grid_dimensions, LayerBand::FULL);
        Self::assemble(
            device,
            queue,
            color_format,
            chunk_meshes,
            source_chunk_grids,
            grid_dimensions,
        )
    }

    /// Build the cuboid renderer from a [`TwoLayerChunk`] per covering chunk (ADR 0010 E3):
    /// a coarse-solid block becomes a ONE-BOX fast path, a boundary block its stored
    /// microblock cuboids, and inter-block / inter-chunk seam faces are culled via the
    /// per-face seam-solidity flags (plus the neighbour coarse layer) — NOT a densified
    /// apron. The emitted exposed-face set is proven identical to the dense
    /// `new_from_chunks` path (the E3 parity gate), so it renders pixel-identical.
    ///
    /// `chunks` is `(absolute_chunk_coord, TwoLayerChunk)` per covering chunk;
    /// `grid_dimensions` is the whole composite voxel dims; `recentre_voxels` is the
    /// resolve's carried recentre (ADR 0008) so the two-layer mesh lands in the SAME world
    /// frame the dense path assembles. The INITIAL build is FULL-band (the E3 fast paths);
    /// the two-layer chunks are RETAINED so a later band reclip (the layer scrubber, ADR
    /// 0010 #53) re-meshes DIRECTLY from the store — no dense source grids needed.
    pub fn new_from_two_layer_chunks(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        chunks: &[([i32; 3], Arc<crate::two_layer_store::TwoLayerChunk>)],
        grid_dimensions: [u32; 3],
        recentre_voxels: RecentreVoxels,
        voxels_per_block: u32,
    ) -> Self {
        // The synchronous path builds the FULL model (no band clip). Delegates to the banded
        // builder with `LayerBand::FULL` so its output is byte-identical to before (goldens
        // + gpu_parity stay pixel-exact).
        Self::new_from_two_layer_chunks_banded(
            device,
            queue,
            color_format,
            chunks,
            grid_dimensions,
            recentre_voxels,
            voxels_per_block,
            LayerBand::FULL,
        )
    }

    /// As `new_from_two_layer_chunks`, but builds the mesh already CLIPPED to `band`
    /// (issue #60 M2). The async worker uses this so the swapped-in renderer already matches
    /// the active `effective_band` — the swap frame then does NOT trigger a full synchronous
    /// `rebuild_for_band` re-mesh on the main thread (the multi-second hitch #60 removed,
    /// which would fire on EVERY async swap during onion-skin scrubbing). Sets `current_band`
    /// so the per-frame `update_uniforms` treats the band as already applied. `LayerBand::FULL`
    /// is identical to the plain builder.
    #[allow(clippy::too_many_arguments)]
    pub fn new_from_two_layer_chunks_banded(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        chunks: &[([i32; 3], Arc<crate::two_layer_store::TwoLayerChunk>)],
        grid_dimensions: [u32; 3],
        recentre_voxels: RecentreVoxels,
        voxels_per_block: u32,
        band: LayerBand,
    ) -> Self {
        profiling::scope!("cuboid_mesh_build_two_layer");
        let chunk_meshes = build_two_layer_chunk_meshes(
            chunks,
            grid_dimensions,
            recentre_voxels,
            voxels_per_block,
            band,
        );
        let mut renderer = Self::assemble(
            device,
            queue,
            color_format,
            chunk_meshes,
            Vec::new(),
            grid_dimensions,
        );
        // Retain the two-layer chunks + frame so `rebuild_for_band` re-meshes the band
        // slab from the store (ADR 0010 #53) — the layer scrubber on the two-layer path.
        renderer.source_two_layer_chunks = chunks.to_vec();
        renderer.source_two_layer_recentre = recentre_voxels;
        renderer.source_two_layer_density = voxels_per_block.max(1);
        // The mesh was built AT `band`, so record it — a same-band `update_uniforms` is then
        // a no-op instead of a full re-mesh (M2). A later band change still re-clips.
        renderer.current_band = band;
        renderer
    }

    /// Shared GPU-resource assembly for both the dense ([`new_from_chunks`]) and two-layer
    /// ([`new_from_two_layer_chunks`]) builders: upload the per-chunk meshes, build the
    /// uniform / per-draw-overlay / atlas / loaded bind groups + pipelines, and assemble
    /// the renderer. `source_chunk_grids` is retained for the band reclip (empty on the
    /// two-layer path, which stays FULL-band until E5).
    fn assemble(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        chunk_meshes: Vec<CuboidChunkMesh>,
        source_chunk_grids: Vec<([i32; 3], VoxelGrid)>,
        grid_dimensions: [u32; 3],
    ) -> Self {
        let total_box_count = chunk_meshes.iter().map(|m| m.box_count).sum();
        let chunk_buffers = upload_chunk_meshes(device, &chunk_meshes);

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cuboid uniforms"),
            size: std::mem::size_of::<CuboidUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cuboid uniform bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cuboid uniform bind group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // ADR 0012 (H1): the onion ghost draw's own uniform buffer + bind group (same
        // layout as the solid, a separate buffer so one frame carries both states).
        let ghost_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cuboid ghost uniforms"),
            size: std::mem::size_of::<CuboidUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ghost_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cuboid ghost uniform bind group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ghost_uniform_buffer.as_entire_binding(),
            }],
        });

        // --- Per-draw on-face-grid overlay-active uniform (group 2, ADR 0003 §3c) ---
        // The overlay flag is no longer a vertex attribute (ADR 0010 E3): a chunk mesh is
        // split into an overlay-off and an overlay-on draw, each selecting this per-draw
        // `u32` via a DYNAMIC OFFSET. Two entries — `0` then `1` — packed one
        // `min_uniform_buffer_offset_alignment` apart, so the off-draw binds offset 0 and
        // the on-draw binds the stride.
        let (overlay_bind_group, overlay_dynamic_stride) =
            build_overlay_bind_group(device, &overlay_bind_group_layout(device));

        // --- Material texture ATLAS (E3c-1 / ADR 0002 O8) ---
        // Pack ALL material textures (Stone/Wood/Plain) into ONE atlas image and
        // bind it as a SINGLE 2D texture, so a chunk of mixed materials is one mesh
        // = one draw (the Vintage Story approach) — no per-material texture bind.
        // Each face's `material_id` maps to its atlas sub-rect (uploaded in the
        // uniforms); the shader tiles the per-voxel slice INTO that sub-rect.
        //
        // Sampler is CLAMP-to-edge + Nearest (matching the instanced texel grid).
        // The per-voxel tiling can NOT use a Repeat sampler here — Repeat would wrap
        // to the WHOLE atlas, i.e. into a neighbour material — so the shader does the
        // `fract`-tiling into the sub-rect itself, and the atlas's replicated-edge
        // gutter + half-texel inset (see `texture_atlas`) defend the cell borders.
        let atlas = MaterialAtlas::from_procedural_materials();
        let atlas_rects = atlas_rects_from(&atlas);
        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cuboid atlas sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let atlas_bind_group_layout = build_atlas_bind_group_layout(device);
        let atlas_texture = upload_atlas_texture(device, queue, &atlas);
        let atlas_view = atlas_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let atlas_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cuboid atlas bind group"),
            layout: &atlas_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&atlas_sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cuboid shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cuboid.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cuboid pipeline layout"),
            bind_group_layouts: &[
                Some(&uniform_bind_group_layout),
                Some(&atlas_bind_group_layout),
                // group(2): the per-draw overlay-active uniform (ADR 0003 §3c).
                Some(&overlay_bind_group_layout(device)),
            ],
            immediate_size: 0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CuboidVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as u64,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as u64,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Uint32,
                },
                // ADR 0003 §3c / ADR 0010 E3: the on-face-grid flag is NO LONGER a vertex
                // attribute — the chunk mesh is split into overlay-off / overlay-on draws,
                // each selecting a per-draw `grid_overlay_active` uniform (group 2).
            ],
        };

        // Build the render pipeline, parameterized by cull mode: the normal pass
        // back-culls; the debug-faces pass disables culling so a back face that is
        // the nearest surface (a winding bug) still draws and is flagged by the
        // shader's `front_facing` marker — exactly like the instanced path's
        // cull-on / cull-off pipeline pair.
        let build_pipeline = |label: &str, cull_mode: Option<wgpu::Face>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vertex_main"),
                    buffers: std::slice::from_ref(&vertex_layout),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fragment_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: MSAA_SAMPLE_COUNT,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline = build_pipeline("cuboid pipeline", Some(wgpu::Face::Back));
        let debug_pipeline = build_pipeline("cuboid debug pipeline", None);

        // ADR 0012 (H1): the onion GHOST pipeline. Same shader + layout as the solid, but
        // alpha-blends the flat-tinted ghost OVER the solid, depth-tested `Less`. Depth WRITE
        // is ON (not off): each pixel then shows only the NEAREST ghost surface, blended once
        // — NOT an order-dependent accumulation of every overlapping translucent face. This
        // makes the ghost render a pure function of the visible surface, so it is IDENTICAL
        // across the display paths whose greedy decomposition / raymarch differ face-for-face
        // (dense vs two-layer mesh, and the brick raymarch) exactly as the OPAQUE solid render
        // already matches — the `brick_golden_matches_dense` / two-layer cross-checks depend
        // on it. Solid geometry (drawn first) still occludes the ghost via the same depth
        // buffer; the ghost may occlude the depth-tested overlays drawn after it, which for a
        // translucent context slab is acceptable. Back-face culled like the solid.
        let ghost_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cuboid onion ghost pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                buffers: std::slice::from_ref(&vertex_layout),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: MSAA_SAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        // --- Loaded-VS-block pipelines (part of #20) ---
        // A second shader + pipeline pair that binds the applied block's 6-layer
        // D2Array at group(1) (built externally by `LoadedMaterial`, against the
        // SAME `build_face_material_layout` descriptor used here, so the bind group
        // is layout-compatible) and selects the per-face layer by the face normal.
        // It shares the uniform group(0) and the same vertex layout, so a loaded
        // block renders pixel-aligned with the procedural geometry — only the
        // texture source differs. The procedural atlas pipelines stay the default.
        let loaded_material_layout = crate::renderer::build_face_material_layout(device);
        // The shared material sampler (nearest, clamp-to-edge) — reused by loaded VS
        // blocks so they slice/filter exactly like the procedural atlas. Retained on
        // the renderer and exposed so the app can build a `LoadedMaterial` against it.
        let loaded_material_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cuboid loaded material sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let loaded_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cuboid loaded-block shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cuboid_loaded.wgsl").into()),
        });
        let loaded_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("cuboid loaded pipeline layout"),
                bind_group_layouts: &[
                    Some(&uniform_bind_group_layout),
                    Some(&loaded_material_layout),
                    // group(2): the per-draw overlay-active uniform (ADR 0003 §3c).
                    Some(&overlay_bind_group_layout(device)),
                ],
                immediate_size: 0,
            });
        let build_loaded_pipeline = |label: &str, cull_mode: Option<wgpu::Face>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&loaded_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &loaded_shader,
                    entry_point: Some("vertex_main"),
                    buffers: std::slice::from_ref(&vertex_layout),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &loaded_shader,
                    entry_point: Some("fragment_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: MSAA_SAMPLE_COUNT,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            })
        };
        let loaded_pipeline = build_loaded_pipeline("cuboid loaded pipeline", Some(wgpu::Face::Back));
        let loaded_debug_pipeline = build_loaded_pipeline("cuboid loaded debug pipeline", None);

        // Every resident chunk visible until the next frustum cull in `update_uniforms`.
        let mut visible_chunks: Vec<[i32; 3]> = chunk_buffers.keys().copied().collect();
        visible_chunks.sort_unstable();

        Self {
            pipeline,
            debug_pipeline,
            loaded_pipeline,
            loaded_debug_pipeline,
            debug_face_mode: false,
            chunk_buffers,
            visible_chunks,
            uniform_buffer,
            uniform_bind_group,
            overlay_bind_group,
            overlay_dynamic_stride,
            atlas_bind_group,
            atlas_rects,
            bound_material: MaterialChoice::Plain,
            source_chunk_grids,
            // The dense builders retain no two-layer chunks; `new_from_two_layer_chunks`
            // overrides these after `assemble` so its band reclip re-meshes from the store.
            source_two_layer_chunks: Vec::new(),
            source_two_layer_recentre: RecentreVoxels::new([0; 3]),
            source_two_layer_density: 1,
            source_grid_dimensions: grid_dimensions,
            total_box_count,
            current_band: LayerBand::FULL,
            loaded_material_layout,
            loaded_material_sampler,
            ghost_pipeline,
            ghost_uniform_buffer,
            ghost_uniform_bind_group,
            ghost_lower_buffers: std::collections::HashMap::new(),
            ghost_upper_buffers: std::collections::HashMap::new(),
            ghost_built_band: None,
        }
    }

    /// Incrementally update the per-chunk buffers for a geometry edit (issue #40):
    /// re-mesh + re-upload ONLY the chunks the edit (and its apron neighbours) touched,
    /// drop vacated chunks, and KEEP every other chunk's existing buffers — instead of
    /// the wholesale `new_from_chunks` recreate (the measured ~600ms/edit GPU cost).
    ///
    /// `chunk_grids` is the FULL post-edit covering set (`resident_render_chunks`),
    /// needed IN FULL so the re-meshed chunks' aprons see every neighbour; `grid_dimensions`
    /// is the whole composite's voxel dims (band-clip mapping); `evicted_dirty` is the
    /// resolve cache's evicted coords for this edit (from `invalidate_aabb`).
    ///
    /// PRECONDITION: the floating origin did NOT shift since the last rebuild. Chunk
    /// grids are stored pre-rebased against the composite recentre, so a recentre shift
    /// staleens EVERY buffer — the caller must fall back to `new_from_chunks` then. The
    /// active layer band is preserved (re-meshes at `self.current_band`).
    pub fn incremental_rebuild_from_chunks(
        &mut self,
        device: &wgpu::Device,
        chunk_grids: &[([i32; 3], &VoxelGrid)],
        grid_dimensions: [u32; 3],
        evicted_dirty: &[[i32; 3]],
    ) {
        profiling::scope!("cuboid_mesh_incremental");
        self.source_grid_dimensions = grid_dimensions;

        // The renderer's KNOWN set is its source grids' coords (includes occupied-but-
        // fully-occluded chunks that carry no buffer), so occluded chunks stay stable
        // instead of being treated as "new" and re-meshed every edit.
        let resident: Vec<[i32; 3]> = self.source_chunk_grids.iter().map(|(c, _)| *c).collect();
        let occupied: Vec<[i32; 3]> = chunk_grids
            .iter()
            .filter(|(_, grid)| !grid.occupied.is_empty())
            .map(|(coord, _)| *coord)
            .collect();
        let plan = cuboid_incremental_plan(&resident, evicted_dirty, &occupied);

        // Re-mesh only the dirty-dilated subset (aprons from the full set) at the
        // active band, then upload those chunks' buffers.
        let rebuild_set: std::collections::HashSet<[i32; 3]> =
            plan.rebuild.iter().copied().collect();
        let meshes = build_chunk_meshes_with_apron_filtered(
            chunk_grids,
            Some(&rebuild_set),
            grid_dimensions,
            self.current_band,
        );
        let rebuilt_buffers = upload_chunk_meshes(device, &meshes);

        // Apply. Drop evicted buffers, then drop EVERY rebuild coord's old buffer (a
        // rebuild coord that now meshes to EMPTY — e.g. fully occluded by new neighbour
        // occupancy — produces no buffer and must lose its stale one), then insert the
        // freshly built buffers. Net result == wholesale rebuild's buffer set.
        let grids_by_coord: std::collections::HashMap<[i32; 3], &VoxelGrid> =
            chunk_grids.iter().map(|(coord, grid)| (*coord, *grid)).collect();
        for coord in &plan.evict {
            self.chunk_buffers.remove(coord);
        }
        for coord in &plan.rebuild {
            self.chunk_buffers.remove(coord);
        }
        self.chunk_buffers.extend(rebuilt_buffers);

        // Keep `source_chunk_grids` the COMPLETE current covering set (a later band
        // re-clip reads it for global occupancy): drop evicted, upsert each rebuilt
        // coord's grid. Untouched chunks are resolve-cache hits → already correct.
        let evict_set: std::collections::HashSet<[i32; 3]> = plan.evict.iter().copied().collect();
        self.source_chunk_grids
            .retain(|(coord, _)| !evict_set.contains(coord));
        for coord in &plan.rebuild {
            if let Some(grid) = grids_by_coord.get(coord) {
                match self.source_chunk_grids.iter_mut().find(|(c, _)| c == coord) {
                    Some(entry) => entry.1 = (*grid).clone(),
                    None => self.source_chunk_grids.push((*coord, (*grid).clone())),
                }
            }
        }

        // Recompute the diagnostics from the (now-correct) full buffer set. All chunks
        // visible until the next frustum cull in `update_uniforms`.
        self.total_box_count = self.chunk_buffers.values().map(|c| c.box_count).sum();
        self.visible_chunks = self.chunk_buffers.keys().copied().collect();
        self.visible_chunks.sort_unstable();
    }

    /// Incrementally update the per-chunk buffers for a geometry edit on the **two-layer**
    /// path (issue #55 — the two-layer analogue of `incremental_rebuild_from_chunks`):
    /// re-mesh + re-upload ONLY the chunks the edit (and its 26-neighbourhood seam footprint)
    /// touched, drop vacated chunks, and KEEP every other chunk's existing buffers — instead
    /// of the wholesale `new_from_two_layer_chunks` recreate that re-meshes + re-uploads the
    /// WHOLE resident set every edit (the exact per-edit latency #40 fixed for the dense path,
    /// regressed onto the two-layer live renderer after E5).
    ///
    /// `chunks` is the FULL post-edit covering set (the `TwoLayerResidentCache`'s resident
    /// chunks), needed IN FULL so the re-meshed chunks' seam-flag culling consults every
    /// neighbour; `recentre_voxels` / `voxels_per_block` are the resolve's carried frame
    /// (ADR 0008); `grid_dimensions` the whole composite's voxel dims (band-clip mapping);
    /// `evicted_dirty` the resident cache's evicted coords for this edit (from
    /// [`TwoLayerResidentCache::invalidate_aabb`](crate::two_layer_store::TwoLayerResidentCache::invalidate_aabb)).
    ///
    /// The dirty set is dilated by the 26-neighbourhood via the SAME
    /// [`cuboid_incremental_plan`] the dense path uses — the seam-solidity dependency footprint
    /// is that same 26-neighbourhood (a neighbour's coarse / microblock face occupancy can cull
    /// this chunk's seam faces). Applying the plan — re-mesh `rebuild`, drop `evict`, keep the
    /// rest — yields a per-chunk buffer set IDENTICAL to a wholesale two-layer rebuild (proven
    /// by `incremental_two_layer_gpu_buffer_rebuild_equals_wholesale`).
    ///
    /// PRECONDITION: this must be the two-layer path (built via
    /// `new_from_two_layer_chunks`). A two-layer chunk is chunk-local-integer (ADR 0008), so
    /// — unlike the dense path — a floating-origin recentre SHIFT does NOT staleen the resident
    /// buffers (the recentre is a pure index offset re-applied here as `recentre_voxels`); the
    /// caller need not fall back on a recentre shift, only on a DENSITY change (which resizes
    /// every chunk's voxel extent and re-keys the whole buffer set). The active layer band is
    /// preserved (re-meshes at `self.current_band`).
    pub fn incremental_rebuild_from_two_layer_chunks(
        &mut self,
        device: &wgpu::Device,
        chunks: &[([i32; 3], Arc<crate::two_layer_store::TwoLayerChunk>)],
        grid_dimensions: [u32; 3],
        recentre_voxels: RecentreVoxels,
        voxels_per_block: u32,
        evicted_dirty: &[[i32; 3]],
    ) {
        profiling::scope!("cuboid_mesh_incremental_two_layer");
        self.source_grid_dimensions = grid_dimensions;
        self.source_two_layer_recentre = recentre_voxels;
        self.source_two_layer_density = voxels_per_block.max(1);

        // The renderer's KNOWN set is its retained two-layer chunks' coords (includes
        // occupied-but-fully-occluded chunks that carry no buffer), so occluded chunks stay
        // stable instead of being treated as "new" and re-meshed every edit.
        let resident: Vec<[i32; 3]> =
            self.source_two_layer_chunks.iter().map(|(c, _)| *c).collect();
        let occupied: Vec<[i32; 3]> = chunks
            .iter()
            .filter(|(_, chunk)| chunk.has_geometry())
            .map(|(coord, _)| *coord)
            .collect();
        let plan = cuboid_incremental_plan(&resident, evicted_dirty, &occupied);

        // Re-mesh only the dirty-dilated subset (seam culling from the full set) at the
        // active band, then upload those chunks' buffers.
        let rebuild_set: std::collections::HashSet<[i32; 3]> =
            plan.rebuild.iter().copied().collect();
        let meshes = build_two_layer_chunk_meshes_filtered(
            chunks,
            Some(&rebuild_set),
            grid_dimensions,
            recentre_voxels,
            voxels_per_block,
            self.current_band,
        );
        let rebuilt_buffers = upload_chunk_meshes(device, &meshes);

        // Apply. Drop evicted buffers, then drop EVERY rebuild coord's old buffer (a rebuild
        // coord that now meshes to EMPTY — e.g. fully occluded by new neighbour occupancy —
        // produces no buffer and must lose its stale one), then insert the freshly built
        // buffers. Net result == wholesale two-layer rebuild's buffer set.
        for coord in &plan.evict {
            self.chunk_buffers.remove(coord);
        }
        for coord in &plan.rebuild {
            self.chunk_buffers.remove(coord);
        }
        self.chunk_buffers.extend(rebuilt_buffers);

        // Keep `source_two_layer_chunks` the COMPLETE current covering set (a later band
        // reclip re-meshes from it): drop evicted, upsert each rebuilt coord's chunk.
        // Untouched chunks are resident-cache hits → already correct. Rebuilding a chunk that
        // went all-air still upserts its (empty) chunk so the retained set matches `chunks`.
        let chunks_by_coord: std::collections::HashMap<[i32; 3], &Arc<crate::two_layer_store::TwoLayerChunk>> =
            chunks.iter().map(|(coord, chunk)| (*coord, chunk)).collect();
        let evict_set: std::collections::HashSet<[i32; 3]> = plan.evict.iter().copied().collect();
        self.source_two_layer_chunks
            .retain(|(coord, _)| !evict_set.contains(coord));
        for coord in &plan.rebuild {
            if let Some(&chunk) = chunks_by_coord.get(coord) {
                // `Arc::clone` (O(1)) — the retained source set shares the resident chunk,
                // never deep-copies it.
                match self
                    .source_two_layer_chunks
                    .iter_mut()
                    .find(|(c, _)| c == coord)
                {
                    Some(entry) => entry.1 = Arc::clone(chunk),
                    None => self.source_two_layer_chunks.push((*coord, Arc::clone(chunk))),
                }
            }
        }

        // Recompute the diagnostics from the (now-correct) full buffer set. All chunks
        // visible until the next frustum cull in `update_uniforms`.
        self.total_box_count = self.chunk_buffers.values().map(|c| c.box_count).sum();
        self.visible_chunks = self.chunk_buffers.keys().copied().collect();
        self.visible_chunks.sort_unstable();
    }

    /// The loaded-VS-block material bind-group layout (6-layer D2Array texture +
    /// sampler). Exposed so a runtime-loaded block (M6) can build a bind group of the
    /// SAME shape (via `LoadedMaterial`) and be drawn by the loaded pipeline.
    pub fn material_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.loaded_material_layout
    }

    /// The shared material sampler (nearest, clamp-to-edge) — reused by loaded
    /// materials so they slice/filter exactly like the procedural atlas.
    pub fn material_sampler(&self) -> &wgpu::Sampler {
        &self.loaded_material_sampler
    }

    /// Re-mesh the stored per-chunk grids CLIPPED to `band` (issue #12 parity) and
    /// re-upload every chunk's buffers, when `band` differs from the last build. The
    /// cuboid band clip masks each chunk's region before decomposition so the band
    /// edges get real cap faces, so it must rebuild geometry (a fragment discard
    /// would leave a merged column's slab open-topped). No-op when the band is
    /// unchanged.
    fn rebuild_for_band(&mut self, device: &wgpu::Device, band: LayerBand) {
        // --- SOLID geometry (clipped to the exact [band_min, band_max]; `onion_depth` is
        // NOT a solid input, so the solid band is unchanged by ADR 0012). Skipped when the
        // band is unchanged (the M2 no-swap-rehitch property). ---
        if band != self.current_band {
            self.current_band = band;
            if let Some(chunk_meshes) = self.build_band_meshes(band) {
                self.total_box_count = chunk_meshes.iter().map(|m| m.box_count).sum();
                self.chunk_buffers = upload_chunk_meshes(device, &chunk_meshes);
                // All chunks visible until the next frustum cull in `update_uniforms`.
                self.visible_chunks = self.chunk_buffers.keys().copied().collect();
                self.visible_chunks.sort_unstable();
            }
            // A source-less (empty) build leaves the geometry in place (matches pre-0012).
        }

        // --- GHOST geometry (ADR 0012 H1): the thin per-slab onion meshes. Rebuilt on a
        // band change OR when never built for this band (the first frame after an async
        // swap that pre-built only the solid — the slabs are cheap, so this is not the
        // multi-second re-mesh #60 removed). ---
        if self.ghost_built_band != Some(band) {
            self.rebuild_ghost_slabs(device, band);
            self.ghost_built_band = Some(band);
        }
    }

    /// Build the per-chunk SOLID meshes clipped to `band` from whichever source the
    /// renderer retains (the two-layer store, else the dense per-chunk grids). `None`
    /// when the renderer has neither source (an empty build). The two-layer analogue of
    /// the dense apron mesher, kept as ONE helper so [`rebuild_for_band`] and the ghost
    /// slab build share the exact same clip semantics (ADR 0012: the two ghost slabs are
    /// just this build at the slab bands).
    fn build_band_meshes(&self, band: LayerBand) -> Option<Vec<CuboidChunkMesh>> {
        if !self.source_two_layer_chunks.is_empty() {
            return Some(build_two_layer_chunk_meshes(
                &self.source_two_layer_chunks,
                self.source_grid_dimensions,
                self.source_two_layer_recentre,
                self.source_two_layer_density,
                band,
            ));
        }
        if self.source_chunk_grids.is_empty() {
            return None;
        }
        let chunk_refs: Vec<([i32; 3], &VoxelGrid)> = self
            .source_chunk_grids
            .iter()
            .map(|(coord, g)| (*coord, g))
            .collect();
        Some(build_chunk_meshes_with_apron(
            &chunk_refs,
            self.source_grid_dimensions,
            band,
        ))
    }

    /// (ADR 0012 H1) Rebuild the two onion GHOST slab meshes for `band`: the layers
    /// `[band_min − depth, band_min)` (lower slab) and `(band_max, band_max + depth]`
    /// (upper slab), the recentred-Z remainder of the onion span `AppCore::onion_fog_params`
    /// derives (floored half, Z-up, depth clamped 1..8). Each slab is meshed by the SAME
    /// banded builder the solid uses, so it carries real cap faces at the slab edges — the
    /// brick raymarch ghost's per-slab traversal clamp produces the same caps, which is what
    /// keeps `brick_golden_matches_dense` green. Empty (both maps cleared) when onion is off
    /// (`onion_depth == 0`) or a slab falls outside the grid.
    fn rebuild_ghost_slabs(&mut self, device: &wgpu::Device, band: LayerBand) {
        self.ghost_lower_buffers.clear();
        self.ghost_upper_buffers.clear();
        if band.onion_depth == 0 {
            return;
        }
        let depth = band.onion_depth;
        let grid_z = self.source_grid_dimensions[2];
        let last_layer = grid_z.saturating_sub(1);
        // Lower slab: layers [band_min − depth, band_min − 1]. Skipped when the band bottom
        // is already layer 0 (nothing below to ghost).
        if band.band_min > 0 {
            let slab = LayerBand {
                band_min: band.band_min.saturating_sub(depth),
                band_max: band.band_min - 1,
                onion_depth: 0,
            };
            if let Some(meshes) = self.build_band_meshes(slab) {
                self.ghost_lower_buffers = upload_chunk_meshes(device, &meshes);
            }
        }
        // Upper slab: layers [band_max + 1, band_max + depth]. Skipped when the band top is
        // already the last layer (nothing above to ghost).
        if band.band_max < last_layer {
            let slab = LayerBand {
                band_min: band.band_max + 1,
                band_max: (band.band_max + depth).min(last_layer),
                onion_depth: 0,
            };
            if let Some(meshes) = self.build_band_meshes(slab) {
                self.ghost_upper_buffers = upload_chunk_meshes(device, &meshes);
            }
        }
    }

    /// Total exposed quad faces across all resident chunks (diagnostic, both overlay runs).
    pub fn face_count(&self) -> u32 {
        self.chunk_buffers
            .values()
            .map(|c| (c.index_count + c.index_count_overlay) / 6)
            .sum()
    }

    /// Total triangles across all resident chunks (diagnostic, both overlay runs).
    pub fn triangle_count(&self) -> u32 {
        self.chunk_buffers
            .values()
            .map(|c| (c.index_count + c.index_count_overlay) / 3)
            .sum()
    }

    /// Total boxes the last build decomposed into across all chunks (diagnostic).
    pub fn box_count(&self) -> u32 {
        self.total_box_count
    }

    /// Number of resident render chunks (non-empty cuboid meshes).
    pub fn chunk_count(&self) -> u32 {
        self.chunk_buffers.len() as u32
    }

    /// Number of chunks that survived the last frustum cull (will be drawn).
    pub fn visible_chunk_count(&self) -> u32 {
        self.visible_chunks.len() as u32
    }

    /// Upload the per-frame uniforms (camera matrix, grid half-extent + density
    /// for the per-voxel texture slice + grid overlay, grid-overlay params +
    /// toggle, per-material base colours) and frustum-cull the mesh chunks.
    ///
    /// `grid_dimensions` give the half-extent so `world + half` is the absolute
    /// voxel position the UV slice + overlay key off. `voxels_per_block` is the
    /// density (slice size + block-line period). `grid_overlay_enabled` reflects
    /// the Display toggle. `bound` is the active procedural material: it selects
    /// the bound texture (E3b-2) AND drives the relative base-colour modulation
    /// (exactly like the instanced step-3b). `None` means a loaded VS block is
    /// active: modulation is disabled here, and the loaded-block pipeline selected in
    /// `draw` (when its 6-layer D2Array bind group is supplied) ignores the
    /// procedural atlas/modulation uniforms entirely (part of #20).
    #[allow(clippy::too_many_arguments)]
    pub fn update_uniforms(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_projection: glam::Mat4,
        grid_dimensions: [u32; 3],
        voxels_per_block: u32,
        grid_overlay_enabled: bool,
        bound: Option<MaterialChoice>,
        band: LayerBand,
        debug_face_mode: bool,
    ) {
        // Layer-range band clip (issue #12 parity): re-mesh the grid clipped to the
        // band (real cap faces at the band edges) when it changed. Debug-faces mode
        // bypasses the band (the instanced check sees the whole model), so force the
        // full band while it is on.
        let effective_band = if debug_face_mode {
            LayerBand::FULL
        } else {
            band
        };
        self.rebuild_for_band(device, effective_band);
        // The bound procedural material drives BOTH the texture binding (selected
        // in `draw`) and the per-box modulation. A `None` (loaded VS block) falls
        // back to Plain's texture + neutral modulation for now (the cuboid path
        // renders a loaded block as a single global material this sub-step).
        // Debug-faces mode forces modulation off (the shader bypasses it anyway),
        // matching the instanced path.
        let (modulation_enabled, base_colors, material) = match bound {
            Some(material) if !debug_face_mode => (
                true,
                crate::renderer::relative_material_base_colors_public(material),
                material,
            ),
            Some(material) => (
                false,
                [[1.0, 1.0, 1.0, 0.0]; MaterialChoice::MATERIAL_COUNT],
                material,
            ),
            None => (
                false,
                [[1.0, 1.0, 1.0, 0.0]; MaterialChoice::MATERIAL_COUNT],
                MaterialChoice::Plain,
            ),
        };
        self.bound_material = material;
        // Record the debug flag so `draw` selects the matching cull-off pipeline.
        self.debug_face_mode = debug_face_mode;

        let overlay = crate::renderer::grid_overlay_params();
        let uniforms = CuboidUniforms {
            view_projection: view_projection.to_cols_array_2d(),
            // Corner-anchoring: the grid's low corner is `−floor(dim/2)`, so the GPU
            // recovers the absolute voxel frame with `world_position + floor(dim/2)`
            // (integer-valued). Using `dim/2.0` would be half a voxel off for an ODD
            // dim, mis-snapping the voxel/block grid overlay and the Z-band clip.
            grid_half_extent: [
                (grid_dimensions[0] / 2) as f32,
                (grid_dimensions[1] / 2) as f32,
                (grid_dimensions[2] / 2) as f32,
            ],
            voxels_per_block: voxels_per_block.max(1) as f32,
            voxel_line_color: overlay.voxel_line_color,
            grid_overlay_enabled: if grid_overlay_enabled { 1.0 } else { 0.0 },
            block_line_color: overlay.block_line_color,
            material_modulation_enabled: if modulation_enabled { 1.0 } else { 0.0 },
            voxel_line_half_width: overlay.voxel_line_half_width,
            block_line_half_width: overlay.block_line_half_width,
            voxel_line_alpha: overlay.voxel_line_alpha,
            block_line_alpha: overlay.block_line_alpha,
            // Layer-range band clip (issue #12 parity): the shader keeps fragments
            // whose voxel layer is in [band_min, band_max] (both INCLUSIVE),
            // matching the instanced voxel pass. `LayerBand::FULL` uses band_max =
            // u32::MAX, so `as f32` (≈ 4.29e9) leaves every layer unclipped.
            band_min: band.band_min as f32,
            band_max: band.band_max as f32,
            debug_face_mode: if debug_face_mode { 1.0 } else { 0.0 },
            // ADR 0012 (H1): the SOLID draw is never the ghost — 0 here keeps the solid
            // uniform bytes identical to pre-onion-ghost (non-onion goldens byte-green).
            ghost_mode: 0.0,
            material_base_colors: base_colors,
            material_atlas_rects: self.atlas_rects,
            ghost_tint: [0.0, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // ADR 0012 (H1) — the onion GHOST uniform. Identical camera/frame to the solid,
        // but `ghost_mode = 1` (flat translucent tint) + the tint colour. Both onion
        // slabs share this ONE uniform (the slab distinction lives in the per-slab GHOST
        // geometry, not the uniform), so a band scrub only re-meshes the thin slabs and
        // never touches this buffer's shape. The tint is the SAME constant the brick
        // ghost binds — `brick_golden_matches_dense` depends on the two matching.
        let ghost_uniforms = CuboidUniforms {
            ghost_mode: 1.0,
            ghost_tint: crate::renderer::onion_ghost_tint(),
            ..uniforms
        };
        queue.write_buffer(&self.ghost_uniform_buffer, 0, bytemuck::bytes_of(&ghost_uniforms));

        // Frustum-cull the per-chunk buffers by their world AABBs (sorted for a
        // deterministic draw order; cross-chunk order is pixel-irrelevant — opaque +
        // depth-tested).
        let frustum = Frustum::from_view_projection(view_projection);
        self.visible_chunks.clear();
        for (coord, chunk) in &self.chunk_buffers {
            if frustum.intersects_aabb(&chunk.aabb) {
                self.visible_chunks.push(*coord);
            }
        }
        self.visible_chunks.sort_unstable();
    }

    /// Record the cuboid draw into an already-begun render pass. Iterates the
    /// frustum-visible per-chunk buffers, one indexed draw per chunk over its own
    /// vertex/index buffer.
    ///
    /// `loaded_material` (part of #20): when an applied/loaded VS block is active,
    /// the caller passes the block's 6-layer D2Array bind group (`LoadedMaterial::
    /// bind_group`); the cuboid path then selects the loaded-block pipeline + shader,
    /// binding that D2Array at group(1) and selecting the per-face layer by the face
    /// normal — so the cuboid path shows the SAME texture the instanced path shows.
    /// `None` (no block applied) keeps the procedural-atlas path, unchanged.
    pub fn draw(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        loaded_material: Option<&wgpu::BindGroup>,
    ) {
        if self.chunk_buffers.is_empty() {
            return;
        }
        // Debug-faces mode selects the cull-off pipeline (matching the uploaded
        // `debug_face_mode` flag) so back faces surviving a winding bug still draw
        // and get the shader's stripe marker — same as the instanced path. The
        // pipeline pair is the loaded-block pair when a block is applied (binds its
        // D2Array at group 1), else the procedural atlas pair.
        let (pipeline, material_bind_group) = match loaded_material {
            Some(loaded_bind_group) => (
                if self.debug_face_mode {
                    &self.loaded_debug_pipeline
                } else {
                    &self.loaded_pipeline
                },
                loaded_bind_group,
            ),
            None => (
                if self.debug_face_mode {
                    &self.debug_pipeline
                } else {
                    &self.pipeline
                },
                &self.atlas_bind_group,
            ),
        };
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        // group(1) is either the procedural ATLAS (per-face `material_id` → atlas
        // sub-rect in the shader, one bind for a mixed-material chunk) or the loaded
        // block's D2Array (per-face layer selected by normal). One bind, one draw/chunk.
        render_pass.set_bind_group(1, material_bind_group, &[]);
        for coord in &self.visible_chunks {
            let Some(chunk) = self.chunk_buffers.get(coord) else {
                continue;
            };
            if chunk.index_count == 0 && chunk.index_count_overlay == 0 {
                continue;
            }
            render_pass.set_vertex_buffer(0, chunk.vertex_buffer.slice(..));
            render_pass.set_index_buffer(chunk.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            // ADR 0003 §3c: two draws per chunk — the overlay-OFF run (group(2) dynamic
            // offset 0 → overlay-active uniform = 0) then the overlay-ON run (dynamic
            // offset `overlay_dynamic_stride` → uniform = 1). The on-run is the second
            // half of the single index buffer (byte offset `index_count * 4`).
            if chunk.index_count > 0 {
                render_pass.set_bind_group(2, &self.overlay_bind_group, &[0]);
                render_pass.draw_indexed(0..chunk.index_count, 0, 0..1);
            }
            if chunk.index_count_overlay > 0 {
                render_pass.set_bind_group(2, &self.overlay_bind_group, &[self.overlay_dynamic_stride]);
                let start = chunk.index_count;
                render_pass
                    .draw_indexed(start..start + chunk.index_count_overlay, 0, 0..1);
            }
        }
    }

    /// (ADR 0012 H1) Draw the onion GHOST pass: the two thin per-slab meshes flat-tinted
    /// translucent, alpha-blended over the solid with the depth test `Less` + depth WRITE ON
    /// (nearest ghost surface wins, builder-independent). MUST be called AFTER `draw`, inside
    /// the same MSAA pass (the solid's depth is what occludes the ghost). A no-op when onion is off (both slab
    /// maps empty). Group(1) binds the procedural atlas even for loaded-material scenes —
    /// the ghost shader flat-tints and never samples it (flat tint even over `cuboid_loaded`).
    /// Both slabs are drawn with the whole index buffer per chunk (overlay-off + overlay-on
    /// runs together): the ghost ignores the on-face grid overlay, so one draw suffices.
    pub fn draw_ghost(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        if self.ghost_lower_buffers.is_empty() && self.ghost_upper_buffers.is_empty() {
            return;
        }
        render_pass.set_pipeline(&self.ghost_pipeline);
        render_pass.set_bind_group(0, &self.ghost_uniform_bind_group, &[]);
        render_pass.set_bind_group(1, &self.atlas_bind_group, &[]);
        // Overlay disabled for the ghost (the shader flat-tints before any overlay); bind
        // the off-slot (0) so group(2) is satisfied.
        render_pass.set_bind_group(2, &self.overlay_bind_group, &[0]);
        // Lower slab THEN upper slab — the same order the brick raymarch ghost draws its
        // two slabs, so any screen overlap of the two blends identically across paths.
        // Within a slab, iterate in SORTED coord order: the ghost writes no depth, so a
        // stable draw order keeps the alpha-blend result deterministic across runs AND
        // identical between the dense and two-layer builds (the two_layer golden gate).
        for buffers in [&self.ghost_lower_buffers, &self.ghost_upper_buffers] {
            let mut coords: Vec<[i32; 3]> = buffers.keys().copied().collect();
            coords.sort_unstable();
            for coord in coords {
                let Some(chunk) = buffers.get(&coord) else {
                    continue;
                };
                let total = chunk.index_count + chunk.index_count_overlay;
                if total == 0 {
                    continue;
                }
                render_pass.set_vertex_buffer(0, chunk.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(chunk.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..total, 0, 0..1);
            }
        }
    }
}

/// Upload built per-chunk meshes into GPU buffers, one [`CuboidChunkBuffers`] per
/// non-empty chunk (issue #20 S6c-2d).
fn upload_chunk_meshes(
    device: &wgpu::Device,
    chunk_meshes: &[CuboidChunkMesh],
) -> std::collections::HashMap<[i32; 3], CuboidChunkBuffers> {
    let mut buffers = std::collections::HashMap::new();
    for mesh in chunk_meshes {
        if mesh.indices.is_empty() && mesh.indices_overlay.is_empty() {
            continue;
        }
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cuboid chunk vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        // One index buffer = overlay-OFF run then overlay-ON run (ADR 0003 §3c); the two
        // draws slice it by count + offset.
        let mut all_indices = mesh.indices.clone();
        all_indices.extend_from_slice(&mesh.indices_overlay);
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cuboid chunk indices"),
            contents: bytemuck::cast_slice(&all_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        buffers.insert(
            mesh.coord,
            CuboidChunkBuffers {
                vertex_buffer,
                index_buffer,
                index_count: mesh.indices.len() as u32,
                index_count_overlay: mesh.indices_overlay.len() as u32,
                aabb: mesh.aabb,
                box_count: mesh.box_count,
            },
        );
    }
    buffers
}

/// Bucket a whole [`VoxelGrid`] into per-chunk sub-grids keyed by integer chunk
/// coord `floor(world_position / chunk_extent)` (issue #20 S6c-2d) — the SAME key
/// [`crate::renderer::VoxelRenderer::rebuild_instances`] uses, so the cuboid `new`
/// wrapper's chunk partition matches the instanced one and the resolve cache's
/// per-chunk accessor. A sub-grid carries only the occupied voxels (its `dimensions`
/// is unused by the apron mesher, which keys off `world_position`).
fn bucket_grid_into_chunk_grids(
    grid: &VoxelGrid,
    voxels_per_block: u32,
) -> Vec<([i32; 3], VoxelGrid)> {
    use std::collections::HashMap;
    let chunk_extent = (crate::core_geom::CHUNK_BLOCKS * voxels_per_block.max(1)) as f32;
    let mut buckets: HashMap<[i32; 3], VoxelGrid> = HashMap::new();
    for voxel in &grid.occupied {
        let position = voxel.world_position();
        let key = [
            (position[0] / chunk_extent).floor() as i32,
            (position[1] / chunk_extent).floor() as i32,
            (position[2] / chunk_extent).floor() as i32,
        ];
        buckets
            .entry(key)
            .or_insert_with(|| VoxelGrid::new([0, 0, 0]))
            .occupied
            .push(*voxel);
    }
    let mut out: Vec<([i32; 3], VoxelGrid)> = buckets.into_iter().collect();
    out.sort_unstable_by_key(|(coord, _)| *coord);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::Voxel;

    /// Perf probe (mesh-display scaling guard): per-size timing + emission counts of the pure
    /// CPU two-layer mesh generation — the path the display takes when a loaded VS material
    /// disengages the brick raymarch. Run:
    /// `cargo test --release mesh_pipeline_scaling_probe -- --ignored --nocapture`.
    #[test]
    #[ignore = "perf probe — run in release with --nocapture"]
    fn mesh_pipeline_scaling_probe() {
        use crate::core_geom::MaterialChoice;
        use crate::scene::{Node, NodeContent, Scene};
        use crate::sketch::{PlaneAxis, Sketch, SketchSolid};
        let density = 16u32;
        for blocks in [50i64, 125, 250, 500] {
            let edge = blocks * density as i64;
            let extrude =
                SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, edge, edge), edge as u32);
            let scene = Scene::from_nodes(vec![Node::new(
                "Box",
                NodeContent::SketchTool { producer: extrude, material: MaterialChoice::Stone },
            )]);
            let chunks =
                crate::two_layer_store::TwoLayerStore::enabled().build_covering_chunks(
                    &scene, density, 0,
                );
            let dims = scene.placed_region_dimensions(density);
            let recentre = scene.recentre_voxels_for_resolve(density);
            let start = std::time::Instant::now();
            let meshes =
                build_two_layer_chunk_meshes(&chunks, dims, recentre, density, LayerBand::FULL);
            let elapsed = start.elapsed();
            let boxes: u64 = meshes.iter().map(|mesh| mesh.box_count as u64).sum();
            let vertices: u64 = meshes.iter().map(|mesh| mesh.vertices.len() as u64).sum();
            let indices: u64 = meshes
                .iter()
                .map(|mesh| (mesh.indices.len() + mesh.indices_overlay.len()) as u64)
                .sum();
            let vertex_megabytes =
                (vertices * std::mem::size_of::<CuboidVertex>() as u64) as f64 / 1.0e6;
            println!(
                "mesh {edge}^3 vx: {} chunk meshes | {boxes} boxes | \
                 {vertices} vertices ({vertex_megabytes:.0} MB) | {indices} indices | \
                 CPU mesh-gen {elapsed:?}",
                meshes.len(),
            );
        }
    }

    /// Build a tiny grid from a set of occupied voxel indices, all one material, with
    /// the given dimensions, in the RECENTRED render frame the live cuboid path sees.
    ///
    /// The stored `local_index` reproduces the retired f32 fixture's
    /// `world_position = index + 0.5 − dim/2` EXACTLY for an EVEN dim (where the centre
    /// is a half-integer): `local_index = floor(index + 0.5 − dim/2)`, so
    /// `world_position()` (= `local_index + 0.5`) equals the old value bit-for-bit and the
    /// band-clip's `half = floor(dim/2)` frame assumption still holds. (An ODD dim's old
    /// centre fell on an INTEGER, which the integer payload — whose centres are always
    /// half-integers — cannot represent; the one odd-dim test below corner-anchors and
    /// reads the world planes directly, since the mesher is anchor-shift-invariant.)
    fn grid_from_indices(dimensions: [u32; 3], cells: &[[u32; 3]], material: u16) -> VoxelGrid {
        let half = [
            dimensions[0] as f32 / 2.0,
            dimensions[1] as f32 / 2.0,
            dimensions[2] as f32 / 2.0,
        ];
        let mut grid = VoxelGrid::new(dimensions);
        for &[i, j, k] in cells {
            grid.occupied.push(Voxel {
                local_index: [
                    (i as f32 + 0.5 - half[0]).floor() as i32,
                    (j as f32 + 0.5 - half[1]).floor() as i32,
                    (k as f32 + 0.5 - half[2]).floor() as i32,
                ],
                block_local_coord: [0, 0, 0],
                block_id: crate::core_geom::BlockId(material),
                attrs: crate::core_geom::BlockAttrs::DEFAULT,
                grid_overlay: false,
            });
        }
        grid
    }

    #[test]
    fn single_voxel_cube_has_six_faces() {
        // A solid 1-voxel "block" in a 3³ grid → 1 box → 6 exposed faces,
        // 12 triangles, 36 indices, 24 vertices.
        let grid = grid_from_indices([3, 3, 3], &[[1, 1, 1]], 0);
        let mesh = build_cuboid_mesh(&grid, 1);
        assert_eq!(mesh.box_count(), 1, "single voxel → one box");
        assert_eq!(mesh.face_count(), 6, "all six faces exposed");
        assert_eq!(mesh.triangle_count(), 12, "6 faces × 2 triangles");
        assert_eq!(mesh.index_count(), 36, "6 faces × 6 indices");
        assert_eq!(mesh.vertex_count(), 24, "6 faces × 4 verts");
    }

    #[test]
    fn two_voxel_run_is_one_box_six_faces() {
        // A 2-voxel run along X (same material) merges into a single box; its
        // exposed-face mesh still has exactly 6 faces (the shared internal face
        // between the two voxels is culled BY merging into one box).
        let grid = grid_from_indices([4, 3, 3], &[[1, 1, 1], [2, 1, 1]], 0);
        let mesh = build_cuboid_mesh(&grid, 1);
        assert_eq!(mesh.box_count(), 1, "2-voxel run → one merged box");
        assert_eq!(mesh.face_count(), 6, "merged box still has 6 exposed faces");
        assert_eq!(mesh.triangle_count(), 12);
        assert_eq!(mesh.index_count(), 36);
    }

    #[test]
    fn solid_block_collapses_to_six_faces() {
        // A solid 4×4×4 single-material block → 1 box → 6 faces (vs 4096 cubes /
        // 24576 instanced triangles): the order-of-magnitude reduction.
        let mut cells = Vec::new();
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    cells.push([x, y, z]);
                }
            }
        }
        let grid = grid_from_indices([4, 4, 4], &cells, 0);
        let mesh = build_cuboid_mesh(&grid, 1);
        assert_eq!(mesh.box_count(), 1);
        assert_eq!(mesh.face_count(), 6);
        assert_eq!(mesh.triangle_count(), 12);
    }

    #[test]
    fn adjacent_solid_faces_are_culled() {
        // Two separate boxes of DIFFERENT materials sharing a face: the shared
        // faces are culled (backed by solid), so the combined silhouette is a
        // 2×1×1 box surface = 6 faces, not 12.
        let mut grid = grid_from_indices([4, 3, 3], &[[1, 1, 1]], 0);
        // Second voxel, different material, adjacent in +X — built in the SAME recentred
        // frame `grid_from_indices` uses (`floor(index + 0.5 − dim/2)`) so it lands next
        // to the first voxel.
        let half = [2.0f32, 1.5, 1.5];
        grid.occupied.push(Voxel {
            local_index: [
                (2.0 + 0.5 - half[0]).floor() as i32,
                (1.0 + 0.5 - half[1]).floor() as i32,
                (1.0 + 0.5 - half[2]).floor() as i32,
            ],
            block_local_coord: [0, 0, 0],
            block_id: crate::core_geom::BlockId(1),
            attrs: crate::core_geom::BlockAttrs::DEFAULT,
            grid_overlay: false,
        });
        let mesh = build_cuboid_mesh(&grid, 1);
        assert_eq!(mesh.box_count(), 2, "different materials → two boxes");
        // 2 boxes × 6 faces = 12, minus the 2 shared (one each side) = 10 faces.
        assert_eq!(
            mesh.face_count(),
            10,
            "the two faces between the adjacent boxes are culled"
        );
    }

    /// E3b-2: the per-voxel UV is the absolute voxel position on the face's two
    /// in-plane axes, so a face spanning N voxels must have vertices whose
    /// absolute index spans 0..N on those axes (the shader divides by density +
    /// Repeat-tiles, giving one texture tile per voxel). Here a 3-voxel X-run in a
    /// 5³ grid merges to one box; its top (Z-up: +Z) face must span 3 voxels along X
    /// and 1 along Y, i.e. world X-extent 3.
    #[test]
    fn merged_face_spans_one_uv_unit_per_voxel() {
        // Use an EVEN grid dim (6) so the recentred fixture lands on half-integer
        // centres the integer payload represents exactly (an odd dim's old centre fell on
        // an integer — see `grid_from_indices`). The X-run [1,2,3] then occupies absolute
        // voxels 1..=3 with planes at 1 and 4, exactly as before.
        let grid = grid_from_indices([6, 6, 6], &[[1, 3, 3], [2, 3, 3], [3, 3, 3]], 0);
        let mesh = build_cuboid_mesh(&grid, 1);
        assert_eq!(mesh.box_count(), 1, "3-voxel X-run merges to one box");

        // Absolute voxel position = world position + half (dims/2). The UV in the
        // shader uses exactly this, so spanning 3 units in X across the face means
        // the texture tiles 3× (once per voxel) with a Repeat sampler.
        let half = [3.0f32, 3.0, 3.0];
        let abs_x: Vec<f32> = mesh
            .vertices
            .iter()
            .map(|v| v.position[0] + half[0])
            .collect();
        let min_x = abs_x.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_x = abs_x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // The run occupies absolute X voxels 1..=3 → planes at 1 and 4.
        assert_eq!(min_x, 1.0, "box min X plane = first voxel index");
        assert_eq!(max_x, 4.0, "box max+1 X plane = last voxel index + 1");
        assert_eq!(max_x - min_x, 3.0, "face spans 3 voxel UV units along X");
    }

    /// E3b-2: the face-normal → texture-array layer mapping must match the loaded
    /// shader's `face_layer` (cuboid_loaded.wgsl) EXACTLY. Z-up: +Z = up (2), -Z =
    /// down (3); ±X = east/west (0/1); -Y = south/front (4), +Y = north/back (5).
    /// Replicated here as a pure function so the shader↔CPU mapping is regression-
    /// guarded (the vertical texture axis is Z, not Y).
    #[test]
    fn face_normal_to_layer_matches_instanced() {
        // Byte-for-byte the same branch order as cuboid_loaded.wgsl::face_layer.
        fn face_layer(normal: [f32; 3]) -> i32 {
            let m = [normal[0].abs(), normal[1].abs(), normal[2].abs()];
            if m[2] > 0.5 {
                // Vertical (Z-up): +Z = up (2), -Z = down (3).
                if normal[2] > 0.0 { 2 } else { 3 }
            } else if m[0] > 0.5 {
                if normal[0] > 0.0 { 0 } else { 1 }
            } else if normal[1] < 0.0 {
                // -Y = south/front.
                4
            } else {
                // +Y = north/back.
                5
            }
        }
        // FACE_TEMPLATES order is +X,-X,+Y,-Y,+Z,-Z → Z-up layers 0,1,5,4,2,3.
        let expected = [0, 1, 5, 4, 2, 3];
        for (face, &want) in FACE_TEMPLATES.iter().zip(expected.iter()) {
            assert_eq!(face_layer(face.normal), want, "normal {:?}", face.normal);
        }
    }

    /// The cuboid decomposition must cover EVERY occupied voxel of the grid — the
    /// box set's total voxel count equals `grid.occupied.len()` — for ANY shape AND
    /// for a recentred cloud. This is the regression guard for the "partial
    /// silhouette" bug (#18): the cuboid cylinder rendered ~1/4 of the disc because
    /// the densifier anchored region index 0 at `dimensions/2` and silently dropped
    /// the voxels of a recentred cloud (the scene resolve path shifts an odd-block
    /// shape off-centre). A wedge means lost coverage; this asserts none is lost.
    #[test]
    fn cuboid_covers_every_voxel_for_all_shapes() {
        use crate::voxel::{SdfShape, ShapeKind, VoxelProducer};

        for &kind in &[
            ShapeKind::Cylinder,
            ShapeKind::Sphere,
            ShapeKind::Torus,
            ShapeKind::Box,
            ShapeKind::Tube,
        ] {
            // 5×1×5 is the default disc (odd X/Z blocks → the recentre that exposed
            // the bug); also exercise an odd-all-axes size to be thorough.
            for &size in &[[5u32, 1, 5], [3, 3, 3], [5, 3, 7]] {
                let voxels_per_block = 8;
                let shape = SdfShape::from_blocks(kind, size, 1, voxels_per_block);
                // Shift-invariance: also run a deliberately recentred copy of the
                // grid (every voxel +8 in each axis, like `resolve_region`'s
                // off-centre composite) — coverage must be identical.
                for shift in [0.0f32, 8.0] {
                    let mut shifted = VoxelGrid::new(shape.grid_dimensions(voxels_per_block));
                    shape.resolve(&mut shifted, voxels_per_block);
                    if shifted.occupied.is_empty() {
                        continue;
                    }
                    for voxel in &mut shifted.occupied {
                        for axis in 0..3 {
                            voxel.local_index[axis] += shift as i32;
                        }
                    }

                    let (region, _world_offset) = region_from_voxel_cloud(&shifted);
                    let region_solid =
                        region.cells.iter().filter(|c| c.is_some()).count();
                    let boxes = decompose_into_boxes(&region);
                    let covered: u64 = boxes.iter().map(|b| b.cell_count()).sum();

                    assert_eq!(
                        region_solid,
                        shifted.occupied.len(),
                        "{kind:?} {size:?} shift={shift}: densified region lost \
                         voxels ({region_solid} of {})",
                        shifted.occupied.len()
                    );
                    assert_eq!(
                        covered,
                        shifted.occupied.len() as u64,
                        "{kind:?} {size:?} shift={shift}: cuboid boxes cover \
                         {covered} of {} voxels (a partial silhouette)",
                        shifted.occupied.len()
                    );
                }
            }
        }
    }

    /// E3b-3: the layer-range band clip masks the densified region to the band's
    /// absolute Z-layer range (Z-up: layers are Z-slices) BEFORE decomposition, so
    /// clipping a solid block to a sub-band yields a thinner block — with NEW cap
    /// faces at the band edges (a fragment discard on the single merged column would
    /// leave it open-topped, with no caps). Here a solid 4×4×4 block (one tall box)
    /// clipped to a 2-layer band must mesh as a 4×4×2 box: still 6 faces, but spanning
    /// exactly 2 voxels in Z.
    #[test]
    fn band_clip_masks_region_and_caps_the_slab() {
        let mut cells = Vec::new();
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    cells.push([x, y, z]);
                }
            }
        }
        // A centred 4³ block: half_z = 2, so absolute layer == region-local Z here.
        let grid = grid_from_indices([4, 4, 4], &cells, 0);

        // Full band → the whole block: 1 box, 6 faces, Z-span 4.
        let full = build_cuboid_mesh_banded(&grid, 1, LayerBand::FULL);
        assert_eq!(full.box_count(), 1);
        assert_eq!(full.face_count(), 6);

        // Band [1, 2] (inclusive) → only Z-layers 1 and 2 survive: a 4×4×2 slab.
        let band = LayerBand {
            band_min: 1,
            band_max: 2,
            onion_depth: 0,
        };
        let clipped = build_cuboid_mesh_banded(&grid, 1, band);
        assert_eq!(clipped.box_count(), 1, "the clipped slab is still one box");
        assert_eq!(
            clipped.face_count(),
            6,
            "the band edges get real cap faces (top + bottom), so still 6 faces"
        );

        // The clipped slab spans EXACTLY 2 voxels in Z (the band height), with new
        // caps — confirming masking, not a fragment discard.
        let half_z = 2.0f32;
        let abs_z: Vec<f32> = clipped
            .vertices
            .iter()
            .map(|v| v.position[2] + half_z)
            .collect();
        let min_z = abs_z.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_z = abs_z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(min_z, 1.0, "slab bottom cap at the band's lower layer");
        assert_eq!(max_z, 3.0, "slab top cap at the band's upper layer + 1");
        assert_eq!(max_z - min_z, 2.0, "slab is exactly the 2-layer band tall");
    }

    /// A band entirely OUTSIDE the occupied layers clips everything away.
    #[test]
    fn band_clip_outside_occupied_layers_is_empty() {
        let grid = grid_from_indices([4, 4, 4], &[[1, 1, 1], [2, 2, 2]], 0);
        let band = LayerBand {
            band_min: 10,
            band_max: 12,
            onion_depth: 0,
        };
        let mesh = build_cuboid_mesh_banded(&grid, 1, band);
        assert_eq!(mesh.box_count(), 0, "no voxel falls in the band");
        assert_eq!(mesh.face_count(), 0);
    }

    /// Vertex-position ↔ voxel-extent correspondence: every emitted face vertex
    /// must land on one of a box's integer corner planes — `min` (the box's
    /// min-corner) or `max + 1` (its exclusive far plane) on each axis — once the
    /// shift-invariant `world_offset` is subtracted back out. This proves the
    /// geometry the mesher emits matches the integer box bounds the decomposition
    /// produced (no off-by-one / wrong-plane vertex), as a pure CPU assertion.
    #[test]
    fn vertex_positions_match_box_voxel_extents() {
        use std::collections::HashSet;

        // A few irregular shapes so vertices come from boxes of varied extents and
        // a multi-box decomposition (different materials force a split).
        let single = grid_from_indices([3, 3, 3], &[[1, 1, 1]], 0);
        let run = grid_from_indices([5, 5, 5], &[[1, 2, 2], [2, 2, 2], [3, 2, 2]], 0);
        // Two adjacent boxes of different materials (a 2-box decomposition).
        let mut two_box = grid_from_indices([4, 3, 3], &[[1, 1, 1]], 0);
        // Adjacent in +X, built in the SAME recentred frame as `grid_from_indices`.
        let half = [2.0f32, 1.5, 1.5];
        two_box.occupied.push(Voxel {
            local_index: [
                (2.0 + 0.5 - half[0]).floor() as i32,
                (1.0 + 0.5 - half[1]).floor() as i32,
                (1.0 + 0.5 - half[2]).floor() as i32,
            ],
            block_local_coord: [0, 0, 0],
            block_id: crate::core_geom::BlockId(1),
            attrs: crate::core_geom::BlockAttrs::DEFAULT,
            grid_overlay: false,
        });

        for grid in [single, run, two_box] {
            let mesh = build_cuboid_mesh(&grid, 1);
            // Recover the exact region + offset + boxes the mesher used.
            let (region, world_offset) = region_from_voxel_cloud(&grid);
            let boxes = decompose_into_boxes(&region);
            assert!(!boxes.is_empty(), "test shape must decompose to ≥1 box");

            // The set of valid corner planes per axis = {min} ∪ {max+1} over boxes.
            let mut valid_plane: [HashSet<i64>; 3] =
                [HashSet::new(), HashSet::new(), HashSet::new()];
            for voxel_box in &boxes {
                for (axis, planes) in valid_plane.iter_mut().enumerate() {
                    planes.insert(voxel_box.min[axis] as i64);
                    planes.insert(voxel_box.max[axis] as i64 + 1);
                }
            }

            for vertex in &mesh.vertices {
                for axis in 0..3 {
                    // Undo the world offset → region-local integer plane.
                    let local_plane = (vertex.position[axis] - world_offset[axis]).round() as i64;
                    // The round must be exact (planes are integers in local space).
                    assert!(
                        (vertex.position[axis] - world_offset[axis] - local_plane as f32).abs()
                            < 1e-4,
                        "vertex {:?} axis {axis} not on an integer local plane",
                        vertex.position
                    );
                    assert!(
                        valid_plane[axis].contains(&local_plane),
                        "vertex {:?} axis {axis} local plane {local_plane} is not a box \
                         min or max+1 plane (valid: {:?})",
                        vertex.position,
                        valid_plane[axis]
                    );
                }
            }

            // Per-box: the box's OWN min and max+1 corner planes must each appear in
            // the emitted vertex set (the box actually contributes its extents).
            let emitted: HashSet<[i64; 3]> = mesh
                .vertices
                .iter()
                .map(|vertex| {
                    [
                        (vertex.position[0] - world_offset[0]).round() as i64,
                        (vertex.position[1] - world_offset[1]).round() as i64,
                        (vertex.position[2] - world_offset[2]).round() as i64,
                    ]
                })
                .collect();
            for voxel_box in &boxes {
                let min_corner = [
                    voxel_box.min[0] as i64,
                    voxel_box.min[1] as i64,
                    voxel_box.min[2] as i64,
                ];
                let max_corner = [
                    voxel_box.max[0] as i64 + 1,
                    voxel_box.max[1] as i64 + 1,
                    voxel_box.max[2] as i64 + 1,
                ];
                assert!(
                    emitted.contains(&min_corner),
                    "box {voxel_box:?} min corner {min_corner:?} missing from vertices"
                );
                assert!(
                    emitted.contains(&max_corner),
                    "box {voxel_box:?} max+1 corner {max_corner:?} missing from vertices"
                );
            }
        }
    }

    #[test]
    fn empty_grid_has_no_mesh() {
        let grid = VoxelGrid::new([4, 4, 4]);
        let mesh = build_cuboid_mesh(&grid, 1);
        assert_eq!(mesh.box_count(), 0);
        assert_eq!(mesh.face_count(), 0);
        assert_eq!(mesh.index_count(), 0);
    }

    // ---- Per-chunk apron meshing structural parity (issue #20 S6c-2d) ----

    /// A single UNIT exposed face: the absolute integer plane coordinate on the
    /// face's axis, the two in-plane unit-cell lower coords, and the face axis +
    /// sign. Canonical regardless of how a co-planar face is split into abutting
    /// quads, so it is the granularity at which whole-region meshing and per-chunk
    /// apron meshing must produce the IDENTICAL set.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    struct UnitFace {
        /// 0 = X face, 1 = Y face, 2 = Z face.
        axis: u8,
        /// `+1` / `-1` outward direction along `axis`.
        sign: i8,
        /// Integer world plane on `axis` (the quad's constant coordinate).
        plane: i64,
        /// The two in-plane unit-cell lower coords (the axes other than `axis`).
        cell: [i64; 2],
    }

    /// Round a world coordinate that must land on an integer plane (box corners are
    /// integer planes in world space once the shift-invariant offset is folded in).
    fn round_plane(value: f32) -> i64 {
        let rounded = value.round();
        assert!(
            (value - rounded).abs() < 1e-3,
            "vertex coord {value} is not on an integer world plane"
        );
        rounded as i64
    }

    /// Explode a vertex/index mesh into its SET of unit exposed faces (the canonical
    /// granularity), in the GLOBAL INDEX frame. Mesh vertices live in world space at
    /// `global_index + world_offset`, so subtracting `world_offset` recovers the
    /// integer global-index planes the ground-truth `genuine_exposed_faces` keys off.
    /// Each quad (6 indices) lies on a plane perpendicular to its normal; it is split
    /// into the unit cells it covers in the two in-plane axes.
    fn unit_faces_in_index_frame(
        vertices: &[CuboidVertex],
        indices: &[u32],
        world_offset: [f32; 3],
    ) -> std::collections::HashSet<UnitFace> {
        let mut faces = std::collections::HashSet::new();
        let to_index = |pos: [f32; 3]| -> [f32; 3] {
            [
                pos[0] - world_offset[0],
                pos[1] - world_offset[1],
                pos[2] - world_offset[2],
            ]
        };
        // Each quad is two triangles emitted as [b, b+1, b+2, b, b+2, b+3], so the
        // four distinct corner vertices are indices[i], [i+1], [i+2], [i+5].
        let mut i = 0;
        while i < indices.len() {
            let corners = [
                vertices[indices[i] as usize],
                vertices[indices[i + 1] as usize],
                vertices[indices[i + 2] as usize],
                vertices[indices[i + 5] as usize],
            ];
            let normal = corners[0].normal;
            let axis = if normal[0].abs() > 0.5 {
                0usize
            } else if normal[1].abs() > 0.5 {
                1
            } else {
                2
            };
            let sign: i8 = if normal[axis] > 0.0 { 1 } else { -1 };
            let (a, b) = match axis {
                0 => (1usize, 2usize),
                1 => (0usize, 2usize),
                _ => (0usize, 1usize),
            };
            let plane = round_plane(to_index(corners[0].position)[axis]);
            // The quad's span in the two in-plane axes (integer index planes).
            let mut a_lo = i64::MAX;
            let mut a_hi = i64::MIN;
            let mut b_lo = i64::MAX;
            let mut b_hi = i64::MIN;
            for corner in &corners {
                let idx = to_index(corner.position);
                let av = round_plane(idx[a]);
                let bv = round_plane(idx[b]);
                a_lo = a_lo.min(av);
                a_hi = a_hi.max(av);
                b_lo = b_lo.min(bv);
                b_hi = b_hi.max(bv);
            }
            for ca in a_lo..a_hi {
                for cb in b_lo..b_hi {
                    faces.insert(UnitFace {
                        axis: axis as u8,
                        sign,
                        plane,
                        cell: [ca, cb],
                    });
                }
            }
            i += 6;
        }
        faces
    }

    /// The world offset (`min_world - 0.5` per axis) the mesher anchors a grid's
    /// vertices on — subtract it from a mesh vertex to get the integer global-index
    /// frame `genuine_exposed_faces` uses.
    fn grid_world_offset(grid: &VoxelGrid) -> [f32; 3] {
        let mut min_world = [f32::INFINITY; 3];
        for v in &grid.occupied {
            let position = v.world_position();
            for (axis, m) in min_world.iter_mut().enumerate() {
                *m = m.min(position[axis]);
            }
        }
        [min_world[0] - 0.5, min_world[1] - 0.5, min_world[2] - 0.5]
    }

    /// Bucket a whole grid into per-chunk sub-grids exactly as the renderer's `new`
    /// wrapper does (`floor(world / chunk_extent)`), so the per-chunk mesher sees the
    /// SAME partition the live path does.
    fn bucket_for_test(grid: &VoxelGrid, voxels_per_block: u32) -> Vec<([i32; 3], VoxelGrid)> {
        super::bucket_grid_into_chunk_grids(grid, voxels_per_block)
    }

    /// The set of GENUINELY-exposed unit faces of an occupancy set: a `(voxel,
    /// direction)` whose neighbour cell is air. This is the VISIBLE silhouette — the
    /// surface that survives back-face culling + depth testing. The cuboid mesher's
    /// `face_is_exposed` emits a whole MERGED box face when ANY cell behind it is
    /// air, so it OVER-DRAWS the sub-faces backed by solid; those over-draw quads are
    /// always either back-face-culled or depth-occluded by the solid they are buried
    /// in, so they never reach a pixel. The genuinely-exposed set is therefore the
    /// invariant that determines the rendered image — and the structural parity claim:
    /// it must be IDENTICAL for whole-region and per-chunk meshing. We derive it
    /// straight from the occupancy (the ground truth) and also use it to filter an
    /// emitted mesh's unit faces down to its visible subset.
    fn genuine_exposed_faces(
        occupied: &std::collections::HashSet<[i64; 3]>,
    ) -> std::collections::HashSet<UnitFace> {
        let dirs: [(usize, i8, [i64; 3]); 6] = [
            (0, 1, [1, 0, 0]),
            (0, -1, [-1, 0, 0]),
            (1, 1, [0, 1, 0]),
            (1, -1, [0, -1, 0]),
            (2, 1, [0, 0, 1]),
            (2, -1, [0, 0, -1]),
        ];
        let mut faces = std::collections::HashSet::new();
        for &v in occupied {
            for (axis, sign, delta) in dirs {
                let neighbor = [v[0] + delta[0], v[1] + delta[1], v[2] + delta[2]];
                if occupied.contains(&neighbor) {
                    continue; // backed by solid → interior, not visible
                }
                // The face plane on `axis`: for +sign it's the voxel's far plane
                // (v[axis] + 1), for -sign the near plane (v[axis]).
                let plane = if sign > 0 { v[axis] + 1 } else { v[axis] };
                let (a, b) = match axis {
                    0 => (1usize, 2usize),
                    1 => (0usize, 2usize),
                    _ => (0usize, 1usize),
                };
                faces.insert(UnitFace {
                    axis: axis as u8,
                    sign,
                    plane,
                    cell: [v[a], v[b]],
                });
            }
        }
        faces
    }

    /// Filter an emitted mesh's unit faces down to the VISIBLE subset (those whose
    /// `(plane, cell, axis, sign)` is a genuinely-exposed face), discarding the
    /// over-draw quads `face_is_exposed` emits for partially-exposed merged boxes.
    fn visible_unit_faces(
        vertices: &[CuboidVertex],
        indices: &[u32],
        world_offset: [f32; 3],
        genuine: &std::collections::HashSet<UnitFace>,
    ) -> std::collections::HashSet<UnitFace> {
        unit_faces_in_index_frame(vertices, indices, world_offset)
            .into_iter()
            .filter(|f| genuine.contains(f))
            .collect()
    }

    /// Absolute integer occupancy (global indices `round(world - min_world)`) of a
    /// grid — the same frame the cuboid mesher's vertices live in, so a `UnitFace`
    /// derived from occupancy and one derived from a mesh vertex compare directly.
    fn occupancy_indices(grid: &VoxelGrid) -> std::collections::HashSet<[i64; 3]> {
        let mut min_world = [f32::INFINITY; 3];
        for v in &grid.occupied {
            let position = v.world_position();
            for (axis, m) in min_world.iter_mut().enumerate() {
                *m = m.min(position[axis]);
            }
        }
        let mut set = std::collections::HashSet::new();
        for v in &grid.occupied {
            let position = v.world_position();
            set.insert([
                (position[0] - min_world[0]).round() as i64,
                (position[1] - min_world[1]).round() as i64,
                (position[2] - min_world[2]).round() as i64,
            ]);
        }
        set
    }

    /// Map a wholesale/filtered mesh build to `coord -> (vertex bytes, indices)` — the
    /// per-chunk GPU buffer set proxy (the renderer uploads exactly these bytes), so a
    /// byte-equal map == a byte-equal buffer set.
    fn mesh_map(
        meshes: &[CuboidChunkMesh],
    ) -> std::collections::HashMap<[i32; 3], (Vec<u8>, Vec<u32>)> {
        meshes
            .iter()
            .map(|m| {
                (
                    m.coord,
                    (bytemuck::cast_slice::<_, u8>(&m.vertices).to_vec(), m.indices.clone()),
                )
            })
            .collect()
    }

    /// Issue #40: an INCREMENTAL rebuild — re-mesh only the apron-dilated dirty subset,
    /// evict vacated chunks, keep every other chunk's buffer — produces a per-chunk
    /// mesh (buffer) set BYTE-IDENTICAL to a wholesale rebuild, for every edit kind
    /// INCLUDING edits at a chunk SEAM (where the 1-voxel apron makes a neighbour's
    /// boundary faces depend on the edited chunk). This is the cuboid analogue of the
    /// deleted instanced `incremental_rebuild_equals_full_rebuild_for_every_edit_kind`
    /// and the real proof that consuming `cuboid_incremental_plan` is output-preserving.
    ///
    /// Both scenes pin the SAME global bounds with anchor voxels at the extremes (so
    /// `world_offset` is identical — modelling the live `incremental_ok` precondition
    /// that the floating origin did NOT shift; a shift forces a wholesale fall-back).
    /// Edits touch only interior / seam voxels. `evicted_dirty` is set to exactly the
    /// chunks that were resident in A AND changed in B — faithfully modelling
    /// `Store::invalidate_aabb`'s evicted set — so the plan's 26-neighbour dilation is
    /// what must catch any stale neighbour; if the dilation were wrong, a seam edit fails.
    #[test]
    fn incremental_cuboid_rebuild_equals_wholesale() {
        // vpb 1 → chunk extent = CHUNK_BLOCKS(4) voxels, so a 12³ grid spans 3 chunks
        // per axis and interior/seam edits exercise real apron seams.
        let dims = [12u32, 12, 12];
        // Anchor frame: the 8 corners, present in EVERY scene → global bounds pinned.
        let anchors: Vec<[u32; 3]> = {
            let e = [0u32, 11];
            let mut cs = Vec::new();
            for &x in &e {
                for &y in &e {
                    for &z in &e {
                        cs.push([x, y, z]);
                    }
                }
            }
            cs
        };
        // (scene_a interior, scene_b interior) edit pairs. x=4/x=8 are chunk seams
        // (chunk extent 4), so an edit at x=3↔4 changes faces in TWO chunks.
        let scene_a_interior: Vec<[u32; 3]> =
            vec![[3, 3, 3], [4, 3, 3], [7, 7, 7], [8, 7, 7], [5, 9, 2]];
        let edits: Vec<(&str, Vec<[u32; 3]>)> = vec![
            // Add an interior voxel (no seam).
            ("add interior", {
                let mut v = scene_a_interior.clone();
                v.push([6, 6, 6]);
                v
            }),
            // Remove a seam voxel → its cross-seam neighbour [3,3,3] re-exposes a face.
            ("remove seam voxel", {
                let mut v = scene_a_interior.clone();
                v.retain(|c| *c != [4, 3, 3]);
                v
            }),
            // Add a voxel straddling a seam next to an existing one → neighbour chunk
            // boundary faces get culled.
            ("add seam voxel", {
                let mut v = scene_a_interior.clone();
                v.push([4, 7, 7]); // abuts [3-or-8?]; sits in chunk x=1 next to [8,7,7] region edits
                v.push([3, 7, 7]);
                v
            }),
            // Move a voxel across a seam (remove one side, add the other).
            ("move across seam", {
                let mut v = scene_a_interior.clone();
                v.retain(|c| *c != [8, 7, 7]);
                v.push([9, 7, 7]);
                v
            }),
        ];

        for (label, interior_b) in &edits {
            let mut cells_a = anchors.clone();
            cells_a.extend(scene_a_interior.iter().copied());
            let mut cells_b = anchors.clone();
            cells_b.extend(interior_b.iter().copied());

            let grid_a = grid_from_indices(dims, &cells_a, 0);
            let grid_b = grid_from_indices(dims, &cells_b, 0);
            let buckets_a = bucket_for_test(&grid_a, 1);
            let buckets_b = bucket_for_test(&grid_b, 1);
            let refs_a: Vec<([i32; 3], &VoxelGrid)> =
                buckets_a.iter().map(|(c, g)| (*c, g)).collect();
            let refs_b: Vec<([i32; 3], &VoxelGrid)> =
                buckets_b.iter().map(|(c, g)| (*c, g)).collect();

            // Wholesale builds (the ground truth) for A (prior state) and B (target).
            let wholesale_a = build_chunk_meshes_with_apron(&refs_a, dims, LayerBand::FULL);
            let wholesale_b = build_chunk_meshes_with_apron(&refs_b, dims, LayerBand::FULL);

            // Per-chunk occupancy index sets, to derive the faithful dirty set.
            let occ = |buckets: &[([i32; 3], VoxelGrid)]| {
                let mut m: std::collections::HashMap<[i32; 3], std::collections::HashSet<[i64; 3]>> =
                    std::collections::HashMap::new();
                for (coord, g) in buckets {
                    m.insert(*coord, occupancy_indices(g));
                }
                m
            };
            let occ_a = occ(&buckets_a);
            let occ_b = occ(&buckets_b);

            // resident = A's covering coords (the renderer's source_chunk_grids coords).
            let resident: Vec<[i32; 3]> = buckets_a.iter().map(|(c, _)| *c).collect();
            // occupied_b = B's non-empty covering coords.
            let occupied_b: Vec<[i32; 3]> = buckets_b
                .iter()
                .filter(|(_, g)| !g.occupied.is_empty())
                .map(|(c, _)| *c)
                .collect();
            // evicted_dirty = chunks resident in A whose occupancy CHANGED in B (exactly
            // what invalidate_aabb evicts: resident ∩ edit-region). Newly-appeared
            // chunks (absent from A) are caught by the plan's own new-appeared term.
            let evicted_dirty: Vec<[i32; 3]> = resident
                .iter()
                .copied()
                .filter(|coord| occ_a.get(coord) != occ_b.get(coord))
                .collect();

            let plan = cuboid_incremental_plan(&resident, &evicted_dirty, &occupied_b);

            // Genuinely incremental: at least the far corners are NOT re-meshed.
            assert!(
                plan.rebuild.len() < occupied_b.len(),
                "[{label}] expected a partial rebuild, got {} of {} chunks",
                plan.rebuild.len(),
                occupied_b.len()
            );

            // Apply the plan to A's mesh map → the incremental buffer set.
            let rebuild_set: std::collections::HashSet<[i32; 3]> =
                plan.rebuild.iter().copied().collect();
            let rebuilt =
                build_chunk_meshes_with_apron_filtered(&refs_b, Some(&rebuild_set), dims, LayerBand::FULL);

            let mut result = mesh_map(&wholesale_a);
            for coord in &plan.evict {
                result.remove(coord);
            }
            for coord in &plan.rebuild {
                result.remove(coord); // a rebuild coord meshing to empty drops out here
            }
            for (coord, entry) in mesh_map(&rebuilt) {
                result.insert(coord, entry);
            }

            let target = mesh_map(&wholesale_b);
            assert_eq!(
                result, target,
                "[{label}] incremental buffer set must byte-equal the wholesale rebuild"
            );
        }
    }

    /// The CORE structural guarantee (the analogue of the decomposition round-trip):
    /// the per-chunk-with-apron VISIBLE exposed-face SET equals the whole-region
    /// mesher's — and both equal the ground-truth genuinely-exposed set derived
    /// straight from occupancy — for many shapes/sizes INCLUDING shapes spanning
    /// multiple chunks. This is what guarantees the RENDERED IMAGE is unchanged,
    /// independent of the goldens: the apron makes seam faces between two solid
    /// chunks culled (no extra visible interior quads), and co-planar seam-spanning
    /// faces split into abutting quads covering the IDENTICAL visible unit-face set.
    ///
    /// "Visible" = the subset of emitted unit faces backed by air. The mesher emits a
    /// whole MERGED box face when ANY cell behind it is air, over-drawing the
    /// sub-faces backed by solid; those over-draw quads are always back-face-culled or
    /// depth-occluded by the solid they are buried in, so they never reach a pixel and
    /// their (merge-order-dependent) count is NOT a rendering invariant. The visible
    /// set IS. We assert the visible set of BOTH paths equals the ground truth, AND
    /// that every genuine face is actually emitted by each path (no real hole).
    #[test]
    fn per_chunk_apron_exposed_face_set_equals_whole_region() {
        use crate::voxel::{SdfShape, ShapeKind, VoxelProducer};

        let mut multi_chunk_seen = false;
        // Densities chosen so shapes span MULTIPLE chunks (chunk = CHUNK_BLOCKS=4
        // blocks × density voxels per axis = 32 voxels at density 8). An 8-block axis
        // = 64 voxels = 2 chunks/axis, so the apron is exercised at real seams.
        for &kind in &[
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Torus,
            ShapeKind::Box,
            ShapeKind::Tube,
        ] {
            for &(size, density) in &[
                ([5u32, 1, 5], 8u32), // the default disc (odd X/Z → recentred cloud)
                ([3, 3, 3], 8),
                ([8, 2, 8], 8), // 64×16×64 voxels → 2 chunks/axis in X/Z (multi-chunk)
                ([5, 3, 7], 8),
            ] {
                let shape = SdfShape::from_blocks(kind, size, 1, density);
                let dims = shape.grid_dimensions(density);
                let mut grid = VoxelGrid::new(dims);
                shape.resolve(&mut grid, density);
                if grid.occupied.is_empty() {
                    continue;
                }

                // Ground-truth genuinely-exposed faces, straight from occupancy.
                let occupancy = occupancy_indices(&grid);
                let genuine = genuine_exposed_faces(&occupancy);
                let world_offset = grid_world_offset(&grid);

                // Whole-region reference mesh → its VISIBLE face subset.
                let whole = build_cuboid_mesh(&grid, density);
                let whole_visible =
                    visible_unit_faces(&whole.vertices, &whole.indices, world_offset, &genuine);

                // Per-chunk-with-apron mesh → its VISIBLE face subset (union over chunks).
                let buckets = bucket_for_test(&grid, density);
                let chunk_refs: Vec<([i32; 3], &VoxelGrid)> =
                    buckets.iter().map(|(c, g)| (*c, g)).collect();
                if buckets.len() > 1 {
                    multi_chunk_seen = true;
                }
                let chunk_meshes =
                    build_chunk_meshes_with_apron(&chunk_refs, dims, LayerBand::FULL);
                let mut per_chunk_visible = std::collections::HashSet::new();
                for mesh in &chunk_meshes {
                    per_chunk_visible.extend(visible_unit_faces(
                        &mesh.vertices,
                        &mesh.indices,
                        world_offset,
                        &genuine,
                    ));
                }

                // Both paths must emit EXACTLY the ground-truth visible surface — no
                // missing genuine face (a hole), no spurious visible face (a stray
                // seam quad). This is the rendering-determining invariant.
                assert_eq!(
                    whole_visible, genuine,
                    "{kind:?} {size:?} density={density}: whole-region visible faces != ground truth"
                );
                assert_eq!(
                    per_chunk_visible, genuine,
                    "{kind:?} {size:?} density={density}: per-chunk apron visible faces != \
                     ground truth ({} per-chunk vs {} genuine)",
                    per_chunk_visible.len(),
                    genuine.len()
                );
            }
        }
        assert!(
            multi_chunk_seen,
            "no test case actually spanned multiple chunks — apron never exercised at a seam"
        );
    }

    /// A solid slab spanning a chunk seam must NOT emit interior seam faces — the
    /// apron culls them. Two abutting solid 32-voxel chunks (density 8, 4 blocks ×
    /// 8 = 32 voxels per chunk axis) form an 8×4×4-block solid box across the X
    /// seam; the whole-region mesh is one box (6 faces' worth of unit faces), and
    /// the per-chunk mesh (two boxes) must produce the IDENTICAL unit-face set with
    /// no faces on the interior seam plane.
    #[test]
    fn solid_slab_across_chunk_seam_has_no_interior_faces() {
        let density = 8u32;
        let chunk_voxels = crate::core_geom::CHUNK_BLOCKS * density; // 32
        let nx = chunk_voxels * 2; // span two chunks in X
        let ny = density; // 8
        let nz = density; // 8
        let dims = [nx, ny, nz];
        let half = [nx as f32 / 2.0, ny as f32 / 2.0, nz as f32 / 2.0];
        let mut grid = VoxelGrid::new(dims);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    grid.occupied.push(crate::voxel::Voxel {
                        local_index: [
                            (i as f32 + 0.5 - half[0]).floor() as i32,
                            (j as f32 + 0.5 - half[1]).floor() as i32,
                            (k as f32 + 0.5 - half[2]).floor() as i32,
                        ],
                        block_local_coord: [0, 0, 0],
                        block_id: crate::core_geom::BlockId(0),
                        attrs: crate::core_geom::BlockAttrs::DEFAULT,
                        grid_overlay: false,
                    });
                }
            }
        }

        let buckets = bucket_for_test(&grid, density);
        assert!(
            buckets.len() >= 2,
            "the solid slab must span at least two chunks in X (got {})",
            buckets.len()
        );

        let occupancy = occupancy_indices(&grid);
        let genuine = genuine_exposed_faces(&occupancy);
        let world_offset = grid_world_offset(&grid);

        let whole = build_cuboid_mesh(&grid, density);
        let whole_visible =
            visible_unit_faces(&whole.vertices, &whole.indices, world_offset, &genuine);

        let chunk_refs: Vec<([i32; 3], &VoxelGrid)> =
            buckets.iter().map(|(c, g)| (*c, g)).collect();
        let chunk_meshes = build_chunk_meshes_with_apron(&chunk_refs, dims, LayerBand::FULL);
        let mut per_chunk_visible = std::collections::HashSet::new();
        for mesh in &chunk_meshes {
            per_chunk_visible.extend(visible_unit_faces(
                &mesh.vertices,
                &mesh.indices,
                world_offset,
                &genuine,
            ));
        }

        // The slab surface is exactly the box's 6 sides: 2*(nx*ny + nx*nz + ny*nz)
        // unit faces. No interior seam plane faces.
        let expected = 2 * (nx * ny + nx * nz + ny * nz) as usize;
        assert_eq!(
            genuine.len(),
            expected,
            "solid box surface should be {expected} unit faces"
        );
        assert_eq!(
            whole_visible, genuine,
            "solid cross-seam slab: whole-region visible faces != ground truth"
        );
        assert_eq!(
            per_chunk_visible, genuine,
            "solid cross-seam slab: per-chunk apron visible faces != ground truth (interior \
             seam faces leaked or a side is missing)"
        );
    }

    /// The per-chunk band clip must match the whole-region band clip's VISIBLE
    /// exposed-face set (real caps at the band edges, per chunk). A torus clipped to a
    /// sub-band that falls INSIDE the chunks must synthesise the cap faces identically
    /// in both paths. Ground truth = the genuinely-exposed faces of the BAND-MASKED
    /// occupancy (cells outside `[band_min, band_max]` removed, so the band edges are
    /// real air boundaries → cap faces).
    #[test]
    fn per_chunk_band_clip_face_set_equals_whole_region() {
        use crate::voxel::{SdfShape, ShapeKind, VoxelProducer};
        let voxels_per_block = 8;
        let shape = SdfShape::from_blocks(ShapeKind::Torus, [8, 2, 8], 1, voxels_per_block);
        let dims = shape.grid_dimensions(voxels_per_block);
        let mut grid = VoxelGrid::new(dims);
        shape.resolve(&mut grid, voxels_per_block);
        assert!(!grid.occupied.is_empty());

        // Z-up: the band is a Z-layer range. The torus [8,2,8] is 128 voxels tall in
        // Z; a band straddling the vertical middle keeps a real slice of the tube.
        let band = LayerBand {
            band_min: 56,
            band_max: 71,
            onion_depth: 0,
        };

        // Ground truth: genuinely-exposed faces of the BAND-MASKED occupancy. The
        // band maps an absolute layer to a global index Z by `gz + base_layer` where
        // base_layer = floor(min_world.z + half_z) and the global index uses
        // `round(world - min_world)`. We mask occupancy to `base_layer + gz ∈ band`.
        let mut min_world = [f32::INFINITY; 3];
        for v in &grid.occupied {
            let position = v.world_position();
            for (axis, m) in min_world.iter_mut().enumerate() {
                *m = m.min(position[axis]);
            }
        }
        let half_z = (dims[2] / 2) as f32; // corner-anchoring: floored half
        let base_layer = (min_world[2] + half_z).floor() as i64;
        let occupancy: std::collections::HashSet<[i64; 3]> = grid
            .occupied
            .iter()
            .map(|v| {
                let position = v.world_position();
                [
                    (position[0] - min_world[0]).round() as i64,
                    (position[1] - min_world[1]).round() as i64,
                    (position[2] - min_world[2]).round() as i64,
                ]
            })
            .filter(|idx| {
                let layer = base_layer + idx[2];
                layer >= band.band_min as i64 && layer <= band.band_max as i64
            })
            .collect();
        assert!(!occupancy.is_empty(), "band must keep some voxels");
        let genuine = genuine_exposed_faces(&occupancy);
        let world_offset = grid_world_offset(&grid);

        let whole = build_cuboid_mesh_banded(&grid, 8, band);
        let whole_visible =
            visible_unit_faces(&whole.vertices, &whole.indices, world_offset, &genuine);

        let buckets = bucket_for_test(&grid, 8);
        assert!(buckets.len() > 1, "torus must span multiple chunks");
        let chunk_refs: Vec<([i32; 3], &VoxelGrid)> =
            buckets.iter().map(|(c, g)| (*c, g)).collect();
        let chunk_meshes = build_chunk_meshes_with_apron(&chunk_refs, dims, band);
        let mut per_chunk_visible = std::collections::HashSet::new();
        for mesh in &chunk_meshes {
            per_chunk_visible.extend(visible_unit_faces(
                &mesh.vertices,
                &mesh.indices,
                world_offset,
                &genuine,
            ));
        }

        assert_eq!(
            whole_visible, genuine,
            "banded torus: whole-region visible faces != band-masked ground truth"
        );
        assert_eq!(
            per_chunk_visible, genuine,
            "banded torus: per-chunk apron visible faces != band-masked ground truth"
        );
    }

    // ---- ADR 0010 E3 — two-layer mesher exposed-face parity (#50) ----

    use crate::core_geom::MaterialChoice as MC;
    use crate::scene::{DefId, Node, NodeContent, NodeTransform, Scene};
    use crate::two_layer_store::TwoLayerStore;
    use crate::voxel::{SdfShape as TwoLayerSdf, ShapeKind as TwoLayerShape};

    /// Every unit face a mesh EMITS that ACTUALLY RENDERS — i.e. whose cell on the NORMAL
    /// (front) side is AIR per the dense occupancy. A face buried in solid (front solid) is
    /// back-face-culled / depth-occluded and never reaches a pixel, so the rendered image is
    /// exactly this set. Unlike [`visible_unit_faces`] (which pre-filters to `genuine`), this
    /// does NOT discard non-genuine faces — so it CATCHES a spurious face that renders
    /// (front air, but no solid behind it), the over-draw bug `visible_unit_faces` hides.
    fn renderable_unit_faces(
        vertices: &[CuboidVertex],
        indices: &[u32],
        world_offset: [f32; 3],
        occupied: &std::collections::HashSet<[i64; 3]>,
    ) -> std::collections::HashSet<UnitFace> {
        unit_faces_in_index_frame(vertices, indices, world_offset)
            .into_iter()
            .filter(|f| {
                // The cell on the NORMAL (front) side of the face. For +sign the plane is the
                // voxel's far edge (front cell index = plane), for -sign the near edge (front
                // cell index = plane - 1). The two in-plane axes carry `f.cell`.
                let (axis, sign) = (f.axis as usize, f.sign);
                let (a, b) = match axis {
                    0 => (1usize, 2usize),
                    1 => (0usize, 2usize),
                    _ => (0usize, 1usize),
                };
                let mut front = [0i64; 3];
                front[axis] = if sign > 0 { f.plane } else { f.plane - 1 };
                front[a] = f.cell[0];
                front[b] = f.cell[1];
                // Renders iff the front cell is AIR (nothing occluding it).
                !occupied.contains(&front)
            })
            .collect()
    }

    /// The two-layer mesher's RENDERABLE exposed-face set (every emitted face whose front cell
    /// is air per the dense occupancy), in the recentred-index frame. Unions over every chunk.
    fn two_layer_renderable_faces(
        scene: &Scene,
        density: u32,
        world_offset: [f32; 3],
        recentre: RecentreVoxels,
        grid_dimensions: [u32; 3],
        occupied: &std::collections::HashSet<[i64; 3]>,
    ) -> std::collections::HashSet<UnitFace> {
        let store = TwoLayerStore::enabled();
        let chunks = store.build_covering_chunks(scene, density, 0);
        let meshes =
            build_two_layer_chunk_meshes(&chunks, grid_dimensions, recentre, density, LayerBand::FULL);
        let mut renderable = std::collections::HashSet::new();
        for mesh in &meshes {
            renderable.extend(renderable_unit_faces(
                &mesh.vertices,
                &mesh.indices,
                world_offset,
                occupied,
            ));
            renderable.extend(renderable_unit_faces(
                &mesh.vertices,
                &mesh.indices_overlay,
                world_offset,
                occupied,
            ));
        }
        renderable
    }

    /// Assert the two-layer mesher's VISIBLE exposed-face set equals the dense path's —
    /// and both equal the ground-truth genuinely-exposed set derived straight from the
    /// dense occupancy — for one scene. Mirrors
    /// [`per_chunk_apron_exposed_face_set_equals_whole_region`] (the apron parity), but for
    /// the two-layer (coarse one-box + microblock cuboid + seam-flag) mesher (ADR 0010 E3).
    fn assert_two_layer_face_parity(scene: &Scene, density: u32, label: &str) {
        // The dense assembled grid: ground truth occupancy + the reference whole-grid mesh.
        let dense = scene.resolve_region(scene.full_extent_blocks(density), density, 0);
        assert!(!dense.occupied.is_empty(), "[{label}] scene resolved empty");
        let occupancy = occupancy_indices(&dense);
        let genuine = genuine_exposed_faces(&occupancy);
        let world_offset = grid_world_offset(&dense);

        // Dense reference mesh → its VISIBLE face subset (the existing parity reference).
        let whole = build_cuboid_mesh(&dense, density);
        let mut whole_visible =
            visible_unit_faces(&whole.vertices, &whole.indices, world_offset, &genuine);
        whole_visible.extend(visible_unit_faces(
            &whole.vertices,
            &whole.indices_overlay,
            world_offset,
            &genuine,
        ));
        assert_eq!(
            whole_visible, genuine,
            "[{label}] dense reference visible faces != ground truth"
        );

        // Two-layer mesher → its RENDERABLE face set (every emitted face whose front cell is
        // air — exactly what reaches a pixel). This must equal the ground-truth genuine
        // surface: a SUPERSET would be a spurious rendered face (over-draw that isn't
        // occluded — the boundary-seam over-emit bug), a SUBSET a hole. Strictly stronger
        // than the visible-subset check (which can't see over-emission).
        let two_layer_renderable = two_layer_renderable_faces(
            scene,
            density,
            world_offset,
            // The dense-oracle grid carries its recentre as a raw triple (raw by rule); mint
            // the frame newtype at this test boundary.
            RecentreVoxels::new(dense.recentre_voxels),
            dense.dimensions,
            &occupancy,
        );
        assert_eq!(
            two_layer_renderable, genuine,
            "[{label}] two-layer mesher RENDERABLE faces != ground truth ({} renderable vs {} \
             genuine) — a hole (missing surface) or a spurious rendered seam face (over-draw)",
            two_layer_renderable.len(),
            genuine.len()
        );
    }

    fn two_layer_tool(
        kind: TwoLayerShape,
        size: [u32; 3],
        offset: [i64; 3],
        material: MC,
        density: u32,
    ) -> Node {
        let shape = TwoLayerSdf::from_blocks(kind, size, 1, density);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset, density);
        node
    }

    /// THE E3 GATE (parity (b)): the two-layer mesher's exposed-face SET equals the dense
    /// mesher's across the gated scene matrix — SDF shapes (incl. flat/odd), a demo scene,
    /// a demo village, a LARGE solid (the one-box coarse path must leave no interior seam),
    /// AND an overlapping multi-material scene (the E2 carry-over: overlaps must render
    /// identically). Multi-chunk shapes exercise the inter-chunk seam-flag culling.
    #[test]
    fn two_layer_mesher_exposed_face_set_equals_dense() {
        let density = 16u32;

        // SDF shapes including flat/odd sizes (multi-chunk at d16: an 8-block axis = 128
        // voxels = 2 chunks/axis).
        for kind in [
            TwoLayerShape::Sphere,
            TwoLayerShape::Cylinder,
            TwoLayerShape::Torus,
            TwoLayerShape::Box,
            TwoLayerShape::Tube,
        ] {
            for size in [[5u32, 5, 5], [5, 1, 5], [3, 1, 3], [8, 2, 8]] {
                let scene = Scene::from_nodes(vec![two_layer_tool(
                    kind,
                    size,
                    [0, 0, 0],
                    MC::Stone,
                    density,
                )]);
                assert_two_layer_face_parity(&scene, density, &format!("{kind:?} {size:?}"));
            }
        }

        // Demo scene: three disjoint Tools (the shot --demo-scene shape set).
        let demo = Scene::from_nodes(vec![
            two_layer_tool(TwoLayerShape::Sphere, [5, 5, 5], [0, 0, 0], MC::Stone, density),
            two_layer_tool(TwoLayerShape::Box, [5, 5, 5], [8, 0, 0], MC::Wood, density),
            two_layer_tool(TwoLayerShape::Torus, [5, 5, 5], [0, 0, 6], MC::Plain, density),
        ]);
        assert_two_layer_face_parity(&demo, density, "demo-scene");

        // Demo village: an instanced definition placed four times (the shot --demo-village).
        {
            let house = DefId(1);
            let mut village = Scene::from_nodes(vec![
                village_instance("House 1", house, [0, 0, 0], density),
                village_instance("House 2", house, [6, 0, 0], density),
                village_instance("House 3", house, [12, 0, 0], density),
                village_instance("House 4", house, [18, 0, 0], density),
            ]);
            village.add_definition(
                house,
                "House".to_string(),
                vec![
                    two_layer_tool(TwoLayerShape::Box, [2, 2, 2], [0, 0, 0], MC::Stone, density),
                    two_layer_tool(TwoLayerShape::Cylinder, [1, 2, 1], [0, 2, 0], MC::Wood, density),
                ],
            );
            assert_two_layer_face_parity(&village, density, "demo-village");
        }

        // A LARGE solid box: the one-box coarse path must leave no interior seam or hole.
        // 6 blocks @ d16 = 96 voxels/axis = 3 chunks/axis, so the interior chunks are fully
        // coarse-solid and the seam-flag culling spans every chunk boundary.
        let large = Scene::from_nodes(vec![two_layer_tool(
            TwoLayerShape::Box,
            [6, 6, 6],
            [0, 0, 0],
            MC::Stone,
            density,
        )]);
        assert_two_layer_face_parity(&large, density, "large-solid-box");

        // A SketchSolid REVOLVE (the shot --demo-sketch-revolve golden): a boundary-only
        // producer (no coarse-solid blocks — its profile fill is not a coarse box), so this
        // pins the microblock-cuboid + seam path for a non-SDF producer.
        {
            use crate::sketch::{PlaneAxis, RevolveAxis, Sketch, SketchPoint, SketchSolid};
            let block = density as i64;
            let r = |b: i64| b * block;
            let h = |b: i64| b * block;
            let profile = vec![
                SketchPoint::new(0, h(0)),
                SketchPoint::new(r(4), h(0)),
                SketchPoint::new(r(4), h(1)),
                SketchPoint::new(r(2), h(3)),
                SketchPoint::new(r(2), h(5)),
                SketchPoint::new(r(4), h(6)),
                SketchPoint::new(r(3), h(8)),
                SketchPoint::new(0, h(8)),
            ];
            let producer =
                SketchSolid::revolve(Sketch::new(PlaneAxis::X, profile), RevolveAxis::InPlane1, 360);
            let revolve = Scene::from_nodes(vec![Node::new(
                "Vase",
                NodeContent::SketchTool {
                    producer,
                    material: MC::Stone,
                },
            )]);
            assert_two_layer_face_parity(&revolve, density, "sketch-revolve");
        }

        // OVERLAPPING multi-material (E2 carry-over): two boxes overlapping with different
        // materials. The overlap region resolves last-writer-wins (document order) and must
        // render the IDENTICAL exposed-face set as the dense path.
        let overlap = Scene::from_nodes(vec![
            two_layer_tool(TwoLayerShape::Box, [3, 3, 3], [0, 0, 0], MC::Stone, density),
            two_layer_tool(TwoLayerShape::Box, [3, 3, 3], [1, 1, 0], MC::Wood, density),
        ]);
        assert_two_layer_face_parity(&overlap, density, "overlap-multi-material");
    }

    /// The RENDERABLE exposed-face set of a supplied two-layer chunk set (every emitted face
    /// whose front cell is air per `occupied`), unioned over every chunk. Shared by the
    /// incremental-edit mesh parity test below so it can mesh a chunk set assembled through the
    /// resident cache (post-edit) rather than a fresh `build_covering_chunks`.
    fn two_layer_chunk_set_renderable_faces(
        chunks: &[([i32; 3], Arc<crate::two_layer_store::TwoLayerChunk>)],
        density: u32,
        world_offset: [f32; 3],
        recentre: RecentreVoxels,
        grid_dimensions: [u32; 3],
        occupied: &std::collections::HashSet<[i64; 3]>,
    ) -> std::collections::HashSet<UnitFace> {
        let meshes =
            build_two_layer_chunk_meshes(chunks, grid_dimensions, recentre, density, LayerBand::FULL);
        let mut renderable = std::collections::HashSet::new();
        for mesh in &meshes {
            renderable.extend(renderable_unit_faces(
                &mesh.vertices,
                &mesh.indices,
                world_offset,
                occupied,
            ));
            renderable.extend(renderable_unit_faces(
                &mesh.vertices,
                &mesh.indices_overlay,
                world_offset,
                occupied,
            ));
        }
        renderable
    }

    /// **ADR 0010 #54 GATE (rendered parity):** a scene edited INCREMENTALLY on the two-layer
    /// path (build scene A into a [`TwoLayerResidentCache`], invalidate the edit's dirty AABB
    /// chunks, re-derive only those) meshes to the SAME renderable exposed-face set as (a) a
    /// FULL from-scratch two-layer rebuild of the edited scene B and (b) the dense ground truth.
    /// This is the mesh-face-set assertion for an edited scene the acceptance criteria call for:
    /// an incremental edit is pixel-identical to a full rebuild and to the dense path.
    ///
    /// Covers move / recolor / resize / add / remove — each keeps the mesher's face set
    /// identical to a full rebuild, proving the resident cache's chunk-granular invalidation
    /// leaves no stale geometry and misses no fresh surface through the mesher.
    #[test]
    fn incremental_two_layer_edit_meshes_identically_to_full_rebuild() {
        use crate::two_layer_store::TwoLayerResidentCache;
        let density = 16u32;

        // A base scene with a wide sphere (many chunks) plus an interior subject box, so an
        // edit touches a strict subset while much of the chunk set stays resident.
        let scene_a = Scene::from_nodes(vec![
            two_layer_tool(TwoLayerShape::Sphere, [6, 6, 6], [0, 0, 0], MC::Stone, density),
            two_layer_tool(TwoLayerShape::Box, [3, 3, 3], [10, 0, 0], MC::Wood, density),
        ]);

        let recolor = {
            let mut b = scene_a.clone();
            if let NodeContent::Tool { material, .. } = &mut b.root_node_mut(1).content {
                *material = MC::Stone;
            }
            ("recolor", b)
        };
        let resize = {
            let mut b = scene_a.clone();
            let replacement =
                two_layer_tool(TwoLayerShape::Box, [2, 2, 2], [10, 0, 0], MC::Wood, density);
            let slot = b.root_node_mut(1);
            slot.content = replacement.content;
            slot.transform = replacement.transform;
            ("resize", b)
        };
        let move_edit = {
            let mut b = scene_a.clone();
            b.root_node_mut(1).transform = NodeTransform::from_blocks([13, 0, 0], density);
            ("move", b)
        };
        let add_edit = {
            let mut b = scene_a.clone();
            b.add_node(two_layer_tool(
                TwoLayerShape::Box,
                [2, 2, 2],
                [16, 0, 0],
                MC::Plain,
                density,
            ));
            ("add", b)
        };
        let remove_edit = {
            let mut b = scene_a.clone();
            let subject = b.roots[1];
            b.remove_node(subject);
            ("remove", b)
        };

        for (label, scene_b) in [recolor, resize, move_edit, add_edit, remove_edit] {
            // Drive the incremental edit through the resident cache exactly as app_core would.
            let mut cache = TwoLayerResidentCache::enabled();
            let _ = cache.resident_two_layer_chunks(&scene_a, density, 0);
            let index_a = scene_a.build_leaf_spatial_index(density);
            let index_b = scene_b.build_leaf_spatial_index(density);
            match index_b.edit_aabb_since(&index_a) {
                Some(edit_aabb) => {
                    cache.invalidate_aabb(&edit_aabb, density);
                }
                None => cache.clear(),
            }
            let incremental_chunks: Vec<_> = cache
                .resident_two_layer_chunks(&scene_b, density, 0)
                .into_iter()
                .map(|(coord, chunk)| (coord, chunk.clone()))
                .collect();

            // The dense ground truth for scene B (the pixel reference).
            let dense = scene_b.resolve_region(scene_b.full_extent_blocks(density), density, 0);
            assert!(!dense.occupied.is_empty(), "[{label}] scene resolved empty");
            let occupancy = occupancy_indices(&dense);
            let genuine = genuine_exposed_faces(&occupancy);
            let world_offset = grid_world_offset(&dense);

            // The incrementally-edited chunk set's renderable face set == dense ground truth.
            let incremental_faces = two_layer_chunk_set_renderable_faces(
                &incremental_chunks,
                density,
                world_offset,
                RecentreVoxels::new(dense.recentre_voxels),
                dense.dimensions,
                &occupancy,
            );
            assert_eq!(
                incremental_faces, genuine,
                "[{label}] incrementally-edited two-layer mesh RENDERABLE faces != dense ground \
                 truth — a stale chunk (over-draw) or a missed fresh surface (hole)"
            );

            // And identical to a FULL from-scratch two-layer rebuild of scene B.
            let full_faces = two_layer_renderable_faces(
                &scene_b,
                density,
                world_offset,
                RecentreVoxels::new(dense.recentre_voxels),
                dense.dimensions,
                &occupancy,
            );
            assert_eq!(
                incremental_faces, full_faces,
                "[{label}] incremental two-layer mesh faces must equal a FULL two-layer rebuild"
            );
        }
    }

    /// One chunk's GPU-buffer proxy: `(vertex bytes, overlay-off indices, overlay-on indices)`.
    type ChunkBufferProxy = (Vec<u8>, Vec<u32>, Vec<u32>);

    /// Map a two-layer mesh build to `coord -> (vertex bytes, off-indices, overlay-indices)` —
    /// the per-chunk GPU buffer set proxy (the renderer uploads exactly these bytes via
    /// [`upload_chunk_meshes`], concatenating the two index runs), so a byte-equal map ==
    /// a byte-equal buffer set. Carries BOTH index runs (unlike the dense `mesh_map`) so the
    /// overlay split is part of the parity claim.
    fn two_layer_mesh_map(
        meshes: &[CuboidChunkMesh],
    ) -> std::collections::HashMap<[i32; 3], ChunkBufferProxy> {
        meshes
            .iter()
            .map(|m| {
                (
                    m.coord,
                    (
                        bytemuck::cast_slice::<_, u8>(&m.vertices).to_vec(),
                        m.indices.clone(),
                        m.indices_overlay.clone(),
                    ),
                )
            })
            .collect()
    }

    /// **THE ISSUE #55 GATE (byte-parity + only-dirty-remeshed).** Drive an edit through the
    /// [`TwoLayerResidentCache`] exactly as `AppCore::rebuild` does, then apply the
    /// [`cuboid_incremental_plan`] on the two-layer chunk source exactly as
    /// [`CuboidMeshRenderer::incremental_rebuild_from_two_layer_chunks`] does, and assert:
    ///
    /// 1. **Byte-parity** — the incrementally-updated per-chunk buffer set (kept-buffers ∪
    ///    freshly-meshed rebuild subset, minus evicted) is BYTE-IDENTICAL to a wholesale
    ///    two-layer re-mesh of the edited scene. This is the GPU-buffer analogue of the dense
    ///    [`incremental_cuboid_rebuild_equals_wholesale`], and of the face-set
    ///    [`incremental_two_layer_edit_meshes_identically_to_full_rebuild`].
    /// 2. **Only-dirty-remeshed (the perf proof)** — the re-meshed set is the plan's dirty +
    ///    26-neighbourhood-dilated rebuild set, STRICTLY smaller than the whole resident set
    ///    (quantified on a many-chunk scene). Without this the slice is unverified: it is what
    ///    proves per-edit mesh cost scales with the dirty set, not the scene size.
    #[test]
    fn incremental_two_layer_gpu_buffer_rebuild_equals_wholesale() {
        use crate::two_layer_store::TwoLayerResidentCache;
        let density = 16u32;

        // Two ANCHOR boxes at fixed extremes PLUS an interior subject box. The anchors are
        // present in EVERY scene, so the composite bounds — hence the recentre (floating origin)
        // — stay PINNED across each edit. That models the live guard: the two-layer GPU-buffer
        // incremental only runs when the recentre did NOT shift (a shift re-frames every kept
        // buffer's baked vertices → the shell falls back to wholesale, `app_core.rs`). With the
        // recentre pinned, the incremental path is genuinely exercised, and the subject box sits
        // far from the anchors so an edit dirties a strict subset while most chunks stay resident.
        let anchors = || {
            vec![
                two_layer_tool(TwoLayerShape::Box, [2, 2, 2], [-14, 0, 0], MC::Stone, density),
                two_layer_tool(TwoLayerShape::Box, [2, 2, 2], [14, 8, 6], MC::Stone, density),
            ]
        };
        // scene_a: anchors + subject box (index 2) at a fixed offset well inside the bounds.
        let scene_a = {
            let mut nodes = anchors();
            nodes.push(two_layer_tool(TwoLayerShape::Box, [3, 3, 3], [4, 2, 2], MC::Wood, density));
            Scene::from_nodes(nodes)
        };
        let subject = 2usize;

        let recolor = {
            let mut b = scene_a.clone();
            if let NodeContent::Tool { material, .. } = &mut b.root_node_mut(subject).content {
                *material = MC::Plain;
            }
            ("recolor", b)
        };
        let resize = {
            // Shrink the subject box; the anchors keep the bounds pinned so the recentre holds.
            let mut b = scene_a.clone();
            let replacement =
                two_layer_tool(TwoLayerShape::Box, [2, 2, 2], [4, 2, 2], MC::Wood, density);
            let slot = b.root_node_mut(subject);
            slot.content = replacement.content;
            slot.transform = replacement.transform;
            ("resize", b)
        };
        let move_edit = {
            // Move the subject WITHIN the anchored bounds (recentre unchanged).
            let mut b = scene_a.clone();
            b.root_node_mut(subject).transform = NodeTransform::from_blocks([6, 2, 2], density);
            ("move", b)
        };
        let remove_edit = {
            let mut b = scene_a.clone();
            let subject_id = b.roots[subject];
            b.remove_node(subject_id);
            ("remove", b)
        };

        for (label, scene_b) in [recolor, resize, move_edit, remove_edit] {
            // Build scene A's resident set + its owned two-layer chunk source (what the renderer
            // retains in `source_two_layer_chunks`).
            let mut cache = TwoLayerResidentCache::enabled();
            let chunks_a: Vec<([i32; 3], Arc<TwoLayerChunk>)> =
                cache.resident_two_layer_chunks(&scene_a, density, 0);
            let dims = scene_a.placed_region_dimensions(density);
            let recentre = scene_a.recentre_voxels_for_resolve(density);
            // The renderer's initial (wholesale) buffer set for A.
            let wholesale_a = build_two_layer_chunk_meshes(
                &chunks_a,
                dims,
                recentre,
                density,
                LayerBand::FULL,
            );

            // The edit: invalidate the dirty AABB (or clear), then re-derive the resident set.
            let index_a = scene_a.build_leaf_spatial_index(density);
            let index_b = scene_b.build_leaf_spatial_index(density);
            let evicted_dirty: Vec<[i32; 3]> = match index_b.edit_aabb_since(&index_a) {
                Some(edit_aabb) => cache.invalidate_aabb(&edit_aabb, density),
                None => {
                    cache.clear();
                    Vec::new()
                }
            };
            // These edits are all localisable (a single moved/edited leaf), so the AABB path
            // must have been taken — the perf claim only holds on the localised path.
            assert!(
                !evicted_dirty.is_empty(),
                "[{label}] expected a localisable edit (non-empty evicted-dirty set)"
            );
            let dims_b = scene_b.placed_region_dimensions(density);
            let recentre_b = scene_b.recentre_voxels_for_resolve(density);
            // The anchors pin the bounds, so the recentre must NOT shift — the precondition
            // under which the GPU-buffer incremental keeps untouched chunks' baked vertices.
            assert_eq!(
                recentre_b, recentre,
                "[{label}] anchors should pin the recentre; a shift would force wholesale fallback"
            );
            let chunks_b: Vec<([i32; 3], Arc<TwoLayerChunk>)> =
                cache.resident_two_layer_chunks(&scene_b, density, 0);

            // The plan — dilate the dirty set by the 26-neighbourhood, keep only non-empty
            // chunks — exactly as `incremental_rebuild_from_two_layer_chunks` computes it.
            let resident: Vec<[i32; 3]> = chunks_a.iter().map(|(c, _)| *c).collect();
            let occupied: Vec<[i32; 3]> = chunks_b
                .iter()
                .filter(|(_, chunk)| chunk.has_geometry())
                .map(|(coord, _)| *coord)
                .collect();
            let plan = cuboid_incremental_plan(&resident, &evicted_dirty, &occupied);

            // --- ONLY-DIRTY-REMESHED (the perf proof) ---
            // The re-meshed set is STRICTLY smaller than the resident set — most chunks keep
            // their buffers. Quantify the gap so a regression to wholesale re-mesh fails here.
            assert!(
                plan.rebuild.len() < resident.len(),
                "[{label}] incremental re-mesh must touch FEWER than every resident chunk \
                 (rebuilt {} of {} resident) — else it's a wholesale re-mesh regression",
                plan.rebuild.len(),
                resident.len(),
            );

            // Re-mesh ONLY the rebuild subset (seam culling from the FULL post-edit set) and
            // confirm the filtered build produced meshes for EXACTLY that subset (∩ non-empty)
            // — never the whole resident set.
            let rebuild_set: std::collections::HashSet<[i32; 3]> =
                plan.rebuild.iter().copied().collect();
            let rebuilt = build_two_layer_chunk_meshes_filtered(
                &chunks_b,
                Some(&rebuild_set),
                dims_b,
                recentre_b,
                density,
                LayerBand::FULL,
            );
            let remeshed_coords: std::collections::HashSet<[i32; 3]> =
                rebuilt.iter().map(|m| m.coord).collect();
            assert!(
                remeshed_coords.is_subset(&rebuild_set),
                "[{label}] the filtered two-layer build meshed a chunk OUTSIDE the dirty-dilated \
                 rebuild set — seam culling must read neighbours but only EMIT the subset"
            );

            // --- BYTE-PARITY ---
            // Apply the plan to A's buffer map (drop evicted + rebuild coords, insert the fresh
            // meshes) and assert it byte-equals a wholesale re-mesh of B — the exact ops
            // `incremental_rebuild_from_two_layer_chunks` performs on `chunk_buffers`.
            let mut result = two_layer_mesh_map(&wholesale_a);
            for coord in &plan.evict {
                result.remove(coord);
            }
            for coord in &plan.rebuild {
                result.remove(coord); // a rebuild coord meshing to empty drops out here
            }
            for (coord, entry) in two_layer_mesh_map(&rebuilt) {
                result.insert(coord, entry);
            }

            let wholesale_b = build_two_layer_chunk_meshes(
                &chunks_b,
                dims_b,
                recentre_b,
                density,
                LayerBand::FULL,
            );
            let target = two_layer_mesh_map(&wholesale_b);
            assert_eq!(
                result, target,
                "[{label}] incremental two-layer buffer set must byte-equal the wholesale rebuild"
            );
        }
    }

    /// Band-masked occupancy of a dense grid, keyed in the recentred-index frame the two-layer
    /// mesher emits in (a voxel at recentred index `v` sits at absolute layer `v[2] + half_z`,
    /// FLOORED half — the SAME map the two-layer band clip inverts). Mirrors the banded-torus
    /// test's masking, restated in the RECENTRED frame so it lines up with `world_offset`.
    fn banded_occupancy_indices(
        dense: &VoxelGrid,
        band: LayerBand,
    ) -> std::collections::HashSet<[i64; 3]> {
        let mut min_world = [f32::INFINITY; 3];
        for v in &dense.occupied {
            let position = v.world_position();
            for (axis, m) in min_world.iter_mut().enumerate() {
                *m = m.min(position[axis]);
            }
        }
        let half_z = (dense.dimensions[2] / 2) as f32;
        let base_layer = (min_world[2] + half_z).floor() as i64;
        dense
            .occupied
            .iter()
            .map(|v| {
                let position = v.world_position();
                [
                    (position[0] - min_world[0]).round() as i64,
                    (position[1] - min_world[1]).round() as i64,
                    (position[2] - min_world[2]).round() as i64,
                ]
            })
            .filter(|idx| {
                let layer = base_layer + idx[2];
                layer >= band.band_min as i64 && layer <= band.band_max as i64
            })
            .collect()
    }

    /// ADR 0010 #53 GATE: the two-layer BANDED mesher's RENDERABLE face set equals the dense
    /// path's band-masked genuine surface — proving the band reclip (coarse clipped one-box,
    /// microblock cuboid clip, cut-plane cap faces) is a pure optimisation on the data seam,
    /// identical to `build_cuboid_mesh_banded` on the dense path. Because `renderable_unit_faces`
    /// tests the front cell against the BAND-MASKED occupancy, a cut-plane cap face (front cell
    /// out of band ⇒ air) MUST be emitted, and a spurious over-emit or a hole both fail.
    fn assert_two_layer_banded_face_parity(
        scene: &Scene,
        density: u32,
        band: LayerBand,
        label: &str,
    ) {
        let dense = scene.resolve_region(scene.full_extent_blocks(density), density, 0);
        assert!(!dense.occupied.is_empty(), "[{label}] scene resolved empty");
        let banded = banded_occupancy_indices(&dense, band);
        assert!(!banded.is_empty(), "[{label}] band kept no voxels");
        let genuine = genuine_exposed_faces(&banded);
        let world_offset = grid_world_offset(&dense);

        // Dense banded reference → its visible face subset must equal the banded ground truth.
        let whole = build_cuboid_mesh_banded(&dense, density, band);
        let mut whole_visible =
            visible_unit_faces(&whole.vertices, &whole.indices, world_offset, &genuine);
        whole_visible.extend(visible_unit_faces(
            &whole.vertices,
            &whole.indices_overlay,
            world_offset,
            &genuine,
        ));
        assert_eq!(
            whole_visible, genuine,
            "[{label}] dense banded reference visible faces != band-masked ground truth"
        );

        // Two-layer banded mesher → its RENDERABLE face set (front cell tested against the
        // band-masked occupancy) must equal the same ground truth.
        let store = TwoLayerStore::enabled();
        let chunks = store.build_covering_chunks(scene, density, 0);
        let meshes = build_two_layer_chunk_meshes(
            &chunks,
            dense.dimensions,
            RecentreVoxels::new(dense.recentre_voxels),
            density,
            band,
        );
        let mut renderable = std::collections::HashSet::new();
        for mesh in &meshes {
            renderable.extend(renderable_unit_faces(
                &mesh.vertices,
                &mesh.indices,
                world_offset,
                &banded,
            ));
            renderable.extend(renderable_unit_faces(
                &mesh.vertices,
                &mesh.indices_overlay,
                world_offset,
                &banded,
            ));
        }
        assert_eq!(
            renderable, genuine,
            "[{label}] two-layer BANDED renderable faces != band-masked ground truth ({} vs {}) \
             — a hole, a spurious cut-plane over-emit, or a missing cap face",
            renderable.len(),
            genuine.len()
        );
    }

    /// THE ADR 0010 #53 GATE: the two-layer mesher honours a layer band identically to the dense
    /// banded path across a matrix of bands — a band that CUTS through coarse-solid interiors (a
    /// large box: the clipped one-box + cut cap face), a band that clips microblock cuboids (a
    /// sphere), a band flush to a block boundary, and a thin single-block band. Multi-chunk at
    /// d16 so the clip crosses chunk seams.
    #[test]
    fn two_layer_banded_mesher_matches_dense() {
        let density = 16u32;
        let band = |lo: u32, hi: u32| LayerBand {
            band_min: lo,
            band_max: hi,
            onion_depth: 0,
        };

        // Large solid box (8 blocks = 128 voxels/axis = 2 chunks/axis): the coarse one-box
        // interior must clip to the band and cap at the cut plane, across the chunk seam.
        let large = Scene::from_nodes(vec![two_layer_tool(
            TwoLayerShape::Box,
            [8, 8, 8],
            [0, 0, 0],
            MC::Stone,
            density,
        )]);
        // A band cutting mid-block (layer 40 is inside block 2 at d16), a block-flush band, and
        // a thin single-layer slice.
        for b in [band(0, 40), band(48, 96), band(0, 63), band(70, 70)] {
            assert_two_layer_banded_face_parity(&large, density, b, "large-box-band");
        }

        // Sphere: a rounded boundary — the microblock cuboids clip to the band and the equator
        // slice exposes a filled cross-section cap.
        let sphere = Scene::from_nodes(vec![two_layer_tool(
            TwoLayerShape::Sphere,
            [5, 5, 5],
            [0, 0, 0],
            MC::Stone,
            density,
        )]);
        for b in [band(0, 40), band(30, 50)] {
            assert_two_layer_banded_face_parity(&sphere, density, b, "sphere-band");
        }
    }

    fn village_instance(name: &str, def: DefId, offset: [i64; 3], density: u32) -> Node {
        let mut node = Node::new(name, NodeContent::Instance(def));
        node.transform = NodeTransform::from_blocks(offset, density);
        node
    }
}
