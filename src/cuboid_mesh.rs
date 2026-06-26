//! Cuboid mesh render path (ADR 0002 E3b-1, part of #18) — BEHIND A FLAG.
//!
//! The instanced renderer ([`crate::renderer::VoxelRenderer`]) draws one cube
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

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::cuboid::{decompose_into_boxes, VoxelBox, VoxelRegion};
use crate::frustum::{Aabb, Frustum};
use crate::panel::MaterialChoice;
use crate::renderer::{bucket_instances_into_chunks, LayerBand, DEPTH_FORMAT, MSAA_SAMPLE_COUNT};
use crate::texture_atlas::MaterialAtlas;
use crate::voxel::VoxelGrid;

/// One mesh vertex of a cuboid face: world position, the face's outward normal,
/// and the box's `material_id` (constant across the face).
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

/// A built CPU mesh of a grid's exposed cuboid faces, plus the per-chunk index
/// ranges + world AABBs for frustum culling (reusing the instanced path's chunk
/// partition).
#[derive(Debug, Default, Clone)]
pub struct CuboidMesh {
    vertices: Vec<CuboidVertex>,
    indices: Vec<u32>,
    /// One entry per render chunk: `(index_start, index_count, world AABB)`.
    chunks: Vec<MeshChunk>,
    /// Number of boxes the grid decomposed into (diagnostic).
    box_count: u32,
}

#[derive(Debug, Clone, Copy)]
struct MeshChunk {
    index_start: u32,
    index_count: u32,
    aabb: Aabb,
}

impl CuboidMesh {
    /// Total number of triangles in the mesh.
    pub fn triangle_count(&self) -> u32 {
        (self.indices.len() / 3) as u32
    }

    /// Total number of exposed quad faces (two triangles each).
    pub fn face_count(&self) -> u32 {
        (self.indices.len() / 6) as u32
    }

    /// Number of vertices.
    pub fn vertex_count(&self) -> u32 {
        self.vertices.len() as u32
    }

    /// Number of indices.
    pub fn index_count(&self) -> u32 {
        self.indices.len() as u32
    }

    /// Number of cuboid boxes the grid decomposed into.
    pub fn box_count(&self) -> u32 {
        self.box_count
    }

    /// Number of render chunks the mesh is partitioned into.
    pub fn chunk_count(&self) -> u32 {
        self.chunks.len() as u32
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
/// Where the instanced path discards fragments outside the band per voxel layer,
/// the cuboid path masks the densified region to the band's absolute Y-layer range
/// `[band.band_min, band.band_max]` (INCLUSIVE) BEFORE decomposition. Masking (not
/// a fragment discard) is required so the band's top/bottom voxels expose real CAP
/// faces: a single tall merged column has only one +Y face — at the model's true
/// top — so discarding its out-of-band fragments would leave the displayed slab
/// open-topped. Masking makes the cells just outside the band air, so the greedy
/// mesher caps the slab exactly like the instanced slab's top/bottom voxel faces.
///
/// `LayerBand::FULL` (band_max = u32::MAX) masks nothing — the full model is built,
/// byte-identical to the unbanded path.
pub fn build_cuboid_mesh_banded(
    grid: &VoxelGrid,
    voxels_per_block: u32,
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
    // Mask region cells whose ABSOLUTE Y-layer falls outside `[band_min, band_max]`
    // to air, so the greedy mesher below produces real cap faces at the band edges
    // (see `build_cuboid_mesh_banded`). The instanced path clips by the absolute
    // layer `floor(world_position.y + half_y)`; a region-local Y index `ly` maps to
    // that absolute layer by a constant `base_layer = floor(min_world.y + half_y)`
    // (= `floor(world_offset.y + 0.5 + half_y)`), so absolute layer = `base_layer +
    // ly`. We invert the band into region-local Y and clear everything outside it.
    if band.band_min > 0 || band.band_max != u32::MAX {
        let half_y = grid_y as f32 / 2.0;
        let base_layer = (world_offset[1] + 0.5 + half_y).floor() as i64;
        // Region-local Y range that maps into [band_min, band_max] (inclusive).
        let local_lo = band.band_min as i64 - base_layer;
        let local_hi = band.band_max as i64 - base_layer;
        let [rx, ry, rz] = region.extent;
        for ly in 0..ry {
            let in_band = (ly as i64) >= local_lo && (ly as i64) <= local_hi;
            if in_band {
                continue;
            }
            for lz in 0..rz {
                for lx in 0..rx {
                    region.set(lx, ly, lz, None);
                }
            }
        }
    }

    let boxes = decompose_into_boxes(&region);

    // Reuse the instanced chunk partition: bucket voxels into chunks and key each
    // box to a chunk by its min-corner voxel. A box never straddles a material
    // change, but it CAN straddle a chunk boundary; we assign it wholesale to the
    // chunk of its min corner and expand that chunk's AABB to contain it, so the
    // frustum test stays conservative (never a false negative).
    let (_instances, instanced_chunks) = bucket_instances_into_chunks(grid, voxels_per_block);
    let chunk_extent = (crate::renderer::CHUNK_BLOCKS * voxels_per_block.max(1)) as f32;
    // `world_offset` (from `occupied_index_bounds`) maps a REGION-LOCAL voxel index
    // to its world min-corner plane at the EXACT location the instanced path draws
    // that voxel, i.e. `min(world_position) - 0.5`. Adding it to a local index `l`
    // gives `(min_voxel_min_plane) + l`, so the cuboid mesh sits pixel-for-pixel on
    // top of the instanced voxels even when the scene recentred the cloud off the
    // geometric centre. (A centred grid yields the old `-dimensions/2`.)

    // Map a chunk integer key → its position in `instanced_chunks` (same sort
    // order). We rebuild the key→slot map by recomputing each chunk's key from its
    // AABB centre is fragile; instead bucket boxes by key into our own map and
    // build chunks from that, computing AABBs from the boxes themselves.
    use std::collections::HashMap;
    let mut buckets: HashMap<[i32; 3], Vec<usize>> = HashMap::new();
    for (box_index, voxel_box) in boxes.iter().enumerate() {
        // World centre of the box's min-corner voxel: local index + 0.5 + offset.
        let key = [
            ((voxel_box.min[0] as f32 + 0.5 + world_offset[0]) / chunk_extent).floor() as i32,
            ((voxel_box.min[1] as f32 + 0.5 + world_offset[1]) / chunk_extent).floor() as i32,
            ((voxel_box.min[2] as f32 + 0.5 + world_offset[2]) / chunk_extent).floor() as i32,
        ];
        buckets.entry(key).or_default().push(box_index);
    }
    // Deterministic chunk order (matches the instanced sort).
    let mut keys: Vec<[i32; 3]> = buckets.keys().copied().collect();
    keys.sort_unstable();
    // Touch `instanced_chunks` so the partition source is unmistakably the shared
    // one; the count is a useful invariant in debug builds.
    debug_assert!(
        instanced_chunks.len() >= keys.len() || boxes.is_empty(),
        "cuboid chunks should not exceed instanced chunks"
    );

    let mut vertices: Vec<CuboidVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut chunks: Vec<MeshChunk> = Vec::new();

    for key in keys {
        let box_indices = &buckets[&key];
        let index_start = indices.len() as u32;
        let mut aabb = Aabb::empty();
        for &box_index in box_indices {
            let voxel_box = &boxes[box_index];
            emit_box_faces(voxel_box, &region, world_offset, &mut vertices, &mut indices, &mut aabb);
        }
        let index_count = indices.len() as u32 - index_start;
        chunks.push(MeshChunk {
            index_start,
            index_count,
            aabb,
        });
    }

    CuboidMesh {
        vertices,
        indices,
        chunks,
        box_count: boxes.len() as u32,
    }
}

/// The mesh's vertices, or a single zeroed placeholder vertex when the mesh is
/// empty (so the GPU vertex buffer is never zero-sized — nothing is drawn anyway,
/// since an empty mesh has no chunks).
fn mesh_vertices_or_placeholder(mesh: &CuboidMesh) -> Vec<CuboidVertex> {
    if mesh.vertices.is_empty() {
        vec![CuboidVertex {
            position: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            material_id: 0,
        }]
    } else {
        mesh.vertices.clone()
    }
}

/// The mesh's indices, or a single placeholder index when the mesh is empty.
fn mesh_indices_or_placeholder(mesh: &CuboidMesh) -> Vec<u32> {
    if mesh.indices.is_empty() {
        vec![0u32]
    } else {
        mesh.indices.clone()
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
        for (axis, min_axis) in min_world.iter_mut().enumerate() {
            *min_axis = min_axis.min(voxel.world_position[axis]);
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
        let index = region_index(voxel.world_position);
        for axis in 0..3 {
            max_index[axis] = max_index[axis].max(index[axis]);
        }
    }
    let extent = [
        (max_index[0] + 1) as u32,
        (max_index[1] + 1) as u32,
        (max_index[2] + 1) as u32,
    ];

    // Pass 3: stamp materials into the dense region.
    let mut region = VoxelRegion::new_empty(extent);
    for voxel in &grid.occupied {
        let [lx, ly, lz] = region_index(voxel.world_position);
        region.set(lx as u32, ly as u32, lz as u32, Some(voxel.material_id));
    }

    // World min-corner plane of region-local index 0 = its centre minus 0.5.
    let world_offset = [
        min_world[0] - 0.5,
        min_world[1] - 0.5,
        min_world[2] - 0.5,
    ];
    (region, world_offset)
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
                material_id: voxel_box.material_id as u32,
            });
        }
        // Two CCW triangles per quad (matching the instanced winding scheme).
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// Is the given face of the box exposed? The face is exposed when ANY voxel cell
/// immediately beyond it is air — i.e. the face is part of the solid's outer
/// surface. Because a box is solid, a face fully backed by solid neighbours is
/// occluded and culled; a box-internal direction (impossible for a single box,
/// but defended) is likewise covered. We scan the slab of neighbour cells across
/// the face's two in-plane axes and expose the whole quad if any neighbour is air.
///
/// This keeps ONE quad per box face (not per voxel), so a merged box stays cheap
/// while the silhouette is correct: if a face is partially exposed, the whole
/// merged quad is emitted (an over-draw of at most the box's own face, never a
/// hole), which is acceptable for shape parity.
fn face_is_exposed(voxel_box: &VoxelBox, region: &VoxelRegion, delta: [i32; 3]) -> bool {
    let [min_x, min_y, min_z] = voxel_box.min;
    let [max_x, max_y, max_z] = voxel_box.max;

    // The neighbour slab is the box's face shifted one cell along `delta`.
    let span = |axis: usize| -> (i64, i64) {
        match axis {
            0 => (min_x as i64, max_x as i64),
            1 => (min_y as i64, max_y as i64),
            _ => (min_z as i64, max_z as i64),
        }
    };
    let (sx0, sx1) = span(0);
    let (sy0, sy1) = span(1);
    let (sz0, sz1) = span(2);

    // For the axis the face faces along, the neighbour plane is a single layer at
    // the box edge + delta; the other two axes scan the box's full extent.
    let scan_axis = |axis: usize, edge_min: i64, edge_max: i64| -> (i64, i64) {
        if delta[axis] != 0 {
            // The single neighbour layer just outside the box on this axis.
            let plane = if delta[axis] > 0 {
                edge_max + 1
            } else {
                edge_min - 1
            };
            (plane, plane)
        } else {
            (edge_min, edge_max)
        }
    };
    let (nx0, nx1) = scan_axis(0, sx0, sx1);
    let (ny0, ny1) = scan_axis(1, sy0, sy1);
    let (nz0, nz1) = scan_axis(2, sz0, sz1);

    for nz in nz0..=nz1 {
        for ny in ny0..=ny1 {
            for nx in nx0..=nx1 {
                if nx < 0 || ny < 0 || nz < 0 {
                    return true; // outside grid → air → exposed
                }
                if region
                    .material_at(nx as u32, ny as u32, nz as u32)
                    .is_none()
                {
                    return true; // an air neighbour → this face is exposed
                }
            }
        }
    }
    false
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
    _band_pad: f32,
    material_base_colors: [[f32; 4]; MaterialChoice::MATERIAL_COUNT],
    /// Per-material atlas sub-rect (ADR 0002 E3c-1 / O8), indexed by `material_id`:
    /// `[inset_min_u, inset_min_v, inset_size_u, inset_size_v]`. The shader maps the
    /// per-voxel slice's `fract`-tiled UV into this window of the single atlas, so a
    /// chunk of mixed materials is ONE mesh = ONE draw (no per-material texture
    /// bind). Each `vec4` is naturally 16-aligned.
    material_atlas_rects: [[f32; 4]; MaterialChoice::MATERIAL_COUNT],
}

/// Convert a packed [`MaterialAtlas`]'s per-material sub-rects into the uniform
/// array layout `[inset_min_u, inset_min_v, inset_size_u, inset_size_v]` the shader
/// indexes by `material_id`. Materials without a packed sub-rect (should not happen
/// for the procedural set) fall back to the WHOLE atlas (`[0,0,1,1]`), so a missing
/// id degrades to "sample the atlas" rather than panicking.
fn atlas_rects_from(atlas: &MaterialAtlas) -> [[f32; 4]; MaterialChoice::MATERIAL_COUNT] {
    let mut rects = [[0.0, 0.0, 1.0, 1.0]; MaterialChoice::MATERIAL_COUNT];
    for (slot, sub_rect) in rects.iter_mut().zip(atlas.sub_rects.iter()) {
        let [size_u, size_v] = sub_rect.inset_size();
        *slot = [sub_rect.inset_min_u, sub_rect.inset_min_v, size_u, size_v];
    }
    rects
}

/// The cuboid atlas bind-group layout: a single 2D texture (binding 0) + sampler
/// (binding 1). One atlas for ALL materials replaces the former per-material
/// D2Array binds (ADR 0002 O8).
fn build_atlas_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
fn upload_atlas_texture(
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

/// All GPU resources for drawing the cuboid mesh (flag-gated alternate path).
pub struct CuboidMeshRenderer {
    pipeline: wgpu::RenderPipeline,
    /// Face-orientation debug pipeline: identical to `pipeline` except
    /// `cull_mode: None`, so a back face that is the nearest surface (a winding
    /// bug) still draws and is flagged by the shader's `front_facing` marker.
    /// Selected in `draw` when `debug_face_mode` is on — mirroring the instanced
    /// path's cull-off debug pipeline.
    debug_pipeline: wgpu::RenderPipeline,
    /// Whether the last `update_uniforms` requested debug-faces mode (selects the
    /// cull-off pipeline in `draw`, matching the uploaded `debug_face_mode` flag).
    debug_face_mode: bool,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
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
    mesh: CuboidMesh,
    /// Indices into `mesh.chunks` that survived the last frustum cull.
    visible_chunks: Vec<usize>,
    /// The grid + density the mesh was built from, retained so the mesh can be
    /// rebuilt CLIPPED to a new layer-range band (issue #12 parity) without the
    /// caller re-supplying the grid. The cuboid band clip masks the region before
    /// decomposition (real cap faces), so a band change re-meshes; we cache the
    /// last band and rebuild only when it differs.
    source_grid: VoxelGrid,
    source_voxels_per_block: u32,
    current_band: LayerBand,
}

impl CuboidMeshRenderer {
    /// Build the cuboid renderer from a grid, decomposing + meshing immediately.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        grid: &VoxelGrid,
        voxels_per_block: u32,
    ) -> Self {
        let mesh = build_cuboid_mesh(grid, voxels_per_block);

        // Always allocate at least one (zeroed) vertex/index so the buffers are
        // valid even for an empty grid (nothing is drawn — no chunks).
        let vertices = mesh_vertices_or_placeholder(&mesh);
        let raw_indices = mesh_indices_or_placeholder(&mesh);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cuboid mesh vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cuboid mesh indices"),
            contents: bytemuck::cast_slice(&raw_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

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

        let visible_chunks: Vec<usize> = (0..mesh.chunks.len()).collect();

        Self {
            pipeline,
            debug_pipeline,
            debug_face_mode: false,
            vertex_buffer,
            index_buffer,
            uniform_buffer,
            uniform_bind_group,
            atlas_bind_group,
            atlas_rects,
            bound_material: MaterialChoice::Plain,
            mesh,
            visible_chunks,
            source_grid: grid.clone(),
            source_voxels_per_block: voxels_per_block,
            current_band: LayerBand::FULL,
        }
    }

    /// Re-mesh the stored grid CLIPPED to `band` (issue #12 parity) and re-upload
    /// the vertex/index buffers, when `band` differs from the last build. The
    /// cuboid band clip masks the region before decomposition so the band edges get
    /// real cap faces, so it must rebuild geometry (a fragment discard would leave a
    /// merged column's slab open-topped). No-op when the band is unchanged.
    fn rebuild_for_band(&mut self, device: &wgpu::Device, band: LayerBand) {
        if band == self.current_band {
            return;
        }
        self.current_band = band;
        self.mesh =
            build_cuboid_mesh_banded(&self.source_grid, self.source_voxels_per_block, band);
        let vertices = mesh_vertices_or_placeholder(&self.mesh);
        let raw_indices = mesh_indices_or_placeholder(&self.mesh);
        self.vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cuboid mesh vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cuboid mesh indices"),
            contents: bytemuck::cast_slice(&raw_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        // All chunks visible until the next frustum cull in `update_uniforms`.
        self.visible_chunks = (0..self.mesh.chunks.len()).collect();
    }

    /// The built mesh (for diagnostics: triangle/box/chunk counts).
    pub fn mesh(&self) -> &CuboidMesh {
        &self.mesh
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
    /// (exactly like the instanced step-3b). `None` (a loaded VS block, rendered
    /// as a single global material for now) binds Plain + disables modulation.
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
            grid_half_extent: [
                grid_dimensions[0] as f32 / 2.0,
                grid_dimensions[1] as f32 / 2.0,
                grid_dimensions[2] as f32 / 2.0,
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
            _band_pad: 0.0,
            material_base_colors: base_colors,
            material_atlas_rects: self.atlas_rects,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Frustum-cull the chunks (reusing the chunk world-AABBs).
        let frustum = Frustum::from_view_projection(view_projection);
        self.visible_chunks.clear();
        for (index, chunk) in self.mesh.chunks.iter().enumerate() {
            if frustum.intersects_aabb(&chunk.aabb) {
                self.visible_chunks.push(index);
            }
        }
    }

    /// Record the cuboid draw into an already-begun render pass. Draws each
    /// frustum-visible chunk as its own indexed range.
    pub fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        if self.mesh.indices.is_empty() {
            return;
        }
        // Debug-faces mode selects the cull-off pipeline (matching the uploaded
        // `debug_face_mode` flag) so back faces surviving a winding bug still draw
        // and get the shader's stripe marker — same as the instanced path.
        let pipeline = if self.debug_face_mode {
            &self.debug_pipeline
        } else {
            &self.pipeline
        };
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        // ONE atlas bind group for ALL materials (E3c-1 / O8): the per-face
        // `material_id` selects its atlas sub-rect in the shader, so a mixed-material
        // chunk needs no per-material rebind — one bind, one draw per chunk.
        render_pass.set_bind_group(1, &self.atlas_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        for &chunk_index in &self.visible_chunks {
            let chunk = &self.mesh.chunks[chunk_index];
            if chunk.index_count == 0 {
                continue;
            }
            let start = chunk.index_start;
            let end = start + chunk.index_count;
            render_pass.draw_indexed(start..end, 0, 0..1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::Voxel;

    /// Build a tiny grid from a set of (absolute index) occupied voxels, all one
    /// material, with the given dimensions.
    fn grid_from_indices(dimensions: [u32; 3], cells: &[[u32; 3]], material: u16) -> VoxelGrid {
        let half = [
            dimensions[0] as f32 / 2.0,
            dimensions[1] as f32 / 2.0,
            dimensions[2] as f32 / 2.0,
        ];
        let mut grid = VoxelGrid::new(dimensions);
        for &[i, j, k] in cells {
            grid.occupied.push(Voxel {
                world_position: [
                    i as f32 + 0.5 - half[0],
                    j as f32 + 0.5 - half[1],
                    k as f32 + 0.5 - half[2],
                ],
                block_local_coord: [0, 0, 0],
                material_id: material,
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
        // Second voxel, different material, adjacent in +X.
        let half = [2.0f32, 1.5, 1.5];
        grid.occupied.push(Voxel {
            world_position: [2.0 + 0.5 - half[0], 1.0 + 0.5 - half[1], 1.0 + 0.5 - half[2]],
            block_local_coord: [0, 0, 0],
            material_id: 1,
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
    /// 5³ grid merges to one box; its top (+Y) face must span 3 voxels along X and
    /// 1 along Z, i.e. world X-extent 3 and Z-extent 1.
    #[test]
    fn merged_face_spans_one_uv_unit_per_voxel() {
        let grid = grid_from_indices([5, 5, 5], &[[1, 2, 2], [2, 2, 2], [3, 2, 2]], 0);
        let mesh = build_cuboid_mesh(&grid, 1);
        assert_eq!(mesh.box_count(), 1, "3-voxel X-run merges to one box");

        // Absolute voxel position = world position + half (dims/2). The UV in the
        // shader uses exactly this, so spanning 3 units in X across the face means
        // the texture tiles 3× (once per voxel) with a Repeat sampler.
        let half = [2.5f32, 2.5, 2.5];
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

    /// E3b-2: the face-normal → texture-array layer mapping the cuboid shader uses
    /// must match the instanced `face_layer` (0 +X, 1 -X, 2 +Y, 3 -Y, 4 +Z, 5 -Z).
    /// Replicated here as a pure function so the mapping is regression-guarded.
    #[test]
    fn face_normal_to_layer_matches_instanced() {
        fn face_layer(normal: [f32; 3]) -> i32 {
            let m = [normal[0].abs(), normal[1].abs(), normal[2].abs()];
            if m[0] > 0.5 {
                if normal[0] > 0.0 { 0 } else { 1 }
            } else if m[1] > 0.5 {
                if normal[1] > 0.0 { 2 } else { 3 }
            } else if normal[2] > 0.0 {
                4
            } else {
                5
            }
        }
        let expected = [0, 1, 2, 3, 4, 5];
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
                let shape = SdfShape {
                    kind,
                    size_blocks: size,
                    voxels_per_block: 8,
                    wall_blocks: 1,
                };
                // Shift-invariance: also run a deliberately recentred copy of the
                // grid (every voxel +8 in each axis, like `resolve_region`'s
                // off-centre composite) — coverage must be identical.
                for shift in [0.0f32, 8.0] {
                    let mut shifted = VoxelGrid::new(shape.grid_dimensions());
                    shape.resolve(&mut shifted);
                    if shifted.occupied.is_empty() {
                        continue;
                    }
                    for voxel in &mut shifted.occupied {
                        for axis in 0..3 {
                            voxel.world_position[axis] += shift;
                        }
                    }

                    let (region, _world_offset) = region_from_voxel_cloud(&shifted);
                    let region_solid =
                        region.cells.iter().filter(|c| c.is_some()).count();
                    let boxes = decompose_into_boxes(&region);
                    let covered: u64 = boxes.iter().map(|b| b.voxel_count()).sum();

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
    /// absolute Y-layer range BEFORE decomposition, so clipping a solid block to a
    /// sub-band yields a thinner block — with NEW cap faces at the band edges, just
    /// like the instanced slab's per-voxel top/bottom faces (a fragment discard on
    /// the single merged column would leave it open-topped, with no caps). Here a
    /// solid 4×4×4 block (one tall box) clipped to a 2-layer band must mesh as a
    /// 4×2×4 box: still 6 faces, but spanning exactly 2 voxels in Y.
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
        // A centred 4³ block: half_y = 2, so absolute layer == region-local Y here.
        let grid = grid_from_indices([4, 4, 4], &cells, 0);

        // Full band → the whole block: 1 box, 6 faces, Y-span 4.
        let full = build_cuboid_mesh_banded(&grid, 1, LayerBand::FULL);
        assert_eq!(full.box_count(), 1);
        assert_eq!(full.face_count(), 6);

        // Band [1, 2] (inclusive) → only layers 1 and 2 survive: a 4×2×4 slab.
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

        // The clipped slab spans EXACTLY 2 voxels in Y (the band height), with new
        // caps — confirming masking, not a fragment discard.
        let half_y = 2.0f32;
        let abs_y: Vec<f32> = clipped
            .vertices
            .iter()
            .map(|v| v.position[1] + half_y)
            .collect();
        let min_y = abs_y.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_y = abs_y.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(min_y, 1.0, "slab bottom cap at the band's lower layer");
        assert_eq!(max_y, 3.0, "slab top cap at the band's upper layer + 1");
        assert_eq!(max_y - min_y, 2.0, "slab is exactly the 2-layer band tall");
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
        let half = [2.0f32, 1.5, 1.5];
        two_box.occupied.push(Voxel {
            world_position: [2.0 + 0.5 - half[0], 1.0 + 0.5 - half[1], 1.0 + 0.5 - half[2]],
            block_local_coord: [0, 0, 0],
            material_id: 1,
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
        assert_eq!(mesh.chunk_count(), 0);
    }
}
