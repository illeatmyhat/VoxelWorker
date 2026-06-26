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
//! SCOPE (this sub-step): SHAPE parity + per-box material colour + basic
//! lighting. NO texture slice, NO grid overlay, NO layer clip, NO debug-faces —
//! those land in later E3 sub-steps. The instanced path stays the DEFAULT and is
//! untouched; this path is selected only when the `cuboid` mesher flag is on.
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

use crate::cuboid::{decompose_into_boxes, region_from_voxel_grid, VoxelBox, VoxelRegion};
use crate::frustum::{Aabb, Frustum};
use crate::panel::MaterialChoice;
use crate::renderer::{bucket_instances_into_chunks, DEPTH_FORMAT, MSAA_SAMPLE_COUNT};
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
    let [grid_x, grid_y, grid_z] = grid.dimensions;
    if grid_x == 0 || grid_y == 0 || grid_z == 0 || grid.occupied.is_empty() {
        return CuboidMesh::default();
    }

    // Decompose the WHOLE grid (origin 0 → region-local index == absolute index).
    let region = region_from_voxel_grid(grid, [0, 0, 0], grid.dimensions);
    let boxes = decompose_into_boxes(&region);

    // Reuse the instanced chunk partition: bucket voxels into chunks and key each
    // box to a chunk by its min-corner voxel. A box never straddles a material
    // change, but it CAN straddle a chunk boundary; we assign it wholesale to the
    // chunk of its min corner and expand that chunk's AABB to contain it, so the
    // frustum test stays conservative (never a false negative).
    let (_instances, instanced_chunks) = bucket_instances_into_chunks(grid, voxels_per_block);
    let chunk_extent = (crate::renderer::CHUNK_BLOCKS * voxels_per_block.max(1)) as f32;
    let half = [
        grid_x as f32 / 2.0,
        grid_y as f32 / 2.0,
        grid_z as f32 / 2.0,
    ];

    // Map a chunk integer key → its position in `instanced_chunks` (same sort
    // order). We rebuild the key→slot map by recomputing each chunk's key from its
    // AABB centre is fragile; instead bucket boxes by key into our own map and
    // build chunks from that, computing AABBs from the boxes themselves.
    use std::collections::HashMap;
    let mut buckets: HashMap<[i32; 3], Vec<usize>> = HashMap::new();
    for (box_index, voxel_box) in boxes.iter().enumerate() {
        // World centre of the box's min-corner voxel: index + 0.5 - half.
        let key = [
            ((voxel_box.min[0] as f32 + 0.5 - half[0]) / chunk_extent).floor() as i32,
            ((voxel_box.min[1] as f32 + 0.5 - half[1]) / chunk_extent).floor() as i32,
            ((voxel_box.min[2] as f32 + 0.5 - half[2]) / chunk_extent).floor() as i32,
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
            emit_box_faces(voxel_box, &region, half, &mut vertices, &mut indices, &mut aabb);
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

/// Emit the exposed faces of one box into the shared vertex/index buffers,
/// expanding `aabb` to contain the box. A face is exposed when the voxel cell
/// immediately beyond it (per axis, across the box's full extent on the other two
/// axes) is air — at minimum this culls box-internal faces; here it also culls
/// faces fully covered by adjacent solid voxels.
fn emit_box_faces(
    voxel_box: &VoxelBox,
    region: &VoxelRegion,
    half: [f32; 3],
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

    // Expand the chunk AABB to this box's world extent.
    aabb.expand(glam::Vec3::new(lo[0] - half[0], lo[1] - half[1], lo[2] - half[2]));
    aabb.expand(glam::Vec3::new(hi[0] - half[0], hi[1] - half[1], hi[2] - half[2]));

    for face in &FACE_TEMPLATES {
        if !face_is_exposed(voxel_box, region, face.neighbor_delta) {
            continue;
        }
        let base = vertices.len() as u32;
        for corner in &face.corners {
            // 0 → min plane (lo), 1 → max+1 plane (hi); shift into world space.
            let world = [
                (if corner[0] == 0 { lo[0] } else { hi[0] }) - half[0],
                (if corner[1] == 0 { lo[1] } else { hi[1] }) - half[1],
                (if corner[2] == 0 { lo[2] } else { hi[2] }) - half[2],
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

/// std140-safe uniform block for the cuboid pass. Mirrors only what this step
/// needs: the camera matrix, the per-material base colours (reused from the
/// instanced step-3b modulation), and a modulation toggle. Field order matches
/// `CuboidUniforms` in `shaders/cuboid.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CuboidUniforms {
    view_projection: [[f32; 4]; 4],
    material_modulation_enabled: f32,
    _pad: [f32; 3],
    material_base_colors: [[f32; 4]; MaterialChoice::MATERIAL_COUNT],
}

/// All GPU resources for drawing the cuboid mesh (flag-gated alternate path).
pub struct CuboidMeshRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    mesh: CuboidMesh,
    /// Indices into `mesh.chunks` that survived the last frustum cull.
    visible_chunks: Vec<usize>,
}

impl CuboidMeshRenderer {
    /// Build the cuboid renderer from a grid, decomposing + meshing immediately.
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        grid: &VoxelGrid,
        voxels_per_block: u32,
    ) -> Self {
        let mesh = build_cuboid_mesh(grid, voxels_per_block);

        // Always allocate at least one (zeroed) vertex/index so the buffers are
        // valid even for an empty grid (nothing is drawn — no chunks).
        let vertices = if mesh.vertices.is_empty() {
            vec![CuboidVertex {
                position: [0.0; 3],
                normal: [0.0, 1.0, 0.0],
                material_id: 0,
            }]
        } else {
            mesh.vertices.clone()
        };
        let raw_indices = if mesh.indices.is_empty() {
            vec![0u32]
        } else {
            mesh.indices.clone()
        };

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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cuboid shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cuboid.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cuboid pipeline layout"),
            bind_group_layouts: &[Some(&uniform_bind_group_layout)],
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

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cuboid pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                buffers: &[vertex_layout],
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

        let visible_chunks: Vec<usize> = (0..mesh.chunks.len()).collect();

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            uniform_buffer,
            uniform_bind_group,
            mesh,
            visible_chunks,
        }
    }

    /// The built mesh (for diagnostics: triangle/box/chunk counts).
    pub fn mesh(&self) -> &CuboidMesh {
        &self.mesh
    }

    /// Number of chunks that survived the last frustum cull (will be drawn).
    pub fn visible_chunk_count(&self) -> u32 {
        self.visible_chunks.len() as u32
    }

    /// Upload the per-frame uniforms (camera matrix + per-material base colours)
    /// and frustum-cull the mesh chunks. `bound` is the active procedural material
    /// (drives the relative base-colour modulation, exactly like the instanced
    /// path's step-3b). When `None`, modulation is off (neutral colours) — e.g. a
    /// loaded VS block, which the cuboid path renders as a single global material
    /// for now.
    pub fn update_uniforms(
        &mut self,
        queue: &wgpu::Queue,
        view_projection: glam::Mat4,
        bound: Option<MaterialChoice>,
    ) {
        let (modulation_enabled, base_colors) = match bound {
            Some(material) => (
                true,
                crate::renderer::relative_material_base_colors_public(material),
            ),
            None => (false, [[1.0, 1.0, 1.0, 0.0]; MaterialChoice::MATERIAL_COUNT]),
        };
        let uniforms = CuboidUniforms {
            view_projection: view_projection.to_cols_array_2d(),
            material_modulation_enabled: if modulation_enabled { 1.0 } else { 0.0 },
            _pad: [0.0; 3],
            material_base_colors: base_colors,
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
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
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
