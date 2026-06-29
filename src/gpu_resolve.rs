//! GPU view-resolve P1 spike (ADR 0007): the repo's first compute pipeline.
//!
//! This is the GPU half of the CPU↔GPU A/B equivalence net (ADR 0007 §5/§6). It
//! evaluates a producer's voxel occupancy on the GPU, per chunk, over the SAME
//! apron'd field the CPU per-chunk fog builder (`build_per_chunk_fog_occupancy`)
//! produces — so the two can be asserted byte-identical. It is **display/test infra
//! only**: nothing here is authoritative, nothing reads back as truth (ADR 0006 §4).
//! Gated behind `--features gpu` so the GPU-less CI runner skips it.
//!
//! The spike deliberately uses a plain `u32`-per-cell readback buffer rather than the
//! production packed-byte storage buffer + `copy_buffer_to_texture`: the A/B test only
//! needs the occupancy values back on the CPU to compare, and a one-cell-per-invocation
//! buffer keeps the parity question (Rust↔WGSL float divergence at the iso-surface)
//! uncluttered by packing mechanics. The packed-texture path is P1's wiring step, after
//! parity is proven.

use wgpu::util::DeviceExt;

use crate::core_geom::CHUNK_BLOCKS;
use crate::sketch::{Operation, RevolveAxis, SketchSolid};
use crate::voxel::{SdfShape, ShapeKind};

/// The producer descriptor uniform. Layout MUST match the `Descriptor` struct in
/// `src/shaders/gpu_resolve.wgsl`: `vec3`/`vec4` fields are 16-byte aligned, trailing
/// scalars pack into the final 48 bytes. Total 144 bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Descriptor {
    grid: [i32; 4],
    local_offset: [i32; 4],
    semi_axes: [f32; 4],
    /// [extrude_min0, extrude_min1, revolve_axial_min, revolve_radial_max].
    profile_ints: [i32; 4],
    /// [in_plane_0, in_plane_1, normal, revolve_axial_world_axis].
    sketch_axes: [u32; 4],
    /// [radial_a, radial_b, revolve_is_inplane0, profile_straddles_axis].
    revolve_axes: [u32; 4],
    /// 0 = SDF primitive, 1 = sketch extrude, 2 = sketch revolve.
    producer_type: u32,
    kind: u32,
    wall_voxels: f32,
    turn: f32,
    profile_count: u32,
    is_partial: u32,
    chunk_extent: i32,
    pad: u32,
    num_chunks: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

impl Descriptor {
    /// The shared per-chunk fields (everything not producer-specific), with all
    /// producer-specific lanes zeroed. Producer builders fill in their own fields.
    fn base(grid: [u32; 3], chunk_extent: i32, num_chunks: u32) -> Self {
        let [grid_x, grid_y, grid_z] = grid;
        Self {
            grid: [grid_x as i32, grid_y as i32, grid_z as i32, 0],
            // A lone untranslated producer in the recentred frame the fog consumes has
            // fog-global == producer-local (the recentre and the fog decode's
            // floor(grid/2) cancel), so the offset is zero (see the WGSL `local_offset`).
            local_offset: [0, 0, 0, 0],
            semi_axes: [grid_x as f32 / 2.0, grid_y as f32 / 2.0, grid_z as f32 / 2.0, 0.0],
            profile_ints: [0; 4],
            sketch_axes: [0; 4],
            revolve_axes: [0; 4],
            producer_type: 0,
            kind: 0,
            wall_voxels: 0.0,
            turn: 0.0,
            profile_count: 0,
            is_partial: 0,
            chunk_extent,
            pad: (chunk_extent + 2) as u32,
            num_chunks,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

/// Map a [`ShapeKind`] to the discriminant the WGSL `signed_distance` switch expects.
/// Pinned here (not `as u32`) so a future enum reorder can't silently desync the shader.
fn shape_kind_discriminant(kind: ShapeKind) -> u32 {
    match kind {
        ShapeKind::Cylinder => 0,
        ShapeKind::Tube => 1,
        ShapeKind::Sphere => 2,
        ShapeKind::Torus => 3,
        ShapeKind::Box => 4,
    }
}

/// Holds the compute pipeline + bind-group layout so a test can resolve many cases
/// against one device without rebuilding the pipeline each call.
pub struct GpuResolver {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuResolver {
    /// Build the compute pipeline from `shaders/gpu_resolve.wgsl`.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_resolve compute"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gpu_resolve.wgsl").into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_resolve bgl"),
                entries: &[
                    // 0: descriptor uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 1: chunk coords (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 2: occupancy output (read-write storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 3: sketch profile vertices (read-only storage; dummy for SDF)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_resolve pll"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_resolve pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// GPU-evaluate the apron'd occupancy of `shape` at document density
    /// `voxels_per_block`, for each chunk in `chunk_coords` (CHUNK_BLOCKS-space integer
    /// coords, same order/contents as `PerChunkFogOccupancy.volumes`). Returns one
    /// `pad³`-byte occupancy Vec per chunk, in `(ak*pad + aj)*pad + ai` order — directly
    /// comparable to `ChunkFogVolume.occupancy`.
    pub fn resolve_sdf_occupancy(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shape: &SdfShape,
        voxels_per_block: u32,
        chunk_coords: &[[i32; 3]],
    ) -> Vec<Vec<u8>> {
        let voxels_per_block = voxels_per_block.max(1);
        let grid = shape.grid_dimensions(voxels_per_block);
        let chunk_extent = (CHUNK_BLOCKS * voxels_per_block) as i32;
        let mut descriptor = Descriptor::base(grid, chunk_extent, chunk_coords.len() as u32);
        descriptor.producer_type = 0;
        descriptor.kind = shape_kind_discriminant(shape.kind);
        descriptor.wall_voxels = (shape.wall_blocks * voxels_per_block) as f32;
        // SDF needs no profile; bind a single dummy vertex.
        self.dispatch(device, queue, descriptor, chunk_coords, &[[0, 0]])
    }

    /// GPU-evaluate the apron'd occupancy of a [`SketchSolid`] (extrude or revolve) at
    /// document density `voxels_per_block`, for each chunk in `chunk_coords`. Returns
    /// one `pad³`-byte occupancy Vec per chunk (same contract as
    /// [`resolve_sdf_occupancy`](Self::resolve_sdf_occupancy)).
    pub fn resolve_sketch_occupancy(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        solid: &SketchSolid,
        voxels_per_block: u32,
        chunk_coords: &[[i32; 3]],
    ) -> Vec<Vec<u8>> {
        let voxels_per_block = voxels_per_block.max(1);
        let grid = solid.grid_dimensions();
        let chunk_extent = (CHUNK_BLOCKS * voxels_per_block) as i32;
        let mut descriptor = Descriptor::base(grid, chunk_extent, chunk_coords.len() as u32);

        let plane = solid.sketch.plane;
        let [in_plane_0, in_plane_1] = plane.in_plane_axes();
        let normal = plane.normal_axis();
        descriptor.sketch_axes[0] = in_plane_0 as u32;
        descriptor.sketch_axes[1] = in_plane_1 as u32;
        descriptor.sketch_axes[2] = normal as u32;
        descriptor.profile_count = solid.sketch.profile.len() as u32;

        // The profile bbox min in-plane (mirrors `SketchSolid::profile_bounds`).
        let profile = &solid.sketch.profile;
        let mut min = [i64::MAX; 2];
        let mut max = [i64::MIN; 2];
        for point in profile {
            for axis in 0..2 {
                min[axis] = min[axis].min(point.offset_voxels[axis]);
                max[axis] = max[axis].max(point.offset_voxels[axis]);
            }
        }

        match solid.operation {
            Operation::Extrude { .. } => {
                descriptor.producer_type = 1;
                descriptor.profile_ints[0] = min[0] as i32;
                descriptor.profile_ints[1] = min[1] as i32;
            }
            Operation::Revolve { axis, sweep } => {
                descriptor.producer_type = 2;
                // Reinterpret the in-plane axes as (axial, radial) per RevolveAxis,
                // mirroring `resolve_revolve`'s setup exactly.
                let (axial_world_axis, axial_min, radial_in_plane_axis, radial_profile_coord) =
                    match axis {
                        RevolveAxis::InPlane0 => (in_plane_0, min[0], in_plane_1, 1usize),
                        RevolveAxis::InPlane1 => (in_plane_1, min[1], in_plane_0, 0usize),
                    };
                let mut radial_world_axes = [radial_in_plane_axis, normal];
                radial_world_axes.sort_unstable();
                let [radial_a, radial_b] = radial_world_axes;

                let mut straddles = false;
                let mut radial_max = 0i64;
                for point in profile {
                    let radial_coord = point.offset_voxels[radial_profile_coord];
                    if radial_coord < 0 {
                        straddles = true;
                    }
                    radial_max = radial_max.max(radial_coord.abs());
                }

                descriptor.sketch_axes[3] = axial_world_axis as u32;
                descriptor.profile_ints[2] = axial_min as i32;
                descriptor.profile_ints[3] = radial_max as i32;
                descriptor.revolve_axes[0] = radial_a as u32;
                descriptor.revolve_axes[1] = radial_b as u32;
                descriptor.revolve_axes[2] = matches!(axis, RevolveAxis::InPlane0) as u32;
                descriptor.revolve_axes[3] = straddles as u32;
                descriptor.turn = sweep.turn_degrees as f32;
                descriptor.is_partial = (sweep.turn_degrees < 360) as u32;
            }
        }

        let profile_vertices: Vec<[i32; 2]> = profile
            .iter()
            .map(|p| [p.offset_voxels[0] as i32, p.offset_voxels[1] as i32])
            .collect();
        self.dispatch(device, queue, descriptor, chunk_coords, &profile_vertices)
    }

    /// Build the buffers + bind group, dispatch, and read the occupancy back, split
    /// into one `pad³` Vec per chunk.
    fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        descriptor: Descriptor,
        chunk_coords: &[[i32; 3]],
        profile_vertices: &[[i32; 2]],
    ) -> Vec<Vec<u8>> {
        let pad = descriptor.pad as usize;
        let cells_per_chunk = pad * pad * pad;
        let num_chunks = chunk_coords.len();
        if num_chunks == 0 {
            return Vec::new();
        }
        let total_cells = cells_per_chunk * num_chunks;

        // The spike dispatches one invocation per apron cell along x only; guard the
        // single-dimension workgroup-count limit so a too-large case fails loudly here
        // (the test matrix keeps high-density cases to a few chunks — see the test).
        let workgroups = total_cells.div_ceil(64);
        let max_dim = device.limits().max_compute_workgroups_per_dimension as usize;
        assert!(
            workgroups <= max_dim,
            "gpu_resolve spike: {workgroups} workgroups exceeds the {max_dim} single-dimension \
             limit; reduce chunk count or density for this case"
        );

        let descriptor_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_resolve descriptor"),
            contents: bytemuck::bytes_of(&descriptor),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let coords_padded: Vec<[i32; 4]> = chunk_coords
            .iter()
            .map(|&[x, y, z]| [x, y, z, 0])
            .collect();
        let coords_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_resolve chunk coords"),
            contents: bytemuck::cast_slice(&coords_padded),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // The profile buffer is always bound (binding 3); SDF cases pass a single dummy
        // vertex so the storage binding is never zero-sized.
        let profile_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_resolve profile"),
            contents: bytemuck::cast_slice(profile_vertices),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_size = (total_cells * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_resolve occupancy"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_resolve readback"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_resolve bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: descriptor_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: coords_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: profile_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_resolve pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
        queue.submit(Some(encoder.finish()));

        // Map and block until ready (headless test path — no frame loop to poll).
        let slice = staging_buffer.slice(..);
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
        let values: &[u32] = bytemuck::cast_slice(&mapped);
        let occupancy: Vec<Vec<u8>> = (0..num_chunks)
            .map(|chunk| {
                let start = chunk * cells_per_chunk;
                values[start..start + cells_per_chunk]
                    .iter()
                    .map(|&v| v as u8)
                    .collect()
            })
            .collect();
        drop(mapped);
        staging_buffer.unmap();

        occupancy
    }
}
