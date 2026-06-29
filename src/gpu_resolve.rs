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
use crate::debug_clouds::{
    DebugCloudField, CLOUD_EDGE_BILLOW, CLOUD_NOISE_GAIN, CLOUD_NOISE_LACUNARITY,
    CLOUD_NOISE_OCTAVES, CLOUD_NOISE_WAVELENGTH_FRACTION,
};
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
    /// DebugClouds fBm constants: [edge_billow, wavelength_fraction, lacunarity, gain].
    cloud_params: [f32; 4],
    /// 0 = SDF primitive, 1 = sketch extrude, 2 = sketch revolve, 3 = debug clouds.
    producer_type: u32,
    kind: u32,
    wall_voxels: f32,
    turn: f32,
    profile_count: u32,
    is_partial: u32,
    chunk_extent: i32,
    pad: u32,
    num_chunks: u32,
    /// Atlas packing (the `main_atlas` entry only). Zero for the per-chunk A/B path.
    tiles_per_axis: u32,
    atlas_dim: u32,
    padded_row: u32,
    /// DebugClouds: fBm octave count + number of puffs in the `cloud_puffs` buffer.
    cloud_octaves: u32,
    num_puffs: u32,
    _pad0: u32,
    _pad1: u32,
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
            cloud_params: [0.0; 4],
            producer_type: 0,
            kind: 0,
            wall_voxels: 0.0,
            turn: 0.0,
            profile_count: 0,
            is_partial: 0,
            chunk_extent,
            pad: (chunk_extent + 2) as u32,
            num_chunks,
            tiles_per_axis: 0,
            atlas_dim: 0,
            padded_row: 0,
            cloud_octaves: 0,
            num_puffs: 0,
            _pad0: 0,
            _pad1: 0,
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

/// `copy_buffer_to_texture` requires each buffer row be a multiple of this (wgpu's
/// `COPY_BYTES_PER_ROW_ALIGNMENT`), so the packed atlas buffer pads its rows to it.
const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

/// Placeholder bindings 3/4/5 for producers that don't use a given input (the layout
/// always binds all three, so no storage binding is ever zero-sized).
const DUMMY_PROFILE: &[[i32; 2]] = &[[0, 0]];
const DUMMY_CLOUDS: &[[f32; 4]] = &[[0.0; 4]];
const DUMMY_PERM: &[u32] = &[0];

/// The producer-specific input buffers bound at bindings 3 (profile), 4 (cloud puffs),
/// and 5 (cloud permutation). Each producer fills the one(s) it uses and leaves the
/// rest as dummies — bundled so the dispatch helpers stay under the argument limit.
struct ProducerInputs<'a> {
    profile: &'a [[i32; 2]],
    cloud_puffs: &'a [[f32; 4]],
    cloud_perm: &'a [u32],
}

impl<'a> ProducerInputs<'a> {
    fn sdf() -> Self {
        Self { profile: DUMMY_PROFILE, cloud_puffs: DUMMY_CLOUDS, cloud_perm: DUMMY_PERM }
    }
    fn sketch(profile: &'a [[i32; 2]]) -> Self {
        Self { profile, cloud_puffs: DUMMY_CLOUDS, cloud_perm: DUMMY_PERM }
    }
    fn clouds(cloud_puffs: &'a [[f32; 4]], cloud_perm: &'a [u32]) -> Self {
        Self { profile: DUMMY_PROFILE, cloud_puffs, cloud_perm }
    }
}

/// The GPU-packed fog atlas: the R8 texture produced via `copy_buffer_to_texture`,
/// plus its bytes read back (unpadded `atlas_dim³`) for the A/B assertion, and the
/// tile geometry so a caller can compare against `upload_grid_per_chunk`'s packing.
pub struct AtlasResult {
    /// The R8Unorm 3D atlas the per-chunk fog raymarch samples.
    pub texture: wgpu::Texture,
    /// `atlas_dim³` occupancy bytes (0/255), read back from `texture`, row-unpadded.
    pub atlas: Vec<u8>,
    /// `tiles_per_axis * pad` — the atlas cube dimension per axis.
    pub atlas_dim: u32,
    /// Resident chunk tiles per atlas axis (`ceil(cbrt(chunk_count))`).
    pub tiles_per_axis: u32,
    /// `chunk_extent + 2` — the apron'd per-axis tile span.
    pub pad: u32,
}

/// Holds the compute pipelines + bind-group layout so a test can resolve many cases
/// against one device without rebuilding the pipelines each call.
pub struct GpuResolver {
    /// The A/B entry: one u32 (0/255) per apron cell, per-chunk-linear order.
    pipeline: wgpu::ComputePipeline,
    /// The atlas entry: packed occupancy bytes in the `upload_grid_per_chunk` atlas
    /// layout, ready for `copy_buffer_to_texture`.
    atlas_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuResolver {
    /// Build the compute pipelines from `shaders/gpu_resolve.wgsl`.
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
                    // 3: sketch profile vertices (read-only storage; dummy otherwise)
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
                    // 4: cloud puffs (read-only storage; dummy otherwise)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 5: cloud permutation table (read-only storage; dummy otherwise)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
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
        let atlas_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_resolve atlas pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main_atlas"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            atlas_pipeline,
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
        let descriptor = Self::sdf_descriptor(shape, voxels_per_block, chunk_coords.len() as u32);
        self.dispatch(device, queue, descriptor, chunk_coords, ProducerInputs::sdf())
    }

    /// As [`resolve_sdf_occupancy`](Self::resolve_sdf_occupancy), but packs the result
    /// into the `upload_grid_per_chunk` atlas via `copy_buffer_to_texture` and returns
    /// the `atlas_dim³`-byte atlas read back from the R8 texture (the production
    /// texture-write mechanic, under the A/B net).
    pub fn resolve_sdf_atlas(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shape: &SdfShape,
        voxels_per_block: u32,
        chunk_coords: &[[i32; 3]],
    ) -> AtlasResult {
        let descriptor = Self::sdf_descriptor(shape, voxels_per_block, chunk_coords.len() as u32);
        self.dispatch_atlas(device, queue, descriptor, chunk_coords, ProducerInputs::sdf())
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
        let (descriptor, profile) =
            Self::sketch_descriptor(solid, voxels_per_block, chunk_coords.len() as u32);
        self.dispatch(device, queue, descriptor, chunk_coords, ProducerInputs::sketch(&profile))
    }

    /// As [`resolve_sketch_occupancy`](Self::resolve_sketch_occupancy), but packs the
    /// result into the atlas texture (see [`resolve_sdf_atlas`](Self::resolve_sdf_atlas)).
    pub fn resolve_sketch_atlas(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        solid: &SketchSolid,
        voxels_per_block: u32,
        chunk_coords: &[[i32; 3]],
    ) -> AtlasResult {
        let (descriptor, profile) =
            Self::sketch_descriptor(solid, voxels_per_block, chunk_coords.len() as u32);
        self.dispatch_atlas(device, queue, descriptor, chunk_coords, ProducerInputs::sketch(&profile))
    }

    /// GPU-evaluate the apron'd occupancy of a [`DebugCloudField`] at document density
    /// `voxels_per_block`, for each chunk in `chunk_coords` (same contract as
    /// [`resolve_sdf_occupancy`](Self::resolve_sdf_occupancy)).
    pub fn resolve_clouds_occupancy(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        field: &DebugCloudField,
        voxels_per_block: u32,
        chunk_coords: &[[i32; 3]],
    ) -> Vec<Vec<u8>> {
        let (descriptor, puffs, perm) =
            Self::cloud_descriptor(field, voxels_per_block, chunk_coords.len() as u32);
        self.dispatch(device, queue, descriptor, chunk_coords, ProducerInputs::clouds(&puffs, &perm))
    }

    /// As [`resolve_clouds_occupancy`](Self::resolve_clouds_occupancy), but packs the
    /// result into the atlas texture (see [`resolve_sdf_atlas`](Self::resolve_sdf_atlas)).
    pub fn resolve_clouds_atlas(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        field: &DebugCloudField,
        voxels_per_block: u32,
        chunk_coords: &[[i32; 3]],
    ) -> AtlasResult {
        let (descriptor, puffs, perm) =
            Self::cloud_descriptor(field, voxels_per_block, chunk_coords.len() as u32);
        self.dispatch_atlas(device, queue, descriptor, chunk_coords, ProducerInputs::clouds(&puffs, &perm))
    }

    /// Build the SDF producer descriptor (atlas fields left zero).
    fn sdf_descriptor(shape: &SdfShape, voxels_per_block: u32, num_chunks: u32) -> Descriptor {
        let voxels_per_block = voxels_per_block.max(1);
        let grid = shape.grid_dimensions(voxels_per_block);
        let chunk_extent = (CHUNK_BLOCKS * voxels_per_block) as i32;
        let mut descriptor = Descriptor::base(grid, chunk_extent, num_chunks);
        descriptor.producer_type = 0;
        descriptor.kind = shape_kind_discriminant(shape.kind);
        descriptor.wall_voxels = (shape.wall_blocks * voxels_per_block) as f32;
        descriptor
    }

    /// Build the sketch producer descriptor + its profile-vertex buffer contents.
    fn sketch_descriptor(
        solid: &SketchSolid,
        voxels_per_block: u32,
        num_chunks: u32,
    ) -> (Descriptor, Vec<[i32; 2]>) {
        let voxels_per_block = voxels_per_block.max(1);
        let grid = solid.grid_dimensions();
        let chunk_extent = (CHUNK_BLOCKS * voxels_per_block) as i32;
        let mut descriptor = Descriptor::base(grid, chunk_extent, num_chunks);

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
        (descriptor, profile_vertices)
    }

    /// Build the DebugClouds descriptor + its puff buffer (2 vec4 per puff) and the
    /// permutation table, computed CPU-side from the field exactly as the CPU resolve
    /// does (so the GPU noise indexes the same table / puffs).
    fn cloud_descriptor(
        field: &DebugCloudField,
        voxels_per_block: u32,
        num_chunks: u32,
    ) -> (Descriptor, Vec<[f32; 4]>, Vec<u32>) {
        let voxels_per_block = voxels_per_block.max(1);
        let chunk_extent = (CHUNK_BLOCKS * voxels_per_block) as i32;
        let mut descriptor = Descriptor::base(field.dimensions, chunk_extent, num_chunks);
        descriptor.producer_type = 3;
        descriptor.cloud_params = [
            CLOUD_EDGE_BILLOW,
            CLOUD_NOISE_WAVELENGTH_FRACTION,
            CLOUD_NOISE_LACUNARITY,
            CLOUD_NOISE_GAIN,
        ];
        descriptor.cloud_octaves = CLOUD_NOISE_OCTAVES;

        let puff_params = field.gpu_puffs();
        descriptor.num_puffs = puff_params.len() as u32;
        let mut puffs: Vec<[f32; 4]> = Vec::with_capacity(puff_params.len() * 2);
        for p in &puff_params {
            puffs.push([p.center[0], p.center[1], p.center[2], p.radius]);
            puffs.push([p.noise_offset[0], p.noise_offset[1], p.noise_offset[2], 0.0]);
        }
        let perm: Vec<u32> = field.permutation_table().iter().map(|&b| b as u32).collect();
        (descriptor, puffs, perm)
    }

    /// Build the buffers + bind group, dispatch, and read the occupancy back, split
    /// into one `pad³` Vec per chunk.
    fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        descriptor: Descriptor,
        chunk_coords: &[[i32; 3]],
        inputs: ProducerInputs,
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

        // Bindings 3/4/5 are always bound; non-matching producers pass single dummies so
        // no storage binding is ever zero-sized.
        let profile_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_resolve profile"),
            contents: bytemuck::cast_slice(inputs.profile),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let cloud_puffs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_resolve cloud puffs"),
            contents: bytemuck::cast_slice(inputs.cloud_puffs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let cloud_perm_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_resolve cloud perm"),
            contents: bytemuck::cast_slice(inputs.cloud_perm),
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
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cloud_puffs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cloud_perm_buffer.as_entire_binding(),
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

    /// Pack the GPU-evaluated occupancy into the `upload_grid_per_chunk` atlas layout as
    /// packed bytes, `copy_buffer_to_texture` it into an R8 atlas, then read the texture
    /// back (row-unpadded) for the A/B assertion. This exercises the production
    /// texture-write mechanic (packed-byte buffer → 256-padded rows → R8 atlas).
    fn dispatch_atlas(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mut descriptor: Descriptor,
        chunk_coords: &[[i32; 3]],
        inputs: ProducerInputs,
    ) -> AtlasResult {
        let pad = descriptor.pad;
        let cells_per_chunk = (pad * pad * pad) as usize;
        let num_chunks = chunk_coords.len();

        // Tile geometry, identical to `upload_grid_per_chunk`.
        let tiles_per_axis = ((num_chunks as f64).cbrt().ceil() as u32).max(1);
        let atlas_dim = tiles_per_axis * pad;
        let padded_row = atlas_dim.div_ceil(COPY_BYTES_PER_ROW_ALIGNMENT) * COPY_BYTES_PER_ROW_ALIGNMENT;
        descriptor.tiles_per_axis = tiles_per_axis;
        descriptor.atlas_dim = atlas_dim;
        descriptor.padded_row = padded_row;

        if num_chunks == 0 {
            return AtlasResult {
                texture: create_empty_atlas(device),
                atlas: vec![0u8; (atlas_dim as usize).pow(3)],
                atlas_dim,
                tiles_per_axis,
                pad,
            };
        }
        let workgroups = (cells_per_chunk * num_chunks).div_ceil(64);
        let max_dim = device.limits().max_compute_workgroups_per_dimension as usize;
        assert!(
            workgroups <= max_dim,
            "gpu_resolve atlas: {workgroups} workgroups exceeds the {max_dim} single-dimension limit"
        );

        // The packed-byte buffer (256-padded rows), as `atomic<u32>` words in the shader.
        let padded_bytes = padded_row as usize * atlas_dim as usize * atlas_dim as usize;

        let descriptor_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_resolve atlas descriptor"),
            contents: bytemuck::bytes_of(&descriptor),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let coords_padded: Vec<[i32; 4]> =
            chunk_coords.iter().map(|&[x, y, z]| [x, y, z, 0]).collect();
        let coords_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_resolve atlas coords"),
            contents: bytemuck::cast_slice(&coords_padded),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let profile_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_resolve atlas profile"),
            contents: bytemuck::cast_slice(inputs.profile),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let cloud_puffs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_resolve atlas cloud puffs"),
            contents: bytemuck::cast_slice(inputs.cloud_puffs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let cloud_perm_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_resolve atlas cloud perm"),
            contents: bytemuck::cast_slice(inputs.cloud_perm),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let packed_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_resolve packed atlas"),
            size: padded_bytes as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gpu_resolve atlas texture"),
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
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_resolve atlas readback"),
            size: padded_bytes as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_resolve atlas bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: descriptor_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: coords_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: packed_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: profile_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cloud_puffs_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: cloud_perm_buffer.as_entire_binding() },
            ],
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // The shader ORs occupancy bytes into a zero buffer, so clear it first.
        encoder.clear_buffer(&packed_buffer, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_resolve atlas pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.atlas_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }
        let copy_layout = wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(padded_row),
            rows_per_image: Some(atlas_dim),
        };
        let extent = wgpu::Extent3d {
            width: atlas_dim,
            height: atlas_dim,
            depth_or_array_layers: atlas_dim,
        };
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo { buffer: &packed_buffer, layout: copy_layout },
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            extent,
        );
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo { buffer: &readback, layout: copy_layout },
            extent,
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

        // Unpad the 256-aligned rows back to a dense `atlas_dim³` byte cube.
        let mapped = slice.get_mapped_range();
        let atlas_dim_usize = atlas_dim as usize;
        let padded_row_usize = padded_row as usize;
        let mut atlas = vec![0u8; atlas_dim_usize.pow(3)];
        for az in 0..atlas_dim_usize {
            for ay in 0..atlas_dim_usize {
                let src = (az * atlas_dim_usize + ay) * padded_row_usize;
                let dst = (az * atlas_dim_usize + ay) * atlas_dim_usize;
                atlas[dst..dst + atlas_dim_usize]
                    .copy_from_slice(&mapped[src..src + atlas_dim_usize]);
            }
        }
        drop(mapped);
        readback.unmap();

        AtlasResult {
            texture,
            atlas,
            atlas_dim,
            tiles_per_axis,
            pad,
        }
    }
}

/// A 1×1×1 R8 atlas for the degenerate (zero-chunk) atlas result.
fn create_empty_atlas(device: &wgpu::Device) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gpu_resolve empty atlas"),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}
