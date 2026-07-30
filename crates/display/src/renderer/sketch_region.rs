//! The sketch region wash renderer (ADR 0030 §3): the resolved 2D material region shaded ON the
//! sketch plane, as a field rather than a fill.
//!
//! A fullscreen triangle whose fragment intersects the sketch plane and evaluates
//! `substrate::geom2d::signed_distance_to_region` there — the shader is a hand-written mirror, and
//! it is handed the SAME `(LoopRole, Vec<[f32; 2]>)` value the resolve folds, so there is one
//! definition of the region and two evaluators of it. Drawn INSIDE the existing MSAA voxel pass
//! with depth compare `Always`, like the placement ghost: the wash is an authoring affordance on
//! the plane, and the solid it describes stands in front of that plane, so depth-testing it would
//! hide it exactly when it matters.
//!
//! Self-gating: [`draw`](SketchRegionRenderer::draw) is a no-op until
//! [`update`](SketchRegionRenderer::update) uploads a region.

use super::*;
use substrate::geom2d::LoopRole;

/// std140 uniform for the wash; field order matches `SketchRegionUniforms` in
/// `sketch_region.wgsl` **byte-for-byte**.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SketchRegionUniforms {
    ray_inverse_unprojection: [[f32; 4]; 4],
    ray_eye: [f32; 4],
    viewport: [f32; 4],
    plane_origin: [f32; 4],
    plane_axis0: [f32; 4],
    plane_axis1: [f32; 4],
    plane_normal: [f32; 4],
    tint: [f32; 4],
    bounds: [f32; 4],
    counts: [u32; 4],
}

/// One loop's slice of the point buffer, as the shader's `RegionLoop`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct RegionLoopSlot {
    role: u32,
    start: u32,
    count: u32,
    padding: u32,
}

/// The `LoopRole` discriminant the shader switches on. **MUST match `LoopRole`'s declaration
/// order** in `substrate::geom2d` — the one place a hand-written mirror drifts without any
/// distance ever being wrong. The exhaustive `match` makes a new variant a compile error here.
pub fn sketch_region_loop_role_discriminant(role: LoopRole) -> u32 {
    match role {
        LoopRole::Fill => 0,
        LoopRole::Hole => 1,
    }
}

/// How far outside the profile's bounding box the wash still evaluates, in voxels — enough for the
/// antialiasing band at the outermost edge to resolve rather than being clipped square.
const BOUNDS_PADDING_VOXELS: f32 = 8.0;

/// The sketch plane in the render frame, as the ONE forward map produces it.
///
/// Every field comes from `SketchHandles::profile_to_render`, so the wash lands on the plane the
/// vertex handles land on by construction rather than by a kept-in-sync mirror (ADR 0008).
#[derive(Debug, Clone, Copy)]
pub struct SketchPlaneFrame {
    /// The render-frame position of profile coordinate `(0, 0)`.
    pub origin: [f32; 3],
    /// The render-frame displacement of `+1` voxel along the profile's first in-plane axis.
    pub axis0: [f32; 3],
    /// The same for the second in-plane axis.
    pub axis1: [f32; 3],
    /// The plane's unit normal in the render frame.
    pub normal: [f32; 3],
}

/// The wash pass over the sketch plane.
pub struct SketchRegionRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    loop_buffer: wgpu::Buffer,
    point_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Loops the storage buffer can hold before it must grow.
    loop_capacity: usize,
    /// Points the storage buffer can hold before it must grow.
    point_capacity: usize,
    /// Whether a region was uploaded. `false` after `new`; set by `update`, cleared by `disarm`.
    armed: bool,
}

/// The smallest buffer either storage array is created at — a zero-sized storage buffer is invalid,
/// and starting non-trivial means the common sketch never reallocates.
const INITIAL_CAPACITY: usize = 256;

impl SketchRegionRenderer {
    /// Create the wash renderer for a colour target. It starts DISARMED.
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sketch region shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sketch_region.wgsl").into()),
        });

        let storage_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sketch region bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                            SketchRegionUniforms,
                        >() as u64),
                    },
                    count: None,
                },
                storage_entry(1),
                storage_entry(2),
            ],
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sketch region uniforms"),
            size: std::mem::size_of::<SketchRegionUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let loop_buffer = storage_buffer::<RegionLoopSlot>(device, "loops", INITIAL_CAPACITY);
        let point_buffer = storage_buffer::<[f32; 2]>(device, "points", INITIAL_CAPACITY);
        let bind_group = bind_region(
            device,
            &bind_group_layout,
            &uniform_buffer,
            &loop_buffer,
            &point_buffer,
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sketch region pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sketch region pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    // The shader outputs PREMULTIPLIED colour (`tint.rgb * alpha`).
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            // Depth compare `Always`, write off — the placement ghost's reasoning, for the same
            // reason: the wash marks the authoring surface, and the solid the sketch produces
            // stands on that surface, so a depth test would hide the wash under the very geometry
            // it is describing. Write stays off so a translucent overlay never occludes anything.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
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

        Self {
            pipeline,
            bind_group_layout,
            uniform_buffer,
            loop_buffer,
            point_buffer,
            bind_group,
            loop_capacity: INITIAL_CAPACITY,
            point_capacity: INITIAL_CAPACITY,
            armed: false,
        }
    }

    /// Arm and upload this frame's region. `region` is the profile's `Fill`/`Hole` loops in PROFILE
    /// voxel coordinates — the same value `signed_distance_to_region` folds on the CPU. `tint` is
    /// LINEAR RGB + source alpha.
    ///
    /// A region with no loops disarms instead, so an empty sketch draws nothing.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ray_inverse_unprojection: glam::Mat4,
        ray_eye: glam::Vec3,
        viewport_px: [u32; 4],
        plane: SketchPlaneFrame,
        region: &[(LoopRole, Vec<[f32; 2]>)],
        tint: [f32; 4],
    ) {
        let total_points: usize = region.iter().map(|(_, points)| points.len()).sum();
        if region.is_empty() || total_points == 0 {
            self.disarm();
            return;
        }
        let mut slots = Vec::with_capacity(region.len());
        let mut points: Vec<[f32; 2]> = Vec::with_capacity(total_points);
        let mut bounds = ([f32::MAX, f32::MAX], [f32::MIN, f32::MIN]);
        for (role, loop_points) in region {
            slots.push(RegionLoopSlot {
                role: sketch_region_loop_role_discriminant(*role),
                start: points.len() as u32,
                count: loop_points.len() as u32,
                padding: 0,
            });
            for point in loop_points {
                for ((low, high), coordinate) in bounds
                    .0
                    .iter_mut()
                    .zip(bounds.1.iter_mut())
                    .zip(point.iter())
                {
                    *low = low.min(*coordinate);
                    *high = high.max(*coordinate);
                }
            }
            points.extend_from_slice(loop_points);
        }

        // Grow either storage buffer if this region outgrew it, then rebind — a bind group holds
        // the buffer it was built against, so a new buffer needs a new group.
        let mut rebind = false;
        if slots.len() > self.loop_capacity {
            self.loop_capacity = slots.len().next_power_of_two();
            self.loop_buffer =
                storage_buffer::<RegionLoopSlot>(device, "loops", self.loop_capacity);
            rebind = true;
        }
        if points.len() > self.point_capacity {
            self.point_capacity = points.len().next_power_of_two();
            self.point_buffer = storage_buffer::<[f32; 2]>(device, "points", self.point_capacity);
            rebind = true;
        }
        if rebind {
            self.bind_group = bind_region(
                device,
                &self.bind_group_layout,
                &self.uniform_buffer,
                &self.loop_buffer,
                &self.point_buffer,
            );
        }
        queue.write_buffer(&self.loop_buffer, 0, bytemuck::cast_slice(&slots));
        queue.write_buffer(&self.point_buffer, 0, bytemuck::cast_slice(&points));

        let uniforms = SketchRegionUniforms {
            ray_inverse_unprojection: ray_inverse_unprojection.to_cols_array_2d(),
            ray_eye: [ray_eye.x, ray_eye.y, ray_eye.z, 0.0],
            viewport: [
                viewport_px[0] as f32,
                viewport_px[1] as f32,
                viewport_px[2] as f32,
                viewport_px[3] as f32,
            ],
            plane_origin: with_padding(plane.origin),
            plane_axis0: with_padding(plane.axis0),
            plane_axis1: with_padding(plane.axis1),
            plane_normal: with_padding(plane.normal),
            tint,
            bounds: [
                bounds.0[0] - BOUNDS_PADDING_VOXELS,
                bounds.0[1] - BOUNDS_PADDING_VOXELS,
                bounds.1[0] + BOUNDS_PADDING_VOXELS,
                bounds.1[1] + BOUNDS_PADDING_VOXELS,
            ],
            counts: [slots.len() as u32, 0, 0, 0],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        self.armed = true;
    }

    /// Disarm the wash (a frame with no sketch open). [`draw`](Self::draw) becomes a no-op again.
    pub fn disarm(&mut self) {
        self.armed = false;
    }

    /// Record the wash into an already-begun (MSAA) pass: one fullscreen triangle. Self-gating.
    pub fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        if !self.armed {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

/// A read-only storage buffer sized for `capacity` elements of `T`.
fn storage_buffer<T>(device: &wgpu::Device, what: &str, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("sketch region {what}")),
        size: (std::mem::size_of::<T>() * capacity.max(1)) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn bind_region(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniforms: &wgpu::Buffer,
    loops: &wgpu::Buffer,
    points: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sketch region bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: loops.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: points.as_entire_binding(),
            },
        ],
    })
}

/// A render-frame vector as the std140 `vec4` the uniform declares; `w` is padding.
fn with_padding(vector: [f32; 3]) -> [f32; 4] {
    [vector[0], vector[1], vector[2], 0.0]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The discriminant order the WGSL mirror switches on MUST match `LoopRole`'s declaration
    /// order — the one place this hand-written mirror drifts without any distance ever being wrong.
    #[test]
    fn discriminant_order_matches_loop_role_declaration() {
        assert_eq!(sketch_region_loop_role_discriminant(LoopRole::Fill), 0);
        assert_eq!(sketch_region_loop_role_discriminant(LoopRole::Hole), 1);
    }

    /// One mat4 (64) + eight vec4 (128) + one vec4<u32> (16) = 208 bytes, a multiple of the
    /// std140 uniform alignment.
    #[test]
    fn uniform_layout_is_std140_sized() {
        assert_eq!(std::mem::size_of::<SketchRegionUniforms>(), 208);
        assert_eq!(std::mem::size_of::<SketchRegionUniforms>() % 16, 0);
    }

    /// The storage slot is four `u32`s, which is what the shader's `RegionLoop` declares.
    #[test]
    fn loop_slot_matches_the_shader_struct() {
        assert_eq!(std::mem::size_of::<RegionLoopSlot>(), 16);
    }
}
