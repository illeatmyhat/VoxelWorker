use super::*;

/// The group(0) camera/frame uniform bind-group layout every cuboid-shader pipeline binds
/// (the solid/ghost draws in [`super::CuboidMeshRenderer::assemble`] and the selected-operand
/// ghost passes, issue #78). ONE builder so the layouts stay bind-compatible.
pub(crate) fn cuboid_uniform_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
    })
}

/// The per-draw on-face-grid overlay-active bind-group layout (group 2, ADR 0003 §3c / ADR
/// 0010 E3): one `u32` uniform read with a DYNAMIC OFFSET, so the overlay-off and overlay-on
/// draws of a chunk select `0` / `1` from a two-entry buffer without a per-vertex flag.
pub(crate) fn overlay_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
pub(crate) fn build_overlay_bind_group(
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
