//! The scene block-material GPU texture + its bind-group layout.
//!
//! This is the pure-wgpu backing for an APPLIED Vintage Story block on the scene
//! renderers. It owns:
//!
//!   * [`LoadedMaterial`] — a runtime-loaded RGBA block texture uploaded as a
//!     6-layer texture array (one layer per cube face) bound exactly like the
//!     procedural Stone/Wood/Plain ones, so the per-voxel slice shader textures
//!     the model with a real VS block.
//!   * [`block_texture_bind_group_layout`] — the shared block-texture bind-group
//!     layout shape (binding 0 = texture, binding 1 = sampler), reused by every
//!     scene material AND by the shell's palette-preview thumbnail renderer.
//!
//! The palette PREVIEW thumbnail renderer (which draws the UI's little 45° cube
//! tiles, not the scene) lives in the shell crate's `thumbnail` module; it reaches
//! down here only for [`block_texture_bind_group_layout`].

use assets::DecodedRgba;

/// A runtime-loaded block material: a 6-layer texture array (one layer per cube
/// face), bound exactly like the procedural materials so the per-voxel slice
/// shader treats it identically. A uniform block puts the same image on all six
/// layers; a per-face block (M7) puts each face's PNG on its own layer.
pub struct LoadedMaterial {
    pub bind_group: wgpu::BindGroup,
    /// The label of the applied block (for the panel readout).
    pub label: String,
    /// Whether the resolved faces are genuinely per-face (top != side); used for
    /// logging / verification (`--list-perface`, `--apply-block`).
    pub is_per_face: bool,
    /// Average RGBA colour of the side face — the representative palette colour
    /// used by the `.vox` export (M8).
    pub average_color: [u8; 4],
}

impl LoadedMaterial {
    /// Build a 6-layer material DIRECTLY from six raw RGBA8 face buffers (part of
    /// #20 verification / synthetic blocks). Each `layers[i]` is a tightly-packed
    /// `width*height*4` RGBA8 buffer in CubeFaceSlot order (0 +X, 1 -X, 2 +Y, 3 -Y,
    /// 4 +Z, 5 -Z); the texture is uploaded as the SAME sRGB D2Array + bind-group
    /// shape `from_faces` produces, so it is interchangeable on both render paths
    /// without needing a real VS install. Used by the headless harness to apply six
    /// distinct solid-colour faces and prove the cuboid path textures per-face.
    #[allow(clippy::too_many_arguments)]
    pub fn from_face_layers(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        material_bind_group_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        width: u32,
        height: u32,
        layers: &[&[u8]; 6],
        label: String,
    ) -> Self {
        let average_color = average_rgba(layers[0]);
        let texture = crate::renderer::upload_face_material_texture(
            device, queue, width, height, layers,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("synthetic loaded block material bind group"),
            layout: material_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        Self {
            bind_group,
            label,
            is_per_face: true,
            average_color,
        }
    }

    /// Build a 6-layer material from resolved per-face PNG paths (M7).
    ///
    /// Each face PNG is decoded and, if face sizes differ, rescaled to the
    /// largest common size so all six layers share one `Extent3d`. The texture is
    /// sRGB so the main shader's lighting + grid overlay mix in linear space,
    /// matching Stone/Wood/Plain. Falls back to a uniform layer set from any face
    /// that fails to decode.
    pub fn from_faces(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        material_bind_group_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        faces: &assets::FaceTextures,
        label: String,
    ) -> Self {
        let is_per_face = !faces.is_uniform();

        // Decode each face PNG (in CubeFaceSlot layer order). A face that fails
        // to decode is left None and filled from a sibling below.
        let decoded_faces: Vec<Option<DecodedRgba>> = faces
            .paths
            .iter()
            .map(|path| assets::decode_rgba(path))
            .collect();

        // Pick a representative decoded face to size the array + fill gaps.
        let representative = decoded_faces
            .iter()
            .flatten()
            .next()
            .cloned()
            .unwrap_or((1, 1, vec![0u8, 0u8, 0u8, 255u8]));
        let (target_width, target_height, _) = representative;

        // Normalise every layer to (target_width, target_height) so the array's
        // layers share one size. Missing faces use the representative image.
        let layers_rgba: Vec<Vec<u8>> = decoded_faces
            .into_iter()
            .map(|face| {
                let decoded = face.unwrap_or_else(|| representative.clone());
                resize_rgba_nearest(&decoded, target_width, target_height)
            })
            .collect();

        // Representative palette colour for the .vox export: the average of the
        // side face (layer 0), which is the most representative of the block.
        let average_color = average_rgba(&layers_rgba[0]);

        let layer_slices: [&[u8]; 6] = [
            &layers_rgba[0],
            &layers_rgba[1],
            &layers_rgba[2],
            &layers_rgba[3],
            &layers_rgba[4],
            &layers_rgba[5],
        ];
        let texture = crate::renderer::upload_face_material_texture(
            device,
            queue,
            target_width,
            target_height,
            &layer_slices,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("loaded block material bind group"),
            layout: material_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        Self {
            bind_group,
            label,
            is_per_face,
            average_color,
        }
    }
}

/// Average RGBA of a tightly-packed RGBA8 buffer (alpha forced opaque). Used for
/// the `.vox` export's representative palette colour (M8).
fn average_rgba(pixels: &[u8]) -> [u8; 4] {
    if pixels.len() < 4 {
        return [0x80, 0x80, 0x80, 0xff];
    }
    let mut sums = [0u64; 3];
    let count = (pixels.len() / 4) as u64;
    for pixel in pixels.chunks_exact(4) {
        sums[0] += pixel[0] as u64;
        sums[1] += pixel[1] as u64;
        sums[2] += pixel[2] as u64;
    }
    let count = count.max(1);
    [
        (sums[0] / count) as u8,
        (sums[1] / count) as u8,
        (sums[2] / count) as u8,
        255,
    ]
}

/// Nearest-neighbour rescale of a decoded RGBA image to `target_width`×
/// `target_height`. A no-op (clone) when the size already matches. Nearest keeps
/// VS block textures crisp and matches the material sampler's filter.
fn resize_rgba_nearest(decoded: &DecodedRgba, target_width: u32, target_height: u32) -> Vec<u8> {
    let (width, height, ref pixels) = *decoded;
    if width == target_width && height == target_height {
        return pixels.clone();
    }
    let width = width.max(1);
    let height = height.max(1);
    let mut out = vec![0u8; (target_width * target_height * 4) as usize];
    for y in 0..target_height {
        let source_y = (y * height / target_height.max(1)).min(height - 1);
        for x in 0..target_width {
            let source_x = (x * width / target_width.max(1)).min(width - 1);
            let source_index = ((source_y * width + source_x) * 4) as usize;
            let dest_index = ((y * target_width + x) * 4) as usize;
            if source_index + 4 <= pixels.len() {
                out[dest_index..dest_index + 4]
                    .copy_from_slice(&pixels[source_index..source_index + 4]);
            }
        }
    }
    out
}

/// Build the standard block-texture bind group layout (binding 0 = texture,
/// binding 1 = sampler) — the SAME shape the voxel renderer's material layout
/// uses, so a loaded texture is interchangeable with the procedural ones. Also
/// reused by the shell's palette-preview thumbnail renderer.
pub fn block_texture_bind_group_layout(
    device: &wgpu::Device,
    label: &str,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label} block texture layout")),
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
