use super::*;

/// Edge length of every procedural material texture (square, no mipmaps).
const MATERIAL_TEXTURE_SIZE: u32 = 32;

// (The instanced renderer's `VoxelUniforms` struct + `voxel.wgsl` shader were
// removed with the legacy mesher — part of #20. The cuboid path uses its own
// `CuboidUniforms`.)

/// The visible layer band (issue #12), in voxel Z-layer indices (Z-up: layers are
/// Z-slices), passed to the mesh band clip. The band is INCLUSIVE on both ends:
/// layers `[band_min, band_max]` render solid. `onion_depth` is the number of layers
/// OUTSIDE the band that render ghosted (screen-door dither); `0` = a hard clip.
///
/// Pass [`LayerBand::FULL`] (or any band whose `band_max >= grid_z - 1` and
/// `band_min == 0`) to draw the whole model unclipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerBand {
    pub band_min: u32,
    pub band_max: u32,
    pub onion_depth: u32,
}

impl LayerBand {
    /// An effectively-unbounded band (the whole grid, no onion skin). `band_max`
    /// is huge so no layer is ever clipped regardless of `grid_z`.
    pub const FULL: LayerBand = LayerBand {
        band_min: 0,
        band_max: u32::MAX,
        onion_depth: 0,
    };
}

// (The instanced `VoxelRenderer` + its per-chunk GPU instance cache were removed
// with the legacy mesher — part of #20. The cuboid path is the sole renderer.)

/// Which texture the voxel pass binds for the active material.
///
/// `Procedural` selects one of the built-in Stone/Wood/Plain textures;
/// `Loaded` overrides with a runtime-loaded VS block's bind group (M6). Both use
/// the identical pipeline + per-voxel slice shader.
#[derive(Clone, Copy)]
pub enum MaterialSource<'a> {
    Procedural(MaterialChoice),
    Loaded(&'a wgpu::BindGroup),
}

/// Build the 6-layer face-material bind-group layout (M7): a `D2Array` texture
/// (binding 0, one layer per cube face) + a sampler (binding 1). Both the
/// procedural materials and a loaded VS block build a bind group of this shape,
/// so the single voxel pipeline draws uniform and per-face materials alike.
pub fn build_face_material_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("voxel face material bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
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

/// Upload six RGBA8 sRGB layers (one per cube face) as a single `D2Array`
/// texture (nearest filter, clamp-to-edge, no mipmaps). Every layer must be the
/// same `width`×`height`; callers that have per-face PNGs of differing sizes
/// rescale to a common size first (see `block_palette::upload_face_layers`).
pub fn upload_face_material_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    layers: &[&[u8]; 6],
) -> wgpu::Texture {
    let width = width.max(1);
    let height = height.max(1);
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 6,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("voxel face material texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // sRGB so the GPU decodes samples to linear; lighting + the grid overlay
        // then run in linear space and the sRGB target re-encodes on write.
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (layer_index, layer_pixels) in layers.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer_index as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            layer_pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }
    texture
}

/// A small deterministic value-noise generator so the procedural textures are
/// stable across runs (the prototype used `Math.random`; we want reproducible
/// screenshots). Returns a float in `[0, 1)`.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_unit(&mut self) -> f32 {
        // Numerical Recipes LCG constants.
        self.state = self.state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.state >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// Pack three components into an opaque RGBA8 pixel (alpha = 255).
fn rgba(r: f32, g: f32, b: f32) -> [u8; 4] {
    [
        r.clamp(0.0, 255.0) as u8,
        g.clamp(0.0, 255.0) as u8,
        b.clamp(0.0, 255.0) as u8,
        255,
    ]
}

/// Stone: 32×32 grey ~rgb(132,126,118) with ±20 per-pixel noise + darker speckles.
/// Port of `makeStone` (chisel-bench-reference.html).
fn generate_stone_texture() -> Vec<u8> {
    let mut rng = Lcg::new(0x5701_3a9f);
    let count = (MATERIAL_TEXTURE_SIZE * MATERIAL_TEXTURE_SIZE) as usize;
    let mut pixels = vec![0u8; count * 4];
    // The prototype iterates i (x) outer, j (y) inner, filling column-major; the
    // exact per-pixel correspondence is cosmetic noise, so we fill row-major.
    for pixel in pixels.chunks_exact_mut(4) {
        let noise = 132.0 + (rng.next_unit() * 40.0 - 20.0).floor();
        pixel.copy_from_slice(&rgba(noise, noise - 6.0, noise - 14.0));
    }
    // ~22 darker speckles.
    for _ in 0..22 {
        let x = (rng.next_unit() * MATERIAL_TEXTURE_SIZE as f32) as u32;
        let y = (rng.next_unit() * MATERIAL_TEXTURE_SIZE as f32) as u32;
        let dark = 90.0 + (rng.next_unit() * 30.0).floor();
        let index = ((y.min(MATERIAL_TEXTURE_SIZE - 1) * MATERIAL_TEXTURE_SIZE
            + x.min(MATERIAL_TEXTURE_SIZE - 1))
            * 4) as usize;
        pixels[index..index + 4].copy_from_slice(&rgba(dark, dark - 8.0, dark - 16.0));
    }
    pixels
}

/// Wood: 32×32 brown base with a horizontal sine grain + per-pixel noise.
/// Port of `makeWood` (chisel-bench-reference.html).
fn generate_wood_texture() -> Vec<u8> {
    let mut rng = Lcg::new(0x00c0_ffee);
    let mut pixels = Vec::with_capacity((MATERIAL_TEXTURE_SIZE * MATERIAL_TEXTURE_SIZE * 4) as usize);
    for row in 0..MATERIAL_TEXTURE_SIZE {
        let grain = (row as f32 * 0.9).sin() * 10.0 + (rng.next_unit() * 10.0 - 5.0);
        for _ in 0..MATERIAL_TEXTURE_SIZE {
            let red = 120.0 + grain + (rng.next_unit() * 8.0 - 4.0);
            pixels.extend_from_slice(&rgba(red.floor(), (red * 0.62).floor(), (red * 0.34).floor()));
        }
    }
    pixels
}

/// Plain: flat warm grey `#b6a079`. Port of `makePlain`.
fn generate_plain_texture() -> Vec<u8> {
    let count = (MATERIAL_TEXTURE_SIZE * MATERIAL_TEXTURE_SIZE) as usize;
    let mut pixels = Vec::with_capacity(count * 4);
    for _ in 0..count {
        pixels.extend_from_slice(&[0xb6, 0xa0, 0x79, 0xff]);
    }
    pixels
}

/// The average RGBA colour of a procedural material's texture — the
/// representative palette colour used by the `.vox` export (M8). A loaded VS
/// block can supply its own average instead; this covers the procedural case.
pub fn procedural_material_average_color(material: MaterialChoice) -> [u8; 4] {
    let pixels = match material {
        MaterialChoice::Stone => generate_stone_texture(),
        MaterialChoice::Wood => generate_wood_texture(),
        MaterialChoice::Plain => generate_plain_texture(),
    };
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

/// The average colour of a material's procedural texture as a LINEAR `[r, g, b]`
/// (the space the shader lights/blends in). Indexed by `material_id` order
/// (Stone/Wood/Plain) via [`MaterialChoice::from_material_id`].
fn material_average_linear(id: u16) -> [f32; 3] {
    let srgb = procedural_material_average_color(MaterialChoice::from_material_id(id));
    [
        srgb_component_to_linear(srgb[0]),
        srgb_component_to_linear(srgb[1]),
        srgb_component_to_linear(srgb[2]),
    ]
}

/// The per-voxel material base colours (ADR 0001 step 3) RELATIVE to the bound
/// texture's own average colour. Slot `id` holds `avg(id) / avg(bound)`, so:
///   * the bound material's own slot is ~`[1,1,1]` (neutral — its texture is
///     shown unchanged, preserving the existing look for a single-material model);
///   * every other material's slot recolours the shared bound texture toward that
///     material's tint, so a Wood node and a Stone node drawn from one bound
///     texture render in visibly distinct colours.
///
/// This is the cheap base-colour-modulation the ADR/task call for, NOT a
/// per-material texture array.
fn relative_material_base_colors(
    bound: MaterialChoice,
) -> [[f32; 4]; MaterialChoice::MATERIAL_COUNT] {
    let bound_avg = material_average_linear(bound.material_id());
    let mut colors = [[1.0, 1.0, 1.0, 0.0]; MaterialChoice::MATERIAL_COUNT];
    for (id, slot) in colors.iter_mut().enumerate() {
        let avg = material_average_linear(id as u16);
        // Guard against a near-zero bound channel (a flat black texture); fall back
        // to a neutral 1.0 so a divide can't explode.
        for axis in 0..3 {
            slot[axis] = if bound_avg[axis] > 1e-4 {
                avg[axis] / bound_avg[axis]
            } else {
                1.0
            };
        }
    }
    colors
}

/// Public access to the per-material relative base colours (step 3b) for the cuboid
/// mesh path (ADR 0002 E3b-1), so it modulates per-box material colour. Returns each
/// material's average colour relative to `bound`'s average (the bound material's
/// own slot is ~neutral white).
pub fn relative_material_base_colors_public(
    bound: MaterialChoice,
) -> [[f32; 4]; MaterialChoice::MATERIAL_COUNT] {
    relative_material_base_colors(bound)
}

/// The grid-overlay tuning the instanced voxel pass uses, exposed so the
/// flag-gated cuboid mesh path (ADR 0002 E3b-2) draws the position-based grid
/// overlay with the EXACT same colours/half-widths/alphas — keeping the merged
/// box faces phase-aligned to the same per-voxel/per-block lines.
#[derive(Debug, Clone, Copy)]
pub struct GridOverlayParams {
    pub voxel_line_color: [f32; 3],
    pub block_line_color: [f32; 3],
    pub voxel_line_half_width: f32,
    pub block_line_half_width: f32,
    pub voxel_line_alpha: f32,
    pub block_line_alpha: f32,
}

/// The instanced path's grid-overlay parameters (colours in LINEAR space, the
/// same the voxel shader receives), for the cuboid path to reuse verbatim.
pub fn grid_overlay_params() -> GridOverlayParams {
    GridOverlayParams {
        voxel_line_color: srgb_hex_to_linear(VOXEL_LINE_COLOR_HEX),
        block_line_color: srgb_hex_to_linear(BLOCK_LINE_COLOR_HEX),
        voxel_line_half_width: VOXEL_LINE_HALF_WIDTH,
        block_line_half_width: BLOCK_LINE_HALF_WIDTH,
        voxel_line_alpha: VOXEL_LINE_ALPHA,
        block_line_alpha: BLOCK_LINE_ALPHA,
    }
}

/// Generate the three procedural material textures (Stone/Wood/Plain) as RGBA8
/// sRGB pixel buffers, in `MaterialChoice` order, so the cuboid path (E3b-2) can
/// upload the SAME procedural textures the instanced path binds.
pub fn procedural_material_pixels() -> [Vec<u8>; 3] {
    [
        generate_stone_texture(),
        generate_wood_texture(),
        generate_plain_texture(),
    ]
}

/// The edge length of every procedural material texture (square), exposed so the
/// cuboid path uploads them at the matching size.
pub fn procedural_material_texture_size() -> u32 {
    MATERIAL_TEXTURE_SIZE
}
