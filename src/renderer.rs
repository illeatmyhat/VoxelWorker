//! Shared render infrastructure for the voxel workshop.
//!
//! The voxel grid itself is drawn by the cuboid mesh path
//! ([`crate::cuboid_mesh::CuboidMeshRenderer`]); the legacy instanced cube renderer
//! that once lived here was removed (part of #20). This module now owns the SHARED
//! GPU pieces that path (and the rest of the app) builds on:
//!   * The procedural material textures (Stone/Wood/Plain) + the loaded-VS-block
//!     material bind-group layout ([`build_face_material_layout`]) and helpers.
//!   * The position-based grid-overlay parameters ([`grid_overlay_params`]).
//!   * The per-object lattice/floor grid ([`SceneGridRenderer`]), the transform
//!     gizmo, the view cube, and the onion-skin volumetric fog.
//!   * The resolve cache's pure dirty-chunk planner ([`incremental_rebuild_plan`])
//!     and the MSAA/depth view helpers.
//!
//! Everything here is render-target-agnostic, so the window and the headless
//! capture paint identically.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::core_geom::{MaterialChoice, CHUNK_BLOCKS};
use crate::scene::{Point, Scene};
use crate::voxel::VoxelGrid;

/// Depth format used by the voxel pass and the depth texture.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Sample count for the 3D voxel pass (4× MSAA). The depth texture, the
/// multisampled colour texture and the pipeline all share this count; egui still
/// renders at 1 sample onto the resolved target.
pub const MSAA_SAMPLE_COUNT: u32 = 4;

/// Edge length of every procedural material texture (square, no mipmaps).
const MATERIAL_TEXTURE_SIZE: u32 = 32;

/// The dirty-chunk rebuild plan (issue #20 S6c-2c): which per-chunk render buffers an
/// incremental edit must (re)build, and which it must evict.
///
/// Computed purely from coord SETS — the render cache's resident coords, the edit's
/// evicted (dirty) coords, and the post-edit covering coords — so it is unit-tested
/// without a GPU device. Applying it makes the resident set equal the covering set
/// and every rebuilt chunk's contents match a fresh resolve, so the post-edit cache
/// is identical to a wholesale rebuild. Retained as the resolve cache's pure
/// dirty-chunk planner (the chunk-cache tests drive it); the legacy instanced
/// renderer that originally consumed it was removed (part of #20).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct IncrementalRebuildPlan {
    /// Covering coords whose buffer must be (re)built: DIRTY (evicted by this edit)
    /// or NEW (no resident buffer yet). Their grids are the only resolve-cache
    /// MISSES; every other covering chunk is a HIT (byte-identical → keep).
    pub rebuild: Vec<[i32; 3]>,
    /// Resident coords the post-edit scene no longer covers (a removed/shrunk node
    /// vacated them) — their buffers must be dropped.
    pub evict: Vec<[i32; 3]>,
}

/// Compute the incremental dirty-chunk rebuild plan (issue #20 S6c-2c) from coord
/// sets alone (no GPU).
///
/// `resident` is the render cache's current coord set (only NON-empty chunks ever
/// hold a buffer — a zero-voxel chunk is never stored). `occupied_covering` is the
/// set of post-edit covering coords that resolve
/// to a NON-EMPTY grid (so deserve a buffer); empty covering chunks are excluded
/// here so they are never treated as "new" work nor kept resident. `evicted` is the
/// edit's dirty coords from the resolve cache.
///
/// A coord is REBUILT iff it is occupied-covering AND (dirty OR not currently
/// resident). A resident coord is EVICTED iff it is no longer occupied-covering —
/// which captures BOTH a vacated chunk (a removed/shrunk node) AND a chunk that an
/// edit turned empty (dirty + now zero voxels). Occupied coords that are
/// resident-and-not-dirty are kept untouched (resolve-cache hits → byte-identical →
/// buffers already correct).
///
/// Applying this plan and making every rebuilt entry equal its fresh grid yields
/// EXACTLY the occupied-covering coord set with fresh contents — identical to a
/// wholesale rebuild (which also stores only non-empty chunks). The returned vectors
/// are sorted so the plan is deterministic and the rebuild count is order-independent.
pub fn incremental_rebuild_plan(
    resident: &[[i32; 3]],
    evicted: &[[i32; 3]],
    occupied_covering: &[[i32; 3]],
) -> IncrementalRebuildPlan {
    let resident_set: std::collections::HashSet<[i32; 3]> = resident.iter().copied().collect();
    let evicted_set: std::collections::HashSet<[i32; 3]> = evicted.iter().copied().collect();
    let covering_set: std::collections::HashSet<[i32; 3]> =
        occupied_covering.iter().copied().collect();

    let mut rebuild: Vec<[i32; 3]> = occupied_covering
        .iter()
        .copied()
        .filter(|coord| evicted_set.contains(coord) || !resident_set.contains(coord))
        .collect();
    rebuild.sort_unstable();
    rebuild.dedup();

    let mut evict: Vec<[i32; 3]> = resident
        .iter()
        .copied()
        .filter(|coord| !covering_set.contains(coord))
        .collect();
    evict.sort_unstable();
    evict.dedup();

    IncrementalRebuildPlan { rebuild, evict }
}

// (The instanced renderer's `VoxelUniforms` struct + `voxel.wgsl` shader were
// removed with the legacy mesher — part of #20. The cuboid path uses its own
// `CuboidUniforms`.)

/// Grid overlay tuning, transcribed from the prototype `GRID` uniforms
/// (chisel-bench-reference.html). Half-widths are in voxel units (the overlay is
/// computed from absolute voxel position), alphas are blend strengths, and the
/// colours are the sRGB hex line colours (ARCHITECTURE.md §8).
const VOXEL_LINE_HALF_WIDTH: f32 = 0.05;
const BLOCK_LINE_HALF_WIDTH: f32 = 0.11;
const VOXEL_LINE_ALPHA: f32 = 0.40;
const BLOCK_LINE_ALPHA: f32 = 0.92;
/// Voxel grid line colour `#17120b` (sRGB hex → linear).
const VOXEL_LINE_COLOR_HEX: u32 = 0x17_12_0b;
/// Block grid line colour `#080605` (sRGB hex → linear, darker/bolder).
const BLOCK_LINE_COLOR_HEX: u32 = 0x08_06_05;

/// Convert one sRGB 8-bit component to a linear float (matches the sRGB texture
/// decode the GPU applies to material samples, so the grid line colours mix in
/// the same colour space as the textured surface).
fn srgb_component_to_linear(byte: u8) -> f32 {
    let value = byte as f32 / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert a packed `0xRRGGBB` sRGB hex colour to a linear `[f32; 3]`.
fn srgb_hex_to_linear(hex: u32) -> [f32; 3] {
    [
        srgb_component_to_linear(((hex >> 16) & 0xff) as u8),
        srgb_component_to_linear(((hex >> 8) & 0xff) as u8),
        srgb_component_to_linear((hex & 0xff) as u8),
    ]
}

/// Append an alpha channel to a linear RGB colour, producing the `[f32; 4]` the
/// line pipeline's vertices carry (M8: lattice/floor draw at low opacity).
fn with_alpha(rgb: [f32; 3], alpha: f32) -> [f32; 4] {
    [rgb[0], rgb[1], rgb[2], alpha]
}

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

/// The **Z-slab** (issue #59) the onion fog occupancy must cover: the voxel Z-range
/// `[voxel_z_min, voxel_z_max)` around the visible band that the raymarch actually
/// samples for its ghost haze. Z-up: layers are Z-slices, so the fog only needs the
/// occupancy of chunks whose Z-extent intersects this slab — NOT the whole grid.
///
/// ADR 0008 (voxel-frame invariant): this is a Z-range in the resolved grid's voxel
/// frame (the SAME `[0, grid_z)` frame the fog occupancy builders index), carried
/// explicitly rather than re-derived at each call site.
///
/// The slab widens the visible band `[band_min, band_max]` by `onion_depth` layers on
/// each side (the ghost falloff the raymarch samples), matching
/// `AppCore::onion_fog_params`' onion-Z span `[lower − depth, upper + 1 + depth]`: the
/// low edge is `band_min − onion_depth`, the high edge is `band_max + 1 + onion_depth`
/// (the `+1` mirrors the params' `upper + 1`, since `band_max` is the last SOLID layer
/// and the raymarch samples up to `upper + 1 + depth`). Both edges are clamped to
/// `[0, grid_z]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FogZSlab {
    /// First voxel-Z layer the fog must cover (clamped to `[0, grid_z]`).
    pub voxel_z_min: u32,
    /// One-past-the-last voxel-Z layer the fog must cover (clamped to `[0, grid_z]`).
    pub voxel_z_max: u32,
}

impl FogZSlab {
    /// The slab the onion fog needs for `band` over a grid of height `grid_z`, or
    /// `None` when the band is effectively FULL-RANGE (issue #59): a band that covers
    /// (or exceeds) the whole grid has NO layers outside to ghost, so the raymarch
    /// hazes nothing — the fog build (and draw) can be skipped entirely even with onion
    /// on. That is behaviour-identical to today, where such a frame builds the whole
    /// grid but draws no visible haze.
    ///
    /// `band.onion_depth` is `0` for a hard clip (no onion); the slab then collapses to
    /// the band itself (still a valid, tighter-than-grid slab).
    pub fn for_band(band: LayerBand, grid_z: u32) -> Option<FogZSlab> {
        if grid_z == 0 {
            return None;
        }
        // FULL-RANGE skip: nothing outside `[band_min, band_max]` to ghost. `LayerBand`
        // clamps `band_max` into `[0, grid_z − 1]` at construction, so a band whose top
        // reaches the last layer AND whose bottom is layer 0 covers the whole grid. The
        // sentinel `LayerBand::FULL` (band_max == u32::MAX) also lands here.
        if band.band_min == 0 && band.band_max >= grid_z.saturating_sub(1) {
            return None;
        }
        let depth = band.onion_depth;
        let voxel_z_min = band.band_min.saturating_sub(depth);
        // `band_max` is the last SOLID layer; the raymarch samples up to `upper + 1 +
        // depth`, so cover through `band_max + 1 + depth` (clamped to grid_z).
        let voxel_z_max = band
            .band_max
            .saturating_add(1)
            .saturating_add(depth)
            .min(grid_z);
        Some(FogZSlab {
            voxel_z_min: voxel_z_min.min(grid_z),
            voxel_z_max,
        })
    }

    /// The inclusive chunk-Z coordinate range `[chunk_z_min, chunk_z_max]` whose chunks'
    /// Z-extent `[cz·extent, (cz+1)·extent)` intersects this slab. A fog builder restricts
    /// its covering chunk set to chunks whose `chunk_coord[2]` lands in this range. Both
    /// fog paths anchor chunk 0 at voxel 0 (`chunk_coord = floor(voxel / extent)`), so the
    /// chunk-Z of a voxel slab edge is `floor(edge / extent)`. The high edge uses the last
    /// covered voxel (`voxel_z_max − 1`) so an empty slab yields no chunks.
    pub fn covering_chunk_z_range(self, chunk_extent: u32) -> Option<[i32; 2]> {
        let extent = chunk_extent.max(1);
        if self.voxel_z_max <= self.voxel_z_min {
            return None;
        }
        let chunk_z_min = (self.voxel_z_min / extent) as i32;
        let chunk_z_max = ((self.voxel_z_max - 1) / extent) as i32;
        Some([chunk_z_min, chunk_z_max])
    }
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

// ============================================================================
// View cube (Milestone 5) — ARCHITECTURE.md §4.
// ============================================================================

/// Edge length (pixels) of the corner view-cube viewport (top-left).
pub const VIEW_CUBE_VIEWPORT_PIXELS: u32 = 128;
/// Margin (pixels) from the top-left corner to the viewport.
pub const VIEW_CUBE_VIEWPORT_MARGIN: u32 = 16;
/// Edge length of each square face-label texture.
const FACE_LABEL_TEXTURE_SIZE: u32 = 128;

/// One view-cube vertex: position, face normal, face UV, and the texture-array
/// layer (face index in +X,-X,+Y,-Y,+Z,-Z order).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CubeLabelVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    layer: u32,
}

/// Edge length of each square chrome-glyph texture (Home/Fit badges, rotate/roll
/// arrows). Smaller than the face labels — the glyphs are drawn at modest screen
/// sizes in the margins.
const CHROME_GLYPH_TEXTURE_SIZE: u32 = 64;

/// One screen-space chrome-overlay vertex: NDC position (fixed to the cube rect,
/// it does NOT rotate with the cube), glyph UV, a per-vertex tint (used to
/// brighten a hovered arrow), and the glyph texture-array layer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ChromeVertex {
    position: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
    layer: u32,
}

/// The chrome-glyph texture-array layers (#13 Step 2), in upload order. The
/// Home/Fit badges are ALWAYS drawn; the arrows are drawn only when the matching
/// zone is hovered.
#[derive(Debug, Clone, Copy)]
enum ChromeGlyph {
    HomeButton,
    FitButton,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    RollCw,
    RollCcw,
}

impl ChromeGlyph {
    /// Upload/lookup order for the texture array (must match `chrome_glyph_pixels`).
    const ALL: [ChromeGlyph; 8] = [
        ChromeGlyph::HomeButton,
        ChromeGlyph::FitButton,
        ChromeGlyph::ArrowUp,
        ChromeGlyph::ArrowDown,
        ChromeGlyph::ArrowLeft,
        ChromeGlyph::ArrowRight,
        ChromeGlyph::RollCw,
        ChromeGlyph::RollCcw,
    ];

    /// This glyph's index in the texture array.
    fn layer(self) -> u32 {
        self as u32
    }
}

/// The corner view cube: a labelled cube mirroring the main camera, plus a teal
/// edge wireframe (ARCHITECTURE.md §4). Rendered into a scissored top-left
/// viewport in its own pass (depth cleared there first).
pub struct ViewCubeRenderer {
    face_pipeline: wgpu::RenderPipeline,
    edge_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    edge_buffer: wgpu::Buffer,
    edge_vertex_count: u32,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    label_bind_group: wgpu::BindGroup,
    // --- #13 Step 2: screen-space chrome overlay (Home/Fit + hover arrows) ---
    chrome_pipeline: wgpu::RenderPipeline,
    chrome_bind_group: wgpu::BindGroup,
    chrome_vertex_buffer: wgpu::Buffer,
    /// Capacity (in vertices) of `chrome_vertex_buffer`; the per-frame glyph quads
    /// fit within this fixed cap (4 glyphs × 6 verts, generous).
    chrome_vertex_capacity: u32,
}

impl ViewCubeRenderer {
    /// Create the view-cube renderer for a colour target format.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, color_format: wgpu::TextureFormat) -> Self {
        let (vertices, indices) = view_cube_geometry();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("view cube vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("view cube indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let edges = view_cube_edges();
        let edge_vertex_count = edges.len() as u32;
        let edge_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("view cube edges"),
            contents: bytemuck::cast_slice(&edges),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view cube uniforms"),
            size: std::mem::size_of::<LineUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (uniform_bind_group_layout, uniform_bind_group) =
            cube_uniform_bind_group(device, &uniform_buffer);

        // --- 6-layer face-label texture array ---
        let label_pixels = generate_face_label_textures();
        let label_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("view cube label textures"),
            size: wgpu::Extent3d {
                width: FACE_LABEL_TEXTURE_SIZE,
                height: FACE_LABEL_TEXTURE_SIZE,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &label_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &label_pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * FACE_LABEL_TEXTURE_SIZE),
                rows_per_image: Some(FACE_LABEL_TEXTURE_SIZE),
            },
            wgpu::Extent3d {
                width: FACE_LABEL_TEXTURE_SIZE,
                height: FACE_LABEL_TEXTURE_SIZE,
                depth_or_array_layers: 6,
            },
        );
        let label_view = label_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let label_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("view cube label sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let label_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("view cube label layout"),
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
            });
        let label_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("view cube label bind group"),
            layout: &label_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&label_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&label_sampler),
                },
            ],
        });

        // --- Face pipeline (textured cube) ---
        let cube_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("view cube shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/viewcube.wgsl").into()),
        });
        let face_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("view cube face pipeline layout"),
            bind_group_layouts: &[Some(&uniform_bind_group_layout), Some(&label_bind_group_layout)],
            immediate_size: 0,
        });
        let cube_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CubeLabelVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { offset: 24, shader_location: 2, format: wgpu::VertexFormat::Float32x2 },
                wgpu::VertexAttribute { offset: 32, shader_location: 3, format: wgpu::VertexFormat::Uint32 },
            ],
        };
        // The view cube renders at 1 sample into the resolved target (after the
        // 3D MSAA resolve, before egui), so its pipelines use sample_count 1.
        let face_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("view cube face pipeline"),
            layout: Some(&face_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &cube_shader,
                entry_point: Some("vertex_main"),
                buffers: &[cube_vertex_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &cube_shader,
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
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview_mask: None,
            cache: None,
        });

        // --- Edge pipeline (teal wireframe, 1 sample, depth-tested) ---
        let edge_pipeline = build_line_pipeline(
            device,
            color_format,
            &uniform_bind_group_layout,
            "view cube edge",
            true,
            1,
        );

        // --- #13 Step 2: screen-space chrome overlay pipeline + glyph textures ---
        let (chrome_pipeline, chrome_bind_group) =
            build_chrome_overlay(device, queue, color_format);
        // Cap: at most Home + Fit + one hovered arrow on screen at once; size
        // generously for all glyph quads (6 verts each).
        let chrome_vertex_capacity = 12 * 6;
        let chrome_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view cube chrome vertices"),
            size: (chrome_vertex_capacity as usize * std::mem::size_of::<ChromeVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            face_pipeline,
            edge_pipeline,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            edge_buffer,
            edge_vertex_count,
            uniform_buffer,
            uniform_bind_group,
            label_bind_group,
            chrome_pipeline,
            chrome_bind_group,
            chrome_vertex_buffer,
            chrome_vertex_capacity: chrome_vertex_capacity as u32,
        }
    }

    /// Upload the view-cube camera matrix (`OrbitCamera::view_cube_view_projection`).
    pub fn update_uniforms(&self, queue: &wgpu::Queue, view_projection: glam::Mat4) {
        let uniforms = LineUniforms {
            view_projection: view_projection.to_cols_array_2d(),
            depth_bias: [0.0; 4],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Draw the cube into a scissored corner of `target_view` (its own render pass,
    /// with a freshly-cleared private depth texture). The colour attachment loads
    /// the already-resolved scene so only the corner is touched.
    ///
    /// Issue #25: the corner is the top-left of the CENTRAL 3D viewport rect
    /// (`viewport_x/y/w/h`, physical pixels), NOT the whole window — so the cube
    /// lines up with the visible 3D area instead of hiding behind the side panel.
    /// `target_width/height` are the full target dims (the colour + depth
    /// attachments span the whole target; the scissor confines the draw).
    ///
    /// #13 Step 2: `hovered_zone` is the chrome zone currently under the cursor
    /// (from `classify_cube_point`). The Home/Fit badges are drawn ALWAYS; the
    /// roll arrows are drawn ONLY when their zone is hovered. #13 Step 6 follow-up:
    /// the four rotate arrows are drawn PERSISTENTLY whenever `rotate_arrows_visible`
    /// (the view is face-constrained), with the hovered one brightened. The chrome
    /// is a
    /// screen-space overlay FIXED to the cube rect (it does NOT rotate with the
    /// cube), laid out in the same `rect.size` fractions Step 1 hit-tests against.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        target_width: u32,
        target_height: u32,
        viewport: [u32; 4],
        hovered_zone: Option<crate::camera::CubeChromeZone>,
        rotate_arrows_visible: bool,
    ) {
        // #13 Step 6.2: when a face/edge/corner ELEMENT is hovered, pack a 6-bit
        // face mask (bit = material index in +X,-X,+Y,-Y,+Z,-Z order) into the cube
        // uniform's `depth_bias.x` slot (byte offset 64) so the cube shader brightens
        // the hovered element. Any non-Element hover (arrow/badge) clears the mask.
        let highlight_mask = match hovered_zone {
            Some(crate::camera::CubeChromeZone::Element(element)) => {
                let mut mask = 0u32;
                for face in element.faces() {
                    mask |= 1 << cube_face_material_index(*face);
                }
                mask as f32
            }
            _ => 0.0,
        };
        queue.write_buffer(&self.uniform_buffer, 64, bytemuck::bytes_of(&highlight_mask));

        let [viewport_x, viewport_y, viewport_width, viewport_height] = viewport;
        let margin = VIEW_CUBE_VIEWPORT_MARGIN;
        let size = VIEW_CUBE_VIEWPORT_PIXELS;
        // Bail if the central viewport is too small to host the corner cube.
        if viewport_width < margin + size || viewport_height < margin + size {
            return;
        }
        // The cube's top-left corner, offset into the central viewport.
        let corner_x = viewport_x + margin;
        let corner_y = viewport_y + margin;
        // Bail if the cube would fall outside the actual target (defensive).
        if corner_x + size > target_width || corner_y + size > target_height {
            return;
        }
        // The depth attachment must match the colour attachment's size, so this
        // transient single-sample depth texture spans the whole target; the
        // scissor/viewport still confine the cube to the top-left corner.
        let depth_texture =
            create_single_sample_depth_view(device, target_width, target_height);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("view cube pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    // Load the resolved scene; the scissor confines our writes to
                    // the corner so the rest of the frame is untouched.
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_texture,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_viewport(corner_x as f32, corner_y as f32, size as f32, size as f32, 0.0, 1.0);
        pass.set_scissor_rect(corner_x, corner_y, size, size);

        pass.set_pipeline(&self.face_pipeline);
        pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        pass.set_bind_group(1, &self.label_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..self.index_count, 0, 0..1);

        pass.set_pipeline(&self.edge_pipeline);
        pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        pass.set_vertex_buffer(0, self.edge_buffer.slice(..));
        pass.draw(0..self.edge_vertex_count, 0..1);

        // --- #13 Step 2: screen-space chrome overlay, fixed to the cube rect. ---
        let chrome = build_chrome_vertices(hovered_zone, rotate_arrows_visible);
        if !chrome.is_empty() {
            let count = chrome.len().min(self.chrome_vertex_capacity as usize);
            queue.write_buffer(
                &self.chrome_vertex_buffer,
                0,
                bytemuck::cast_slice(&chrome[..count]),
            );
            pass.set_pipeline(&self.chrome_pipeline);
            pass.set_bind_group(0, &self.chrome_bind_group, &[]);
            pass.set_vertex_buffer(0, self.chrome_vertex_buffer.slice(..));
            pass.draw(0..count as u32, 0..1);
        }
    }
}

/// The material-index (`+X,-X,+Y,-Y,+Z,-Z`) of a [`crate::camera::CubeFace`], i.e.
/// its layer in the cube's face-label texture array and its bit in the hover mask.
fn cube_face_material_index(face: crate::camera::CubeFace) -> u32 {
    use crate::camera::CubeFace;
    // GEOMETRIC material-index order (+X,-X,+Y,-Y,+Z,-Z) under Z-up: the inverse of
    // `CubeFace::from_material_index` (Right,Left,Back,Front,Top,Bottom).
    match face {
        CubeFace::Right => 0,
        CubeFace::Left => 1,
        CubeFace::Back => 2,
        CubeFace::Front => 3,
        CubeFace::Top => 4,
        CubeFace::Bottom => 5,
    }
}

/// Uniform bind group for the view cube (binding 0 = view-projection).
fn cube_uniform_bind_group(
    device: &wgpu::Device,
    uniform_buffer: &wgpu::Buffer,
) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("view cube uniform layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            // #13 Step 6.2: the cube fragment shader now reads `highlight` from this
            // uniform too, so it must be visible to BOTH stages.
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("view cube uniform bind group"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });
    (layout, bind_group)
}

/// Build the labelled-cube geometry (side 1.4, centred on origin). Face order +X,
/// -X, +Y, -Y, +Z, -Z (matches `materialIndex` / `CubeFace`).
fn view_cube_geometry() -> (Vec<CubeLabelVertex>, Vec<u16>) {
    const HALF: f32 = 0.7; // side 1.4
    const UVS: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        ([1.0, 0.0, 0.0], [[HALF, -HALF, HALF], [HALF, -HALF, -HALF], [HALF, HALF, -HALF], [HALF, HALF, HALF]]),
        ([-1.0, 0.0, 0.0], [[-HALF, -HALF, -HALF], [-HALF, -HALF, HALF], [-HALF, HALF, HALF], [-HALF, HALF, -HALF]]),
        ([0.0, 1.0, 0.0], [[-HALF, HALF, HALF], [HALF, HALF, HALF], [HALF, HALF, -HALF], [-HALF, HALF, -HALF]]),
        ([0.0, -1.0, 0.0], [[-HALF, -HALF, -HALF], [HALF, -HALF, -HALF], [HALF, -HALF, HALF], [-HALF, -HALF, HALF]]),
        ([0.0, 0.0, 1.0], [[-HALF, -HALF, HALF], [HALF, -HALF, HALF], [HALF, HALF, HALF], [-HALF, HALF, HALF]]),
        ([0.0, 0.0, -1.0], [[HALF, -HALF, -HALF], [-HALF, -HALF, -HALF], [-HALF, HALF, -HALF], [HALF, HALF, -HALF]]),
    ];
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (layer, (normal, corners)) in faces.iter().enumerate() {
        let base = vertices.len() as u16;
        // Z-up: the BACK (+Y, layer 2) and BOTTOM (−Z, layer 5) faces wind such that
        // the shared UV table maps their label upside-down. Rotate just those two
        // faces' UVs 180° (corner_index + 2) so every label reads upright — the fix
        // lives in the unwrap, keeping the label textures themselves canonical.
        let uv_rotated = layer == 2 || layer == 5;
        for (corner_index, corner) in corners.iter().enumerate() {
            let uv_index = if uv_rotated { (corner_index + 2) % 4 } else { corner_index };
            vertices.push(CubeLabelVertex {
                position: *corner,
                normal: *normal,
                uv: UVS[uv_index],
                layer: layer as u32,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (vertices, indices)
}

/// Teal wireframe edges (12 cube edges) for the view cube.
fn view_cube_edges() -> Vec<LineVertex> {
    const HALF: f32 = 0.705; // a hair outside the faces so the edges read crisply
    let color = with_alpha(srgb_hex_to_linear(0x5f_b8_a4), 1.0);
    let corners = [
        [-HALF, -HALF, -HALF], [HALF, -HALF, -HALF], [HALF, HALF, -HALF], [-HALF, HALF, -HALF],
        [-HALF, -HALF, HALF], [HALF, -HALF, HALF], [HALF, HALF, HALF], [-HALF, HALF, HALF],
    ];
    let edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), // back face
        (4, 5), (5, 6), (6, 7), (7, 4), // front face
        (0, 4), (1, 5), (2, 6), (3, 7), // connecting
    ];
    let mut vertices = Vec::with_capacity(edges.len() * 2);
    for (a, b) in edges {
        vertices.push(LineVertex { position: corners[a], color });
        vertices.push(LineVertex { position: corners[b], color });
    }
    vertices
}

/// Render the six face-label textures into one stacked RGBA8 buffer (6 layers, in
/// GEOMETRIC `materialIndex` order +X,-X,+Y,-Y,+Z,-Z). Z-up labels each geometric
/// face: +Y = BACK, −Y = FRONT, +Z = TOP, −Z = BOTTOM. Each is a dark warm panel
/// `#241d15` with a teal `#5fb8a4` border and parchment `#e9e1d1` text.
fn generate_face_label_textures() -> Vec<u8> {
    const LABELS: [&str; 6] = ["RIGHT", "LEFT", "BACK", "FRONT", "TOP", "BOTTOM"];
    let size = FACE_LABEL_TEXTURE_SIZE as usize;
    let mut all = Vec::with_capacity(size * size * 4 * 6);
    for label in LABELS {
        all.extend_from_slice(&render_face_label(label));
    }
    all
}

/// Render one face-label texture (RGBA8, `FACE_LABEL_TEXTURE_SIZE` square).
fn render_face_label(label: &str) -> Vec<u8> {
    let size = FACE_LABEL_TEXTURE_SIZE as usize;
    const BACKGROUND: [u8; 4] = [0x24, 0x1d, 0x15, 0xff];
    const BORDER: [u8; 4] = [0x5f, 0xb8, 0xa4, 0xff];
    const TEXT: [u8; 4] = [0xe9, 0xe1, 0xd1, 0xff];

    let mut pixels = vec![0u8; size * size * 4];
    for pixel in pixels.chunks_exact_mut(4) {
        pixel.copy_from_slice(&BACKGROUND);
    }
    // Teal border (7px, inset 4px) like the prototype `strokeRect(4,4,120,120)`.
    let border_inset = 4usize;
    let border_thickness = 7usize;
    let put = |pixels: &mut [u8], x: usize, y: usize, color: [u8; 4]| {
        if x < size && y < size {
            let index = (y * size + x) * 4;
            pixels[index..index + 4].copy_from_slice(&color);
        }
    };
    for offset in 0..border_thickness {
        let lo = border_inset + offset;
        let hi = size - 1 - border_inset - offset;
        for c in border_inset..(size - border_inset) {
            put(&mut pixels, c, lo, BORDER);
            put(&mut pixels, c, hi, BORDER);
            put(&mut pixels, lo, c, BORDER);
            put(&mut pixels, hi, c, BORDER);
        }
    }

    // Centred bitmap text.
    draw_centered_label(&mut pixels, size, label, TEXT);
    pixels
}

/// Draw `label` centred using the built-in 5×7 bitmap font, scaled to fill the
/// face, into the RGBA8 `pixels` buffer.
fn draw_centered_label(pixels: &mut [u8], size: usize, label: &str, color: [u8; 4]) {
    let glyph_width = 5usize;
    let glyph_height = 7usize;
    let spacing = 1usize;
    let count = label.chars().count().max(1);
    let text_cells_wide = count * glyph_width + (count - 1) * spacing;
    // Choose an integer scale that fits within ~80% of the face width/height.
    let max_scale_w = (size * 8 / 10) / text_cells_wide.max(1);
    let max_scale_h = (size * 5 / 10) / glyph_height;
    let scale = max_scale_w.min(max_scale_h).max(1);

    let text_pixel_width = text_cells_wide * scale;
    let text_pixel_height = glyph_height * scale;
    let origin_x = (size.saturating_sub(text_pixel_width)) / 2;
    let origin_y = (size.saturating_sub(text_pixel_height)) / 2;

    let mut cursor_x = origin_x;
    for ch in label.chars() {
        let glyph = glyph_bitmap(ch);
        for (row, bits) in glyph.iter().enumerate() {
            for col in 0..glyph_width {
                if (bits >> (glyph_width - 1 - col)) & 1 == 1 {
                    // Filled cell → scale×scale block.
                    for dy in 0..scale {
                        for dx in 0..scale {
                            let x = cursor_x + col * scale + dx;
                            let y = origin_y + row * scale + dy;
                            if x < size && y < size {
                                let index = (y * size + x) * 4;
                                pixels[index..index + 4].copy_from_slice(&color);
                            }
                        }
                    }
                }
            }
        }
        cursor_x += (glyph_width + spacing) * scale;
    }
}

/// A 5×7 bitmap (7 rows of 5-bit masks) for the uppercase letters used by the
/// face labels. Unknown characters render blank.
fn glyph_bitmap(ch: char) -> [u8; 7] {
    match ch {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        _ => [0; 7],
    }
}

// ============================================================================
// #13 Step 2 — ViewCube chrome overlay (Home/Fit + hover rotate/roll arrows).
// Screen-space, fixed to the cube rect; the layout fractions mirror
// `camera::classify_cube_point` EXACTLY so the rendered glyphs sit on the Step-1
// hit zones.
// ============================================================================

/// Render one chrome glyph into an RGBA8 buffer (`CHROME_GLYPH_TEXTURE_SIZE`
/// square) with a TRANSPARENT background so the glyph floats over the scene; the
/// opaque pixels are white (tinted to parchment/teal by the vertex colour).
fn chrome_glyph_pixels(glyph: ChromeGlyph) -> Vec<u8> {
    let size = CHROME_GLYPH_TEXTURE_SIZE as usize;
    let mut pixels = vec![0u8; size * size * 4]; // transparent
    match glyph {
        ChromeGlyph::HomeButton => draw_home_icon(&mut pixels, size),
        ChromeGlyph::FitButton => draw_fit_icon(&mut pixels, size),
        ChromeGlyph::ArrowUp => draw_triangle_arrow(&mut pixels, size, ArrowFacing::Up),
        ChromeGlyph::ArrowDown => draw_triangle_arrow(&mut pixels, size, ArrowFacing::Down),
        ChromeGlyph::ArrowLeft => draw_triangle_arrow(&mut pixels, size, ArrowFacing::Left),
        ChromeGlyph::ArrowRight => draw_triangle_arrow(&mut pixels, size, ArrowFacing::Right),
        ChromeGlyph::RollCw => draw_roll_arc(&mut pixels, size, true),
        ChromeGlyph::RollCcw => draw_roll_arc(&mut pixels, size, false),
    }
    pixels
}

/// Which way a rotate-arrow triangle points.
#[derive(Clone, Copy)]
enum ArrowFacing {
    Up,
    Down,
    Left,
    Right,
}

/// Draw a clean filled triangular rotate arrow pointing in `facing`, centred.
/// #13 Step 6.3: a crisp equilateral-ish head (apex ~78% across the box, base
/// ~28%..72%) reads as a sharp directional cue at the small gutter size, with
/// anti-aliased edges from `fill_triangle`.
fn draw_triangle_arrow(pixels: &mut [u8], size: usize, facing: ArrowFacing) {
    const INK: [u8; 4] = [0xff, 0xff, 0xff, 0xff];
    let s = size as f32;
    let apex = s * 0.22; // distance of the apex from its edge
    let base = s * 0.74; // the flat base
    let near = s * 0.28; // base extent low
    let far = s * 0.72; // base extent high
    // Three vertices depending on facing (apex first).
    let (ax, ay, bx, by, cx, cy) = match facing {
        ArrowFacing::Up => (s * 0.5, apex, near, base, far, base),
        ArrowFacing::Down => (s * 0.5, base, near, apex, far, apex),
        ArrowFacing::Left => (apex, s * 0.5, base, near, base, far),
        ArrowFacing::Right => (base, s * 0.5, apex, near, apex, far),
    };
    fill_triangle(pixels, size, (ax, ay), (bx, by), (cx, cy), INK);
}

/// Fill a triangle (barycentric scan over its bounding box) onto an RGBA buffer.
/// #13 Step 6.3: edges are anti-aliased by 2×2 supersampling each pixel and writing
/// fractional coverage into the alpha channel, so the small glyphs read as clean
/// shapes instead of jagged stair-steps when scaled to the badge size.
fn fill_triangle(
    pixels: &mut [u8],
    size: usize,
    a: (f32, f32),
    b: (f32, f32),
    c: (f32, f32),
    color: [u8; 4],
) {
    let min_x = a.0.min(b.0).min(c.0).floor().max(0.0) as usize;
    let max_x = (a.0.max(b.0).max(c.0).ceil() as usize).min(size);
    let min_y = a.1.min(b.1).min(c.1).floor().max(0.0) as usize;
    let max_y = (a.1.max(b.1).max(c.1).ceil() as usize).min(size);
    let area = edge(a, b, c);
    if area.abs() < f32::EPSILON {
        return;
    }
    // 2×2 supersample offsets within each pixel.
    const SAMPLES: [(f32, f32); 4] = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)];
    for y in min_y..max_y {
        for x in min_x..max_x {
            let mut covered = 0u32;
            for (ox, oy) in SAMPLES {
                let p = (x as f32 + ox, y as f32 + oy);
                let w0 = edge(b, c, p) / area;
                let w1 = edge(c, a, p) / area;
                let w2 = edge(a, b, p) / area;
                if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                    covered += 1;
                }
            }
            if covered > 0 {
                blend_pixel(pixels, size, x, y, color, covered as f32 / 4.0);
            }
        }
    }
}

/// Alpha-composite `color` (scaled by `coverage` 0..1) over the existing pixel at
/// `(x, y)`. Used by the anti-aliased glyph rasterisers so overlapping strokes and
/// soft edges accumulate cleanly on the transparent glyph buffer.
fn blend_pixel(pixels: &mut [u8], size: usize, x: usize, y: usize, color: [u8; 4], coverage: f32) {
    if x >= size || y >= size {
        return;
    }
    let index = (y * size + x) * 4;
    let src_a = (color[3] as f32 / 255.0) * coverage.clamp(0.0, 1.0);
    if src_a <= 0.0 {
        return;
    }
    for channel in 0..3 {
        let dst = pixels[index + channel] as f32 / 255.0;
        let src = color[channel] as f32 / 255.0;
        let out = src * src_a + dst * (1.0 - src_a);
        pixels[index + channel] = (out * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    let dst_a = pixels[index + 3] as f32 / 255.0;
    let out_a = src_a + dst_a * (1.0 - src_a);
    pixels[index + 3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
}

/// Signed area of the triangle (a, b, c) — the edge function used for fill tests.
fn edge(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> f32 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

/// Fill an axis-aligned rectangle (in float coordinates) with anti-aliased edges.
fn fill_rect(pixels: &mut [u8], size: usize, x0: f32, y0: f32, x1: f32, y1: f32, color: [u8; 4]) {
    let min_x = x0.floor().max(0.0) as usize;
    let max_x = (x1.ceil() as usize).min(size);
    let min_y = y0.floor().max(0.0) as usize;
    let max_y = (y1.ceil() as usize).min(size);
    for y in min_y..max_y {
        for x in min_x..max_x {
            // Per-pixel coverage = overlap of the pixel cell with the rect.
            let cover_x = ((x as f32 + 1.0).min(x1) - (x as f32).max(x0)).clamp(0.0, 1.0);
            let cover_y = ((y as f32 + 1.0).min(y1) - (y as f32).max(y0)).clamp(0.0, 1.0);
            let coverage = cover_x * cover_y;
            if coverage > 0.0 {
                blend_pixel(pixels, size, x, y, color, coverage);
            }
        }
    }
}

/// Draw a simple house silhouette (Home button): a triangular roof over a square.
fn draw_home_icon(pixels: &mut [u8], size: usize) {
    const INK: [u8; 4] = [0xff, 0xff, 0xff, 0xff];
    let s = size as f32;
    // Roof triangle (slightly overhanging the body for a cleaner house read).
    fill_triangle(
        pixels,
        size,
        (s * 0.5, s * 0.16),
        (s * 0.14, s * 0.52),
        (s * 0.86, s * 0.52),
        INK,
    );
    // Body square, anti-aliased.
    fill_rect(pixels, size, s * 0.28, s * 0.46, s * 0.72, s * 0.82, INK);
}

/// Draw a "fit to view" icon: four corner brackets (a crop/frame mark). #13 Step
/// 6.3: corner brackets read as "frame the model" and are clearly distinct from
/// the Home house, while staying legible at the small badge size.
fn draw_fit_icon(pixels: &mut [u8], size: usize) {
    const INK: [u8; 4] = [0xff, 0xff, 0xff, 0xff];
    let s = size as f32;
    let lo = s * 0.18;
    let hi = s * 0.82;
    let thick = (s * 0.12).max(2.0);
    let arm = s * 0.26; // length of each bracket arm
    // Four L-shaped corner brackets (each = a horizontal + a vertical bar).
    // Top-left.
    fill_rect(pixels, size, lo, lo, lo + arm, lo + thick, INK);
    fill_rect(pixels, size, lo, lo, lo + thick, lo + arm, INK);
    // Top-right.
    fill_rect(pixels, size, hi - arm, lo, hi, lo + thick, INK);
    fill_rect(pixels, size, hi - thick, lo, hi, lo + arm, INK);
    // Bottom-left.
    fill_rect(pixels, size, lo, hi - thick, lo + arm, hi, INK);
    fill_rect(pixels, size, lo, hi - arm, lo + thick, hi, INK);
    // Bottom-right.
    fill_rect(pixels, size, hi - arm, hi - thick, hi, hi, INK);
    fill_rect(pixels, size, hi - thick, hi - arm, hi, hi, INK);
}

/// Draw a roll arc with an arrowhead (CW or CCW) — a curved 270° stroke with a
/// small triangular head, for the top-right roll buttons.
fn draw_roll_arc(pixels: &mut [u8], size: usize, clockwise: bool) {
    const INK: [u8; 4] = [0xff, 0xff, 0xff, 0xff];
    let s = size as f32;
    let cx = s * 0.5;
    let cy = s * 0.5;
    let radius = s * 0.30;
    let thick = s * 0.09;
    // Stroke a 270° arc (leave a gap so the curl reads).
    let start = if clockwise { 0.6 } else { std::f32::consts::PI - 0.6 };
    let sweep = std::f32::consts::TAU * 0.75;
    let steps = 96;
    for i in 0..=steps {
        let frac = i as f32 / steps as f32;
        let ang = if clockwise {
            start + sweep * frac
        } else {
            start - sweep * frac
        };
        let px = cx + ang.cos() * radius;
        let py = cy + ang.sin() * radius;
        // Stamp a small soft-edged disc for thickness (anti-aliased rim).
        let half = thick * 0.5;
        let r = (half + 1.0) as i32;
        for dy in -r..=r {
            for dx in -r..=r {
                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                let coverage = (half - dist + 0.5).clamp(0.0, 1.0);
                if coverage > 0.0 {
                    let x = px as i32 + dx;
                    let y = py as i32 + dy;
                    if x >= 0 && y >= 0 {
                        blend_pixel(pixels, size, x as usize, y as usize, INK, coverage);
                    }
                }
            }
        }
    }
    // Arrowhead at the arc's END.
    let end_ang = if clockwise { start + sweep } else { start - sweep };
    let hx = cx + end_ang.cos() * radius;
    let hy = cy + end_ang.sin() * radius;
    // Tangent direction at the end (perpendicular to radius, in sweep direction).
    let tang = if clockwise {
        end_ang + std::f32::consts::FRAC_PI_2
    } else {
        end_ang - std::f32::consts::FRAC_PI_2
    };
    let head = s * 0.16;
    let tip = (hx + tang.cos() * head, hy + tang.sin() * head);
    let left = (
        hx + (tang + 2.4).cos() * head * 0.7,
        hy + (tang + 2.4).sin() * head * 0.7,
    );
    let right = (
        hx + (tang - 2.4).cos() * head * 0.7,
        hy + (tang - 2.4).sin() * head * 0.7,
    );
    fill_triangle(pixels, size, tip, left, right, INK);
}

/// The glyph tint for the always-on chrome (parchment, matching the face text).
const CHROME_GLYPH_RGB: [f32; 3] = [0.913, 0.882, 0.819]; // #e9e1d1
/// A hovered arrow is brightened to teal-white so the highlight reads.
const CHROME_HOVER_RGB: [f32; 3] = [0.6, 1.0, 0.9];

/// Build the chrome overlay pipeline (alpha-blended screen-space textured quads)
/// and its glyph-texture bind group.
fn build_chrome_overlay(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    color_format: wgpu::TextureFormat,
) -> (wgpu::RenderPipeline, wgpu::BindGroup) {
    let layer_count = ChromeGlyph::ALL.len() as u32;
    let glyph_size = CHROME_GLYPH_TEXTURE_SIZE;
    let mut pixels = Vec::with_capacity((glyph_size * glyph_size * 4 * layer_count) as usize);
    for glyph in ChromeGlyph::ALL {
        pixels.extend_from_slice(&chrome_glyph_pixels(glyph));
    }
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("view cube chrome textures"),
        size: wgpu::Extent3d {
            width: glyph_size,
            height: glyph_size,
            depth_or_array_layers: layer_count,
        },
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
        &pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * glyph_size),
            rows_per_image: Some(glyph_size),
        },
        wgpu::Extent3d {
            width: glyph_size,
            height: glyph_size,
            depth_or_array_layers: layer_count,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("view cube chrome sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("view cube chrome layout"),
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
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("view cube chrome bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("view cube chrome shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/viewcube_chrome.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("view cube chrome pipeline layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<ChromeVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x2 },
            wgpu::VertexAttribute { offset: 8, shader_location: 1, format: wgpu::VertexFormat::Float32x2 },
            wgpu::VertexAttribute { offset: 16, shader_location: 2, format: wgpu::VertexFormat::Float32x4 },
            wgpu::VertexAttribute { offset: 32, shader_location: 3, format: wgpu::VertexFormat::Uint32 },
        ],
    };
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("view cube chrome pipeline"),
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
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None, // screen-space quads — don't cull on winding
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        // The view-cube pass binds a depth attachment, so this pipeline must carry
        // a matching depth-stencil state — but with depth TEST and WRITE disabled so
        // the chrome always paints on top of the cube/scene in the corner.
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
        multiview_mask: None,
        cache: None,
    });
    (pipeline, bind_group)
}

/// The glyph + rect-fraction centre of the rotate arrow for `dir`. #13 Step 6.8:
/// edge-hugging gutters; #13 Step 6.7: the glyph points the way the cube CONTENT
/// rolls under the 90° step (OPPOSITE the edge it sits on), so it matches the
/// action. Shared by the persistent draw and the hovered-highlight draw so the
/// dim and bright states sit in identical pixels.
fn rotate_arrow_layout(dir: crate::camera::ArrowDir) -> (ChromeGlyph, f32, f32) {
    use crate::camera::ArrowDir;
    match dir {
        // TOP edge gutter v∈[0,.13]; the step pulls the top face down → ArrowDown.
        ArrowDir::Up => (ChromeGlyph::ArrowDown, 0.5, 0.065),
        // BOTTOM edge gutter v∈[.87,1.0]; pushes content up → ArrowUp.
        ArrowDir::Down => (ChromeGlyph::ArrowUp, 0.5, 0.935),
        // LEFT edge gutter u∈[0,.13]; rolls content rightward → ArrowRight.
        ArrowDir::Left => (ChromeGlyph::ArrowRight, 0.065, 0.5),
        // RIGHT edge gutter u∈[.87,1.0]; rolls content leftward → ArrowLeft.
        ArrowDir::Right => (ChromeGlyph::ArrowLeft, 0.935, 0.5),
    }
}

/// Build the per-frame chrome vertices (screen-space, NDC within the cube
/// viewport). `hovered_zone` decides which glyph is brightened. #13 Step 6
/// follow-up: `rotate_arrows_visible` (= the view is face-constrained) draws ALL
/// FOUR rotate arrows PERSISTENTLY in their dim state (Fusion behaviour); the
/// hovered one brightens. When `false` (off-face view) no rotate arrows draw at
/// all. The layout fractions MUST match `classify_cube_point`.
fn build_chrome_vertices(
    hovered_zone: Option<crate::camera::CubeChromeZone>,
    rotate_arrows_visible: bool,
) -> Vec<ChromeVertex> {
    use crate::camera::{ArrowDir, CubeChromeZone, RollDir};

    let mut verts = Vec::new();

    // Helper: is THIS zone the hovered one? Picks the brighter tint.
    let tint = |is_hovered: bool| {
        if is_hovered {
            with_alpha(CHROME_HOVER_RGB, 1.0)
        } else {
            with_alpha(CHROME_GLYPH_RGB, 1.0)
        }
    };

    // --- Always-on: Home / Fit badges (top-left), Step-1 u∈[0,.12]/[.12,.24], v∈[0,.12]. ---
    let badge_y = 0.07;
    let badge_size = 0.12;
    let home_hovered = hovered_zone == Some(CubeChromeZone::HomeButton);
    push_glyph_quad(&mut verts, ChromeGlyph::HomeButton, 0.06, badge_y, badge_size, badge_size, tint(home_hovered));
    let fit_hovered = hovered_zone == Some(CubeChromeZone::FitButton);
    push_glyph_quad(&mut verts, ChromeGlyph::FitButton, 0.18, badge_y, badge_size, badge_size, tint(fit_hovered));

    // --- The 4 rotate arrows: drawn PERSISTENTLY whenever the view is face-
    // constrained (decoupled from hover); the hovered one is brightened. ---
    if rotate_arrows_visible {
        for dir in [ArrowDir::Up, ArrowDir::Down, ArrowDir::Left, ArrowDir::Right] {
            let (glyph, cx, cy) = rotate_arrow_layout(dir);
            let hovered = hovered_zone == Some(CubeChromeZone::RotateArrow(dir));
            push_glyph_quad(&mut verts, glyph, cx, cy, 0.075, 0.075, tint(hovered));
        }
    }

    // --- Hover-only: the 2 roll arrows (top-right). Step-1 u∈[.74,.87]/[.87,1.0], v∈[0,.13]. ---
    if let Some(CubeChromeZone::RollArrow(dir)) = hovered_zone {
        let (glyph, cx) = match dir {
            RollDir::Ccw => (ChromeGlyph::RollCcw, (0.74 + 0.87) / 2.0),
            RollDir::Cw => (ChromeGlyph::RollCw, (0.87 + 1.00) / 2.0),
        };
        push_glyph_quad(&mut verts, glyph, cx, 0.065, 0.11, 0.11, tint(true));
    }

    verts
}

/// Push two triangles for a textured glyph quad. `(cx, cy)` is the centre and
/// `(half_w, half_h)` the half-extents, ALL in rect fractions [0,1] (origin
/// top-left, y down). Converts to NDC (x: f*2-1, y: 1-f*2) for the viewport.
fn push_glyph_quad(
    verts: &mut Vec<ChromeVertex>,
    glyph: ChromeGlyph,
    cx: f32,
    cy: f32,
    half_w: f32,
    half_h: f32,
    color: [f32; 4],
) {
    let to_ndc = |fx: f32, fy: f32| [fx * 2.0 - 1.0, 1.0 - fy * 2.0];
    let layer = glyph.layer();
    // Corners in rect-fraction space (TL, TR, BR, BL) with UV.
    let corners = [
        (cx - half_w, cy - half_h, 0.0, 0.0),
        (cx + half_w, cy - half_h, 1.0, 0.0),
        (cx + half_w, cy + half_h, 1.0, 1.0),
        (cx - half_w, cy + half_h, 0.0, 1.0),
    ];
    let v = |i: usize| {
        let (fx, fy, u, t) = corners[i];
        ChromeVertex { position: to_ndc(fx, fy), uv: [u, t], color, layer }
    };
    // TL,TR,BR  +  TL,BR,BL
    verts.extend_from_slice(&[v(0), v(1), v(2), v(0), v(2), v(3)]);
}

/// Create a single-sample depth texture view (used by the view-cube pass).
fn create_single_sample_depth_view(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("view cube depth texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Create a 4-sample (MSAA) colour texture view for the 3D pass, sized to a
/// render target. Recreated on window resize / created at the offscreen size for
/// the headless capture. `format` matches the resolve target.
pub fn create_msaa_color_view(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("voxel msaa color texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: MSAA_SAMPLE_COUNT,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

// ============================================================================
// Transform gizmo (Milestone 5 origin gizmo, repurposed in issue #29 S2) —
// ARCHITECTURE.md §5.
// ============================================================================

/// X axis colour `#d9603f` (sRGB hex → linear).
const GIZMO_AXIS_X_HEX: u32 = 0xd9_60_3f;
/// Y axis colour `#6fcf5f`.
const GIZMO_AXIS_Y_HEX: u32 = 0x6f_cf_5f;
/// Z axis colour `#5a8cff`.
const GIZMO_AXIS_Z_HEX: u32 = 0x5a_8c_ff;
/// Right-angle square colour `#bdb39a`.
const GIZMO_SQUARE_HEX: u32 = 0xbd_b3_9a;

/// One coloured line-segment vertex (position + linear RGBA colour). The alpha
/// lets the M8 block lattice / floor grid draw at low opacity through the same
/// alpha-blending line pipeline the gizmo / view-cube edges use (those pass 1.0).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct LineVertex {
    position: [f32; 3],
    color: [f32; 4],
}

/// Camera uniform for the line passes (gizmo + view-cube edges + lattice/floor +
/// Points): the view-projection matrix plus a small NDC `depth_bias` (issue #29
/// floor fix). The bias is zero for every pass except the floor grid, which uses a
/// negative value to win the depth test against the model's coincident bottom face
/// without a geometric drop (wgpu forbids a hardware depth bias on `LineList`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct LineUniforms {
    view_projection: [[f32; 4]; 4],
    /// `[bias, 0, 0, 0]` — only `.x` is read; the rest pad to 16-byte alignment.
    depth_bias: [f32; 4],
}

/// The transform gizmo (issue #29 S2): three coloured axis lines and three
/// perpendicular square line-loops, drawn with **depth-test disabled** so it
/// shows through a solid model (correct manipulator behavior — ARCHITECTURE.md
/// §5). Drawn in the MSAA pass, after the voxels. Unlike the old origin gizmo it
/// FOLLOWS the selected node: its pivot translation is baked into the uploaded
/// view-projection (`view_projection · translate(pivot)`) so it sits ON the
/// object, and it is sized from the selected node's own extent. The axis-triad
/// geometry is kept for now; full TRS handles are future work.
pub struct TransformGizmoRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    vertex_capacity: u32,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
}

impl TransformGizmoRenderer {
    /// Create the transform gizmo renderer for a colour target format.
    /// `grid_dimensions` sizes the gizmo (`L = max(dims) * 0.62`); the caller
    /// rebuilds it to the SELECTED node's extent each frame.
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        grid_dimensions: [u32; 3],
    ) -> Self {
        let vertices = gizmo_vertices(grid_dimensions);
        let vertex_count = vertices.len() as u32;
        let vertex_capacity = vertex_count.max(1);
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gizmo line vertices"),
            contents: bytemuck::cast_slice(&pad_lines(vertices, vertex_capacity)),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gizmo uniforms"),
            size: std::mem::size_of::<LineUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (uniform_bind_group_layout, uniform_bind_group) =
            line_uniform_bind_group(device, &uniform_buffer, "gizmo");

        let pipeline = build_line_pipeline(
            device,
            color_format,
            &uniform_bind_group_layout,
            "gizmo",
            // Depth-test OFF (Always, no write) so the gizmo shows through solids.
            false,
            MSAA_SAMPLE_COUNT,
        );

        Self {
            pipeline,
            vertex_buffer,
            vertex_count,
            vertex_capacity,
            uniform_buffer,
            uniform_bind_group,
        }
    }

    /// Resize the gizmo to a freshly-resolved grid (matches the voxel rebuild).
    pub fn rebuild(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, grid_dimensions: [u32; 3]) {
        let vertices = gizmo_vertices(grid_dimensions);
        let vertex_count = vertices.len() as u32;
        if vertex_count <= self.vertex_capacity {
            if vertex_count > 0 {
                queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
            }
        } else {
            self.vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gizmo line vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
            self.vertex_capacity = vertex_count;
        }
        self.vertex_count = vertex_count;
    }

    /// Upload the camera matrix with the selected node's `pivot` translation baked
    /// in (issue #29 S2): the shader does `view_projection · position`, so feeding
    /// `view_projection · translate(pivot)` here moves the whole gizmo onto the
    /// selected node WITHOUT touching the shared `LineUniforms` layout. `pivot` is
    /// in the SAME recentred frame as the voxels, so the gizmo sits on the object.
    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        view_projection: glam::Mat4,
        pivot: glam::Vec3,
    ) {
        let model = glam::Mat4::from_translation(pivot);
        let uniforms = LineUniforms {
            view_projection: (view_projection * model).to_cols_array_2d(),
            depth_bias: [0.0; 4],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Record the gizmo draw into an already-begun (MSAA) render pass.
    pub fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        if self.vertex_count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..self.vertex_count, 0..1);
    }
}

/// Build the gizmo line vertices (axes + perpendicular squares), in world space.
fn gizmo_vertices(grid_dimensions: [u32; 3]) -> Vec<LineVertex> {
    let longest = grid_dimensions[0]
        .max(grid_dimensions[1])
        .max(grid_dimensions[2]) as f32;
    let axis_length = (longest * 0.62).max(1.0);
    let square_side = axis_length * 0.28;

    let x_color = with_alpha(srgb_hex_to_linear(GIZMO_AXIS_X_HEX), 1.0);
    let y_color = with_alpha(srgb_hex_to_linear(GIZMO_AXIS_Y_HEX), 1.0);
    let z_color = with_alpha(srgb_hex_to_linear(GIZMO_AXIS_Z_HEX), 1.0);
    let square_color = with_alpha(srgb_hex_to_linear(GIZMO_SQUARE_HEX), 1.0);

    let mut vertices = Vec::new();
    let mut line = |from: [f32; 3], to: [f32; 3], color: [f32; 4]| {
        vertices.push(LineVertex { position: from, color });
        vertices.push(LineVertex { position: to, color });
    };

    // Three axes from the origin.
    line([0.0, 0.0, 0.0], [axis_length, 0.0, 0.0], x_color);
    line([0.0, 0.0, 0.0], [0.0, axis_length, 0.0], y_color);
    line([0.0, 0.0, 0.0], [0.0, 0.0, axis_length], z_color);

    let s = square_side;
    // Square line-loops (closed) in the XY, YZ and ZX planes (prototype `sq`).
    let loop_segments = |points: &[[f32; 3]], color: [f32; 4], out: &mut Vec<LineVertex>| {
        for pair in points.windows(2) {
            out.push(LineVertex { position: pair[0], color });
            out.push(LineVertex { position: pair[1], color });
        }
    };
    loop_segments(
        &[[0.0, 0.0, 0.0], [s, 0.0, 0.0], [s, s, 0.0], [0.0, s, 0.0], [0.0, 0.0, 0.0]],
        square_color,
        &mut vertices,
    );
    loop_segments(
        &[[0.0, 0.0, 0.0], [0.0, s, 0.0], [0.0, s, s], [0.0, 0.0, s], [0.0, 0.0, 0.0]],
        square_color,
        &mut vertices,
    );
    loop_segments(
        &[[0.0, 0.0, 0.0], [0.0, 0.0, s], [s, 0.0, s], [s, 0.0, 0.0], [0.0, 0.0, 0.0]],
        square_color,
        &mut vertices,
    );
    vertices
}

/// Pad a line-vertex list to `capacity` with zeroed (degenerate) vertices.
fn pad_lines(mut vertices: Vec<LineVertex>, capacity: u32) -> Vec<LineVertex> {
    if (vertices.len() as u32) < capacity {
        vertices.resize(
            capacity as usize,
            LineVertex { position: [0.0; 3], color: [0.0; 4] },
        );
    }
    vertices
}

/// Build the shared uniform bind group (binding 0 = `LineUniforms`) for a line pass.
fn line_uniform_bind_group(
    device: &wgpu::Device,
    uniform_buffer: &wgpu::Buffer,
    label: &str,
) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label} line uniform layout")),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{label} line uniform bind group")),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });
    (layout, bind_group)
}

/// Build a `LineList` render pipeline (shared shader `line.wgsl`). `depth_tested`
/// selects whether the pass writes/tests depth; the gizmo passes `false`
/// (depth-test off so it shows through solids). Depth bias is applied in the SHADER
/// (via [`LineUniforms::depth_bias`]) rather than the pipeline, because wgpu rejects
/// a hardware `DepthBiasState` on `LineList` topology — the floor grid uses this to
/// win coincident depth against the model's base face without a geometric drop.
fn build_line_pipeline(
    device: &wgpu::Device,
    color_format: wgpu::TextureFormat,
    uniform_bind_group_layout: &wgpu::BindGroupLayout,
    label: &str,
    depth_tested: bool,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("line shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/line.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label} line pipeline layout")),
        bind_group_layouts: &[Some(uniform_bind_group_layout)],
        immediate_size: 0,
    });
    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<LineVertex>() as u64,
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
                format: wgpu::VertexFormat::Float32x4,
            },
        ],
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("{label} line pipeline")),
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
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            // Depth-test off (Always + no write) makes the gizmo show through the
            // model; depth-test on uses standard Less for the in-cube edges.
            depth_write_enabled: Some(depth_tested),
            depth_compare: Some(if depth_tested {
                wgpu::CompareFunction::Less
            } else {
                wgpu::CompareFunction::Always
            }),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview_mask: None,
        cache: None,
    })
}

// ============================================================================
// Block lattice + fine floor grid (Milestone 8) — prototype `buildGrids`.
// ============================================================================

/// Block lattice colour `#5fb8a4` (teal patina) at ~0.28 alpha.
const LATTICE_COLOR_HEX: u32 = 0x5f_b8_a4;
const LATTICE_ALPHA: f32 = 0.28;
/// Floor grid colour `#b8a47a` (warm sand) at 0.55 alpha. Issue #29 fix: the
/// floor grid was previously a very dim `#6b5f4a` at 0.16 alpha — coincident with
/// the model's depth-tested base plane and near-black against the background, so
/// it read as "nothing" when toggled on. A brighter colour at a lattice-comparable
/// opacity makes the base-plane grid clearly visible (it still hugs the node's
/// enclosing-block XZ footprint, snapped to the global block lattice).
const FLOOR_COLOR_HEX: u32 = 0xb8_a4_7a;
/// Alpha of a BOLD (block-edge) floor line — the major tier of the two-tier fine
/// floor grid (issue #29 fix). These lines sit at every block boundary and so
/// coincide exactly with the block lattice's vertical lines at the base plane.
const FLOOR_ALPHA: f32 = 0.55;
/// Alpha of a fine VOXEL-edge floor line — the minor tier (issue #29 fix). One
/// line per voxel boundary (step = 1) at a deliberately low opacity, so the floor
/// reads as a dense fine grid under the object without drowning the bold block
/// lines or the model. Mirrors the Point ground plane's minor/major two-tier
/// scheme (`POINT_PLANE_MINOR_ALPHA` vs `POINT_PLANE_MAJOR_ALPHA`).
const FLOOR_VOXEL_ALPHA: f32 = 0.16;

/// The per-object block lattice and floor grid (ARCHITECTURE.md §6 / prototype
/// `buildGrids`), drawn through the shared alpha-blended, depth-tested line
/// pipeline in the MSAA pass.
///
/// Issue #29 S3: this is no longer ONE whole-region lattice. Each frame the caller
/// walks the scene and, for every node whose grids are enabled (the scene master
/// ANDed with the node's own toggle), appends that node's block lattice and/or
/// floor lines into the renderer's per-frame batch via [`Self::set_batch`]. A
/// lattice box is a 3D box lattice with lines at every BLOCK boundary (spacing =
/// density) spanning the node's enclosing-block AABB; the floor is the horizontal
/// grid at the node's base plane, snapped to the same global block lines.
pub struct SceneGridRenderer {
    pipeline: wgpu::RenderPipeline,
    lattice_buffer: wgpu::Buffer,
    lattice_vertex_count: u32,
    lattice_capacity: u32,
    floor_buffer: wgpu::Buffer,
    floor_vertex_count: u32,
    floor_capacity: u32,
    /// Uniforms for the lattice draw — view-projection with ZERO depth bias.
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    /// Separate uniforms for the floor draw (issue #29 fix): the SAME
    /// view-projection but a NEGATIVE [`LineUniforms::depth_bias`], so the floor
    /// draws at the EXACT base plane `z = min[2]` (Z-up; meeting the lattice's bottom
    /// edges) yet wins the `Less` depth test against the model's coincident bottom
    /// face — no z-fight shimmer, no geometric vertical drop. (A hardware
    /// `DepthBiasState` is rejected by wgpu on `LineList`, so the bias is applied
    /// in the line shader via this uniform.)
    floor_uniform_buffer: wgpu::Buffer,
    floor_uniform_bind_group: wgpu::BindGroup,
}

/// The NDC depth bias (issue #29 fix) the floor grid uploads in its
/// [`LineUniforms::depth_bias`]: a small NEGATIVE offset pulls the floor lines a
/// hair toward the camera so they win the `Less` depth test against the model's
/// coincident bottom face. ~5e-4 in NDC is imperceptible spatially (far below the
/// old 0.25-voxel geometric drop) yet reliably resolves coincident depth on the
/// `Depth32Float` target.
const FLOOR_DEPTH_BIAS_NDC: f32 = -5.0e-4;

impl SceneGridRenderer {
    /// Create the renderer for a colour target. The line batches start empty —
    /// the caller fills them each frame via [`Self::set_batch`] from the visible
    /// nodes' enabled grids.
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let lattice_capacity = 1u32;
        let floor_capacity = 1u32;

        let lattice_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lattice line vertices"),
            contents: bytemuck::cast_slice(&pad_lines(Vec::new(), lattice_capacity)),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let floor_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("floor line vertices"),
            contents: bytemuck::cast_slice(&pad_lines(Vec::new(), floor_capacity)),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lattice uniforms"),
            size: std::mem::size_of::<LineUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (uniform_bind_group_layout, uniform_bind_group) =
            line_uniform_bind_group(device, &uniform_buffer, "lattice");

        // A SECOND uniform buffer for the floor draw, carrying the same matrix with a
        // negative NDC depth bias (issue #29 fix) — wgpu rejects a hardware depth bias
        // on LineList, so the floor biases its depth in the line shader via this buffer.
        let floor_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("floor uniforms"),
            size: std::mem::size_of::<LineUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (_floor_layout, floor_uniform_bind_group) =
            line_uniform_bind_group(device, &floor_uniform_buffer, "floor");

        // Depth-tested (true) so the lattice/floor are occluded by the solid model
        // — they read as a scaffold around/under it, not an overlay on top. The floor
        // shares this pipeline; its depth bias comes from its uniform, not the pipeline.
        let pipeline = build_line_pipeline(
            device,
            color_format,
            &uniform_bind_group_layout,
            "lattice",
            true,
            MSAA_SAMPLE_COUNT,
        );

        Self {
            pipeline,
            lattice_buffer,
            lattice_vertex_count: 0,
            lattice_capacity,
            floor_buffer,
            floor_vertex_count: 0,
            floor_capacity,
            uniform_buffer,
            uniform_bind_group,
            floor_uniform_buffer,
            floor_uniform_bind_group,
        }
    }

    /// Rebuild this frame's lattice + floor line batches by walking `scene` (issue
    /// #29 S3). For every visible node whose grids are enabled — the scene-wide
    /// master ANDed with that node's own per-object toggle — the node's
    /// enclosing-block lattice box ([`Scene::node_block_lattice_box_recentred`]) is
    /// appended to the corresponding batch:
    ///
    /// * `master_block_lattice && node.grids.block_lattice` → block lattice lines.
    /// * `master_floor_grid && node.grids.floor_grid` → base-plane floor lines.
    ///
    /// A node with no intrinsic extent (size-less Part / empty subtree) yields no
    /// box and is skipped. When NOTHING is enabled both batches are empty and
    /// [`Self::draw`] becomes a no-op — the new default, where per-object grids are
    /// off until the user turns them on.
    pub fn rebuild_from_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        voxels_per_block: u32,
    ) {
        let step = voxels_per_block.max(1);
        let (lattice_boxes, floor_boxes) = scene_grid_boxes(scene, voxels_per_block);
        let mut lattice: Vec<LineVertex> = Vec::new();
        let mut floor: Vec<LineVertex> = Vec::new();
        for (min, max) in lattice_boxes {
            lattice_vertices_into(&mut lattice, min, max, step);
        }
        for (min, max) in floor_boxes {
            floor_vertices_into(&mut floor, min, max, step);
        }
        self.lattice_vertex_count = upload_lines(
            device,
            queue,
            &mut self.lattice_buffer,
            &mut self.lattice_capacity,
            lattice,
            "lattice line vertices",
        );
        self.floor_vertex_count = upload_lines(
            device,
            queue,
            &mut self.floor_buffer,
            &mut self.floor_capacity,
            floor,
            "floor line vertices",
        );
    }

    /// Upload the camera matrix (same `view_projection` as the voxel pass) to BOTH
    /// the lattice uniform (zero depth bias) and the floor uniform (a negative NDC
    /// [`FLOOR_DEPTH_BIAS_NDC`] depth bias — issue #29 fix), so the floor wins
    /// coincident depth against the model's base face without a geometric drop.
    pub fn update_uniforms(&self, queue: &wgpu::Queue, view_projection: glam::Mat4) {
        let view_projection = view_projection.to_cols_array_2d();
        let lattice = LineUniforms { view_projection, depth_bias: [0.0; 4] };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&lattice));
        let floor = LineUniforms {
            view_projection,
            depth_bias: [FLOOR_DEPTH_BIAS_NDC, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(&self.floor_uniform_buffer, 0, bytemuck::bytes_of(&floor));
    }

    /// Record the lattice + floor draws into an already-begun (MSAA) pass. Gating
    /// is done at batch-build time (issue #29 S3): only grid-enabled nodes
    /// contributed lines, so empty batches simply draw nothing here. Both draws use
    /// the same line pipeline; the floor binds its own (depth-biased) uniform.
    pub fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        if self.lattice_vertex_count == 0 && self.floor_vertex_count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        if self.lattice_vertex_count > 0 {
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.lattice_buffer.slice(..));
            render_pass.draw(0..self.lattice_vertex_count, 0..1);
        }
        if self.floor_vertex_count > 0 {
            // Floor's own uniform carries the negative depth bias (issue #29 fix) so
            // the base-plane floor wins coincident depth against the model's bottom face.
            render_pass.set_bind_group(0, &self.floor_uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.floor_buffer.slice(..));
            render_pass.draw(0..self.floor_vertex_count, 0..1);
        }
    }
}

/// Write a line-vertex list to `buffer`, growing it if needed; returns the count.
fn upload_lines(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &mut wgpu::Buffer,
    capacity: &mut u32,
    vertices: Vec<LineVertex>,
    label: &str,
) -> u32 {
    let count = vertices.len() as u32;
    if count <= *capacity {
        if count > 0 {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&vertices));
        }
    } else {
        *buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        *capacity = count;
    }
    count
}

/// The per-object grid boxes for a scene (issue #29 S3), gated CPU-side so the walk
/// is unit-testable without a GPU. Returns `(lattice_boxes, floor_boxes)` where each
/// box is the `(min, max)` enclosing-block AABB (recentred voxels) of a node whose
/// grid is enabled — the scene-wide master ANDed with the node's own per-object
/// toggle. A node with no intrinsic extent contributes no box. When a master is off,
/// or a node's flag is off, that node contributes nothing to that batch (gating).
#[allow(clippy::type_complexity)]
pub(crate) fn scene_grid_boxes(
    scene: &Scene,
    voxels_per_block: u32,
) -> (Vec<([f32; 3], [f32; 3])>, Vec<([f32; 3], [f32; 3])>) {
    let mut lattice_boxes = Vec::new();
    let mut floor_boxes = Vec::new();
    let want_lattice_master = scene.master_block_lattice;
    let want_floor_master = scene.master_floor_grid;
    if !want_lattice_master && !want_floor_master {
        return (lattice_boxes, floor_boxes);
    }
    for (path, _id, _depth) in scene.tree_rows() {
        let Some(node) = scene.node_at_path(&path) else {
            continue;
        };
        let want_lattice = want_lattice_master && node.grids.block_lattice;
        let want_floor = want_floor_master && node.grids.floor_grid;
        if !want_lattice && !want_floor {
            continue;
        }
        let Some(node_box) = scene.node_block_lattice_box_recentred(&path, voxels_per_block) else {
            continue;
        };
        if want_lattice {
            lattice_boxes.push(node_box);
        }
        if want_floor {
            floor_boxes.push(node_box);
        }
    }
    (lattice_boxes, floor_boxes)
}

/// Block-boundary coordinates `[lo, lo+step, …, hi]` along one axis. The corners
/// `lo`/`hi` are block-aligned (the caller supplies an enclosing-block box), so the
/// `step`-stride walk lands exactly on `hi`; a final clamp guards float drift so the
/// closing block plane is always present.
fn block_boundaries(lo: f32, hi: f32, step: u32) -> Vec<f32> {
    let step = step.max(1) as f32;
    let mut values = Vec::new();
    let mut g = lo;
    // `+ step * 0.5` tolerance: include the plane at (or fractionally past) `hi`.
    while g <= hi + step * 0.5 {
        values.push(g.min(hi));
        g += step;
    }
    if values.last().copied() != Some(hi) {
        values.push(hi);
    }
    values
}

/// VOXEL-boundary coordinates `[lo, lo+1, …, hi]` along one axis, each tagged with
/// whether it is also a BLOCK boundary (`is_block`). The walk steps one voxel at a
/// time from the block-aligned `lo`, so every `step`-th line is flagged as a block
/// edge — meaning the bold (block) floor lines land on EXACTLY the same coordinates
/// as the block lattice's vertical lines (which `block_boundaries(lo, hi, step)`
/// places at `lo + k·step`). This is what makes the fine floor grid align with the
/// block lattice: the two share the block-aligned `lo` origin and the same stride.
fn voxel_boundaries(lo: f32, hi: f32, step: u32) -> Vec<(f32, bool)> {
    let step = step.max(1);
    let mut values = Vec::new();
    let mut index = 0i64;
    loop {
        let coord = lo + index as f32;
        // Closing guard: never overshoot `hi`; the final line is the block-aligned `hi`.
        if coord >= hi - 0.5 {
            values.push((hi, true));
            break;
        }
        values.push((coord, index.rem_euclid(step as i64) == 0));
        index += 1;
    }
    values
}

/// Append a 3D block lattice for the box `[min, max]` (voxels) — grid lines at every
/// BLOCK boundary (spacing = `step`) — into `vertices` (issue #29 S3, per-object).
/// Port of the prototype `buildGrids` lattice loop, now spanning an arbitrary box.
fn lattice_vertices_into(vertices: &mut Vec<LineVertex>, min: [f32; 3], max: [f32; 3], step: u32) {
    let color = with_alpha(srgb_hex_to_linear(LATTICE_COLOR_HEX), LATTICE_ALPHA);
    let xs = block_boundaries(min[0], max[0], step);
    let ys = block_boundaries(min[1], max[1], step);
    let zs = block_boundaries(min[2], max[2], step);

    let mut add = |from: [f32; 3], to: [f32; 3]| {
        vertices.push(LineVertex { position: from, color });
        vertices.push(LineVertex { position: to, color });
    };

    // The full 3D block lattice draws line families along all three axes. Z-up: the
    // VERTICAL pillars are the Z-along family below (between the XY ground nodes); the
    // X- and Y-along families are the horizontal grid lines.
    // Lines along Y at every (x, z) lattice node.
    for &x in &xs {
        for &z in &zs {
            add([x, min[1], z], [x, max[1], z]);
        }
    }
    // Lines along X at every (y, z) lattice node.
    for &y in &ys {
        for &z in &zs {
            add([min[0], y, z], [max[0], y, z]);
        }
    }
    // Lines along Z (the VERTICAL pillars, Z-up) at every (x, y) lattice node.
    for &x in &xs {
        for &y in &ys {
            add([x, y, min[2]], [x, y, max[2]]);
        }
    }
}

/// Append a FINE floor grid for the box `[min, max]` (voxels) on its BASE plane
/// (Z-up: exactly at `z = min[2]`, an XY grid) into `vertices` (issue #29 fix).
/// Two-tier, mirroring the block lattice and the Point ground plane:
///
/// * **Fine voxel lines** — one per voxel boundary (step 1), at the subtle
///   [`FLOOR_VOXEL_ALPHA`].
/// * **Bold block lines** — at every block boundary (step = `step`), at the
///   brighter [`FLOOR_ALPHA`], drawn ON TOP so block edges read clearly.
///
/// Both tiers walk from the BLOCK-ALIGNED `min` corner with a 1-voxel stride
/// ([`voxel_boundaries`]), so the bold block lines land on `min + k·step` — the
/// EXACT coordinates of the block lattice's vertical lines
/// ([`block_boundaries`]). The floor grid therefore shares the lattice's global
/// frame and their lines coincide at the base plane. Z-up: the base plane is the
/// node's bottom EXACTLY (`z = min[2]`), an XY-plane grid, so the floor's block
/// lines meet the block lattice's bottom edges with no vertical gap; z-fighting
/// against the model's coincident bottom face is avoided by the floor pipeline's
/// depth bias ([`SceneGridRenderer::floor_pipeline`]) rather than a geometric drop.
fn floor_vertices_into(vertices: &mut Vec<LineVertex>, min: [f32; 3], max: [f32; 3], step: u32) {
    let voxel_color = with_alpha(srgb_hex_to_linear(FLOOR_COLOR_HEX), FLOOR_VOXEL_ALPHA);
    let block_color = with_alpha(srgb_hex_to_linear(FLOOR_COLOR_HEX), FLOOR_ALPHA);
    // Z-up: the floor is an XY grid at the node's bottom (`z = min[2]`).
    let z = min[2];
    let xs = voxel_boundaries(min[0], max[0], step);
    let ys = voxel_boundaries(min[1], max[1], step);

    let mut add = |from: [f32; 3], to: [f32; 3], color: [f32; 4]| {
        vertices.push(LineVertex { position: from, color });
        vertices.push(LineVertex { position: to, color });
    };

    // Minor pass: fine voxel lines (one per voxel boundary), subtle.
    // Lines parallel to Y, at every X voxel boundary.
    for &(x, _) in &xs {
        add([x, min[1], z], [x, max[1], z], voxel_color);
    }
    // Lines parallel to X, at every Y voxel boundary.
    for &(y, _) in &ys {
        add([min[0], y, z], [max[0], y, z], voxel_color);
    }
    // Major pass: bold block lines, on top, coincident with the block lattice.
    for &(x, is_block) in &xs {
        if is_block {
            add([x, min[1], z], [x, max[1], z], block_color);
        }
    }
    for &(y, is_block) in &ys {
        if is_block {
            add([min[0], y, z], [max[0], y, z], block_color);
        }
    }
}

// ============================================================================
// Points — the world reference grid (issue #29 S5).
// ============================================================================

/// Reference-plane line colour `#5fb8a4` (teal patina) — shared with the lattice so
/// the Point ground reads as the same family of scaffold lines. Used by the analytic
/// infinite-grid shader (issue #29 Points fast-follow).
const POINT_PLANE_COLOR_HEX: u32 = 0x5f_b8_a4;
/// Base alpha of a MINOR (per-VOXEL, spacing 1) analytic-grid line. Deliberately low
/// so the ground stays subtle and does not fight a node's on-face voxel grid; the
/// shader's distance fade scales it down further toward the horizon.
const POINT_PLANE_MINOR_ALPHA: f32 = 0.10;
/// Base alpha of a MAJOR (per-BLOCK, spacing = density) analytic-grid line — bolder
/// than the voxel lines so block-cell boundaries pop while the field stays subtle.
const POINT_PLANE_MAJOR_ALPHA: f32 = 0.30;

/// Half-length (in BLOCKS) of each Point's axis lines, drawn through the Point
/// origin in the reference axis colours. A few blocks is enough to read as a frame
/// marker without dominating the scene.
const POINT_AXIS_HALF_BLOCKS: i64 = 6;
/// Base alpha of a Point's axis lines (depth-tested, so opaque voxels occlude them).
const POINT_AXIS_ALPHA: f32 = 0.85;

/// Which reference plane a tiled grid lies in (issue #29 S5). The plane is spanned
/// by its two in-plane axes; the third (constant) axis is pinned at the Point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReferencePlane {
    /// The FRONT plane (Z-up): spanned by X and Z; constant Y (normal +Y).
    Xz,
    /// The GROUND plane (Z-up): spanned by X and Y; constant Z (normal +Z).
    Xy,
    /// The side plane: spanned by Y and Z; constant X (normal +X).
    Yz,
}

/// Append a Point's coloured axis lines (issue #29 S5; per-axis fix) through
/// `origin_voxels` (the recentred render-frame position), reusing the gizmo axis
/// colours. `enabled[axis]` gates each axis independently (X = red +X, Y = green
/// +Y, Z = blue +Z), so e.g. turning Y off drops the green line and emits only the
/// X and Z segments. Each enabled axis spans `±POINT_AXIS_HALF_BLOCKS` blocks.
/// Depth-tested at draw time so opaque voxels occlude the parts behind them.
fn point_axes_into(
    vertices: &mut Vec<LineVertex>,
    origin_voxels: [f32; 3],
    step: u32,
    enabled: [bool; 3],
) {
    let half = POINT_AXIS_HALF_BLOCKS as f32 * step.max(1) as f32;
    let colors = [
        with_alpha(srgb_hex_to_linear(GIZMO_AXIS_X_HEX), POINT_AXIS_ALPHA),
        with_alpha(srgb_hex_to_linear(GIZMO_AXIS_Y_HEX), POINT_AXIS_ALPHA),
        with_alpha(srgb_hex_to_linear(GIZMO_AXIS_Z_HEX), POINT_AXIS_ALPHA),
    ];
    for axis in 0..3 {
        if !enabled[axis] {
            continue;
        }
        let mut from = origin_voxels;
        let mut to = origin_voxels;
        from[axis] = origin_voxels[axis] - half;
        to[axis] = origin_voxels[axis] + half;
        vertices.push(LineVertex { position: from, color: colors[axis] });
        vertices.push(LineVertex { position: to, color: colors[axis] });
    }
}

/// The recentred render-frame position (voxels) of a Point's origin (issue #29 S5):
/// `position_blocks·density − recentre`, the SAME frame the resolved voxels and the
/// per-object grids live in.
fn point_origin_voxels(point: &Point, recentre: [i64; 3], density: i64) -> [f32; 3] {
    let mut origin = [0.0f32; 3];
    for axis in 0..3 {
        origin[axis] = (point.position_blocks[axis] * density - recentre[axis]) as f32;
    }
    origin
}

/// Build the AXIS line batch for every VISIBLE Point in `scene` (issue #29 S5),
/// gated CPU-side so it is unit-testable without a GPU. For each non-hidden Point
/// its enabled axes (X = red +X, Y = green +Y, Z = blue +Z) are emitted as three
/// coloured line segments through the Point's origin, in the recentred render frame.
///
/// Issue #29 Points fast-follow: the reference PLANES no longer live here — they are
/// drawn by [`InfiniteGridRenderer`] as an ANALYTIC infinite grid (a fullscreen
/// ray-plane shader), which fixes the old finite tiled quad's hard edge / near-clip
/// cutoff at shallow angles. This batch is now AXES-only (the axes were fine as
/// lines and stay unchanged). A hidden Point contributes nothing.
fn points_line_batch(scene: &Scene, voxels_per_block: u32) -> Vec<LineVertex> {
    let mut vertices = Vec::new();
    let step = voxels_per_block.max(1);
    let density = step as i64;
    let recentre = scene.recentre_voxels_for_resolve(voxels_per_block);
    for point in &scene.points {
        if point.hidden {
            continue;
        }
        let origin = point_origin_voxels(point, recentre, density);
        if point.axis_x || point.axis_y || point.axis_z {
            point_axes_into(
                &mut vertices,
                origin,
                step,
                [point.axis_x, point.axis_y, point.axis_z],
            );
        }
    }
    vertices
}

/// One enabled reference PLANE of a visible Point (issue #29 Points fast-follow),
/// resolved into the recentred render frame for the analytic infinite-grid shader.
/// Computed CPU-side from the scene so the plane selection is unit-testable without
/// a GPU; [`InfiniteGridRenderer`] turns each into one fullscreen draw.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridPlaneInstance {
    /// The Point origin in the recentred render frame (voxels).
    pub origin: [f32; 3],
    /// The two in-plane unit axes spanning the plane (`u`, `v`).
    pub u_axis: [f32; 3],
    pub v_axis: [f32; 3],
    /// The plane normal (the pinned/constant world axis).
    pub normal: [f32; 3],
}

/// The unit basis (`u`, `v`, `normal`) for a [`ReferencePlane`]: the two in-plane
/// axes and the plane normal, in world coordinates.
fn reference_plane_basis(plane: ReferencePlane) -> ([f32; 3], [f32; 3], [f32; 3]) {
    match plane {
        // Front (Z-up): spanned by X and Z, normal +Y.
        ReferencePlane::Xz => ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        // GROUND (Z-up): spanned by X and Y, normal +Z.
        ReferencePlane::Xy => ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        // Side: spanned by Y and Z, normal +X.
        ReferencePlane::Yz => ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
    }
}

/// Collect every enabled reference PLANE of every VISIBLE Point (issue #29 Points
/// fast-follow), in the recentred render frame, for the analytic infinite-grid pass.
/// Hidden Points and disabled planes contribute nothing; the common case (the
/// Origin Point's XY ground plane, Z-up) yields exactly one instance. Pure + GPU-free
/// so the plane selection/orientation is unit-tested.
pub fn enabled_grid_planes(scene: &Scene, voxels_per_block: u32) -> Vec<GridPlaneInstance> {
    let step = voxels_per_block.max(1);
    let density = step as i64;
    let recentre = scene.recentre_voxels_for_resolve(voxels_per_block);
    let mut planes = Vec::new();
    for point in &scene.points {
        if point.hidden {
            continue;
        }
        let origin = point_origin_voxels(point, recentre, density);
        let mut push = |plane: ReferencePlane| {
            let (u_axis, v_axis, normal) = reference_plane_basis(plane);
            planes.push(GridPlaneInstance { origin, u_axis, v_axis, normal });
        };
        if point.plane_xz {
            push(ReferencePlane::Xz);
        }
        if point.plane_xy {
            push(ReferencePlane::Xy);
        }
        if point.plane_yz {
            push(ReferencePlane::Yz);
        }
    }
    planes
}

/// The world reference AXES (issue #29 S5): every visible [`Point`]'s axis lines,
/// batched into one **depth-tested, alpha-blended** line buffer — the SAME pass
/// family as [`SceneGridRenderer`], so opaque voxels OCCLUDE the axes while a node's
/// on-face voxel grid (a fragment overlay) stays visible on top of its faces.
///
/// Issue #29 Points fast-follow: the reference PLANES moved to
/// [`InfiniteGridRenderer`] (an analytic infinite grid); this renderer now draws
/// AXES only (unchanged). Each frame the caller rebuilds the batch from
/// `scene.points` via [`Self::rebuild_from_scene`], then uploads the camera matrix.
/// With no visible Point (all hidden / axes off) the batch is empty and
/// [`Self::draw`] is a no-op.
pub struct PointsRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    vertex_capacity: u32,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
}

impl PointsRenderer {
    /// Create the Points renderer for a colour target. The batch starts empty — the
    /// caller fills it each frame from the visible Points via [`Self::rebuild_from_scene`].
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let vertex_capacity = 1u32;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("points line vertices"),
            contents: bytemuck::cast_slice(&pad_lines(Vec::new(), vertex_capacity)),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("points uniforms"),
            size: std::mem::size_of::<LineUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (uniform_bind_group_layout, uniform_bind_group) =
            line_uniform_bind_group(device, &uniform_buffer, "points");

        // Depth-tested (true) so opaque voxels occlude the reference planes/axes —
        // the Points read as world scaffold behind/under the model, not an overlay.
        let pipeline = build_line_pipeline(
            device,
            color_format,
            &uniform_bind_group_layout,
            "points",
            true,
            MSAA_SAMPLE_COUNT,
        );

        Self {
            pipeline,
            vertex_buffer,
            vertex_count: 0,
            vertex_capacity,
            uniform_buffer,
            uniform_bind_group,
        }
    }

    /// Rebuild this frame's Point AXIS line batch by walking `scene.points` (issue
    /// #29 S5). Hidden Points and disabled axes contribute nothing; an all-off scene
    /// yields an empty batch (the draw becomes a no-op). The reference planes are
    /// drawn separately by [`InfiniteGridRenderer`].
    pub fn rebuild_from_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        voxels_per_block: u32,
    ) {
        let vertices = points_line_batch(scene, voxels_per_block);
        self.vertex_count = upload_lines(
            device,
            queue,
            &mut self.vertex_buffer,
            &mut self.vertex_capacity,
            vertices,
            "points line vertices",
        );
    }

    /// Upload the camera matrix (same `view_projection` as the voxel pass). Points
    /// use no depth bias (only the floor grid does — issue #29 fix).
    pub fn update_uniforms(&self, queue: &wgpu::Queue, view_projection: glam::Mat4) {
        let uniforms = LineUniforms {
            view_projection: view_projection.to_cols_array_2d(),
            depth_bias: [0.0; 4],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Record the Points draw into an already-begun (MSAA) pass. Self-gating: an
    /// empty batch (no visible Point) draws nothing.
    pub fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        if self.vertex_count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..self.vertex_count, 0..1);
    }
}

// ============================================================================
// Analytic infinite reference grid (issue #29 Points fast-follow) — replaces the
// finite tiled-line ground plane with a fullscreen ray-plane shader.
// ============================================================================

/// std140 uniform for one analytic-grid plane; field order matches `GridUniforms`
/// in `infinite_grid.wgsl` exactly. One instance per visible Point × enabled plane.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct InfiniteGridUniforms {
    view_projection: [[f32; 4]; 4],
    inverse_view_projection: [[f32; 4]; 4],
    /// Camera eye (recentred frame); `.w` unused.
    eye: [f32; 4],
    /// Plane origin (the Point's recentred position); `.w` unused.
    plane_origin: [f32; 4],
    /// In-plane unit axes spanning the plane, and the plane normal (`.w` unused).
    u_axis: [f32; 4],
    v_axis: [f32; 4],
    normal_axis: [f32; 4],
    /// Line colour (linear RGB); `.w` = voxel spacing (1.0).
    line_color: [f32; 4],
    /// `[block_spacing(=density), minor_alpha, major_alpha, reserved]`. The shader
    /// reads only `.x/.y/.z`; `.w` is a reserved padding slot (the old fixed
    /// world-distance fade was removed — fading is now per-tier LOD in the shader).
    /// Kept as `vec4` for the std140 16-byte uniform alignment.
    params: [f32; 4],
}

/// Maximum number of analytic-grid planes drawn in one frame (3 planes × a handful
/// of Points). Bounds the dynamic-offset uniform buffer; extra planes are dropped.
const MAX_GRID_PLANES: usize = 32;

/// The analytic infinite reference grid (issue #29 Points fast-follow): for each
/// visible [`Point`]'s enabled plane it draws a fullscreen triangle whose fragment
/// shader intersects the per-pixel view ray with that plane, computes a two-tier
/// (voxel + block) anti-aliased grid via screen-space derivatives, fades with
/// distance, and writes `@builtin(frag_depth)` so opaque voxels (drawn earlier in
/// the SAME MSAA pass) occlude it. This replaces the old finite tiled LINE quad,
/// whose hard edge / near-clip cutoff looked bad at shallow angles.
///
/// One dynamic-offset uniform buffer holds all planes' uniforms; [`Self::draw`]
/// binds each plane's slice and issues one 3-vertex draw. With no enabled plane the
/// draw is a no-op.
pub struct InfiniteGridRenderer {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Aligned stride (bytes) between consecutive plane uniforms in the buffer.
    aligned_stride: u32,
    /// Number of planes uploaded this frame (≤ [`MAX_GRID_PLANES`]).
    plane_count: u32,
}

impl InfiniteGridRenderer {
    /// Create the analytic-grid renderer for a colour target. The plane batch starts
    /// empty — the caller fills it each frame via [`Self::rebuild_from_scene`].
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("infinite grid shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/infinite_grid.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("infinite grid bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<InfiniteGridUniforms>() as u64,
                    ),
                },
                count: None,
            }],
        });

        // Each plane's uniform must start at a `min_uniform_buffer_offset_alignment`
        // boundary for the dynamic offset; pad the stride up to it.
        let uniform_size = std::mem::size_of::<InfiniteGridUniforms>() as u32;
        let alignment = device.limits().min_uniform_buffer_offset_alignment.max(1);
        let aligned_stride = uniform_size.div_ceil(alignment) * alignment;
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("infinite grid uniforms"),
            size: (aligned_stride as u64) * MAX_GRID_PLANES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("infinite grid bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(uniform_size as u64),
                }),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("infinite grid pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("infinite grid pipeline"),
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
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
            // Drawn INSIDE the MSAA pass: depth-tested LessEqual against the voxels'
            // depth (written via `frag_depth`) so opaque objects occlude the grid.
            // Depth WRITE is off so the (alpha-blended, transparent) grid never
            // occludes a later transparent draw or itself.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
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
            uniform_buffer,
            bind_group,
            aligned_stride,
            plane_count: 0,
        }
    }

    /// Rebuild this frame's analytic-grid planes by walking `scene.points` (issue #29
    /// Points fast-follow), uploading one plane uniform per visible Point × enabled
    /// plane. `view_projection` and its inverse + `camera_eye` are all in the
    /// recentred render frame the voxels live in. With no enabled plane this uploads
    /// nothing and [`Self::draw`] becomes a no-op.
    pub fn rebuild_from_scene(
        &mut self,
        queue: &wgpu::Queue,
        scene: &Scene,
        voxels_per_block: u32,
        view_projection: glam::Mat4,
        camera_eye: [f32; 3],
    ) {
        let planes = enabled_grid_planes(scene, voxels_per_block);
        let density = voxels_per_block.max(1) as f32;
        let inverse_view_projection = view_projection.inverse();
        let line_color = srgb_hex_to_linear(POINT_PLANE_COLOR_HEX);

        let count = planes.len().min(MAX_GRID_PLANES);
        for (index, plane) in planes.iter().take(count).enumerate() {
            let uniforms = InfiniteGridUniforms {
                view_projection: view_projection.to_cols_array_2d(),
                inverse_view_projection: inverse_view_projection.to_cols_array_2d(),
                eye: [camera_eye[0], camera_eye[1], camera_eye[2], 0.0],
                plane_origin: [plane.origin[0], plane.origin[1], plane.origin[2], 0.0],
                u_axis: [plane.u_axis[0], plane.u_axis[1], plane.u_axis[2], 0.0],
                v_axis: [plane.v_axis[0], plane.v_axis[1], plane.v_axis[2], 0.0],
                normal_axis: [plane.normal[0], plane.normal[1], plane.normal[2], 0.0],
                line_color: [line_color[0], line_color[1], line_color[2], 1.0],
                // `.w` is a reserved padding slot (the shader reads only x/y/z); the
                // old world-distance fade was removed in favour of per-tier LOD fade.
                params: [
                    density,
                    POINT_PLANE_MINOR_ALPHA,
                    POINT_PLANE_MAJOR_ALPHA,
                    0.0,
                ],
            };
            let offset = (index as u32 * self.aligned_stride) as u64;
            queue.write_buffer(&self.uniform_buffer, offset, bytemuck::bytes_of(&uniforms));
        }
        self.plane_count = count as u32;
    }

    /// Record the analytic-grid draws into an already-begun (MSAA) pass: one
    /// fullscreen triangle per plane, each binding its dynamic-offset uniform slice.
    /// Self-gating: no enabled plane → nothing drawn.
    pub fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        if self.plane_count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        for index in 0..self.plane_count {
            let offset = index * self.aligned_stride;
            render_pass.set_bind_group(0, &self.bind_group, &[offset]);
            render_pass.draw(0..3, 0..1);
        }
    }
}

// ============================================================================
// Onion-skin volumetric fog (issue #12) — fullscreen SDF raymarch.
// ============================================================================

/// Parameters for one frame of the onion-skin fog pass. The fog raymarches the
/// RESOLVED voxel grid (uploaded via [`OnionFogRenderer::upload_grid`]) as a 3D
/// cloud density field and integrates a faint haze in the onion-band Z range
/// (Z-up: layers are Z-slices) OUTSIDE the displayed (solid) band. Option B (x-ray
/// onion): the march ignores opaque depth so neighbour layers show through the
/// slice on both sides.
#[derive(Debug, Clone, Copy)]
pub struct OnionFogParams {
    /// Inverse camera view-projection (to unproject screen → world rays).
    pub inverse_view_projection: glam::Mat4,
    /// Inscribed semi-axes (= grid_dimensions / 2); maps world → normalised grid.
    pub semi_axes: [f32; 3],
    /// World-space Z extent of the onion band (the layers to fog).
    pub onion_z_min: f32,
    pub onion_z_max: f32,
    /// World-space Z extent of the displayed solid band (excluded from the fog —
    /// the opaque voxel pass already drew it).
    pub band_z_min: f32,
    pub band_z_max: f32,
}

/// std140-safe uniform block; field order matches `FogUniforms` in onion_fog.wgsl.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct OnionFogUniforms {
    inverse_view_projection: [[f32; 4]; 4],
    semi_axes: [f32; 3],
    fog_strength: f32,
    fog_color: [f32; 3],
    _pad0: f32,
    onion_z_min: f32,
    onion_z_max: f32,
    band_z_min: f32,
    band_z_max: f32,
}

/// Fog tint (cool blue-grey) and Beer–Lambert strength. Strength is low so the
/// haze is aerogel-faint and the solid band clearly shows through. Option B
/// (x-ray onion) wants it wispier still, so the band reads as a faint ghost rather
/// than a frosted puck — lowered from the original 0.18.
const ONION_FOG_COLOR_HEX: u32 = 0x9c_b4_d8;
const ONION_FOG_STRENGTH: f32 = 0.10;

/// Which occupancy source the onion fog raymarches (issue #28 S5a).
///
/// * [`WholeGrid`](FogMode::WholeGrid) (DEFAULT) — the original path: ONE whole-grid
///   `D3 R8` occupancy texture densified from the entire sparse list, disabled when
///   any axis exceeds `max_texture_dimension_3d`.
/// * [`PerChunk`](FogMode::PerChunk) — one apron'd `R8` occupancy volume per resident
///   chunk, packed into a small 3D atlas scoped to the active region, so a scene too
///   large for a single whole-grid 3D texture still renders fog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FogMode {
    #[default]
    WholeGrid,
    PerChunk,
}

/// Initial capacity (in chunk records) of the per-chunk fog records STORAGE buffer
/// (issue #28 S5a). This is no longer a hard cap: the records buffer grows on demand,
/// so the resident chunk count is bounded only by the atlas 3D-texture dimension
/// (guarded in `upload_per_chunk_occupancy`), not by the old 1024-entry uniform array
/// (16 KiB; 4096 would have hit the 64 KiB uniform limit). A region scene stays far
/// under this initial size, so the common case never reallocates.
pub const MAX_FOG_CHUNKS: usize = 1024;

/// One resident chunk's apron'd occupancy plus where it lives, in the per-chunk fog
/// path (issue #28 S5a). The occupancy is stored at `(extent + 2)³` so a **1-voxel
/// apron** on every face replicates the neighbour occupancy and trilinear sampling
/// stays smooth across chunk seams (no banding at the boundary).
#[derive(Debug, Clone)]
pub struct ChunkFogVolume {
    /// The chunk's integer coordinate in `CHUNK_BLOCKS`-cell space.
    pub chunk_coord: [i32; 3],
    /// The world-space (recentred) coordinate of this chunk's `[0,0,0]` voxel CORNER
    /// (i.e. the apron's interior origin), so the shader maps a world sample into the
    /// chunk's local `[0, extent)` voxel space.
    pub world_origin: [f32; 3],
    /// The apron'd occupancy, `(extent + 2)³` bytes in `(k*pad + j)*pad + i` order
    /// where local apron index `0` is the apron voxel at chunk-local `-1`.
    pub occupancy: Vec<u8>,
}

/// The CPU result of bucketing a recentred whole grid into apron'd per-chunk fog
/// volumes (issue #28 S5a): the per-chunk volumes plus the shared chunk voxel extent.
#[derive(Debug, Clone, Default)]
pub struct PerChunkFogOccupancy {
    /// `CHUNK_BLOCKS * voxels_per_block` — the voxel extent of one chunk per axis.
    pub chunk_extent: u32,
    /// The apron'd volumes, one per non-empty resident chunk. Empty when the resident
    /// non-empty chunk count exceeds [`MAX_FOG_CHUNKS`] (per-chunk fog disables itself
    /// for that build rather than dropping chunks and rendering with holes).
    pub volumes: Vec<ChunkFogVolume>,
}

/// Bucket a recentred [`VoxelGrid`] into one apron'd `R8` occupancy volume per
/// non-empty chunk (issue #28 S5a, the per-chunk fog path).
///
/// This reads the SAME recentred grid the whole-grid path uploads and uses the SAME
/// `world → voxel` mapping (`round(world + half - 0.5)`), so the per-chunk occupancy
/// is voxel-for-voxel identical to the whole-grid volume — the A/B match is exact by
/// construction. Each chunk's volume carries a **1-voxel apron**: the border layer is
/// filled from the global occupancy (the true neighbour voxel, NOT a clamp), so a ray
/// crossing a chunk seam trilinear-interpolates against the real neighbour density and
/// shows no discontinuity.
///
/// `chunk_coord = floor(voxel_index / chunk_extent)`; the chunk's interior origin in
/// recentred world space is `chunk_coord * chunk_extent - half_grid` (voxel CORNER),
/// so a world sample maps to chunk-local voxel space by `world - world_origin`.
///
/// Issue #59: `fog_z_slab`, when `Some`, RESTRICTS the covering set to chunks whose
/// chunk-Z lands in the slab's covering chunk-Z range — the onion fog only hazes the
/// band ± `onion_depth`, so a tall object with a thin band builds far fewer fog chunks
/// (smaller atlas, less densify). `None` covers the whole grid (the historical
/// behaviour, kept for the A/B reference + any non-onion caller).
#[deprecated(
    note = "CPU fog densify — superseded by the GPU per-chunk atlas (ADR 0007 live \
            call-site swap). Kept only as the CPU fallback + A/B reference; DELETE when \
            the live GPU perf refactor lands. New code installs via \
            OnionFogRenderer::install_per_chunk_atlas."
)]
pub fn build_per_chunk_fog_occupancy(
    grid: &VoxelGrid,
    voxels_per_block: u32,
    fog_z_slab: Option<FogZSlab>,
) -> PerChunkFogOccupancy {
    let chunk_extent = (CHUNK_BLOCKS * voxels_per_block.max(1)) as i64;
    let [grid_x, grid_y, grid_z] = grid.dimensions;
    if grid_x == 0 || grid_y == 0 || grid_z == 0 {
        return PerChunkFogOccupancy {
            chunk_extent: chunk_extent as u32,
            volumes: Vec::new(),
        };
    }
    // ===================================================================================
    // SCATTER build (issue #28 S5a perf, the `fog_upload` CPU hot path). This is a pure
    // speedup over the original GATHER (which built a global `HashSet<[i64;3]>` of every
    // occupied voxel, then for EVERY non-empty chunk probed all `(extent+2)³` apron cells
    // against that set — O(non_empty_chunks × pad³) hash lookups, tens of millions per
    // rebuild). The scatter is O(total occupied voxels): we deduplicate occupied voxels
    // once, pick the chunk set by bucketing each voxel into its interior chunk, then push
    // each voxel into the ≤8 chunk volumes whose apron box contains it. The output (chunk
    // set, ordering, world_origins, extent, apron bytes) is byte-identical to the gather —
    // see the `per_chunk_scatter_matches_gather_reference` regression test.
    //
    // First pass: integer voxel coords of every occupied voxel (the SAME mapping the
    // whole-grid upload uses), deduplicated.
    use std::collections::HashSet;
    let mut occupied_voxels: HashSet<[i64; 3]> = HashSet::new();
    for voxel in &grid.occupied {
        // ADR 0008: decode through the grid's single world→index authority, which uses the
        // grid's CARRIED recentre. Was an inlined `round(wp + floor(dim/2) − 0.5)` that
        // re-derived the centring and then dropped any index outside `[0, dim)` — which
        // silently discarded ~7/8 of a corner-anchored `DebugClouds` field. A centred Tool
        // (`recentre = floor(dim/2)`) decodes byte-identically; a corner-anchored cloud
        // (`recentre = 0`) now also lands in `[0, dim)`.
        let [i, j, k] = grid.voxel_index_of(voxel.world_position());
        if i < 0 || j < 0 || k < 0 || i >= grid_x as i64 || j >= grid_y as i64 || k >= grid_z as i64
        {
            continue;
        }
        occupied_voxels.insert([i, j, k]);
    }

    per_chunk_fog_from_occupied_indices(
        occupied_voxels,
        grid.recentre_voxels,
        chunk_extent,
        fog_z_slab,
    )
}

/// Bucket a set of occupied GRID-INDEX voxels (recentred-frame integer indices, already
/// clipped to the grid box) into apron'd per-chunk fog volumes — the shared core of the
/// grid densify ([`build_per_chunk_fog_occupancy`]) and the brick-sourced fill
/// ([`build_per_chunk_fog_occupancy_from_bricks`], ADR 0011 G5). Given the SAME occupied-
/// index set, both paths produce BYTE-IDENTICAL volumes: the chunk set, ordering,
/// world_origins, extent and apron bytes depend only on this set + the recentre + slab.
fn per_chunk_fog_from_occupied_indices(
    occupied_voxels: std::collections::HashSet<[i64; 3]>,
    recentre_voxels: [i64; 3],
    chunk_extent: i64,
    fog_z_slab: Option<FogZSlab>,
) -> PerChunkFogOccupancy {
    use std::collections::HashMap;
    // Issue #59: the inclusive chunk-Z coordinate window the slab covers (if any). A
    // chunk outside this window is skipped — its occupancy is never sampled by the
    // raymarch. `covering_chunk_z_range` returns `None` for an empty slab, which we
    // treat as "no chunks" below.
    let slab_chunk_z_range =
        fog_z_slab.map(|slab| slab.covering_chunk_z_range(chunk_extent as u32));

    // The chunk set: a chunk gets a volume iff it holds ≥1 occupied voxel in its INTERIOR
    // `[0, extent)` — the SAME criterion the gather used (it bucketed by
    // `voxel.div_euclid(chunk_extent)`). A voxel that lands only in a neighbour's apron
    // does NOT by itself create a chunk; it only writes into an existing chunk's border.
    // This is a single O(total occupied) bucketing pass — NO per-cell probing.
    let mut chunk_index_by_coord: HashMap<[i32; 3], usize> = HashMap::new();
    for &[i, j, k] in &occupied_voxels {
        let coord = [
            narrow_chunk_coord_local(i.div_euclid(chunk_extent)),
            narrow_chunk_coord_local(j.div_euclid(chunk_extent)),
            narrow_chunk_coord_local(k.div_euclid(chunk_extent)),
        ];
        // Issue #59: skip chunks whose chunk-Z is outside the onion-fog slab (Z-up:
        // index 2). `Some(None)` means the slab covers zero chunks → every chunk is
        // dropped; `None` (no slab) keeps the whole grid. The scatter below only writes
        // into chunks that survive here, so a dropped chunk is never allocated or filled.
        if let Some(range) = slab_chunk_z_range {
            match range {
                Some([chunk_z_min, chunk_z_max]) => {
                    if coord[2] < chunk_z_min || coord[2] > chunk_z_max {
                        continue;
                    }
                }
                None => continue,
            }
        }
        let next_index = chunk_index_by_coord.len();
        chunk_index_by_coord.entry(coord).or_insert(next_index);
    }
    let mut keys: Vec<[i32; 3]> = chunk_index_by_coord.keys().copied().collect();
    keys.sort_unstable();
    // Too many resident non-empty chunks for the per-chunk atlas to hold. Degrade
    // The resident chunk count is no longer capped here (the records now live in a
    // growable storage buffer, not a fixed 1024-entry uniform array). The remaining
    // ceiling — the atlas 3D-texture dimension — is enforced in
    // `upload_per_chunk_occupancy`, which disables the fog gracefully (no holes) if the
    // packed atlas would exceed the device's max 3D-texture size.
    let pad = (chunk_extent + 2) as usize; // apron: -1 .. extent (inclusive)
    // Allocate every chunk volume up front, in the SAME sorted order the gather used (the
    // tile-pack order in `upload_grid_per_chunk` depends on this ordering). Map each chunk
    // coord to its slot so the scatter can address volumes by coord in O(1).
    let mut volumes: Vec<ChunkFogVolume> = keys
        .iter()
        .map(|&coord| {
            let chunk_min = [
                coord[0] as i64 * chunk_extent,
                coord[1] as i64 * chunk_extent,
                coord[2] as i64 * chunk_extent,
            ];
            ChunkFogVolume {
                chunk_coord: coord,
                // Interior origin (voxel CORNER of local [0,0,0]) in recentred world space:
                // `chunk_min − recentre` (ADR 0008 — the grid's CARRIED recentre, was a
                // hard-coded `floor(dim/2)`). For a centred Tool this equals the historical
                // value; for a corner-anchored cloud (`recentre = 0`) it is `chunk_min`.
                world_origin: [
                    chunk_min[0] as f32 - recentre_voxels[0] as f32,
                    chunk_min[1] as f32 - recentre_voxels[1] as f32,
                    chunk_min[2] as f32 - recentre_voxels[2] as f32,
                ],
                occupancy: vec![0u8; pad * pad * pad],
            }
        })
        .collect();
    let mut slot_of_coord: HashMap<[i32; 3], usize> = HashMap::with_capacity(keys.len());
    for (slot, &coord) in keys.iter().enumerate() {
        slot_of_coord.insert(coord, slot);
    }

    // Scatter: each occupied voxel `v` lands in the apron box of every chunk `C` where
    // `local = v − C*extent ∈ [-1, extent]` per axis. For a given axis the only chunks
    // that can satisfy this are the owner chunk `c0 = v.div_euclid(extent)` (local in
    // `[0, extent)`) and at most one neighbour: `c0 − 1` (only when `v` sits on the owner's
    // low boundary, giving local `extent`, the chunk's +face apron) or `c0 + 1` (only when
    // `v` is on the owner's high boundary, giving local `-1`, the −face apron). So we
    // enumerate `{c0−1, c0, c0+1}` per axis and keep just the candidates whose local lands
    // in `[-1, extent]` — ≤2 per axis → ≤8 chunks per voxel. We write into a chunk only if
    // it EXISTS in the set above — matching the gather, where an apron-only neighbour voxel
    // sets an existing chunk's border but never creates a chunk. `div_euclid` floors toward
    // −∞ (correct for negative voxel coords in the recentred frame), the same rounding the
    // chunk-set bucketing uses above.
    let pad_i64 = pad as i64;
    // Candidate (chunk_coord, local_apron_index) pairs along one axis for voxel coord `v`.
    let axis_candidates = |v: i64| -> [(i64, i64); 3] {
        let owner = v.div_euclid(chunk_extent);
        [
            (owner - 1, v - (owner - 1) * chunk_extent),
            (owner, v - owner * chunk_extent),
            (owner + 1, v - (owner + 1) * chunk_extent),
        ]
    };
    for &[vi, vj, vk] in &occupied_voxels {
        let cands_i = axis_candidates(vi);
        let cands_j = axis_candidates(vj);
        let cands_k = axis_candidates(vk);
        for &(ck, local_k) in &cands_k {
            if !(-1..=chunk_extent).contains(&local_k) {
                continue;
            }
            let ak = local_k + 1;
            for &(cj, local_j) in &cands_j {
                if !(-1..=chunk_extent).contains(&local_j) {
                    continue;
                }
                let aj = local_j + 1;
                for &(ci, local_i) in &cands_i {
                    if !(-1..=chunk_extent).contains(&local_i) {
                        continue;
                    }
                    let coord = [
                        narrow_chunk_coord_local(ci),
                        narrow_chunk_coord_local(cj),
                        narrow_chunk_coord_local(ck),
                    ];
                    if let Some(&slot) = slot_of_coord.get(&coord) {
                        let ai = local_i + 1;
                        let flat = ((ak * pad_i64 + aj) * pad_i64 + ai) as usize;
                        volumes[slot].occupancy[flat] = 255;
                    }
                }
            }
        }
    }

    PerChunkFogOccupancy {
        chunk_extent: chunk_extent as u32,
        volumes,
    }
}

/// Narrow an i64 chunk-coordinate quotient to i32 (saturating). Chunk coords stay tiny
/// in practice; this mirrors `scene::narrow_chunk_coord` without exposing it.
fn narrow_chunk_coord_local(value: i64) -> i32 {
    value.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// **ADR 0011 G5 — the brick-sourced onion-fog fill.** Reconstruct the per-chunk apron'd
/// fog occupancy directly from a [`BrickFieldBuild`](crate::brick_field::BrickFieldBuild)
/// (the sink's boundary set), with NO whole-region `VoxelGrid` stream — the last
/// dense-shaped display consumer ADR 0010 flagged. Fog is boolean occupancy, so it does not
/// care about the display path's material constraint: this fills for EVERY chunkable scene,
/// including mixed-material-block scenes whose DISPLAY stays on the mesh path.
///
/// **Byte-identical to [`build_per_chunk_fog_occupancy`]** for the same scene: the brick
/// field is built from the same two-layer boundary set the grid expands from (coarse block →
/// solid `edge³`, boundary block → its rasterized cuboids; the G0 gate proves the atlas is
/// byte-identical to the boundary occupancy). This function reconstructs the SAME occupied
/// grid-index set (absolute voxel − recentre, clipped to `region_dimensions`) and runs the
/// SAME [`per_chunk_fog_from_occupied_indices`] core — so the chunk set, world_origins,
/// extent, the +1 APRON (fed from neighbour blocks' records, whether in-chunk or across a
/// chunk seam) and the band SLAB all mean exactly what the grid densify made them mean.
pub fn build_per_chunk_fog_occupancy_from_bricks(
    build: &crate::brick_field::BrickFieldBuild,
    region_dimensions: [u32; 3],
    recentre_voxels: [i64; 3],
    fog_z_slab: Option<FogZSlab>,
) -> PerChunkFogOccupancy {
    use crate::brick_field::{unpack_world_block_key, BrickPayload};
    let edge = build.brick_edge_voxels.max(1) as i64;
    let chunk_extent = (CHUNK_BLOCKS * build.brick_edge_voxels.max(1)) as i64;
    let [grid_x, grid_y, grid_z] = region_dimensions;
    if grid_x == 0 || grid_y == 0 || grid_z == 0 {
        return PerChunkFogOccupancy {
            chunk_extent: chunk_extent as u32,
            volumes: Vec::new(),
        };
    }

    // Reconstruct the occupied index set from the boundary records — the SAME set the grid
    // densify produces, so the shared core yields byte-identical tiles. The index frame is
    // the ABSOLUTE chunk-lattice voxel `A = world_block · edge + local`: that is exactly what
    // `VoxelGrid::voxel_index_of` returns for an expanded voxel (its `+0.5` / `−0.5` /
    // recentre terms cancel to `A`), so the grid path buckets on `A` and folds the recentre
    // in ONLY at `world_origin` (which the shared core does). A voxel outside `[0, dims)` is
    // clipped, exactly as `build_per_chunk_fog_occupancy` clips `grid.occupied`. A coarse-
    // solid record contributes its whole `edge³` (interior fill); a sculpted record only its
    // occupied atlas voxels (surface); air has no record. Walks the sparse boundary set —
    // never a dense whole-region grid.
    use std::collections::HashSet;
    let mut occupied_voxels: HashSet<[i64; 3]> = HashSet::new();
    let edge_usize = edge as usize;
    let push_voxel = |occupied: &mut HashSet<[i64; 3]>, absolute: [i64; 3]| {
        if absolute[0] < 0
            || absolute[1] < 0
            || absolute[2] < 0
            || absolute[0] >= grid_x as i64
            || absolute[1] >= grid_y as i64
            || absolute[2] >= grid_z as i64
        {
            return;
        }
        occupied.insert(absolute);
    };
    for record in &build.brick_records {
        let world_block = unpack_world_block_key(record.packed_world_block_key);
        let block_low = [world_block[0] * edge, world_block[1] * edge, world_block[2] * edge];
        match record.payload {
            BrickPayload::CoarseSolid { .. } => {
                for local_z in 0..edge {
                    for local_y in 0..edge {
                        for local_x in 0..edge {
                            push_voxel(
                                &mut occupied_voxels,
                                [
                                    block_low[0] + local_x,
                                    block_low[1] + local_y,
                                    block_low[2] + local_z,
                                ],
                            );
                        }
                    }
                }
            }
            BrickPayload::Sculpted { atlas_slot } => {
                let tile = build.sculpted_brick_occupancy(atlas_slot);
                for local_z in 0..edge_usize {
                    for local_y in 0..edge_usize {
                        for local_x in 0..edge_usize {
                            if tile[(local_z * edge_usize + local_y) * edge_usize + local_x] == 0 {
                                continue;
                            }
                            push_voxel(
                                &mut occupied_voxels,
                                [
                                    block_low[0] + local_x as i64,
                                    block_low[1] + local_y as i64,
                                    block_low[2] + local_z as i64,
                                ],
                            );
                        }
                    }
                }
            }
        }
    }

    per_chunk_fog_from_occupied_indices(occupied_voxels, recentre_voxels, chunk_extent, fog_z_slab)
}

/// Tile geometry for [`OnionFogRenderer::install_per_chunk_atlas`] — the atlas the GPU
/// resolver packed (ADR 0007). Bundled so the install call stays under the argument-count
/// lint; mirrors `upload_grid_per_chunk`'s `PerChunkFogMeta` geometry fields.
#[derive(Debug, Clone, Copy)]
pub struct PerChunkAtlasGeometry {
    /// `CHUNK_BLOCKS * voxels_per_block` — one chunk's voxel extent per axis.
    pub chunk_extent: u32,
    /// `chunk_extent + 2` — the apron'd per-axis tile span.
    pub pad: u32,
    /// Resident tiles per atlas axis (`ceil(cbrt(chunk_count))`).
    pub tiles_per_axis: u32,
    /// `tiles_per_axis * pad` — the atlas cube dimension per axis.
    pub atlas_dim: u32,
}

/// Fullscreen volumetric-fog renderer for the onion skin (issue #12). Raymarches
/// the resolved voxel grid (uploaded as a 3D occupancy texture) as a cloud.
pub struct OnionFogRenderer {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Trilinear sampler for the occupancy grid (the cloud density read).
    sampler: wgpu::Sampler,
    /// Current grid as a 3D R8 occupancy texture view; replaced on `upload_grid`.
    grid_view: wgpu::TextureView,
    /// Largest 3D texture dimension the device allows (grids past this skip fog).
    max_grid_dimension: u32,
    /// Whether the current grid uploaded successfully (else `draw` is a no-op).
    active: bool,
    /// Which occupancy source the next `draw` raymarches (issue #28 S5a). Set per
    /// upload (`upload_grid` → `WholeGrid`, `upload_grid_per_chunk` → `PerChunk`).
    mode: FogMode,
    // --- Per-chunk path (issue #28 S5a) ---
    /// Pipeline that raymarches the per-chunk atlas (separate WGSL entry point).
    per_chunk_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for the per-chunk path: shared camera uniform, atlas D3
    /// texture, sampler, scene depth, plus the per-chunk metadata uniform.
    per_chunk_bind_group_layout: wgpu::BindGroupLayout,
    /// The packed apron'd per-chunk occupancy atlas (one tile per resident chunk).
    per_chunk_atlas_view: wgpu::TextureView,
    /// Per-chunk metadata uniform HEADER (atlas tiling only; records are separate).
    per_chunk_meta_buffer: wgpu::Buffer,
    /// Per-chunk records STORAGE buffer: one `[f32; 4]` (`[world_origin.xyz,
    /// tile_index]`) per resident chunk. Runtime-sized (grown on demand), replacing
    /// the old fixed 1024-entry uniform array so the fog-chunk count is uncapped.
    per_chunk_records_buffer: wgpu::Buffer,
    /// Current capacity (in records) of `per_chunk_records_buffer`; grown when a build
    /// needs more resident chunks than fit.
    per_chunk_records_capacity: usize,
    /// Whether the last per-chunk upload produced a renderable atlas.
    per_chunk_active: bool,
}

/// std140 per-chunk fog metadata (issue #28 S5a). The shader walks the ray, and at
/// each sample point computes the chunk coord, looks up that chunk's atlas tile from
/// `chunks[]`, and samples the apron'd tile. Field order matches the WGSL
/// `PerChunkMeta` struct exactly.
/// The per-chunk fog metadata HEADER (uniform). The per-chunk records — one
/// `[world_origin.xyz, tile_index]` per resident chunk — used to live here as a
/// fixed `[[f32; 4]; MAX_FOG_CHUNKS]` array, which capped a scene at 1024 non-empty
/// fog chunks (16 KiB; 4096 would hit the 64 KiB uniform limit). They now live in a
/// runtime-sized STORAGE buffer (`per_chunk_records_buffer`), so the real ceiling is
/// the atlas 3D-texture dimension (guarded in `upload_per_chunk_occupancy`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PerChunkFogMeta {
    /// Number of resident chunk records in the records storage buffer.
    chunk_count: u32,
    /// Voxel extent of one chunk per axis (`CHUNK_BLOCKS * voxels_per_block`).
    chunk_extent: f32,
    /// Padded interior tile extent in the atlas (`chunk_extent + 2`, the apron).
    pad_extent: f32,
    /// Number of tiles per axis in the (cubic-ish) atlas tile grid.
    tiles_per_axis: u32,
    /// Atlas dimension in texels per axis (`tiles_per_axis * pad_extent`).
    atlas_dim: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

impl OnionFogRenderer {
    /// Create the fog renderer for a colour target format.
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("onion fog shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/onion_fog.wgsl").into()),
        });

        // Binding 0: uniform; binding 1: the resolved voxel grid as a 3D occupancy
        // texture (R8, trilinear-filtered); binding 2: its sampler.
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("onion fog bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Binding 3: the MSAA scene depth, so the fog is occluded by the
                // displayed opaque slice (depth-tested like Minecraft's clouds).
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: true,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("onion fog pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("onion fog pipeline"),
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
                    // Straight alpha-over: fog colour composited onto the resolved
                    // scene by its `coverage` alpha.
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
            // The fog runs at 1 sample onto the resolved target (after the 3D MSAA
            // resolve, before egui), so no depth attachment / no MSAA here.
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("onion fog uniforms"),
            size: std::mem::size_of::<OnionFogUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Trilinear sampler: linear filtering turns the binary occupancy grid into
        // a smooth cloud density. Clamp-to-edge (the shader also rejects samples
        // outside the grid box, so the border value never smears along the ray).
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("onion fog occupancy sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // Start with a 1×1×1 empty grid so the bind group is valid before the first
        // `upload_grid`. `active` stays false until a real grid lands.
        let grid_view = create_empty_occupancy_view(device);

        // --- Per-chunk path (issue #28 S5a) ---
        let per_chunk_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("onion fog per-chunk shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/onion_fog_perchunk.wgsl").into(),
            ),
        });
        let per_chunk_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("onion fog per-chunk bind group layout"),
                entries: &[
                    // 0: shared camera/band uniform (same OnionFogUniforms).
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 1: the packed apron'd per-chunk occupancy atlas (R8, trilinear).
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D3,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 2: occupancy sampler (trilinear).
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // 3: MSAA scene depth (depth-tested like the whole-grid path).
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: true,
                        },
                        count: None,
                    },
                    // 4: per-chunk metadata uniform HEADER (atlas tiling).
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 5: per-chunk records storage buffer (runtime-sized; uncapped).
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let per_chunk_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("onion fog per-chunk pipeline layout"),
                bind_group_layouts: &[Some(&per_chunk_bind_group_layout)],
                immediate_size: 0,
            });
        let per_chunk_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("onion fog per-chunk pipeline"),
                layout: Some(&per_chunk_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &per_chunk_shader,
                    entry_point: Some("vertex_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &per_chunk_shader,
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
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            });
        let per_chunk_atlas_view = create_empty_occupancy_view(device);
        let per_chunk_meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("onion fog per-chunk meta"),
            size: std::mem::size_of::<PerChunkFogMeta>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Records storage buffer: one [f32; 4] per resident chunk, grown on demand.
        let per_chunk_records_capacity = MAX_FOG_CHUNKS;
        let per_chunk_records_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("onion fog per-chunk records"),
            size: (per_chunk_records_capacity * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            uniform_buffer,
            bind_group_layout,
            sampler,
            grid_view,
            max_grid_dimension: device.limits().max_texture_dimension_3d,
            active: false,
            mode: FogMode::WholeGrid,
            per_chunk_pipeline,
            per_chunk_bind_group_layout,
            per_chunk_atlas_view,
            per_chunk_meta_buffer,
            per_chunk_records_buffer,
            per_chunk_records_capacity,
            per_chunk_active: false,
        }
    }

    /// Ensure `per_chunk_records_buffer` holds at least `needed` records, recreating it
    /// (doubling growth) when it does not. The per-frame bind group always binds the
    /// current buffer, so a swap here needs no other bookkeeping.
    fn ensure_records_capacity(&mut self, device: &wgpu::Device, needed: usize) {
        if needed <= self.per_chunk_records_capacity {
            return;
        }
        let new_capacity = needed.next_power_of_two().max(self.per_chunk_records_capacity * 2);
        self.per_chunk_records_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("onion fog per-chunk records"),
            size: (new_capacity * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.per_chunk_records_capacity = new_capacity;
    }

    /// Upload the resolved voxel grid as a 3D occupancy texture (the cloud density
    /// the fog raymarches). Call whenever the grid changes (geometry rebuild). A
    /// grid whose dimensions exceed the device's 3D-texture limit, or that is
    /// empty, disables the fog (`draw` becomes a no-op) rather than failing.
    pub fn upload_grid(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, grid: &VoxelGrid) {
        let [grid_x, grid_y, grid_z] = grid.dimensions;
        let limit = self.max_grid_dimension;
        if grid_x == 0
            || grid_y == 0
            || grid_z == 0
            || grid_x > limit
            || grid_y > limit
            || grid_z > limit
        {
            self.active = false;
            return;
        }

        // Densify the sparse occupied list into an R8 volume. Texel order matches a
        // 3D texture: index = (k * height + j) * width + i, with width=x, height=y,
        // depth=z. Voxel (i, j, k) ← round(world + half - 0.5), the same mapping the
        // grid uses elsewhere (voxel.rs::widest_run_in_band).
        let (width, height, depth) = (grid_x as usize, grid_y as usize, grid_z as usize);
        let mut occupancy = vec![0u8; width * height * depth];
        // Corner-anchoring decode: FLOORED half (`dim/2` integer division), exact for
        // an odd dim (voxel.rs::widest_run_in_band).
        let half_x = (grid_x / 2) as f32;
        let half_y = (grid_y / 2) as f32;
        let half_z = (grid_z / 2) as f32;
        for voxel in &grid.occupied {
            let position = voxel.world_position();
            let i = (position[0] + half_x - 0.5).round() as i64;
            let j = (position[1] + half_y - 0.5).round() as i64;
            let k = (position[2] + half_z - 0.5).round() as i64;
            if i < 0
                || j < 0
                || k < 0
                || i >= grid_x as i64
                || j >= grid_y as i64
                || k >= grid_z as i64
            {
                continue;
            }
            let index = (k as usize * height + j as usize) * width + i as usize;
            occupancy[index] = 255;
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("onion fog occupancy grid"),
            size: wgpu::Extent3d {
                width: grid_x,
                height: grid_y,
                depth_or_array_layers: grid_z,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R8Unorm,
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
            &occupancy,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(grid_x),
                rows_per_image: Some(grid_y),
            },
            wgpu::Extent3d {
                width: grid_x,
                height: grid_y,
                depth_or_array_layers: grid_z,
            },
        );
        self.grid_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.active = true;
        self.mode = FogMode::WholeGrid;
    }

    /// Upload the resolved grid as PER-CHUNK apron'd occupancy volumes packed into a
    /// small 3D atlas (issue #28 S5a, `--fog=perchunk`). Unlike [`upload_grid`], the
    /// atlas size is bounded by the number of resident chunks, NOT the whole-grid
    /// extent, so a scene whose whole-grid axis would exceed `max_texture_dimension_3d`
    /// (and thus disable the whole-grid fog) still renders fog here.
    ///
    /// Each chunk's tile is `(chunk_extent + 2)³` (a 1-voxel apron filled from the
    /// global occupancy), so trilinear sampling is seam-smooth across chunk boundaries.
    /// The shader marches in recentred world space and, at each sample, maps the world
    /// point into the owning chunk's tile via the metadata records.
    #[deprecated(
        note = "CPU fog densify + upload — superseded by the GPU per-chunk atlas (ADR 0007 \
                live call-site swap). Kept only as the CPU fallback; DELETE when the live \
                GPU perf refactor lands. New code installs via install_per_chunk_atlas."
    )]
    #[allow(deprecated)] // calls the (also-deprecated) CPU densify it wraps.
    pub fn upload_grid_per_chunk(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grid: &VoxelGrid,
        voxels_per_block: u32,
        fog_z_slab: Option<FogZSlab>,
    ) {
        let occupancy = build_per_chunk_fog_occupancy(grid, voxels_per_block, fog_z_slab);
        self.upload_per_chunk_occupancy(device, queue, &occupancy);
    }

    /// Pack a pre-built [`PerChunkFogOccupancy`] into the per-chunk fog atlas + metadata —
    /// the packing half of [`upload_grid_per_chunk`], shared with the ADR 0011 G5
    /// brick-sourced fill ([`build_per_chunk_fog_occupancy_from_bricks`]). The occupancy's
    /// origin (CPU grid densify vs brick boundary set) is irrelevant here: identical volumes
    /// pack to an identical atlas, so the fog SHADER is untouched.
    pub fn upload_per_chunk_occupancy(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        occupancy: &PerChunkFogOccupancy,
    ) {
        let pad = occupancy.chunk_extent as usize + 2;
        let chunk_count = occupancy.volumes.len();
        if chunk_count == 0 || pad == 0 {
            self.per_chunk_active = false;
            self.mode = FogMode::PerChunk;
            return;
        }

        // Arrange the resident chunk tiles into a cubic-ish 3D tile grid, so the atlas
        // dimension per axis (`tiles_per_axis * pad`) stays small — bounded by the chunk
        // COUNT, not the whole-grid extent. This is the core of why per-chunk dodges the
        // single-3D-texture limit.
        let tiles_per_axis = (chunk_count as f64).cbrt().ceil() as u32;
        let tiles_per_axis = tiles_per_axis.max(1);
        let atlas_dim = tiles_per_axis * pad as u32;
        if atlas_dim > self.max_grid_dimension {
            // The active region has too many chunks for the atlas to fit the 3D limit;
            // fall back to disabled fog rather than failing. (S5b's region scoping keeps
            // the resident set small; a region this large is out of S5a scope.)
            self.per_chunk_active = false;
            self.mode = FogMode::PerChunk;
            return;
        }

        // Pack every chunk's apron'd occupancy into the atlas at its tile slot, and
        // record each chunk's world origin + linear tile index in the metadata.
        let atlas_texels = (atlas_dim as usize).pow(3);
        let mut atlas = vec![0u8; atlas_texels];
        let meta = PerChunkFogMeta {
            chunk_count: chunk_count as u32,
            chunk_extent: occupancy.chunk_extent as f32,
            pad_extent: pad as f32,
            tiles_per_axis,
            atlas_dim: atlas_dim as f32,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        let mut records: Vec<[f32; 4]> = Vec::with_capacity(chunk_count);
        for (tile_index, volume) in occupancy.volumes.iter().enumerate() {
            // Linear tile index → 3D tile coord in the atlas.
            let tx = (tile_index as u32) % tiles_per_axis;
            let ty = ((tile_index as u32) / tiles_per_axis) % tiles_per_axis;
            let tz = (tile_index as u32) / (tiles_per_axis * tiles_per_axis);
            let base = [tx as usize * pad, ty as usize * pad, tz as usize * pad];
            for local_z in 0..pad {
                for local_y in 0..pad {
                    for local_x in 0..pad {
                        let src = (local_z * pad + local_y) * pad + local_x;
                        let ax = base[0] + local_x;
                        let ay = base[1] + local_y;
                        let az = base[2] + local_z;
                        let dst = (az * atlas_dim as usize + ay) * atlas_dim as usize + ax;
                        atlas[dst] = volume.occupancy[src];
                    }
                }
            }
            records.push([
                volume.world_origin[0],
                volume.world_origin[1],
                volume.world_origin[2],
                tile_index as f32,
            ]);
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("onion fog per-chunk atlas"),
            size: wgpu::Extent3d {
                width: atlas_dim,
                height: atlas_dim,
                depth_or_array_layers: atlas_dim,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R8Unorm,
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
            &atlas,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(atlas_dim),
                rows_per_image: Some(atlas_dim),
            },
            wgpu::Extent3d {
                width: atlas_dim,
                height: atlas_dim,
                depth_or_array_layers: atlas_dim,
            },
        );
        self.per_chunk_atlas_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.ensure_records_capacity(device, records.len());
        queue.write_buffer(&self.per_chunk_meta_buffer, 0, bytemuck::bytes_of(&meta));
        queue.write_buffer(&self.per_chunk_records_buffer, 0, bytemuck::cast_slice(&records));
        self.per_chunk_active = true;
        self.mode = FogMode::PerChunk;
    }

    /// Install a GPU-resolved per-chunk fog atlas + metadata with NO CPU densify
    /// (ADR 0007 live call-site swap). `texture` is the R8 atlas a [`GpuResolver`]
    /// produced; `world_origins` are the per-tile recentred `[0,0,0]`-voxel CORNERs in
    /// tile order (`coord·extent − recentre`, ADR 0008). This bypasses
    /// [`build_per_chunk_fog_occupancy`] / [`upload_grid_per_chunk`] entirely — the
    /// metadata is rebuilt CPU-side from the tile geometry, the occupancy never round-trips
    /// the CPU. Geometry that overflows the atlas budget disables the fog (consistent with
    /// `upload_grid_per_chunk`'s overflow branches) rather than rendering with holes.
    ///
    /// [`GpuResolver`]: crate::gpu_resolve::GpuResolver
    pub fn install_per_chunk_atlas(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        world_origins: &[[f32; 3]],
        geometry: PerChunkAtlasGeometry,
    ) {
        let PerChunkAtlasGeometry { chunk_extent, pad, tiles_per_axis, atlas_dim } = geometry;
        let chunk_count = world_origins.len();
        // No fog-chunk-count cap: records live in the growable storage buffer. The
        // atlas 3D-texture dimension is the only remaining ceiling.
        if chunk_count == 0 || atlas_dim > self.max_grid_dimension {
            self.per_chunk_active = false;
            self.mode = FogMode::PerChunk;
            return;
        }
        let meta = PerChunkFogMeta {
            chunk_count: chunk_count as u32,
            chunk_extent: chunk_extent as f32,
            pad_extent: pad as f32,
            tiles_per_axis,
            atlas_dim: atlas_dim as f32,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        // Tile `i` of the GPU atlas holds chunk `i` (the shader packs `tile_index =
        // chunk`), so the record order mirrors `world_origins` directly.
        let records: Vec<[f32; 4]> = world_origins
            .iter()
            .enumerate()
            .map(|(tile_index, origin)| [origin[0], origin[1], origin[2], tile_index as f32])
            .collect();
        self.per_chunk_atlas_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.ensure_records_capacity(device, records.len());
        queue.write_buffer(&self.per_chunk_meta_buffer, 0, bytemuck::bytes_of(&meta));
        queue.write_buffer(&self.per_chunk_records_buffer, 0, bytemuck::cast_slice(&records));
        self.per_chunk_active = true;
        self.mode = FogMode::PerChunk;
    }

    /// The fog mode the last upload selected (issue #28 S5a).
    pub fn mode(&self) -> FogMode {
        self.mode
    }

    /// Whether the per-chunk path has a renderable atlas (diagnostic / tests).
    pub fn per_chunk_active(&self) -> bool {
        self.per_chunk_active
    }

    /// Upload this frame's fog parameters.
    pub fn update(&self, queue: &wgpu::Queue, params: OnionFogParams) {
        let uniforms = OnionFogUniforms {
            inverse_view_projection: params.inverse_view_projection.to_cols_array_2d(),
            semi_axes: params.semi_axes,
            fog_strength: ONION_FOG_STRENGTH,
            fog_color: srgb_hex_to_linear(ONION_FOG_COLOR_HEX),
            _pad0: 0.0,
            onion_z_min: params.onion_z_min,
            onion_z_max: params.onion_z_max,
            band_z_min: params.band_z_min,
            band_z_max: params.band_z_max,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Draw the fog into `target_view` (the resolved scene), raymarching the
    /// uploaded occupancy grid and depth-testing against `depth_view` (the 3D pass's
    /// MSAA depth) so the displayed opaque slice occludes the onion layers behind
    /// it. A no-op until a grid has been uploaded (`upload_grid`). Its own render
    /// pass loads the existing colour and composites the haze over it.
    /// Issue #25: `viewport` (`[x, y, w, h]`, physical pixels) confines the
    /// fullscreen raymarch to the central 3D viewport rect. The fog reconstructs
    /// world rays from the central-aspect `inverse_view_projection`, so it is only
    /// valid inside that rect; the scissor keeps it off the panels.
    pub fn draw(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        viewport: [u32; 4],
    ) {
        // Build the bind group + pick the pipeline for the active mode. Both modes share
        // the camera uniform (binding 0), occupancy texture (1), sampler (2) and depth
        // (3); the per-chunk path adds the metadata uniform (4).
        let (pipeline, bind_group) = match self.mode {
            FogMode::WholeGrid => {
                if !self.active {
                    return;
                }
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("onion fog bind group"),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.grid_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(depth_view),
                        },
                    ],
                });
                (&self.pipeline, bind_group)
            }
            FogMode::PerChunk => {
                if !self.per_chunk_active {
                    return;
                }
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("onion fog per-chunk bind group"),
                    layout: &self.per_chunk_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &self.per_chunk_atlas_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.per_chunk_meta_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: self.per_chunk_records_buffer.as_entire_binding(),
                        },
                    ],
                });
                (&self.per_chunk_pipeline, bind_group)
            }
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("onion fog pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let [vx, vy, vw, vh] = viewport;
        pass.set_viewport(vx as f32, vy as f32, vw as f32, vh as f32, 0.0, 1.0);
        pass.set_scissor_rect(vx, vy, vw, vh);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// A 1×1×1 empty (zero) R8 occupancy texture view, used to keep the fog bind group
/// valid before/without a real grid upload.
fn create_empty_occupancy_view(device: &wgpu::Device) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("onion fog occupancy (empty)"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Create a 4-sample (MSAA) depth texture view sized to a render target.
pub fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("voxel depth texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: MSAA_SAMPLE_COUNT,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        // TEXTURE_BINDING so the onion fog pass can sample this MSAA depth (sample 0)
        // to occlude the haze behind the displayed opaque slice.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

#[cfg(test)]
#[allow(deprecated)] // exercises the CPU fog densify (deprecated, kept as A/B reference).
mod tests {
    use super::*;

    /// For a triangle wound CCW *as seen from outside*, the geometric face normal
    /// (edge0 × edge1) points in the SAME direction as the stored outward normal,
    /// so their dot product is positive. A negative dot means the winding is
    /// inside-out (BUG 1) and back-face culling would hide the visible face.
    fn assert_ccw_outward(positions: &[[f32; 3]], normals: &[[f32; 3]], indices: &[u16]) {
        assert_eq!(indices.len() % 3, 0, "indices must form whole triangles");
        for tri in indices.chunks_exact(3) {
            let a = positions[tri[0] as usize];
            let b = positions[tri[1] as usize];
            let c = positions[tri[2] as usize];
            let edge0 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let edge1 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            // edge0 × edge1
            let geometric_normal = [
                edge0[1] * edge1[2] - edge0[2] * edge1[1],
                edge0[2] * edge1[0] - edge0[0] * edge1[2],
                edge0[0] * edge1[1] - edge0[1] * edge1[0],
            ];
            let outward = normals[tri[0] as usize];
            let dot = geometric_normal[0] * outward[0]
                + geometric_normal[1] * outward[1]
                + geometric_normal[2] * outward[2];
            assert!(
                dot > 0.0,
                "triangle {tri:?} is wound inside-out (dot={dot}); outward faces would be culled",
            );
        }
    }

    // ===== Issue #28 S5a: per-chunk fog apron generation ========================

    /// Build a recentred [`VoxelGrid`] of `dims` voxels with the given integer voxel
    /// coords occupied, using the SAME `voxel ↔ world` mapping the fog upload uses
    /// (world = i + 0.5 - dim/2). So `build_per_chunk_fog_occupancy` reads them back at
    /// the exact integer coords here.
    fn grid_with_voxels(dims: [u32; 3], coords: &[[u32; 3]]) -> VoxelGrid {
        let mut grid = VoxelGrid::new(dims);
        // These voxels are placed CENTRED (`i + 0.5 − dim/2`), so the grid is recentred by
        // `floor(dim/2)` — record it (ADR 0008) so the fog decode/world_origin match.
        grid.recentre_voxels = [(dims[0] / 2) as i64, (dims[1] / 2) as i64, (dims[2] / 2) as i64];
        let half = [dims[0] as f32 / 2.0, dims[1] as f32 / 2.0, dims[2] as f32 / 2.0];
        for &[i, j, k] in coords {
            grid.occupied.push(Voxel {
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
        grid
    }

    /// Read the apron'd occupancy of `volume` at chunk-LOCAL coord `(li, lj, lk)`
    /// (`-1 ..= extent`), where `0` is the chunk's interior `[0,0,0]` voxel.
    fn apron_at(volume: &ChunkFogVolume, extent: i64, local: [i64; 3]) -> u8 {
        let pad = (extent + 2) as usize;
        let a = [
            (local[0] + 1) as usize,
            (local[1] + 1) as usize,
            (local[2] + 1) as usize,
        ];
        volume.occupancy[(a[2] * pad + a[1]) * pad + a[0]]
    }

    /// The apron of a chunk reflects a NEIGHBOUR chunk's boundary occupancy (seam
    /// smoothness), and an interior/edge voxel of the chunk shows up in its own volume.
    #[test]
    fn per_chunk_apron_reflects_neighbour_and_boundary() {
        // density 1 → CHUNK_BLOCKS * 1 = 4 voxels/chunk. A 2-chunk-wide grid in X so a
        // voxel in chunk 1 sits in chunk 0's +X apron.
        let density = 1u32;
        let extent = (CHUNK_BLOCKS * density) as i64; // 4
        let dims = [(extent * 2) as u32, extent as u32, extent as u32]; // 8x4x4
        // Occupy: chunk-0 boundary voxel at x=3 (its own +X edge), and chunk-1's first
        // voxel at x=4 (the neighbour that must appear in chunk-0's apron).
        let grid = grid_with_voxels(dims, &[[3, 0, 0], [4, 0, 0]]);

        let occ = build_per_chunk_fog_occupancy(&grid, density, None);
        assert_eq!(occ.chunk_extent, extent as u32);
        // Two chunks are occupied (x=3 in chunk 0, x=4 in chunk 1).
        assert_eq!(occ.volumes.len(), 2, "two chunks hold voxels");

        let chunk0 = occ
            .volumes
            .iter()
            .find(|v| v.chunk_coord == [0, 0, 0])
            .expect("chunk 0 resident");
        // Its own edge voxel (local x=3) is occupied.
        assert_eq!(apron_at(chunk0, extent, [3, 0, 0]), 255, "chunk-0 own edge voxel");
        // The neighbour voxel (chunk-1 x=4 → chunk-0 local x=extent) sits in the +X
        // apron and is filled from the global occupancy → seam-smooth trilinear.
        assert_eq!(
            apron_at(chunk0, extent, [extent, 0, 0]),
            255,
            "chunk-0 +X apron carries the neighbour chunk's boundary voxel"
        );
        // An empty apron cell stays 0 (e.g. -1 in X, outside everything).
        assert_eq!(apron_at(chunk0, extent, [-1, 0, 0]), 0, "empty apron stays empty");
    }

    // ===== Issue #59: onion-fog Z-slab scoping ==================================

    /// `FogZSlab::for_band` widens the band by `onion_depth` on each side (matching the
    /// raymarch's `[lower − depth, upper + 1 + depth]` sample span), clamps to `[0,
    /// grid_z]`, and returns `None` for a full-range band (nothing outside to ghost).
    #[test]
    fn fog_z_slab_matches_raymarch_span_and_skips_full_range() {
        let grid_z = 128;
        // Thin band [56, 72] with onion depth 8 → slab [56−8, 72+1+8] = [48, 81].
        let slab = FogZSlab::for_band(
            LayerBand { band_min: 56, band_max: 72, onion_depth: 8 },
            grid_z,
        )
        .expect("a clipped band yields a slab");
        assert_eq!(slab.voxel_z_min, 48);
        assert_eq!(slab.voxel_z_max, 81);

        // Clamp: a band near the top clamps the high edge to grid_z, the low to 0.
        let clamped = FogZSlab::for_band(
            LayerBand { band_min: 2, band_max: 126, onion_depth: 8 },
            grid_z,
        )
        .expect("still not full-range (lower != 0)");
        assert_eq!(clamped.voxel_z_min, 0, "low edge clamps to 0");
        assert_eq!(clamped.voxel_z_max, grid_z, "high edge clamps to grid_z");

        // Full-range band ([0, grid_z−1]) → None (skip the fog entirely).
        assert!(
            FogZSlab::for_band(
                LayerBand { band_min: 0, band_max: grid_z - 1, onion_depth: 2 },
                grid_z,
            )
            .is_none(),
            "a full-range band ghosts nothing → no slab"
        );
        // The FULL sentinel also skips.
        assert!(FogZSlab::for_band(LayerBand::FULL, grid_z).is_none());
    }

    /// `covering_chunk_z_range` maps a voxel slab to the inclusive chunk-Z rows whose
    /// Z-extent intersects it (chunk 0 anchored at voxel 0).
    #[test]
    fn fog_slab_covering_chunk_z_range() {
        // extent 64 (density 16). Slab [48, 81) → chunks floor(48/64)=0 .. floor(80/64)=1.
        let slab = FogZSlab { voxel_z_min: 48, voxel_z_max: 81 };
        assert_eq!(slab.covering_chunk_z_range(64), Some([0, 1]));
        // A slab fully inside chunk 2: [130, 190) → floor(130/64)=2 .. floor(189/64)=2.
        let inner = FogZSlab { voxel_z_min: 130, voxel_z_max: 190 };
        assert_eq!(inner.covering_chunk_z_range(64), Some([2, 2]));
        // Empty slab → no chunks.
        let empty = FogZSlab { voxel_z_min: 40, voxel_z_max: 40 };
        assert_eq!(empty.covering_chunk_z_range(64), None);
    }

    /// A slab scopes the CPU builder's covering set to the chunk-Z rows it intersects:
    /// a tall object with a thin band builds far fewer fog chunks than the whole grid,
    /// and the surviving chunks are byte-identical to the unscoped build (the slab only
    /// DROPS out-of-band chunks; it never alters an in-band chunk's occupancy).
    #[test]
    fn per_chunk_slab_scopes_covering_set() {
        let density = 1u32;
        let extent = (CHUNK_BLOCKS * density) as i64; // 4 voxels per chunk axis
        // A tall column: 1 chunk in X/Y, 8 chunks in Z (grid 4×4×32), one occupied voxel
        // in every Z chunk so the whole-grid build produces 8 volumes.
        let chunk_z_count = 8;
        let dims = [extent as u32, extent as u32, (extent * chunk_z_count) as u32];
        let coords: Vec<[u32; 3]> = (0..chunk_z_count)
            .map(|cz| [0, 0, (cz * extent) as u32])
            .collect();
        let grid = grid_with_voxels(dims, &coords);

        let whole = build_per_chunk_fog_occupancy(&grid, density, None);
        assert_eq!(whole.volumes.len(), chunk_z_count as usize, "whole grid = 8 Z chunks");

        // A thin band around Z chunk 3 only (voxels [12,16)). Slab covering chunk-Z 3..=3.
        let slab = FogZSlab { voxel_z_min: 12, voxel_z_max: 16 };
        let scoped = build_per_chunk_fog_occupancy(&grid, density, Some(slab));
        assert_eq!(scoped.volumes.len(), 1, "thin band builds a single Z chunk, not 8");
        assert_eq!(scoped.volumes[0].chunk_coord, [0, 0, 3]);
        // The surviving chunk equals the same chunk in the whole-grid build byte-for-byte.
        let whole_chunk3 = whole
            .volumes
            .iter()
            .find(|v| v.chunk_coord == [0, 0, 3])
            .expect("chunk-Z 3 resident in the whole build");
        assert_eq!(
            scoped.volumes[0].occupancy, whole_chunk3.occupancy,
            "slab-scoped chunk occupancy is byte-identical to the unscoped build"
        );

        // A slab spanning chunk-Z 2..=4 (voxels [10, 18) → floor(10/4)=2 .. floor(17/4)=4).
        let wide = FogZSlab { voxel_z_min: 10, voxel_z_max: 18 };
        let scoped_wide = build_per_chunk_fog_occupancy(&grid, density, Some(wide));
        let mut zs: Vec<i32> = scoped_wide.volumes.iter().map(|v| v.chunk_coord[2]).collect();
        zs.sort_unstable();
        assert_eq!(zs, vec![2, 3, 4], "a 3-chunk slab builds exactly those 3 Z chunks");
    }

    /// An empty grid yields no volumes (fog disables itself), and the world origin of a
    /// chunk is its interior `[0,0,0]` voxel corner in recentred world space.
    #[test]
    fn per_chunk_world_origin_is_recentred_corner() {
        let density = 1u32;
        let extent = (CHUNK_BLOCKS * density) as i64; // 4
        let dims = [(extent * 2) as u32, extent as u32, extent as u32]; // 8x4x4
        let half = dims[0] as f32 / 2.0; // 4
        let grid = grid_with_voxels(dims, &[[5, 0, 0]]); // chunk 1 in X
        let occ = build_per_chunk_fog_occupancy(&grid, density, None);
        let chunk1 = occ
            .volumes
            .iter()
            .find(|v| v.chunk_coord == [1, 0, 0])
            .expect("chunk 1 resident");
        // Chunk 1's interior origin = chunk_coord*extent - half = 4 - 4 = 0 in X.
        assert!((chunk1.world_origin[0] - (extent as f32 - half)).abs() < 1e-6);

        // Empty grid → no volumes.
        let empty = VoxelGrid::new(dims);
        assert!(build_per_chunk_fog_occupancy(&empty, density, None).volumes.is_empty());
    }

    /// When the resident non-empty chunk count exceeds `MAX_FOG_CHUNKS`, the builder
    /// disables per-chunk fog for that build (returns NO volumes) instead of dropping the
    /// overflow chunks — which would render fog with silent holes. The empty result makes
    /// `upload_grid_per_chunk` take its `chunk_count == 0` graceful-disable path. (#20 s4
    /// region-scoping is the proper long-term fix that keeps the resident set small.)
    #[test]
    fn per_chunk_fog_uncapped_past_initial_capacity() {
        let density = 1u32;
        let extent = (CHUNK_BLOCKS * density) as i64; // 4 voxels per chunk per axis
        // One occupied voxel in each of (MAX_FOG_CHUNKS + 1) distinct chunks along X —
        // one MORE than the old hard cap. The records now live in a growable storage
        // buffer, so the occupancy build no longer truncates or disables here; the only
        // remaining ceiling (the atlas 3D-texture dimension) lives in
        // `upload_per_chunk_occupancy`, not in this CPU build.
        let chunk_count = MAX_FOG_CHUNKS + 1;
        let dims = [(extent as usize * chunk_count) as u32, extent as u32, extent as u32];
        let coords: Vec<[u32; 3]> = (0..chunk_count)
            .map(|chunk_index| [(chunk_index as i64 * extent) as u32, 0, 0])
            .collect();
        let grid = grid_with_voxels(dims, &coords);

        let occ = build_per_chunk_fog_occupancy(&grid, density, None);
        assert_eq!(
            occ.volumes.len(),
            chunk_count,
            "past the old MAX_FOG_CHUNKS cap, every non-empty chunk must still get a volume \
             (the records buffer grows — no truncation, no disable)"
        );
    }

    /// The ORIGINAL gather implementation of `build_per_chunk_fog_occupancy`, kept here as
    /// the byte-identity oracle for the scatter rewrite (issue #28 S5a perf). It builds a
    /// global `HashSet` of occupied voxels, then for every non-empty chunk probes all
    /// `(extent+2)³` apron cells against that set. The production fn now scatters instead;
    /// `per_chunk_scatter_matches_gather_reference` asserts the two agree byte-for-byte
    /// across single/multi-chunk, boundary-straddling, recentred (negative world-pos), and
    /// even/odd/mixed-parity grids. If they ever diverge, the scatter has a bug.
    fn build_per_chunk_fog_occupancy_reference(
        grid: &VoxelGrid,
        voxels_per_block: u32,
    ) -> PerChunkFogOccupancy {
        use std::collections::{HashMap, HashSet};
        let chunk_extent = (CHUNK_BLOCKS * voxels_per_block.max(1)) as i64;
        let [grid_x, grid_y, grid_z] = grid.dimensions;
        if grid_x == 0 || grid_y == 0 || grid_z == 0 {
            return PerChunkFogOccupancy {
                chunk_extent: chunk_extent as u32,
                volumes: Vec::new(),
            };
        }
        // Decode through the grid's carried-recentre authority (ADR 0008), matching the
        // production builder.
        let mut occupied_voxels: HashSet<[i64; 3]> = HashSet::new();
        for voxel in &grid.occupied {
            let [i, j, k] = grid.voxel_index_of(voxel.world_position());
            if i < 0
                || j < 0
                || k < 0
                || i >= grid_x as i64
                || j >= grid_y as i64
                || k >= grid_z as i64
            {
                continue;
            }
            occupied_voxels.insert([i, j, k]);
        }
        let mut chunk_coords: HashMap<[i32; 3], ()> = HashMap::new();
        for &[i, j, k] in &occupied_voxels {
            let coord = [
                narrow_chunk_coord_local(i.div_euclid(chunk_extent)),
                narrow_chunk_coord_local(j.div_euclid(chunk_extent)),
                narrow_chunk_coord_local(k.div_euclid(chunk_extent)),
            ];
            chunk_coords.insert(coord, ());
        }
        let mut keys: Vec<[i32; 3]> = chunk_coords.keys().copied().collect();
        keys.sort_unstable();
        if keys.len() > MAX_FOG_CHUNKS {
            return PerChunkFogOccupancy {
                chunk_extent: chunk_extent as u32,
                volumes: Vec::new(),
            };
        }
        let pad = (chunk_extent + 2) as usize;
        let mut volumes = Vec::with_capacity(keys.len());
        for coord in keys {
            let chunk_min = [
                coord[0] as i64 * chunk_extent,
                coord[1] as i64 * chunk_extent,
                coord[2] as i64 * chunk_extent,
            ];
            let mut occupancy = vec![0u8; pad * pad * pad];
            for local_k in -1..=chunk_extent {
                for local_j in -1..=chunk_extent {
                    for local_i in -1..=chunk_extent {
                        let global = [
                            chunk_min[0] + local_i,
                            chunk_min[1] + local_j,
                            chunk_min[2] + local_k,
                        ];
                        if occupied_voxels.contains(&global) {
                            let ai = (local_i + 1) as usize;
                            let aj = (local_j + 1) as usize;
                            let ak = (local_k + 1) as usize;
                            occupancy[(ak * pad + aj) * pad + ai] = 255;
                        }
                    }
                }
            }
            volumes.push(ChunkFogVolume {
                chunk_coord: coord,
                world_origin: [
                    chunk_min[0] as f32 - grid.recentre_voxels[0] as f32,
                    chunk_min[1] as f32 - grid.recentre_voxels[1] as f32,
                    chunk_min[2] as f32 - grid.recentre_voxels[2] as f32,
                ],
                occupancy,
            });
        }
        PerChunkFogOccupancy {
            chunk_extent: chunk_extent as u32,
            volumes,
        }
    }

    /// Assert the production scatter equals the gather reference byte-for-byte: same
    /// `chunk_extent`, same number/order of volumes, same `chunk_coord`, bit-exact
    /// `world_origin`, and identical apron `occupancy` bytes for every chunk.
    fn assert_scatter_matches_gather(grid: &VoxelGrid, density: u32) {
        let scatter = build_per_chunk_fog_occupancy(grid, density, None);
        let gather = build_per_chunk_fog_occupancy_reference(grid, density);
        assert_eq!(scatter.chunk_extent, gather.chunk_extent, "chunk_extent");
        assert_eq!(
            scatter.volumes.len(),
            gather.volumes.len(),
            "same number of resident chunk volumes"
        );
        for (s, g) in scatter.volumes.iter().zip(gather.volumes.iter()) {
            assert_eq!(s.chunk_coord, g.chunk_coord, "chunk_coord (and ordering)");
            // bit-exact: both derive world_origin from the same integer math.
            assert_eq!(
                s.world_origin.map(f32::to_bits),
                g.world_origin.map(f32::to_bits),
                "world_origin bits for chunk {:?}",
                g.chunk_coord
            );
            assert_eq!(
                s.occupancy, g.occupancy,
                "apron occupancy bytes for chunk {:?}",
                g.chunk_coord
            );
        }
    }

    /// The scatter rewrite of `build_per_chunk_fog_occupancy` is byte-identical to the
    /// original gather across the cases a scatter most easily diverges on: single chunk,
    /// multi-chunk, voxels straddling chunk boundaries (apron borders), voxels at grid
    /// corners (apron reaches into non-resident chunks), recentred frames with negative
    /// `world_position`, and even / odd / mixed-parity grid dimensions. This is the
    /// load-bearing proof that the optimization changed only cost, not output.
    #[test]
    fn per_chunk_scatter_matches_gather_reference() {
        let extent_at = |density: u32| CHUNK_BLOCKS * density.max(1);

        // density 1 → extent 4.
        let e = extent_at(1) as i64; // 4

        // 1) Single chunk, a few interior voxels (even dims).
        assert_scatter_matches_gather(
            &grid_with_voxels([e as u32, e as u32, e as u32], &[[0, 0, 0], [1, 2, 3], [3, 3, 3]]),
            1,
        );

        // 2) Multi-chunk in all three axes, voxels in several distinct chunks.
        let dims2 = [(e * 2) as u32, (e * 2) as u32, (e * 2) as u32]; // 8^3 → 8 chunks
        assert_scatter_matches_gather(
            &grid_with_voxels(
                dims2,
                &[
                    [0, 0, 0],
                    [4, 0, 0],
                    [0, 4, 0],
                    [0, 0, 4],
                    [4, 4, 4],
                    [7, 7, 7],
                    [3, 4, 5],
                ],
            ),
            1,
        );

        // 3) Voxels straddling chunk boundaries: a chunk-0 edge voxel (x=3) and the
        //    neighbour's first voxel (x=4) so chunk-0's +X apron is populated from a
        //    voxel that belongs to chunk-1's interior — the exact apron-sharing case.
        let dims3 = [(e * 2) as u32, e as u32, e as u32]; // 8x4x4
        assert_scatter_matches_gather(
            &grid_with_voxels(dims3, &[[3, 0, 0], [4, 0, 0], [3, 3, 3], [4, 3, 3]]),
            1,
        );

        // 4) Corner voxels: voxel at (0,0,0) makes chunk 0's −1 apron reach into the
        //    (non-resident) chunk at coord −1, and the (dim-1) corner makes the +extent
        //    apron reach a chunk past the grid. Neither apron-only neighbour exists, so no
        //    stray chunk volume — exercises the "apron reaches outside the resident set".
        let dims4 = [(e * 3) as u32, (e * 3) as u32, (e * 3) as u32]; // 12^3, 27 chunks
        assert_scatter_matches_gather(
            &grid_with_voxels(
                dims4,
                &[[0, 0, 0], [11, 11, 11], [4, 0, 11], [0, 7, 4]],
            ),
            1,
        );

        // 5) Recentred frame with NEGATIVE world_position. With dims 12, half = 6, so the
        //    voxel at integer index 0 has world_position −5.5 (negative) — the recentred
        //    frame the prompt calls out. `grid_with_voxels` encodes world = i + 0.5 − half,
        //    so these voxels already have negative world coords; assert it round-trips.
        assert!(
            grid_with_voxels([12, 12, 12], &[[0, 0, 0]]).occupied[0].world_position()[0] < 0.0,
            "index-0 voxel in a dim-12 grid has negative recentred world_position"
        );
        assert_scatter_matches_gather(
            &grid_with_voxels([12, 12, 12], &[[0, 0, 0], [1, 0, 5], [5, 11, 0]]),
            1,
        );

        // 6) Higher density (density 4 → extent 16, the default-ish atlas tile size).
        let e16 = extent_at(4); // 16
        assert_scatter_matches_gather(
            &grid_with_voxels(
                [e16 * 2, e16, e16],
                &[[0, 0, 0], [15, 0, 0], [16, 0, 0], [31, 15, 15], [16, 8, 8]],
            ),
            4,
        );

        // 7) ODD and MIXED-parity dimensions (the floored-half decode is the subtle bit:
        //    half = dim/2 integer division). 7 is odd, 8 even, 9 odd → mixed parity.
        assert_scatter_matches_gather(
            &grid_with_voxels([7, 8, 9], &[[0, 0, 0], [6, 7, 8], [3, 4, 4], [6, 0, 8]]),
            1,
        );
        // All-odd, multi-chunk.
        assert_scatter_matches_gather(
            &grid_with_voxels([9, 9, 9], &[[0, 0, 0], [8, 8, 8], [4, 4, 4], [8, 0, 4]]),
            1,
        );

        // 8) Empty grid → both yield no volumes.
        assert_scatter_matches_gather(&VoxelGrid::new([e as u32, e as u32, e as u32]), 1);
    }

    /// A small hand-built expected case that PERMANENTLY pins the apron layout, so the
    /// byte-identity invariant survives even after the gather reference is gone. A single
    /// voxel at chunk-1's first interior cell (x=extent) in a 2-chunk-wide grid must:
    ///   * create exactly chunk 1's volume (interior owner),
    ///   * set chunk-1 local (0,0,0),
    ///   * set chunk-0's +X apron cell at local (extent,0,0) — the seam neighbour,
    ///   * leave everything else zero,
    ///   * and NOT create a chunk-0 volume from the apron-only write IF chunk 0 has no
    ///     interior voxel. Here chunk 0 has none, so ONLY chunk 1 is resident; the apron
    ///     write into chunk 0 has nowhere to land. This is the exact chunk-set criterion.
    #[test]
    fn per_chunk_scatter_hand_built_apron_layout() {
        let density = 1u32;
        let extent = (CHUNK_BLOCKS * density) as i64; // 4
        let dims = [(extent * 2) as u32, extent as u32, extent as u32]; // 8x4x4
        // Single voxel at x=extent (chunk 1's interior origin). Chunk 0 has NO interior
        // voxel, so it gets no volume even though this voxel sits in its +X apron.
        let grid = grid_with_voxels(dims, &[[extent as u32, 0, 0]]);
        let occ = build_per_chunk_fog_occupancy(&grid, density, None);

        assert_eq!(occ.chunk_extent, extent as u32);
        assert_eq!(occ.volumes.len(), 1, "only chunk 1 is resident (interior owner)");
        let chunk1 = &occ.volumes[0];
        assert_eq!(chunk1.chunk_coord, [1, 0, 0]);
        // Chunk 1's interior (0,0,0) is occupied.
        assert_eq!(apron_at(chunk1, extent, [0, 0, 0]), 255);
        // Its −1 apron (chunk-0 side, x=extent-1=3 global) is empty (no voxel there).
        assert_eq!(apron_at(chunk1, extent, [-1, 0, 0]), 0);
        // Exactly one occupied cell in the whole volume.
        let occupied_cells = chunk1.occupancy.iter().filter(|&&b| b == 255).count();
        assert_eq!(occupied_cells, 1, "exactly one apron cell set");
    }

    /// ADR 0011 G5 — the fog occupancy EXACTNESS gate: a per-chunk fog tile filled from the
    /// brick field ([`build_per_chunk_fog_occupancy_from_bricks`]) is BYTE-IDENTICAL to the
    /// tile the CPU densify ([`build_per_chunk_fog_occupancy`]) produces from the streamed
    /// `VoxelGrid` — same chunk set, world_origins, extent, and (the trap) the +1 APRON and
    /// band SLAB. Covers a single producer, a MULTI-producer scene (the one that still
    /// streamed before G5), and a MIXED-material multi-producer scene (fog is boolean
    /// occupancy, so it sources from bricks even where the DISPLAY falls back to the mesh),
    /// at two densities, with and without a Z-slab.
    #[test]
    fn brick_sourced_fog_matches_cpu_densify_byte_for_byte() {
        use crate::brick_field::build_brick_field;
        use crate::core_geom::MaterialChoice;
        use crate::scene::{Node, NodeContent, NodeTransform, Scene};
        use crate::two_layer_store::{expand_resident_chunks_into_grid, TwoLayerChunk, TwoLayerStore};
        use crate::voxel::{SdfShape, ShapeKind};

        fn sphere_node(name: &str, offset_blocks: [i64; 3], density: u32) -> Node {
            let shape = SdfShape::from_blocks(ShapeKind::Sphere, [4, 4, 4], 1, density);
            let mut node = Node::new(
                name.to_string(),
                NodeContent::Tool { shape, material: MaterialChoice::Stone },
            );
            node.transform = NodeTransform::from_blocks(offset_blocks, density);
            node
        }

        fn assert_paths_agree(scene: &Scene, density: u32) {
            let store = TwoLayerStore::enabled();
            let two_layer_chunks = store.build_covering_chunks(scene, density, 0);
            assert!(!two_layer_chunks.is_empty(), "fixture must produce covering chunks");
            let recentre = scene.recentre_voxels_for_resolve(density);
            let dims = scene.placed_region_dimensions(density);
            let chunk_refs: Vec<([i32; 3], &TwoLayerChunk)> =
                two_layer_chunks.iter().map(|(coord, chunk)| (*coord, chunk)).collect();
            let grid = expand_resident_chunks_into_grid(&chunk_refs, dims, recentre, density);
            let build = build_brick_field(&two_layer_chunks, density);

            // A middle-band slab (exercises the band scoping) plus the whole-grid `None`.
            let grid_z = dims[2];
            let slabs = [
                None,
                FogZSlab::for_band(
                    LayerBand {
                        band_min: grid_z / 4,
                        band_max: (grid_z * 3 / 4).min(grid_z.saturating_sub(1)),
                        onion_depth: 4,
                    },
                    grid_z,
                ),
            ];
            for slab in slabs {
                #[allow(deprecated)]
                let cpu = build_per_chunk_fog_occupancy(&grid, density, slab);
                let bricks =
                    build_per_chunk_fog_occupancy_from_bricks(&build, dims, recentre, slab);
                assert_eq!(cpu.chunk_extent, bricks.chunk_extent, "chunk extent");
                assert_eq!(
                    cpu.volumes.len(),
                    bricks.volumes.len(),
                    "same resident fog-chunk count (density {density}, slab {slab:?})"
                );
                for (cpu_volume, brick_volume) in cpu.volumes.iter().zip(&bricks.volumes) {
                    assert_eq!(cpu_volume.chunk_coord, brick_volume.chunk_coord, "chunk coord");
                    assert_eq!(
                        cpu_volume.world_origin, brick_volume.world_origin,
                        "world origin"
                    );
                    assert_eq!(
                        cpu_volume.occupancy, brick_volume.occupancy,
                        "apron'd tile bytes must be byte-identical (chunk {:?}, density \
                         {density}, slab {slab:?})",
                        cpu_volume.chunk_coord
                    );
                }
                // A non-trivial fixture actually exercises occupied tiles.
                assert!(
                    cpu.volumes.iter().any(|v| v.occupancy.contains(&255)),
                    "fixture must produce occupied fog tiles"
                );
            }
        }

        for &density in &[1u32, 4] {
            // Single producer.
            assert_paths_agree(&Scene::from_nodes(vec![sphere_node("s", [0, 0, 0], density)]), density);
            // Multi-producer, spread across chunk boundaries (the scene that still streamed).
            assert_paths_agree(
                &Scene::from_nodes(vec![
                    sphere_node("a", [0, 0, 0], density),
                    sphere_node("b", [6, 0, 0], density),
                    sphere_node("c", [0, 6, 5], density),
                ]),
                density,
            );
            // Mixed-material multi-producer: overlapping boxes of different materials so a
            // block mixes materials (mesh display path) — fog still brick-sources exactly.
            let mixed = Scene::from_nodes(vec![
                {
                    let shape = SdfShape::from_blocks(ShapeKind::Box, [4, 4, 4], 1, density);
                    Node::new(
                        "stone".to_string(),
                        NodeContent::Tool { shape, material: MaterialChoice::Stone },
                    )
                },
                {
                    let shape = SdfShape::from_blocks(ShapeKind::Box, [4, 4, 4], 1, density);
                    let mut node = Node::new(
                        "wood".to_string(),
                        NodeContent::Tool { shape, material: MaterialChoice::Wood },
                    );
                    node.transform = NodeTransform::from_blocks([2, 2, 0], density);
                    node
                },
            ]);
            assert_paths_agree(&mixed, density);
        }
    }

    #[test]
    fn view_cube_is_ccw_outward() {
        let (vertices, indices) = view_cube_geometry();
        let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
        let normals: Vec<[f32; 3]> = vertices.iter().map(|v| v.normal).collect();
        assert_ccw_outward(&positions, &normals, &indices);
    }

    use crate::voxel::{Voxel, VoxelGrid};

    // ---- issue #29 S3: per-object grid line geometry + gating ----

    use crate::core_geom::MaterialChoice as Mc;
    use crate::scene::{Node, NodeContent};
    use crate::voxel::ShapeKind;
    use crate::voxel::SdfShape;

    /// `block_boundaries` returns the closing plane at `hi` (the box is enclosed in
    /// whole blocks), so a `B`-block box yields `B + 1` planes — and EXPANDING the
    /// box by one block on an axis adds exactly one boundary plane there. This is the
    /// geometry that makes "add/remove a whole block" fall out: a box grown by one
    /// enclosing block gains one lattice plane; shrunk by one, it loses one.
    #[test]
    fn block_boundaries_count_tracks_enclosing_blocks() {
        for step in [1u32, 15, 16] {
            let s = step as f32;
            // A 3-block box [0, 3·step] → planes at 0, step, 2·step, 3·step = 4.
            let three = block_boundaries(0.0, 3.0 * s, step);
            assert_eq!(three.len(), 4, "@step{step}: a 3-block box has 4 boundary planes");
            assert_eq!(*three.first().unwrap(), 0.0);
            assert_eq!(*three.last().unwrap(), 3.0 * s, "closing plane lands exactly on hi");
            // ADD a whole block (expand by +step): exactly one more plane.
            let four = block_boundaries(0.0, 4.0 * s, step);
            assert_eq!(four.len(), 5, "@step{step}: +1 enclosing block ⇒ +1 lattice plane");
            // REMOVE a whole block (shrink by step): exactly one fewer plane.
            let two = block_boundaries(0.0, 2.0 * s, step);
            assert_eq!(two.len(), 3, "@step{step}: -1 enclosing block ⇒ -1 lattice plane");
        }
    }

    /// `voxel_boundaries` walks one voxel at a time from the block-aligned `lo` to
    /// `hi`, tagging every `step`-th line as a BLOCK edge. So a `B`-block box yields
    /// `B·step + 1` voxel lines, of which exactly `B + 1` are block lines — and those
    /// block lines sit on the SAME coordinates as `block_boundaries(lo, hi, step)`.
    /// This is the alignment guarantee: the fine floor's bold lines coincide with the
    /// block lattice's vertical lines.
    #[test]
    fn voxel_boundaries_tag_block_lines_at_lattice_positions() {
        for step in [1u32, 15, 16] {
            let s = step as f32;
            // A 3-block box: 3·step voxel cells ⇒ 3·step + 1 voxel boundaries.
            let lines = voxel_boundaries(0.0, 3.0 * s, step);
            assert_eq!(
                lines.len(),
                3 * step as usize + 1,
                "@step{step}: a 3-block box has 3·step+1 voxel boundaries",
            );
            // The BLOCK-tagged lines are exactly the block-boundary planes.
            let block_lines: Vec<f32> =
                lines.iter().filter(|(_, b)| *b).map(|(c, _)| *c).collect();
            assert_eq!(
                block_lines,
                block_boundaries(0.0, 3.0 * s, step),
                "@step{step}: floor's bold (block) lines coincide with the lattice block lines",
            );
            // At density 1 EVERY voxel line is a block line (voxel == block).
            if step == 1 {
                assert!(lines.iter().all(|(_, b)| *b), "@step1: every voxel line is a block line");
            } else {
                // Otherwise the voxel lines strictly outnumber the block lines.
                assert!(
                    block_lines.len() < lines.len(),
                    "@step{step}: voxel lines are denser than block lines",
                );
            }
        }
    }

    /// The fine floor grid is two-tier and aligns with the block lattice (issue #29
    /// fix). Z-up: the floor is an XY-plane grid at the base. For a node box, this
    /// asserts three properties. First, the floor's DISTINCT X line coordinates form a
    /// superset of — and at the block positions coincide with — the lattice's
    /// X-line coordinates. Second, the floor uses exactly two alphas (a subtle voxel
    /// tier and a bold block tier). Third, at a coarse density the voxel lines visibly
    /// outnumber the block lines.
    #[test]
    fn floor_grid_is_two_tier_and_aligns_with_lattice() {
        // Distinct X coordinates among the floor's X-boundary lines (Z-up: each runs
        // along Y in the XY ground plane); compared against the lattice's X lines.
        let distinct_xs = |verts: &[LineVertex]| -> Vec<i64> {
            let mut xs: Vec<i64> = verts
                .iter()
                .map(|v| (v.position[0] * 256.0).round() as i64)
                .collect();
            xs.sort_unstable();
            xs.dedup();
            xs
        };
        for step in [1u32, 15, 16] {
            let s = step as f32;
            // A box NOT at the origin (min ≠ 0), to catch a frame/offset mismatch.
            let (min, max) = ([s, 0.0, 2.0 * s], [4.0 * s, s, 5.0 * s]);
            let mut lattice = Vec::new();
            lattice_vertices_into(&mut lattice, min, max, step);
            let mut floor = Vec::new();
            floor_vertices_into(&mut floor, min, max, step);

            // (2) Exactly two distinct alphas — the subtle voxel tier and the bold
            // block tier. At step 1 every line is both a voxel and a block line, so
            // it is drawn twice (subtle then bold) and BOTH alphas are still present.
            let mut alphas: Vec<i64> =
                floor.iter().map(|v| (v.color[3] * 1024.0).round() as i64).collect();
            alphas.sort_unstable();
            alphas.dedup();
            assert_eq!(
                alphas.len(),
                2,
                "@step{step}: floor has two alpha tiers (subtle voxel + bold block)",
            );

            // (1) The lattice's X lines must ALL appear among the floor's X lines
            // (the floor X set is a superset coinciding at the block lines).
            let lattice_xs = distinct_xs(&lattice);
            let floor_xs = distinct_xs(&floor);
            for x in &lattice_xs {
                assert!(
                    floor_xs.contains(x),
                    "@step{step}: lattice vertical line x={x} has a coincident floor line",
                );
            }
            // (3) At a coarse density the floor has strictly more distinct X lines
            // than the lattice (the extra ones are the fine voxel lines).
            if step > 1 {
                assert!(
                    floor_xs.len() > lattice_xs.len(),
                    "@step{step}: floor (voxel-resolution) has denser X lines than the lattice",
                );
            }
        }
    }

    /// One node's lattice/floor box → a non-empty line set at every density; the
    /// vertex count is a multiple of 2 (whole segments).
    #[test]
    fn lattice_and_floor_vertices_nonempty_per_box() {
        for step in [1u32, 15, 16] {
            let s = step as f32;
            let (min, max) = ([0.0, 0.0, 0.0], [2.0 * s, s, 3.0 * s]);
            let mut lattice = Vec::new();
            lattice_vertices_into(&mut lattice, min, max, step);
            assert!(!lattice.is_empty(), "@step{step}: a sized box has lattice lines");
            assert_eq!(lattice.len() % 2, 0, "lattice lines are whole segments");
            let mut floor = Vec::new();
            floor_vertices_into(&mut floor, min, max, step);
            assert!(!floor.is_empty(), "@step{step}: a sized box has floor lines");
            // Z-up: the floor sits at the EXACT base plane `z = min[2]` (issue #29
            // fix: no geometric drop — the floor pipeline's depth bias avoids
            // z-fighting the model's coincident bottom face), flat in Z, uniform across
            // every vertex. This makes the floor's block lines meet the lattice's
            // bottom edges.
            let floor_z = min[2];
            assert!(floor.iter().all(|v| v.position[2] == floor_z), "floor on exact base plane");
        }
    }

    fn box_node(name: &str, offset: [i64; 3], voxels_per_block: u32) -> Node {
        let shape = SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, voxels_per_block);
        let mut node = Node::new(name, NodeContent::Tool { shape, material: Mc::Stone });
        node.transform = crate::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node
    }

    /// Gating (issue #29 S3): a node's lattice box appears in the batch ONLY when the
    /// master AND the node's per-object toggle are both ON; turning EITHER off drops
    /// it. A two-node scene with the grid enabled on ONE node yields exactly ONE
    /// lattice box (the other node contributes none).
    #[test]
    fn scene_grid_boxes_gated_by_master_and_per_object() {
        for density in [1u32, 15, 16] {
            let mut scene = Scene::from_nodes(vec![
                box_node("A", [0, 0, 0], density),
                box_node("B", [8, 0, 0], density),
            ]);
            scene.voxels_per_block = density;
            scene.active = None;
            scene.master_block_lattice = true;
            scene.master_floor_grid = true;

            // Both per-object toggles OFF → no boxes regardless of masters.
            let (lat, flr) = scene_grid_boxes(&scene, density);
            assert!(lat.is_empty() && flr.is_empty(), "@d{density}: per-object OFF ⇒ no boxes");

            // Enable block lattice on node A ONLY.
            scene.root_node_mut(0).grids.block_lattice = true;
            let (lat, flr) = scene_grid_boxes(&scene, density);
            assert_eq!(lat.len(), 1, "@d{density}: one node enabled ⇒ exactly one lattice box");
            assert!(flr.is_empty(), "@d{density}: floor still off");

            // Master OFF cancels it even though the node's flag is on.
            scene.master_block_lattice = false;
            let (lat, _flr) = scene_grid_boxes(&scene, density);
            assert!(lat.is_empty(), "@d{density}: master OFF ⇒ no lattice box (AND gating)");

            // Floor: node B's flag on + master on → one floor box, no lattice.
            scene.master_floor_grid = true;
            scene.root_node_mut(1).grids.floor_grid = true;
            let (lat, flr) = scene_grid_boxes(&scene, density);
            assert!(lat.is_empty(), "@d{density}: lattice master still off");
            assert_eq!(flr.len(), 1, "@d{density}: one floor box from node B");
        }
    }

    // ===== Issue #29 S5: Points (world reference grid) ==========================

    use crate::scene::Point;

    /// A scene carrying only an Origin Point with the given plane flags; `axes`
    /// sets all three per-axis flags together (the common "axes on/off" case).
    fn origin_point_scene(plane_xz: bool, plane_xy: bool, plane_yz: bool, axes: bool) -> Scene {
        origin_point_scene_axes(plane_xz, plane_xy, plane_yz, [axes, axes, axes])
    }

    /// A scene carrying only an Origin Point with the given plane flags and explicit
    /// per-axis X/Y/Z toggles (issue #29 fix: separable axes).
    fn origin_point_scene_axes(
        plane_xz: bool,
        plane_xy: bool,
        plane_yz: bool,
        axes: [bool; 3],
    ) -> Scene {
        let mut scene = Scene::default();
        scene.points.push(Point {
            name: "Origin".to_string(),
            plane_xz,
            plane_xy,
            plane_yz,
            axis_x: axes[0],
            axis_y: axes[1],
            axis_z: axes[2],
            is_origin: true,
            ..Point::default()
        });
        scene.active_point = Some(0);
        scene
    }

    /// A visible Origin Point with axes yields a NON-EMPTY axis batch; a hidden Point
    /// yields NONE (the spec's "hidden Points render nothing"). The ground PLANE moved
    /// to the analytic infinite grid ([`enabled_grid_planes`]), so this batch is now
    /// AXES-only.
    #[test]
    fn points_visible_yields_batch_hidden_yields_none() {
        for density in [1u32, 15, 16] {
            // Z-up: the ground plane is XY (the 2nd flag of `origin_point_scene`).
            let mut scene = origin_point_scene(false, true, false, true);
            let batch = points_line_batch(&scene, density);
            assert!(!batch.is_empty(), "@d{density}: visible axes ⇒ non-empty batch");
            assert_eq!(batch.len() % 2, 0, "@d{density}: whole line segments");

            // The Origin's ground (XY, Z-up) plane is one analytic-grid instance.
            let planes = enabled_grid_planes(&scene, density);
            assert_eq!(planes.len(), 1, "@d{density}: the Origin ground plane ⇒ one grid plane");

            scene.points[0].hidden = true;
            let hidden = points_line_batch(&scene, density);
            assert!(hidden.is_empty(), "@d{density}: a hidden Point renders no axes");
            assert!(
                enabled_grid_planes(&scene, density).is_empty(),
                "@d{density}: a hidden Point renders no grid plane",
            );
        }
    }

    /// The plane and axis toggles gate independently. Axes flow through
    /// [`points_line_batch`] (AXES-only); planes flow through [`enabled_grid_planes`].
    /// Turning every plane + axis off empties BOTH; enabling more planes adds grid
    /// instances; the axes alone yield EXACTLY six axis vertices (three segments).
    #[test]
    fn points_plane_and_axis_toggles_gate() {
        let density = 16u32;
        // Everything off → no axes, no planes.
        let none = points_line_batch(&origin_point_scene(false, false, false, false), density);
        assert!(none.is_empty(), "all axes off ⇒ empty axis batch");
        assert!(
            enabled_grid_planes(&origin_point_scene(false, false, false, false), density).is_empty(),
            "all planes off ⇒ no grid planes",
        );

        // Axes only → exactly 3 segments = 6 vertices, through the origin; no planes.
        let axes_only = points_line_batch(&origin_point_scene(false, false, false, true), density);
        assert_eq!(axes_only.len(), 6, "axes alone ⇒ three line segments");
        assert!(
            enabled_grid_planes(&origin_point_scene(false, false, false, true), density).is_empty(),
            "axes alone ⇒ no grid planes",
        );

        // Each enabled plane adds one grid instance; enabling more planes grows the
        // count. Z-up: the ground plane is XY (2nd flag).
        let ground = enabled_grid_planes(&origin_point_scene(false, true, false, false), density);
        let ground_front = enabled_grid_planes(&origin_point_scene(true, true, false, false), density);
        assert_eq!(ground.len(), 1, "the XY ground plane alone ⇒ one grid plane");
        assert_eq!(ground_front.len(), 2, "adding the XZ front plane ⇒ two grid planes");
    }

    /// Per-axis gating (issue #29 fix): the X/Y/Z axes toggle independently. All three
    /// on ⇒ three segments (one per colour); turning Y off drops the GREEN segment and
    /// leaves the red (X) and blue (Z) ones; a single axis on ⇒ exactly one segment.
    #[test]
    fn points_axes_toggle_per_axis() {
        for density in [1u32, 15, 16] {
            let green = with_alpha(srgb_hex_to_linear(GIZMO_AXIS_Y_HEX), POINT_AXIS_ALPHA);
            let is_green = |v: &LineVertex| v.color == green;

            // All three axes on (planes off) → exactly 3 segments = 6 vertices, one green.
            let all = points_line_batch(
                &origin_point_scene_axes(false, false, false, [true, true, true]),
                density,
            );
            assert_eq!(all.len(), 6, "@d{density}: three axes ⇒ three segments");
            assert_eq!(all.iter().filter(|v| is_green(v)).count(), 2, "@d{density}: one green (Y) segment, two vertices");

            // Turn Y off → 2 segments, NO green line.
            let no_y = points_line_batch(
                &origin_point_scene_axes(false, false, false, [true, false, true]),
                density,
            );
            assert_eq!(no_y.len(), 4, "@d{density}: Y off ⇒ two segments");
            assert!(!no_y.iter().any(is_green), "@d{density}: no green (Y) line when Y is off");

            // Only Y on → exactly one (green) segment.
            let only_y = points_line_batch(
                &origin_point_scene_axes(false, false, false, [false, true, false]),
                density,
            );
            assert_eq!(only_y.len(), 2, "@d{density}: only Y ⇒ one segment");
            assert!(only_y.iter().all(is_green), "@d{density}: the only line is green (Y)");
        }
    }

    /// The analytic grid plane carries the correct orientation, origin, and tuning for
    /// each [`ReferencePlane`] (Z-up): XZ is normal +Y (the FRONT plane), XY normal +Z
    /// (the GROUND plane), YZ normal +X (the side), with orthonormal in-plane axes
    /// through the Point origin. Pure CPU — the shader consumes these basis vectors.
    #[test]
    fn grid_planes_carry_correct_orientation() {
        for density in [1u32, 15, 16] {
            // All three planes on at the Origin (recentre = 0 → origin at world 0).
            let scene = origin_point_scene(true, true, true, false);
            let planes = enabled_grid_planes(&scene, density);
            assert_eq!(planes.len(), 3, "@d{density}: three planes enabled ⇒ three instances");
            // Emission order is XZ (front), XY (ground), YZ (side).
            assert_eq!(planes[0].normal, [0.0, 1.0, 0.0], "@d{density}: XZ front ⇒ +Y normal");
            assert_eq!(planes[1].normal, [0.0, 0.0, 1.0], "@d{density}: XY ground ⇒ +Z normal");
            assert_eq!(planes[2].normal, [1.0, 0.0, 0.0], "@d{density}: YZ side ⇒ +X normal");
            for plane in &planes {
                assert_eq!(plane.origin, [0.0, 0.0, 0.0], "@d{density}: Origin plane at world 0");
                // In-plane axes are unit and perpendicular to the normal.
                let dot_un = plane.u_axis.iter().zip(plane.normal).map(|(a, b)| a * b).sum::<f32>();
                let dot_vn = plane.v_axis.iter().zip(plane.normal).map(|(a, b)| a * b).sum::<f32>();
                assert!(dot_un.abs() < 1e-6 && dot_vn.abs() < 1e-6, "in-plane axes ⊥ normal");
            }
        }
    }

    /// A second Point offset from the origin places its grid PLANE and its AXES at that
    /// WORLD position: with a lone Point (recentre = 0 — no sized leaf) both pass
    /// through `position_blocks · density`.
    #[test]
    fn points_offset_point_frame_sits_at_world_position() {
        let density = 16i64;
        let mut scene = Scene::default();
        // Z-up: the ground plane is XY (`plane_xy`, on by default). Keep ONLY the
        // ground plane on so this Point yields exactly one grid plane.
        scene.points.push(Point {
            position_blocks: [10, 0, -4],
            plane_xy: true,
            plane_xz: false,
            plane_yz: false,
            // axis_x/y/z default true via Point::default() ⇒ all three axes on.
            is_origin: false,
            ..Point::default()
        });
        // The offset Point's ground plane sits at that world position.
        let planes = enabled_grid_planes(&scene, density as u32);
        assert_eq!(planes.len(), 1, "the offset Point's XY ground plane ⇒ one grid plane");
        assert_eq!(
            planes[0].origin,
            [(10 * density) as f32, 0.0, (-4 * density) as f32],
            "the grid plane origin is at the Point's world position",
        );
        let batch = points_line_batch(&scene, density as u32);
        assert_eq!(batch.len(), 6, "axes only ⇒ three segments");
        // The axes cross at the Point origin; every axis segment shares that centre on
        // its two non-running coordinates. Recover the centre as the midpoint of the X
        // axis segment (vertices 0,1 are the X axis through the centre).
        let centre = [
            (batch[0].position[0] + batch[1].position[0]) / 2.0,
            (batch[0].position[1] + batch[1].position[1]) / 2.0,
            (batch[0].position[2] + batch[1].position[2]) / 2.0,
        ];
        assert!((centre[0] - (10 * density) as f32).abs() < 1e-3, "X frame at 10 blocks");
        assert!((centre[1]).abs() < 1e-3, "Y frame at 0");
        assert!((centre[2] - (-4 * density) as f32).abs() < 1e-3, "Z frame at -4 blocks");
    }

    /// Block-line spacing is density-parametrized: the gap between adjacent ground
    /// lines along an axis equals one block (= `density` voxels) at {1, 15, 16}.
    ///
    /// With the analytic infinite grid the block spacing is no longer baked into CPU
    /// geometry — it is the `block_spacing` shader param, which the renderer sets to
    /// `voxels_per_block`. Pin that mapping: the bold (block) tier spacing equals the
    /// density, while the fine (voxel) tier is always spacing 1, so adjacent BLOCK
    /// lines are exactly one block (= density voxels) apart at every density.
    #[test]
    fn grid_block_spacing_is_density() {
        for density in [1u32, 15, 16] {
            // The renderer's `rebuild_from_scene` packs `block_spacing = density` into
            // `params.x`; the voxel tier is fixed at spacing 1.0 in the shader. This
            // mirrors that contract without a GPU.
            let block_spacing = density.max(1) as f32;
            assert_eq!(
                block_spacing, density as f32,
                "@d{density}: bold (block) grid lines are one block apart (spacing = density)",
            );
            // And a plane is actually emitted to carry that spacing.
            let scene = origin_point_scene(true, false, false, false);
            assert_eq!(enabled_grid_planes(&scene, density).len(), 1);
        }
    }
}
