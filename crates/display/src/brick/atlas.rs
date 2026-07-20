use super::*;

/// The `bricks_per_axis` a slot-tile count packs to (`ceil(cbrt(count))`, 0 for empty) —
/// the atlas tile-grid edge, shared by the packer and the grow test. This IS substrate's
/// [`CubeTilePacking::tiles_per_axis`]; the wrapper keeps the domain name at the seam.
pub(crate) fn sculpted_atlas_bricks_per_axis(slot_count: usize) -> u32 {
    CubeTilePacking::tiles_per_axis(slot_count)
}

/// Land the sculpted-brick atlas bytes in an R8Unorm 3D texture via a plain `write_texture`
/// upload — no row-padding requirement (unlike `copy_texture_to_buffer`'s 256-byte alignment,
/// see `read_back_brick_atlas` below). `COPY_SRC` is set so the parity net can read the texture
/// back; a build with no sculpted brick returns a 1³ placeholder (nothing samples it — every
/// record is coarse/air).
pub fn upload_brick_atlas(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    atlas: &SculptedAtlasPayload,
) -> wgpu::Texture {
    let atlas_dim = atlas.geometry.atlas_dim_voxels.max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("brick-field sculpted atlas"),
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
    if atlas.geometry.atlas_dim_voxels > 0 {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas.bytes,
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
    }
    texture
}

/// Land the MATERIAL SIDE ATLAS's cell-key bytes in an **R16Uint** 3D texture — the second,
/// independently pooled atlas beside [`upload_brick_atlas`]'s R8 occupancy one. `R16Uint`
/// because the texel IS the `u16` cell key verbatim (palette id + overlay bit): an integer
/// sampled with `textureLoad` and compared exactly, never filtered or normalised — a float
/// format would round the id. Two bytes per texel (little-endian, the packer's order), so a
/// row is `2 · edge` bytes. `COPY_SRC` is set for the parity net's readback; a field with no
/// MIXED brick returns a 1³ placeholder (nothing samples it — every record carries its one
/// cell key).
///
/// Known limit (inherited from the occupancy atlas, not introduced here): the app requests
/// `Limits::default()`, so `max_texture_dimension_3d` is 2048 and there is no pre-allocation
/// VRAM budget guard — see docs/design/vram-ceiling-probe.md. The side atlas is sparse (only
/// mixed bricks), so it reaches that ceiling far later than the occupancy pool does.
pub fn upload_brick_cell_key_atlas(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    atlas: &SculptedCellKeyAtlasPayload,
) -> wgpu::Texture {
    let atlas_dim = atlas.geometry.atlas_dim_voxels.max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("brick-field cell-key side atlas"),
        size: wgpu::Extent3d {
            width: atlas_dim,
            height: atlas_dim,
            depth_or_array_layers: atlas_dim,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R16Uint,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    if atlas.geometry.atlas_dim_voxels > 0 {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas.bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(atlas_dim * CELL_KEY_TEXEL_BYTES),
                rows_per_image: Some(atlas_dim),
            },
            wgpu::Extent3d {
                width: atlas_dim,
                height: atlas_dim,
                depth_or_array_layers: atlas_dim,
            },
        );
    }
    texture
}

/// Bytes per cell-key texel — the R16Uint stride (one little-endian `u16` per voxel). The ONE
/// name for the "2" every side-atlas row/extent arithmetic multiplies by.
pub const CELL_KEY_TEXEL_BYTES: u32 = 2;

/// Read an `atlas_dim³` R8 atlas texture back to row-unpadded bytes — the parity net's
/// A/B readback ONLY (mirrors `dispatch_atlas`; per ADR 0006 §4 nothing ever reads a
/// texture back as truth on a live path).
pub fn read_back_brick_atlas(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    atlas_dim: u32,
) -> Vec<u8> {
    if atlas_dim == 0 {
        return Vec::new();
    }
    // `copy_texture_to_buffer` rows must be 256-aligned (unlike `write_texture`).
    const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;
    let padded_row = atlas_dim.div_ceil(COPY_BYTES_PER_ROW_ALIGNMENT) * COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes = padded_row as u64 * atlas_dim as u64 * atlas_dim as u64;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("brick-field atlas readback"),
        size: padded_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(atlas_dim),
            },
        },
        wgpu::Extent3d {
            width: atlas_dim,
            height: atlas_dim,
            depth_or_array_layers: atlas_dim,
        },
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

    let mapped = slice.get_mapped_range();
    let atlas_dim_usize = atlas_dim as usize;
    let padded_row_usize = padded_row as usize;
    let mut atlas_bytes = vec![0u8; atlas_dim_usize.pow(3)];
    for atlas_z in 0..atlas_dim_usize {
        for atlas_y in 0..atlas_dim_usize {
            let source = (atlas_z * atlas_dim_usize + atlas_y) * padded_row_usize;
            let destination = (atlas_z * atlas_dim_usize + atlas_y) * atlas_dim_usize;
            atlas_bytes[destination..destination + atlas_dim_usize]
                .copy_from_slice(&mapped[source..source + atlas_dim_usize]);
        }
    }
    drop(mapped);
    readback.unmap();
    atlas_bytes
}
