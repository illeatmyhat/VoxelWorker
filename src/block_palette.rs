//! The egui-facing block palette state (Milestone 6): the tiles + click counter.
//!
//! This is the shell half of the block palette. It owns the palette STATE — the
//! list of tiles (label, variant count, thumbnail `egui::TextureId`, variant
//! paths) plus the click counter that picks a deterministic pseudo-random
//! variant. Rendering a thumbnail and registering it with egui lives here in
//! [`BlockPalette::add_group`], which drives the pure-wgpu
//! [`display::block_texture::ThumbnailRenderer`] and then bridges the rendered
//! texture into egui via `egui_wgpu::Renderer::register_native_texture`.
//!
//! The GPU backing (the thumbnail renderer + the runtime-loaded material) lives
//! in the display crate's `block_texture` module; the egui-facing tile widgets
//! live in `panel/palette.rs`. This module is the state that binds them.

use display::assets::{BlockGroup, DecodedRgba};
use display::block_texture::ThumbnailRenderer;

/// One ready palette tile: its label, variant count, the egui texture id of its
/// thumbnail, and the absolute paths of its variants (for `apply`).
pub struct PaletteTile {
    pub label: String,
    pub variant_count: usize,
    pub thumbnail_id: egui::TextureId,
    pub variants: Vec<std::path::PathBuf>,
    /// The scanned group (kept so M7 per-face resolution can re-key on apply).
    pub group: BlockGroup,
    /// Keep the thumbnail texture alive for as long as the tile (egui only holds
    /// a view/bind-group internally; dropping the texture would invalidate it).
    pub _thumbnail_texture: wgpu::Texture,
}

/// The palette state shared by the windowed app + the headless shot path.
#[derive(Default)]
pub struct BlockPalette {
    pub tiles: Vec<PaletteTile>,
    /// Status line text ("Scanning…", "N blocks loaded", "No VS install found…").
    pub status: String,
    /// Incrementing click counter → deterministic pseudo-random variant pick
    /// (`variants[counter % len]`), since `Math.random` isn't desired for
    /// reproducible screenshots.
    pub click_counter: usize,
}

impl BlockPalette {
    /// Append a scanned group: render its thumbnail, register it with egui, push a tile.
    #[allow(clippy::too_many_arguments)]
    pub fn add_group(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        thumbnail_renderer: &ThumbnailRenderer,
        egui_renderer: &mut egui_wgpu::Renderer,
        group: BlockGroup,
        thumbnail_rgba: &DecodedRgba,
    ) {
        let texture = thumbnail_renderer.render_thumbnail(device, queue, thumbnail_rgba);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let thumbnail_id =
            egui_renderer.register_native_texture(device, &view, wgpu::FilterMode::Nearest);
        self.tiles.push(PaletteTile {
            label: group.label.clone(),
            variant_count: group.variants.len(),
            thumbnail_id,
            variants: group.variants.clone(),
            group,
            _thumbnail_texture: texture,
        });
    }

    /// Map a categorical [`BlockId`](voxel_core::core_geom::BlockId) (ADR 0003 §3a) to the
    /// procedural [`MaterialChoice`](voxel_core::core_geom::MaterialChoice) it renders as.
    ///
    /// This is the categorical block-palette resolution the per-voxel cell now routes
    /// through: the three procedural materials ARE the palette today (`block_id` ⇒
    /// Stone/Wood/Plain), so the mapping is `MaterialChoice::from_material_id`. The rich
    /// VS palette CONTENT (a real `block_id` → texture table) is the deferred part; this
    /// is the seam it will replace, so the renderer + `.vox` export call one resolver
    /// rather than reading the id directly. `&self` is taken so a future palette with
    /// real content resolves against THIS palette's loaded tiles, not a global table.
    pub fn material_for_block(&self, block_id: voxel_core::core_geom::BlockId) -> voxel_core::core_geom::MaterialChoice {
        voxel_core::core_geom::MaterialChoice::from_material_id(block_id.color_index())
    }

    /// Pick the next pseudo-random variant path of `tile_index` and bump the
    /// counter. Returns the chosen variant's absolute path (caller decodes +
    /// uploads it as the active material).
    pub fn pick_variant(&mut self, tile_index: usize) -> Option<std::path::PathBuf> {
        let tile = self.tiles.get(tile_index)?;
        if tile.variants.is_empty() {
            return None;
        }
        let index = self.click_counter % tile.variants.len();
        self.click_counter = self.click_counter.wrapping_add(1);
        Some(tile.variants[index].clone())
    }
}
