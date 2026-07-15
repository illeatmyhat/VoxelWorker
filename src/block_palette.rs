//! The block palette, split across the shell/UI boundary (ADR 0016 Phase 8b).
//!
//! The palette has two halves that this module keeps index-aligned:
//!
//!   * The **UI-facing state** — [`BlockPalette`] / [`PaletteTile`] — is pure egui +
//!     std + `voxel_core`: the list of tiles (label, variant count, thumbnail
//!     `egui::TextureId`, variant paths) plus the click counter that picks a
//!     deterministic pseudo-random variant. It links NO wgpu. This is the half that
//!     lives in the `ui` crate ([`ui::palette`]); the shell re-exports it here for the
//!     GPU host below and its consumers.
//!   * The **shell-side GPU host** — [`PaletteHost`] — owns the wgpu backing the tiles
//!     cannot name: the [`ThumbnailRenderer`] that draws each 45° cube thumbnail, the
//!     `wgpu::Texture` keep-alives (egui only holds a view/bind-group internally, so
//!     dropping the texture would invalidate the tile), and the scanned
//!     [`BlockGroup`]s (kept so M7 per-face resolution can re-key on apply). It renders
//!     + registers a thumbnail with egui, then pushes one entry to ALL THREE vecs.
//!
//! The INVARIANT the host upholds: `ui.tiles`, `keepalive`, and `groups` are always
//! the same length and index-aligned — a tile at index `i` owns `keepalive[i]` and was
//! scanned from `groups[i]`. Only [`PaletteHost::add_group`] (push) and
//! [`PaletteHost::clear`] (truncate all three) mutate them, so the alignment holds.

use display::assets::{BlockGroup, DecodedRgba};
use display::block_texture::ThumbnailRenderer;
pub use ui::palette::{BlockPalette, PaletteTile};

/// The shell's GPU host for the palette: it owns the wgpu resources the UI-facing
/// [`BlockPalette`] cannot name (the thumbnail renderer, the texture keep-alives) and
/// the scanned [`BlockGroup`]s used for per-face resolution, and it keeps them
/// index-aligned with `ui.tiles` (see the module invariant).
pub struct PaletteHost {
    /// The UI-facing palette state (tiles + status + click counter) — the half handed
    /// to `ui::panel::build_panel`.
    pub ui: BlockPalette,
    /// Keep each tile's thumbnail texture alive for as long as the tile: egui only
    /// holds a view/bind-group internally, so dropping the texture invalidates it.
    /// Index-aligned with `ui.tiles`.
    keepalive: Vec<wgpu::Texture>,
    /// The scanned group behind each tile (M7 per-face resolution re-keys on apply).
    /// Index-aligned with `ui.tiles`.
    groups: Vec<BlockGroup>,
    /// Offscreen renderer for the 45° palette cube thumbnails (M6).
    thumbnail_renderer: ThumbnailRenderer,
}

impl PaletteHost {
    /// Build an empty host with the given initial status line, constructing the
    /// thumbnail renderer against the device/queue.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, status: String) -> Self {
        Self {
            ui: BlockPalette {
                status,
                ..BlockPalette::default()
            },
            keepalive: Vec::new(),
            groups: Vec::new(),
            thumbnail_renderer: ThumbnailRenderer::new(device, queue),
        }
    }

    /// Append a scanned group: render its thumbnail, register it with egui, and push an
    /// index-aligned entry to all three of `ui.tiles`, `keepalive`, and `groups`.
    pub fn add_group(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        egui_renderer: &mut egui_wgpu::Renderer,
        group: BlockGroup,
        thumbnail_rgba: &DecodedRgba,
    ) {
        let texture = self
            .thumbnail_renderer
            .render_thumbnail(device, queue, thumbnail_rgba);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let thumbnail_id =
            egui_renderer.register_native_texture(device, &view, wgpu::FilterMode::Nearest);
        self.ui.tiles.push(PaletteTile {
            label: group.label.clone(),
            variant_count: group.variants.len(),
            thumbnail_id,
            variants: group.variants.clone(),
        });
        self.keepalive.push(texture);
        self.groups.push(group);
    }

    /// Pick the next pseudo-random variant path of `tile_index` (delegates to the
    /// UI-facing [`BlockPalette::pick_variant`]).
    pub fn pick_variant(&mut self, tile_index: usize) -> Option<std::path::PathBuf> {
        self.ui.pick_variant(tile_index)
    }

    /// The scanned [`BlockGroup`] behind tile `tile_index` (for M7 per-face
    /// resolution), or `None` if out of range.
    pub fn group(&self, tile_index: usize) -> Option<&BlockGroup> {
        self.groups.get(tile_index)
    }

    /// Clear all tiles: truncates `ui.tiles`, `keepalive`, and `groups` in lockstep so
    /// the index-alignment invariant holds.
    pub fn clear(&mut self) {
        self.ui.tiles.clear();
        self.keepalive.clear();
        self.groups.clear();
    }
}
