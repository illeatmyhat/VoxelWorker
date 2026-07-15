//! The UI-facing block palette state: the tiles + status + click counter.
//!
//! This is the egui-facing half of the block palette (ADR 0016 Phase 8b). It owns the
//! palette STATE — the list of [`PaletteTile`]s (label, variant count, thumbnail
//! `egui::TextureId`, variant paths) plus the click counter that picks a deterministic
//! pseudo-random variant. It links NO wgpu: the tiles reference their thumbnails only as
//! already-registered [`egui::TextureId`]s handed in by the shell.
//!
//! The GPU backing (the thumbnail renderer that draws each cube, the `wgpu::Texture`
//! keep-alives, the scanned `BlockGroup`s used for per-face resolution) lives in the
//! shell's `PaletteHost`, which keeps those resources index-aligned with [`tiles`]
//! (`PaletteHost::add_group` pushes one entry to each in lockstep).
//!
//! [`tiles`]: BlockPalette::tiles

/// One ready palette tile: its label, variant count, the egui texture id of its
/// thumbnail, and the absolute paths of its variants (for the apply path).
///
/// It carries NO wgpu type: the thumbnail is an [`egui::TextureId`] the shell
/// registered against egui-wgpu, and the texture it points at is kept alive alongside
/// this tile by the shell's `PaletteHost`.
pub struct PaletteTile {
    pub label: String,
    pub variant_count: usize,
    pub thumbnail_id: egui::TextureId,
    pub variants: Vec<std::path::PathBuf>,
}

/// The UI-facing palette state shared by the windowed app + the headless shot path.
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
    pub fn material_for_block(
        &self,
        block_id: voxel_core::core_geom::BlockId,
    ) -> voxel_core::core_geom::MaterialChoice {
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
