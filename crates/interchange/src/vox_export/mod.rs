//! MagicaVoxel `.vox` export (Milestone 8).
//!
//! Serialises a resolved [`VoxelGrid`] to a MagicaVoxel
//! `.vox` file so the result can be ingested by the **Automatic Chiselling
//! REBORN** Vintage Story mod. The chunked binary is
//! hand-written (no crate dependency) — it is a `VOX ` magic + version 150
//! header followed by one `MAIN` chunk that contains, per model, a `SIZE` and an
//! `XYZI` chunk, plus a single trailing `RGBA` palette chunk.
//!
//! ## Axis convention (documented, the bit that bites)
//!
//! MagicaVoxel uses a **Z-up** right-handed coordinate system; our world is **also
//! Z-up** (vertical = +Z, index 2). So our grid index `(i, j, k)` maps DIRECTLY to
//! a vox coordinate `(x, y, z) = (i, j, k)` — **no axis swap**. (Before the Z-up
//! reorientation our world was Y-up and this path swapped Y/Z to stand the model
//! upright; now the model is already Z-up so the swap is gone.)
//!
//! ## 256 limit
//!
//! A single MagicaVoxel model caps at 256 voxels per axis (coords are `u8`). If
//! any grid dimension exceeds 256 we **split** along that axis into a tiled set
//! of models inside the one MAIN chunk (each ≤ 256), rather than truncating.
//! Typical grids (e.g. 80×16×80) are a single model. The split is documented in
//! [`VoxExport::model_count`].
//!
//! ## Colour (ADR 0003 §3a — block-palette mapping)
//!
//! Each voxel's categorical `block_id` (ADR 0003 §3a) maps through the active block
//! palette to a `.vox` palette slot: file palette index `block_id + 1` carries that
//! block's colour, and every voxel references its own block's slot. The palette is
//! built over the existing three procedural materials (the categorical CAPABILITY; the
//! rich VS palette content stays deferred), so a single-material scene (every voxel
//! `block_id 0`) still references ONE slot and is byte-identical to the old single-
//! representative-colour export. A multi-material scene now exports each block in its
//! own palette colour rather than collapsing to one.

use voxel_core::voxel::VoxelGrid;

/// The per-`block_id` RGBA palette the `.vox` export writes (ADR 0003 §3a). Index `i`
/// is the colour for `block_id == i`; it is written to `.vox` file palette slot `i + 1`
/// (MagicaVoxel palette is 1-based, 0 = empty). Sized to the procedural material set —
/// the categorical capability over the existing three materials.
pub type BlockPaletteColors = [[u8; 4]; voxel_core::core_geom::MaterialChoice::MATERIAL_COUNT];

/// MagicaVoxel per-axis maximum (coordinates are stored as `u8`, 0..=255).
pub const VOX_AXIS_MAX: u32 = 256;

/// One occupied voxel placed into vox (Z-up) coordinate space, ready to write.
/// Coordinates are the LOCAL coordinates within the owning model after any
/// 256-split tiling has been applied.
#[derive(Debug, Clone, Copy)]
struct VoxVoxel {
    x: u8,
    y: u8,
    z: u8,
    /// File palette index (1-based; 0 = empty).
    color_index: u8,
}

/// One model destined for the `.vox` file: a size plus its voxels (local coords).
#[derive(Debug, Clone)]
struct VoxModel {
    size: [u32; 3],
    voxels: Vec<VoxVoxel>,
}

/// The prepared export: a set of models (one unless the 256-limit forced a
/// split) plus the palette. Build it with [`VoxExport::from_grid`], then either
/// [`VoxExport::to_bytes`] or [`VoxExport::write`].
pub struct VoxExport {
    models: Vec<VoxModel>,
    /// Per-`block_id` RGBA palette (ADR 0003 §3a): slot `block_id` written to `.vox`
    /// file palette index `block_id + 1`, which each voxel of that block references.
    palette_colors: BlockPaletteColors,
    /// Total occupied voxels written across all models (== grid.occupied.len()).
    voxel_count: usize,
}

impl VoxExport {
    /// Build the export from a resolved grid (Z-up, no axis swap) and tiling into
    /// ≤256 models if any dimension exceeds [`VOX_AXIS_MAX`].
    ///
    /// `palette_colors` maps each `block_id` to its RGBA palette colour (ADR 0003 §3a;
    /// build it with `block_palette_from_active` or pass the procedural material
    /// colours). Each voxel references `block_id + 1` in the `.vox` palette.
    pub fn from_grid(grid: &VoxelGrid, palette_colors: BlockPaletteColors) -> Self {
        // One bucketing path: the whole-grid case is the region case with a single
        // grid covering the whole region (issue #20 S6d). Keeping ONE code path
        // guarantees the region-scoped export can never drift from the whole-grid
        // export — they are literally the same function.
        Self::from_region_voxels(
            grid.dimensions,
            std::iter::once(&grid.occupied[..]),
            palette_colors,
        )
    }

    /// Build a [`BlockPaletteColors`] in which the ACTIVE material's slot carries
    /// `representative_rgba` and the other procedural materials carry a neutral grey.
    /// This is the categorical seam the single-material `.vox` export used to inline as
    /// one representative colour (ADR 0003 §3a): a single-material scene (every voxel
    /// `block_id == active`) still references one slot, so its file bytes are unchanged.
    pub fn block_palette_from_active(
        active: voxel_core::core_geom::MaterialChoice,
        representative_rgba: [u8; 4],
    ) -> BlockPaletteColors {
        let mut palette = [[0x80, 0x80, 0x80, 0xff]; voxel_core::core_geom::MaterialChoice::MATERIAL_COUNT];
        palette[active.material_id() as usize] = representative_rgba;
        palette
    }

    /// **Region-scoped `.vox` export (issue #20 S6d).** Build the SAME export
    /// [`from_grid`](Self::from_grid) would build for the assembled monolithic
    /// region grid, but from a SET of per-chunk voxel slices — so the export
    /// consumer no longer needs the whole grid materialised once the S6c
    /// monolithic bridge is gone.
    ///
    /// `region_dimensions` are the region's voxel dimensions (exactly the assembled
    /// monolithic grid's `dimensions`): they define the tiling, the model sizes and
    /// the half-extents used to recover integer grid indices from each voxel's
    /// centred `world_position`. `chunk_voxels` yields each covering chunk's
    /// `occupied` slice; the chunks' voxels are in the SAME (recentred) coordinate
    /// frame the monolithic grid uses.
    ///
    /// ## Why this equals the whole-grid export
    ///
    /// The union of the per-chunk occupied slices is EXACTLY the monolithic region
    /// grid's occupied set (the S2 cache-assembly equivalence proof), and every
    /// voxel is bucketed by the identical `i = round(world_x + grid_x/2 − 0.5)`
    /// arithmetic [`from_grid`](Self::from_grid) uses, against the same
    /// `region_dimensions`. Each voxel therefore lands in the same model at the same
    /// local coordinate. The per-model voxel SET (and the model sizes, palette and
    /// counts) is identical; only the per-model voxel emission ORDER may differ
    /// (chunk-iteration order vs the monolithic stamp order), which a MagicaVoxel
    /// reader treats as the same model. The region export test asserts model-set
    /// equality.
    pub fn from_region_voxels<'voxels>(
        region_dimensions: [u32; 3],
        chunk_voxels: impl IntoIterator<Item = &'voxels [voxel_core::voxel::Voxel]>,
        palette_colors: BlockPaletteColors,
    ) -> Self {
        Self::from_region_voxel_iter(
            region_dimensions,
            chunk_voxels.into_iter().flatten().copied(),
            palette_colors,
        )
    }

    /// **Cacheless STREAMING `.vox` export (ADR 0010 E4).** Build the SAME export
    /// [`from_region_voxels`](Self::from_region_voxels) would build, but from a stream
    /// of OWNED per-chunk voxel `Vec`s — each chunk buffer is bucketed then DROPPED, so
    /// no whole-region dense grid is ever assembled. This is the seam the export button
    /// drives over the cacheless two-layer evaluator
    /// ([`evaluation::two_layer_store::stream_vox_occupancy`]): a coarse-solid block is a
    /// fast `d³` fill, a boundary block is per-voxel — and the **6M whole-region cap
    /// dissolves on the export path** because the only transient is one chunk's voxels.
    ///
    /// The bucketing is the SAME core `from_region_voxels` uses (same
    /// `i = round(world + floor(dim/2) − 0.5)` decode, same 256-tiling), so a streamed
    /// export is model-set-identical to the dense-path region export for any scene that
    /// fits the dense path — the E4 parity gate.
    pub fn from_region_voxel_chunks(
        region_dimensions: [u32; 3],
        chunk_voxels: impl IntoIterator<Item = Vec<voxel_core::voxel::Voxel>>,
        palette_colors: BlockPaletteColors,
    ) -> Self {
        Self::from_region_voxel_iter(
            region_dimensions,
            chunk_voxels.into_iter().flatten(),
            palette_colors,
        )
    }

    /// The shared bucketing core: stream EVERY occupied [`Voxel`](voxel_core::voxel::Voxel)
    /// into its 256-tile model at the corner-anchored decode index, dropping each as it
    /// is consumed (so an owned per-chunk stream never holds the whole region). Both
    /// [`from_region_voxels`](Self::from_region_voxels) (borrowed slices) and
    /// [`from_region_voxel_chunks`](Self::from_region_voxel_chunks) (owned chunk Vecs)
    /// flatten into this one path — which drives the same incremental
    /// [`VoxExportBuilder`] the live streaming export button feeds one chunk at a time,
    /// so the streamed and dense exports can never drift.
    fn from_region_voxel_iter(
        region_dimensions: [u32; 3],
        voxels: impl Iterator<Item = voxel_core::voxel::Voxel>,
        palette_colors: BlockPaletteColors,
    ) -> Self {
        let mut builder = VoxExportBuilder::new(region_dimensions, palette_colors);
        for voxel in voxels {
            builder.ingest_voxel(voxel);
        }
        builder.finish()
    }

    /// Number of models written (1 unless the 256-limit forced a tiled split).
    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Total occupied voxels written across all models.
    pub fn voxel_count(&self) -> usize {
        self.voxel_count
    }

    /// Serialise to the in-memory `.vox` byte stream.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        // Header: "VOX " + version 150.
        out.extend_from_slice(b"VOX ");
        write_u32(&mut out, 150);

        // Build the MAIN chunk's children first so we know its content size.
        let mut children = Vec::new();
        // PACK chunk (number of models) — required when >1 model.
        if self.models.len() > 1 {
            let mut pack = Vec::new();
            write_u32(&mut pack, self.models.len() as u32);
            write_chunk(&mut children, b"PACK", &pack, &[]);
        }
        for model in &self.models {
            // SIZE chunk.
            let mut size = Vec::new();
            write_u32(&mut size, model.size[0]);
            write_u32(&mut size, model.size[1]);
            write_u32(&mut size, model.size[2]);
            write_chunk(&mut children, b"SIZE", &size, &[]);

            // XYZI chunk: count then (x, y, z, colorIndex) per voxel. The file
            // colour index is 1-based.
            let mut xyzi = Vec::new();
            write_u32(&mut xyzi, model.voxels.len() as u32);
            for voxel in &model.voxels {
                xyzi.push(voxel.x);
                xyzi.push(voxel.y);
                xyzi.push(voxel.z);
                xyzi.push(voxel.color_index);
            }
            write_chunk(&mut children, b"XYZI", &xyzi, &[]);
        }

        // RGBA palette chunk: 256 entries. MagicaVoxel reads palette[i] for file index
        // i+1, so a voxel of `block_id` (which references file index `block_id + 1`)
        // reads array entry `block_id` (ADR 0003 §3a — the categorical block-palette
        // mapping). The procedural block colours fill the leading slots; the rest are a
        // neutral grey so the file stays valid.
        let mut rgba = Vec::with_capacity(256 * 4);
        for entry in 0..256 {
            if entry < self.palette_colors.len() {
                rgba.extend_from_slice(&self.palette_colors[entry]);
            } else {
                rgba.extend_from_slice(&[0x80, 0x80, 0x80, 0xff]);
            }
        }
        write_chunk(&mut children, b"RGBA", &rgba, &[]);

        // MAIN chunk: empty content, all of `children` as child content.
        write_chunk(&mut out, b"MAIN", &[], &children);
        out
    }

    /// Serialise and write the `.vox` to `path`, creating parent dirs.
    ///
    /// **Atomic write (data-loss guard).** The bytes go to a UNIQUE sibling temp file in
    /// the same directory (same filesystem ⇒ the rename is atomic), then the temp is moved
    /// onto the final path. The final path therefore only ever holds a COMPLETE export: a
    /// process killed mid-write — e.g. the window closed during a multi-second background
    /// export, which detaches and kills the export worker thread — leaves at WORST a stray
    /// temp, never a truncated `.vox` the Vintage Story mod would ingest as a corrupt model.
    ///
    /// - **Unique temp name** `.<final-name>.<pid>-<nanos>.tmp` (leading dot keeps it out of
    ///   the way; `pid` + wall-clock nanos avoid clobbering a real user file OR a concurrent
    ///   export's temp — no extra dependency).
    /// - **Move via `rename`, with a `copy` fallback.** `rename` is the atomic fast path.
    ///   On Windows it fails when the destination is open WITHOUT delete sharing; the old
    ///   in-place `std::fs::write` overwrote such a file fine (it needs only write sharing),
    ///   so we fall back to `fs::copy` (an in-place overwrite) to preserve that behaviour.
    /// - **On total failure the complete temp is KEPT**, and its path is named in the error
    ///   so the user can recover the export by hand rather than silently losing it.
    pub fn write(&self, path: &std::path::Path) -> std::io::Result<usize> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let bytes = self.to_bytes();
        let temp_path = Self::unique_temp_path(path);
        if let Err(error) = std::fs::write(&temp_path, &bytes) {
            // The temp is incomplete — remove it and surface the write error.
            let _ = std::fs::remove_file(&temp_path);
            return Err(error);
        }
        // Fast path: atomic rename onto the final name.
        if std::fs::rename(&temp_path, path).is_ok() {
            return Ok(bytes.len());
        }
        // Rename failed (typical Windows cause: the destination is open without delete
        // sharing). Fall back to an in-place copy, which needs only the write sharing the
        // old direct write needed — restoring the old overwrite semantics.
        match std::fs::copy(&temp_path, path) {
            Ok(_) => {
                let _ = std::fs::remove_file(&temp_path);
                Ok(bytes.len())
            }
            Err(error) => {
                // Both moves failed. KEEP the temp (it is the complete export) and name it
                // in the error so the user can recover it rather than lose the work.
                Err(std::io::Error::new(
                    error.kind(),
                    format!(
                        "could not place export at {} ({error}); complete export preserved at {}",
                        path.display(),
                        temp_path.display()
                    ),
                ))
            }
        }
    }

    /// A unique sibling temp path for `path` (`.<final-name>.<pid>-<nanos>.tmp`), in the
    /// SAME directory so the rename in [`write`](Self::write) stays on one filesystem. The
    /// pid + wall-clock nanos make the name collision-proof against both a real user file
    /// and a concurrent export, with no extra dependency.
    fn unique_temp_path(path: &std::path::Path) -> std::path::PathBuf {
        let final_name = path
            .file_name()
            .map(|name| name.to_os_string())
            .unwrap_or_default();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or(0);
        let mut temp_name = std::ffi::OsString::from(".");
        temp_name.push(&final_name);
        temp_name.push(format!(".{}-{nanos}.tmp", std::process::id()));
        match path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent.join(temp_name),
            _ => std::path::PathBuf::from(temp_name),
        }
    }
}

/// **Incremental streaming `.vox` builder (ADR 0010 E4).** Buckets a stream of occupied
/// voxels into the `.vox` 256-tile model set ONE chunk (or one voxel) at a time, so the
/// export never holds more than a single streamed chunk's voxels plus the per-model
/// output buffers — the inherent `.vox` output cost. This is the seam the live export
/// button drives over [`evaluation::two_layer_store::stream_vox_occupancy`]: each covering
/// chunk's freshly-expanded `Vec<Voxel>` is ingested then DROPPED, so peak transient
/// memory is O(one chunk + the output buffers), never O(all occupied voxels). It
/// dissolves the `Vec<Vec<Voxel>>` accumulate-then-convert intermediate the button used
/// to materialise before calling [`VoxExport::from_region_voxel_chunks`] — the owner's
/// peak-memory law: no O(volume) accumulation on any path.
///
/// ## Why the model set can be pre-created (single pass, no bounds pre-scan)
///
/// The model COUNT and SIZES are a pure function of `region_dimensions` (the 256-tiling),
/// and the palette is supplied explicitly (ADR 0003 §3a) — neither depends on scanning the
/// occupancy. So [`new`](Self::new) builds the full (empty) model set up front and every
/// voxel lands in its model by the SAME corner-anchored decode
/// [`VoxExport::from_region_voxels`] uses. One streaming pass suffices; the format never
/// forces a second pass over the voxels.
///
/// The dense [`VoxExport::from_region_voxels`] / [`VoxExport::from_region_voxel_chunks`]
/// paths flatten their voxels into this SAME builder, so a live streamed export is
/// byte-identical to the accumulate-then-convert export for the same voxel stream (the
/// order in which chunks are ingested is the order `stream_vox_occupancy` emits them, which
/// is exactly the order `from_region_voxel_chunks` flattens the accumulated `Vec`s).
pub struct VoxExportBuilder {
    /// The region's voxel dimensions (the tiling/decode frame) — the exact value
    /// [`document::scene::Scene::placed_region_dimensions`] and
    /// [`evaluation::two_layer_store::stream_vox_occupancy`] produce.
    region_dimensions: [u32; 3],
    /// Corner-anchoring half-extents (FLOORED `dim/2`) reused per voxel: the decode
    /// `round(world + floor(dim/2) − 0.5)` recovers the exact index for an odd dim too.
    half: [f32; 3],
    /// The pre-created 256-tile model set (tile_x, tile_y, tile_z order), each growing
    /// only its own per-model voxel buffer as voxels are ingested.
    models: Vec<VoxModel>,
    /// `(tile_x, tile_y, tile_z) -> models[index]` so a decoded voxel finds its model.
    model_index: std::collections::HashMap<(u32, u32, u32), usize>,
    palette_colors: BlockPaletteColors,
    voxel_count: usize,
}

impl VoxExportBuilder {
    /// Pre-create the full 256-tile model set for `region_dimensions` (empty voxel
    /// buffers) and the palette, ready to [`ingest_chunk`](Self::ingest_chunk) a stream.
    /// The model sizes/count are fixed here — they are a pure function of the region.
    pub fn new(region_dimensions: [u32; 3], palette_colors: BlockPaletteColors) -> Self {
        let [grid_x, grid_y, grid_z] = region_dimensions;
        // Corner-anchoring decode: FLOORED half (`dim/2` integer division), so
        // `round(world + floor(dim/2) − 0.5)` recovers the exact index for an odd dim
        // too (see voxel.rs::widest_run_in_band).
        let half = [(grid_x / 2) as f32, (grid_y / 2) as f32, (grid_z / 2) as f32];

        // Number of tiles along each grid axis so every tile is ≤ 256.
        let tiles_x = grid_x.div_ceil(VOX_AXIS_MAX).max(1);
        let tiles_y = grid_y.div_ceil(VOX_AXIS_MAX).max(1);
        let tiles_z = grid_z.div_ceil(VOX_AXIS_MAX).max(1);

        // Each tile's grid-space size (≤ 256); the last tile on an axis is the
        // remainder.
        let tile_size = |total: u32, index: u32| -> u32 {
            let origin = index * VOX_AXIS_MAX;
            (total - origin).min(VOX_AXIS_MAX)
        };

        // Build an indexable list of models (tile_x, tile_y, tile_z order).
        let mut models: Vec<VoxModel> = Vec::new();
        let mut model_index = std::collections::HashMap::new();
        for ty in 0..tiles_y {
            for tz in 0..tiles_z {
                for tx in 0..tiles_x {
                    let sx = tile_size(grid_x, tx);
                    let sy = tile_size(grid_y, ty);
                    let sz = tile_size(grid_z, tz);
                    model_index.insert((tx, ty, tz), models.len());
                    // Z-up: vox size = (our X, our Y, our Z) — no swap.
                    models.push(VoxModel {
                        size: [sx, sy, sz],
                        voxels: Vec::new(),
                    });
                }
            }
        }

        Self {
            region_dimensions,
            half,
            models,
            model_index,
            palette_colors,
            voxel_count: 0,
        }
    }

    /// Bucket every voxel in one STREAMED chunk into its model, so the caller can DROP
    /// the chunk buffer afterward (ADR 0010 E4): only one chunk's voxels are ever
    /// resident. This is the sink [`evaluation::two_layer_store::stream_vox_occupancy`] drives.
    pub fn ingest_chunk(&mut self, chunk_voxels: &[voxel_core::voxel::Voxel]) {
        for voxel in chunk_voxels {
            self.ingest_voxel(*voxel);
        }
    }

    /// Decode one voxel's corner-anchored grid index, tile it, and push it into its
    /// model (dropping it if it falls outside the region — the same guard the dense
    /// path used). The block id selects the `.vox` palette slot (ADR 0003 §3a).
    fn ingest_voxel(&mut self, voxel: voxel_core::voxel::Voxel) {
        let [grid_x, grid_y, grid_z] = self.region_dimensions;
        // Recover non-negative integer grid indices from the world-centred
        // voxel-centre position: `i = round(world_x + dim_x/2 - 0.5)`.
        let position = voxel.world_position();
        let i = (position[0] + self.half[0] - 0.5).round();
        let j = (position[1] + self.half[1] - 0.5).round();
        let k = (position[2] + self.half[2] - 0.5).round();
        if i < 0.0 || j < 0.0 || k < 0.0 {
            return;
        }
        let i = i as u32;
        let j = j as u32;
        let k = k as u32;
        if i >= grid_x || j >= grid_y || k >= grid_z {
            return;
        }

        let tx = i / VOX_AXIS_MAX;
        let ty = j / VOX_AXIS_MAX;
        let tz = k / VOX_AXIS_MAX;
        let local_i = i % VOX_AXIS_MAX;
        let local_j = j % VOX_AXIS_MAX;
        let local_k = k % VOX_AXIS_MAX;

        let Some(&model_pos) = self.model_index.get(&(tx, ty, tz)) else {
            return;
        };
        // Z-up: vox (x, y, z) = (our i, our j, our k) — no swap. The categorical
        // block id selects the `.vox` palette slot (`block_id + 1`; ADR 0003 §3a),
        // so a multi-material model exports each block in its own colour. Clamp to
        // the procedural palette so a stray id stays in range.
        let palette_slot = voxel
            .color_index()
            .min(voxel_core::core_geom::MaterialChoice::MATERIAL_COUNT as u16 - 1)
            as u8
            + 1;
        self.models[model_pos].voxels.push(VoxVoxel {
            x: local_i as u8,
            y: local_j as u8,
            z: local_k as u8,
            color_index: palette_slot,
        });
        self.voxel_count += 1;
    }

    /// Finalise the streamed export: drop the empty tiles a sparse split leaves behind,
    /// keeping at least one (possibly empty) model so the file stays valid.
    pub fn finish(mut self) -> VoxExport {
        // Drop empty models (a sparse split can leave some tiles with nothing).
        self.models.retain(|model| !model.voxels.is_empty());
        // Always emit at least one (possibly empty) model so the file is valid.
        if self.models.is_empty() {
            let [grid_x, grid_y, grid_z] = self.region_dimensions;
            self.models.push(VoxModel {
                // Z-up: no swap — vox size = (our X, our Y, our Z).
                size: [
                    grid_x.clamp(1, VOX_AXIS_MAX),
                    grid_y.clamp(1, VOX_AXIS_MAX),
                    grid_z.clamp(1, VOX_AXIS_MAX),
                ],
                voxels: Vec::new(),
            });
        }

        VoxExport {
            models: self.models,
            palette_colors: self.palette_colors,
            voxel_count: self.voxel_count,
        }
    }
}

/// Append a little-endian `u32`.
fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Append one chunk: 4-byte id, content size (LE u32), children size (LE u32),
/// then the content bytes, then the children bytes.
fn write_chunk(out: &mut Vec<u8>, id: &[u8; 4], content: &[u8], children: &[u8]) {
    out.extend_from_slice(id);
    write_u32(out, content.len() as u32);
    write_u32(out, children.len() as u32);
    out.extend_from_slice(content);
    out.extend_from_slice(children);
}

#[cfg(test)]
mod tests;
