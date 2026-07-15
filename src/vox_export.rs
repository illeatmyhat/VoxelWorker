//! MagicaVoxel `.vox` export (Milestone 8).
//!
//! Serialises a resolved [`VoxelGrid`] to a MagicaVoxel
//! `.vox` file so the result can be ingested by the **Automatic Chiselling
//! REBORN** Vintage Story mod (DATA.md §".vox export"). The chunked binary is
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
    /// ([`crate::two_layer_store::stream_vox_occupancy`]): a coarse-solid block is a
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
/// button drives over [`crate::two_layer_store::stream_vox_occupancy`]: each covering
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
    /// [`crate::scene::Scene::placed_region_dimensions`] and
    /// [`crate::two_layer_store::stream_vox_occupancy`] produce.
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
    /// resident. This is the sink [`crate::two_layer_store::stream_vox_occupancy`] drives.
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
mod tests {
    use super::*;
    use voxel_core::core_geom::MaterialChoice;
    use voxel_core::voxel::{ShapeKind};
    use crate::voxel::{SdfShape};

    /// Resolve a small cylinder and round-trip it through `.vox`, asserting the
    /// voxel count and dimensions survive (Z-up, no axis swap).
    ///
    /// Corner-anchoring: `from_grid`'s decode (`round(world + floor(dim/2) − 0.5)`)
    /// expects the grid in the RECENTRED frame (low corner `−floor(dim/2)`), which is
    /// what production produces. So resolve through a one-node scene (recentred), NOT
    /// the bare producer grid (whose low corner is 0).
    #[test]
    fn vox_round_trip_matches_grid() {
        let scene = crate::scene::Scene::from_geometry(
            GeometryParams {
                shape: ShapeKind::Cylinder,
                size_voxels: [80, 16, 80],
                size_measurements: None,
                voxels_per_block: 16,
                wall_blocks: 1,
            },
            voxel_core::core_geom::MaterialChoice::Stone,
        );
        let grid = scene.resolve_region(scene.full_extent_blocks(16), 16, 0);
        assert!(grid.occupied_count() > 0, "expected a non-empty grid");

        let export = VoxExport::from_grid(&grid, VoxExport::block_palette_from_active(MaterialChoice::Stone, [132, 126, 118, 255]));
        assert_eq!(export.model_count(), 1, "small grid is a single model");
        assert_eq!(export.voxel_count(), grid.occupied_count());

        let bytes = export.to_bytes();
        let parsed = dot_vox::load_bytes(&bytes).expect("dot_vox should parse our file");

        assert_eq!(parsed.models.len(), 1);
        let model = &parsed.models[0];
        // Z-up: vox size = (our X, our Y, our Z) — no swap. Grid 80×16×80 → vox 80×16×80.
        let [gx, gy, gz] = grid.dimensions;
        assert_eq!(model.size.x, gx);
        assert_eq!(model.size.y, gy);
        assert_eq!(model.size.z, gz);
        // Every occupied voxel was written exactly once.
        assert_eq!(model.voxels.len(), grid.occupied_count());
        // All coordinates are within the model's declared size.
        for voxel in &model.voxels {
            assert!((voxel.x as u32) < model.size.x);
            assert!((voxel.y as u32) < model.size.y);
            assert!((voxel.z as u32) < model.size.z);
        }
    }

    /// The atomic write's post-conditions (findings 2/3/4): a successful `write` leaves
    /// the final file present and NO stray temp behind, and the unique temp name it picks
    /// is a dot-prefixed sibling in the SAME directory (so the rename stays on one
    /// filesystem). The Windows rename→copy fallback on a share-violating destination is
    /// not portably simulatable, so it is reasoned in `write`'s doc comment rather than
    /// exercised here.
    #[test]
    fn atomic_write_leaves_no_temp_and_temp_is_a_dir_sibling() {
        let scene = crate::scene::Scene::from_geometry(
            GeometryParams {
                shape: ShapeKind::Cylinder,
                size_voxels: [16, 16, 16],
                size_measurements: None,
                voxels_per_block: 8,
                wall_blocks: 1,
            },
            voxel_core::core_geom::MaterialChoice::Stone,
        );
        let grid = scene.resolve_region(scene.full_extent_blocks(8), 8, 0);
        let export = VoxExport::from_grid(
            &grid,
            VoxExport::block_palette_from_active(MaterialChoice::Stone, [132, 126, 118, 255]),
        );

        let dir = std::env::temp_dir()
            .join(format!("voxel_worker_write_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create the test dir");
        let final_path = dir.join("model.vox");

        // The unique temp name is a dot-prefixed sibling in the same directory.
        let temp = VoxExport::unique_temp_path(&final_path);
        assert_eq!(temp.parent(), Some(dir.as_path()), "temp is a sibling of the final file");
        assert!(
            temp.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with('.') && n.ends_with(".tmp")),
            "temp name is dot-prefixed and .tmp-suffixed: {temp:?}"
        );

        // A successful write leaves the final file and NO temp behind.
        let bytes = export.write(&final_path).expect("write succeeds");
        assert!(bytes > 0, "wrote a non-empty file");
        assert!(final_path.exists(), "the final file is present");
        let leftover_temps: Vec<_> = std::fs::read_dir(&dir)
            .expect("read the test dir")
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.ends_with(".tmp"))
            })
            .collect();
        assert!(
            leftover_temps.is_empty(),
            "a successful write leaves no temp file behind, found: {leftover_temps:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Z-up convention pin: an ASYMMETRIC shape (tall in Z) puts its vertical extent
    /// on vox-Z with NO swap. A cylinder 2×2×5 blocks (5 blocks tall along the +Z
    /// axis) must export to a vox model whose Z size is the largest — proving the
    /// vertical axis lands on vox-Z directly, not relocated to vox-Y.
    #[test]
    fn vox_export_puts_vertical_on_vox_z_no_swap() {
        let scene = crate::scene::Scene::from_geometry(
            GeometryParams {
                shape: ShapeKind::Cylinder,
                size_voxels: [2 * 16, 2 * 16, 5 * 16],
                size_measurements: None,
                voxels_per_block: 16,
                wall_blocks: 1,
            },
            voxel_core::core_geom::MaterialChoice::Stone,
        );
        let grid = scene.resolve_region(scene.full_extent_blocks(16), 16, 0);
        let [gx, gy, gz] = grid.dimensions; // 32 × 32 × 80
        let export = VoxExport::from_grid(&grid, VoxExport::block_palette_from_active(MaterialChoice::Stone, [132, 126, 118, 255]));
        let bytes = export.to_bytes();
        let parsed = dot_vox::load_bytes(&bytes).expect("dot_vox should parse our file");
        let model = &parsed.models[0];
        // No swap: vox size matches our (X, Y, Z) exactly, with Z the tallest axis.
        assert_eq!([model.size.x, model.size.y, model.size.z], [gx, gy, gz]);
        assert!(
            model.size.z > model.size.x && model.size.z > model.size.y,
            "the tall (Z) axis must land on vox-Z (got {:?})",
            (model.size.x, model.size.y, model.size.z)
        );
    }

    /// A grid wider than 256 voxels on an axis must split into multiple models,
    /// never silently truncate.
    #[test]
    fn vox_splits_models_over_256() {
        // 17 blocks × 16 vx = 272 > 256 on X. Resolve through a scene (recentred
        // frame) so `from_grid`'s corner-anchored decode lands every voxel in range.
        let scene = crate::scene::Scene::from_geometry(
            GeometryParams {
                shape: ShapeKind::Box,
                size_voxels: [272, 16, 16],
                size_measurements: None,
                voxels_per_block: 16,
                wall_blocks: 1,
            },
            voxel_core::core_geom::MaterialChoice::Stone,
        );
        let grid = scene.resolve_region(scene.full_extent_blocks(16), 16, 0);

        let export = VoxExport::from_grid(&grid, VoxExport::block_palette_from_active(MaterialChoice::Stone, [200, 200, 200, 255]));
        assert!(export.model_count() >= 2, "272-wide grid should split");
        // No voxels lost across the split.
        assert_eq!(export.voxel_count(), grid.occupied_count());

        let bytes = export.to_bytes();
        let parsed = dot_vox::load_bytes(&bytes).expect("dot_vox should parse split file");
        let total: usize = parsed.models.iter().map(|m| m.voxels.len()).sum();
        assert_eq!(total, grid.occupied_count());
        for model in &parsed.models {
            assert!(model.size.x <= 256 && model.size.y <= 256 && model.size.z <= 256);
        }
    }

    // ===== Issue #20 S6d: region-scoped `.vox` export ============================

    use crate::chunk_cache::ChunkResolveCache;
    use crate::scene::{Node, NodeContent, Scene};
    use crate::voxel::GeometryParams;

    /// Parse a `.vox` byte stream into a per-model SORTED multiset of
    /// `(size, voxel (x, y, z, color))`, so two exports compare equal regardless of
    /// per-model voxel emission ORDER (chunk-iteration order vs monolithic stamp
    /// order) — a MagicaVoxel reader treats reordered voxels as the same model.
    type ModelVoxelSet = std::collections::BTreeSet<(u8, u8, u8, u8)>;
    type ModelSets = std::collections::BTreeSet<([u32; 3], ModelVoxelSet)>;
    fn parsed_model_sets(bytes: &[u8]) -> ModelSets {
        let parsed = dot_vox::load_bytes(bytes).expect("dot_vox should parse our file");
        parsed
            .models
            .iter()
            .map(|model| {
                let size = [model.size.x, model.size.y, model.size.z];
                let voxels = model
                    .voxels
                    .iter()
                    .map(|v| (v.x, v.y, v.z, v.i))
                    .collect::<std::collections::BTreeSet<_>>();
                (size, voxels)
            })
            .collect()
    }

    /// Parse a `.vox` byte stream into a per-model **last-writer-wins** map
    /// `(x, y, z) -> colour` — the occupancy a MagicaVoxel reader actually renders. The
    /// dense-path export writes DUPLICATE voxels at positions where leaves overlap (the
    /// dense occupied Vec keeps both leaves' entries; the LATER one in document order is
    /// the resolved winner a reader shows); the streamed two-layer export is one-id-per-
    /// cell (Union later-wins resolved). Reducing both to last-writer-per-coord compares
    /// the TRUE resolved file: for every non-overlapping scene each coord has one writer,
    /// so this is identical to [`parsed_model_sets`]; only genuine overlap differs, and
    /// there the last-writer map is the correct comparison (ADR 0010 parity-gate canonical
    /// form, mirroring `two_layer_store.rs::resolved_occupancy_set`).
    type ModelLastWriter = std::collections::BTreeMap<(u8, u8, u8), u8>;
    type ModelLastWriterSets = std::collections::BTreeSet<([u32; 3], Vec<((u8, u8, u8), u8)>)>;
    fn parsed_model_last_writer_sets(bytes: &[u8]) -> ModelLastWriterSets {
        let parsed = dot_vox::load_bytes(bytes).expect("dot_vox should parse our file");
        parsed
            .models
            .iter()
            .map(|model| {
                let size = [model.size.x, model.size.y, model.size.z];
                // Voxels are in write order; the LAST entry at a coord wins (insert
                // overwrites), reproducing the MagicaVoxel reader's resolved occupancy.
                let mut last: ModelLastWriter = std::collections::BTreeMap::new();
                for v in &model.voxels {
                    last.insert((v.x, v.y, v.z), v.i);
                }
                (size, last.into_iter().collect::<Vec<_>>())
            })
            .collect()
    }

    fn assert_region_vox_export_equals_whole_grid(scene: &Scene, vpb: u32, label: &str) {
        let rgba = VoxExport::block_palette_from_active(MaterialChoice::Stone, [132, 126, 118, 255]);

        // Whole-grid export: assemble the monolithic region grid, export via the
        // existing `from_grid` path.
        let region = scene.full_extent_blocks(vpb);
        let whole = scene.resolve_region(region, vpb, 0);
        let whole_export = VoxExport::from_grid(&whole, rgba);

        // Region export: from the per-chunk grids, no monolithic grid assembled.
        let mut cache = ChunkResolveCache::new();
        let (dims, occupied) = cache.bound_region_occupied(scene, vpb, 0);
        let region_export = VoxExport::from_region_voxels(dims, occupied, rgba);

        assert_eq!(
            region_export.voxel_count(),
            whole_export.voxel_count(),
            "[{label}] region export voxel count must equal whole-grid"
        );
        assert_eq!(
            region_export.model_count(),
            whole_export.model_count(),
            "[{label}] region export model count must equal whole-grid"
        );
        assert_eq!(
            parsed_model_sets(&region_export.to_bytes()),
            parsed_model_sets(&whole_export.to_bytes()),
            "[{label}] region export model-set (sizes + voxels) must equal whole-grid"
        );
    }

    fn shape_scene(kind: ShapeKind, vpb: u32, size: [u32; 3]) -> Scene {
        Scene::from_geometry(
            GeometryParams {
                shape: kind,
                size_voxels: [size[0] * vpb, size[1] * vpb, size[2] * vpb],
                size_measurements: None,
                voxels_per_block: vpb,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        )
    }

    /// The region-scoped `.vox` export equals the whole-grid export for the bounded
    /// SDF shapes (single-model cases).
    #[test]
    fn region_vox_export_equals_whole_grid_for_shapes() {
        for kind in [
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Tube,
            ShapeKind::Torus,
            ShapeKind::Box,
        ] {
            let scene = shape_scene(kind, 16, [5, 5, 5]);
            assert_region_vox_export_equals_whole_grid(&scene, 16, &format!("{kind:?}"));
        }
    }

    /// The region-scoped `.vox` export equals the whole-grid export even when the
    /// region is wide enough to FORCE a 256-split into multiple models — proving the
    /// per-chunk bucketing tiles identically to the monolithic path.
    #[test]
    fn region_vox_export_equals_whole_grid_when_split_over_256() {
        // 20 blocks × 16 = 320 voxels > 256 on X → splits into 2 models.
        let scene = shape_scene(ShapeKind::Box, 16, [20, 1, 1]);
        assert_region_vox_export_equals_whole_grid(&scene, 16, "wide-box-split");
    }

    /// A multi-leaf demo scene (spans several chunks across leaves) exports
    /// identically through the region path.
    #[test]
    fn region_vox_export_equals_whole_grid_for_demo_scene() {
        let vpb = 16u32;
        let make_tool = |kind, offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, vpb);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = crate::scene::NodeTransform::from_blocks(offset, vpb);
            node
        };
        let scene = Scene::from_nodes(vec![
            make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
            make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
            make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
        ]);
        assert_region_vox_export_equals_whole_grid(&scene, vpb, "demo-scene");
    }

    // ===== Issue #20 Step 2: far-offset export ===================================

    /// Build a two-node scene whose composite is centred FAR from the world origin:
    /// one node at the origin and one node `offset_blocks` away on X. The composite
    /// centre lands at the midpoint, so each node sits ~`offset/2 × vpb` voxels from
    /// the recentred frame's origin. The second node is placed `offset_blocks` blocks
    /// away on X.
    fn far_offset_two_box_scene(vpb: u32, offset_blocks: i64) -> Scene {
        let make_box = |offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(ShapeKind::Box, [3, 3, 3], 1, vpb);
            let mut node = Node::new("Box", NodeContent::Tool { shape, material });
            node.transform = crate::scene::NodeTransform::from_blocks(offset, vpb);
            node
        };
        let mut scene = Scene::from_nodes(vec![
            make_box([0, 0, 0], MaterialChoice::Stone),
            make_box([offset_blocks, 0, 0], MaterialChoice::Wood),
        ]);
        scene.voxels_per_block = vpb;
        scene
    }

    /// **The rewired export is behaviour-equivalent to the old monolithic export, far
    /// from the origin (issue #20 Step 2).** The live export button now routes through
    /// `ChunkResolveCache::vox_export` instead of a dense whole-region resolve + `from_grid`. This
    /// proves the rewiring is safe at far offset: for a scene whose composite is
    /// centred ~250,000 blocks out (4e6 voxels — well into the f32 large-magnitude
    /// regime), the region-scoped export's model SET (sizes + per-model voxels) equals
    /// the old whole-grid export's, AND both keep the full voxel count (the per-chunk
    /// ground truth). So the wiring change is a true no-op on the written file.
    ///
    /// NOTE (finding, issue #20 Low #1): routing through `vox_export` does NOT make a
    /// genuinely region-WIDE far scene more accurate than the monolithic path. Both
    /// bucket into the region-relative `[0, grid_x)` frame, so both add `half_x` (≈ the
    /// region half-width) in f32; once the region exceeds ~2^24 voxels on an axis the
    /// voxel-centre `.5` is unrepresentable and BOTH paths collapse identically (the
    /// exports stay model-set-equal). The f32-`.5` loss is inherent to the f32
    /// `world_position` at large magnitude, not to which assembly path is used. The
    /// rewiring's value is the Step-4 decoupling from the monolithic grid, not a
    /// far-offset accuracy gain.
    #[test]
    fn far_offset_region_export_equals_monolithic() {
        let vpb = 16u32;
        // 500,000-block separation → composite centred ~250,000 blocks out → each box
        // ~4e6 voxels from origin. Region grid stays under 2^24 voxels wide so the full
        // voxel set survives (the per-chunk ground truth is matched exactly).
        let scene = far_offset_two_box_scene(vpb, 500_000);
        let rgba = VoxExport::block_palette_from_active(MaterialChoice::Stone, [132, 126, 118, 255]);

        // Ground-truth voxel count (frame-independent): the per-chunk assembly rebases
        // each chunk in i64, so its occupied count is the TRUE distinct-voxel count.
        let expected_voxels = scene.resolve_region_via_chunks(vpb, 0).occupied_count();

        // New (region-scoped) path — what the export button now calls.
        let mut cache = ChunkResolveCache::new();
        let (dims, occupied) = cache.bound_region_occupied(&scene, vpb, 0);
        let region_export = VoxExport::from_region_voxels(dims, occupied, rgba);
        assert_eq!(
            region_export.voxel_count(),
            expected_voxels,
            "region export must keep every voxel at this far offset"
        );

        // Old (monolithic) path the button used before.
        let region = scene.full_extent_blocks(vpb);
        let whole = scene.resolve_region(region, vpb, 0);
        let monolithic_export = VoxExport::from_grid(&whole, rgba);

        // The rewiring is a no-op on the written file: same model set (sizes + voxels),
        // same counts. (Per-model voxel ORDER may differ — chunk-iteration vs
        // monolithic stamp order — which a MagicaVoxel reader treats as the same model.)
        assert_eq!(
            region_export.voxel_count(),
            monolithic_export.voxel_count(),
            "rewired export voxel count must equal the old monolithic export far out"
        );
        assert_eq!(
            parsed_model_sets(&region_export.to_bytes()),
            parsed_model_sets(&monolithic_export.to_bytes()),
            "rewired export model-set must equal the old monolithic export far out"
        );
    }

    /// The far-offset region export, once parsed and re-read, round-trips to the same
    /// total voxel count the per-chunk ground truth holds — exercising the full
    /// build → serialise → `dot_vox::load_bytes` path the export button drives (minus
    /// the file dialog), so the wiring is verified end to end headlessly.
    #[test]
    fn far_offset_region_export_round_trips_full_voxel_set() {
        let vpb = 16u32;
        let scene = far_offset_two_box_scene(vpb, 500_000);
        let rgba = VoxExport::block_palette_from_active(MaterialChoice::Stone, [132, 126, 118, 255]);

        let mut cache = ChunkResolveCache::new();
        let (dims, occupied) = cache.bound_region_occupied(&scene, vpb, 0);
        let region_export = VoxExport::from_region_voxels(dims, occupied, rgba);

        let parsed = dot_vox::load_bytes(&region_export.to_bytes())
            .expect("dot_vox should parse the far-offset export");

        let expected_voxels = scene.resolve_region_via_chunks(vpb, 0).occupied_count();
        let total: usize = parsed.models.iter().map(|model| model.voxels.len()).sum();
        assert_eq!(
            total, expected_voxels,
            "far-offset export must round-trip every voxel"
        );

        // The two far-separated boxes occupy different 256-tiles on X, so the parsed
        // file must contain at least two non-empty models.
        let nonempty_models = parsed
            .models
            .iter()
            .filter(|model| !model.voxels.is_empty())
            .count();
        assert!(
            nonempty_models >= 2,
            "the two far-separated boxes must land in >=2 distinct tiles (got {nonempty_models})"
        );
    }

    // ===== ADR 0010 E4: cacheless STREAMING `.vox` export ========================

    use voxel_core::core_geom::MaterialChoice as Mat;
    use crate::two_layer_store::{stream_vox_occupancy, TwoLayerStore};

    /// Build the `.vox` export by STREAMING the cacheless two-layer evaluator (coarse
    /// `d³` fast-fill + boundary per-voxel) — the E4 path the export button drives.
    fn streamed_vox_export(scene: &Scene, vpb: u32, rgba: BlockPaletteColors) -> VoxExport {
        let store = TwoLayerStore::enabled();
        let mut chunks: Vec<Vec<voxel_core::voxel::Voxel>> = Vec::new();
        let dims = stream_vox_occupancy(&store, scene, vpb, |chunk| chunks.push(chunk))
            .expect("the two-layer capability is enabled");
        VoxExport::from_region_voxel_chunks(dims, chunks, rgba)
    }

    /// **THE E4 `.vox` PARITY GATE:** the streamed export's written `.vox` (model set =
    /// sizes + per-voxel `(x, y, z, colour)`) is IDENTICAL to today's dense-path region
    /// export, for the gated scene. Mirrors
    /// `assert_region_vox_export_equals_whole_grid` on the streaming path.
    fn assert_streamed_vox_export_equals_dense(scene: &Scene, vpb: u32, label: &str) {
        let rgba = VoxExport::block_palette_from_active(Mat::Stone, [132, 126, 118, 255]);

        // Dense path (today's export): per-chunk `bound_region_occupied` → `from_region_voxels`.
        let mut cache = ChunkResolveCache::new();
        let (dims, occupied) = cache.bound_region_occupied(scene, vpb, 0);
        let dense_export = VoxExport::from_region_voxels(dims, occupied, rgba);

        // Streamed path (E4): the cacheless two-layer evaluator.
        let streamed_export = streamed_vox_export(scene, vpb, rgba);

        assert_eq!(
            streamed_export.model_count(),
            dense_export.model_count(),
            "[{label}] streamed export model count must equal the dense-path export"
        );
        // The faithful parity comparison is the RESOLVED occupancy a MagicaVoxel reader
        // renders — last-writer-per-coord (position + palette colour). For every
        // non-overlapping scene each coord has one writer, so this is bit-identical to
        // the raw per-voxel set; only genuine leaf overlap differs (the dense file keeps
        // duplicate entries there, the streamed file is resolved), and the last-writer
        // map is the correct comparison (ADR 0010 parity-gate canonical form).
        let streamed_bytes = streamed_export.to_bytes();
        let dense_bytes = dense_export.to_bytes();
        assert_eq!(
            parsed_model_last_writer_sets(&streamed_bytes),
            parsed_model_last_writer_sets(&dense_bytes),
            "[{label}] streamed export resolved occupancy (last-writer position + palette \
             colour) must be IDENTICAL to the dense-path `.vox` export"
        );
        // The streamed export is one-id-per-cell: it writes NO duplicate voxels, so its
        // raw voxel count equals its resolved count (the dense path over-counts at
        // overlaps; the streamed path never does — that is the elision win).
        let streamed_resolved_count: usize = parsed_model_last_writer_sets(&streamed_bytes)
            .iter()
            .map(|(_, last)| last.len())
            .sum();
        assert_eq!(
            streamed_export.voxel_count(),
            streamed_resolved_count,
            "[{label}] the streamed export must write one voxel per resolved cell (no \
             duplicate-at-overlap entries)"
        );
    }

    /// **THE STREAMED-SINK PEAK-MEMORY PROOF (ADR 0010 E4).** The live export button now
    /// buckets each streamed chunk DIRECTLY into a [`VoxExportBuilder`] then drops it
    /// (peak = O(one chunk + output buffers)), instead of accumulating every chunk into a
    /// `Vec<Vec<Voxel>>` before one `from_region_voxel_chunks` conversion (peak =
    /// O(all voxels)). This asserts the two produce a BYTE-IDENTICAL `.vox` for a
    /// multi-chunk scene: both drive the SAME `stream_vox_occupancy` (identical chunk
    /// order), and the incremental builder IS the core `from_region_voxel_chunks` flattens
    /// into — so the memory fix is a pure no-op on the written file, down to voxel emission
    /// order and palette bytes. The accumulate-then-convert path is kept here as the oracle.
    fn assert_streamed_builder_matches_accumulated(scene: &Scene, vpb: u32, label: &str) {
        let rgba = VoxExport::block_palette_from_active(Mat::Stone, [132, 126, 118, 255]);
        let store = TwoLayerStore::enabled();

        // Incremental streaming sink (the live button): bucket each chunk, then drop it —
        // only one chunk's voxels are ever resident.
        let region_dimensions = scene.placed_region_dimensions(vpb);
        let mut builder = VoxExportBuilder::new(region_dimensions, rgba);
        let dims_stream =
            stream_vox_occupancy(&store, scene, vpb, |chunk| builder.ingest_chunk(&chunk))
                .expect("the two-layer capability is enabled");
        assert_eq!(
            dims_stream, region_dimensions,
            "[{label}] the builder must be pre-created with the SAME dims the stream emits"
        );
        let streamed = builder.finish();

        // Accumulate-then-convert ORACLE (the retired path): push every chunk into a
        // Vec<Vec<Voxel>> before converting — O(all voxels) peak.
        let mut accumulated_chunks: Vec<Vec<voxel_core::voxel::Voxel>> = Vec::new();
        stream_vox_occupancy(&store, scene, vpb, |chunk| accumulated_chunks.push(chunk))
            .expect("the two-layer capability is enabled");
        let accumulated =
            VoxExport::from_region_voxel_chunks(region_dimensions, accumulated_chunks, rgba);

        assert_eq!(
            streamed.voxel_count(),
            accumulated.voxel_count(),
            "[{label}] streamed-sink voxel count must equal the accumulate-then-convert path"
        );
        assert_eq!(
            streamed.model_count(),
            accumulated.model_count(),
            "[{label}] streamed-sink model count must equal the accumulate-then-convert path"
        );
        assert_eq!(
            streamed.to_bytes(),
            accumulated.to_bytes(),
            "[{label}] the streamed-sink `.vox` bytes must be IDENTICAL to the \
             accumulate-then-convert export (the peak-memory fix is a no-op on the file)"
        );
    }

    #[test]
    fn streamed_builder_matches_accumulated_for_multi_chunk_scenes() {
        let vpb = 16u32;
        let make_tool = |kind, offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, vpb);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = crate::scene::NodeTransform::from_blocks(offset, vpb);
            node
        };
        // A multi-leaf scene spanning several chunks across leaves.
        let demo = Scene::from_nodes(vec![
            make_tool(ShapeKind::Sphere, [0, 0, 0], Mat::Stone),
            make_tool(ShapeKind::Box, [8, 0, 0], Mat::Wood),
            make_tool(ShapeKind::Torus, [0, 0, 6], Mat::Plain),
        ]);
        assert_streamed_builder_matches_accumulated(&demo, vpb, "demo-scene");

        // A wide box forcing a 256-split (multiple models across many chunks).
        let wide = shape_scene(ShapeKind::Box, vpb, [20, 1, 1]);
        assert_streamed_builder_matches_accumulated(&wide, vpb, "wide-box-split");
    }

    #[test]
    fn streamed_vox_export_equals_dense_for_shapes() {
        for kind in [
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Tube,
            ShapeKind::Torus,
            ShapeKind::Box,
        ] {
            let scene = shape_scene(kind, 16, [5, 5, 5]);
            assert_streamed_vox_export_equals_dense(&scene, 16, &format!("{kind:?}"));
        }
    }

    /// FLAT / odd-sized shapes (a 1-block axis straddling two chunks) stream identically.
    #[test]
    fn streamed_vox_export_equals_dense_for_flat_and_odd_shapes() {
        for kind in [ShapeKind::Cylinder, ShapeKind::Sphere, ShapeKind::Torus] {
            for size in [[5u32, 1, 5], [3, 1, 3], [5, 3, 5], [1, 1, 1]] {
                let scene = shape_scene(kind, 16, size);
                assert_streamed_vox_export_equals_dense(
                    &scene,
                    16,
                    &format!("{kind:?} {size:?}"),
                );
            }
        }
    }

    /// A wide box forcing a 256-split streams the same multi-model set as the dense path.
    #[test]
    fn streamed_vox_export_equals_dense_when_split_over_256() {
        let scene = shape_scene(ShapeKind::Box, 16, [20, 1, 1]);
        assert_streamed_vox_export_equals_dense(&scene, 16, "wide-box-split");
    }

    #[test]
    fn streamed_vox_export_equals_dense_for_demo_scene() {
        let vpb = 16u32;
        let make_tool = |kind, offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, vpb);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = crate::scene::NodeTransform::from_blocks(offset, vpb);
            node
        };
        let scene = Scene::from_nodes(vec![
            make_tool(ShapeKind::Sphere, [0, 0, 0], Mat::Stone),
            make_tool(ShapeKind::Box, [8, 0, 0], Mat::Wood),
            make_tool(ShapeKind::Torus, [0, 0, 6], Mat::Plain),
        ]);
        assert_streamed_vox_export_equals_dense(&scene, vpb, "demo-scene");
    }

    #[test]
    fn streamed_vox_export_equals_dense_for_demo_village() {
        use crate::scene::DefId;
        let vpb = 16u32;
        let house_def_id = DefId(1);
        let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, size, 1, vpb);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = crate::scene::NodeTransform::from_blocks(offset, vpb);
            node
        };
        let instance = |name: &str, offset: [i64; 3]| {
            let mut node = Node::new(name, NodeContent::Instance(house_def_id));
            node.transform = crate::scene::NodeTransform::from_blocks(offset, vpb);
            node
        };
        let mut scene = Scene::from_nodes(vec![
            instance("House 1", [0, 0, 0]),
            instance("House 2", [6, 0, 0]),
            instance("House 3", [12, 0, 0]),
            instance("House 4", [18, 0, 0]),
        ]);
        scene.add_definition(
            house_def_id,
            "House".to_string(),
            vec![
                tool(ShapeKind::Box, [2, 2, 2], [0, 0, 0], Mat::Stone),
                tool(ShapeKind::Cylinder, [1, 2, 1], [0, 2, 0], Mat::Wood),
            ],
        );
        assert_streamed_vox_export_equals_dense(&scene, vpb, "demo-village");
    }

    /// A sketch-revolve solid (always classifies BOUNDARY — its polygon fill is not a
    /// coarse box) streams its per-voxel boundary path identically to the dense export.
    #[test]
    fn streamed_vox_export_equals_dense_for_sketch_solid() {
        use crate::sketch::{PlaneAxis, RevolveAxis, Sketch, SketchSolid};
        let vpb = 16u32;
        let profile = Sketch::rectangle(PlaneAxis::Z, 24, 16);
        let producer = SketchSolid::revolve(profile, RevolveAxis::InPlane0, 360);
        let node = Node::new(
            "Revolve",
            NodeContent::SketchTool {
                producer,
                material: Mat::Stone,
            },
        );
        let scene = Scene::from_nodes(vec![node]);
        assert_streamed_vox_export_equals_dense(&scene, vpb, "sketch-revolve");
    }

    /// An OVERLAP multi-material scene (two boxes of different materials overlapping):
    /// the overlap blocks classify BOUNDARY (Union later-wins material is per-voxel), so
    /// each voxel's `.vox` palette colour must match the dense export through the palette.
    #[test]
    fn streamed_vox_export_equals_dense_for_overlap_multi_material() {
        let vpb = 16u32;
        let make_tool = |offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(ShapeKind::Box, [4, 4, 4], 1, vpb);
            let mut node = Node::new("Box", NodeContent::Tool { shape, material });
            node.transform = crate::scene::NodeTransform::from_blocks(offset, vpb);
            node
        };
        let scene = Scene::from_nodes(vec![
            make_tool([0, 0, 0], Mat::Stone),
            make_tool([2, 0, 0], Mat::Wood),
        ]);
        assert_streamed_vox_export_equals_dense(&scene, vpb, "overlap-multi-material");
    }

    /// **6M-CAP DISSOLUTION (the E4 headline):** an 800×800-revolve-class solid box —
    /// 50³ blocks @ d16 = 800³ voxels, whose dense whole-region count (~5.1e8) blows the
    /// 6M `MAX_GRID_VOXELS` cap — EXPORTS SUCCESSFULLY via the streaming path. We assert
    /// (a) the dense single-shape guard WOULD reject it, and (b) the streamed export
    /// produces the full surface+interior occupancy without a whole-region densify.
    #[test]
    fn streamed_vox_export_dissolves_6m_cap_on_large_solid() {
        let vpb = 16u32;
        let blocks = 50u32;
        let shape = SdfShape::from_blocks(ShapeKind::Box, [blocks, blocks, blocks], 1, vpb);
        // (a) The dense single-shape cap WOULD reject this scene outright.
        assert!(
            shape.exceeds_voxel_cap(vpb),
            "the large solid must exceed the dense 6M cap to prove the point"
        );
        let node = Node::new("BigBox", NodeContent::Tool { shape, material: Mat::Stone });
        let scene = Scene::from_nodes(vec![node]);
        let rgba = VoxExport::block_palette_from_active(Mat::Stone, [132, 126, 118, 255]);

        // (b) The streamed export succeeds — no whole-region grid is ever built.
        let export = streamed_vox_export(&scene, vpb, rgba);

        // The export holds the FULL occupancy (surface shell + coarse interior fast-fill).
        // 800³ voxels = 5.1e8; the dense path could never assemble it. (The `.vox` 256
        // cap tiles the 800-axis into ceil(800/256)=4 models per axis.)
        let region_voxels = (blocks as u64 * vpb as u64).pow(3);
        assert_eq!(
            export.voxel_count() as u64,
            region_voxels,
            "the streamed export must hold the FULL solid occupancy (surface + interior \
             coarse fast-fill), far past the dense 6M cap"
        );
        // The file parses and tiles correctly (800 > 256 on every axis → 4³ = 64 models).
        let parsed = dot_vox::load_bytes(&export.to_bytes())
            .expect("dot_vox should parse the large streamed export");
        let total: u64 = parsed.models.iter().map(|m| m.voxels.len() as u64).sum();
        assert_eq!(total, region_voxels, "every voxel must survive the 256-split tiling");
        for model in &parsed.models {
            assert!(model.size.x <= 256 && model.size.y <= 256 && model.size.z <= 256);
        }
    }
}
