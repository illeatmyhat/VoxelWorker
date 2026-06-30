//! MagicaVoxel `.vox` export (Milestone 8).
//!
//! Serialises a resolved [`VoxelGrid`](crate::voxel::VoxelGrid) to a MagicaVoxel
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

use crate::voxel::VoxelGrid;

/// The per-`block_id` RGBA palette the `.vox` export writes (ADR 0003 §3a). Index `i`
/// is the colour for `block_id == i`; it is written to `.vox` file palette slot `i + 1`
/// (MagicaVoxel palette is 1-based, 0 = empty). Sized to the procedural material set —
/// the categorical capability over the existing three materials.
pub type BlockPaletteColors = [[u8; 4]; crate::core_geom::MaterialChoice::MATERIAL_COUNT];

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
    /// build it with [`block_palette_from_active`] or pass the procedural material
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
        active: crate::core_geom::MaterialChoice,
        representative_rgba: [u8; 4],
    ) -> BlockPaletteColors {
        let mut palette = [[0x80, 0x80, 0x80, 0xff]; crate::core_geom::MaterialChoice::MATERIAL_COUNT];
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
        chunk_voxels: impl IntoIterator<Item = &'voxels [crate::voxel::Voxel]>,
        palette_colors: BlockPaletteColors,
    ) -> Self {
        let [grid_x, grid_y, grid_z] = region_dimensions;
        // Corner-anchoring decode: FLOORED half (`dim/2` integer division), so
        // `round(world + floor(dim/2) − 0.5)` recovers the exact index for an odd dim
        // too (see voxel.rs::widest_run_in_band).
        let half_x = (grid_x / 2) as f32;
        let half_y = (grid_y / 2) as f32;
        let half_z = (grid_z / 2) as f32;

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

        let mut voxel_count = 0usize;
        for voxel in chunk_voxels.into_iter().flatten() {
            // Recover non-negative integer grid indices from the world-centred
            // voxel-centre position: `i = round(world_x + dim_x/2 - 0.5)`.
            let position = voxel.world_position();
            let i = (position[0] + half_x - 0.5).round();
            let j = (position[1] + half_y - 0.5).round();
            let k = (position[2] + half_z - 0.5).round();
            if i < 0.0 || j < 0.0 || k < 0.0 {
                continue;
            }
            let i = i as u32;
            let j = j as u32;
            let k = k as u32;
            if i >= grid_x || j >= grid_y || k >= grid_z {
                continue;
            }

            let tx = i / VOX_AXIS_MAX;
            let ty = j / VOX_AXIS_MAX;
            let tz = k / VOX_AXIS_MAX;
            let local_i = i % VOX_AXIS_MAX;
            let local_j = j % VOX_AXIS_MAX;
            let local_k = k % VOX_AXIS_MAX;

            let Some(&model_pos) = model_index.get(&(tx, ty, tz)) else {
                continue;
            };
            // Z-up: vox (x, y, z) = (our i, our j, our k) — no swap. The categorical
            // block id selects the `.vox` palette slot (`block_id + 1`; ADR 0003 §3a),
            // so a multi-material model exports each block in its own colour. Clamp to
            // the procedural palette so a stray id stays in range.
            let palette_slot = voxel
                .color_index()
                .min(crate::core_geom::MaterialChoice::MATERIAL_COUNT as u16 - 1)
                as u8
                + 1;
            models[model_pos].voxels.push(VoxVoxel {
                x: local_i as u8,
                y: local_j as u8,
                z: local_k as u8,
                color_index: palette_slot,
            });
            voxel_count += 1;
        }

        // Drop empty models (a sparse split can leave some tiles with nothing).
        models.retain(|model| !model.voxels.is_empty());
        // Always emit at least one (possibly empty) model so the file is valid.
        if models.is_empty() {
            models.push(VoxModel {
                // Z-up: no swap — vox size = (our X, our Y, our Z).
                size: [
                    grid_x.clamp(1, VOX_AXIS_MAX),
                    grid_y.clamp(1, VOX_AXIS_MAX),
                    grid_z.clamp(1, VOX_AXIS_MAX),
                ],
                voxels: Vec::new(),
            });
        }

        Self {
            models,
            palette_colors,
            voxel_count,
        }
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
    pub fn write(&self, path: &std::path::Path) -> std::io::Result<usize> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let bytes = self.to_bytes();
        std::fs::write(path, &bytes)?;
        Ok(bytes.len())
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
    use crate::core_geom::MaterialChoice;
    use crate::voxel::{SdfShape, ShapeKind};

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
            crate::core_geom::MaterialChoice::Stone,
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
            crate::core_geom::MaterialChoice::Stone,
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
            crate::core_geom::MaterialChoice::Stone,
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
    /// `ChunkResolveCache::vox_export` instead of `resolve_scene` + `from_grid`. This
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
}
