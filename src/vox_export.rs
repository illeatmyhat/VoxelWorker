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
//! MagicaVoxel uses a **Z-up** right-handed coordinate system; our world is
//! **Y-up**. So our grid index `(i, j_up, k)` maps to a vox coordinate
//! `(x, z, y_up) = (i, k, j_up)` — our vertical axis becomes vox Z, and our
//! depth axis (Z) becomes vox Y. This makes the model stand upright in
//! MagicaVoxel and the mod instead of lying on its side.
//!
//! ## 256 limit
//!
//! A single MagicaVoxel model caps at 256 voxels per axis (coords are `u8`). If
//! any grid dimension exceeds 256 we **split** along that axis into a tiled set
//! of models inside the one MAIN chunk (each ≤ 256), rather than truncating.
//! Typical grids (e.g. 80×16×80) are a single model. The split is documented in
//! [`VoxExport::model_count`].
//!
//! ## Colour
//!
//! A minimal palette: the model's representative colour (the average colour of
//! the active material) is written to palette index 1 and every voxel references
//! it. `material_id` is currently always 0, so a single representative entry is
//! sufficient for v1.

use crate::voxel::VoxelGrid;

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
    /// RGBA palette entry written at file index 1 (the representative colour).
    representative_rgba: [u8; 4],
    /// Total occupied voxels written across all models (== grid.occupied.len()).
    voxel_count: usize,
}

impl VoxExport {
    /// Build the export from a resolved grid, mapping Y-up → Z-up and tiling into
    /// ≤256 models if any dimension exceeds [`VOX_AXIS_MAX`].
    ///
    /// `representative_rgba` is the single palette colour every voxel references
    /// (e.g. the average colour of the active material texture).
    pub fn from_grid(grid: &VoxelGrid, representative_rgba: [u8; 4]) -> Self {
        let [grid_x, grid_y, grid_z] = grid.dimensions;
        let half_x = grid_x as f32 / 2.0;
        let half_y = grid_y as f32 / 2.0;
        let half_z = grid_z as f32 / 2.0;

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
                    // vox size = (our X, our Z, our Y) after Y-up→Z-up swap.
                    models.push(VoxModel {
                        size: [sx, sz, sy],
                        voxels: Vec::new(),
                    });
                }
            }
        }

        let mut voxel_count = 0usize;
        for voxel in &grid.occupied {
            // Recover non-negative integer grid indices from the world-centred
            // voxel-centre position: `i = round(world_x + dim_x/2 - 0.5)`.
            let i = (voxel.world_position[0] + half_x - 0.5).round();
            let j = (voxel.world_position[1] + half_y - 0.5).round();
            let k = (voxel.world_position[2] + half_z - 0.5).round();
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
            // Y-up → Z-up: vox (x, y, z) = (our i, our k, our j_up).
            models[model_pos].voxels.push(VoxVoxel {
                x: local_i as u8,
                y: local_k as u8,
                z: local_j as u8,
                color_index: 1,
            });
            voxel_count += 1;
        }

        // Drop empty models (a sparse split can leave some tiles with nothing).
        models.retain(|model| !model.voxels.is_empty());
        // Always emit at least one (possibly empty) model so the file is valid.
        if models.is_empty() {
            models.push(VoxModel {
                size: [
                    grid_x.clamp(1, VOX_AXIS_MAX),
                    grid_z.clamp(1, VOX_AXIS_MAX),
                    grid_y.clamp(1, VOX_AXIS_MAX),
                ],
                voxels: Vec::new(),
            });
        }

        Self {
            models,
            representative_rgba,
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

        // RGBA palette chunk: 256 entries. MagicaVoxel reads palette[i] for file
        // index i+1, so our representative colour at file index 1 is the first
        // array entry. The rest are a neutral grey so the file stays valid.
        let mut rgba = Vec::with_capacity(256 * 4);
        for entry in 0..256 {
            if entry == 0 {
                rgba.extend_from_slice(&self.representative_rgba);
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
    use crate::voxel::{SdfShape, ShapeKind, VoxelProducer, VoxelGrid};

    /// Resolve a small cylinder and round-trip it through `.vox`, asserting the
    /// voxel count and dimensions survive the Y-up → Z-up mapping.
    #[test]
    fn vox_round_trip_matches_grid() {
        let shape = SdfShape {
            kind: ShapeKind::Cylinder,
            size_blocks: [5, 1, 5],
            voxels_per_block: 16,
            wall_blocks: 1,
        };
        let mut grid = VoxelGrid::new(shape.grid_dimensions());
        shape.resolve(&mut grid);
        assert!(grid.occupied_count() > 0, "expected a non-empty grid");

        let export = VoxExport::from_grid(&grid, [132, 126, 118, 255]);
        assert_eq!(export.model_count(), 1, "small grid is a single model");
        assert_eq!(export.voxel_count(), grid.occupied_count());

        let bytes = export.to_bytes();
        let parsed = dot_vox::load_bytes(&bytes).expect("dot_vox should parse our file");

        assert_eq!(parsed.models.len(), 1);
        let model = &parsed.models[0];
        // vox size = (our X, our Z, our Y). Grid is 80×16×80 → vox 80×80×16.
        let [gx, gy, gz] = grid.dimensions;
        assert_eq!(model.size.x, gx);
        assert_eq!(model.size.y, gz);
        assert_eq!(model.size.z, gy);
        // Every occupied voxel was written exactly once.
        assert_eq!(model.voxels.len(), grid.occupied_count());
        // All coordinates are within the model's declared size.
        for voxel in &model.voxels {
            assert!((voxel.x as u32) < model.size.x);
            assert!((voxel.y as u32) < model.size.y);
            assert!((voxel.z as u32) < model.size.z);
        }
    }

    /// A grid wider than 256 voxels on an axis must split into multiple models,
    /// never silently truncate.
    #[test]
    fn vox_splits_models_over_256() {
        // 17 blocks × 16 vx = 272 > 256 on X.
        let shape = SdfShape {
            kind: ShapeKind::Box,
            size_blocks: [17, 1, 1],
            voxels_per_block: 16,
            wall_blocks: 1,
        };
        let mut grid = VoxelGrid::new(shape.grid_dimensions());
        shape.resolve(&mut grid);

        let export = VoxExport::from_grid(&grid, [200, 200, 200, 255]);
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
}
