//! CPU material-texture atlas packer (ADR 0002 E3c-1, part of #18, decision O8).
//!
//! Vintage Story packs every block texture into ONE atlas and emits atlas UVs per
//! vertex, so a whole chunk of mixed-material geometry becomes a SINGLE mesh = a
//! single draw call regardless of how many materials it contains. This module is
//! the CPU side of that for the flag-gated cuboid mesher: it lays each material's
//! texture into one atlas image and records, per material, the UV sub-rectangle it
//! occupies. The cuboid mesher then maps each face's `material_id` to its sub-rect
//! and emits atlas UVs (`cuboid_mesh.rs`); the cuboid shader samples the one atlas
//! (`shaders/cuboid.wgsl`).
//!
//! ## Packing strategy: gutter-padded shelf/grid packer
//! All current procedural materials (Stone/Wood/Plain) are the SAME square size,
//! so a simple **shelf packer** that places tiles left-to-right, wrapping to a new
//! shelf (row) when the current shelf would overflow a target atlas width, is both
//! sufficient and trivially correct. A grid is just the degenerate shelf case where
//! every tile is identical in size — which is exactly our case — but the shelf
//! algorithm is written generally so a future loaded-VS-block material of a
//! different size still packs without a rewrite. The choice is documented here per
//! the task: a shelf packer is chosen over a full rectangle-bin-packing (MaxRects /
//! guillotine) because our tile set is tiny and near-uniform — the packing-density
//! win of a smarter packer is irrelevant, and a shelf packer has no failure modes.
//!
//! ## Texture bleeding / half-texel inset (the load-bearing correctness detail)
//! The cuboid per-voxel texture slice TILES a material's tile once per voxel across
//! a merged face. With the whole atlas in one texture, that tiling can NOT use the
//! GPU `Repeat` address mode (it would wrap to the WHOLE atlas, i.e. into a
//! neighbouring material). The shader instead computes `fract(per_voxel_uv)` itself
//! and maps that `[0,1)` into the material's sub-rect. Two artifacts must be
//! defended against at atlas-cell borders:
//!   1. **Filter/derivative spill across the cell border.** Even with NEAREST
//!      filtering, a fragment exactly on a tile seam can sample the adjacent cell.
//!      We surround every tile with a **replicated-edge gutter** (the tile's own
//!      border pixels copied outward by `GUTTER_TEXELS`), so a one-texel spill lands
//!      on a copy of the correct edge, never the neighbour material.
//!   2. **Wrap seam within a tile.** Because the shader tiles with `fract`, the
//!      sampled sub-rect is reported with a **half-texel inset** on each side
//!      (`inset_uv_*`): sampling is clamped to texel centres, so `fract`→0 and
//!      `fract`→1 both land inside the tile rather than on its outer edge where they
//!      could round into the gutter.
//!
//! The packer therefore returns BOTH the full tile UV rect (for diagnostics) and
//! the half-texel-inset sampling window the shader actually uses.

use crate::core_geom::MaterialChoice;
// The pure packer geometry — shelf layout, half-texel-inset UV rects, and the
// replicated-edge blit — is textbook rectangle packing with no material/texture
// vocabulary, so it lives in substrate (see the material-atlas handling in
// docs/architecture). This module is the domain adapter: it owns the material
// ordering, the gutter choice, and the `MaterialAtlas`/`AtlasSubRect` names.
use substrate::occupancy::shelf_bin_pack::{ShelfBinPack, TileImage, TileSize};

/// Texels of replicated-edge gutter padded around every tile in the atlas. One
/// texel is enough to absorb a single-texel filter/derivative spill at a cell
/// border; we use a small constant rather than 0 so the seam defence is explicit.
pub const GUTTER_TEXELS: u32 = 1;

/// One material's place in the atlas, in atlas UV space (`[0,1]` across the whole
/// atlas image). `min_*`/`max_*` are the tile's OUTER bounds (the original tile,
/// excluding the gutter); `inset_*` are those bounds pulled in by half a texel so
/// the shader's `fract`-based per-voxel tiling never samples the outermost texel
/// edge (where it could round into the gutter / neighbour). The shader maps a
/// per-voxel `fract` in `[0,1)` linearly into the inset window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtlasSubRect {
    /// Outer tile bounds in atlas UV (excludes the gutter).
    pub min_u: f32,
    pub min_v: f32,
    pub max_u: f32,
    pub max_v: f32,
    /// Half-texel-inset sampling window the shader tiles into.
    pub inset_min_u: f32,
    pub inset_min_v: f32,
    pub inset_max_u: f32,
    pub inset_max_v: f32,
}

impl AtlasSubRect {
    /// The inset window's UV size (`max - min`) on each axis; the shader scales a
    /// per-voxel `fract` by this and offsets by `inset_min_*`.
    pub fn inset_size(&self) -> [f32; 2] {
        [
            self.inset_max_u - self.inset_min_u,
            self.inset_max_v - self.inset_min_v,
        ]
    }
}

/// The packed atlas: one RGBA8 image plus the per-material UV sub-rects (indexed by
/// `material_id`, i.e. [`MaterialChoice`] order).
#[derive(Debug, Clone)]
pub struct MaterialAtlas {
    pub width: u32,
    pub height: u32,
    /// Row-major RGBA8 (4 bytes/texel), `width * height * 4` long.
    pub pixels: Vec<u8>,
    /// One sub-rect per material, in `material_id` order.
    pub sub_rects: Vec<AtlasSubRect>,
}

/// A source tile handed to the packer: its pixel dimensions and RGBA8 pixels.
#[derive(Debug, Clone)]
pub struct AtlasSourceTile {
    pub width: u32,
    pub height: u32,
    /// Row-major RGBA8, `width * height * 4` long.
    pub pixels: Vec<u8>,
}

impl MaterialAtlas {
    /// Pack the procedural Stone/Wood/Plain material textures into one atlas, in
    /// [`MaterialChoice`] order so the sub-rect for `material_id == m` is
    /// `sub_rects[m]`. This is the cuboid path's atlas source (it binds the SAME
    /// procedural textures the instanced path uses, just packed).
    pub fn from_procedural_materials() -> MaterialAtlas {
        let texture_size = crate::renderer::procedural_material_texture_size();
        let tiles: Vec<AtlasSourceTile> = crate::renderer::procedural_material_pixels()
            .into_iter()
            .map(|pixels| AtlasSourceTile {
                width: texture_size,
                height: texture_size,
                pixels,
            })
            .collect();
        debug_assert_eq!(
            tiles.len(),
            MaterialChoice::MATERIAL_COUNT,
            "atlas expects one tile per material"
        );
        Self::pack_tiles(&tiles)
    }

    /// Pack arbitrary RGBA8 tiles into one atlas with a replicated-edge gutter and
    /// half-texel-inset sampling windows (see the module docs). The shelf layout,
    /// UV-rect math, and gutter blit are substrate's [`ShelfBinPack`]; this adapter
    /// supplies the tile sizes and the [`GUTTER_TEXELS`] gutter, then names the
    /// result in material-atlas vocabulary. Byte-identical to the pre-extraction
    /// output (the app's texture rendering + goldens pin it).
    pub fn pack_tiles(tiles: &[AtlasSourceTile]) -> MaterialAtlas {
        if tiles.is_empty() {
            return MaterialAtlas {
                width: 1,
                height: 1,
                pixels: vec![0, 0, 0, 255],
                sub_rects: Vec::new(),
            };
        }

        let sizes: Vec<TileSize> = tiles
            .iter()
            .map(|tile| TileSize {
                width: tile.width,
                height: tile.height,
            })
            .collect();
        let layout = ShelfBinPack::plan(&sizes, GUTTER_TEXELS);

        // Blit each tile + its replicated-edge gutter and record its UV sub-rect
        // (RGBA8 = 4 bytes/texel).
        let mut pixels = vec![0u8; (layout.sheet_width * layout.sheet_height * 4) as usize];
        let mut sub_rects = Vec::with_capacity(tiles.len());
        for (tile, placement) in tiles.iter().zip(layout.placements.iter()) {
            ShelfBinPack::blit_with_replicated_edge(
                &mut pixels,
                layout.sheet_width,
                4,
                &TileImage {
                    width: tile.width,
                    height: tile.height,
                    pixels: &tile.pixels,
                },
                *placement,
                GUTTER_TEXELS,
            );
            let rect = ShelfBinPack::normalized_rect(
                placement,
                tile.width,
                tile.height,
                layout.sheet_width,
                layout.sheet_height,
            );
            sub_rects.push(AtlasSubRect {
                min_u: rect.min_u,
                min_v: rect.min_v,
                max_u: rect.max_u,
                max_v: rect.max_v,
                inset_min_u: rect.inset_min_u,
                inset_min_v: rect.inset_min_v,
                inset_max_u: rect.inset_max_u,
                inset_max_v: rect.inset_max_v,
            });
        }

        MaterialAtlas {
            width: layout.sheet_width,
            height: layout.sheet_height,
            pixels,
            sub_rects,
        }
    }
}

// The pure packer-geometry tests (one-placement-per-tile, unit-square/disjoint
// rects, gutter edge replication) moved with the geometry to
// `substrate::occupancy::shelf_bin_pack`. The tests that stay here exercise the DOMAIN adapter:
// the material ordering and the empty-list placeholder.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn procedural_atlas_has_three_materials() {
        let atlas = MaterialAtlas::from_procedural_materials();
        assert_eq!(atlas.sub_rects.len(), MaterialChoice::MATERIAL_COUNT);
        assert!(atlas.width > 0 && atlas.height > 0);
        assert_eq!(
            atlas.pixels.len(),
            (atlas.width * atlas.height * 4) as usize
        );
    }

    #[test]
    fn empty_tile_list_yields_placeholder() {
        let atlas = MaterialAtlas::pack_tiles(&[]);
        assert!(atlas.sub_rects.is_empty());
        assert_eq!(atlas.width, 1);
        assert_eq!(atlas.height, 1);
    }
}
