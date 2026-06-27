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
    /// half-texel-inset sampling windows (see the module docs). Tiles are laid out
    /// left-to-right on shelves, wrapping to a new shelf when the current one would
    /// exceed `TARGET_ATLAS_WIDTH_TILES` tiles' worth of width; for our uniform
    /// tiles this lays a near-square grid.
    pub fn pack_tiles(tiles: &[AtlasSourceTile]) -> MaterialAtlas {
        if tiles.is_empty() {
            return MaterialAtlas {
                width: 1,
                height: 1,
                pixels: vec![0, 0, 0, 255],
                sub_rects: Vec::new(),
            };
        }

        // --- Plan the shelf layout (positions only; no pixels yet). ---
        // Wrap to a new shelf to keep the atlas roughly square: cap a shelf at
        // `tiles_per_shelf` tiles, chosen as ceil(sqrt(count)) so N near-uniform
        // tiles form an R×C grid with R ≈ C.
        let tile_count = tiles.len();
        let tiles_per_shelf = (tile_count as f32).sqrt().ceil() as usize;
        let tiles_per_shelf = tiles_per_shelf.max(1);

        // Each tile is padded with a gutter on every side. The cell a tile occupies
        // is `tile + 2 * gutter` on each axis.
        let padded = |dimension: u32| -> u32 { dimension + 2 * GUTTER_TEXELS };

        // Walk the shelves to size the atlas: a shelf's height is its tallest padded
        // tile; the atlas width is the widest padded shelf.
        let mut placements: Vec<TilePlacement> = Vec::with_capacity(tile_count);
        let mut atlas_width: u32 = 0;
        let mut shelf_origin_y: u32 = 0;
        let mut index = 0;
        while index < tile_count {
            let shelf_end = (index + tiles_per_shelf).min(tile_count);
            let mut shelf_cursor_x: u32 = 0;
            let mut shelf_height: u32 = 0;
            for tile in &tiles[index..shelf_end] {
                // Tile's inner (gutter-excluded) top-left in atlas pixels.
                let inner_x = shelf_cursor_x + GUTTER_TEXELS;
                let inner_y = shelf_origin_y + GUTTER_TEXELS;
                placements.push(TilePlacement { inner_x, inner_y });
                shelf_cursor_x += padded(tile.width);
                shelf_height = shelf_height.max(padded(tile.height));
            }
            atlas_width = atlas_width.max(shelf_cursor_x);
            shelf_origin_y += shelf_height;
            index = shelf_end;
        }
        let atlas_height = shelf_origin_y.max(1);
        let atlas_width = atlas_width.max(1);

        // --- Blit the tiles + their replicated-edge gutters into the atlas image. ---
        let mut pixels = vec![0u8; (atlas_width * atlas_height * 4) as usize];
        let mut sub_rects = Vec::with_capacity(tile_count);
        for (tile, placement) in tiles.iter().zip(placements.iter()) {
            blit_tile_with_gutter(
                &mut pixels,
                atlas_width,
                tile,
                placement.inner_x,
                placement.inner_y,
            );
            sub_rects.push(sub_rect_for(
                placement,
                tile.width,
                tile.height,
                atlas_width,
                atlas_height,
            ));
        }

        MaterialAtlas {
            width: atlas_width,
            height: atlas_height,
            pixels,
            sub_rects,
        }
    }
}

/// Where a tile's inner (gutter-excluded) region sits in the atlas, in pixels.
struct TilePlacement {
    inner_x: u32,
    inner_y: u32,
}

/// Compute a material's UV sub-rect (outer bounds + half-texel-inset sampling
/// window) from its pixel placement and the atlas dimensions.
fn sub_rect_for(
    placement: &TilePlacement,
    tile_width: u32,
    tile_height: u32,
    atlas_width: u32,
    atlas_height: u32,
) -> AtlasSubRect {
    let atlas_w = atlas_width as f32;
    let atlas_h = atlas_height as f32;
    // Outer tile bounds in atlas UV (the inner region, excluding the gutter).
    let min_u = placement.inner_x as f32 / atlas_w;
    let min_v = placement.inner_y as f32 / atlas_h;
    let max_u = (placement.inner_x + tile_width) as f32 / atlas_w;
    let max_v = (placement.inner_y + tile_height) as f32 / atlas_h;
    // Half-texel inset: pull each edge in by half an atlas texel so the shader's
    // fract-tiling lands on texel centres, never the outermost edge (which could
    // round into the gutter under interpolation/derivatives).
    let half_texel_u = 0.5 / atlas_w;
    let half_texel_v = 0.5 / atlas_h;
    AtlasSubRect {
        min_u,
        min_v,
        max_u,
        max_v,
        inset_min_u: min_u + half_texel_u,
        inset_min_v: min_v + half_texel_v,
        inset_max_u: max_u - half_texel_u,
        inset_max_v: max_v - half_texel_v,
    }
}

/// Copy one RGBA8 tile into the atlas at `(inner_x, inner_y)` AND replicate its
/// edge pixels outward by [`GUTTER_TEXELS`] on every side, so a one-texel sample
/// spill at a cell border reads a copy of the correct edge, never a neighbour
/// material's pixels.
fn blit_tile_with_gutter(
    atlas_pixels: &mut [u8],
    atlas_width: u32,
    tile: &AtlasSourceTile,
    inner_x: u32,
    inner_y: u32,
) {
    let gutter = GUTTER_TEXELS as i64;
    let tile_w = tile.width as i64;
    let tile_h = tile.height as i64;

    // Fill the padded cell (tile + gutter on every side). For a destination texel,
    // clamp its tile-local coordinate into the tile so border texels replicate.
    for cell_dy in -gutter..(tile_h + gutter) {
        for cell_dx in -gutter..(tile_w + gutter) {
            let source_x = cell_dx.clamp(0, tile_w - 1);
            let source_y = cell_dy.clamp(0, tile_h - 1);
            let source_index = ((source_y * tile_w + source_x) * 4) as usize;

            let dest_x = inner_x as i64 + cell_dx;
            let dest_y = inner_y as i64 + cell_dy;
            if dest_x < 0 || dest_y < 0 {
                continue;
            }
            let dest_index = ((dest_y * atlas_width as i64 + dest_x) * 4) as usize;
            if dest_index + 4 > atlas_pixels.len() || source_index + 4 > tile.pixels.len() {
                continue;
            }
            atlas_pixels[dest_index..dest_index + 4]
                .copy_from_slice(&tile.pixels[source_index..source_index + 4]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A solid-colour RGBA8 tile of the given size.
    fn solid_tile(size: u32, color: [u8; 4]) -> AtlasSourceTile {
        AtlasSourceTile {
            width: size,
            height: size,
            pixels: color.repeat((size * size) as usize),
        }
    }

    #[test]
    fn packs_one_sub_rect_per_tile() {
        let tiles = vec![
            solid_tile(4, [255, 0, 0, 255]),
            solid_tile(4, [0, 255, 0, 255]),
            solid_tile(4, [0, 0, 255, 255]),
        ];
        let atlas = MaterialAtlas::pack_tiles(&tiles);
        assert_eq!(atlas.sub_rects.len(), 3, "one sub-rect per tile");
        // Atlas image is the right byte length for its declared size.
        assert_eq!(
            atlas.pixels.len(),
            (atlas.width * atlas.height * 4) as usize
        );
    }

    #[test]
    fn sub_rects_are_inside_unit_uv_and_disjoint() {
        let tiles = vec![
            solid_tile(8, [10, 20, 30, 255]),
            solid_tile(8, [40, 50, 60, 255]),
            solid_tile(8, [70, 80, 90, 255]),
            solid_tile(8, [100, 110, 120, 255]),
        ];
        let atlas = MaterialAtlas::pack_tiles(&tiles);
        for rect in &atlas.sub_rects {
            assert!(rect.min_u >= 0.0 && rect.max_u <= 1.0);
            assert!(rect.min_v >= 0.0 && rect.max_v <= 1.0);
            // Inset window is strictly inside the outer bounds.
            assert!(rect.inset_min_u > rect.min_u && rect.inset_max_u < rect.max_u);
            assert!(rect.inset_min_v > rect.min_v && rect.inset_max_v < rect.max_v);
        }
        // No two outer rects overlap (the gutter guarantees a 1-texel gap → strict).
        for a in 0..atlas.sub_rects.len() {
            for b in (a + 1)..atlas.sub_rects.len() {
                let ra = &atlas.sub_rects[a];
                let rb = &atlas.sub_rects[b];
                let disjoint = ra.max_u <= rb.min_u
                    || rb.max_u <= ra.min_u
                    || ra.max_v <= rb.min_v
                    || rb.max_v <= ra.min_v;
                assert!(disjoint, "sub-rects {a} and {b} overlap");
            }
        }
    }

    #[test]
    fn gutter_replicates_edge_pixels() {
        // A single 2×2 tile: its gutter ring must be copies of the nearest edge.
        let tile = AtlasSourceTile {
            width: 2,
            height: 2,
            // (0,0)=red (1,0)=green (0,1)=blue (1,1)=white
            pixels: vec![
                255, 0, 0, 255, // (0,0)
                0, 255, 0, 255, // (1,0)
                0, 0, 255, 255, // (0,1)
                255, 255, 255, 255, // (1,1)
            ],
        };
        let atlas = MaterialAtlas::pack_tiles(&[tile]);
        // With gutter=1, the inner tile sits at (1,1); the top-left gutter texel
        // (0,0) must replicate the tile's (0,0) corner = red.
        let texel = |x: u32, y: u32| -> [u8; 4] {
            let index = ((y * atlas.width + x) * 4) as usize;
            [
                atlas.pixels[index],
                atlas.pixels[index + 1],
                atlas.pixels[index + 2],
                atlas.pixels[index + 3],
            ]
        };
        assert_eq!(texel(0, 0), [255, 0, 0, 255], "corner gutter = corner pixel");
        assert_eq!(texel(1, 0), [255, 0, 0, 255], "top gutter above (0,0)=red");
        assert_eq!(texel(2, 0), [0, 255, 0, 255], "top gutter above (1,0)=green");
    }

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
