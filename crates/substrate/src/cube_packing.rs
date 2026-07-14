//! Packing linear slots into a cube-shaped tile grid: `slot → 3D tile origin`.
//!
//! `CubeTilePacking` lays out a linear sequence of equal-edge tiles into the smallest cubic
//! grid that holds them. Given `tile_count` tiles of edge `e`, it chooses `tiles_per_axis =
//! ceil(cbrt(tile_count))` so the `tiles_per_axis³` grid covers every slot, giving a cube of
//! side `tiles_per_axis · e`. Slot `s` occupies the tile at grid coordinate `(s mod t, (s /
//! t) mod t, s / t²)` (x-fastest, the standard row-major delinearization), whose low corner
//! in cube units is that coordinate times `e`. This is the space-packing an atlas/texture
//! layout wants: a 1D allocation index turned into a 3D placement that keeps the containing
//! volume near-cubic (so no axis blows past a hardware dimension limit) and grows by whole
//! tiles.
//!
//! The cube-root sizing wastes at most one partial shell of tiles (the grid rounds up), the
//! informed trade for a single scalar `tiles_per_axis` that both the packer and a
//! "did the grid grow?" check derive identically.
//!
//! Cite: row-major / space-filling delinearization folklore (the mixed-radix index split);
//! texture-atlas packing practice (a linear allocation tiled into an N³ volume). No single
//! canonical citation — the geometry is `count → ceil(cbrt) → mixed-radix origin`.

use crate::bit_cube::BitCube;

/// The geometry of a cube-shaped tile grid: how many tiles per axis, and the resulting cube
/// side in cells. Both derive from a tile count and a tile edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CubeTilePacking {
    /// Tiles along each axis of the grid (`ceil(cbrt(tile_count))`, `0` for no tiles).
    pub tiles_per_axis: u32,
    /// The packed cube's side in cells (`tiles_per_axis · tile_edge`, `0` for no tiles).
    pub cube_dim_cells: u32,
}

impl CubeTilePacking {
    /// The tiles-per-axis a slot count packs to: `ceil(cbrt(count))`, or `0` for an empty
    /// set. The grid-edge scalar both the packer and a grow test read.
    pub fn tiles_per_axis(tile_count: usize) -> u32 {
        if tile_count == 0 {
            0
        } else {
            ((tile_count as f64).cbrt().ceil() as u32).max(1)
        }
    }

    /// The packing geometry for `tile_count` tiles of edge `tile_edge`.
    pub fn for_tile_count(tile_count: usize, tile_edge: u32) -> Self {
        let tiles_per_axis = Self::tiles_per_axis(tile_count);
        Self {
            tiles_per_axis,
            cube_dim_cells: tiles_per_axis * tile_edge,
        }
    }

    /// The low-corner cell of `slot`'s tile in the cube (linear slot → 3D tile coord,
    /// x-fastest, times the tile edge).
    pub fn tile_origin_cells(&self, slot: u32, tile_edge: u32) -> [usize; 3] {
        let tiles = self.tiles_per_axis;
        let edge = tile_edge as usize;
        [
            (slot % tiles) as usize * edge,
            ((slot / tiles) % tiles) as usize * edge,
            (slot / (tiles * tiles)) as usize * edge,
        ]
    }

    /// Scatter a slot-indexed slice of [`BitCube`] tiles into one cube-shaped byte buffer:
    /// each tile lands at its slot's origin, its set bits written as `set_byte` (via the
    /// [`BitCube`] row-expand seam), clear cells left `0`. Returns `(tiles_per_axis,
    /// cube_dim_cells, bytes)`. Every tile must share `tile_edge`. A count of `0` yields
    /// `(0, 0, empty)`.
    pub fn pack_bit_cubes(
        tiles: &[BitCube],
        tile_edge: u32,
        set_byte: u8,
    ) -> (u32, u32, Vec<u8>) {
        let edge = tile_edge as usize;
        let packing = Self::for_tile_count(tiles.len(), tile_edge);
        let cube_dim = packing.cube_dim_cells as usize;
        let mut bytes = vec![0u8; cube_dim * cube_dim * cube_dim];
        for (slot, tile) in tiles.iter().enumerate() {
            debug_assert_eq!(
                tile.edge(),
                tile_edge,
                "every packed tile must share the given tile edge"
            );
            let origin = packing.tile_origin_cells(slot as u32, tile_edge);
            for local_z in 0..edge {
                for local_y in 0..edge {
                    let row_index = local_z * edge + local_y;
                    let dest = ((origin[2] + local_z) * cube_dim + origin[1] + local_y) * cube_dim
                        + origin[0];
                    tile.expand_row_into(row_index, &mut bytes[dest..dest + edge], set_byte);
                }
            }
        }
        (packing.tiles_per_axis, packing.cube_dim_cells, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `tiles_per_axis` is `ceil(cbrt(count))`: 0→0, 1→1, 8→2, 9→3 (rounds up past a full
    /// cube), 27→3, 28→4.
    #[test]
    fn tiles_per_axis_is_ceil_cube_root() {
        for (count, expected) in [(0, 0), (1, 1), (7, 2), (8, 2), (9, 3), (27, 3), (28, 4)] {
            assert_eq!(CubeTilePacking::tiles_per_axis(count), expected, "count {count}");
        }
    }

    /// The slot → origin map is the x-fastest mixed-radix split, and packing a set of
    /// single-cell tiles places each tile's one set cell at exactly its slot origin — the
    /// index bijection an atlas consumer relies on.
    #[test]
    fn pack_places_each_tile_at_its_slot_origin() {
        // Five edge-2 tiles, each with only cell (0,0,0) set. tiles_per_axis = ceil(cbrt 5) = 2,
        // cube side = 4.
        let tile_edge = 2;
        let tiles: Vec<BitCube> = (0..5)
            .map(|_| {
                let mut cube = BitCube::empty(tile_edge);
                cube.set_x_run(0, 0, 0, 0);
                cube
            })
            .collect();
        let (tiles_per_axis, cube_dim, bytes) =
            CubeTilePacking::pack_bit_cubes(&tiles, tile_edge, 255);
        assert_eq!(tiles_per_axis, 2);
        assert_eq!(cube_dim, 4);
        let cube_dim = cube_dim as usize;

        let packing = CubeTilePacking::for_tile_count(tiles.len(), tile_edge);
        // Exactly the five slot-origin cells are set; everything else is clear.
        let mut expected = vec![0u8; cube_dim * cube_dim * cube_dim];
        for slot in 0..tiles.len() as u32 {
            let origin = packing.tile_origin_cells(slot, tile_edge);
            expected[(origin[2] * cube_dim + origin[1]) * cube_dim + origin[0]] = 255;
        }
        assert_eq!(bytes, expected);
    }

    /// An empty tile set packs to a zero-sized cube.
    #[test]
    fn empty_packs_to_a_zero_cube() {
        let (tiles_per_axis, cube_dim, bytes) = CubeTilePacking::pack_bit_cubes(&[], 4, 255);
        assert_eq!((tiles_per_axis, cube_dim), (0, 0));
        assert!(bytes.is_empty());
    }
}
