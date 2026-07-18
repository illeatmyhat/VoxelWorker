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
//! The geometry is independent of what a cell HOLDS: the same slot → origin map scatters
//! occupancy tiles ([`BitCube`], one byte per cell) and payload tiles ([`ValueCube<u16>`], two
//! little-endian bytes per cell) — a texel format enters only as its bytes-per-cell stride and
//! how it fills one X-row. Two pools packed this way therefore agree cell-for-cell on where
//! slot `s` lives, whatever their texel widths.
//!
//! Cite: row-major / space-filling delinearization folklore (the mixed-radix index split);
//! texture-atlas packing practice (a linear allocation tiled into an N³ volume). No single
//! canonical citation — the geometry is `count → ceil(cbrt) → mixed-radix origin`.

use crate::occupancy::bit_cube::BitCube;
use crate::occupancy::value_cube::ValueCube;

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
    pub fn pack_bit_cubes(tiles: &[BitCube], tile_edge: u32, set_byte: u8) -> (u32, u32, Vec<u8>) {
        debug_assert!(
            tiles.iter().all(|tile| tile.edge() == tile_edge),
            "every packed tile must share the given tile edge"
        );
        Self::scatter_tile_rows(tiles.len(), tile_edge, 1, |slot, row_index, out_row| {
            tiles[slot].expand_row_into(row_index, out_row, set_byte)
        })
    }

    /// Scatter a slot-indexed slice of [`ValueCube<u16>`] tiles into one cube-shaped buffer of
    /// **little-endian 16-bit texels** (2 bytes per cell): the payload sibling of
    /// [`pack_bit_cubes`](Self::pack_bit_cubes), same slot → tile-origin geometry, so a linear
    /// payload pool and a linear occupancy pool of the same slot count land cell-for-cell in
    /// the same places. Cells outside any tile stay `0`. Returns `(tiles_per_axis,
    /// cube_dim_cells, bytes)` — note the dimensions are in CELLS, while `bytes.len()` is
    /// `2 · cube_dim_cells³`. Every tile must share `tile_edge`; a count of `0` yields
    /// `(0, 0, empty)`.
    pub fn pack_u16_value_cubes(tiles: &[ValueCube<u16>], tile_edge: u32) -> (u32, u32, Vec<u8>) {
        debug_assert!(
            tiles.iter().all(|tile| tile.edge() == tile_edge),
            "every packed tile must share the given tile edge"
        );
        Self::scatter_tile_rows(tiles.len(), tile_edge, 2, |slot, row_index, out_row| {
            for (value, texel) in tiles[slot]
                .row(row_index)
                .iter()
                .zip(out_row.chunks_exact_mut(2))
            {
                texel.copy_from_slice(&value.to_le_bytes());
            }
        })
    }

    /// The shared scatter: walk every slot's `edge²` X-rows, hand each row's destination byte
    /// span to `write_row` (`(slot, row_index, out_row)`), and return the packed cube. The ONE
    /// place the cube-root sizing, the slot → origin map, and the row-span arithmetic live —
    /// a texel format enters only as its `bytes_per_texel` stride and how it fills a row, so
    /// two formats can never drift on the geometry.
    fn scatter_tile_rows<F>(
        tile_count: usize,
        tile_edge: u32,
        bytes_per_texel: usize,
        mut write_row: F,
    ) -> (u32, u32, Vec<u8>)
    where
        F: FnMut(usize, usize, &mut [u8]),
    {
        let edge = tile_edge as usize;
        let packing = Self::for_tile_count(tile_count, tile_edge);
        let cube_dim = packing.cube_dim_cells as usize;
        let mut bytes = vec![0u8; cube_dim * cube_dim * cube_dim * bytes_per_texel];
        let row_bytes = edge * bytes_per_texel;
        for slot in 0..tile_count {
            let origin = packing.tile_origin_cells(slot as u32, tile_edge);
            for local_z in 0..edge {
                for local_y in 0..edge {
                    let row_index = local_z * edge + local_y;
                    let dest_cell = ((origin[2] + local_z) * cube_dim + origin[1] + local_y)
                        * cube_dim
                        + origin[0];
                    let start = dest_cell * bytes_per_texel;
                    write_row(slot, row_index, &mut bytes[start..start + row_bytes]);
                }
            }
        }
        (packing.tiles_per_axis, packing.cube_dim_cells, bytes)
    }
}

/// Kani bounded-model-checking proof of [`CubeTilePacking::tile_origin_cells`]'s **slot → tile
/// bijection** — the addressing an atlas consumer trusts: two slots must never scatter to the
/// same tile (aliased occupancy/payload), and every tile must land inside the packed cube. Proved
/// over every slot of a representative grid and every tile edge `1..=64` (the density bound = the
/// verification bound). The packing struct is built directly with a CONCRETE `tiles_per_axis`,
/// which both keeps the slot-split divisions constant (cheap) and sidesteps
/// [`CubeTilePacking::tiles_per_axis`], whose `f64::cbrt` is a foreign function CBMC cannot model
/// (the sizing scalar is a separate, un-verified concern). `#[cfg(kani)]` keeps this inactive in
/// ordinary builds. Run under WSL: `cargo kani -p substrate`.
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// **The linear slot → 3D tile-origin map is an injective, in-bounds bijection.** For a
    /// representative `tiles_per_axis = 3` grid (a partial shell, `3³ = 27` slots) and every tile
    /// edge `1..=64`: each slot's tile fits wholly inside the cube (`origin[axis] + edge <=
    /// cube_dim`), and two slots share an origin IFF they are the same slot — the mixed-radix
    /// (x-fastest) uniqueness that stops an atlas from scattering two tiles onto one another.
    #[kani::proof]
    fn tile_origin_is_an_injective_in_bounds_slot_map() {
        const TILES_PER_AXIS: u32 = 3;
        const SLOT_COUNT: u32 = TILES_PER_AXIS * TILES_PER_AXIS * TILES_PER_AXIS; // 27
        let edge: u32 = {
            let value: u32 = kani::any();
            kani::assume(value >= 1 && value <= 64);
            value
        };
        let packing = CubeTilePacking {
            tiles_per_axis: TILES_PER_AXIS,
            cube_dim_cells: TILES_PER_AXIS * edge,
        };
        let cube_dim = (TILES_PER_AXIS * edge) as usize;

        let slot_a: u32 = kani::any();
        let slot_b: u32 = kani::any();
        kani::assume(slot_a < SLOT_COUNT && slot_b < SLOT_COUNT);
        let origin_a = packing.tile_origin_cells(slot_a, edge);
        let origin_b = packing.tile_origin_cells(slot_b, edge);

        // In bounds: every slot's whole tile `[origin, origin + edge)` fits inside the cube.
        let mut axis = 0;
        while axis < 3 {
            assert!(origin_a[axis] + edge as usize <= cube_dim);
            axis += 1;
        }
        // Injective: distinct slots land at distinct origins (no atlas aliasing).
        assert!((origin_a == origin_b) == (slot_a == slot_b));
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

    /// An empty tile set packs to a zero-sized cube — for both texel formats.
    #[test]
    fn empty_packs_to_a_zero_cube() {
        let (tiles_per_axis, cube_dim, bytes) = CubeTilePacking::pack_bit_cubes(&[], 4, 255);
        assert_eq!((tiles_per_axis, cube_dim), (0, 0));
        assert!(bytes.is_empty());
        let (tiles_per_axis, cube_dim, bytes) = CubeTilePacking::pack_u16_value_cubes(&[], 4);
        assert_eq!((tiles_per_axis, cube_dim), (0, 0));
        assert!(bytes.is_empty());
    }

    /// The 16-bit payload pack: every tile's every cell lands at `slot_origin + local`, stored
    /// low byte first, and cells outside any tile stay zero — checked against an independent
    /// per-cell oracle that re-derives each destination texel from the slot → origin map.
    #[test]
    fn pack_u16_lands_each_cell_at_its_slot_origin_little_endian() {
        // Five edge-2 tiles: cell (x,y,z) of slot s carries a value unique to (s,x,y,z).
        let tile_edge = 2;
        let tile_count = 5;
        let value_at = |slot: u32, x: u32, y: u32, z: u32| -> u16 {
            (slot as u16) << 12 | (z as u16) << 8 | (y as u16) << 4 | x as u16 | 0x0801
        };
        let tiles: Vec<ValueCube<u16>> = (0..tile_count)
            .map(|slot| {
                let mut cube = ValueCube::new_filled(tile_edge, 0u16);
                for z in 0..tile_edge {
                    for y in 0..tile_edge {
                        for x in 0..tile_edge {
                            cube.set(x, y, z, value_at(slot, x, y, z));
                        }
                    }
                }
                cube
            })
            .collect();

        let (tiles_per_axis, cube_dim, bytes) =
            CubeTilePacking::pack_u16_value_cubes(&tiles, tile_edge);
        assert_eq!(tiles_per_axis, 2, "ceil(cbrt 5) = 2");
        assert_eq!(cube_dim, 4, "2 tiles/axis × edge 2");
        let cube_dim = cube_dim as usize;
        assert_eq!(bytes.len(), 2 * cube_dim.pow(3), "two bytes per texel");

        // The oracle: an independent per-cell scatter, values written as LE pairs.
        let packing = CubeTilePacking::for_tile_count(tile_count as usize, tile_edge);
        let mut expected = vec![0u8; 2 * cube_dim.pow(3)];
        for slot in 0..tile_count {
            let origin = packing.tile_origin_cells(slot, tile_edge);
            for z in 0..tile_edge as usize {
                for y in 0..tile_edge as usize {
                    for x in 0..tile_edge as usize {
                        let cell = ((origin[2] + z) * cube_dim + origin[1] + y) * cube_dim
                            + origin[0]
                            + x;
                        let value = value_at(slot, x as u32, y as u32, z as u32);
                        expected[cell * 2..cell * 2 + 2].copy_from_slice(&value.to_le_bytes());
                    }
                }
            }
        }
        assert_eq!(bytes, expected);
    }

    /// The two texel formats share ONE geometry: for the same slot count and edge, the cells a
    /// bit tile's set bits land in are exactly the cells the payload pack writes a tile's
    /// values into (byte index `i` ⇔ texel index `i`) — the identity a paired occupancy/payload
    /// pool depends on.
    #[test]
    fn bit_and_u16_packs_agree_on_where_each_slot_lands() {
        let tile_edge = 4;
        let tile_count = 9; // ceil(cbrt 9) = 3 → a partial shell, the interesting case
        let bit_tiles: Vec<BitCube> = (0..tile_count)
            .map(|_| {
                let mut cube = BitCube::empty(tile_edge);
                for z in 0..tile_edge {
                    for y in 0..tile_edge {
                        cube.set_x_run(y, z, 0, tile_edge - 1);
                    }
                }
                cube
            })
            .collect();
        let value_tiles: Vec<ValueCube<u16>> = (0..tile_count)
            .map(|_| ValueCube::new_filled(tile_edge, 0xABCD))
            .collect();

        let (bit_axis, bit_dim, occupancy) =
            CubeTilePacking::pack_bit_cubes(&bit_tiles, tile_edge, 255);
        let (value_axis, value_dim, payload) =
            CubeTilePacking::pack_u16_value_cubes(&value_tiles, tile_edge);
        assert_eq!((bit_axis, bit_dim), (value_axis, value_dim));
        for cell in 0..(bit_dim as usize).pow(3) {
            let occupied = occupancy[cell] == 255;
            let value = u16::from_le_bytes([payload[cell * 2], payload[cell * 2 + 1]]);
            assert_eq!(
                occupied,
                value == 0xABCD,
                "cell {cell}: the two pools must agree on which cells belong to a tile"
            );
        }
    }
}
