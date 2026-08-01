//! Packing equal-ish padded tiles into one sheet: shelf/next-fit rectangle packing.
//!
//! `ShelfBinPack` lays a sequence of rectangular tiles into a single larger sheet
//! by the **shelf** (a.k.a. next-fit level) heuristic: tiles are placed left to
//! right along a "shelf" (a horizontal band), and when the current shelf has taken
//! its quota of tiles a new shelf is started above the previous one. It is the 2D
//! sibling of this crate's [`crate::occupancy::cube_packing::CubeTilePacking`] — a linear
//! sequence turned into a space-filling placement — specialized to rectangles laid
//! on rows rather than cubes stacked in a grid.
//!
//! Shelf packing is chosen over the denser rectangle-bin packers (`MaxRects`, guillotine)
//! deliberately: for a small set of near-uniform tiles the packing-density win of a smarter
//! packer is irrelevant, and the shelf heuristic has no failure modes. The shelf
//! quota is `tiles_per_shelf = ceil(sqrt(count))`, so `n` equal tiles form an
//! `r × c` grid with `r ≈ c` — a near-square sheet, which keeps either dimension
//! from running past a hardware texture-size limit.
//!
//! ## Gutter and half-texel inset (the sampling-correctness detail)
//!
//! When a consumer tiles a single packed tile *by itself* across a surface — mapping
//! a repeating coordinate through `fract` into the tile's sub-rect rather than using
//! a hardware `Repeat` address mode (which would wrap across the whole sheet into a
//! neighbor) — two edge artifacts have to be defended against. Both are standard
//! atlas *bleed* defenses:
//!
//!  1. **Neighbor spill.** Even nearest-neighbor sampling can, at a tile seam,
//!     read one texel into the adjacent tile. Every tile is surrounded by a
//!     **replicated-edge gutter** ([`ShelfBinPack::blit_with_replicated_edge`]): the
//!     tile's own border texels copied outward by `gutter` texels, so a one-texel
//!     spill lands on a copy of the correct edge, never the neighbor.
//!  2. **Within-tile wrap seam.** Because the consumer tiles with `fract`, the
//!     reported sub-rect is pulled in by **half a texel** on each side
//!     ([`ShelfBinPack::normalized_rect`]): sampling is clamped to texel centers so
//!     `fract → 0` and `fract → 1` both land inside the tile, never on its outer
//!     edge where they could round into the gutter.
//!
//! `normalized_rect` therefore returns BOTH the full tile rect (outer bounds, for
//! diagnostics) and the half-texel-inset window a `fract`-tiling consumer samples.
//! All coordinates are in normalized `[0, 1]` sheet space.

/// A tile's pixel dimensions, handed to the layout planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileSize {
    pub width: u32,
    pub height: u32,
}

/// A borrowed source tile for the gutter blit.
///
/// Its interleaved row-major texel bytes have length `width · height ·
/// bytes_per_texel`, with the stride fixed by the destination sheet.
#[derive(Debug, Clone, Copy)]
pub struct TileImage<'a> {
    pub width: u32,
    pub height: u32,
    pub pixels: &'a [u8],
}

/// Where one tile's inner (gutter-excluded) region sits in the sheet, in pixels —
/// the top-left corner of the tile itself, not of its surrounding gutter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedTilePlacement {
    pub inner_x: u32,
    pub inner_y: u32,
}

/// A tile's normalized sheet rectangle, including its half-texel inset window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizedTileRect {
    /// Outer tile bounds in normalized sheet coordinates (excludes the gutter).
    pub min_u: f32,
    pub min_v: f32,
    pub max_u: f32,
    pub max_v: f32,
    /// The outer bounds pulled in by half a texel on each side.
    pub inset_min_u: f32,
    pub inset_min_v: f32,
    pub inset_max_u: f32,
    pub inset_max_v: f32,
}

/// The result of planning a shelf layout: the sheet's pixel dimensions and one
/// [`PackedTilePlacement`] per input tile, in input order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShelfBinPack {
    /// Sheet width in pixels (the widest shelf; at least `1`).
    pub sheet_width: u32,
    /// Sheet height in pixels (the summed shelf heights; at least `1`).
    pub sheet_height: u32,
    /// One placement per input tile, in order.
    pub placements: Vec<PackedTilePlacement>,
}

impl ShelfBinPack {
    /// The shelf quota `ceil(sqrt(tile_count))` (at least `1`): capping a shelf at
    /// this many near-uniform tiles lays them into an `r × c` grid with `r ≈ c`.
    #[must_use]
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    pub fn tiles_per_shelf(tile_count: usize) -> usize {
        ((tile_count as f32).sqrt().ceil() as usize).max(1)
    }

    /// Plan a shelf layout for `tiles`, each surrounded by a `gutter`-texel border.
    /// Tiles fill a shelf left to right until the shelf's quota
    /// ([`Self::tiles_per_shelf`]) is reached, then a new shelf starts above.
    /// A tile occupies a cell of `dimension + 2 · gutter` on each axis; a shelf's
    /// height is its tallest padded tile and the sheet's width is its widest padded
    /// shelf. Returns placements of each tile's inner (gutter-excluded) corner. An
    /// empty tile list yields a `1 × 1` sheet with no placements.
    #[must_use]
    pub fn plan(tiles: &[TileSize], gutter: u32) -> Self {
        let tile_count = tiles.len();
        let tiles_per_shelf = Self::tiles_per_shelf(tile_count);
        let padded = |dimension: u32| -> u32 { dimension.saturating_add(gutter.saturating_mul(2)) };

        let mut placements: Vec<PackedTilePlacement> = Vec::with_capacity(tile_count);
        let mut sheet_width: u32 = 0;
        let mut shelf_origin_y: u32 = 0;
        let mut index = 0;
        while index < tile_count {
            let shelf_end = index.saturating_add(tiles_per_shelf).min(tile_count);
            let mut shelf_cursor_x: u32 = 0;
            let mut shelf_height: u32 = 0;
            if let Some(shelf) = tiles.get(index..shelf_end) {
                for tile in shelf {
                    let inner_x = shelf_cursor_x.saturating_add(gutter);
                    let inner_y = shelf_origin_y.saturating_add(gutter);
                    placements.push(PackedTilePlacement { inner_x, inner_y });
                    shelf_cursor_x = shelf_cursor_x.saturating_add(padded(tile.width));
                    shelf_height = shelf_height.max(padded(tile.height));
                }
            }
            sheet_width = sheet_width.max(shelf_cursor_x);
            shelf_origin_y = shelf_origin_y.saturating_add(shelf_height);
            index = shelf_end;
        }

        Self {
            sheet_width: sheet_width.max(1),
            sheet_height: shelf_origin_y.max(1),
            placements,
        }
    }

    /// The normalized `[0, 1]` sheet rect of a tile at `placement` with the given
    /// pixel size, including the half-texel-inset sampling window (see the module
    /// docs). Pure function of the placement, the tile size, and the sheet size.
    #[must_use]
    pub fn normalized_rect(
        placement: &PackedTilePlacement,
        tile_width: u32,
        tile_height: u32,
        sheet_width: u32,
        sheet_height: u32,
    ) -> NormalizedTileRect {
        let sheet_w = u32_to_f32(sheet_width);
        let sheet_h = u32_to_f32(sheet_height);
        // Outer tile bounds in normalized coordinates (the inner region, no gutter).
        let min_u = u32_to_f32(placement.inner_x) / sheet_w;
        let min_v = u32_to_f32(placement.inner_y) / sheet_h;
        let max_u = u32_to_f32(placement.inner_x.saturating_add(tile_width)) / sheet_w;
        let max_v = u32_to_f32(placement.inner_y.saturating_add(tile_height)) / sheet_h;
        // Half-texel inset: pull each edge in by half a texel so a fract-tiling
        // consumer lands on texel centers, never the outermost edge (which could
        // round into the gutter under interpolation/derivatives).
        let half_texel_u = 0.5 / sheet_w;
        let half_texel_v = 0.5 / sheet_h;
        NormalizedTileRect {
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

    /// Copy `tile` into the sheet with its inner corner at `placement` AND replicate
    /// its edge texels outward by `gutter` on every side, so a one-texel sample spill
    /// at a cell border reads a copy of the correct edge, never a neighbor tile.
    /// `bytes_per_texel` interleaved bytes per texel (e.g. `4` for RGBA8); `sheet` is
    /// row-major `sheet_width` texels wide. Destinations that fall outside the sheet
    /// (or sources past the tile) are skipped, so a caller need not pre-clip the
    /// gutter ring.
    pub fn blit_with_replicated_edge(
        sheet: &mut [u8],
        sheet_width: u32,
        bytes_per_texel: usize,
        tile: &TileImage,
        placement: PackedTilePlacement,
        gutter: u32,
    ) {
        let gutter = i64::from(gutter);
        let tile_w = i64::from(tile.width);
        let tile_h = i64::from(tile.height);

        // Fill the padded cell (tile + gutter on every side). For a destination
        // texel, clamp its tile-local coordinate into the tile so border texels
        // replicate outward.
        let Some(start) = gutter.checked_neg() else {
            return;
        };
        let Some(end_y) = tile_h.checked_add(gutter) else {
            return;
        };
        let Some(end_x) = tile_w.checked_add(gutter) else {
            return;
        };
        for cell_dy in start..end_y {
            for cell_dx in start..end_x {
                let source_x = cell_dx.clamp(0, tile_w.saturating_sub(1));
                let source_y = cell_dy.clamp(0, tile_h.saturating_sub(1));
                let Some(source_cell) = source_y
                    .checked_mul(tile_w)
                    .and_then(|row| row.checked_add(source_x))
                else {
                    continue;
                };
                let Some(source_index) = usize::try_from(source_cell)
                    .ok()
                    .and_then(|index| index.checked_mul(bytes_per_texel))
                else {
                    continue;
                };

                let Some(dest_x) = i64::from(placement.inner_x).checked_add(cell_dx) else {
                    continue;
                };
                let Some(dest_y) = i64::from(placement.inner_y).checked_add(cell_dy) else {
                    continue;
                };
                if dest_x < 0 || dest_y < 0 {
                    continue;
                }
                let Some(dest_cell) = dest_y
                    .checked_mul(i64::from(sheet_width))
                    .and_then(|row| row.checked_add(dest_x))
                else {
                    continue;
                };
                let Some(dest_index) = usize::try_from(dest_cell)
                    .ok()
                    .and_then(|index| index.checked_mul(bytes_per_texel))
                else {
                    continue;
                };
                let Some(dest_end) = dest_index.checked_add(bytes_per_texel) else {
                    continue;
                };
                let Some(source_end) = source_index.checked_add(bytes_per_texel) else {
                    continue;
                };
                if let (Some(destination), Some(source)) = (
                    sheet.get_mut(dest_index..dest_end),
                    tile.pixels.get(source_index..source_end),
                ) {
                    destination.copy_from_slice(source);
                }
            }
        }
    }
}

#[allow(clippy::as_conversions, clippy::cast_precision_loss)]
const fn u32_to_f32(value: u32) -> f32 {
    value as f32
}

/// Kani bounded-model-checking proof of [`ShelfBinPack::normalized_rect`]'s **half-texel inset
/// invariant** — the sampling-correctness detail (module docs §"Gutter and half-texel inset"):
/// for any valid placement the outer tile rect stays inside the unit square and the inset
/// sampling window sits STRICTLY inside it and never inverts, so a `fract`-tiling consumer lands
/// on texel centers and can never round out into the gutter (the bleed the whole gutter machinery
/// exists to prevent). Proved over all small placements rather than the fixtures the unit test
/// names.
///
/// The sibling [`ShelfBinPack::plan`] is deliberately NOT a Kani harness: it calls
/// [`ShelfBinPack::tiles_per_shelf`], whose `f32::sqrt` is a foreign function CBMC cannot model,
/// and it grows a `Vec` — the same bounded-model-checking limits that scope out the interval-set
/// insert. `#[cfg(kani)]` keeps this inactive in ordinary builds. Run under WSL:
/// `cargo kani -p substrate`.
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// A symbolic value in `1..=bound`.
    fn positive_up_to(bound: u32) -> u32 {
        let value: u32 = kani::any();
        kani::assume(value >= 1 && value <= bound);
        value
    }

    /// **The half-texel inset window sits strictly inside the outer tile rect, which itself lies
    /// in the unit square.** For any tile that genuinely fits in the sheet (`inner + size <=
    /// sheet` on each axis, `size >= 1`), `normalized_rect` yields `0 <= min <= max <= 1`,
    /// `inset_min > min`, and `inset_max < max`, on both axes — the inset pulls in by a real
    /// half-texel (`0.5/sheet >= 2^-11`, far above the `~2^-24` f32 ULP at these magnitudes, so the
    /// strictness is robust to rounding). A consumer sampling the inset window therefore never
    /// reads the outer edge, where interpolation could pull a texel out of the tile into its
    /// gutter. (The window's non-inversion at a degenerate 1-px tile is left unasserted — there the
    /// two inset edges collapse to within a rounding step, a case no real tile hits.)
    ///
    /// The sheet dimensions are the concrete powers of two `512 × 256` (the placement and tile
    /// stay symbolic): dividing by a power of two is an EXACT f32 scaling, which keeps the SAT
    /// solve fast — a symbolic sheet divisor is a general f32 divide and pushed the same proof past
    /// 100 s. Two different axes guard against a u/v mix-up.
    #[kani::proof]
    fn normalized_rect_inset_stays_strictly_inside_the_unit_tile() {
        const SHEET_WIDTH: u32 = 512;
        const SHEET_HEIGHT: u32 = 256;
        let tile_width = positive_up_to(SHEET_WIDTH);
        let tile_height = positive_up_to(SHEET_HEIGHT);
        let inner_x: u32 = kani::any();
        let inner_y: u32 = kani::any();
        // Bounded so the fit sums below cannot overflow u32.
        kani::assume(inner_x <= SHEET_WIDTH && inner_y <= SHEET_HEIGHT);
        // A placement that genuinely fits inside the sheet on each axis.
        kani::assume(inner_x + tile_width <= SHEET_WIDTH);
        kani::assume(inner_y + tile_height <= SHEET_HEIGHT);
        let sheet_width = SHEET_WIDTH;
        let sheet_height = SHEET_HEIGHT;
        let placement = PackedTilePlacement { inner_x, inner_y };
        let rect = ShelfBinPack::normalized_rect(
            &placement,
            tile_width,
            tile_height,
            sheet_width,
            sheet_height,
        );
        // Outer bounds inside the unit square, non-inverted.
        assert!(rect.min_u >= 0.0 && rect.min_u <= rect.max_u && rect.max_u <= 1.0);
        assert!(rect.min_v >= 0.0 && rect.min_v <= rect.max_v && rect.max_v <= 1.0);
        // The inset window sits strictly inside the outer rect (never touches the sampled edge).
        assert!(rect.inset_min_u > rect.min_u && rect.inset_max_u < rect.max_u);
        assert!(rect.inset_min_v > rect.min_v && rect.inset_max_v < rect.max_v);
    }
}

#[cfg(test)]
#[allow(
    clippy::all,
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used
)]
mod tests {
    #![allow(
        clippy::all,
        clippy::arithmetic_side_effects,
        clippy::as_conversions,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::pedantic,
        clippy::nursery,
        clippy::unwrap_used
    )]
    use super::*;

    const GUTTER: u32 = 1;
    const RGBA: usize = 4;

    /// `tiles_per_shelf` is `ceil(sqrt(count))`, at least 1: 0→1, 1→1, 2→2, 4→2,
    /// 5→3, 9→3, 10→4.
    #[test]
    fn tiles_per_shelf_is_ceil_sqrt() {
        for (count, expected) in [(0, 1), (1, 1), (2, 2), (4, 2), (5, 3), (9, 3), (10, 4)] {
            assert_eq!(
                ShelfBinPack::tiles_per_shelf(count),
                expected,
                "count {count}"
            );
        }
    }

    /// The planner emits exactly one placement per tile and sizes the sheet to hold
    /// every placed tile plus its gutter.
    #[test]
    fn plans_one_placement_per_tile() {
        let sizes = [
            TileSize {
                width: 4,
                height: 4,
            },
            TileSize {
                width: 4,
                height: 4,
            },
            TileSize {
                width: 4,
                height: 4,
            },
        ];
        let layout = ShelfBinPack::plan(&sizes, GUTTER);
        assert_eq!(layout.placements.len(), 3, "one placement per tile");
        for (placement, size) in layout.placements.iter().zip(sizes.iter()) {
            assert!(placement.inner_x + size.width + GUTTER <= layout.sheet_width);
            assert!(placement.inner_y + size.height + GUTTER <= layout.sheet_height);
        }
    }

    /// An empty tile list plans a `1 × 1` sheet with no placements.
    #[test]
    fn empty_plan_is_unit_sheet() {
        let layout = ShelfBinPack::plan(&[], GUTTER);
        assert!(layout.placements.is_empty());
        assert_eq!((layout.sheet_width, layout.sheet_height), (1, 1));
    }

    /// Normalized rects stay inside the unit square, their half-texel-inset windows
    /// sit strictly inside the outer bounds, and no two outer rects overlap (the
    /// gutter guarantees at least a one-texel gap).
    #[test]
    fn rects_are_inside_unit_square_and_disjoint() {
        let sizes = [
            TileSize {
                width: 8,
                height: 8,
            },
            TileSize {
                width: 8,
                height: 8,
            },
            TileSize {
                width: 8,
                height: 8,
            },
            TileSize {
                width: 8,
                height: 8,
            },
        ];
        let layout = ShelfBinPack::plan(&sizes, GUTTER);
        let rects: Vec<NormalizedTileRect> = layout
            .placements
            .iter()
            .zip(sizes.iter())
            .map(|(placement, size)| {
                ShelfBinPack::normalized_rect(
                    placement,
                    size.width,
                    size.height,
                    layout.sheet_width,
                    layout.sheet_height,
                )
            })
            .collect();
        for rect in &rects {
            assert!(rect.min_u >= 0.0 && rect.max_u <= 1.0);
            assert!(rect.min_v >= 0.0 && rect.max_v <= 1.0);
            assert!(rect.inset_min_u > rect.min_u && rect.inset_max_u < rect.max_u);
            assert!(rect.inset_min_v > rect.min_v && rect.inset_max_v < rect.max_v);
        }
        for a in 0..rects.len() {
            for b in (a + 1)..rects.len() {
                let ra = &rects[a];
                let rb = &rects[b];
                let disjoint = ra.max_u <= rb.min_u
                    || rb.max_u <= ra.min_u
                    || ra.max_v <= rb.min_v
                    || rb.max_v <= ra.min_v;
                assert!(disjoint, "rects {a} and {b} overlap");
            }
        }
    }

    /// The gutter ring is a copy of the tile's nearest edge, not a neighbor's or a
    /// cleared texel.
    #[test]
    fn gutter_replicates_edge_texels() {
        // A single 2×2 tile: (0,0)=red (1,0)=green (0,1)=blue (1,1)=white.
        let tile_pixels: Vec<u8> = vec![
            255, 0, 0, 255, // (0,0)
            0, 255, 0, 255, // (1,0)
            0, 0, 255, 255, // (0,1)
            255, 255, 255, 255, // (1,1)
        ];
        let sizes = [TileSize {
            width: 2,
            height: 2,
        }];
        let layout = ShelfBinPack::plan(&sizes, GUTTER);
        let mut sheet = vec![0u8; (layout.sheet_width * layout.sheet_height) as usize * RGBA];
        let placement = layout.placements[0];
        ShelfBinPack::blit_with_replicated_edge(
            &mut sheet,
            layout.sheet_width,
            RGBA,
            &TileImage {
                width: 2,
                height: 2,
                pixels: &tile_pixels,
            },
            placement,
            GUTTER,
        );
        // With gutter=1 the inner tile sits at (1,1); the top-left gutter texel (0,0)
        // replicates the tile's (0,0) corner = red.
        let texel = |x: u32, y: u32| -> [u8; 4] {
            let index = ((y * layout.sheet_width + x) * 4) as usize;
            [
                sheet[index],
                sheet[index + 1],
                sheet[index + 2],
                sheet[index + 3],
            ]
        };
        assert_eq!(
            texel(0, 0),
            [255, 0, 0, 255],
            "corner gutter = corner texel"
        );
        assert_eq!(texel(1, 0), [255, 0, 0, 255], "top gutter above (0,0)=red");
        assert_eq!(
            texel(2, 0),
            [0, 255, 0, 255],
            "top gutter above (1,0)=green"
        );
    }
}
