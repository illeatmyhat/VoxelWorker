//! Whole-region + cacheless streaming over resident two-layer chunks (the dense oracle, the export/measure streams).


use rayon::prelude::*;
use substrate::interval::DisjointIntervalSet;

use voxel_core::core_geom::CHUNK_BLOCKS;
use document::scene::Scene;
use voxel_core::voxel::{RecentreVoxels, Voxel};
// Used only by the compile-gated whole-region oracles (`resolve_region_two_layer` under
// `oracle`, `expand_resident_chunks_into_grid` under `test-support`); gated so the plain
// production build does not carry an unused import.
#[cfg(any(test, feature = "oracle", feature = "test-support"))]
use voxel_core::voxel::VoxelGrid;
#[cfg(any(test, feature = "test-support"))]
use std::sync::Arc;

#[allow(unused_imports)]
use super::*;

/// Stream a whole scene's two-layer chunks back to one recentred [`VoxelGrid`], in the
/// EXACT frame the dense [`Store::resolve_region`](crate::store::Store::resolve_region)
/// assembles — the parity gate's "two-layer round-trip" (ADR 0010 parity (a)). Builds
/// each covering chunk via the capability, expands it (coarse fast-fill + boundary
/// per-voxel), and rebases by `chunk_min_voxels − recentre` so the occupied SET matches
/// the dense path bit-for-bit (position + block id).
///
/// Returns `None` when the capability is OFF (the caller stays on the dense path) or the
/// scene has no covering chunk range (a Part-only scene — handled by the dense path).
///
/// **Oracle — compile-gated.** This streams a whole-region dense [`VoxelGrid`] purely so
/// the parity gate can compare it against `Store::resolve_region`; no runtime display
/// path assembles a whole-region grid. It is excluded from production builds behind the
/// `oracle` feature (tests reach it via `cfg(test)`), so a dense whole-region resolve is
/// a compile error in production — see the proof chapter's "Oracles" section
/// (`docs/architecture/05-proof.md`).
#[cfg(any(test, feature = "oracle"))]
pub fn resolve_region_two_layer(
    store: &TwoLayerStore,
    scene: &Scene,
    voxels_per_block: u32,
    lod: u32,
) -> Option<VoxelGrid> {
    if !store.is_enabled() {
        return None;
    }
    debug_assert_eq!(lod, 0, "E2 only resolves full resolution (lod 0)");

    let region_dimensions = scene.placed_region_dimensions(voxels_per_block);
    let mut output = VoxelGrid::new(region_dimensions);
    // ADR 0008: carry the recentre so consumers decode world→index without re-deriving it
    // — the same value the dense `resolve_region` stamps (the parity gate compares it).
    let recentre = scene.recentre_voxels_for_resolve(voxels_per_block);
    output.recentre_voxels = recentre.voxels();

    let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(voxels_per_block) else {
        return Some(output); // No composite extent (Part-only): empty region.
    };

    // Unwrap the frame at the per-chunk rebase arithmetic below.
    let recentre_voxels = recentre.voxels();
    let chunk_extent_voxels = (CHUNK_BLOCKS * voxels_per_block.max(1)) as i64;
    // Visit chunks in the SAME z,y,x order the dense store assembles them in, so the
    // emitted voxel order matches too (the multiset compare is order-independent, but
    // keeping the order identical makes a future ordered compare cheap).
    for chunk_z in min_chunk[2]..=max_chunk[2] {
        for chunk_y in min_chunk[1]..=max_chunk[1] {
            for chunk_x in min_chunk[0]..=max_chunk[0] {
                let chunk_coord = [chunk_x, chunk_y, chunk_z];
                let chunk = store
                    .build_chunk(chunk_coord, scene, voxels_per_block, lod)
                    .expect("capability is enabled");
                // Rebase chunk-local indices into the recentred frame: a voxel at
                // chunk-local index `l` has absolute index `chunk_min + l`, and the
                // recentred frame subtracts `recentre` — so the offset is
                // `chunk_min − recentre` (i64, before the i32 downcast). This is exactly
                // `resolve_chunk_rebased`'s rebase with floating origin = recentre.
                let index_offset = [
                    chunk_x as i64 * chunk_extent_voxels - recentre_voxels[0],
                    chunk_y as i64 * chunk_extent_voxels - recentre_voxels[1],
                    chunk_z as i64 * chunk_extent_voxels - recentre_voxels[2],
                ];
                chunk.expand_occupancy_into(&mut output.occupied, index_offset);
            }
        }
    }

    Some(output)
}

/// Expand an already-resident two-layer chunk set (ADR 0010 E5 — the
/// [`TwoLayerResidentCache`] display path) into one recentred [`VoxelGrid`], in the
/// EXACT frame the retired dense `Store::resolve_region` assembled. Unlike
/// [`resolve_region_two_layer`] (which re-classifies each chunk from the scene via a
/// stateless [`TwoLayerStore`]), this reuses the caller's ALREADY-BUILT resident chunks.
///
/// `chunks` is `(absolute_chunk_coord, Arc<TwoLayerChunk>)` per covering chunk (the
/// [`TwoLayerResidentCache::resident_two_layer_chunks`] output); `region_dimensions` is
/// the composite voxel extent ([`Scene::placed_region_dimensions`]); `recentre`
/// is the composite recentre frame (ADR 0008). The occupied SET is bit-identical to
/// [`resolve_region_two_layer`]'s (the E2 round-trip parity gate proves the shared expand
/// path).
///
/// **ADR 0011 G5 — demoted to a TEST-ONLY oracle (`#[cfg(test)]`).** With the fog
/// `VoxelGrid` stream retired, no runtime path expands the resident set into a dense grid;
/// this survives only so parity tests (the brick-vs-densify fog nets, the render-frame
/// coordinate guard, the grid-overlay invalidation test) can materialise the same occupancy
/// the retired stream produced, and assert against it. A runtime caller reappearing here would
/// be the O(volume) densify coming back.
///
/// It ships only in TEST builds: `cfg(test)` inside this crate, and the `test-support`
/// feature for downstream crates' tests (the app crate's `AppCore` grid-overlay test
/// reaches it across the boundary).
#[cfg(any(test, feature = "test-support"))]
pub fn expand_resident_chunks_into_grid(
    chunks: &[([i32; 3], Arc<TwoLayerChunk>)],
    region_dimensions: [u32; 3],
    recentre: RecentreVoxels,
    voxels_per_block: u32,
) -> VoxelGrid {
    let mut output = VoxelGrid::new(region_dimensions);
    // Unwrap at the chunk-rebase arithmetic (the index offset below) and the grid's carried
    // raw frame field.
    let recentre_voxels = recentre.voxels();
    output.recentre_voxels = recentre_voxels;
    let chunk_extent_voxels = (CHUNK_BLOCKS * voxels_per_block.max(1)) as i64;
    for (chunk_coord, chunk) in chunks {
        // Rebase chunk-local indices into the recentred frame: a voxel at chunk-local
        // index `l` sits at absolute `chunk_min + l`, and the recentred frame subtracts
        // the recentre — the SAME `chunk_min − recentre` offset `resolve_region_two_layer`
        // applies (ADR 0008).
        let index_offset = [
            chunk_coord[0] as i64 * chunk_extent_voxels - recentre_voxels[0],
            chunk_coord[1] as i64 * chunk_extent_voxels - recentre_voxels[1],
            chunk_coord[2] as i64 * chunk_extent_voxels - recentre_voxels[2],
        ];
        chunk.expand_occupancy_into(&mut output.occupied, index_offset);
    }
    output
}

// ===== ADR 0010 E4 — the cacheless STREAMING exact sinks =====================
//
// The display sink (E3) CACHES the two layers; the exact sinks (`.vox` export and
// the diameter/widest-run query) read the SAME evaluator region-scoped, cacheless,
// streaming (ADR 0010 Decision 3 — "many sinks by policy"). They drive the E2
// classifier ([`classify_chunk_block`]) block-by-block and NEVER assemble a dense
// whole-region `VoxelGrid`: a coarse-solid block is a fast `d³` fill (export) or an
// analytic run contribution (query: `run += d`, no per-voxel expansion); a boundary
// block is per-voxel field eval. This is why the `.vox` 6M whole-region cap
// dissolves on the export path — no dense interior is ever materialised.

/// Build ONE covering chunk in the recentred frame and stream its occupancy into a
/// FRESH `Vec<Voxel>` (coarse `d³` fast-fill + boundary per-voxel), in the EXACT
/// frame the dense [`Store::resolve_region`](crate::store::Store::resolve_region)
/// assembles. Returns `None` when the capability is OFF.
///
/// This is the per-chunk streaming primitive both exact sinks share: the export
/// buckets each chunk's `Vec` then DROPS it (never holding the whole region), and a
/// caller wanting the whole occupancy just concatenates. No dense whole-region grid
/// is ever allocated — the chunk buffer is the only transient, bounded to one
/// chunk's surface + (for export) its coarse interior.
pub(crate) fn stream_chunk_recentred(
    store: &TwoLayerStore,
    scene: &Scene,
    chunk_coord: [i32; 3],
    voxels_per_block: u32,
    recentre: RecentreVoxels,
) -> Option<Vec<Voxel>> {
    let chunk = store.build_chunk(chunk_coord, scene, voxels_per_block, 0)?;
    let chunk_extent_voxels = (CHUNK_BLOCKS * voxels_per_block.max(1)) as i64;
    // Rebase chunk-local indices into the recentred frame (mirrors
    // `resolve_region_two_layer` / `resolve_chunk_rebased`): a chunk-local voxel `l`
    // has absolute index `chunk_min + l`, and the recentred frame subtracts the
    // recentre, so the offset is `chunk_min − recentre` (i64, before the i32 downcast).
    // Unwrap at this arithmetic.
    let recentre_voxels = recentre.voxels();
    let index_offset = [
        chunk_coord[0] as i64 * chunk_extent_voxels - recentre_voxels[0],
        chunk_coord[1] as i64 * chunk_extent_voxels - recentre_voxels[1],
        chunk_coord[2] as i64 * chunk_extent_voxels - recentre_voxels[2],
    ];
    let mut output = Vec::new();
    chunk.expand_occupancy_into(&mut output, index_offset);
    Some(output)
}

/// **Cacheless `.vox` streaming source (ADR 0010 E4).** Stream the scene's exact
/// occupancy region-scoped, ONE covering chunk at a time, in the recentred frame the
/// dense path produces, invoking `sink` with each chunk's freshly-expanded
/// `Vec<Voxel>` (coarse `d³` fast-fill + boundary per-voxel) before dropping it. The
/// caller's `sink` buckets each chunk into the `.vox` model set (so no whole-region
/// dense grid is ever assembled — the 6M whole-region cap dissolves on this path).
///
/// Returns the region's voxel `dimensions` (the SAME value
/// [`Scene::placed_region_dimensions`](document::scene::Scene::placed_region_dimensions)
/// produces — the `.vox` tiling/decode frame). Returns `None` when the capability is
/// OFF (the caller falls back to the dense path).
pub fn stream_vox_occupancy<Sink: FnMut(Vec<Voxel>)>(
    store: &TwoLayerStore,
    scene: &Scene,
    voxels_per_block: u32,
    mut sink: Sink,
) -> Option<[u32; 3]> {
    if !store.is_enabled() {
        return None;
    }
    let region_dimensions = scene.placed_region_dimensions(voxels_per_block);
    // Carry the frame newtype through the per-chunk stream; it is unwrapped inside
    // `stream_chunk_recentred`'s per-chunk rebase, never at this return.
    let recentre = scene.recentre_voxels_for_resolve(voxels_per_block);
    let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(voxels_per_block) else {
        // No composite extent (Part-only): an empty occupancy is still a valid export.
        return Some(region_dimensions);
    };
    for chunk_z in min_chunk[2]..=max_chunk[2] {
        for chunk_y in min_chunk[1]..=max_chunk[1] {
            for chunk_x in min_chunk[0]..=max_chunk[0] {
                if let Some(chunk_voxels) = stream_chunk_recentred(
                    store,
                    scene,
                    [chunk_x, chunk_y, chunk_z],
                    voxels_per_block,
                    recentre,
                ) {
                    sink(chunk_voxels);
                }
            }
        }
    }
    Some(region_dimensions)
}

/// **Cacheless diameter / widest-run query (ADR 0010 E4).** Compute the widest
/// occupied run in the layer band `[band_min, band_max]` (Z-slices, Z-up) by
/// streaming the classifier block-by-block — accounting a **coarse-solid block
/// ANALYTICALLY** (a fully-solid block sets a contiguous `density`-long X span in
/// every `(y, z)` row it covers, with NO per-voxel expansion) and a boundary block
/// per-voxel. Returns the SAME value
/// [`Store::widest_run_in_band`](crate::store::Store::widest_run_in_band) /
/// [`VoxelGrid::widest_run_in_band`](voxel_core::voxel::VoxelGrid::widest_run_in_band)
/// returns for the assembled region, but never assembles a dense grid.
///
/// Returns `None` when the capability is OFF (the caller falls back to the dense
/// path).
///
/// ## Frame / decode (identical to the dense readout, ADR 0008)
///
/// The shared per-`(y, z)` occupancy rows are keyed by the GLOBAL X index the dense
/// [`VoxelGrid::widest_run_in_band`](voxel_core::voxel::VoxelGrid::widest_run_in_band) computes
/// (`i = round(world_x + floor(grid_x/2) − 0.5)`). A coarse-solid block is stamped
/// at the SAME indices its per-voxel expansion would land — the recentred chunk-local
/// voxel index `chunk_min + block_low + local − recentre`, whose `world = index + 0.5`
/// decodes back to `index − region_low = index + floor(dim/2)` — so the analytic span
/// is bit-identical to expanding the block and scanning. A boundary block's per-voxel
/// fill uses the identical decode, so a run crossing a coarse↔boundary seam is one
/// contiguous span in the shared bitset.
pub fn streamed_widest_run_in_band(
    store: &TwoLayerStore,
    scene: &Scene,
    voxels_per_block: u32,
    band_min: u32,
    band_max: u32,
) -> Option<u32> {
    if !store.is_enabled() {
        return None;
    }
    let [grid_x, grid_y, grid_z] = scene.placed_region_dimensions(voxels_per_block);
    if grid_x == 0 || grid_y == 0 || grid_z == 0 {
        return Some(0);
    }
    // Unwrap the frame at this cacheless query's per-block rebase arithmetic.
    let recentre_voxels = scene.recentre_voxels_for_resolve(voxels_per_block).voxels();
    let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(voxels_per_block) else {
        return Some(0);
    };

    let width = grid_x as usize;
    // The dense decode: `idx = round(world + floor(dim/2) − 0.5)`. Because every
    // streamed voxel index `n` decodes `world = n + 0.5` ⇒ `idx = n + floor(dim/2)`,
    // the recentred index maps to the global grid index by ADDING `floor(dim/2)`.
    let half = [
        (grid_x / 2) as i64,
        (grid_y / 2) as i64,
        (grid_z / 2) as i64,
    ];
    let density = voxels_per_block.max(1) as i64;
    let chunk_extent_voxels = CHUNK_BLOCKS as i64 * density;
    let grid_y_i = grid_y as i64;
    let band_min_i = band_min as i64;
    let band_max_i = band_max as i64;

    // ADR 0010 E5 — BLOCK-ROW DEDUP (the O(volume)→O(total blocks) diameter fix).
    //
    // The widest run is folded one **(chunk_z, chunk_y) band at a time**: every global
    // (z, y) row maps to EXACTLY ONE band (the recentred z/y ranges partition disjointly),
    // so a band is complete once its `chunk_x` sweep finishes. Bands are independent, so the
    // fold runs across them in PARALLEL with a deterministic `max` reduction — live state is
    // bounded per band, never a region-wide bitset (the prior `Vec<bool>` per (z,y) row was
    // O(volume): a 8000×800×800 solid was ≈5 GB and OOM-hung startup).
    //
    // The fatal remaining cost was per-VOXEL-row work WITHIN a band: a coarse-solid block
    // stamped its `[x0, x0+d)` span into every one of the `d²` voxel rows it covers, so a
    // solid cube was O(voxels) (edge² rows × the blocks per row) — 64M rows at 8000³, a 127s
    // main-thread freeze. THE DEDUP: every `d²` voxel row under the SAME BLOCK-ROW (same
    // block_y, block_z) receives IDENTICAL coarse spans, so each block-row's coarse span set
    // is accumulated ONCE — block-granular (one `[x0, x0+d)` per coarse block, ×density
    // widths) — and its widest contiguous run is counted once for the whole block-row (valid
    // iff its z-range meets the band and its y-range meets `[0, grid_y)`, exactly the old
    // per-voxel band/grid clip). Per-voxel work survives ONLY for the voxel rows a BOUNDARY
    // (microblock) block actually intersects: those are expanded per voxel and then MERGED
    // with their block-row's coarse spans (so a run crossing a coarse↔boundary seam is one
    // contiguous span). A solid cube is then O(total blocks) — the chunk-enumeration floor —
    // not O(voxels).
    //
    // Each accumulator holds occupied X as a sorted, disjoint, non-touching INTERVAL list
    // (half-open `[lo, hi)`); the widest contiguous run in a row is `max(hi − lo)`.
    // Byte-identical to the dense `widest_run_in_band` oracle (the
    // `streamed_widest_run_matches_dense_*` parity gate) — the same global X indices,
    // stitched the same way across seams, folded to the same max.
    const BLOCK_ROWS: usize = CHUNK_BLOCKS as usize * CHUNK_BLOCKS as usize;

    let mut bands: Vec<(i32, i32)> = Vec::new();
    for chunk_z in min_chunk[2]..=max_chunk[2] {
        for chunk_y in min_chunk[1]..=max_chunk[1] {
            bands.push((chunk_z, chunk_y));
        }
    }

    // The widest occupied run within ONE (chunk_z, chunk_y) band. Pure over the shared
    // immutable inputs (store/scene/frame scalars), so it parallelises across bands.
    let band_widest = |(chunk_z, chunk_y): (i32, i32)| -> u32 {
        // The band's global-row origins (`to_global[1..=2]`, constant over `chunk_x`).
        let z_base = chunk_z as i64 * chunk_extent_voxels - recentre_voxels[2] + half[2];
        let y_base = chunk_y as i64 * chunk_extent_voxels - recentre_voxels[1] + half[1];

        // Per block-row (index `block_z * CHUNK_BLOCKS + block_y`) coarse X spans — the ONE
        // span set shared by every voxel row the block-row covers. Boundary cuboid spans are
        // expanded per voxel row, keyed by their global `(z, y)` (sparse — surface rows only),
        // and merged with the block-row's coarse spans at fold time.
        let mut coarse_block_runs: [DisjointIntervalSet; BLOCK_ROWS] =
            std::array::from_fn(|_| DisjointIntervalSet::new());
        let mut boundary_rows: std::collections::HashMap<(i64, i64), DisjointIntervalSet> =
            std::collections::HashMap::new();

        for chunk_x in min_chunk[0]..=max_chunk[0] {
            let chunk_coord = [chunk_x, chunk_y, chunk_z];
            let Some(chunk) = store.build_chunk(chunk_coord, scene, voxels_per_block, 0) else {
                continue;
            };
            // The recentred→global X origin: global_x = chunk_min_x + block_low_x + local
            // − recentre + half. Spans arrive in ascending X (chunk_x, block_x, local all
            // increase), so `DisjointIntervalSet::insert`'s append fast path coalesces a
            // solid row in O(1).
            let to_global_x = chunk_x as i64 * chunk_extent_voxels - recentre_voxels[0] + half[0];
            for block_z in 0..CHUNK_BLOCKS {
                for block_y in 0..CHUNK_BLOCKS {
                    let block_row =
                        block_z as usize * CHUNK_BLOCKS as usize + block_y as usize;
                    for block_x in 0..CHUNK_BLOCKS {
                        let block = [block_x, block_y, block_z];
                        let block_low_x = block_x as i64 * density;
                        if chunk.coarse_block(block).is_some() {
                            // ANALYTIC / block-granular: the contiguous span `[x0, x0+d)` is
                            // identical for every voxel row this coarse block covers — recorded
                            // ONCE for the whole block-row (no per-voxel expansion).
                            let x0 = to_global_x + block_low_x;
                            let lo = x0.max(0);
                            let hi = (x0 + density).min(width as i64);
                            if lo < hi {
                                coarse_block_runs[block_row].insert(lo, hi);
                            }
                        } else if let Some(geometry) = chunk.microblocks.get(&block) {
                            // BOUNDARY: per-cuboid spans expanded into every voxel row they
                            // cover, band/grid/width-clipped exactly as the dense readout.
                            for cuboid in &geometry.cuboids {
                                for local_z in cuboid.min[2]..=cuboid.max[2] {
                                    let gz = z_base + block_z as i64 * density + local_z as i64;
                                    if gz < band_min_i || gz > band_max_i {
                                        continue;
                                    }
                                    for local_y in cuboid.min[1]..=cuboid.max[1] {
                                        let gy =
                                            y_base + block_y as i64 * density + local_y as i64;
                                        if gy < 0 || gy >= grid_y_i {
                                            continue;
                                        }
                                        let x_lo = (to_global_x + block_low_x
                                            + cuboid.min[0] as i64)
                                            .max(0);
                                        let x_hi = (to_global_x
                                            + block_low_x
                                            + cuboid.max[0] as i64
                                            + 1)
                                            .min(width as i64);
                                        if x_lo >= x_hi {
                                            continue;
                                        }
                                        boundary_rows
                                            .entry((gz, gy))
                                            .or_default()
                                            .insert(x_lo, x_hi);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut widest = 0u32;
        // (1) All-coarse contribution: each block-row with coarse runs AND ≥1 valid voxel row
        // (its z-range meets the band, its y-range meets `[0, grid_y)`) carries that widest
        // coarse run in every one of its voxel rows.
        for block_z in 0..CHUNK_BLOCKS {
            for block_y in 0..CHUNK_BLOCKS {
                let block_row = block_z as usize * CHUNK_BLOCKS as usize + block_y as usize;
                let runs = &coarse_block_runs[block_row];
                if runs.is_empty() {
                    continue;
                }
                let z_lo = z_base + block_z as i64 * density;
                if z_lo + density - 1 < band_min_i || z_lo > band_max_i {
                    continue;
                }
                let y_lo = y_base + block_y as i64 * density;
                if y_lo + density - 1 < 0 || y_lo >= grid_y_i {
                    continue;
                }
                widest = widest.max(runs.widest_span() as u32);
            }
        }
        // (2) Boundary rows: merge each with its block-row's coarse spans, fold the widest —
        // a run crossing a coarse↔boundary seam is one contiguous span. The global (z, y) key
        // recovers its block-row: `block_z = (gz − z_base) / d`, `block_y = (gy − y_base) / d`.
        for (&(gz, gy), spans) in &boundary_rows {
            let block_z = (gz - z_base) / density;
            let block_y = (gy - y_base) / density;
            let block_row = block_z as usize * CHUNK_BLOCKS as usize + block_y as usize;
            let mut merged = coarse_block_runs[block_row].clone();
            for &(lo, hi) in spans.intervals() {
                merged.insert(lo, hi);
            }
            widest = widest.max(merged.widest_span() as u32);
        }
        widest
    };

    let widest = bands.into_par_iter().map(band_widest).max().unwrap_or(0);
    Some(widest)
}

