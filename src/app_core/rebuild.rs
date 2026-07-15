//! The headless geometry rebuild — [`AppCore::rebuild`] (ADR 0010 E5 / ADR 0011 G5).

use std::sync::Arc;

use voxel_core::core_geom::CHUNK_BLOCKS;
use voxel_core::voxel::chunk_extent_exceeds_bound;
use document::scene::Scene;
use evaluation::two_layer_store::TwoLayerChunk;

use super::{AppCore, RebuildOutcome, RebuildOutput};

impl AppCore {
    /// **The headless geometry rebuild (A2e).** Route the resolve through the
    /// per-chunk store with issue #27 S3 TARGETED invalidation: build the new
    /// scene's leaf spatial index, diff it against the last rebuild's to get the
    /// edit's dirty world-AABB, and evict ONLY the chunks that AABB touches (every
    /// other cached chunk stays resident). Fall back to a wholesale `clear()` when a
    /// precise AABB can't be computed — the first rebuild (no previous index), a
    /// density change, or a region-spanning Part edit (no localisable box, see
    /// `LeafSpatialIndex::edit_aabb_since`). The reassembled grid is byte-identical
    /// either way (the same chunks are re-resolved; untouched chunks are reused).
    ///
    /// Returns the region dimensions + recentre + the per-chunk render accessor, which
    /// BORROWS the store. The returned [`RebuildOutcome`] therefore borrows `self`, so the
    /// shell must consume it (build the cuboid mesh, refresh the brick field) BEFORE the
    /// next `&mut AppCore` call. A density whose single-chunk voxel capacity exceeds the bound
    /// is rejected WITHOUT touching the store, returning the offending count so the shell can
    /// surface the cap warning (AppCore never writes panel state).
    ///
    /// **ADR 0011 G5 — no dense grid is ever assembled.** A rebuild produces ONLY the sparse
    /// two-layer covering chunks + scalar metadata; the whole-region `VoxelGrid` expansion
    /// (ADR 0010's flagged per-edit densify debt) is GONE. The brick sink packs from the same
    /// `two_layer_chunks` the display meshes from, and the camera / scrubber read
    /// `region_dimensions` — nothing reads a dense occupancy array. The only surviving dense
    /// resolve is the test oracles.
    pub fn rebuild(&mut self, scene: &Scene, density: u32) -> RebuildOutcome {
        profiling::scope!("app_core_rebuild");
        // Issue #27 S2: the resolve is chunked + lazy, so the voxel bound is a
        // PER-CHUNK bound, not a whole-scene total. Only a pathological density
        // (one chunk's voxel capacity alone exceeds the bound) is rejected.
        if chunk_extent_exceeds_bound(density) {
            let chunk_extent = (CHUNK_BLOCKS * density.max(1)) as u64;
            let chunk_voxels = chunk_extent * chunk_extent * chunk_extent;
            return RebuildOutcome::DensityRejected {
                chunk_voxels_millions: chunk_voxels as f32 / 1_000_000.0,
            };
        }

        // ADR 0010 E5: S3 targeted invalidation on the TWO-LAYER resident cache (#54).
        // `invalidate_aabb` evicts the edit's dirty chunks (so the next
        // `resident_two_layer_chunks` re-classifies only them); `clear()` handles the
        // first build / density change / region-spanning Part edit where there is no
        // localisable AABB. A two-layer chunk is chunk-local-integer (ADR 0008), so —
        // unlike the retired dense store — a floating-origin SHIFT does NOT invalidate
        // the cache (the recentre is a pure index offset applied at expand/mesh time).
        let new_leaf_index = scene.build_leaf_spatial_index(density);
        // The ONE mint point returns the recentre already carrying its frame (finding #7);
        // unwrap to the raw triple only for the shift arithmetic + the `[i64; 3]` previous
        // recentre state below. The `RecentreVoxels` itself flows straight into the output.
        let new_recentre = scene.recentre_voxels_for_resolve(density);
        let new_recentre_voxels = new_recentre.voxels();
        // The floating-origin shift since the last rebuild (render-frame voxels). The
        // first rebuild has no previous recentre, so it shifts nothing (the camera is
        // framed explicitly at startup, not compensated). The shell subtracts this
        // from `camera.target` so the view stays put as the origin floats.
        let previous_recentre = self.previous_recentre_voxels.unwrap_or(new_recentre_voxels);
        let recentre_shift_voxels = [
            new_recentre_voxels[0] - previous_recentre[0],
            new_recentre_voxels[1] - previous_recentre[1],
            new_recentre_voxels[2] - previous_recentre[2],
        ];
        // The chunk-granular GPU-buffer incremental (#55) reuses UNTOUCHED chunks' baked
        // buffers verbatim, so it is only valid when those buffers are still in the right
        // frame. Two guards force a wholesale re-mesh even for a localisable edit:
        //   * DENSITY change — re-keys every chunk (chunk extent = CHUNK_BLOCKS × density),
        //     so the whole resident buffer set is in a different voxel frame.
        //   * RECENTRE (floating-origin) SHIFT — although a two-layer chunk is chunk-local-
        //     integer (so the resident CACHE stays valid across a shift), the MESHER bakes the
        //     recentre into each vertex's world position at emit time. A shift therefore
        //     staleens every kept buffer's vertices (an untouched chunk's mesh would sit at the
        //     old origin), exactly the dense `incremental_rebuild_from_chunks` precondition.
        //     The cache invalidation below still runs (it is frame-independent); only the
        //     GPU-buffer incremental falls back.
        let density_changed = self.previous_density != Some(density);
        let recentre_shifted = recentre_shift_voxels != [0; 3];
        let buffers_reframed = density_changed || recentre_shifted;
        // The incremental GPU-buffer re-mesh hint (#55): `Some(evicted_dirty)` only when the
        // edit LOCALISED (an `invalidate_aabb` path) AND the resident buffers stayed in frame.
        // Any wholesale `clear()` — first build, region-spanning Part edit — and any reframing
        // (density change / recentre shift) yields `None`, so the shell re-meshes wholesale.
        let incremental_dirty_chunks: Option<Vec<[i32; 3]>> = match self
            .previous_leaf_index
            .as_ref()
        {
            Some(previous) => match new_leaf_index.edit_aabb_since(previous) {
                Some(edit_aabb) => {
                    profiling::scope!("invalidate_aabb");
                    let evicted = self.two_layer_cache.invalidate_aabb(&edit_aabb, density);
                    // `invalidate_aabb` clears everything on a density mismatch (returning all
                    // resident coords); either way, a reframing forces a wholesale re-mesh.
                    if buffers_reframed {
                        None
                    } else {
                        Some(evicted)
                    }
                }
                None => {
                    profiling::scope!("invalidate_clear");
                    self.two_layer_cache.clear();
                    None
                }
            },
            None => {
                profiling::scope!("invalidate_clear");
                self.two_layer_cache.clear();
                None
            }
        };
        self.previous_recentre_voxels = Some(new_recentre_voxels);
        self.previous_leaf_index = Some(new_leaf_index);
        self.previous_density = Some(density);

        // Ensure every covering chunk is resident (re-classifying only the dirty /
        // missing ones); the SAME `Arc`-shared set feeds both the mesher and the brick
        // sink in the shell (classified once). The two-layer mesher re-meshes wholesale from
        // this set each rebuild (the resident cache is the incremental seam).
        //
        // ADR 0011 G5: NO whole-region `VoxelGrid` is expanded here anymore — the last
        // per-edit densify (ADR 0010's flagged debt) is retired. The resident set is the sole
        // display truth.
        let two_layer_chunks: Vec<([i32; 3], Arc<TwoLayerChunk>)> = {
            profiling::scope!("resident_two_layer_chunks");
            // The resident cache hands out an OWNED, `Arc`-shared covering set (an O(1)
            // refcount bump per chunk — NOT the old O(all-blocks) deep clone). It already
            // outlives the `&mut self` cache borrow, so it becomes `RebuildOutput.
            // two_layer_chunks` directly, with no further copy.
            self.two_layer_cache.resident_two_layer_chunks(scene, density, 0)
        };
        let region_dimensions = Self::region_dimensions_for(scene, density);
        RebuildOutcome::Built(RebuildOutput {
            region_dimensions,
            two_layer_chunks,
            recentre_voxels: new_recentre,
            recentre_shift_voxels,
            incremental_dirty_chunks,
        })
    }
}
