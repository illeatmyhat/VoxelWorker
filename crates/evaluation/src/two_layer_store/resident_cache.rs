//! The TwoLayerResidentCache — the live incremental resident set of two-layer chunks.

use std::collections::BTreeMap;
use std::sync::Arc;

use document::scene::Scene;
use voxel_core::spatial_index::{ChunkCoverage, VoxelAabb};

#[allow(unused_imports)]
use super::*;

/// **The resident two-layer display cache (ADR 0010 #54 — chunk-granular incremental edits).**
///
/// The [`TwoLayerStore`] above is a *stateless* builder (every call re-classifies a chunk from
/// the scene); this is its **incremental-edit counterpart**, the two-layer analogue of the dense
/// [`crate::store::Store`]. It holds the resident [`TwoLayerChunk`]s across edits and re-derives
/// **only the chunks an edit's world-AABB intersects** (chunk-granular, ADR 0002 Decision 3),
/// mirroring [`Store::invalidate_aabb`](crate::store::Store::invalidate_aabb) exactly. Untouched
/// chunks stay resident.
///
/// ## Why a dirty chunk re-runs the whole build
///
/// A dirty chunk drops its cached [`TwoLayerChunk`] and, on next access, re-runs the block
/// classifier + two-layer build ([`build_two_layer_chunk`]) from scratch. Chunk-granular is
/// sufficient to unblock E5 (retire the dense path); a **block-granular dirty-brick recompute**
/// (re-classify only the blocks the edit AABB touches, keeping the rest of the chunk's coarse
/// layer) is a later optimization, NOT this slice (ADR 0010 Consequences).
///
/// ## Frame (ADR 0008) — why a recentre shift does NOT invalidate the cache
///
/// A [`TwoLayerChunk`] is stored in **chunk-local integer** frame (its coarse ids + block-local
/// cuboids never mention the absolute origin — that lives in the chunk COORD key). The recentre /
/// floating origin is applied only at *expand* time as a pure index offset
/// ([`TwoLayerChunk::expand_occupancy_into`]). So — unlike the dense [`Store`], which caches
/// PRE-REBASED grids and must clear on a floating-origin shift — a recentre shift leaves every
/// resident two-layer chunk VALID. Only a **density change** (which resizes each chunk's voxel
/// extent) forces a wholesale clear; that is the one binding this cache tracks.
///
/// [`Store`]: crate::store::Store
#[derive(Debug, Clone, Default)]
pub struct TwoLayerResidentCache {
    /// The two-layer capability flag (ADR 0010 Decision 6), forwarded to the stateless builder.
    /// `false` (the default) means the cache stays empty and [`resident_two_layer_chunks`] is a
    /// no-op, so a caller falls back to the dense path.
    ///
    /// [`resident_two_layer_chunks`]: Self::resident_two_layer_chunks
    enabled: bool,
    /// Resident chunks keyed by ABSOLUTE chunk coord (the only LOD in use is 0, ADR 0002 S4a).
    ///
    /// Stored as `Arc<TwoLayerChunk>` so [`resident_two_layer_chunks`](Self::resident_two_layer_chunks)
    /// can hand the owned covering set out to the readers (mesh / brick / fog) and into the async
    /// `GeometryRebuildRequest` with an O(1) refcount bump each, never a deep chunk copy per rebuild.
    ///
    /// **Mutation discipline (why an `Arc` is safe here).** A dirty chunk is never mutated
    /// through its `Arc` while shell copies are alive: the cache only ever REPLACES a chunk's
    /// entry with a freshly built `Arc` (evict via [`invalidate_aabb`](Self::invalidate_aabb) /
    /// [`clear`](Self::clear), then re-`insert` in [`resident_two_layer_chunks`]), so an
    /// outstanding shared copy keeps seeing the exact chunk it was handed. No `Arc::make_mut` /
    /// in-place edit path exists — the resident chunk is immutable once built.
    resident: BTreeMap<[i32; 3], Arc<TwoLayerChunk>>,
    /// The density the resident chunks were built at. A change resizes every chunk's voxel
    /// extent, so it forces a wholesale [`clear`](Self::clear) (mirrors
    /// [`Store::rebind_if_changed`](crate::store::Store)'s density guard).
    bound_density: Option<u32>,
}

impl TwoLayerResidentCache {
    /// A resident cache with the two-layer capability ENABLED. The default ([`Default`]) is
    /// DISABLED (empty, no-op), matching the ADR's "OFF by default, dense fallback" coexistence.
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            resident: BTreeMap::new(),
            bound_density: None,
        }
    }

    /// Whether the two-layer capability is engaged.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// The number of chunks currently resident (a diagnostic / test-observability count).
    pub fn resident_len(&self) -> usize {
        self.resident.len()
    }

    /// Drop every cached chunk (the all-or-nothing invalidation seam) — the two-layer analogue
    /// of [`Store::clear`](crate::store::Store::clear). Used for the first build (no previous
    /// scene to diff) and the edit kinds [`invalidate_aabb`](Self::invalidate_aabb) can't
    /// localise (a density change, or a region-spanning VoxelBody edit).
    pub fn clear(&mut self) {
        self.resident.clear();
        self.bound_density = None;
    }

    /// **Targeted invalidation (ADR 0010 #54, mirroring
    /// [`Store::invalidate_aabb`](crate::store::Store::invalidate_aabb)).** Drop exactly the
    /// cached chunks whose half-open box intersects the edit world-AABB `edit_aabb` (absolute
    /// voxels, producer-true frame), at `voxels_per_block` — ADR 0002 Decision 3's whole-chunk
    /// dirty granularity. Every other cached chunk stays resident untouched, so the next
    /// [`resident_two_layer_chunks`](Self::resident_two_layer_chunks) re-runs the classifier +
    /// build only for the evicted (dirty) chunks.
    ///
    /// `edit_aabb` is what
    /// [`LeafSpatialIndex::edit_aabb_since`](voxel_core::spatial_index::LeafSpatialIndex::edit_aabb_since)
    /// returns: the union of an edit's old and new leaf boxes, so a moved node dirties chunks
    /// around BOTH its source and destination. An empty `edit_aabb` evicts nothing.
    ///
    /// A density mismatch against the bound density is treated conservatively (the AABB was
    /// computed at a different chunk size) by clearing everything — belt-and-braces, as the
    /// caller already falls back to [`clear`](Self::clear) for a density change.
    ///
    /// **Returns the chunk coords actually evicted** (resident AND intersecting the edit AABB),
    /// so the mesher's incremental plan (`cuboid_incremental_plan`, up in the display layer) can
    /// dilate exactly this dirty set by the 26-neighbourhood. The density-mismatch path returns
    /// every previously-resident coord.
    pub fn invalidate_aabb(
        &mut self,
        edit_aabb: &VoxelAabb,
        voxels_per_block: u32,
    ) -> Vec<[i32; 3]> {
        if let Some(bound) = self.bound_density {
            if bound != voxels_per_block {
                let evicted: Vec<[i32; 3]> = self.resident.keys().copied().collect();
                self.clear();
                return evicted;
            }
        }
        let Some((min_chunk, max_chunk)) = edit_aabb.covering_chunk_range(voxels_per_block) else {
            return Vec::new(); // empty edit AABB — nothing to invalidate.
        };
        let mut evicted = Vec::new();
        self.resident.retain(|coord, _| {
            let inside = (0..3).all(|axis| coord[axis] >= min_chunk[axis] && coord[axis] <= max_chunk[axis]);
            if inside {
                evicted.push(*coord);
            }
            !inside
        });
        evicted
    }

    /// **Per-chunk two-layer accessor — the incremental analogue of
    /// [`Store::resident_render_chunks`](crate::store::Store::resident_render_chunks).** Ensure
    /// every covering chunk of `(scene, voxels_per_block, lod)` is resident (re-run the
    /// classifier + build for any DIRTY or MISSING chunk, reuse resident HITs verbatim), then
    /// return every covering chunk as `([i32; 3] absolute_chunk_coord, Arc<TwoLayerChunk>)` in the
    /// SAME z,y,x order the dense store assembles.
    ///
    /// Because a two-layer chunk is chunk-local-integer (frame-independent), a resident HIT is
    /// reused across a recentre shift; only [`invalidate_aabb`](Self::invalidate_aabb) (a dirty
    /// edit) or a density change ([`clear`](Self::clear)) re-derives a chunk. The returned chunks
    /// are `Arc`-SHARED (an O(1) refcount bump per covering chunk, NOT a deep copy), so the caller
    /// owns a covering set that outlives this `&mut self` borrow and can be meshed, fog-expanded,
    /// brick-packed AND moved into the async mesh request without cloning a single chunk. The fill
    /// (needing `&mut self`) runs FIRST, then the gather clones the resident `Arc`s.
    ///
    /// Returns an empty `Vec` when the capability is OFF (dense fallback) or the scene has no
    /// covering chunk range (a VoxelBody-only scene).
    pub fn resident_two_layer_chunks(
        &mut self,
        scene: &Scene,
        voxels_per_block: u32,
        lod: u32,
    ) -> Vec<([i32; 3], Arc<TwoLayerChunk>)> {
        debug_assert_eq!(lod, 0, "E2 only builds full resolution (lod 0)");
        if !self.enabled {
            return Vec::new();
        }
        // A density change resizes every chunk's voxel extent; drop the stale residents.
        if self.bound_density != Some(voxels_per_block) {
            self.resident.clear();
            self.bound_density = Some(voxels_per_block);
        }

        let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(voxels_per_block) else {
            return Vec::new();
        };

        // Fill misses (dirty-evicted or never-built). The build (`build_two_layer_chunk`)
        // is the ~3.5s cost and is pure given the scene, so parallelise the WHOLESALE fill
        // (#57): gather the missing coords, build them in parallel into a Vec, THEN insert
        // serially (the insert is cheap next to the build). This keeps the incremental
        // dirty-set path (#54) intact — only chunks actually absent are (re)built, resident
        // HITs are reused verbatim — while the initial build / density-change / recentre
        // fallback (which re-fills many chunks at once) now runs across threads. Each chunk
        // is deterministic given the scene, so the resident map is identical to the serial
        // one-by-one fill regardless of thread count.
        //
        // #63: HOIST the leaf list out of the per-chunk build (compute ONCE, not per missing
        // chunk); #66: the EDIT BROADPHASE (ADR 0011 Decision 4b) — a stateless per-build BVH
        // over the leaf world-AABBs, queried per missing chunk, so each is built from only
        // its overlapping candidate leaves. Only chunks actually absent are (re)built,
        // resident HITs are reused verbatim (the #54 dirty-set path is intact).
        let leaves = scene.leaf_producers(voxels_per_block);
        let broadphase = leaf_edit_broadphase(&leaves, voxels_per_block);
        let missing_coords: Vec<[i32; 3]> =
            enumerate_covering_chunk_coords(min_chunk, max_chunk)
                .into_iter()
                .filter(|coord| !self.resident.contains_key(coord))
                .collect();
        let freshly_built =
            build_chunks_parallel(missing_coords, &leaves, &broadphase, voxels_per_block);
        for (coord, chunk) in freshly_built {
            self.resident.insert(coord, chunk);
        }

        // Gather the covering chunks as O(1) `Arc` clones (all HITs after the fill above) — the
        // caller gets an owned, shareable covering set with no deep chunk copy.
        let resident = &self.resident;
        let mut chunks = Vec::new();
        for chunk_z in min_chunk[2]..=max_chunk[2] {
            for chunk_y in min_chunk[1]..=max_chunk[1] {
                for chunk_x in min_chunk[0]..=max_chunk[0] {
                    let coord = [chunk_x, chunk_y, chunk_z];
                    if let Some(chunk) = resident.get(&coord) {
                        chunks.push((coord, Arc::clone(chunk)));
                    }
                }
            }
        }
        chunks
    }
}

