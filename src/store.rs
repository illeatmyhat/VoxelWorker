//! ADR 0003 (foundation rework) data layer — the store. Relocated here from
//! `chunk_cache` in slice A2b (the type `ChunkResolveCache` was renamed to
//! [`Store`]); a re-export shim at `crate::chunk_cache` keeps existing call
//! sites compiling until later slices migrate them.
//!
//! A per-chunk resolve cache (ADR 0002 Decision 3, issue #27 S2).
//!
//! S0 made resolve **chunk-addressable** ([`Scene::resolve_chunk`]) and proved a
//! whole-region resolve can be reassembled from per-chunk pieces
//! (`Scene::resolve_region_via_chunks`). S2 turns that decomposition into the
//! **resolve mechanism**: a cache keyed by `(chunk_coord, lod)` that resolves a
//! chunk **on demand** (lazily) and stores the result, so a second request for the
//! same chunk is a map lookup instead of a re-resolve.
//!
//! ## What S2 changes (and what it does NOT)
//!
//! * **Lazy per-chunk resolve + cache.** [`Store::chunk`] returns the
//!   cached per-chunk [`VoxelGrid`] (in **absolute** composite voxel coordinates,
//!   exactly as [`Scene::resolve_chunk`] produces), resolving + storing it on a
//!   miss.
//! * **Per-chunk voxel bound.** The old whole-region `MAX_GRID_VOXELS` guard is now
//!   a *per-chunk* bound (a single chunk can't exceed it), so a scene whose TOTAL
//!   voxel count is far beyond the old 6M ceiling resolves fine as long as every
//!   individual chunk is small. See [`voxel_core::voxel::MAX_CHUNK_VOXELS`].
//! * **Identical render output.** `Store::resolve_region` rebuilds the
//!   SAME recentred monolithic grid the renderer/mesher/fog consume today — but
//!   assembled from cached chunks. The bytes downstream are unchanged (see the
//!   module-level invariant in `Store::resolve_region`).
//!
//! **S3 (#27) added smart invalidation** on top of this seam:
//! [`Store::invalidate_aabb`] evicts exactly the chunks an edit's
//! world-AABB intersects (whole-chunk dirty granularity, ADR 0002 Decision 3). The
//! edit AABB is computed by
//! [`LeafSpatialIndex::edit_aabb_since`](voxel_core::spatial_index::LeafSpatialIndex::edit_aabb_since)
//! (diffing the scene's leaf spatial index before vs after the edit);
//! [`Store::clear`] remains the fallback for edits that can't be
//! localised (a density change or a region-spanning Part edit).
//!
//! What is **deferred** (do NOT look for it here):
//!
//! * **Recentre removal, camera-relative rebasing, renderer consuming per-chunk
//!   meshes directly** — S4 / #27. S2 still hands the renderer one recentred
//!   monolithic grid.

use std::collections::HashMap;

use crate::chunk_storage::{compress, decompress};
use crate::disk_chunk_store::DiskChunkStore;
use crate::scene::Scene;
use voxel_core::spatial_index::{ChunkCoverage, VoxelAabb};
use voxel_core::voxel::{Voxel, VoxelGrid};

/// Back-compat alias for the pre-A2b name. Existing call sites refer to the
/// store as `ChunkResolveCache`; later slices migrate them to [`Store`].
pub type ChunkResolveCache = Store;

/// The cache key: a chunk coordinate (in `CHUNK_BLOCKS`-cell space) plus its
/// level-of-detail. `lod` is the parked LOD seam (ADR 0002 Decision 2): it is
/// always `0` today and is carried so a future down-sampling LOD level is a
/// behavioural change, not a key-shape change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkCacheKey {
    /// The chunk's integer cell coordinate (see [`Scene::resolve_chunk`]).
    pub chunk_coord: [i32; 3],
    /// Level of detail (always `0` for now).
    pub lod: u32,
}

impl ChunkCacheKey {
    /// A key for `chunk_coord` at the given `lod`.
    pub fn new(chunk_coord: [i32; 3], lod: u32) -> Self {
        Self { chunk_coord, lod }
    }
}

/// An in-memory cache of per-chunk resolved [`VoxelGrid`]s, keyed by
/// `(chunk_coord, lod)`.
///
/// A cache instance is bound to one density (`voxels_per_block`): the chunk extent
/// in voxels is a function of density, so mixing densities in one cache would key
/// chunks of different physical sizes under the same coordinate. A density change
/// therefore [`clear`](Self::clear)s and re-binds the cache (see
/// `resolve_region`).
///
/// **S3 seam:** invalidation is currently all-or-nothing
/// ([`clear`](Self::clear)) plus a single-chunk drop
/// ([`invalidate_chunk`](Self::invalidate_chunk)); neither tracks WHICH edit
/// touched WHICH chunk. S3 (#27) adds edit-world-AABB → dirty-chunk invalidation
/// on top of this seam.
#[derive(Debug, Default)]
pub struct Store {
    /// The resolved per-chunk grids, in coordinates **rebased to the bound floating
    /// origin** (ADR 0002 Decision 2, S4b). With the default floating origin
    /// `[0, 0, 0]` these are absolute composite coordinates (the S0 contract); the
    /// render path binds the origin to the composite recentre so the chunks come out
    /// already rebased (and far chunks keep f32 precision — the subtraction is done in
    /// i64 inside [`Scene::resolve_chunk_rebased`], not in f32 here).
    chunks: HashMap<ChunkCacheKey, VoxelGrid>,
    /// The density every cached chunk was resolved at, set on first use. `None`
    /// until the first resolve. A request at a different density clears the cache
    /// and re-binds to the new density (a chunk's voxel extent depends on density).
    bound_density: Option<u32>,
    /// The floating origin (in absolute voxels) every cached chunk was rebased
    /// around (ADR 0002 S4b). `[0, 0, 0]` until bound otherwise. A request at a
    /// different origin clears + re-binds (every cached chunk's stored positions are
    /// relative to it). `resolve_region` binds it to the composite recentre.
    bound_floating_origin: [i64; 3],

    // ===== Out-of-core spill (issue #20 Step 3) ==============================
    /// The maximum number of resolved chunks that may stay **resident** in RAM at
    /// once. `None` (the default, [`new`](Self::new)) means UNBOUNDED — the cache
    /// never spills and behaves exactly as before this step (every existing caller,
    /// every golden, every parity test). When `Some(cap)`, a resident insert that
    /// would push the resident count over `cap` spills the least-recently-used
    /// resident chunk to [`disk_store`](Self::disk_store) (compressing it), and a
    /// later access reloads it transparently.
    max_resident_chunks: Option<usize>,
    /// The backing disk store for spilled chunks. `None` until the first spill is
    /// possible (i.e. only ever created when [`max_resident_chunks`](Self::max_resident_chunks)
    /// is set). Keyed by [`ChunkCacheKey`] in the cache's CURRENT density+origin
    /// binding: a [`rebind_if_changed`](Self::rebind_if_changed) (density / origin
    /// change) clears BOTH the resident map and this store, so a reloaded chunk can
    /// never carry a stale binding (the S6c wiring note's correctness condition).
    disk_store: Option<DiskChunkStore>,
    /// Per-resident-chunk last-use tick, for least-recently-used spill selection.
    /// A monotonically increasing logical clock ([`access_clock`](Self::access_clock))
    /// stamps the touched chunk on every access; the resident chunk with the smallest
    /// tick is the spill victim. Only populated when spilling is active.
    last_used_tick: HashMap<ChunkCacheKey, u64>,
    /// The monotonic logical clock backing [`last_used_tick`](Self::last_used_tick).
    access_clock: u64,
    /// Lifetime count of chunks spilled from RAM to disk (one per over-cap insert).
    spill_count: u64,
    /// Lifetime count of chunks reloaded from disk back into RAM (a hit on a spilled
    /// chunk — NOT a recompute, NOT a resident hit).
    disk_reload_count: u64,
    /// Lifetime count of chunks resolved from scratch via the scene resolver (a miss
    /// in BOTH the resident map and the disk store).
    recompute_count: u64,
}

impl Store {
    /// A fresh, empty cache that NEVER spills (unbounded resident set). This is the
    /// behaviour every existing caller relies on (the renderer, `shot`, `vox_export`,
    /// every golden and parity test) — identical to before issue #20 Step 3.
    pub fn new() -> Self {
        Self::default()
    }

    /// A fresh, empty cache that keeps at most `max_resident_chunks` resolved chunks
    /// resident in RAM, **spilling the least-recently-used to disk** (issue #20 Step 3)
    /// under `disk_store_directory` and reloading them transparently on the next
    /// access. The public chunk-fetch API and the data returned are UNCHANGED — a
    /// chunk fetched after a spill+reload is byte-identical to one that stayed
    /// resident (the spill compresses via [`crate::chunk_storage`], whose round-trip
    /// is lossless to the f32 bit).
    ///
    /// # Panics
    /// Panics if `max_resident_chunks == 0` (a cache that can hold nothing resident
    /// is a misconfiguration — same contract as [`DiskChunkStore::new`]).
    ///
    /// # Errors
    /// Returns the I/O error if the disk-store directory cannot be created.
    pub fn with_resident_cap(
        max_resident_chunks: usize,
        disk_store_directory: impl AsRef<std::path::Path>,
    ) -> std::io::Result<Self> {
        assert!(
            max_resident_chunks >= 1,
            "Store resident cap must be at least 1"
        );
        // The disk store needs its own resident cap; it is only ever used as cold
        // storage (the cache spills INTO it and reloads OUT of it), so give it the
        // same cap — its own internal LRU is harmless here because every chunk handed
        // to it is immediately one we dropped from RAM.
        let disk_store = DiskChunkStore::new(disk_store_directory, max_resident_chunks)?;
        Ok(Self {
            max_resident_chunks: Some(max_resident_chunks),
            disk_store: Some(disk_store),
            ..Self::default()
        })
    }

    /// Number of chunks currently resident in RAM (for tests / diagnostics). When a
    /// resident cap is set this never exceeds it.
    pub fn resident_chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Lifetime count of chunks spilled from RAM to disk (issue #20 Step 3). Always
    /// `0` for an unbounded cache.
    pub fn spill_count(&self) -> u64 {
        self.spill_count
    }

    /// Lifetime count of chunks reloaded from disk back into RAM (a hit on a spilled
    /// chunk). Always `0` for an unbounded cache.
    pub fn disk_reload_count(&self) -> u64 {
        self.disk_reload_count
    }

    /// Lifetime count of chunks resolved from scratch via the scene resolver (a miss
    /// in BOTH the resident map and the disk store).
    pub fn recompute_count(&self) -> u64 {
        self.recompute_count
    }

    /// The per-chunk resolved grid for `chunk_coord` at `lod`, resolving + caching
    /// it on a miss and returning the cached grid on a hit.
    ///
    /// The returned grid is in **absolute** composite voxel coordinates (a chunk's
    /// voxels keep their true scene positions — they are NOT rebased to the chunk
    /// origin), exactly as [`Scene::resolve_chunk`] produces. The first call binds
    /// the cache to `voxels_per_block`; a later call at a different density clears
    /// and re-binds (a chunk's voxel extent is density-dependent).
    ///
    /// A chunk whose resolved voxel count would exceed the per-chunk bound
    /// ([`voxel_core::voxel::MAX_CHUNK_VOXELS`]) is rejected by the call sites BEFORE
    /// resolving (the bound is a guard on the chunk's voxel *capacity*, evaluated
    /// from the chunk's voxel extent); this method itself does not re-check it (it
    /// resolves whatever the scene yields for that chunk).
    pub fn chunk(&mut self, chunk_coord: [i32; 3], scene: &Scene, voxels_per_block: u32, lod: u32) -> &VoxelGrid {
        // The public entry binds the cache to the ABSOLUTE frame (floating origin
        // `[0, 0, 0]`) — the S0 contract a bare `chunk()` caller expects. The render
        // path goes through `resolve_region`, which binds the floating origin to the
        // composite recentre first and then pulls chunks via `chunk_for_current_binding`.
        self.rebind_if_changed(voxels_per_block, [0, 0, 0]);
        self.chunk_for_current_binding(chunk_coord, scene, voxels_per_block, lod)
    }

    /// Pull (or resolve) one chunk for the cache's CURRENT density + floating-origin
    /// binding, WITHOUT re-binding. The caller is responsible for having bound the
    /// cache (via `rebind_if_changed`) to the intended density/origin first — this is
    /// what lets `resolve_region` bind to the composite recentre once and then pull
    /// every covering chunk already rebased.
    fn chunk_for_current_binding(
        &mut self,
        chunk_coord: [i32; 3],
        scene: &Scene,
        voxels_per_block: u32,
        lod: u32,
    ) -> &VoxelGrid {
        let key = ChunkCacheKey::new(chunk_coord, lod);
        self.ensure_resident(key, scene, voxels_per_block);
        // `ensure_resident` guarantees the key is in `self.chunks`; the borrow is split
        // out here so the spill (which also borrows `self` mutably) is already done.
        self.chunks
            .get(&key)
            .expect("ensure_resident left the requested chunk resident")
    }

    /// Guarantee the chunk for `key` is **resident** in `self.chunks`, counting the
    /// access for LRU and (when a resident cap is set) spilling the least-recently-used
    /// OTHER resident chunk to disk if the insert would breach the cap.
    ///
    /// The three lookup tiers (issue #20 Step 3):
    /// 1. **Resident hit** — already in RAM. Just refresh its LRU tick.
    /// 2. **Disk hit** — spilled earlier: decompress it back to a [`VoxelGrid`] and
    ///    promote it to resident (counts as `disk_reload`).
    /// 3. **Miss in both** — resolve it from scratch via the scene (counts as a
    ///    `recompute`).
    ///
    /// For an unbounded cache (`max_resident_chunks == None`) this is exactly the old
    /// `entry().or_insert_with()` resolve-on-miss with no disk tier and no LRU
    /// bookkeeping.
    fn ensure_resident(&mut self, key: ChunkCacheKey, scene: &Scene, voxels_per_block: u32) {
        // Tier 1: resident hit.
        if self.chunks.contains_key(&key) {
            self.touch_resident(key);
            return;
        }

        let origin = self.bound_floating_origin;

        // Tier 2: spilled to disk? Reload + decompress. (Only possible when spilling.)
        if let Some(store) = self.disk_store.as_mut() {
            if let Some(compressed) = store
                .get(key)
                .expect("disk store reload must not fail")
            {
                let grid = decompress(&compressed);
                self.disk_reload_count += 1;
                self.insert_resident(key, grid);
                return;
            }
        }

        // Tier 3: miss in both — resolve from scratch.
        let grid = scene.resolve_chunk_rebased(key.chunk_coord, voxels_per_block, key.lod, origin);
        self.recompute_count += 1;
        self.insert_resident(key, grid);
    }

    /// Insert a freshly-resident `grid` under `key`, stamping its LRU tick and, if a
    /// resident cap is set and this insert breaches it, first spilling the
    /// least-recently-used OTHER resident chunk to disk.
    fn insert_resident(&mut self, key: ChunkCacheKey, grid: VoxelGrid) {
        if self.max_resident_chunks.is_some() {
            self.spill_until_room_for_one();
        }
        self.chunks.insert(key, grid);
        self.touch_resident(key);
    }

    /// Stamp `key` with the next LRU clock tick (its most-recent use). A no-op when
    /// spilling is disabled (no LRU bookkeeping is needed for an unbounded cache).
    fn touch_resident(&mut self, key: ChunkCacheKey) {
        if self.max_resident_chunks.is_none() {
            return;
        }
        self.access_clock += 1;
        self.last_used_tick.insert(key, self.access_clock);
    }

    /// While the resident set is at (or over) the cap, spill the single
    /// least-recently-used resident chunk to disk. Called BEFORE a new resident insert
    /// so the post-insert resident count never exceeds the cap.
    fn spill_until_room_for_one(&mut self) {
        let Some(cap) = self.max_resident_chunks else {
            return;
        };
        while self.chunks.len() >= cap {
            let Some(victim_key) = self
                .last_used_tick
                .iter()
                .filter(|(key, _)| self.chunks.contains_key(key))
                .min_by_key(|(_, &tick)| tick)
                .map(|(&key, _)| key)
            else {
                break; // Nothing resident to spill (cap == 0 is rejected at construction).
            };
            let grid = self
                .chunks
                .remove(&victim_key)
                .expect("the LRU victim was just observed resident");
            self.last_used_tick.remove(&victim_key);
            let compressed = compress(&grid);
            self.disk_store
                .as_mut()
                .expect("a cap implies a disk store")
                .put(victim_key, compressed)
                .expect("spilling a chunk to disk must not fail");
            self.spill_count += 1;
        }
    }

    /// Rebuild the SAME recentred monolithic [`VoxelGrid`] the renderer, mesher and
    /// onion fog consume today — but assembled by pulling each covering chunk from
    /// the cache (resolving misses on demand) instead of stamping every leaf into
    /// one grid in a single pass.
    ///
    /// ## Identical-output invariant
    ///
    /// The render path's truth is [`Scene::resolve_region`], which (a) sizes the
    /// output to the composite extent and (b) **recentres** the composite on the
    /// origin by subtracting `recentre_voxels` from every voxel. This method
    /// reproduces both:
    ///
    /// 1. Pull each covering chunk from the cache. A cached chunk holds voxels in
    ///    **absolute** composite coordinates (`producer_local + world_offset ×
    ///    density`), the exact value [`Scene::resolve_chunk`] emits — and, by the S0
    ///    equivalence proof, the union of all covering chunks is the exact occupied
    ///    SET of [`Scene::resolve_region`] **before** its recentre.
    /// 2. Apply the SAME recentre offset [`Scene::resolve_region`] uses, subtracting
    ///    it from each voxel to land the composite centred in the output.
    ///
    /// For every scene whose occupied voxels sit at coordinates exactly
    /// representable in `f32` (all near-origin scenes — every golden, and every
    /// scene in the S2 parity tests: sphere/cylinder/torus/village/demo), the
    /// recentre subtraction is exact, so the reassembled grid's `(position,
    /// material_id)` set is **bit-identical** to [`Scene::resolve_region`]'s. The
    /// parity tests assert this equality directly; if a scene ever moved the
    /// goldens it would mean an observable geometry change, not a rebaseline.
    ///
    /// (Far-offset scenes — voxels at ~1e6 — lose `f32` precision INSIDE the
    /// absolute chunk before the subtraction, so the two frames can differ there.
    /// That is the very precision problem S4's camera-relative rebasing exists to
    /// solve; it is out of scope for S2 and is NOT a golden.)
    ///
    /// **Oracle — compile-gated.** This is the dense reference resolver the sparse
    /// runtime path is cross-checked against (and the `shot` golden tool renders from);
    /// it is excluded from production builds behind the `oracle` feature (tests reach it
    /// via `cfg(test)`), so production code cannot reach a dense path — see the proof
    /// chapter's "Oracles" section (`docs/architecture/05-proof.md`).
    #[cfg(any(test, feature = "oracle"))]
    pub fn resolve_region(&mut self, scene: &Scene, voxels_per_block: u32, lod: u32) -> VoxelGrid {
        debug_assert_eq!(lod, 0, "S2 only resolves full resolution (lod 0)");

        // Thin wrapper over the [`bind_region`](Self::bind_region) primitive (A2c):
        // bind the cache to the composite recentre/floating origin (ADR 0002
        // Decision 2 / S4b) and make every covering chunk resident — the only step
        // needing `&mut self` — then assemble the union of the resident chunks'
        // voxels. Each chunk comes out ALREADY rebased by `resolve_chunk_rebased`,
        // with the recentre subtracted in i64 BEFORE the f32 downcast, so a
        // far-placed scene keeps full f32 precision (the S1 speckle is gone). For a
        // near scene this is bit-identical to the previous direct f32 subtract (the
        // recentre is integer-block-aligned and positions are small), so the goldens
        // are unchanged. The covering chunks are visited in the same z,y,x order the
        // bind resolved them in, so the assembled voxel order is identical too.
        let region_dimensions = self.bind_region(scene, voxels_per_block, lod);
        let mut output = VoxelGrid::new(region_dimensions);
        // ADR 0008: carry the recentre the chunks were rebased by, so the fog (and any
        // other consumer) decodes `world → index` without re-deriving `floor(dim/2)`. This
        // matches `Scene::resolve_region`'s output exactly (the S2 identical-output net).
        output.recentre_voxels = scene.recentre_voxels_for_resolve(voxels_per_block).voxels();
        for grid in self.covering_chunk_grids(scene, voxels_per_block, lod) {
            // The cached chunk is already rebased to the floating origin
            // (= recentre), so its voxels drop straight into the output.
            output.occupied.extend_from_slice(&grid.occupied);
        }
        output
    }

    /// **Per-chunk render accessor (issue #20 S6c step 4).** Bind the cache to the
    /// composite recentre/floating-origin for `(scene, density, lod)` EXACTLY as
    /// `resolve_region` does, then return every covering
    /// chunk as `([i32; 3] absolute_chunk_coord, &VoxelGrid rebased_grid)`.
    ///
    /// The returned grids are the SAME rebased per-chunk grids whose union
    /// `resolve_region` assembles — byte-identical (each one
    /// is already rebased to the floating origin = composite recentre, with the
    /// subtraction done in i64 inside
    /// [`Scene::resolve_chunk_rebased`](crate::scene::Scene::resolve_chunk_rebased)
    /// before the f32 downcast). The union of all returned chunks' occupied voxels
    /// (position + `material_id`, in the recentred frame) therefore equals
    /// `resolve_region`'s assembled grid voxel-for-voxel; this is the seam the
    /// upcoming per-chunk renderer consumes instead of one monolithic grid.
    ///
    /// Each returned coord is the absolute chunk coord that OWNS that grid's voxels
    /// (the half-open box `[c·E, (c+1)·E)` per axis, `E = CHUNK_BLOCKS × density`),
    /// and the returned coord set equals the scene's
    /// [`covering_chunk_range`](crate::scene::Scene::covering_chunk_range) for the
    /// region (empty for a Part-only scene with no composite extent).
    ///
    /// The grids are **borrowed** from the cache (`&VoxelGrid`), so the returned
    /// `Vec` borrows `self` immutably for its lifetime. Resolving misses needs
    /// `&mut self`, so binding + resolving happens FIRST (in
    /// [`bind_region`](Self::bind_region), all-mut), and the
    /// borrows are gathered only AFTER every covering chunk is resident (so the
    /// returned slice is all cache HITs — no interleaved mut/shared borrow). The
    /// resolved chunks stay CACHED, so a later `resolve_region` reuses them.
    pub fn resident_render_chunks(
        &mut self,
        scene: &Scene,
        voxels_per_block: u32,
        lod: u32,
    ) -> Vec<([i32; 3], &VoxelGrid)> {
        // Bind to the recentre + resolve/resident every covering chunk (the only
        // step needing `&mut self`); after this the gather below is all HITs.
        let _region_dimensions = self.bind_region(scene, voxels_per_block, lod);

        let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(voxels_per_block) else {
            return Vec::new();
        };

        let chunks = &self.chunks;
        let mut rendered = Vec::new();
        for chunk_z in min_chunk[2]..=max_chunk[2] {
            for chunk_y in min_chunk[1]..=max_chunk[1] {
                for chunk_x in min_chunk[0]..=max_chunk[0] {
                    let coord = [chunk_x, chunk_y, chunk_z];
                    if let Some(grid) = chunks.get(&ChunkCacheKey::new(coord, lod)) {
                        rendered.push((coord, grid));
                    }
                }
            }
        }
        rendered
    }

    /// **Region-scoped diameter readout (issue #20 S6d).** Compute the widest
    /// occupied run in the layer band `[band_min, band_max]` (the scrubber/diameter
    /// readout) from the scene's per-chunk grids, WITHOUT assembling a monolithic
    /// grid — returning the SAME value
    /// [`VoxelGrid::widest_run_in_band`](voxel_core::voxel::VoxelGrid::widest_run_in_band)
    /// returns for the assembled region.
    ///
    /// The cache is bound to the composite recentre (exactly as
    /// `resolve_region`) so each covering chunk's voxels are
    /// in the recentred frame, then every chunk's voxels are bucketed into ONE
    /// shared per-`(y, z)` occupancy row keyed by the GLOBAL X index — so a run
    /// crossing a chunk seam is one contiguous span (see
    /// [`widest_run_in_band_over_chunks`](voxel_core::voxel::widest_run_in_band_over_chunks)
    /// for the stitching detail). The chunks resolved on a miss are CACHED, so a
    /// later `resolve_region` / re-measure reuses them.
    pub fn widest_run_in_band(
        &mut self,
        scene: &Scene,
        voxels_per_block: u32,
        lod: u32,
        band_min: u32,
        band_max: u32,
    ) -> u32 {
        let region_dimensions = self.bind_region(scene, voxels_per_block, lod);
        voxel_core::voxel::widest_run_in_band_over_chunks(
            region_dimensions,
            self.covering_chunk_grids(scene, voxels_per_block, lod),
            band_min,
            band_max,
        )
    }

    /// **Bound-region read primitive (ADR 0003 store seam; issue #20 S6d).** Bind
    /// the cache to the scene's ACTIVE region (recentre + density, as
    /// `resolve_region` does), ensure every covering chunk
    /// is resolved + resident, and return the region's voxel dimensions alongside
    /// the resident covering chunks' occupied slices — WITHOUT assembling a
    /// monolithic grid. This is the cache/export-agnostic primitive the `.vox`
    /// export is a thin wrapper over; the export glue itself lives at the call
    /// site over [`VoxExport::from_region_voxels`](crate::vox_export::VoxExport::from_region_voxels),
    /// so the cache no longer depends on the export module. The union of the
    /// returned occupied slices is exactly the monolithic region grid's occupied
    /// set (the S2 cache-assembly equivalence proof).
    pub fn bound_region_occupied(
        &mut self,
        scene: &Scene,
        voxels_per_block: u32,
        lod: u32,
    ) -> ([u32; 3], Vec<&[Voxel]>) {
        let region_dimensions = self.bind_region(scene, voxels_per_block, lod);
        let occupied = self
            .covering_chunk_grids(scene, voxels_per_block, lod)
            .map(|grid| &grid.occupied[..])
            .collect();
        (region_dimensions, occupied)
    }

    /// **The bound-region primitive (ADR 0003 store seam).** Bind the cache to the
    /// composite recentre/floating origin + density for `(scene, voxels_per_block)`
    /// and ensure every covering chunk is resolved + resident, returning the
    /// region's voxel dimensions. This is the shared `&mut self` step the four
    /// consumer-shaped reads ([`resolve_region`](Self::resolve_region),
    /// [`resident_render_chunks`](Self::resident_render_chunks),
    /// [`widest_run_in_band`](Self::widest_run_in_band),
    /// [`bound_region_occupied`](Self::bound_region_occupied)) are thin wrappers
    /// over. After this, [`covering_chunk_grids`](Self::covering_chunk_grids) yields
    /// the resident covering chunks (all cache HITs) in the recentred frame.
    fn bind_region(
        &mut self,
        scene: &Scene,
        voxels_per_block: u32,
        lod: u32,
    ) -> [u32; 3] {
        debug_assert_eq!(lod, 0, "S6d only operates at full resolution (lod 0)");
        let recentre_voxels = scene.recentre_voxels_for_resolve(voxels_per_block).voxels();
        self.rebind_if_changed(voxels_per_block, recentre_voxels);

        if let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(voxels_per_block) {
            for chunk_z in min_chunk[2]..=max_chunk[2] {
                for chunk_y in min_chunk[1]..=max_chunk[1] {
                    for chunk_x in min_chunk[0]..=max_chunk[0] {
                        let _ = self.chunk_for_current_binding(
                            [chunk_x, chunk_y, chunk_z],
                            scene,
                            voxels_per_block,
                            lod,
                        );
                    }
                }
            }
        }
        scene.placed_region_dimensions(voxels_per_block)
    }

    /// Iterate the per-chunk grids covering the scene's region, in chunk order.
    /// Assumes [`bind_region`](Self::bind_region) has
    /// already resolved + resident every covering chunk at the cache's current
    /// binding (so these are all map lookups, no resolves), and that the cache is
    /// bound to the recentre (so the voxels are in the recentred frame).
    fn covering_chunk_grids<'cache>(
        &'cache self,
        scene: &Scene,
        voxels_per_block: u32,
        lod: u32,
    ) -> impl Iterator<Item = &'cache VoxelGrid> {
        let range = scene.covering_chunk_range(voxels_per_block);
        let chunks = &self.chunks;
        range.into_iter().flat_map(move |(min_chunk, max_chunk)| {
            (min_chunk[2]..=max_chunk[2]).flat_map(move |chunk_z| {
                (min_chunk[1]..=max_chunk[1]).flat_map(move |chunk_y| {
                    (min_chunk[0]..=max_chunk[0]).filter_map(move |chunk_x| {
                        chunks.get(&ChunkCacheKey::new([chunk_x, chunk_y, chunk_z], lod))
                    })
                })
            })
        })
    }

    /// Drop every cached chunk (the all-or-nothing invalidation seam).
    ///
    /// Still used for the edit kinds [`invalidate_aabb`](Self::invalidate_aabb) can't
    /// localise (a density change, or a region-spanning Part edit) and on the very
    /// first rebuild (no previous scene to diff against). For a localisable edit,
    /// prefer `invalidate_aabb`.
    pub fn clear(&mut self) {
        self.chunks.clear();
        self.bound_density = None;
        self.bound_floating_origin = [0, 0, 0];
        // Purge spilled chunks + LRU state too, so a stale spilled chunk can never
        // resurface across a clear (issue #20 Step 3).
        self.last_used_tick.clear();
        if let Some(store) = self.disk_store.as_mut() {
            store.clear().expect("clearing the disk store must not fail");
        }
    }

    /// Drop a single cached chunk across all LODs (a finer-grained seam) — from BOTH
    /// the resident set AND the disk store, so an edit can never let a stale spilled
    /// chunk resurface (issue #20 Step 3).
    ///
    /// [`invalidate_aabb`](Self::invalidate_aabb) calls this for each chunk an edit's
    /// world-AABB intersects.
    pub fn invalidate_chunk(&mut self, chunk_coord: [i32; 3]) {
        self.evict_coord_everywhere(chunk_coord);
    }

    /// Purge every cached entry (resident, spilled, and LRU bookkeeping) for
    /// `chunk_coord` across all LODs. The disk-store purge is what stops a stale
    /// spilled chunk from reloading after an edit (issue #20 Step 3).
    fn evict_coord_everywhere(&mut self, chunk_coord: [i32; 3]) {
        // Gather the keys at this coord BEFORE mutating (a coord can hold several LODs).
        let purged: Vec<ChunkCacheKey> = self
            .chunks
            .keys()
            .copied()
            .filter(|key| key.chunk_coord == chunk_coord)
            .collect();
        self.chunks.retain(|key, _| key.chunk_coord != chunk_coord);
        self.last_used_tick.retain(|key, _| key.chunk_coord != chunk_coord);
        if let Some(store) = self.disk_store.as_mut() {
            // The disk store may hold this coord at LODs not currently resident, so
            // purge the resident-derived keys AND defensively re-derive nothing extra:
            // a chunk is only ever spilled under the same key it was resident at, and
            // the only LOD in use today is 0, so the resident-key sweep covers it. Purge
            // each known key, plus lod 0 unconditionally (the parked LOD seam).
            for key in &purged {
                store.remove(*key).expect("disk store remove must not fail");
            }
            store
                .remove(ChunkCacheKey::new(chunk_coord, 0))
                .expect("disk store remove must not fail");
        }
    }

    /// **Targeted invalidation (issue #27 S3).** Drop exactly the cached chunks whose
    /// half-open box intersects the edit world-AABB `edit_aabb` (in absolute voxels,
    /// the producer-true frame), at `voxels_per_block` — ADR 0002 Decision 3's
    /// whole-chunk dirty granularity. Every other cached chunk stays resident
    /// untouched.
    ///
    /// `edit_aabb` is what
    /// [`LeafSpatialIndex::edit_aabb_since`](voxel_core::spatial_index::LeafSpatialIndex::edit_aabb_since)
    /// returns: the union of the old and new boxes of whatever the edit changed
    /// (moved / added / removed / edited leaves), so a moved node dirties chunks
    /// around BOTH its source and destination. An empty `edit_aabb` (nothing changed)
    /// evicts nothing.
    ///
    /// A density mismatch against the cache's bound density is treated
    /// conservatively (the AABB was computed at a different chunk size) by clearing
    /// everything — but the caller [`main`] already falls back to `clear` for a
    /// density change, so this path is belt-and-braces.
    ///
    /// **Returns the set of chunk-coords actually evicted** (those that were
    /// resident AND intersected the edit AABB), so the GPU cache (issue #20 S6c) can
    /// later evict exactly those coords in lockstep with this resolve cache. The
    /// belt-and-braces density-mismatch path returns every coord that was resident
    /// before the clear. An empty edit AABB (or a clear of an empty cache) returns an
    /// empty `Vec`.
    ///
    /// [`main`]: crate
    pub fn invalidate_aabb(&mut self, edit_aabb: &VoxelAabb, voxels_per_block: u32) -> Vec<[i32; 3]> {
        if let Some(bound) = self.bound_density {
            if bound != voxels_per_block {
                // Density mismatch: everything is dropped, so the evicted set is
                // every resident coord (gathered before the clear).
                let evicted: Vec<[i32; 3]> = self.chunks.keys().map(|key| key.chunk_coord).collect();
                self.clear();
                return evicted;
            }
        }
        let Some((min_chunk, max_chunk)) = edit_aabb.covering_chunk_range(voxels_per_block) else {
            return Vec::new(); // empty edit AABB — nothing to invalidate.
        };
        // Resident coords inside the edit range — the reported (and dropped) set.
        let mut evicted = Vec::new();
        self.chunks.retain(|key, _| {
            let coord = key.chunk_coord;
            let inside = (0..3).all(|axis| coord[axis] >= min_chunk[axis] && coord[axis] <= max_chunk[axis]);
            if inside {
                evicted.push(coord);
            }
            !inside
        });
        self.last_used_tick.retain(|key, _| {
            let coord = key.chunk_coord;
            !(0..3).all(|axis| coord[axis] >= min_chunk[axis] && coord[axis] <= max_chunk[axis])
        });
        // Purge spilled chunks across the edit range too, so an evicted-then-spilled
        // chunk cannot reload stale after an edit (issue #20 Step 3). A spilled chunk
        // is NOT in `self.chunks`, so it does not appear in `evicted` (the reported
        // resident set is unchanged), but it must still be dropped from disk. The store
        // exposes no key iterator, so purge by walking the (bounded) edit coord range.
        if let Some(store) = self.disk_store.as_mut() {
            for chunk_x in min_chunk[0]..=max_chunk[0] {
                for chunk_y in min_chunk[1]..=max_chunk[1] {
                    for chunk_z in min_chunk[2]..=max_chunk[2] {
                        store
                            .remove(ChunkCacheKey::new([chunk_x, chunk_y, chunk_z], 0))
                            .expect("disk store remove must not fail");
                    }
                }
            }
        }
        evicted
    }

    /// Clear + re-bind the cache when the requested density OR floating origin differs
    /// from the one the resident chunks were resolved at. A chunk's voxel extent
    /// depends on density, and its stored positions are relative to the floating
    /// origin (ADR 0002 S4b), so a change in either invalidates every cached chunk.
    fn rebind_if_changed(&mut self, voxels_per_block: u32, floating_origin: [i64; 3]) {
        let density_matches = self.bound_density == Some(voxels_per_block);
        let origin_matches = self.bound_floating_origin == floating_origin;
        if density_matches && origin_matches {
            return;
        }
        if self.bound_density.is_some() && !(density_matches && origin_matches) {
            // Re-binding from a previous binding: drop the now-stale chunks from RAM,
            // disk and LRU state. A spilled chunk is keyed/serialised in the OLD
            // binding, so it must not survive a rebind (the S6c wiring-note correctness
            // condition — otherwise a far chunk would reload mis-placed; issue #20 Step 3).
            self.chunks.clear();
            self.last_used_tick.clear();
            if let Some(store) = self.disk_store.as_mut() {
                store.clear().expect("clearing the disk store must not fail");
            }
        }
        self.bound_density = Some(voxels_per_block);
        self.bound_floating_origin = floating_origin;
    }
}

/// The residency decision an incremental edit forces on a per-chunk render cache:
/// which chunks' buffers to (re)build, and which to drop. This is the store's
/// pure, GPU-free residency planner — set-difference glue over three coord sets,
/// with the eviction semantics (below) as the domain content. Relocated from the
/// renderer by ADR 0016 (retiring the store → renderer edge); it originated as
/// issue #20 S6c-2c.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct IncrementalRebuildPlan {
    /// Covering coords whose buffer must be (re)built: DIRTY (evicted by this edit)
    /// or NEW (no resident buffer yet). Their grids are the only resolve-cache
    /// MISSES; every other covering chunk is a HIT (byte-identical → keep).
    pub rebuild: Vec<[i32; 3]>,
    /// Resident coords the post-edit scene no longer covers (a removed/shrunk node
    /// vacated them) — their buffers must be dropped.
    pub evict: Vec<[i32; 3]>,
}

/// Compute the incremental dirty-chunk rebuild plan from coord sets alone (no GPU).
///
/// `resident` is the render cache's current coord set (only NON-empty chunks ever
/// hold a buffer — a zero-voxel chunk is never stored). `occupied_covering` is the
/// set of post-edit covering coords that resolve to a NON-EMPTY grid (so deserve a
/// buffer); empty covering chunks are excluded here so they are never treated as
/// "new" work nor kept resident. `evicted` is the edit's dirty coords from the
/// resolve cache (see [`Store::invalidate_aabb`]).
///
/// A coord is REBUILT iff it is occupied-covering AND (dirty OR not currently
/// resident). A resident coord is EVICTED iff it is no longer occupied-covering —
/// which captures BOTH a vacated chunk (a removed/shrunk node) AND a chunk that an
/// edit turned empty (dirty + now zero voxels). Occupied coords that are
/// resident-and-not-dirty are kept untouched (resolve-cache hits → byte-identical →
/// buffers already correct).
///
/// Applying this plan and making every rebuilt entry equal its fresh grid yields
/// EXACTLY the occupied-covering coord set with fresh contents — identical to a
/// wholesale rebuild (which also stores only non-empty chunks). The returned vectors
/// are sorted so the plan is deterministic and the rebuild count is order-independent.
pub fn incremental_rebuild_plan(
    resident: &[[i32; 3]],
    evicted: &[[i32; 3]],
    occupied_covering: &[[i32; 3]],
) -> IncrementalRebuildPlan {
    let resident_set: std::collections::HashSet<[i32; 3]> = resident.iter().copied().collect();
    let evicted_set: std::collections::HashSet<[i32; 3]> = evicted.iter().copied().collect();
    let covering_set: std::collections::HashSet<[i32; 3]> =
        occupied_covering.iter().copied().collect();

    let mut rebuild: Vec<[i32; 3]> = occupied_covering
        .iter()
        .copied()
        .filter(|coord| evicted_set.contains(coord) || !resident_set.contains(coord))
        .collect();
    rebuild.sort_unstable();
    rebuild.dedup();

    let mut evict: Vec<[i32; 3]> = resident
        .iter()
        .copied()
        .filter(|coord| !covering_set.contains(coord))
        .collect();
    evict.sort_unstable();
    evict.dedup();

    IncrementalRebuildPlan { rebuild, evict }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::core_geom::MaterialChoice;
    use crate::voxel::GeometryParams;
    use crate::scene::{
        DefId, Node, NodeContent, RegionBlocks,
    };
    use voxel_core::voxel::{ShapeKind, VoxelGrid};
    use crate::voxel::{SdfShape};

    /// Canonicalise an occupied set into a sorted multiset of
    /// `(bit_exact_voxel_position, material_id)`, so two resolves compare equal
    /// regardless of voxel emission ORDER but **byte-for-byte** on each `f32`
    /// position. Keying on the raw `f32` bits (`to_bits`) — not a rounded integer —
    /// means this asserts the bytes the renderer/mesher/fog consume are IDENTICAL,
    /// the S2 bit-identical-output guarantee (not merely the same rounded voxel
    /// set). A sub-ULP shift in any position fails the comparison.
    fn occupied_multiset(grid: &VoxelGrid) -> std::collections::BTreeMap<([u32; 3], u16), usize> {
        let mut multiset = std::collections::BTreeMap::new();
        for voxel in &grid.occupied {
            let position = voxel.world_position();
            let key = [
                position[0].to_bits(),
                position[1].to_bits(),
                position[2].to_bits(),
            ];
            *multiset.entry((key, voxel.color_index())).or_insert(0) += 1;
        }
        multiset
    }

    fn shape_scene(kind: ShapeKind, voxels_per_block: u32) -> Scene {
        Scene::from_geometry(
            GeometryParams {
                shape: kind,
                size_voxels: [5 * voxels_per_block, 5 * voxels_per_block, 5 * voxels_per_block],
                size_measurements: None,
                voxels_per_block,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        )
    }

    /// (a) A cache HIT returns a grid identical to a fresh `Scene::resolve_chunk`.
    #[test]
    fn cache_hit_matches_fresh_resolve_chunk() {
        let scene = shape_scene(ShapeKind::Sphere, 16);
        let mut cache = Store::new();
        let chunk_coord = [0, 0, 0];

        let fresh = scene.resolve_chunk(chunk_coord, 16, 0);

        // First call: a miss (resolves + stores).
        assert_eq!(cache.resident_chunk_count(), 0);
        let first = cache.chunk(chunk_coord, &scene, 16, 0).clone();
        assert_eq!(cache.resident_chunk_count(), 1);
        // Second call: a hit (no new resident chunk).
        let second = cache.chunk(chunk_coord, &scene, 16, 0).clone();
        assert_eq!(cache.resident_chunk_count(), 1, "a hit must not add a chunk");

        assert_eq!(first.dimensions, fresh.dimensions);
        assert_eq!(
            occupied_multiset(&first),
            occupied_multiset(&fresh),
            "a cached chunk must equal a fresh resolve_chunk"
        );
        assert_eq!(
            occupied_multiset(&second),
            occupied_multiset(&fresh),
            "a cache HIT must return the same grid as the miss"
        );
    }

    /// (b) The cache-assembled `resolve_region` output is IDENTICAL (occupied set +
    /// material_id, same recentre) to the monolithic `Scene::resolve_region` — for
    /// every required scene: all SDF shapes, demo-scene, demo-village.
    fn assert_cache_region_matches_monolithic(scene: &Scene, voxels_per_block: u32, label: &str) {
        let monolithic = scene.resolve_region(
            scene.full_extent_blocks(voxels_per_block),
            voxels_per_block,
            0,
        );
        let mut cache = Store::new();
        let assembled = cache.resolve_region(scene, voxels_per_block, 0);

        assert_eq!(
            assembled.dimensions, monolithic.dimensions,
            "[{label}] cache-assembled dimensions must match monolithic"
        );
        assert_eq!(
            assembled.occupied_count(),
            monolithic.occupied_count(),
            "[{label}] cache-assembled occupied count must match monolithic"
        );
        assert_eq!(
            occupied_multiset(&assembled),
            occupied_multiset(&monolithic),
            "[{label}] cache-assembled occupied set (position + material) must be \
             BIT-IDENTICAL to monolithic resolve_region (same recentre)"
        );
    }

    #[test]
    fn cache_region_matches_monolithic_for_all_shapes() {
        for kind in [
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Tube,
            ShapeKind::Torus,
            ShapeKind::Box,
        ] {
            let scene = shape_scene(kind, 16);
            assert_cache_region_matches_monolithic(&scene, 16, &format!("{kind:?}"));
        }
    }

    /// FLAT / odd-sized shapes (e.g. a 5×1×5 cylinder — the app default) are the
    /// regression case for the S0 covering-range bug S2 fixed: the producer centres
    /// its grid on the origin, so a 1-block (odd) axis straddles two chunks, but the
    /// old block-AABB covering range (`floor(size/2)` per block) missed one of them
    /// and dropped half the voxels. This pins that the cache covers the
    /// producer-true voxel extent and reassembles bit-identically.
    #[test]
    fn cache_region_matches_monolithic_for_flat_and_odd_shapes() {
        for kind in [ShapeKind::Cylinder, ShapeKind::Sphere, ShapeKind::Torus] {
            for size in [[5u32, 1, 5], [3, 1, 3], [5, 3, 5], [1, 1, 1]] {
                let scene = Scene::from_geometry(
                    GeometryParams {
                        shape: kind,
                        size_voxels: [size[0] * 16, size[1] * 16, size[2] * 16],
                        size_measurements: None,
                        voxels_per_block: 16,
                        wall_blocks: 1,
                    },
                    MaterialChoice::Stone,
                );
                assert_cache_region_matches_monolithic(&scene, 16, &format!("{kind:?} {size:?}"));
            }
        }
    }

    #[test]
    fn cache_region_matches_monolithic_for_demo_scene() {
        let voxels_per_block = 16;
        let make_tool = |kind, offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, voxels_per_block);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = crate::scene::NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let scene = Scene::from_nodes(vec![
            make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
            make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
            make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
        ]);
        assert_cache_region_matches_monolithic(&scene, voxels_per_block, "demo-scene");
    }

    #[test]
    fn cache_region_matches_monolithic_for_demo_village() {
        let voxels_per_block = 16;
        let house_def_id = DefId(1);
        let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, size, 1, voxels_per_block);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = crate::scene::NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let instance = |name: &str, offset: [i64; 3]| {
            let mut node = Node::new(name, NodeContent::Instance(house_def_id));
            node.transform = crate::scene::NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let mut scene = Scene::from_nodes(vec![
            instance("House 1", [0, 0, 0]),
            instance("House 2", [6, 0, 0]),
            instance("House 3", [12, 0, 0]),
            instance("House 4", [18, 0, 0]),
        ]);
        scene.add_definition(
            house_def_id,
            "House".to_string(),
            vec![
                tool(ShapeKind::Box, [2, 2, 2], [0, 0, 0], MaterialChoice::Stone),
                tool(ShapeKind::Cylinder, [1, 2, 1], [0, 2, 0], MaterialChoice::Wood),
            ],
        );
        assert_cache_region_matches_monolithic(&scene, voxels_per_block, "demo-village");
    }

    /// A density change clears + re-binds the cache (a chunk's voxel extent depends
    /// on density), and the re-resolve still matches the monolithic at the new
    /// density.
    #[test]
    fn density_change_rebinds_cache() {
        let scene = shape_scene(ShapeKind::Torus, 16);
        let mut cache = Store::new();
        let _ = cache.resolve_region(&scene, 16, 0);
        assert!(cache.resident_chunk_count() > 0);

        let scene_8 = shape_scene(ShapeKind::Torus, 8);
        let assembled_8 = cache.resolve_region(&scene_8, 8, 0);
        let monolithic_8 =
            scene_8.resolve_region(scene_8.full_extent_blocks(8), 8, 0);
        assert_eq!(
            occupied_multiset(&assembled_8),
            occupied_multiset(&monolithic_8),
            "after a density change the cache re-resolves correctly at the new density"
        );
    }

    /// `clear` empties the cache (the S3 invalidation seam).
    #[test]
    fn clear_empties_cache() {
        let scene = shape_scene(ShapeKind::Sphere, 16);
        let mut cache = Store::new();
        let _ = cache.chunk([0, 0, 0], &scene, 16, 0);
        assert!(cache.resident_chunk_count() > 0);
        cache.clear();
        assert_eq!(cache.resident_chunk_count(), 0, "clear drops every chunk");
    }

    /// (c) A synthetic scene whose TOTAL voxel count exceeds the old 6M whole-region
    /// cap, but whose individual chunks are each small, resolves successfully under
    /// the new PER-CHUNK bound — proving total scene size is no longer capped at 6M.
    ///
    /// The scene is two small boxes pushed to opposite corners of a cube spaced 16
    /// blocks apart on EVERY axis. The composite AABB is a 17³-block cube → at
    /// density 16 that is `(17·16)³ ≈ 20M` whole-region voxels (well past the old 6M
    /// total cap), yet only the two corner chunks hold any voxels and each holds one
    /// tiny box — far under the per-chunk bound.
    ///
    /// (Spreading the boxes DIAGONALLY rather than in a long row keeps the same
    /// "total ≫ 6M, chunks tiny" coverage while the covering-chunk grid stays a small
    /// ~5³ cube — the row-of-64 form this replaced spanned ~500 chunks on one axis
    /// and dominated the lib-test wall-time. See issue #27 S3.)
    #[test]
    fn scene_exceeding_old_total_cap_resolves_under_per_chunk_bound() {
        let voxels_per_block = 16u32;
        // Two 1-block stone cubes at opposite corners of a 16-block cube, so the
        // composite spans a huge cubic extent while each chunk holds at most one box.
        let spacing_blocks = 16i64;
        let shape = SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, voxels_per_block);
        let corner = |label: &str, offset: [i64; 3]| {
            let mut node = Node::new(
                label,
                NodeContent::Tool { shape: shape.clone(), material: MaterialChoice::Stone },
            );
            node.transform = crate::scene::NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let scene = Scene::from_nodes(vec![
            corner("Box lo", [0, 0, 0]),
            corner("Box hi", [spacing_blocks, spacing_blocks, spacing_blocks]),
        ]);

        // The OLD whole-region cap would reject this: the composite AABB voxel count
        // is far beyond 6M.
        let region = scene.full_extent_blocks(voxels_per_block);
        let whole_region_voxels = region.size_blocks[0] as u64
            * region.size_blocks[1] as u64
            * region.size_blocks[2] as u64
            * (voxels_per_block as u64).pow(3);
        assert!(
            whole_region_voxels > voxel_core::voxel::MAX_GRID_VOXELS,
            "the synthetic scene's whole-region voxel count ({whole_region_voxels}) must \
             exceed the OLD 6M total cap to prove the point"
        );

        // Every individual chunk is small (one small box at most) — under the new
        // per-chunk bound, so the lazy per-chunk resolve succeeds.
        let mut cache = Store::new();
        let (min_chunk, max_chunk) = scene
            .covering_chunk_range(voxels_per_block)
            .expect("a placed scene has a covering chunk range");
        let mut total_resolved = 0usize;
        for chunk_z in min_chunk[2]..=max_chunk[2] {
            for chunk_y in min_chunk[1]..=max_chunk[1] {
                for chunk_x in min_chunk[0]..=max_chunk[0] {
                    let chunk = cache.chunk([chunk_x, chunk_y, chunk_z], &scene, voxels_per_block, 0);
                    assert!(
                        (chunk.occupied_count() as u64) <= voxel_core::voxel::MAX_CHUNK_VOXELS,
                        "every chunk must stay under the per-chunk bound"
                    );
                    total_resolved += chunk.occupied_count();
                }
            }
        }
        assert!(
            total_resolved > 0,
            "the lazy per-chunk resolve must produce voxels for a scene the old total \
             cap would have rejected outright"
        );
    }

    /// One whose SINGLE chunk exceeds the per-chunk bound is still rejected (the cap
    /// did not simply vanish — it moved to per-chunk granularity).
    #[test]
    fn single_chunk_exceeding_per_chunk_bound_is_rejected() {
        // The per-chunk bound is the chunk's voxel CAPACITY (one chunk's voxel
        // extent cubed). A density large enough that one chunk's capacity exceeds
        // the bound must be rejected by the guard helper.
        let chunk_capacity_at = |voxels_per_block: u32| -> u64 {
            let extent = (voxel_core::core_geom::CHUNK_BLOCKS * voxels_per_block) as u64;
            extent * extent * extent
        };
        // Density 16: chunk extent = 64 voxels → 64³ = 262_144 voxels/chunk, well
        // under the bound — NOT rejected.
        assert!(chunk_capacity_at(16) <= voxel_core::voxel::MAX_CHUNK_VOXELS);
        assert!(
            !voxel_core::voxel::chunk_extent_exceeds_bound(16),
            "a normal density-16 chunk is under the per-chunk bound"
        );

        // A density whose single chunk capacity exceeds the bound IS rejected.
        // chunk extent = CHUNK_BLOCKS × density; pick a density making 64³·k > bound.
        let huge_density = 64u32; // extent = 256 → 256³ = 16_777_216 voxels/chunk.
        assert!(
            chunk_capacity_at(huge_density) > voxel_core::voxel::MAX_CHUNK_VOXELS,
            "the chosen huge density must make one chunk exceed the per-chunk bound"
        );
        assert!(
            voxel_core::voxel::chunk_extent_exceeds_bound(huge_density),
            "a chunk whose voxel capacity exceeds the per-chunk bound must be rejected"
        );
    }

    // ===== Issue #27 S3: targeted edit-AABB invalidation ========================

    fn three_tool_scene(voxels_per_block: u32, box_offset_x: i64) -> Scene {
        let make_tool = |kind, offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, voxels_per_block);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = crate::scene::NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let mut scene = Scene::from_nodes(vec![
            make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
            make_tool(ShapeKind::Box, [box_offset_x, 0, 0], MaterialChoice::Wood),
            make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
        ]);
        scene.voxels_per_block = voxels_per_block;
        scene
    }

    /// The set of chunk coords currently resident in the cache (for assertions).
    fn resident_coords(cache: &Store) -> std::collections::BTreeSet<[i32; 3]> {
        cache.chunks.keys().map(|key| key.chunk_coord).collect()
    }

    /// After an edit at offset X, exactly the chunks intersecting the edit AABB are
    /// evicted; every other chunk stays resident; and a re-resolve after the
    /// targeted invalidation yields a grid IDENTICAL to a full fresh resolve.
    #[test]
    fn targeted_invalidation_evicts_only_intersecting_chunks() {
        let density = 16u32;
        // A scene spread far enough in X that the Box occupies chunks no other leaf
        // touches (so moving it is a clean, localised edit).
        let scene_a = three_tool_scene(density, 40);
        let mut cache = Store::new();
        let _ = cache.resolve_region(&scene_a, density, 0);
        let all_resident = resident_coords(&cache);
        assert!(!all_resident.is_empty());

        // Move the Box from +40X to +80X. Compute the edit AABB via the spatial-index
        // diff, exactly as `main::rebuild_geometry` does.
        let mut scene_b = scene_a.clone();
        scene_b.root_node_mut(1).transform = crate::scene::NodeTransform::from_blocks([80, 0, 0], density);
        let index_a = scene_a.build_leaf_spatial_index(density);
        let index_b = scene_b.build_leaf_spatial_index(density);
        let edit_aabb = index_b.edit_aabb_since(&index_a).expect("same density");

        // The chunks the edit AABB intersects — the EXPECTED evicted set (those that
        // were resident).
        let (min_chunk, max_chunk) = edit_aabb
            .covering_chunk_range(density)
            .expect("a non-empty edit AABB has a chunk range");
        let mut expected_evicted = std::collections::BTreeSet::new();
        for &coord in &all_resident {
            let inside = (0..3).all(|axis| coord[axis] >= min_chunk[axis] && coord[axis] <= max_chunk[axis]);
            if inside {
                expected_evicted.insert(coord);
            }
        }
        assert!(!expected_evicted.is_empty(), "the move must dirty at least one resident chunk");

        cache.invalidate_aabb(&edit_aabb, density);
        let after = resident_coords(&cache);

        // Every expected-evicted chunk is gone; every other chunk is still resident.
        for coord in &expected_evicted {
            assert!(!after.contains(coord), "chunk {coord:?} intersecting the edit must be evicted");
        }
        for coord in &all_resident {
            if !expected_evicted.contains(coord) {
                assert!(after.contains(coord), "chunk {coord:?} outside the edit must stay resident");
            }
        }

        // A re-resolve after targeted invalidation == a full fresh resolve of B.
        let reresolved = cache.resolve_region(&scene_b, density, 0);
        let mut fresh_cache = Store::new();
        let fresh = fresh_cache.resolve_region(&scene_b, density, 0);
        assert_eq!(
            occupied_multiset(&reresolved),
            occupied_multiset(&fresh),
            "re-resolve after targeted invalidation must equal a full fresh resolve"
        );
    }

    /// Moving a node from A to B invalidates chunks around BOTH A and B (the diff
    /// unions the old and new boxes).
    #[test]
    fn move_invalidates_chunks_around_both_endpoints() {
        let density = 16u32;
        let scene_a = three_tool_scene(density, 40);
        let mut cache = Store::new();
        let _ = cache.resolve_region(&scene_a, density, 0);

        let mut scene_b = scene_a.clone();
        scene_b.root_node_mut(1).transform = crate::scene::NodeTransform::from_blocks([80, 0, 0], density);
        let index_a = scene_a.build_leaf_spatial_index(density);
        let index_b = scene_b.build_leaf_spatial_index(density);
        let edit_aabb = index_b.edit_aabb_since(&index_a).expect("same density");

        // The chunk owning the OLD Box centre (40·16 = 640 voxels) and the chunk
        // owning the NEW centre (80·16 = 1280 voxels) must BOTH be in the edit range.
        let chunk_extent = (voxel_core::core_geom::CHUNK_BLOCKS * density) as i32;
        let old_chunk_x = (640i32).div_euclid(chunk_extent);
        let new_chunk_x = (1280i32).div_euclid(chunk_extent);
        let (min_chunk, max_chunk) = edit_aabb.covering_chunk_range(density).unwrap();
        assert!(min_chunk[0] <= old_chunk_x && old_chunk_x <= max_chunk[0], "edit range must cover OLD chunk");
        assert!(min_chunk[0] <= new_chunk_x && new_chunk_x <= max_chunk[0], "edit range must cover NEW chunk");
    }

    /// An empty edit AABB (nothing changed) evicts nothing.
    #[test]
    fn empty_edit_aabb_evicts_nothing() {
        let density = 16u32;
        let scene = three_tool_scene(density, 8);
        let mut cache = Store::new();
        let _ = cache.resolve_region(&scene, density, 0);
        let before = resident_coords(&cache);
        let empty = voxel_core::spatial_index::VoxelAabb::new([0, 0, 0], [0, 0, 0]);
        let evicted = cache.invalidate_aabb(&empty, density);
        assert!(evicted.is_empty(), "an empty edit AABB reports an empty evicted set");
        assert_eq!(resident_coords(&cache), before, "an empty edit AABB evicts nothing");
    }

    /// **S6c-2a: the evicted-set return.** `invalidate_aabb` returns exactly the
    /// coords spanned by the edit AABB's `covering_chunk_range` that were resident —
    /// the same set the cache actually drops — so the GPU cache can evict in lockstep.
    #[test]
    fn invalidate_aabb_returns_exactly_the_evicted_coords() {
        let density = 16u32;
        let scene_a = three_tool_scene(density, 40);
        let mut cache = Store::new();
        let _ = cache.resolve_region(&scene_a, density, 0);
        let all_resident = resident_coords(&cache);

        let mut scene_b = scene_a.clone();
        scene_b.root_node_mut(1).transform = crate::scene::NodeTransform::from_blocks([80, 0, 0], density);
        let index_a = scene_a.build_leaf_spatial_index(density);
        let index_b = scene_b.build_leaf_spatial_index(density);
        let edit_aabb = index_b.edit_aabb_since(&index_a).expect("same density");

        // The expected evicted set: resident coords inside the edit's chunk range.
        let (min_chunk, max_chunk) = edit_aabb
            .covering_chunk_range(density)
            .expect("a non-empty edit AABB has a chunk range");
        let mut expected: std::collections::BTreeSet<[i32; 3]> = std::collections::BTreeSet::new();
        for &coord in &all_resident {
            let inside = (0..3).all(|axis| coord[axis] >= min_chunk[axis] && coord[axis] <= max_chunk[axis]);
            if inside {
                expected.insert(coord);
            }
        }
        assert!(!expected.is_empty(), "the move must dirty at least one resident chunk");

        let returned: std::collections::BTreeSet<[i32; 3]> =
            cache.invalidate_aabb(&edit_aabb, density).into_iter().collect();
        assert_eq!(
            returned, expected,
            "the returned evicted set must equal exactly the resident coords inside \
             the edit AABB's covering_chunk_range"
        );
        // And the returned set is exactly what was dropped.
        let after = resident_coords(&cache);
        for coord in &returned {
            assert!(!after.contains(coord), "a returned coord must no longer be resident");
        }
        assert_eq!(
            after.len() + returned.len(),
            all_resident.len(),
            "evicted + remaining must partition the originally-resident set"
        );
    }

    /// A density mismatch (the belt-and-braces clear path) reports EVERY resident
    /// coord as evicted.
    #[test]
    fn invalidate_aabb_density_mismatch_reports_all_resident_evicted() {
        let scene = three_tool_scene(16, 8);
        let mut cache = Store::new();
        let _ = cache.resolve_region(&scene, 16, 0);
        let before = resident_coords(&cache);
        assert!(!before.is_empty());

        // Invalidate at a DIFFERENT density than the cache is bound to → clear path.
        let aabb = voxel_core::spatial_index::VoxelAabb::new([0, 0, 0], [16, 16, 16]);
        let returned: std::collections::BTreeSet<[i32; 3]> =
            cache.invalidate_aabb(&aabb, 8).into_iter().collect();
        assert_eq!(returned, before, "a density mismatch evicts (and reports) every resident coord");
        assert_eq!(cache.resident_chunk_count(), 0, "the cache is cleared");
    }

    // ===== Issue #20 S6c step 4: per-chunk render accessor ========================

    /// (S6c-2a parity) The union of `resident_render_chunks` (occupied cells +
    /// material_id, in each chunk's rebased frame) equals `resolve_region`'s
    /// assembled grid BYTE-FOR-BYTE, AND each returned coord is the absolute chunk
    /// coord that owns its grid's voxels, AND the coord set equals the scene's
    /// `covering_chunk_range`.
    fn assert_render_chunks_match_resolve_region(scene: &Scene, voxels_per_block: u32, label: &str) {
        // The truth: the assembled monolithic grid the renderer consumes today.
        let mut region_cache = Store::new();
        let assembled = region_cache.resolve_region(scene, voxels_per_block, 0);

        let mut render_cache = Store::new();
        let chunks = render_cache.resident_render_chunks(scene, voxels_per_block, 0);

        // Parity: the union of the per-chunk grids' occupied sets (already rebased,
        // same frame as the assembled grid) is bit-identical to the assembled grid.
        let mut union: std::collections::BTreeMap<([u32; 3], u16), usize> =
            std::collections::BTreeMap::new();
        for (_coord, grid) in &chunks {
            for (key, count) in occupied_multiset(grid) {
                *union.entry(key).or_insert(0) += count;
            }
        }
        assert_eq!(
            union,
            occupied_multiset(&assembled),
            "[{label}] union of resident_render_chunks must be BIT-IDENTICAL to \
             resolve_region's assembled grid (same rebased frame)"
        );

        // Coord set equals the scene's covering_chunk_range.
        let returned_coords: std::collections::BTreeSet<[i32; 3]> =
            chunks.iter().map(|(coord, _)| *coord).collect();
        let mut expected_coords: std::collections::BTreeSet<[i32; 3]> =
            std::collections::BTreeSet::new();
        if let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(voxels_per_block) {
            for chunk_z in min_chunk[2]..=max_chunk[2] {
                for chunk_y in min_chunk[1]..=max_chunk[1] {
                    for chunk_x in min_chunk[0]..=max_chunk[0] {
                        expected_coords.insert([chunk_x, chunk_y, chunk_z]);
                    }
                }
            }
        }
        assert_eq!(
            returned_coords, expected_coords,
            "[{label}] returned coord set must equal the scene's covering_chunk_range"
        );

        // Coord correctness: each returned coord is the absolute chunk coord that
        // owns its grid's voxels. The accessor binds to the recentre, so a chunk
        // coord `c` owns rebased voxels in `[c·E - recentre, (c+1)·E - recentre)`.
        let chunk_extent = (voxel_core::core_geom::CHUNK_BLOCKS * voxels_per_block) as i64;
        let recentre = scene.recentre_voxels_for_resolve(voxels_per_block).voxels();
        for (coord, grid) in &chunks {
            for voxel in &grid.occupied {
                let position = voxel.world_position();
                for axis in 0..3 {
                    // Rebased absolute voxel index = floor(position) + recentre.
                    let absolute = position[axis].floor() as i64 + recentre[axis];
                    let owner = absolute.div_euclid(chunk_extent) as i32;
                    assert_eq!(
                        owner, coord[axis],
                        "[{label}] voxel at {:?} (axis {axis}) must be owned by chunk \
                         coord {coord:?}, not {owner}",
                        position
                    );
                }
            }
        }
    }

    #[test]
    fn render_chunks_match_resolve_region_for_all_shapes() {
        for kind in [
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Tube,
            ShapeKind::Torus,
            ShapeKind::Box,
        ] {
            let scene = shape_scene(kind, 16);
            assert_render_chunks_match_resolve_region(&scene, 16, &format!("{kind:?}"));
        }
    }

    #[test]
    fn render_chunks_match_resolve_region_for_demo_scene() {
        let scene = three_tool_scene(16, 8);
        assert_render_chunks_match_resolve_region(&scene, 16, "demo-scene");
    }

    #[test]
    fn render_chunks_match_resolve_region_for_demo_village() {
        let vpb = 16u32;
        let house_def_id = DefId(1);
        let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, size, 1, vpb);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = crate::scene::NodeTransform::from_blocks(offset, vpb);
            node
        };
        let instance = |name: &str, offset: [i64; 3]| {
            let mut node = Node::new(name, NodeContent::Instance(house_def_id));
            node.transform = crate::scene::NodeTransform::from_blocks(offset, vpb);
            node
        };
        let mut scene = Scene::from_nodes(vec![
            instance("House 1", [0, 0, 0]),
            instance("House 2", [6, 0, 0]),
            instance("House 3", [12, 0, 0]),
            instance("House 4", [18, 0, 0]),
        ]);
        scene.add_definition(
            house_def_id,
            "House".to_string(),
            vec![
                tool(ShapeKind::Box, [2, 2, 2], [0, 0, 0], MaterialChoice::Stone),
                tool(ShapeKind::Cylinder, [1, 2, 1], [0, 2, 0], MaterialChoice::Wood),
            ],
        );
        assert_render_chunks_match_resolve_region(&scene, vpb, "demo-village");
    }

    /// A Part-only scene (no composite extent) yields an empty render-chunk set,
    /// matching `resolve_region`'s empty grid.
    #[test]
    fn render_chunks_empty_for_part_only_scene() {
        let scene = Scene::single_node(Node::new(
            "Clouds",
            NodeContent::Part(crate::scene::Part::DebugClouds { seed: 0 }),
        ));
        let mut cache = Store::new();
        let chunks = cache.resident_render_chunks(&scene, 16, 0);
        assert!(chunks.is_empty(), "a Part-only scene has no covering chunks");
    }

    /// **ADR 0002 S4b — origin-rebased rendering, far-offset precision.** A box
    /// placed a HUGE distance from the origin must resolve to a grid whose voxel
    /// positions are **byte-identical** to the SAME box at the origin — because the
    /// render frame is rebased to the floating origin (= the composite recentre) in
    /// i64 BEFORE the f32 downcast, so the absolute distance never reaches the f32
    /// data.
    ///
    /// The offset is **1_000_000 blocks** = 16_000_000 voxels at density 16, PAST the
    /// f32 exact-integer ceiling (2²⁴ ≈ 16.7M). Under the OLD recentre-AFTER-f32-add
    /// path the absolute position `local + 1.6e7` lost the voxel-centre `.5` on EVERY
    /// voxel (the S1 far-lands jitter — verified at ~13% of the 3D viewport in the
    /// headless render). This test is the durable CPU regression guard that the
    /// rebased path keeps far == near to the LAST BIT (replacing S1's degraded
    /// far-offset behaviour). The bit-exact key (`f32::to_bits`) fails on any sub-ULP
    /// shift, so it would catch a regression that a rounded-voxel comparison misses.
    #[test]
    fn far_offset_resolves_byte_identical_to_near_after_rebase() {
        let vpb = 16u32;
        let box_scene = |offset_x: i64| -> Scene {
            let shape = SdfShape::from_blocks(ShapeKind::Box, [4, 4, 4], 1, vpb);
            let mut node = Node::new(
                "box",
                NodeContent::Tool { shape, material: MaterialChoice::Stone },
            );
            node.transform = crate::scene::NodeTransform::from_blocks([offset_x, 0, 0], vpb);
            Scene::single_node(node)
        };

        let mut near_cache = Store::new();
        let near = near_cache.resolve_region(&box_scene(0), vpb, 0);
        // 1_000_000 blocks → 16M voxels, past the f32 exact-integer ceiling.
        let mut far_cache = Store::new();
        let far = far_cache.resolve_region(&box_scene(1_000_000), vpb, 0);

        assert_eq!(near.occupied_count(), far.occupied_count(), "same shape");
        assert!(near.occupied_count() > 0, "the box must resolve to voxels");
        // Every voxel-centre `.5` fraction must survive the rebase (would be lost to
        // f32 rounding at 1.6e7 under the old subtract-AFTER-f32 path).
        for voxel in &far.occupied {
            let position = voxel.world_position();
            for axis in 0..3 {
                let frac = position[axis].fract().abs();
                assert!(
                    (frac - 0.5).abs() < 1e-4,
                    "far voxel centre lost its .5 fraction (f32 jitter): {:?}",
                    position
                );
            }
        }
        assert_eq!(
            occupied_multiset(&far),
            occupied_multiset(&near),
            "the far box must resolve BYTE-IDENTICAL to the near box — the rebase \
             subtracts the floating origin in i64 before the f32 downcast, so the \
             absolute distance never degrades the rendered f32 positions (S4b)"
        );
    }

    /// A Part-only scene (no intrinsic-size leaf) resolves to an empty recentred
    /// grid through the cache, exactly as monolithic `resolve_region` does.
    #[test]
    fn part_only_scene_resolves_empty_through_cache() {
        let scene = Scene::single_node(Node::new(
            "Clouds",
            NodeContent::Part(crate::scene::Part::DebugClouds { seed: 0 }),
        ));
        let mut cache = Store::new();
        let assembled = cache.resolve_region(&scene, 16, 0);
        // A Part-only scene has no composite AABB → resolve_region returns a
        // zero-sized empty grid; the cache path matches.
        let monolithic = scene.resolve_region(RegionBlocks::new([0, 0, 0]), 16, 0);
        assert_eq!(assembled.occupied_count(), monolithic.occupied_count());
        assert_eq!(assembled.occupied_count(), 0);
    }

    // ===== Issue #20 S6d: region-scoped consumers =================================

    /// The whole-grid diameter readout for a scene's full active region — the
    /// reference value the region-scoped variants must reproduce.
    fn whole_grid_widest_run(scene: &Scene, vpb: u32, band: (u32, u32)) -> u32 {
        let region = scene.full_extent_blocks(vpb);
        let grid = scene.resolve_region(region, vpb, 0);
        grid.widest_run_in_band(band.0, band.1)
    }

    /// The cache's region-scoped `widest_run_in_band` returns the SAME value as the
    /// whole-grid `VoxelGrid::widest_run_in_band` for every required scene, across
    /// several layer bands.
    fn assert_region_widest_run_matches_whole_grid(scene: &Scene, vpb: u32, label: &str) {
        let dims = scene.placed_region_dimensions(vpb);
        // Z-up: layers are Z-slices, so the band spans the Z dimension (index 2).
        let grid_z = dims[2];
        // A spread of bands: the whole stack, the bottom layer, the top layer, the
        // exact mid-Z layer (the old slice), a thin interior band, and an
        // out-of-range band (above the grid → empty).
        let mid = grid_z.saturating_sub(1) / 2;
        let bands = [
            (0, grid_z.saturating_sub(1)),
            (0, 0),
            (grid_z.saturating_sub(1), grid_z.saturating_sub(1)),
            (mid, mid),
            (mid, (mid + 2).min(grid_z.saturating_sub(1))),
            (grid_z + 10, grid_z + 20),
        ];
        for band in bands {
            let expected = whole_grid_widest_run(scene, vpb, band);
            let mut cache = Store::new();
            let actual = cache.widest_run_in_band(scene, vpb, 0, band.0, band.1);
            assert_eq!(
                actual, expected,
                "[{label}] region widest_run_in_band band {band:?} must equal whole-grid"
            );
        }
    }

    /// **Far-offset diameter (issue #20 Step 2).** Two 3-block boxes 20,000 blocks
    /// apart on X: the composite is centred ~10,000 blocks out, so each box sits
    /// ~160,000 voxels from the recentred origin — far beyond any object the camera
    /// frames, while keeping the whole-grid reference cheap. The live diameter readout
    /// now routes through
    /// the region-scoped `widest_run_in_band`; it must report the box's TRUE width (a
    /// 48-voxel face row), confirming the rewired readout is correct far from the
    /// origin. It also equals the whole-grid value (the parity reference) — the two
    /// stay in lockstep until the region grid exceeds ~2^24 voxels on an axis, beyond
    /// which f32 collapses both identically (see the export test's NOTE).
    #[test]
    fn region_widest_run_correct_at_far_offset() {
        let vpb = 16u32;
        let make_box = |offset: [i64; 3]| {
            let shape = SdfShape::from_blocks(ShapeKind::Box, [3, 3, 3], 1, vpb);
            let mut node = Node::new("box", NodeContent::Tool { shape, material: MaterialChoice::Stone });
            node.transform = crate::scene::NodeTransform::from_blocks(offset, vpb);
            node
        };
        // 20,000-block separation → composite centred ~10,000 blocks out → each box
        // ~160,000 voxels from the origin (far beyond any normal scene), while the
        // whole-grid reference (an O(grid_x)-per-row bitset) stays cheap to assemble.
        let scene = Scene::from_nodes(vec![make_box([0, 0, 0]), make_box([20_000, 0, 0])]);

        let dims = scene.placed_region_dimensions(vpb);
        // Z-up: layers are Z-slices, so the band spans the Z stack (both boxes at z=0).
        let band = (0, dims[2].saturating_sub(1));
        let true_box_width = 3 * vpb; // each box spans a full 48-voxel face row.

        let mut cache = Store::new();
        let region = cache.widest_run_in_band(&scene, vpb, 0, band.0, band.1);
        assert_eq!(
            region, true_box_width,
            "region widest_run must report the box's true 48-voxel width at far offset"
        );
        // And it equals the whole-grid reference at this (still f32-safe) far offset.
        assert_eq!(
            region,
            whole_grid_widest_run(&scene, vpb, band),
            "region widest_run must equal whole-grid at far offset"
        );
    }

    #[test]
    fn region_widest_run_matches_whole_grid_for_all_shapes() {
        for kind in [
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Tube,
            ShapeKind::Torus,
            ShapeKind::Box,
        ] {
            let scene = shape_scene(kind, 16);
            assert_region_widest_run_matches_whole_grid(&scene, 16, &format!("{kind:?}"));
        }
    }

    #[test]
    fn region_widest_run_matches_whole_grid_for_demo_scene() {
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
        assert_region_widest_run_matches_whole_grid(&scene, vpb, "demo-scene");
    }

    #[test]
    fn region_widest_run_matches_whole_grid_for_demo_village() {
        let vpb = 16u32;
        let house_def_id = DefId(1);
        let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, size, 1, vpb);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = crate::scene::NodeTransform::from_blocks(offset, vpb);
            node
        };
        let instance = |name: &str, offset: [i64; 3]| {
            let mut node = Node::new(name, NodeContent::Instance(house_def_id));
            node.transform = crate::scene::NodeTransform::from_blocks(offset, vpb);
            node
        };
        let mut scene = Scene::from_nodes(vec![
            instance("House 1", [0, 0, 0]),
            instance("House 2", [6, 0, 0]),
            instance("House 3", [12, 0, 0]),
            instance("House 4", [18, 0, 0]),
        ]);
        scene.add_definition(
            house_def_id,
            "House".to_string(),
            vec![
                tool(ShapeKind::Box, [2, 2, 2], [0, 0, 0], MaterialChoice::Stone),
                tool(ShapeKind::Cylinder, [1, 2, 1], [0, 2, 0], MaterialChoice::Wood),
            ],
        );
        assert_region_widest_run_matches_whole_grid(&scene, vpb, "demo-village");
    }

    /// **The cross-seam stitching case (the one that catches a stitching bug).** A
    /// long thin horizontal bar that deliberately spans MANY chunks on X: a box of
    /// 20 blocks × density 16 = 320 voxels wide, while a chunk is `CHUNK_BLOCKS=4 ×
    /// 16 = 64` voxels wide — so the bar crosses ~5 chunk seams. The widest run in a
    /// band through the bar must be the FULL bar width (one contiguous run), not the
    /// per-chunk fragment width. A naive per-chunk-max-then-combine implementation
    /// would report ~64 (one chunk's worth); the correct stitched answer equals the
    /// whole-grid run. We assert both: region == whole-grid AND the run is wider than
    /// a single chunk's voxel extent (proving the seam was actually crossed).
    #[test]
    fn region_widest_run_stitches_runs_across_chunk_seams() {
        let vpb = 16u32;
        let bar_blocks_x = 20u32; // 20 × 16 = 320 voxels wide.
        let shape = SdfShape::from_blocks(ShapeKind::Box, [bar_blocks_x, 1, 1], 1, vpb);
        let scene = Scene::from_nodes(vec![Node::new(
            "bar",
            NodeContent::Tool { shape, material: MaterialChoice::Stone },
        )]);

        let dims = scene.placed_region_dimensions(vpb);
        // Z-up: layers are Z-slices, so the band spans the Z dimension (index 2).
        let band = (0, dims[2].saturating_sub(1));

        let expected = whole_grid_widest_run(&scene, vpb, band);
        let mut cache = Store::new();
        let actual = cache.widest_run_in_band(&scene, vpb, 0, band.0, band.1);

        let chunk_extent_voxels = voxel_core::core_geom::CHUNK_BLOCKS * vpb; // 64
        assert!(
            expected > chunk_extent_voxels,
            "the bar's widest run ({expected}) must exceed one chunk's voxel extent \
             ({chunk_extent_voxels}) so the run genuinely crosses chunk seams"
        );
        assert_eq!(
            actual, expected,
            "region widest_run must stitch the run across chunk seams to equal the \
             whole-grid full-bar width"
        );
        // And the bar is the full grid width (a solid 320-voxel box row).
        assert_eq!(actual, dims[0], "the bar fills the whole X extent");
    }

    /// Single-voxel and empty bands: a 1×1×1 box (one voxel) reports a widest run of
    /// 1 in its band and 0 outside it; the region variant matches the whole grid.
    #[test]
    fn region_widest_run_single_voxel_and_empty_band() {
        let vpb = 16u32;
        // A 1-block box at density 16 is a 16³ solid; pick density 1 for a true
        // single voxel so the run is exactly 1.
        let shape = SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, 1);
        let _ = vpb;
        let scene = Scene::from_nodes(vec![Node::new(
            "dot",
            NodeContent::Tool { shape, material: MaterialChoice::Stone },
        )]);
        let dims = scene.placed_region_dimensions(1);
        assert_eq!(dims, [1, 1, 1], "a 1×1×1@1 box is a single voxel");

        // In-band: widest run 1.
        let expected_in = whole_grid_widest_run(&scene, 1, (0, 0));
        let mut cache = Store::new();
        let actual_in = cache.widest_run_in_band(&scene, 1, 0, 0, 0);
        assert_eq!(expected_in, 1);
        assert_eq!(actual_in, expected_in);

        // Out-of-range band: empty → 0.
        let expected_out = whole_grid_widest_run(&scene, 1, (5, 9));
        let mut cache2 = Store::new();
        let actual_out = cache2.widest_run_in_band(&scene, 1, 0, 5, 9);
        assert_eq!(expected_out, 0);
        assert_eq!(actual_out, expected_out);

        // A wholly empty scene (Part-only, no occupied voxels): region run is 0.
        let empty_scene = Scene::single_node(Node::new(
            "Clouds",
            NodeContent::Part(crate::scene::Part::DebugClouds { seed: 0 }),
        ));
        let mut cache3 = Store::new();
        assert_eq!(cache3.widest_run_in_band(&empty_scene, 16, 0, 0, 100), 0);
    }

    // ===== Issue #20 S6c-2c: incremental dirty-chunk rebuild ======================

    /// A per-chunk GPU instance cache, MODELLED on CPU as `coord → that chunk's
    /// occupied multiset` (the multiset is the byte-identical proxy for the GPU
    /// buffer's contents — `renderer::instances_for_chunk` builds one VoxelInstance
    /// per occupied voxel, so two chunks with equal occupied multisets produce
    /// byte-identical instance buffers). This lets the incremental-rebuild decision
    /// logic ([`incremental_rebuild_plan`], the EXACT function the GPU path
    /// uses) be exercised without a wgpu device, while still proving the post-edit
    /// cache CONTENTS match a full rebuild.
    type RenderCache = std::collections::BTreeMap<[i32; 3], ChunkMultiset>;
    type ChunkMultiset = std::collections::BTreeMap<([u32; 3], u16), usize>;

    /// Build the render cache a WHOLESALE rebuild produces for `scene`: every
    /// covering chunk's grid as a multiset (skipping zero-voxel chunks, exactly as
    /// `renderer::rebuild_chunk` drops them — no buffer is allocated for an empty
    /// chunk).
    fn full_render_cache(scene: &Scene, density: u32) -> RenderCache {
        let mut cache = Store::new();
        let chunks = cache.resident_render_chunks(scene, density, 0);
        chunks
            .iter()
            .filter(|(_, grid)| !grid.occupied.is_empty())
            .map(|(coord, grid)| (*coord, occupied_multiset(grid)))
            .collect()
    }

    /// Apply ONE incremental edit (scene_a → scene_b) to `render_cache` IN PLACE,
    /// driving the GPU-cache decisions through [`incremental_rebuild_plan`]
    /// — the same plan `VoxelRenderer::incremental_rebuild_from_chunks` applies.
    /// Returns the number of chunks rebuilt (the observability count). The resolve
    /// cache (`resolve_cache`) carries state across edits exactly as the live app's
    /// does, so a HIT chunk is reused verbatim.
    fn apply_incremental_edit(
        render_cache: &mut RenderCache,
        resolve_cache: &mut Store,
        scene_a: &Scene,
        scene_b: &Scene,
        density: u32,
    ) -> usize {
        // 1. Edit AABB → evicted (dirty) coords, exactly as main::rebuild_geometry.
        let index_a = scene_a.build_leaf_spatial_index(density);
        let index_b = scene_b.build_leaf_spatial_index(density);
        let edit_aabb = index_b
            .edit_aabb_since(&index_a)
            .expect("same-density localisable edit");
        let evicted = resolve_cache.invalidate_aabb(&edit_aabb, density);

        // A recentre shift rebases EVERY chunk's contents, so the incremental path is
        // invalid — main::rebuild_geometry falls back to a full rebuild. Model that.
        let recentre_changed = scene_a.recentre_voxels_for_resolve(density)
            != scene_b.recentre_voxels_for_resolve(density);

        // 2. Freshly-resolved covering chunks for scene B (resolves the dirty/new
        //    chunks, reuses HITs).
        let render_chunks = resolve_cache.resident_render_chunks(scene_b, density, 0);

        if recentre_changed {
            // Full rebuild: clear + restore every non-empty covering chunk.
            render_cache.clear();
            for (coord, grid) in &render_chunks {
                if !grid.occupied.is_empty() {
                    render_cache.insert(*coord, occupied_multiset(grid));
                }
            }
            return render_chunks.len();
        }

        let resident: Vec<[i32; 3]> = render_cache.keys().copied().collect();
        // Only NON-EMPTY covering chunks deserve a buffer (matching the renderer).
        let occupied_covering: Vec<[i32; 3]> = render_chunks
            .iter()
            .filter(|(_, grid)| !grid.occupied.is_empty())
            .map(|(coord, _)| *coord)
            .collect();

        // 3. The plan — the SAME pure function the renderer drives the GPU from.
        let plan = incremental_rebuild_plan(&resident, &evicted, &occupied_covering);

        // 4. Rebuild only the planned coords (dirty ∪ new); evict the vacated ones.
        let rebuild_set: std::collections::BTreeSet<[i32; 3]> =
            plan.rebuild.iter().copied().collect();
        for (coord, grid) in &render_chunks {
            if rebuild_set.contains(coord) {
                render_cache.insert(*coord, occupied_multiset(grid));
            }
        }
        for coord in &plan.evict {
            render_cache.remove(coord);
        }
        plan.rebuild.len()
    }

    /// A tool node at the given offset, for building edit scenes.
    fn tool_node(kind: ShapeKind, size: [u32; 3], offset: [i64; 3], material: MaterialChoice) -> Node {
        let shape = SdfShape::from_blocks(kind, size, 1, 16);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = crate::scene::NodeTransform::from_blocks(offset, 16);
        node
    }

    /// **The key S6c-2c correctness test.** For a sequence of localised edits, the
    /// render cache built INCREMENTALLY (rebuild only dirty/new chunks, evict
    /// vacated) is IDENTICAL — coord set AND every chunk's instance multiset — to a
    /// full wholesale rebuild of the post-edit scene. Proves no stale chunk survives
    /// and no fresh chunk is missed. Also asserts the dirty-chunk count is STRICTLY
    /// LESS than the total resident count for a localised edit (so it is genuinely
    /// incremental, not a disguised full rebuild).
    #[test]
    fn incremental_rebuild_equals_full_rebuild_for_every_edit_kind() {
        let density = 16u32;

        // The base scene: three tools spread far apart in X so each occupies chunks
        // the others don't touch (clean localised edits). Start the render + resolve
        // caches as a wholesale build of scene A.
        // Two STATIC anchor nodes at the X extremes pin the composite extent (hence
        // the recentre / floating origin) so the interior edits below keep it FIXED —
        // that is the regime where the incremental dirty-only path is valid (a
        // recentre shift rebases every chunk and forces a full rebuild instead; see
        // `apply_incremental_edit`). The interior "subject" box sits between them.
        let anchor_lo = || tool_node(ShapeKind::Sphere, [5, 5, 5], [0, 0, 0], MaterialChoice::Stone);
        let anchor_hi = || tool_node(ShapeKind::Torus, [5, 5, 5], [120, 0, 0], MaterialChoice::Plain);
        let scene_a = Scene::from_nodes(vec![
            anchor_lo(),
            tool_node(ShapeKind::Box, [5, 5, 5], [60, 0, 0], MaterialChoice::Wood),
            anchor_hi(),
        ]);

        // Each case mutates scene_a → scene_b by ONE edit kind, all keeping the
        // composite extent (recentre) fixed via the anchors, so all are genuinely
        // incremental. Each is checked independently from a fresh wholesale build of A.
        let recolor = {
            let mut b = scene_a.clone();
            // In-place recolor of the interior Box (material change, same geometry).
            if let NodeContent::Tool { material, .. } = &mut b.root_node_mut(1).content {
                *material = MaterialChoice::Stone;
            }
            ("recolor", b)
        };
        let resize = {
            let mut b = scene_a.clone();
            // In-place resize of the interior Box (few dirty chunks around it).
            // Replace content + transform in place so the node keeps its arena id.
            let replacement = tool_node(ShapeKind::Box, [3, 3, 3], [60, 0, 0], MaterialChoice::Wood);
            let slot = b.root_node_mut(1);
            slot.content = replacement.content;
            slot.transform = replacement.transform;
            ("resize", b)
        };
        let move_node = {
            let mut b = scene_a.clone();
            // Move the interior Box from +60X to +70X (still interior → recentre
            // fixed; dirty around BOTH endpoints).
            b.root_node_mut(1).transform = crate::scene::NodeTransform::from_blocks([70, 0, 0], density);
            ("move", b)
        };
        let add_node = {
            let mut b = scene_a.clone();
            // ADD a new INTERIOR tool (brand-new covering chunks; extent unchanged).
            b.add_node(tool_node(ShapeKind::Box, [3, 3, 3], [90, 0, 0], MaterialChoice::Stone));
            ("add", b)
        };
        let remove_node = {
            let mut b = scene_a.clone();
            // REMOVE the interior Box (its chunks must be evicted/vacated; the
            // anchors keep the extent so the recentre is unchanged).
            let interior_id = b.roots[1];
            b.remove_node(interior_id);
            ("remove", b)
        };

        for (label, scene_b) in [recolor, resize, move_node, add_node, remove_node] {
            // Precondition: every edit keeps the recentre fixed (so the incremental
            // path applies — a recentre shift would force a full rebuild and the
            // dirty-count assertion below would not hold).
            assert_eq!(
                scene_a.recentre_voxels_for_resolve(density),
                scene_b.recentre_voxels_for_resolve(density),
                "[{label}] this edit must keep the composite recentre fixed"
            );

            // Incremental: wholesale-build A, then apply the single edit to B.
            let mut resolve_cache = Store::new();
            let mut render_cache: RenderCache = {
                let chunks = resolve_cache.resident_render_chunks(&scene_a, density, 0);
                chunks
                    .iter()
                    .filter(|(_, grid)| !grid.occupied.is_empty())
                    .map(|(coord, grid)| (*coord, occupied_multiset(grid)))
                    .collect()
            };
            let total_before = render_cache.len();
            let rebuilt = apply_incremental_edit(
                &mut render_cache,
                &mut resolve_cache,
                &scene_a,
                &scene_b,
                density,
            );

            // The full wholesale rebuild for the post-edit scene B (the truth).
            let full = full_render_cache(&scene_b, density);

            assert_eq!(
                render_cache, full,
                "[{label}] incremental render cache (coords + each chunk's instance \
                 multiset) MUST equal a full wholesale rebuild of scene B — a stale \
                 chunk or a missed fresh chunk would differ here"
            );

            // Dirty-count-is-less: a localised edit rebuilds strictly fewer chunks
            // than the scene's total resident chunks (proving it is incremental, not
            // a disguised full rebuild). `total_before` and `full.len()` are both the
            // scene's full per-chunk count (A and B differ by one localised node), so
            // a genuine incremental edit touches a strict subset.
            let scene_chunks = total_before.max(full.len());
            assert!(
                rebuilt < scene_chunks,
                "[{label}] a localised edit must rebuild strictly FEWER chunks \
                 ({rebuilt}) than the scene's total ({scene_chunks}) — else it is a \
                 disguised full rebuild"
            );
        }
    }

    /// A focused dirty-count assertion: an in-place recolor of ONE SMALL far-flung
    /// node dirties only the handful of chunks that node occupies, NOT the whole
    /// scene — so a localised edit rebuilds far fewer than half the resident chunks.
    #[test]
    fn localized_recolor_rebuilds_few_chunks() {
        let density = 16u32;
        // A wide sphere (many chunks) plus a tiny 1-block box pushed far out in X,
        // so the box owns only ~1 chunk no other leaf touches.
        let scene_a = Scene::from_nodes(vec![
            tool_node(ShapeKind::Sphere, [9, 9, 9], [0, 0, 0], MaterialChoice::Stone),
            tool_node(ShapeKind::Box, [1, 1, 1], [80, 0, 0], MaterialChoice::Wood),
        ]);
        let mut scene_b = scene_a.clone();
        if let NodeContent::Tool { material, .. } = &mut scene_b.root_node_mut(1).content {
            *material = MaterialChoice::Stone;
        }

        let mut resolve_cache = Store::new();
        let mut render_cache: RenderCache = {
            let chunks = resolve_cache.resident_render_chunks(&scene_a, density, 0);
            chunks
                .iter()
                .filter(|(_, grid)| !grid.occupied.is_empty())
                .map(|(coord, grid)| (*coord, occupied_multiset(grid)))
                .collect()
        };
        let total = render_cache.len();
        let rebuilt =
            apply_incremental_edit(&mut render_cache, &mut resolve_cache, &scene_a, &scene_b, density);

        assert!(total >= 8, "the spread scene has many resident chunks ({total})");
        assert!(
            rebuilt * 2 < total,
            "a localised recolor of a small node must rebuild far fewer than half the \
             chunks: rebuilt {rebuilt} of {total}"
        );
        // And the result still matches a full rebuild.
        assert_eq!(render_cache, full_render_cache(&scene_b, density));
    }

    // ===== Issue #20 Step 3: out-of-core spill to DiskChunkStore ==================

    /// A unique temp directory under the system temp dir, removed on drop so no spill
    /// test leaves disk litter (mirrors the disk-store tests' RAII guard).
    struct TempDir {
        path: std::path::PathBuf,
    }
    impl TempDir {
        fn new(label: &str) -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "voxelworker_chunk_cache_spill_test_{label}_{}_{unique}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            Self { path }
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    /// The covering chunk coords of `scene` at `density`, in chunk order.
    fn covering_coords(scene: &Scene, density: u32) -> Vec<[i32; 3]> {
        let mut coords = Vec::new();
        if let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(density) {
            for chunk_z in min_chunk[2]..=max_chunk[2] {
                for chunk_y in min_chunk[1]..=max_chunk[1] {
                    for chunk_x in min_chunk[0]..=max_chunk[0] {
                        coords.push([chunk_x, chunk_y, chunk_z]);
                    }
                }
            }
        }
        coords
    }

    /// (a) A chunk fetched AFTER being spilled-and-reloaded is BYTE-IDENTICAL (every
    /// f32 position bit + material) to the resident result — the spill/reload round-trip
    /// is transparent. A capacity of 1 forces every-other access to spill the prior.
    #[test]
    fn spilled_and_reloaded_chunk_is_byte_identical() {
        let density = 16u32;
        // A scene spread across several chunks in X so we have >1 covering chunk.
        let scene = three_tool_scene(density, 40);
        let coords = covering_coords(&scene, density);
        assert!(coords.len() >= 2, "need at least two covering chunks to force a spill");

        // The reference: every covering chunk's grid from an UNBOUNDED cache (no spill).
        let mut reference = Store::new();
        let expected: std::collections::HashMap<[i32; 3], _> = coords
            .iter()
            .map(|&coord| (coord, occupied_multiset(reference.chunk(coord, &scene, density, 0))))
            .collect();

        // A capacity-1 spilling cache: fetch every coord once (filling + spilling), then
        // re-fetch every coord — each re-fetch reloads from disk (or recomputes) and
        // must equal the unbounded reference byte-for-byte.
        let temp = TempDir::new("byte_identical");
        let mut cache = Store::with_resident_cap(1, &temp.path).unwrap();
        for &coord in &coords {
            let _ = cache.chunk(coord, &scene, density, 0);
            assert!(cache.resident_chunk_count() <= 1, "capacity 1 keeps at most one resident");
        }
        assert!(cache.spill_count() >= 1, "filling past capacity 1 must spill");

        for &coord in &coords {
            let got = occupied_multiset(cache.chunk(coord, &scene, density, 0));
            assert_eq!(
                got, expected[&coord],
                "chunk {coord:?} after spill+reload must be byte-identical to the resident result"
            );
        }
    }

    /// (b) The resident cap is honored: under sustained load over many chunks the
    /// resident count NEVER exceeds the cap, and every chunk remains correct.
    #[test]
    fn resident_cap_is_never_exceeded() {
        let density = 16u32;
        let cap = 3usize;
        let scene = three_tool_scene(density, 80); // a wide spread → many chunks.
        let coords = covering_coords(&scene, density);
        assert!(coords.len() > cap, "the scene must have more chunks than the cap to exercise spill");

        let temp = TempDir::new("cap_honored");
        let mut cache = Store::with_resident_cap(cap, &temp.path).unwrap();
        // Repeat the sweep twice so reloads (which also insert) are stress-tested.
        for _ in 0..2 {
            for &coord in &coords {
                let _ = cache.chunk(coord, &scene, density, 0);
                assert!(
                    cache.resident_chunk_count() <= cap,
                    "resident count {} exceeded cap {cap}",
                    cache.resident_chunk_count()
                );
            }
        }
        assert_eq!(cache.resident_chunk_count(), cap.min(coords.len()), "fills to the cap");
    }

    /// (c) LRU order: the LEAST-recently-used chunk is the one spilled. Touch A, then B,
    /// then fetch a third over a cap of 2 — A (the LRU) is the spill victim, not B.
    #[test]
    fn least_recently_used_chunk_is_spilled() {
        let density = 16u32;
        let scene = three_tool_scene(density, 80);
        let coords = covering_coords(&scene, density);
        assert!(coords.len() >= 3, "need at least three covering chunks");
        let (a, b, c) = (coords[0], coords[1], coords[2]);

        let temp = TempDir::new("lru_order");
        let mut cache = Store::with_resident_cap(2, &temp.path).unwrap();

        // Fetch A then B (both resident, cap 2); A is now the LRU.
        let _ = cache.chunk(a, &scene, density, 0);
        let _ = cache.chunk(b, &scene, density, 0);
        assert_eq!(cache.resident_chunk_count(), 2);
        assert_eq!(cache.spill_count(), 0, "two chunks fit the cap of 2");

        // Fetch C over capacity → A (the LRU) is spilled, B stays resident.
        let _ = cache.chunk(c, &scene, density, 0);
        assert_eq!(cache.spill_count(), 1, "exactly one chunk spilled");
        assert_eq!(cache.resident_chunk_count(), 2);

        // Re-fetch B: resident → NO reload. Re-fetch A: spilled → exactly one reload.
        let reloads_before = cache.disk_reload_count();
        let _ = cache.chunk(b, &scene, density, 0);
        assert_eq!(
            cache.disk_reload_count(), reloads_before,
            "B stayed resident (A was the LRU victim) — no reload"
        );
        let _ = cache.chunk(a, &scene, density, 0);
        assert_eq!(
            cache.disk_reload_count(), reloads_before + 1,
            "A was the spilled LRU — fetching it reloads exactly once"
        );
    }

    /// (d) Invalidation purges BOTH resident and disk: a spilled chunk that an edit
    /// dirties must NOT resurface (a later fetch recomputes it, it does not reload the
    /// stale disk copy). Verified through both `invalidate_chunk` and `invalidate_aabb`.
    #[test]
    fn invalidation_purges_resident_and_disk() {
        let density = 16u32;
        let scene = three_tool_scene(density, 80);
        let coords = covering_coords(&scene, density);
        assert!(coords.len() >= 2);

        // --- invalidate_chunk path ---
        let temp = TempDir::new("invalidate_chunk");
        let mut cache = Store::with_resident_cap(1, &temp.path).unwrap();
        // Fill so coords[0] gets spilled to disk (cap 1, fetch a second coord after).
        let _ = cache.chunk(coords[0], &scene, density, 0);
        let _ = cache.chunk(coords[1], &scene, density, 0);
        assert!(cache.spill_count() >= 1, "coords[0] must be spilled to disk");
        let reloads_before = cache.disk_reload_count();

        // Invalidate the spilled coord, then fetch it: it must RECOMPUTE, not reload.
        cache.invalidate_chunk(coords[0]);
        let recomputes_before = cache.recompute_count();
        let _ = cache.chunk(coords[0], &scene, density, 0);
        assert_eq!(
            cache.disk_reload_count(), reloads_before,
            "an invalidated spilled chunk must NOT reload the stale disk copy"
        );
        assert_eq!(
            cache.recompute_count(), recomputes_before + 1,
            "the invalidated chunk is recomputed from the scene"
        );

        // --- invalidate_aabb path ---
        let temp2 = TempDir::new("invalidate_aabb");
        let mut cache2 = Store::with_resident_cap(1, &temp2.path).unwrap();
        let _ = cache2.resolve_region(&scene, density, 0); // resolves + spills all but one.
        assert!(cache2.spill_count() >= 1, "resolve_region over cap 1 must spill");

        // An edit AABB spanning the whole covering chunk grid purges every chunk
        // (resident + disk). The AABB is in absolute (producer-true) voxels, the frame
        // `invalidate_aabb` expects.
        let region_aabb = {
            let (lo, hi) = scene.covering_chunk_range(density).unwrap();
            let chunk_extent = (voxel_core::core_geom::CHUNK_BLOCKS * density) as i64;
            let min_v = [
                lo[0] as i64 * chunk_extent,
                lo[1] as i64 * chunk_extent,
                lo[2] as i64 * chunk_extent,
            ];
            let max_v = [
                (hi[0] as i64 + 1) * chunk_extent,
                (hi[1] as i64 + 1) * chunk_extent,
                (hi[2] as i64 + 1) * chunk_extent,
            ];
            voxel_core::spatial_index::VoxelAabb::new(min_v, max_v)
        };
        let _ = cache2.invalidate_aabb(&region_aabb, density);
        let reloads_before2 = cache2.disk_reload_count();
        // Re-resolve: every chunk must recompute, none reload a purged disk copy.
        let _ = cache2.resolve_region(&scene, density, 0);
        assert_eq!(
            cache2.disk_reload_count(), reloads_before2,
            "after invalidate_aabb no chunk reloads a stale spilled copy"
        );
    }

    /// (e) Counters tally an expected access sequence: spill / reload / recompute counts
    /// match a hand-traced sequence over a capacity-1 cache and two distinct chunks.
    #[test]
    fn counters_tally_an_expected_access_sequence() {
        let density = 16u32;
        let scene = three_tool_scene(density, 80);
        let coords = covering_coords(&scene, density);
        assert!(coords.len() >= 2);
        let (a, b) = (coords[0], coords[1]);

        let temp = TempDir::new("counters");
        let mut cache = Store::with_resident_cap(1, &temp.path).unwrap();

        // 1. Fetch A (miss in both → recompute 1; nothing to spill yet).
        let _ = cache.chunk(a, &scene, density, 0);
        assert_eq!((cache.recompute_count(), cache.spill_count(), cache.disk_reload_count()), (1, 0, 0));

        // 2. Fetch A again (resident hit → no counter moves).
        let _ = cache.chunk(a, &scene, density, 0);
        assert_eq!((cache.recompute_count(), cache.spill_count(), cache.disk_reload_count()), (1, 0, 0));

        // 3. Fetch B (recompute 2; inserting over cap 1 spills A → spill 1).
        let _ = cache.chunk(b, &scene, density, 0);
        assert_eq!((cache.recompute_count(), cache.spill_count(), cache.disk_reload_count()), (2, 1, 0));

        // 4. Fetch A (spilled → reload 1; inserting over cap 1 spills B → spill 2).
        let _ = cache.chunk(a, &scene, density, 0);
        assert_eq!((cache.recompute_count(), cache.spill_count(), cache.disk_reload_count()), (2, 2, 1));

        // 5. Fetch B (spilled → reload 2; spills A → spill 3). No recompute (both exist).
        let _ = cache.chunk(b, &scene, density, 0);
        assert_eq!((cache.recompute_count(), cache.spill_count(), cache.disk_reload_count()), (2, 3, 2));
    }

    /// An unbounded cache (the default `new()`) NEVER spills, reloads or tracks LRU —
    /// proving the live path / goldens are untouched by Step 3.
    #[test]
    fn unbounded_cache_never_spills() {
        let density = 16u32;
        let scene = three_tool_scene(density, 80);
        let mut cache = Store::new();
        let _ = cache.resolve_region(&scene, density, 0);
        assert!(cache.resident_chunk_count() > 1, "an unbounded cache keeps every chunk resident");
        assert_eq!(cache.spill_count(), 0);
        assert_eq!(cache.disk_reload_count(), 0);
        assert!(cache.recompute_count() > 0, "recompute count tracks first-time resolves");
    }

    /// A zero resident cap is rejected at construction (a cache that holds nothing
    /// resident is a misconfiguration).
    #[test]
    fn zero_resident_cap_panics() {
        let temp = TempDir::new("zero_cap");
        let result = std::panic::catch_unwind(|| {
            Store::with_resident_cap(0, &temp.path)
        });
        assert!(result.is_err(), "a zero resident cap must panic");
    }
}
