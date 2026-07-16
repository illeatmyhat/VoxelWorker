//! The residency + per-chunk resolve cache — the store proper.
//!
//! A per-chunk resolve cache (ADR 0002 Decision 3, issue #27 S2): a cache keyed by
//! `(chunk_coord, lod)` that resolves a chunk **on demand** (lazily) and stores the
//! result, so a second request for the same chunk is a map lookup instead of a
//! re-resolve. [`Store::resolve_region`] (the dense whole-region oracle, compile-gated
//! behind the `oracle` feature) rebuilds the recentred monolithic grid from cached
//! chunks; [`Store::invalidate_aabb`] evicts exactly the chunks an edit's world-AABB
//! intersects (whole-chunk dirty granularity), and [`Store::clear`] is the wholesale
//! fallback. Out-of-core spill (issue #20 Step 3) moves least-recently-used resident
//! chunks to the backing [`DiskChunkStore`](crate::disk_chunk_store::DiskChunkStore).

use std::collections::HashMap;

use crate::chunk_storage::{compress, decompress};
use crate::disk_chunk_store::DiskChunkStore;
use document::scene::Scene;
use voxel_core::spatial_index::{ChunkCoverage, VoxelAabb};
use voxel_core::voxel::{Voxel, VoxelGrid};

use super::ChunkCacheKey;

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
    // `pub(crate)` so the sibling `tests` submodule (split out of this module by ADR 0016
    // Phase 3) can inspect resident chunk coords; not part of the public API.
    pub(crate) chunks: HashMap<ChunkCacheKey, VoxelGrid>,
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
    /// [`Scene::resolve_chunk_rebased`](document::scene::Scene::resolve_chunk_rebased)
    /// before the f32 downcast). The union of all returned chunks' occupied voxels
    /// (position + `material_id`, in the recentred frame) therefore equals
    /// `resolve_region`'s assembled grid voxel-for-voxel; this is the seam the
    /// upcoming per-chunk renderer consumes instead of one monolithic grid.
    ///
    /// Each returned coord is the absolute chunk coord that OWNS that grid's voxels
    /// (the half-open box `[c·E, (c+1)·E)` per axis, `E = CHUNK_BLOCKS × density`),
    /// and the returned coord set equals the scene's
    /// [`covering_chunk_range`](document::scene::Scene::covering_chunk_range) for the
    /// region (empty for a VoxelBody-only scene with no composite extent).
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
    /// site over `VoxExport::from_region_voxels` (up in the interchange layer),
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
    /// localise (a density change, or a region-spanning VoxelBody edit) and on the very
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
