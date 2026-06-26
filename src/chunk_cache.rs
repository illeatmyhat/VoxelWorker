//! A per-chunk resolve cache (ADR 0002 Decision 3, issue #27 S2).
//!
//! S0 made resolve **chunk-addressable** ([`Scene::resolve_chunk`]) and proved a
//! whole-region resolve can be reassembled from per-chunk pieces
//! ([`Scene::resolve_region_via_chunks`]). S2 turns that decomposition into the
//! **resolve mechanism**: a cache keyed by `(chunk_coord, lod)` that resolves a
//! chunk **on demand** (lazily) and stores the result, so a second request for the
//! same chunk is a map lookup instead of a re-resolve.
//!
//! ## What S2 changes (and what it does NOT)
//!
//! * **Lazy per-chunk resolve + cache.** [`ChunkResolveCache::chunk`] returns the
//!   cached per-chunk [`VoxelGrid`] (in **absolute** composite voxel coordinates,
//!   exactly as [`Scene::resolve_chunk`] produces), resolving + storing it on a
//!   miss.
//! * **Per-chunk voxel bound.** The old whole-region `MAX_GRID_VOXELS` guard is now
//!   a *per-chunk* bound (a single chunk can't exceed it), so a scene whose TOTAL
//!   voxel count is far beyond the old 6M ceiling resolves fine as long as every
//!   individual chunk is small. See [`crate::voxel::MAX_CHUNK_VOXELS`].
//! * **Identical render output.** [`ChunkResolveCache::resolve_region`] rebuilds the
//!   SAME recentred monolithic grid the renderer/mesher/fog consume today — but
//!   assembled from cached chunks. The bytes downstream are unchanged (see the
//!   module-level invariant in [`ChunkResolveCache::resolve_region`]).
//!
//! **S3 (#27) added smart invalidation** on top of this seam:
//! [`ChunkResolveCache::invalidate_aabb`] evicts exactly the chunks an edit's
//! world-AABB intersects (whole-chunk dirty granularity, ADR 0002 Decision 3). The
//! edit AABB is computed by
//! [`LeafSpatialIndex::edit_aabb_since`](crate::spatial_index::LeafSpatialIndex::edit_aabb_since)
//! (diffing the scene's leaf spatial index before vs after the edit);
//! [`ChunkResolveCache::clear`] remains the fallback for edits that can't be
//! localised (a density change or a region-spanning Part edit).
//!
//! What is **deferred** (do NOT look for it here):
//!
//! * **Recentre removal, camera-relative rebasing, renderer consuming per-chunk
//!   meshes directly** — S4 / #27. S2 still hands the renderer one recentred
//!   monolithic grid.

use std::collections::HashMap;

use crate::scene::Scene;
use crate::spatial_index::VoxelAabb;
use crate::voxel::VoxelGrid;

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
/// [`resolve_region`](Self::resolve_region)).
///
/// **S3 seam:** invalidation is currently all-or-nothing
/// ([`clear`](Self::clear)) plus a single-chunk drop
/// ([`invalidate_chunk`](Self::invalidate_chunk)); neither tracks WHICH edit
/// touched WHICH chunk. S3 (#27) adds edit-world-AABB → dirty-chunk invalidation
/// on top of this seam.
#[derive(Debug, Default)]
pub struct ChunkResolveCache {
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
}

impl ChunkResolveCache {
    /// A fresh, empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of chunks currently resident (for tests / diagnostics).
    pub fn resident_chunk_count(&self) -> usize {
        self.chunks.len()
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
    /// ([`crate::voxel::MAX_CHUNK_VOXELS`]) is rejected by the call sites BEFORE
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
        let origin = self.bound_floating_origin;
        self.chunks
            .entry(key)
            .or_insert_with(|| scene.resolve_chunk_rebased(chunk_coord, voxels_per_block, lod, origin))
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
    pub fn resolve_region(&mut self, scene: &Scene, voxels_per_block: u32, lod: u32) -> VoxelGrid {
        debug_assert_eq!(lod, 0, "S2 only resolves full resolution (lod 0)");

        // The render floating origin (ADR 0002 Decision 2 / S4b) IS the composite
        // recentre offset `resolve_region` subtracts to centre the composite on the
        // origin. Binding the cache to it makes every chunk come out ALREADY rebased
        // by `resolve_chunk_rebased` — with the subtraction done in i64 BEFORE the f32
        // downcast, so a far-placed scene keeps full f32 precision (the S1 speckle is
        // gone). For a near scene this is bit-identical to the previous f32 subtract
        // (the recentre is integer-block-aligned and positions are small), so the
        // goldens are unchanged.
        let recentre_voxels = scene.recentre_voxels_for_resolve(voxels_per_block);
        self.rebind_if_changed(voxels_per_block, recentre_voxels);

        let region_dimensions = scene.placed_region_dimensions(voxels_per_block);
        let mut output = VoxelGrid::new(region_dimensions);

        let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(voxels_per_block) else {
            // No leaf has an intrinsic size (a Part-only scene with no Tools): no
            // composite AABB, so there are no chunks — an empty recentred grid,
            // exactly as `resolve_region` returns for the same scene.
            return output;
        };

        for chunk_z in min_chunk[2]..=max_chunk[2] {
            for chunk_y in min_chunk[1]..=max_chunk[1] {
                for chunk_x in min_chunk[0]..=max_chunk[0] {
                    let chunk = self.chunk_for_current_binding(
                        [chunk_x, chunk_y, chunk_z],
                        scene,
                        voxels_per_block,
                        lod,
                    );
                    // The cached chunk is already rebased to the floating origin
                    // (= recentre), so its voxels drop straight into the output.
                    output.occupied.extend_from_slice(&chunk.occupied);
                }
            }
        }
        output
    }

    /// Drop every cached chunk (the all-or-nothing invalidation seam).
    ///
    /// Still used for the edit kinds [`invalidate_aabb`](Self::invalidate_aabb) can't
    /// localise (a density change, or a region-spanning Part edit) and on the very
    /// first rebuild (no previous scene to diff against). For a localisable edit,
    /// prefer [`invalidate_aabb`].
    pub fn clear(&mut self) {
        self.chunks.clear();
        self.bound_density = None;
        self.bound_floating_origin = [0, 0, 0];
    }

    /// Drop a single cached chunk across all LODs (a finer-grained seam).
    ///
    /// [`invalidate_aabb`](Self::invalidate_aabb) calls this for each chunk an edit's
    /// world-AABB intersects.
    pub fn invalidate_chunk(&mut self, chunk_coord: [i32; 3]) {
        self.chunks.retain(|key, _| key.chunk_coord != chunk_coord);
    }

    /// **Targeted invalidation (issue #27 S3).** Drop exactly the cached chunks whose
    /// half-open box intersects the edit world-AABB `edit_aabb` (in absolute voxels,
    /// the producer-true frame), at `voxels_per_block` — ADR 0002 Decision 3's
    /// whole-chunk dirty granularity. Every other cached chunk stays resident
    /// untouched.
    ///
    /// `edit_aabb` is what
    /// [`LeafSpatialIndex::edit_aabb_since`](crate::spatial_index::LeafSpatialIndex::edit_aabb_since)
    /// returns: the union of the old and new boxes of whatever the edit changed
    /// (moved / added / removed / edited leaves), so a moved node dirties chunks
    /// around BOTH its source and destination. An empty `edit_aabb` (nothing changed)
    /// evicts nothing.
    ///
    /// A density mismatch against the cache's bound density is treated
    /// conservatively (the AABB was computed at a different chunk size) by clearing
    /// everything — but the caller [`main`] already falls back to [`clear`] for a
    /// density change, so this path is belt-and-braces.
    ///
    /// [`main`]: crate
    pub fn invalidate_aabb(&mut self, edit_aabb: &VoxelAabb, voxels_per_block: u32) {
        if let Some(bound) = self.bound_density {
            if bound != voxels_per_block {
                self.clear();
                return;
            }
        }
        let Some((min_chunk, max_chunk)) = edit_aabb.covering_chunk_range(voxels_per_block) else {
            return; // empty edit AABB — nothing to invalidate.
        };
        self.chunks.retain(|key, _| {
            let coord = key.chunk_coord;
            let inside = (0..3).all(|axis| coord[axis] >= min_chunk[axis] && coord[axis] <= max_chunk[axis]);
            !inside
        });
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
            // Re-binding from a previous binding: drop the now-stale chunks.
            self.chunks.clear();
        }
        self.bound_density = Some(voxels_per_block);
        self.bound_floating_origin = floating_origin;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panel::{GeometryParams, MaterialChoice};
    use crate::scene::{
        AssemblyDef, DefId, Node, NodeContent, NodePath, RegionBlocks,
    };
    use crate::voxel::{SdfShape, ShapeKind, VoxelGrid};

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
            let key = [
                voxel.world_position[0].to_bits(),
                voxel.world_position[1].to_bits(),
                voxel.world_position[2].to_bits(),
            ];
            *multiset.entry((key, voxel.material_id)).or_insert(0) += 1;
        }
        multiset
    }

    fn shape_scene(kind: ShapeKind, voxels_per_block: u32) -> Scene {
        Scene::from_geometry(
            GeometryParams {
                shape: kind,
                size_blocks: [5, 5, 5],
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
        let mut cache = ChunkResolveCache::new();
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
        let mut cache = ChunkResolveCache::new();
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
                        size_blocks: size,
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
            let shape = SdfShape {
                kind,
                size_blocks: [5, 5, 5],
                voxels_per_block,
                wall_blocks: 1,
            };
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform.offset_blocks = offset;
            node
        };
        let scene = Scene {
            nodes: vec![
                make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
                make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
                make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
            ],
            active: Some(NodePath::root_index(0)),
            ..Scene::default()
        };
        assert_cache_region_matches_monolithic(&scene, voxels_per_block, "demo-scene");
    }

    #[test]
    fn cache_region_matches_monolithic_for_demo_village() {
        let voxels_per_block = 16;
        let house_def_id = DefId(1);
        let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
            let shape = SdfShape {
                kind,
                size_blocks: size,
                voxels_per_block,
                wall_blocks: 1,
            };
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform.offset_blocks = offset;
            node
        };
        let house = AssemblyDef {
            id: house_def_id,
            name: "House".to_string(),
            children: vec![
                tool(ShapeKind::Box, [2, 2, 2], [0, 0, 0], MaterialChoice::Stone),
                tool(ShapeKind::Cylinder, [1, 2, 1], [0, 2, 0], MaterialChoice::Wood),
            ],
        };
        let instance = |name: &str, offset: [i64; 3]| {
            let mut node = Node::new(name, NodeContent::Instance(house_def_id));
            node.transform.offset_blocks = offset;
            node
        };
        let scene = Scene {
            nodes: vec![
                instance("House 1", [0, 0, 0]),
                instance("House 2", [6, 0, 0]),
                instance("House 3", [12, 0, 0]),
                instance("House 4", [18, 0, 0]),
            ],
            definitions: vec![house],
            active: Some(NodePath::root_index(0)),
        };
        assert_cache_region_matches_monolithic(&scene, voxels_per_block, "demo-village");
    }

    /// A density change clears + re-binds the cache (a chunk's voxel extent depends
    /// on density), and the re-resolve still matches the monolithic at the new
    /// density.
    #[test]
    fn density_change_rebinds_cache() {
        let scene = shape_scene(ShapeKind::Torus, 16);
        let mut cache = ChunkResolveCache::new();
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
        let mut cache = ChunkResolveCache::new();
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
        let shape = SdfShape {
            kind: ShapeKind::Box,
            size_blocks: [1, 1, 1],
            voxels_per_block,
            wall_blocks: 1,
        };
        let corner = |label: &str, offset: [i64; 3]| {
            let mut node = Node::new(
                label,
                NodeContent::Tool { shape, material: MaterialChoice::Stone },
            );
            node.transform.offset_blocks = offset;
            node
        };
        let scene = Scene {
            nodes: vec![
                corner("Box lo", [0, 0, 0]),
                corner("Box hi", [spacing_blocks, spacing_blocks, spacing_blocks]),
            ],
            active: Some(NodePath::root_index(0)),
            ..Scene::default()
        };

        // The OLD whole-region cap would reject this: the composite AABB voxel count
        // is far beyond 6M.
        let region = scene.full_extent_blocks(voxels_per_block);
        let whole_region_voxels = region.size_blocks[0] as u64
            * region.size_blocks[1] as u64
            * region.size_blocks[2] as u64
            * (voxels_per_block as u64).pow(3);
        assert!(
            whole_region_voxels > crate::voxel::MAX_GRID_VOXELS,
            "the synthetic scene's whole-region voxel count ({whole_region_voxels}) must \
             exceed the OLD 6M total cap to prove the point"
        );

        // Every individual chunk is small (one small box at most) — under the new
        // per-chunk bound, so the lazy per-chunk resolve succeeds.
        let mut cache = ChunkResolveCache::new();
        let (min_chunk, max_chunk) = scene
            .covering_chunk_range(voxels_per_block)
            .expect("a placed scene has a covering chunk range");
        let mut total_resolved = 0usize;
        for chunk_z in min_chunk[2]..=max_chunk[2] {
            for chunk_y in min_chunk[1]..=max_chunk[1] {
                for chunk_x in min_chunk[0]..=max_chunk[0] {
                    let chunk = cache.chunk([chunk_x, chunk_y, chunk_z], &scene, voxels_per_block, 0);
                    assert!(
                        (chunk.occupied_count() as u64) <= crate::voxel::MAX_CHUNK_VOXELS,
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
            let extent = (crate::renderer::CHUNK_BLOCKS * voxels_per_block) as u64;
            extent * extent * extent
        };
        // Density 16: chunk extent = 64 voxels → 64³ = 262_144 voxels/chunk, well
        // under the bound — NOT rejected.
        assert!(chunk_capacity_at(16) <= crate::voxel::MAX_CHUNK_VOXELS);
        assert!(
            !crate::voxel::chunk_extent_exceeds_bound(16),
            "a normal density-16 chunk is under the per-chunk bound"
        );

        // A density whose single chunk capacity exceeds the bound IS rejected.
        // chunk extent = CHUNK_BLOCKS × density; pick a density making 64³·k > bound.
        let huge_density = 64u32; // extent = 256 → 256³ = 16_777_216 voxels/chunk.
        assert!(
            chunk_capacity_at(huge_density) > crate::voxel::MAX_CHUNK_VOXELS,
            "the chosen huge density must make one chunk exceed the per-chunk bound"
        );
        assert!(
            crate::voxel::chunk_extent_exceeds_bound(huge_density),
            "a chunk whose voxel capacity exceeds the per-chunk bound must be rejected"
        );
    }

    // ===== Issue #27 S3: targeted edit-AABB invalidation ========================

    fn three_tool_scene(voxels_per_block: u32, box_offset_x: i64) -> Scene {
        let make_tool = |kind, offset: [i64; 3], material| {
            let shape = SdfShape {
                kind,
                size_blocks: [5, 5, 5],
                voxels_per_block,
                wall_blocks: 1,
            };
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform.offset_blocks = offset;
            node
        };
        Scene {
            nodes: vec![
                make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
                make_tool(ShapeKind::Box, [box_offset_x, 0, 0], MaterialChoice::Wood),
                make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
            ],
            active: Some(NodePath::root_index(0)),
            ..Scene::default()
        }
    }

    /// The set of chunk coords currently resident in the cache (for assertions).
    fn resident_coords(cache: &ChunkResolveCache) -> std::collections::BTreeSet<[i32; 3]> {
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
        let mut cache = ChunkResolveCache::new();
        let _ = cache.resolve_region(&scene_a, density, 0);
        let all_resident = resident_coords(&cache);
        assert!(!all_resident.is_empty());

        // Move the Box from +40X to +80X. Compute the edit AABB via the spatial-index
        // diff, exactly as `main::rebuild_geometry` does.
        let mut scene_b = scene_a.clone();
        scene_b.nodes[1].transform.offset_blocks = [80, 0, 0];
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
        let mut fresh_cache = ChunkResolveCache::new();
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
        let mut cache = ChunkResolveCache::new();
        let _ = cache.resolve_region(&scene_a, density, 0);

        let mut scene_b = scene_a.clone();
        scene_b.nodes[1].transform.offset_blocks = [80, 0, 0];
        let index_a = scene_a.build_leaf_spatial_index(density);
        let index_b = scene_b.build_leaf_spatial_index(density);
        let edit_aabb = index_b.edit_aabb_since(&index_a).expect("same density");

        // The chunk owning the OLD Box centre (40·16 = 640 voxels) and the chunk
        // owning the NEW centre (80·16 = 1280 voxels) must BOTH be in the edit range.
        let chunk_extent = (crate::renderer::CHUNK_BLOCKS * density) as i32;
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
        let mut cache = ChunkResolveCache::new();
        let _ = cache.resolve_region(&scene, density, 0);
        let before = resident_coords(&cache);
        let empty = crate::spatial_index::VoxelAabb::new([0, 0, 0], [0, 0, 0]);
        cache.invalidate_aabb(&empty, density);
        assert_eq!(resident_coords(&cache), before, "an empty edit AABB evicts nothing");
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
            let shape = SdfShape {
                kind: ShapeKind::Box,
                size_blocks: [4, 4, 4],
                voxels_per_block: vpb,
                wall_blocks: 1,
            };
            let mut node = Node::new(
                "box",
                NodeContent::Tool { shape, material: MaterialChoice::Stone },
            );
            node.transform.offset_blocks = [offset_x, 0, 0];
            Scene::single_node(node)
        };

        let mut near_cache = ChunkResolveCache::new();
        let near = near_cache.resolve_region(&box_scene(0), vpb, 0);
        // 1_000_000 blocks → 16M voxels, past the f32 exact-integer ceiling.
        let mut far_cache = ChunkResolveCache::new();
        let far = far_cache.resolve_region(&box_scene(1_000_000), vpb, 0);

        assert_eq!(near.occupied_count(), far.occupied_count(), "same shape");
        assert!(near.occupied_count() > 0, "the box must resolve to voxels");
        // Every voxel-centre `.5` fraction must survive the rebase (would be lost to
        // f32 rounding at 1.6e7 under the old subtract-AFTER-f32 path).
        for voxel in &far.occupied {
            for axis in 0..3 {
                let frac = voxel.world_position[axis].fract().abs();
                assert!(
                    (frac - 0.5).abs() < 1e-4,
                    "far voxel centre lost its .5 fraction (f32 jitter): {:?}",
                    voxel.world_position
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
        let mut cache = ChunkResolveCache::new();
        let assembled = cache.resolve_region(&scene, 16, 0);
        // A Part-only scene has no composite AABB → resolve_region returns a
        // zero-sized empty grid; the cache path matches.
        let monolithic = scene.resolve_region(RegionBlocks::new([0, 0, 0]), 16, 0);
        assert_eq!(assembled.occupied_count(), monolithic.occupied_count());
        assert_eq!(assembled.occupied_count(), 0);
    }
}
