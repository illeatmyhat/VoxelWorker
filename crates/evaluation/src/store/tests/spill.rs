use super::*;

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
