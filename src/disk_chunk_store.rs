//! Disk-backed chunk store with bounded-RAM LRU eviction (issue #20 S6b).
//!
//! S6a ([`crate::chunk_storage`]) gave us a compact, serde-serialisable on-disk
//! form for one resolved chunk grid ([`CompressedChunk`]). This module is the
//! **store** that uses it to bound RAM: it keeps at most a configured number of
//! chunks resident in memory and, when a [`put`](DiskChunkStore::put) would push
//! the resident set over that capacity, it **evicts the least-recently-used**
//! resident chunks by serialising them to disk and dropping them from RAM. A
//! later [`get`](DiskChunkStore::get) for an evicted key transparently reloads it
//! from disk back into RAM (and counts as a use, refreshing its LRU position).
//!
//! ## Standalone (NOT yet wired into the live path)
//!
//! This is a self-contained, thoroughly-tested component. It is **not** wired into
//! the live resolve/render path — that integration is S6c, when the
//! monolithic-grid bridge is removed and the floating-origin/rebasing coupling is
//! reworked (see the module note at the bottom). The render path is untouched, so
//! the goldens are trivially unaffected.
//!
//! ## Capacity model: resident chunk COUNT (not a byte budget)
//!
//! The cap is a maximum number of **resident chunks**, not a byte budget. The
//! rationale: the store's job is to decouple scene size from RAM, and the natural
//! unit the future caller ([`crate::chunk_cache::ChunkResolveCache`]) thinks in is
//! "how many chunks may I hold". A byte budget would force a per-chunk size
//! estimate (a `CompressedChunk` has no cheap exact in-RAM byte size) and make the
//! bounded-RAM invariant fuzzy ("≤ budget ± one chunk"). A count cap gives a crisp,
//! testable invariant — `resident_count() <= capacity` ALWAYS — and the caller can
//! pick a count from a target RAM / typical-chunk-size ratio if it wants a byte
//! feel. Capacity `0` is rejected (a store that can hold nothing is a bug, not a
//! configuration).
//!
//! ## Serialisation format: `serde_json` (an existing dependency)
//!
//! Chunks are written as `serde_json` (already a dependency; `CompressedChunk` is
//! already serde-round-trip tested through JSON in S6a). No new dependency is
//! added. JSON is not the most compact on-disk form — a binary `bincode` would be
//! smaller — but adding a binary codec is out of scope for this standalone step
//! and noted for S6c. The store is format-agnostic behind two private helpers
//! ([`write_chunk_file`] / [`read_chunk_file`]), so swapping the codec later is a
//! two-function change.
//!
//! ## Windows specifics
//!
//! * **Path-safe filenames for negative / large coords.** A key
//!   `([i32;3], u32)` is encoded into a filename that uses only `[0-9a-z_]` (no
//!   `-`, no path separators): negatives are encoded with an `n` prefix per axis
//!   (e.g. `chunk_n5_0_12__lod0.json`). Distinct keys map to distinct filenames.
//! * **No file-lock issues.** Every file handle is opened, used and dropped within
//!   the single helper call that touches it (`std::fs::write` / `read`), so no
//!   handle is held across calls — Windows will not refuse a later delete/rewrite.
//! * **Idempotent directory creation.** The store directory is created with
//!   `create_dir_all` on construction (a no-op if it already exists).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::chunk_cache::ChunkCacheKey;
use crate::chunk_storage::CompressedChunk;

/// Observable counters for the store's eviction / reload behaviour, so tests (and
/// future diagnostics) can prove the LRU/disk machinery does exactly what it
/// claims — nothing more (no needless reloads), nothing less (the invariant holds).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DiskChunkStoreStats {
    /// Total chunks evicted from RAM to disk over the store's lifetime (each
    /// over-capacity `put` evicts exactly one LRU resident chunk).
    pub evictions: u64,
    /// Total chunks reloaded from disk back into RAM. Increments ONLY on a `get`
    /// for a key that is currently evicted (resident on disk, not in RAM) — a hit
    /// on a resident chunk does NOT increment it. This is the "we don't reload
    /// needlessly" proof.
    pub disk_reloads: u64,
    /// Resident (in-RAM) chunk count right now. Never exceeds the capacity.
    pub resident_count: usize,
    /// Keys currently stored on disk (evicted, not resident). A key reloaded into
    /// RAM by a `get` is removed from this set; a re-eviction adds it back.
    pub on_disk_count: usize,
}

/// One resident chunk plus the logical clock tick at which it was last used, so the
/// least-recently-used resident chunk is the one with the smallest `last_used`.
#[derive(Debug)]
struct ResidentChunk {
    chunk: CompressedChunk,
    last_used: u64,
}

/// A disk-backed store of [`CompressedChunk`]s keyed by `(chunk_coord, lod)` with a
/// bounded number of resident (in-RAM) chunks and least-recently-used eviction to
/// disk.
///
/// See the module docs for the capacity model (resident chunk count), the
/// serialisation format (`serde_json`) and the Windows handling.
#[derive(Debug)]
pub struct DiskChunkStore {
    /// Where evicted chunks are serialised. Created on construction.
    directory: PathBuf,
    /// Maximum number of chunks that may be resident in RAM at once (≥ 1).
    capacity: usize,
    /// The resident chunks, by key. `resident.len() <= capacity` is the invariant.
    resident: HashMap<ChunkCacheKey, ResidentChunk>,
    /// Keys that are currently on disk (evicted, NOT resident). Disjoint from
    /// `resident`: a key is in exactly one of the two (or neither, if never put).
    on_disk: std::collections::HashSet<ChunkCacheKey>,
    /// A monotonically increasing logical clock; each access stamps the touched
    /// chunk with the next tick so LRU is "smallest last_used".
    clock: u64,
    /// Lifetime counters (evictions, reloads).
    evictions: u64,
    disk_reloads: u64,
}

impl DiskChunkStore {
    /// Create a store that keeps at most `capacity` chunks resident in RAM,
    /// spilling the rest to `directory` (created if absent).
    ///
    /// # Panics
    /// Panics if `capacity == 0` (a store that can hold nothing is a misconfiguration).
    ///
    /// # Errors
    /// Returns the I/O error if the directory cannot be created.
    pub fn new(directory: impl AsRef<Path>, capacity: usize) -> std::io::Result<Self> {
        assert!(capacity >= 1, "DiskChunkStore capacity must be at least 1");
        let directory = directory.as_ref().to_path_buf();
        std::fs::create_dir_all(&directory)?;
        Ok(Self {
            directory,
            capacity,
            resident: HashMap::new(),
            on_disk: std::collections::HashSet::new(),
            clock: 0,
            evictions: 0,
            disk_reloads: 0,
        })
    }

    /// The store's resident-RAM capacity (max resident chunks).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The number of chunks currently resident in RAM (`<= capacity`, always).
    pub fn resident_count(&self) -> usize {
        self.resident.len()
    }

    /// A snapshot of the observability counters.
    pub fn stats(&self) -> DiskChunkStoreStats {
        DiskChunkStoreStats {
            evictions: self.evictions,
            disk_reloads: self.disk_reloads,
            resident_count: self.resident.len(),
            on_disk_count: self.on_disk.len(),
        }
    }

    /// Whether `key` is present anywhere in the store (resident OR on disk).
    pub fn contains(&self, key: ChunkCacheKey) -> bool {
        self.resident.contains_key(&key) || self.on_disk.contains(&key)
    }

    /// Store `chunk` under `key`, making it the most-recently-used resident chunk.
    ///
    /// If `key` was already on disk, it is brought resident (and removed from disk;
    /// the stale file is deleted). If inserting would exceed capacity, the
    /// least-recently-used OTHER resident chunk is evicted to disk first, so the
    /// resident count never exceeds capacity.
    ///
    /// # Errors
    /// Returns an I/O error if an eviction write (or a stale-file delete) fails.
    pub fn put(&mut self, key: ChunkCacheKey, chunk: CompressedChunk) -> std::io::Result<()> {
        let tick = self.next_tick();

        // If this key lived on disk, it is becoming resident: forget the disk copy
        // (and remove its now-stale file) so the two sets stay disjoint.
        if self.on_disk.remove(&key) {
            let path = self.path_for(key);
            remove_file_if_present(&path)?;
        }

        // Overwriting an existing resident key does not grow the resident set, so it
        // never needs an eviction. A genuinely new key might.
        let is_new_resident = !self.resident.contains_key(&key);
        if is_new_resident && self.resident.len() >= self.capacity {
            self.evict_least_recently_used()?;
        }

        self.resident.insert(key, ResidentChunk { chunk, last_used: tick });
        Ok(())
    }

    /// Fetch the chunk for `key`, or `None` if the store has never held it.
    ///
    /// A resident hit returns a clone and refreshes the key's LRU position. A key
    /// that was evicted to disk is **reloaded** into RAM (incrementing
    /// [`DiskChunkStoreStats::disk_reloads`]), which may evict the current LRU
    /// resident chunk to keep within capacity, then returned. Reloading refreshes
    /// the key's LRU position too (it was just used).
    ///
    /// Returns a clone rather than a reference so the reload path (which mutates the
    /// resident map / may evict) is not fighting the borrow checker; chunks are
    /// already cheap-to-clone compact structures.
    ///
    /// # Errors
    /// Returns an I/O error if a reload read, its deserialise, or an eviction write
    /// triggered by the reload fails.
    pub fn get(&mut self, key: ChunkCacheKey) -> std::io::Result<Option<CompressedChunk>> {
        let tick = self.next_tick();

        // Fast path: a resident hit. Refresh LRU, clone out. NO disk_reloads bump.
        if let Some(entry) = self.resident.get_mut(&key) {
            entry.last_used = tick;
            return Ok(Some(entry.chunk.clone()));
        }

        // Evicted? Reload it from disk. This is the ONLY place disk_reloads grows.
        if self.on_disk.contains(&key) {
            let chunk = read_chunk_file(&self.path_for(key))?;
            self.disk_reloads += 1;

            // Reloading brings a chunk resident, so it may need to make room first.
            if self.resident.len() >= self.capacity {
                self.evict_least_recently_used()?;
            }
            // It is resident now; its disk copy is consumed (delete the file so the
            // sets stay disjoint and a future eviction rewrites a fresh file).
            self.on_disk.remove(&key);
            remove_file_if_present(&self.path_for(key))?;
            self.resident.insert(key, ResidentChunk { chunk: chunk.clone(), last_used: tick });
            return Ok(Some(chunk));
        }

        // Never stored under this key.
        Ok(None)
    }

    /// Evict the single least-recently-used resident chunk to disk, dropping it from
    /// RAM. Called when an insert/reload would otherwise breach capacity.
    ///
    /// Picks the resident entry with the smallest `last_used`. A no-op (returns
    /// `Ok`) if nothing is resident — which only happens with `capacity` so small
    /// that even the chunk being inserted is the only one; that single insert then
    /// stays within capacity on its own.
    fn evict_least_recently_used(&mut self) -> std::io::Result<()> {
        let Some((&victim_key, _)) = self
            .resident
            .iter()
            .min_by_key(|(_, entry)| entry.last_used)
        else {
            return Ok(());
        };
        // Remove from RAM, serialise to disk, record it as on-disk.
        let victim = self
            .resident
            .remove(&victim_key)
            .expect("the min key was just observed in the resident map");
        write_chunk_file(&self.path_for(victim_key), &victim.chunk)?;
        self.on_disk.insert(victim_key);
        self.evictions += 1;
        Ok(())
    }

    /// The next logical clock tick (monotonic; stamps the just-touched chunk).
    fn next_tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// The on-disk file path for `key` — a Windows-path-safe filename under the
    /// store directory (see [`encode_key_filename`]).
    fn path_for(&self, key: ChunkCacheKey) -> PathBuf {
        self.directory.join(encode_key_filename(key))
    }
}

/// Encode a chunk key into a path-safe filename using only `[0-9a-z_]`.
///
/// Negative axes are encoded with an `n` prefix (so `-5` → `n5`), avoiding the `-`
/// character entirely (harmless on Windows but kept clean), and there are no path
/// separators, so a far / negative chunk coord still produces one valid file. The
/// encoding is injective: the three axes and the lod are separated by single `_`s
/// (with `__` before the lod), and the magnitude digits are unambiguous given the
/// `n`/no-`n` sign marker, so distinct keys never collide.
fn encode_key_filename(key: ChunkCacheKey) -> String {
    fn axis(value: i32) -> String {
        if value < 0 {
            // Use unsigned magnitude to handle i32::MIN without overflow.
            format!("n{}", (value as i64).unsigned_abs())
        } else {
            value.to_string()
        }
    }
    let [x, y, z] = key.chunk_coord;
    format!(
        "chunk_{}_{}_{}__lod{}.json",
        axis(x),
        axis(y),
        axis(z),
        key.lod
    )
}

/// Serialise a [`CompressedChunk`] to `path` as `serde_json`. The file handle is
/// opened and closed entirely within `std::fs::write`, so no handle is held across
/// store calls (Windows-safe re-delete/rewrite).
fn write_chunk_file(path: &Path, chunk: &CompressedChunk) -> std::io::Result<()> {
    let json = serde_json::to_vec(chunk).map_err(std::io::Error::other)?;
    std::fs::write(path, json)
}

/// Read + deserialise a [`CompressedChunk`] from `path` (inverse of
/// [`write_chunk_file`]). Handle closed within `std::fs::read`.
fn read_chunk_file(path: &Path) -> std::io::Result<CompressedChunk> {
    let bytes = std::fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(std::io::Error::other)
}

/// Delete `path` if it exists, treating "already gone" as success (idempotent).
fn remove_file_if_present(path: &Path) -> std::io::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk_storage::{compress, decompress};
    use crate::voxel::{Voxel, VoxelGrid};

    /// A unique temp directory under the system temp dir, cleaned up by [`TempDir`]
    /// on drop so no test leaves disk litter.
    struct TempDir {
        path: PathBuf,
    }
    impl TempDir {
        fn new(label: &str) -> Self {
            // A unique name: label + pid + a process-monotonic counter (no rand dep).
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "voxelworker_disk_store_test_{label}_{}_{unique}",
                std::process::id()
            ));
            // Start clean even if a previous crashed run left the dir behind.
            let _ = std::fs::remove_dir_all(&path);
            Self { path }
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    /// A small, distinct single-material chunk whose material id encodes `seed`, so
    /// two chunks built from different seeds are easy to tell apart after a
    /// round-trip. Dimensions vary with `seed` too, exercising the dims round-trip.
    fn make_chunk(seed: u16) -> CompressedChunk {
        let dim = 2 + (seed % 3) as u32; // 2..=4 per axis
        let dimensions = [dim, dim, dim];
        let mut grid = VoxelGrid::new(dimensions);
        let half = [dim as f32 / 2.0; 3];
        for z in 0..dim {
            for y in 0..dim {
                for x in 0..dim {
                    grid.occupied.push(Voxel {
                        world_position: [
                            x as f32 + 0.5 - half[0],
                            y as f32 + 0.5 - half[1],
                            z as f32 + 0.5 - half[2],
                        ],
                        block_local_coord: [
                            (x % 4) as u8,
                            (y % 4) as u8,
                            (z % 4) as u8,
                        ],
                        material_id: seed,
                    });
                }
            }
        }
        compress(&grid)
    }

    /// A decoded grid's comparable fingerprint: dimensions plus the occupied set as
    /// a sorted multiset of `(position-bits, block-local-coord, material_id)`.
    type GridFingerprint = (
        [u32; 3],
        std::collections::BTreeMap<([u32; 3], [u8; 3], u16), usize>,
    );

    /// Canonicalise a chunk's decoded grid into a comparable form (dims + sorted
    /// occupied set of position-bits/coord/material), so a disk round-trip can be
    /// asserted equal to the original byte-for-byte on positions.
    fn grid_fingerprint(chunk: &CompressedChunk) -> GridFingerprint {
        let grid = decompress(chunk);
        let mut occupied = std::collections::BTreeMap::new();
        for voxel in &grid.occupied {
            let position_bits = [
                voxel.world_position[0].to_bits(),
                voxel.world_position[1].to_bits(),
                voxel.world_position[2].to_bits(),
            ];
            *occupied
                .entry((position_bits, voxel.block_local_coord, voxel.material_id))
                .or_insert(0) += 1;
        }
        (grid.dimensions, occupied)
    }

    fn key(coord: [i32; 3]) -> ChunkCacheKey {
        ChunkCacheKey::new(coord, 0)
    }

    #[test]
    fn capacity_zero_panics() {
        let temp = TempDir::new("cap_zero");
        let result = std::panic::catch_unwind(|| DiskChunkStore::new(&temp.path, 0));
        assert!(result.is_err(), "capacity 0 must panic");
    }

    #[test]
    fn put_then_get_resident_does_not_count_as_reload() {
        let temp = TempDir::new("resident_hit");
        let mut store = DiskChunkStore::new(&temp.path, 4).unwrap();
        let chunk = make_chunk(7);
        store.put(key([0, 0, 0]), chunk.clone()).unwrap();

        let got = store.get(key([0, 0, 0])).unwrap().expect("just put it");
        assert_eq!(got, chunk, "a resident get returns the stored chunk");
        let stats = store.stats();
        assert_eq!(stats.disk_reloads, 0, "a resident hit must NOT reload from disk");
        assert_eq!(stats.evictions, 0, "nothing exceeded capacity, so no eviction");
        assert_eq!(stats.resident_count, 1);
    }

    /// Round-trip THROUGH disk: put, force eviction, get back, decompress, and the
    /// grid equals the original (occupied + material + dims).
    #[test]
    fn round_trip_through_disk_eviction_preserves_grid() {
        let temp = TempDir::new("disk_round_trip");
        let mut store = DiskChunkStore::new(&temp.path, 1).unwrap();
        let original = make_chunk(42);
        let original_fingerprint = grid_fingerprint(&original);

        store.put(key([1, 2, 3]), original.clone()).unwrap();
        // A second put over capacity 1 evicts key [1,2,3] to disk.
        store.put(key([9, 9, 9]), make_chunk(1)).unwrap();
        assert!(store.stats().evictions >= 1, "the over-capacity put must evict");
        assert_eq!(store.resident_count(), 1, "capacity 1 keeps exactly one resident");

        // Get the evicted key back: it must reload from disk and decode identically.
        let reloaded = store.get(key([1, 2, 3])).unwrap().expect("still stored on disk");
        assert_eq!(store.stats().disk_reloads, 1, "the evicted get reloads exactly once");
        assert_eq!(reloaded, original, "the reloaded CompressedChunk equals the original");
        assert_eq!(
            grid_fingerprint(&reloaded),
            original_fingerprint,
            "decompress(reloaded) equals the original grid (dims + occupied + material)"
        );
    }

    /// Bounded RAM: after many puts, resident_count never exceeds capacity, and
    /// every evicted chunk is retrievable from disk decoding to its original.
    #[test]
    fn bounded_ram_invariant_and_all_evicted_retrievable() {
        let temp = TempDir::new("bounded_ram");
        let capacity = 3usize;
        let mut store = DiskChunkStore::new(&temp.path, capacity).unwrap();

        let count = 25u16;
        let mut originals = Vec::new();
        for index in 0..count {
            let coord = [index as i32, 0, 0];
            let chunk = make_chunk(index);
            originals.push((coord, chunk.clone()));
            store.put(key(coord), chunk).unwrap();
            assert!(
                store.resident_count() <= capacity,
                "resident_count {} exceeded capacity {capacity} after put {index}",
                store.resident_count()
            );
        }
        assert_eq!(store.resident_count(), capacity, "the store fills to capacity");

        // Every chunk ever put is still retrievable (resident or reloaded) and equal.
        for (coord, original) in &originals {
            let got = store.get(key(*coord)).unwrap().expect("every put chunk is retrievable");
            assert_eq!(&got, original, "chunk at {coord:?} survived the round-trip");
            assert!(store.resident_count() <= capacity, "invariant holds across reloads too");
        }
    }

    /// LRU correctness: touching a chunk makes it most-recently-used so it survives
    /// the next eviction; the genuinely least-recently-used chunk is the victim.
    #[test]
    fn lru_touch_survives_and_least_recent_is_evicted() {
        let temp = TempDir::new("lru_order");
        let mut store = DiskChunkStore::new(&temp.path, 3).unwrap();

        // Fill: A, B, C resident (A oldest, C newest).
        store.put(key([0, 0, 0]), make_chunk(0)).unwrap(); // A
        store.put(key([1, 0, 0]), make_chunk(1)).unwrap(); // B
        store.put(key([2, 0, 0]), make_chunk(2)).unwrap(); // C

        // Touch A (a get) → A becomes most-recently-used; B is now the LRU.
        let _ = store.get(key([0, 0, 0])).unwrap().expect("A resident");
        assert_eq!(store.stats().disk_reloads, 0, "A was resident — no reload");

        // Insert D over capacity → the LRU (B) is evicted, NOT the touched A.
        store.put(key([3, 0, 0]), make_chunk(3)).unwrap(); // D
        assert_eq!(store.resident_count(), 3);
        assert_eq!(store.stats().evictions, 1, "exactly one chunk was evicted");

        // B is the one on disk; A, C, D are resident (a get on them does NOT reload).
        let reloads_before = store.stats().disk_reloads;
        let _ = store.get(key([0, 0, 0])).unwrap().expect("A still resident");
        let _ = store.get(key([2, 0, 0])).unwrap().expect("C still resident");
        let _ = store.get(key([3, 0, 0])).unwrap().expect("D resident");
        assert_eq!(
            store.stats().disk_reloads,
            reloads_before,
            "A, C, D were resident (B was the LRU victim) — none reloaded"
        );

        // B IS on disk: getting it reloads exactly once.
        let _ = store.get(key([1, 0, 0])).unwrap().expect("B reloads from disk");
        assert_eq!(
            store.stats().disk_reloads,
            reloads_before + 1,
            "only B (the evicted LRU) reloads"
        );
    }

    /// Negative and large chunk coords produce valid, distinct files and round-trip.
    #[test]
    fn negative_and_large_coords_round_trip_distinctly() {
        let temp = TempDir::new("neg_large_coords");
        let mut store = DiskChunkStore::new(&temp.path, 1).unwrap();

        let coords = [
            [-5, 0, 12],
            [i32::MIN, i32::MAX, 0],
            [-1, -1, -1],
            [1_000_000, -1_000_000, 7],
            [0, 0, 0],
        ];
        // Distinct filenames for distinct keys (a sanity check on the encoder).
        let mut filenames = std::collections::HashSet::new();
        for coord in &coords {
            assert!(
                filenames.insert(encode_key_filename(key(*coord))),
                "filename collision for coord {coord:?}"
            );
        }

        let mut originals = Vec::new();
        for (index, coord) in coords.iter().enumerate() {
            let chunk = make_chunk(index as u16 + 50);
            originals.push((*coord, chunk.clone()));
            store.put(key(*coord), chunk).unwrap(); // capacity 1 → each evicts the prior
        }
        // All but the last were evicted to disk; every one decodes to its original.
        for (coord, original) in &originals {
            let got = store.get(key(*coord)).unwrap().expect("retrievable");
            assert_eq!(&got, original, "coord {coord:?} round-tripped via disk");
            assert_eq!(grid_fingerprint(&got), grid_fingerprint(original));
        }
    }

    /// disk_reloads increments ONLY on access to an evicted key (proves no needless
    /// reloads): repeated gets of a resident chunk never bump it, and a get of a
    /// never-stored key returns None without a reload.
    #[test]
    fn reload_counter_only_on_evicted_access() {
        let temp = TempDir::new("reload_counter");
        let mut store = DiskChunkStore::new(&temp.path, 2).unwrap();
        store.put(key([0, 0, 0]), make_chunk(0)).unwrap();
        store.put(key([1, 0, 0]), make_chunk(1)).unwrap();

        // Many resident gets: no reloads.
        for _ in 0..10 {
            let _ = store.get(key([0, 0, 0])).unwrap();
            let _ = store.get(key([1, 0, 0])).unwrap();
        }
        assert_eq!(store.stats().disk_reloads, 0, "resident gets never reload");

        // A get for a key never stored: None, no reload.
        assert!(store.get(key([99, 99, 99])).unwrap().is_none());
        assert_eq!(store.stats().disk_reloads, 0, "a missing key does not reload");

        // Force an eviction, then access the evicted key: exactly one reload.
        store.put(key([2, 0, 0]), make_chunk(2)).unwrap(); // evicts the LRU of {0,1}
        assert_eq!(store.stats().evictions, 1);
        // The LRU among [0] and [1] after the loop: both were touched in the loop in
        // order 0 then 1, so [0] is older → [0] is the victim.
        let _ = store.get(key([0, 0, 0])).unwrap().expect("evicted [0] reloads");
        assert_eq!(store.stats().disk_reloads, 1, "exactly one reload, on the evicted key");
    }

    /// Overwriting an existing resident key updates its value without evicting
    /// (the resident set did not grow) and refreshes its LRU position.
    #[test]
    fn overwrite_resident_key_does_not_evict() {
        let temp = TempDir::new("overwrite");
        let mut store = DiskChunkStore::new(&temp.path, 2).unwrap();
        store.put(key([0, 0, 0]), make_chunk(0)).unwrap();
        store.put(key([1, 0, 0]), make_chunk(1)).unwrap();

        // Overwrite [0] with a new value — at capacity, but NOT a new key.
        let replacement = make_chunk(123);
        store.put(key([0, 0, 0]), replacement.clone()).unwrap();
        assert_eq!(store.stats().evictions, 0, "overwriting a resident key evicts nothing");
        assert_eq!(store.resident_count(), 2);
        let got = store.get(key([0, 0, 0])).unwrap().expect("resident");
        assert_eq!(got, replacement, "the overwrite took effect");

        // Because the overwrite refreshed [0]'s LRU, inserting a third key evicts
        // [1] (now the LRU), not [0].
        store.put(key([2, 0, 0]), make_chunk(2)).unwrap();
        assert_eq!(store.stats().disk_reloads, 0);
        let _ = store.get(key([0, 0, 0])).unwrap().expect("[0] stayed resident");
        assert_eq!(store.stats().disk_reloads, 0, "[0] was MRU after overwrite, not evicted");
        let _ = store.get(key([1, 0, 0])).unwrap().expect("[1] reloads from disk");
        assert_eq!(store.stats().disk_reloads, 1, "[1] was the LRU victim");
    }

    /// Re-putting an evicted key brings it resident and deletes its stale disk file
    /// (a later get returns the NEW value without a reload).
    #[test]
    fn reput_evicted_key_supersedes_disk_copy() {
        let temp = TempDir::new("reput_evicted");
        let mut store = DiskChunkStore::new(&temp.path, 1).unwrap();
        store.put(key([0, 0, 0]), make_chunk(0)).unwrap();
        store.put(key([1, 0, 0]), make_chunk(1)).unwrap(); // evicts [0] to disk
        assert!(store.stats().on_disk_count >= 1);

        // Re-put [0] with a NEW value: it becomes resident (evicting [1]).
        let fresh = make_chunk(200);
        store.put(key([0, 0, 0]), fresh.clone()).unwrap();
        let got = store.get(key([0, 0, 0])).unwrap().expect("resident again");
        assert_eq!(store.stats().disk_reloads, 0, "[0] was resident — no reload");
        assert_eq!(got, fresh, "the re-put value supersedes the stale disk copy");
    }

    /// The store directory is created idempotently (constructing twice on the same
    /// dir is fine, and a pre-existing dir is reused).
    #[test]
    fn directory_creation_is_idempotent() {
        let temp = TempDir::new("idempotent_dir");
        let _store_a = DiskChunkStore::new(&temp.path, 2).unwrap();
        // Constructing again on the same (now-existing) directory must succeed.
        let store_b = DiskChunkStore::new(&temp.path, 2);
        assert!(store_b.is_ok(), "re-creating the store on an existing dir is a no-op");
    }
}
