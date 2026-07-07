//! A spatial index over leaf nodes' world-AABBs (issue #27 S3).
//!
//! S2 made resolve chunk-addressable + cached, but invalidation stayed
//! all-or-nothing: every edit `clear()`s the whole cache. S3 narrows that to
//! **whole-chunk dirty invalidation** (ADR 0002 Decision 3): an edit dirties only
//! the chunks whose AABB its world-AABB intersects.
//!
//! Three pieces live here:
//!
//! * [`VoxelAabb`] — a half-open integer **absolute-voxel** box `[min, max)`, the
//!   exact frame [`Scene::resolve_chunk`](crate::scene::Scene::resolve_chunk) and
//!   chunk ownership (`floor(position / chunk_extent)`) live in. Using the same
//!   frame is what lets a query AABB be turned into the precise set of chunk
//!   coordinates it touches.
//! * [`LeafSpatialIndex`] — a flat list of `(leaf_world_aabb, fingerprint)` built
//!   by one `for_each_leaf` walk of the scene. It answers "which leaves' world-
//!   AABBs intersect a query AABB" (a linear overlap scan), and "which chunks did
//!   an edit dirty" by diffing two indices (the scene before vs after the edit).
//! * [`EditBroadphaseBvh`] — THE edit broadphase (ADR 0011 Decision 4b, #66): a
//!   per-build BVH over producer world-AABBs answering "which producers overlap
//!   this box" for the two-layer wholesale build (and, later, G3's dirty-AABB →
//!   producers query).
//!
//! ## Why a flat list (not an octree / grid)
//!
//! The index must return the **same** leaf set a full `for_each_leaf` walk filtered
//! by AABB returns — that is the correctness contract S3 must prove. The simplest
//! structure that *is* exactly that walk, filtered, is a flat `Vec` of the walk's
//! per-leaf AABBs scanned linearly. Leaf counts are small (tens for the demo
//! scene, low hundreds for `--demo-village`'s instanced houses), so a linear scan
//! per chunk is cheap and obviously correct; a fancier acceleration structure (a
//! uniform grid or loose octree) would add a divergence risk (a leaf dropped or
//! double-counted by a bucketing bug) for no measurable win at v1 scene sizes. We
//! keep it flat and provably-equal-to-the-walk; if scenes ever grow to where the
//! linear scan dominates, the structure can be swapped behind this same API.

use crate::core_geom::CHUNK_BLOCKS;

/// A half-open integer box `[min, max)` in **absolute voxel** coordinates — the
/// frame the chunk decomposition owns (a voxel at absolute position `p` belongs to
/// chunk `floor(p / chunk_extent)`). Empty when any `min[axis] >= max[axis]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoxelAabb {
    /// Inclusive minimum corner (absolute voxels).
    ///
    /// **i64 (S4a, 64-bit world addressing):** absolute voxels = block offset ×
    /// density, so a far-placed leaf (a node at ±10⁹ blocks × density 16 ≈ ±1.6×10¹⁰
    /// voxels) overflows i32 — the corner MUST be i64 or the producer-true frame
    /// silently truncates. The derived CHUNK coordinate stays i32 (see
    /// [`Self::covering_chunk_range`]).
    pub min: [i64; 3],
    /// Exclusive maximum corner (absolute voxels).
    pub max: [i64; 3],
}

impl VoxelAabb {
    /// A box spanning `[min, max)`.
    pub fn new(min: [i64; 3], max: [i64; 3]) -> Self {
        Self { min, max }
    }

    /// Whether the box is empty (no voxel lies inside it on some axis).
    pub fn is_empty(&self) -> bool {
        (0..3).any(|axis| self.min[axis] >= self.max[axis])
    }

    /// Whether two half-open boxes overlap (share at least one voxel cell). Touching
    /// faces (one box's `max` equals the other's `min`) do **not** overlap — the
    /// half-open convention, matching chunk ownership.
    pub fn intersects(&self, other: &VoxelAabb) -> bool {
        if self.is_empty() || other.is_empty() {
            return false;
        }
        (0..3).all(|axis| self.min[axis] < other.max[axis] && other.min[axis] < self.max[axis])
    }

    /// The smallest box containing both inputs (an empty box contributes nothing).
    pub fn union(&self, other: &VoxelAabb) -> VoxelAabb {
        if self.is_empty() {
            return *other;
        }
        if other.is_empty() {
            return *self;
        }
        VoxelAabb {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    /// The inclusive range of chunk coordinates `[min_chunk, max_chunk]` whose
    /// half-open boxes this AABB intersects, at the given density. `None` when the
    /// box is empty. Mirrors
    /// [`Scene::covering_chunk_range`](crate::scene::Scene::covering_chunk_range):
    /// the lowest chunk owns `min`, the highest owns `max - 1` (the last occupied
    /// voxel of the half-open box).
    pub fn covering_chunk_range(&self, voxels_per_block: u32) -> Option<([i32; 3], [i32; 3])> {
        if self.is_empty() {
            return None;
        }
        // Voxel corners are i64 (a far-placed leaf); the chunk extent is small, so
        // the division happens in i64 and the chunk-coord QUOTIENT narrows to i32
        // safely (≤ ±2.5×10⁸ for offsets up to ±10⁹ blocks — S4a).
        let chunk_extent_voxels = (CHUNK_BLOCKS * voxels_per_block.max(1)) as i64;
        let mut min_chunk = [0i32; 3];
        let mut max_chunk = [0i32; 3];
        for axis in 0..3 {
            min_chunk[axis] = narrow_chunk_coord(self.min[axis].div_euclid(chunk_extent_voxels));
            max_chunk[axis] = narrow_chunk_coord((self.max[axis] - 1).div_euclid(chunk_extent_voxels));
        }
        Some((min_chunk, max_chunk))
    }
}

/// A content fingerprint distinguishing two leaves that occupy the SAME world-AABB
/// but emit DIFFERENT voxels (e.g. a recoloured Tool, or a swapped shape kind at an
/// identical bounding box). The edit diff ([`LeafSpatialIndex::edit_aabb_since`])
/// must dirty a leaf whose voxels changed even when its box did not, so the
/// fingerprint is compared alongside the AABB.
///
/// It is derived from the bytes of the leaf's [`NodeContent`] that affect the
/// resolved voxels. `RegionSpanning` marks a leaf with no intrinsic AABB (a Part
/// such as the debug-cloud field, whose voxels fill the whole composite region):
/// such a leaf cannot be localised to chunks, so any edit touching it forces a
/// wholesale clear (see [`LeafSpatialIndex::edit_aabb_since`]).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LeafFingerprint {
    /// A localisable leaf with a concrete world-AABB; the payload identifies its
    /// resolved content so a same-box content change is still detected.
    Bounded(String),
    /// A leaf with no intrinsic AABB (region-spanning), e.g. a Part. Carries its
    /// content bytes so a Part edit is still seen as a change, but its presence in a
    /// diff forces a wholesale clear (it cannot be chunk-localised).
    RegionSpanning(String),
}

/// One leaf's entry in the index: its world-AABB (absolute voxels) plus a content
/// fingerprint. The AABB is `None`/empty for a region-spanning leaf.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeafEntry {
    /// The leaf's world-AABB in absolute voxels (the producer-true frame). Empty for
    /// a region-spanning leaf (it has no intrinsic box).
    pub world_aabb: VoxelAabb,
    /// What the leaf resolves to (so a same-box content change is detected).
    pub fingerprint: LeafFingerprint,
}

/// A flat spatial index over a scene's leaf world-AABBs at a fixed density.
///
/// Built by [`Scene::build_leaf_spatial_index`](crate::scene::Scene::build_leaf_spatial_index)
/// from a single `for_each_leaf` walk, so the set of entries is — by construction —
/// exactly the leaves that walk yields. Queried by AABB overlap
/// ([`leaves_intersecting`](Self::leaves_intersecting)); diffed against a previous
/// index to compute an edit's dirty AABB ([`edit_aabb_since`](Self::edit_aabb_since)).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LeafSpatialIndex {
    /// The per-leaf entries, in `for_each_leaf` (depth-first) order.
    pub entries: Vec<LeafEntry>,
    /// The density the AABBs were computed at (an index is only comparable to
    /// another at the same density).
    pub voxels_per_block: u32,
    /// Whether the scene contains a region-spanning leaf (a Part). When `true`, a
    /// precise edit AABB can't always be computed; see [`edit_aabb_since`].
    pub has_region_spanning_leaf: bool,
}

impl LeafSpatialIndex {
    /// The leaves whose world-AABBs intersect `query` (a linear overlap scan).
    /// Region-spanning leaves (empty AABB) never match an AABB query — they are not
    /// localisable; callers that must account for them use
    /// [`has_region_spanning_leaf`](Self::has_region_spanning_leaf).
    pub fn leaves_intersecting(&self, query: &VoxelAabb) -> Vec<&LeafEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.world_aabb.intersects(query))
            .collect()
    }

    /// The union of every leaf world-AABB in the index (the scene's whole occupied
    /// box), or an empty box when there are no bounded leaves.
    pub fn bounding_aabb(&self) -> VoxelAabb {
        let mut acc = VoxelAabb::new([0; 3], [0; 3]);
        for entry in &self.entries {
            acc = acc.union(&entry.world_aabb);
        }
        acc
    }

    /// The world-AABB an edit dirtied, computed by diffing this index (the scene
    /// AFTER the edit) against `previous` (the scene BEFORE it).
    ///
    /// The dirty AABB is the **union of every leaf whose (AABB, fingerprint) pair
    /// changed** — present in exactly one of the two indices (the symmetric
    /// difference as multisets). This captures every edit kind uniformly:
    ///
    /// * **Move / offset** — the leaf's old box and new box are both in the
    ///   difference, so the union spans BOTH locations (dirtying chunks around the
    ///   source and the destination).
    /// * **Add** — the new leaf is only in `self`; its box is dirtied.
    /// * **Remove** — the old leaf is only in `previous`; its box is dirtied.
    /// * **Edit in place (resize / recolour / shape swap)** — the (AABB,
    ///   fingerprint) pair differs, so both old and new boxes are dirtied.
    ///
    /// Returns:
    /// * `Some(aabb)` — invalidate exactly the chunks `aabb` intersects.
    /// * `Some(empty)` (an empty AABB) — nothing changed; invalidate nothing.
    /// * `None` — a **conservative fallback**: the caller must `clear()` the whole
    ///   cache. This happens when (a) the two indices were built at different
    ///   densities (every chunk's voxel extent changed), or (b) a **region-spanning**
    ///   leaf (a Part) was added, removed, or edited — it has no localisable box, so
    ///   its dirty region is "everywhere".
    pub fn edit_aabb_since(&self, previous: &LeafSpatialIndex) -> Option<VoxelAabb> {
        if self.voxels_per_block != previous.voxels_per_block {
            // A density change resizes every chunk; nothing is reusable.
            return None;
        }

        // Multiset symmetric difference on (AABB, fingerprint). Two leaves that are
        // byte-identical in placement AND content cancel out; everything else is a
        // change that must be dirtied. A region-spanning leaf appearing in the
        // difference forces a wholesale clear.
        use std::collections::HashMap;
        let mut counts: HashMap<(VoxelAabbKey, &LeafFingerprint), i64> = HashMap::new();
        for entry in &self.entries {
            *counts
                .entry((VoxelAabbKey::from(entry.world_aabb), &entry.fingerprint))
                .or_insert(0) += 1;
        }
        for entry in &previous.entries {
            *counts
                .entry((VoxelAabbKey::from(entry.world_aabb), &entry.fingerprint))
                .or_insert(0) -= 1;
        }

        let mut dirty = VoxelAabb::new([0; 3], [0; 3]);
        for ((aabb_key, fingerprint), count) in counts {
            if count == 0 {
                continue; // unchanged leaf — cancels between the two indices.
            }
            match fingerprint {
                LeafFingerprint::RegionSpanning(_) => {
                    // A Part changed (added/removed/edited): its dirty region is the
                    // whole scene — fall back to a wholesale clear.
                    return None;
                }
                LeafFingerprint::Bounded(_) => {
                    dirty = dirty.union(&aabb_key.into());
                }
            }
        }
        Some(dirty)
    }
}

/// **The edit broadphase (ADR 0011 Decision 4b, #66): a BVH over producer world-AABBs.**
///
/// Answers "which input boxes overlap this query box" — the ONE broadphase query on the
/// edit side. The two-layer wholesale build queries it once per covering chunk (chunk box
/// → candidate producers), and G3's incremental dirty-brick recompute (#69) later reuses
/// the same query for dirty-edit-AABB → overlapping producers. It replaces the #63
/// per-chunk bucketing (`bucket_leaves_into_chunks`): one query, one structure.
///
/// **Stateless by ruling:** rebuilt from scratch per build/edit, never persisted or
/// refitted across edits — no invalidation obligation, no stale-cache bug class (the C1
/// lesson). Construction is a median-split over box centroids, O(N log N) (~1ms at the
/// 10k-producer target); a persistent/refitted BVH is a measured future option only if
/// the rebuild ever shows hot.
///
/// **Exactness:** a query returns EXACTLY the input indices whose box overlaps the query
/// (the unit net asserts equality with a naive linear filter), **sorted ascending** — so
/// a caller that supplied boxes in document order gets candidates back as a document-order
/// subsequence (a filter, never a reorder; Union later-wins material resolution is
/// preserved). Empty input boxes overlap nothing and are excluded at construction, the
/// same half-open [`VoxelAabb::intersects`] convention as every other consumer.
#[derive(Debug, Clone, Default)]
pub struct EditBroadphaseBvh {
    /// Depth-first flattened nodes: a node's LEFT child is `node_index + 1`; the RIGHT
    /// child index is stored. Empty when no input box was non-empty.
    nodes: Vec<EditBroadphaseNode>,
    /// The input indices of the non-empty boxes, reordered by construction; a leaf node
    /// owns the contiguous `[entry_start, entry_start + entry_count)` slice.
    entry_input_indices: Vec<u32>,
    /// The boxes parallel to `entry_input_indices` (so the per-entry overlap test reads
    /// the reordered slice, never the caller's).
    entry_aabbs: Vec<VoxelAabb>,
}

/// One BVH node: the bounds of every entry under it, plus its internal/leaf payload.
#[derive(Debug, Clone, Copy)]
struct EditBroadphaseNode {
    /// The union of every entry box in this subtree.
    aabb: VoxelAabb,
    kind: EditBroadphaseNodeKind,
}

#[derive(Debug, Clone, Copy)]
enum EditBroadphaseNodeKind {
    /// Two children: the left is the next node depth-first, the right is stored.
    Internal { right_child: u32 },
    /// A run of entries in [`EditBroadphaseBvh::entry_input_indices`] / `entry_aabbs`.
    Leaf { entry_start: u32, entry_count: u32 },
}

/// Entries per leaf node before a subtree stops splitting. Small enough that the
/// per-leaf linear overlap test stays trivial, large enough to keep the node count
/// (and construction cost) down.
const EDIT_BROADPHASE_LEAF_CAPACITY: usize = 8;

impl EditBroadphaseBvh {
    /// Build the BVH over `input_aabbs`; index `i` of the slice is the index a query
    /// reports. Empty boxes are excluded (they overlap nothing).
    pub fn build(input_aabbs: &[VoxelAabb]) -> Self {
        let mut entries: Vec<(u32, VoxelAabb)> = input_aabbs
            .iter()
            .enumerate()
            .filter(|(_, aabb)| !aabb.is_empty())
            .map(|(input_index, aabb)| (input_index as u32, *aabb))
            .collect();
        let mut nodes = Vec::new();
        if !entries.is_empty() {
            build_edit_broadphase_subtree(&mut nodes, &mut entries, 0);
        }
        Self {
            nodes,
            entry_input_indices: entries.iter().map(|(input_index, _)| *input_index).collect(),
            entry_aabbs: entries.iter().map(|(_, aabb)| *aabb).collect(),
        }
    }

    /// Every input index whose box overlaps `query`, **sorted ascending** (= input order:
    /// a document-order caller gets a document-order subsequence). Exactly the set a
    /// naive linear `intersects` filter over the input slice returns.
    pub fn overlapping_input_indices(&self, query: &VoxelAabb) -> Vec<usize> {
        let mut overlapping = Vec::new();
        if self.nodes.is_empty() || query.is_empty() {
            return overlapping;
        }
        let mut pending_nodes: Vec<u32> = vec![0];
        while let Some(node_index) = pending_nodes.pop() {
            let node = &self.nodes[node_index as usize];
            if !node.aabb.intersects(query) {
                continue;
            }
            match node.kind {
                EditBroadphaseNodeKind::Internal { right_child } => {
                    pending_nodes.push(node_index + 1);
                    pending_nodes.push(right_child);
                }
                EditBroadphaseNodeKind::Leaf {
                    entry_start,
                    entry_count,
                } => {
                    for entry in entry_start..entry_start + entry_count {
                        if self.entry_aabbs[entry as usize].intersects(query) {
                            overlapping.push(self.entry_input_indices[entry as usize] as usize);
                        }
                    }
                }
            }
        }
        // Traversal order is tree order, not input order — restore input (document) order.
        overlapping.sort_unstable();
        overlapping
    }
}

/// Recursively emit the subtree over `entries` (which it reorders in place; the subtree's
/// leaf runs index into the final reordered array at `entry_offset`). Returns the emitted
/// root's node index. Median split on the longest axis of the CENTROID bounds — a
/// balanced tree of depth `log2(N / capacity)`, so recursion stays shallow (~10 at 10k).
fn build_edit_broadphase_subtree(
    nodes: &mut Vec<EditBroadphaseNode>,
    entries: &mut [(u32, VoxelAabb)],
    entry_offset: usize,
) -> u32 {
    let node_index = nodes.len() as u32;
    let mut bounds = entries[0].1;
    for (_, aabb) in entries.iter().skip(1) {
        bounds = bounds.union(aabb);
    }

    if entries.len() <= EDIT_BROADPHASE_LEAF_CAPACITY {
        nodes.push(EditBroadphaseNode {
            aabb: bounds,
            kind: EditBroadphaseNodeKind::Leaf {
                entry_start: entry_offset as u32,
                entry_count: entries.len() as u32,
            },
        });
        return node_index;
    }

    // Split axis = the widest spread of the DOUBLED centroids (min + max; halving would
    // only lose parity information). Coincident centroids on every axis still split fine:
    // the median partition then just halves the run arbitrarily.
    let doubled_centroid =
        |aabb: &VoxelAabb, axis: usize| aabb.min[axis] + aabb.max[axis];
    let split_axis = (0..3)
        .max_by_key(|&axis| {
            let low = entries
                .iter()
                .map(|(_, aabb)| doubled_centroid(aabb, axis))
                .min()
                .expect("entries is non-empty");
            let high = entries
                .iter()
                .map(|(_, aabb)| doubled_centroid(aabb, axis))
                .max()
                .expect("entries is non-empty");
            high - low
        })
        .expect("three axes");
    let middle = entries.len() / 2;
    entries.select_nth_unstable_by_key(middle, |(_, aabb)| doubled_centroid(aabb, split_axis));

    // Placeholder; the right child index is known only after the left subtree is emitted.
    nodes.push(EditBroadphaseNode {
        aabb: bounds,
        kind: EditBroadphaseNodeKind::Internal { right_child: 0 },
    });
    let (left_entries, right_entries) = entries.split_at_mut(middle);
    build_edit_broadphase_subtree(nodes, left_entries, entry_offset);
    let right_child =
        build_edit_broadphase_subtree(nodes, right_entries, entry_offset + middle);
    nodes[node_index as usize].kind = EditBroadphaseNodeKind::Internal { right_child };
    node_index
}

/// A hashable/orderable mirror of [`VoxelAabb`] for use as a map key in the diff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct VoxelAabbKey {
    min: [i64; 3],
    max: [i64; 3],
}

/// Narrow an `i64` chunk coordinate to `i32` (the cache-key / chunk-index width).
/// See the audit note on
/// [`Scene::covering_chunk_range`](crate::scene::Scene::covering_chunk_range): the
/// absolute-voxel math is i64, but the chunk coordinate stays well inside i32 for
/// the supported offset range (S4a).
fn narrow_chunk_coord(chunk_coord: i64) -> i32 {
    debug_assert!(
        chunk_coord >= i32::MIN as i64 && chunk_coord <= i32::MAX as i64,
        "chunk coordinate {chunk_coord} overflows i32 — block offset past the \
         supported range (S4a)"
    );
    chunk_coord.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

impl From<VoxelAabb> for VoxelAabbKey {
    fn from(aabb: VoxelAabb) -> Self {
        Self {
            min: aabb.min,
            max: aabb.max,
        }
    }
}

impl From<VoxelAabbKey> for VoxelAabb {
    fn from(key: VoxelAabbKey) -> Self {
        VoxelAabb::new(key.min, key.max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersects_is_half_open() {
        let a = VoxelAabb::new([0, 0, 0], [10, 10, 10]);
        // Overlapping box.
        assert!(a.intersects(&VoxelAabb::new([5, 5, 5], [15, 15, 15])));
        // Touching faces (b.min == a.max) do NOT overlap (half-open).
        assert!(!a.intersects(&VoxelAabb::new([10, 0, 0], [20, 10, 10])));
        // Fully separate.
        assert!(!a.intersects(&VoxelAabb::new([100, 0, 0], [110, 10, 10])));
        // Empty box never intersects.
        assert!(!a.intersects(&VoxelAabb::new([0, 0, 0], [0, 0, 0])));
    }

    #[test]
    fn union_ignores_empty() {
        let empty = VoxelAabb::new([0, 0, 0], [0, 0, 0]);
        let b = VoxelAabb::new([3, 3, 3], [7, 7, 7]);
        assert_eq!(empty.union(&b), b);
        assert_eq!(b.union(&empty), b);
        let a = VoxelAabb::new([-2, 0, 0], [4, 4, 4]);
        assert_eq!(a.union(&b), VoxelAabb::new([-2, 0, 0], [7, 7, 7]));
    }

    /// The BVH's whole contract: a query returns EXACTLY the input indices a naive linear
    /// `intersects` filter returns, sorted ascending (input = document order). Exercised
    /// over a deterministic pseudo-random population including empty boxes, duplicates,
    /// nested boxes, and far-flung outliers, with queries of every flavour (miss, point-ish,
    /// spanning, everything, empty).
    #[test]
    fn edit_broadphase_bvh_matches_naive_filter() {
        // Small deterministic LCG so the population is reproducible without a rand dep.
        let mut lcg_state = 0x1234_5678_9abc_def0_u64;
        let mut next_in = |range: i64| -> i64 {
            lcg_state = lcg_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((lcg_state >> 33) as i64).rem_euclid(range)
        };

        let mut boxes = Vec::new();
        for _ in 0..300 {
            let min = [
                next_in(400) - 200,
                next_in(400) - 200,
                next_in(400) - 200,
            ];
            let extent = [next_in(60), next_in(60), next_in(60)]; // 0 ⇒ empty box
            boxes.push(VoxelAabb::new(
                min,
                [min[0] + extent[0], min[1] + extent[1], min[2] + extent[2]],
            ));
        }
        // Duplicates, a nested pair, an everything-box, and far-flung outliers (i64 range).
        boxes.push(boxes[0]);
        boxes.push(VoxelAabb::new([-500, -500, -500], [500, 500, 500]));
        boxes.push(VoxelAabb::new([-10, -10, -10], [-5, -5, -5]));
        boxes.push(VoxelAabb::new([-9, -9, -9], [-6, -6, -6]));
        boxes.push(VoxelAabb::new(
            [16_000_000_000, 0, 0],
            [16_000_000_064, 64, 64],
        ));

        let bvh = EditBroadphaseBvh::build(&boxes);
        let queries = [
            VoxelAabb::new([0, 0, 0], [64, 64, 64]),
            VoxelAabb::new([-200, -200, -200], [200, 200, 200]),
            VoxelAabb::new([7, 7, 7], [8, 8, 8]),
            VoxelAabb::new([10_000, 10_000, 10_000], [10_064, 10_064, 10_064]), // miss
            VoxelAabb::new([15_999_999_999, 0, 0], [16_000_000_001, 1, 1]), // outlier hit
            VoxelAabb::new([0, 0, 0], [0, 0, 0]),                           // empty query
        ];
        for query in &queries {
            let naive: Vec<usize> = boxes
                .iter()
                .enumerate()
                .filter(|(_, aabb)| aabb.intersects(query))
                .map(|(input_index, _)| input_index)
                .collect();
            assert_eq!(
                bvh.overlapping_input_indices(query),
                naive,
                "BVH candidates for {query:?} must equal the naive filter, ascending"
            );
        }
    }

    /// Degenerate populations: no boxes at all, and all-empty boxes, both yield an index
    /// that answers every query with nothing (and never panics).
    #[test]
    fn edit_broadphase_bvh_handles_empty_populations() {
        let query = VoxelAabb::new([-100, -100, -100], [100, 100, 100]);
        assert!(EditBroadphaseBvh::build(&[])
            .overlapping_input_indices(&query)
            .is_empty());
        let all_empty = [
            VoxelAabb::new([0, 0, 0], [0, 0, 0]),
            VoxelAabb::new([5, 5, 5], [5, 9, 9]),
        ];
        assert!(EditBroadphaseBvh::build(&all_empty)
            .overlapping_input_indices(&query)
            .is_empty());
    }

    #[test]
    fn covering_chunk_range_matches_chunk_ownership() {
        // density 16, CHUNK_BLOCKS 4 → chunk extent 64 voxels.
        let extent = (CHUNK_BLOCKS * 16) as i32;
        assert_eq!(extent, 64);
        // A box wholly inside chunk 0.
        let a = VoxelAabb::new([1, 1, 1], [10, 10, 10]);
        assert_eq!(a.covering_chunk_range(16), Some(([0, 0, 0], [0, 0, 0])));
        // A box straddling chunk -1 and 0 on X (negative coords use div_euclid).
        let b = VoxelAabb::new([-1, 1, 1], [10, 10, 10]);
        assert_eq!(b.covering_chunk_range(16), Some(([-1, 0, 0], [0, 0, 0])));
        // A box ending exactly on a chunk boundary covers only the lower chunk
        // (half-open: last occupied voxel is max-1 = 63 → chunk 0).
        let c = VoxelAabb::new([0, 0, 0], [64, 64, 64]);
        assert_eq!(c.covering_chunk_range(16), Some(([0, 0, 0], [0, 0, 0])));
        // One voxel past the boundary reaches chunk 1.
        let d = VoxelAabb::new([0, 0, 0], [65, 64, 64]);
        assert_eq!(d.covering_chunk_range(16), Some(([0, 0, 0], [1, 0, 0])));
        // Empty box → no range.
        assert_eq!(VoxelAabb::new([0, 0, 0], [0, 0, 0]).covering_chunk_range(16), None);
    }
}
