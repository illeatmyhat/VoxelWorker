//! A spatial index over leaf nodes' world-AABBs (issue #27 S3).
//!
//! S2 made resolve chunk-addressable + cached, but invalidation stayed
//! all-or-nothing: every edit `clear()`s the whole cache. S3 narrows that to
//! **whole-chunk dirty invalidation** (ADR 0002 Decision 3): an edit dirties only
//! the chunks whose AABB its world-AABB intersects.
//!
//! The domain seam for the substrate spatial primitives lives here:
//!
//! * [`VoxelAabb`] — the domain name for substrate's half-open integer box, read as an
//!   **absolute-voxel** box `[min, max)`: the exact frame
//!   [`Scene::resolve_chunk`](crate::scene::Scene::resolve_chunk) and chunk ownership
//!   (`floor(position / chunk_extent)`) live in. [`ChunkCoverage`] adds the domain-only
//!   reading (a box → the chunk coordinates it touches) that substrate's pure box omits.
//! * [`EditBroadphaseBvh`] — the domain name for substrate's `Bvh` used as THE edit
//!   broadphase (ADR 0011 Decision 4b, #66): a per-build BVH over producer world-AABBs
//!   answering "which producers overlap this box" for the two-layer wholesale build
//!   (and, later, G3's dirty-AABB → producers query).
//! * [`LeafSpatialIndex`] — the genuinely domain-shaped piece (it must equal the
//!   `for_each_leaf` walk, by design): a flat list of `(leaf_world_aabb, fingerprint)`
//!   built by one walk of the scene. It answers "which leaves' world-AABBs intersect a
//!   query AABB" (a linear overlap scan), and "which chunks did an edit dirty" by
//!   diffing two indices (the scene before vs after the edit).
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

/// The domain name for the substrate half-open integer box [`substrate::Aabb`], read
/// as an **absolute-voxel** box `[min, max)` — the frame the chunk decomposition owns
/// (a voxel at absolute position `p` belongs to chunk `floor(p / chunk_extent)`).
///
/// Kept as a domain alias rather than renamed to the bare `Aabb` deliberately: the
/// codebase already has a *floating-point* [`crate::frustum::Aabb`] for frustum
/// culling, and `VoxelAabb` keeps the integer-voxel box unambiguous at every call site.
/// The corners are `i64` (64-bit world addressing): absolute voxels = block offset ×
/// density, so a far-placed leaf (±10⁹ blocks × density) overflows i32; the derived
/// CHUNK coordinate stays i32 (see [`ChunkCoverage::covering_chunk_range`]). See
/// `docs/architecture/data-structures.md` (the Substrate section) for the box itself.
pub use substrate::Aabb as VoxelAabb;

/// The domain name for the substrate [`substrate::Bvh`] used as **THE edit broadphase**
/// (ADR 0011 Decision 4b, #66): a bounding-volume hierarchy over producer world-AABBs
/// answering "which producers overlap this chunk box", rebuilt per wholesale build /
/// edit and never persisted (the C1 stale-cache lesson). The domain construction and
/// per-chunk query live at the seam in `two_layer_store` (`leaf_edit_broadphase`,
/// `chunk_candidate_leaves`); see `docs/architecture/02-evaluation.md`.
pub use substrate::Bvh as EditBroadphaseBvh;

/// The domain reading of a [`VoxelAabb`]: the inclusive chunk-coordinate range
/// `[min_chunk, max_chunk]` whose half-open chunk boxes the AABB intersects, at a given
/// density. This lives in the domain (not on substrate's pure box) because it is
/// defined by the chunk decomposition — `CHUNK_BLOCKS` and the density — see
/// `docs/architecture/02-evaluation.md` (chunk addressing).
pub trait ChunkCoverage {
    /// The inclusive `[min_chunk, max_chunk]` range, or `None` when the box is empty.
    /// The lowest chunk owns `min`, the highest owns `max - 1` (the last occupied voxel
    /// of the half-open box). Mirrors
    /// [`Scene::covering_chunk_range`](crate::scene::Scene::covering_chunk_range).
    fn covering_chunk_range(&self, voxels_per_block: u32) -> Option<([i32; 3], [i32; 3])>;
}

impl ChunkCoverage for VoxelAabb {
    fn covering_chunk_range(&self, voxels_per_block: u32) -> Option<([i32; 3], [i32; 3])> {
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
            max_chunk[axis] =
                narrow_chunk_coord((self.max[axis] - 1).div_euclid(chunk_extent_voxels));
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
