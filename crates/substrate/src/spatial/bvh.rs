//! A bounding-volume hierarchy over integer AABBs, built by spatial-median split.
//!
//! `Bvh` is the textbook bounding-volume hierarchy (BVH): a binary tree whose leaves
//! own a run of input boxes and whose internal nodes carry the union of their
//! subtree's boxes, so a box query prunes whole subtrees that cannot overlap. It
//! answers the broadphase question "which input boxes overlap this query box" in
//! sub-linear expected time, returning the input indices **sorted ascending** (so a
//! caller that fed boxes in a meaningful order gets a subsequence of that order back,
//! never a reordering).
//!
//! ## Build policy — spatial-median split, no SAH
//!
//! Construction partitions the entries at the **median of their centroids along the
//! longest-spread axis** (`select_nth_unstable_by_key`, O(N) partition, O(N log N)
//! overall), stopping when a run reaches [`BVH_LEAF_CAPACITY`]. It uses neither a
//! surface-area heuristic nor any query-cost model: this hierarchy is **rebuilt from
//! scratch on demand** rather than persisted and refitted, so build speed dominates
//! and query optimality does not pay for the extra build cost. A persistent,
//! refitted, or SAH-partitioned BVH is a measured future option, not this structure.
//!
//! Nodes are stored in one flat `Vec` in depth-first pre-order: a node's LEFT child is
//! the next node (`node_index + 1`), and the RIGHT child index is stored in the node —
//! the standard flattened-DFS layout that makes traversal an index walk with no
//! per-node pointers.
//!
//! The build policy is the build-speed-over-query trade for a hierarchy rebuilt per use:
//! spatial-median split on the centroid of the longest axis, fixed leaf cap 8, no SAH.

use crate::spatial::aabb::LatticeAabb;

/// Entries per leaf node before a subtree stops splitting. Small enough that the
/// per-leaf linear overlap test stays trivial, large enough to keep the node count
/// (and construction cost) down.
pub const BVH_LEAF_CAPACITY: usize = 8;

/// A bounding-volume hierarchy over a slice of [`LatticeAabb`]s, built by spatial-median
/// split and queried by box overlap. Empty input boxes overlap nothing and are
/// excluded at construction, following the half-open [`LatticeAabb::intersects`] convention.
#[derive(Debug, Clone, Default)]
pub struct Bvh {
    /// Depth-first flattened nodes: a node's LEFT child is `node_index + 1`; the RIGHT
    /// child index is stored. Empty when no input box was non-empty.
    nodes: Vec<BvhNode>,
    /// The input indices of the non-empty boxes, reordered by construction; a leaf node
    /// owns the contiguous `[entry_start, entry_start + entry_count)` slice.
    entry_input_indices: Vec<u32>,
    /// The boxes parallel to `entry_input_indices` (so the per-entry overlap test reads
    /// the reordered slice, never the caller's).
    entry_aabbs: Vec<LatticeAabb>,
}

/// One BVH node: the bounds of every entry under it, plus its internal/leaf payload.
#[derive(Debug, Clone, Copy)]
struct BvhNode {
    /// The union of every entry box in this subtree.
    aabb: LatticeAabb,
    kind: BvhNodeKind,
}

#[derive(Debug, Clone, Copy)]
enum BvhNodeKind {
    /// Two children: the left is the next node depth-first, the right is stored.
    Internal { right_child: u32 },
    /// A run of entries in [`Bvh::entry_input_indices`] / `entry_aabbs`.
    Leaf { entry_start: u32, entry_count: u32 },
}

impl Bvh {
    /// Build the BVH over `input_aabbs`; index `i` of the slice is the index a query
    /// reports. Empty boxes are excluded (they overlap nothing).
    #[must_use]
    pub fn build(input_aabbs: &[LatticeAabb]) -> Self {
        let mut entries: Vec<(u32, LatticeAabb)> = input_aabbs
            .iter()
            .enumerate()
            .filter(|(_, aabb)| !aabb.is_empty())
            .filter_map(|(input_index, aabb)| {
                u32::try_from(input_index).ok().map(|index| (index, *aabb))
            })
            .collect();
        let mut nodes = Vec::new();
        if !entries.is_empty() {
            build_bvh_subtree(&mut nodes, &mut entries, 0);
        }
        Self {
            nodes,
            entry_input_indices: entries
                .iter()
                .map(|(input_index, _)| *input_index)
                .collect(),
            entry_aabbs: entries.iter().map(|(_, aabb)| *aabb).collect(),
        }
    }

    /// Every input index whose box overlaps `query`, **sorted ascending** (= input order:
    /// an ordered caller gets an ordered subsequence). Exactly the set a naive linear
    /// `intersects` filter over the input slice returns.
    #[must_use]
    pub fn overlapping_input_indices(&self, query: &LatticeAabb) -> Vec<usize> {
        let mut overlapping = Vec::new();
        if self.nodes.is_empty() || query.is_empty() {
            return overlapping;
        }
        let mut pending_nodes: Vec<u32> = vec![0];
        while let Some(node_index) = pending_nodes.pop() {
            let Some(node) = self
                .nodes
                .get(usize::try_from(node_index).unwrap_or_default())
            else {
                continue;
            };
            if !node.aabb.intersects(query) {
                continue;
            }
            match node.kind {
                BvhNodeKind::Internal { right_child } => {
                    pending_nodes.push(node_index.saturating_add(1));
                    pending_nodes.push(right_child);
                }
                BvhNodeKind::Leaf {
                    entry_start,
                    entry_count,
                } => {
                    for entry in entry_start..entry_start.saturating_add(entry_count) {
                        let index = usize::try_from(entry).unwrap_or_default();
                        if let (Some(aabb), Some(input_index)) = (
                            self.entry_aabbs.get(index),
                            self.entry_input_indices.get(index),
                        ) {
                            if aabb.intersects(query) {
                                overlapping.push(usize::try_from(*input_index).unwrap_or_default());
                            }
                        }
                    }
                }
            }
        }
        // Traversal order is tree order, not input order — restore input order.
        overlapping.sort_unstable();
        overlapping
    }
}

/// Recursively emit the subtree over `entries` (which it reorders in place; the subtree's
/// leaf runs index into the final reordered array at `entry_offset`). Returns the emitted
/// root's node index. Median split on the longest axis of the CENTROID bounds — a
/// balanced tree of depth `log2(N / capacity)`, so recursion stays shallow (~10 at 10k).
fn build_bvh_subtree(
    nodes: &mut Vec<BvhNode>,
    entries: &mut [(u32, LatticeAabb)],
    entry_offset: usize,
) -> u32 {
    let node_index = u32::try_from(nodes.len()).unwrap_or(u32::MAX);
    let Some((_, first_aabb)) = entries.first() else {
        return node_index;
    };
    let mut bounds = *first_aabb;
    for (_, aabb) in entries.iter().skip(1) {
        bounds = bounds.union(aabb);
    }

    if entries.len() <= BVH_LEAF_CAPACITY {
        nodes.push(BvhNode {
            aabb: bounds,
            kind: BvhNodeKind::Leaf {
                entry_start: u32::try_from(entry_offset).unwrap_or(u32::MAX),
                entry_count: u32::try_from(entries.len()).unwrap_or(u32::MAX),
            },
        });
        return node_index;
    }

    // Split axis = the widest spread of the DOUBLED centroids (min + max; halving would
    // only lose parity information). Coincident centroids on every axis still split fine:
    // the median partition then just halves the run arbitrarily.
    let doubled_centroid = |aabb: &LatticeAabb, axis: usize| {
        aabb.min
            .get(axis)
            .copied()
            .unwrap_or_default()
            .saturating_add(aabb.max.get(axis).copied().unwrap_or_default())
    };
    let split_axis = (0..3)
        .max_by_key(|&axis| {
            let low = entries
                .iter()
                .map(|(_, aabb)| doubled_centroid(aabb, axis))
                .min()
                .unwrap_or_default();
            let high = entries
                .iter()
                .map(|(_, aabb)| doubled_centroid(aabb, axis))
                .max()
                .unwrap_or_default();
            high.saturating_sub(low)
        })
        .unwrap_or_default();
    let middle = entries.len() / 2;
    entries.select_nth_unstable_by_key(middle, |(_, aabb)| doubled_centroid(aabb, split_axis));

    // Placeholder; the right child index is known only after the left subtree is emitted.
    nodes.push(BvhNode {
        aabb: bounds,
        kind: BvhNodeKind::Internal { right_child: 0 },
    });
    let (left_entries, right_entries) = entries.split_at_mut(middle);
    build_bvh_subtree(nodes, left_entries, entry_offset);
    let right_child = build_bvh_subtree(nodes, right_entries, entry_offset.saturating_add(middle));
    if let Some(node) = nodes.get_mut(usize::try_from(node_index).unwrap_or_default()) {
        node.kind = BvhNodeKind::Internal { right_child };
    }
    node_index
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::all,
        clippy::arithmetic_side_effects,
        clippy::as_conversions,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::pedantic,
        clippy::nursery,
        clippy::unwrap_used
    )]
    use super::*;

    /// The BVH's whole contract: a query returns EXACTLY the input indices a naive linear
    /// `intersects` filter returns, sorted ascending (input order). Exercised over a
    /// deterministic pseudo-random population including empty boxes, duplicates, nested
    /// boxes, and far-flung outliers, with queries of every flavor (miss, point-ish,
    /// spanning, everything, empty).
    #[test]
    fn bvh_matches_naive_filter() {
        // Small deterministic LCG so the population is reproducible without a rand dep.
        let mut lcg_state = 0x1234_5678_9abc_def0_u64;
        let mut next_in = |range: i64| -> i64 {
            lcg_state = lcg_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((lcg_state >> 33) as i64).rem_euclid(range)
        };

        let mut boxes = Vec::new();
        for _ in 0..300 {
            let min = [next_in(400) - 200, next_in(400) - 200, next_in(400) - 200];
            let extent = [next_in(60), next_in(60), next_in(60)]; // 0 ⇒ empty box
            boxes.push(LatticeAabb::new(
                min,
                [min[0] + extent[0], min[1] + extent[1], min[2] + extent[2]],
            ));
        }
        // Duplicates, a nested pair, an everything-box, and far-flung outliers (i64 range).
        boxes.push(boxes[0]);
        boxes.push(LatticeAabb::new([-500, -500, -500], [500, 500, 500]));
        boxes.push(LatticeAabb::new([-10, -10, -10], [-5, -5, -5]));
        boxes.push(LatticeAabb::new([-9, -9, -9], [-6, -6, -6]));
        boxes.push(LatticeAabb::new(
            [16_000_000_000, 0, 0],
            [16_000_000_064, 64, 64],
        ));

        let bvh = Bvh::build(&boxes);
        let queries = [
            LatticeAabb::new([0, 0, 0], [64, 64, 64]),
            LatticeAabb::new([-200, -200, -200], [200, 200, 200]),
            LatticeAabb::new([7, 7, 7], [8, 8, 8]),
            LatticeAabb::new([10_000, 10_000, 10_000], [10_064, 10_064, 10_064]), // miss
            LatticeAabb::new([15_999_999_999, 0, 0], [16_000_000_001, 1, 1]),     // outlier hit
            LatticeAabb::new([0, 0, 0], [0, 0, 0]),                               // empty query
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

    /// Degenerate populations: no boxes at all, and all-empty boxes, both yield a hierarchy
    /// that answers every query with nothing (and never panics).
    #[test]
    fn bvh_handles_empty_populations() {
        let query = LatticeAabb::new([-100, -100, -100], [100, 100, 100]);
        assert!(Bvh::build(&[]).overlapping_input_indices(&query).is_empty());
        let all_empty = [
            LatticeAabb::new([0, 0, 0], [0, 0, 0]),
            LatticeAabb::new([5, 5, 5], [5, 9, 9]),
        ];
        assert!(Bvh::build(&all_empty)
            .overlapping_input_indices(&query)
            .is_empty());
    }
}
