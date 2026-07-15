//! The store's pure, GPU-free residency planner: the set-difference over coord sets
//! that decides which chunks a per-chunk render cache must (re)build and which to drop.

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
/// resolve cache (see [`Store::invalidate_aabb`](super::Store::invalidate_aabb)).
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
