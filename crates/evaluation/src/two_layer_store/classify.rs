//! The interval-bound block classifier (air / coarse-solid / boundary) + boundary-block per-voxel resolve + seam-solidity computation.


use substrate::solids::{
    CellCombineOp, CellContribution, ScopedCellClassification, ScopedCellEvent,
};

use voxel_core::core_geom::{BlockId, CellKey};
use crate::cuboid::{decompose_into_boxes, VoxelRegion};
use document::scene::{CombineOp, LeafProducer, ScopeFrame};
use voxel_core::spatial_index::VoxelAabb;
use voxel_core::voxel::{VoxelGrid, SURFACE_ISOLEVEL};
use document::voxel::FieldClassification;

#[allow(unused_imports)]
use super::*;

/// One step of the depth-first **scoped ordered fold** (ADR 0017 Decision 3, issue #74)
/// over a document-order leaf subsequence: the scope-open / scope-close markers of the
/// push/pop evaluation, reconstructed from each leaf's carried
/// [`scope_path`](LeafProducer::scope_path) by [`scoped_leaf_steps`].
pub(crate) enum ScopedLeafStep<'leaf> {
    /// A sealed scope opens: push a fresh accumulator; subsequent leaves compose into it.
    OpenScope,
    /// The innermost scope closes: pop its accumulator and fold the composed body into
    /// the parent under the SCOPE's own operation.
    CloseScope(CombineOp),
    /// One leaf composes into the CURRENT (innermost open) accumulator under its own
    /// operation.
    Leaf(&'leaf LeafProducer),
}

/// Linearize a document-order leaf subsequence into the depth-first scoped fold's step
/// list. Scopes are contiguous in the depth-first walk, so comparing each leaf's carried
/// scope path against the currently-open frames recovers the exact close/open marker
/// sequence the walk would have emitted — including for a FILTERED subsequence (the
/// broadphase / per-block overlap filters): a dropped leaf contributes nothing to the
/// cell being evaluated, and a scope all of whose leaves were dropped simply never opens
/// (an empty scope folds to nothing under Union/Subtract — see the substrate kernel's ∅
/// identities).
///
/// `leaves` MUST be a document-order subsequence of one `Scene::leaf_producers` list (a
/// filter, never a reorder) — the same precondition the callers already carry.
pub(crate) fn scoped_leaf_steps<'leaf>(
    leaves: &[&'leaf LeafProducer],
) -> Vec<ScopedLeafStep<'leaf>> {
    let mut steps = Vec::new();
    let mut open_frames: Vec<ScopeFrame> = Vec::new();
    for &leaf in leaves {
        // The longest prefix of open frames this leaf's path keeps open.
        let mut common = 0;
        while common < open_frames.len()
            && common < leaf.scope_path.len()
            && open_frames[common] == leaf.scope_path[common]
        {
            common += 1;
        }
        // Close the scopes the leaf is no longer inside, innermost first (each close
        // carries the SCOPE's own operation, ADR 0017 Decision 3).
        while open_frames.len() > common {
            let frame = open_frames.pop().expect("len checked by the loop condition");
            steps.push(ScopedLeafStep::CloseScope(frame.operation));
        }
        // Open the leaf's scopes beyond the common prefix, outermost first.
        for frame in &leaf.scope_path[common..] {
            open_frames.push(*frame);
            steps.push(ScopedLeafStep::OpenScope);
        }
        steps.push(ScopedLeafStep::Leaf(leaf));
    }
    // Close everything still open after the last leaf.
    while let Some(frame) = open_frames.pop() {
        steps.push(ScopedLeafStep::CloseScope(frame.operation));
    }
    steps
}

/// Whether this leaf can ADD occupancy to the scene's root accumulator: its own operation
/// is `Union` and every enclosing scope folds under `Union` (ADR 0017 Decision 3). A
/// `Subtract` anywhere on the path makes the leaf's root-level influence purely
/// subtractive — e.g. a Union leaf inside a Group placed under Subtract only ever CARVES
/// the parent (its body enters the group's composed occupancy, which is then removed
/// from the parent). Purely additive leaves are also the only leaves that ever STAMP
/// material at the root (booleans never stamp — Decision 1).
pub(crate) fn leaf_is_purely_additive(leaf: &LeafProducer) -> bool {
    leaf.operation == CombineOp::Union
        && leaf
            .scope_path
            .iter()
            .all(|frame| frame.operation == CombineOp::Union)
}

/// Map the document's [`CombineOp`] onto the substrate kernel's [`CellCombineOp`] role.
fn cell_combine_role(operation: CombineOp) -> CellCombineOp {
    match operation {
        CombineOp::Union => CellCombineOp::Union,
        CombineOp::Subtract => CellCombineOp::Subtract,
    }
}

/// Classify ONE block (the absolute-voxel box `block_abs_voxels`) against the scene's
/// leaves (ADR 0010 Decision 2), composing each boundable leaf's field interval by CSG
/// interval arithmetic and overriding to BOUNDARY for any unboundable leaf.
///
/// `block_abs_voxels` is the block's half-open `[min, max)` box in the SCENE's ABSOLUTE
/// voxel frame (the frame [`Scene::resolve_chunk`](document::scene::Scene::resolve_chunk)
/// clips against — recentre is applied later as a pure index offset, so classification is
/// frame-independent). Each leaf's interval is taken in its OWN local voxel-index frame by
/// subtracting the leaf's `world_offset_voxels` (the same map
/// [`stamp_producer_into_chunk`](crate::scene) uses for its resolve window).
///
/// The fold is SCOPED (ADR 0017 Decision 3, issue #74): each leaf's interval folds into
/// its innermost enclosing Group / definition-body scope, and a closing scope's composed
/// interval folds into its parent under the SCOPE's own operation — so a boolean inside a
/// scope can never affect the verdict outside it.
///
/// Returns:
/// * [`BlockClassification::Air`] iff the scoped fold provably leaves the block empty
///   (every overlapping purely-additive leaf misses it, or a subtractive operand provably
///   carves it whole; no leaf unboundable) — the conservative interval guarantees brute
///   force finds zero voxels.
/// * [`BlockClassification::CoarseSolid`] iff a single purely-additive leaf provably
///   fills the WHOLE block solid AND the fold (which carries every boolean's intervals,
///   ADR 0017) still proves solidity — every voxel occupied, one material.
/// * [`BlockClassification::Boundary`] otherwise (straddling, multi-additive overlap that
///   can't be proven uniformly solid, or any unboundable leaf) — the always-safe verdict.
///
/// The classifier is CONSERVATIVE: a block it calls AIR or COARSE-SOLID is occupancy-
/// IDENTICAL to brute force; a block it calls BOUNDARY is resolved per-voxel and is exact
/// regardless. This is what makes the round-trip bit-identical to the dense path.
pub(crate) fn classify_chunk_block(
    leaves: &[&LeafProducer],
    block_abs_voxels: VoxelAabb,
    voxels_per_block: u32,
) -> BlockClassification {
    // Gather the leaves whose own grid AABB overlaps this block (others contribute
    // nothing to any cell in it: a dropped Union adds no occupancy, a dropped
    // Subtract carves none, and a scope whose leaves are all dropped never opens).
    // A leaf's absolute span is `[off, off + grid)` (corner-anchored).
    let overlapping: Vec<&LeafProducer> = leaves
        .iter()
        .copied()
        .filter(|leaf| leaf_world_box(leaf, voxels_per_block).intersects(&block_abs_voxels))
        .collect();

    // With the Union/Subtract roles that exist today, occupancy at the root can only
    // be CREATED by a purely additive leaf (ADR 0017: booleans — at any scope depth —
    // only ever remove). No overlapping purely-additive leaf ⇒ provably empty.
    if !overlapping.iter().any(|leaf| leaf_is_purely_additive(leaf)) {
        return BlockClassification::Air;
    }

    // Fold the per-leaf conservative field intervals through the substrate SCOPED
    // black/white/grey classifier (`ScopedCellClassification`, issue #74): each leaf's
    // interval folds into its innermost open scope under the leaf's CSG role (Union =
    // min-of-fields, Subtract = the conservative difference `max(running, −operand)`
    // — Duff 1992 interval arithmetic), and a closing scope folds its composed
    // interval into the parent under the SCOPE's role — so a cutter inside a scope
    // can never degrade a verdict outside it, and a subtract operand can only degrade
    // a verdict toward Boundary/Air-it-can-prove, never claim solidity. The kernel's
    // ∅ identities absorb a cutter that precedes everything in its scope
    // (subtract-from-nothing), so no leading-cutter drop is needed here. `None` iff
    // any leaf is unboundable ⇒ BOUNDARY. Leaf iteration, the scope-marker
    // reconstruction ([`scoped_leaf_steps`]) and the local-frame map stay HERE
    // (domain); the fold algebra lives in substrate.
    let verdict = ScopedCellClassification::classify(
        scoped_leaf_steps(&overlapping).into_iter().map(|step| match step {
            ScopedLeafStep::OpenScope => ScopedCellEvent::OpenScope,
            ScopedLeafStep::CloseScope(operation) => {
                ScopedCellEvent::CloseScope(cell_combine_role(operation))
            }
            ScopedLeafStep::Leaf(leaf) => {
                // Map the absolute block box into THIS leaf's local voxel-index frame
                // `[0, full)` by subtracting its world offset — the exact frame
                // `cell_field_interval` expects (ADR 0008: the frame is carried,
                // never re-derived).
                let cell_local = VoxelAabb::new(
                    [
                        block_abs_voxels.min[0] - leaf.world_offset_voxels[0],
                        block_abs_voxels.min[1] - leaf.world_offset_voxels[1],
                        block_abs_voxels.min[2] - leaf.world_offset_voxels[2],
                    ],
                    [
                        block_abs_voxels.max[0] - leaf.world_offset_voxels[0],
                        block_abs_voxels.max[1] - leaf.world_offset_voxels[1],
                        block_abs_voxels.max[2] - leaf.world_offset_voxels[2],
                    ],
                );
                ScopedCellEvent::Contribution(CellContribution {
                    field_interval: leaf
                        .producer
                        .cell_field_interval(cell_local, voxels_per_block),
                    combine: cell_combine_role(leaf.operation),
                })
            }
        }),
        SURFACE_ISOLEVEL,
    );

    let Some(classification) = verdict else {
        // An unboundable leaf in the fold (cannot classify) ⇒ resolve the block per-voxel.
        return BlockClassification::Boundary;
    };

    match classification {
        FieldClassification::Air => BlockClassification::Air,
        FieldClassification::Boundary => BlockClassification::Boundary,
        FieldClassification::CoarseSolid => {
            // A composed-solid verdict is only SAFELY coarse when EXACTLY ONE
            // PURELY ADDITIVE leaf overlaps the block: with two additive leaves the
            // per-voxel MATERIAL is "later wins on overlap", which the composed
            // interval cannot resolve (it proves geometric solidity, not which id
            // each voxel takes). Under ADR 0017 no boolean ever stamps material —
            // neither a Subtract leaf nor a Union leaf whose scope folds under
            // Subtract (its body only ever carves the parent) — so overlapping
            // subtractive-influence leaves do NOT disqualify the elision: the
            // scoped fold already carried their intervals, and a CoarseSolid
            // verdict PROVES they carve nothing here, so every voxel keeps the
            // single additive leaf's material. Any other mix (multiple additive
            // leaves; a single additive leaf with no single-material id, i.e. a
            // Part's per-voxel materials) is forced BOUNDARY so the per-voxel pass
            // decides — still exact, just unelided. When in doubt, Boundary: it is
            // always exact.
            let mut additive_leaves = overlapping
                .iter()
                .filter(|leaf| leaf_is_purely_additive(leaf));
            match (additive_leaves.next(), additive_leaves.next()) {
                // Single single-material additive leaf provably filling the block ⇒ coarse.
                (Some(leaf), None) => match leaf.material {
                    Some(block_id) => BlockClassification::CoarseSolid(block_id),
                    None => BlockClassification::Boundary,
                },
                // Multiple additive leaves ⇒ per-voxel later-wins material ⇒ boundary.
                _ => BlockClassification::Boundary,
            }
        }
    }
}

/// The leaf's grid AABB in the SCENE's absolute voxel frame: `[off, off + grid)`,
/// corner-anchored at its `world_offset_voxels`. The single box construction shared by the
/// classify / overlap / whole-chunk paths (they must all test the SAME leaf extent).
pub(crate) fn leaf_world_box(leaf: &LeafProducer, voxels_per_block: u32) -> VoxelAabb {
    let grid_dimensions = leaf.producer.full_dimensions(voxels_per_block);
    VoxelAabb::new(
        leaf.world_offset_voxels,
        [
            leaf.world_offset_voxels[0] + grid_dimensions[0] as i64,
            leaf.world_offset_voxels[1] + grid_dimensions[1] as i64,
            leaf.world_offset_voxels[2] + grid_dimensions[2] as i64,
        ],
    )
}

/// The FIRST **purely additive** leaf (see [`leaf_is_purely_additive`]) whose grid AABB
/// overlaps `block_abs_voxels`, or `None` if none does. The overlap test mirrors
/// [`classify_chunk_block`]'s exactly, so the same leaf is found. A coarse-solid block is
/// owned by exactly one purely additive leaf (the classifier forces any multi-additive
/// overlap to boundary, and no boolean — at any scope depth — ever stamps, ADR 0017), so
/// for a coarse verdict this first additive hit IS the single owning leaf.
pub(crate) fn single_overlapping_leaf<'a>(
    leaves: &[&'a LeafProducer],
    block_abs_voxels: VoxelAabb,
    voxels_per_block: u32,
) -> Option<&'a LeafProducer> {
    leaves
        .iter()
        .copied()
        .filter(|leaf| leaf_is_purely_additive(leaf))
        .find(|leaf| leaf_world_box(leaf, voxels_per_block).intersects(&block_abs_voxels))
}

/// The on-face-grid overlay (ADR 0003 §3c) of the SINGLE purely additive leaf overlapping
/// `block_abs_voxels`, or `false` if none overlaps (an unreachable case for a coarse-solid
/// verdict — guarded defensively).
pub(crate) fn single_overlapping_leaf_overlay(
    leaves: &[&LeafProducer],
    block_abs_voxels: VoxelAabb,
    voxels_per_block: u32,
) -> bool {
    single_overlapping_leaf(leaves, block_abs_voxels, voxels_per_block)
        .is_some_and(|leaf| leaf.grid_overlay)
}

/// The whole-CHUNK fast-path verdict (ADR 0010 Decision 2 — chunk-granular interval
/// elision). Evaluating the composed field interval ONCE at the whole-chunk cell can
/// decide the ENTIRE chunk without the 64 per-block calls, but only when the chunk verdict
/// PROVABLY implies the identical per-block outcome (CONSERVATIVE-NEVER-NARROW).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WholeChunkVerdict {
    /// The whole chunk is provably AIR: every block is empty, so the chunk is the empty
    /// [`TwoLayerChunk`] (no coarse ids, no microblocks).
    AllAir,
    /// The whole chunk is provably one coarse-solid material: every block is
    /// [`BlockClassification::CoarseSolid`] at `block_id` with the same `overlay`.
    AllCoarse { block_id: BlockId, overlay: bool },
    /// The chunk straddles the surface / is multi-leaf ambiguous / unboundable: fall back
    /// to the per-block classify (the always-safe path).
    PerBlock,
}

/// Classify a whole chunk from ONE composed interval at the chunk cell (ADR 0010 Decision 2).
///
/// **Why the chunk verdict is byte-identical to the per-block sweep.** The composed field
/// interval is *inclusion-monotone*: for a sub-block box `B ⊆ chunk` every operand's bound
/// over `B` nests inside its bound over the chunk (the Lipschitz-centre bound because
/// `dist(centre_chunk, centre_B) + circumradius(B) ≤ circumradius(chunk)` for nested
/// axis-aligned boxes; the sketch discrete bound because `B`'s footprint rectangle is a
/// SUBSET of the chunk's), and a sub-block's overlapping-leaf set is a SUBSET of the chunk's
/// (`B ⊆ chunk`). Therefore:
///
/// * **AIR** (`minimum > isolevel`) ⇒ every block's `minimum` is `≥` the chunk's ⇒ every
///   block is AIR ⇒ the empty chunk. (Even a block the per-block sweep called BOUNDARY would
///   resolve to zero voxels here, since the chunk is provably all-outside — still empty.)
/// * **COARSE-SOLID** — [`classify_chunk_block`] only yields it for a SINGLE single-material
///   leaf, so the resolution is provably uniform across the chunk. We ADDITIONALLY require
///   that leaf's grid AABB to CONTAIN the whole chunk, so every sub-block lies inside the
///   leaf's extent ⇒ that one leaf overlaps every block (no block collapses to AIR) ⇒ every
///   block's `maximum` is `≤` the chunk's ⇒ every block is `CoarseSolid(block_id)` with the
///   same `overlay`. (For `SketchSolid`/`SdfShape` a coarse chunk is always fully interior,
///   so the containment guard never rejects a legitimate coarse chunk; it only defends
///   against a hypothetical producer whose field is negative OUTSIDE its own AABB.)
/// * anything else ⇒ `PerBlock`.
pub(crate) fn classify_whole_chunk(
    leaves: &[&LeafProducer],
    chunk_abs_voxels: VoxelAabb,
    voxels_per_block: u32,
) -> WholeChunkVerdict {
    // ADR 0017 Decision 3 (issue #74): the chunk-verdict→per-block TRANSFER arguments
    // below were proven for the FLAT fold of the sibling-level slice. They carry over
    // verbatim exactly when the scoped fold over this chunk's overlapping leaves is
    // occupancy-equivalent to that flat fold, i.e. when every overlapping
    // subtractive-influence leaf sits at ROOT scope (its interval folds at the root,
    // as before) and every other overlapping leaf is purely additive (Union scope
    // closes are occupancy-associative — pre-composing a union body and unioning it
    // in equals unioning its leaves directly, so pure-Union scopes are transparent to
    // the fold; the provable-equivalence regression the pure-Union goldens pin). Any
    // chunk touched by a SCOPED boolean degrades to the always-exact per-block sweep
    // — conservative, never narrow; the per-block classifier still elides its
    // interiors block-by-block.
    if !chunk_fold_is_scope_transparent(leaves, chunk_abs_voxels, voxels_per_block) {
        return WholeChunkVerdict::PerBlock;
    }
    match classify_chunk_block(leaves, chunk_abs_voxels, voxels_per_block) {
        BlockClassification::Air => {
            // ADR 0017: a chunk-level AIR verdict can lean on a SUBTRACT operand (a
            // cutter provably carving the whole chunk pushes the fold's minimum
            // above the isolevel). That proof only transfers to every sub-block if
            // the cutter stays IN each sub-block's own fold — i.e. its grid AABB
            // overlaps every sub-block — because a sub-block the cutter's box misses
            // DROPS the operand (mirroring the per-block overlap filter) and would
            // re-decide from the additive leaves alone. Sound in practice (a
            // producer's field is only deeply-inside within its own AABB), but
            // conservatively require every overlapping subtractive leaf's box to
            // CONTAIN the chunk — the same defensive containment posture as the
            // AllCoarse guard below; anything less degrades to the always-exact
            // per-block sweep. A chunk with NO overlapping purely-additive leaf is
            // trivially all-air regardless (booleans only remove; every sub-block
            // sees no additive leaf either).
            let any_additive_overlaps = leaves.iter().any(|leaf| {
                leaf_is_purely_additive(leaf)
                    && leaf_world_box(leaf, voxels_per_block).intersects(&chunk_abs_voxels)
            });
            if !any_additive_overlaps
                || subtractive_leaves_contain_chunk(leaves, chunk_abs_voxels, voxels_per_block)
            {
                WholeChunkVerdict::AllAir
            } else {
                WholeChunkVerdict::PerBlock
            }
        }
        BlockClassification::Boundary => WholeChunkVerdict::PerBlock,
        BlockClassification::CoarseSolid(block_id) => {
            // classify_chunk_block returns CoarseSolid ONLY for a single single-material
            // PURELY ADDITIVE leaf; recover it (the sole overlapping additive leaf) to
            // read its overlay AND to prove its grid AABB encloses the whole chunk.
            // Overlapping root-scope SUBTRACT leaves need no extra guard here: the chunk
            // fold PROVED they carve nothing in the chunk (their negated interval stayed
            // at-or-below the isolevel), a sub-block's operand intervals nest inside the
            // chunk's (the inclusion monotonicity above extends to the subtract role —
            // `max` preserves interval inclusion), and a sub-block that DROPS a cutter
            // from its fold is only MORE solid — so every sub-block re-proves
            // CoarseSolid at the same material. (Scoped booleans were already routed to
            // the per-block sweep by the transparency guard above.)
            match single_overlapping_leaf(leaves, chunk_abs_voxels, voxels_per_block) {
                Some(leaf)
                    if leaf_world_box(leaf, voxels_per_block).contains_box(&chunk_abs_voxels) =>
                {
                    WholeChunkVerdict::AllCoarse { block_id, overlay: leaf.grid_overlay }
                }
                _ => WholeChunkVerdict::PerBlock,
            }
        }
    }
}

/// Whether the scoped fold over the leaves overlapping `chunk_abs_voxels` is provably
/// occupancy-equivalent to the FLAT sibling-level fold, so the whole-chunk fast-path
/// transfer arguments (proven for the flat fold) apply verbatim: every overlapping leaf
/// is either purely additive (pure-Union scopes are transparent — union is associative,
/// so pre-composing changes nothing about occupancy) or sits at ROOT scope (its boolean
/// folds at the root exactly as in the flat fold). A chunk overlapped by any SCOPED
/// boolean — a Subtract inside a Group, or any leaf under a Subtract-folding scope —
/// fails this and takes the always-exact per-block sweep instead (see
/// [`classify_whole_chunk`]).
fn chunk_fold_is_scope_transparent(
    leaves: &[&LeafProducer],
    chunk_abs_voxels: VoxelAabb,
    voxels_per_block: u32,
) -> bool {
    leaves
        .iter()
        .filter(|leaf| leaf_world_box(leaf, voxels_per_block).intersects(&chunk_abs_voxels))
        .all(|leaf| leaf_is_purely_additive(leaf) || leaf.scope_path.is_empty())
}

/// Whether every SUBTRACTIVE-influence leaf (any leaf that is not purely additive) whose
/// grid AABB overlaps `chunk_abs_voxels` also CONTAINS it — the defensive containment
/// guard the whole-chunk AIR fast path requires under ADR 0017 (see
/// [`classify_whole_chunk`]): only a cutter present in EVERY sub-block's fold transfers a
/// chunk-level carved-to-air proof to each sub-block. Vacuously true with no overlapping
/// cutters (the pure-Union case, where AIR was proven by the additive intervals alone and
/// is inclusion-monotone as before).
fn subtractive_leaves_contain_chunk(
    leaves: &[&LeafProducer],
    chunk_abs_voxels: VoxelAabb,
    voxels_per_block: u32,
) -> bool {
    leaves
        .iter()
        .filter(|leaf| !leaf_is_purely_additive(leaf))
        .map(|leaf| leaf_world_box(leaf, voxels_per_block))
        .all(|cutter_box| {
            !cutter_box.intersects(&chunk_abs_voxels)
                || cutter_box.contains_box(&chunk_abs_voxels)
        })
}


/// Resolve a boundary block per-voxel into a dense `density³` [`VoxelRegion`] (the
/// material at each occupied voxel), decompose it to cuboids, and compute its per-face
/// seam-solidity flags. `block_min_abs` is the block's low corner in absolute voxels.
///
/// Per-voxel resolution reuses each overlapping leaf's [`VoxelProducer::resolve_into`]
/// over the block window, composed by the SAME scoped ordered-fold semantics the dense
/// path uses (ADR 0017 / issue #74: within a scope, a Union leaf stamps later-wins on
/// overlap and a Subtract leaf CLEARS the cells its body covers; a closing scope's
/// composed body folds into its parent as a unit under the SCOPE's operation) — so the
/// materialised block is bit-identical to the dense store's voxels for that block.
pub(crate) fn resolve_boundary_block(
    leaves: &[&LeafProducer],
    block_min_abs: [i64; 3],
    density: u32,
    voxels_per_block: u32,
) -> MicroblockGeometry {
    let extent = [density, density, density];
    let mut region = VoxelRegion::new_empty(extent);
    let block_abs = VoxelAabb::new(
        block_min_abs,
        [
            block_min_abs[0] + density as i64,
            block_min_abs[1] + density as i64,
            block_min_abs[2] + density as i64,
        ],
    );

    // Only the leaves whose grid AABB overlaps this block can touch its cells; dropping
    // the rest cannot change the fold (a dropped Union adds nothing here, a dropped
    // Subtract carves nothing here, and a scope whose leaves are all dropped never
    // opens — see `scoped_leaf_steps`).
    let overlapping: Vec<&LeafProducer> = leaves
        .iter()
        .copied()
        .filter(|leaf| leaf_world_box(leaf, voxels_per_block).intersects(&block_abs))
        .collect();

    // Compose in DOCUMENT ORDER (the order `leaf_producers` yields them, which is
    // `for_each_leaf`'s walk order) as a stack-evaluated scoped fold: each open scope
    // composes into its own block-local scratch region; a closing scope folds the
    // composed body into its parent under the scope's operation — exactly the dense
    // scoped fold, restricted to this block (composition is cell-local, so the
    // restriction commutes).
    let mut scope_stack: Vec<VoxelRegion> = Vec::new();
    for step in scoped_leaf_steps(&overlapping) {
        match step {
            ScopedLeafStep::OpenScope => scope_stack.push(VoxelRegion::new_empty(extent)),
            ScopedLeafStep::CloseScope(operation) => {
                let closed = scope_stack.pop().expect("scoped_leaf_steps emits balanced markers");
                let parent = scope_stack.last_mut().unwrap_or(&mut region);
                fold_closed_scope_into_region(parent, operation, &closed, extent);
            }
            ScopedLeafStep::Leaf(leaf) => {
                let target = scope_stack.last_mut().unwrap_or(&mut region);
                compose_leaf_into_region(target, leaf, block_min_abs, density, voxels_per_block);
            }
        }
    }

    let cuboids = decompose_into_boxes(&region);
    let seam_solidity = compute_seam_solidity(&region);
    MicroblockGeometry {
        cuboids,
        seam_solidity,
    }
}

/// Resolve one leaf's cells inside the block window and compose them into `region` (the
/// innermost open scope's block-local accumulator) under the LEAF's own operation: a
/// Union leaf stamps its render key (later document-order write wins — a plain overwrite
/// reproduces the dense Union), a Subtract leaf CLEARS every cell its body covers
/// (occupancy-only, ADR 0017 Decision 1 — it never writes a render key, so the material
/// of every surviving cell is untouched).
fn compose_leaf_into_region(
    region: &mut VoxelRegion,
    leaf: &LeafProducer,
    block_min_abs: [i64; 3],
    density: u32,
    voxels_per_block: u32,
) {
    // Resolve JUST this block's window in the leaf's local voxel-index frame.
    let window_local = VoxelAabb::new(
        [
            block_min_abs[0] - leaf.world_offset_voxels[0],
            block_min_abs[1] - leaf.world_offset_voxels[1],
            block_min_abs[2] - leaf.world_offset_voxels[2],
        ],
        [
            block_min_abs[0] + density as i64 - leaf.world_offset_voxels[0],
            block_min_abs[1] + density as i64 - leaf.world_offset_voxels[1],
            block_min_abs[2] + density as i64 - leaf.world_offset_voxels[2],
        ],
    );
    let mut local = VoxelGrid::default();
    leaf.producer
        .resolve_into(&mut local, voxels_per_block, window_local);

    // Stamp each emitted voxel into the block-local region at its material (a Tool
    // overrides every voxel's id; a Part keeps its own per-voxel id). The voxel's
    // local index is in the LEAF's frame, so shift back to block-local by adding the
    // leaf offset and subtracting the block's absolute low corner.
    for voxel in &local.occupied {
        let block_local = [
            voxel.local_index[0] as i64 + leaf.world_offset_voxels[0] - block_min_abs[0],
            voxel.local_index[1] as i64 + leaf.world_offset_voxels[1] - block_min_abs[1],
            voxel.local_index[2] as i64 + leaf.world_offset_voxels[2] - block_min_abs[2],
        ];
        if block_local.iter().any(|&c| c < 0 || c >= density as i64) {
            continue; // Outside this block (the window clamps, but guard anyway).
        }
        // ADR 0017: a Subtract leaf is an occupancy-only mask — every cell its
        // body covers is CLEARED from the result accumulated so far in ITS scope.
        if leaf.operation == CombineOp::Subtract {
            region.set(
                block_local[0] as u32,
                block_local[1] as u32,
                block_local[2] as u32,
                None,
            );
            continue;
        }
        let block_id = match leaf.material {
            Some(id) => id.0,
            None => voxel.block_id.0,
        };
        // Stamp the cuboid mesher's RENDER-CELL key (ADR 0003 §3c): the clean
        // categorical id in the low bits, this leaf's on-face-grid overlay in the
        // dedicated bit. So `decompose_into_boxes` splits a box across differing
        // overlay states exactly like the dense mesher, and the E3 mesher reads the
        // box's clean id + overlay back out — without the render flag ever entering
        // the categorical cell (the E2 occupancy expansion masks the bit off).
        let render_key = CellKey::compose(block_id, leaf.grid_overlay).raw();
        // Later document-order leaf wins on overlap: a plain overwrite reproduces
        // the dense Union (the walk visits in document order, last write persists).
        region.set(
            block_local[0] as u32,
            block_local[1] as u32,
            block_local[2] as u32,
            Some(render_key),
        );
    }
}

/// Fold one CLOSED scope's composed block-local body into its parent accumulator under
/// the scope's own operation (ADR 0017 Decision 3) — the [`VoxelRegion`] mirror of the
/// dense path's `fold_closed_scope_into`:
///
/// * `Union` — every occupied cell of the body OVERWRITES the parent's cell (the body
///   closed at the walk position AFTER everything already in the parent, so the
///   overwrite is exactly the later-wins rule of the flat fold).
/// * `Subtract` — every occupied cell of the body CLEARS the parent's cell
///   (occupancy-only; the body's render keys never enter the parent).
fn fold_closed_scope_into_region(
    parent: &mut VoxelRegion,
    operation: CombineOp,
    closed: &VoxelRegion,
    extent: [u32; 3],
) {
    for z in 0..extent[2] {
        for y in 0..extent[1] {
            for x in 0..extent[0] {
                let Some(render_key) = closed.cell_at(x, y, z) else {
                    continue;
                };
                match operation {
                    CombineOp::Union => parent.set(x, y, z, Some(render_key)),
                    CombineOp::Subtract => parent.set(x, y, z, None),
                }
            }
        }
    }
}

/// Compute the per-face seam-solidity flags for a resolved boundary block: a face is solid
/// iff EVERY voxel cell on that face of the `density³` region is occupied.
pub(crate) fn compute_seam_solidity(region: &VoxelRegion) -> SeamSolidity {
    let [width, height, depth] = region.extent;
    let mut solid = [[true; 2]; 3];
    // Degenerate (zero-extent) region: no face can be solid.
    if width == 0 || height == 0 || depth == 0 {
        return SeamSolidity {
            solid: [[false; 2]; 3],
        };
    }

    // X faces (axis 0): low x == 0, high x == width - 1.
    for &(side, x) in &[(0usize, 0u32), (1usize, width - 1)] {
        let mut face_solid = true;
        'scan: for z in 0..depth {
            for y in 0..height {
                if region.cell_at(x, y, z).is_none() {
                    face_solid = false;
                    break 'scan;
                }
            }
        }
        solid[0][side] = face_solid;
    }
    // Y faces (axis 1): low y == 0, high y == height - 1.
    for &(side, y) in &[(0usize, 0u32), (1usize, height - 1)] {
        let mut face_solid = true;
        'scan: for z in 0..depth {
            for x in 0..width {
                if region.cell_at(x, y, z).is_none() {
                    face_solid = false;
                    break 'scan;
                }
            }
        }
        solid[1][side] = face_solid;
    }
    // Z faces (axis 2): low z == 0, high z == depth - 1.
    for &(side, z) in &[(0usize, 0u32), (1usize, depth - 1)] {
        let mut face_solid = true;
        'scan: for y in 0..height {
            for x in 0..width {
                if region.cell_at(x, y, z).is_none() {
                    face_solid = false;
                    break 'scan;
                }
            }
        }
        solid[2][side] = face_solid;
    }

    SeamSolidity { solid }
}

