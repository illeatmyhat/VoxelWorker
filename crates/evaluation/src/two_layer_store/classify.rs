//! The interval-bound block classifier (air / coarse-solid / boundary) + boundary-block per-voxel resolve + seam-solidity computation.


use substrate::solids::{
    CellCombineOp, CellContribution, ScopedCellClassification, ScopedCellEvent,
};
use glam::{Quat, Vec3};
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
/// identities). An INTERSECT-closing scope must NOT disappear that way (`A ∩ ∅ = ∅`
/// annihilates the parent, so the close must happen even where no scope leaf emits);
/// the filters above guarantee it cannot, because every leaf inside such a scope is
/// Intersect-influence ([`LeafProducer::masks_beyond_bounds`]) and is never dropped.
///
/// FIXTURE definitions (ADR 0017 Decision 4, issue #77) need no machinery here: a
/// fixture's expansion contributes NO scope frame, so its leaves arrive carrying the
/// HOSTING scope's path plus their own operations — to this fold (and to every
/// conservative fast path reasoning over `operation` + `scope_path`) a fixture's
/// Subtract child is indistinguishable from a scoped-or-root cutter authored in place.
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
/// boolean anywhere on the path makes the leaf's root-level influence purely
/// removing — e.g. a Union leaf inside a Group placed under Subtract only ever CARVES
/// the parent (its body enters the group's composed occupancy, which is then removed
/// from the parent), and a Union leaf inside a Group placed under Intersect (#75) only
/// ever PRESERVES parent cells its scope's body covers (it never creates root
/// occupancy of its own). Purely additive leaves are also the only leaves that ever
/// STAMP material at the root (booleans never stamp — Decision 1).
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
        CombineOp::Intersect => CellCombineOp::Intersect,
        // Unreachable: a scope containing an Emboss node is pre-composed into a single
        // CompositeProducer leaf before classification, so the kernel never sees this arm.
        // The role is irrelevant anyway — the caller forces such a leaf's interval to `None`,
        // which makes the whole cell BOUNDARY (resolve per-voxel), the always-safe fallback.
        CombineOp::Emboss { .. } => CellCombineOp::Union,
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
    //
    // ADR 0017 (#75): an Intersect-INFLUENCE leaf (its own operation is Intersect, or
    // any enclosing scope closes under Intersect) is NEVER dropped: its mask kills
    // cells OUTSIDE its body, so a block its box misses is exactly where it still
    // applies (dropping it would err toward SOLID — never conservative). Keeping it
    // also guarantees every Intersect-closing scope opens in the fold, so the ∅-body
    // close annihilates the parent interval (the kernel's `A ∩ ∅ = ∅`). Its interval
    // over a far block comes from the producer as usual: provably-air for the shipped
    // producers (an SDF's Lipschitz bound / the sketch's outside-extent arm), or
    // `None` ⇒ the always-exact per-voxel fallback.
    let overlapping: Vec<&LeafProducer> = leaves
        .iter()
        .copied()
        .filter(|leaf| {
            leaf_world_box(leaf, voxels_per_block).intersects(&block_abs_voxels)
                || leaf.masks_beyond_bounds()
        })
        .collect();

    // Occupancy at the root can only be CREATED by a purely additive leaf (ADR 0017:
    // booleans — Subtract and Intersect, at any scope depth — only ever remove). No
    // overlapping purely-additive leaf ⇒ provably empty.
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
                // Map the absolute block box into THIS leaf's producer-local voxel-index
                // frame `[0, full)` via the inverse affine (ADR 0027) — the exact frame
                // `cell_field_interval` expects (ADR 0008: the frame is carried, never
                // re-derived). Exact for the axis-aligned leaves; a conservative enclosing
                // box for a genuine rotation (the isometry keeps the interval bound sound).
                let cell_local =
                    abs_box_to_producer_local(leaf, block_abs_voxels, voxels_per_block);
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
            // not a Subtract leaf, not an Intersect mask (#75), nor a Union leaf
            // whose scope folds under a boolean (its body only ever removes /
            // preserves parent cells) — so overlapping boolean-influence leaves do
            // NOT disqualify the elision: the scoped fold already carried their
            // intervals, and a CoarseSolid verdict PROVES the cutters carve nothing
            // and the masks cover everything here, so every voxel keeps the
            // single additive leaf's material. Any other mix (multiple additive
            // leaves; a single additive leaf with no single-material id, i.e. a
            // VoxelBody's per-voxel materials) is forced BOUNDARY so the per-voxel pass
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

/// The ADR 0027 continuous placement affine — [`substrate::spatial::LeafPlacement`], the world↔
/// producer-local map every classify / resolve / broadphase function in this module routes
/// through. It generalizes the ADR 0026 discrete
/// [`LatticeOrientation`](substrate::spatial::LatticeOrientation) permutation to an arbitrary
/// `Quat` rotation plus a float offset; for an axis-aligned rotation it reproduces the pre-0027
/// lattice `turn_point_in_box + offset` EXACTLY (ADR 0027 §4 — a rotation is an isometry, so
/// per-voxel occupancy stays exact). Hoisted to substrate so the dense reference oracle
/// (`document`) builds the identical map instead of a divergent translation-only copy. Construct
/// it for a leaf with [`leaf_affine`].
pub(crate) type LeafAffine = substrate::spatial::LeafPlacement;

/// Build the placement [`LeafAffine`] for `leaf` at the document's `voxels_per_block` — the
/// evaluation-layer adapter that reads the leaf's producer dimensions, continuous rotation and
/// integer-plus-float world offset and hands them to the substrate constructor. The composed-field
/// point-eval ([`super::composed_field_at`], ADR 0027 §5) and the dense reference oracle build the
/// SAME [`LeafAffine`] from the same components, so no path re-derives the map.
pub(crate) fn leaf_affine(leaf: &LeafProducer, voxels_per_block: u32) -> LeafAffine {
    let full_dimensions = leaf.producer.full_dimensions(voxels_per_block);
    let full = Vec3::new(
        full_dimensions[0] as f32,
        full_dimensions[1] as f32,
        full_dimensions[2] as f32,
    );
    // ADR 0027 §1 wandering origin: keep the integer `world_offset_voxels` and the fractional
    // `offset_local_voxels` SPLIT into substrate, so a far-out leaf's translation stays exact in
    // i64 rather than collapsing to a precision-losing f32 sum here (the ADR 0008 (c)-violation).
    LeafAffine::from_origin_and_local(
        leaf.rotation,
        full,
        leaf.world_offset_voxels,
        leaf.offset_local_voxels,
    )
}

/// The world offset (in ABSOLUTE voxels) that seats a producer of local dimensions `full`,
/// rotated by `rotation`, so its local CENTRE `full/2` lands at world `target_centre` under the
/// SAME corner-anchored [`LeafAffine`] the classifier folds through (ADR 0027 §5 placement).
///
/// It is the inverse of `leaf_affine(..).world_of(full/2) == target_centre`: placement picks a
/// rotation and a surface contact, and this returns the `world_offset` (⇒ a node's `offset_voxels`
/// `+` `offset_local_voxels`) that makes the classifier resolve the producer with its centre
/// exactly there. Delegates to [`substrate::spatial::seat_centre_at`], which shares the
/// `min_rotated_corner` corner anchor with [`LeafAffine`] — ONE definition, so a dropped node
/// resolves where it previewed (no "two impls of one predicate" drift).
pub fn seat_centre_at(rotation: Quat, full: Vec3, target_centre: Vec3) -> Vec3 {
    substrate::spatial::seat_centre_at(rotation, full, target_centre)
}

/// The leaf's grid AABB in the SCENE's absolute voxel frame — the integer enclosing box of the
/// [`LeafAffine`] applied to the 8 corners of the producer's local `[0, full]` box (ADR 0027).
/// The single box construction shared by the classify / overlap / whole-chunk paths (they must
/// all test the SAME leaf extent).
///
/// For an AXIS-ALIGNED rotation this equals the pre-0027 `[off, off + turn_extent(full))`
/// EXACTLY: the affine sends the corners to integer world positions, so rounding to nearest
/// recovers them bit-for-bit (the `an_oriented_leaf_occupies_the_turned_cells_of_the_upright_one`
/// golden pins it). For a genuine rotation the corners land off-lattice, so the min is FLOORED and
/// the max CEILED to conservatively enclose the rotated box (SOUND: the true occupied set ⊆ this
/// AABB, ADR 0027 §4).
pub(crate) fn leaf_world_box(leaf: &LeafProducer, voxels_per_block: u32) -> VoxelAabb {
    let (min, max) = leaf_affine(leaf, voxels_per_block).world_aabb();
    VoxelAabb::new(min, max)
}

/// Map an absolute voxel box into the leaf's **producer-local** `[0, full)` frame — the integer
/// enclosing box of the inverse [`LeafAffine::local_of`] applied to the 8 corners of `abs` (ADR
/// 0027). This is the frame `cell_field_interval` / `resolve_into` expect (the producer never
/// learns the leaf is turned).
///
/// For an AXIS-ALIGNED rotation this equals the pre-0027 `unturn_box` EXACTLY (the corners map to
/// integer local coordinates, recovered by rounding to nearest). For a genuine rotation the mapped
/// box is a ROTATED box in the local frame; flooring the min and ceiling the max conservatively
/// encloses it — SOUND because the true region ⊆ this AABB, so `cell_field_interval` over it still
/// brackets the field (the isometry keeps the cell radius invariant, ADR 0027 §4). The box may
/// fall partly outside `[0, full)` (a block straddling the leaf edge); the producer bounds/clamps
/// it exactly as before.
pub(crate) fn abs_box_to_producer_local(
    leaf: &LeafProducer,
    abs: VoxelAabb,
    voxels_per_block: u32,
) -> VoxelAabb {
    let (min, max) = leaf_affine(leaf, voxels_per_block).local_aabb(abs.min, abs.max);
    VoxelAabb::new(min, max)
}

/// Map a **producer-local** voxel index to its absolute voxel index (ADR 0027): the absolute
/// cell the local cell's CENTRE lands in, `world_of(index + 0.5).floor()`. The inverse of
/// [`abs_box_to_producer_local`] for a single cell — the forward-emit direction.
///
/// **Why `+0.5`/`floor` reproduces the lattice byte-for-byte.** For a positive-sign lattice axis
/// the affine's `world_of` already equals `local + offset`, and `floor(local + 0.5 + offset) =
/// local + offset`. For a NEGATED axis the corner-anchored affine gives `full − local + offset`
/// while the lattice `turn_point_in_box` gives `full − 1 − local + offset`; the centre sample makes
/// the affine value `full − local − 0.5 + offset`, whose `floor` is `full − 1 − local + offset` —
/// exactly the lattice. So every axis-aligned turn emits into the identical absolute cells the ADR
/// 0026 permutation did, and the half-unit margin absorbs the `Quat` round-off.
pub(crate) fn producer_local_voxel_to_abs(
    leaf: &LeafProducer,
    local_index: [i32; 3],
    voxels_per_block: u32,
) -> [i64; 3] {
    // ADR 0027 §1 wandering origin: `world_cell_of_local_centre` re-adds the integer origin in i64
    // after flooring the small origin-relative image, so a far-out leaf's cells stay exact.
    leaf_affine(leaf, voxels_per_block).world_cell_of_local_centre(local_index)
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
    // ADR 0017 (#75): the whole-chunk TRANSFER arguments below were proven WITHOUT
    // Intersect masks in the fold, and a mask breaks their shared premise — that a
    // sub-block dropping an operand from its fold can only become MORE solid (true for
    // a dropped cutter, false for a mask, which is precisely never dropped). Rather
    // than re-prove the transfer under masks, any chunk evaluated against an
    // Intersect-influence leaf degrades to the always-exact per-block sweep (which
    // still elides coarse interiors block-by-block, mask intervals included) — when in
    // doubt, degrade; never err toward air/solid (Decision 6). The one still-provable
    // escape is kept: a chunk NO purely-additive leaf overlaps is trivially all-air
    // regardless of masks (booleans only remove — the same argument as the per-block
    // early return), so the vast empty space around a masked scene stays one call.
    // Extending the fast path to root-scope masks (the transfer arguments do appear to
    // survive: mask operands are never dropped and `max` preserves interval inclusion)
    // is a recorded follow-up, deliberately not this slice.
    if leaves.iter().any(|leaf| leaf.masks_beyond_bounds()) {
        let any_additive_overlaps = leaves.iter().any(|leaf| {
            leaf_is_purely_additive(leaf)
                && leaf_world_box(leaf, voxels_per_block).intersects(&chunk_abs_voxels)
        });
        return if any_additive_overlaps {
            WholeChunkVerdict::PerBlock
        } else {
            WholeChunkVerdict::AllAir
        };
    }
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
    // opens — see `scoped_leaf_steps`) — EXCEPT an Intersect-influence leaf (ADR 0017
    // #75), which is never dropped: its mask kills cells outside its body (a box miss
    // means its window resolves EMPTY here and everything accumulated in its scope
    // dies), and keeping it guarantees every Intersect-closing scope opens.
    let overlapping: Vec<&LeafProducer> = leaves
        .iter()
        .copied()
        .filter(|leaf| {
            leaf_world_box(leaf, voxels_per_block).intersects(&block_abs)
                || leaf.masks_beyond_bounds()
        })
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
/// reproduces the dense Union), a Subtract leaf CLEARS every cell its body covers, and an
/// Intersect leaf (#75) KEEPS ONLY the cells its body covers — clearing every other cell
/// of the accumulator, including the whole block when its body misses it (`A ∩ ∅ = ∅`).
/// Both booleans are occupancy-only (ADR 0017 Decision 1 — they never write a render key,
/// so the material of every surviving cell is untouched).
/// Clear every accumulator cell in the block-local `[0, density)³` extent that the
/// mask's `body_covers` set MISSES — the Intersect clearing sweep shared by
/// [`compose_leaf_into_region`] (forward emit) and [`gather_rotated_leaf_into_region`]
/// (inverse gather). Surviving cells keep the render key they already carry (the mask
/// never stamps); a body covering nothing clears the whole block (`A ∩ ∅ = ∅`).
fn clear_uncovered_cells(
    region: &mut VoxelRegion,
    density: u32,
    body_covers: &std::collections::HashSet<[u32; 3]>,
) {
    for z in 0..density {
        for y in 0..density {
            for x in 0..density {
                if region.cell_at(x, y, z).is_some() && !body_covers.contains(&[x, y, z]) {
                    region.set(x, y, z, None);
                }
            }
        }
    }
}

pub(crate) fn compose_leaf_into_region(
    region: &mut VoxelRegion,
    leaf: &LeafProducer,
    block_min_abs: [i64; 3],
    density: u32,
    voxels_per_block: u32,
) {
    // ADR 0027: a FIELD producer whose producer-local lattice is OUT OF PHASE with the abs
    // lattice cannot be emitted by the forward affine (its `[0, full)` cells no longer land
    // one-per-abs-cell), so resample it by INVERSE GATHER instead. Two things throw it out of
    // phase: a genuine rotation, OR a fractional `offset_local_voxels` (a sub-voxel seat under
    // ADR 0027 continuous placement). A fractional TRANSLATION dephases the lattice exactly like
    // a rotation — `abs_box_to_producer_local`'s round-to-nearest then disagrees with the
    // floor-based per-cell emit and drops the boundary-voxel layer at every block seam (a
    // one-block-thick tube wall grids into fragments). Every whole-phase leaf (integer offset,
    // no rotation — most placements) and every fieldless producer (cloud / VoxelBody, which is
    // NEVER continuously rotated) takes the exact forward-emit path below — byte-identical to
    // ADR 0026.
    let in_phase = substrate::spatial::is_in_phase(leaf.rotation, leaf.offset_local_voxels);
    if !in_phase && leaf.producer.as_field().is_some() {
        gather_rotated_leaf_into_region(region, leaf, block_min_abs, density, voxels_per_block);
        return;
    }
    debug_assert!(
        in_phase,
        "an out-of-phase (rotated or sub-voxel-seated) fieldless producer cannot reach the \
         forward-emit path (ADR 0027)"
    );

    // Resolve JUST this block's window in the leaf's producer-local voxel-index frame
    // (ADR 0027: the inverse affine `abs_box_to_producer_local`, exact for the axis-aligned
    // leaves that reach here).
    let block_abs = VoxelAabb::new(
        block_min_abs,
        [
            block_min_abs[0] + density as i64,
            block_min_abs[1] + density as i64,
            block_min_abs[2] + density as i64,
        ],
    );
    let window_local = abs_box_to_producer_local(leaf, block_abs, voxels_per_block);
    let mut local = VoxelGrid::default();
    leaf.producer
        .resolve_into(&mut local, voxels_per_block, window_local);

    // ADR 0017 (#75): an Intersect leaf keeps ONLY the accumulator cells its body also
    // covers in this block. Collect the body's block-local cells, then sweep the whole
    // block-local extent clearing every occupied cell the body misses — surviving cells
    // keep the render key they already carry (the mask never stamps). A body whose box
    // misses the block resolves an EMPTY window, so the sweep clears everything: the
    // block-local reading of `A ∩ ∅ = ∅`.
    if leaf.operation == CombineOp::Intersect {
        let mut body_covers: std::collections::HashSet<[u32; 3]> =
            std::collections::HashSet::with_capacity(local.occupied.len());
        for voxel in &local.occupied {
            // ADR 0027: the voxel's index is in the producer's local frame; map it to absolute
            // via the forward affine (exact for an axis-aligned leaf) before rebasing to block-local.
            let abs = producer_local_voxel_to_abs(leaf, voxel.local_index, voxels_per_block);
            let block_local: [i64; 3] = std::array::from_fn(|axis| abs[axis] - block_min_abs[axis]);
            if block_local.iter().any(|&c| c < 0 || c >= density as i64) {
                continue; // Outside this block (the window clamps, but guard anyway).
            }
            body_covers.insert(std::array::from_fn(|axis| block_local[axis] as u32));
        }
        clear_uncovered_cells(region, density, &body_covers);
        return;
    }

    // Stamp each emitted voxel into the block-local region at its material (a Tool
    // overrides every voxel's id; a VoxelBody keeps its own per-voxel id). The voxel's
    // local index is in the LEAF's frame, so shift back to block-local by adding the
    // leaf offset and subtracting the block's absolute low corner.
    for voxel in &local.occupied {
        // ADR 0027: map the producer-local index into absolute via the forward affine, then
        // rebase to block-local. Identity leaves reproduce the old `index + offset − block`.
        let abs = producer_local_voxel_to_abs(leaf, voxel.local_index, voxels_per_block);
        let block_local: [i64; 3] = std::array::from_fn(|axis| abs[axis] - block_min_abs[axis]);
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

/// The ADR 0027 **inverse-resample gather** for a genuinely-rotated field producer: for each
/// absolute voxel in the block window, inverse-map its CENTRE into the producer-local frame and
/// test the field, applying the SAME per-operation logic the forward emit does (Union stamps,
/// Subtract clears, Intersect keeps-only-covered). Only a FIELD producer (an SDF Tool, a sketch
/// solid, a composed Part) reaches here — a fieldless producer is never continuously rotated
/// (ADR 0027), so [`compose_leaf_into_region`] routes it to the forward-emit path.
///
/// Occupancy is exact per voxel (a rotation is an isometry, ADR 0027 §4): `local_if_covered`
/// samples the field the producer resolves, so the gathered set equals the forward-emitted turn
/// wherever the rotation IS axis-aligned (proven by the gather-vs-permutation oracle test) and
/// extends it smoothly to off-axis rotations.
pub(crate) fn gather_rotated_leaf_into_region(
    region: &mut VoxelRegion,
    leaf: &LeafProducer,
    block_min_abs: [i64; 3],
    density: u32,
    voxels_per_block: u32,
) {
    let affine = leaf_affine(leaf, voxels_per_block);
    let field = leaf
        .producer
        .as_field()
        .expect("the caller gathers only field producers (ADR 0027)");

    // The producer-local coordinate of a block-local voxel's CENTRE, returned only when the
    // field classifies that centre inside-or-on the surface. The `[f32; 3]` is handed to
    // `material_at` for a per-voxel-material producer.
    let local_if_covered = |x: u32, y: u32, z: u32| -> Option<[f32; 3]> {
        // ADR 0027 §1 wandering origin: rebase the absolute cell against the leaf origin in i64
        // before the inverse rotation, so a far-out block keeps full sub-voxel precision.
        let abs_cell =
            [block_min_abs[0] + x as i64, block_min_abs[1] + y as i64, block_min_abs[2] + z as i64];
        let local = affine.local_of_abs_cell_centre(abs_cell).voxels().to_array();
        (field.signed_distance(local, voxels_per_block) <= SURFACE_ISOLEVEL).then_some(local)
    };

    // ADR 0017 (#75): an Intersect leaf keeps ONLY the accumulator cells its body also covers.
    // Build the body-covers set from the gather, then reuse the exact clearing sweep the forward
    // path uses (surviving cells keep their render key; the mask never stamps). A body that
    // covers nothing clears the whole block — the block-local reading of `A ∩ ∅ = ∅`.
    if leaf.operation == CombineOp::Intersect {
        let mut body_covers: std::collections::HashSet<[u32; 3]> =
            std::collections::HashSet::new();
        for z in 0..density {
            for y in 0..density {
                for x in 0..density {
                    if local_if_covered(x, y, z).is_some() {
                        body_covers.insert([x, y, z]);
                    }
                }
            }
        }
        clear_uncovered_cells(region, density, &body_covers);
        return;
    }

    for z in 0..density {
        for y in 0..density {
            for x in 0..density {
                let Some(local) = local_if_covered(x, y, z) else {
                    continue;
                };
                // ADR 0017: a Subtract leaf is an occupancy-only mask — clear every covered cell.
                if leaf.operation == CombineOp::Subtract {
                    region.set(x, y, z, None);
                    continue;
                }
                // Union: stamp the render key. Material precedence mirrors the forward path — the
                // leaf's single-material override, else the producer's per-voxel material, else the
                // default id.
                let block_id = match leaf.material {
                    Some(id) => id.0,
                    None => leaf
                        .producer
                        .material_at(local, voxels_per_block)
                        .map(|id| id.0)
                        .unwrap_or(BlockId::DEFAULT.0),
                };
                let render_key = CellKey::compose(block_id, leaf.grid_overlay).raw();
                region.set(x, y, z, Some(render_key));
            }
        }
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
/// * `Intersect` (#75) — the parent KEEPS ONLY the cells the body also occupies: every
///   parent cell the body does NOT cover is cleared (occupancy-only; surviving cells
///   keep their parent render key). A body that composed to EMPTY clears the whole
///   parent — `A ∩ ∅ = ∅`, the ∅ identity of the substrate kernel.
fn fold_closed_scope_into_region(
    parent: &mut VoxelRegion,
    operation: CombineOp,
    closed: &VoxelRegion,
    extent: [u32; 3],
) {
    for z in 0..extent[2] {
        for y in 0..extent[1] {
            for x in 0..extent[0] {
                match (operation, closed.cell_at(x, y, z)) {
                    (CombineOp::Union, Some(render_key)) => {
                        parent.set(x, y, z, Some(render_key));
                    }
                    (CombineOp::Subtract, Some(_)) => parent.set(x, y, z, None),
                    (CombineOp::Intersect, None) => parent.set(x, y, z, None),
                    // Union/Subtract ignore cells the body left empty; Intersect
                    // keeps (does not touch) the parent cells the body covers.
                    (CombineOp::Union, None)
                    | (CombineOp::Subtract, None)
                    | (CombineOp::Intersect, Some(_)) => {}
                    // Unreachable: an Emboss scope is pre-composed (it needs the accumulated
                    // FIELD, which a region of resolved cells no longer carries).
                    (CombineOp::Emboss { .. }, _) => {}
                }
            }
        }
    }
}

/// Whether EVERY cell on the face of `region` where axis `axis` (0=x/1=y/2=z) equals
/// `fixed` is occupied. Sweeps the other two axes over their full extent, assembling
/// each `(x, y, z)` so `cell_at`'s argument order is preserved exactly — the shared
/// scan the three per-axis face checks used to open-code (the sweep ORDER is
/// immaterial: "all cells occupied" is order-independent, the early return just prunes).
fn face_is_solid(region: &VoxelRegion, axis: usize, fixed: u32) -> bool {
    let extent = region.extent;
    let swept_a = (axis + 1) % 3;
    let swept_b = (axis + 2) % 3;
    for u in 0..extent[swept_a] {
        for v in 0..extent[swept_b] {
            let mut coord = [0u32; 3];
            coord[axis] = fixed;
            coord[swept_a] = u;
            coord[swept_b] = v;
            if region.cell_at(coord[0], coord[1], coord[2]).is_none() {
                return false;
            }
        }
    }
    true
}

/// Compute the per-face seam-solidity flags for a resolved boundary block: a face is solid
/// iff EVERY voxel cell on that face of the `density³` region is occupied.
pub(crate) fn compute_seam_solidity(region: &VoxelRegion) -> SeamSolidity {
    let extent = region.extent;
    // Degenerate (zero-extent) region: no face can be solid.
    if extent.contains(&0) {
        return SeamSolidity {
            solid: [[false; 2]; 3],
        };
    }

    // Per axis: side 0 is the low face (fixed = 0), side 1 the high face (extent − 1).
    let mut solid = [[true; 2]; 3];
    for axis in 0..3 {
        solid[axis][0] = face_is_solid(region, axis, 0);
        solid[axis][1] = face_is_solid(region, axis, extent[axis] - 1);
    }

    SeamSolidity { solid }
}

