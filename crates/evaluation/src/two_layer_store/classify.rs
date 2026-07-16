//! The interval-bound block classifier (air / coarse-solid / boundary) + boundary-block per-voxel resolve + seam-solidity computation.


use substrate::solids::{CellClassification, CellCombineOp, CellContribution};

use voxel_core::core_geom::{BlockId, CellKey};
use crate::cuboid::{decompose_into_boxes, VoxelRegion};
use document::scene::{CombineOp, LeafProducer};
use voxel_core::spatial_index::VoxelAabb;
use voxel_core::voxel::{VoxelGrid, SURFACE_ISOLEVEL};
use document::voxel::FieldClassification;

#[allow(unused_imports)]
use super::*;

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
/// Returns:
/// * [`BlockClassification::Air`] iff the ordered fold provably leaves the block empty
///   (every overlapping Union leaf misses it, or an overlapping Subtract leaf provably
///   carves it whole; no leaf unboundable) — the conservative interval guarantees brute
///   force finds zero voxels.
/// * [`BlockClassification::CoarseSolid`] iff a single Union leaf provably fills the
///   WHOLE block solid AND the fold (which carries any Subtract intervals, ADR 0017)
///   still proves solidity — every voxel occupied, one material.
/// * [`BlockClassification::Boundary`] otherwise (straddling, multi-Union overlap that
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
    // nothing). A leaf's absolute span is `[off, off + grid)` (corner-anchored).
    let mut overlapping: Vec<&LeafProducer> = Vec::new();
    for &leaf in leaves {
        let grid_dimensions = leaf.producer.full_dimensions(voxels_per_block);
        let leaf_box = VoxelAabb::new(
            leaf.world_offset_voxels,
            [
                leaf.world_offset_voxels[0] + grid_dimensions[0] as i64,
                leaf.world_offset_voxels[1] + grid_dimensions[1] as i64,
                leaf.world_offset_voxels[2] + grid_dimensions[2] as i64,
            ],
        );
        if leaf_box.intersects(&block_abs_voxels) {
            overlapping.push(leaf);
        }
    }

    // ADR 0017: composition is an ordered document-order fold. Within THIS block,
    // the accumulated result before a leaf is the fold of the earlier OVERLAPPING
    // leaves (a non-overlapping leaf contributes nothing here: a dropped Union adds
    // no occupancy, a dropped Subtract carves none). A Subtract leaf that precedes
    // the first overlapping Union leaf therefore carves from NOTHING — drop it so
    // the substrate fold (which seeds its running interval with the FIRST
    // contribution regardless of role) never seeds on a cutter's body.
    let leading_subtract_count = overlapping
        .iter()
        .take_while(|leaf| leaf.operation == CombineOp::Subtract)
        .count();
    let contributions: &[&LeafProducer] = &overlapping[leading_subtract_count..];

    if contributions.is_empty() {
        // No Union leaf touches the block (only cutters, or nothing at all) ⇒
        // provably empty: a Subtract can only remove occupancy, never add it.
        return BlockClassification::Air;
    }

    // Fold the per-leaf conservative field intervals through the substrate
    // black/white/grey classifier (`CellClassification`) under each leaf's CSG role
    // (ADR 0017: Union = min-of-fields, Subtract = the conservative difference
    // `max(running, −operand)` — Duff 1992 interval arithmetic, so a Subtract
    // operand can only degrade a verdict toward Boundary/Air-it-can-prove, never
    // claim solidity). `None` iff any leaf is unboundable ⇒ BOUNDARY. Leaf
    // iteration + the local-frame map stay HERE (domain).
    let verdict = CellClassification::classify(
        contributions.iter().map(|leaf| {
            // Map the absolute block box into THIS leaf's local voxel-index frame `[0, full)`
            // by subtracting its world offset — the exact frame `cell_field_interval` expects
            // (ADR 0008: the frame is carried, never re-derived).
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
            CellContribution {
                field_interval: leaf
                    .producer
                    .cell_field_interval(cell_local, voxels_per_block),
                combine: match leaf.operation {
                    CombineOp::Union => CellCombineOp::Union,
                    CombineOp::Subtract => CellCombineOp::Subtract,
                },
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
            // A composed-solid verdict is only SAFELY coarse when EXACTLY ONE UNION
            // leaf overlaps the block: with two Union leaves the per-voxel MATERIAL
            // is "later wins on overlap", which the composed interval cannot resolve
            // (it proves geometric solidity, not which id each voxel takes). Under
            // ADR 0017 a Subtract leaf never stamps material, so overlapping cutters
            // do NOT disqualify the elision — the kernel's fold already carried
            // their subtract intervals, and a CoarseSolid verdict PROVES the cutters
            // carve nothing here, so every voxel keeps the single Union leaf's
            // material. Any other mix (multiple Union leaves; a single Union leaf
            // with no single-material id, i.e. a Part's per-voxel materials) is
            // forced BOUNDARY so the per-voxel pass decides — still exact, just
            // unelided. When in doubt, Boundary: it is always exact.
            let mut union_leaves = contributions
                .iter()
                .filter(|leaf| leaf.operation == CombineOp::Union);
            match (union_leaves.next(), union_leaves.next()) {
                // Single single-material Union leaf provably filling the block ⇒ coarse.
                (Some(leaf), None) => match leaf.material {
                    Some(block_id) => BlockClassification::CoarseSolid(block_id),
                    None => BlockClassification::Boundary,
                },
                // Multiple Union leaves ⇒ per-voxel later-wins material ⇒ boundary.
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

/// The FIRST **Union** leaf whose grid AABB overlaps `block_abs_voxels`, or `None` if none
/// does. The overlap test mirrors [`classify_chunk_block`]'s exactly, so the same leaf is
/// found. A coarse-solid block is owned by exactly one UNION leaf (the classifier forces
/// any multi-Union overlap to boundary, and a Subtract leaf never stamps — ADR 0017), so
/// for a coarse verdict this first Union hit IS the single owning leaf.
pub(crate) fn single_overlapping_leaf<'a>(
    leaves: &[&'a LeafProducer],
    block_abs_voxels: VoxelAabb,
    voxels_per_block: u32,
) -> Option<&'a LeafProducer> {
    leaves
        .iter()
        .copied()
        .filter(|leaf| leaf.operation == CombineOp::Union)
        .find(|leaf| leaf_world_box(leaf, voxels_per_block).intersects(&block_abs_voxels))
}

/// The on-face-grid overlay (ADR 0003 §3c) of the SINGLE Union leaf overlapping
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
    match classify_chunk_block(leaves, chunk_abs_voxels, voxels_per_block) {
        BlockClassification::Air => {
            // ADR 0017: a chunk-level AIR verdict can now lean on a SUBTRACT
            // operand (a cutter provably carving the whole chunk pushes the fold's
            // minimum above the isolevel). That proof only transfers to every
            // sub-block if the cutter stays IN each sub-block's own fold — i.e. its
            // grid AABB overlaps every sub-block — because a sub-block the cutter's
            // box misses DROPS the operand (mirroring the per-block overlap filter)
            // and would re-decide from the Union leaves alone. Sound in practice
            // (a producer's field is only deeply-inside within its own AABB), but
            // conservatively require every overlapping Subtract leaf's box to
            // CONTAIN the chunk — the same defensive containment posture as the
            // AllCoarse guard below; anything less degrades to the always-exact
            // per-block sweep. A chunk with NO overlapping Union leaf is trivially
            // all-air regardless (Subtract only removes; every sub-block sees no
            // Union leaf either).
            let any_union_overlaps = leaves.iter().any(|leaf| {
                leaf.operation == CombineOp::Union
                    && leaf_world_box(leaf, voxels_per_block).intersects(&chunk_abs_voxels)
            });
            if !any_union_overlaps || subtract_leaves_contain_chunk(leaves, chunk_abs_voxels, voxels_per_block)
            {
                WholeChunkVerdict::AllAir
            } else {
                WholeChunkVerdict::PerBlock
            }
        }
        BlockClassification::Boundary => WholeChunkVerdict::PerBlock,
        BlockClassification::CoarseSolid(block_id) => {
            // classify_chunk_block returns CoarseSolid ONLY for a single single-material
            // UNION leaf; recover it (the sole overlapping Union leaf) to read its overlay
            // AND to prove its grid AABB encloses the whole chunk. Overlapping SUBTRACT
            // leaves need no extra guard here: the chunk fold PROVED they carve nothing in
            // the chunk (their negated interval stayed at-or-below the isolevel), a
            // sub-block's operand intervals nest inside the chunk's (the inclusion
            // monotonicity above extends to the subtract role — `max` preserves interval
            // inclusion), and a sub-block that DROPS a cutter from its fold is only MORE
            // solid — so every sub-block re-proves CoarseSolid at the same material.
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

/// Whether every SUBTRACT leaf whose grid AABB overlaps `chunk_abs_voxels` also CONTAINS
/// it — the defensive containment guard the whole-chunk AIR fast path requires under ADR
/// 0017 (see [`classify_whole_chunk`]): only a cutter present in EVERY sub-block's fold
/// transfers a chunk-level carved-to-air proof to each sub-block. Vacuously true with no
/// overlapping cutters (the pure-Union case, where AIR was proven by the Union intervals
/// alone and is inclusion-monotone as before).
fn subtract_leaves_contain_chunk(
    leaves: &[&LeafProducer],
    chunk_abs_voxels: VoxelAabb,
    voxels_per_block: u32,
) -> bool {
    leaves
        .iter()
        .filter(|leaf| leaf.operation == CombineOp::Subtract)
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
/// over the block window, composed by the SAME ordered-fold semantics the dense path
/// uses (document order — a Union leaf stamps later-wins on overlap, a Subtract leaf
/// CLEARS the cells its body covers, ADR 0017) — so the materialised block is
/// bit-identical to the dense store's voxels for that block.
pub(crate) fn resolve_boundary_block(
    leaves: &[&LeafProducer],
    block_min_abs: [i64; 3],
    density: u32,
    voxels_per_block: u32,
) -> MicroblockGeometry {
    let extent = [density, density, density];
    let mut region = VoxelRegion::new_empty(extent);

    // Compose leaves in DOCUMENT ORDER (the order `leaf_producers` yields them, which is
    // `for_each_leaf`'s walk order): a Union leaf stamps (later wins on overlap), a
    // Subtract leaf clears — exactly the dense ordered fold (ADR 0017).
    for &leaf in leaves {
        let grid_dimensions = leaf.producer.full_dimensions(voxels_per_block);
        let leaf_box = VoxelAabb::new(
            leaf.world_offset_voxels,
            [
                leaf.world_offset_voxels[0] + grid_dimensions[0] as i64,
                leaf.world_offset_voxels[1] + grid_dimensions[1] as i64,
                leaf.world_offset_voxels[2] + grid_dimensions[2] as i64,
            ],
        );
        let block_abs = VoxelAabb::new(
            block_min_abs,
            [
                block_min_abs[0] + density as i64,
                block_min_abs[1] + density as i64,
                block_min_abs[2] + density as i64,
            ],
        );
        if !leaf_box.intersects(&block_abs) {
            continue;
        }

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
            // body covers is CLEARED from the accumulated result, in the same
            // document-order walk. It never writes a render key, so the material of
            // every surviving cell is untouched.
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

    let cuboids = decompose_into_boxes(&region);
    let seam_solidity = compute_seam_solidity(&region);
    MicroblockGeometry {
        cuboids,
        seam_solidity,
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

