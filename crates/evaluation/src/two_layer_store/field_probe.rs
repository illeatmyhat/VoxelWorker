//! The composed-field **point-eval** (ADR 0027 §5): the scene's composed signed distance
//! at a single world point.
//!
//! The continuous-placement surface-raycast (a separate ADR 0027 §5 slice) marches the
//! scene's composed signed-distance field along the eye ray; it needs the field VALUE at a
//! world point, and this is that value. [`composed_field_at`] folds the SAME scoped ordered
//! CSG composition [`classify_chunk_block`](super::classify_chunk_block) folds — reusing
//! [`scoped_leaf_steps`](super::scoped_leaf_steps) for the step sequence and
//! [`LeafAffine`](super::LeafAffine) for the absolute→producer-local frame map — but folds
//! on FIELD VALUES instead of occupancy intervals. Sharing the classifier's step
//! reconstruction and affine is deliberate: a second copy of either would be a
//! "two impls of one predicate" drift trap (`[[measure-before-rejecting]]` sibling).
//!
//! ## Why the field agrees with the classifier
//!
//! At [`SURFACE_ISOLEVEL`](voxel_core::voxel::SURFACE_ISOLEVEL)` == 0`,
//! `composed_field_at(point) <= 0` equals "`point` is occupied" per the classifier, because
//! each CSG fold's sign predicate matches the occupancy predicate exactly (Duff 1992 — the
//! same algebra the substrate interval kernel folds):
//!
//! * **Union** = `min(accumulator, value)` — `min(a, b) <= 0  ⟺  a <= 0 ∨ b <= 0` (OR).
//! * **Subtract** = `max(accumulator, −value)` — `max(a, −b) <= 0  ⟺  a <= 0 ∧ b >= 0`
//!   (inside the base AND outside the cutter — AND-NOT).
//! * **Intersect** = `max(accumulator, value)` — `max(a, b) <= 0  ⟺  a <= 0 ∧ b <= 0` (AND).
//!
//! A scope opens a fresh accumulator seeded `+INFINITY` (the empty-body identity: a scope
//! that contributes no field is "nothing here", `min`-neutral for Union and `max(·, −∞→+∞)`
//! -harmless for the booleans, exactly the ∅ identities the classifier's kernel carries);
//! `CloseScope(operation)` folds the closed accumulator into its parent under the SCOPE's
//! own operation. The root accumulator also starts `+INFINITY`, and the result is the root
//! accumulator after every step — `+INFINITY` where nothing is present (an empty scene, or
//! one whose every leaf is fieldless).

use glam::Vec3;

use document::scene::{CombineOp, LeafProducer};

use super::{scoped_leaf_steps, LeafAffine, ScopedLeafStep};

/// Fold one `operand` field value into a scope `accumulator` under `operation` — the
/// pointwise CSG algebra (Duff 1992) whose sign matches the classifier's occupancy
/// predicate (see the module docs).
fn fold_operand(accumulator: f32, operand: f32, operation: CombineOp) -> f32 {
    match operation {
        CombineOp::Union => accumulator.min(operand),
        CombineOp::Subtract => accumulator.max(-operand),
        CombineOp::Intersect => accumulator.max(operand),
        // An Emboss scope is pre-composed into a single `Composed` leaf BEFORE
        // classification (ADR 0020 Decision 7 — the same absorption `classify_chunk_block`
        // relies on), so a raw `Emboss` step never reaches this fold over raw leaves. If
        // one ever did, Union is the safe fallback: `min` can only enlarge the composed
        // body (never wrongly carve a placement surface away), matching the classifier's
        // own `CombineOp::Emboss => Union` role for the same unreachable arm.
        CombineOp::Emboss { .. } => accumulator.min(operand),
    }
}

/// The scene's **composed signed distance** at `world_point` (in ABSOLUTE voxel
/// coordinates): negative inside the composed body, positive outside, `~0` on the surface,
/// and [`f32::INFINITY`] where nothing is present (an empty scene, or one whose every leaf
/// is fieldless). ADR 0027 §5 — the field-value half of the CPU continuous-placement
/// surface-raycast.
///
/// `leaves` MUST be a document-order [`Scene::leaf_producers`](document::scene::Scene::leaf_producers)
/// subsequence (the same precondition [`scoped_leaf_steps`](super::scoped_leaf_steps)
/// carries). The fold is the SAME scoped ordered CSG composition
/// [`classify_chunk_block`](super::classify_chunk_block) uses (ADR 0017 Decision 3), on
/// FIELD VALUES rather than occupancy intervals, so `composed_field_at(point) <= 0` agrees
/// with the classifier's occupancy at `point` (Union = `min`, Subtract = `max(·, −v)`,
/// Intersect = `max` — see the module docs for the sign-equivalence).
///
/// Each `Leaf` contributes its producer's [`Field::signed_distance`](document::voxel::Field::signed_distance)
/// at the point mapped into the producer's own local voxel frame by the ADR 0027 inverse
/// [`LeafAffine`](super::LeafAffine) — the exact frame map the classifier folds through
/// (never re-derived here). A **fieldless** leaf
/// ([`as_field`](document::voxel::VoxelProducer::as_field)` == None`, e.g. a cloud /
/// `VoxelBody`) contributes NOTHING and is skipped — it is never a placement surface. The
/// leaf's producer is already outset-wrapped by
/// [`leaf_producers`](document::scene::Scene::leaf_producers), so its field already carries
/// any outset (never re-applied here).
pub fn composed_field_at(
    leaves: &[&LeafProducer],
    world_point: Vec3,
    voxels_per_block: u32,
) -> f32 {
    // The stack of scope accumulators: index 0 is the ROOT accumulator, deeper entries are
    // the currently-open sealed scopes. Each starts at `+INFINITY` (the empty-body
    // identity). The stack is never empty — the root accumulator never closes.
    let mut accumulators: Vec<f32> = vec![f32::INFINITY];

    for step in scoped_leaf_steps(leaves) {
        match step {
            ScopedLeafStep::OpenScope => accumulators.push(f32::INFINITY),
            ScopedLeafStep::CloseScope(operation) => {
                let closed = accumulators
                    .pop()
                    .expect("scoped_leaf_steps emits balanced open/close markers");
                let parent = accumulators
                    .last_mut()
                    .expect("the root accumulator never closes, so a parent always remains");
                *parent = fold_operand(*parent, closed, operation);
            }
            ScopedLeafStep::Leaf(leaf) => {
                // A fieldless leaf (cloud / VoxelBody) contributes NOTHING to the field —
                // it is never a placement surface (ADR 0027 §5).
                let Some(field) = leaf.producer.as_field() else {
                    continue;
                };
                // Map the absolute world point into THIS leaf's producer-local voxel frame
                // via the classifier's own inverse affine (ADR 0027 — the frame is carried,
                // never re-derived), then read the producer's signed distance there.
                let affine = LeafAffine::of(leaf, voxels_per_block);
                let local = affine.local_of(world_point);
                let value = field.signed_distance(local.to_array(), voxels_per_block);
                let accumulator = accumulators
                    .last_mut()
                    .expect("the innermost open accumulator always exists");
                *accumulator = fold_operand(*accumulator, value, leaf.operation);
            }
        }
    }

    accumulators
        .pop()
        .expect("the root accumulator remains after every balanced step")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    use document::scene::{Node, NodeContent, NodeTransform, Scene};
    use document::voxel::SdfShape;
    use voxel_core::core_geom::MaterialChoice;
    use voxel_core::voxel::ShapeKind;

    const DENSITY: u32 = 8;

    /// A whole-block `kind` Tool of `size_blocks` at `offset_blocks` carrying `operation`.
    fn tool(
        kind: ShapeKind,
        size_blocks: [u32; 3],
        offset_blocks: [i64; 3],
        material: MaterialChoice,
        operation: CombineOp,
    ) -> Node {
        let shape = SdfShape::from_blocks(kind, size_blocks, 1, DENSITY);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset_blocks, DENSITY);
        node.operation = operation;
        node
    }

    /// The scene's resolved occupancy as a set of ABSOLUTE voxel indices — the INDEPENDENT
    /// oracle the composed field is held against. `resolve_region` yields voxels in its
    /// recentred frame (`local_index`), so each is rebased to absolute by adding the grid's
    /// carried `recentre_voxels` (ADR 0008: absolute = local + recentre).
    fn resolved_occupancy_abs(scene: &Scene, density: u32) -> BTreeSet<[i64; 3]> {
        let dense = scene.resolve_region(scene.full_extent_blocks(density), density, 0);
        let recentre = dense.recentre_voxels;
        dense
            .occupied
            .iter()
            .map(|voxel| {
                [
                    voxel.local_index[0] as i64 + recentre[0],
                    voxel.local_index[1] as i64 + recentre[1],
                    voxel.local_index[2] as i64 + recentre[2],
                ]
            })
            .collect()
    }

    /// Assert `composed_field_at(centre) <= 0` agrees with `occupied` at EVERY absolute
    /// voxel in the axis-aligned box `[min, max)` (an integer voxel index box). Sampling at
    /// the voxel CENTRE (`index + 0.5`) is the point the dense resolve classifies each voxel
    /// at, so the two answer the SAME question — the consistency proof.
    fn assert_field_sign_matches_occupancy(
        leaves: &[&LeafProducer],
        occupied: &BTreeSet<[i64; 3]>,
        min: [i64; 3],
        max: [i64; 3],
        density: u32,
        label: &str,
    ) {
        let mut checked = 0u64;
        for z in min[2]..max[2] {
            for y in min[1]..max[1] {
                for x in min[0]..max[0] {
                    let centre = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                    let field_says_inside = composed_field_at(leaves, centre, density) <= 0.0;
                    let occupied_here = occupied.contains(&[x, y, z]);
                    assert_eq!(
                        field_says_inside, occupied_here,
                        "[{label}] voxel {:?}: composed_field_at<=0 ({field_says_inside}) must \
                         agree with the resolved occupancy ({occupied_here})",
                        [x, y, z]
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 0, "[{label}] the sweep box must be non-empty");
    }

    /// The bounding box of `occupied`, expanded by `margin` voxels on every side, so the
    /// sweep covers deep-interior (negative), surface (`~0`), and a shell of exterior
    /// (positive) voxels.
    fn occupancy_bounds(occupied: &BTreeSet<[i64; 3]>, margin: i64) -> ([i64; 3], [i64; 3]) {
        let mut min = [i64::MAX; 3];
        let mut max = [i64::MIN; 3];
        for voxel in occupied {
            for axis in 0..3 {
                min[axis] = min[axis].min(voxel[axis]);
                max[axis] = max[axis].max(voxel[axis] + 1); // half-open
            }
        }
        (
            [min[0] - margin, min[1] - margin, min[2] - margin],
            [max[0] + margin, max[1] + margin, max[2] + margin],
        )
    }

    /// (1) A single Sphere Tool: the centre is deep inside (negative), far outside is
    /// positive, AND — the consistency proof — `composed_field_at <= 0` agrees with the
    /// resolved occupancy at EVERY voxel of a box covering the sphere plus a shell.
    #[test]
    fn single_sphere_field_matches_occupancy_everywhere() {
        // A 4×4×4-block sphere at the origin → voxel extent [0, 32) per axis, centre (16,16,16).
        let scene = Scene::from_nodes(vec![tool(
            ShapeKind::Sphere,
            [4, 4, 4],
            [0, 0, 0],
            MaterialChoice::Stone,
            CombineOp::Union,
        )]);
        let leaves = scene.leaf_producers(DENSITY);
        let leaves: Vec<&LeafProducer> = leaves.iter().collect();

        let centre = Vec3::new(16.0, 16.0, 16.0);
        assert!(
            composed_field_at(&leaves, centre, DENSITY) < 0.0,
            "the sphere centre is well inside the body (negative)"
        );
        let far_outside = Vec3::new(500.0, 16.0, 16.0);
        assert!(
            composed_field_at(&leaves, far_outside, DENSITY) > 0.0,
            "a point far outside every leaf is positive"
        );

        let occupied = resolved_occupancy_abs(&scene, DENSITY);
        let (min, max) = occupancy_bounds(&occupied, 3);
        assert_field_sign_matches_occupancy(&leaves, &occupied, min, max, DENSITY, "sphere");
    }

    /// An off-origin ellipsoid Tool (a non-cube SDF shape at a non-zero offset) — the
    /// consistency proof again, exercising the affine's translation and non-uniform extents.
    #[test]
    fn offset_ellipsoid_field_matches_occupancy_everywhere() {
        // A 5×3×4-block sphere (⇒ ellipsoid) at block offset (2,1,3).
        let scene = Scene::from_nodes(vec![tool(
            ShapeKind::Sphere,
            [5, 3, 4],
            [2, 1, 3],
            MaterialChoice::Stone,
            CombineOp::Union,
        )]);
        let leaves = scene.leaf_producers(DENSITY);
        let leaves: Vec<&LeafProducer> = leaves.iter().collect();

        let occupied = resolved_occupancy_abs(&scene, DENSITY);
        assert!(!occupied.is_empty(), "the ellipsoid must occupy some voxels");
        let (min, max) = occupancy_bounds(&occupied, 3);
        assert_field_sign_matches_occupancy(&leaves, &occupied, min, max, DENSITY, "ellipsoid");
    }

    /// (2) Union of two boxes: the composed field is the `min` of the two — a point inside
    /// EITHER box is negative, a point inside neither is positive, and the sign agrees with
    /// the resolved (unioned) occupancy everywhere.
    #[test]
    fn union_of_two_boxes_is_the_min_field() {
        // Two 3-block boxes, separated so each has a private interior and they share a seam.
        let scene = Scene::from_nodes(vec![
            tool(ShapeKind::Box, [3, 3, 3], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            tool(ShapeKind::Box, [3, 3, 3], [3, 0, 0], MaterialChoice::Wood, CombineOp::Union),
        ]);
        let leaves = scene.leaf_producers(DENSITY);
        let leaves: Vec<&LeafProducer> = leaves.iter().collect();

        // A point inside the FIRST box only ([0,24) on X) is negative.
        let inside_first = Vec3::new(4.0, 12.0, 12.0);
        assert!(
            composed_field_at(&leaves, inside_first, DENSITY) < 0.0,
            "a point inside the first box is negative (Union = min)"
        );
        // A point inside the SECOND box only ([24,48) on X) is negative.
        let inside_second = Vec3::new(44.0, 12.0, 12.0);
        assert!(
            composed_field_at(&leaves, inside_second, DENSITY) < 0.0,
            "a point inside the second box is negative (Union = min)"
        );
        // A point outside BOTH (above them in Z) is positive.
        let outside_both = Vec3::new(12.0, 12.0, 100.0);
        assert!(
            composed_field_at(&leaves, outside_both, DENSITY) > 0.0,
            "a point inside neither box is positive"
        );

        let occupied = resolved_occupancy_abs(&scene, DENSITY);
        let (min, max) = occupancy_bounds(&occupied, 3);
        assert_field_sign_matches_occupancy(&leaves, &occupied, min, max, DENSITY, "union");
    }

    /// (3) Subtract — a cutter carving a box: a point inside the cutter-carved region is
    /// POSITIVE (outside the composed body) though it is inside the base box, proving the
    /// `max(accumulator, −value)` Subtract fold. The sign agrees with the carved occupancy
    /// everywhere.
    #[test]
    fn subtract_carves_a_positive_region_inside_the_base() {
        // A 4-block Stone base with a 2-block Wood cutter wholly inside it (block [1,3)³).
        let scene = Scene::from_nodes(vec![
            tool(ShapeKind::Box, [4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            tool(ShapeKind::Box, [2, 2, 2], [1, 1, 1], MaterialChoice::Wood, CombineOp::Subtract),
        ]);
        let leaves = scene.leaf_producers(DENSITY);
        let leaves: Vec<&LeafProducer> = leaves.iter().collect();

        // The cutter occupies voxels [8,24)³; its centre (16,16,16) is inside the BASE box
        // ([0,32)³) yet carved away — so the composed field there is POSITIVE.
        let carved_centre = Vec3::new(16.0, 16.0, 16.0);
        assert!(
            composed_field_at(&leaves, carved_centre, DENSITY) > 0.0,
            "a point inside the cutter is OUTSIDE the composed body (Subtract = max(acc, -v))"
        );
        // The base box's own centre lies inside the cutter too; a corner of the base far
        // from the cutter (voxel (2,2,2)) is still solid ⇒ negative.
        let uncarved = Vec3::new(2.5, 2.5, 2.5);
        assert!(
            composed_field_at(&leaves, uncarved, DENSITY) < 0.0,
            "a base-box point the cutter does not reach stays inside (negative)"
        );

        let occupied = resolved_occupancy_abs(&scene, DENSITY);
        assert!(
            !occupied.iter().any(|voxel| *voxel == [16, 16, 16]),
            "the carved centre voxel must NOT be occupied (sanity on the oracle)"
        );
        let (min, max) = occupancy_bounds(&occupied, 3);
        assert_field_sign_matches_occupancy(&leaves, &occupied, min, max, DENSITY, "subtract");
    }

    /// An empty scene (no leaves) and a scene of only a fieldless leaf both report
    /// `+INFINITY` everywhere — nothing is present, so there is no placement surface.
    #[test]
    fn empty_and_fieldless_scenes_are_infinite() {
        let empty: Vec<&LeafProducer> = Vec::new();
        assert_eq!(
            composed_field_at(&empty, Vec3::new(1.0, 2.0, 3.0), DENSITY),
            f32::INFINITY,
            "an empty scene has no field anywhere"
        );

        // A DebugClouds VoxelBody is fieldless (`as_field() == None`) — it contributes
        // nothing to the composed field, so a cloud-only scene is also everywhere infinite.
        use document::scene::VoxelBody;
        let scene = Scene::from_nodes(vec![Node::new(
            "Clouds",
            NodeContent::VoxelBody(VoxelBody::DebugClouds { seed: 7 }),
        )]);
        let leaves = scene.leaf_producers(DENSITY);
        let leaves: Vec<&LeafProducer> = leaves.iter().collect();
        assert_eq!(
            composed_field_at(&leaves, Vec3::new(4.0, 4.0, 4.0), DENSITY),
            f32::INFINITY,
            "a scene of only a fieldless leaf has no composed field"
        );
    }
}
