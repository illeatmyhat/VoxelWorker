//! Boolean-operand ghost derivation (ADR 0018 Decision 6 — "Show booleans" mode):
//! selection → ghost bodies + frame.
//!
//! The app_core half of the seam: read the boolean-operand body slices of the ACTIVE
//! selection's subtree from the document ([`Scene::boolean_operand_body_slices`]) —
//! every Subtract/Intersect operand within the selected subtree (the root part selects
//! the whole scene) — evaluate each through the two-layer evaluator (bounded by that
//! operand's covering chunks — never a whole-scene resolve, and never a dense grid), and
//! hand the display layer plain meshes-to-be + styles
//! ([`display::mesh::SelectedOperandGhostBody`]). Display renders, app_core derives, the
//! document stays pure (ADR 0016).
//!
//! Re-derived only on selection / geometry / MODE change (the shell + `shot` call this at
//! those seams), never per frame. The mode gate (only Show-booleans mode ghosts) lives at
//! the call site — this derivation is mode-agnostic.

use display::mesh::SelectedOperandGhostBody;
use display::renderer::OperandGhostStyle;
use document::scene::{CombineOp, NodeId, Scene};
use evaluation::two_layer_store::TwoLayerStore;
use substrate::spatial::{LeafPlacement, ProducerLocalVoxelPoint};
use voxel_core::voxel::RecentreVoxels;

use super::AppCore;

/// Everything the display's [`SelectedOperandGhostRenderer`] rebuild needs: the ghost
/// bodies plus the COMPOSED scene's frame (ADR 0008 — the slice chunks are in absolute
/// composite coords, so meshing them against the composed recentre lands the ghost
/// voxel-exact on the operand's place in the render frame).
///
/// [`SelectedOperandGhostRenderer`]: display::mesh::SelectedOperandGhostRenderer
pub struct SelectedOperandGhost {
    /// One body per boolean operand in the selected subtree (a fixture-instance selection
    /// contributes one per spliced boolean child).
    pub bodies: Vec<SelectedOperandGhostBody>,
    /// The composed scene's voxel extent (the shader's corner-anchoring scalar).
    pub grid_dimensions: [u32; 3],
    /// The composed scene's resolve recentre — the render frame the ghost meshes into.
    pub recentre: RecentreVoxels,
    /// The document density the bodies were evaluated at.
    pub density: u32,
}

/// Map the document's combine operation onto display's ghost-style vocabulary (the
/// display layer never reads `CombineOp` — ADR 0016 layering). The boolean-operand walk
/// only ever emits mask operands, so Union never reaches here.
fn operand_ghost_style_for(operation: CombineOp) -> OperandGhostStyle {
    match operation {
        CombineOp::Subtract => OperandGhostStyle::Subtract,
        CombineOp::Intersect => OperandGhostStyle::Intersect,
        // `is_boolean_operand` (scene::operand_body) admits only Subtract and Intersect, so
        // neither of these reaches the mapper. Emboss has a footprint and could plausibly
        // earn an x-ray ghost of its own later, but it would need its own style rather than
        // borrowing a mask's — it neither removes nor keeps, it MOVES the surface.
        CombineOp::Union | CombineOp::Emboss { .. } => {
            unreachable!("the boolean-operand walk only emits Subtract/Intersect operands")
        }
    }
}

/// Everything the display's [`SelectionOutlineRenderer`] rebuild needs (ADR 0032 —
/// viewport selection feedback): the selected nodes' standalone bodies plus the COMPOSED
/// scene's frame (ADR 0008, same contract as [`SelectedOperandGhost`]).
///
/// [`SelectionOutlineRenderer`]: display::mesh::SelectionOutlineRenderer
pub struct SelectedBodyCel {
    /// One body per surviving selected node (selection-roots filtered: a node whose
    /// ancestor is also selected contributes nothing — the ancestor's composed body
    /// already covers it; the bodies union in the outline's shared depth map).
    pub bodies: Vec<display::mesh::SelectedBodyChunks>,
    /// The composed scene's voxel extent (the shader's corner-anchoring scalar).
    pub grid_dimensions: [u32; 3],
    /// The composed scene's resolve recentre — the render frame the cel meshes into.
    pub recentre: RecentreVoxels,
    /// The document density the bodies were evaluated at.
    pub density: u32,
    /// Analytic feature-edge segments of the selected shapes (flat endpoint pairs) in
    /// RENDER-frame voxels — the authored geometry's own edges (a box's 12, a
    /// cylinder's 2 rim ellipses, a tube's 4), not anything derived from the voxel
    /// surface. Empty when no selected shape catalogues any edge.
    pub edge_segments: Vec<[f32; 3]>,
    /// A cap cut the junction tracing short (pair cap, seed or step budget) —
    /// some crease lines may be missing. Surfaced so a missing-line report is a
    /// lookup, not a hunt; the catalogue edges are never truncated.
    pub edge_trace_truncated: bool,
}

/// Segments per tessellated rim ellipse. Fixed (not screen-adaptive) so the polyline
/// is world-stable under orbit — the whole point of the analytic edges.
const EDGE_CIRCLE_SEGMENTS: u32 = 96;

/// Emit every leaf's authored edge catalogue ([`VoxelProducer::edge_polylines_local`])
/// as segment endpoint pairs in TRUE-WORLD voxels. Consumes the SAME `leaf_producers`
/// walk the evaluator reads — never a hand-mirrored descent — so placement, instance
/// expansion, the cycle guard and fixture splicing stay single-sourced. A pre-composed
/// scope or an outset-wrapped leaf arrives as a producer that honestly catalogues
/// nothing (its authored edges are gone from the surface it resolves).
///
/// [`VoxelProducer::edge_polylines_local`]: document::voxel::VoxelProducer::edge_polylines_local
fn collect_edge_segments_true_world(slice: &Scene, density: u32, out: &mut Vec<[f32; 3]>) {
    for leaf in slice.leaf_producers(density) {
        let polylines = leaf
            .producer
            .edge_polylines_local(density, EDGE_CIRCLE_SEGMENTS);
        if polylines.is_empty() {
            continue;
        }
        let grid = leaf.producer.full_dimensions(density);
        let full = glam::Vec3::new(grid[0] as f32, grid[1] as f32, grid[2] as f32);
        let placement = LeafPlacement::from_origin_and_local(
            leaf.rotation,
            full,
            leaf.world_offset_voxels,
            leaf.offset_local_voxels,
        );
        for polyline in polylines {
            for pair in polyline.windows(2) {
                for point in pair {
                    out.push(
                        placement
                            .world_of(ProducerLocalVoxelPoint::from_voxels(
                                glam::Vec3::from_array(*point),
                            ))
                            .voxels()
                            .to_array(),
                    );
                }
            }
        }
    }
}

/// Hard cap on traced surface pairs per selection change (runaway guard for a
/// selection overlapping very many bodies). Deterministic truncation: pairs
/// enumerate `(i, j < i)` over the DOCUMENT-ORDER leaf walk, so the same scene
/// and selection always trace the same 64 — the cut costs junction lines (and
/// raises the truncation flag), never correctness or golden stability.
const JUNCTION_PAIR_CAP: usize = 64;

/// Trace the CSG junction curves of the selection — where a selected leaf's
/// surface crosses ANOTHER leaf's surface (a cutter's wall meeting the face it
/// carved) — as segment endpoint pairs in TRUE-WORLD voxels, appended to `out`.
///
/// Pairs come from the SAME `leaf_producers` walk the evaluator reads, pruned by
/// inflated `world_aabb` overlap; a pair traces when at least one side belongs to
/// the selected subtree (an external cutter's crease on a selected host counts,
/// and vice versa). Fold order is deliberately NOT consulted: the junction lives
/// on `F = 0 ∩ G = 0` regardless of which leaf folded first, and the shader's
/// hull + scene-depth gates discard any span over carved-away or occluded space.
/// Emboss leaves skip (their resolved surface has moved off the authored field);
/// a fieldless producer skips by type. The prune brackets are the evaluator's own
/// `cell_field_interval` — conservative-never-narrow per producer, so the seeding
/// never silently loses a curve to an optimistic Lipschitz guess.
///
/// Returns whether any cap truncated the result (the pair cap here, or a seed /
/// step budget inside the tracer) — surfaced, never silent.
fn collect_junction_segments_true_world(
    scene: &Scene,
    picked: &std::collections::BTreeSet<NodeId>,
    density: u32,
    out: &mut Vec<[f32; 3]>,
) -> bool {
    let mut truncated = false;
    let in_selection = |id: NodeId| {
        if picked.contains(&id) {
            return true;
        }
        let mut current = id;
        while let Some((Some(parent), _)) = scene.parent_and_index_of(current) {
            if picked.contains(&parent) {
                return true;
            }
            current = parent;
        }
        false
    };
    struct PairSide {
        placement: LeafPlacement,
        aabb: ([i64; 3], [i64; 3]),
        producer: Box<dyn document::voxel::VoxelProducer>,
        selected: bool,
    }
    let sides: Vec<PairSide> = scene
        .leaf_producers(density)
        .into_iter()
        .filter(|leaf| !matches!(leaf.operation, CombineOp::Emboss { .. }))
        .filter(|leaf| leaf.producer.as_field().is_some())
        .map(|leaf| {
            let grid = leaf.producer.full_dimensions(density);
            let placement = LeafPlacement::from_origin_and_local(
                leaf.rotation,
                glam::Vec3::new(grid[0] as f32, grid[1] as f32, grid[2] as f32),
                leaf.world_offset_voxels,
                leaf.offset_local_voxels,
            );
            let aabb = placement.world_aabb();
            let selected = in_selection(leaf.origin.node)
                || leaf.origin.instance_host.is_some_and(in_selection);
            PairSide {
                placement,
                aabb,
                producer: leaf.producer,
                selected,
            }
        })
        .collect();

    // A field + brackets over one side, in TRUE-WORLD voxels: the field maps the
    // sample through the placement's inverse; the bracket maps the world cube's
    // corners to a local enclosing cell and asks the producer's own conservative
    // interval (an unknown interval keeps the cell — never prunes).
    fn world_field(side: &PairSide, density: u32) -> impl Fn(glam::Vec3) -> f32 + '_ {
        let placement = side.placement;
        let field = side
            .producer
            .as_field()
            .expect("filtered to field-bearing producers");
        move |point: glam::Vec3| -> f32 {
            let local = placement
                .local_of(substrate::spatial::TrueWorldVoxelPoint::from_voxels(point))
                .voxels();
            field.signed_distance(local.to_array(), density)
        }
    }
    fn world_bracket(side: &PairSide, density: u32) -> impl Fn(glam::Vec3, f32) -> (f32, f32) + '_ {
        let placement = side.placement;
        let producer = &side.producer;
        move |cell_min: glam::Vec3, size: f32| -> (f32, f32) {
            let mut local_min = [f32::MAX; 3];
            let mut local_max = [f32::MIN; 3];
            for corner in 0..8u8 {
                let world = cell_min
                    + glam::Vec3::new(
                        if corner & 1 != 0 { size } else { 0.0 },
                        if corner & 2 != 0 { size } else { 0.0 },
                        if corner & 4 != 0 { size } else { 0.0 },
                    );
                let local = placement
                    .local_of(substrate::spatial::TrueWorldVoxelPoint::from_voxels(world))
                    .voxels();
                for axis in 0..3 {
                    local_min[axis] = local_min[axis].min(local[axis]);
                    local_max[axis] = local_max[axis].max(local[axis]);
                }
            }
            let cell = voxel_core::spatial_index::VoxelAabb::new(
                std::array::from_fn(|axis| local_min[axis].floor() as i64),
                std::array::from_fn(|axis| local_max[axis].ceil() as i64),
            );
            match producer.cell_field_interval(cell, density) {
                Some(interval) => (interval.minimum, interval.maximum),
                None => (-1.0, 1.0),
            }
        }
    }

    let config = substrate::spatial::SurfaceIntersectionConfig::default();
    let mut traced_pairs = 0usize;
    for first in 0..sides.len() {
        for second in (first + 1)..sides.len() {
            let (a, b) = (&sides[first], &sides[second]);
            if !(a.selected || b.selected) {
                continue;
            }
            // Inflated-AABB broadphase: surfaces can only cross inside the overlap.
            let mut overlap_min = [0f32; 3];
            let mut overlap_max = [0f32; 3];
            let mut disjoint = false;
            for axis in 0..3 {
                let low = (a.aabb.0[axis].max(b.aabb.0[axis]) - 1) as f32;
                let high = (a.aabb.1[axis].min(b.aabb.1[axis]) + 1) as f32;
                if high <= low {
                    disjoint = true;
                    break;
                }
                overlap_min[axis] = low;
                overlap_max[axis] = high;
            }
            if disjoint {
                continue;
            }
            if traced_pairs >= JUNCTION_PAIR_CAP {
                return true;
            }
            traced_pairs += 1;
            let field_a = world_field(a, density);
            let field_b = world_field(b, density);
            let bracket_a = world_bracket(a, density);
            let bracket_b = world_bracket(b, density);
            let outcome = substrate::spatial::trace_intersection_curves(
                &substrate::spatial::ImplicitSurfacePair {
                    field_f: &field_a,
                    field_g: &field_b,
                    bracket_f: &bracket_a,
                    bracket_g: &bracket_b,
                },
                glam::Vec3::from_array(overlap_min),
                glam::Vec3::from_array(overlap_max),
                &config,
            );
            truncated |= outcome.seed_budget_exhausted || outcome.step_budget_exhausted;
            for curve in outcome.curves {
                for pair in curve.points.windows(2) {
                    out.push(pair[0].to_array());
                    out.push(pair[1].to_array());
                }
                if curve.closed && curve.points.len() > 2 {
                    out.push(curve.points[curve.points.len() - 1].to_array());
                    out.push(curve.points[0].to_array());
                }
            }
        }
    }
    truncated
}

impl AppCore {
    /// Derive the boolean-operand ghost for `target`'s subtree (ADR 0018 Decision 6 —
    /// "Show booleans" mode), or `None` when the subtree covers no boolean with
    /// geometry.
    ///
    /// Cost bound: each operand slice is evaluated over ITS OWN covering chunk range
    /// (the operand body's extent) via the stateless two-layer evaluator — a selection
    /// change never re-resolves the whole scene, and no dense whole-region grid is ever
    /// assembled (the user law).
    pub fn boolean_operand_ghost(
        scene: &Scene,
        target: NodeId,
        density: u32,
    ) -> Option<SelectedOperandGhost> {
        evaluate_operand_ghost_slices(scene, scene.boolean_operand_body_slices(target), density)
    }

    /// Derive the selection-cel bodies for `targets` (ADR 0032 — viewport selection
    /// feedback, all view modes), or `None` when no target yields a body with geometry.
    ///
    /// Selection-roots filtered: a target with a selected ancestor is skipped (the
    /// ancestor's composed body already covers it — drawing both would double the cel
    /// alpha). The root part, stale ids and disabled nodes derive nothing
    /// ([`Scene::node_body_slice`]). Same cost bound as the operand ghost: each body is
    /// evaluated over its OWN covering chunks only.
    pub fn selected_body_cel(
        scene: &Scene,
        targets: &[NodeId],
        density: u32,
    ) -> Option<SelectedBodyCel> {
        let picked: std::collections::BTreeSet<NodeId> = targets.iter().copied().collect();
        let has_picked_ancestor = |id: NodeId| {
            let mut current = id;
            while let Some((Some(parent), _)) = scene.parent_and_index_of(current) {
                if picked.contains(&parent) {
                    return true;
                }
                current = parent;
            }
            false
        };
        let store = TwoLayerStore::enabled();
        let mut bodies = Vec::new();
        let mut edge_segments_true_world = Vec::new();
        for &target in targets {
            if has_picked_ancestor(target) {
                continue;
            }
            let Some(slice) = scene.node_body_slice(target) else {
                continue;
            };
            let chunks = store.build_covering_chunks(&slice, density, 0);
            if chunks.iter().all(|(_, chunk)| !chunk.has_geometry()) {
                continue;
            }
            bodies.push(chunks);
            collect_edge_segments_true_world(&slice, density, &mut edge_segments_true_world);
        }
        if bodies.is_empty() {
            return None;
        }
        let edge_trace_truncated = collect_junction_segments_true_world(
            scene,
            &picked,
            density,
            &mut edge_segments_true_world,
        );
        let recentre = scene.recentre_voxels_for_resolve(density);
        let recentre_f32 = recentre.voxels().map(|axis| axis as f32);
        let edge_segments = edge_segments_true_world
            .into_iter()
            .map(|point| std::array::from_fn(|axis| point[axis] - recentre_f32[axis]))
            .collect();
        Some(SelectedBodyCel {
            bodies,
            grid_dimensions: scene.placed_region_dimensions(density),
            recentre,
            density,
            edge_segments,
            edge_trace_truncated,
        })
    }
}

/// The evaluation half of the ghost derivation: run each `(operation, slice)` through the
/// stateless two-layer evaluator — bounded by the SLICE's covering chunks, never a
/// whole-scene resolve, never a dense grid — and package the surviving bodies with the
/// COMPOSED scene's frame (ADR 0008: the slices are in absolute composite coords, so
/// meshing against the composed recentre lands each ghost voxel-exact).
fn evaluate_operand_ghost_slices(
    scene: &Scene,
    slices: Vec<(CombineOp, Scene)>,
    density: u32,
) -> Option<SelectedOperandGhost> {
    if slices.is_empty() {
        return None;
    }
    let store = TwoLayerStore::enabled();
    let mut bodies = Vec::new();
    for (operation, slice) in &slices {
        let chunks = store.build_covering_chunks(slice, density, 0);
        // A body that evaluates to nothing (e.g. an empty definition) ghosts nothing.
        if chunks.iter().all(|(_, chunk)| !chunk.has_geometry()) {
            continue;
        }
        bodies.push(SelectedOperandGhostBody {
            style: operand_ghost_style_for(*operation),
            chunks,
        });
    }
    if bodies.is_empty() {
        return None;
    }
    Some(SelectedOperandGhost {
        bodies,
        grid_dimensions: scene.placed_region_dimensions(density),
        recentre: scene.recentre_voxels_for_resolve(density),
        density,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use document::scene::{DefId, Node, NodeContent, NodeTransform, ROOT_NODE_ID};
    use document::voxel::{SdfShape, VoxelProducer};
    use voxel_core::core_geom::MaterialChoice;
    use voxel_core::voxel::ShapeKind;

    const DENSITY: u32 = 8;

    fn box_tool(
        size_blocks: [u32; 3],
        offset_blocks: [i64; 3],
        operation: CombineOp,
        name: &str,
    ) -> Node {
        let shape = SdfShape::from_blocks(ShapeKind::Box, size_blocks, 1, DENSITY);
        let mut node = Node::new(
            name,
            NodeContent::Tool {
                shape,
                material: MaterialChoice::Stone,
            },
        );
        node.transform = NodeTransform::from_blocks(offset_blocks, DENSITY);
        node.operation = operation;
        node
    }

    fn host_and_cutter_scene() -> Scene {
        let mut scene = Scene::from_nodes(vec![
            box_tool([4, 4, 4], [0, 0, 0], CombineOp::Union, "Host"),
            box_tool([2, 2, 2], [1, 1, 1], CombineOp::Subtract, "Cutter"),
        ]);
        scene.voxels_per_block = DENSITY;
        scene.ensure_node_ids();
        scene
    }

    /// No ghost for a stale target.
    #[test]
    fn stale_target_derives_no_ghost() {
        let scene = host_and_cutter_scene();
        assert!(AppCore::boolean_operand_ghost(&scene, NodeId(9999), DENSITY).is_none());
    }

    /// A boolean operand ghosts in its operation style; a Union selection has no boolean
    /// operand in its (leaf) subtree, so it ghosts nothing (never a Union tint).
    #[test]
    fn styles_follow_the_selected_operation() {
        let scene = host_and_cutter_scene();
        let ghost =
            AppCore::boolean_operand_ghost(&scene, scene.roots[1], DENSITY).expect("cutter ghosts");
        assert_eq!(ghost.bodies.len(), 1);
        assert_eq!(ghost.bodies[0].style, OperandGhostStyle::Subtract);

        // The Union host is a non-boolean leaf: nothing to reveal.
        assert!(AppCore::boolean_operand_ghost(&scene, scene.roots[0], DENSITY).is_none());
    }

    /// Re-derivation on selection change resolves ONLY the selected operand's covering
    /// chunks — the derivation seam's no-whole-scene-re-resolve bound: the small cutter's
    /// ghost holds one chunk while the scene spans many.
    #[test]
    fn derivation_is_bounded_by_the_selected_body() {
        let mut scene = Scene::from_nodes(vec![
            box_tool([4, 4, 4], [40, 0, 0], CombineOp::Union, "Far host"),
            box_tool([2, 2, 2], [0, 0, 0], CombineOp::Subtract, "Cutter"),
        ]);
        scene.voxels_per_block = DENSITY;
        scene.ensure_node_ids();

        let ghost =
            AppCore::boolean_operand_ghost(&scene, scene.roots[1], DENSITY).expect("cutter ghosts");
        assert_eq!(
            ghost.bodies[0].chunks.len(),
            1,
            "the 2-block cutter covers ONE chunk; the far host's extent is never evaluated"
        );
        // The frame handed to display is the COMPOSED scene's (ADR 0008), so the ghost
        // mesh lands in the same render frame as the solid.
        assert_eq!(
            ghost.grid_dimensions,
            scene.placed_region_dimensions(DENSITY)
        );
        assert_eq!(
            ghost.recentre.voxels(),
            scene.recentre_voxels_for_resolve(DENSITY).voxels()
        );
    }

    /// A buried cutter's ghost body is the cutter's OWN full body (the two-layer chunks
    /// carry its geometry even though the composed scene swallows it entirely).
    #[test]
    fn buried_cutter_still_derives_its_body() {
        let scene = host_and_cutter_scene();
        let ghost =
            AppCore::boolean_operand_ghost(&scene, scene.roots[1], DENSITY).expect("cutter ghosts");
        let stored: u64 = ghost.bodies[0]
            .chunks
            .iter()
            .map(|(_, chunk)| chunk.stored_voxel_count())
            .sum();
        assert!(
            stored > 0,
            "the fully-buried cutter's own body must not be empty"
        );
    }

    /// ADR 0032 selection cel: every selected node derives its OWN standalone body —
    /// including a Union host (which never ghosts in Show-booleans) and a Subtract
    /// cutter (its root op neutralised so the body resolves constructively).
    #[test]
    fn cel_derives_a_body_per_selected_node() {
        let scene = host_and_cutter_scene();
        let cel = AppCore::selected_body_cel(&scene, &[scene.roots[0], scene.roots[1]], DENSITY)
            .expect("both nodes derive bodies");
        assert_eq!(cel.bodies.len(), 2);
        for chunks in &cel.bodies {
            let stored: u64 = chunks
                .iter()
                .map(|(_, chunk)| chunk.stored_voxel_count())
                .sum();
            assert!(stored > 0, "each selected body carries its own geometry");
        }
        assert_eq!(cel.grid_dimensions, scene.placed_region_dimensions(DENSITY));
        assert_eq!(
            cel.recentre.voxels(),
            scene.recentre_voxels_for_resolve(DENSITY).voxels()
        );
    }

    /// The root part, a stale id, and a disabled node derive no cel body: the root IS
    /// the render, a stale id names nothing, and a disabled node shows no surface for a
    /// depth-tested overlay to sit on.
    #[test]
    fn cel_skips_root_stale_and_disabled_targets() {
        let mut scene = host_and_cutter_scene();
        assert!(AppCore::selected_body_cel(&scene, &[ROOT_NODE_ID], DENSITY).is_none());
        assert!(AppCore::selected_body_cel(&scene, &[NodeId(9999)], DENSITY).is_none());
        let host = scene.roots[0];
        scene.arena.get_mut(&host).unwrap().enabled = false;
        assert!(AppCore::selected_body_cel(&scene, &[host], DENSITY).is_none());
    }

    /// Analytic edge catalogue (ADR 0032 V1): a box lists its 12 straight edges on
    /// the `[0, full]` corners; a sphere and a torus are smooth and list nothing.
    #[test]
    fn box_catalogues_twelve_edges_and_smooth_kinds_none() {
        let catalogue = |kind, blocks| {
            SdfShape::from_blocks(kind, blocks, 1, DENSITY)
                .edge_polylines_local(DENSITY, EDGE_CIRCLE_SEGMENTS)
        };
        let edges = catalogue(ShapeKind::Box, [4, 4, 4]);
        assert_eq!(edges.len(), 12);
        for polyline in &edges {
            assert_eq!(polyline.len(), 2, "a box edge is one straight segment");
            for point in polyline {
                for axis in 0..3 {
                    assert!(
                        point[axis] == 0.0 || point[axis] == 32.0,
                        "box edge endpoints sit on the box corners, got {point:?}"
                    );
                }
            }
        }
        assert!(catalogue(ShapeKind::Sphere, [4, 4, 4]).is_empty());
        assert!(catalogue(ShapeKind::Torus, [4, 4, 2]).is_empty());
    }

    /// A tube catalogues 4 rim ellipses (outer + inner × top + bottom, axis along Z);
    /// a wall consuming the whole cross-section closes the hole and drops the inner
    /// pair. A cylinder is the outer pair alone.
    #[test]
    fn tube_catalogues_four_rims_until_the_wall_closes_the_hole() {
        let catalogue = |kind, blocks, wall_blocks| {
            SdfShape::from_blocks(kind, blocks, wall_blocks, DENSITY)
                .edge_polylines_local(DENSITY, EDGE_CIRCLE_SEGMENTS)
        };
        let rims = catalogue(ShapeKind::Tube, [8, 8, 4], 1);
        assert_eq!(rims.len(), 4);
        for rim in &rims {
            let z = rim[0][2];
            assert!(z == 0.0 || z == 32.0, "rims sit on the tube's caps");
            assert!(rim.iter().all(|point| point[2] == z), "each rim is planar");
        }
        // Angle 0 of each rim: centre (32, 32) + radius along +X — outer 32, inner 24.
        let radii: Vec<f32> = rims.iter().map(|rim| rim[0][0] - 32.0).collect();
        assert_eq!(radii, vec![32.0, 32.0, 24.0, 24.0]);

        let walled_shut = catalogue(ShapeKind::Tube, [4, 4, 4], 2);
        assert_eq!(walled_shut.len(), 2, "no hole, no inner rims");
        assert_eq!(catalogue(ShapeKind::Cylinder, [8, 8, 4], 0).len(), 2);
    }

    /// The cel's edge segments land in the RENDER frame: the host box's 12 edges
    /// (24 endpoints) sit on the `[0, 32]` true-world corners minus the recentre.
    #[test]
    fn cel_edge_segments_land_in_the_render_frame() {
        let scene = host_and_cutter_scene();
        let cel = AppCore::selected_body_cel(&scene, &[scene.roots[0]], DENSITY)
            .expect("host derives a body");
        assert_eq!(cel.edge_segments.len(), 24, "12 edges, 2 endpoints each");
        let recentre = cel.recentre.voxels();
        for point in &cel.edge_segments {
            for axis in 0..3 {
                let true_world = point[axis] + recentre[axis] as f32;
                assert!(
                    true_world == 0.0 || true_world == 32.0,
                    "endpoint {point:?} must be a box corner in true world"
                );
            }
        }
    }

    /// A rotated node's edges turn with it under the corner-anchor convention: a
    /// 64×32×32 box turned 90° about Z re-anchors so its edges span 32×64×32 from
    /// the node's world offset.
    #[test]
    fn cel_edges_follow_the_node_rotation() {
        let mut scene = Scene::from_nodes(vec![box_tool(
            [8, 4, 4],
            [0, 0, 0],
            CombineOp::Union,
            "Turned",
        )]);
        scene.voxels_per_block = DENSITY;
        scene.ensure_node_ids();
        let target = scene.roots[0];
        let quarter_turn = glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let node = scene.arena.get_mut(&target).unwrap();
        node.transform = node.transform.clone().with_rotation(quarter_turn);

        let cel = AppCore::selected_body_cel(&scene, &[target], DENSITY).expect("derives a body");
        let recentre = cel.recentre.voxels();
        let mut max = [f32::MIN; 3];
        let mut min = [f32::MAX; 3];
        for point in &cel.edge_segments {
            for axis in 0..3 {
                let true_world = point[axis] + recentre[axis] as f32;
                min[axis] = min[axis].min(true_world);
                max[axis] = max[axis].max(true_world);
            }
        }
        for axis in 0..3 {
            assert!(
                min[axis].abs() < 1e-3,
                "low corner anchors on the world offset, got {min:?}"
            );
        }
        let span: Vec<f32> = (0..3).map(|axis| max[axis] - min[axis]).collect();
        assert!(
            (span[0] - 32.0).abs() < 1e-3
                && (span[1] - 64.0).abs() < 1e-3
                && (span[2] - 32.0).abs() < 1e-3,
            "a 64×32×32 box turned 90° about Z spans 32×64×32, got {span:?}"
        );
    }

    /// An instance's edges are the definition's catalogue under the instance's
    /// placement: a definition holding a rotated 64×32×32 box, instanced at a block
    /// offset, shows 12 edges spanning 32×64×32 anchored on the instance offset —
    /// translation-only composition, exactly like the resolve walk.
    #[test]
    fn instance_edges_land_at_the_instanced_placement() {
        let quarter_turn = glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let mut inner = box_tool([8, 4, 4], [0, 0, 0], CombineOp::Union, "Turned");
        inner.transform = inner.transform.clone().with_rotation(quarter_turn);
        let mut instance = Node::new("House 1", NodeContent::Instance(DefId(7)));
        instance.transform = NodeTransform::from_blocks([2, 0, 0], DENSITY);
        let mut scene = Scene::from_nodes(vec![instance]);
        scene.voxels_per_block = DENSITY;
        scene.add_definition(DefId(7), "House", [inner]);
        scene.ensure_node_ids();

        let target = scene.roots[0];
        let cel = AppCore::selected_body_cel(&scene, &[target], DENSITY).expect("instance body");
        assert_eq!(
            cel.edge_segments.len(),
            24,
            "12 box edges, 2 endpoints each"
        );
        let recentre = cel.recentre.voxels();
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for point in &cel.edge_segments {
            for axis in 0..3 {
                let true_world = point[axis] + recentre[axis] as f32;
                min[axis] = min[axis].min(true_world);
                max[axis] = max[axis].max(true_world);
            }
        }
        assert!(
            (min[0] - 16.0).abs() < 1e-3 && min[1].abs() < 1e-3 && min[2].abs() < 1e-3,
            "low corner anchors on the instance offset, got {min:?}"
        );
        let span: Vec<f32> = (0..3).map(|axis| max[axis] - min[axis]).collect();
        assert!(
            (span[0] - 32.0).abs() < 1e-3
                && (span[1] - 64.0).abs() < 1e-3
                && (span[2] - 32.0).abs() < 1e-3,
            "the definition's turned box spans 32×64×32, got {span:?}"
        );
    }

    /// Two instances of one definition each catalogue their own copy; a fixture
    /// definition's spliced Subtract child catalogues too (the walk never reads
    /// operations — the shader clips edges to where they crease the composed body).
    #[test]
    fn every_instance_and_spliced_cutter_catalogues_edges() {
        let host = box_tool([4, 4, 4], [0, 0, 0], CombineOp::Union, "Host");
        let cutter = box_tool([2, 2, 2], [1, 1, 1], CombineOp::Subtract, "Cutter");
        let mut first = Node::new("First", NodeContent::Instance(DefId(3)));
        first.transform = NodeTransform::from_blocks([0, 0, 0], DENSITY);
        let mut second = Node::new("Second", NodeContent::Instance(DefId(3)));
        second.transform = NodeTransform::from_blocks([10, 0, 0], DENSITY);
        let mut scene = Scene::from_nodes(vec![first, second]);
        scene.voxels_per_block = DENSITY;
        scene.add_definition(DefId(3), "Notched", [host, cutter]);
        scene.set_definition_fixture(DefId(3), true);
        scene.ensure_node_ids();

        let cel = AppCore::selected_body_cel(&scene, &[scene.roots[0], scene.roots[1]], DENSITY)
            .expect("both instances derive bodies");
        assert_eq!(
            cel.edge_segments.len(),
            96,
            "2 instances × (host 24 + spliced cutter 24) endpoints"
        );
        let recentre = cel.recentre.voxels();
        let xs: Vec<f32> = cel
            .edge_segments
            .iter()
            .map(|point| point[0] + recentre[0] as f32)
            .collect();
        assert!(
            xs.iter().any(|&x| x < 40.0) && xs.iter().any(|&x| x >= 80.0),
            "each instance's edges sit at its own placement"
        );
    }

    /// A sketch solid's edges arrive through the same leaf walk (the producer answers
    /// the catalogue itself): an L extrude catalogues 2 cap outlines (6 windows each)
    /// + 6 laterals = 36 segment endpoints, corner-anchored at the node's offset.
    #[test]
    fn sketch_solid_cel_edges_flow_through_the_leaf_walk() {
        use document::sketch::{PlaneAxis, Sketch, SketchPoint, SketchSolid};
        let profile = vec![
            SketchPoint::new(0, 0),
            SketchPoint::new(4, 0),
            SketchPoint::new(4, 2),
            SketchPoint::new(2, 2),
            SketchPoint::new(2, 4),
            SketchPoint::new(0, 4),
        ];
        let node = Node::new(
            "L",
            NodeContent::SketchTool {
                producer: SketchSolid::extrude(Sketch::new(PlaneAxis::Z, profile), 3),
                material: MaterialChoice::Stone,
            },
        );
        let mut scene = Scene::from_nodes(vec![node]);
        scene.voxels_per_block = DENSITY;
        scene.ensure_node_ids();

        let cel = AppCore::selected_body_cel(&scene, &[scene.roots[0]], DENSITY)
            .expect("sketch solid derives a body");
        assert_eq!(cel.edge_segments.len(), 36);
        let recentre = cel.recentre.voxels();
        for point in &cel.edge_segments {
            let z = point[2] + recentre[2] as f32;
            assert!(z == 0.0 || z == 3.0, "every edge point sits on a cap plane");
        }
    }

    /// Selecting the ROOT PART x-rays every boolean in the whole scene (the scene-wide
    /// master): two hosts each with their own cutter → two ghost bodies.
    #[test]
    fn root_part_selection_covers_every_boolean() {
        let mut scene = Scene::from_nodes(vec![
            box_tool([4, 4, 4], [0, 0, 0], CombineOp::Union, "Host A"),
            box_tool([2, 2, 2], [1, 1, 1], CombineOp::Subtract, "Cutter A"),
            box_tool([4, 4, 4], [20, 0, 0], CombineOp::Union, "Host B"),
            box_tool([2, 2, 2], [21, 1, 1], CombineOp::Subtract, "Cutter B"),
        ]);
        scene.voxels_per_block = DENSITY;
        scene.ensure_node_ids();
        let ghost = AppCore::boolean_operand_ghost(&scene, ROOT_NODE_ID, DENSITY)
            .expect("both cutters ghost");
        assert_eq!(ghost.bodies.len(), 2);
        assert!(ghost
            .bodies
            .iter()
            .all(|b| b.style == OperandGhostStyle::Subtract));
    }

    /// A cutter poking through the host's +X face: their surfaces cross on the
    /// rectangle perimeter at x = 32, y and z on the cutter's walls {8, 24}.
    fn notch_scene() -> Scene {
        let mut scene = Scene::from_nodes(vec![
            box_tool([4, 4, 4], [0, 0, 0], CombineOp::Union, "Host"),
            box_tool([2, 2, 2], [3, 1, 1], CombineOp::Subtract, "Cutter"),
        ]);
        scene.voxels_per_block = DENSITY;
        scene.ensure_node_ids();
        scene
    }

    /// The traced junction lands on BOTH surfaces: every endpoint sits on the host
    /// face x = 32 AND on a cutter wall, and all four walls of the notch mouth
    /// appear. Selecting either side of the pair traces the same junction.
    #[test]
    fn junction_curves_trace_the_notch_mouth() {
        let scene = notch_scene();
        for select in [0usize, 1] {
            let picked: std::collections::BTreeSet<NodeId> =
                [scene.roots[select]].into_iter().collect();
            let mut segments = Vec::new();
            collect_junction_segments_true_world(&scene, &picked, DENSITY, &mut segments);
            assert!(
                !segments.is_empty(),
                "the cutter's walls cross the host's +X face"
            );
            let mut walls_hit = [false; 4];
            for point in &segments {
                assert!(
                    (point[0] - 32.0).abs() < 0.05,
                    "junction lives on the host face x=32, got {point:?}"
                );
                let wall = [
                    (point[1] - 8.0).abs() < 0.05,
                    (point[1] - 24.0).abs() < 0.05,
                    (point[2] - 8.0).abs() < 0.05,
                    (point[2] - 24.0).abs() < 0.05,
                ];
                assert!(
                    wall.iter().any(|&on| on),
                    "junction lives on a cutter wall, got {point:?}"
                );
                for (hit, on) in walls_hit.iter_mut().zip(wall) {
                    *hit |= on;
                }
            }
            assert_eq!(
                walls_hit, [true; 4],
                "the full notch-mouth perimeter is traced (selected root {select})"
            );
        }
    }

    /// A pair with NEITHER side in the selection traces nothing — the junction
    /// derivation is selection-scoped, never scene-wide.
    #[test]
    fn junctions_skip_pairs_outside_the_selection() {
        let mut scene = Scene::from_nodes(vec![
            box_tool([4, 4, 4], [0, 0, 0], CombineOp::Union, "Host"),
            box_tool([2, 2, 2], [3, 1, 1], CombineOp::Subtract, "Cutter"),
            box_tool([2, 2, 2], [10, 0, 0], CombineOp::Union, "Bystander"),
        ]);
        scene.voxels_per_block = DENSITY;
        scene.ensure_node_ids();
        let picked: std::collections::BTreeSet<NodeId> = [scene.roots[2]].into_iter().collect();
        let mut segments = Vec::new();
        collect_junction_segments_true_world(&scene, &picked, DENSITY, &mut segments);
        assert!(
            segments.is_empty(),
            "the disjoint bystander shares no junction; the unselected host/cutter pair skips"
        );
    }

    /// The `--demo-subtract` corner-octant pair: the cutter is FLUSH with the body on
    /// three faces, and the flush patches must neither trace (tangency floor) nor
    /// flood the seed budget (the seed screen regression — the top face's L once
    /// vanished because flush cells exhausted `max_seeds` first). The junction is
    /// three open L curves, one per body face the cutter's walls cross.
    #[test]
    fn flush_corner_notch_traces_all_three_faces() {
        let density = 16u32;
        let body = SdfShape::from_blocks(ShapeKind::Box, [4, 4, 4], 1, density);
        let cutter = SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, density);
        let field_body = |p: glam::Vec3| {
            body.as_field()
                .unwrap()
                .signed_distance(p.to_array(), density)
        };
        let field_cutter = |p: glam::Vec3| {
            cutter
                .as_field()
                .unwrap()
                .signed_distance([p.x - 32.0, p.y - 32.0, p.z - 32.0], density)
        };
        let bracket_body = |cell_min: glam::Vec3, size: f32| -> (f32, f32) {
            let centre = cell_min + glam::Vec3::splat(size / 2.0);
            substrate::spatial::lipschitz_cell_bracket(
                field_body(centre),
                1.5,
                size * 3f32.sqrt() / 2.0,
            )
        };
        let bracket_cutter = |cell_min: glam::Vec3, size: f32| -> (f32, f32) {
            let centre = cell_min + glam::Vec3::splat(size / 2.0);
            substrate::spatial::lipschitz_cell_bracket(
                field_cutter(centre),
                1.5,
                size * 3f32.sqrt() / 2.0,
            )
        };
        let curves = substrate::spatial::trace_intersection_curves(
            &substrate::spatial::ImplicitSurfacePair {
                field_f: &field_body,
                field_g: &field_cutter,
                bracket_f: &bracket_body,
                bracket_g: &bracket_cutter,
            },
            glam::Vec3::new(31.0, 31.0, 31.0),
            glam::Vec3::new(65.0, 65.0, 65.0),
            &substrate::spatial::SurfaceIntersectionConfig::default(),
        )
        .curves;
        assert_eq!(curves.len(), 3, "one L per body face, no flush phantoms");
        for curve in &curves {
            assert!(!curve.closed, "each L is open (aborts at the body edges)");
            assert!(
                curve.points.len() > 100,
                "each L covers its two 32-voxel lines"
            );
            let flat_axis = (0..3)
                .find(|&axis| {
                    curve
                        .points
                        .iter()
                        .all(|point| (point[axis] - 64.0).abs() < 0.05)
                })
                .expect("each L lies flat on one body face at 64");
            for point in &curve.points {
                let on_wall = (0..3)
                    .filter(|&axis| axis != flat_axis)
                    .any(|axis| (point[axis] - 32.0).abs() < 0.05);
                assert!(on_wall, "every point sits on a cutter wall, got {point:?}");
            }
        }
    }

    /// The bore mouth traces as ONE closed circle — the corrected-point dedup
    /// regression: bracket-admitted seeds farther out than the consume radius must
    /// not re-trace the same curve (this pair once yielded it 10 times).
    #[test]
    fn bore_junction_traces_one_circle() {
        let density = 16u32;
        let body = SdfShape::from_blocks(ShapeKind::Box, [4, 4, 4], 1, density);
        let bore = SdfShape::from_blocks(ShapeKind::Cylinder, [2, 2, 5], 1, density);
        let field_body = |p: glam::Vec3| {
            body.as_field()
                .unwrap()
                .signed_distance(p.to_array(), density)
        };
        let field_bore = |p: glam::Vec3| {
            bore.as_field()
                .unwrap()
                .signed_distance([p.x - 16.0, p.y - 16.0, p.z - 16.0], density)
        };
        let bracket_body = |cell_min: glam::Vec3, size: f32| -> (f32, f32) {
            let centre = cell_min + glam::Vec3::splat(size / 2.0);
            substrate::spatial::lipschitz_cell_bracket(
                field_body(centre),
                1.5,
                size * 3f32.sqrt() / 2.0,
            )
        };
        let bracket_bore = |cell_min: glam::Vec3, size: f32| -> (f32, f32) {
            let centre = cell_min + glam::Vec3::splat(size / 2.0);
            substrate::spatial::lipschitz_cell_bracket(
                field_bore(centre),
                1.5,
                size * 3f32.sqrt() / 2.0,
            )
        };
        let curves = substrate::spatial::trace_intersection_curves(
            &substrate::spatial::ImplicitSurfacePair {
                field_f: &field_body,
                field_g: &field_bore,
                bracket_f: &bracket_body,
                bracket_g: &bracket_bore,
            },
            glam::Vec3::new(15.0, 15.0, 15.0),
            glam::Vec3::new(49.0, 49.0, 65.0),
            &substrate::spatial::SurfaceIntersectionConfig::default(),
        )
        .curves;
        assert_eq!(curves.len(), 1, "one bore mouth, traced once");
        let circle = &curves[0];
        assert!(circle.closed);
        assert!(circle.points.len() > 100, "the full 16-voxel-radius circle");
        for point in &circle.points {
            assert!(
                (point.z - 64.0).abs() < 0.05,
                "on the top face, got {point:?}"
            );
            let radius = ((point.x - 32.0).powi(2) + (point.y - 32.0).powi(2)).sqrt();
            assert!(
                (radius - 16.0).abs() < 0.05,
                "on the bore wall, got {point:?}"
            );
        }
    }
}
