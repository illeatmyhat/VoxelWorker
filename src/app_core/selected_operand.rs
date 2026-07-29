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
use document::scene::{CombineOp, NodeContent, NodeId, Scene};
use evaluation::two_layer_store::TwoLayerStore;
use substrate::spatial::{LeafPlacement, ProducerLocalVoxelPoint};
use voxel_core::voxel::{RecentreVoxels, ShapeKind};

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
}

/// Segments per tessellated rim ellipse. Fixed (not screen-adaptive) so the polyline
/// is world-stable under orbit — the whole point of the analytic edges.
const EDGE_CIRCLE_SEGMENTS: u32 = 96;

/// The analytic feature-edge catalogue of one shape, as polylines in the producer's
/// LOCAL `[0, full]` voxel frame. Only edges the authored geometry actually has: a box
/// its 12 edges, a cylinder its 2 rim ellipses (axis along Z), a tube those plus the 2
/// inner rim ellipses; a sphere and a torus are smooth everywhere and catalogue none.
fn shape_edge_polylines(kind: ShapeKind, grid: [u32; 3], wall_voxels: f32) -> Vec<Vec<[f32; 3]>> {
    let full = [grid[0] as f32, grid[1] as f32, grid[2] as f32];
    let half = [full[0] / 2.0, full[1] / 2.0, full[2] / 2.0];
    let rim_pair = |semi_x: f32, semi_y: f32| -> Vec<Vec<[f32; 3]>> {
        [0.0, full[2]]
            .into_iter()
            .map(|z| {
                (0..=EDGE_CIRCLE_SEGMENTS)
                    .map(|step| {
                        let angle =
                            step as f32 / EDGE_CIRCLE_SEGMENTS as f32 * std::f32::consts::TAU;
                        [
                            half[0] + semi_x * angle.cos(),
                            half[1] + semi_y * angle.sin(),
                            z,
                        ]
                    })
                    .collect()
            })
            .collect()
    };
    match kind {
        ShapeKind::Box => {
            let corner = |x: f32, y: f32, z: f32| [x * full[0], y * full[1], z * full[2]];
            let mut polylines = Vec::with_capacity(12);
            for a in [0.0, 1.0] {
                for b in [0.0, 1.0] {
                    polylines.push(vec![corner(0.0, a, b), corner(1.0, a, b)]);
                    polylines.push(vec![corner(a, 0.0, b), corner(a, 1.0, b)]);
                    polylines.push(vec![corner(a, b, 0.0), corner(a, b, 1.0)]);
                }
            }
            polylines
        }
        ShapeKind::Cylinder => rim_pair(half[0], half[1]),
        ShapeKind::Tube => {
            let mut polylines = rim_pair(half[0], half[1]);
            // The inner wall is the SDF's inner elliptical cylinder (semi-axes reduced
            // by the wall). A wall consuming the whole cross-section leaves no hole —
            // and no inner rims.
            let inner_x = half[0] - wall_voxels;
            let inner_y = half[1] - wall_voxels;
            if inner_x > 0.01 && inner_y > 0.01 {
                polylines.extend(rim_pair(inner_x, inner_y));
            }
            polylines
        }
        ShapeKind::Sphere | ShapeKind::Torus => Vec::new(),
    }
}

/// Walk a `node_body_slice` and emit every Tool leaf's edge catalogue as segment
/// endpoint pairs in TRUE-WORLD voxels (the slice root is absolutely placed; Group
/// descent accumulates child offsets the same way the resolve walk does). Leaves the
/// catalogue can't describe yet — sketch solids, voxel bodies, instances — emit
/// nothing.
fn collect_edge_segments_true_world(slice: &Scene, density: u32, out: &mut Vec<[f32; 3]>) {
    fn visit(
        slice: &Scene,
        node_id: NodeId,
        offset_voxels: [i64; 3],
        offset_local: [f32; 3],
        density: u32,
        out: &mut Vec<[f32; 3]>,
    ) {
        let Some(node) = slice.arena.get(&node_id) else {
            return;
        };
        if !node.enabled {
            return;
        }
        let offset_voxels: [i64; 3] =
            std::array::from_fn(|axis| offset_voxels[axis] + node.transform.offset_voxels[axis]);
        let offset_local: [f32; 3] = std::array::from_fn(|axis| {
            offset_local[axis] + node.transform.offset_local_voxels[axis]
        });
        match &node.content {
            NodeContent::Group(children) => {
                for &child in children {
                    visit(slice, child, offset_voxels, offset_local, density, out);
                }
            }
            NodeContent::Tool { shape, .. } => {
                let grid = shape.grid_dimensions(density);
                let full = glam::Vec3::new(grid[0] as f32, grid[1] as f32, grid[2] as f32);
                let placement = LeafPlacement::from_origin_and_local(
                    node.transform.rotation(),
                    full,
                    offset_voxels,
                    offset_local,
                );
                let wall_voxels = (shape.wall_blocks * density) as f32;
                for polyline in shape_edge_polylines(shape.kind, grid, wall_voxels) {
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
            _ => {}
        }
    }
    for &root in &slice.roots {
        visit(slice, root, [0; 3], [0.0; 3], density, out);
    }
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
    use document::scene::{Node, NodeContent, NodeTransform, ROOT_NODE_ID};
    use document::voxel::SdfShape;
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
        let edges = shape_edge_polylines(ShapeKind::Box, [32, 32, 32], 8.0);
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
        assert!(shape_edge_polylines(ShapeKind::Sphere, [32, 32, 32], 8.0).is_empty());
        assert!(shape_edge_polylines(ShapeKind::Torus, [32, 32, 16], 8.0).is_empty());
    }

    /// A tube catalogues 4 rim ellipses (outer + inner × top + bottom, axis along Z);
    /// a wall consuming the whole cross-section closes the hole and drops the inner
    /// pair. A cylinder is the outer pair alone.
    #[test]
    fn tube_catalogues_four_rims_until_the_wall_closes_the_hole() {
        let rims = shape_edge_polylines(ShapeKind::Tube, [64, 64, 32], 8.0);
        assert_eq!(rims.len(), 4);
        for rim in &rims {
            let z = rim[0][2];
            assert!(z == 0.0 || z == 32.0, "rims sit on the tube's caps");
            assert!(rim.iter().all(|point| point[2] == z), "each rim is planar");
        }
        // Angle 0 of each rim: centre (32, 32) + radius along +X — outer 32, inner 24.
        let radii: Vec<f32> = rims.iter().map(|rim| rim[0][0] - 32.0).collect();
        assert_eq!(radii, vec![32.0, 32.0, 24.0, 24.0]);

        let walled_shut = shape_edge_polylines(ShapeKind::Tube, [32, 32, 32], 16.0);
        assert_eq!(walled_shut.len(), 2, "no hole, no inner rims");
        assert_eq!(
            shape_edge_polylines(ShapeKind::Cylinder, [64, 64, 32], 0.0).len(),
            2
        );
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
}
