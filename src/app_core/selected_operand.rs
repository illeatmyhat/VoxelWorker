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

/// Everything the display's [`SelectedBodyCelRenderer`] rebuild needs (ADR 0032 —
/// viewport selection feedback): the selected nodes' standalone bodies plus the COMPOSED
/// scene's frame (ADR 0008, same contract as [`SelectedOperandGhost`]).
///
/// [`SelectedBodyCelRenderer`]: display::mesh::SelectedBodyCelRenderer
pub struct SelectedBodyCel {
    /// One body per surviving selected node (selection-roots filtered: a node whose
    /// ancestor is also selected contributes nothing — the ancestor's composed body
    /// already covers it, and drawing both would double the cel alpha).
    pub bodies: Vec<display::mesh::SelectedBodyChunks>,
    /// The composed scene's voxel extent (the shader's corner-anchoring scalar).
    pub grid_dimensions: [u32; 3],
    /// The composed scene's resolve recentre — the render frame the cel meshes into.
    pub recentre: RecentreVoxels,
    /// The document density the bodies were evaluated at.
    pub density: u32,
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
        }
        if bodies.is_empty() {
            return None;
        }
        Some(SelectedBodyCel {
            bodies,
            grid_dimensions: scene.placed_region_dimensions(density),
            recentre: scene.recentre_voxels_for_resolve(density),
            density,
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
