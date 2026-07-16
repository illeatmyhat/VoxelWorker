//! Selected-operand ghost derivation (issue #78) — selection → ghost bodies + frame.
//!
//! The app_core half of the seam: read the ACTIVE selection's standalone body slices
//! from the document ([`Scene::active_operand_body_slices`]), evaluate each through the
//! two-layer evaluator (bounded by the SELECTED SUBTREE's covering chunks — never a
//! whole-scene resolve, and never a dense grid), and hand the display layer plain
//! meshes-to-be + styles ([`display::mesh::SelectedOperandGhostBody`]). Display renders,
//! app_core derives, the document stays pure (ADR 0016).
//!
//! Re-derived only on selection/geometry change (the shell + `shot` call this at those
//! seams), never per frame.
//!
//! Issue #79 adds [`AppCore::child_boolean_ghost`]: the PERSISTENT sibling — the same
//! evaluation over [`Scene::shown_child_boolean_body_slices`] (every Subtract/Intersect
//! operand inside the "Show child booleans"-checked subtrees) instead of the active
//! selection. Same bodies-plus-frame product, same renderer, re-derived at the same
//! seams (plus the `SetShowChildBooleans` toggle's `operand_ghosts_changed` effect).

use display::mesh::SelectedOperandGhostBody;
use display::renderer::OperandGhostStyle;
use document::scene::{CombineOp, Scene};
use evaluation::two_layer_store::TwoLayerStore;
use voxel_core::voxel::RecentreVoxels;

use super::AppCore;

/// Everything the display's [`SelectedOperandGhostRenderer`] rebuild needs: the ghost
/// bodies plus the COMPOSED scene's frame (ADR 0008 — the slice chunks are in absolute
/// composite coords, so meshing them against the composed recentre lands the ghost
/// voxel-exact on the selected node's place in the render frame).
///
/// [`SelectedOperandGhostRenderer`]: display::mesh::SelectedOperandGhostRenderer
pub struct SelectedOperandGhost {
    /// One body per operand: a plain selection is one; a fixture-instance selection is
    /// one per spliced child (each under its own operation's style).
    pub bodies: Vec<SelectedOperandGhostBody>,
    /// The composed scene's voxel extent (the shader's corner-anchoring scalar).
    pub grid_dimensions: [u32; 3],
    /// The composed scene's resolve recentre — the render frame the ghost meshes into.
    pub recentre: RecentreVoxels,
    /// The document density the bodies were evaluated at.
    pub density: u32,
}

/// Map the document's combine operation onto display's ghost-style vocabulary (the
/// display layer never reads `CombineOp` — ADR 0016 layering).
fn operand_ghost_style_for(operation: CombineOp) -> OperandGhostStyle {
    match operation {
        CombineOp::Union => OperandGhostStyle::Union,
        CombineOp::Subtract => OperandGhostStyle::Subtract,
        CombineOp::Intersect => OperandGhostStyle::Intersect,
    }
}

impl AppCore {
    /// Derive the selected-operand ghost for the active selection (issue #78), or `None`
    /// when nothing is selected / the selection is hidden / its body is empty.
    ///
    /// Cost bound: each slice is evaluated over ITS OWN covering chunk range (the
    /// selected subtree's extent) via the stateless two-layer evaluator — a selection
    /// change never re-resolves the whole scene, and no dense whole-region grid is ever
    /// assembled (the user law).
    pub fn selected_operand_ghost(scene: &Scene, density: u32) -> Option<SelectedOperandGhost> {
        evaluate_operand_ghost_slices(scene, scene.active_operand_body_slices(), density)
    }

    /// Derive the persistent child-boolean ghost (issue #79): the standalone body of
    /// EVERY Subtract/Intersect operand inside the "Show child booleans"-checked
    /// subtrees ([`Scene::shown_child_boolean_body_slices`]), or `None` when no flag is
    /// set / no covered boolean has geometry. The active selection's own body is
    /// already excluded document-side (the #78 selection ghost draws it — the
    /// cross-overlay dedupe rule), so drawing both overlays never doubles an alpha.
    ///
    /// Same cost bound as the selection ghost: each slice evaluates over ITS OWN
    /// covering chunk range — toggling the checkbox never re-resolves the whole scene
    /// (the #79 acceptance bound), and no dense grid is ever assembled (the user law).
    pub fn child_boolean_ghost(scene: &Scene, density: u32) -> Option<SelectedOperandGhost> {
        evaluate_operand_ghost_slices(scene, scene.shown_child_boolean_body_slices(), density)
    }
}

/// The shared evaluation half of both ghost derivations (#78 selection / #79
/// persistent): run each `(operation, slice)` through the stateless two-layer
/// evaluator — bounded by the SLICE's covering chunks, never a whole-scene resolve,
/// never a dense grid — and package the surviving bodies with the COMPOSED scene's
/// frame (ADR 0008: the slices are in absolute composite coords, so meshing against
/// the composed recentre lands each ghost voxel-exact).
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
    use document::scene::{Node, NodeContent, NodeTransform};
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
            NodeContent::Tool { shape, material: MaterialChoice::Stone },
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

    /// Issue #78 acceptance: no ghost with an empty selection.
    #[test]
    fn empty_selection_derives_no_ghost() {
        let mut scene = host_and_cutter_scene();
        scene.active = None;
        assert!(AppCore::selected_operand_ghost(&scene, DENSITY).is_none());
    }

    /// The style mapping: a Union selection ghosts as the SUBTLE union style (never
    /// red/amber), a Subtract cutter as red, an Intersect mask as amber.
    #[test]
    fn styles_follow_the_selected_operation() {
        let mut scene = host_and_cutter_scene();
        scene.active = Some(scene.roots[0]);
        let ghost = AppCore::selected_operand_ghost(&scene, DENSITY).expect("union body ghosts");
        assert_eq!(ghost.bodies.len(), 1);
        assert_eq!(ghost.bodies[0].style, OperandGhostStyle::Union);

        scene.active = Some(scene.roots[1]);
        let ghost = AppCore::selected_operand_ghost(&scene, DENSITY).expect("cutter ghosts");
        assert_eq!(ghost.bodies[0].style, OperandGhostStyle::Subtract);
    }

    /// Re-derivation on selection change resolves ONLY the selected subtree's covering
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
        scene.active = Some(scene.roots[1]);

        let ghost = AppCore::selected_operand_ghost(&scene, DENSITY).expect("cutter ghosts");
        assert_eq!(
            ghost.bodies[0].chunks.len(),
            1,
            "the 2-block cutter covers ONE chunk; the far host's extent is never evaluated"
        );
        // The frame handed to display is the COMPOSED scene's (ADR 0008), so the ghost
        // mesh lands in the same render frame as the solid.
        assert_eq!(ghost.grid_dimensions, scene.placed_region_dimensions(DENSITY));
        assert_eq!(
            ghost.recentre.voxels(),
            scene.recentre_voxels_for_resolve(DENSITY).voxels()
        );
    }

    /// A buried cutter's ghost body is the cutter's OWN full body (the two-layer chunks
    /// carry its geometry even though the composed scene swallows it entirely).
    #[test]
    fn buried_cutter_still_derives_its_body() {
        let mut scene = host_and_cutter_scene();
        scene.active = Some(scene.roots[1]);
        let ghost = AppCore::selected_operand_ghost(&scene, DENSITY).expect("cutter ghosts");
        let stored: u64 = ghost.bodies[0]
            .chunks
            .iter()
            .map(|(_, chunk)| chunk.stored_voxel_count())
            .sum();
        assert!(stored > 0, "the fully-buried cutter's own body must not be empty");
    }

    // ---- Issue #79: the persistent child-boolean ghost derivation ----

    /// Flag off (the default) → no persistent ghost; flag on → the covered boolean
    /// operands ghost in their operation styles (never the Union body).
    #[test]
    fn child_boolean_ghost_derives_only_when_the_flag_is_set() {
        let mut scene = host_and_cutter_scene();
        scene.active = None;
        assert!(AppCore::child_boolean_ghost(&scene, DENSITY).is_none());

        // Check the HOST's flag: the flag covers the checked node's subtree only (the
        // sibling cutter is outside it), so still nothing…
        let host_id = scene.roots[0];
        scene.node_by_id_mut(host_id).expect("host resolves").show_child_booleans = true;
        assert!(AppCore::child_boolean_ghost(&scene, DENSITY).is_none());

        // …while checking the CUTTER itself ghosts it (the node itself is included
        // when it is a boolean).
        let cutter_id = scene.roots[1];
        scene.node_by_id_mut(cutter_id).expect("cutter resolves").show_child_booleans = true;
        let ghost = AppCore::child_boolean_ghost(&scene, DENSITY).expect("the cutter ghosts");
        assert_eq!(ghost.bodies.len(), 1);
        assert_eq!(ghost.bodies[0].style, OperandGhostStyle::Subtract);
    }

    /// The #79 no-re-resolve bound at the derivation seam: re-deriving the persistent
    /// ghost evaluates ONLY the covered boolean bodies' covering chunks — a toggle on
    /// a small cutter never touches the far host's extent — and hands display the
    /// COMPOSED scene's frame (ADR 0008), exactly like the selection ghost.
    #[test]
    fn child_boolean_derivation_is_bounded_by_the_covered_bodies() {
        let mut scene = Scene::from_nodes(vec![
            box_tool([4, 4, 4], [40, 0, 0], CombineOp::Union, "Far host"),
            box_tool([2, 2, 2], [0, 0, 0], CombineOp::Subtract, "Cutter"),
        ]);
        scene.voxels_per_block = DENSITY;
        scene.ensure_node_ids();
        scene.active = None;
        let cutter_id = scene.roots[1];
        scene.node_by_id_mut(cutter_id).expect("cutter resolves").show_child_booleans = true;

        let ghost = AppCore::child_boolean_ghost(&scene, DENSITY).expect("cutter ghosts");
        assert_eq!(
            ghost.bodies[0].chunks.len(),
            1,
            "the 2-block cutter covers ONE chunk; the far host's extent is never evaluated"
        );
        assert_eq!(ghost.grid_dimensions, scene.placed_region_dimensions(DENSITY));
        assert_eq!(
            ghost.recentre.voxels(),
            scene.recentre_voxels_for_resolve(DENSITY).voxels()
        );
    }

    /// The cross-overlay dedupe: with the flagged cutter ALSO selected, the selection
    /// ghost owns its body and the persistent set is empty — one body, one overlay,
    /// never a doubled alpha.
    #[test]
    fn selected_body_moves_to_the_selection_ghost_not_both() {
        let mut scene = host_and_cutter_scene();
        let cutter_id = scene.roots[1];
        scene.node_by_id_mut(cutter_id).expect("cutter resolves").show_child_booleans = true;
        scene.active = Some(cutter_id);
        assert!(AppCore::selected_operand_ghost(&scene, DENSITY).is_some());
        assert!(AppCore::child_boolean_ghost(&scene, DENSITY).is_none());
    }
}
