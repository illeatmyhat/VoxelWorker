//! The boolean-operand body slices of the selected subtree (ADR 0018 Decision 6 — the
//! "Show booleans" viewer mode's document-side derivation).
//!
//! In Show-booleans mode the SELECTED subtree x-rays its boolean operands: every
//! Subtract/Intersect operand body within the selected node's subtree (the node itself
//! included when it is a boolean) renders as an operation-coded ghost over the finished
//! scene. Selecting the root part ([`ROOT_NODE_ID`]) covers the whole scene; no / hidden
//! / stale selection yields no ghost.
//!
//! Each operand body is derived as a **scene slice**: a clone of the document re-rooted
//! on that operand, keeping its ABSOLUTE placement (the ancestor Group offsets are baked
//! into the slice root's transform — ADR 0008: the frame is carried, never re-derived),
//! so the slice's covering chunks land on the composed scene's chunk lattice and the
//! ghost mesh sits voxel-exact on the operand's place. The subtree's INTERNAL booleans
//! still spend themselves (a covered boolean Group emits its own sealed, internally-carved
//! body), while the emitted operand's own combine operation is neutralised to `Union` so
//! its body composes constructively when resolved standalone (a Subtract root at fold
//! start would yield nothing).
//!
//! Resolving each slice is bounded by that operand's own covering chunk range, never the
//! whole scene's — the derivation cost scales with the ghosted bodies.

use super::*;

/// Whether `operation` is one of the boolean masks the ghost shows — the operands that
/// are invisible by success (a Subtract carves itself away, an Intersect survives only
/// as the kept overlap). Union bodies are already visible, so they never ghost.
fn operation_is_boolean_mask(operation: CombineOp) -> bool {
    matches!(operation, CombineOp::Subtract | CombineOp::Intersect)
}

impl Scene {
    /// The boolean-operand body slices for the ACTIVE selection's subtree (ADR 0018
    /// Decision 6 — "Show booleans" mode): `(operation, slice)` pairs where `operation`
    /// is the boolean role the body folds under (Subtract/Intersect — picking the ghost
    /// style) and `slice` is a scene whose sole root is that body, placed absolutely.
    ///
    /// The walk is rooted at the selection and unconditional within its subtree:
    ///
    /// * **Root part** (`active == ROOT_NODE_ID`) — every boolean operand in the WHOLE
    ///   scene (the scene-wide master).
    /// * **A regular node** — every boolean operand inside that node's subtree, the node
    ///   itself included when it is a boolean. A non-boolean leaf selection is degenerate
    ///   but consistent (it scopes the walk to that ingredient — nothing to reveal).
    /// * **Covered boolean Group** — emits its sealed composed (internally-carved) body
    ///   AND the walk descends (its internal cutters are subtree operands too).
    /// * **Sealed-definition Instance** — a leaf operand (the reusable cutter, issue #76:
    ///   it folds the definition's FINISHED body under its own operation).
    /// * **Fixture Instance** (inert own operation, issue #77) — its definition children
    ///   splice into the host fold, so the walk descends into them under the instance's
    ///   transform.
    /// * **No / hidden / stale selection** — empty (no ghost). A hidden node stamps
    ///   nothing into the composition, so there is no invisible-by-success body to reveal.
    ///
    /// Each node is visited once per placement path (no body is ever emitted twice — a
    /// body drawn twice would read as doubled ghost alpha). The slice keeps the
    /// document's `definitions` (an Instance root still expands) and density.
    pub fn boolean_operand_body_slices(&self) -> Vec<(CombineOp, Scene)> {
        let Some(active) = self.active else {
            return Vec::new();
        };
        let mut slices = Vec::new();
        if active == ROOT_NODE_ID {
            // The root part: every boolean operand in the whole scene.
            self.collect_boolean_operands(&self.roots, [0i64; 3], &mut slices);
            return slices;
        }

        // A regular node: descend the id-spine to the selection, accumulating the
        // ANCESTOR world voxel offset (the same descent `node_subtree_extent_voxels`
        // performs), then walk that node's subtree rooted at it.
        let Some(path) = self.active_path() else {
            return Vec::new();
        };
        let mut siblings: &[NodeId] = &self.roots;
        let mut ancestor_offset_voxels = [0i64; 3];
        let mut target_id: Option<NodeId> = None;
        for (depth, &index) in path.indices.iter().enumerate() {
            let Some(&child_id) = siblings.get(index) else {
                return Vec::new();
            };
            let Some(node) = self.arena.get(&child_id) else {
                return Vec::new();
            };
            if depth + 1 == path.indices.len() {
                target_id = Some(child_id);
            } else if let NodeContent::Group(children) = &node.content {
                for (accumulated, offset) in ancestor_offset_voxels
                    .iter_mut()
                    .zip(node.transform.offset_voxels)
                {
                    *accumulated += offset;
                }
                siblings = children;
            } else {
                return Vec::new();
            }
        }
        let Some(target_id) = target_id else {
            return Vec::new();
        };
        self.collect_boolean_operands(
            std::slice::from_ref(&target_id),
            ancestor_offset_voxels,
            &mut slices,
        );
        slices
    }

    /// The recursive walk behind [`boolean_operand_body_slices`](Self::boolean_operand_body_slices):
    /// collect the `(operation, slice)` of every enabled Subtract/Intersect operand in
    /// `siblings` and their descendants. `ancestor_offset_voxels` is the accumulated world
    /// offset of the siblings' parent (baked into each emitted slice root — ADR 0008
    /// carried frames).
    fn collect_boolean_operands(
        &self,
        siblings: &[NodeId],
        ancestor_offset_voxels: [i64; 3],
        slices: &mut Vec<(CombineOp, Scene)>,
    ) {
        for &node_id in siblings {
            let Some(node) = self.arena.get(&node_id) else {
                continue;
            };
            if !node.enabled {
                // A disabled subtree contributes no body to the composition — no ghost.
                continue;
            }
            let emit_own_body = operation_is_boolean_mask(node.operation);
            let offset_under_node = [
                ancestor_offset_voxels[0] + node.transform.offset_voxels[0],
                ancestor_offset_voxels[1] + node.transform.offset_voxels[1],
                ancestor_offset_voxels[2] + node.transform.offset_voxels[2],
            ];
            // A fixture instance's own operation is INERT (issue #77): its definition
            // children splice into the host fold, so it never emits its own body even
            // under a boolean operation — it only contributes via descent below.
            let is_fixture_instance = matches!(&node.content, NodeContent::Instance(def_id)
                if self.def_by_id(*def_id).is_some_and(|def| def.fixture));
            // The one operand-body emit: a boolean Group's sealed composed body, or a
            // sealed-definition Instance's finished body (issue #76, the reusable
            // cutter), or a leaf producer's body — all the SAME push, gated identically.
            if emit_own_body && !is_fixture_instance {
                slices.push((
                    node.operation,
                    self.operand_body_slice(node.id, ancestor_offset_voxels),
                ));
            }
            // Descend into subtree operands: a Group's children (whether or not the
            // Group itself is a boolean), or a fixture instance's spliced definition
            // children under the instance's transform. A sealed Instance / leaf has no
            // subtree operands to descend into.
            match &node.content {
                NodeContent::Group(children) => {
                    self.collect_boolean_operands(children, offset_under_node, slices);
                }
                NodeContent::Instance(def_id) if is_fixture_instance => {
                    if let Some(def) = self.def_by_id(*def_id) {
                        self.collect_boolean_operands(&def.children, offset_under_node, slices);
                    }
                }
                _ => {}
            }
        }
    }

    /// Build the standalone slice for the body rooted at `root_id`: the document cloned,
    /// re-rooted on that node, with `ancestor_offset_voxels` baked into the root's
    /// transform (absolute placement preserved) and the root's operation neutralised to
    /// `Union` (its body must compose constructively when resolved alone — a Subtract
    /// root at fold start would yield nothing).
    fn operand_body_slice(&self, root_id: NodeId, ancestor_offset_voxels: [i64; 3]) -> Scene {
        let mut slice = self.clone();
        slice.roots = vec![root_id];
        slice.active = None;
        if let Some(root) = slice.arena.get_mut(&root_id) {
            for (offset, ancestor) in root
                .transform
                .offset_voxels
                .iter_mut()
                .zip(ancestor_offset_voxels)
            {
                *offset += ancestor;
            }
            root.operation = CombineOp::Union;
        }
        slice
    }
}
