//! The selected node's standalone operand body (issue #78 — the selected-operand ghost's
//! document-side derivation).
//!
//! The ghost visualises the active node's OWN body — its subtree resolved standalone,
//! under the composition semantics of ADR 0017 ("Composition", `docs/architecture/
//! 01-document.md`): the subtree's INTERNAL booleans still spend themselves (a Group's
//! cutter carves the Group's own body), but the node's own combine operation is
//! neutralised (a Subtract cutter standalone would carve nothing — its body IS what we
//! want to see). This module derives that body as a **scene slice**: a clone of the
//! document re-rooted on the selection, keeping the node's ABSOLUTE placement (the
//! ancestor Group offsets are baked into the slice root's transform — ADR 0008: the frame
//! is carried, never re-derived), so the slice's covering chunks land on the composed
//! scene's chunk lattice and the ghost mesh sits voxel-exact on the node's place.
//!
//! Resolving a slice is bounded by the SELECTED SUBTREE's extent (its covering chunk
//! range), never the whole scene's — the derivation cost scales with the selected body.

use super::*;

impl Scene {
    /// The standalone body slices of the ACTIVE selection: `(operation, slice)` pairs
    /// where `operation` is the role the body folds under (picking the ghost style) and
    /// `slice` is a scene whose sole root is that body, placed absolutely.
    ///
    /// * **Leaf / Group / sealed-definition Instance** — ONE slice: the node's subtree,
    ///   its own operation neutralised to `Union` so the body composes constructively
    ///   standalone (internal booleans inside the subtree still apply — the honest "own
    ///   body" of a carved Group is the carved body).
    /// * **Fixture-definition Instance** (its own operation is inert, ADR 0017 Decision
    ///   4 / [`node_operation_is_inert`](Self::node_operation_is_inert)) — one slice PER
    ///   spliced child, each under the CHILD's own operation and the instance's
    ///   transform. This is the honest reading of "what IS this instance": a fixture is
    ///   its children spliced into the host fold, each carrying its own role (the window
    ///   ghosts its opening red and its frame as the subtle constructive tint).
    /// * **No / hidden / stale selection** — empty (no ghost).
    ///
    /// The slice keeps the document's `definitions` (an Instance root still expands) and
    /// its density; the slice root's retained offset *measurements* are NOT re-targeted
    /// (the canonical `offset_voxels` always wins for geometry, and a slice never
    /// re-evaluates a density change).
    pub fn active_operand_body_slices(&self) -> Vec<(CombineOp, Scene)> {
        let Some(path) = self.active_path() else {
            return Vec::new();
        };
        // Walk the id-spine down to the selection, accumulating the ANCESTOR world
        // voxel offset (the same descent `node_subtree_extent_voxels` performs).
        let mut siblings: &[NodeId] = &self.roots;
        let mut ancestor_offset_voxels = [0i64; 3];
        let mut target: Option<&Node> = None;
        for (depth, &index) in path.indices.iter().enumerate() {
            let Some(&child_id) = siblings.get(index) else {
                return Vec::new();
            };
            let Some(node) = self.arena.get(&child_id) else {
                return Vec::new();
            };
            if depth + 1 == path.indices.len() {
                target = Some(node);
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
        let Some(target) = target else {
            return Vec::new();
        };
        if !target.visible {
            // A hidden node contributes no body to the composition — no ghost.
            return Vec::new();
        }

        // A fixture instance's own operation is inert: ghost each SPLICED child under
        // its own operation, placed under the instance's transform.
        if let NodeContent::Instance(def_id) = &target.content {
            if let Some(def) = self.def_by_id(*def_id) {
                if def.fixture {
                    let instance_offset_voxels = [
                        ancestor_offset_voxels[0] + target.transform.offset_voxels[0],
                        ancestor_offset_voxels[1] + target.transform.offset_voxels[1],
                        ancestor_offset_voxels[2] + target.transform.offset_voxels[2],
                    ];
                    return def
                        .children
                        .iter()
                        .filter_map(|&child_id| {
                            let child = self.arena.get(&child_id)?;
                            if !child.visible {
                                return None;
                            }
                            Some((
                                child.operation,
                                self.operand_body_slice(child_id, instance_offset_voxels),
                            ))
                        })
                        .collect();
                }
            }
        }

        vec![(
            target.operation,
            self.operand_body_slice(target.id, ancestor_offset_voxels),
        )]
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
