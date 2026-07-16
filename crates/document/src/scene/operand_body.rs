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
//!
//! Issue #79 adds the PERSISTENT sibling of the same derivation:
//! [`Scene::shown_child_boolean_body_slices`] walks every subtree whose root carries the
//! per-node "Show child booleans" flag ([`Node::show_child_booleans`]) and collects the
//! standalone body slice of EVERY Subtract/Intersect operand inside it — the same slice
//! mechanics, keyed by a checkbox instead of the selection, and restricted to the
//! boolean masks (never the constructive Union bodies: those are already visible).

use super::*;

/// Whether `operation` is one of the boolean masks the issue #79 persistent ghost
/// shows — the operands that are invisible by success (a Subtract carves itself away,
/// an Intersect survives only as the kept overlap). Union bodies are already visible,
/// so the persistent ghost never tints them (unlike the #78 selection ghost).
fn operation_is_boolean_mask(operation: CombineOp) -> bool {
    matches!(operation, CombineOp::Subtract | CombineOp::Intersect)
}

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

    /// The persistent child-boolean ghost's body slices (issue #79): for every subtree
    /// covered by a node with "Show child booleans" checked, the `(operation, slice)`
    /// pair of EVERY visible Subtract/Intersect operand inside it — the checked node
    /// itself included when it is a boolean. Each slice is built by the same
    /// [`operand_body_slice`](Self::operand_body_slice) mechanics as the #78 selection
    /// ghost (absolute placement kept, root operation neutralised to `Union`).
    ///
    /// * **Nesting / dedupe** — ONE walk over the tree: a node is visited once per
    ///   placement path, so an outer checked node covers inner subtrees and a
    ///   redundantly-checked inner node can never emit its operands twice (a body
    ///   drawn twice would read as doubled ghost alpha — a style bug).
    /// * **Hidden subtrees** — contribute nothing (a hidden node stamps nothing into
    ///   the composition, so there is no invisible-by-success operand to reveal).
    /// * **Groups** — a boolean Group emits its sealed composed body AND the walk
    ///   descends: its internal cutters are boolean operands within the subtree too.
    /// * **Instances** — a sealed-definition Instance is a leaf (it emits when its own
    ///   operation is a boolean — the reusable cutter, issue #76; booleans INTERNAL to
    ///   the sealed definition spend themselves inside its finished body and are not
    ///   operands of this scope). A FIXTURE Instance (inert own operation, issue #77)
    ///   splices its definition children into the host fold, so the walk descends into
    ///   them under the instance's transform.
    /// * **The active selection is excluded** — the #78 selection ghost already draws
    ///   the ACTIVE node's body in the same style, so the persistent set skips it (the
    ///   dedupe rule above, across the two overlays). For an active fixture Instance
    ///   the whole splice is skipped: the selection ghost draws one body per spliced
    ///   child itself.
    pub fn shown_child_boolean_body_slices(&self) -> Vec<(CombineOp, Scene)> {
        let mut slices = Vec::new();
        self.collect_shown_boolean_operands(&self.roots, false, [0i64; 3], &mut slices);
        slices
    }

    /// The recursive walk behind
    /// [`shown_child_boolean_body_slices`](Self::shown_child_boolean_body_slices):
    /// `ancestors_shown` carries whether any
    /// ancestor (or spliced-through fixture instance) had the checkbox on;
    /// `ancestor_offset_voxels` the accumulated world offset of the ancestors (baked
    /// into each emitted slice root — ADR 0008 carried frames).
    fn collect_shown_boolean_operands(
        &self,
        siblings: &[NodeId],
        ancestors_shown: bool,
        ancestor_offset_voxels: [i64; 3],
        slices: &mut Vec<(CombineOp, Scene)>,
    ) {
        for &node_id in siblings {
            let Some(node) = self.arena.get(&node_id) else {
                continue;
            };
            if !node.visible {
                // A hidden subtree contributes no body to the composition — no ghost.
                continue;
            }
            let shown = ancestors_shown || node.show_child_booleans;
            // Cross-overlay dedupe: the #78 selection ghost already draws the ACTIVE
            // node's own body (same style for a boolean), so the persistent set skips
            // it — otherwise the two overlays would double the ghost alpha.
            let drawn_by_selection_ghost = self.active == Some(node.id);
            let emit_own_body = shown
                && operation_is_boolean_mask(node.operation)
                && !drawn_by_selection_ghost;
            let offset_under_node = [
                ancestor_offset_voxels[0] + node.transform.offset_voxels[0],
                ancestor_offset_voxels[1] + node.transform.offset_voxels[1],
                ancestor_offset_voxels[2] + node.transform.offset_voxels[2],
            ];
            match &node.content {
                NodeContent::Group(children) => {
                    if emit_own_body {
                        // A boolean Group's own operand body: its sealed composed
                        // (internally-carved) body, exactly like the #78 selection slice.
                        slices.push((
                            node.operation,
                            self.operand_body_slice(node.id, ancestor_offset_voxels),
                        ));
                    }
                    // Always descend: a Group's children are subtree operands whether
                    // or not the Group itself is a boolean or the active selection.
                    self.collect_shown_boolean_operands(
                        children,
                        shown,
                        offset_under_node,
                        slices,
                    );
                }
                NodeContent::Instance(def_id) => {
                    if self.def_by_id(*def_id).is_some_and(|def| def.fixture) {
                        // A fixture instance's own operation is inert (issue #77): its
                        // definition children splice into the host fold, so descend
                        // into them under the instance's transform. An ACTIVE fixture
                        // instance is skipped wholesale — the selection ghost derives
                        // one body per spliced child itself (the dedupe rule; the
                        // simplification accepted for booleans buried DEEPER than the
                        // spliced children, which that ghost shows carved-in-place).
                        if !drawn_by_selection_ghost {
                            if let Some(def) = self.def_by_id(*def_id) {
                                self.collect_shown_boolean_operands(
                                    &def.children,
                                    shown,
                                    offset_under_node,
                                    slices,
                                );
                            }
                        }
                    } else if emit_own_body {
                        // A sealed-definition Instance is a leaf operand: it folds the
                        // definition's FINISHED body under its own operation (issue
                        // #76 — the reusable cutter), so that whole body is the ghost.
                        slices.push((
                            node.operation,
                            self.operand_body_slice(node.id, ancestor_offset_voxels),
                        ));
                    }
                }
                _ => {
                    // Leaf producers (Tool / SketchTool / VoxelBody).
                    if emit_own_body {
                        slices.push((
                            node.operation,
                            self.operand_body_slice(node.id, ancestor_offset_voxels),
                        ));
                    }
                }
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
