//! Structural edits over the assembly graph: add / remove / group / ungroup, the
//! definition + instance workflow (ADR 0001 step 4), fixtures (ADR 0017 Decision 4),
//! and the subtree capture/reinsert primitives the undo path relies on.

use voxel_core::core_geom::MaterialChoice;

use crate::voxel::{GeometryParams, SdfShape};

use super::*;

impl Scene {
    /// Look up a reusable definition by its [`DefId`] (ADR 0001 step 4). Returns
    /// `None` when no definition carries that id — an `Instance` pointing at a
    /// missing definition resolves to nothing.
    pub fn def_by_id(&self, id: DefId) -> Option<&AssemblyDef> {
        self.definitions.iter().find(|def| def.id == id)
    }

    /// Set the [`fixture`](AssemblyDef::fixture) flag of the definition `id` (ADR
    /// 0017 Decision 4, issue #77 — the `SetDefinitionFixture` intent's field
    /// write). Returns whether a definition carried the id (a dangling id is a
    /// no-op, like every other field write to a missing target).
    pub fn set_definition_fixture(&mut self, id: DefId, fixture: bool) -> bool {
        match self.definitions.iter_mut().find(|def| def.id == id) {
            Some(def) => {
                def.fixture = fixture;
                true
            }
            None => false,
        }
    }

    /// Whether `node`'s own [`CombineOp`] is **inert** (ADR 0017 Decision 4, issue
    /// #77): true exactly for an `Instance` of a FIXTURE definition, whose children
    /// splice into the hosting scope's fold under their OWN operations — the
    /// instance's operation is never consulted by the resolver, so the inspector
    /// hides the Operation selector (no dead control). Every other node kind (and an
    /// instance of a sealed definition, or of a missing one) folds under its own
    /// operation as usual.
    pub fn node_operation_is_inert(&self, node: &Node) -> bool {
        match &node.content {
            NodeContent::Instance(def_id) => {
                self.def_by_id(*def_id).is_some_and(|def| def.fixture)
            }
            _ => false,
        }
    }

    /// Append `node` to the TOP-LEVEL list and make it the active selection.
    /// Returns its top-level index.
    ///
    /// ADR 0003 Phase B3: selection is keyed by [`NodeId`], so the appended node is
    /// minted a stable id here ([`mint_node_id`](Self::mint_node_id)) before
    /// `active` is pointed at it — a freshly-added node is selectable by identity
    /// immediately, surviving any later reorder.
    pub fn add_node(&mut self, node: Node) -> usize {
        // The arena insert (mint id, stamp it, store) is exactly `insert_subtree`.
        let id = self.insert_subtree(node);
        self.roots.push(id);
        let index = self.roots.len() - 1;
        self.active = Some(id);
        index
    }

    /// Append `node` as a child of the Group identified by `group_id` and select
    /// it. Returns `true` if the target was a Group and the node was added. A no-op
    /// (returns `false`) when the id does not resolve to a Group.
    pub fn add_child_to_group(&mut self, group_id: NodeId, mut node: Node) -> bool {
        // ADR 0003 Phase B4: the op targets a NodeId; resolve it to the positional
        // path the internal storage still needs (the positional bridge survives
        // until B5). A stale id → no-op (mirrors the old out-of-range path bail).
        let Some(group_path) = self.path_of(group_id) else {
            return false;
        };
        let group_path = &group_path;
        // Bail before minting if the target is not a Group, so a no-op neither adds
        // a node nor burns a counter value.
        match self.node_at_path(group_path).map(|node| &node.content) {
            Some(NodeContent::Group(_)) => {}
            _ => return false,
        }
        // Mint the child's stable id (ADR 0003 Phase B3) so selection can point at
        // it by identity; minting BEFORE the mutable group borrow releases the
        // `&mut next_node_id` borrow so it can't overlap the arena borrow (B5).
        let id = self.mint_node_id();
        node.id = id;
        // Insert the child into the arena (its `Node` lives there now), then splice
        // its id onto the Group's spine. The arena insert is independent of the group
        // borrow, so the two `&mut arena` accesses are sequential, not overlapping.
        self.arena.insert(id, node);
        let Some(group_node) = self.node_at_path_mut(group_path) else {
            // Unreachable (we checked it is a Group above), but keep the arena clean.
            self.arena.remove(&id);
            return false;
        };
        let NodeContent::Group(children) = &mut group_node.content else {
            self.arena.remove(&id);
            return false;
        };
        children.push(id);
        self.active = Some(id);
        true
    }

    /// Remove the node identified by `target_id` (top-level or a Group child),
    /// keeping the `active` selection sensible: after a removal the selection falls
    /// back to the removed node's parent (so a Group's last child deletion selects
    /// the Group), or to a surviving top-level node, or `None` when the scene
    /// empties. A stale id (no longer in the tree) is ignored.
    /// Detach `id` from its parent spine and purge its WHOLE subtree from the arena,
    /// WITHOUT touching `active` (ADR 0003 Phase B4/B5). Resolves the id to its
    /// positional path, splices it out of its parent spine (top-level `roots` or a
    /// Group's `Vec<NodeId>`), then drops the removed node + every descendant (a
    /// shared-borrow DFS into a `Vec` so no arena borrow is held during removal —
    /// leaving any behind would orphan it). Returns the removed node's former slot
    /// `(parent_indices, last_index)` so [`remove_node`](Self::remove_node) can
    /// re-derive a selection there; a stale id / already-detached slot → `None`.
    fn detach_and_purge_subtree(&mut self, id: NodeId) -> Option<(Vec<usize>, usize)> {
        let path = self.path_of(id)?;
        let (&last_index, parent_indices) = path.indices.split_last()?;
        let parent_path = NodePath::from_indices(parent_indices.to_vec());
        let removed_id = match self.siblings_mut(&parent_path) {
            Some(spine) if last_index < spine.len() => spine.remove(last_index),
            _ => return None,
        };
        let mut to_remove = Vec::new();
        self.collect_subtree_ids(removed_id, &mut to_remove);
        for descendant in to_remove {
            self.arena.remove(&descendant);
        }
        Some((parent_indices.to_vec(), last_index))
    }

    pub fn remove_node(&mut self, target_id: NodeId) {
        let Some((parent_indices, last_index)) = self.detach_and_purge_subtree(target_id) else {
            return;
        };
        // Re-derive a valid selection. Prefer the sibling now occupying the removed
        // slot (a Group, or the scene root → a surviving top-level node); fall back
        // to the parent Group, then None when empty. ADR 0003 Phase B3: the fallback
        // yields a NodePath, which we resolve to the surviving node's stable id.
        self.active = self
            .fallback_selection_after_remove(&parent_indices, last_index)
            .and_then(|path| self.id_at_path(&path));
    }

    /// Clone the detached subtree rooted at `root_id` (the node + every descendant
    /// through [`NodeContent::Group`] spines) into a `Vec<Node>`, root first, in the
    /// SAME DFS order as [`collect_subtree_ids`](Self::collect_subtree_ids) (ADR 0003
    /// Phase C C2 undo support). Captured BEFORE a `remove_node` so the inverse can
    /// re-insert every `Node` under its ORIGINAL id. Definition bodies are NOT followed
    /// (an `Instance` references a def stored separately).
    pub fn clone_subtree_nodes(&self, root_id: NodeId) -> Vec<Node> {
        let mut ids = Vec::new();
        self.collect_subtree_ids(root_id, &mut ids);
        ids.into_iter()
            .filter_map(|id| self.arena.get(&id).cloned())
            .collect()
    }

    /// Remove the node `id` (and its whole subtree) from the arena + splice its id out
    /// of its parent spine, WITHOUT re-deriving the `active` selection (ADR 0003 Phase
    /// C C2). The undo path restores selection itself from the command's captured
    /// `selection_before`, so unlike [`remove_node`](Self::remove_node) this must not
    /// touch `active`. Used to reverse a single-node mint (`Inverse::RemoveAdded`). A
    /// stale id is a no-op.
    pub fn remove_node_exact(&mut self, id: NodeId) {
        // Same detach + purge as `remove_node`, but drop the returned slot — the undo
        // path restores selection from the command's captured `selection_before`.
        self.detach_and_purge_subtree(id);
    }

    /// Reverse [`group_active`](Self::group_active) (ADR 0003 Phase C C2): the fresh
    /// `group` node took `target`'s spine slot and adopted `target` as its sole child.
    /// Put `target`'s id back in the slot `group` occupies and drop `group` from the
    /// arena. Does NOT touch `active` (the undo path restores it). A no-op if `group`
    /// no longer resolves.
    pub fn ungroup_node(&mut self, group: NodeId, target: NodeId) {
        let Some(path) = self.path_of(group) else {
            return;
        };
        let Some((&last_index, parent_indices)) = path.indices.split_last() else {
            return;
        };
        let parent_path = NodePath::from_indices(parent_indices.to_vec());
        if let Some(spine) = self.siblings_mut(&parent_path) {
            if last_index < spine.len() {
                spine[last_index] = target;
            }
        }
        self.arena.remove(&group);
    }

    /// Re-insert a detached subtree captured by [`clone_subtree_nodes`](Self::clone_subtree_nodes)
    /// (ADR 0003 Phase C C2): store every `Node` back in the arena under its ORIGINAL
    /// id (safe — the monotonic counter never reuses an id), then splice the root id
    /// (`nodes[0]`) into `parent`'s spine (`None` = top-level `roots`) at `index`.
    /// Reverses a [`remove_node`](Self::remove_node). Does NOT touch `active`.
    pub fn reinsert_subtree(&mut self, parent: Option<NodeId>, index: usize, nodes: &[Node]) {
        let Some(root) = nodes.first() else {
            return;
        };
        let root_id = root.id;
        for node in nodes {
            self.arena.insert(node.id, node.clone());
        }
        match parent {
            None => {
                let clamped = index.min(self.roots.len());
                self.roots.insert(clamped, root_id);
            }
            Some(parent_id) => {
                if let Some(parent_node) = self.arena.get_mut(&parent_id) {
                    if let NodeContent::Group(children) = &mut parent_node.content {
                        let clamped = index.min(children.len());
                        children.insert(clamped, root_id);
                    }
                }
            }
        }
    }

    /// Collect `root_id` and every descendant id (through [`NodeContent::Group`]
    /// spines) into `out`, via a shared-borrow DFS over the arena (ADR 0003 Phase B5).
    /// Used by [`remove_node`](Self::remove_node) to gather a detached subtree's ids
    /// up front so the arena entries can be dropped without holding a borrow across
    /// the removal. Definition bodies are NOT followed (an `Instance` references a
    /// def stored separately; deleting an instance never deletes the shared body).
    fn collect_subtree_ids(&self, root_id: NodeId, out: &mut Vec<NodeId>) {
        out.push(root_id);
        // Snapshot the Group's spine length, then re-fetch each child id by position
        // for the recursive descent — so no `&self.arena.get` borrow is held across
        // the recursive `&self` call (and no per-group spine clone is allocated).
        let child_count = match self.arena.get(&root_id).map(|node| &node.content) {
            Some(NodeContent::Group(children)) => children.len(),
            _ => return,
        };
        for child_index in 0..child_count {
            let Some(NodeContent::Group(children)) =
                self.arena.get(&root_id).map(|node| &node.content)
            else {
                return;
            };
            let child_id = children[child_index];
            self.collect_subtree_ids(child_id, out);
        }
    }

    /// Wrap the active node in a new [`NodeContent::Group`] in place (ADR 0001
    /// step 4 authoring): the active node becomes the sole child of a fresh Group
    /// that takes its slot among its siblings. The Group inherits an identity
    /// transform (the child keeps its own offset, so the composite is unchanged),
    /// and the wrapped child becomes the new active selection. Returns the new
    /// Group's [`NodeId`] on success; `None` when there is no active node.
    ///
    /// Grouping a node that is itself a Group simply nests it one level deeper —
    /// the recursion handles arbitrary depth.
    pub fn group_active(&mut self) -> Option<NodeId> {
        // ADR 0003 Phase B3: selection is a NodeId; resolve it to the child's
        // current position to do the positional wrap. The child keeps its id (and
        // thus stays selected by identity); only the new Group needs a fresh id.
        let path = self.active_path()?;
        let (&index, parent_indices) = path.indices.split_last()?;
        let group_id = self.mint_node_id();
        let parent_path = NodePath::from_indices(parent_indices.to_vec());
        // B5: the spine carries child IDS. Swap the child's id at `index` for the new
        // Group's id (capturing the child id), so the child `Node` never leaves the
        // arena (only its id moves down one level into the Group's spine) — it keeps
        // its stable identity and stays the active selection.
        let child_id = {
            let siblings = self.siblings_mut(&parent_path)?;
            if index >= siblings.len() {
                return None;
            }
            let child_id = siblings.remove(index);
            siblings.insert(index, group_id);
            child_id
        };
        // The new Group owns the wrapped child by id; store it in the arena. Named
        // "Part" (ADR 0018 Decision 1: the composition container is user-facing "Part").
        let mut group = Node::new("Part", NodeContent::Group(vec![child_id]));
        group.id = group_id;
        self.arena.insert(group_id, group);
        // ADR 0003 Phase B4: return the new Group's stable id (minted above) rather
        // than its positional path.
        Some(group_id)
    }

    /// The smallest unused [`DefId`] (one past the current max, or `DefId(1)` when
    /// there are no definitions — id 0 is reserved/unused for clarity).
    pub fn next_def_id(&self) -> DefId {
        let max = self
            .definitions
            .iter()
            .map(|def| def.id.0)
            .max()
            .unwrap_or(0);
        DefId(max + 1)
    }

    /// Turn the active node into a reusable [`AssemblyDef`] and REPLACE it with an
    /// [`NodeContent::Instance`] of that definition (ADR 0001 step 4: "make
    /// definition from this Group/node"). The active node's content moves into the
    /// new definition's children (a Group's children become the def body; a single
    /// leaf becomes a one-node def); the active node keeps its transform but its
    /// content becomes an `Instance(new_def_id)`. Returns the new [`DefId`] on
    /// success; `None` when there is no active node.
    ///
    /// After this, the active selection stays on the (now-instance) node, and the
    /// definition can be placed again via [`add_instance`](Self::add_instance) —
    /// the village workflow: one stored body, many placements.
    pub fn make_definition_from_active(&mut self, name: impl Into<String>) -> Option<DefId> {
        // ADR 0018 Decision 2: the root part is never a definition target (a definition
        // of the whole scene is out of scope) — reject it before touching anything.
        if self.active == Some(ROOT_NODE_ID) {
            return None;
        }
        let def_id = self.next_def_id();
        // ADR 0003 Phase B3: resolve the selected NodeId to its current position.
        // The node keeps its id while only its content becomes an Instance, so the
        // selection stays valid (still the same node by identity) with no re-point.
        let active_id = self.active?;
        // The edit is by id (B5); the `node_by_id_mut` lookup below already bails
        // (`?`) on a stale selection, so no separate presence guard is needed.
        // The definition body, as a spine of arena ids:
        // * a Group DONATES its child id spine (`mem::take` empties the Group's
        //   `Vec<NodeId>`); the child `Node`s STAY in the arena — the def now owns
        //   them by reference, none are orphaned (B5).
        // * any other content becomes a single-node body: a fresh "Body" node
        //   wrapping a clone of the content, inserted into the arena under a new id.
        // First mutate the node's content to the Instance and extract either the
        // donated child-id spine (Group) or a fresh "Body" node to insert (leaf),
        // dropping the `&mut node` arena borrow before any further `&mut self` use.
        enum Body {
            Donated(Vec<NodeId>),
            // Boxed: a `Node` is an order of magnitude larger than the donated id
            // spine, and an unboxed variant would size this temporary to the larger
            // of the two on every extraction (clippy::large_enum_variant).
            Leaf(Box<Node>),
        }
        let body = {
            let node = self.node_by_id_mut(active_id)?;
            let body = match &mut node.content {
                NodeContent::Group(children) => Body::Donated(std::mem::take(children)),
                other => Body::Leaf(Box::new(Node::new("Body", other.clone()))),
            };
            node.content = NodeContent::Instance(def_id);
            body
        };
        let child_ids: Vec<NodeId> = match body {
            Body::Donated(ids) => ids,
            Body::Leaf(node) => vec![self.insert_subtree(*node)],
        };
        self.definitions.push(AssemblyDef {
            id: def_id,
            name: name.into(),
            children: child_ids,
            // A freshly-extracted part is SEALED (ADR 0017 Decision 3) — splicing
            // is a deliberate per-definition opt-in (Decision 4, issue #77).
            fixture: false,
        });
        Some(def_id)
    }

    /// Place another [`NodeContent::Instance`] of the definition `def_id` as a new
    /// top-level node (ADR 0001 step 4: "Add Instance"). The instance is named
    /// after the definition and gets a default offset that nudges it clear of
    /// earlier instances of the same def (so a freshly-added village house does not
    /// land exactly on top of the previous one). Selects the new node. Returns its
    /// [`NodeId`], or `None` when no definition carries `def_id`.
    pub fn add_instance(&mut self, def_id: DefId) -> Option<NodeId> {
        let def = self.def_by_id(def_id)?;
        let name = format!("{} instance", def.name);
        // Nudge each new instance of this def along +X so it does not overlap the
        // previous one. Count existing top-level instances of this def for the step.
        let existing = self
            .roots
            .iter()
            .filter_map(|id| self.arena.get(id))
            .filter(|node| matches!(node.content, NodeContent::Instance(id) if id == def_id))
            .count();
        let mut node = Node::new(name, NodeContent::Instance(def_id));
        // Block-granular auto-spacing → canonical voxels at the document density.
        let spacing_blocks = (existing as i64 + 1) * DEFAULT_INSTANCE_SPACING_BLOCKS as i64;
        node.transform = NodeTransform::from_blocks([spacing_blocks, 0, 0], self.voxels_per_block);
        let index = self.add_node(node);
        // ADR 0003 Phase B4: return the appended node's stable id rather than its
        // positional path. `add_node` minted its id and pointed `active` at it, and
        // `id_at_path` reads it back from the slot it now occupies.
        self.id_at_path(&NodePath::root_index(index))
    }

    /// Build the one-node Tool scene that reproduces today's single-shape
    /// behaviour from the panel's [`GeometryParams`] plus the active
    /// [`MaterialChoice`]. The node is a [`NodeContent::Tool`] wrapping the SDF
    /// shape, carrying `material` as its single material.
    ///
    /// Step 2 removed the `debug_clouds: bool` selector — "Clouds" is now an
    /// Add-a-VoxelBody action in the node list ([`VoxelBody::DebugClouds`]), not a mode of
    /// the geometry. So this constructor only ever builds a Tool; the back-compat
    /// config load (a single persisted geometry) routes through here.
    pub fn from_geometry(geometry: GeometryParams, material: MaterialChoice) -> Self {
        // Capture the density before `from_geometry` consumes the params (it is no
        // longer `Copy` — it owns an optional boxed retained-size expression).
        let voxels_per_block = geometry.voxels_per_block;
        let mut scene = Self::single_node(Node::new(
            "Shape",
            NodeContent::Tool {
                shape: SdfShape::from_geometry(geometry),
                material,
            },
        ));
        // Density is document-level (ADR 0003 §3f(0)): carry the UI control value
        // onto the scene, not the shape.
        scene.voxels_per_block = voxels_per_block;
        scene
    }
}
