//! Path/id navigation and selection over the id-keyed arena: resolving a
//! [`NodePath`] or [`NodeId`] to a node, the inverse [`Scene::path_of`], the tree-row
//! projection, active-selection accessors, and the id-minting / spine-repointing
//! helpers the load path and edit ops share.

use super::*;

impl Scene {
    /// Mint a stable [`NodeId`] for every still-unassigned node (ADR 0003 Phase B).
    /// Walks the top-level nodes, every [`NodeContent::Group`]'s children, and every
    /// definition's nodes; any node carrying the `NodeId(0)` sentinel gets a fresh id
    /// from [`next_node_id`](Self::next_node_id). The counter is first advanced past
    /// any ids ALREADY present (a loaded scene may carry minted ids) so new ids never
    /// collide. **Idempotent:** a second call mints nothing (every node already has a
    /// non-zero id). Called on the load/normalization path alongside
    /// [`ensure_origin_point`](Self::ensure_origin_point).
    pub fn ensure_node_ids(&mut self) {
        // Advance the counter past any ids already present, so freshly-minted ids
        // never collide with ones a loaded scene already carries. The arena keys ARE
        // every node's id (BTreeMap → ascending order, so the scan is stable), so a
        // single pass over the arena values + the definition spines covers it. Note:
        // a node carrying the `NodeId(0)` sentinel is stored UNDER key 0 in the arena
        // (a fresh-by-value insert always mints, but a deserialized arena could carry
        // a single 0-keyed node), so the `max` ignores 0 naturally.
        let mut max_existing = 0u64;
        for id in self.arena.keys() {
            max_existing = max_existing.max(id.0);
        }
        // `.max(2)`: id `1` is reserved for the root part ([`ROOT_NODE_ID`]), so the
        // first real id a load-path mint hands out is `2`.
        self.next_node_id = self.next_node_id.max(max_existing + 1).max(2);

        // Re-key any still-unassigned node out of the `NodeId(0)` sentinel slot. With
        // the arena keyed by id, minting a fresh id means MOVING the arena entry AND
        // repointing the one spine slot (`roots`, a Group's children, or a definition's
        // children) that referenced it — otherwise the spine keeps pointing at slot 0
        // while the node lives elsewhere, silently orphaning it on load (it would never
        // render, list, or select). At most one node can sit under key 0 (BTreeMap keys
        // are unique). In practice every arena/def node is minted at insert time
        // (`insert_subtree`), so this is a safety net for a deserialized scene that
        // carries a `NodeId(0)` node.
        if self.arena.contains_key(&NodeId(0)) {
            let fresh = NodeId(self.next_node_id);
            self.next_node_id += 1;
            // Repoint the spine FIRST (while the node is still at key 0), then move the
            // arena entry. Mutating the `Vec<NodeId>` spines never borrows another arena
            // node, so no nested-borrow dance is needed.
            let repointed = self.repoint_spine_id(NodeId(0), fresh);
            debug_assert!(
                repointed,
                "a NodeId(0) arena node must be referenced by some spine slot",
            );
            if let Some(mut node) = self.arena.remove(&NodeId(0)) {
                node.id = fresh;
                self.arena.insert(fresh, node);
            }
        }
    }

    /// Replace every spine reference to `old` with `new` across the top-level
    /// [`roots`](Self::roots), every [`NodeContent::Group`]'s children, and every
    /// definition's children. Returns whether any slot was repointed. Used when
    /// re-keying a node in the arena (its id is its key, so the references that name
    /// it must move with it). Touches only the `Vec<NodeId>` spines — it never looks
    /// up another arena node, so it borrows the arena mutably without nesting.
    fn repoint_spine_id(&mut self, old: NodeId, new: NodeId) -> bool {
        let mut repointed = false;
        for slot in self.roots.iter_mut() {
            if *slot == old {
                *slot = new;
                repointed = true;
            }
        }
        for node in self.arena.values_mut() {
            if let NodeContent::Group(children) = &mut node.content {
                for slot in children.iter_mut() {
                    if *slot == old {
                        *slot = new;
                        repointed = true;
                    }
                }
            }
        }
        for definition in self.definitions.iter_mut() {
            for slot in definition.children.iter_mut() {
                if *slot == old {
                    *slot = new;
                    repointed = true;
                }
            }
        }
        repointed
    }

    /// The node at `path`, walking from `nodes` down through Group
    /// children. `None` when any index along the path is out of range or the path
    /// tries to descend through a non-Group (a Tool / VoxelBody / Instance has no
    /// addressable children).
    pub fn node_at_path(&self, path: &NodePath) -> Option<&Node> {
        // Walk the id-spine (`roots`, then each Group's `Vec<NodeId>`) for ORDER,
        // fetching each node's content from the arena. ADR 0003 Phase B5.
        let mut siblings: &[NodeId] = &self.roots;
        let mut found: Option<&Node> = None;
        for (depth, &index) in path.indices.iter().enumerate() {
            let &child_id = siblings.get(index)?;
            let node = self.arena.get(&child_id)?;
            let is_last = depth + 1 == path.indices.len();
            if is_last {
                found = Some(node);
            } else if let NodeContent::Group(children) = &node.content {
                siblings = children;
            } else {
                return None;
            }
        }
        found
    }

    /// The node at `path`, mutably (the inspector edits through this). ADR 0003
    /// Phase B5: resolve the path to a single [`NodeId`] over the id-spine (a shared
    /// walk), then take ONE mutable arena borrow at the end — so the descent never
    /// holds an aliasing `&mut` into the arena.
    pub fn node_at_path_mut(&mut self, path: &NodePath) -> Option<&mut Node> {
        let id = self.id_at_path(path)?;
        self.arena.get_mut(&id)
    }

    /// The [`NodeId`] of the node at `path` — the top-level-tree inverse of
    /// [`path_of`](Self::path_of) — or `None` if the path doesn't resolve (ADR 0003
    /// Phase B2). A convenience bridge while selection/commands migrate off
    /// [`NodePath`] onto [`NodeId`].
    pub fn id_at_path(&self, path: &NodePath) -> Option<NodeId> {
        self.node_at_path(path).map(|node| node.id)
    }

    /// The node with the given [`NodeId`] in the **top-level assembly tree**
    /// (top-level nodes + [`NodeContent::Group`] children — the same scope
    /// [`NodePath`] addresses), or `None` (ADR 0003 Phase B2). `NodeId(0)` (the
    /// unassigned sentinel) never matches. O(n) DFS; Phase B5 swaps the storage for
    /// an arena so this becomes a direct lookup.
    pub fn node_by_id(&self, id: NodeId) -> Option<&Node> {
        // ADR 0003 Phase B5: the arena IS keyed by NodeId, so this is a direct
        // lookup (was an O(n) DFS). The `NodeId(0)` unassigned sentinel never matches.
        if id == NodeId(0) {
            return None;
        }
        // ADR 0018 Decision 2: the root part lives on `self.root` (a field, not the
        // arena), so resolve its reserved id there — this is what makes it selectable
        // (`active_node`) and inspectable like any other node.
        if id == ROOT_NODE_ID {
            return Some(&self.root);
        }
        self.arena.get(&id)
    }

    /// The node with the given [`NodeId`], mutably (ADR 0003 Phase B2). Same scope +
    /// caveats as [`node_by_id`](Self::node_by_id).
    pub fn node_by_id_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        // ADR 0003 Phase B5: direct id-keyed arena lookup.
        if id == NodeId(0) {
            return None;
        }
        // ADR 0018 Decision 2: the root part is a field, not an arena entry — its
        // reserved id edits `self.root` (e.g. a rename via `SetName`). Its children
        // are `self.roots`, never mutated through this handle.
        if id == ROOT_NODE_ID {
            return Some(&mut self.root);
        }
        self.arena.get_mut(&id)
    }

    /// Set the `enabled` flag of the node identified by `id` (ADR 0003 Phase B4),
    /// returning whether the id resolved to a node. A NodeId-typed edit op so the
    /// panel's checkbox can mutate by identity rather than by path. Because the flag
    /// gates participation rather than display, flipping it changes the composed body
    /// and the caller must re-resolve.
    pub fn set_node_enabled(&mut self, id: NodeId, enabled: bool) -> bool {
        match self.node_by_id_mut(id) {
            Some(node) => {
                node.enabled = enabled;
                true
            }
            None => false,
        }
    }

    /// The [`NodePath`] addressing the node with the given [`NodeId`] in the
    /// top-level assembly tree, or `None` (ADR 0003 Phase B2). The inverse of
    /// [`id_at_path`](Self::id_at_path): `path_of(id_at_path(path)) == Some(path)`
    /// for every path that resolves. While `NodePath` is still the identity of
    /// record, this lets callers hold a stable [`NodeId`] and recover its current
    /// position on demand.
    pub fn path_of(&self, id: NodeId) -> Option<NodePath> {
        // ADR 0003 Phase B5: walk the id-spine (`roots`, then each Group's spine) for
        // ORDER, fetching content from the arena — the canonical render-time NodePath
        // projection. The arena is get-only here.
        fn search(scene: &Scene, spine: &[NodeId], id: NodeId, prefix: &mut Vec<usize>) -> bool {
            for (index, &child_id) in spine.iter().enumerate() {
                prefix.push(index);
                if child_id == id {
                    return true;
                }
                if let Some(NodeContent::Group(children)) =
                    scene.arena.get(&child_id).map(|node| &node.content)
                {
                    if search(scene, children, id, prefix) {
                        return true;
                    }
                }
                prefix.pop();
            }
            false
        }
        if id == NodeId(0) {
            return None;
        }
        let mut prefix = Vec::new();
        search(self, &self.roots, id, &mut prefix).then(|| NodePath::from_indices(prefix))
    }

    /// Flatten the top-level assembly into a depth-first list of `(path, id, depth)`
    /// rows for the tree UI (ADR 0001 step 4): every top-level node, and — for a
    /// [`NodeContent::Group`] — its children recursively at increasing depth. The
    /// rows are in display order (a parent immediately precedes its children).
    /// `Instance` nodes are leaves here (their definition's body is stored
    /// separately and rendered in the Definitions list, not inlined into the tree).
    ///
    /// ADR 0003 Phase B4: each row also carries the node's stable [`NodeId`] so the
    /// panel can feed the now-NodeId-typed select/delete/visibility ops directly,
    /// without a `path → id` round-trip; the `NodePath` stays for depth/path display.
    pub fn tree_rows(&self) -> Vec<(NodePath, NodeId, usize)> {
        let mut rows = Vec::new();
        // ADR 0018 Decision 2: the root part is the TOP ROW (depth 0), addressed by
        // the empty `NodePath` (it is not in the `roots` spine — `node_at_path` on the
        // empty path returns `None`, so geometry consumers like the grid batch skip it
        // harmlessly). Its children — the top-level nodes — indent one level beneath it.
        rows.push((NodePath::from_indices(Vec::new()), ROOT_NODE_ID, 0));
        collect_tree_rows(self, &self.roots, &mut Vec::new(), 1, &mut rows);
        rows
    }

    /// The active node, if any. ADR 0003 Phase B3: resolves the selected
    /// [`NodeId`] via [`node_by_id`](Self::node_by_id) (a stale id → `None`).
    pub fn active_node(&self) -> Option<&Node> {
        self.active.and_then(|id| self.node_by_id(id))
    }

    // `active_node_mut` was DELETED 2026-07-18 with zero callers. Its doc claimed "the
    // inspector edits through this", which was never true in this form: an inspector edit is
    // an Intent carrying its TARGET id, applied via `node_by_id_mut(target)` (see
    // `app_core::intent`), so the edit path never consults `active`. Reading the active node
    // is still a real need (`active_node`); mutating THROUGH the selection is not, and would
    // in fact be the wrong shape — it would let an edit silently retarget when the selection
    // moves. Do not reintroduce it; take the id.

    /// The [`NodePath`] currently addressing the active node, or `None` when nothing
    /// is selected (or the selected [`NodeId`] no longer resolves). ADR 0003 Phase
    /// B3: a positional bridge for the few call sites + tests that still reason in
    /// paths, now that [`active`](Self::active) stores an id.
    pub fn active_path(&self) -> Option<NodePath> {
        self.active.and_then(|id| self.path_of(id))
    }

    /// Mint the next fresh [`NodeId`] from the document counter (ADR 0003 Phase B3),
    /// advancing it past the value handed out. Matches the
    /// [`ensure_node_ids`](Self::ensure_node_ids) convention: ids start at `1`
    /// (`0` is the unassigned sentinel). Used by the `add_*` edit ops so a new node
    /// carries a stable id the moment it joins the tree.
    pub(super) fn mint_node_id(&mut self) -> NodeId {
        // `.max(2)`: `1` is the reserved root-part id ([`ROOT_NODE_ID`]); user nodes
        // mint from `2` upward so one can never collide with the root.
        self.next_node_id = self.next_node_id.max(2);
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        id
    }

    /// The parent of the node `id` in the top-level assembly tree, and its index in
    /// that parent's spine (ADR 0003 Phase C C2 undo support): `(Some(parent_id),
    /// index)` for a Group child, `(None, index)` for a top-level node. `None` when the
    /// id does not resolve. Used to CAPTURE a node's slot before a structural edit so
    /// the inverse can splice it back at the same place.
    pub fn parent_and_index_of(&self, id: NodeId) -> Option<(Option<NodeId>, usize)> {
        let path = self.path_of(id)?;
        let (&last_index, parent_indices) = path.indices.split_last()?;
        if parent_indices.is_empty() {
            return Some((None, last_index));
        }
        // The parent is the node the parent-prefix path resolves to (always a Group,
        // since a non-Group has no addressable children).
        let parent_path = NodePath::from_indices(parent_indices.to_vec());
        let parent_id = self.id_at_path(&parent_path)?;
        Some((Some(parent_id), last_index))
    }

    /// The mutable id-spine addressed by `parent_path` (the empty path → the
    /// top-level [`roots`](Self::roots); otherwise the [`Vec<NodeId>`] of the Group
    /// the path resolves to). `None` when the path does not resolve to a Group.
    /// ADR 0003 Phase B5: returns the SPINE of child ids, not the child `Node`s.
    pub(super) fn siblings_mut(&mut self, parent_path: &NodePath) -> Option<&mut Vec<NodeId>> {
        if parent_path.indices.is_empty() {
            return Some(&mut self.roots);
        }
        match self.node_at_path_mut(parent_path) {
            Some(node) => match &mut node.content {
                NodeContent::Group(children) => Some(children),
                _ => None,
            },
            None => None,
        }
    }

    /// Choose a valid `active` path after removing the child at `removed_index`
    /// from the sibling list at `parent_indices`.
    pub(super) fn fallback_selection_after_remove(
        &self,
        parent_indices: &[usize],
        removed_index: usize,
    ) -> Option<NodePath> {
        let parent_path = NodePath::from_indices(parent_indices.to_vec());
        let sibling_count = if parent_indices.is_empty() {
            self.roots.len()
        } else {
            match self.node_at_path(&parent_path).map(|n| &n.content) {
                Some(NodeContent::Group(children)) => children.len(),
                _ => 0,
            }
        };
        if sibling_count > 0 {
            // Select the sibling now occupying the removed slot (clamped to last).
            let new_index = removed_index.min(sibling_count - 1);
            let mut indices = parent_indices.to_vec();
            indices.push(new_index);
            Some(NodePath::from_indices(indices))
        } else if parent_indices.is_empty() {
            // The whole scene emptied.
            None
        } else {
            // A Group lost its last child — select the (now empty) Group itself.
            Some(parent_path)
        }
    }

    /// Test helper (ADR 0003 Phase B5): the top-level node at positional `index`, via
    /// the [`roots`](Self::roots) spine + arena. Replaces the old `scene.nodes[index]`
    /// positional read now that storage is id-keyed.
    #[cfg(any(test, feature = "test-support"))]
    pub fn root_node(&self, index: usize) -> &Node {
        let id = self.roots[index];
        &self.arena[&id]
    }

    /// Test helper (ADR 0003 Phase B5): the top-level node at positional `index`,
    /// mutably. Replaces the old `scene.nodes[index]` positional `&mut`.
    #[cfg(any(test, feature = "test-support"))]
    pub fn root_node_mut(&mut self, index: usize) -> &mut Node {
        let id = self.roots[index];
        self.arena.get_mut(&id).expect("root id present in arena")
    }
}

/// Depth-first worker for [`Scene::tree_rows`]: append `(path, depth)` for each
/// node in `nodes`, descending into Group children (a Group's children follow it
/// at `depth + 1`). `prefix` is the path of the assembly that owns `nodes`.
fn collect_tree_rows(
    scene: &Scene,
    spine: &[NodeId],
    prefix: &mut Vec<usize>,
    depth: usize,
    rows: &mut Vec<(NodePath, NodeId, usize)>,
) {
    // Iterate the id-spine for ORDER, fetching content from the arena (ADR 0003 B5).
    for (index, &child_id) in spine.iter().enumerate() {
        prefix.push(index);
        rows.push((NodePath::from_indices(prefix.clone()), child_id, depth));
        if let Some(NodeContent::Group(children)) =
            scene.arena.get(&child_id).map(|node| &node.content)
        {
            collect_tree_rows(scene, children, prefix, depth + 1, rows);
        }
        prefix.pop();
    }
}
