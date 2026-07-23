//! Scene construction, arena insertion, and reference [`Point`] management: the
//! terse by-value constructors ([`Scene::from_nodes`], [`Scene::single_node`]), the
//! [`NodeBuilder`] flatten-into-arena path, definition registration, and the Origin
//! Point synthesis.

use super::*;

impl Scene {
    /// A scene with a single node — the shape every one-node call site builds. The
    /// lone node is the active selection.
    ///
    /// ADR 0003 Phase B3: selection is keyed by [`NodeId`], so the lone node is
    /// minted a stable id here ([`ensure_node_ids`](Self::ensure_node_ids)) and
    /// `active` is set to that id — the scene is born already-normalised, so the
    /// selection resolves immediately without a separate load-path mint.
    pub fn single_node(node: Node) -> Self {
        let mut scene = Self::from_nodes(vec![node]);
        scene.active = scene.roots.first().copied();
        scene
    }

    /// Build a scene from a list of top-level [`Node`]s (ADR 0003 Phase B5), inserting
    /// each (and its `Group` descendants) into the [`arena`](Self::arena) under a
    /// freshly-minted [`NodeId`] and recording the top-level ids as the
    /// [`roots`](Self::roots) spine in order. The terse constructor the demo builders
    /// and test fixtures use so they keep building `Node` trees by value while the
    /// storage underneath is the id-keyed arena. `active` is left `None` (callers set
    /// it). Equivalent in effect to the old `Scene { nodes, .. }` + `ensure_node_ids`.
    pub fn from_nodes<I, B>(nodes: I) -> Self
    where
        I: IntoIterator<Item = B>,
        B: Into<NodeBuilder>,
    {
        let mut scene = Self::default();
        for spec in nodes {
            let id = scene.insert_builder(spec.into());
            scene.roots.push(id);
        }
        scene
    }

    /// Insert a [`Node`] (and, for a [`NodeContent::Group`], its child subtrees) into
    /// the [`arena`](Self::arena) under a freshly-minted [`NodeId`], returning the id
    /// the node itself took. Does NOT touch [`roots`](Self::roots) or any parent spine.
    /// Used by the edit ops (a pre-built `Node` with an already-id spine Group content
    /// is inserted as-is — its descendants already live in the arena).
    pub(super) fn insert_subtree(&mut self, mut node: Node) -> NodeId {
        let id = self.mint_node_id();
        node.id = id;
        self.arena.insert(id, node);
        id
    }

    /// Flatten a [`NodeBuilder`] spec into the [`arena`](Self::arena), returning the
    /// id the spec's node took (ADR 0003 Phase B5). For a [`NodeBuilder::Group`] the
    /// children are inserted first (depth-first), then the Group node is stored with
    /// its spine of minted child ids. Does NOT touch [`roots`](Self::roots) — the
    /// caller splices the returned id where it belongs.
    fn insert_builder(&mut self, spec: NodeBuilder) -> NodeId {
        match spec {
            NodeBuilder::Leaf(node) => self.insert_subtree(node),
            NodeBuilder::Group {
                name,
                transform,
                enabled,
                children,
            } => {
                let child_ids: Vec<NodeId> =
                    children.into_iter().map(|child| self.insert_builder(child)).collect();
                let mut group = Node::new(name, NodeContent::Group(child_ids));
                group.transform = transform;
                group.enabled = enabled;
                self.insert_subtree(group)
            }
        }
    }

    /// Register a reusable [`AssemblyDef`] from `children` built by value (ADR 0003
    /// Phase B5): each child subtree is inserted into the scene [`arena`](Self::arena)
    /// and the def stores their ids as its spine. The terse test/demo helper mirroring
    /// [`from_nodes`](Self::from_nodes) for definition bodies.
    pub fn add_definition<I, B>(&mut self, id: DefId, name: impl Into<String>, children: I)
    where
        I: IntoIterator<Item = B>,
        B: Into<NodeBuilder>,
    {
        let child_ids: Vec<NodeId> = children
            .into_iter()
            .map(|child| self.insert_builder(child.into()))
            .collect();
        self.definitions.push(AssemblyDef {
            id,
            name: name.into(),
            children: child_ids,
            // Sealed by default (ADR 0017 Decision 3); flip via
            // `set_definition_fixture` to opt a part into splicing (Decision 4).
            fixture: false,
        });
    }

    /// Ensure the scene has exactly one **Origin** Point (issue #29). If no Point
    /// has `is_origin == true`, insert one at index 0 with the spec defaults
    /// (ground plane + axes on; positioned at the world origin). Idempotent: a
    /// second call (or a load of a scene that already carries an Origin) does
    /// nothing. Called on every load path so every scene gains its Origin.
    pub fn ensure_origin_point(&mut self) {
        if self.points.iter().any(|point| point.is_origin) {
            return;
        }
        self.points.insert(
            0,
            Point {
                name: "Origin".to_string(),
                position_blocks: [0, 0, 0],
                offset_voxels: [0, 0, 0],
                // Z-up: the ground plane is XY (`plane_xy`).
                plane_xz: false,
                plane_xy: true,
                plane_yz: false,
                axis_x: true,
                axis_y: true,
                axis_z: true,
                hidden: false,
                is_origin: true,
            },
        );
    }

    /// Erase structurally-invalid sketch entities across every [`NodeContent::SketchTool`]
    /// node (ADR 0030 load policy — erase invalid objects rather than fail the load),
    /// returning `(node name, dropped segment count)` for each node that had drops. Called on
    /// the load path beside [`ensure_node_ids`](Self::ensure_node_ids); the caller emits the
    /// CLI warning. Idempotent — a clean scene drops nothing and returns empty.
    pub fn repair_sketches(&mut self) -> Vec<(String, usize)> {
        let mut warnings = Vec::new();
        for node in self.arena.values_mut() {
            if let NodeContent::SketchTool { producer, .. } = &mut node.content {
                let dropped = producer.sketch.repair();
                if dropped > 0 {
                    warnings.push((node.name.clone(), dropped));
                }
            }
        }
        warnings
    }

    /// Append a reference [`Point`] to the scene (issue #29). A newly-added user
    /// Point defaults to **all planes OFF** (XZ/XY/YZ) with its **axes ON** (issue
    /// #29 fix): only the Origin keeps the ground (XY, Z-up) plane on by default (via
    /// [`ensure_origin_point`](Self::ensure_origin_point)). The plane/axis flags on
    /// the passed `point` are overridden here so every "+ Add Point" path gets the
    /// clean default; the caller controls only the point's name/position/identity.
    pub fn add_point(&mut self, mut point: Point) {
        point.plane_xz = false;
        point.plane_xy = false;
        point.plane_yz = false;
        point.axis_x = true;
        point.axis_y = true;
        point.axis_z = true;
        self.points.push(point);
    }

    /// Remove the Point at `index` (issue #29). **No-op if it is the Origin** (the
    /// Origin is undeletable) or the index is out of range. Hiding the Origin is
    /// done by setting its `hidden` flag (see [`toggle_point_hidden`]), not by
    /// removal.
    ///
    /// [`toggle_point_hidden`]: Self::toggle_point_hidden
    pub fn remove_point(&mut self, index: usize) {
        match self.points.get(index) {
            Some(point) if !point.is_origin => {
                self.points.remove(index);
            }
            _ => {}
        }
    }

    /// Toggle the `hidden` flag of the Point at `index` (issue #29). Works for the
    /// Origin too — the Origin is hideable (just not deletable). No-op for an
    /// out-of-range index.
    pub fn toggle_point_hidden(&mut self, index: usize) {
        if let Some(point) = self.points.get_mut(index) {
            point.hidden = !point.hidden;
        }
    }
}
