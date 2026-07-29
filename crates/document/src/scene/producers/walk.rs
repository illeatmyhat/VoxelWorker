//! Tree walk + scope pre-composition: the depth-first [`Scene::for_each_leaf`] /
//! [`Scene::walk_nodes`] traversal that composes placed leaves, the flat
//! [`Scene::leaf_producers`] op-stack it feeds, and the sealed-scope pre-composition
//! ([`Scene::composed_scope_leaf`] and friends) ADR 0019 Decision 7 requires.

use super::*;
use crate::scene::*;

impl Scene {
    /// Walk the whole node tree depth-first, invoking
    /// `visitor(world_offset_voxels, leaf)` once for every **enabled leaf** (`Tool`
    /// / `VoxelBody`) with its accumulated **world** VOXEL offset (`parent_offset +
    /// node.offset_voxels`, summed down the tree — translation-only composition,
    /// ADR 0001 step 4; voxels at the document density, ADR 0003 §3f(0)).
    ///
    /// `Group` children inherit the group's world offset; an `Instance(def)` resolves
    /// the referenced [`AssemblyDef`]'s children under the instance's world offset, so
    /// the SAME definition placed by N instances is visited N times at N locations
    /// (the village-of-reused-houses case). The cycle guard (an `Instance` may not
    /// reference an ancestor definition) lives in [`walk_nodes`].
    ///
    /// [`walk_nodes`]: Self::walk_nodes
    /// The visitor receives, per enabled leaf: its accumulated world VOXEL
    /// offset, its content, its own `grids.voxel_grid_on_faces` flag (issue
    /// #29 S4 — the resolver stamps it as each voxel's dedicated `grid_overlay`
    /// attribute, so the on-face voxel grid travels with each voxel through
    /// chunk bucketing), its own
    /// [`CombineOp`] (ADR 0017: the leaf's role in the ordered document-order
    /// fold — a `Subtract` leaf removes occupancy from everything accumulated
    /// before it *in its scope*), and its **scope path** — the chain of
    /// enclosing sealed composition scopes (ADR 0017 Decision 3, issue #74):
    /// every `Group` node and every `Instance`-expanded SEALED definition body
    /// on the way down, outermost first, each carrying the SCOPE node's own
    /// operation (see [`ScopeFrame`]). A scope pre-composes its leaves into one
    /// body; a boolean inside it can never affect geometry outside it. A
    /// FIXTURE definition's expansion contributes NO frame (ADR 0017 Decision
    /// 4, issue #77): its leaves splice into the hosting scope's fold carrying
    /// the HOST's path, so they compose directly at the instance's position.
    pub(crate) fn for_each_leaf(&self, visitor: &mut LeafVisitor<'_>) {
        let mut def_path: Vec<DefId> = Vec::new();
        let mut scope_path: Vec<ScopeFrame> = Vec::new();
        // The ROOT PART is a scope too (ADR 0018 Decision 2 makes it a field rather than an
        // arena entry, but it is still the container the top-level nodes compose in). So it
        // pre-composes on the same two triggers a Group does: its own outset dilates the
        // whole scene, and a top-level Emboss needs the accumulated field its siblings form.
        //
        // Without this a top-level emboss reaches the voxel-set fold, which has no `A − N`
        // to read, and no-ops — and most scenes put their nodes at the top level.
        if let Some(composed) =
            self.composed_scope_leaf(&self.root, &self.roots, [0, 0, 0], &mut def_path, None)
        {
            visitor(VisitedLeaf {
                world_offset_voxels: composed.origin_voxels,
                // ADR 0027: the root composite sits at its integer origin with no continuous
                // slide of its own (its members' float offsets are baked into the composed
                // producer, not carried here), and carries no rotation — so it is upright.
                offset_local_voxels: [0.0, 0.0, 0.0],
                rotation: glam::Quat::IDENTITY,
                body: LeafBody::Composed {
                    producer: composed.producer,
                    fingerprint: composed.fingerprint,
                },
                grid_on_faces: self.root.grids.voxel_grid_on_faces,
                operation: CombineOp::Union,
                outset: self.root.outset,
                scope_path: &scope_path,
                // The whole scene collapsed into the root part: the composite IS the root,
                // and its members carry the per-node origins a pick descends into.
                origin: LeafOrigin::authored(self.root.id),
            });
            return;
        }
        self.walk_nodes(
            &self.roots,
            AccumulatedOffset::origin(),
            &mut def_path,
            &mut scope_path,
            None,
            visitor,
        );
    }

    /// Collect every enabled leaf as a [`LeafProducer`] (ADR 0010 E2): its world voxel
    /// offset, a boxed [`VoxelProducer`](crate::voxel::VoxelProducer), and its single-material
    /// override id. This is the
    /// op-stack the two-layer classifier / boundary-resolve evaluate over — the SAME
    /// leaves [`resolve_chunk_rebased`](Self::resolve_chunk_rebased) stamps, in the SAME
    /// document (walk) order, so the two-layer round-trip composes identically (later-wins
    /// Union on overlap). A region-sized VoxelBody (the cloud field) is sized to the composite
    /// `placed_region_dimensions` exactly as the dense chunk resolve sizes it.
    ///
    /// `pub` — the evaluator seam ADR 0010 E2's `two_layer_store` (up in the app crate) reads;
    /// the dense store keeps using the private [`for_each_leaf`](Self::for_each_leaf).
    pub fn leaf_producers(&self, voxels_per_block: u32) -> Vec<LeafProducer> {
        let region_dimensions = self.placed_region_dimensions(voxels_per_block);
        let mut leaves = Vec::new();
        self.for_each_leaf(&mut |VisitedLeaf {
                                     world_offset_voxels,
                                     offset_local_voxels,
                                     rotation,
                                     body,
                                     grid_on_faces,
                                     operation,
                                     outset,
                                     scope_path,
                                     origin,
                                 }| {
            // ADR 0019 Decision 7: the outset dilates the body BEFORE it folds. Wrapping the
            // producer (rather than teaching the fold a new arm) means the classifier's
            // `cell_field_interval` call below and the voxel-set fold both see one definition
            // of what outset means — see `OutsetProducer`.
            let outset_voxels = outset_voxels_at(outset, voxels_per_block);
            let Some((material, producer)) =
                body.into_producer(region_dimensions, voxels_per_block, outset_voxels)
            else {
                return;
            };
            leaves.push(LeafProducer {
                // The dilated body grows on every side, so its low corner moves DOWN by the
                // outset — the wrapper's frame origin sits `N` below the inner producer's
                // (ADR 0008: the frame is carried, never re-derived).
                world_offset_voxels: std::array::from_fn(|axis| {
                    world_offset_voxels[axis] - outset_voxels
                }),
                // ADR 0027: the continuous rotation and the float local offset. The
                // `- outset_voxels` frame adjustment is an INTEGER-frame move only — the float
                // slide is relative to the (unadjusted) integer origin, so it is NOT re-based here.
                rotation,
                offset_local_voxels,
                producer,
                material,
                grid_overlay: grid_on_faces,
                operation,
                scope_path: scope_path.to_vec(),
                origin,
            });
        });
        leaves
    }

    /// Pre-compose a sealed scope into ONE producer, when it carries an outset and can be
    /// composed. Returns the composed body and the world offset of its low corner.
    ///
    /// `None` means "walk it normally": either the scope carries no outset (the overwhelming
    /// common case — nothing to dilate, so nothing to change) or its subtree contains a body
    /// that cannot participate. A `VoxelBody` is the latter: the cloud field sizes itself
    /// from the region, which a scope has no notion of, and it is fieldless anyway — so a
    /// Part containing one could not be outset even if it were composed (ADR 0020 Decision
    /// 1). Declining leaves that Part's existing behaviour exactly as it was.
    /// The shared tail of [`composed_scope_leaf`](Self::composed_scope_leaf) and
    /// [`composed_subtree`](Self::composed_subtree): given the collected `members` and a
    /// caller-built `fingerprint`, derive the composite's frame origin as the low corner
    /// of its UNION members (the only ones that can bound the body, ADR 0020 Decision 3 —
    /// voxel-valued so density-independent), rebase every member onto it (ADR 0008), and
    /// wrap it in a [`ComposedScope`]. `None` when there are no members, or no union
    /// member (the scope composes to nothing it could dilate). One definition of how a
    /// composite's origin is derived; the two callers differ only in the fingerprint.
    fn finish_composite(
        mut members: Vec<crate::voxel::CompositeMember>,
        fingerprint: String,
    ) -> Option<ComposedScope> {
        if members.is_empty() {
            return None;
        }
        let mut origin = [i64::MAX; 3];
        for member in &members {
            if member.operation != CombineOp::Union {
                continue;
            }
            for (lowest, member_offset) in origin.iter_mut().zip(member.offset_voxels) {
                *lowest = (*lowest).min(member_offset);
            }
        }
        if origin[0] == i64::MAX {
            return None;
        }
        for member in &mut members {
            member.offset_voxels =
                std::array::from_fn(|axis| member.offset_voxels[axis] - origin[axis]);
        }
        Some(ComposedScope {
            origin_voxels: origin,
            fingerprint,
            producer: crate::voxel::CompositeProducer::new(members),
        })
    }

    fn composed_scope_leaf(
        &self,
        scope: &Node,
        children: &[NodeId],
        world_offset_voxels: [i64; 3],
        def_path: &mut Vec<DefId>,
        instance_host: Option<NodeId>,
    ) -> Option<ComposedScope> {
        // Two reasons to pre-compose: the scope is DILATED as a whole (ADR 0019 Decision 7),
        // or one of its members EMBOSSES and so needs the accumulated body as a field rather
        // than as a voxel set (ADR 0020 Decision 4).
        let embosses = children.iter().any(|child| {
            self.arena
                .get(child)
                .is_some_and(|node| node.operation.needs_accumulated_field())
        });
        if scope.outset == voxel_core::units::Measurement::default() && !embosses {
            return None;
        }
        let mut members = Vec::new();
        let mut fingerprints = Vec::new();
        self.collect_composite_members(
            children,
            world_offset_voxels,
            def_path,
            instance_host,
            &mut members,
            &mut fingerprints,
        )?;
        Self::finish_composite(
            members,
            format!(":composed[{}]={}", scope.id.0, fingerprints.join("|")),
        )
    }

    /// Collect a scope's members in document order, mirroring [`walk_nodes`]'s expansion
    /// rules exactly — nested Groups and sealed definition bodies become nested composites,
    /// a FIXTURE definition splices its children inline (ADR 0017 Decision 4), and the cycle
    /// guard is the same. `None` aborts the whole composition (see
    /// [`composed_scope_leaf`](Self::composed_scope_leaf)).
    ///
    /// [`walk_nodes`]: Self::walk_nodes
    fn collect_composite_members(
        &self,
        spine: &[NodeId],
        parent_offset: [i64; 3],
        def_path: &mut Vec<DefId>,
        instance_host: Option<NodeId>,
        members: &mut Vec<crate::voxel::CompositeMember>,
        fingerprints: &mut Vec<String>,
    ) -> Option<()> {
        for &node_id in spine {
            let Some(node) = self.arena.get(&node_id) else {
                continue;
            };
            if !node.enabled {
                continue;
            }
            let world_offset_voxels: [i64; 3] = std::array::from_fn(|axis| {
                parent_offset[axis] + node.transform.offset_voxels[axis]
            });
            match &node.content {
                // A fieldless / region-sized body cannot compose (see the caller).
                NodeContent::VoxelBody(_) => return None,
                NodeContent::Tool { .. } | NodeContent::SketchTool { .. } => {
                    let (material, producer) = LeafBody::Content(&node.content).into_producer(
                        [0, 0, 0],
                        0,
                        // A member's OWN outset still applies inside the scope: it dilates
                        // that member before the scope's fold sees it, and the scope's outset
                        // then dilates the composed result.
                        node.outset.to_voxels(1).unwrap_or(0),
                    )?;
                    fingerprints.push(format!("{}", node.id.0));
                    members.push(crate::voxel::CompositeMember {
                        offset_voxels: world_offset_voxels,
                        operation: node.operation,
                        material,
                        producer,
                        source: LeafOrigin {
                            node: node.id,
                            instance_host,
                        },
                    });
                }
                NodeContent::Group(children) => {
                    let nested = self.composed_subtree(
                        children,
                        world_offset_voxels,
                        def_path,
                        instance_host,
                    )?;
                    fingerprints.push(format!("{}({})", node.id.0, nested.fingerprint));
                    members.push(crate::voxel::CompositeMember {
                        offset_voxels: nested.origin_voxels,
                        operation: node.operation,
                        material: None,
                        producer: crate::voxel::OutsetProducer::wrap(
                            Box::new(nested.producer),
                            node.outset.to_voxels(1).unwrap_or(0),
                        ),
                        source: LeafOrigin {
                            node: node.id,
                            instance_host,
                        },
                    });
                }
                NodeContent::Instance(def_id) => {
                    if def_path.contains(def_id) {
                        continue;
                    }
                    let def = self.def_by_id(*def_id)?;
                    def_path.push(*def_id);
                    // ADR 0032: the OUTERMOST instance wins, so an already-set host is
                    // kept — a definition containing another instance still redirects a pick
                    // to the placement the user can see.
                    let inner_host = instance_host.or(Some(node.id));
                    let outcome = if def.fixture {
                        // ADR 0017 Decision 4: a fixture does NOT pre-compose — its children
                        // splice into the hosting scope's fold under their own operations.
                        self.collect_composite_members(
                            &def.children,
                            world_offset_voxels,
                            def_path,
                            inner_host,
                            members,
                            fingerprints,
                        )
                    } else {
                        self.composed_subtree(
                            &def.children,
                            world_offset_voxels,
                            def_path,
                            inner_host,
                        )
                        .map(|nested| {
                            fingerprints.push(format!("{}[{}]", node.id.0, nested.fingerprint));
                            members.push(crate::voxel::CompositeMember {
                                offset_voxels: nested.origin_voxels,
                                operation: node.operation,
                                material: None,
                                producer: crate::voxel::OutsetProducer::wrap(
                                    Box::new(nested.producer),
                                    node.outset.to_voxels(1).unwrap_or(0),
                                ),
                                source: LeafOrigin {
                                    node: node.id,
                                    instance_host,
                                },
                            });
                        })
                    };
                    def_path.pop();
                    outcome?;
                }
            }
        }
        Some(())
    }

    /// Compose a sub-scope unconditionally (its outset is applied by the CALLER, which owns
    /// the member entry) — the recursive half of [`composed_scope_leaf`].
    ///
    /// [`composed_scope_leaf`]: Self::composed_scope_leaf
    fn composed_subtree(
        &self,
        children: &[NodeId],
        world_offset_voxels: [i64; 3],
        def_path: &mut Vec<DefId>,
        instance_host: Option<NodeId>,
    ) -> Option<ComposedScope> {
        let mut members = Vec::new();
        let mut fingerprints = Vec::new();
        self.collect_composite_members(
            children,
            world_offset_voxels,
            def_path,
            instance_host,
            &mut members,
            &mut fingerprints,
        )?;
        Self::finish_composite(members, fingerprints.join("|"))
    }

    /// Recursive worker for [`for_each_leaf`](Self::for_each_leaf). `parent_offset`
    /// is the accumulated placement offset of the assembly that owns `nodes` — the
    /// integer world VOXEL offset plus the continuous float slide relative to it (ADR
    /// 0027), carried together and handed to the visitor (additive — resolve still reads
    /// the integer offset, so occupancy is unchanged);
    /// `instance_host` is the outermost `Instance` this spine is being expanded under, or
    /// `None` in the scene proper — what a viewport pick redirects to (ADR 0032);
    /// `def_path` is the stack of definition ids currently being expanded (for the
    /// cycle guard — an `Instance` that would re-enter a definition already on the
    /// path is skipped instead of recursing forever); `scope_path` is the stack of
    /// enclosing sealed-scope frames (ADR 0017 Decision 3 — pushed on entering a
    /// `Group` or an `Instance`'s definition body, popped on leaving) handed to the
    /// visitor per leaf.
    pub(crate) fn walk_nodes(
        &self,
        spine: &[NodeId],
        parent_offset: AccumulatedOffset,
        def_path: &mut Vec<DefId>,
        scope_path: &mut Vec<ScopeFrame>,
        instance_host: Option<NodeId>,
        visitor: &mut LeafVisitor<'_>,
    ) {
        // GOLDEN-CRITICAL (ADR 0003 B5): iterate the id-spine for ORDER (document
        // order = later-wins on overlap), fetching each node's content from the
        // arena. NEVER iterate the arena to produce this walk — that visits in id
        // order and would reorder Union material on overlap, moving the goldens.
        for &node_id in spine {
            let Some(node) = self.arena.get(&node_id) else {
                continue;
            };
            if !node.enabled {
                continue;
            }
            // ADR 0027: the integer offset and the continuous float slide accumulate the
            // same way, additively, and travel together to the visitor.
            let world_offset = parent_offset.plus(&node.transform);
            let world_offset_voxels = world_offset.world_voxels;
            match &node.content {
                NodeContent::Tool { .. }
                | NodeContent::SketchTool { .. }
                | NodeContent::VoxelBody(_) => {
                    // ADR 0017: the leaf carries its OWN `operation` plus the chain
                    // of enclosing sealed-scope frames (issue #74) into the flat
                    // walk — consumers reconstruct the scoped fold from the paths.
                    visitor(VisitedLeaf {
                        world_offset_voxels,
                        offset_local_voxels: world_offset.local_voxels,
                        // ADR 0027: the leaf's continuous rotation, the whole tilt seated
                        // against the surface it was dropped on. The classifier reads this
                        // quaternion directly (a lattice turn is just a rotation on the exact
                        // classifier path).
                        rotation: node.transform.rotation(),
                        body: LeafBody::Content(&node.content),
                        grid_on_faces: node.grids.voxel_grid_on_faces,
                        operation: node.operation,
                        outset: node.outset,
                        scope_path,
                        origin: LeafOrigin {
                            node: node.id,
                            instance_host,
                        },
                    });
                }
                NodeContent::Group(children) => {
                    // ADR 0019 Decision 7: an outset on a SCOPE dilates the scope's COMPOSED
                    // body, so the scope is evaluated as one producer and handed to the
                    // visitor as a single leaf rather than recursed into. Per-member
                    // dilation is a different operation and the ADR rejects it: it would
                    // make an internal Subtract cutter carve MORE, where dilating the
                    // composed Part grows the finished body and partly closes that cut.
                    if let Some(composed) = self.composed_scope_leaf(
                        node,
                        children,
                        world_offset_voxels,
                        def_path,
                        instance_host,
                    ) {
                        visitor(VisitedLeaf {
                            world_offset_voxels: composed.origin_voxels,
                            // ADR 0027: a composed scope carries no continuous slide of its own
                            // (members' float offsets are baked into the composite producer)
                            // and no rotation of its own — the composite is upright.
                            offset_local_voxels: [0.0, 0.0, 0.0],
                            rotation: glam::Quat::IDENTITY,
                            body: LeafBody::Composed {
                                producer: composed.producer,
                                fingerprint: composed.fingerprint,
                            },
                            grid_on_faces: node.grids.voxel_grid_on_faces,
                            operation: node.operation,
                            outset: node.outset,
                            scope_path,
                            origin: LeafOrigin {
                                node: node.id,
                                instance_host,
                            },
                        });
                        continue;
                    }
                    // ADR 0017 Decision 3 (issue #74): a Group is a SEALED
                    // composition scope — its frame (identity + the GROUP node's own
                    // operation) encloses every leaf below it, so the group's
                    // children pre-compose into one body that folds into the parent
                    // under the group's operation.
                    scope_path.push(ScopeFrame {
                        scope_node: node.id,
                        operation: node.operation,
                    });
                    self.walk_nodes(
                        children,
                        world_offset,
                        def_path,
                        scope_path,
                        instance_host,
                        visitor,
                    );
                    scope_path.pop();
                }
                NodeContent::Instance(def_id) => {
                    // Cycle guard: an Instance may not reference an ancestor
                    // definition. If this id is already being expanded on the
                    // current path, skip it (never recurse into a cycle).
                    if def_path.contains(def_id) {
                        eprintln!(
                            "scene: skipping Instance({def_id:?}) — cyclic reference \
                             to an ancestor definition (path {def_path:?})"
                        );
                        continue;
                    }
                    let Some(def) = self.def_by_id(*def_id) else {
                        // An Instance pointing at a missing definition resolves to
                        // nothing (no panic — the model stays robust to dangling ids).
                        continue;
                    };
                    def_path.push(*def_id);
                    // ADR 0032: the OUTERMOST instance wins — set on first entry and kept
                    // through any nested instance, so a pick redirects to the placement the
                    // user can actually see rather than one buried in a definition.
                    let inner_host = instance_host.or(Some(node.id));
                    if def.fixture {
                        // ADR 0017 Decision 4 (issue #77): a FIXTURE definition does
                        // NOT pre-compose — NO scope frame is pushed, so its children
                        // SPLICE into the HOSTING scope's fold at this instance's
                        // spine position, in order, under the instance's transform
                        // (`world_offset_voxels` composes exactly as for a sealed
                        // body — the carried-frame discipline of ADR 0008; the host
                        // is POSITIONAL, never a stored reference). The instance's
                        // own `operation` is INERT (never consulted): each spliced
                        // leaf folds under its OWN operation. The fixture pierces
                        // exactly this ONE level of pre-composition — every scope
                        // frame already on `scope_path` (the host scope's seal and
                        // above) stays absolute. Invalidation: flipping a def's
                        // fixture flag changes every expanded leaf's carried scope
                        // path (the instance frame appears/disappears), so their
                        // fingerprints change and the store re-classifies exactly
                        // those leaves' AABBs — which contain every cell a splice
                        // can differ in (Union adds / Subtract carves only within a
                        // leaf's own body; an Intersect-influence leaf is already a
                        // wholesale-clear fingerprint kind either way).
                        self.walk_nodes(
                            &def.children,
                            world_offset,
                            def_path,
                            scope_path,
                            inner_host,
                            visitor,
                        );
                    } else {
                        // ADR 0017 Decision 3 (issue #74): a definition body is a
                        // SEALED scope — it pre-composes (internal booleans are fully
                        // spent inside it), and the finished body folds into the
                        // parent under the INSTANCE node's operation. The frame's
                        // identity is the INSTANCE node (unique per placement), so
                        // two expansions of the same definition are distinct scopes.
                        scope_path.push(ScopeFrame {
                            scope_node: node.id,
                            operation: node.operation,
                        });
                        self.walk_nodes(
                            &def.children,
                            world_offset,
                            def_path,
                            scope_path,
                            inner_host,
                            visitor,
                        );
                        scope_path.pop();
                    }
                    def_path.pop();
                }
            }
        }
    }
}
