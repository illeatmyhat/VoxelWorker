#![allow(
    clippy::doc_markdown,
    clippy::too_long_first_doc_paragraph,
    clippy::use_self
)]

//! The linear command stack — inverse-based undo/redo.
//!
//! Undo/redo is a **linear command stack of inverse commands**, NOT snapshots: the
//! trajectory targets 10k+ node scenes, so a per-edit full-`Scene` clone is too heavy.
//! Instead every undoable [`Intent`] is paired, at apply time, with a fine-grained
//! [`Inverse`] that reverses exactly what the forward op changed. Real-time collaboration
//! is ruled out, so the stack is strictly linear (an `undo` Vec + a `redo` Vec, no
//! branching).
//!
//! **The counter rule (byte-equal invariant).** A minting intent advances
//! [`Scene::next_node_id`](crate::scene::Scene::next_node_id). For `apply; undo` to
//! restore the scene EXACTLY (so the round-trip tests can assert full `Scene`
//! `PartialEq`), `AppCore::undo` restores `next_node_id` to the value captured BEFORE the
//! apply ([`Command::counter_before`]). Because the stack is linear (a `redo` only ever
//! follows an `undo` with nothing applied between), rewinding the counter is safe and
//! makes `redo` re-mint byte-identical ids by replaying the forward intent.
//!
//! **The stacks live in the shell.** This module owns only the pair that needs `Scene`
//! internals — [`Command`] and [`Inverse`]. Which stack a command sits on, how
//! transactions batch, and the selection to restore on undo are runtime workspace state
//! owned by `AppCore`.

use crate::intent::Intent;
use crate::scene::{DefId, Node, NodeContent, NodeId, Point};

/// The reverse of one applied [`Intent`] — enough captured state to restore the
/// scene EXACTLY to what it was before the forward op ran.
///
/// Each arm is the minimal capture for a class of forward op: a field-set is
/// reversed by the SAME intent carrying the PRIOR value; a mint is reversed by
/// removing the minted node; a structural rearrange is reversed by the captured
/// before-shape. The structural arms capture detached `Node`s by value and re-insert
/// them under their ORIGINAL ids (safe — the monotonic counter never reuses an id,
/// and `undo` rewinds the counter so a later `redo` re-mints the same ids).
pub enum Inverse {
    /// Reverse a field-set intent (SetEnabled / SetShape / SetMaterial / SetOffset /
    /// SetName / SetCloudSeed / SetNodeGrids / SetDensity / SetGridMasters and the point
    /// field-sets SetPointHidden / SetPointPlanes / SetPointAxes / SetPointPosition) by
    /// replaying the SAME intent carrying the field's PRIOR value, captured before the
    /// mutate.
    Field(Intent),
    /// Reverse a single-node mint (AddNode / AddChild / AddInstance): the forward op
    /// minted exactly one node + spliced its id onto the end of a spine; the inverse
    /// removes that node (which `remove_node`'s spine splice + arena drop does), then
    /// `undo` rewinds the counter so the id is free again.
    RemoveAdded {
        /// The minted node's id.
        id: NodeId,
    },
    /// Reverse `GroupNode`: [`wrap_node_in_group`](crate::scene::Scene::wrap_node_in_group) wrapped
    /// `target` in a fresh Group `group` that took `target`'s slot. The inverse drops
    /// the Group node from the arena and puts `target`'s id back in the Group's spine
    /// slot.
    UngroupNode {
        /// The wrapped node, restored to the Group's slot.
        target: NodeId,
        /// The fresh Group node to remove.
        group: NodeId,
    },
    /// Reverse `MakeDefinition`:
    /// [`make_definition_from_node`](crate::scene::Scene::make_definition_from_node)
    /// overwrote the node's content with `Instance(def)` and pushed an `AssemblyDef`.
    /// The inverse restores the node's captured `prior_content`, pops the def, and (for
    /// the leaf case, where the op minted a fresh "Body" node) drops that body node.
    UndoMakeDefinition {
        /// The node whose content became an Instance.
        node: NodeId,
        /// The node's content before it became `Instance(def)`.
        prior_content: NodeContent,
        /// The def to pop.
        def: DefId,
        /// The fresh "Body" node the leaf path minted, to drop (the Group path donated
        /// its existing children instead, so this is `None` there).
        minted_body: Option<NodeId>,
    },
    /// Reverse `RemoveNode`: re-insert the captured detached subtree's `Node`s into the
    /// arena under their ORIGINAL ids and splice the root id back into its parent spine
    /// at its original index.
    InsertSubtree {
        /// The parent the subtree hung under (`None` = top-level `roots`).
        parent: Option<NodeId>,
        /// The root id's original index in its parent spine.
        index: usize,
        /// The detached subtree, by value (root first), re-inserted under original ids.
        nodes: Vec<Node>,
    },
    /// Reverse `AddPoint`: [`add_point`](crate::scene::Scene::add_point) appends a
    /// Point at `index` (= `points.len()` before the add). The inverse removes the
    /// Point at that index (it carries no Origin flag, so this never hits the
    /// undeletable-Origin guard).
    RemoveAddedPoint {
        /// The appended Point's index.
        index: usize,
    },
    /// Reverse `RemovePoint`: re-insert the captured `point` at its original `index`.
    InsertPoint {
        /// The removed point's original index.
        index: usize,
        /// The removed point, by value.
        point: Point,
    },
    /// A no-op inverse — the forward intent changed nothing (a field-set to a missing
    /// id / kind-mismatched node, a `RemoveNode` of a stale id, a `RemovePoint` of the
    /// Origin / out-of-range index). Undo restores nothing.
    NoOp,
}

impl Inverse {
    /// Apply this inverse to `scene`, reversing the forward op — the STRUCTURAL arms
    /// only. The counter + selection restore is the caller's (`AppCore::undo`) job; this
    /// touches only the document structure the forward op mutated.
    ///
    /// **[`Inverse::Field`] is NOT handled here**: a field-set is reversed by routing its
    /// prior-value [`Intent`] back through `AppCore::dispatch` — the single owner of the
    /// field-write mutations — so there is no re-implemented copy to silently diverge from
    /// `dispatch` (which would break the byte-equal invariant). `undo`/`redo` intercept
    /// `Field` before calling this, so reaching the `Field` arm here is a construction bug.
    pub fn apply(&self, scene: &mut crate::scene::Scene) {
        match self {
            Inverse::Field(_) => {
                debug_assert!(
                    false,
                    "Inverse::Field must be routed through AppCore::dispatch, not apply()"
                );
            }
            Inverse::RemoveAdded { id } => {
                scene.remove_node_exact(*id);
            }
            Inverse::UngroupNode { target, group } => {
                scene.ungroup_node(*group, *target);
            }
            Inverse::UndoMakeDefinition {
                node,
                prior_content,
                def,
                minted_body,
            } => {
                // Restore the node's content FIRST (it re-adopts the donated children,
                // for the Group path), then pop the def, then drop any minted body.
                if let Some(target) = scene.arena.get_mut(node) {
                    target.content = prior_content.clone();
                }
                scene.definitions.retain(|definition| definition.id != *def);
                if let Some(body_id) = minted_body {
                    scene.arena.remove(body_id);
                }
            }
            Inverse::InsertSubtree {
                parent,
                index,
                nodes,
            } => {
                scene.reinsert_subtree(*parent, *index, nodes);
            }
            Inverse::RemoveAddedPoint { index } => {
                if *index < scene.points.len() {
                    scene.points.remove(*index);
                }
            }
            Inverse::InsertPoint { index, point } => {
                let clamped = (*index).min(scene.points.len());
                scene.points.insert(clamped, point.clone());
            }
            Inverse::NoOp => {}
        }
    }
}

/// One applied document mutation paired with its [`Inverse`] and the counter state to
/// restore on undo. A `redo` re-`dispatch`es the forward `intent`; an `undo` applies
/// `inverse` then restores the captured counter.
///
/// The SELECTION to restore alongside it is workspace state, captured by the shell's
/// `RecordedCommand` wrapper — the document neither owns nor reverses it.
pub struct Command {
    /// The forward intent (re-dispatched on redo).
    pub intent: Intent,
    /// The captured reverse of `intent`.
    pub inverse: Inverse,
    /// `scene.next_node_id` BEFORE the forward op — restored on undo so a minting op's
    /// id is freed, and a later redo re-mints byte-identical ids (the counter rule).
    pub counter_before: u64,
}
