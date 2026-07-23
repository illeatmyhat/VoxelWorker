//! Tests for the sketch-editing undo GROUP (ADR 0028 §4, issue #94).
//!
//! The group is the tracer bullet's transaction core: a mode opens it on enter, in-mode edits
//! route into its own session history through the SAME [`apply_intent`] door (so apply and undo
//! never disagree on which stack an edit lives on), fine-grained in-mode undo/redo reverse
//! individual edits, Finish moves the whole session onto the main stack as ONE transaction, and
//! Cancel reverses the session back to the enter-state and discards it. Every assertion maps to
//! one of #94's acceptance criteria and runs headless (no GPU — these only touch the borrowed
//! scene + the owned command stack).
//!
//! [`apply_intent`]: super::AppCore::apply_intent

use super::AppCore;
use camera::OrbitCamera;
use document::intent::Intent;
use document::scene::{Node, NodeContent, NodeId, Scene};
use document::sketch::{PlaneAxis, Sketch, SketchSolid};
use voxel_core::core_geom::MaterialChoice;

fn test_core() -> AppCore {
    AppCore::new(OrbitCamera::default())
}

/// A rectangle-footprint extrude producer (profile in XY, extruded along +Z). The three args
/// vary independently so successive edits are distinguishable.
fn box_sketch(width_voxels: i64, depth_voxels: i64, height_voxels: u32) -> SketchSolid {
    SketchSolid::extrude(
        Sketch::rectangle(PlaneAxis::Z, width_voxels, depth_voxels),
        height_voxels,
    )
}

/// A one-sketch scene (stable ids + Origin), the sketch node active.
fn single_sketch_scene() -> (Scene, NodeId) {
    let mut scene = Scene::from_nodes(vec![Node::new(
        "Sketch",
        NodeContent::SketchTool {
            producer: box_sketch(32, 32, 32),
            material: MaterialChoice::Stone,
        },
    )]);
    scene.ensure_node_ids();
    scene.ensure_origin_point();
    scene.active = scene.roots.first().copied();
    let target = scene.roots[0];
    (scene, target)
}

/// The `SketchSolid` currently on `id` (panics if not a sketch node).
fn producer_of(scene: &Scene, id: NodeId) -> SketchSolid {
    match &scene.node_by_id(id).expect("live node").content {
        NodeContent::SketchTool { producer, .. } => producer.clone(),
        _ => panic!("not a sketch node"),
    }
}

/// The `MaterialChoice` currently on `id`.
fn material_of(scene: &Scene, id: NodeId) -> MaterialChoice {
    match &scene.node_by_id(id).expect("live node").content {
        NodeContent::SketchTool { material, .. } => *material,
        _ => panic!("not a sketch node"),
    }
}

/// Apply an in-mode sketch edit through the SINGLE apply door — while a group is open this
/// routes into the session (exactly what the live inspector / vertex drag do).
fn edit(core: &mut AppCore, scene: &mut Scene, target: NodeId, producer: SketchSolid) {
    core.apply_intent(scene, Intent::SetSketch { target, producer });
}

#[test]
fn finish_commits_the_session_as_one_main_entry() {
    let mut core = test_core();
    let (mut scene, target) = single_sketch_scene();
    let enter = producer_of(&scene, target);

    core.begin_sketch_group();
    // Three coalesced edits (three drag releases / inspector nudges) — all through apply_intent.
    edit(&mut core, &mut scene, target, box_sketch(48, 32, 32));
    edit(&mut core, &mut scene, target, box_sketch(48, 48, 32));
    let final_producer = box_sketch(48, 48, 64);
    edit(&mut core, &mut scene, target, final_producer.clone());
    // In-mode edits route into the session, NEVER the main stack (the bug the review caught:
    // they must not reach `apply_intent`'s main-stack path while a group is open).
    assert_eq!(core.undo_depth(), 0, "in-mode edits must not reach the main stack");

    core.finish_sketch_group();
    assert!(!core.in_sketch_group(), "Finish closes the group");
    assert_eq!(
        core.undo_depth(),
        1,
        "Finish commits the whole session as ONE main-stack transaction"
    );
    assert_eq!(
        producer_of(&scene, target),
        final_producer,
        "the final producer stands after Finish"
    );

    // A SINGLE main-stack undo reverses the ENTIRE session back to the enter-state.
    core.undo(&mut scene);
    assert_eq!(
        producer_of(&scene, target),
        enter,
        "one undo past the sketch reverses all of it"
    );
    // And a single redo re-applies the whole session.
    core.redo(&mut scene);
    assert_eq!(producer_of(&scene, target), final_producer);
}

#[test]
fn cancel_rolls_the_session_back_to_enter() {
    let mut core = test_core();
    let (mut scene, target) = single_sketch_scene();
    let enter = producer_of(&scene, target);

    core.begin_sketch_group();
    edit(&mut core, &mut scene, target, box_sketch(48, 32, 32));
    edit(&mut core, &mut scene, target, box_sketch(64, 64, 64));
    core.cancel_sketch_group(&mut scene);

    assert!(!core.in_sketch_group(), "Cancel closes the group");
    assert_eq!(
        producer_of(&scene, target),
        enter,
        "Cancel restores the enter-state"
    );
    assert_eq!(core.undo_depth(), 0, "Cancel leaves nothing on the main stack");
}

#[test]
fn cancel_restores_the_enter_selection() {
    // Regression for the review's confirmed finding [5]: Cancel must restore the enter
    // SELECTION, not just the producer — the general transaction model gets this for free
    // because each reversed command restores its own captured `selection_before`.
    let mut core = test_core();
    let (mut scene, target) = single_sketch_scene();
    // A second node so selection can move to something other than the sketch during the session.
    scene.active = Some(target);

    core.begin_sketch_group();
    edit(&mut core, &mut scene, target, box_sketch(48, 48, 48));
    // The shell moves the selection mid-session (e.g. a sub-element pick), then the user Cancels.
    scene.active = None;
    core.cancel_sketch_group(&mut scene);
    assert_eq!(
        scene.active,
        Some(target),
        "Cancel restores the selection captured before the first in-mode edit"
    );
}

#[test]
fn a_non_producer_edit_mid_session_is_captured() {
    // The general transaction model captures ANY in-mode edit, not only producer changes — a
    // material edit mid-session is reversed by Cancel and committed by Finish. (The old
    // net-producer-collapse silently dropped this.)
    let mut core = test_core();
    let (mut scene, target) = single_sketch_scene();
    assert_eq!(material_of(&scene, target), MaterialChoice::Stone);

    core.begin_sketch_group();
    core.apply_intent(&mut scene, Intent::SetMaterial { target, material: MaterialChoice::Wood });
    assert_eq!(material_of(&scene, target), MaterialChoice::Wood, "live during the session");
    assert_eq!(core.undo_depth(), 0, "the material edit stays in the session");

    core.cancel_sketch_group(&mut scene);
    assert_eq!(
        material_of(&scene, target),
        MaterialChoice::Stone,
        "Cancel reverses a non-producer in-mode edit too"
    );
}

#[test]
fn in_mode_undo_redo_is_fine_grained() {
    let mut core = test_core();
    let (mut scene, target) = single_sketch_scene();
    let enter = producer_of(&scene, target);
    let a = box_sketch(48, 32, 32);
    let b = box_sketch(48, 48, 64);

    core.begin_sketch_group();
    edit(&mut core, &mut scene, target, a.clone());
    edit(&mut core, &mut scene, target, b.clone());

    // In-mode undo reverses ONE edit, staying in the mode and never touching the main stack.
    core.undo(&mut scene);
    assert_eq!(
        producer_of(&scene, target),
        a,
        "in-mode undo reverses the last edit only"
    );
    assert!(core.in_sketch_group(), "in-mode undo stays in the mode");
    assert_eq!(core.undo_depth(), 0, "in-mode undo never touches the main stack");

    core.undo(&mut scene);
    assert_eq!(
        producer_of(&scene, target),
        enter,
        "a second in-mode undo reaches the enter-state"
    );
    // Undo past the enter-state is a no-op — the enter is not itself an edit.
    core.undo(&mut scene);
    assert_eq!(producer_of(&scene, target), enter, "undo past enter is a no-op");

    // In-mode redo re-applies each edit in turn.
    core.redo(&mut scene);
    assert_eq!(producer_of(&scene, target), a);
    core.redo(&mut scene);
    assert_eq!(producer_of(&scene, target), b);
}

#[test]
fn a_net_zero_session_commits_nothing() {
    // Edited then in-mode-undone back to enter: Finish files no empty main-stack entry.
    let mut core = test_core();
    let (mut scene, target) = single_sketch_scene();
    core.begin_sketch_group();
    edit(&mut core, &mut scene, target, box_sketch(48, 48, 48));
    core.undo(&mut scene); // back to enter (the edit sits on session_redo)
    core.finish_sketch_group();
    assert_eq!(
        core.undo_depth(),
        0,
        "a net-zero session must leave no undo entry"
    );
}

#[test]
fn edits_outside_a_group_stay_singleton_main_transactions() {
    // The main stack is unchanged for ordinary edits: each is its own singleton transaction, so
    // one undo reverses one edit (no behaviour change for the non-sketch path).
    let mut core = test_core();
    let (mut scene, target) = single_sketch_scene();
    let enter = producer_of(&scene, target);
    let a = box_sketch(48, 48, 48);
    edit(&mut core, &mut scene, target, a.clone());
    edit(&mut core, &mut scene, target, box_sketch(64, 64, 64));
    assert_eq!(core.undo_depth(), 2, "two ordinary edits = two transactions");
    core.undo(&mut scene);
    assert_eq!(producer_of(&scene, target), a, "one undo reverses one ordinary edit");
    core.undo(&mut scene);
    assert_eq!(producer_of(&scene, target), enter);
}

#[test]
fn begin_does_not_mutate_the_document() {
    // The group is non-document (ADR 0022): opening it mutates NOTHING in the scene.
    let mut core = test_core();
    let (scene, _target) = single_sketch_scene();
    let before = scene.clone();
    core.begin_sketch_group();
    assert_eq!(scene, before, "opening a sketch group must not touch the document");
    assert!(core.in_sketch_group());
}

#[test]
fn a_fresh_in_mode_edit_clears_the_in_mode_redo() {
    // The linear-stack rule, scoped to the session: an edit after an in-mode undo drops the
    // in-mode redo future.
    let mut core = test_core();
    let (mut scene, target) = single_sketch_scene();
    core.begin_sketch_group();
    edit(&mut core, &mut scene, target, box_sketch(48, 32, 32));
    core.undo(&mut scene); // to enter; the undone edit sits on session_redo
    let c = box_sketch(64, 64, 64);
    edit(&mut core, &mut scene, target, c.clone()); // a fresh edit clears the redo
    core.redo(&mut scene); // no-op — the redo future was invalidated
    assert_eq!(
        producer_of(&scene, target),
        c,
        "a fresh in-mode edit invalidates the in-mode redo future"
    );
}
