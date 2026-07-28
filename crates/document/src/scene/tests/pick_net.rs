use super::*;
use std::collections::BTreeMap;
use voxel_core::core_geom::MaterialChoice;

// ---- ADR 0032 slice 4: the adversarial net ----
//
// `pick.rs` pins the picked-node rule by EXAMPLE, and every example there was chosen by the
// same mental model that wrote the resolver — so they confirm it rather than attack it. This
// module differentials instead: it derives, from the dense oracle alone, which node owns each
// cell, then asserts `picked_node_at_voxel` agrees at EVERY cell in the scene's box —
// including the empty ones, where it must name nothing. Both directions matter: the first bug
// it caught was stamped-but-unpickable, the second was pickable-but-carved.
//
// **How node identity comes out of a material oracle.** The oracle records materials, not
// nodes. Rather than cap a scene at one body per palette entry, each scene is resolved once
// per tool with that tool's material set to a MARKER and every other tool's set to the
// BACKGROUND: the cells stamped MARKER are exactly the cells that tool owns. Materials never
// affect occupancy — a boolean does not stamp one at all (ADR 0017 Decision 1) — so every
// variant resolves identical geometry and only the read-out changes. Two materials is the
// whole requirement, so the net is bounded by neither the palette nor the scene size.
//
// **What this can and cannot catch.** For an out-of-phase leaf, pick and oracle sample the
// same field through the same `dense_leaf_placement`, so per-leaf coverage is shared-fate: the
// net is blind to a bug in the placement affine itself, which `substrate`'s translation-
// invariance tests and `evaluation`'s affine oracle already own. What it attacks is the
// ownership FOLD — later-wins order, scope open/close, and the descent into a pre-composed
// scope — which is slice 2's new code and shares nothing with the oracle's grid fold.

/// The material marking the tool under test in an ownership variant, and the material every
/// other tool wears. Any two distinct entries work; nothing here depends on the palette size.
const MARKER: MaterialChoice = MaterialChoice::Wood;
const BACKGROUND: MaterialChoice = MaterialChoice::Stone;

/// A whole-block box at an explicit density, so the sweep can vary density without touching
/// the shared `box_tool` fixture (which is pinned to the module's authoring density).
fn box_at(
    size_blocks: [u32; 3],
    offset_blocks: [i64; 3],
    operation: CombineOp,
    density: u32,
) -> Node {
    let shape = SdfShape::from_blocks(ShapeKind::Box, size_blocks, 1, density);
    let mut node = Node::new(
        "Box",
        NodeContent::Tool {
            shape,
            material: BACKGROUND,
        },
    );
    node.transform = NodeTransform::from_blocks(offset_blocks, density);
    node.operation = operation;
    node
}

/// `box_at` carrying an adversarial placement on top of its whole-block seat.
fn placed_box(
    size_blocks: [u32; 3],
    offset_blocks: [i64; 3],
    operation: CombineOp,
    rotation: glam::Quat,
    slide: [f32; 3],
    density: u32,
) -> Node {
    let mut node = box_at(size_blocks, offset_blocks, operation, density);
    node.transform.rotation_quaternion = Some(rotation.to_array());
    node.transform.offset_local_voxels = slide;
    node
}

/// Every `Tool` node in `scene`, in document order. The net authors no instances, so a scene's
/// tools are all reachable from its roots.
fn tool_nodes(scene: &Scene) -> Vec<NodeId> {
    fn descend(scene: &Scene, id: NodeId, found: &mut Vec<NodeId>) {
        let Some(node) = scene.node_by_id(id) else {
            return;
        };
        match &node.content {
            NodeContent::Tool { .. } => found.push(id),
            NodeContent::Group(children) => {
                for child in children {
                    descend(scene, *child, found);
                }
            }
            _ => {}
        }
    }
    let mut found = Vec::new();
    for root in &scene.roots {
        descend(scene, *root, &mut found);
    }
    found
}

fn child_of(scene: &Scene, parent: NodeId, index: usize) -> NodeId {
    match &scene
        .node_by_id(parent)
        .expect("the parent resolves")
        .content
    {
        NodeContent::Group(children) => children[index],
        other => panic!("expected a Group, got {other:?}"),
    }
}

fn set_operation(scene: &mut Scene, node: NodeId, operation: CombineOp) {
    scene
        .node_by_id_mut(node)
        .expect("the node is in the scene")
        .operation = operation;
}

fn with_material(scene: &mut Scene, tool: NodeId, choice: MaterialChoice) {
    match &mut scene
        .node_by_id_mut(tool)
        .expect("the tool is in the scene")
        .content
    {
        NodeContent::Tool { material, .. } => *material = choice,
        other => panic!("expected a Tool, got {other:?}"),
    }
}

/// The material the dense resolve STAMPED at each absolute cell: a last-wins fold over the
/// oracle's occupied list, which is emitted in document order and never deduplicated, so the
/// final writer at a cell is the one whose colour the user sees there (ADR 0017).
fn stamped_materials(scene: &Scene, density: u32) -> BTreeMap<[i64; 3], u16> {
    let grid = scene.resolve_region(scene.full_extent_blocks(density), density, 0);
    let recentre = grid.recentre_voxels;
    let mut stamped = BTreeMap::new();
    for voxel in &grid.occupied {
        let position = voxel.world_position();
        let cell: [i64; 3] =
            std::array::from_fn(|axis| (position[axis] - 0.5).round() as i64 + recentre[axis]);
        stamped.insert(cell, voxel.color_index());
    }
    stamped
}

/// Which node owns each occupied cell, according to the dense oracle alone — the independent
/// answer `picked_node_at_voxel` is held against. See the module header for the marker trick.
fn owner_of_each_cell(scene: &Scene, density: u32) -> BTreeMap<[i64; 3], NodeId> {
    let tools = tool_nodes(scene);
    let mut owners: BTreeMap<[i64; 3], NodeId> = BTreeMap::new();
    for &marked in &tools {
        let mut variant = scene.clone();
        for &tool in &tools {
            let choice = if tool == marked { MARKER } else { BACKGROUND };
            with_material(&mut variant, tool, choice);
        }
        for (cell, material) in stamped_materials(&variant, density) {
            if material == MARKER.material_id() {
                let previous = owners.insert(cell, marked);
                assert!(
                    previous.is_none(),
                    "two tools both claim to be the last stamper at {cell:?}, so the marker \
                     read-out is not a function"
                );
            }
        }
    }
    owners
}

/// Assert the law over every cell in the scene's placed box, padded by one so the empty rim is
/// checked too: an owned cell picks its owner, an unowned cell picks nothing. Returns how many
/// cells were owned, so a caller can prove the sweep bit on something.
///
/// The box comes from the scene's own placed extent rather than from the owned cells, so a
/// case whose booleans annihilate everything is still swept — "the pick names nothing anywhere"
/// is a real assertion, and the shapes below include orderings that reach it.
fn assert_the_pick_names_the_owner(scene: &Scene, density: u32, case: &str) -> usize {
    let owners = owner_of_each_cell(scene, density);
    let Some((extent_low, extent_high)) = scene.placed_extent_voxels(density) else {
        panic!("{case}: no leaf has an intrinsic size, so the case proves nothing");
    };
    let low: [i64; 3] = std::array::from_fn(|axis| extent_low[axis] - 1);
    let high: [i64; 3] = std::array::from_fn(|axis| extent_high[axis] + 1);

    for z in low[2]..=high[2] {
        for y in low[1]..=high[1] {
            for x in low[0]..=high[0] {
                let cell = [x, y, z];
                let picked = scene.picked_node_at_voxel(cell, density);
                match (owners.get(&cell), picked) {
                    (Some(&owner), Some(picked)) => assert_eq!(
                        picked, owner,
                        "{case}: the resolve gave {cell:?} to {owner:?}, the pick named {picked:?}"
                    ),
                    (Some(&owner), None) => panic!(
                        "{case}: the resolve gave {cell:?} to {owner:?}, the pick named nothing"
                    ),
                    (None, Some(picked)) => {
                        panic!("{case}: the resolve left {cell:?} empty, the pick named {picked:?}")
                    }
                    (None, None) => {}
                }
            }
        }
    }
    owners.len()
}

/// The placements the sweep drives every scene shape through. The oblique quats are the ones
/// `evaluation`'s affine oracle already trusts, so a disagreement here is the pick's fold, not
/// the affine underneath it.
///
/// Every entry carrying a rotation also carries a FRACTIONAL slide, deliberately. A pure
/// lattice turn on a whole-voxel seat is classified in phase
/// (`substrate::spatial::is_in_phase`), which routes the dense oracle down `stamp_producer` — a
/// translation that drops the turn — while the pick honours it, as the live classifier does.
/// That divergence is real and is reported as a finding against the ORACLE; pairing turns with
/// a slide forces the gather path and keeps this net pointed at the pick. The lattice-turn case
/// itself is held by `pick::a_quarter_turned_body_picks_on_its_turned_footprint`, whose
/// expectation is hand-derived and so does not depend on the oracle at all.
fn adversarial_placements() -> Vec<(&'static str, glam::Quat, [f32; 3])> {
    let quarter_turn = glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
    let tilt = glam::Quat::from_rotation_x(0.7);
    let compound = glam::Quat::from_rotation_z(0.6) * glam::Quat::from_rotation_x(0.3);
    vec![
        (
            "seated square on the lattice",
            glam::Quat::IDENTITY,
            [0.0; 3],
        ),
        ("half-voxel slide", glam::Quat::IDENTITY, [0.5, 0.0, 0.0]),
        (
            "fractional slide on two axes",
            glam::Quat::IDENTITY,
            [0.25, 0.0, 0.75],
        ),
        (
            "quarter turn, slid off phase",
            quarter_turn,
            [0.5, 0.0, 0.0],
        ),
        ("oblique tilt", tilt, [0.25, 0.0, 0.0]),
        ("oblique compound", compound, [0.25, 0.0, 0.75]),
    ]
}

/// The purest case: one placed body, nothing to fold against, swept across densities. Any
/// disagreement here is the per-leaf coverage test rather than the fold. Density 1 is included
/// because it collapses a block to a voxel, which is where parity assumptions surface.
#[test]
fn a_lone_placed_body_is_owned_by_itself_everywhere() {
    let mut owned = 0;
    for density in [1, 2, 8] {
        for (case, rotation, slide) in adversarial_placements() {
            let scene = with_minted_ids(Scene::from_nodes(vec![placed_box(
                [4, 6, 4],
                [0, 0, 0],
                CombineOp::Union,
                rotation,
                slide,
                density,
            )]));
            owned +=
                assert_the_pick_names_the_owner(&scene, density, &format!("d{density} {case}"));
        }
    }
    assert!(
        owned > 0,
        "the sweep resolved nothing, so it proved nothing"
    );
}

/// Later-wins under an adversarial placement: the overlap must hand every cell to the LATER
/// body, cell by cell, along a boundary that no longer follows the lattice.
#[test]
fn an_overlap_hands_every_cell_to_the_later_body() {
    for (case, rotation, slide) in adversarial_placements() {
        let scene = with_minted_ids(Scene::from_nodes(vec![
            box_at([3, 3, 3], [0, 0, 0], CombineOp::Union, DENSITY),
            placed_box(
                [2, 2, 2],
                [1, 0, 0],
                CombineOp::Union,
                rotation,
                slide,
                DENSITY,
            ),
        ]));
        assert!(assert_the_pick_names_the_owner(&scene, DENSITY, case) > 0);
    }
}

/// A placed cutter: every surviving cell keeps its host, every carved cell picks nothing. The
/// cavity wall is what a nearest-hit rule gets wrong, swept here over a wall a rotation has
/// taken off the lattice.
#[test]
fn a_placed_cutter_leaves_the_host_owning_exactly_what_survived() {
    for (case, rotation, slide) in adversarial_placements() {
        let scene = with_minted_ids(Scene::from_nodes(vec![
            box_at([4, 4, 4], [0, 0, 0], CombineOp::Union, DENSITY),
            placed_box(
                [2, 2, 2],
                [1, 1, 1],
                CombineOp::Subtract,
                rotation,
                slide,
                DENSITY,
            ),
        ]));
        assert!(assert_the_pick_names_the_owner(&scene, DENSITY, case) > 0);
    }
}

/// The scope orderings a hand-picked example set does not reach, each folded through every
/// placement. `sync_owner_scope_stack` / `fold_owner_into` are a hand-mirror of the dense
/// resolvers' `sync_grid_scope_stack` / `fold_closed_scope_into`, and mirrors drift at the
/// edges — an empty accumulator, a mask closing a scope, a scope closing into another scope.
fn scope_shapes(placed: &Node, density: u32) -> Vec<(&'static str, Scene)> {
    let ground = box_at([4, 4, 4], [0, 0, 0], CombineOp::Union, density);
    let cut = |operation| box_at([2, 2, 2], [1, 1, 1], operation, density);
    let mut shapes = Vec::new();

    // A mask is the FIRST leaf of its scope, so the accumulator is empty when it runs:
    // `A ∩ ∅ = ∅` must annihilate the scope's body without touching what lies beneath it.
    shapes.push((
        "a mask opens the scope",
        with_minted_ids(Scene::from_nodes(vec![
            NodeBuilder::Leaf(ground.clone()),
            NodeBuilder::group(
                "Part",
                vec![
                    NodeBuilder::Leaf(cut(CombineOp::Intersect)),
                    NodeBuilder::Leaf(placed.clone()),
                ],
            ),
        ])),
    ));

    // The scope itself closes under Subtract: its composed body — the placed leaf carved by its
    // own sibling — becomes one cutter against the ground beneath it.
    {
        let mut scene = Scene::from_nodes(vec![
            NodeBuilder::Leaf(ground.clone()),
            NodeBuilder::group(
                "Cut",
                vec![
                    NodeBuilder::Leaf(placed.clone()),
                    NodeBuilder::Leaf(cut(CombineOp::Subtract)),
                ],
            ),
        ]);
        let group = scene.roots[1];
        set_operation(&mut scene, group, CombineOp::Subtract);
        shapes.push(("the scope closes under Subtract", with_minted_ids(scene)));
    }

    // Three levels with a mask at each closing edge: the stack must open and close in the right
    // order, and the innermost surviving body must still be the node that names the cell.
    {
        let mut scene = Scene::from_nodes(vec![
            NodeBuilder::Leaf(ground.clone()),
            NodeBuilder::group(
                "Outer",
                vec![
                    NodeBuilder::Leaf(box_at([3, 3, 3], [0, 0, 0], CombineOp::Union, density)),
                    NodeBuilder::group(
                        "Middle",
                        vec![
                            NodeBuilder::Leaf(placed.clone()),
                            NodeBuilder::group(
                                "Inner",
                                vec![NodeBuilder::Leaf(cut(CombineOp::Union))],
                            ),
                        ],
                    ),
                ],
            ),
        ]);
        let outer = scene.roots[1];
        let middle = child_of(&scene, outer, 1);
        let inner = child_of(&scene, middle, 1);
        set_operation(&mut scene, middle, CombineOp::Intersect);
        set_operation(&mut scene, inner, CombineOp::Subtract);
        shapes.push(("three scopes deep, mixed ops", with_minted_ids(scene)));
    }

    shapes
}

#[test]
fn every_scope_ordering_folds_ownership_the_way_it_folds_occupancy() {
    let mut owned = 0;
    for (placement_case, rotation, slide) in adversarial_placements() {
        let placed = placed_box(
            [2, 3, 2],
            [1, 0, 0],
            CombineOp::Union,
            rotation,
            slide,
            DENSITY,
        );
        for (shape_case, scene) in scope_shapes(&placed, DENSITY) {
            owned += assert_the_pick_names_the_owner(
                &scene,
                DENSITY,
                &format!("{shape_case} / {placement_case}"),
            );
        }
    }
    assert!(
        owned > 0,
        "every scope ordering annihilated its scene, so the sweep proved nothing"
    );
}

/// A PRE-COMPOSED scope (forced by an outset on the Group — ADR 0019 Decision 7) is ONE leaf
/// to the fold, so ownership inside it is not answered by the fold at all: it goes to
/// `CompositeProducer::origin_at_point`. That function and the composite's own `material_at`
/// are two hand-written mirrors of the same rule — last-inside-Union, else nearest — and the
/// oracle stamps a composite per-voxel through `material_at` while the pick reads `origin_at`.
/// The differential is therefore the exact probe for the two drifting apart, and the OUTSET
/// SHELL is where it bites hardest, because out there "nearest" is the only rule in play.
///
/// The members carry no rotation or sub-voxel slide on purpose: `CompositeMember` keeps only an
/// integer `offset_voxels`, so a placed member is a pre-existing upstream gap rather than this
/// resolver's, and sweeping one here would fail for the wrong reason.
#[test]
fn a_pre_composed_scope_agrees_with_the_material_its_shell_wears() {
    for outset_voxels in [1, 3] {
        for gap_blocks in [0, 1, 3] {
            let mut scene = Scene::from_nodes(vec![NodeBuilder::group(
                "Part",
                vec![
                    NodeBuilder::Leaf(box_at([2, 2, 2], [0, 0, 0], CombineOp::Union, DENSITY)),
                    NodeBuilder::Leaf(box_at(
                        [2, 2, 2],
                        [2 + gap_blocks, 0, 0],
                        CombineOp::Union,
                        DENSITY,
                    )),
                ],
            )]);
            let group = scene.roots[0];
            scene
                .node_by_id_mut(group)
                .expect("the group resolves")
                .outset = voxel_core::units::Measurement::from_voxels(outset_voxels);
            let scene = with_minted_ids(scene);
            assert!(
                assert_the_pick_names_the_owner(
                    &scene,
                    DENSITY,
                    &format!("outset {outset_voxels}, gap {gap_blocks} blocks"),
                ) > 0
            );
        }
    }
}

/// ADR 0008 wandering origin: the whole sweep, re-run a hundred thousand blocks out. The
/// placement affine rebases in i64 before any f32 rotation, so ownership must be identical to
/// the same scenes at the origin — a differential is what proves that, since a hand-derived
/// expectation far out is just the origin expectation plus an offset.
#[test]
fn the_sweep_holds_a_hundred_thousand_blocks_out() {
    let far_blocks = 100_000;
    let mut owned = 0;
    for (case, rotation, slide) in adversarial_placements() {
        let scene = with_minted_ids(Scene::from_nodes(vec![
            box_at([3, 3, 3], [far_blocks, 0, 0], CombineOp::Union, DENSITY),
            placed_box(
                [2, 2, 2],
                [far_blocks + 1, 0, 0],
                CombineOp::Union,
                rotation,
                slide,
                DENSITY,
            ),
            placed_box(
                [2, 2, 2],
                [far_blocks + 1, 1, 1],
                CombineOp::Subtract,
                rotation,
                slide,
                DENSITY,
            ),
        ]));
        owned += assert_the_pick_names_the_owner(&scene, DENSITY, &format!("far {case}"));
    }
    assert!(
        owned > 0,
        "the far sweep resolved nothing, so it proved nothing"
    );
}
