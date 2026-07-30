use super::*;
use voxel_core::core_geom::MaterialChoice;

// ---- ADR 0032: which node a viewport pick selects ----
//
// The raycast names a solid ABSOLUTE voxel; the resolver names the node. The law under
// test is "the pick follows the material": the answer is the last purely-additive leaf in
// document order covering the cell, which is exactly the leaf whose material the resolve
// stamped there (ADR 0017, later-wins). The interesting cases are the ones where a
// nearest-hit or first-hit rule would give a DIFFERENT answer — an overlap, the wall of a
// carved cavity, and a body reached through a definition or a pre-composed scope.

const CUTTER_DEF: DefId = DefId(51);

/// A `size_blocks`-cube Union body of `material` at `offset_blocks`.
fn body(size_blocks: u32, offset_blocks: [i64; 3], material: MaterialChoice) -> Node {
    box_tool([size_blocks; 3], offset_blocks, material, CombineOp::Union)
}

fn picked(scene: &Scene, absolute_voxel: [i64; 3]) -> Option<NodeId> {
    scene.picked_node_at_voxel(absolute_voxel, DENSITY)
}

/// The baseline: a click on a body selects it, and a click on empty space selects nothing.
#[test]
fn a_solid_cell_picks_its_body_and_empty_space_picks_nothing() {
    let scene = with_minted_ids(Scene::from_nodes(vec![body(
        2,
        [0, 0, 0],
        MaterialChoice::Stone,
    )]));

    assert_eq!(picked(&scene, [8, 8, 8]), Some(scene.roots[0]));
    assert_eq!(
        picked(&scene, [64, 8, 8]),
        None,
        "a cell no leaf covers belongs to no node"
    );
}

/// Where two Union bodies overlap, the LATER one owns the cell — the same later-wins rule
/// that decided the material the user can see there (ADR 0017). A first-hit scan would
/// return the earlier body and select something the click does not look like.
#[test]
fn overlapping_bodies_pick_the_later_one() {
    let scene = with_minted_ids(Scene::from_nodes(vec![
        body(2, [0, 0, 0], MaterialChoice::Stone),
        body(2, [1, 0, 0], MaterialChoice::Wood),
    ]));
    let (earlier, later) = (scene.roots[0], scene.roots[1]);

    assert_eq!(
        picked(&scene, [12, 8, 8]),
        Some(later),
        "in the overlap the later body wins, exactly as its material does"
    );
    assert_eq!(
        picked(&scene, [4, 8, 8]),
        Some(earlier),
        "outside the overlap the earlier body still owns its own cells"
    );
    assert_eq!(picked(&scene, [20, 8, 8]), Some(later));
}

/// The cavity case, and the reason booleans are excluded. A `Subtract` cutter's surface IS
/// the wall of the hole it carved, so every cell on that wall lies on the cutter's boundary
/// too — a nearest-hit rule would hand back the cutter for every click inside the cavity.
#[test]
fn the_wall_of_a_carved_cavity_picks_the_host_not_the_cutter() {
    let mut scene = Scene::from_nodes(vec![
        body(4, [0, 0, 0], MaterialChoice::Stone),
        box_tool(
            [2, 2, 2],
            [1, 1, 1],
            MaterialChoice::Plain,
            CombineOp::Subtract,
        ),
    ]);
    scene = with_minted_ids(scene);
    let host = scene.roots[0];

    // The cutter spans absolute voxels [8, 24)³; the cell just outside it on the low side
    // is host wall, and it sits exactly on the cutter's surface.
    assert_eq!(
        picked(&scene, [7, 12, 12]),
        Some(host),
        "the cavity wall belongs to the body that was carved, never to the cutter"
    );
    assert_eq!(
        picked(&scene, [12, 12, 12]),
        None,
        "a cell inside the cavity is not solid, so it belongs to no node"
    );
}

/// A body reached through an `Instance` picks as the INSTANCE (ADR 0017 / ADR 0032):
/// selecting the definition's child would address geometry shared by every placement.
#[test]
fn an_instanced_body_picks_as_its_placement() {
    let mut scene = Scene::from_nodes(vec![instance_node(
        CUTTER_DEF,
        [4, 0, 0],
        CombineOp::Union,
        "Placement",
    )]);
    scene.add_definition(
        CUTTER_DEF,
        "Block",
        vec![body(2, [0, 0, 0], MaterialChoice::Stone)],
    );
    let scene = with_minted_ids(scene);

    assert_eq!(
        picked(&scene, [40, 8, 8]),
        Some(scene.roots[0]),
        "the instance answers for the definition body it placed"
    );
}

/// A PRE-COMPOSED scope is one leaf to the walk (here forced by an outset on the Group, ADR
/// 0019 Decision 7). The pick must still descend into its members and name the authored
/// leaf — ADR 0032 picks the leaf at any depth, and a single top-level `Emboss` would
/// otherwise collapse a whole document into one unpickable body.
#[test]
fn a_pre_composed_scope_picks_its_inner_member() {
    let mut scene = Scene::from_nodes(vec![NodeBuilder::group(
        "Part",
        vec![
            body(2, [0, 0, 0], MaterialChoice::Stone).into(),
            body(2, [2, 0, 0], MaterialChoice::Wood).into(),
        ],
    )]);
    let group_id = scene.roots[0];
    // An outset on the scope is what triggers pre-composition; two voxels keeps the shell
    // thin enough that the cells asserted below sit inside the members themselves.
    scene
        .node_by_id_mut(group_id)
        .expect("the group resolves")
        .outset = parametric::units::Measurement::from_voxels(2);
    let scene = with_minted_ids(scene);
    let members = match &scene
        .node_by_id(group_id)
        .expect("the group resolves")
        .content
    {
        NodeContent::Group(children) => children.clone(),
        other => panic!("expected a Group, got {other:?}"),
    };

    assert_eq!(
        picked(&scene, [8, 8, 8]),
        Some(members[0]),
        "a click inside the first member names that member, not the composed Part"
    );
    assert_eq!(
        picked(&scene, [24, 8, 8]),
        Some(members[1]),
        "and a click inside the second names the second"
    );
}

/// ADR 0027: a sub-voxel seat puts the leaf OUT OF PHASE with the absolute lattice, which
/// routes coverage through the placement affine instead of a plain translation. The pick
/// must take the same branch the resolve does, or a nudged body stops being clickable.
#[test]
fn a_sub_voxel_seated_body_is_still_pickable() {
    let mut node = body(2, [0, 0, 0], MaterialChoice::Stone);
    node.transform.offset_local_voxels = [0.5, 0.0, 0.0];
    let scene = with_minted_ids(Scene::from_nodes(vec![node]));

    assert_eq!(
        picked(&scene, [8, 8, 8]),
        Some(scene.roots[0]),
        "a half-voxel slide changes where the body's cells land, not whether they pick"
    );
}

/// ADR 0008 wandering origin: a body placed far from the world origin keeps full precision,
/// because the placement affine rebases in i64 before any f32 rotation math. A pick a
/// million blocks out must name the same node a pick at the origin does.
#[test]
fn a_far_placed_body_picks_the_same_way() {
    let far_blocks = 100_000;
    let scene = with_minted_ids(Scene::from_nodes(vec![body(
        2,
        [far_blocks, 0, 0],
        MaterialChoice::Stone,
    )]));

    let far_voxel = far_blocks * DENSITY as i64;
    assert_eq!(picked(&scene, [far_voxel + 8, 8, 8]), Some(scene.roots[0]));
    assert_eq!(picked(&scene, [far_voxel - 8, 8, 8]), None);
}

/// A `Subtract` SIBLING inside a scope carves that scope's composed body before it folds
/// into the parent, so a cell it removes shows the material of whatever lies BENEATH the
/// scope. The pick must follow: naming the carved-away leaf would select a node with no
/// visible geometry at the click.
#[test]
fn a_scope_sibling_carve_hands_the_cell_to_the_body_beneath() {
    let mut scene = Scene::from_nodes(vec![
        NodeBuilder::leaf(body(2, [0, 0, 0], MaterialChoice::Stone)),
        NodeBuilder::group(
            "Part",
            vec![
                body(2, [0, 0, 0], MaterialChoice::Wood).into(),
                box_tool(
                    [2, 2, 2],
                    [0, 0, 0],
                    MaterialChoice::Plain,
                    CombineOp::Subtract,
                )
                .into(),
            ],
        ),
    ]);
    scene = with_minted_ids(scene);
    let beneath = scene.roots[0];

    assert_eq!(
        picked(&scene, [8, 8, 8]),
        Some(beneath),
        "the group's body was fully carved by its own sibling, so the cell is the \
         root-level body's"
    );
}

/// ADR 0026/0027: an axis-aligned turn is a real rotation — the display emits the TURNED
/// cells. A pick that ignored it would answer for the unturned footprint: dead clicks on the
/// visible body, live clicks in the empty air beside it.
#[test]
fn a_quarter_turned_body_picks_on_its_turned_footprint() {
    let mut node = box_tool(
        [4, 2, 2],
        [0, 0, 0],
        MaterialChoice::Stone,
        CombineOp::Union,
    );
    node.transform.rotation_quaternion =
        Some(glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2).to_array());
    let scene = with_minted_ids(Scene::from_nodes(vec![node]));

    // Unturned the body spans 32x16x16 voxels; the quarter turn about Z swaps X and Y, so
    // the placed body spans 16x32x16 from the same low corner.
    assert_eq!(
        picked(&scene, [8, 24, 8]),
        Some(scene.roots[0]),
        "a cell the turned body covers is pickable"
    );
    assert_eq!(
        picked(&scene, [24, 8, 8]),
        None,
        "a cell only the UNTURNED footprint would have covered is empty air"
    );
}

/// A click on a Part's OUTSET SHELL — cells the dilation added, which no member's body
/// reaches — selects the member the shell grew from, because that is whose material the
/// shell wears (ADR 0019 Decision 7). The scope node carrying the outset is one navigation
/// step away, not a different pick.
#[test]
fn the_outset_shell_of_a_part_picks_the_member_beneath_it() {
    let mut scene = Scene::from_nodes(vec![NodeBuilder::group(
        "Part",
        vec![body(2, [0, 0, 0], MaterialChoice::Stone).into()],
    )]);
    let group_id = scene.roots[0];
    scene
        .node_by_id_mut(group_id)
        .expect("the group resolves")
        .outset = parametric::units::Measurement::from_voxels(3);
    let scene = with_minted_ids(scene);
    let member = match &scene
        .node_by_id(group_id)
        .expect("the group resolves")
        .content
    {
        NodeContent::Group(children) => children[0],
        other => panic!("expected a Group, got {other:?}"),
    };

    // The member spans absolute voxels [0, 16); the shell reaches 3 further on every side.
    assert_eq!(
        picked(&scene, [-2, 8, 8]),
        Some(member),
        "a shell cell belongs to the member it grew from"
    );
    assert_eq!(
        picked(&scene, [-5, 8, 8]),
        None,
        "past the shell there is nothing"
    );
}
