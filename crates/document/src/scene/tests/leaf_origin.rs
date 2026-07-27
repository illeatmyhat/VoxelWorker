use super::*;
use voxel_core::core_geom::MaterialChoice;

// ---- ADR 0032: the walk reports which node each body came from ----
//
// `LeafOrigin` answers the question a viewport pick asks: the ray lands on a voxel, the
// voxel belongs to a leaf, and the leaf must name a node the user can select. Two facts
// decide it — the leaf's own node, and the outermost `Instance` it was expanded under.
// These tests pin the second, because it is the one with rules: an instance picks as
// ITSELF (selecting a node inside a definition would address geometry shared by every
// other placement of it), the OUTERMOST instance wins, and a FIXTURE expansion — which
// pushes no `ScopeFrame` and is otherwise indistinguishable from a host leaf — redirects
// exactly like a sealed one.

const OUTER_DEF: DefId = DefId(41);
const INNER_DEF: DefId = DefId(42);

/// The `origin` of every leaf a scene walks, in document order.
fn walked_origins(scene: &Scene) -> Vec<LeafOrigin> {
    scene
        .leaf_producers(DENSITY)
        .into_iter()
        .map(|leaf| leaf.origin)
        .collect()
}

/// The child node ids a definition was registered with.
fn definition_children(scene: &Scene, def_id: DefId) -> Vec<NodeId> {
    scene
        .definitions
        .iter()
        .find(|def| def.id == def_id)
        .expect("the definition was registered")
        .children
        .clone()
}

fn stone_box(offset_blocks: [i64; 3]) -> Node {
    box_tool(
        [2, 2, 2],
        offset_blocks,
        MaterialChoice::Stone,
        CombineOp::Union,
    )
}

/// A body authored directly in the scene is under no instance, so it picks as itself.
#[test]
fn an_authored_leaf_picks_as_its_own_node() {
    let scene = with_minted_ids(Scene::from_nodes(vec![
        stone_box([0, 0, 0]),
        stone_box([4, 0, 0]),
    ]));

    assert_eq!(
        walked_origins(&scene),
        vec![
            LeafOrigin::authored(scene.roots[0]),
            LeafOrigin::authored(scene.roots[1]),
        ],
        "a top-level leaf's origin is its own node, under no instance host"
    );
}

/// A SEALED definition's bodies redirect to the instance that placed them: the definition's
/// children are shared by every placement, so naming one would address all of them.
#[test]
fn a_sealed_instances_bodies_pick_as_the_instance() {
    let mut scene = Scene::from_nodes(vec![instance_node(
        INNER_DEF,
        [0, 0, 0],
        CombineOp::Union,
        "A",
    )]);
    scene.add_definition(
        INNER_DEF,
        "Part",
        vec![stone_box([0, 0, 0]), stone_box([4, 0, 0])],
    );
    let scene = with_minted_ids(scene);
    let instance = scene.roots[0];

    let picked: Vec<NodeId> = walked_origins(&scene)
        .into_iter()
        .map(LeafOrigin::picked_node)
        .collect();
    assert_eq!(
        picked,
        vec![instance, instance],
        "both definition bodies pick as the ONE instance that placed them (ADR 0017)"
    );
}

/// A FIXTURE definition pushes no `ScopeFrame` (ADR 0017 Decision 4) — its children splice
/// into the host's fold as if authored there. The redirect must still happen, or a spliced
/// body would be indistinguishable from a host leaf and pick into the shared definition.
#[test]
fn a_fixture_instances_bodies_still_pick_as_the_instance() {
    let mut scene = Scene::from_nodes(vec![
        stone_box([8, 0, 0]),
        instance_node(INNER_DEF, [0, 0, 0], CombineOp::Union, "W"),
    ]);
    scene.add_definition(INNER_DEF, "Window", vec![stone_box([0, 0, 0])]);
    assert!(scene.set_definition_fixture(INNER_DEF, true));
    let scene = with_minted_ids(scene);
    let (host, instance) = (scene.roots[0], scene.roots[1]);

    let spliced_body = definition_children(&scene, INNER_DEF)[0];
    assert_eq!(
        walked_origins(&scene),
        vec![
            LeafOrigin::authored(host),
            LeafOrigin {
                node: spliced_body,
                instance_host: Some(instance)
            },
        ],
        "the spliced fixture body keeps its own node identity AND names the instance"
    );
}

/// The OUTERMOST instance wins. A definition containing another instance is still reached
/// through the one placement the user can see — redirecting to the inner instance would
/// select a node inside a definition, which is the thing the rule exists to prevent.
#[test]
fn a_nested_instance_picks_as_the_outermost_placement() {
    let mut scene = Scene::from_nodes(vec![instance_node(
        OUTER_DEF,
        [0, 0, 0],
        CombineOp::Union,
        "Bracket",
    )]);
    scene.add_definition(INNER_DEF, "Bolt", vec![stone_box([0, 0, 0])]);
    scene.add_definition(
        OUTER_DEF,
        "Bracket",
        vec![
            NodeBuilder::Leaf(stone_box([4, 0, 0])),
            NodeBuilder::Leaf(instance_node(
                INNER_DEF,
                [0, 0, 0],
                CombineOp::Union,
                "Bolt",
            )),
        ],
    );
    let scene = with_minted_ids(scene);
    let placement = scene.roots[0];

    let picked: Vec<NodeId> = walked_origins(&scene)
        .into_iter()
        .map(LeafOrigin::picked_node)
        .collect();
    assert_eq!(
        picked,
        vec![placement, placement],
        "the bracket's own body AND the bolt nested two definitions deep both pick as the \
         one placement in the scene"
    );
}
