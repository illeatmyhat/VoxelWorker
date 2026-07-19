use super::*;
use voxel_core::core_geom::MaterialChoice;
use crate::sketch::SketchSolid;
use voxel_core::voxel::ShapeKind;
use crate::voxel::GeometryParams;
use crate::voxel::SdfShape;

    // ---- S0: chunk-addressable resolve (issue #27) ---------------------------
    //
    // These tests prove the ADDITIVE chunked resolve path reconstructs EXACTLY
    // what the monolithic `resolve_region` produces, after normalising for the
    // recentre offset that `resolve_region` applies and the chunk path does not.
    // The render path (`resolve_region`) is untouched; only these new functions
    // are exercised.


    /// Assert the chunk-reassembled occupied set EXACTLY equals the monolithic
    /// `resolve_region`'s set (position + material), after recentre normalisation,
    /// AND that no chunk emits a voxel outside its own chunk AABB.
    fn assert_chunked_matches_monolithic(scene: &Scene, voxels_per_block: u32, label: &str) {
        let monolithic = scene.resolve_region(
            scene.full_extent_blocks(voxels_per_block),
            voxels_per_block,
            0,
        );
        let chunked = scene.resolve_region_via_chunks(voxels_per_block, 0);

        let recentre = scene.recentre_voxels(voxels_per_block);
        let monolithic_set = occupied_multiset(&monolithic, recentre);
        let chunked_set = occupied_multiset(&chunked, [0, 0, 0]);

        assert_eq!(
            chunked_set, monolithic_set,
            "[{label}] chunked occupied set must equal monolithic resolve (recentre-normalised)"
        );
        // Cross-check the count too (a multiset equality already implies it, but
        // this pins the failure message to the simplest symptom first).
        assert_eq!(
            chunked.occupied_count(),
            monolithic.occupied_count(),
            "[{label}] chunked occupied count must equal monolithic"
        );

        // Each per-chunk resolve must keep every voxel inside its OWN chunk AABB
        // (exactly-one-chunk ownership). Walk the covering range and re-resolve.
        let chunk_extent_voxels =
            (voxel_core::core_geom::CHUNK_BLOCKS * voxels_per_block.max(1)) as i32;
        if let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(voxels_per_block) {
            let mut total_from_chunks = 0usize;
            for chunk_z in min_chunk[2]..=max_chunk[2] {
                for chunk_y in min_chunk[1]..=max_chunk[1] {
                    for chunk_x in min_chunk[0]..=max_chunk[0] {
                        let chunk_coord = [chunk_x, chunk_y, chunk_z];
                        let chunk = scene.resolve_chunk(chunk_coord, voxels_per_block, 0);
                        total_from_chunks += chunk.occupied_count();
                        for voxel in &chunk.occupied {
                            let world_position = voxel.world_position();
                            for axis in 0..3 {
                                let lo = (chunk_coord[axis] * chunk_extent_voxels) as f32;
                                let hi = lo + chunk_extent_voxels as f32;
                                let position = world_position[axis];
                                assert!(
                                    position >= lo && position < hi,
                                    "[{label}] voxel {position} on axis {axis} escaped chunk \
                                     {chunk_coord:?} box [{lo}, {hi})"
                                );
                            }
                        }
                    }
                }
            }
            // Every monolithic voxel is accounted for by exactly one chunk (no
            // double-counting, no drops): the chunk total equals the whole count.
            assert_eq!(
                total_from_chunks,
                monolithic.occupied_count(),
                "[{label}] summed per-chunk counts must equal the monolithic count \
                 (each voxel in exactly one chunk)"
            );
        }
    }

    fn shape_scene(kind: ShapeKind, voxels_per_block: u32) -> Scene {
        Scene::from_geometry(
            GeometryParams {
                shape: kind,
                size_voxels: [5 * voxels_per_block, 5 * voxels_per_block, 5 * voxels_per_block],
                size_measurements: None,
                voxels_per_block,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        )
    }

    /// Single-shape parity, all five SDF kinds — mirrors the all-shapes coverage
    /// style. (Single-node zero-offset scenes also exercise the recentre
    /// normalisation, since `resolve_region` recentres even a lone node.)
    #[test]
    fn chunked_resolve_matches_monolithic_for_all_shapes() {
        for kind in [
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Tube,
            ShapeKind::Torus,
            ShapeKind::Box,
        ] {
            let scene = shape_scene(kind, 16);
            assert_chunked_matches_monolithic(&scene, 16, &format!("{kind:?}"));
        }
    }

    /// A multi-node placed scene (the `--demo-scene` shape: a Sphere + an offset
    /// Box + an offset Torus, three materials) — proves the chunked path composes
    /// several leaves at distinct offsets and materials.
    #[test]
    fn chunked_resolve_matches_monolithic_for_demo_scene() {
        let voxels_per_block = 16;
        let make_tool = |kind, offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, voxels_per_block);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let scene = scene_with_top_level_selected(
            Scene::from_nodes(vec![
                make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
                make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
                make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
            ]),
            0,
        );
        assert_chunked_matches_monolithic(&scene, voxels_per_block, "demo-scene");
    }

    /// The `--demo-village` scene: four `Instance`s of one `House` definition (a
    /// Box body + a Cylinder chimney `Group`) — proves the chunked path follows
    /// instance + group transform composition (reuse-by-reference).
    #[test]
    fn chunked_resolve_matches_monolithic_for_demo_village() {
        let voxels_per_block = 16;
        let house_def_id = DefId(1);
        let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, size, 1, voxels_per_block);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let instance = |name: &str, offset: [i64; 3]| {
            let mut node = Node::new(name, NodeContent::Instance(house_def_id));
            node.transform = NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let mut scene_build = Scene::from_nodes(vec![
            instance("House 1", [0, 0, 0]),
            instance("House 2", [6, 0, 0]),
            instance("House 3", [12, 0, 0]),
            instance("House 4", [18, 0, 0]),
        ]);
        scene_build.add_definition(
            house_def_id,
            "House".to_string(),
            vec![
                tool(ShapeKind::Box, [2, 2, 2], [0, 0, 0], MaterialChoice::Stone),
                tool(ShapeKind::Cylinder, [1, 2, 1], [0, 2, 0], MaterialChoice::Wood),
            ],
        );
        let scene = scene_with_top_level_selected(scene_build, 0);
        assert_chunked_matches_monolithic(&scene, voxels_per_block, "demo-village");
    }

    /// ADR 0003 §3i Slice 2a: the new sketch→extrude producer composes through the
    /// chunked resolve identically to the monolithic one — mirrors the SDF parity
    /// harness for a SketchTool leaf. Two cases: a plain rectangle extrude (the box
    /// sugar) and a concave L-shape extrude (the added-value path), both at the app
    /// density and at an off-origin placement so the recentre/cover math is real.
    #[test]
    fn chunked_resolve_matches_monolithic_for_sketch_extrude() {
        use crate::sketch::{PlaneAxis, Sketch, SketchPoint};
        let voxels_per_block = 16;
        let density = voxels_per_block as i64;

        // (a) Rectangle extrude (box sugar), placed off-origin on X. Z-up:
        // footprint-extrude-up uses PlaneAxis::Z (profile in XY, extruded along +Z).
        let rect = SketchSolid::extrude(
            Sketch::rectangle(PlaneAxis::Z, 3 * density, 2 * density),
            2 * density as u32,
        );
        let mut rect_node = Node::new(
            "Sketch rect",
            NodeContent::SketchTool {
                producer: rect,
                material: MaterialChoice::Stone,
            },
        );
        rect_node.transform = NodeTransform::from_blocks([5, 0, 0], voxels_per_block);
        let rect_scene = Scene::single_node(rect_node);
        assert_chunked_matches_monolithic(&rect_scene, voxels_per_block, "sketch-rect");

        // (b) Concave L-shape extrude (the added value a box can't make).
        let two = 2 * density;
        let four = 4 * density;
        let l_profile = vec![
            SketchPoint::new(0, 0),
            SketchPoint::new(four, 0),
            SketchPoint::new(four, two),
            SketchPoint::new(two, two),
            SketchPoint::new(two, four),
            SketchPoint::new(0, four),
        ];
        let l_extrude =
            SketchSolid::extrude(Sketch::new(PlaneAxis::Z, l_profile), 3 * density as u32);
        let mut l_node = Node::new(
            "Sketch L",
            NodeContent::SketchTool {
                producer: l_extrude,
                material: MaterialChoice::Wood,
            },
        );
        // Off-origin (crossing chunk boundaries on both in-plane axes X and Y) so the
        // off-origin chunked path is proven on the concave/reflex shape, not just the
        // convex rectangle above. (Z-up: the L footprint lives in the XY ground plane.)
        l_node.transform = NodeTransform::from_blocks([5, 5, 0], voxels_per_block);
        let l_scene = Scene::single_node(l_node);
        assert_chunked_matches_monolithic(&l_scene, voxels_per_block, "sketch-L");
    }

    /// ADR 0003 §3i: the revolve operation composes through the chunked resolve
    /// identically to the monolithic one — mirrors the extrude parity harness for a
    /// solid of revolution. A rectangle revolved 360° about Z (a cylinder) placed
    /// off-origin on X+Y so the recentre/cover math is real and the disc crosses
    /// chunk boundaries on both radial axes.
    #[test]
    fn chunked_resolve_matches_monolithic_for_sketch_revolve() {
        use crate::sketch::{PlaneAxis, RevolveAxis, Sketch};
        let voxels_per_block = 16;
        let density = voxels_per_block as i64;

        // PlaneAxis::X + RevolveAxis::InPlane1 ⇒ axial = Z (vertical), radial = {X, Y}.
        // (a) Profile (radial, axial) = rectangle(radial = 2 blocks, axial = 3 blocks)
        // ⇒ a 4-block-diameter, 3-block-tall cylinder. EVEN radial + whole-block axial.
        let revolve = SketchSolid::revolve(
            Sketch::rectangle(PlaneAxis::X, 2 * density, 3 * density),
            RevolveAxis::InPlane1,
            360,
        );
        let mut node = Node::new(
            "Sketch revolve",
            NodeContent::SketchTool {
                producer: revolve,
                material: MaterialChoice::Stone,
            },
        );
        // Off-origin so the covering chunk range and recentre offset are non-trivial.
        node.transform = NodeTransform::from_blocks([5, 5, 0], voxels_per_block);
        let scene = Scene::single_node(node);
        assert_chunked_matches_monolithic(&scene, voxels_per_block, "sketch-revolve");

        // (b) ODD axial extent (NOT a whole number of blocks) with an even radial, so
        // the even-radial diameter + odd-axial block-rounding combo is exercised through
        // the chunked path. Radial 30 voxels (diameter 60), axial 2·16 + 5 = 37 voxels.
        let revolve_odd_axial = SketchSolid::revolve(
            Sketch::rectangle(PlaneAxis::X, 30, 2 * density + 5),
            RevolveAxis::InPlane1,
            360,
        );
        let mut odd_node = Node::new(
            "Sketch revolve odd axial",
            NodeContent::SketchTool {
                producer: revolve_odd_axial,
                material: MaterialChoice::Wood,
            },
        );
        odd_node.transform = NodeTransform::from_blocks([5, 5, 0], voxels_per_block);
        let odd_scene = Scene::single_node(odd_node);
        assert_chunked_matches_monolithic(&odd_scene, voxels_per_block, "sketch-revolve-odd-axial");
    }

    /// A scene with a single node shifted well OFF the origin (+8 blocks on X) —
    /// proves the chunked path handles off-centre placement (the AABB does not
    /// start at the origin, so the covering chunk range is non-trivial and the
    /// recentre offset is non-zero).
    #[test]
    fn chunked_resolve_matches_monolithic_for_offset_node() {
        let voxels_per_block = 16;
        let shape = SdfShape::from_blocks(ShapeKind::Sphere, [4, 4, 4], 1, voxels_per_block);
        let mut node = Node::new(
            "Offset sphere",
            NodeContent::Tool {
                shape,
                material: MaterialChoice::Wood,
            },
        );
        node.transform = NodeTransform::from_blocks([8, 0, 0], voxels_per_block);
        let scene = Scene::single_node(node);

        // Sanity: the recentre is genuinely non-zero for this off-centre scene, so
        // the normalisation is actually exercised (a zero recentre would make the
        // test vacuous on that axis).
        let recentre = scene.recentre_voxels(voxels_per_block);
        assert_ne!(
            recentre, [0, 0, 0],
            "an off-centre node must produce a non-zero recentre (else the \
             normalisation is untested)"
        );
        assert_chunked_matches_monolithic(&scene, voxels_per_block, "offset-node");
    }

    /// A chunk that no leaf overlaps resolves to an EMPTY grid (no panic), and its
    /// dimensions are still one chunk's extent.
    #[test]
    fn empty_chunk_resolves_to_empty_grid() {
        let scene = shape_scene(ShapeKind::Sphere, 16);
        // A chunk far outside the (origin-area) composite AABB.
        let chunk = scene.resolve_chunk([1000, 1000, 1000], 16, 0);
        assert_eq!(chunk.occupied_count(), 0, "a far-off chunk must be empty");
        let chunk_extent = voxel_core::core_geom::CHUNK_BLOCKS * 16;
        assert_eq!(
            chunk.dimensions,
            [chunk_extent, chunk_extent, chunk_extent],
            "an empty chunk still reports one chunk's voxel extent"
        );
    }

    /// Parity holds at a non-default density too (16 is the app default; this pins
    /// that the chunk-extent / ownership math is density-correct).
    #[test]
    fn chunked_resolve_matches_monolithic_at_density_8() {
        let scene = shape_scene(ShapeKind::Torus, 8);
        assert_chunked_matches_monolithic(&scene, 8, "torus@8");
    }


    // ---- Outset: the per-node dilation (ADR 0019 Decision 7) ------------------

    /// A box with an outset of `N` resolves to the box grown by `N` on EVERY side.
    ///
    /// This is the whole feature in one assertion. A box is the case where the answer is
    /// exact and countable: `SdfShape::Box` measures in Chebyshev, and a Chebyshev dilation
    /// of a box IS a box, so `S³` voxels must become `(S + 2N)³` — no approximation to hide
    /// an off-by-one in.
    #[test]
    fn outset_grows_a_box_by_the_outset_on_every_side() {
        let voxels_per_block = 16;
        let size_blocks = 2u32;
        let side = size_blocks * voxels_per_block;

        let resolved_side = |outset_voxels: i64| {
            let shape =
                SdfShape::from_blocks(ShapeKind::Box, [size_blocks; 3], 1, voxels_per_block);
            let mut node =
                Node::new("Box", NodeContent::Tool { shape, material: MaterialChoice::Stone });
            node.outset = voxel_core::units::Measurement::from_voxels(outset_voxels);
            let scene = Scene::from_nodes(vec![node]);
            let grid =
                scene.resolve_region(scene.full_extent_blocks(voxels_per_block), voxels_per_block, 0);
            let count = grid.occupied_count() as f64;
            // Cube-root back to a side length so a failure reads as "37 instead of 40"
            // rather than as a six-digit voxel count.
            count.cbrt().round() as i64
        };

        assert_eq!(resolved_side(0), side as i64, "an outset of 0 must not change the body");
        assert_eq!(
            resolved_side(4),
            side as i64 + 8,
            "an outset of 4 voxels must grow the box by 4 on each of the two sides per axis"
        );
        assert_eq!(
            resolved_side(-4),
            side as i64 - 8,
            "a NEGATIVE outset must INSET, shrinking the box by 4 on each side (ADR 0019 \
             Decision 7 — this is how a deliberate gap between chiselled pieces is authored)"
        );
    }

    /// The chunked and monolithic resolves must agree under outset too.
    ///
    /// The chunked path skips leaves whose AABB misses the chunk, and that skip is only
    /// bit-identical to stamping-then-clipping if the AABB it tests is the DILATED one. A
    /// cutter whose outset reaches into a chunk its own bounds miss would otherwise be
    /// skipped there and its mask silently lost — visible as geometry that survives in one
    /// chunk and not its neighbour.
    #[test]
    fn chunked_resolve_matches_monolithic_under_outset() {
        let voxels_per_block = 8;
        let mut solid = Node::new(
            "Solid",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Box, [4, 4, 4], 1, voxels_per_block),
                material: MaterialChoice::Stone,
            },
        );
        solid.outset = voxel_core::units::Measurement::from_voxels(3);

        // An outset SUBTRACT cutter: the dilation must carve MORE than the cutter's own
        // bounds, in both paths, including across a chunk seam.
        let mut cutter = Node::new(
            "Cutter",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Sphere, [2, 2, 2], 1, voxels_per_block),
                material: MaterialChoice::Wood,
            },
        );
        cutter.operation = CombineOp::Subtract;
        cutter.outset = voxel_core::units::Measurement::from_voxels(2);
        cutter.transform = NodeTransform::from_blocks([1, 0, 0], voxels_per_block);

        let scene = scene_with_top_level_selected(Scene::from_nodes(vec![solid, cutter]), 0);
        assert_chunked_matches_monolithic(&scene, voxels_per_block, "outset-union-and-subtract");
    }

    /// An outset SUBTRACT cutter removes strictly more than the same cutter without one.
    ///
    /// This is the authoring case outset exists for — clearance around a chiselled cut — and
    /// it pins the DIRECTION of the dilation. A sign error would still round-trip between the
    /// two resolve paths and still bracket soundly; only comparing against the undilated cut
    /// catches it.
    #[test]
    fn an_outset_cutter_carves_more_than_an_undilated_one() {
        let voxels_per_block = 8;
        let carved_count = |outset_voxels: i64| {
            let solid = Node::new(
                "Solid",
                NodeContent::Tool {
                    shape: SdfShape::from_blocks(ShapeKind::Box, [4, 4, 4], 1, voxels_per_block),
                    material: MaterialChoice::Stone,
                },
            );
            let mut cutter = Node::new(
                "Cutter",
                NodeContent::Tool {
                    shape: SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, voxels_per_block),
                    material: MaterialChoice::Wood,
                },
            );
            cutter.operation = CombineOp::Subtract;
            cutter.outset = voxel_core::units::Measurement::from_voxels(outset_voxels);
            let scene = Scene::from_nodes(vec![solid, cutter]);
            scene
                .resolve_region(scene.full_extent_blocks(voxels_per_block), voxels_per_block, 0)
                .occupied_count()
        };

        let plain = carved_count(0);
        let dilated = carved_count(2);
        assert!(
            dilated < plain,
            "an outset cutter must remove MORE than an undilated one \
             (outset 2 left {dilated} voxels, outset 0 left {plain})"
        );
    }

    /// **An outset on a Part dilates the Part's COMPOSED body** (ADR 0019 Decision 7).
    ///
    /// A Part (`NodeContent::Group` — ADR 0018 Decision 1 names the composition container
    /// "Part") is a sealed scope: it pre-composes its children into one body. Its outset must
    /// therefore dilate that composed body, not each member separately. ADR 0019 Decision 7
    /// rejects leaf-only outset precisely so a reusable composed cutter can be given
    /// clearance as a whole.
    ///
    /// The two are NOT interchangeable. Dilation distributes over union, so for a
    /// pure-union Part the per-member and per-Part answers agree — but a Part with an
    /// internal `Subtract` diverges sharply: dilating members individually makes the inner
    /// cutter carve MORE, while dilating the composed Part grows the finished body and
    /// partly closes that cut. This test uses the union case for the baseline and the
    /// carved case for the distinction.
    #[test]
    fn an_outset_on_a_part_dilates_the_parts_composed_body() {
        let voxels_per_block = 8;
        let part_scene = |outset_voxels: i64, with_internal_cut: bool| {
            let mut children = vec![NodeBuilder::Leaf({
                let shape =
                    SdfShape::from_blocks(ShapeKind::Box, [4, 4, 4], 1, voxels_per_block);
                Node::new("Body", NodeContent::Tool { shape, material: MaterialChoice::Stone })
            })];
            if with_internal_cut {
                children.push(NodeBuilder::Leaf({
                    let shape =
                        SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, voxels_per_block);
                    let mut node = Node::new(
                        "Cut",
                        NodeContent::Tool { shape, material: MaterialChoice::Wood },
                    );
                    node.operation = CombineOp::Subtract;
                    node.transform = NodeTransform::from_blocks([3, 3, 3], voxels_per_block);
                    node
                }));
            }
            let mut scene = Scene::from_nodes(vec![NodeBuilder::group("Part", children)]);
            // The Part is the only top-level node, so path [0] is the group itself.
            let path = crate::scene::NodePath::from_indices(vec![0]);
            scene
                .node_at_path_mut(&path)
                .expect("the Part resolves at path [0]")
                .outset = voxel_core::units::Measurement::from_voxels(outset_voxels);
            scene
        };

        let occupancy = |scene: &Scene| {
            scene
                .resolve_region(scene.full_extent_blocks(voxels_per_block), voxels_per_block, 0)
                .occupied_count()
        };

        // A Part wrapping ONE box must dilate exactly like the box would: 32³ → 40³.
        let plain = occupancy(&part_scene(0, false));
        let dilated = occupancy(&part_scene(4, false));
        assert_eq!(plain, 32 * 32 * 32, "the undilated Part is a plain 4-block box");
        assert_eq!(
            dilated,
            40 * 40 * 40,
            "a Part with an outset of 4 must grow its composed body by 4 on every side"
        );

        // With an internal cut, dilating the COMPOSED body must partly close the notch, so
        // the result exceeds what dilating the members separately would leave.
        let carved_plain = occupancy(&part_scene(0, true));
        let carved_dilated = occupancy(&part_scene(4, true));
        assert!(
            carved_dilated > carved_plain,
            "an outset Part with an internal cut must still grow overall \
             ({carved_dilated} vs {carved_plain})"
        );
    }

    /// The outset shell of a mixed-material Part inherits the NEAREST member's material.
    ///
    /// Inside the body the later Union member still wins on overlap (ADR 0017), so an outset
    /// Part's interior is coloured exactly as the same Part at outset zero. Outside it there
    /// is no "later" to appeal to — no member contains the point — so the shell takes the
    /// material of the surface it grew from. Flattening the Part to one material would
    /// visibly recolour it the moment a user typed a clearance.
    #[test]
    fn an_outset_shell_takes_the_nearest_members_material() {
        let voxels_per_block = 8;
        let stone = {
            let shape = SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, voxels_per_block);
            Node::new("Stone", NodeContent::Tool { shape, material: MaterialChoice::Stone })
        };
        let wood = {
            let shape = SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, voxels_per_block);
            let mut node =
                Node::new("Wood", NodeContent::Tool { shape, material: MaterialChoice::Wood });
            node.transform = NodeTransform::from_blocks([6, 0, 0], voxels_per_block);
            node
        };
        let mut scene = Scene::from_nodes(vec![NodeBuilder::group(
            "Part",
            vec![NodeBuilder::Leaf(stone), NodeBuilder::Leaf(wood)],
        )]);
        let path = crate::scene::NodePath::from_indices(vec![0]);
        scene
            .node_at_path_mut(&path)
            .expect("the Part resolves at path [0]")
            .outset = voxel_core::units::Measurement::from_voxels(2);

        let grid =
            scene.resolve_region(scene.full_extent_blocks(voxels_per_block), voxels_per_block, 0);
        let stone_id = MaterialChoice::Stone.block_id();
        let wood_id = MaterialChoice::Wood.block_id();
        let stone_count = grid.occupied.iter().filter(|v| v.block_id == stone_id).count();
        let wood_count = grid.occupied.iter().filter(|v| v.block_id == wood_id).count();

        // Both materials must survive the dilation — a single-material Part would show one.
        assert!(stone_count > 0, "the dilated Part kept no Stone voxels");
        assert!(wood_count > 0, "the dilated Part kept no Wood voxels");
        // The two boxes are identical and symmetric about the gap, so their dilated shells
        // are too: neither material may dominate.
        let ratio = stone_count as f64 / wood_count as f64;
        assert!(
            (0.9..1.1).contains(&ratio),
            "symmetric members must dilate symmetrically \
             (stone {stone_count} vs wood {wood_count})"
        );
    }

    /// The chunked and monolithic resolves agree on a Part outset too.
    ///
    /// A composed scope arrives at the chunked path as a single leaf whose AABB is the
    /// composed-then-dilated box. If that box were wrong, the chunk-skip would drop the Part
    /// in chunks its members reach — the same seam bug per-leaf outset could have had, one
    /// level up.
    #[test]
    fn chunked_resolve_matches_monolithic_for_an_outset_part() {
        let voxels_per_block = 8;
        let member = |offset_blocks: [i64; 3], material| {
            let shape = SdfShape::from_blocks(ShapeKind::Box, [3, 2, 2], 1, voxels_per_block);
            let mut node = Node::new("M", NodeContent::Tool { shape, material });
            node.transform = NodeTransform::from_blocks(offset_blocks, voxels_per_block);
            NodeBuilder::Leaf(node)
        };
        let mut scene = Scene::from_nodes(vec![NodeBuilder::group(
            "Part",
            vec![
                member([0, 0, 0], MaterialChoice::Stone),
                member([4, 1, 0], MaterialChoice::Wood),
            ],
        )]);
        let path = crate::scene::NodePath::from_indices(vec![0]);
        scene
            .node_at_path_mut(&path)
            .expect("the Part resolves at path [0]")
            .outset = voxel_core::units::Measurement::from_voxels(3);
        let scene = scene_with_top_level_selected(scene, 0);
        assert_chunked_matches_monolithic(&scene, voxels_per_block, "outset-part");
    }
