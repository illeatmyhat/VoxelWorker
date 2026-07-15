use super::*;
use voxel_core::voxel::Voxel;

/// Perf probe (mesh-display scaling guard): per-size timing + emission counts of the pure
/// CPU two-layer mesh generation — the path the display takes when a loaded VS material
/// disengages the brick raymarch. Run:
/// `cargo test --release mesh_pipeline_scaling_probe -- --ignored --nocapture`.
#[test]
#[ignore = "perf probe — run in release with --nocapture"]
fn mesh_pipeline_scaling_probe() {
    use voxel_core::core_geom::MaterialChoice;
    use document::scene::{Node, NodeContent, Scene};
    use document::sketch::{PlaneAxis, Sketch, SketchSolid};
    let density = 16u32;
    for blocks in [50i64, 125, 250, 500] {
        let edge = blocks * density as i64;
        let extrude =
            SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, edge, edge), edge as u32);
        let scene = Scene::from_nodes(vec![Node::new(
            "Box",
            NodeContent::SketchTool { producer: extrude, material: MaterialChoice::Stone },
        )]);
        let chunks =
            evaluation::two_layer_store::TwoLayerStore::enabled().build_covering_chunks(
                &scene, density, 0,
            );
        let dims = scene.placed_region_dimensions(density);
        let recentre = scene.recentre_voxels_for_resolve(density);
        let start = std::time::Instant::now();
        let meshes =
            build_two_layer_chunk_meshes(&chunks, dims, recentre, density, LayerBand::FULL);
        let elapsed = start.elapsed();
        let boxes: u64 = meshes.iter().map(|mesh| mesh.box_count as u64).sum();
        let vertices: u64 = meshes.iter().map(|mesh| mesh.vertices.len() as u64).sum();
        let indices: u64 = meshes
            .iter()
            .map(|mesh| (mesh.indices.len() + mesh.indices_overlay.len()) as u64)
            .sum();
        let vertex_megabytes =
            (vertices * std::mem::size_of::<CuboidVertex>() as u64) as f64 / 1.0e6;
        println!(
            "mesh {edge}^3 vx: {} chunk meshes | {boxes} boxes | \
             {vertices} vertices ({vertex_megabytes:.0} MB) | {indices} indices | \
             CPU mesh-gen {elapsed:?}",
            meshes.len(),
        );
    }
}

#[test]
fn single_voxel_cube_has_six_faces() {
    // A solid 1-voxel "block" in a 3³ grid → 1 box → 6 exposed faces,
    // 12 triangles, 36 indices, 24 vertices.
    let grid = grid_from_indices([3, 3, 3], &[[1, 1, 1]], 0);
    let mesh = build_cuboid_mesh(&grid, 1);
    assert_eq!(mesh.box_count(), 1, "single voxel → one box");
    assert_eq!(mesh.face_count(), 6, "all six faces exposed");
    assert_eq!(mesh.triangle_count(), 12, "6 faces × 2 triangles");
    assert_eq!(mesh.index_count(), 36, "6 faces × 6 indices");
    assert_eq!(mesh.vertex_count(), 24, "6 faces × 4 verts");
}

#[test]
fn two_voxel_run_is_one_box_six_faces() {
    // A 2-voxel run along X (same material) merges into a single box; its
    // exposed-face mesh still has exactly 6 faces (the shared internal face
    // between the two voxels is culled BY merging into one box).
    let grid = grid_from_indices([4, 3, 3], &[[1, 1, 1], [2, 1, 1]], 0);
    let mesh = build_cuboid_mesh(&grid, 1);
    assert_eq!(mesh.box_count(), 1, "2-voxel run → one merged box");
    assert_eq!(mesh.face_count(), 6, "merged box still has 6 exposed faces");
    assert_eq!(mesh.triangle_count(), 12);
    assert_eq!(mesh.index_count(), 36);
}

#[test]
fn solid_block_collapses_to_six_faces() {
    // A solid 4×4×4 single-material block → 1 box → 6 faces (vs 4096 cubes /
    // 24576 instanced triangles): the order-of-magnitude reduction.
    let mut cells = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                cells.push([x, y, z]);
            }
        }
    }
    let grid = grid_from_indices([4, 4, 4], &cells, 0);
    let mesh = build_cuboid_mesh(&grid, 1);
    assert_eq!(mesh.box_count(), 1);
    assert_eq!(mesh.face_count(), 6);
    assert_eq!(mesh.triangle_count(), 12);
}

#[test]
fn adjacent_solid_faces_are_culled() {
    // Two separate boxes of DIFFERENT materials sharing a face: the shared
    // faces are culled (backed by solid), so the combined silhouette is a
    // 2×1×1 box surface = 6 faces, not 12.
    let mut grid = grid_from_indices([4, 3, 3], &[[1, 1, 1]], 0);
    // Second voxel, different material, adjacent in +X — built in the SAME recentred
    // frame `grid_from_indices` uses (`floor(index + 0.5 − dim/2)`) so it lands next
    // to the first voxel.
    let half = [2.0f32, 1.5, 1.5];
    grid.occupied.push(Voxel {
        local_index: [
            (2.0 + 0.5 - half[0]).floor() as i32,
            (1.0 + 0.5 - half[1]).floor() as i32,
            (1.0 + 0.5 - half[2]).floor() as i32,
        ],
        block_local_coord: [0, 0, 0],
        block_id: voxel_core::core_geom::BlockId(1),
        attrs: voxel_core::core_geom::BlockAttrs::DEFAULT,
        grid_overlay: false,
    });
    let mesh = build_cuboid_mesh(&grid, 1);
    assert_eq!(mesh.box_count(), 2, "different materials → two boxes");
    // 2 boxes × 6 faces = 12, minus the 2 shared (one each side) = 10 faces.
    assert_eq!(
        mesh.face_count(),
        10,
        "the two faces between the adjacent boxes are culled"
    );
}

/// E3b-2: the per-voxel UV is the absolute voxel position on the face's two
/// in-plane axes, so a face spanning N voxels must have vertices whose
/// absolute index spans 0..N on those axes (the shader divides by density +
/// Repeat-tiles, giving one texture tile per voxel). Here a 3-voxel X-run in a
/// 5³ grid merges to one box; its top (Z-up: +Z) face must span 3 voxels along X
/// and 1 along Y, i.e. world X-extent 3.
#[test]
fn merged_face_spans_one_uv_unit_per_voxel() {
    // Use an EVEN grid dim (6) so the recentred fixture lands on half-integer
    // centres the integer payload represents exactly (an odd dim's old centre fell on
    // an integer — see `grid_from_indices`). The X-run [1,2,3] then occupies absolute
    // voxels 1..=3 with planes at 1 and 4, exactly as before.
    let grid = grid_from_indices([6, 6, 6], &[[1, 3, 3], [2, 3, 3], [3, 3, 3]], 0);
    let mesh = build_cuboid_mesh(&grid, 1);
    assert_eq!(mesh.box_count(), 1, "3-voxel X-run merges to one box");

    // Absolute voxel position = world position + half (dims/2). The UV in the
    // shader uses exactly this, so spanning 3 units in X across the face means
    // the texture tiles 3× (once per voxel) with a Repeat sampler.
    let half = [3.0f32, 3.0, 3.0];
    let abs_x: Vec<f32> = mesh
        .vertices
        .iter()
        .map(|v| v.position[0] + half[0])
        .collect();
    let min_x = abs_x.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_x = abs_x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    // The run occupies absolute X voxels 1..=3 → planes at 1 and 4.
    assert_eq!(min_x, 1.0, "box min X plane = first voxel index");
    assert_eq!(max_x, 4.0, "box max+1 X plane = last voxel index + 1");
    assert_eq!(max_x - min_x, 3.0, "face spans 3 voxel UV units along X");
}

/// E3b-2: the face-normal → texture-array layer mapping must match the loaded
/// shader's `face_layer` (cuboid_loaded.wgsl) EXACTLY. Z-up: +Z = up (2), -Z =
/// down (3); ±X = east/west (0/1); -Y = south/front (4), +Y = north/back (5).
/// Replicated here as a pure function so the shader↔CPU mapping is regression-
/// guarded (the vertical texture axis is Z, not Y).
#[test]
fn face_normal_to_layer_matches_instanced() {
    // Byte-for-byte the same branch order as cuboid_loaded.wgsl::face_layer.
    fn face_layer(normal: [f32; 3]) -> i32 {
        let m = [normal[0].abs(), normal[1].abs(), normal[2].abs()];
        if m[2] > 0.5 {
            // Vertical (Z-up): +Z = up (2), -Z = down (3).
            if normal[2] > 0.0 { 2 } else { 3 }
        } else if m[0] > 0.5 {
            if normal[0] > 0.0 { 0 } else { 1 }
        } else if normal[1] < 0.0 {
            // -Y = south/front.
            4
        } else {
            // +Y = north/back.
            5
        }
    }
    // FACE_TEMPLATES order is +X,-X,+Y,-Y,+Z,-Z → Z-up layers 0,1,5,4,2,3.
    let expected = [0, 1, 5, 4, 2, 3];
    for (face, &want) in FACE_TEMPLATES.iter().zip(expected.iter()) {
        assert_eq!(face_layer(face.normal), want, "normal {:?}", face.normal);
    }
}

/// The cuboid decomposition must cover EVERY occupied voxel of the grid — the
/// box set's total voxel count equals `grid.occupied.len()` — for ANY shape AND
/// for a recentred cloud. This is the regression guard for the "partial
/// silhouette" bug (#18): the cuboid cylinder rendered ~1/4 of the disc because
/// the densifier anchored region index 0 at `dimensions/2` and silently dropped
/// the voxels of a recentred cloud (the scene resolve path shifts an odd-block
/// shape off-centre). A wedge means lost coverage; this asserts none is lost.
#[test]
fn cuboid_covers_every_voxel_for_all_shapes() {
    use voxel_core::voxel::{ShapeKind};
    use document::voxel::{SdfShape, VoxelProducer};

    for &kind in &[
        ShapeKind::Cylinder,
        ShapeKind::Sphere,
        ShapeKind::Torus,
        ShapeKind::Box,
        ShapeKind::Tube,
    ] {
        // 5×1×5 is the default disc (odd X/Z blocks → the recentre that exposed
        // the bug); also exercise an odd-all-axes size to be thorough.
        for &size in &[[5u32, 1, 5], [3, 3, 3], [5, 3, 7]] {
            let voxels_per_block = 8;
            let shape = SdfShape::from_blocks(kind, size, 1, voxels_per_block);
            // Shift-invariance: also run a deliberately recentred copy of the
            // grid (every voxel +8 in each axis, like `resolve_region`'s
            // off-centre composite) — coverage must be identical.
            for shift in [0.0f32, 8.0] {
                let mut shifted = VoxelGrid::new(shape.grid_dimensions(voxels_per_block));
                shape.resolve(&mut shifted, voxels_per_block);
                if shifted.occupied.is_empty() {
                    continue;
                }
                for voxel in &mut shifted.occupied {
                    for axis in 0..3 {
                        voxel.local_index[axis] += shift as i32;
                    }
                }

                let (region, _world_offset) = region_from_voxel_cloud(&shifted);
                let region_solid =
                    region.cells.iter().filter(|c| c.is_some()).count();
                let boxes = decompose_into_boxes(&region);
                let covered: u64 = boxes.iter().map(|b| b.cell_count()).sum();

                assert_eq!(
                    region_solid,
                    shifted.occupied.len(),
                    "{kind:?} {size:?} shift={shift}: densified region lost \
                     voxels ({region_solid} of {})",
                    shifted.occupied.len()
                );
                assert_eq!(
                    covered,
                    shifted.occupied.len() as u64,
                    "{kind:?} {size:?} shift={shift}: cuboid boxes cover \
                     {covered} of {} voxels (a partial silhouette)",
                    shifted.occupied.len()
                );
            }
        }
    }
}

/// E3b-3: the layer-range band clip masks the densified region to the band's
/// absolute Z-layer range (Z-up: layers are Z-slices) BEFORE decomposition, so
/// clipping a solid block to a sub-band yields a thinner block — with NEW cap
/// faces at the band edges (a fragment discard on the single merged column would
/// leave it open-topped, with no caps). Here a solid 4×4×4 block (one tall box)
/// clipped to a 2-layer band must mesh as a 4×4×2 box: still 6 faces, but spanning
/// exactly 2 voxels in Z.
#[test]
fn band_clip_masks_region_and_caps_the_slab() {
    let mut cells = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                cells.push([x, y, z]);
            }
        }
    }
    // A centred 4³ block: half_z = 2, so absolute layer == region-local Z here.
    let grid = grid_from_indices([4, 4, 4], &cells, 0);

    // Full band → the whole block: 1 box, 6 faces, Z-span 4.
    let full = build_cuboid_mesh_banded(&grid, 1, LayerBand::FULL);
    assert_eq!(full.box_count(), 1);
    assert_eq!(full.face_count(), 6);

    // Band [1, 2] (inclusive) → only Z-layers 1 and 2 survive: a 4×4×2 slab.
    let band = LayerBand {
        band_min: 1,
        band_max: 2,
        onion_depth: 0,
    };
    let clipped = build_cuboid_mesh_banded(&grid, 1, band);
    assert_eq!(clipped.box_count(), 1, "the clipped slab is still one box");
    assert_eq!(
        clipped.face_count(),
        6,
        "the band edges get real cap faces (top + bottom), so still 6 faces"
    );

    // The clipped slab spans EXACTLY 2 voxels in Z (the band height), with new
    // caps — confirming masking, not a fragment discard.
    let half_z = 2.0f32;
    let abs_z: Vec<f32> = clipped
        .vertices
        .iter()
        .map(|v| v.position[2] + half_z)
        .collect();
    let min_z = abs_z.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_z = abs_z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert_eq!(min_z, 1.0, "slab bottom cap at the band's lower layer");
    assert_eq!(max_z, 3.0, "slab top cap at the band's upper layer + 1");
    assert_eq!(max_z - min_z, 2.0, "slab is exactly the 2-layer band tall");
}

/// A band entirely OUTSIDE the occupied layers clips everything away.
#[test]
fn band_clip_outside_occupied_layers_is_empty() {
    let grid = grid_from_indices([4, 4, 4], &[[1, 1, 1], [2, 2, 2]], 0);
    let band = LayerBand {
        band_min: 10,
        band_max: 12,
        onion_depth: 0,
    };
    let mesh = build_cuboid_mesh_banded(&grid, 1, band);
    assert_eq!(mesh.box_count(), 0, "no voxel falls in the band");
    assert_eq!(mesh.face_count(), 0);
}

/// Vertex-position ↔ voxel-extent correspondence: every emitted face vertex
/// must land on one of a box's integer corner planes — `min` (the box's
/// min-corner) or `max + 1` (its exclusive far plane) on each axis — once the
/// shift-invariant `world_offset` is subtracted back out. This proves the
/// geometry the mesher emits matches the integer box bounds the decomposition
/// produced (no off-by-one / wrong-plane vertex), as a pure CPU assertion.
#[test]
fn vertex_positions_match_box_voxel_extents() {
    use std::collections::HashSet;

    // A few irregular shapes so vertices come from boxes of varied extents and
    // a multi-box decomposition (different materials force a split).
    let single = grid_from_indices([3, 3, 3], &[[1, 1, 1]], 0);
    let run = grid_from_indices([5, 5, 5], &[[1, 2, 2], [2, 2, 2], [3, 2, 2]], 0);
    // Two adjacent boxes of different materials (a 2-box decomposition).
    let mut two_box = grid_from_indices([4, 3, 3], &[[1, 1, 1]], 0);
    // Adjacent in +X, built in the SAME recentred frame as `grid_from_indices`.
    let half = [2.0f32, 1.5, 1.5];
    two_box.occupied.push(Voxel {
        local_index: [
            (2.0 + 0.5 - half[0]).floor() as i32,
            (1.0 + 0.5 - half[1]).floor() as i32,
            (1.0 + 0.5 - half[2]).floor() as i32,
        ],
        block_local_coord: [0, 0, 0],
        block_id: voxel_core::core_geom::BlockId(1),
        attrs: voxel_core::core_geom::BlockAttrs::DEFAULT,
        grid_overlay: false,
    });

    for grid in [single, run, two_box] {
        let mesh = build_cuboid_mesh(&grid, 1);
        // Recover the exact region + offset + boxes the mesher used.
        let (region, world_offset) = region_from_voxel_cloud(&grid);
        let boxes = decompose_into_boxes(&region);
        assert!(!boxes.is_empty(), "test shape must decompose to ≥1 box");

        // The set of valid corner planes per axis = {min} ∪ {max+1} over boxes.
        let mut valid_plane: [HashSet<i64>; 3] =
            [HashSet::new(), HashSet::new(), HashSet::new()];
        for voxel_box in &boxes {
            for (axis, planes) in valid_plane.iter_mut().enumerate() {
                planes.insert(voxel_box.min[axis] as i64);
                planes.insert(voxel_box.max[axis] as i64 + 1);
            }
        }

        for vertex in &mesh.vertices {
            for axis in 0..3 {
                // Undo the world offset → region-local integer plane.
                let local_plane = (vertex.position[axis] - world_offset[axis]).round() as i64;
                // The round must be exact (planes are integers in local space).
                assert!(
                    (vertex.position[axis] - world_offset[axis] - local_plane as f32).abs()
                        < 1e-4,
                    "vertex {:?} axis {axis} not on an integer local plane",
                    vertex.position
                );
                assert!(
                    valid_plane[axis].contains(&local_plane),
                    "vertex {:?} axis {axis} local plane {local_plane} is not a box \
                     min or max+1 plane (valid: {:?})",
                    vertex.position,
                    valid_plane[axis]
                );
            }
        }

        // Per-box: the box's OWN min and max+1 corner planes must each appear in
        // the emitted vertex set (the box actually contributes its extents).
        let emitted: HashSet<[i64; 3]> = mesh
            .vertices
            .iter()
            .map(|vertex| {
                [
                    (vertex.position[0] - world_offset[0]).round() as i64,
                    (vertex.position[1] - world_offset[1]).round() as i64,
                    (vertex.position[2] - world_offset[2]).round() as i64,
                ]
            })
            .collect();
        for voxel_box in &boxes {
            let min_corner = [
                voxel_box.min[0] as i64,
                voxel_box.min[1] as i64,
                voxel_box.min[2] as i64,
            ];
            let max_corner = [
                voxel_box.max[0] as i64 + 1,
                voxel_box.max[1] as i64 + 1,
                voxel_box.max[2] as i64 + 1,
            ];
            assert!(
                emitted.contains(&min_corner),
                "box {voxel_box:?} min corner {min_corner:?} missing from vertices"
            );
            assert!(
                emitted.contains(&max_corner),
                "box {voxel_box:?} max+1 corner {max_corner:?} missing from vertices"
            );
        }
    }
}

#[test]
fn empty_grid_has_no_mesh() {
    let grid = VoxelGrid::new([4, 4, 4]);
    let mesh = build_cuboid_mesh(&grid, 1);
    assert_eq!(mesh.box_count(), 0);
    assert_eq!(mesh.face_count(), 0);
    assert_eq!(mesh.index_count(), 0);
}

