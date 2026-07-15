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

/// Build a tiny grid from a set of occupied voxel indices, all one material, with
/// the given dimensions, in the RECENTRED render frame the live cuboid path sees.
///
/// The stored `local_index` reproduces the retired f32 fixture's
/// `world_position = index + 0.5 − dim/2` EXACTLY for an EVEN dim (where the centre
/// is a half-integer): `local_index = floor(index + 0.5 − dim/2)`, so
/// `world_position()` (= `local_index + 0.5`) equals the old value bit-for-bit and the
/// band-clip's `half = floor(dim/2)` frame assumption still holds. (An ODD dim's old
/// centre fell on an INTEGER, which the integer payload — whose centres are always
/// half-integers — cannot represent; the one odd-dim test below corner-anchors and
/// reads the world planes directly, since the mesher is anchor-shift-invariant.)
fn grid_from_indices(dimensions: [u32; 3], cells: &[[u32; 3]], material: u16) -> VoxelGrid {
    let half = [
        dimensions[0] as f32 / 2.0,
        dimensions[1] as f32 / 2.0,
        dimensions[2] as f32 / 2.0,
    ];
    let mut grid = VoxelGrid::new(dimensions);
    for &[i, j, k] in cells {
        grid.occupied.push(Voxel {
            local_index: [
                (i as f32 + 0.5 - half[0]).floor() as i32,
                (j as f32 + 0.5 - half[1]).floor() as i32,
                (k as f32 + 0.5 - half[2]).floor() as i32,
            ],
            block_local_coord: [0, 0, 0],
            block_id: voxel_core::core_geom::BlockId(material),
            attrs: voxel_core::core_geom::BlockAttrs::DEFAULT,
            grid_overlay: false,
        });
    }
    grid
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

// ---- Per-chunk apron meshing structural parity (issue #20 S6c-2d) ----

/// A single UNIT exposed face: the absolute integer plane coordinate on the
/// face's axis, the two in-plane unit-cell lower coords, and the face axis +
/// sign. Canonical regardless of how a co-planar face is split into abutting
/// quads, so it is the granularity at which whole-region meshing and per-chunk
/// apron meshing must produce the IDENTICAL set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct UnitFace {
    /// 0 = X face, 1 = Y face, 2 = Z face.
    axis: u8,
    /// `+1` / `-1` outward direction along `axis`.
    sign: i8,
    /// Integer world plane on `axis` (the quad's constant coordinate).
    plane: i64,
    /// The two in-plane unit-cell lower coords (the axes other than `axis`).
    cell: [i64; 2],
}

/// Round a world coordinate that must land on an integer plane (box corners are
/// integer planes in world space once the shift-invariant offset is folded in).
fn round_plane(value: f32) -> i64 {
    let rounded = value.round();
    assert!(
        (value - rounded).abs() < 1e-3,
        "vertex coord {value} is not on an integer world plane"
    );
    rounded as i64
}

/// Explode a vertex/index mesh into its SET of unit exposed faces (the canonical
/// granularity), in the GLOBAL INDEX frame. Mesh vertices live in world space at
/// `global_index + world_offset`, so subtracting `world_offset` recovers the
/// integer global-index planes the ground-truth `genuine_exposed_faces` keys off.
/// Each quad (6 indices) lies on a plane perpendicular to its normal; it is split
/// into the unit cells it covers in the two in-plane axes.
fn unit_faces_in_index_frame(
    vertices: &[CuboidVertex],
    indices: &[u32],
    world_offset: [f32; 3],
) -> std::collections::HashSet<UnitFace> {
    let mut faces = std::collections::HashSet::new();
    let to_index = |pos: [f32; 3]| -> [f32; 3] {
        [
            pos[0] - world_offset[0],
            pos[1] - world_offset[1],
            pos[2] - world_offset[2],
        ]
    };
    // Each quad is two triangles emitted as [b, b+1, b+2, b, b+2, b+3], so the
    // four distinct corner vertices are indices[i], [i+1], [i+2], [i+5].
    let mut i = 0;
    while i < indices.len() {
        let corners = [
            vertices[indices[i] as usize],
            vertices[indices[i + 1] as usize],
            vertices[indices[i + 2] as usize],
            vertices[indices[i + 5] as usize],
        ];
        let normal = corners[0].normal;
        let axis = if normal[0].abs() > 0.5 {
            0usize
        } else if normal[1].abs() > 0.5 {
            1
        } else {
            2
        };
        let sign: i8 = if normal[axis] > 0.0 { 1 } else { -1 };
        let (a, b) = match axis {
            0 => (1usize, 2usize),
            1 => (0usize, 2usize),
            _ => (0usize, 1usize),
        };
        let plane = round_plane(to_index(corners[0].position)[axis]);
        // The quad's span in the two in-plane axes (integer index planes).
        let mut a_lo = i64::MAX;
        let mut a_hi = i64::MIN;
        let mut b_lo = i64::MAX;
        let mut b_hi = i64::MIN;
        for corner in &corners {
            let idx = to_index(corner.position);
            let av = round_plane(idx[a]);
            let bv = round_plane(idx[b]);
            a_lo = a_lo.min(av);
            a_hi = a_hi.max(av);
            b_lo = b_lo.min(bv);
            b_hi = b_hi.max(bv);
        }
        for ca in a_lo..a_hi {
            for cb in b_lo..b_hi {
                faces.insert(UnitFace {
                    axis: axis as u8,
                    sign,
                    plane,
                    cell: [ca, cb],
                });
            }
        }
        i += 6;
    }
    faces
}

/// The world offset (`min_world - 0.5` per axis) the mesher anchors a grid's
/// vertices on — subtract it from a mesh vertex to get the integer global-index
/// frame `genuine_exposed_faces` uses.
fn grid_world_offset(grid: &VoxelGrid) -> [f32; 3] {
    let mut min_world = [f32::INFINITY; 3];
    for v in &grid.occupied {
        let position = v.world_position();
        for (axis, m) in min_world.iter_mut().enumerate() {
            *m = m.min(position[axis]);
        }
    }
    [min_world[0] - 0.5, min_world[1] - 0.5, min_world[2] - 0.5]
}

/// Bucket a whole grid into per-chunk sub-grids exactly as the renderer's `new`
/// wrapper does (`floor(world / chunk_extent)`), so the per-chunk mesher sees the
/// SAME partition the live path does.
fn bucket_for_test(grid: &VoxelGrid, voxels_per_block: u32) -> Vec<([i32; 3], VoxelGrid)> {
    super::bucket_grid_into_chunk_grids(grid, voxels_per_block)
}

/// The set of GENUINELY-exposed unit faces of an occupancy set: a `(voxel,
/// direction)` whose neighbour cell is air. This is the VISIBLE silhouette — the
/// surface that survives back-face culling + depth testing. The cuboid mesher's
/// `face_is_exposed` emits a whole MERGED box face when ANY cell behind it is
/// air, so it OVER-DRAWS the sub-faces backed by solid; those over-draw quads are
/// always either back-face-culled or depth-occluded by the solid they are buried
/// in, so they never reach a pixel. The genuinely-exposed set is therefore the
/// invariant that determines the rendered image — and the structural parity claim:
/// it must be IDENTICAL for whole-region and per-chunk meshing. We derive it
/// straight from the occupancy (the ground truth) and also use it to filter an
/// emitted mesh's unit faces down to its visible subset.
fn genuine_exposed_faces(
    occupied: &std::collections::HashSet<[i64; 3]>,
) -> std::collections::HashSet<UnitFace> {
    let dirs: [(usize, i8, [i64; 3]); 6] = [
        (0, 1, [1, 0, 0]),
        (0, -1, [-1, 0, 0]),
        (1, 1, [0, 1, 0]),
        (1, -1, [0, -1, 0]),
        (2, 1, [0, 0, 1]),
        (2, -1, [0, 0, -1]),
    ];
    let mut faces = std::collections::HashSet::new();
    for &v in occupied {
        for (axis, sign, delta) in dirs {
            let neighbor = [v[0] + delta[0], v[1] + delta[1], v[2] + delta[2]];
            if occupied.contains(&neighbor) {
                continue; // backed by solid → interior, not visible
            }
            // The face plane on `axis`: for +sign it's the voxel's far plane
            // (v[axis] + 1), for -sign the near plane (v[axis]).
            let plane = if sign > 0 { v[axis] + 1 } else { v[axis] };
            let (a, b) = match axis {
                0 => (1usize, 2usize),
                1 => (0usize, 2usize),
                _ => (0usize, 1usize),
            };
            faces.insert(UnitFace {
                axis: axis as u8,
                sign,
                plane,
                cell: [v[a], v[b]],
            });
        }
    }
    faces
}

/// Filter an emitted mesh's unit faces down to the VISIBLE subset (those whose
/// `(plane, cell, axis, sign)` is a genuinely-exposed face), discarding the
/// over-draw quads `face_is_exposed` emits for partially-exposed merged boxes.
fn visible_unit_faces(
    vertices: &[CuboidVertex],
    indices: &[u32],
    world_offset: [f32; 3],
    genuine: &std::collections::HashSet<UnitFace>,
) -> std::collections::HashSet<UnitFace> {
    unit_faces_in_index_frame(vertices, indices, world_offset)
        .into_iter()
        .filter(|f| genuine.contains(f))
        .collect()
}

/// Absolute integer occupancy (global indices `round(world - min_world)`) of a
/// grid — the same frame the cuboid mesher's vertices live in, so a `UnitFace`
/// derived from occupancy and one derived from a mesh vertex compare directly.
fn occupancy_indices(grid: &VoxelGrid) -> std::collections::HashSet<[i64; 3]> {
    let mut min_world = [f32::INFINITY; 3];
    for v in &grid.occupied {
        let position = v.world_position();
        for (axis, m) in min_world.iter_mut().enumerate() {
            *m = m.min(position[axis]);
        }
    }
    let mut set = std::collections::HashSet::new();
    for v in &grid.occupied {
        let position = v.world_position();
        set.insert([
            (position[0] - min_world[0]).round() as i64,
            (position[1] - min_world[1]).round() as i64,
            (position[2] - min_world[2]).round() as i64,
        ]);
    }
    set
}

/// Map a wholesale/filtered mesh build to `coord -> (vertex bytes, indices)` — the
/// per-chunk GPU buffer set proxy (the renderer uploads exactly these bytes), so a
/// byte-equal map == a byte-equal buffer set.
fn mesh_map(
    meshes: &[CuboidChunkMesh],
) -> std::collections::HashMap<[i32; 3], (Vec<u8>, Vec<u32>)> {
    meshes
        .iter()
        .map(|m| {
            (
                m.coord,
                (bytemuck::cast_slice::<_, u8>(&m.vertices).to_vec(), m.indices.clone()),
            )
        })
        .collect()
}

/// Issue #40: an INCREMENTAL rebuild — re-mesh only the apron-dilated dirty subset,
/// evict vacated chunks, keep every other chunk's buffer — produces a per-chunk
/// mesh (buffer) set BYTE-IDENTICAL to a wholesale rebuild, for every edit kind
/// INCLUDING edits at a chunk SEAM (where the 1-voxel apron makes a neighbour's
/// boundary faces depend on the edited chunk). This is the cuboid analogue of the
/// deleted instanced `incremental_rebuild_equals_full_rebuild_for_every_edit_kind`
/// and the real proof that consuming `cuboid_incremental_plan` is output-preserving.
///
/// Both scenes pin the SAME global bounds with anchor voxels at the extremes (so
/// `world_offset` is identical — modelling the live `incremental_ok` precondition
/// that the floating origin did NOT shift; a shift forces a wholesale fall-back).
/// Edits touch only interior / seam voxels. `evicted_dirty` is set to exactly the
/// chunks that were resident in A AND changed in B — faithfully modelling
/// `Store::invalidate_aabb`'s evicted set — so the plan's 26-neighbour dilation is
/// what must catch any stale neighbour; if the dilation were wrong, a seam edit fails.
#[test]
fn incremental_cuboid_rebuild_equals_wholesale() {
    // vpb 1 → chunk extent = CHUNK_BLOCKS(4) voxels, so a 12³ grid spans 3 chunks
    // per axis and interior/seam edits exercise real apron seams.
    let dims = [12u32, 12, 12];
    // Anchor frame: the 8 corners, present in EVERY scene → global bounds pinned.
    let anchors: Vec<[u32; 3]> = {
        let e = [0u32, 11];
        let mut cs = Vec::new();
        for &x in &e {
            for &y in &e {
                for &z in &e {
                    cs.push([x, y, z]);
                }
            }
        }
        cs
    };
    // (scene_a interior, scene_b interior) edit pairs. x=4/x=8 are chunk seams
    // (chunk extent 4), so an edit at x=3↔4 changes faces in TWO chunks.
    let scene_a_interior: Vec<[u32; 3]> =
        vec![[3, 3, 3], [4, 3, 3], [7, 7, 7], [8, 7, 7], [5, 9, 2]];
    let edits: Vec<(&str, Vec<[u32; 3]>)> = vec![
        // Add an interior voxel (no seam).
        ("add interior", {
            let mut v = scene_a_interior.clone();
            v.push([6, 6, 6]);
            v
        }),
        // Remove a seam voxel → its cross-seam neighbour [3,3,3] re-exposes a face.
        ("remove seam voxel", {
            let mut v = scene_a_interior.clone();
            v.retain(|c| *c != [4, 3, 3]);
            v
        }),
        // Add a voxel straddling a seam next to an existing one → neighbour chunk
        // boundary faces get culled.
        ("add seam voxel", {
            let mut v = scene_a_interior.clone();
            v.push([4, 7, 7]); // abuts [3-or-8?]; sits in chunk x=1 next to [8,7,7] region edits
            v.push([3, 7, 7]);
            v
        }),
        // Move a voxel across a seam (remove one side, add the other).
        ("move across seam", {
            let mut v = scene_a_interior.clone();
            v.retain(|c| *c != [8, 7, 7]);
            v.push([9, 7, 7]);
            v
        }),
    ];

    for (label, interior_b) in &edits {
        let mut cells_a = anchors.clone();
        cells_a.extend(scene_a_interior.iter().copied());
        let mut cells_b = anchors.clone();
        cells_b.extend(interior_b.iter().copied());

        let grid_a = grid_from_indices(dims, &cells_a, 0);
        let grid_b = grid_from_indices(dims, &cells_b, 0);
        let buckets_a = bucket_for_test(&grid_a, 1);
        let buckets_b = bucket_for_test(&grid_b, 1);
        let refs_a: Vec<([i32; 3], &VoxelGrid)> =
            buckets_a.iter().map(|(c, g)| (*c, g)).collect();
        let refs_b: Vec<([i32; 3], &VoxelGrid)> =
            buckets_b.iter().map(|(c, g)| (*c, g)).collect();

        // Wholesale builds (the ground truth) for A (prior state) and B (target).
        let wholesale_a = build_chunk_meshes_with_apron(&refs_a, dims, LayerBand::FULL);
        let wholesale_b = build_chunk_meshes_with_apron(&refs_b, dims, LayerBand::FULL);

        // Per-chunk occupancy index sets, to derive the faithful dirty set.
        let occ = |buckets: &[([i32; 3], VoxelGrid)]| {
            let mut m: std::collections::HashMap<[i32; 3], std::collections::HashSet<[i64; 3]>> =
                std::collections::HashMap::new();
            for (coord, g) in buckets {
                m.insert(*coord, occupancy_indices(g));
            }
            m
        };
        let occ_a = occ(&buckets_a);
        let occ_b = occ(&buckets_b);

        // resident = A's covering coords (the renderer's source_chunk_grids coords).
        let resident: Vec<[i32; 3]> = buckets_a.iter().map(|(c, _)| *c).collect();
        // occupied_b = B's non-empty covering coords.
        let occupied_b: Vec<[i32; 3]> = buckets_b
            .iter()
            .filter(|(_, g)| !g.occupied.is_empty())
            .map(|(c, _)| *c)
            .collect();
        // evicted_dirty = chunks resident in A whose occupancy CHANGED in B (exactly
        // what invalidate_aabb evicts: resident ∩ edit-region). Newly-appeared
        // chunks (absent from A) are caught by the plan's own new-appeared term.
        let evicted_dirty: Vec<[i32; 3]> = resident
            .iter()
            .copied()
            .filter(|coord| occ_a.get(coord) != occ_b.get(coord))
            .collect();

        let plan = cuboid_incremental_plan(&resident, &evicted_dirty, &occupied_b);

        // Genuinely incremental: at least the far corners are NOT re-meshed.
        assert!(
            plan.rebuild.len() < occupied_b.len(),
            "[{label}] expected a partial rebuild, got {} of {} chunks",
            plan.rebuild.len(),
            occupied_b.len()
        );

        // Apply the plan to A's mesh map → the incremental buffer set.
        let rebuild_set: std::collections::HashSet<[i32; 3]> =
            plan.rebuild.iter().copied().collect();
        let rebuilt =
            build_chunk_meshes_with_apron_filtered(&refs_b, Some(&rebuild_set), dims, LayerBand::FULL);

        let mut result = mesh_map(&wholesale_a);
        for coord in &plan.evict {
            result.remove(coord);
        }
        for coord in &plan.rebuild {
            result.remove(coord); // a rebuild coord meshing to empty drops out here
        }
        for (coord, entry) in mesh_map(&rebuilt) {
            result.insert(coord, entry);
        }

        let target = mesh_map(&wholesale_b);
        assert_eq!(
            result, target,
            "[{label}] incremental buffer set must byte-equal the wholesale rebuild"
        );
    }
}

/// The CORE structural guarantee (the analogue of the decomposition round-trip):
/// the per-chunk-with-apron VISIBLE exposed-face SET equals the whole-region
/// mesher's — and both equal the ground-truth genuinely-exposed set derived
/// straight from occupancy — for many shapes/sizes INCLUDING shapes spanning
/// multiple chunks. This is what guarantees the RENDERED IMAGE is unchanged,
/// independent of the goldens: the apron makes seam faces between two solid
/// chunks culled (no extra visible interior quads), and co-planar seam-spanning
/// faces split into abutting quads covering the IDENTICAL visible unit-face set.
///
/// "Visible" = the subset of emitted unit faces backed by air. The mesher emits a
/// whole MERGED box face when ANY cell behind it is air, over-drawing the
/// sub-faces backed by solid; those over-draw quads are always back-face-culled or
/// depth-occluded by the solid they are buried in, so they never reach a pixel and
/// their (merge-order-dependent) count is NOT a rendering invariant. The visible
/// set IS. We assert the visible set of BOTH paths equals the ground truth, AND
/// that every genuine face is actually emitted by each path (no real hole).
#[test]
fn per_chunk_apron_exposed_face_set_equals_whole_region() {
    use voxel_core::voxel::{ShapeKind};
    use document::voxel::{SdfShape, VoxelProducer};

    let mut multi_chunk_seen = false;
    // Densities chosen so shapes span MULTIPLE chunks (chunk = CHUNK_BLOCKS=4
    // blocks × density voxels per axis = 32 voxels at density 8). An 8-block axis
    // = 64 voxels = 2 chunks/axis, so the apron is exercised at real seams.
    for &kind in &[
        ShapeKind::Sphere,
        ShapeKind::Cylinder,
        ShapeKind::Torus,
        ShapeKind::Box,
        ShapeKind::Tube,
    ] {
        for &(size, density) in &[
            ([5u32, 1, 5], 8u32), // the default disc (odd X/Z → recentred cloud)
            ([3, 3, 3], 8),
            ([8, 2, 8], 8), // 64×16×64 voxels → 2 chunks/axis in X/Z (multi-chunk)
            ([5, 3, 7], 8),
        ] {
            let shape = SdfShape::from_blocks(kind, size, 1, density);
            let dims = shape.grid_dimensions(density);
            let mut grid = VoxelGrid::new(dims);
            shape.resolve(&mut grid, density);
            if grid.occupied.is_empty() {
                continue;
            }

            // Ground-truth genuinely-exposed faces, straight from occupancy.
            let occupancy = occupancy_indices(&grid);
            let genuine = genuine_exposed_faces(&occupancy);
            let world_offset = grid_world_offset(&grid);

            // Whole-region reference mesh → its VISIBLE face subset.
            let whole = build_cuboid_mesh(&grid, density);
            let whole_visible =
                visible_unit_faces(&whole.vertices, &whole.indices, world_offset, &genuine);

            // Per-chunk-with-apron mesh → its VISIBLE face subset (union over chunks).
            let buckets = bucket_for_test(&grid, density);
            let chunk_refs: Vec<([i32; 3], &VoxelGrid)> =
                buckets.iter().map(|(c, g)| (*c, g)).collect();
            if buckets.len() > 1 {
                multi_chunk_seen = true;
            }
            let chunk_meshes =
                build_chunk_meshes_with_apron(&chunk_refs, dims, LayerBand::FULL);
            let mut per_chunk_visible = std::collections::HashSet::new();
            for mesh in &chunk_meshes {
                per_chunk_visible.extend(visible_unit_faces(
                    &mesh.vertices,
                    &mesh.indices,
                    world_offset,
                    &genuine,
                ));
            }

            // Both paths must emit EXACTLY the ground-truth visible surface — no
            // missing genuine face (a hole), no spurious visible face (a stray
            // seam quad). This is the rendering-determining invariant.
            assert_eq!(
                whole_visible, genuine,
                "{kind:?} {size:?} density={density}: whole-region visible faces != ground truth"
            );
            assert_eq!(
                per_chunk_visible, genuine,
                "{kind:?} {size:?} density={density}: per-chunk apron visible faces != \
                 ground truth ({} per-chunk vs {} genuine)",
                per_chunk_visible.len(),
                genuine.len()
            );
        }
    }
    assert!(
        multi_chunk_seen,
        "no test case actually spanned multiple chunks — apron never exercised at a seam"
    );
}

/// A solid slab spanning a chunk seam must NOT emit interior seam faces — the
/// apron culls them. Two abutting solid 32-voxel chunks (density 8, 4 blocks ×
/// 8 = 32 voxels per chunk axis) form an 8×4×4-block solid box across the X
/// seam; the whole-region mesh is one box (6 faces' worth of unit faces), and
/// the per-chunk mesh (two boxes) must produce the IDENTICAL unit-face set with
/// no faces on the interior seam plane.
#[test]
fn solid_slab_across_chunk_seam_has_no_interior_faces() {
    let density = 8u32;
    let chunk_voxels = voxel_core::core_geom::CHUNK_BLOCKS * density; // 32
    let nx = chunk_voxels * 2; // span two chunks in X
    let ny = density; // 8
    let nz = density; // 8
    let dims = [nx, ny, nz];
    let half = [nx as f32 / 2.0, ny as f32 / 2.0, nz as f32 / 2.0];
    let mut grid = VoxelGrid::new(dims);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                grid.occupied.push(voxel_core::voxel::Voxel {
                    local_index: [
                        (i as f32 + 0.5 - half[0]).floor() as i32,
                        (j as f32 + 0.5 - half[1]).floor() as i32,
                        (k as f32 + 0.5 - half[2]).floor() as i32,
                    ],
                    block_local_coord: [0, 0, 0],
                    block_id: voxel_core::core_geom::BlockId(0),
                    attrs: voxel_core::core_geom::BlockAttrs::DEFAULT,
                    grid_overlay: false,
                });
            }
        }
    }

    let buckets = bucket_for_test(&grid, density);
    assert!(
        buckets.len() >= 2,
        "the solid slab must span at least two chunks in X (got {})",
        buckets.len()
    );

    let occupancy = occupancy_indices(&grid);
    let genuine = genuine_exposed_faces(&occupancy);
    let world_offset = grid_world_offset(&grid);

    let whole = build_cuboid_mesh(&grid, density);
    let whole_visible =
        visible_unit_faces(&whole.vertices, &whole.indices, world_offset, &genuine);

    let chunk_refs: Vec<([i32; 3], &VoxelGrid)> =
        buckets.iter().map(|(c, g)| (*c, g)).collect();
    let chunk_meshes = build_chunk_meshes_with_apron(&chunk_refs, dims, LayerBand::FULL);
    let mut per_chunk_visible = std::collections::HashSet::new();
    for mesh in &chunk_meshes {
        per_chunk_visible.extend(visible_unit_faces(
            &mesh.vertices,
            &mesh.indices,
            world_offset,
            &genuine,
        ));
    }

    // The slab surface is exactly the box's 6 sides: 2*(nx*ny + nx*nz + ny*nz)
    // unit faces. No interior seam plane faces.
    let expected = 2 * (nx * ny + nx * nz + ny * nz) as usize;
    assert_eq!(
        genuine.len(),
        expected,
        "solid box surface should be {expected} unit faces"
    );
    assert_eq!(
        whole_visible, genuine,
        "solid cross-seam slab: whole-region visible faces != ground truth"
    );
    assert_eq!(
        per_chunk_visible, genuine,
        "solid cross-seam slab: per-chunk apron visible faces != ground truth (interior \
         seam faces leaked or a side is missing)"
    );
}

/// The per-chunk band clip must match the whole-region band clip's VISIBLE
/// exposed-face set (real caps at the band edges, per chunk). A torus clipped to a
/// sub-band that falls INSIDE the chunks must synthesise the cap faces identically
/// in both paths. Ground truth = the genuinely-exposed faces of the BAND-MASKED
/// occupancy (cells outside `[band_min, band_max]` removed, so the band edges are
/// real air boundaries → cap faces).
#[test]
fn per_chunk_band_clip_face_set_equals_whole_region() {
    use voxel_core::voxel::{ShapeKind};
    use document::voxel::{SdfShape, VoxelProducer};
    let voxels_per_block = 8;
    let shape = SdfShape::from_blocks(ShapeKind::Torus, [8, 2, 8], 1, voxels_per_block);
    let dims = shape.grid_dimensions(voxels_per_block);
    let mut grid = VoxelGrid::new(dims);
    shape.resolve(&mut grid, voxels_per_block);
    assert!(!grid.occupied.is_empty());

    // Z-up: the band is a Z-layer range. The torus [8,2,8] is 128 voxels tall in
    // Z; a band straddling the vertical middle keeps a real slice of the tube.
    let band = LayerBand {
        band_min: 56,
        band_max: 71,
        onion_depth: 0,
    };

    // Ground truth: genuinely-exposed faces of the BAND-MASKED occupancy. The
    // band maps an absolute layer to a global index Z by `gz + base_layer` where
    // base_layer = floor(min_world.z + half_z) and the global index uses
    // `round(world - min_world)`. We mask occupancy to `base_layer + gz ∈ band`.
    let mut min_world = [f32::INFINITY; 3];
    for v in &grid.occupied {
        let position = v.world_position();
        for (axis, m) in min_world.iter_mut().enumerate() {
            *m = m.min(position[axis]);
        }
    }
    let half_z = (dims[2] / 2) as f32; // corner-anchoring: floored half
    let base_layer = (min_world[2] + half_z).floor() as i64;
    let occupancy: std::collections::HashSet<[i64; 3]> = grid
        .occupied
        .iter()
        .map(|v| {
            let position = v.world_position();
            [
                (position[0] - min_world[0]).round() as i64,
                (position[1] - min_world[1]).round() as i64,
                (position[2] - min_world[2]).round() as i64,
            ]
        })
        .filter(|idx| {
            let layer = base_layer + idx[2];
            layer >= band.band_min as i64 && layer <= band.band_max as i64
        })
        .collect();
    assert!(!occupancy.is_empty(), "band must keep some voxels");
    let genuine = genuine_exposed_faces(&occupancy);
    let world_offset = grid_world_offset(&grid);

    let whole = build_cuboid_mesh_banded(&grid, 8, band);
    let whole_visible =
        visible_unit_faces(&whole.vertices, &whole.indices, world_offset, &genuine);

    let buckets = bucket_for_test(&grid, 8);
    assert!(buckets.len() > 1, "torus must span multiple chunks");
    let chunk_refs: Vec<([i32; 3], &VoxelGrid)> =
        buckets.iter().map(|(c, g)| (*c, g)).collect();
    let chunk_meshes = build_chunk_meshes_with_apron(&chunk_refs, dims, band);
    let mut per_chunk_visible = std::collections::HashSet::new();
    for mesh in &chunk_meshes {
        per_chunk_visible.extend(visible_unit_faces(
            &mesh.vertices,
            &mesh.indices,
            world_offset,
            &genuine,
        ));
    }

    assert_eq!(
        whole_visible, genuine,
        "banded torus: whole-region visible faces != band-masked ground truth"
    );
    assert_eq!(
        per_chunk_visible, genuine,
        "banded torus: per-chunk apron visible faces != band-masked ground truth"
    );
}

// ---- ADR 0010 E3 — two-layer mesher exposed-face parity (#50) ----

use voxel_core::core_geom::MaterialChoice as MC;
use document::scene::{DefId, Node, NodeContent, NodeTransform, Scene};
use evaluation::two_layer_store::TwoLayerStore;
use voxel_core::voxel::{ShapeKind as TwoLayerShape};
use document::voxel::{SdfShape as TwoLayerSdf};

/// Every unit face a mesh EMITS that ACTUALLY RENDERS — i.e. whose cell on the NORMAL
/// (front) side is AIR per the dense occupancy. A face buried in solid (front solid) is
/// back-face-culled / depth-occluded and never reaches a pixel, so the rendered image is
/// exactly this set. Unlike [`visible_unit_faces`] (which pre-filters to `genuine`), this
/// does NOT discard non-genuine faces — so it CATCHES a spurious face that renders
/// (front air, but no solid behind it), the over-draw bug `visible_unit_faces` hides.
fn renderable_unit_faces(
    vertices: &[CuboidVertex],
    indices: &[u32],
    world_offset: [f32; 3],
    occupied: &std::collections::HashSet<[i64; 3]>,
) -> std::collections::HashSet<UnitFace> {
    unit_faces_in_index_frame(vertices, indices, world_offset)
        .into_iter()
        .filter(|f| {
            // The cell on the NORMAL (front) side of the face. For +sign the plane is the
            // voxel's far edge (front cell index = plane), for -sign the near edge (front
            // cell index = plane - 1). The two in-plane axes carry `f.cell`.
            let (axis, sign) = (f.axis as usize, f.sign);
            let (a, b) = match axis {
                0 => (1usize, 2usize),
                1 => (0usize, 2usize),
                _ => (0usize, 1usize),
            };
            let mut front = [0i64; 3];
            front[axis] = if sign > 0 { f.plane } else { f.plane - 1 };
            front[a] = f.cell[0];
            front[b] = f.cell[1];
            // Renders iff the front cell is AIR (nothing occluding it).
            !occupied.contains(&front)
        })
        .collect()
}

/// The two-layer mesher's RENDERABLE exposed-face set (every emitted face whose front cell
/// is air per the dense occupancy), in the recentred-index frame. Unions over every chunk.
fn two_layer_renderable_faces(
    scene: &Scene,
    density: u32,
    world_offset: [f32; 3],
    recentre: RecentreVoxels,
    grid_dimensions: [u32; 3],
    occupied: &std::collections::HashSet<[i64; 3]>,
) -> std::collections::HashSet<UnitFace> {
    let store = TwoLayerStore::enabled();
    let chunks = store.build_covering_chunks(scene, density, 0);
    let meshes =
        build_two_layer_chunk_meshes(&chunks, grid_dimensions, recentre, density, LayerBand::FULL);
    let mut renderable = std::collections::HashSet::new();
    for mesh in &meshes {
        renderable.extend(renderable_unit_faces(
            &mesh.vertices,
            &mesh.indices,
            world_offset,
            occupied,
        ));
        renderable.extend(renderable_unit_faces(
            &mesh.vertices,
            &mesh.indices_overlay,
            world_offset,
            occupied,
        ));
    }
    renderable
}

/// Assert the two-layer mesher's VISIBLE exposed-face set equals the dense path's —
/// and both equal the ground-truth genuinely-exposed set derived straight from the
/// dense occupancy — for one scene. Mirrors
/// [`per_chunk_apron_exposed_face_set_equals_whole_region`] (the apron parity), but for
/// the two-layer (coarse one-box + microblock cuboid + seam-flag) mesher (ADR 0010 E3).
fn assert_two_layer_face_parity(scene: &Scene, density: u32, label: &str) {
    // The dense assembled grid: ground truth occupancy + the reference whole-grid mesh.
    let dense = scene.resolve_region(scene.full_extent_blocks(density), density, 0);
    assert!(!dense.occupied.is_empty(), "[{label}] scene resolved empty");
    let occupancy = occupancy_indices(&dense);
    let genuine = genuine_exposed_faces(&occupancy);
    let world_offset = grid_world_offset(&dense);

    // Dense reference mesh → its VISIBLE face subset (the existing parity reference).
    let whole = build_cuboid_mesh(&dense, density);
    let mut whole_visible =
        visible_unit_faces(&whole.vertices, &whole.indices, world_offset, &genuine);
    whole_visible.extend(visible_unit_faces(
        &whole.vertices,
        &whole.indices_overlay,
        world_offset,
        &genuine,
    ));
    assert_eq!(
        whole_visible, genuine,
        "[{label}] dense reference visible faces != ground truth"
    );

    // Two-layer mesher → its RENDERABLE face set (every emitted face whose front cell is
    // air — exactly what reaches a pixel). This must equal the ground-truth genuine
    // surface: a SUPERSET would be a spurious rendered face (over-draw that isn't
    // occluded — the boundary-seam over-emit bug), a SUBSET a hole. Strictly stronger
    // than the visible-subset check (which can't see over-emission).
    let two_layer_renderable = two_layer_renderable_faces(
        scene,
        density,
        world_offset,
        // The dense-oracle grid carries its recentre as a raw triple (raw by rule); mint
        // the frame newtype at this test boundary.
        RecentreVoxels::new(dense.recentre_voxels),
        dense.dimensions,
        &occupancy,
    );
    assert_eq!(
        two_layer_renderable, genuine,
        "[{label}] two-layer mesher RENDERABLE faces != ground truth ({} renderable vs {} \
         genuine) — a hole (missing surface) or a spurious rendered seam face (over-draw)",
        two_layer_renderable.len(),
        genuine.len()
    );
}

fn two_layer_tool(
    kind: TwoLayerShape,
    size: [u32; 3],
    offset: [i64; 3],
    material: MC,
    density: u32,
) -> Node {
    let shape = TwoLayerSdf::from_blocks(kind, size, 1, density);
    let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
    node.transform = NodeTransform::from_blocks(offset, density);
    node
}

/// THE E3 GATE (parity (b)): the two-layer mesher's exposed-face SET equals the dense
/// mesher's across the gated scene matrix — SDF shapes (incl. flat/odd), a demo scene,
/// a demo village, a LARGE solid (the one-box coarse path must leave no interior seam),
/// AND an overlapping multi-material scene (the E2 carry-over: overlaps must render
/// identically). Multi-chunk shapes exercise the inter-chunk seam-flag culling.
#[test]
fn two_layer_mesher_exposed_face_set_equals_dense() {
    let density = 16u32;

    // SDF shapes including flat/odd sizes (multi-chunk at d16: an 8-block axis = 128
    // voxels = 2 chunks/axis).
    for kind in [
        TwoLayerShape::Sphere,
        TwoLayerShape::Cylinder,
        TwoLayerShape::Torus,
        TwoLayerShape::Box,
        TwoLayerShape::Tube,
    ] {
        for size in [[5u32, 5, 5], [5, 1, 5], [3, 1, 3], [8, 2, 8]] {
            let scene = Scene::from_nodes(vec![two_layer_tool(
                kind,
                size,
                [0, 0, 0],
                MC::Stone,
                density,
            )]);
            assert_two_layer_face_parity(&scene, density, &format!("{kind:?} {size:?}"));
        }
    }

    // Demo scene: three disjoint Tools (the shot --demo-scene shape set).
    let demo = Scene::from_nodes(vec![
        two_layer_tool(TwoLayerShape::Sphere, [5, 5, 5], [0, 0, 0], MC::Stone, density),
        two_layer_tool(TwoLayerShape::Box, [5, 5, 5], [8, 0, 0], MC::Wood, density),
        two_layer_tool(TwoLayerShape::Torus, [5, 5, 5], [0, 0, 6], MC::Plain, density),
    ]);
    assert_two_layer_face_parity(&demo, density, "demo-scene");

    // Demo village: an instanced definition placed four times (the shot --demo-village).
    {
        let house = DefId(1);
        let mut village = Scene::from_nodes(vec![
            village_instance("House 1", house, [0, 0, 0], density),
            village_instance("House 2", house, [6, 0, 0], density),
            village_instance("House 3", house, [12, 0, 0], density),
            village_instance("House 4", house, [18, 0, 0], density),
        ]);
        village.add_definition(
            house,
            "House".to_string(),
            vec![
                two_layer_tool(TwoLayerShape::Box, [2, 2, 2], [0, 0, 0], MC::Stone, density),
                two_layer_tool(TwoLayerShape::Cylinder, [1, 2, 1], [0, 2, 0], MC::Wood, density),
            ],
        );
        assert_two_layer_face_parity(&village, density, "demo-village");
    }

    // A LARGE solid box: the one-box coarse path must leave no interior seam or hole.
    // 6 blocks @ d16 = 96 voxels/axis = 3 chunks/axis, so the interior chunks are fully
    // coarse-solid and the seam-flag culling spans every chunk boundary.
    let large = Scene::from_nodes(vec![two_layer_tool(
        TwoLayerShape::Box,
        [6, 6, 6],
        [0, 0, 0],
        MC::Stone,
        density,
    )]);
    assert_two_layer_face_parity(&large, density, "large-solid-box");

    // A SketchSolid REVOLVE (the shot --demo-sketch-revolve golden): a boundary-only
    // producer (no coarse-solid blocks — its profile fill is not a coarse box), so this
    // pins the microblock-cuboid + seam path for a non-SDF producer.
    {
        use document::sketch::{PlaneAxis, RevolveAxis, Sketch, SketchPoint, SketchSolid};
        let block = density as i64;
        let r = |b: i64| b * block;
        let h = |b: i64| b * block;
        let profile = vec![
            SketchPoint::new(0, h(0)),
            SketchPoint::new(r(4), h(0)),
            SketchPoint::new(r(4), h(1)),
            SketchPoint::new(r(2), h(3)),
            SketchPoint::new(r(2), h(5)),
            SketchPoint::new(r(4), h(6)),
            SketchPoint::new(r(3), h(8)),
            SketchPoint::new(0, h(8)),
        ];
        let producer =
            SketchSolid::revolve(Sketch::new(PlaneAxis::X, profile), RevolveAxis::InPlane1, 360);
        let revolve = Scene::from_nodes(vec![Node::new(
            "Vase",
            NodeContent::SketchTool {
                producer,
                material: MC::Stone,
            },
        )]);
        assert_two_layer_face_parity(&revolve, density, "sketch-revolve");
    }

    // OVERLAPPING multi-material (E2 carry-over): two boxes overlapping with different
    // materials. The overlap region resolves last-writer-wins (document order) and must
    // render the IDENTICAL exposed-face set as the dense path.
    let overlap = Scene::from_nodes(vec![
        two_layer_tool(TwoLayerShape::Box, [3, 3, 3], [0, 0, 0], MC::Stone, density),
        two_layer_tool(TwoLayerShape::Box, [3, 3, 3], [1, 1, 0], MC::Wood, density),
    ]);
    assert_two_layer_face_parity(&overlap, density, "overlap-multi-material");
}

/// The RENDERABLE exposed-face set of a supplied two-layer chunk set (every emitted face
/// whose front cell is air per `occupied`), unioned over every chunk. Shared by the
/// incremental-edit mesh parity test below so it can mesh a chunk set assembled through the
/// resident cache (post-edit) rather than a fresh `build_covering_chunks`.
fn two_layer_chunk_set_renderable_faces(
    chunks: &[([i32; 3], Arc<evaluation::two_layer_store::TwoLayerChunk>)],
    density: u32,
    world_offset: [f32; 3],
    recentre: RecentreVoxels,
    grid_dimensions: [u32; 3],
    occupied: &std::collections::HashSet<[i64; 3]>,
) -> std::collections::HashSet<UnitFace> {
    let meshes =
        build_two_layer_chunk_meshes(chunks, grid_dimensions, recentre, density, LayerBand::FULL);
    let mut renderable = std::collections::HashSet::new();
    for mesh in &meshes {
        renderable.extend(renderable_unit_faces(
            &mesh.vertices,
            &mesh.indices,
            world_offset,
            occupied,
        ));
        renderable.extend(renderable_unit_faces(
            &mesh.vertices,
            &mesh.indices_overlay,
            world_offset,
            occupied,
        ));
    }
    renderable
}

/// **ADR 0010 #54 GATE (rendered parity):** a scene edited INCREMENTALLY on the two-layer
/// path (build scene A into a [`TwoLayerResidentCache`], invalidate the edit's dirty AABB
/// chunks, re-derive only those) meshes to the SAME renderable exposed-face set as (a) a
/// FULL from-scratch two-layer rebuild of the edited scene B and (b) the dense ground truth.
/// This is the mesh-face-set assertion for an edited scene the acceptance criteria call for:
/// an incremental edit is pixel-identical to a full rebuild and to the dense path.
///
/// Covers move / recolor / resize / add / remove — each keeps the mesher's face set
/// identical to a full rebuild, proving the resident cache's chunk-granular invalidation
/// leaves no stale geometry and misses no fresh surface through the mesher.
#[test]
fn incremental_two_layer_edit_meshes_identically_to_full_rebuild() {
    use evaluation::two_layer_store::TwoLayerResidentCache;
    let density = 16u32;

    // A base scene with a wide sphere (many chunks) plus an interior subject box, so an
    // edit touches a strict subset while much of the chunk set stays resident.
    let scene_a = Scene::from_nodes(vec![
        two_layer_tool(TwoLayerShape::Sphere, [6, 6, 6], [0, 0, 0], MC::Stone, density),
        two_layer_tool(TwoLayerShape::Box, [3, 3, 3], [10, 0, 0], MC::Wood, density),
    ]);

    let recolor = {
        let mut b = scene_a.clone();
        if let NodeContent::Tool { material, .. } = &mut b.root_node_mut(1).content {
            *material = MC::Stone;
        }
        ("recolor", b)
    };
    let resize = {
        let mut b = scene_a.clone();
        let replacement =
            two_layer_tool(TwoLayerShape::Box, [2, 2, 2], [10, 0, 0], MC::Wood, density);
        let slot = b.root_node_mut(1);
        slot.content = replacement.content;
        slot.transform = replacement.transform;
        ("resize", b)
    };
    let move_edit = {
        let mut b = scene_a.clone();
        b.root_node_mut(1).transform = NodeTransform::from_blocks([13, 0, 0], density);
        ("move", b)
    };
    let add_edit = {
        let mut b = scene_a.clone();
        b.add_node(two_layer_tool(
            TwoLayerShape::Box,
            [2, 2, 2],
            [16, 0, 0],
            MC::Plain,
            density,
        ));
        ("add", b)
    };
    let remove_edit = {
        let mut b = scene_a.clone();
        let subject = b.roots[1];
        b.remove_node(subject);
        ("remove", b)
    };

    for (label, scene_b) in [recolor, resize, move_edit, add_edit, remove_edit] {
        // Drive the incremental edit through the resident cache exactly as app_core would.
        let mut cache = TwoLayerResidentCache::enabled();
        let _ = cache.resident_two_layer_chunks(&scene_a, density, 0);
        let index_a = scene_a.build_leaf_spatial_index(density);
        let index_b = scene_b.build_leaf_spatial_index(density);
        match index_b.edit_aabb_since(&index_a) {
            Some(edit_aabb) => {
                cache.invalidate_aabb(&edit_aabb, density);
            }
            None => cache.clear(),
        }
        let incremental_chunks: Vec<_> = cache
            .resident_two_layer_chunks(&scene_b, density, 0)
            .into_iter()
            .map(|(coord, chunk)| (coord, chunk.clone()))
            .collect();

        // The dense ground truth for scene B (the pixel reference).
        let dense = scene_b.resolve_region(scene_b.full_extent_blocks(density), density, 0);
        assert!(!dense.occupied.is_empty(), "[{label}] scene resolved empty");
        let occupancy = occupancy_indices(&dense);
        let genuine = genuine_exposed_faces(&occupancy);
        let world_offset = grid_world_offset(&dense);

        // The incrementally-edited chunk set's renderable face set == dense ground truth.
        let incremental_faces = two_layer_chunk_set_renderable_faces(
            &incremental_chunks,
            density,
            world_offset,
            RecentreVoxels::new(dense.recentre_voxels),
            dense.dimensions,
            &occupancy,
        );
        assert_eq!(
            incremental_faces, genuine,
            "[{label}] incrementally-edited two-layer mesh RENDERABLE faces != dense ground \
             truth — a stale chunk (over-draw) or a missed fresh surface (hole)"
        );

        // And identical to a FULL from-scratch two-layer rebuild of scene B.
        let full_faces = two_layer_renderable_faces(
            &scene_b,
            density,
            world_offset,
            RecentreVoxels::new(dense.recentre_voxels),
            dense.dimensions,
            &occupancy,
        );
        assert_eq!(
            incremental_faces, full_faces,
            "[{label}] incremental two-layer mesh faces must equal a FULL two-layer rebuild"
        );
    }
}

/// One chunk's GPU-buffer proxy: `(vertex bytes, overlay-off indices, overlay-on indices)`.
type ChunkBufferProxy = (Vec<u8>, Vec<u32>, Vec<u32>);

/// Map a two-layer mesh build to `coord -> (vertex bytes, off-indices, overlay-indices)` —
/// the per-chunk GPU buffer set proxy (the renderer uploads exactly these bytes via
/// [`upload_chunk_meshes`], concatenating the two index runs), so a byte-equal map ==
/// a byte-equal buffer set. Carries BOTH index runs (unlike the dense `mesh_map`) so the
/// overlay split is part of the parity claim.
fn two_layer_mesh_map(
    meshes: &[CuboidChunkMesh],
) -> std::collections::HashMap<[i32; 3], ChunkBufferProxy> {
    meshes
        .iter()
        .map(|m| {
            (
                m.coord,
                (
                    bytemuck::cast_slice::<_, u8>(&m.vertices).to_vec(),
                    m.indices.clone(),
                    m.indices_overlay.clone(),
                ),
            )
        })
        .collect()
}

/// **THE ISSUE #55 GATE (byte-parity + only-dirty-remeshed).** Drive an edit through the
/// [`TwoLayerResidentCache`] exactly as `AppCore::rebuild` does, then apply the
/// [`cuboid_incremental_plan`] on the two-layer chunk source exactly as
/// [`CuboidMeshRenderer::incremental_rebuild_from_two_layer_chunks`] does, and assert:
///
/// 1. **Byte-parity** — the incrementally-updated per-chunk buffer set (kept-buffers ∪
///    freshly-meshed rebuild subset, minus evicted) is BYTE-IDENTICAL to a wholesale
///    two-layer re-mesh of the edited scene. This is the GPU-buffer analogue of the dense
///    [`incremental_cuboid_rebuild_equals_wholesale`], and of the face-set
///    [`incremental_two_layer_edit_meshes_identically_to_full_rebuild`].
/// 2. **Only-dirty-remeshed (the perf proof)** — the re-meshed set is the plan's dirty +
///    26-neighbourhood-dilated rebuild set, STRICTLY smaller than the whole resident set
///    (quantified on a many-chunk scene). Without this the slice is unverified: it is what
///    proves per-edit mesh cost scales with the dirty set, not the scene size.
#[test]
fn incremental_two_layer_gpu_buffer_rebuild_equals_wholesale() {
    use evaluation::two_layer_store::TwoLayerResidentCache;
    let density = 16u32;

    // Two ANCHOR boxes at fixed extremes PLUS an interior subject box. The anchors are
    // present in EVERY scene, so the composite bounds — hence the recentre (floating origin)
    // — stay PINNED across each edit. That models the live guard: the two-layer GPU-buffer
    // incremental only runs when the recentre did NOT shift (a shift re-frames every kept
    // buffer's baked vertices → the shell falls back to wholesale, `app_core.rs`). With the
    // recentre pinned, the incremental path is genuinely exercised, and the subject box sits
    // far from the anchors so an edit dirties a strict subset while most chunks stay resident.
    let anchors = || {
        vec![
            two_layer_tool(TwoLayerShape::Box, [2, 2, 2], [-14, 0, 0], MC::Stone, density),
            two_layer_tool(TwoLayerShape::Box, [2, 2, 2], [14, 8, 6], MC::Stone, density),
        ]
    };
    // scene_a: anchors + subject box (index 2) at a fixed offset well inside the bounds.
    let scene_a = {
        let mut nodes = anchors();
        nodes.push(two_layer_tool(TwoLayerShape::Box, [3, 3, 3], [4, 2, 2], MC::Wood, density));
        Scene::from_nodes(nodes)
    };
    let subject = 2usize;

    let recolor = {
        let mut b = scene_a.clone();
        if let NodeContent::Tool { material, .. } = &mut b.root_node_mut(subject).content {
            *material = MC::Plain;
        }
        ("recolor", b)
    };
    let resize = {
        // Shrink the subject box; the anchors keep the bounds pinned so the recentre holds.
        let mut b = scene_a.clone();
        let replacement =
            two_layer_tool(TwoLayerShape::Box, [2, 2, 2], [4, 2, 2], MC::Wood, density);
        let slot = b.root_node_mut(subject);
        slot.content = replacement.content;
        slot.transform = replacement.transform;
        ("resize", b)
    };
    let move_edit = {
        // Move the subject WITHIN the anchored bounds (recentre unchanged).
        let mut b = scene_a.clone();
        b.root_node_mut(subject).transform = NodeTransform::from_blocks([6, 2, 2], density);
        ("move", b)
    };
    let remove_edit = {
        let mut b = scene_a.clone();
        let subject_id = b.roots[subject];
        b.remove_node(subject_id);
        ("remove", b)
    };

    for (label, scene_b) in [recolor, resize, move_edit, remove_edit] {
        // Build scene A's resident set + its owned two-layer chunk source (what the renderer
        // retains in `source_two_layer_chunks`).
        let mut cache = TwoLayerResidentCache::enabled();
        let chunks_a: Vec<([i32; 3], Arc<TwoLayerChunk>)> =
            cache.resident_two_layer_chunks(&scene_a, density, 0);
        let dims = scene_a.placed_region_dimensions(density);
        let recentre = scene_a.recentre_voxels_for_resolve(density);
        // The renderer's initial (wholesale) buffer set for A.
        let wholesale_a = build_two_layer_chunk_meshes(
            &chunks_a,
            dims,
            recentre,
            density,
            LayerBand::FULL,
        );

        // The edit: invalidate the dirty AABB (or clear), then re-derive the resident set.
        let index_a = scene_a.build_leaf_spatial_index(density);
        let index_b = scene_b.build_leaf_spatial_index(density);
        let evicted_dirty: Vec<[i32; 3]> = match index_b.edit_aabb_since(&index_a) {
            Some(edit_aabb) => cache.invalidate_aabb(&edit_aabb, density),
            None => {
                cache.clear();
                Vec::new()
            }
        };
        // These edits are all localisable (a single moved/edited leaf), so the AABB path
        // must have been taken — the perf claim only holds on the localised path.
        assert!(
            !evicted_dirty.is_empty(),
            "[{label}] expected a localisable edit (non-empty evicted-dirty set)"
        );
        let dims_b = scene_b.placed_region_dimensions(density);
        let recentre_b = scene_b.recentre_voxels_for_resolve(density);
        // The anchors pin the bounds, so the recentre must NOT shift — the precondition
        // under which the GPU-buffer incremental keeps untouched chunks' baked vertices.
        assert_eq!(
            recentre_b, recentre,
            "[{label}] anchors should pin the recentre; a shift would force wholesale fallback"
        );
        let chunks_b: Vec<([i32; 3], Arc<TwoLayerChunk>)> =
            cache.resident_two_layer_chunks(&scene_b, density, 0);

        // The plan — dilate the dirty set by the 26-neighbourhood, keep only non-empty
        // chunks — exactly as `incremental_rebuild_from_two_layer_chunks` computes it.
        let resident: Vec<[i32; 3]> = chunks_a.iter().map(|(c, _)| *c).collect();
        let occupied: Vec<[i32; 3]> = chunks_b
            .iter()
            .filter(|(_, chunk)| chunk.has_geometry())
            .map(|(coord, _)| *coord)
            .collect();
        let plan = cuboid_incremental_plan(&resident, &evicted_dirty, &occupied);

        // --- ONLY-DIRTY-REMESHED (the perf proof) ---
        // The re-meshed set is STRICTLY smaller than the resident set — most chunks keep
        // their buffers. Quantify the gap so a regression to wholesale re-mesh fails here.
        assert!(
            plan.rebuild.len() < resident.len(),
            "[{label}] incremental re-mesh must touch FEWER than every resident chunk \
             (rebuilt {} of {} resident) — else it's a wholesale re-mesh regression",
            plan.rebuild.len(),
            resident.len(),
        );

        // Re-mesh ONLY the rebuild subset (seam culling from the FULL post-edit set) and
        // confirm the filtered build produced meshes for EXACTLY that subset (∩ non-empty)
        // — never the whole resident set.
        let rebuild_set: std::collections::HashSet<[i32; 3]> =
            plan.rebuild.iter().copied().collect();
        let rebuilt = build_two_layer_chunk_meshes_filtered(
            &chunks_b,
            Some(&rebuild_set),
            dims_b,
            recentre_b,
            density,
            LayerBand::FULL,
        );
        let remeshed_coords: std::collections::HashSet<[i32; 3]> =
            rebuilt.iter().map(|m| m.coord).collect();
        assert!(
            remeshed_coords.is_subset(&rebuild_set),
            "[{label}] the filtered two-layer build meshed a chunk OUTSIDE the dirty-dilated \
             rebuild set — seam culling must read neighbours but only EMIT the subset"
        );

        // --- BYTE-PARITY ---
        // Apply the plan to A's buffer map (drop evicted + rebuild coords, insert the fresh
        // meshes) and assert it byte-equals a wholesale re-mesh of B — the exact ops
        // `incremental_rebuild_from_two_layer_chunks` performs on `chunk_buffers`.
        let mut result = two_layer_mesh_map(&wholesale_a);
        for coord in &plan.evict {
            result.remove(coord);
        }
        for coord in &plan.rebuild {
            result.remove(coord); // a rebuild coord meshing to empty drops out here
        }
        for (coord, entry) in two_layer_mesh_map(&rebuilt) {
            result.insert(coord, entry);
        }

        let wholesale_b = build_two_layer_chunk_meshes(
            &chunks_b,
            dims_b,
            recentre_b,
            density,
            LayerBand::FULL,
        );
        let target = two_layer_mesh_map(&wholesale_b);
        assert_eq!(
            result, target,
            "[{label}] incremental two-layer buffer set must byte-equal the wholesale rebuild"
        );
    }
}

/// Band-masked occupancy of a dense grid, keyed in the recentred-index frame the two-layer
/// mesher emits in (a voxel at recentred index `v` sits at absolute layer `v[2] + half_z`,
/// FLOORED half — the SAME map the two-layer band clip inverts). Mirrors the banded-torus
/// test's masking, restated in the RECENTRED frame so it lines up with `world_offset`.
fn banded_occupancy_indices(
    dense: &VoxelGrid,
    band: LayerBand,
) -> std::collections::HashSet<[i64; 3]> {
    let mut min_world = [f32::INFINITY; 3];
    for v in &dense.occupied {
        let position = v.world_position();
        for (axis, m) in min_world.iter_mut().enumerate() {
            *m = m.min(position[axis]);
        }
    }
    let half_z = (dense.dimensions[2] / 2) as f32;
    let base_layer = (min_world[2] + half_z).floor() as i64;
    dense
        .occupied
        .iter()
        .map(|v| {
            let position = v.world_position();
            [
                (position[0] - min_world[0]).round() as i64,
                (position[1] - min_world[1]).round() as i64,
                (position[2] - min_world[2]).round() as i64,
            ]
        })
        .filter(|idx| {
            let layer = base_layer + idx[2];
            layer >= band.band_min as i64 && layer <= band.band_max as i64
        })
        .collect()
}

/// ADR 0010 #53 GATE: the two-layer BANDED mesher's RENDERABLE face set equals the dense
/// path's band-masked genuine surface — proving the band reclip (coarse clipped one-box,
/// microblock cuboid clip, cut-plane cap faces) is a pure optimisation on the data seam,
/// identical to `build_cuboid_mesh_banded` on the dense path. Because `renderable_unit_faces`
/// tests the front cell against the BAND-MASKED occupancy, a cut-plane cap face (front cell
/// out of band ⇒ air) MUST be emitted, and a spurious over-emit or a hole both fail.
fn assert_two_layer_banded_face_parity(
    scene: &Scene,
    density: u32,
    band: LayerBand,
    label: &str,
) {
    let dense = scene.resolve_region(scene.full_extent_blocks(density), density, 0);
    assert!(!dense.occupied.is_empty(), "[{label}] scene resolved empty");
    let banded = banded_occupancy_indices(&dense, band);
    assert!(!banded.is_empty(), "[{label}] band kept no voxels");
    let genuine = genuine_exposed_faces(&banded);
    let world_offset = grid_world_offset(&dense);

    // Dense banded reference → its visible face subset must equal the banded ground truth.
    let whole = build_cuboid_mesh_banded(&dense, density, band);
    let mut whole_visible =
        visible_unit_faces(&whole.vertices, &whole.indices, world_offset, &genuine);
    whole_visible.extend(visible_unit_faces(
        &whole.vertices,
        &whole.indices_overlay,
        world_offset,
        &genuine,
    ));
    assert_eq!(
        whole_visible, genuine,
        "[{label}] dense banded reference visible faces != band-masked ground truth"
    );

    // Two-layer banded mesher → its RENDERABLE face set (front cell tested against the
    // band-masked occupancy) must equal the same ground truth.
    let store = TwoLayerStore::enabled();
    let chunks = store.build_covering_chunks(scene, density, 0);
    let meshes = build_two_layer_chunk_meshes(
        &chunks,
        dense.dimensions,
        RecentreVoxels::new(dense.recentre_voxels),
        density,
        band,
    );
    let mut renderable = std::collections::HashSet::new();
    for mesh in &meshes {
        renderable.extend(renderable_unit_faces(
            &mesh.vertices,
            &mesh.indices,
            world_offset,
            &banded,
        ));
        renderable.extend(renderable_unit_faces(
            &mesh.vertices,
            &mesh.indices_overlay,
            world_offset,
            &banded,
        ));
    }
    assert_eq!(
        renderable, genuine,
        "[{label}] two-layer BANDED renderable faces != band-masked ground truth ({} vs {}) \
         — a hole, a spurious cut-plane over-emit, or a missing cap face",
        renderable.len(),
        genuine.len()
    );
}

/// THE ADR 0010 #53 GATE: the two-layer mesher honours a layer band identically to the dense
/// banded path across a matrix of bands — a band that CUTS through coarse-solid interiors (a
/// large box: the clipped one-box + cut cap face), a band that clips microblock cuboids (a
/// sphere), a band flush to a block boundary, and a thin single-block band. Multi-chunk at
/// d16 so the clip crosses chunk seams.
#[test]
fn two_layer_banded_mesher_matches_dense() {
    let density = 16u32;
    let band = |lo: u32, hi: u32| LayerBand {
        band_min: lo,
        band_max: hi,
        onion_depth: 0,
    };

    // Large solid box (8 blocks = 128 voxels/axis = 2 chunks/axis): the coarse one-box
    // interior must clip to the band and cap at the cut plane, across the chunk seam.
    let large = Scene::from_nodes(vec![two_layer_tool(
        TwoLayerShape::Box,
        [8, 8, 8],
        [0, 0, 0],
        MC::Stone,
        density,
    )]);
    // A band cutting mid-block (layer 40 is inside block 2 at d16), a block-flush band, and
    // a thin single-layer slice.
    for b in [band(0, 40), band(48, 96), band(0, 63), band(70, 70)] {
        assert_two_layer_banded_face_parity(&large, density, b, "large-box-band");
    }

    // Sphere: a rounded boundary — the microblock cuboids clip to the band and the equator
    // slice exposes a filled cross-section cap.
    let sphere = Scene::from_nodes(vec![two_layer_tool(
        TwoLayerShape::Sphere,
        [5, 5, 5],
        [0, 0, 0],
        MC::Stone,
        density,
    )]);
    for b in [band(0, 40), band(30, 50)] {
        assert_two_layer_banded_face_parity(&sphere, density, b, "sphere-band");
    }
}

fn village_instance(name: &str, def: DefId, offset: [i64; 3], density: u32) -> Node {
    let mut node = Node::new(name, NodeContent::Instance(def));
    node.transform = NodeTransform::from_blocks(offset, density);
    node
}
