use super::*;

// ---- Per-chunk apron meshing structural parity (issue #20 S6c-2d) ----

/// Bucket a whole grid into per-chunk sub-grids exactly as the renderer's `new`
/// wrapper does (`floor(world / chunk_extent)`), so the per-chunk mesher sees the
/// SAME partition the live path does.
fn bucket_for_test(grid: &VoxelGrid, voxels_per_block: u32) -> Vec<([i32; 3], VoxelGrid)> {
    crate::mesh::bucket_grid_into_chunk_grids(grid, voxels_per_block)
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
        let wholesale_a = build_chunk_meshes_with_apron(&refs_a, dims, LayerBand::FULL, None);
        let wholesale_b = build_chunk_meshes_with_apron(&refs_b, dims, LayerBand::FULL, None);

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
            build_chunk_meshes_with_apron_filtered(&refs_b, Some(&rebuild_set), dims, LayerBand::FULL, None);

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
                build_chunk_meshes_with_apron(&chunk_refs, dims, LayerBand::FULL, None);
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
    let chunk_meshes = build_chunk_meshes_with_apron(&chunk_refs, dims, LayerBand::FULL, None);
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
    let chunk_meshes = build_chunk_meshes_with_apron(&chunk_refs, dims, band, None);
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
