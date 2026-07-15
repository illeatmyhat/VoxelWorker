use super::*;

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
