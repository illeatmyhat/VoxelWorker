//! ADR 0011 slice G3 — the incremental dirty-brick atlas update net. The load-bearing
//! assertion: an [`IncrementalBrickField`] patched edit-by-edit (only dirty chunks
//! re-evaluated, slots free-listed) is byte-exact vs a from-scratch [`build_brick_field`]
//! of the SAME scene, after EVERY step, across explicit block-kind transitions
//! (air↔sculpted↔coarse) and add / move / recolour / delete edits.
use crate::brick::*;
use voxel_core::core_geom::MaterialChoice;
use document::scene::{Node, NodeContent, NodeTransform, Scene};
use evaluation::two_layer_store::{TwoLayerChunk, TwoLayerResidentCache};
use voxel_core::voxel::ShapeKind;
use document::voxel::SdfShape;

/// The owned covering set the shell feeds `apply_dirty_update` / `build_brick_field`
/// (the resident cache borrows, so clone out — exactly as `AppCore::rebuild` does).
fn covering_owned(
    cache: &mut TwoLayerResidentCache,
    scene: &Scene,
    density: u32,
) -> Vec<([i32; 3], Arc<TwoLayerChunk>)> {
    cache.resident_two_layer_chunks(scene, density, 0)
}

/// A tool node (single material, so the scene stays brick-representable) of `blocks³`
/// at a block offset — the small edited object.
fn tool(kind: ShapeKind, offset_blocks: [i64; 3], material: MaterialChoice, density: u32) -> Node {
    let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, density);
    let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
    node.transform = NodeTransform::from_blocks(offset_blocks, density);
    node
}

/// The set of atlas slots the live sculpted records reference, plus a check that no
/// two live records share a slot (a "ghost brick" would show as a duplicate).
fn live_slots(build: &BrickFieldBuild) -> std::collections::BTreeSet<u32> {
    let mut slots = std::collections::BTreeSet::new();
    for record in &build.brick_records {
        if let BrickPayload::Sculpted { atlas_slot } = record.payload {
            assert!(
                slots.insert(atlas_slot),
                "live slot {atlas_slot} referenced twice (ghost brick)"
            );
        }
    }
    slots
}

/// Assert the incremental field materialisation is byte-exact vs the wholesale build
/// of the same scene: SAME record keys, kinds, materials, seam flags; each sculpted
/// record's atlas bytes equal (slot NUMBERS differ — the free-list vs dense `0..count`
/// — so compare the occupancy, not the slot). Free slots may hold garbage: they are
/// asserted unreachable from live records (the `live_slots` uniqueness check).
fn assert_incremental_matches_wholesale(
    incremental: &BrickFieldBuild,
    wholesale: &BrickFieldBuild,
    label: &str,
) {
    assert_eq!(
        incremental.brick_edge_voxels, wholesale.brick_edge_voxels,
        "[{label}] brick edge must match"
    );
    assert_eq!(
        incremental.brick_records.len(),
        wholesale.brick_records.len(),
        "[{label}] record count must match wholesale"
    );
    let _ = live_slots(incremental); // no ghost bricks (live slots unique)
    for whole_record in &wholesale.brick_records {
        let block = unpack_world_block_key(whole_record.packed_world_block_key);
        let inc_record = incremental
            .find_record(block)
            .unwrap_or_else(|| panic!("[{label}] incremental missing record at {block:?}"));
        assert_eq!(
            inc_record.packed_world_block_key, whole_record.packed_world_block_key,
            "[{label}] key mismatch at {block:?}"
        );
        assert_eq!(
            inc_record.material_id, whole_record.material_id,
            "[{label}] material mismatch at {block:?}"
        );
        assert_eq!(
            inc_record.seam_solidity, whole_record.seam_solidity,
            "[{label}] seam-solidity mismatch at {block:?}"
        );
        assert_eq!(
            inc_record.payload.kind_discriminant(),
            whole_record.payload.kind_discriminant(),
            "[{label}] kind mismatch at {block:?}"
        );
        match (inc_record.payload, whole_record.payload) {
            (
                BrickPayload::CoarseSolid { block_id: a },
                BrickPayload::CoarseSolid { block_id: b },
            ) => assert_eq!(a, b, "[{label}] coarse block id mismatch at {block:?}"),
            (
                BrickPayload::Sculpted { atlas_slot: inc_slot },
                BrickPayload::Sculpted { atlas_slot: whole_slot },
            ) => {
                // Slot NUMBERS differ (free-list vs dense) — compare the bytes.
                assert_eq!(
                    incremental.sculpted_brick_occupancy(inc_slot),
                    wholesale.sculpted_brick_occupancy(whole_slot),
                    "[{label}] sculpted occupancy bytes mismatch at {block:?}"
                );
            }
            _ => panic!("[{label}] payload kind disagreement at {block:?}"),
        }
    }
}

/// THE parity gate for G3 (issue #69 acceptance): drive a scripted sequence of edits
/// — recolour, move, shape-swap, delete, re-add — applying each INCREMENTALLY, and
/// after every step assert the incremental field equals a from-scratch wholesale build
/// of the same scene. Two fixed anchor tools at the extremes pin the covering set so an
/// incremental edit never grows it (the app's reframe guard — a growth routes wholesale).
/// A non-16 density exercises the block-denominated granule.
#[test]
fn incremental_dirty_update_equals_wholesale_after_every_step() {
    let density = 4u32;
    let material = MaterialChoice::Stone;
    // Two anchors far apart fix the covering chunk range; the middle tool is edited.
    let anchor_lo = tool(ShapeKind::Box, [-14, 0, 0], material, density);
    let anchor_hi = tool(ShapeKind::Box, [14, 0, 0], material, density);
    let scene_with = |middle: Option<Node>| {
        let mut nodes = vec![anchor_lo.clone(), anchor_hi.clone()];
        if let Some(m) = middle {
            nodes.push(m);
        }
        Scene::from_nodes(nodes)
    };

    // The scripted edits (each keeps the anchors, edits the middle) — chosen to force
    // block-kind transitions: add (air→sculpted/coarse), move (sculpted↔air↔coarse),
    // recolour (sculpted/coarse material change), shape-swap (occupancy change), delete.
    let scenes = [
        ("initial", scene_with(None)),
        ("add-sphere", scene_with(Some(tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Wood, density)))),
        ("recolour", scene_with(Some(tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Plain, density)))),
        ("move", scene_with(Some(tool(ShapeKind::Sphere, [2, 1, 0], MaterialChoice::Plain, density)))),
        ("shape-swap", scene_with(Some(tool(ShapeKind::Box, [2, 1, 0], MaterialChoice::Plain, density)))),
        ("delete", scene_with(None)),
        ("re-add", scene_with(Some(tool(ShapeKind::Torus, [0, 0, 0], MaterialChoice::Wood, density)))),
    ];

    let mut cache = TwoLayerResidentCache::enabled();
    cache.clear();
    let scene0 = &scenes[0].1;
    let mut previous_index = scene0.build_leaf_spatial_index(density);
    let fresh0 = covering_owned(&mut cache, scene0, density);
    let build0 = build_brick_field(&fresh0, density);
    let mut field = IncrementalBrickField::from_wholesale(build0.clone()).0;
    let mut covering: std::collections::BTreeSet<[i32; 3]> =
        fresh0.iter().map(|(coord, _)| *coord).collect();
    assert_incremental_matches_wholesale(&field.to_build(), &build0, scenes[0].0);

    let mut incremental_steps = 0usize;
    for (label, scene) in &scenes[1..] {
        let new_index = scene.build_leaf_spatial_index(density);
        let edit_aabb = new_index.edit_aabb_since(&previous_index);
        // Mirror `AppCore::rebuild`: localisable edit → invalidate its chunks; a `None`
        // (wholesale) edit clears. Build the fresh covering set AFTER invalidation.
        let dirty = match &edit_aabb {
            Some(aabb) => cache.invalidate_aabb(aabb, density),
            None => {
                cache.clear();
                Vec::new()
            }
        };
        let fresh = covering_owned(&mut cache, scene, density);
        let new_covering: std::collections::BTreeSet<[i32; 3]> =
            fresh.iter().map(|(coord, _)| *coord).collect();

        // Incremental applies only when localisable AND the covering set is invariant
        // (the app routes a growth/reframe wholesale). Otherwise reset from wholesale.
        if edit_aabb.is_some() && new_covering == covering {
            field.apply_dirty_update(&fresh, &dirty);
            incremental_steps += 1;
        } else {
            let build = build_brick_field(&fresh, density);
            field = IncrementalBrickField::from_wholesale(build).0;
        }
        covering = new_covering;

        let wholesale = build_brick_field(&fresh, density);
        assert_incremental_matches_wholesale(&field.to_build(), &wholesale, label);
        previous_index = new_index;
    }
    assert!(
        incremental_steps >= 4,
        "the script must exercise the INCREMENTAL path on most steps (was {incremental_steps})"
    );
}

/// Untouched-slot discipline (issue #69 acceptance): an edit confined to ONE chunk
/// writes only that chunk's blocks' slots (+ frees), never the whole scene's — the
/// "per-edit cost ∝ dirty region" claim made testable. A recolour keeps occupancy
/// identical, so exactly the dirty chunk's sculpted blocks are freed + rewritten.
#[test]
fn one_chunk_edit_writes_only_that_chunks_slots() {
    let density = 4u32;
    // Anchors fix the covering set; a compact middle tool occupies its own chunks.
    let anchor_lo = tool(ShapeKind::Box, [-14, 0, 0], MaterialChoice::Stone, density);
    let anchor_hi = tool(ShapeKind::Box, [14, 0, 0], MaterialChoice::Stone, density);
    let scene_a = Scene::from_nodes(vec![
        anchor_lo.clone(),
        anchor_hi.clone(),
        tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Wood, density),
    ]);
    let scene_b = Scene::from_nodes(vec![
        anchor_lo,
        anchor_hi,
        // Same shape/placement, DIFFERENT material — a pure recolour (occupancy fixed).
        tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Plain, density),
    ]);

    let mut cache = TwoLayerResidentCache::enabled();
    cache.clear();
    let index_a = scene_a.build_leaf_spatial_index(density);
    let fresh_a = covering_owned(&mut cache, &scene_a, density);
    let build_a = build_brick_field(&fresh_a, density);
    let mut field = IncrementalBrickField::from_wholesale(build_a.clone()).0;
    let total_sculpted = build_a.sculpted_brick_count();

    let index_b = scene_b.build_leaf_spatial_index(density);
    let edit_aabb = index_b
        .edit_aabb_since(&index_a)
        .expect("a recolour is a localisable edit");
    let dirty = cache.invalidate_aabb(&edit_aabb, density);
    let fresh_b = covering_owned(&mut cache, &scene_b, density);

    // Count the sculpted blocks living in the dirty chunks (the recolour re-writes
    // exactly these — occupancy is unchanged, only the record material differs).
    let dirty_set: std::collections::BTreeSet<[i32; 3]> = dirty.iter().copied().collect();
    let expected_written: usize = fresh_b
        .iter()
        .filter(|(coord, _)| dirty_set.contains(coord))
        .map(|(_, chunk)| chunk.microblocks.len())
        .sum();

    let update = field.apply_dirty_update(&fresh_b, &dirty);

    assert!(
        !dirty.is_empty() && dirty.len() < covering_owned(&mut cache, &scene_b, density).len(),
        "the edit must dirty SOME but not ALL chunks (dirtied {} of the covering set)",
        dirty.len()
    );
    assert_eq!(
        update.written_slots.len(),
        expected_written,
        "an edit must write exactly the dirty chunks' sculpted slots, no more"
    );
    assert!(
        update.written_slots.len() < total_sculpted,
        "a one-region edit must write FEWER than every scene slot ({} of {})",
        update.written_slots.len(),
        total_sculpted
    );
    // A pure recolour keeps occupancy, so freed == rewritten (slots recycled in place)
    // and the atlas does not grow.
    assert_eq!(update.freed_slots.len(), expected_written, "recolour frees what it rewrites");
    assert!(!update.atlas_grew, "a recolour does not grow the atlas");
    // And the result is still byte-exact vs wholesale.
    let wholesale = build_brick_field(&fresh_b, density);
    assert_incremental_matches_wholesale(&field.to_build(), &wholesale, "one-chunk-recolour");
}

/// **The patch-parity witness (item 9).** The renderer's patch path no longer materialises
/// `to_build()` per edit — it reads each dirty slot's bytes and the atlas geometry straight
/// from the mirror. This pins those owner-side accessors to what a `to_build()`
/// materialisation would have produced: after an incremental edit, every written slot's
/// `sculpted_slot_bytes` equals `to_build().sculpted_brick_occupancy` for that slot, and
/// `atlas_geometry()` matches the build's tile geometry. If these ever drift, the GPU patch
/// would upload the wrong texels while the parity gate (which still uses `to_build`) stayed
/// green — so this is the guard the deleted per-edit `to_build` used to provide implicitly.
#[test]
fn patched_slot_bytes_and_geometry_match_to_build_materialisation() {
    let density = 4u32;
    let anchor_lo = tool(ShapeKind::Box, [-14, 0, 0], MaterialChoice::Stone, density);
    let anchor_hi = tool(ShapeKind::Box, [14, 0, 0], MaterialChoice::Stone, density);
    let scene_a = Scene::from_nodes(vec![
        anchor_lo.clone(),
        anchor_hi.clone(),
        tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Wood, density),
    ]);
    // A pure recolour (occupancy fixed) — writes the dirty chunk's slots without growing.
    let scene_b = Scene::from_nodes(vec![
        anchor_lo,
        anchor_hi,
        tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Plain, density),
    ]);

    let mut cache = TwoLayerResidentCache::enabled();
    cache.clear();
    let index_a = scene_a.build_leaf_spatial_index(density);
    let fresh_a = covering_owned(&mut cache, &scene_a, density);
    let build_a = build_brick_field(&fresh_a, density);
    let mut field = IncrementalBrickField::from_wholesale(build_a).0;

    let index_b = scene_b.build_leaf_spatial_index(density);
    let edit_aabb = index_b
        .edit_aabb_since(&index_a)
        .expect("a recolour is a localisable edit");
    let dirty = cache.invalidate_aabb(&edit_aabb, density);
    let fresh_b = covering_owned(&mut cache, &scene_b, density);

    let update = field.apply_dirty_update(&fresh_b, &dirty);
    assert!(!update.written_slots.is_empty(), "the recolour must write some slots");

    // The materialisation the patch path used to build per edit — the witness.
    let materialised = field.to_build();
    let geometry = field.atlas_geometry();
    assert_eq!(
        geometry.brick_edge_voxels, materialised.brick_edge_voxels,
        "mirror edge matches the materialisation"
    );
    assert_eq!(
        geometry.bricks_per_axis, materialised.bricks_per_axis,
        "mirror tile-grid edge matches the materialisation"
    );
    assert_eq!(
        geometry.atlas_dim_voxels, materialised.atlas_dim_voxels,
        "mirror atlas dimension matches the materialisation"
    );
    for &slot in &update.written_slots {
        assert_eq!(
            field.sculpted_slot_bytes(slot),
            materialised.sculpted_brick_occupancy(slot),
            "written slot {slot} bytes must equal the to_build() materialisation"
        );
    }
    // The full re-pack payload equals the materialisation's atlas byte-for-byte.
    assert_eq!(
        field.pack_atlas_payload().bytes,
        materialised.sculpted_atlas_bytes,
        "the grow-path re-pack equals the materialised atlas"
    );
}

/// **The occlusion-dilation seam (ADR 0011 interior elision).** Under the surface-only
/// record contract, an edit can flip records in NON-dirty neighbour chunks: carving away
/// a block un-occludes the face-adjacent blocks across the chunk boundary (their records
/// must APPEAR), and filling it back occludes them again (their records must VANISH).
/// Two chunk-filling solid boxes abut across a chunk boundary; deleting the second is
/// the carve, re-adding it the fill. After each step the incrementally-patched field
/// must equal a from-scratch surface-only wholesale build byte-for-byte — this is what
/// the 26-neighbourhood ring re-derivation in `apply_dirty_update` exists for. The test
/// also asserts the scenario is REAL: the carve changes the record set of a chunk that
/// was NOT in the dirty set (else the fixture is vacuous).
#[test]
fn incremental_carve_across_chunk_boundary_flips_neighbour_occlusion() {
    let density = 4u32;
    let material = MaterialChoice::Stone;
    let chunk_span = CHUNK_BLOCKS as i64;
    // A solid SKETCH-EXTRUDE cube of exactly CHUNK_BLOCKS³ blocks at a chunk-aligned
    // offset — the sketch producer classifies COARSE-solid blocks to the very face
    // (unlike an SDF Box tool, whose 1-block shell resolves as boundary microblocks and
    // would never exercise coarse-record occlusion flips at the interface).
    let chunk_filling_box = |offset_blocks: [i64; 3]| -> Node {
        let edge_voxels = chunk_span * density as i64;
        let producer = document::sketch::SketchSolid::extrude(
            document::sketch::Sketch::rectangle(
                document::sketch::PlaneAxis::Z,
                edge_voxels,
                edge_voxels,
            ),
            edge_voxels as u32,
        );
        let mut node = Node::new(
            format!("box@{offset_blocks:?}"),
            NodeContent::SketchTool { producer, material },
        );
        node.transform = NodeTransform::from_blocks(offset_blocks, density);
        node
    };
    // Anchors pin the covering set so the delete / re-add stays an incremental edit.
    let anchor_lo = chunk_filling_box([-4 * chunk_span, 0, 0]);
    let anchor_hi = chunk_filling_box([4 * chunk_span, 0, 0]);
    // The resident pair: box A and box B abutting on +X across a chunk boundary. Box A's
    // +X-face blocks are occluded exactly while box B exists.
    let box_a = chunk_filling_box([0, 0, 0]);
    let box_b = chunk_filling_box([chunk_span, 0, 0]);
    let scene_with_b = Scene::from_nodes(vec![
        anchor_lo.clone(),
        anchor_hi.clone(),
        box_a.clone(),
        box_b.clone(),
    ]);
    let scene_without_b =
        Scene::from_nodes(vec![anchor_lo.clone(), anchor_hi.clone(), box_a.clone()]);

    let mut cache = TwoLayerResidentCache::enabled();
    cache.clear();
    let index_with_b = scene_with_b.build_leaf_spatial_index(density);
    let fresh_with_b = covering_owned(&mut cache, &scene_with_b, density);
    let build_with_b = build_brick_field(&fresh_with_b, density);
    let mut field = IncrementalBrickField::from_wholesale(build_with_b.clone()).0;

    // --- Step 1: CARVE (delete box B) — exposes box A's face blocks across the seam.
    let index_without_b = scene_without_b.build_leaf_spatial_index(density);
    let carve_aabb = index_without_b
        .edit_aabb_since(&index_with_b)
        .expect("a node delete is a localisable edit");
    let carve_dirty = cache.invalidate_aabb(&carve_aabb, density);
    let fresh_without_b = covering_owned(&mut cache, &scene_without_b, density);
    assert_eq!(
        fresh_with_b.len(),
        fresh_without_b.len(),
        "the anchors must pin the covering set (incremental precondition)"
    );
    field.apply_dirty_update(&fresh_without_b, &carve_dirty);
    let wholesale_without_b = build_brick_field(&fresh_without_b, density);
    assert_incremental_matches_wholesale(
        &field.to_build(),
        &wholesale_without_b,
        "carve-across-boundary",
    );

    // The scenario must be REAL: some chunk OUTSIDE the dirty set changed its record
    // set (box A's face blocks un-occluded) — else the ring re-derivation is untested.
    let dirty_set: std::collections::BTreeSet<[i32; 3]> =
        carve_dirty.iter().copied().collect();
    let records_by_chunk = |build: &BrickFieldBuild| {
        let mut by_chunk: std::collections::BTreeMap<[i32; 3], Vec<u64>> =
            std::collections::BTreeMap::new();
        for record in &build.brick_records {
            let chunk = {
                let block = unpack_world_block_key(record.packed_world_block_key);
                [
                    block[0].div_euclid(CHUNK_BLOCKS as i64) as i32,
                    block[1].div_euclid(CHUNK_BLOCKS as i64) as i32,
                    block[2].div_euclid(CHUNK_BLOCKS as i64) as i32,
                ]
            };
            by_chunk.entry(chunk).or_default().push(record.packed_world_block_key);
        }
        by_chunk
    };
    let before_by_chunk = records_by_chunk(&build_with_b);
    let after_by_chunk = records_by_chunk(&wholesale_without_b);
    let non_dirty_chunk_changed = before_by_chunk
        .iter()
        .any(|(chunk, keys)| {
            !dirty_set.contains(chunk) && after_by_chunk.get(chunk) != Some(keys)
        });
    assert!(
        non_dirty_chunk_changed,
        "fixture must flip records in a NON-dirty chunk (the occlusion ring); \
         dirty set: {dirty_set:?}"
    );

    // --- Step 2: FILL (re-add box B) — re-occludes box A's face blocks.
    let fill_aabb = index_with_b
        .edit_aabb_since(&index_without_b)
        .expect("a node re-add is a localisable edit");
    let fill_dirty = cache.invalidate_aabb(&fill_aabb, density);
    let fresh_refilled = covering_owned(&mut cache, &scene_with_b, density);
    field.apply_dirty_update(&fresh_refilled, &fill_dirty);
    let wholesale_refilled = build_brick_field(&fresh_refilled, density);
    assert_incremental_matches_wholesale(
        &field.to_build(),
        &wholesale_refilled,
        "fill-across-boundary",
    );
    // Fill restores the original record keys (slot numbers may differ — free-listed).
    assert_eq!(
        wholesale_refilled
            .brick_records
            .iter()
            .map(|r| r.packed_world_block_key)
            .collect::<Vec<_>>(),
        build_with_b
            .brick_records
            .iter()
            .map(|r| r.packed_world_block_key)
            .collect::<Vec<_>>(),
        "re-adding box B must restore the original surface record keys"
    );
}

/// Perf probe (issue #69, `#[ignore]`d — run in release): a ~1–2k-block scene, a
/// one-region recolour, incremental patch vs a full `build_brick_field`. The headless
/// stand-in for the Tracy live latency measurement; numbers go in the commit message.
/// Run: `cargo test --release incremental_vs_wholesale_perf_probe -- --ignored --nocapture`.
#[test]
#[ignore = "perf probe — run in release with --nocapture"]
fn incremental_vs_wholesale_perf_probe() {
    use std::time::Instant;
    let density = 8u32;
    let anchor_lo = tool(ShapeKind::Box, [-20, 0, 0], MaterialChoice::Stone, density);
    let anchor_hi = tool(ShapeKind::Box, [20, 0, 0], MaterialChoice::Stone, density);
    let scene_a = Scene::from_nodes(vec![
        anchor_lo.clone(),
        anchor_hi.clone(),
        tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Wood, density),
    ]);
    let scene_b = Scene::from_nodes(vec![
        anchor_lo,
        anchor_hi,
        tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Plain, density),
    ]);
    let mut cache = TwoLayerResidentCache::enabled();
    cache.clear();
    let index_a = scene_a.build_leaf_spatial_index(density);
    let fresh_a = covering_owned(&mut cache, &scene_a, density);
    let build_a = build_brick_field(&fresh_a, density);
    let mut field = IncrementalBrickField::from_wholesale(build_a.clone()).0;

    let index_b = scene_b.build_leaf_spatial_index(density);
    let edit_aabb = index_b.edit_aabb_since(&index_a).expect("localisable");
    let dirty = cache.invalidate_aabb(&edit_aabb, density);
    let fresh_b = covering_owned(&mut cache, &scene_b, density);

    let started = Instant::now();
    let update = field.apply_dirty_update(&fresh_b, &dirty);
    let _incremental_build = field.to_build();
    let incremental = started.elapsed();

    let started = Instant::now();
    let _ = build_brick_field(&fresh_b, density);
    let wholesale = started.elapsed();

    println!(
        "G3 perf probe: scene {} records, edit dirtied {} chunk(s) / {} slots — \
         incremental {:?} vs wholesale {:?} ({:.1}× )",
        build_a.brick_records.len(),
        dirty.len(),
        update.written_slots.len(),
        incremental,
        wholesale,
        wholesale.as_secs_f64() / incremental.as_secs_f64().max(1e-9),
    );
    assert!(update.written_slots.len() < build_a.sculpted_brick_count());
}
