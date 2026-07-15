//! ADR 0011 slice G3 — the incremental dirty-brick atlas update net. The load-bearing
//! assertion: an [`IncrementalBrickField`] patched edit-by-edit (only dirty chunks
//! re-evaluated, slots free-listed) is byte-exact vs a from-scratch [`build_brick_field`]
//! of the SAME scene, after EVERY step, across explicit block-kind transitions
//! (air↔sculpted↔coarse) and add / move / recolour / delete edits.
use crate::brick::*;
use voxel_core::core_geom::MaterialChoice;
use evaluation::cuboid::VoxelBox;
use document::scene::{Node, NodeContent, NodeTransform, Scene};
use evaluation::two_layer_store::{
    MicroblockGeometry, TwoLayerChunk, TwoLayerResidentCache, TwoLayerStore,
};
use voxel_core::voxel::{ShapeKind};
use document::voxel::{GeometryParams, SdfShape};

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

// ========================================================================
// The per-voxel cell-key side atlas (the CPU half): emission classifies a
// sculpted block uniform vs MIXED, and only a mixed block owns a cell-key tile.
//
// These fixtures drive the emission builder DIRECTLY with hand-built two-layer chunks —
// the tightest test of the CPU classifier. (The representability gate is now deleted, so a
// mixed scene DOES reach the brick path through the renderer; the rendering side is proven by
// the mixed-material golden + parity test. This module remains the CPU mirror's own contract.)
// ========================================================================

/// The density the hand-built fixtures use — small enough to state a block's cuboids
/// by hand, and deliberately NOT 16 (the brick edge follows the density).
const HAND_DENSITY: u32 = 4;

/// A hand-built covering set of ONE chunk at `[0, 0, 0]`: `coarse_blocks` are
/// `(block, block_id, overlay)`, `sculpted_blocks` are `(block, cuboids)` whose cuboid
/// labels are render-cell keys ([`CellKey::compose`]).
fn hand_built_chunk(
    coarse_blocks: &[([u32; 3], u16, bool)],
    sculpted_blocks: &[([u32; 3], Vec<VoxelBox>)],
) -> Vec<([i32; 3], Arc<TwoLayerChunk>)> {
    let block_count = (CHUNK_BLOCKS as usize).pow(3);
    let mut chunk = TwoLayerChunk {
        voxels_per_block: HAND_DENSITY,
        coarse: vec![None; block_count],
        coarse_overlay: vec![false; block_count],
        microblocks: std::collections::BTreeMap::new(),
    };
    for (block, block_id, overlay) in coarse_blocks {
        let flat = (block[2] as usize * CHUNK_BLOCKS as usize + block[1] as usize)
            * CHUNK_BLOCKS as usize
            + block[0] as usize;
        chunk.coarse[flat] = Some(BlockId(*block_id));
        chunk.coarse_overlay[flat] = *overlay;
    }
    for (block, cuboids) in sculpted_blocks {
        chunk.microblocks.insert(
            *block,
            MicroblockGeometry {
                cuboids: cuboids.clone(),
                seam_solidity: SeamSolidity::default(),
            },
        );
    }
    vec![([0, 0, 0], Arc::new(chunk))]
}

/// A block-local cuboid carrying the render-cell key `(block_id, overlay)`.
fn cell_box(min: [u32; 3], max: [u32; 3], block_id: u16, overlay: bool) -> VoxelBox {
    VoxelBox {
        min,
        max,
        label: CellKey::compose(block_id, overlay).raw(),
    }
}

/// The independent oracle for one block's per-voxel cell keys: paint each cuboid's key
/// into a dense `edge³` array in cuboid order (air stays [`AIR_CELL_KEY_DONT_CARE`]).
fn expected_cell_keys(cuboids: &[VoxelBox]) -> Vec<u16> {
    let edge = HAND_DENSITY as usize;
    let mut keys = vec![AIR_CELL_KEY_DONT_CARE; edge.pow(3)];
    for cuboid in cuboids {
        for z in cuboid.min[2]..=cuboid.max[2] {
            for y in cuboid.min[1]..=cuboid.max[1] {
                for x in cuboid.min[0]..=cuboid.max[0] {
                    keys[(z as usize * edge + y as usize) * edge + x as usize] = cuboid.label;
                }
            }
        }
    }
    keys
}

/// The independent oracle for one block's occupancy bytes (the same walk, occupancy only).
fn expected_occupancy_bytes(cuboids: &[VoxelBox]) -> Vec<u8> {
    let edge = HAND_DENSITY as usize;
    let mut bytes = vec![0u8; edge.pow(3)];
    for cuboid in cuboids {
        for z in cuboid.min[2]..=cuboid.max[2] {
            for y in cuboid.min[1]..=cuboid.max[1] {
                for x in cuboid.min[0]..=cuboid.max[0] {
                    bytes[(z as usize * edge + y as usize) * edge + x as usize] =
                        SCULPTED_BRICK_OCCUPIED;
                }
            }
        }
    }
    bytes
}

/// **Emission classifies uniform vs MIXED (the slice's core claim).** A block whose
/// microblock cuboids all share one cell key is UNIFORM: its material + overlay ride on
/// the record and it owns NO cell-key tile. A block whose cuboids disagree — on the
/// material OR on the overlay bit alone — is MIXED: it additionally carries a per-voxel
/// cell-key tile whose keys match its cuboids exactly, while its occupancy tile is
/// unchanged (byte-identical to the occupancy-only rasterization). A coarse block carries
/// its id + its chunk overlay marker and owns neither tile.
#[test]
fn emission_classifies_uniform_and_mixed_sculpted_blocks() {
    let uniform_block = [0u32, 0, 0];
    let uniform_cuboids = vec![
        cell_box([0, 0, 0], [1, 3, 3], 1, true),
        cell_box([2, 0, 0], [3, 1, 3], 1, true), // same cell key ⇒ still uniform
    ];
    let mixed_material_block = [1u32, 0, 0];
    let mixed_material_cuboids = vec![
        cell_box([0, 0, 0], [1, 3, 3], 1, false),
        cell_box([2, 0, 0], [3, 3, 3], 2, false), // different block id ⇒ MIXED
    ];
    let mixed_overlay_block = [2u32, 0, 0];
    let mixed_overlay_cuboids = vec![
        cell_box([0, 0, 0], [3, 3, 1], 1, false),
        cell_box([0, 0, 2], [3, 3, 3], 1, true), // same id, overlay differs ⇒ MIXED
    ];
    let coarse_block = [3u32, 0, 0];
    let chunks = hand_built_chunk(
        &[(coarse_block, 2, true)],
        &[
            (uniform_block, uniform_cuboids.clone()),
            (mixed_material_block, mixed_material_cuboids.clone()),
            (mixed_overlay_block, mixed_overlay_cuboids.clone()),
        ],
    );
    let build = build_brick_field(&chunks, HAND_DENSITY);

    // Exactly the two mixed blocks own a cell-key tile.
    assert_eq!(build.mixed_brick_count(), 2);
    assert_eq!(build.cell_key_tiles.len(), 2);
    assert_eq!(build.sculpted_brick_count(), 3, "all three boundary blocks are sculpted");

    // (a) The uniform block: one cell key on the record, NO cell-key tile.
    let record = build
        .find_record([uniform_block[0] as i64, uniform_block[1] as i64, uniform_block[2] as i64])
        .expect("uniform boundary block must have a record");
    assert_eq!(record.material_id, 1);
    assert!(record.overlay, "the uniform block's single cell key sets the overlay bit");
    assert_eq!(record.payload.cell_key_slot(), None, "a uniform brick owns no cell-key tile");
    assert!(matches!(record.payload, BrickPayload::Sculpted { .. }));
    assert_eq!(record.payload.kind_discriminant(), 1);

    // (b) + (c) The mixed blocks: a cell-key tile whose per-voxel keys are exactly the
    // cuboids', an occupancy tile unchanged by the classification.
    for (block, cuboids) in [
        (mixed_material_block, &mixed_material_cuboids),
        (mixed_overlay_block, &mixed_overlay_cuboids),
    ] {
        let record = build
            .find_record([block[0] as i64, block[1] as i64, block[2] as i64])
            .expect("mixed boundary block must have a record");
        let BrickPayload::SculptedMixed {
            atlas_slot,
            cell_key_slot,
        } = record.payload
        else {
            panic!("a block whose cuboids disagree on their cell key must emit MIXED");
        };
        assert_eq!(
            record.payload.kind_discriminant(),
            2,
            "a MIXED brick is its own GPU record kind (it traverses like a sculpted one, \
             but shades from its cell-key tile)"
        );
        assert_eq!(
            build.cell_key_tiles[cell_key_slot as usize].as_slice(),
            expected_cell_keys(cuboids).as_slice(),
            "the cell-key tile must carry each voxel's own cuboid key at {block:?}"
        );
        assert_eq!(
            build.sculpted_brick_occupancy(atlas_slot),
            expected_occupancy_bytes(cuboids),
            "the occupancy tile is unchanged by the material classification at {block:?}"
        );
    }

    // (d) The coarse block: id + the chunk's overlay marker, no slot of either pool.
    let record = build
        .find_record([coarse_block[0] as i64, coarse_block[1] as i64, coarse_block[2] as i64])
        .expect("coarse block must have a record");
    assert_eq!(record.material_id, 2);
    assert!(record.overlay, "a coarse block carries its chunk's per-block overlay marker");
    assert_eq!(record.payload.occupancy_atlas_slot(), None);
    assert_eq!(record.payload.cell_key_slot(), None);

    // The two mixed bricks' cell-key slots are DISTINCT and dense in the wholesale build.
    let mut cell_key_slots: Vec<u32> = build
        .brick_records
        .iter()
        .filter_map(|record| record.payload.cell_key_slot())
        .collect();
    cell_key_slots.sort_unstable();
    assert_eq!(cell_key_slots, vec![0, 1]);
}

/// Resolve every live record's cell-key tile through the mirror's own slot numbering and
/// compare it against a from-scratch wholesale build of the same chunks (whose numbering
/// is dense and unrelated) — the cell-key half of the incremental-vs-wholesale parity
/// oracle: same kind, same material/overlay, same occupancy bytes, same per-voxel keys.
fn assert_cell_key_parity(
    mirror: &IncrementalBrickField,
    chunks: &[([i32; 3], Arc<TwoLayerChunk>)],
    label: &str,
) {
    let wholesale = build_brick_field(chunks, HAND_DENSITY);
    let incremental = mirror.to_build();
    assert_eq!(
        incremental.brick_records.len(),
        wholesale.brick_records.len(),
        "[{label}] record count must match wholesale"
    );
    assert_eq!(
        mirror.mixed_brick_count(),
        wholesale.mixed_brick_count(),
        "[{label}] live mixed-brick count must match wholesale"
    );
    for (mirrored, whole) in incremental
        .brick_records
        .iter()
        .zip(wholesale.brick_records.iter())
    {
        let block = unpack_world_block_key(whole.packed_world_block_key);
        assert_eq!(
            mirrored.packed_world_block_key, whole.packed_world_block_key,
            "[{label}] record order must match wholesale"
        );
        assert_eq!(
            (mirrored.material_id, mirrored.overlay),
            (whole.material_id, whole.overlay),
            "[{label}] record cell key at {block:?}"
        );
        assert_eq!(
            mirrored.payload.cell_key_slot().is_some(),
            whole.payload.cell_key_slot().is_some(),
            "[{label}] uniform/mixed verdict at {block:?}"
        );
        match (
            mirrored.payload.occupancy_atlas_slot(),
            whole.payload.occupancy_atlas_slot(),
        ) {
            (Some(mirror_slot), Some(whole_slot)) => assert_eq!(
                incremental.sculpted_brick_occupancy(mirror_slot),
                wholesale.sculpted_brick_occupancy(whole_slot),
                "[{label}] occupancy bytes at {block:?} (slots renumber, bytes do not)"
            ),
            (None, None) => {}
            _ => panic!("[{label}] payload kind disagreement at {block:?}"),
        }
        if let (Some(mirror_slot), Some(whole_slot)) = (
            mirrored.payload.cell_key_slot(),
            whole.payload.cell_key_slot(),
        ) {
            assert_eq!(
                mirror.cell_key_tile(mirror_slot).as_slice(),
                wholesale.cell_key_tiles[whole_slot as usize].as_slice(),
                "[{label}] cell-key tile at {block:?} (slots renumber, keys do not)"
            );
        }
    }
}

/// **A block flipping uniform↔mixed under an incremental edit allocates/frees its
/// cell-key slot** — in the SEPARATE material pool, leaving the occupancy pool alone (the
/// block stays a sculpted brick either way, so its occupancy slot is merely rewritten).
/// After every step the mirror agrees with a from-scratch wholesale build, cell-key tiles
/// included.
#[test]
fn incremental_uniform_mixed_flip_churns_only_the_cell_key_pool() {
    let block_a = [0u32, 0, 0];
    let block_b = [1u32, 0, 0];
    let uniform_a = vec![cell_box([0, 0, 0], [3, 3, 3], 1, false)];
    let mixed_a = vec![
        cell_box([0, 0, 0], [3, 3, 1], 1, false),
        cell_box([0, 0, 2], [3, 3, 3], 2, false),
    ];
    let mixed_b = vec![
        cell_box([0, 0, 0], [1, 3, 3], 2, false),
        cell_box([2, 0, 0], [3, 3, 3], 2, true), // overlay-only mix
    ];
    let uniform_b = vec![cell_box([0, 0, 0], [3, 3, 3], 2, true)];

    // Step 0 (wholesale seed): A uniform, B mixed — one cell-key slot in use.
    let step_0 = hand_built_chunk(
        &[],
        &[(block_a, uniform_a.clone()), (block_b, mixed_b.clone())],
    );
    let build = build_brick_field(&step_0, HAND_DENSITY);
    let (mut mirror, _atlas) = IncrementalBrickField::from_wholesale(build);
    assert_eq!(mirror.mixed_brick_count(), 1);
    assert_eq!(mirror.cell_key_slot_high_water(), 1);
    let occupancy_high_water = mirror.slot_high_water();
    assert_eq!(occupancy_high_water, 2, "both blocks are sculpted bricks");
    assert_cell_key_parity(&mirror, &step_0, "step 0 (wholesale seed)");

    // Step 1 (the FLIP): A becomes mixed, B becomes uniform. B's cell-key slot is freed
    // and A's allocation reuses it — the material pool churns, its high-water mark does
    // not grow, and the occupancy pool's does not move at all.
    let step_1 = hand_built_chunk(
        &[],
        &[(block_a, mixed_a.clone()), (block_b, uniform_b.clone())],
    );
    let update = mirror.apply_dirty_update(&step_1, &[[0, 0, 0]]);
    assert!(!update.atlas_grew, "the occupancy atlas must not grow on a material flip");
    assert_eq!(mirror.slot_high_water(), occupancy_high_water);
    assert_eq!(mirror.mixed_brick_count(), 1, "exactly one block is mixed after the flip");
    assert_eq!(
        mirror.cell_key_slot_high_water(),
        1,
        "the freed cell-key slot must be reused, not appended to"
    );
    let record_a = mirror
        .records()
        .iter()
        .find(|record| unpack_world_block_key(record.packed_world_block_key) == [0, 0, 0])
        .expect("block A must still have a record");
    let cell_key_slot = record_a
        .payload
        .cell_key_slot()
        .expect("block A is MIXED after the flip");
    assert_eq!(
        mirror.cell_key_tile(cell_key_slot).as_slice(),
        expected_cell_keys(&mixed_a).as_slice()
    );
    let record_b = mirror
        .records()
        .iter()
        .find(|record| unpack_world_block_key(record.packed_world_block_key) == [1, 0, 0])
        .expect("block B must still have a record");
    assert_eq!(record_b.payload.cell_key_slot(), None, "block B is UNIFORM after the flip");
    assert_eq!((record_b.material_id, record_b.overlay), (2, true));
    assert_cell_key_parity(&mirror, &step_1, "step 1 (uniform↔mixed flip)");

    // Step 2 (GROW): both blocks mixed — the second mixed brick appends a new cell-key
    // slot (the pool grows independently of the occupancy pool, which stays put).
    let step_2 = hand_built_chunk(&[], &[(block_a, mixed_a), (block_b, mixed_b)]);
    mirror.apply_dirty_update(&step_2, &[[0, 0, 0]]);
    assert_eq!(mirror.mixed_brick_count(), 2);
    assert_eq!(mirror.cell_key_slot_high_water(), 2);
    assert_eq!(mirror.slot_high_water(), occupancy_high_water);
    assert_cell_key_parity(&mirror, &step_2, "step 2 (both mixed)");

    // Step 3 (FREE): both blocks uniform — every cell-key slot is freed (the mixed count
    // drops to zero; the high-water mark keeps the freed holes, as the occupancy pool does).
    let step_3 = hand_built_chunk(&[], &[(block_a, uniform_a), (block_b, uniform_b)]);
    mirror.apply_dirty_update(&step_3, &[[0, 0, 0]]);
    assert_eq!(mirror.mixed_brick_count(), 0, "no block is mixed any more");
    assert_eq!(
        mirror.cell_key_slot_high_water(),
        2,
        "freed slots keep their (dead) tiles until reallocated"
    );
    assert!(mirror
        .records()
        .iter()
        .all(|record| record.payload.cell_key_slot().is_none()));
    assert_cell_key_parity(&mirror, &step_3, "step 3 (both uniform again)");
}

/// A scene whose sculpted blocks are all UNIFORM (every scene the brick path renders
/// today) emits NO cell-key tile at all — the sparse-side-atlas contract, and the reason
/// the GPU bytes cannot move in this slice: `pack_gpu_records` reads the occupancy slot +
/// the record material, both untouched.
#[test]
fn a_uniform_scene_emits_no_cell_key_tiles() {
    let voxels_per_block = 4;
    let scene = Scene::from_geometry(
        GeometryParams {
            shape: ShapeKind::Sphere,
            size_voxels: [33, 33, 33],
            size_measurements: None,
            voxels_per_block,
            wall_blocks: 1,
        },
        MaterialChoice::Stone,
    );
    let chunks = TwoLayerStore::enabled().build_covering_chunks(&scene, voxels_per_block, 0);
    let build = build_brick_field(&chunks, voxels_per_block);
    assert!(build.sculpted_brick_count() > 0, "the fixture must have sculpted bricks");
    assert!(
        build.cell_key_tiles.is_empty(),
        "a single-material scene must pay no per-voxel material cost"
    );
    assert_eq!(build.mixed_brick_count(), 0);
    assert!(
        build
            .brick_records
            .iter()
            .all(|record| !matches!(record.payload, BrickPayload::SculptedMixed { .. })),
        "no record may be mixed in a single-material scene"
    );

    // …and therefore packs a ZERO-LENGTH side atlas: the second pool costs such a scene
    // nothing at all (not even a tile grid).
    let side_atlas = build.cell_key_atlas_payload();
    assert!(side_atlas.bytes.is_empty(), "no mixed brick ⇒ no side-atlas bytes");
    assert_eq!(side_atlas.cell_key_slot_count, 0);
    assert_eq!(side_atlas.geometry.bricks_per_axis, 0);
    assert_eq!(side_atlas.geometry.atlas_dim_voxels, 0);
}

/// Read one cell-key slot's `edge³` keys back out of a PACKED side atlas — an INDEPENDENT
/// re-derivation of the GPU's addressing (linear slot → 3D tile origin, x-fastest; texels
/// little-endian u16, two bytes each), so a bug in the packer cannot hide behind the
/// packer's own arithmetic.
fn packed_cell_keys_at_slot(bytes: &[u8], bricks_per_axis: u32, slot: u32) -> Vec<u16> {
    let edge = HAND_DENSITY as usize;
    let tiles = bricks_per_axis.max(1) as usize;
    let atlas_dim = tiles * edge;
    let slot = slot as usize;
    let origin = [
        (slot % tiles) * edge,
        ((slot / tiles) % tiles) * edge,
        (slot / (tiles * tiles)) * edge,
    ];
    let mut keys = Vec::with_capacity(edge.pow(3));
    for local_z in 0..edge {
        for local_y in 0..edge {
            for local_x in 0..edge {
                let texel = ((origin[2] + local_z) * atlas_dim + origin[1] + local_y)
                    * atlas_dim
                    + origin[0]
                    + local_x;
                keys.push(u16::from_le_bytes([bytes[texel * 2], bytes[texel * 2 + 1]]));
            }
        }
    }
    keys
}

/// **The R16 side atlas packs each mixed brick's cell-key tile at its own slot origin.**
/// The pool is sized from ITS OWN slot count (two mixed bricks ⇒ a 2-tile grid), holds two
/// little-endian bytes per voxel, and every live slot reads back — through an independent
/// addressing oracle — as exactly that block's per-voxel cuboid keys. Bricks that are
/// uniform or coarse consume no texel here, whatever their occupancy slot.
#[test]
fn mixed_bricks_pack_the_r16_side_atlas_at_their_own_slot_origins() {
    let uniform_block = [0u32, 0, 0];
    let uniform_cuboids = vec![cell_box([0, 0, 0], [3, 3, 3], 1, true)];
    let mixed_material_block = [1u32, 0, 0];
    let mixed_material_cuboids = vec![
        cell_box([0, 0, 0], [1, 3, 3], 1, false),
        cell_box([2, 0, 0], [3, 3, 3], 2, false),
    ];
    let mixed_overlay_block = [2u32, 0, 0];
    let mixed_overlay_cuboids = vec![
        cell_box([0, 0, 0], [3, 3, 1], 1, false),
        cell_box([0, 0, 2], [3, 3, 3], 1, true),
    ];
    let chunks = hand_built_chunk(
        &[([3, 0, 0], 2, true)],
        &[
            (uniform_block, uniform_cuboids),
            (mixed_material_block, mixed_material_cuboids.clone()),
            (mixed_overlay_block, mixed_overlay_cuboids.clone()),
        ],
    );
    let build = build_brick_field(&chunks, HAND_DENSITY);
    let side_atlas = build.cell_key_atlas_payload();

    // The pool's OWN geometry: two mixed bricks ⇒ ceil(cbrt 2) = 2 tiles/axis, 8 voxels/axis
    // — while the occupancy pool holds THREE sculpted bricks (its own, larger, tile grid).
    assert_eq!(side_atlas.cell_key_slot_count, 2);
    assert_eq!(side_atlas.geometry.bricks_per_axis, 2);
    assert_eq!(side_atlas.geometry.atlas_dim_voxels, 2 * HAND_DENSITY);
    assert_eq!(side_atlas.geometry.brick_edge_voxels, HAND_DENSITY);
    assert_eq!(
        side_atlas.bytes.len(),
        2 * (2 * HAND_DENSITY as usize).pow(3),
        "two bytes per texel — the R16Uint stride"
    );
    assert_eq!(build.sculpted_brick_count(), 3);

    // Every live slot reads back as that block's own per-voxel keys.
    for (block, cuboids) in [
        (mixed_material_block, &mixed_material_cuboids),
        (mixed_overlay_block, &mixed_overlay_cuboids),
    ] {
        let record = build
            .find_record([block[0] as i64, block[1] as i64, block[2] as i64])
            .expect("a mixed block must have a record");
        let slot = record
            .payload
            .cell_key_slot()
            .expect("a mixed block must own a cell-key slot");
        assert_eq!(
            packed_cell_keys_at_slot(
                &side_atlas.bytes,
                side_atlas.geometry.bricks_per_axis,
                slot
            ),
            expected_cell_keys(cuboids),
            "the packed side atlas must carry {block:?}'s keys at its slot origin"
        );
    }

    // The two mixed slots occupy DISJOINT texel spans (the slot → origin map is injective):
    // exactly the two tiles' worth of texels are non-zero-keyed, the rest of the cube is
    // untouched fill.
    let occupied_texels = side_atlas
        .bytes
        .chunks_exact(2)
        .filter(|texel| u16::from_le_bytes([texel[0], texel[1]]) != AIR_CELL_KEY_DONT_CARE)
        .count();
    let expected_keyed: usize = [&mixed_material_cuboids, &mixed_overlay_cuboids]
        .iter()
        .map(|cuboids| {
            expected_cell_keys(cuboids)
                .iter()
                .filter(|key| **key != AIR_CELL_KEY_DONT_CARE)
                .count()
        })
        .sum();
    assert_eq!(
        occupied_texels, expected_keyed,
        "no key may land outside its own slot's tile"
    );
}

/// **The incremental pool's GPU work-list.** A uniform↔mixed flip reports exactly the
/// cell-key slots the sink must free and rewrite (the second pool's own lists, independent
/// of the occupancy atlas's), and the bytes it packs are the bytes a from-scratch build
/// packs — tile-for-tile at each live record's slot (the pools renumber across the two
/// paths; the texels do not). The `to_build()` parity-oracle style, for the side atlas.
#[test]
fn incremental_cell_key_pool_reports_its_work_list_and_packs_like_wholesale() {
    let block_a = [0u32, 0, 0];
    let block_b = [1u32, 0, 0];
    let uniform_a = vec![cell_box([0, 0, 0], [3, 3, 3], 1, false)];
    let mixed_a = vec![
        cell_box([0, 0, 0], [3, 3, 1], 1, false),
        cell_box([0, 0, 2], [3, 3, 3], 2, false),
    ];
    let mixed_b = vec![
        cell_box([0, 0, 0], [1, 3, 3], 2, false),
        cell_box([2, 0, 0], [3, 3, 3], 2, true),
    ];
    let uniform_b = vec![cell_box([0, 0, 0], [3, 3, 3], 2, true)];

    // Seed: A uniform, B mixed — one cell-key slot, a 1-tile side atlas.
    let step_0 = hand_built_chunk(
        &[],
        &[(block_a, uniform_a.clone()), (block_b, mixed_b.clone())],
    );
    let (mut mirror, _atlas) =
        IncrementalBrickField::from_wholesale(build_brick_field(&step_0, HAND_DENSITY));
    assert_eq!(mirror.cell_key_atlas_geometry().bricks_per_axis, 1);

    // The FLIP: A becomes mixed, B becomes uniform. B's slot is freed and A's allocation
    // reuses it — so the sink frees slot 0 and rewrites slot 0, and neither tile grid grows.
    let step_1 = hand_built_chunk(&[], &[(block_a, mixed_a.clone()), (block_b, uniform_b)]);
    let update = mirror.apply_dirty_update(&step_1, &[[0, 0, 0]]);
    assert_eq!(update.freed_cell_key_slots, vec![0]);
    assert_eq!(update.written_cell_key_slots, vec![0]);
    assert!(!update.cell_key_atlas_grew, "a reused slot cannot grow the side atlas");
    assert!(!update.atlas_grew, "the occupancy pool is untouched by a material flip");

    // The dirty-slot bytes the sink uploads ARE the tile's little-endian texels.
    let mut expected_bytes = Vec::new();
    for key in expected_cell_keys(&mixed_a) {
        expected_bytes.extend_from_slice(&key.to_le_bytes());
    }
    assert_eq!(mirror.cell_key_slot_bytes(0), expected_bytes);

    // GROW: B becomes mixed too — the side atlas's OWN grow signal fires (2 slots ⇒ a
    // 2-tile grid), while the occupancy pool's does not.
    let step_2 = hand_built_chunk(&[], &[(block_a, mixed_a), (block_b, mixed_b)]);
    let update = mirror.apply_dirty_update(&step_2, &[[0, 0, 0]]);
    assert!(
        update.cell_key_atlas_grew,
        "the second mixed brick must grow the side atlas's tile grid"
    );
    assert!(!update.atlas_grew, "the occupancy pool's grid is unchanged");
    assert_eq!(update.written_cell_key_slots.len(), 2);
    assert_eq!(mirror.cell_key_atlas_geometry().bricks_per_axis, 2);

    // The wholesale-parity bar: the mirror's packed side atlas carries, at every live mixed
    // record's slot, exactly the tile a from-scratch build packs at ITS slot.
    let packed = mirror.pack_cell_key_atlas_payload();
    assert_eq!(
        packed, mirror.to_build().cell_key_atlas_payload(),
        "the two materialisations of one mirror must be byte-identical"
    );
    let wholesale = build_brick_field(&step_2, HAND_DENSITY);
    let wholesale_atlas = wholesale.cell_key_atlas_payload();
    assert_eq!(packed.geometry, wholesale_atlas.geometry);
    assert_eq!(packed.cell_key_slot_count, wholesale_atlas.cell_key_slot_count);
    for record in mirror.records() {
        let Some(mirror_slot) = record.payload.cell_key_slot() else {
            continue;
        };
        let block = unpack_world_block_key(record.packed_world_block_key);
        let whole_slot = wholesale
            .find_record(block)
            .and_then(|whole| whole.payload.cell_key_slot())
            .expect("the wholesale build must call the same block mixed");
        assert_eq!(
            packed_cell_keys_at_slot(
                &packed.bytes,
                packed.geometry.bricks_per_axis,
                mirror_slot
            ),
            packed_cell_keys_at_slot(
                &wholesale_atlas.bytes,
                wholesale_atlas.geometry.bricks_per_axis,
                whole_slot
            ),
            "packed cell keys at {block:?} (slots renumber, texels do not)"
        );
    }
}
