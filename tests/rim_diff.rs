//! Grazing-rim brick-raymarch regression (bug fixed 2026-07-17). At a GRAZING ortho view
//! of the tube — the pose the head-on `gpu_parity` case never sampled — the brick raymarch
//! block-stepped the top rim into a +Z terraced staircase: the inner voxel-DDA seed floored
//! a max-face grazing entry one voxel PAST the block, so the per-block clip skipped the block
//! holding the true surface (see `VoxelDda::seed_in_box`). This three-way triage pins the fix:
//! GPU hit-identity == CPU exact evaluator (truth) == CPU brick-field march (the shader's f32
//! mirror) on EVERY pixel. The split (algo vs shader) is retained so a regression is triaged
//! at a glance: gpu==cpu_brick but != exact ⇒ raycast algorithm; gpu != cpu_brick ⇒ WGSL only.
//!
//! Run: cargo test --test rim_diff -- --nocapture (skips loudly without a GPU adapter)

mod common;

use document::voxel::{GeometryParams, SdfShape};
use voxel_core::voxel::ShapeKind;
use voxel_worker::{
    build_brick_field, cpu_march_brick_field, cpu_march_exact_occupancy, pack_gpu_records, AppCore,
    BrickRaymarchRenderer, ClipmapPyramid, LayerBand, MaterialChoice, OrbitCamera,
    ProjectionMode, Scene, TwoLayerStore, COLOR_TARGET_FORMAT,
};

fn exact_occupancy_set(
    two_layer_chunks: &[(
        [i32; 3],
        std::sync::Arc<evaluation::two_layer_store::TwoLayerChunk>,
    )],
    voxels_per_block: u32,
) -> std::collections::HashSet<[i64; 3]> {
    use voxel_core::core_geom::CHUNK_BLOCKS;
    let chunk_extent = (CHUNK_BLOCKS * voxels_per_block) as i64;
    let mut occupied = std::collections::HashSet::new();
    let mut expanded = Vec::new();
    for (chunk_coord, chunk) in two_layer_chunks {
        expanded.clear();
        chunk.expand_occupancy_into(&mut expanded, [0, 0, 0]);
        for voxel in &expanded {
            occupied.insert([
                chunk_coord[0] as i64 * chunk_extent + voxel.local_index[0] as i64,
                chunk_coord[1] as i64 * chunk_extent + voxel.local_index[1] as i64,
                chunk_coord[2] as i64 * chunk_extent + voxel.local_index[2] as i64,
            ]);
        }
    }
    occupied
}

#[test]
fn brick_raymarch_matches_exact_at_grazing_rim() {
    if skip_without_gpu("brick_raymarch_matches_exact_at_grazing_rim") {
        return;
    }
    let gpu = common::shared_gpu();
    let width = 900u32;
    let height = 600u32;
    let vpb = 16u32;

    // The repro tube: 50×10×50 blocks, wall 1 (size_voxels [800,160,800]).
    let shape = SdfShape::from_blocks(ShapeKind::Tube, [50, 10, 50], 1, vpb);
    let geometry = GeometryParams {
        shape: ShapeKind::Tube,
        size_voxels: shape.size_voxels,
        size_measurements: None,
        voxels_per_block: vpb,
        wall_blocks: 1,
    };
    let scene = Scene::from_geometry(geometry, MaterialChoice::Stone);

    let two_layer_chunks = TwoLayerStore::enabled().build_covering_chunks(&scene, vpb, 0);
    assert!(!two_layer_chunks.is_empty(), "empty two-layer build");
    let build = build_brick_field(&two_layer_chunks, vpb);
    assert!(!build.brick_records.is_empty(), "empty brick field");

    let recentre = scene.recentre_voxels_for_resolve(vpb);
    let grid_dimensions = scene.placed_region_dimensions(vpb);
    eprintln!("grid_dimensions = {grid_dimensions:?}, records = {}", build.brick_records.len());

    let occupied = exact_occupancy_set(&two_layer_chunks, vpb);
    let occupied_fn = |absolute: [i64; 3]| occupied.contains(&absolute);

    // GRAZING ortho view (the repro pose family): phi≈1.47 (near-horizontal), so the
    // upper wall + top rim are seen edge-on — where the artifact lives.
    let mut app_core = AppCore::new(OrbitCamera::default());
    app_core.camera.target = glam::Vec3::ZERO;
    app_core.camera.orbit_theta = 5.9963;
    app_core.camera.orbit_phi = 1.47;
    app_core.camera.projection_mode = ProjectionMode::Orthographic;
    app_core.camera.orbit_distance = OrbitCamera::auto_framed_distance(grid_dimensions);
    let aspect_ratio = width as f32 / height as f32;
    let view_projection = app_core.view_projection(aspect_ratio, grid_dimensions);
    let viewport_px = [0u32, 0, width, height];
    let band = LayerBand::FULL;

    let gpu_records = pack_gpu_records(&build.brick_records, |_| false);
    let pyramid = ClipmapPyramid::from_chunks(&two_layer_chunks);
    let mut renderer = BrickRaymarchRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
    renderer.install_brick_field(
        &gpu.device,
        &gpu.queue,
        &build.brick_records,
        &build.atlas_payload(),
        &gpu_records,
        &pyramid,
        recentre,
    );
    let frame = renderer.update_uniforms(
        &gpu.queue,
        view_projection,
        viewport_px,
        grid_dimensions,
        band,
        None,
        false,
        Some(MaterialChoice::default()),
    );
    let gpu_image = renderer.render_hit_identity_image(&gpu.device, &gpu.queue, width, height);

    let mut gpu_hits = 0usize;
    let mut gpu_vs_exact_disagree = 0usize;
    let mut algo_bug = 0usize; // gpu != exact && gpu == cpu_brick  → raycast algorithm wrong
    let mut shader_bug = 0usize; // gpu != cpu_brick               → WGSL-only divergence
    let mut brick_vs_exact_disagree = 0usize; // cpu_brick != exact (algorithm, GPU aside)
    let mut examples = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let pixel_index = (y * width + x) as usize;
            let gpu_pixel = gpu_image[pixel_index];
            let gpu_hit = gpu_pixel[0] == 1;
            let gpu_voxel = [gpu_pixel[1] as i32, gpu_pixel[2] as i32, gpu_pixel[3] as i32];
            if gpu_hit {
                gpu_hits += 1;
            }
            let pixel = glam::Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let exact = cpu_march_exact_occupancy(&frame, &occupied_fn, pixel);
            let brick = cpu_march_brick_field(&frame, &gpu_records, &build, &pyramid, pixel);

            let gpu_agrees_exact = match exact {
                Some(h) => gpu_hit && h.absolute_voxel == gpu_voxel,
                None => !gpu_hit,
            };
            let gpu_agrees_brick = match brick {
                Some(h) => gpu_hit && h.absolute_voxel == gpu_voxel,
                None => !gpu_hit,
            };
            let brick_agrees_exact = match (brick, exact) {
                (Some(a), Some(b)) => a.absolute_voxel == b.absolute_voxel,
                (None, None) => true,
                _ => false,
            };
            if !brick_agrees_exact {
                brick_vs_exact_disagree += 1;
            }
            if !gpu_agrees_exact {
                gpu_vs_exact_disagree += 1;
                if !gpu_agrees_brick {
                    shader_bug += 1;
                } else {
                    algo_bug += 1;
                }
                if examples.len() < 14 && gpu_hit {
                    examples.push(format!(
                        "  px=({x:>3},{y:>3}) gpu={gpu_voxel:?} exact={:?} cpu_brick={:?} \
                         faces[gpu?/ex/br]={:?}/{:?}",
                        exact.map(|h| h.absolute_voxel),
                        brick.map(|h| h.absolute_voxel),
                        exact.map(|h| h.face_normal),
                        brick.map(|h| h.face_normal),
                    ));
                }
            }
        }
    }

    eprintln!(
        "gpu_hits={gpu_hits}  gpu!=exact={gpu_vs_exact_disagree} (algo={algo_bug} shader={shader_bug})  cpu_brick!=exact={brick_vs_exact_disagree}"
    );

    assert!(gpu_hits > 0, "zero brick hits — the grazing camera missed the tube");
    // The regression: at this grazing view the brick raymarch — CPU mirror AND GPU shader —
    // must match the exact evaluator on EVERY pixel. Before the `seed_in_box` fix (2026-07-17)
    // this diverged on ~1800 rim pixels (the +Z tread staircase); the head-on parity case in
    // gpu_parity.rs never sampled a grazing rim, so it missed the bug entirely.
    let report = examples.join("\n");
    assert_eq!(
        brick_vs_exact_disagree, 0,
        "CPU brick march diverges from the exact evaluator on {brick_vs_exact_disagree} px (raycast algorithm bug)\n{report}"
    );
    assert_eq!(
        gpu_vs_exact_disagree, 0,
        "GPU brick raymarch diverges from the exact evaluator on {gpu_vs_exact_disagree} px (algo={algo_bug}, shader={shader_bug})\n{report}"
    );
}

/// Runtime GPU-availability probe — the replacement for the deleted `gpu` Cargo feature.
///
/// These tests used to be compiled out entirely behind `#![cfg(feature = "gpu")]`, which
/// meant a GPU-less machine did not skip them, it LOST them (and forgetting the flag made
/// the suite pass vacuously). Now they always compile and skip loudly here instead.
fn skip_without_gpu(test: &str) -> bool {
    static ADAPTER: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ADAPTER.get_or_init(voxel_worker::gpu::adapter_available) {
        return false;
    }
    eprintln!("skipping {test}: no GPU adapter on this machine");
    true
}
