// ADR 0016 Phase 3: this brick-pipeline perf probe was relocated here from the evaluation
// crate's `two_layer_store::tests`. It exercises the FULL brick pipeline (a DISPLAY-layer
// concern: `build_brick_field` / `pack_gpu_records` / `ClipmapPyramid` /
// `IncrementalBrickField`) fed from two-layer chunks, so it straddles the evaluation↔display
// boundary and cannot live in the evaluation crate (whose law forbids naming any display
// type). It reaches the two-layer classify via the public `evaluation::two_layer_store` path.
use document::scene::{Node, NodeContent, Scene};
use evaluation::two_layer_store::TwoLayerStore;
use voxel_core::core_geom::MaterialChoice;

/// Perf probe (`#[ignore]`d — run in release): per-stage timing of the FULL brick
/// pipeline a wholesale rebuild runs after the two-layer classify, on solid
/// sketch-extrude cubes of growing block span. This is the interior-elision regression
/// guard for the 8000³-cube freeze fix (ADR 0011 surface-only record contract): at
/// density 16 the 500-blk/axis cube is 125M blocks, and before the surface-only build
/// every stage was O(all blocks) — ~12.5s of serial main-thread work and ~6 GB of
/// transient record traffic per rebuild. With the record set ∝ surface, every stage
/// must stay sub-second and the record count ~1.5M (the shell), not 125M.
///
/// `cargo test --release brick_pipeline_scaling_probe -- --ignored --nocapture`
/// The 500-blk/axis case (the actual 8000³ user scene) is opt-in via
/// `VOXELWORKER_PROBE_LARGE=1` — it is the slowest case and the smaller spans already
/// expose any O(volume) regression as a super-quadratic jump between rows.
#[test]
#[ignore = "perf probe — run in release with --nocapture"]
fn brick_pipeline_scaling_probe() {
    use crate::brick::{
        build_brick_field, BrickRecord, ClipmapPyramid, IncrementalBrickField,
    };
    use crate::brick::pack_gpu_records;
    use document::sketch::{PlaneAxis, Sketch, SketchSolid};
    let density = 16u32;
    let mut block_spans = vec![50i64, 125, 250];
    if std::env::var_os("VOXELWORKER_PROBE_LARGE").is_some() {
        block_spans.push(500); // the 8000³-voxel user cube
    }
    for blocks in block_spans {
        let edge = blocks * density as i64;
        let extrude =
            SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, edge, edge), edge as u32);
        let scene = Scene::from_nodes(vec![Node::new(
            "Box",
            NodeContent::SketchTool { producer: extrude, material: MaterialChoice::Stone },
        )]);
        let stage_start = std::time::Instant::now();
        let chunks = TwoLayerStore::enabled().build_covering_chunks(&scene, density, 0);
        let classify_elapsed = stage_start.elapsed();
        let stage_start = std::time::Instant::now();
        let build = build_brick_field(&chunks, density);
        let field_elapsed = stage_start.elapsed();
        let record_count = build.brick_records.len();
        let record_megabytes =
            (record_count * std::mem::size_of::<BrickRecord>()) as f64 / 1.0e6;
        let stage_start = std::time::Instant::now();
        let gpu_records = pack_gpu_records(&build.brick_records, |_| false);
        let pack_elapsed = stage_start.elapsed();
        let stage_start = std::time::Instant::now();
        let pyramid = ClipmapPyramid::from_chunks(&chunks);
        let pyramid_elapsed = stage_start.elapsed();
        let stage_start = std::time::Instant::now();
        // Single-owner rework (item 9): from_wholesale now MOVES the records + atlas bytes
        // (no records clone), seeding only the bit tiles.
        let (incremental_mirror, _atlas) = IncrementalBrickField::from_wholesale(build);
        let wholesale_elapsed = stage_start.elapsed();
        println!(
            "brick pipeline probe {edge}^3 vx ({blocks} blk/axis): classify {} chunks \
             {classify_elapsed:?} | brick_field {} surface records ({record_megabytes:.0} MB) \
             {field_elapsed:?} | gpu_pack {} records {pack_elapsed:?} | \
             pyramid(from_chunks) {pyramid_elapsed:?} | \
             from_wholesale (records move + tile seed) {wholesale_elapsed:?}",
            chunks.len(),
            record_count,
            gpu_records.len(),
        );
        drop(incremental_mirror);
        drop(pyramid);
    }
}
