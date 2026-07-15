//! Headless coverage of the ASYNC `.vox` export worker (slow-paths item 2).
//!
//! The export build + file write used to run inline on the event-loop thread, freezing
//! the UI for the whole multi-second export. It now runs on the shared background
//! [`Worker`](voxel_worker::Worker) via [`spawn_vox_export_worker`]. Unlike the display
//! workers it carries NO supersede generation — a `.vox` is a user-chosen file, so the
//! shell serialises exports rather than draining to the latest.
//!
//! These tests drive the REAL worker (a spawned thread, real mpsc channels) headlessly —
//! no window, no GPU — and assert the guarantees a reviewer would otherwise check by hand:
//!
//! 1. **Parity**: the bytes the worker writes are IDENTICAL to an inline export of the
//!    same scene (same builder calls, no worker), and the summary counts match.
//! 2. **Progress**: the worker's per-chunk counter ends at exactly the covering-chunk
//!    total the shell uses as the progress denominator.
//! 3. **Failure + liveness**: an unwritable destination yields an `Err` outcome AND the
//!    worker still serves a subsequent good request (a failure never wedges the loop).
//!
//! Run: `cargo test --test vox_export_worker_async`

use std::path::PathBuf;
use std::time::Duration;

use voxel_worker::{
    spawn_vox_export_worker, MaterialChoice, Scene, TwoLayerStore, VoxExportBuilder,
    VoxExportRequest, VoxExportSummary,
};

mod common;

/// Bounded ceiling any poll-loop waits before failing LOUDLY — a hang is a bug, never an
/// acceptable unbounded wait. Generous so a slow machine doesn't flake.
const WORKER_TIMEOUT: Duration = Duration::from_secs(30);

/// The `.vox` palette the tests export with — a genuinely DISTINCT colour per procedural
/// material (the red channel is index-derived) so a multi-material export exercises the
/// per-`block_id` palette mapping rather than collapsing every slot to one colour. Any fixed
/// scheme works; parity only needs both paths to use the SAME palette.
fn test_palette() -> interchange::vox_export::BlockPaletteColors {
    let mut palette = [[0x40, 0x50, 0x60, 0xff]; MaterialChoice::MATERIAL_COUNT];
    for (index, slot) in palette.iter_mut().enumerate() {
        // Vary one channel per slot so no two materials share a colour.
        slot[0] = 0x40u8.wrapping_add((index as u8).wrapping_mul(0x30));
    }
    palette
}

/// The canonical small export fixture: a solid box, resolved the SAME way the shell
/// exports (through a recentred scene). Small so the covering set is a handful of chunks.
fn export_fixture() -> (Scene, u32) {
    let vpb = 4u32;
    (common::box_scene(2, vpb, MaterialChoice::default()), vpb)
}

/// Build the export INLINE (no worker) to `path`, returning the written summary. This is
/// the parity oracle — the exact builder calls the worker makes, run on this thread.
fn inline_export(scene: &Scene, density: u32, path: &std::path::Path) -> VoxExportSummary {
    let two_layer = TwoLayerStore::enabled();
    let region_dimensions = scene.placed_region_dimensions(density);
    let mut builder = VoxExportBuilder::new(region_dimensions, test_palette());
    voxel_worker::stream_vox_occupancy(&two_layer, scene, density, |chunk_voxels| {
        builder.ingest_chunk(&chunk_voxels);
    })
    .expect("the two-layer capability is enabled");
    let export = builder.finish();
    let bytes = export.write(path).expect("inline export writes to a temp file");
    VoxExportSummary {
        path: path.to_path_buf(),
        voxel_count: export.voxel_count(),
        model_count: export.model_count(),
        bytes,
    }
}

/// A unique temp path (process id + a per-call nonce) so parallel test runs don't collide.
fn unique_temp_path(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NONCE: AtomicU64 = AtomicU64::new(0);
    let nonce = NONCE.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("voxel_worker_export_{tag}_{pid}_{nonce}.vox"))
}

// ===========================================================================
// Test 1 — parity: worker bytes == inline bytes, summary counts match
// ===========================================================================

#[test]
fn worker_export_bytes_match_inline_export() {
    let (scene, density) = export_fixture();

    // Inline oracle.
    let inline_path = unique_temp_path("inline");
    let inline = inline_export(&scene, density, &inline_path);

    // Through the worker.
    let worker = spawn_vox_export_worker();
    let worker_path = unique_temp_path("worker");
    let progress = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    worker.dispatch(VoxExportRequest {
        scene: scene.clone(),
        density,
        palette_colors: test_palette(),
        path: worker_path.clone(),
        progress_chunks: std::sync::Arc::clone(&progress),
    });
    let result = common::poll_until_result(&worker, WORKER_TIMEOUT, "worker export parity");
    let summary = result
        .outcome
        .expect("the worker export of a valid scene succeeds");

    // Summary counts match the inline export.
    assert_eq!(summary.voxel_count, inline.voxel_count, "voxel counts match");
    assert_eq!(summary.model_count, inline.model_count, "model counts match");
    assert_eq!(summary.bytes, inline.bytes, "byte counts match");

    // The FILES are byte-identical.
    let inline_bytes = std::fs::read(&inline_path).expect("read inline .vox");
    let worker_bytes = std::fs::read(&worker_path).expect("read worker .vox");
    assert_eq!(
        inline_bytes, worker_bytes,
        "the worker's .vox bytes are identical to the inline export's"
    );

    let _ = std::fs::remove_file(&inline_path);
    let _ = std::fs::remove_file(&worker_path);
}

// ===========================================================================
// Test 2 — progress: the counter ends at the covering-chunk total
// ===========================================================================

#[test]
fn progress_counter_reaches_covering_chunk_total() {
    let (scene, density) = export_fixture();
    // The denominator the shell computes for the readout — every covering chunk yields
    // under the always-on two-layer capability, so the counter must land on exactly this.
    let total_chunks = scene.covering_chunk_count(density);
    assert!(
        total_chunks > 0,
        "the fixture must have covering chunks so the assertion is meaningful"
    );

    let worker = spawn_vox_export_worker();
    let path = unique_temp_path("progress");
    let progress = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    worker.dispatch(VoxExportRequest {
        scene: scene.clone(),
        density,
        palette_colors: test_palette(),
        path: path.clone(),
        progress_chunks: std::sync::Arc::clone(&progress),
    });
    let result = common::poll_until_result(&worker, WORKER_TIMEOUT, "worker export progress");
    result.outcome.expect("the export succeeds");

    assert_eq!(
        progress.load(std::sync::atomic::Ordering::Relaxed),
        total_chunks,
        "the per-chunk progress counter reaches exactly the covering-chunk total"
    );

    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Test 3 — failure path + liveness after a failed export
// ===========================================================================

#[test]
fn unwritable_path_fails_and_worker_survives_for_next_request() {
    let (scene, density) = export_fixture();
    let worker = spawn_vox_export_worker();

    // An unwritable destination: put a regular FILE where a parent directory would need to
    // be, so `write`'s `create_dir_all(parent)` fails (a file is not a directory).
    let blocker = unique_temp_path("blocker");
    std::fs::write(&blocker, b"not a directory").expect("create the blocking file");
    let bad_path = blocker.join("inner.vox");

    let progress = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    worker.dispatch(VoxExportRequest {
        scene: scene.clone(),
        density,
        palette_colors: test_palette(),
        path: bad_path,
        progress_chunks: std::sync::Arc::clone(&progress),
    });
    let result = common::poll_until_result(&worker, WORKER_TIMEOUT, "worker export failure");
    assert!(
        result.outcome.is_err(),
        "an export to an unwritable path yields an Err outcome (not a panic / wedge)"
    );

    // Liveness: a subsequent GOOD request still succeeds — the failure did not wedge the loop.
    let good_path = unique_temp_path("after_failure");
    let progress2 = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    worker.dispatch(VoxExportRequest {
        scene,
        density,
        palette_colors: test_palette(),
        path: good_path.clone(),
        progress_chunks: std::sync::Arc::clone(&progress2),
    });
    let result = common::poll_until_result(&worker, WORKER_TIMEOUT, "worker export post-failure");
    assert!(
        result.outcome.is_ok(),
        "the worker serves a good request after a failed one (it did not wedge)"
    );

    let _ = std::fs::remove_file(&blocker);
    let _ = std::fs::remove_file(&good_path);
}
