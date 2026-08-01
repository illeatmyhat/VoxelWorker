//! Async `MagicaVoxel` `.vox` export worker.
//!
//! Writing a `.vox` re-streams the scene's exact occupancy region-scoped (one covering
//! chunk at a time — a coarse-solid block is a fast `d³` fill, a boundary block is
//! per-voxel) and serializes the result to a user-chosen file. At the user's current
//! scene scale (an 8000³ region, ~1.95M covering chunks) that build + write is a
//! multi-second job; running it inline on the event-loop thread (the button handler used
//! to) froze the UI for the whole export. This module moves it onto the shared background
//! [`Worker`]: the shell dispatches an owned [`Scene`] clone plus
//! the already-chosen path, keeps drawing, and reads a per-chunk progress counter until
//! the finished [`VoxExportResult`] lands. See the work chapter
//! (`docs/architecture/04-work.md`) for the worker plumbing and the evaluation chapter
//! (`docs/architecture/02-evaluation.md`) for the two-layer streaming export source.
//!
//! ## No supersede generation — the shell serializes instead (a deliberate divergence)
//!
//! Every other background worker (geometry, diameter, brick) carries a monotonic
//! generation and the loop **drains to the latest**, dropping superseded requests — the
//! right policy for a display rebuild, where only the newest matters. An export is
//! different: it is a **user-chosen file**. Drain-to-latest would silently drop a real
//! export the moment a second one was queued, losing a file the user asked for. So this
//! worker carries NO generation; instead the shell **serializes** — it disables the
//! export button while a request is outstanding, so a second export can never be queued
//! and drain-to-latest never bites. The [`build_catching`]
//! generation argument is therefore a fixed `0` (there is no generation to report); it
//! still serves its real purpose here — mapping a build panic to a failure result the
//! shell can show, rather than wedging the worker thread.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::workers::{build_catching, Worker};
use document::scene::Scene;
use evaluation::two_layer_store::{stream_vox_occupancy, TwoLayerStore};
use interchange::vox_export::{BlockPaletteColors, VoxExportBuilder};

/// A request to build and write one `.vox` file.
///
/// It owns the scene clone and plain build data; the main thread has already produced `path`
/// through the save dialog.
pub struct VoxExportRequest {
    /// The scene to export, cloned out of the document so the worker owns it.
    pub scene: Scene,
    /// The document density (voxels per block) the export resolves at.
    pub density: u32,
    /// The per-`block_id` `.vox` palette, computed on the main thread from
    /// the active material's representative color, exactly as the inline path did.
    pub palette_colors: BlockPaletteColors,
    /// The user-chosen destination file (from the rfd save dialog, which stays on the main
    /// thread — a native modal, not the slow part).
    pub path: PathBuf,
    /// Per-chunk progress counter: the worker increments it once per ingested covering
    /// chunk. The shell holds a clone and reads it each frame for the "Exporting… N/M
    /// chunks" readout. Its final value equals the covering-chunk total (every covering
    /// chunk yields under the always-on two-layer capability).
    pub progress_chunks: Arc<AtomicU64>,
}

/// The three numbers the export reports, plus the path, for the shell's
/// completion readout.
pub struct VoxExportSummary {
    /// The file that was written.
    pub path: PathBuf,
    /// Total occupied voxels written across all models.
    pub voxel_count: usize,
    /// Models written (1 unless the 256-limit forced a tiled split).
    pub model_count: usize,
    /// Bytes written to disk.
    pub bytes: usize,
}

/// A finished export: the summary on success, or a human-readable error string (a build
/// panic or an IO failure) the shell surfaces as status text.
pub struct VoxExportResult {
    pub outcome: Result<VoxExportSummary, String>,
}

/// The background `.vox` export worker.
///
/// Its [`Worker`] closure streams the scene into [`VoxExportBuilder`] and writes the file. The
/// shell serializes exports, so this worker does not carry a supersede generation.
pub type VoxExportWorker = Worker<VoxExportRequest, VoxExportResult>;

/// Spawn the `.vox` export worker on a dedicated thread.
///
/// The closure builds the two-layer stream, updates `progress_chunks`, finishes the
/// [`VoxExportBuilder`], and writes the result under [`build_catching`].
pub fn spawn_vox_export_worker() -> VoxExportWorker {
    Worker::spawn("voxel-worker vox export", |request: VoxExportRequest| {
        let VoxExportRequest {
            scene,
            density,
            palette_colors,
            path,
            progress_chunks,
        } = request;
        // `build_catching`'s generation is a fixed 0: this worker has no supersede
        // generation (the shell serializes — see the module doc). The catch still earns
        // its keep — a panic anywhere below becomes an Err the shell shows, not a dead
        // thread that would wedge `export_outstanding` forever.
        //
        // The ENTIRE job — stream + build + write — runs inside the ONE catch, so even a
        // serialization/IO panic in `write` (not just an `io::Error`) still delivers a
        // `VoxExportResult` and re-enables the Export button. `build_catching` maps the
        // panic case to `None`, which becomes the "panicked" Err below.
        let built: Option<Result<VoxExportSummary, String>> = build_catching(0, move || {
            let two_layer = TwoLayerStore::enabled();
            let region_dimensions = scene.placed_region_dimensions(density);
            let mut builder = VoxExportBuilder::new(region_dimensions, palette_colors);
            if stream_vox_occupancy(&two_layer, &scene, density, |chunk_voxels| {
                builder.ingest_chunk(&chunk_voxels);
                progress_chunks.fetch_add(1, Ordering::Relaxed);
            })
            .is_none()
            {
                return Err("the two-layer capability is disabled".to_string());
            }
            let export = builder.finish();
            match export.write(&path) {
                Ok(bytes) => Ok(VoxExportSummary {
                    path,
                    voxel_count: export.voxel_count(),
                    model_count: export.model_count(),
                    bytes,
                }),
                Err(error) => Err(error.to_string()),
            }
        });
        // A build panic (`None`) still ships a result — never a silently wedged export.
        let outcome = built.unwrap_or_else(|| Err("export panicked — see stderr".to_string()));
        VoxExportResult { outcome }
    })
}
