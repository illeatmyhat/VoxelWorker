//! Async MagicaVoxel `.vox` export worker.
//!
//! Writing a `.vox` re-streams the scene's exact occupancy region-scoped (one covering
//! chunk at a time — a coarse-solid block is a fast `d³` fill, a boundary block is
//! per-voxel) and serialises the result to a user-chosen file. At the user's current
//! scene scale (an 8000³ region, ~1.95M covering chunks) that build + write is a
//! multi-second job; running it inline on the event-loop thread (the button handler used
//! to) froze the UI for the whole export. This module moves it onto the shared background
//! [`Worker`](crate::workers::Worker): the shell dispatches an owned [`Scene`] clone plus
//! the already-chosen path, keeps drawing, and reads a per-chunk progress counter until
//! the finished [`VoxExportResult`] lands. See the display chapter
//! (`docs/architecture/03-display.md`) for the worker plumbing and the two-layer chapter
//! (`docs/architecture/04-storage.md`) for the streaming export source.
//!
//! ## No supersede generation — the shell serialises instead (a deliberate divergence)
//!
//! Every other background worker (geometry, diameter, brick) carries a monotonic
//! generation and the loop **drains to the latest**, dropping superseded requests — the
//! right policy for a display rebuild, where only the newest matters. An export is
//! different: it is a **user-chosen file**. Drain-to-latest would silently drop a real
//! export the moment a second one was queued, losing a file the user asked for. So this
//! worker carries NO generation; instead the shell **serialises** — it disables the
//! export button while a request is outstanding, so a second export can never be queued
//! and drain-to-latest never bites. The [`build_catching`](crate::workers::build_catching)
//! generation argument is therefore a fixed `0` (there is no generation to report); it
//! still serves its real purpose here — mapping a build panic to a failure result the
//! shell can show, rather than wedging the worker thread.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::scene::Scene;
use crate::two_layer_store::{stream_vox_occupancy, TwoLayerStore};
use crate::vox_export::{BlockPaletteColors, VoxExportBuilder};
use crate::workers::{build_catching, Worker};

/// A request to build + write one `.vox` file. Carries an OWNED scene clone plus all the
/// plain data the build needs — the save dialog (a native modal, the fast part) already
/// ran on the main thread and produced `path`.
pub struct VoxExportRequest {
    /// The scene to export, cloned out of the document so the worker owns it.
    pub scene: Scene,
    /// The document density (voxels per block) the export resolves at.
    pub density: u32,
    /// The per-`block_id` `.vox` palette (ADR 0003 §3a), computed on the main thread from
    /// the active material's representative colour, exactly as the inline path did.
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

/// The three numbers the old inline `println!` reported, plus the path, for the shell's
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

/// The background `.vox` export worker: a [`Worker`] whose build closure streams the
/// scene into a [`VoxExportBuilder`] and writes the file. Spawn it via
/// [`spawn_vox_export_worker`]. Unlike the display workers it carries no supersede
/// generation — the shell serialises exports (see the module doc).
pub type VoxExportWorker = Worker<VoxExportRequest, VoxExportResult>;

/// Spawn the `.vox` export worker on a dedicated thread. The closure mirrors the body of
/// the old synchronous `export_vox` AFTER the save dialog: build the always-on
/// [`TwoLayerStore`], pre-create the [`VoxExportBuilder`] model set from the region
/// dimensions, [`stream_vox_occupancy`] each covering chunk into it (bumping
/// `progress_chunks` per chunk), finish, and write. The whole build runs under
/// [`build_catching`](crate::workers::build_catching) so a panic becomes a failure result
/// (not a wedged thread); an IO error maps to its `to_string()`.
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
        // generation (the shell serialises — see the module doc). The catch still earns
        // its keep — a panic anywhere below becomes an Err the shell shows, not a dead
        // thread that would wedge `export_outstanding` forever.
        //
        // The ENTIRE job — stream + build + write — runs inside the ONE catch, so even a
        // serialisation/IO panic in `write` (not just an `io::Error`) still delivers a
        // `VoxExportResult` and re-enables the Export button. `build_catching` maps the
        // panic case to `None`, which becomes the "panicked" Err below.
        let built: Option<Result<VoxExportSummary, String>> = build_catching(0, move || {
            let two_layer = TwoLayerStore::enabled();
            let region_dimensions = scene.placed_region_dimensions(density);
            let mut builder = VoxExportBuilder::new(region_dimensions, palette_colors);
            stream_vox_occupancy(&two_layer, &scene, density, |chunk_voxels| {
                builder.ingest_chunk(&chunk_voxels);
                progress_chunks.fetch_add(1, Ordering::Relaxed);
            })
            .expect("the two-layer capability is enabled");
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
