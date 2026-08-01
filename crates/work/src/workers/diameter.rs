//! Async diameter / widest-run measurement worker.
//!
//! The layer-scrubber readout ("Ø N vx") is the widest occupied run in the current layer
//! band, computed by [`streamed_widest_run_in_band`].
//! Even after the block-row dedup that query is O(total blocks) — sub-second on a huge
//! solid but NOT free (~0.5s at 8000³), and it fires on every band scrub and every
//! grid-changing edit. Running it inline on the event-loop thread froze the UI for the
//! whole measurement. This module moves it onto a dedicated background thread: the shell
//! keeps showing the PREVIOUS (stale) diameter until the fresh value lands, so the UI
//! never blocks at any scene scale.
//!
//! ## What crosses the channel
//! A request carries an OWNED [`Scene`] clone (`Send`) plus the density + band scalars.
//! The worker builds its own [`TwoLayerStore::enabled()`](evaluation::two_layer_store::TwoLayerStore)
//! (cheap, stateless) and streams the widest run — the SAME call the synchronous path made,
//! so the value is identical.
//!
//! ## Supersede / generation (drain-to-latest)
//! Every request carries a monotonic `generation`. A burst of edits/scrubs collapses to one
//! measurement of the NEWEST request via the shared [`crate::workers::Worker`]
//! drain-to-latest loop, and the shell discards any result whose generation is not the
//! newest it dispatched — reusing
//! [`GenerationTracker`](crate::engagement::routing::GenerationTracker) on the shell side, like
//! every other display worker.

use crate::workers::Worker;
use document::scene::Scene;
use evaluation::two_layer_store::{streamed_widest_run_in_band, TwoLayerStore};

/// A request to measure the widest occupied run in a layer band (the diameter readout).
/// Carries an OWNED scene clone + the frame scalars — all `Send` plain data.
pub struct DiameterRequest {
    /// Monotonic generation stamp (supersede key). A result is accepted only when its
    /// generation is still the newest the shell dispatched.
    pub generation: u64,
    /// The scene to measure, cloned out of the document so the worker owns it.
    pub scene: Scene,
    /// The document density (voxels per block) the measurement resolves at.
    pub density: u32,
    /// The inclusive layer band `[lower, upper]` (Z-slices, Z-up) to measure the run over.
    pub band: (u32, u32),
}

/// A finished diameter measurement, tagged with the request generation it was built for so
/// the shell can discard a stale one.
pub struct DiameterResult {
    /// The generation of the [`DiameterRequest`] this result answers.
    pub generation: u64,
    /// The widest occupied run in the requested band (voxels).
    pub diameter: u32,
}

/// The background diameter worker.
///
/// Its [`Worker`] closure streams the widest run; the shell dispatches requests and polls the
/// shared drain-to-latest loop each frame.
pub type DiameterWorker = Worker<DiameterRequest, DiameterResult>;

/// Spawn the diameter worker on a dedicated thread.
///
/// It uses [`streamed_widest_run_in_band`] with the same two-layer capability as the synchronous
/// readout. Empty or VoxelBody-only scenes produce zero; unlike geometry and brick builds, this
/// measurement does not use `build_catching`.
pub fn spawn_diameter_worker() -> DiameterWorker {
    Worker::spawn(
        "voxel-worker diameter measure",
        |request: DiameterRequest| {
            let diameter = streamed_widest_run_in_band(
                &TwoLayerStore::enabled(),
                &request.scene,
                request.density,
                request.band.0,
                request.band.1,
            )
            .unwrap_or(0);
            DiameterResult {
                generation: request.generation,
                diameter,
            }
        },
    )
}
