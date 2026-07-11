//! Async diameter / widest-run measurement worker (ADR 0010 E5 follow-up).
//!
//! The layer-scrubber readout ("Ø N vx") is the widest occupied run in the current layer
//! band, computed by [`streamed_widest_run_in_band`](crate::two_layer_store::streamed_widest_run_in_band).
//! Even after the block-row dedup that query is O(total blocks) — sub-second on a huge
//! solid but NOT free (~0.5s at 8000³), and it fires on every band scrub and every
//! grid-changing edit. Running it inline on the event-loop thread froze the UI for the
//! whole measurement. This module moves it onto a dedicated background thread: the shell
//! keeps showing the PREVIOUS (stale) diameter until the fresh value lands, so the UI
//! never blocks at any scene scale.
//!
//! ## What crosses the channel
//! A request carries an OWNED [`Scene`] clone (`Send`) plus the density + band scalars.
//! The worker builds its own [`TwoLayerStore::enabled()`](crate::two_layer_store::TwoLayerStore)
//! (cheap, stateless) and streams the widest run — the SAME call the synchronous path made,
//! so the value is identical.
//!
//! ## Supersede / generation (drain-to-latest)
//! Every request carries a monotonic `generation`. A burst of edits/scrubs collapses to one
//! measurement of the NEWEST request (the worker drains its queue to the latest), and the
//! shell discards any result whose generation is not the newest it dispatched — mirrors the
//! [`GeometryWorker`](crate::geometry_worker::GeometryWorker) supersede contract, reusing
//! [`GenerationTracker`](crate::geometry_worker::GenerationTracker) on the shell side.

use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;

use crate::scene::Scene;
use crate::two_layer_store::{streamed_widest_run_in_band, TwoLayerStore};

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

/// The background diameter worker: owns a build loop on a dedicated thread. The shell sends
/// [`DiameterRequest`]s and polls [`try_recv_result`](Self::try_recv_result) each frame.
pub struct DiameterWorker {
    request_sender: Sender<DiameterRequest>,
    result_receiver: Receiver<DiameterResult>,
    /// Kept so the worker thread's lifetime is tied to the handle; the channel close on drop
    /// signals the loop to exit (its `recv` errors and it returns).
    _worker: JoinHandle<()>,
}

impl DiameterWorker {
    /// Spawn the worker on a dedicated thread. The loop drains its request channel to the
    /// latest (a burst of scrubs collapses to one measurement of the newest request), streams
    /// the widest run, and sends the result back. It exits when the request channel closes
    /// (the shell, hence the `DiameterWorker`, dropped).
    pub fn spawn() -> Self {
        let (request_sender, request_receiver) = std::sync::mpsc::channel::<DiameterRequest>();
        let (result_sender, result_receiver) = std::sync::mpsc::channel::<DiameterResult>();
        let worker = std::thread::Builder::new()
            .name("voxel-worker diameter measure".to_string())
            .spawn(move || run_diameter_worker(&request_receiver, &result_sender))
            .expect("failed to spawn diameter worker");
        Self {
            request_sender,
            result_receiver,
            _worker: worker,
        }
    }

    /// Dispatch a measurement request (non-blocking). The worker drains to the latest, so
    /// sending a newer request while one is in flight supersedes it. A send error (worker
    /// gone) is ignored — the shell keeps its stale diameter, never blocks.
    pub fn dispatch(&self, request: DiameterRequest) {
        let _ = self.request_sender.send(request);
    }

    /// Poll the result channel WITHOUT blocking: return the latest finished result if one has
    /// arrived, else `None`. Drains to the newest available (an in-flight supersede can leave
    /// more than one queued) so the shell never integrates a stale measurement when a newer
    /// one is ready.
    pub fn try_recv_result(&self) -> Option<DiameterResult> {
        let mut latest = None;
        while let Ok(result) = self.result_receiver.try_recv() {
            latest = Some(result);
        }
        latest
    }
}

/// The worker loop: block on the request channel, drain to the newest pending request,
/// measure, send the result. Exits when the channel closes.
fn run_diameter_worker(
    request_receiver: &Receiver<DiameterRequest>,
    result_sender: &Sender<DiameterResult>,
) {
    while let Ok(first) = request_receiver.recv() {
        // Drain-to-latest: collapse a burst of queued requests to the NEWEST so we never
        // measure a superseded generation.
        let mut request = first;
        while let Ok(newer) = request_receiver.try_recv() {
            request = newer;
        }
        // The SAME call the synchronous readout made; `unwrap_or(0)` covers the Part-only /
        // empty scene (no covering chunk range). The two-layer capability is always ON.
        let diameter = streamed_widest_run_in_band(
            &TwoLayerStore::enabled(),
            &request.scene,
            request.density,
            request.band.0,
            request.band.1,
        )
        .unwrap_or(0);
        if result_sender
            .send(DiameterResult {
                generation: request.generation,
                diameter,
            })
            .is_err()
        {
            // The shell is gone; stop.
            return;
        }
    }
}
