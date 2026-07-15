//! The shell's async-worker poll seams: the diameter/widest-run measurement, the background
//! VS block scan drain + bounded thumbnail build, and the `.vox` export completion poll. Split
//! out of `windowed/mod.rs` (ADR 0016).

use super::*;

impl WindowedState {
    /// ADR 0010 E5 follow-up: poll the diameter worker for a finished widest-run measurement
    /// and, if it is still the newest dispatched (not superseded by a later scrub/edit), swap
    /// it into `measured_diameter` + request a redraw so the readout updates this frame.
    /// Non-blocking — the app never waits on the worker; the previous (stale) value shows
    /// meanwhile. A superseded result is discarded via the [`GenerationTracker`].
    pub(super) fn poll_diameter_worker(&mut self) {
        let Some(result) = self.diameter_worker.try_recv_result() else {
            return;
        };
        if !self.diameter_generation.accepts(result.generation) {
            // A later scrub/edit superseded this measurement — discard it (the newer one is
            // in flight; the stale readout keeps showing until it lands).
            return;
        }
        self.measured_diameter = result.diameter;
        self.window.request_redraw();
    }

    /// Drain the background scan channel into a pending queue, then build a
    /// BOUNDED number of thumbnails per frame so a few-hundred-block scan never
    /// stalls a frame. All GPU work (thumbnail render, egui registration) happens
    /// here on the main thread; with the cap it is amortised across frames.
    pub(super) fn poll_scan(&mut self) {
        // Cap the thumbnail GPU work per frame. The PNG decode already happens on
        // the scan worker; this only bounds the main-thread render+register so a
        // burst of groups arriving at once can't hitch the frame.
        const THUMBNAILS_PER_FRAME: usize = 8;

        // Move everything the worker has produced so far into the pending queue.
        if let Some(handle) = self.scan_handle.as_ref() {
            for message in handle.drain() {
                match message {
                    ScanMessage::Group { group, thumbnail_rgba } => {
                        self.pending_groups.push_back((group, thumbnail_rgba));
                    }
                    ScanMessage::Done { group_count, source_name } => {
                        self.scan_total = Some(group_count);
                        self.scan_source_name = source_name;
                        self.scan_handle = None;
                    }
                }
            }
        }

        // Build at most a few thumbnails this frame; the rest wait for later
        // frames (we keep redrawing each frame via `about_to_wait`).
        for _ in 0..THUMBNAILS_PER_FRAME {
            let Some((group, thumbnail_rgba)) = self.pending_groups.pop_front() else {
                break;
            };
            self.palette.add_group(
                &self.gpu.device,
                &self.gpu.queue,
                &mut self.egui_bridge.renderer,
                group,
                &thumbnail_rgba,
            );
        }

        // Status line: still working while groups are arriving or queued; settle
        // to the final count once the worker is done AND the queue is drained.
        if self.scan_handle.is_none() && self.pending_groups.is_empty() {
            if let Some(total) = self.scan_total.take() {
                self.palette.ui.status = match self.scan_source_name.take() {
                    Some(name) => format!("{total} blocks loaded — {name}"),
                    None => "No VS install found — use Connect folder".to_string(),
                };
            }
        } else {
            self.palette.ui.status = format!("{} blocks loaded…", self.palette.ui.tiles.len());
        }
    }

    /// Poll the `.vox` export worker for a finished write (slow-paths item 2). On a
    /// result, clear the in-flight flag + progress pair and set `export_status` to the
    /// summary or the error, then request a redraw so the panel readout updates. While an
    /// export is still in flight, request a redraw anyway so the "Exporting… done/total"
    /// count keeps advancing even if the app would otherwise idle. Non-blocking — the UI
    /// never waits on the worker.
    pub(super) fn poll_vox_export_worker(&mut self) {
        if let Some(result) = self.vox_export_worker.try_recv_result() {
            self.export_outstanding = false;
            self.export_progress = None;
            match result.outcome {
                Ok(summary) => {
                    self.export_status = Some(format!(
                        "wrote {} ({} voxels, {} model(s), {} bytes)",
                        summary.path.display(),
                        summary.voxel_count,
                        summary.model_count,
                        summary.bytes
                    ));
                }
                Err(error) => {
                    // Finding #1: if the user asked to close while this export was in
                    // flight, a FAILURE must CANCEL the deferred close. The guard's whole
                    // promise is "you won't silently lose the file"; exiting on a failed
                    // write breaks exactly that. Clear the deferral so the exit check below
                    // (top of `RedrawRequested`) doesn't fire, and tell the user the close
                    // was cancelled so they can react. A success still exits as before.
                    if self.close_requested_while_exporting {
                        self.close_requested_while_exporting = false;
                        self.export_status = Some(format!(
                            "export .vox FAILED: {error} — close cancelled so you can see \
                             this; close again to exit"
                        ));
                    } else {
                        self.export_status = Some(format!("export .vox failed: {error}"));
                    }
                }
            }
            self.window.request_redraw();
        } else if self.export_outstanding {
            // Keep frames coming so the progress readout refreshes while we wait.
            self.window.request_redraw();
        }
    }
}
