//! VoxelWorker — the windowed application (default binary).
//!
//! A thin entry point: all the windowed logic (`WindowedState`, `App`, the winit
//! `ApplicationHandler`, the per-frame render, and the async-worker poll seams) lives in the
//! shell LIB module tree [`voxel_worker::windowed`] (ADR 0016 — the bin depends on the lib, and
//! the lib already carries the winit/egui/wgpu deps, so the app logic is dep-clean there). This
//! binary just hands control to [`voxel_worker::windowed::run`].

fn main() {
    voxel_worker::windowed::run();
}
