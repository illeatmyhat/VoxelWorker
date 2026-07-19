//! Shared fixtures for the async worker integration tests (`geometry_worker_async`,
//! `brick_worker_async`).
//!
//! Both targets drive a REAL [`Worker`] headlessly and need the same scaffolding: a
//! from-geometry box scene / its covering chunks, and a bounded poll loop that fails
//! LOUDLY on a hang (a worker that never answers is a bug, never an acceptable unbounded
//! wait). This module owns those once; each target declares `mod common;` and uses the
//! subset it needs.
//!
//! Each integration target is its own crate and links this module independently, so an
//! item a given target doesn't use would trip `dead_code` under `-D warnings`; the
//! module-wide allow below is the standard `tests/common` remedy.
#![allow(dead_code)]

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use voxel_worker::{
    GeometryParams, GpuContext, MaterialChoice, Scene, ShapeKind, TwoLayerChunk, TwoLayerStore,
    Worker,
};

/// The ONE wgpu context per test process — every GPU test target shares it.
///
/// **This exists to keep the process from creating and destroying wgpu instances
/// concurrently, which corrupts the heap under a software driver.** `cargo test` runs a
/// target's tests on parallel threads, and each test used to build its own
/// `GpuContext` — i.e. its own `wgpu::Instance`, adapter and device — so a target like
/// `gpu_parity` stood up and tore down fourteen of them at once. On Linux CI that aborted
/// the process with `double free or corruption (fasttop)` AFTER every test had reported
/// `ok`; reproduced locally in WSL as an intermittent SIGSEGV (2/40 runs pinned to two
/// cores, 0/30 with `--test-threads=1`).
///
/// The corruption is not ours — there is no `unsafe` anywhere in our GPU or worker code.
/// It is the Mesa software driver's global teardown racing itself, which a real driver on
/// Windows never exercised. Sharing one context removes the concurrency that provokes it,
/// and is what wgpu expects anyway: `Device` and `Queue` are `Send + Sync` handles built
/// to be used from many threads. Being a `static`, it is also never dropped, so the
/// teardown path simply does not run.
///
/// Returns a `&'static GpuContext`; clone the `device`/`queue` handles off it as before.
pub fn shared_gpu() -> &'static GpuContext {
    static SHARED: OnceLock<GpuContext> = OnceLock::new();
    SHARED.get_or_init(|| pollster::block_on(GpuContext::new(None)))
}

/// How long a NON-blocking `dispatch` is allowed to take before we call it "blocking". The
/// full build is many milliseconds; dispatch is a single channel `send`, so this is a wide
/// margin that still catches a regression that made dispatch wait on the build.
pub const DISPATCH_NONBLOCK_CEILING: Duration = Duration::from_millis(250);

/// A from-geometry solid box of `blocks³` blocks at `vpb` voxels/block — the canonical test
/// scene both worker suites resolve. `wall_blocks: 1` matches the live shell's default.
pub fn box_scene(blocks: u32, vpb: u32, material: MaterialChoice) -> Scene {
    Scene::from_geometry(
        GeometryParams {
            shape: ShapeKind::Box,
            size_voxels: [blocks * vpb; 3],
            size_measurements: None,
            voxels_per_block: vpb,
            wall_blocks: 1,
        },
        material,
    )
}

/// The two-layer covering chunks for a [`box_scene`] — the OWNED input a wholesale rebuild
/// dispatches, exactly as `WindowedState` builds it (`build_covering_chunks`). Sizing
/// `blocks` lands the covering set above or below the async threshold deterministically.
pub fn box_covering_chunks(
    blocks: u32,
    vpb: u32,
    material: MaterialChoice,
) -> Vec<([i32; 3], Arc<TwoLayerChunk>)> {
    TwoLayerStore::enabled().build_covering_chunks(&box_scene(blocks, vpb, material), vpb, 0)
}

/// Poll a `condition` closure with a BOUNDED wait, yielding between polls (mirrors the event
/// loop's per-frame try-recv). Returns once `condition` returns `true`, or fails LOUDLY once
/// `timeout` elapses — a hang is a bug, never an acceptable unbounded wait. The closure variant
/// of [`poll_until_result`], for callers whose "done" signal isn't a bare [`Worker`] result (e.g.
/// an orchestrator poll that installs into its OWN state and returns a `bool`). Same 1ms sleep
/// interval as [`poll_until_result`].
pub fn poll_until(mut condition: impl FnMut() -> bool, timeout: Duration, context: &str) {
    let deadline = Instant::now() + timeout;
    loop {
        if condition() {
            return;
        }
        if Instant::now() >= deadline {
            panic!(
                "{context}: the condition never held within {timeout:?} — the loop hung (a \
                 bug), never an acceptable unbounded wait"
            );
        }
        std::thread::yield_now();
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Poll a worker with a BOUNDED wait, yielding between polls (mirrors the event loop's
/// per-frame `try_recv_result`). Returns the first result, or fails loudly once `timeout`
/// elapses — a hang is a bug, so we never wait unbounded. Generic over the worker's
/// request/result so every async-worker suite shares one poll loop.
pub fn poll_until_result<Request, Response>(
    worker: &Worker<Request, Response>,
    timeout: Duration,
    context: &str,
) -> Response
where
    Request: Send + 'static,
    Response: Send + 'static,
{
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(result) = worker.try_recv_result() {
            return result;
        }
        if Instant::now() >= deadline {
            panic!(
                "{context}: worker produced no result within {timeout:?} — the loop hung (a \
                 bug), never an acceptable unbounded wait"
            );
        }
        std::thread::yield_now();
        std::thread::sleep(Duration::from_millis(1));
    }
}
