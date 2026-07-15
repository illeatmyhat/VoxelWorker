//! Async wholesale geometry rebuild worker (issue #60, ADR 0003 §7).
//!
//! The live app's WHOLESALE geometry rebuild — two-layer classify + the per-chunk
//! cuboid mesh's CPU build + GPU buffer upload — is ~3s for a large object on an
//! initial-create / resize / density / recentre edit. Doing it inline blocks the main
//! thread and freezes the UI. This module moves that build onto a background worker so
//! the UI never stalls: the main thread keeps rendering the CURRENT mesh
//! (stale-while-rebuilding) until the worker's freshly-built [`CuboidMeshRenderer`]
//! arrives, then swaps it in.
//!
//! ## What crosses the channel (why this is sound in wgpu 29)
//! wgpu 29's `Device`/`Queue` — and every GPU handle a [`CuboidMeshRenderer`] holds
//! (`RenderPipeline`, `Buffer`, `BindGroup`, `Sampler`, `BindGroupLayout`) — are
//! `Send + Sync + Clone` (Arc-backed). So the worker **clones `device`/`queue` and
//! builds the WHOLE renderer off-thread**, GPU buffers included, and the finished
//! renderer crosses the channel intact. Only surface acquire/present stays on the main
//! thread (never touched here). The mesh build calls the SAME
//! [`CuboidMeshRenderer::new_from_two_layer_chunks`] the synchronous path calls, so the
//! output is byte-identical (the build-equivalence net — see the tests).
//!
//! ## Division of labour (what the main thread still does synchronously)
//! The two-layer resolve/classify (`AppCore::rebuild`) runs on the main thread — it
//! mutates the resident cache (`&mut AppCore`, the sole document-adjacent writer) and
//! is comparatively cheap; it produces the OWNED `two_layer_chunks` (`Send`). Only the
//! **mesh CPU build + GPU upload** (the heavy `CuboidMeshRenderer` construction) is
//! dispatched here. Fog stays demand-driven on the main thread (#56–#59), unchanged.
//!
//! ## Supersede / generation (drain-to-latest)
//! Every request carries a monotonic [`generation`](GeometryRebuildRequest::generation).
//! The shared drain-to-latest/supersede plumbing lives in [`crate::workers::Worker`]: the
//! worker builds only the newest pending request. The shell **discards any received result
//! whose generation is stale** (an older generation than the newest request it has
//! dispatched); that accept/discard decision is factored into
//! [`GenerationTracker`](crate::engagement::routing::GenerationTracker) so it is unit-testable
//! without a live window.

use std::sync::Arc;

use voxel_core::voxel::RecentreVoxels;
use display::cuboid_mesh::CuboidMeshRenderer;
use display::renderer::LayerBand;
use evaluation::two_layer_store::TwoLayerChunk;
use crate::workers::{build_catching, Worker};

/// A request to build a wholesale cuboid mesh on the worker (issue #60). Carries the
/// OWNED two-layer chunks the resolve produced plus the frame parameters
/// [`CuboidMeshRenderer::new_from_two_layer_chunks`] needs — all `Send` plain data.
pub struct GeometryRebuildRequest {
    /// Monotonic generation stamp (supersede key). A result is accepted only when its
    /// generation matches the newest request the shell has dispatched (see
    /// [`GenerationTracker`](crate::engagement::routing::GenerationTracker)).
    pub generation: u64,
    /// The two-layer covering chunks the resolve produced, `Arc`-shared out of the resident
    /// cache (`Arc<TwoLayerChunk>` is `Send + Sync`, so the request stays `Send` — the move
    /// into the worker is O(chunks) refcount bumps, not a deep chunk copy). Meshed via coarse
    /// one-box + microblock cuboids + seam-flag culling — the sole runtime mesh path.
    pub two_layer_chunks: Vec<([i32; 3], Arc<TwoLayerChunk>)>,
    /// The whole composite grid's voxel dims (the band-clip layer mapping).
    pub grid_dimensions: [u32; 3],
    /// The composite recentre (floating origin, voxels; ADR 0008) the mesh lands in.
    /// Carried as [`RecentreVoxels`] (the frame law): the `CuboidMeshRenderer` builder now
    /// takes the newtype, so the worker hands the frame value straight through — the unwrap
    /// happens only at the mesher's positional rebase arithmetic.
    pub recentre_voxels: RecentreVoxels,
    /// The document density (voxels per block) the chunks were resolved at.
    pub density: u32,
    /// The CURRENT layer-clip band at dispatch (issue #60 M2). The worker builds the
    /// renderer already clipped to THIS band, so the swap frame does NOT trigger a full
    /// synchronous `rebuild_for_band` re-mesh on the main thread (the multi-second hitch
    /// #60 removed). During onion-skin scrubbing a clipped band is common, so a swapped-in
    /// FULL-band renderer would otherwise re-mesh every chunk the instant it arrived. If the
    /// band moved between dispatch and swap the per-frame `rebuild_for_band` still corrects
    /// it — this only optimises the common stable-band case.
    pub band: LayerBand,
}

/// A finished wholesale mesh built by the worker (issue #60): the whole
/// [`CuboidMeshRenderer`] (GPU buffers included) tagged with the request generation it
/// was built for, so the shell can discard a stale result and swap in a fresh one.
pub struct GeometryRebuildResult {
    /// The generation of the [`GeometryRebuildRequest`] this result was built for.
    pub generation: u64,
    /// The freshly built renderer, or `None` if the build PANICKED on the worker (issue
    /// #60 M1: GPU OOM, an internal assert, a bad dimension). A panicked build is caught
    /// (the worker stays alive) and surfaced as a `None` result + a stderr log rather than
    /// silently wedging the worker forever. The shell keeps its current (stale) renderer on
    /// a `None` and does NOT clear the outstanding flag, so the next edit re-dispatches.
    pub renderer: Option<CuboidMeshRenderer>,
}

/// The background geometry worker (issue #60): a [`Worker`] whose build closure owns the
/// cloned `device`/`queue` and turns each [`GeometryRebuildRequest`] into a
/// [`GeometryRebuildResult`]. Spawn it via [`spawn_geometry_worker`]. The shell dispatches
/// requests and polls each frame; the shared drain-to-latest/supersede loop is
/// [`Worker`]'s.
pub type GeometryWorker = Worker<GeometryRebuildRequest, GeometryRebuildResult>;

/// Spawn the geometry worker with cloned GPU handles (issue #60). `device`/`queue` are
/// cloned (wgpu 29 Arc-backed) so the worker can create the mesh's GPU buffers off the main
/// thread; `color_format` is the render target format the pipelines are built for. The
/// closure captures all three and builds via the SAME
/// [`CuboidMeshRenderer::new_from_two_layer_chunks`] the sync path uses, so the output is
/// byte-identical.
///
/// A build panic (GPU OOM, an internal assert, a bad dimension) must NOT wedge the worker,
/// so the build runs under [`build_catching`]: a panic is caught, logged, and surfaced as a
/// `None`-renderer result the shell can react to, and the worker loop stays alive.
pub fn spawn_geometry_worker(
    device: wgpu::Device,
    queue: wgpu::Queue,
    color_format: wgpu::TextureFormat,
) -> GeometryWorker {
    Worker::spawn("voxel-worker geometry rebuild", move |request: GeometryRebuildRequest| {
        let generation = request.generation;
        let renderer =
            build_catching(generation, || build_geometry(&device, &queue, color_format, &request));
        GeometryRebuildResult {
            generation,
            renderer,
        }
    })
}

/// Build the wholesale cuboid mesh for a request (issue #60) — the SAME call the
/// synchronous path makes, so the built renderer is byte-identical (the build-equivalence
/// net asserts this). Factored out so the worker loop and the build-equivalence test share
/// one build entry.
pub fn build_geometry(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    color_format: wgpu::TextureFormat,
    request: &GeometryRebuildRequest,
) -> CuboidMeshRenderer {
    // Issue #60 M2: build already clipped to the request's band so the swap frame does not
    // re-mesh on the main thread. `LayerBand::FULL` (the common no-onion case) is identical
    // to the plain `new_from_two_layer_chunks` output, so goldens/parity stay pixel-exact.
    CuboidMeshRenderer::new_from_two_layer_chunks_banded(
        device,
        queue,
        color_format,
        &request.two_layer_chunks,
        request.grid_dimensions,
        request.recentre_voxels,
        request.density,
        request.band,
    )
}
