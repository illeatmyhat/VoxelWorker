//! Async wholesale brick-pipeline worker (perf follow-up to epic #64, on the issue #60
//! stale-while-rebuilding pattern).
//!
//! A WHOLESALE brick rebuild — [`build_brick_field`] (records + sculpted atlas bytes) +
//! the [`ClipmapPyramid`] + the representability classify + the GPU record pack — ran
//! synchronously inside `rebuild_geometry` and cost ~2s of main-thread hitch per
//! wholesale route on a giant scene (8000³ vx: record build ~1.15s, pyramid ~0.7s,
//! classify ~0.17s). This module moves the WHOLE CPU side onto a background worker: the
//! main thread keeps drawing the CURRENT display (the stale brick field, or the mesh)
//! until the freshly built artifacts arrive, then installs them — only the GPU upload
//! (`install_brick_field`, milliseconds) stays on the main thread. Incremental edits
//! keep their inline patch path ([`IncrementalBrickField::apply_dirty_update`]).
//!
//! ## What crosses the channel
//! A request carries the resolve's covering chunks `Arc`-shared out of the resident
//! cache (`Arc<TwoLayerChunk>` is `Send + Sync` — O(chunks) refcount bumps, no deep
//! copy) plus plain frame scalars. The worker calls the SAME pure builders the
//! synchronous path calls ([`build_brick_rebuild`] is that shared entry), so the
//! artifacts are byte-identical to an inline build. Everything is plain CPU data — no
//! GPU handles — so the worker exists on non-gpu builds too and maintains the CPU
//! mirror there ([`BrickRebuildOutcome::MirrorOnly`]).
//!
//! ## Supersede / generation (drain-to-latest)
//! The same contract as every display worker, on the shared [`crate::workers::Worker`]:
//! every request carries a monotonic generation, the worker drains its queue to the
//! latest, and the shell (via
//! [`GenerationTracker`](crate::display::routing::GenerationTracker)) discards any result
//! whose generation a later edit superseded.
//!
//! ## The interlock (the fog/mesh-era law: NEVER patch a stale artifact)
//! Where a brick edit is routed — and its DELIBERATE divergence from the geometry mesh
//! (a mid-flight SMALL wholesale rebuilds INLINE for bricks, sound only because the
//! shell's inline install seam `finish_brick_install` bumps the generation) — is decided
//! by [`route_brick_rebuild`](crate::display::routing::route_brick_rebuild). The interlock
//! law and that divergence note live next to the function, in the
//! [`crate::display::routing`] module doc.

use std::sync::Arc;

use crate::brick_field::{
    build_brick_field, BrickFieldBuild, ClipmapPyramid, IncrementalBrickField,
};
use crate::brick_raymarch::{brick_representable_overlay, pack_gpu_records, BrickGpuRecord};
use crate::two_layer_store::TwoLayerChunk;
use crate::workers::{build_catching, Worker};

/// A request to rebuild the brick pipeline WHOLESALE on the worker. Carries the
/// resolve's covering chunks (`Arc`-shared, `Send`) plus the frame scalars — the same
/// inputs the synchronous path fed `build_brick_field` / `ClipmapPyramid::from_chunks`.
pub struct BrickRebuildRequest {
    /// Monotonic generation stamp (supersede key). A result is accepted only when its
    /// generation is still the newest the shell dispatched.
    pub generation: u64,
    /// The two-layer covering chunks the resolve produced, `Arc`-shared out of the
    /// resident cache — O(chunks) refcount bumps to move, never a deep chunk copy.
    pub two_layer_chunks: Vec<([i32; 3], Arc<TwoLayerChunk>)>,
    /// The document density (voxels per block) the chunks were resolved at.
    pub density: u32,
    /// The composite recentre (floating origin, voxels; ADR 0008) the field lands in.
    /// Carried through to the result so the install uses the recentre THIS build was
    /// resolved at, never a re-derived one (the ADR 0008 frame law).
    pub recentre_voxels: [i64; 3],
    /// Whether to build the DISPLAY artifacts (classify + pyramid + GPU record pack) on
    /// top of the mirror. `true` on `--features gpu` (the raymarch consumes them);
    /// `false` on a non-gpu build, which maintains only the CPU mirror — matching the
    /// synchronous path, where the display block is compiled out.
    pub build_display_artifacts: bool,
}

/// What a finished wholesale brick rebuild produced — everything the shell's install
/// seam needs, built off-thread so the main thread only uploads.
pub enum BrickRebuildOutcome {
    /// The scene emptied (no brick records). The shell drops the mirror and clears any
    /// live display field (the mesh — trivially cheap for an empty scene — takes over).
    Empty,
    /// The request did not want display artifacts (a non-gpu build): only the CPU
    /// mirror was (re)built. No classify ran — this says nothing about representability.
    MirrorOnly {
        /// The fresh incremental mirror, seeded from the wholesale build
        /// ([`IncrementalBrickField::from_wholesale`]) so the next inline edit patches
        /// from a known-good full field.
        mirror: IncrementalBrickField,
    },
    /// The scene is not display-representable (a block mixes materials across its
    /// microblocks, or blocks disagree on the on-face grid — ADR 0011 G2). The mirror
    /// is still maintained; the display hands over to the cuboid mesh.
    NotRepresentable {
        /// The fresh incremental mirror (maintained regardless of representability).
        mirror: IncrementalBrickField,
    },
    /// The scene is representable: the full display install set, ready for the
    /// main-thread `install_brick_field` upload. Boxed so the enum stays small on the
    /// channel (the install set dwarfs the other variants).
    Display(Box<BrickDisplayInstall>),
}

/// The complete display install set a representable wholesale build produced — every
/// argument the main-thread `install_brick_field` upload needs, built off-thread.
pub struct BrickDisplayInstall {
    /// The wholesale field (records + sculpted atlas bytes) — `mirror.to_build()`
    /// equals this by construction (the G3 gate); both are shipped so the install
    /// does not re-derive either on the main thread.
    pub build: BrickFieldBuild,
    /// The packed GPU record set (all-resident, surface-only per ADR 0011).
    pub gpu_records: Vec<BrickGpuRecord>,
    /// The L1–L3 clip-map pyramid derived from the same chunks.
    pub pyramid: ClipmapPyramid,
    /// The scene-wide on-face-grid overlay state the shader binds.
    pub overlay_active: bool,
    /// The fresh incremental mirror, seeded from `build`.
    pub mirror: IncrementalBrickField,
}

/// A finished wholesale brick rebuild, tagged with the request generation so the shell
/// can discard a superseded one.
pub struct BrickRebuildResult {
    /// The generation of the [`BrickRebuildRequest`] this result was built for.
    pub generation: u64,
    /// The recentre the request carried — the frame the install lands the field in
    /// (ADR 0008: the value travels with the build, never re-derived at install time).
    pub recentre_voxels: [i64; 3],
    /// The built artifacts, or `None` if the build PANICKED on the worker (caught via
    /// [`build_catching`] — the worker stays alive, the shell keeps its stale field and
    /// leaves the outstanding flag set so the next edit re-dispatches).
    pub outcome: Option<BrickRebuildOutcome>,
}

/// Build a wholesale brick rebuild's artifacts — the SAME calls the synchronous path
/// makes, in the same order (record build → representability classify → pyramid +
/// record pack), so the outcome is byte-identical to an inline build (asserted by the
/// build-equivalence test). Factored out so the worker loop and the equivalence test
/// share one entry, like [`build_geometry`](crate::workers::geometry::build_geometry).
pub fn build_brick_rebuild(request: &BrickRebuildRequest) -> BrickRebuildOutcome {
    let build = build_brick_field(&request.two_layer_chunks, request.density);
    if build.brick_records.is_empty() {
        return BrickRebuildOutcome::Empty;
    }
    // Seed the fresh mirror from the wholesale build so the shell's next inline edit
    // patches from a known-good full field (`to_build()` == `build`, the G3 gate).
    let mirror = IncrementalBrickField::from_wholesale(&build);
    if !request.build_display_artifacts {
        return BrickRebuildOutcome::MirrorOnly { mirror };
    }
    match brick_representable_overlay(&request.two_layer_chunks) {
        Some(overlay_active) => {
            let pyramid = ClipmapPyramid::from_chunks(&request.two_layer_chunks);
            // Surface-only by construction (ADR 0011 interior elision fused into
            // emission) — a plain all-resident 1:1 pack.
            let gpu_records = pack_gpu_records(&build, |_| false);
            BrickRebuildOutcome::Display(Box::new(BrickDisplayInstall {
                build,
                gpu_records,
                pyramid,
                overlay_active,
                mirror,
            }))
        }
        None => BrickRebuildOutcome::NotRepresentable { mirror },
    }
}

/// The background brick-pipeline worker: a [`Worker`] whose pure-CPU build closure turns
/// each [`BrickRebuildRequest`] into a [`BrickRebuildResult`]. Spawn it via
/// [`spawn_brick_worker`]. The shell dispatches requests and polls each frame; the shared
/// drain-to-latest/supersede loop is [`Worker`]'s.
pub type BrickWorker = Worker<BrickRebuildRequest, BrickRebuildResult>;

/// Spawn the brick-pipeline worker on a dedicated thread. The closure builds via
/// [`build_brick_rebuild`] and carries the request's recentre through to the result (ADR
/// 0008: the frame value travels with the build, never re-derived at install). Like the
/// geometry worker, the build runs under [`build_catching`] so a build panic is caught and
/// surfaced as a `None` outcome the shell can react to, keeping the loop alive.
pub fn spawn_brick_worker() -> BrickWorker {
    Worker::spawn("voxel-worker brick rebuild", |request: BrickRebuildRequest| {
        let generation = request.generation;
        let recentre_voxels = request.recentre_voxels;
        let outcome = build_catching(generation, || build_brick_rebuild(&request));
        BrickRebuildResult {
            generation,
            recentre_voxels,
            outcome,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_geom::MaterialChoice;
    use crate::scene::Scene;
    use crate::two_layer_store::TwoLayerStore;
    use crate::voxel::{GeometryParams, ShapeKind};

    // --- build_brick_rebuild: byte-equivalence with the synchronous path ---

    /// The covering chunks for a small from-geometry box scene (the worker's input).
    fn box_chunks(blocks: u32, vpb: u32) -> Vec<([i32; 3], Arc<TwoLayerChunk>)> {
        let scene = Scene::from_geometry(
            GeometryParams {
                shape: ShapeKind::Box,
                size_voxels: [blocks * vpb; 3],
                size_measurements: None,
                voxels_per_block: vpb,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        );
        TwoLayerStore::enabled().build_covering_chunks(&scene, vpb, 0)
    }

    /// The worker's build entry produces artifacts byte-identical to the synchronous
    /// path's calls: same field, same pyramid, same GPU record pack, same overlay, and
    /// a mirror whose `to_build()` round-trips the field (the G3 gate).
    #[test]
    fn display_outcome_equals_synchronous_build() {
        let vpb = 4u32;
        let chunks = box_chunks(6, vpb);
        assert!(!chunks.is_empty(), "the box must cover chunks");
        let request = BrickRebuildRequest {
            generation: 1,
            two_layer_chunks: chunks.clone(),
            density: vpb,
            recentre_voxels: [0; 3],
            build_display_artifacts: true,
        };
        let BrickRebuildOutcome::Display(install) = build_brick_rebuild(&request) else {
            panic!("a single-material box is representable — expected Display");
        };
        let BrickDisplayInstall {
            build,
            gpu_records,
            pyramid,
            overlay_active,
            mirror,
        } = *install;
        let sync_build = build_brick_field(&chunks, vpb);
        assert_eq!(build, sync_build, "field build matches the synchronous call");
        assert_eq!(
            pyramid,
            ClipmapPyramid::from_chunks(&chunks),
            "pyramid matches the synchronous call"
        );
        assert_eq!(
            gpu_records,
            pack_gpu_records(&sync_build, |_| false),
            "GPU record pack matches the synchronous call"
        );
        assert_eq!(
            Some(overlay_active),
            brick_representable_overlay(&chunks),
            "overlay state matches the synchronous classify"
        );
        assert_eq!(
            mirror.to_build(),
            sync_build,
            "the shipped mirror round-trips the wholesale build (the G3 gate)"
        );
    }

    /// Without display artifacts (a non-gpu build) only the mirror is produced — no
    /// classify runs, matching the synchronous path where the display block is
    /// compiled out.
    #[test]
    fn mirror_only_when_display_artifacts_not_wanted() {
        let vpb = 4u32;
        let chunks = box_chunks(6, vpb);
        let request = BrickRebuildRequest {
            generation: 1,
            two_layer_chunks: chunks.clone(),
            density: vpb,
            recentre_voxels: [0; 3],
            build_display_artifacts: false,
        };
        let BrickRebuildOutcome::MirrorOnly { mirror } = build_brick_rebuild(&request) else {
            panic!("without display artifacts the outcome is MirrorOnly");
        };
        assert_eq!(mirror.to_build(), build_brick_field(&chunks, vpb));
    }

    /// An empty covering set yields `Empty` — the shell drops the mirror and clears
    /// the display field.
    #[test]
    fn empty_covering_set_yields_empty() {
        let request = BrickRebuildRequest {
            generation: 1,
            two_layer_chunks: Vec::new(),
            density: 4,
            recentre_voxels: [0; 3],
            build_display_artifacts: true,
        };
        assert!(matches!(
            build_brick_rebuild(&request),
            BrickRebuildOutcome::Empty
        ));
    }
}
