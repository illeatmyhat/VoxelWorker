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
//! The same contract as every display worker, on the shared [`crate::worker::Worker`]:
//! every request carries a monotonic generation, the worker drains its queue to the
//! latest, and the shell (via [`GenerationTracker`](crate::geometry_worker::GenerationTracker))
//! discards any result whose generation a later edit superseded.
//!
//! ## The interlock (the fog/mesh-era law: NEVER patch a stale artifact)
//! While an async wholesale brick build is OUTSTANDING the resident
//! [`IncrementalBrickField`] mirror (and the renderer's live field) reflect S0 while
//! the worker builds S1 — so an incremental edit must NOT patch them (that would strand
//! every brick that differs S0→S1 but isn't in the new dirty set, the Frankenstein
//! field). [`route_brick_rebuild`] therefore routes EVERY mid-flight edit WHOLESALE.
//! One DELIBERATE divergence from
//! [`route_geometry_rebuild`](crate::geometry_worker::route_geometry_rebuild) (which
//! sends every mid-flight edit async): a mid-flight wholesale whose covering set is
//! SMALL rebuilds INLINE — immediately current, no worker latency. That is sound ONLY
//! because the shell's inline install seam (`finish_brick_install` in `main.rs`) bumps
//! the generation, so the superseded in-flight result is discarded on arrival; do not
//! remove that bump. The decision is pure so the interlock is unit-testable.

use std::sync::Arc;

use crate::brick_field::{
    build_brick_field, BrickFieldBuild, ClipmapPyramid, IncrementalBrickField,
};
use crate::brick_raymarch::{brick_representable_overlay, pack_gpu_records, BrickGpuRecord};
use crate::two_layer_store::TwoLayerChunk;
use crate::worker::{build_catching, Worker};

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

/// Where a brick rebuild is routed. The brick analogue of
/// [`RebuildRoute`](crate::geometry_worker::RebuildRoute), with the patch precondition
/// (a resident mirror) folded in so the shell reads ONE decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrickRebuildAction {
    /// Patch the resident mirror in place via `apply_dirty_update` (the G3 fast path).
    /// Sound ONLY when nothing is outstanding AND a mirror is resident — the mirror
    /// then reflects the latest resolve.
    PatchInline,
    /// Rebuild the field WHOLESALE inline on the main thread (a small covering set —
    /// cheap enough not to hitch a frame, and it avoids the worker's swap latency).
    WholesaleInline,
    /// Dispatch a WHOLESALE rebuild of the CURRENT full covering set to the async
    /// worker (stale-while-rebuilding). Chosen for a large covering set AND — the
    /// interlock — for ANY edit while an async brick build is outstanding.
    WholesaleAsync,
}

/// Decide where an edit's brick rebuild is routed. Pure — no GPU, no window — so the
/// outstanding-interlock is unit-testable, like `route_geometry_rebuild`.
///
/// The load-bearing rule: while an async brick build is outstanding the resident mirror
/// (and the renderer's live field) do NOT reflect the latest resolve, so an incremental
/// edit must NOT patch them — route EVERY mid-flight edit to a fresh WHOLESALE rebuild
/// instead. An incremental edit with NO resident mirror (the field emptied earlier, or
/// startup dispatched async) also has nothing sound to patch, so it goes wholesale. The
/// covering-set size then gates inline vs async, matching the mesh threshold gate —
/// INCLUDING while outstanding: a small mid-flight wholesale rebuilds inline (it is
/// immediately current; the shell's inline install seam bumps the generation so the
/// superseded in-flight result is discarded — see the module doc's divergence note).
pub fn route_brick_rebuild(
    async_outstanding: bool,
    incremental_edit: bool,
    mirror_resident: bool,
    covering_chunk_count: usize,
    async_threshold: usize,
) -> BrickRebuildAction {
    if !async_outstanding && mirror_resident && incremental_edit {
        return BrickRebuildAction::PatchInline;
    }
    if covering_chunk_count > async_threshold {
        BrickRebuildAction::WholesaleAsync
    } else {
        BrickRebuildAction::WholesaleInline
    }
}

/// Build a wholesale brick rebuild's artifacts — the SAME calls the synchronous path
/// makes, in the same order (record build → representability classify → pyramid +
/// record pack), so the outcome is byte-identical to an inline build (asserted by the
/// build-equivalence test). Factored out so the worker loop and the equivalence test
/// share one entry, like [`build_geometry`](crate::geometry_worker::build_geometry).
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
    use crate::geometry_worker::ASYNC_REBUILD_CHUNK_THRESHOLD;
    use crate::scene::Scene;
    use crate::two_layer_store::TwoLayerStore;
    use crate::voxel::{GeometryParams, ShapeKind};

    const THRESHOLD: usize = ASYNC_REBUILD_CHUNK_THRESHOLD;

    // --- route_brick_rebuild: the outstanding-interlock decision table ---

    /// The interlock — while an async brick build is outstanding, EVERY edit (even an
    /// incremental one with a resident mirror) rebuilds wholesale: patching the
    /// S0 mirror while the worker builds S1 would strand every S0→S1 brick outside the
    /// new dirty set (the Frankenstein field). A large covering set re-dispatches async;
    /// a SMALL one rebuilds wholesale INLINE (immediately current — the shell's inline
    /// install seam bumps the generation so the in-flight result is discarded).
    #[test]
    fn outstanding_forces_wholesale_never_patch() {
        for &incremental_edit in &[false, true] {
            for &mirror_resident in &[false, true] {
                let large =
                    route_brick_rebuild(true, incremental_edit, mirror_resident, THRESHOLD + 1, THRESHOLD);
                assert_eq!(
                    large,
                    BrickRebuildAction::WholesaleAsync,
                    "outstanding + a large covering set re-dispatches async"
                );
                assert_ne!(large, BrickRebuildAction::PatchInline);
                let small = route_brick_rebuild(true, incremental_edit, mirror_resident, 1, THRESHOLD);
                assert_eq!(small, BrickRebuildAction::WholesaleInline);
            }
        }
    }

    /// The quiet fast path: nothing outstanding + a resident mirror + an incremental
    /// edit patches in place (the G3 per-edit cost, proportional to the dirty set).
    #[test]
    fn quiet_incremental_with_mirror_patches_inline() {
        assert_eq!(
            route_brick_rebuild(false, true, true, THRESHOLD + 1, THRESHOLD),
            BrickRebuildAction::PatchInline,
            "the covering-set size is irrelevant to a patch (its cost is the dirty set)"
        );
    }

    /// An incremental edit with NO resident mirror (emptied earlier, or startup
    /// dispatched async and nothing landed yet) has nothing sound to patch — it goes
    /// wholesale, threshold-gated between inline and async like any wholesale.
    #[test]
    fn incremental_without_mirror_goes_wholesale() {
        assert_eq!(
            route_brick_rebuild(false, true, false, THRESHOLD + 1, THRESHOLD),
            BrickRebuildAction::WholesaleAsync
        );
        assert_eq!(
            route_brick_rebuild(false, true, false, THRESHOLD, THRESHOLD),
            BrickRebuildAction::WholesaleInline,
            "a wholesale AT the threshold builds inline (matches the mesh gate)"
        );
    }

    /// A wholesale-shaped edit ignores the mirror and gates on the covering-set size.
    #[test]
    fn wholesale_edit_threshold_gates_inline_vs_async() {
        assert_eq!(
            route_brick_rebuild(false, false, true, THRESHOLD + 1, THRESHOLD),
            BrickRebuildAction::WholesaleAsync
        );
        assert_eq!(
            route_brick_rebuild(false, false, true, THRESHOLD, THRESHOLD),
            BrickRebuildAction::WholesaleInline
        );
    }

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
