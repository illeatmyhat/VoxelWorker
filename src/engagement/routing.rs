//! Display routing policy — the pure per-edit decision functions for the two display
//! pipelines (the cuboid fallback mesh and the ADR 0011 brick raymarch).
//!
//! Each function decides, for one edit, WHERE its derived display artifacts are
//! (re)built — inline on the main thread, or dispatched to a background worker
//! (stale-while-rebuilding) — and how a stale artifact is guarded against an unsound
//! inline patch. They are pure (no GPU, no window), so every routing invariant is
//! unit-testable in the lib.
//!
//! The three per-pipeline functions ([`route_geometry_rebuild`], [`route_mesh_build`],
//! [`route_brick_rebuild`]) are thin wrappers over ONE shared policy,
//! [`route_derived_artifact`], applied to a per-artifact [`DerivedArtifactState`]. They differ
//! only in which staleness inputs they fold into that state (mesh staleness, mirror residency,
//! engagement) and in the single load-bearing divergence recorded as
//! `inline_install_supersedes_in_flight` — so the interlock, the one rule that must never be
//! dialectal, is defined and tested exactly once. The state machine that ACTS on these decisions
//! lives in `src/display/orchestrator.rs` (the [`DisplayOrchestrator`](super::orchestrator::DisplayOrchestrator));
//! the async workers that execute a dispatched rebuild live in [`crate::workers::geometry`] and
//! [`crate::workers::brick`]; and the generation bookkeeping behind supersede is
//! [`GenerationTracker`].
//!
//! ## The interlock (the fog/mesh-era law: NEVER patch a stale artifact)
//! While an async wholesale brick build is OUTSTANDING the resident
//! [`IncrementalBrickField`](display::brick::IncrementalBrickField) mirror (and the
//! renderer's live field) reflect S0 while the worker builds S1 — so an incremental edit
//! must NOT patch them (that would strand every brick that differs S0→S1 but isn't in the
//! new dirty set, the Frankenstein field). [`route_brick_rebuild`] therefore routes EVERY
//! mid-flight edit WHOLESALE. One DELIBERATE divergence from [`route_geometry_rebuild`]
//! (which sends every mid-flight edit async): a mid-flight wholesale whose covering set is
//! SMALL rebuilds INLINE — immediately current, no worker latency. That is sound ONLY
//! because the shell's inline install seam (`DisplayOrchestrator::finish_brick_install` in
//! `src/display/orchestrator.rs`) bumps the generation, so the superseded in-flight result is
//! discarded on arrival; do not remove that bump. The decision is pure so the interlock is
//! unit-testable.

/// The covering-chunk count above which a WHOLESALE geometry rebuild is dispatched to
/// the background worker instead of built inline (issue #60).
///
/// A rebuild covering at most this many chunks is cheap enough to build synchronously on
/// the main thread without a perceptible hitch (the small-object common case), so it
/// avoids the worker hop + its one-frame swap latency. Only a rebuild whose covering set
/// EXCEEDS this — a large object's initial create / resize / density / recentre, the ~3s
/// case the issue targets — goes async. Chosen conservatively: at the default density a
/// chunk is `4×4×4` blocks, so 128 chunks is a large multi-hundred-block object, well
/// past the point where an inline build stalls a frame. (Incremental dirty-chunk edits —
/// the #54/#55 fast path — stay inline REGARDLESS of this threshold; only WHOLESALE
/// rebuilds consult it.)
pub const ASYNC_REBUILD_CHUNK_THRESHOLD: usize = 128;

/// The shape of the edit the resolve produced (issue #60 C1), consumed by
/// [`route_geometry_rebuild`]. Either the edit localised to a few dirty chunks (an inline
/// incremental fast-path candidate) or it needs a wholesale rebuild of `chunk_count`
/// covering chunks (threshold-gated between inline and async).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditShape {
    /// The edit localised — the resolve returned `incremental_dirty_chunks = Some(..)`.
    Incremental,
    /// The edit needs a full rebuild (`incremental_dirty_chunks = None`), covering
    /// `chunk_count` chunks. The threshold decides inline-vs-async.
    Wholesale { chunk_count: usize },
}

/// Where an edit's geometry rebuild is routed (issue #60 C1). Extracted as a pure decision
/// so the C1 interlock — "do NOT inline-patch the currently-installed renderer while an
/// async wholesale build is OUTSTANDING" — is unit-testable without a live window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebuildRoute {
    /// Apply an incremental dirty-chunk re-mesh to the CURRENTLY-installed renderer in
    /// place (the #54/#55 fast path). Sound ONLY when no async build is outstanding — the
    /// installed renderer then reflects the latest resolve.
    InlineIncremental,
    /// Rebuild the WHOLE renderer inline on the main thread (small wholesale, at/below the
    /// async threshold — cheap enough not to hitch a frame).
    WholesaleInline,
    /// Dispatch a WHOLESALE rebuild from the CURRENT full covering set to the async worker
    /// (stale-while-rebuilding). Chosen for a large wholesale edit AND — the C1 interlock —
    /// for ANY edit (even an incremental one) while an async build is outstanding: the
    /// installed renderer is STALE (S0) while the worker builds S1, so inline-patching it
    /// would strand every chunk that differs S0→S1 but isn't in the new dirty set (the
    /// Frankenstein mesh). Re-dispatching a fresh wholesale from the current `AppCore`
    /// resident cache (always current on the main thread) is correct; the worker's
    /// drain-to-latest converges once the user stops editing.
    WholesaleAsync,
}

/// The routing-relevant state of ONE derived display artifact — the cuboid fallback mesh, the
/// brick raymarch field, and every future one the display grows (the per-voxel material atlas,
/// an export snapshot, a nav/occupancy summary for agents). The four booleans are the entire
/// input surface the shared routing policy needs; each artifact's wrapper fills them from its
/// own residency bookkeeping. See [`route_derived_artifact`] for the policy that consumes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DerivedArtifactState {
    /// The resident artifact reflects the LATEST resolve — it is neither stale nor a
    /// placeholder. When false the resident copy cannot be inline-patched (patching a stale
    /// artifact strands every change it does not reflect — the Frankenstein-artifact hazard the
    /// interlock guards); a wholesale rebuild is forced.
    pub current: bool,
    /// An async WHOLESALE rebuild is in flight — dispatched to a worker, not yet accepted and
    /// installed. Like a non-`current` artifact, an outstanding rebuild means the resident copy
    /// is stale (it reflects S0 while the worker builds S1), so it must not be inline-patched.
    pub outstanding: bool,
    /// A patch TARGET is resident — the mesh's re-meshable buffers, or the brick's incremental
    /// mirror. When false there is nothing sound to patch in place, so even a localised edit
    /// rebuilds wholesale.
    pub patchable: bool,
    /// The shell's INLINE install seam bumps the supersede generation, so building a small
    /// wholesale INLINE while a rebuild is still outstanding is sound: the in-flight worker
    /// result is discarded on arrival (its generation is stale). True for the brick, whose
    /// `finish_brick_install` seam bumps the generation; false for the mesh/geometry, which have
    /// no such inline seam and so must route EVERY mid-flight edit to the worker for convergence.
    /// THE one load-bearing difference between the artifacts' dialects — see the interlock note
    /// in the module doc; never erase it or fold it away.
    pub inline_install_supersedes_in_flight: bool,
}

/// The one routing policy shared by every derived display artifact — the single rule the three
/// per-artifact wrappers ([`route_geometry_rebuild`], [`route_mesh_build`],
/// [`route_brick_rebuild`]) now express: *patch inline iff the resident artifact is current and
/// the edit is localised; otherwise rebuild wholesale — inline below the chunk threshold, async
/// above it; while the resident copy is stale (outstanding OR not current) never patch, and —
/// unless this artifact's inline install seam supersedes an in-flight result — never build a
/// wholesale inline either.* Pure (no GPU, no window) so the interlock is unit-testable.
///
/// The decision, in table form:
/// * `Incremental` edit AND the resident copy is current AND patchable AND nothing outstanding →
///   [`InlineIncremental`](RebuildRoute::InlineIncremental) (the patch fast path).
/// * Otherwise a wholesale is needed. It is *interlocked* — forced to the worker — when the
///   resident copy is stale (`outstanding || !current`) AND this artifact does NOT supersede an
///   in-flight result on inline install. Route [`WholesaleAsync`](RebuildRoute::WholesaleAsync)
///   when interlocked OR the wholesale's covering count exceeds the threshold; else
///   [`WholesaleInline`](RebuildRoute::WholesaleInline).
///
/// An `Incremental` edit that cannot take the fast path carries no wholesale covering count
/// (`EditShape::Incremental` is count-less). For the geometry/mesh artifacts that case is always
/// interlocked (their `patchable` is always true, so failing the fast path implies stale, and
/// they do not supersede), so it routes async and the absent count is never consulted. The brick
/// never reaches the core with a count-less incremental wholesale — its wrapper converts an
/// incremental edit that cannot patch into an explicit `Wholesale { chunk_count }` first. So the
/// count-less wholesale path is only reachable by a direct call with an artifact-shape no wrapper
/// produces; there it routes conservatively to the worker (no size to justify an inline build).
pub fn route_derived_artifact(
    state: DerivedArtifactState,
    edit: EditShape,
    async_threshold: usize,
) -> RebuildRoute {
    if edit == EditShape::Incremental && !state.outstanding && state.current && state.patchable {
        return RebuildRoute::InlineIncremental;
    }
    // A wholesale is needed. It is forced to the worker when the resident copy is stale AND this
    // artifact cannot soundly install a wholesale inline over an in-flight result.
    let interlocked =
        (state.outstanding || !state.current) && !state.inline_install_supersedes_in_flight;
    let exceeds_threshold = match edit {
        EditShape::Wholesale { chunk_count } => chunk_count > async_threshold,
        // A count-less incremental wholesale (see the doc): no size justifies an inline build,
        // so route to the worker conservatively.
        EditShape::Incremental => true,
    };
    if interlocked || exceeds_threshold {
        RebuildRoute::WholesaleAsync
    } else {
        RebuildRoute::WholesaleInline
    }
}

/// Decide where an edit's geometry rebuild is routed (issue #60 C1), given whether an async
/// wholesale build is currently OUTSTANDING (dispatched but not yet accepted/installed) and the
/// [`EditShape`] the resolve produced. A thin wrapper over [`route_derived_artifact`]: the
/// installed renderer is always current and always inline-patchable, so an outstanding build is
/// geometry's only stale-forcing input (the C1 interlock). Pure — no GPU, no window.
pub fn route_geometry_rebuild(
    async_outstanding: bool,
    edit: EditShape,
    async_threshold: usize,
) -> RebuildRoute {
    // The installed renderer is always current on the main thread and always inline-patchable
    // (re-meshable buffers), and geometry has no inline supersede seam — so an outstanding async
    // build is the only thing that forces a stale rebuild here (the C1 interlock).
    route_derived_artifact(
        DerivedArtifactState {
            current: true,
            outstanding: async_outstanding,
            patchable: true,
            inline_install_supersedes_in_flight: false,
        },
        edit,
        async_threshold,
    )
}

/// Whether — and how — an edit must (re)build the fallback CUBOID MESH, given that the
/// ADR 0011 brick raymarch is the actual display sink (perf follow-up to epic #64). The
/// mesh is drawn ONLY when the brick raymarch is not engaged (no installed field, debug-face
/// mode, or a loaded VS material); when the brick IS the display the mesh is pure redundant
/// per-edit work — the ~333ms serial build on a big scene — and is SKIPPED, leaving it stale.
///
/// A skipped-stale mesh is exactly as untrustworthy as a still-building async result: it does
/// NOT reflect the latest resolve, so an incremental edit must NOT inline-patch it (that would
/// strand every chunk that changed while the mesh was skipped — the same Frankenstein-mesh
/// hazard the C1 interlock guards). Staleness therefore composes into the interlock by OR-ing
/// with `async_outstanding` before delegating to [`route_geometry_rebuild`], forcing a fresh
/// wholesale build the moment the mesh is next needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshBuildRoute {
    /// The brick raymarch is engaged — the fallback mesh is not drawn. Skip building it and
    /// mark it STALE (the caller records staleness so the next build-required edit goes
    /// wholesale, never an inline patch of the stranded buffers).
    Skip,
    /// The mesh is (or is about to become) the display — build it via this underlying route.
    Build(RebuildRoute),
}

/// Decide whether the fallback cuboid mesh needs building for this edit (the brick-display
/// perf follow-up to epic #64). Pure — no GPU, no window — so the skip/stale/interlock rule
/// is unit-testable in the lib, like [`route_geometry_rebuild`].
///
/// * `brick_display_engaged` — the brick raymarch will draw this frame (a field is installed,
///   no debug-face mode, no loaded VS material). When `true` the mesh is redundant → `Skip`.
/// * `mesh_stale` — the currently-installed mesh was previously SKIPPED (or otherwise does not
///   reflect the latest resolve). A stale mesh cannot be inline-patched, so it is OR-ed into
///   the C1 interlock, forcing a wholesale build.
pub fn route_mesh_build(
    brick_display_engaged: bool,
    mesh_stale: bool,
    async_outstanding: bool,
    edit: EditShape,
    async_threshold: usize,
) -> MeshBuildRoute {
    if brick_display_engaged {
        return MeshBuildRoute::Skip;
    }
    // The mesh is the display this frame — it must become valid. A stale (previously-skipped)
    // mesh is `current: false`, which forces a wholesale rebuild exactly as an outstanding async
    // build does; the mesh has no inline supersede seam, so both stale conditions interlock.
    MeshBuildRoute::Build(route_derived_artifact(
        DerivedArtifactState {
            current: !mesh_stale,
            outstanding: async_outstanding,
            patchable: true,
            inline_install_supersedes_in_flight: false,
        },
        edit,
        async_threshold,
    ))
}

/// The brick display's fate when a rebuild did NOT (re)install it (F1 — the deferred handover
/// decision, brick-display perf follow-up to epic #64). Pure so the "keep the stale brick
/// drawing until the async replacement mesh installs" rule is unit-testable without a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrickDisplayHandover {
    /// The brick raymarch is (still) the live display this rebuild — no handover; any pending
    /// deferred clear is cancelled (the brick is drawing, not being replaced).
    KeepAsDisplay,
    /// Hand the display back to the cuboid mesh NOW: clear the stale brick field this frame.
    /// Chosen when the replacement mesh is already current, OR the brick can't/needn't draw (a
    /// mesh-only display mode is active — debug-face / loaded material), OR no live field remains.
    ClearNow,
    /// DEFER the clear (F1): a stale brick field is still live, the replacement mesh is building
    /// ASYNC, and the brick would still draw — keep it on-screen so the model never blanks for the
    /// seconds the worker takes, and clear it in the mesh-install seam once the fresh mesh lands.
    DeferUntilInstall,
}

/// Decide the brick display's handover when a rebuild did not (re)install the brick sink (F1).
/// Pure — no GPU, no window — so the deferred-clear rule is unit-testable like the routing.
///
/// * `brick_reinstalled_this_rebuild` — the brick installed/patched a field this rebuild (it is
///   the live display). Then it KEEPS the frame; there is nothing to hand over.
/// * `replacement_mesh_current_this_frame` — the fallback mesh became current this frame (an
///   inline build/patch). Then the brick can be cleared immediately (the mesh draws instead).
/// * `brick_would_draw_if_kept` — the brick WOULD draw if its field were kept (no debug-face,
///   no loaded material). When false, keeping a stale field is pointless AND risks a stale patch
///   (F2), so clear now.
/// * `has_live_brick_field` — a non-empty brick field is actually resident to keep. When false
///   there is nothing to defer, so clear now (a no-op on the empty field).
///
/// Only when the brick is NOT the display, the replacement is still building async, the brick
/// would draw, and a live field exists does the clear DEFER — the one case the model would
/// otherwise blank.
pub fn brick_display_handover(
    brick_reinstalled_this_rebuild: bool,
    replacement_mesh_current_this_frame: bool,
    brick_would_draw_if_kept: bool,
    has_live_brick_field: bool,
) -> BrickDisplayHandover {
    if brick_reinstalled_this_rebuild {
        return BrickDisplayHandover::KeepAsDisplay;
    }
    if replacement_mesh_current_this_frame || !brick_would_draw_if_kept || !has_live_brick_field {
        return BrickDisplayHandover::ClearNow;
    }
    BrickDisplayHandover::DeferUntilInstall
}

/// Whether an incremental brick edit may PATCH the resident GPU field in place, or must INSTALL
/// a fresh field (F2 — brick-display perf follow-up to epic #64). Pure so the "a cleared/empty/
/// placeholder field cannot be patched" rule is unit-testable.
///
/// Patch iff an incremental `update` was produced AND the renderer actually HOLDS A LIVE FIELD
/// AND that field is not a stale handover placeholder. The two staleness inputs:
/// * `renderer_holds_live_field` — gating on live residency (not merely renderer-present) is
///   the F2 fix: during a loaded-material (or any cleared) window the renderer's field was
///   zeroed while the CPU mirror kept patching, so a patch would write only the LAST edit's
///   slots over an empty field — a stale atlas. A present-but-empty renderer must re-INSTALL.
/// * `field_pending_replacement` — during an F1 deferred-handover window the live field is a
///   STALE visual placeholder kept drawing only until the replacement mesh lands; it does not
///   reflect the latest resolve, so an edit that restores representability must INSTALL a
///   fresh field, never patch the placeholder (the same Frankenstein hazard, one level up).
pub fn brick_patch_in_place(
    has_incremental_update: bool,
    renderer_holds_live_field: bool,
    field_pending_replacement: bool,
) -> bool {
    has_incremental_update && renderer_holds_live_field && !field_pending_replacement
}

// The monotonic-generation accept/discard bookkeeping behind supersede is the substrate
// `supersede` protocol; the display keeps its `GenerationTracker` vocabulary at this seam so
// the orchestrator/shell call sites stay put. See docs/architecture/04-work.md (the work
// chapter) for how routing pairs a `GenerationTracker` with each async worker.
pub use substrate::GenerationTracker;

/// Where a brick rebuild is routed. The brick analogue of [`RebuildRoute`], with the patch
/// precondition (a resident mirror) folded in so the shell reads ONE decision.
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

/// Decide where an edit's brick rebuild is routed. A thin wrapper over the shared
/// [`route_derived_artifact`] policy, translating the brick's residency inputs into a
/// [`DerivedArtifactState`] and mapping the result to the brick's action enum. Pure — no GPU, no
/// window — so the interlock is unit-testable.
///
/// The load-bearing divergence from the mesh/geometry artifacts, carried through the shared
/// policy as `inline_install_supersedes_in_flight: true`: while an async brick build is
/// outstanding a SMALL mid-flight wholesale rebuilds INLINE rather than re-dispatching. That is
/// sound ONLY because the shell's inline install seam (`DisplayOrchestrator::finish_brick_install`
/// in `src/display/orchestrator.rs`) bumps
/// the supersede generation, so the superseded in-flight result is discarded on arrival — see the
/// module doc's interlock note. A large mid-flight wholesale still re-dispatches async.
///
/// The brick alone must feed the shared core an explicit covering count when an incremental edit
/// cannot patch (no resident mirror, or mid-flight): `EditShape::Incremental` is count-less, so
/// this wrapper converts an incremental edit that will NOT patch into an explicit
/// `Wholesale { chunk_count: covering_chunk_count }`, threshold-gated like any wholesale. The core
/// then owns the inline-vs-async decision; the wrapper only owns this artifact-specific
/// incremental-vs-wholesale translation, which requires the brick's mirror-residency knowledge.
pub fn route_brick_rebuild(
    async_outstanding: bool,
    incremental_edit: bool,
    mirror_resident: bool,
    covering_chunk_count: usize,
    async_threshold: usize,
) -> BrickRebuildAction {
    // An incremental edit is expressible as a localised patch only when a mirror is resident and
    // no rebuild is outstanding; otherwise it must be realised as a wholesale of its covering set
    // (which carries the count the shared policy needs to gate inline vs async).
    let edit = if incremental_edit && mirror_resident && !async_outstanding {
        EditShape::Incremental
    } else {
        EditShape::Wholesale {
            chunk_count: covering_chunk_count,
        }
    };
    let route = route_derived_artifact(
        DerivedArtifactState {
            // The live brick field is always current on the main thread; residency of a patch
            // target is the brick's `patchable`, and the inline install seam supersedes in flight.
            current: true,
            outstanding: async_outstanding,
            patchable: mirror_resident,
            inline_install_supersedes_in_flight: true,
        },
        edit,
        async_threshold,
    );
    match route {
        RebuildRoute::InlineIncremental => BrickRebuildAction::PatchInline,
        RebuildRoute::WholesaleInline => BrickRebuildAction::WholesaleInline,
        RebuildRoute::WholesaleAsync => BrickRebuildAction::WholesaleAsync,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const THRESHOLD: usize = ASYNC_REBUILD_CHUNK_THRESHOLD;

    /// C1 interlock — the core fix. With an async wholesale build OUTSTANDING, an
    /// incremental edit must NOT inline-patch the stale (S0) renderer (that strands every
    /// chunk that differs S0→S1 but isn't in the new dirty set — the Frankenstein mesh).
    /// It routes to a fresh WHOLESALE-async dispatch from the current resident cache.
    #[test]
    fn outstanding_incremental_routes_to_wholesale_async_not_inline() {
        let route = route_geometry_rebuild(true, EditShape::Incremental, THRESHOLD);
        assert_eq!(
            route,
            RebuildRoute::WholesaleAsync,
            "an incremental edit while a build is outstanding must re-dispatch wholesale, \
             never inline-patch the stale renderer (C1)"
        );
        assert_ne!(route, RebuildRoute::InlineIncremental);
    }

    /// C1 — a SMALL wholesale edit that would normally build inline ALSO routes to async
    /// while outstanding: building it inline would overwrite the S0 renderer just as the
    /// outstanding S1 is about to (or the reverse), so route to the worker for convergence.
    #[test]
    fn outstanding_small_wholesale_routes_to_wholesale_async() {
        let small = EditShape::Wholesale { chunk_count: 1 };
        assert_eq!(
            route_geometry_rebuild(true, small, THRESHOLD),
            RebuildRoute::WholesaleAsync,
            "with a build outstanding EVERY edit re-dispatches wholesale-async"
        );
    }

    /// C1 — a large wholesale edit while outstanding is also async (it would be anyway).
    #[test]
    fn outstanding_large_wholesale_routes_to_wholesale_async() {
        let large = EditShape::Wholesale {
            chunk_count: THRESHOLD + 1,
        };
        assert_eq!(
            route_geometry_rebuild(true, large, THRESHOLD),
            RebuildRoute::WholesaleAsync
        );
    }

    /// No build outstanding: the inline incremental fast-path resumes (the installed
    /// renderer reflects the latest resolve, so patching it in place is sound).
    #[test]
    fn not_outstanding_incremental_routes_inline() {
        assert_eq!(
            route_geometry_rebuild(false, EditShape::Incremental, THRESHOLD),
            RebuildRoute::InlineIncremental,
            "with nothing outstanding an incremental edit patches in place (the fast path)"
        );
    }

    /// No build outstanding + a SMALL wholesale (at/below threshold) → inline wholesale.
    #[test]
    fn not_outstanding_small_wholesale_routes_inline() {
        let at = EditShape::Wholesale {
            chunk_count: THRESHOLD,
        };
        assert_eq!(
            route_geometry_rebuild(false, at, THRESHOLD),
            RebuildRoute::WholesaleInline,
            "a wholesale rebuild AT the threshold builds inline"
        );
    }

    /// No build outstanding + a LARGE wholesale (exceeds threshold) → async (the #60 case).
    #[test]
    fn not_outstanding_large_wholesale_routes_async() {
        let large = EditShape::Wholesale {
            chunk_count: THRESHOLD + 1,
        };
        assert_eq!(
            route_geometry_rebuild(false, large, THRESHOLD),
            RebuildRoute::WholesaleAsync,
            "a wholesale rebuild exceeding the threshold dispatches to the worker"
        );
    }

    // --- route_mesh_build: skip the fallback mesh while the brick display is engaged ---

    /// Brick engaged → the fallback mesh is never drawn, so skip building it regardless of the
    /// edit's shape, staleness, or an outstanding async — this is the ~333ms-per-edit win.
    #[test]
    fn brick_engaged_skips_mesh_regardless() {
        for &stale in &[false, true] {
            for &outstanding in &[false, true] {
                for edit in [
                    EditShape::Incremental,
                    EditShape::Wholesale { chunk_count: 1 },
                    EditShape::Wholesale {
                        chunk_count: THRESHOLD + 1,
                    },
                ] {
                    assert_eq!(
                        route_mesh_build(true, stale, outstanding, edit, THRESHOLD),
                        MeshBuildRoute::Skip,
                        "engaged brick display always skips the redundant mesh build"
                    );
                }
            }
        }
    }

    /// Brick NOT engaged + a fresh (non-stale) mesh + nothing outstanding → the normal
    /// `route_geometry_rebuild` decision applies unchanged (the mesh is the live display).
    #[test]
    fn mesh_display_fresh_uses_normal_routing() {
        assert_eq!(
            route_mesh_build(false, false, false, EditShape::Incremental, THRESHOLD),
            MeshBuildRoute::Build(RebuildRoute::InlineIncremental),
            "a fresh live mesh takes the inline incremental fast-path"
        );
        assert_eq!(
            route_mesh_build(
                false,
                false,
                false,
                EditShape::Wholesale { chunk_count: 1 },
                THRESHOLD
            ),
            MeshBuildRoute::Build(RebuildRoute::WholesaleInline),
        );
    }

    /// The core new rule: an incremental edit onto a SKIPPED-STALE mesh must NOT inline-patch
    /// (the mesh was skipped, so its buffers are stranded/empty — patching strands every
    /// intervening change, the Frankenstein mesh). Staleness forces a wholesale build exactly
    /// like the C1 interlock does for an outstanding async build.
    #[test]
    fn stale_mesh_incremental_forces_wholesale_not_inline() {
        let route = route_mesh_build(false, true, false, EditShape::Incremental, THRESHOLD);
        assert_eq!(
            route,
            MeshBuildRoute::Build(RebuildRoute::WholesaleAsync),
            "a stale mesh must rebuild wholesale when it becomes the display, never inline-patch"
        );
        assert_ne!(route, MeshBuildRoute::Build(RebuildRoute::InlineIncremental));
    }

    /// Interlock composition: staleness OR an outstanding async — either one forces wholesale.
    /// A stale mesh with nothing outstanding still routes wholesale (staleness alone suffices).
    #[test]
    fn stale_composes_with_c1_interlock() {
        // Stale + not-outstanding small wholesale still routes async (stale ⇒ no inline patch).
        assert_eq!(
            route_mesh_build(
                false,
                true,
                false,
                EditShape::Wholesale { chunk_count: 1 },
                THRESHOLD
            ),
            MeshBuildRoute::Build(RebuildRoute::WholesaleAsync),
        );
        // Not-stale but outstanding: the existing C1 interlock still forces async.
        assert_eq!(
            route_mesh_build(false, false, true, EditShape::Incremental, THRESHOLD),
            MeshBuildRoute::Build(RebuildRoute::WholesaleAsync),
        );
    }

    // --- brick_display_handover: the F1 deferred-clear rule ---

    /// When the brick (re)installed this rebuild it IS the live display — keep it, cancel any
    /// pending deferred clear, regardless of the other flags.
    #[test]
    fn brick_reinstalled_keeps_display() {
        for &mesh_current in &[false, true] {
            for &would_draw in &[false, true] {
                for &has_field in &[false, true] {
                    assert_eq!(
                        brick_display_handover(true, mesh_current, would_draw, has_field),
                        BrickDisplayHandover::KeepAsDisplay,
                        "a brick that installed this rebuild is the display"
                    );
                }
            }
        }
    }

    /// THE F1 CASE: brick disengaged, the replacement mesh is building ASYNC (not current this
    /// frame), the brick would still draw, and a live field remains → DEFER the clear so the
    /// stale brick keeps drawing until the fresh mesh installs (the model never blanks).
    #[test]
    fn disengaged_async_live_brick_defers_clear() {
        assert_eq!(
            brick_display_handover(false, false, true, true),
            BrickDisplayHandover::DeferUntilInstall,
            "keep the stale brick drawing until the async replacement mesh lands"
        );
    }

    /// The replacement mesh became current THIS frame (an inline build/patch) → clear the brick
    /// now; the fresh mesh draws instead, no blank.
    #[test]
    fn disengaged_mesh_current_clears_now() {
        assert_eq!(
            brick_display_handover(false, true, true, true),
            BrickDisplayHandover::ClearNow,
            "an inline replacement mesh is current — hand over immediately"
        );
    }

    /// A mesh-only mode is active (debug-face / loaded material) so the brick would NOT draw even
    /// if kept → clear now (keeping a stale field is pointless and risks a stale patch, F2). This
    /// preserves the pre-F1 behaviour for the loaded-material window.
    #[test]
    fn disengaged_brick_would_not_draw_clears_now() {
        assert_eq!(
            brick_display_handover(false, false, false, true),
            BrickDisplayHandover::ClearNow,
            "if the brick can't draw anyway, don't defer — clear (avoids the F2 stale patch)"
        );
    }

    /// No live brick field remains (the edit emptied it) → nothing to defer, clear now (a no-op
    /// on the already-empty field).
    #[test]
    fn disengaged_no_live_field_clears_now() {
        assert_eq!(
            brick_display_handover(false, false, true, false),
            BrickDisplayHandover::ClearNow,
            "no live field to keep — clear now"
        );
    }

    // --- brick_patch_in_place: the F2 stale-patch gate ---

    /// Patch only when an incremental update exists AND the renderer holds a LIVE field
    /// AND that field is not a stale F1-handover placeholder.
    #[test]
    fn patch_requires_update_and_live_current_field() {
        assert!(
            brick_patch_in_place(true, true, false),
            "an incremental update onto a live, current resident field patches in place"
        );
        // F2: a present-but-CLEARED renderer (no live field) must INSTALL fresh, never patch —
        // patching would write only the last edit's slots over the emptied atlas (a stale atlas).
        assert!(
            !brick_patch_in_place(true, false, false),
            "an update onto a cleared/empty field must re-install, not patch (F2)"
        );
        // F1 placeholder: a live field awaiting a deferred handover clear is a STALE visual
        // placeholder — an edit that restores representability must INSTALL fresh, never patch
        // the placeholder (patching writes one edit's slots over a field reflecting neither
        // the old nor the new resolve).
        assert!(
            !brick_patch_in_place(true, true, true),
            "an update onto a pending-replacement placeholder must re-install, not patch"
        );
        // No incremental update (a wholesale build) always installs.
        assert!(!brick_patch_in_place(false, true, false));
        assert!(!brick_patch_in_place(false, false, false));
        assert!(!brick_patch_in_place(false, true, true));
    }

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

    // --- route_derived_artifact: the shared policy's exhaustive decision table ---

    /// The three edit shapes the exhaustive table iterates: a localised edit, a wholesale AT the
    /// threshold (builds inline when not interlocked), and a wholesale ABOVE it (async).
    const EDIT_SHAPES: [EditShape; 3] = [
        EditShape::Incremental,
        EditShape::Wholesale {
            chunk_count: THRESHOLD,
        },
        EditShape::Wholesale {
            chunk_count: THRESHOLD + 1,
        },
    ];

    /// An independent restatement of the routing table, used to pin [`route_derived_artifact`]
    /// exhaustively. Kept deliberately as a readable if-ladder (not a copy of the implementation's
    /// expression form) so a refactor that changes the policy must change BOTH to stay green.
    fn expected_route(
        state: DerivedArtifactState,
        edit: EditShape,
        threshold: usize,
    ) -> RebuildRoute {
        // The patch fast path: a localised edit onto a current, patchable, settled artifact.
        let can_patch = matches!(edit, EditShape::Incremental)
            && state.current
            && state.patchable
            && !state.outstanding;
        if can_patch {
            return RebuildRoute::InlineIncremental;
        }
        // A wholesale is needed. The resident copy is stale when a rebuild is outstanding or it is
        // not current; that forces the worker UNLESS this artifact supersedes an inline install.
        let stale = state.outstanding || !state.current;
        let forced_to_worker = stale && !state.inline_install_supersedes_in_flight;
        let big_enough_for_worker = match edit {
            EditShape::Wholesale { chunk_count } => chunk_count > threshold,
            EditShape::Incremental => true,
        };
        if forced_to_worker || big_enough_for_worker {
            RebuildRoute::WholesaleAsync
        } else {
            RebuildRoute::WholesaleInline
        }
    }

    /// Iterate every combination of the four state booleans × the three edit shapes and assert
    /// [`route_derived_artifact`] reproduces the independent table — the policy tested once,
    /// exhaustively (map item 3).
    #[test]
    fn derived_artifact_table_is_exhaustive() {
        for &current in &[false, true] {
            for &outstanding in &[false, true] {
                for &patchable in &[false, true] {
                    for &supersedes in &[false, true] {
                        let state = DerivedArtifactState {
                            current,
                            outstanding,
                            patchable,
                            inline_install_supersedes_in_flight: supersedes,
                        };
                        for edit in EDIT_SHAPES {
                            assert_eq!(
                                route_derived_artifact(state, edit, THRESHOLD),
                                expected_route(state, edit, THRESHOLD),
                                "state={state:?} edit={edit:?} disagrees with the table"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The one cross-check that pins the load-bearing divergence: a mesh-shaped artifact
    /// (`inline_install_supersedes_in_flight = false`) with a rebuild outstanding routes even a
    /// SMALL wholesale to the worker — it cannot soundly install inline over an in-flight result.
    #[test]
    fn mesh_shaped_outstanding_small_wholesale_is_async() {
        let mesh_shaped = DerivedArtifactState {
            current: true,
            outstanding: true,
            patchable: true,
            inline_install_supersedes_in_flight: false,
        };
        assert_eq!(
            route_derived_artifact(mesh_shaped, EditShape::Wholesale { chunk_count: 1 }, THRESHOLD),
            RebuildRoute::WholesaleAsync,
            "no inline supersede seam → a mid-flight small wholesale re-dispatches"
        );
    }

    /// The mirror-image cross-check: a brick-shaped artifact
    /// (`inline_install_supersedes_in_flight = true`) with a rebuild outstanding builds the SAME
    /// small wholesale INLINE — the inline install seam bumps the generation, so the superseded
    /// in-flight result is discarded on arrival. THE divergence the wrappers must preserve.
    #[test]
    fn brick_shaped_outstanding_small_wholesale_is_inline() {
        let brick_shaped = DerivedArtifactState {
            current: true,
            outstanding: true,
            patchable: true,
            inline_install_supersedes_in_flight: true,
        };
        assert_eq!(
            route_derived_artifact(brick_shaped, EditShape::Wholesale { chunk_count: 1 }, THRESHOLD),
            RebuildRoute::WholesaleInline,
            "an inline supersede seam makes a mid-flight small wholesale safe to build inline"
        );
    }

    // --- wrapper equivalence: each dialect equals the shared policy on the artifact's state ---

    /// [`route_geometry_rebuild`] equals the shared policy on geometry's state (always current,
    /// always patchable, never superseding inline) across a small input grid.
    #[test]
    fn geometry_wrapper_matches_shared_policy() {
        for &outstanding in &[false, true] {
            for edit in EDIT_SHAPES {
                let expected = route_derived_artifact(
                    DerivedArtifactState {
                        current: true,
                        outstanding,
                        patchable: true,
                        inline_install_supersedes_in_flight: false,
                    },
                    edit,
                    THRESHOLD,
                );
                assert_eq!(route_geometry_rebuild(outstanding, edit, THRESHOLD), expected);
            }
        }
    }

    /// [`route_mesh_build`] equals `Skip` when the brick is engaged, else the shared policy on the
    /// mesh's state (staleness feeds `current`, no inline supersede seam), across a small grid.
    #[test]
    fn mesh_wrapper_matches_shared_policy() {
        for &engaged in &[false, true] {
            for &stale in &[false, true] {
                for &outstanding in &[false, true] {
                    for edit in EDIT_SHAPES {
                        let got = route_mesh_build(engaged, stale, outstanding, edit, THRESHOLD);
                        let expected = if engaged {
                            MeshBuildRoute::Skip
                        } else {
                            MeshBuildRoute::Build(route_derived_artifact(
                                DerivedArtifactState {
                                    current: !stale,
                                    outstanding,
                                    patchable: true,
                                    inline_install_supersedes_in_flight: false,
                                },
                                edit,
                                THRESHOLD,
                            ))
                        };
                        assert_eq!(got, expected);
                    }
                }
            }
        }
    }

    /// [`route_brick_rebuild`] equals the shared policy on the brick's state (mirror residency
    /// feeds `patchable`, inline supersede seam is true) with the wrapper's incremental→wholesale
    /// covering-count conversion, mapped to the brick action enum, across a small grid.
    #[test]
    fn brick_wrapper_matches_shared_policy() {
        for &outstanding in &[false, true] {
            for &incremental in &[false, true] {
                for &mirror in &[false, true] {
                    for &covering in &[1usize, THRESHOLD, THRESHOLD + 1] {
                        let got = route_brick_rebuild(
                            outstanding, incremental, mirror, covering, THRESHOLD,
                        );
                        let edit = if incremental && mirror && !outstanding {
                            EditShape::Incremental
                        } else {
                            EditShape::Wholesale {
                                chunk_count: covering,
                            }
                        };
                        let core = route_derived_artifact(
                            DerivedArtifactState {
                                current: true,
                                outstanding,
                                patchable: mirror,
                                inline_install_supersedes_in_flight: true,
                            },
                            edit,
                            THRESHOLD,
                        );
                        let expected = match core {
                            RebuildRoute::InlineIncremental => BrickRebuildAction::PatchInline,
                            RebuildRoute::WholesaleInline => BrickRebuildAction::WholesaleInline,
                            RebuildRoute::WholesaleAsync => BrickRebuildAction::WholesaleAsync,
                        };
                        assert_eq!(got, expected);
                    }
                }
            }
        }
    }
}
