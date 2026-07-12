//! Display routing policy — the pure per-edit decision functions for the two display
//! pipelines (the cuboid fallback mesh and the ADR 0011 brick raymarch).
//!
//! Each function decides, for one edit, WHERE its derived display artifacts are
//! (re)built — inline on the main thread, or dispatched to a background worker
//! (stale-while-rebuilding) — and how a stale artifact is guarded against an unsound
//! inline patch. They are pure (no GPU, no window), so every routing invariant is
//! unit-testable in the lib. The state machine that ACTS on these decisions still lives
//! in `main.rs` (a later slice extracts a DisplayOrchestrator alongside this module); the
//! async workers that execute a dispatched rebuild live in [`crate::geometry_worker`] and
//! [`crate::brick_worker`]; and the generation bookkeeping behind supersede is
//! [`GenerationTracker`].
//!
//! ## The interlock (the fog/mesh-era law: NEVER patch a stale artifact)
//! While an async wholesale brick build is OUTSTANDING the resident
//! [`IncrementalBrickField`](crate::brick_field::IncrementalBrickField) mirror (and the
//! renderer's live field) reflect S0 while the worker builds S1 — so an incremental edit
//! must NOT patch them (that would strand every brick that differs S0→S1 but isn't in the
//! new dirty set, the Frankenstein field). [`route_brick_rebuild`] therefore routes EVERY
//! mid-flight edit WHOLESALE. One DELIBERATE divergence from [`route_geometry_rebuild`]
//! (which sends every mid-flight edit async): a mid-flight wholesale whose covering set is
//! SMALL rebuilds INLINE — immediately current, no worker latency. That is sound ONLY
//! because the shell's inline install seam (`finish_brick_install` in `main.rs`) bumps
//! the generation, so the superseded in-flight result is discarded on arrival; do not
//! remove that bump. The decision is pure so the interlock is unit-testable.

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

/// Decide where an edit's geometry rebuild is routed (issue #60 C1), given whether an async
/// wholesale build is currently OUTSTANDING (dispatched but not yet accepted/installed) and
/// the [`EditShape`] the resolve produced. Pure — no GPU, no window — so the C1 interlock is
/// unit-testable.
///
/// The load-bearing rule: while an async build is outstanding the currently-installed
/// renderer does NOT reflect the latest resolve (it is still S0 while the worker builds S1),
/// so an incremental edit must NOT inline-patch it — that produces the Frankenstein mesh
/// described in C1. Route EVERY edit to a fresh wholesale-async dispatch instead. Only when
/// nothing is outstanding is the installed renderer current, so the inline incremental
/// fast-path (and the small-wholesale-inline path) is safe to resume.
pub fn route_geometry_rebuild(
    async_outstanding: bool,
    edit: EditShape,
    async_threshold: usize,
) -> RebuildRoute {
    if async_outstanding {
        // C1 interlock: never inline-patch a stale renderer. Re-dispatch a fresh wholesale
        // async build from the current resident cache, regardless of the edit's shape.
        return RebuildRoute::WholesaleAsync;
    }
    match edit {
        EditShape::Incremental => RebuildRoute::InlineIncremental,
        EditShape::Wholesale { chunk_count } if chunk_count > async_threshold => {
            RebuildRoute::WholesaleAsync
        }
        EditShape::Wholesale { .. } => RebuildRoute::WholesaleInline,
    }
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
    // The mesh is the display this frame — it must become valid. Fold staleness into the
    // interlock: a stale mesh, like an outstanding async build, must not be inline-patched.
    MeshBuildRoute::Build(route_geometry_rebuild(
        async_outstanding || mesh_stale,
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

/// The monotonic generation bookkeeping behind supersede (issue #60) — factored out of
/// the live shell so the accept/discard decision is unit-testable without a window.
///
/// The shell holds one of these. On each WHOLESALE async dispatch it calls
/// [`next_generation`](Self::next_generation) to stamp the request; when a result arrives
/// it calls [`accepts`](Self::accepts) to decide whether to swap it in. A result is
/// accepted only when its generation is the NEWEST dispatched — an older generation (a
/// build that a later edit superseded) is discarded, so the stale mesh is never swapped
/// in over a fresher scene.
#[derive(Debug, Default, Clone, Copy)]
pub struct GenerationTracker {
    /// The generation of the most recent request dispatched to the worker. `0` before any
    /// dispatch (no async build is outstanding, so nothing is accepted).
    latest_dispatched: u64,
}

impl GenerationTracker {
    /// A fresh tracker (no async rebuild dispatched yet).
    pub fn new() -> Self {
        Self { latest_dispatched: 0 }
    }

    /// Mint the next generation for a wholesale async dispatch and record it as the newest
    /// outstanding (issue #60). Generations are strictly increasing from 1, so a later
    /// edit's request always outranks an earlier one still in flight.
    pub fn next_generation(&mut self) -> u64 {
        self.latest_dispatched += 1;
        self.latest_dispatched
    }

    /// Whether a result of `generation` should be accepted (swapped in) or discarded as
    /// stale (issue #60). Accepted iff it matches the newest dispatched generation — a
    /// result from a superseded (older) request is discarded. A result arriving before any
    /// dispatch (or after the counter moved past it) is never accepted.
    pub fn accepts(&self, generation: u64) -> bool {
        generation != 0 && generation == self.latest_dispatched
    }

    /// The newest dispatched generation (diagnostic / test support).
    pub fn latest_dispatched(&self) -> u64 {
        self.latest_dispatched
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    const THRESHOLD: usize = ASYNC_REBUILD_CHUNK_THRESHOLD;

    /// A fresh tracker accepts nothing — no async rebuild is outstanding, so any result
    /// (which could only be a phantom) is discarded.
    #[test]
    fn fresh_tracker_accepts_nothing() {
        let tracker = GenerationTracker::new();
        assert!(!tracker.accepts(0), "generation 0 is never a valid result");
        assert!(!tracker.accepts(1), "no dispatch yet → accept nothing");
    }

    /// The newest dispatched generation is accepted; every earlier one is discarded as
    /// stale. This is the supersede invariant: a mid-build edit dispatches a newer
    /// generation, so the older in-flight result must NOT swap in over the fresher scene.
    #[test]
    fn newest_wins_stale_discarded() {
        let mut tracker = GenerationTracker::new();
        let first = tracker.next_generation();
        assert_eq!(first, 1);
        // A result for the first request is accepted while it is the newest.
        assert!(tracker.accepts(first));

        // The user edits again mid-build → a newer request is dispatched.
        let second = tracker.next_generation();
        assert_eq!(second, 2, "generations strictly increase");
        // Now the FIRST (in-flight) result is stale and must be discarded…
        assert!(!tracker.accepts(first), "superseded generation is discarded");
        // …and only the newest is accepted.
        assert!(tracker.accepts(second));
    }

    /// Several supersedes in a row: only the final generation is accepted; every
    /// intermediate one is stale (the drain-to-latest + newest-wins contract).
    #[test]
    fn only_final_generation_accepted_after_burst() {
        let mut tracker = GenerationTracker::new();
        let mut last = 0;
        for _ in 0..5 {
            last = tracker.next_generation();
        }
        assert_eq!(last, 5);
        for stale in 1..last {
            assert!(!tracker.accepts(stale), "generation {stale} is superseded");
        }
        assert!(tracker.accepts(last), "only the newest generation wins");
        assert_eq!(tracker.latest_dispatched(), 5);
    }

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
}
