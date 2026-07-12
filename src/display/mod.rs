//! The display subsystem — the policy that decides how the two display pipelines (the
//! cuboid fallback mesh and the ADR 0011 brick raymarch) turn each edit into fresh
//! on-screen geometry.
//!
//! This subsystem owns the *decisions*, not the mechanics. The pure per-edit routing
//! functions — where an edit's derived display artifacts are (re)built (inline on the
//! main thread vs dispatched to a background worker, stale-while-rebuilding) and how a
//! stale artifact is guarded against an unsound inline patch — live in [`routing`]. The
//! state machine that ACTS on those decisions (the `AppState` rebuild path) still lives
//! in `main.rs`; a later slice extracts a DisplayOrchestrator into this module to join
//! the routing policy. The async workers that EXECUTE a dispatched rebuild remain in
//! `crate::geometry_worker` and `crate::brick_worker` (they hold the GPU handles and the
//! build closures); this subsystem only tells the shell which of them to feed.

pub mod routing;
