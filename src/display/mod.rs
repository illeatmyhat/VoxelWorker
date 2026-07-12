//! The display subsystem — the policy that decides how the two display pipelines (the
//! cuboid fallback mesh and the ADR 0011 brick raymarch) turn each edit into fresh
//! on-screen geometry.
//!
//! This subsystem owns the *decisions* AND the state machine that acts on them. The pure
//! per-edit routing functions — where an edit's derived display artifacts are (re)built
//! (inline on the main thread vs dispatched to a background worker, stale-while-rebuilding)
//! and how a stale artifact is guarded against an unsound inline patch — live in [`routing`].
//! The [`DisplayOrchestrator`](orchestrator::DisplayOrchestrator) in [`orchestrator`] owns
//! both display renderers, both async workers, and all the per-edit display bookkeeping that
//! acts on the routing decisions; it is constructible without a window, so the state machine
//! — not just its pure fragments — is unit-testable. The winit shell keeps input, surface,
//! egui, and camera, and calls the orchestrator at its (few) integration points. The async
//! workers that EXECUTE a dispatched rebuild remain in `crate::workers::geometry` and
//! `crate::workers::brick` (they hold the GPU handles and the build closures).

pub mod orchestrator;
pub mod routing;
