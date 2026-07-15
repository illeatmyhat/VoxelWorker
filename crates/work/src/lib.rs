//! The WORK layer — the tempo that keeps the shell at 60 Hz (architecture chapter 04,
//! `docs/architecture/04-work.md`; law 7, "the interface never waits").
//!
//! This crate COORDINATES the derivation pipeline — evaluation -> display -> pixels — via a pool
//! of async workers and the engagement state machine that acts on their results:
//!
//!   * [`workers`] — the generic drain-to-latest/supersede [`Worker`](workers::Worker) (the
//!     substrate `supersede` protocol) plus the domain workers that run each heavy rebuild
//!     off-thread: the wholesale cuboid-mesh build ([`geometry`](workers::geometry)), the
//!     wholesale brick-pipeline build ([`brick`](workers::brick)), the diameter / widest-run
//!     measurement ([`diameter`](workers::diameter)), the `.vox` export
//!     ([`export`](workers::export)), and the one-shot streaming face scan ([`scan`](workers::scan)).
//!   * [`engagement`] — the pure per-edit routing policy ([`routing`](engagement::routing):
//!     inline-vs-async, the stale-artifact interlock, generation bookkeeping) and the
//!     [`DisplayOrchestrator`](engagement::orchestrator::DisplayOrchestrator) that owns both
//!     display renderers + both async workers and acts on the routing decisions. The
//!     orchestrator is constructible without a window, so the whole state machine — not just its
//!     pure fragments — is unit-testable.
//!
//! ## The layer law
//! Data flows downward only. This crate sits ABOVE the display + interchange sinks and BELOW the
//! shell: `{display, interchange} <- work <- shell`. It imports NO shell (`app_core`, `panel`,
//! `settings`, `gpu`, `main`) — the upward edge is compile-enforced out.
//!
//! ## Why this crate links wgpu
//! Deliberately. The geometry / brick workers build GPU meshes + brick fields off-thread, and the
//! orchestrator owns the device (handed in from the shell) and drives the display renderers. So
//! `display` is the only crate that links wgpu FOR RENDERING, while `work` links it to BUILD and
//! DRIVE GPU resources; the shell owns the device. That is not a boundary violation — it is the
//! work layer doing its job (law 4: "the CPU owns truth; the GPU owns the frame" — the frame is
//! driven here, off the event-loop thread).

// A public item's doc may link to a private helper to explain how the two relate; that
// cross-reference is deliberate and stays a navigable link under `--document-private-items`.
// The CI doc gate denies broken and redundant links but permits these.
#![allow(rustdoc::private_intra_doc_links)]

pub mod engagement;
pub mod workers;
