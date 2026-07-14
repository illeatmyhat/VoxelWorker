//! The generic background worker behind the async display pipelines.
//!
//! Three heavy rebuilds — the wholesale cuboid-mesh build (geometry), the diameter /
//! widest-run measurement, and the wholesale brick-pipeline build — each run the SAME
//! machine: a dedicated thread that blocks on a request channel, drains a burst of
//! queued requests down to the newest, builds one result, and ships it back. Doing any
//! of these inline on the event-loop thread freezes the UI, so each moves onto a
//! [`Worker`]: the shell keeps drawing the CURRENT (stale) artifact until the freshly
//! built one arrives (stale-while-rebuilding), then swaps it in.
//!
//! Each domain module supplies only its request/result types and a build closure. The
//! generic drain-to-latest/supersede machinery itself is the substrate
//! [`supersede`](substrate::supersede) protocol — [`Worker`] is
//! [`substrate::CoalescingWorker`] and [`build_catching`] is
//! [`substrate::catch_unwind_or_log`], re-exported here so the worker call sites stay put.
//! The supersede/interlock contracts that decide WHICH rebuild is dispatched, and whether
//! an arriving result is accepted, live with the domains (their `route_*` decisions +
//! [`crate::display::routing::GenerationTracker`]). See `docs/architecture/04-work.md` (the
//! work chapter) for the seam.
//!
//! The domain workers themselves live in the submodules: [`brick`], [`diameter`],
//! [`export`], [`geometry`], and the one-shot streaming [`scan`]. The [`export`] worker
//! diverges from the supersede contract — a `.vox` is a user-chosen file, so the shell
//! serialises exports rather than draining to the latest (see its module doc).

pub mod brick;
pub mod diameter;
pub mod export;
pub mod geometry;
pub mod scan;

// The generic drain-to-latest/supersede worker is the substrate `supersede` protocol; the
// domain keeps its `Worker` / `build_catching` vocabulary at this seam so the worker call
// sites stay put. See docs/architecture/04-work.md (the work chapter).
pub use substrate::CoalescingWorker as Worker;
pub(crate) use substrate::catch_unwind_or_log as build_catching;
