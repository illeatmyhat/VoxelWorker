//! Residency + per-chunk resolve + bound-region read primitive — the true cache job.
//!
//! ADR 0003 (foundation rework) data layer. `store` owns the chunked sparse
//! absolute-i64 block store: residency of resolved render chunks, per-chunk
//! resolve, and the `bind_region` read primitive that consumer-shaped queries
//! (`resolve_region`, `resident_render_chunks`, `widest_run_in_band`, export)
//! are thin wrappers over.
//!
//! Currently an empty placeholder created in slice **A2a**. `ChunkResolveCache`
//! relocates here in **A2b** (rename → `store::Store`), the consumer wrappers
//! split off in **A2c**, and resolve state + the `rebuild` body land via
//! `AppCore::rebuild` in **A2e**.
