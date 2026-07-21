//! # evaluation — the one evaluator (document → classified boundary set)
//!
//! This crate is the hinge of the whole system: everything above it is authored intent,
//! everything below it is presentation. It holds the **one evaluator** that turns the
//! document (a program) into occupancy (an answer) — and nothing consumes a producer
//! anywhere else. The mesh, the brick field, the exporter, and the measurement queries
//! all read this crate's output; none of them evaluate the scene themselves. That is Law
//! 6 ("classified once, consumed everywhere"): if the evaluator is right, every sink is
//! right, and two sinks can never drift apart because neither has an opinion of its own.
//!
//! ## The boundary law
//!
//! A component belongs here if and only if it **consumes the document and produces the
//! classified boundary set** — expressible with no *display*, *work*, or *shell* concept
//! in sight. Meshing, raymarching, exporting, and the frame loop are all *downstream*
//! concerns that name this crate's chunks but are named by nothing in it. So the
//! evaluator imports no display, no work, no shell: nothing here may name a renderer, a
//! cuboid mesh, a brick field, a texture atlas, a block palette, an asset, the exporter,
//! a worker, the app core, a panel, settings, or a GPU context. The dependency edge is
//! one-way: `voxel_core·document ← evaluation ← {display, interchange}`, compile-enforced
//! — an upward `use` fails to build. The only dependencies are `document` (the scene +
//! producers it resolves), `voxel_core` (the value vocabulary), `substrate` (the CSG
//! interval arithmetic + cell-classification kernel + cuboid decomposition), `rayon`
//! (the parallel per-chunk build), and `serde`/`serde_json` (the persisted chunk codec).
//! It never depends on the sibling `camera` or `raycast` crates.
//!
//! The one deliberate exception is a *test-only* pair of oracles: `Store::resolve_region`
//! and `two_layer_store::resolve_region_two_layer` are dense O(volume) whole-region
//! resolvers, the measuring sticks the sparse runtime path is held against. They are
//! compile-gated behind the `oracle` feature (and `cfg(test)`), so they are absent from
//! production builds and the evaluator stays memory-follows-the-surface at runtime.
//!
//! ## The chapter it serves
//!
//! These are the nouns and verbs of the architecture's evaluation layer — see
//! `docs/architecture/02-evaluation.md` (block classification by interval bound, the
//! two-layer chunk, residency and targeted invalidation, frames and the floating origin)
//! for the timeless statement, and
//! `docs/design/per-layer-crates-extraction-map.md` (the evaluation row) for the dated
//! provenance of each module.
//!
//! ## Modules
//!
//! * [`store`] — the residency + per-chunk resolve cache ([`store::Store`], aliased
//!   [`store::ChunkResolveCache`]): lazy per-chunk resolve, whole-chunk targeted
//!   invalidation, out-of-core spill, the dense whole-region resolve oracle, and the
//!   GPU-free incremental rebuild planner.
//! * [`two_layer_store`] — the boundary-aware two-layer chunk ([`two_layer_store::TwoLayerChunk`]:
//!   coarse block-ID grid + sparse microblock cuboids + seam flags), the interval-bound
//!   block classifier, the resident cache, and the cacheless export/measure streams.
//! * [`chunk_cache`] — the thin re-export shim keeping historical `chunk_cache::*` call
//!   sites resolving at [`store`].
//! * [`chunk_storage`] — the sparse per-chunk codec (compress/decompress, `CompressedChunk`).
//! * [`disk_chunk_store`] — the out-of-core spill store backing the resolve cache.
//! * [`cuboid`] — the greedy boundary cuboid decomposition (`VoxelBox`, `VoxelRegion`).

// A public item's doc may link to a private helper to explain how the two relate; that
// cross-reference stays a navigable link under `--document-private-items`. The CI doc
// gate denies broken and redundant links but permits these.
#![allow(rustdoc::private_intra_doc_links)]

pub mod chunk_cache;
pub mod chunk_storage;
pub mod cuboid;
pub mod disk_chunk_store;
pub mod store;
pub mod two_layer_store;

/// The composed-field point-eval (ADR 0027 §5): the scene's composed signed distance at a
/// world point, for the CPU continuous-placement surface-raycast. Re-exported at the crate
/// root so the app crate calls `evaluation::composed_field_at`.
pub use two_layer_store::composed_field_at;
