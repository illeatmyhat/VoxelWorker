//! Headless interchange: serialise the evaluated boundary set to external formats.
//!
//! This crate is the **export sink** — it takes the evaluation layer's classified two-layer
//! chunks and writes them to formats other tools ingest. Today that is one writer: the
//! MagicaVoxel [`vox_export`] `.vox` serialiser the Vintage Story mod reads.
//!
//! ## The law: interchange is headless — it never links wgpu
//!
//! Export is CPU-owned truth. Under the architecture's fourth law — *the CPU owns truth; the
//! GPU owns the frame* — evaluation, classification, **export**, and measurement are CPU code,
//! correct without any GPU present (`docs/architecture/README.md`). Interchange therefore names
//! no wgpu device, queue, pipeline, or shader, and carries no dependency that could pull one in.
//! That is exactly what earns it its own crate rather than a module inside `display`: a
//! serialiser here *cannot* accidentally depend on a GPU device, because the crate cannot name
//! one. A headless build exports the same bytes a windowed build does.
//!
//! It reads the SAME classified set every other sink reads — the sixth law, *classified once,
//! consumed everywhere*: the mesh, the brick field, and this exporter all consume one evaluator's
//! output, so the export can never disagree with what is drawn. See the Derivations layer in
//! `docs/architecture/README.md` and `docs/design/per-layer-crates-extraction-map.md` (the
//! interchange row) for provenance.
//!
//! The dependency edge is one-way — `evaluation <- interchange <- {work, shell}` — and
//! compile-enforced: interchange imports no display / work / shell / wgpu type.

// A public item's doc may link to a private helper to explain how the two relate; that
// cross-reference is deliberate and stays a navigable link under `--document-private-items`.
// The CI doc gate denies broken and redundant links but permits these.
#![allow(rustdoc::private_intra_doc_links)]

pub mod vox_export;
