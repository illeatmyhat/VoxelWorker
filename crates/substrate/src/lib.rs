//! # substrate — the pure computer-science / mathematics library
//!
//! This crate holds the load-bearing data structures whose identity is purely
//! *algorithmic*, split out of the domain so they can be identified, read, and
//! reasoned about (including their performance) in isolation. It is not intended
//! for release; it is intended for reading. The application crate depends on
//! substrate; substrate depends on no domain code, and that direction is
//! compile-enforced by the crate boundary.
//!
//! See the **Substrate** section of `docs/architecture/data-structures.md` for
//! the timeless statement of the same rules; the dated component inventory and
//! slice order live in `docs/design/substrate-extraction-map.md`.
//!
//! ## The boundary law
//!
//! A component belongs in this crate if and only if it is describable *entirely*
//! in textbook computer-science / mathematics vocabulary — a bounding-volume
//! hierarchy, an axis-aligned box, a bit-packed occupancy cube, interval
//! arithmetic, a min-mip pyramid, a slot allocator, a space-filling key codec, a
//! rational, a supersede protocol — and is parameterized only by plain numbers
//! and generics, **never by domain types**. Anything that must name a scene, a
//! producer, a chunk, or a brick-as-block is a domain adapter and stays in the
//! application crate at its own seam.
//!
//! ## Naming rule
//!
//! Each component lives in its own module, and the well-known name from the
//! scientific literature *is* the type's name (`MedianSplitBvh`, `IntegerAabb`,
//! `BitCube`, `DisjointIntervalSet`, `ExactRational`, …). The explanation of the
//! structure and the citations to the canonical literature — together with a note
//! on how this implementation's variant deviates — live in the component's own
//! definition, not here. Domain vocabulary survives only at the adapter seams in
//! the application crate.
//!
//! ## Benches
//!
//! Criterion microbenches (`crates/substrate/benches/`) exist for the *hot*
//! components only, and are run on demand — never part of the commit gates.
//!
//! ---
//!
//! Components arrive one extraction slice at a time, each carrying its own oracles,
//! per the extraction map referenced above. Extracted so far (slice S1, spatial):
//! [`Aabb`], [`Bvh`], and the [`lattice_key`] packing codec.

pub mod aabb;
pub mod bvh;
pub mod lattice_key;

pub use aabb::Aabb;
pub use bvh::Bvh;
