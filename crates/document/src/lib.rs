//! # document — the authored TRUTH layer
//!
//! This crate holds everything the user's edits *produce* and nothing they *derive*:
//! the authored document. The scene graph — nodes, groups, instances, transforms, and
//! the combine operations that assemble them — is the spine; hanging off it are the
//! producers that turn authored intent into voxels ([`voxel::SdfShape`] over the
//! signed-distance primitives, the [`sketch::SketchSolid`] extrude/revolve, the
//! [`debug_clouds::DebugCloudField`] part), the 2D [`sketch::Sketch`] those solids are
//! authored from, the serializable [`intent::Intent`] that is the single mutation
//! boundary, and the inverse-[`command::Command`] stack behind undo/redo. It is the
//! second per-layer crate, above `voxel_core` and below everything else.
//!
//! ## The boundary law
//!
//! A component belongs here if and only if it is part of what the user **authors** —
//! expressible with no *evaluation*, *derivation*, *display*, or *wgpu* concept in
//! sight. The scene is the intent; resolving it into resident chunks, meshing it,
//! raymarching it, exporting it are all *downstream* concerns that name this truth but
//! are named by nothing in it. So the truth layer imports no evaluation, no display, no
//! wgpu: nothing here may name a store, a chunk cache, a two-layer store, a cuboid or
//! its mesh, a brick field, a texture atlas, a renderer, a worker, a panel, or a GPU
//! context. The dependency edge is one-way: `voxel_core ← document ← the rest of the
//! app`, compile-enforced — an upward `use` fails to build. The only dependencies are
//! `voxel_core` (the value vocabulary the producers speak), `substrate` (its
//! [`substrate::noise`] generators and CSG [`substrate::interval`] the producers name),
//! `glam` (the algebra), `serde` (persistence), and `rayon` (the parallel resolve).
//! It never depends on the sibling `camera` or `raycast` crates.
//!
//! The one deliberate exception is a *test-only* oracle: `Scene::resolve_region`
//! is a dense O(volume) whole-region resolver, the measuring stick the sparse runtime
//! path (which lives up in the evaluation layer) is held against. It is compile-gated
//! behind the `oracle` feature (and `cfg(test)`), so it is absent from production
//! builds and the truth layer stays behaviour-free at runtime.
//!
//! ## The chapter it serves
//!
//! These are the nouns and verbs of the architecture's document layer — see
//! `docs/architecture/01-document.md` (the scene graph, producers, sketches, intents,
//! commands) for the timeless statement, and
//! `docs/design/per-layer-crates-extraction-map.md` (the document row) for the dated
//! provenance of each module.
//!
//! ## Modules
//!
//! * [`scene`] — the authored scene graph: [`scene::Scene`], its [`scene::Node`] tree
//!   (tools, groups, instances, parts), [`scene::NodeTransform`] placement, the
//!   [`scene::CombineOp`] assembly, extent + placed-region queries, the flat-leaf
//!   spatial reading, and the producer resolve (with the dense oracle).
//! * [`voxel`] — the producer half of the old `voxel` module: the [`voxel::VoxelProducer`]
//!   trait, [`voxel::SdfShape`] and its [`voxel::GeometryParams`], and the
//!   conservative cell-field interval bound (re-exporting `voxel_core`'s value types).
//! * [`sketch`] — the 2D [`sketch::Sketch`] and the [`sketch::SketchSolid`] producer
//!   that extrudes or revolves it into a volume.
//! * [`debug_clouds`] — the [`debug_clouds::DebugCloudField`] procedural part producer.
//! * [`intent`] — the serializable [`intent::Intent`] mutation boundary (the single
//!   door every human / agent / GPU-brush edit passes through).
//! * [`command`] — the linear inverse-[`command::Command`] stack behind undo/redo.

// A public item's doc may link to a private helper to explain how the two relate; that
// cross-reference stays a navigable link under `--document-private-items`. The CI doc
// gate denies broken and redundant links but permits these.
#![allow(rustdoc::private_intra_doc_links)]

pub mod command;
pub mod debug_clouds;
pub mod intent;
pub mod scene;
pub mod sketch;
pub mod voxel;
