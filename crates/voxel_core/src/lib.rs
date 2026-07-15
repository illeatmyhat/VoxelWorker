//! # voxel_core — the foundational value vocabulary
//!
//! This crate holds the plain domain *values* every higher layer of the planner is
//! phrased in: the resolved voxel cell and its sparse grid, the frame-bearing
//! recentre, the primitive-kind tag and the pure signed-distance functions the
//! producers sample, the categorical block-palette id with its cell-key codec and
//! procedural-material choice, the absolute-voxel box and the flat leaf spatial
//! index, and the parametric blocks/voxels measurement core. It is the first
//! per-layer crate above `substrate` and below everything else in the application.
//!
//! ## The boundary law
//!
//! A component belongs here if and only if it is describable as a domain **value** —
//! a measurement, a piece of geometry, a coordinate frame, a categorical id — without
//! naming any *evaluation*, *display*, *wgpu*, or *scene* type. It is vocabulary, not
//! behaviour: nothing here evaluates the operation stack, resolves a chunk, meshes a
//! surface, or touches a GPU. Anything that must name a scene, a producer, a chunk, or
//! a store is a higher-layer concern and stays out. The dependency edge is one-way:
//! `substrate ← voxel_core ← the rest of the app`. The only dependencies are `glam`
//! (the algebra the geometry is written in), `serde` (persistence derives), and
//! `substrate`, whose pure CS/math structures this crate names at the domain seam — a
//! half-open integer box becomes a [`spatial_index::VoxelAabb`], an exact
//! [`substrate::interval::Rational`] becomes the block term of a [`units::Measurement`].
//! It never depends on the sibling `camera` or `raycast` crates.
//!
//! ## The chapter it serves
//!
//! These values are the nouns the architecture's document layer is written in — see
//! `docs/architecture/01-document.md` (producers, materials, units) for the timeless
//! statement, and `docs/design/per-layer-crates-extraction-map.md` (the voxel_core
//! row) for the dated provenance of each module.
//!
//! ## Modules
//!
//! * [`core_geom`] — the dependency-free geometry primitives: the [`core_geom::CHUNK_BLOCKS`]
//!   streaming quantum, the [`core_geom::MaterialChoice`] procedural palette, the
//!   categorical [`core_geom::BlockId`] with its [`core_geom::BlockAttrs`] and the
//!   render-side [`core_geom::CellKey`] codec.
//! * [`voxel`] — the resolved-cell value layer: the [`voxel::Voxel`] and its sparse
//!   [`voxel::VoxelGrid`], the frame-bearing [`voxel::RecentreVoxels`], the
//!   [`voxel::ShapeKind`] primitive tag, and the pure signed-distance functions
//!   ([`voxel::signed_distance`] and its per-kind arms).
//! * [`spatial_index`] — the [`spatial_index::VoxelAabb`] absolute-voxel box and its
//!   chunk-coverage reading, the [`spatial_index::EditBroadphaseBvh`] alias, and the
//!   flat [`spatial_index::LeafSpatialIndex`] over a scene's leaf world-AABBs.
//! * [`units`] — the parametric blocks/voxels [`units::Measurement`] core with its
//!   strict parser and formatter over an exact [`units::ExactRational`].

pub mod core_geom;
pub mod spatial_index;
pub mod units;
pub mod voxel;
