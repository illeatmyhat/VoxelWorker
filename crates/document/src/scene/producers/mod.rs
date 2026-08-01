//! Leaf producers and resolution: the [`VoxelBody`] / [`NodeContent`] leaf kinds, the
//! tree walk that composes placed leaves, the monolithic and chunk-scoped resolve
//! paths (region resolve is a test/oracle-gated oracle), and the per-leaf stamp
//! helpers that write a producer's voxels into an output grid or chunk.

#![allow(clippy::redundant_pub_crate)]

mod gather;
mod model;
mod pick;
mod resolve_chunk;
#[cfg(any(test, feature = "oracle"))]
mod resolve_oracle;
mod scope_fold;
mod walk;

pub use model::{
    leaf_content_fingerprint, outset_voxels_at, AccumulatedOffset, ComposedScope, LeafBody,
    LeafVisitor, VisitedLeaf,
};
pub use model::{
    operation_masks_beyond_bounds, quat_from_lattice, LeafProducer, NodeContent, ScopeFrame,
    VoxelBody,
};
