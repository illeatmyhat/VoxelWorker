//! Leaf producers and resolution: the [`VoxelBody`] / [`NodeContent`] leaf kinds, the
//! tree walk that composes placed leaves, the monolithic and chunk-scoped resolve
//! paths (region resolve is a test/oracle-gated oracle), and the per-leaf stamp
//! helpers that write a producer's voxels into an output grid or chunk.

mod model;
mod walk;
mod scope_fold;
mod gather;
mod resolve_chunk;
#[cfg(any(test, feature = "oracle"))]
mod resolve_oracle;

pub use model::{
    operation_masks_beyond_bounds, quat_from_lattice, LeafProducer, NodeContent, ScopeFrame,
    VoxelBody,
};
pub(crate) use model::{
    leaf_content_fingerprint, outset_voxels_at, ComposedScope, LeafBody, LeafVisitor,
};
