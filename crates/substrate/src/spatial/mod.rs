//! Spatial primitives and acceleration structures: axis-aligned boxes, the
//! bounding-volume hierarchy over them, the space-filling lattice-key codec, the
//! ray primitive with its slab-method box test, and the sparse min-mip occupancy
//! pyramid. Each module carries its own literature citations.

pub mod aabb;
pub mod bvh;
pub mod lattice_key;
pub mod min_mip_pyramid;
pub mod orientation;
pub mod ray;

pub use aabb::{LatticeAabb, RealAabb};
pub use bvh::Bvh;
pub use min_mip_pyramid::{MinMipLevel, SparseMinMipPyramid};
pub use orientation::LatticeOrientation;
pub use ray::{guarded_direction, Ray, RayBoxIntersection, SLAB_ZERO_DIRECTION_GUARD};
