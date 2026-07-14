//! Domain seam for view-frustum culling (see the viewing/projection chapter of the
//! architecture docs).
//!
//! The Gribb–Hartmann plane extraction and the positive-vertex AABB culling test now
//! live in the wgpu-free `camera` crate (`crates/camera`, the `frustum` module); the
//! f32 box they operate on is substrate's closed continuous [`substrate::RealAabb`],
//! co-located there beside its half-open integer twin `LatticeAabb`. This module is
//! the thin re-export — with `RealAabb` under its historical domain name `Aabb` — so
//! the chunk renderer keeps importing `crate::frustum::{Aabb, Frustum}` unchanged.

pub use ::camera::frustum::*;
/// The historical domain name for the closed continuous box the frustum test
/// consumes ([`substrate::RealAabb`]).
pub use substrate::RealAabb as Aabb;
