//! The resolved voxel grid and the producers that fill it.
//!
//! ## Coordinate convention (PROJECT-WIDE — Z-up, right-handed)
//!
//! **Vertical / up = +Z** ([`glam::Vec3::Z`], array index **2**) EVERYWHERE in this
//! crate — camera, SDFs, fog, layers, diameter, mesh and `.vox` export all agree.
//! The ground plane is **XY** (normal +Z); **front = −Y** (the front view looks
//! along +Y); LEFT/RIGHT = ±X; TOP/BOTTOM = ±Z. Panel X/Y/Z fields map directly to
//! indices 0/1/2 with Z genuinely the vertical axis — no relabel shim.
//!
//! Consequences pinned by tests: a tall cylinder/tube/torus has its axis along Z
//! (`size_voxels[2]` is the vertical extent), layer slices are Z-slices, the onion
//! fog band is a Z-range, and the `.vox` export writes our Z straight to vox-Z with
//! NO axis swap (MagicaVoxel is itself Z-up).
//!
//! This module implements the architectural seam required by `REPRESENTATION.md`:
//! **the renderer never calls the SDF directly.** Instead a [`VoxelProducer`]
//! resolves a parametric shape (or, in a later milestone, a sculpt overlay) into
//! a [`VoxelGrid`] — the one consumed truth. The renderer, the layer-range
//! diameter readout (issue #12) and the `.vox` export (M8) all read the grid, so
//! adding a second producer later touches nothing downstream.
//!
//! Milestone 2 has exactly one producer: [`SdfShape`], which runs the sampling
//! triple-loop transcribed from `ARCHITECTURE.md` §1/§2 and writes occupied
//! voxels into the grid.
//!
//! ## The value ⊥ producer split (ADR 0016)
//!
//! The module is split along the future crate seam: [`value`] is the foundational
//! voxel-value layer (the resolved cell, its sparse grid, the frame-bearing
//! recentre, the primitive-kind tag, and the pure signed-distance functions) bound
//! for `voxel_core`; [`producer`] is the document-bound half (the
//! [`VoxelProducer`] trait and its [`SdfShape`] implementor) bound for `document`.
//! `producer` depends downward on `value`; `value` never names anything in
//! `producer`. Both halves re-export here so every call site keeps its
//! `crate::voxel::…` path unchanged.

mod producer;
mod value;

pub use value::{
    chunk_extent_exceeds_bound, signed_distance, signed_distance_box, signed_distance_ellipsoid,
    signed_distance_elliptical_cylinder, widest_run_in_band_over_chunks, BlockAttrs, BlockId,
    RecentreVoxels, ShapeKind, Voxel, VoxelGrid, MAX_CHUNK_VOXELS, MAX_GRID_VOXELS,
    SURFACE_ISOLEVEL,
};

pub use producer::{FieldClassification, FieldInterval, GeometryParams, SdfShape, VoxelProducer};

pub(crate) use producer::clamp_window_to_grid;
