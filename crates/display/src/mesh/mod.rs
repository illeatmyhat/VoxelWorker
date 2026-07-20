//! Cuboid mesh render path (ADR 0002 E3b-1, part of #18). This module started as a
//! flagged alternative to the instanced per-voxel-cube renderer
//! (`crate::renderer::VoxelRenderer`, one cube per occupied voxel); that instanced
//! path and its flag were later removed with the legacy mesher (#20), and this
//! Vintage-Story-style **cuboid mesher** is now the sole mesh render path (the brick
//! raymarch, ADR 0011, is the primary display sink — this path is its A/B parity
//! oracle and understudy, both live). It decomposes the resolved grid into a
//! small set of single-material axis-aligned boxes ([`evaluation::cuboid`]) and builds
//! a triangle mesh of each box's **exposed faces only** (faces internal to the
//! solid set are culled). Each face vertex carries the box's `material_id` and a
//! face normal; the shader (`shaders/cuboid.wgsl`) flat-shades it with
//! normal-based lighting + per-material base-colour modulation.
//!
//! SCOPE (E3b-1): SHAPE parity + per-box material colour + basic lighting.
//! SCOPE (E3b-2): added the per-voxel TEXTURE SLICE (block texture tiled once per
//! voxel across a merged box face, via a voxel-unit UV + a Repeat sampler,
//! replicating the old instanced path's per-face UV direction so even non-symmetric
//! textures land texel-exact), the per-face D2Array layer selection from the face
//! normal, and the position-based per-voxel/per-block GRID OVERLAY. The layer-range
//! band clip (`build_cuboid_mesh_banded` below, issue #12) and debug-faces (this
//! crate's `pipeline` submodule) landed in later work.
//!
//! ## Geometry / coordinate mapping
//! A voxel at region-local index `l = (x, y, z)` occupies the world-space cell
//! `[world_offset + l, world_offset + l + 1]` per axis, so a box spanning voxels
//! `min..=max` becomes the world AABB `[world_offset + min, world_offset + max + 1]`
//! (`emit_box_faces`'s `world_offset` parameter). `world_offset` is NOT a fixed
//! `dimensions/2` centring: it is the cloud-anchored offset `region_from_voxel_cloud`
//! (in `builder.rs`) computes per grid, so the mesh lands exactly where the grid's own
//! `world_position` places that same voxel even when the composite is recentred off
//! its geometric centre — see that function's doc for why the old fixed-centre,
//! origin-at-0 assumption this module used to make was wrong.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;
use wgpu::util::DeviceExt;

use voxel_core::core_geom::{MaterialChoice, CHUNK_BLOCKS};
use evaluation::cuboid::{decompose_into_boxes, VoxelBox, VoxelBoxMaterial, VoxelRegion};
use substrate::solids::CulledBoxMeshing;
use camera::frustum::Frustum;
use substrate::spatial::RealAabb as Aabb;
use crate::renderer::{LayerBand, RegionClip, RegionRole, DEPTH_FORMAT, MSAA_SAMPLE_COUNT};
use crate::texture_atlas::MaterialAtlas;
use voxel_core::core_geom::CellKey;
use evaluation::two_layer_store::{MicroblockGeometry, SeamSolidity, TwoLayerChunk};
use voxel_core::voxel::{RecentreVoxels, VoxelGrid};

mod geometry;
mod builder;
mod two_layer;
mod emit;
mod pipeline;
mod selected_operand;
#[cfg(test)]
mod tests;

// Public API of the cuboid mesh path (ADR 0016 Phase 4b carve).
pub use builder::{
    build_cuboid_mesh, build_cuboid_mesh_banded, cuboid_incremental_plan, CuboidChunkMesh,
    CuboidMesh, CuboidRebuildPlan,
};
pub use pipeline::CuboidMeshRenderer;
pub use selected_operand::{SelectedOperandGhostBody, SelectedOperandGhostRenderer};

// Internal cross-submodule glue: each submodule reaches its siblings (and the shared
// imports above) through `use super::*`.
pub(crate) use builder::*;
pub(crate) use emit::*;
pub(crate) use geometry::*;
pub(crate) use pipeline::*;
pub(crate) use two_layer::*;
