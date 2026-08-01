//! The **cuboid mesher**.
//!
//! This is the sole mesh render path. It decomposes the resolved grid
//! into a small set of single-material axis-aligned boxes ([`evaluation::cuboid`]) and
//! builds a triangle mesh of each box's **exposed faces only** (faces internal to the
//! solid set are culled). Each face vertex carries the box's `material_id` and a face
//! normal; the shader (`shaders/cuboid.wgsl`) flat-shades it with normal-based lighting +
//! per-material base-color modulation, tiles the block texture once per voxel across a
//! merged box face (a voxel-unit UV + a `Repeat` sampler), selects the per-face `D2Array`
//! layer from the face normal, and draws the position-based per-voxel/per-block GRID
//! OVERLAY. A layer-range band clip (`build_cuboid_mesh_banded`) and a debug-faces mode
//! ride on the same path.
//!
//! The brick raymarch is the primary display sink; this path is its A/B parity oracle and
//! the no-GPU-capable understudy, both live.
//!
//! ## Geometry / coordinate mapping
//! A voxel at region-local index `l = (x, y, z)` occupies the world-space cell
//! `[world_offset + l, world_offset + l + 1]` per axis, so a box spanning voxels
//! `min..=max` becomes the world AABB `[world_offset + min, world_offset + max + 1]`
//! (`emit_box_faces`'s `world_offset` parameter). `world_offset` is NOT a fixed
//! `dimensions/2` centering: it is the cloud-anchored offset `region_from_voxel_cloud`
//! (in `builder.rs`) computes per grid, so the mesh lands exactly where the grid's own
//! `world_position` places that same voxel even when the composite is recentered off
//! its geometric center — see that function's doc for why a fixed-center, origin-at-0
//! assumption is wrong.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;
use wgpu::util::DeviceExt;

use crate::renderer::{LayerBand, RegionClip, RegionRole, DEPTH_FORMAT, MSAA_SAMPLE_COUNT};
use crate::texture_atlas::MaterialAtlas;
use camera::frustum::Frustum;
use evaluation::cuboid::{decompose_into_boxes, VoxelBox, VoxelBoxMaterial, VoxelRegion};
use evaluation::two_layer_store::{MicroblockGeometry, SeamSolidity, TwoLayerChunk};
use substrate::solids::CulledBoxMeshing;
use substrate::spatial::RealAabb as Aabb;
use voxel_core::core_geom::CellKey;
use voxel_core::core_geom::{MaterialChoice, CHUNK_BLOCKS};
use voxel_core::voxel::{RecenterVoxels, VoxelGrid};

mod builder;
mod emit;
mod geometry;
mod pipeline;
mod selected_operand;
mod selection_outline;
#[cfg(test)]
mod tests;
mod two_layer;

// Public API of the cuboid mesh path.
pub use builder::{
    build_cuboid_mesh, build_cuboid_mesh_banded, cuboid_incremental_plan, CuboidChunkMesh,
    CuboidMesh, CuboidRebuildPlan,
};
pub use pipeline::CuboidMeshRenderer;
pub use selected_operand::{SelectedOperandGhostBody, SelectedOperandGhostRenderer};
pub use selection_outline::{SelectedBodyChunks, SelectionOutlineRenderer};

// Internal cross-submodule glue: each submodule reaches its siblings (and the shared
// imports above) through `use super::*`.
pub(crate) use builder::*;
pub(crate) use emit::*;
pub(crate) use geometry::*;
pub(crate) use pipeline::*;
pub(crate) use two_layer::*;
