//! Cuboid mesh render path (ADR 0002 E3b-1, part of #18) — BEHIND A FLAG.
//!
//! The instanced renderer (`crate::renderer::VoxelRenderer`) draws one cube
//! per occupied voxel. This module is the FIRST step of replacing that with a
//! Vintage-Story-style **cuboid mesher**: it decomposes the resolved grid into a
//! small set of single-material axis-aligned boxes ([`evaluation::cuboid`]) and builds
//! a triangle mesh of each box's **exposed faces only** (faces internal to the
//! solid set are culled). Each face vertex carries the box's `material_id` and a
//! face normal; the shader (`shaders/cuboid.wgsl`) flat-shades it with the same
//! normal-based lighting + per-material base-colour modulation the instanced
//! path uses.
//!
//! SCOPE (E3b-1): SHAPE parity + per-box material colour + basic lighting.
//! SCOPE (E3b-2, this sub-step): adds the per-voxel TEXTURE SLICE (block texture
//! tiled once per voxel across a merged box face, via a voxel-unit UV + a Repeat
//! sampler, replicating the instanced per-face UV direction so even non-symmetric
//! textures land texel-exact), the per-face D2Array layer selection from the face
//! normal, and the position-based per-voxel/per-block GRID OVERLAY — all matching
//! the instanced path. STILL NO layer-range clip, NO debug-faces (later E3 sub-
//! steps). The instanced path stays the DEFAULT and is untouched; this path is
//! selected only when the `cuboid` mesher flag is on.
//!
//! ## Geometry / coordinate mapping
//! A voxel at region-local index `(x, y, z)` occupies the world-space cell
//! `[i - half, i+1 - half]` per axis, where `i` is the ABSOLUTE voxel index and
//! `half = dimensions / 2`. This matches the instanced path, where a voxel cube
//! is centred at `world_position = i + 0.5 - half` and spans centre ± 0.5. Since
//! we decompose the whole grid with `origin = [0,0,0]`, the region-local index IS
//! the absolute index, so a box spanning voxels `min..=max` becomes the world AABB
//! `[min - half, (max+1) - half]`.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;
use wgpu::util::DeviceExt;

use voxel_core::core_geom::{MaterialChoice, CHUNK_BLOCKS};
use evaluation::cuboid::{decompose_into_boxes, VoxelBox, VoxelBoxMaterial, VoxelRegion};
use substrate::solids::CulledBoxMeshing;
use camera::frustum::Frustum;
use substrate::spatial::RealAabb as Aabb;
use crate::renderer::{LayerBand, DEPTH_FORMAT, MSAA_SAMPLE_COUNT};
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
