//! ADR 0011 — the **brick display path**: the brick-field BUILD, its clip-map pyramid,
//! block-occupancy masks and atlases, the GPU raymarch renderer, and the CPU reference
//! march, carved into one cohesive folder (ADR 0016 Phase 4c — the former `brick_field`
//! and `brick_raymarch` modules fused into `brick`).
//!
//! ## The build (`clipmap`, `occupancy`, `record`, `build`, `incremental`, `atlas`)
//!
//! ADR 0011 G0 packs ADR 0010's two-layer boundary set into a sorted [`BrickRecord`]
//! array (keyed by a packed absolute world-block key — the frame is world-fixed, ADR 0008)
//! plus an R8 3D texture atlas of sculpted-brick occupancy, **surface-only** (ADR 0011
//! interior elision):
//!
//! * **air block** → no record (the ray skips it via the clip-map).
//! * **coarse-solid block** → one [`BrickPayload::CoarseSolid`] record (no atlas slot),
//!   UNLESS fully occluded (all six face-neighbours present + solid), in which case it
//!   emits nothing — a ray can never reach it first.
//! * **boundary block** → one [`BrickPayload::Sculpted`] record whose atlas slot holds the
//!   block's voxel occupancy, rasterized from its cuboids; a MIXED block also owns a
//!   cell-key tile ([`BrickCellKeyTile`]) in the material side atlas.
//!
//! The **brick granule is ONE BLOCK** (ADR 0011 Decision 1): the brick edge is
//! `voxels_per_block`, correct at ANY density. The clip-map pyramid ([`ClipmapPyramid`],
//! L1–L3) derives from the CHUNKS (interiors included) so the ray can skip empty space;
//! [`BlockOccupancyMasks`] carry per-cell interior occupancy. The interior-inclusive build
//! ([`build_brick_field_all_blocks`]) survives as the parity oracle. [`IncrementalBrickField`]
//! maintains the record set + atlas across edits, emitting a [`BrickFieldUpdate`] naming only
//! the dirty slots so the GPU sink patches the minimum.
//!
//! ## The raymarch sink (`gpu_record`, `raymarch`, `cpu_march`)
//!
//! ADR 0011 G1's [`BrickRaymarchRenderer`] is a fullscreen pass that walks a block-space DDA
//! per pixel over the packed [`BrickGpuRecord`] set + atlas, clip-map skipping empty space:
//!
//! * **coarse** records hit as a solid block-cube; **sculpted** records descend to a voxel
//!   DDA over the brick's atlas slot; a lookup miss steps on (air).
//! * a sculpted record whose `atlas_slot` is [`NON_RESIDENT_ATLAS_SLOT`] renders its COARSE
//!   form (the residency-miss contract, ADR 0011 4a) — degraded-but-correct, never skipped.
//! * the pass runs INSIDE the shared 4× MSAA voxel pass and writes per-sample ray-hit depth,
//!   so the rasterized overlays composite exactly as over the mesh; shading transcribes
//!   `cuboid.wgsl` (parity gate clause (c)).
//!
//! Per ADR 0006 the sink is a **display derivation**: records + atlas are built from CPU
//! truth and nothing is read back as truth. The module also hosts the **CPU reference march**
//! ([`cpu_march_brick_field`], [`cpu_march_exact_occupancy`]) — an f32 mirror of the WGSL
//! traversal `tests/gpu_parity.rs` gates against the exact evaluator, and the CPU two-layer
//! mesh stays the headless/no-GPU fallback (ADR 0011 Decision 6).

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;
use wgpu::util::DeviceExt;

use voxel_core::core_geom::{BlockId, CellKey, MaterialChoice, CHUNK_BLOCKS};
use voxel_core::voxel::RecentreVoxels;
use evaluation::cuboid::VoxelBoxMaterial;
use evaluation::two_layer_store::{SeamSolidity, TwoLayerChunk};
use crate::renderer::{LayerBand, RegionClip, DEPTH_FORMAT, MSAA_SAMPLE_COUNT};

// The brick-record key codec IS substrate's `lattice_key`: an absolute world-block
// coordinate packed into one sortable `u64` in z-major lexicographic (z, y, x) order,
// 21 bits/axis, so the record array's integer order IS block order (sortable on the CPU,
// binary-searchable as a `(hi, lo)` u32 pair on the GPU). See
// docs/architecture/data-structures.md (Substrate) for the codec itself.
pub use substrate::spatial::lattice_key::{
    pack_lattice_key as pack_world_block_key, unpack_lattice_key as unpack_world_block_key,
};

// A boundary block's occupancy tile IS substrate's `BitCube` (edge-≤64, one `u64` per X-row);
// the sculpted-atlas scatter reuses substrate's `CubeTilePacking` (linear slot → cubic tile
// grid) and the per-slot store is a `SlotFreeList` (stable-index free-list). See
// docs/architecture/03-display.md (the brick-field atlas).
use substrate::occupancy::{CubeTilePacking, SlotFreeList};
pub use substrate::occupancy::BitCube as BrickOccupancyTile;

// A MIXED block's per-voxel cell-key tile IS substrate's `ValueCube<u16>` — the payload
// sibling of the occupancy `BitCube`, one `u16` per voxel; the occupancy tile gates it
// cell-for-cell and ONE rasterizing walk fills both. See docs/architecture/03-display.md.
pub use substrate::occupancy::ValueCube as ValueTile;

mod clipmap;
mod occupancy;
mod record;
mod build;
mod incremental;
mod atlas;
mod gpu_record;
mod raymarch;
mod cpu_march;
#[cfg(test)]
mod tests;

// The brick path's items, re-exported into `display::brick::*`. Public items surface as the
// crate's public API; the `pub(crate)` helpers stay crate-internal but resolve at this path.
// Sibling submodules reach each other through `use super::*` + these globs (ADR 0016 carve).
pub use clipmap::*;
pub use occupancy::*;
pub use record::*;
pub use build::*;
pub use incremental::*;
pub use atlas::*;
pub use gpu_record::*;
pub use raymarch::*;
pub use cpu_march::*;
