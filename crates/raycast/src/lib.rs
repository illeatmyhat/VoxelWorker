//! # raycast — ray–volume traversal (wgpu-free)
//!
//! This crate holds the graphics **mathematics** of casting a ray through a sparse
//! voxel/brick volume: the [`VoxelDda`] stepping loop (Amanatides & Woo), the
//! hierarchical empty-space skip over a min-mip occupancy pyramid (GigaVoxels / VDB
//! lineage), the composed [`march_brick_hierarchy`] driver that threads a slab entry,
//! a block-scale DDA with the hierarchical skip, and a per-block descent to an inner
//! voxel-scale DDA, and the independent [`march_exact_occupancy`] reference march. It
//! also holds the pure [`pick_view_cube_slab`] element picker. It is a sibling of
//! `substrate` and `camera` — a read-first library of well-known concepts under their
//! literature names — but graphics-specific, and the consuming end of the ray a
//! [`substrate::Ray`] carries (a `camera` unprojection produces one, this crate marches
//! it).
//!
//! ## The graphics-crate boundary law
//!
//! A component belongs here if and only if it is describable entirely in the
//! vocabulary of *ray–volume traversal* — a parametric ray, a cell lattice, a DDA
//! step, an occupancy query — parameterised only by plain numbers, `glam` vectors, and
//! **injected occupancy closures**, never by wgpu, winit, egui, or any domain type (a
//! record, an atlas byte, a residency policy). The dependency edge is one-way:
//! `substrate ← raycast ← the application`. The only non-`glam` dependency is
//! `substrate`, for the shared [`substrate::Ray`] / [`substrate::RealAabb`] geometry
//! and the [`substrate::min_mip_pyramid`] cell-key search the hierarchical skip folds
//! against; this crate never depends on the sibling `camera` crate.
//!
//! The domain's brick march (`voxel_worker::brick_raymarch`) is the adapter: it holds
//! the carried march frame, the record binary search + atlas byte fetch, and the
//! empty-level occupancy policy, and it builds those into the closures this kernel
//! consumes. The WGSL shader `shaders/brick_raymarch.wgsl` is a maintained *mirror* of
//! this kernel — this crate is the readable specification of that shader — and the
//! application's `gpu_parity` suite is the mechanical link that keeps the two
//! byte-identical: the CPU march must produce identical hit voxels, entry-face
//! normals, and block-step counts through this kernel as the GPU shader does.
//!
//! ## Naming rule and citations
//!
//! Each module names the concept it implements and cites the canonical literature in
//! its own module documentation (definition-of-done):
//!
//! * [`voxel_dda`] — the Amanatides & Woo (1987) fast voxel traversal stepping loop.
//! * [`brick_march`] — the composed hierarchical march (Kay–Kajiya slab entry; the
//!   Crassin et al. 2009 / Museth 2013 hierarchical empty-space skip; the per-block
//!   inner voxel DDA) and the flat exact-occupancy reference march.
//! * [`view_cube_pick`] — the ViewCube element picker's ray-slab-with-entry-axis test.

pub mod brick_march;
pub mod view_cube_pick;
pub mod voxel_dda;

pub use brick_march::{
    entry_face_normal, march_brick_hierarchy, march_exact_occupancy, BlockContents,
    ExactMarchParams, HierarchicalMarchParams, MarchHit,
};
pub use view_cube_pick::{
    pick_view_cube_slab, view_cube_hot_zone_neighbours, ViewCubeSlabHit, VIEW_CUBE_HALF_EXTENT,
    VIEW_CUBE_ZONE_THRESHOLD,
};
pub use voxel_dda::VoxelDda;
