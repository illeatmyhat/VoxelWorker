//! Shared render infrastructure for the voxel workshop.
//!
//! The voxel grid itself is drawn by the cuboid mesh path
//! ([`crate::mesh::CuboidMeshRenderer`]); the legacy instanced cube renderer
//! that once lived here was removed (part of #20). This module now owns the SHARED
//! GPU pieces that path (and the rest of the app) builds on:
//!   * The procedural material textures (Stone/Wood/Plain) + the loaded-VS-block
//!     material bind-group layout ([`build_face_material_layout`]) and helpers.
//!   * The position-based grid-overlay parameters ([`grid_overlay_params`]).
//!   * The per-object lattice/floor grid ([`SceneGridRenderer`]), the transform
//!     gizmo, and the view cube.
//!   * The MSAA/depth view helpers.
//!
//! Everything here is render-target-agnostic, so the window and the headless
//! capture paint identically.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use voxel_core::core_geom::MaterialChoice;
use document::scene::{Point, Scene};
use voxel_core::voxel::RecentreVoxels;
// The sRGB↔linear transfer function is textbook math with no domain content, so it
// lives in substrate (see the material/colour handling in docs/architecture); the
// call sites below keep their names via this import.
use substrate::srgb::{srgb_component_to_linear, srgb_hex_to_linear};

mod materials;
mod view_cube;
mod chrome;
mod targets;
mod lines;
mod gizmo;
mod grid;
mod points;
mod infinite_grid;
mod onion;
#[cfg(test)]
mod tests;

// Public API of the shared render infrastructure (ADR 0016 Phase 4d carve). Every
// `crate::renderer::…` / `display::renderer::…` path a consumer named before the
// carve resolves through these re-exports unchanged.
pub use materials::{
    build_face_material_layout, grid_overlay_params, procedural_material_average_color,
    procedural_material_pixels, procedural_material_texture_size,
    relative_material_base_colors_public, upload_face_material_texture, GridOverlayParams,
    LayerBand, MaterialSource,
};
pub use view_cube::{ViewCubeRenderer, VIEW_CUBE_VIEWPORT_MARGIN, VIEW_CUBE_VIEWPORT_PIXELS};
pub use targets::{create_depth_view, create_msaa_color_view, DEPTH_FORMAT, MSAA_SAMPLE_COUNT};
pub use gizmo::TransformGizmoRenderer;
pub use grid::SceneGridRenderer;
pub use points::{enabled_grid_planes, GridPlaneInstance, PointsRenderer};
pub use infinite_grid::InfiniteGridRenderer;
pub use onion::{onion_ghost_tint, OnionFogParams};

// Internal cross-submodule glue: each submodule reaches its siblings' non-`pub`
// (`pub(crate)`) items — and the shared imports above — through `use super::*`.
// Only submodules that expose such items to a sibling or the tests are re-globbed;
// `materials`/`infinite_grid`/`onion` publish only the `pub` API above, so a glob
// would re-export nothing.
pub(crate) use chrome::*;
pub(crate) use gizmo::*;
pub(crate) use lines::*;
pub(crate) use points::*;
pub(crate) use targets::*;
// `grid`/`view_cube` expose their `pub(crate)` geometry helpers ONLY to the unit
// tests (their in-crate render callers reach them within their own module), so the
// glue glob is test-only — unconditional, it would be an unused import in the lib build.
#[cfg(test)]
pub(crate) use grid::*;
#[cfg(test)]
pub(crate) use view_cube::*;

/// Append an alpha channel to a linear RGB colour, producing the `[f32; 4]` the
/// line pipeline's vertices carry (M8: lattice/floor draw at low opacity).
pub(crate) fn with_alpha(rgb: [f32; 3], alpha: f32) -> [f32; 4] {
    [rgb[0], rgb[1], rgb[2], alpha]
}
