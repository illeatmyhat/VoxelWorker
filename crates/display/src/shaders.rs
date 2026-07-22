//! Shared WGSL snippets composed into multiple shader modules at load time.
//!
//! WGSL has no `#include`, and this crate loads each shader as a standalone module
//! via `include_str!` + `create_shader_module`. When a piece of shader math is used
//! by more than one module, copy-pasting it lets the copies drift — the on-face grid
//! overlay coverage math lived in three shaders and had to be bug-fixed three times
//! separately (the brick copy was missed once and shipped broken to the live app).
//!
//! The fix is the lightest possible composition: keep the shared math in ONE `.wgsl`
//! file and string-concatenate it onto each shader source here. No preprocessor, no
//! `naga_oil`, no build script — just [`with_grid_overlay`].

/// The one definition of `fn grid_overlay_color` (see `shaders/grid_overlay.wgsl`).
const GRID_OVERLAY_WGSL: &str = include_str!("shaders/grid_overlay.wgsl");

/// Prepend the shared grid-overlay function to a shader body, yielding a single
/// self-contained WGSL module source.
///
/// Every shader that draws the on-face voxel/block grid (`cuboid.wgsl`,
/// `cuboid_loaded.wgsl`, `brick_raymarch.wgsl`) composes its source through here, so
/// the coverage math has exactly one definition and cannot drift between copies.
/// WGSL permits module-scope declarations in any order, so the prepended function is
/// callable from the entry point in `shader_body`.
pub(crate) fn with_grid_overlay(shader_body: &str) -> String {
    format!("{GRID_OVERLAY_WGSL}\n{shader_body}")
}
