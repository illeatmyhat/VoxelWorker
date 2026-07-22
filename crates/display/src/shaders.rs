//! Shared WGSL snippets composed into multiple shader modules at load time.
//!
//! WGSL has no `#include`, and this crate loads each shader as a standalone module
//! via `include_str!` + `create_shader_module`. When a piece of shader math is used
//! by more than one module, copy-pasting it lets the copies drift — the on-face grid
//! overlay coverage math lived in three shaders and had to be bug-fixed three times
//! separately (the brick copy was missed once and shipped broken to the live app).
//!
//! The fix is the lightest possible composition: keep the shared math in ONE `.wgsl`
//! file per concern and string-concatenate it onto each shader source here. No
//! preprocessor, no `naga_oil`, no build script — just [`with_shared_shading`].

/// The one definition of `fn grid_overlay_color` (see `shaders/grid_overlay.wgsl`).
const GRID_OVERLAY_WGSL: &str = include_str!("shaders/grid_overlay.wgsl");

/// The one definition of the shared cuboid-face shading math — `coord_component`,
/// `cuboid_face_uv`, `face_layer`, `lambert_lighting` (see
/// `shaders/cuboid_face_shading.wgsl`).
const CUBOID_FACE_SHADING_WGSL: &str = include_str!("shaders/cuboid_face_shading.wgsl");

/// Prepend the shared shading snippets (grid-overlay coverage + cuboid-face UV /
/// layer / lighting math) to a shader body, yielding a single self-contained WGSL
/// module source.
///
/// Every shader that shades a merged cuboid face and draws the on-face voxel/block
/// grid (`cuboid.wgsl`, `cuboid_loaded.wgsl`, `brick_raymarch.wgsl`) composes its
/// source through here, so each shared function has exactly one definition and cannot
/// drift between copies. WGSL permits module-scope declarations in any order, so the
/// prepended functions are callable from the entry point in `shader_body`.
pub(crate) fn with_shared_shading(shader_body: &str) -> String {
    format!("{GRID_OVERLAY_WGSL}\n{CUBOID_FACE_SHADING_WGSL}\n{shader_body}")
}
