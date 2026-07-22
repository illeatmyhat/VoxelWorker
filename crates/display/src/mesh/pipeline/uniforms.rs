use super::*;

/// std140-safe uniform block for the cuboid pass (ADR 0002 E3b-2). Carries the
/// camera matrix, the grid half-extent and density (driving the per-voxel texture
/// slice and the position-based grid overlay), the grid-overlay parameters, and
/// the per-material base colours (reused from the instanced step-3b modulation).
/// Every `vec3` is followed by a scalar so it never straddles a 16-byte boundary;
/// the four grid-line scalars then fill the slot before the `vec4` array (which
/// must be 16-aligned). Field order matches the WGSL `CuboidUniforms` exactly.
///
/// Fields are `pub(super)` so the renderer impl in the parent [`super`] module can
/// construct + functional-update this block; they are otherwise an implementation
/// detail of the uniform layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct CuboidUniforms {
    pub(super) view_projection: [[f32; 4]; 4],
    pub(super) grid_half_extent: [f32; 3],
    pub(super) voxels_per_block: f32,
    pub(super) voxel_line_color: [f32; 3],
    pub(super) grid_overlay_enabled: f32,
    pub(super) block_line_color: [f32; 3],
    pub(super) material_modulation_enabled: f32,
    pub(super) voxel_line_half_width: f32,
    pub(super) block_line_half_width: f32,
    pub(super) voxel_line_alpha: f32,
    pub(super) block_line_alpha: f32,
    // Layer-range band clip (issue #12 parity) + debug-faces flag. The two band
    // bounds plus the debug flag plus a pad fill one 16-byte slot, so the colour
    // array below stays 16-aligned (matching the WGSL `CuboidUniforms`).
    pub(super) band_min: f32,
    pub(super) band_max: f32,
    pub(super) debug_face_mode: f32,
    /// ADR 0012 (H1): the onion GHOST flag (0 = normal solid render, 1 = flat
    /// translucent ghost tint). Occupies the former `_band_pad` slot; `0.0` for the
    /// solid draw keeps the solid uniform bytes identical (non-onion goldens byte-green).
    pub(super) ghost_mode: f32,
    pub(super) material_base_colors: [[f32; 4]; MaterialChoice::MATERIAL_COUNT],
    /// Per-material atlas sub-rect (ADR 0002 E3c-1 / O8), indexed by `material_id`:
    /// `[inset_min_u, inset_min_v, inset_size_u, inset_size_v]`. The shader maps the
    /// per-voxel slice's `fract`-tiled UV into this window of the single atlas, so a
    /// chunk of mixed materials is ONE mesh = ONE draw (no per-material texture
    /// bind). Each `vec4` is naturally 16-aligned.
    pub(super) material_atlas_rects: [[f32; 4]; MaterialChoice::MATERIAL_COUNT],
    /// ADR 0012 (H1): the onion ghost tint (linear RGB + src alpha), read only when
    /// `ghost_mode > 0.5`. Appended so the solid draw's uniform layout is unchanged.
    pub(super) ghost_tint: [f32; 4],
    /// Added to `voxel_absolute_position` INSIDE the on-face grid overlay to recover
    /// the TRUE world voxel frame (`= recentre − grid_half_extent`), so the overlay's
    /// voxel and block lines anchor to the world block lattice — the SAME lattice the
    /// per-object block-lattice cage draws on — instead of the render grid's local
    /// half-extent frame (which is out of block phase whenever `recentre` is not a
    /// whole block). Only the overlay reads it; the texture/UV slice keeps `absolute`,
    /// so material tiling is unchanged (goldens byte-green while the overlay is off).
    pub(super) overlay_world_offset: [f32; 3],
    pub(super) _overlay_pad: f32,
}

/// Convert a packed [`MaterialAtlas`]'s per-material sub-rects into the uniform
/// array layout `[inset_min_u, inset_min_v, inset_size_u, inset_size_v]` the shader
/// indexes by `material_id`. Materials without a packed sub-rect (should not happen
/// for the procedural set) fall back to the WHOLE atlas (`[0,0,1,1]`), so a missing
/// id degrades to "sample the atlas" rather than panicking.
pub(crate) fn atlas_rects_from(atlas: &MaterialAtlas) -> [[f32; 4]; MaterialChoice::MATERIAL_COUNT] {
    let mut rects = [[0.0, 0.0, 1.0, 1.0]; MaterialChoice::MATERIAL_COUNT];
    for (slot, sub_rect) in rects.iter_mut().zip(atlas.sub_rects.iter()) {
        let [size_u, size_v] = sub_rect.inset_size();
        *slot = [sub_rect.inset_min_u, sub_rect.inset_min_v, size_u, size_v];
    }
    rects
}

/// Build a ghost-only [`CuboidUniforms`] block (issue #78 — the selected-operand ghost
/// passes; ADR 0012 H1 is the ghost-branch precedent): `ghost_mode = 1` + `ghost_tint`,
/// with the camera + frame scalars the vertex stage reads. The `cuboid.wgsl` ghost branch
/// returns the flat tint before any texture / material / overlay / band read, so every
/// other field is filled with inert values (overlay + modulation off, band FULL).
pub(crate) fn flat_ghost_uniforms(
    view_projection: glam::Mat4,
    grid_dimensions: [u32; 3],
    voxels_per_block: u32,
    ghost_tint: [f32; 4],
) -> CuboidUniforms {
    let overlay = crate::renderer::grid_overlay_params();
    CuboidUniforms {
        view_projection: view_projection.to_cols_array_2d(),
        // FLOORED half, matching the solid draw's corner-anchoring (an odd dim's
        // `dim/2.0` would sit half a voxel off — see `update_uniforms`).
        grid_half_extent: substrate::spatial::GridHalfExtent::of_grid_dimensions(grid_dimensions)
            .voxels(),
        voxels_per_block: voxels_per_block.max(1) as f32,
        voxel_line_color: overlay.voxel_line_color,
        grid_overlay_enabled: 0.0,
        block_line_color: overlay.block_line_color,
        material_modulation_enabled: 0.0,
        voxel_line_half_width: overlay.voxel_line_half_width,
        block_line_half_width: overlay.block_line_half_width,
        voxel_line_alpha: overlay.voxel_line_alpha,
        block_line_alpha: overlay.block_line_alpha,
        band_min: 0.0,
        band_max: u32::MAX as f32,
        debug_face_mode: 0.0,
        ghost_mode: 1.0,
        material_base_colors: [[1.0, 1.0, 1.0, 0.0]; MaterialChoice::MATERIAL_COUNT],
        material_atlas_rects: [[0.0, 0.0, 1.0, 1.0]; MaterialChoice::MATERIAL_COUNT],
        ghost_tint,
        // The ghost branch returns before the overlay, so the anchor is inert here.
        overlay_world_offset: [0.0; 3],
        _overlay_pad: 0.0,
    }
}
