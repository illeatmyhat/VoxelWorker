// Cuboid mesh shader — LOADED VS BLOCK variant (part of #20).
//
// This is the cuboid path's `MaterialSource::Loaded` counterpart to `cuboid.wgsl`:
// it binds a 6-layer D2Array (one PNG per cube face, `MaterialSource::Loaded`) and
// selects the per-face layer FROM THE FACE NORMAL, reproducing the per-face layer
// selection the since-removed instanced path (`shaders/voxel.wgsl`, deleted with the
// legacy mesher, #20) used to do for a loaded block. The default `cuboid.wgsl`
// samples a packed PROCEDURAL atlas (Stone/Wood/Plain) and cannot show a runtime-
// loaded block texture; when a VS block is applied, the renderer selects THIS
// pipeline instead.
//
// Everything else — the per-voxel texture slice (absolute-position UV ÷ density),
// lighting, the position-based grid overlay, the band clip done at mesh-build time,
// and the debug-faces mode — is IDENTICAL to `cuboid.wgsl`, so a loaded block
// renders pixel-aligned with the procedural path's geometry. The procedural-only
// fields of the shared `CuboidUniforms` (material modulation + atlas rects) are
// simply unused here.

// std140-safe; field order matches `CuboidUniforms` in cuboid_mesh.rs EXACTLY (the
// same uniform buffer is bound to both the atlas and the loaded pipeline).
struct CuboidUniforms {
    view_projection: mat4x4<f32>,
    grid_half_extent: vec3<f32>,
    voxels_per_block: f32,
    voxel_line_color: vec3<f32>,
    grid_overlay_enabled: f32,
    block_line_color: vec3<f32>,
    material_modulation_enabled: f32,
    voxel_line_half_width: f32,
    block_line_half_width: f32,
    voxel_line_alpha: f32,
    block_line_alpha: f32,
    band_min: f32,
    band_max: f32,
    debug_face_mode: f32,
    // The full Rust `CuboidUniforms` carries `ghost_mode` here then a `ghost_tint`
    // vec4 after the atlas rects; the loaded path never ghosts, so both are declared
    // only as pads to reach `overlay_world_offset` at the correct std140 offset.
    _ghost_mode_pad: f32,
    material_base_colors: array<vec4<f32>, 3>,
    material_atlas_rects: array<vec4<f32>, 3>,
    _ghost_tint_pad: vec4<f32>,
    // Added to `voxel_absolute_position` inside the on-face overlay to recover the TRUE
    // world voxel frame (= recentre − grid_half_extent), anchoring the lines to the
    // world block lattice. Must match the Rust `CuboidUniforms` tail exactly.
    overlay_world_offset: vec3<f32>,
    _overlay_pad: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: CuboidUniforms;

// ADR 0003 §3c / ADR 0010 E3: the on-face-grid flag is NEITHER in `material_id` (the
// retired `GRID_OVERLAY_BIT` mirror) NOR a per-vertex attribute. A loaded VS block selects
// its per-face texture layer from the outward normal; the on-face-grid flag is the per-draw
// group(2) uniform (the chunk mesh is split into overlay-off / overlay-on draws).
struct OverlayActive { value: u32 };
@group(2) @binding(0)
var<uniform> draw_overlay: OverlayActive;

// Whether this face's on-face grid should draw: the per-DRAW overlay-active flag (group 2)
// ANDed with the scene-wide master uniform (`grid_overlay_enabled`).
fn on_face_grid_enabled() -> bool {
    return uniforms.grid_overlay_enabled > 0.5 && draw_overlay.value != 0u;
}

// The loaded block's 6-layer face texture array (one layer per cube face). Z-up:
// layer order matches the renderer's CubeFaceSlot / `face_layer`: 0 +X(east),
// 1 -X(west), 2 +Z(up), 3 -Z(down), 4 -Y(south/front), 5 +Y(north/back). The
// VERTICAL texture axis is Z: a grass block's `up` PNG lands on its +Z top, NOT a
// +Y wall. A uniform block puts the same PNG on all six layers; a per-face block
// puts each face PNG on its own layer.
@group(1) @binding(0)
var material_texture: texture_2d_array<f32>;
@group(1) @binding(1)
var material_sampler: sampler;

// `face_layer` (per-face D2Array layer from the normal) and `coord_component`
// (per-voxel UV component) are shared — see `shaders/cuboid_face_shading.wgsl`,
// prepended by `with_shared_shading`.

// Signed-axis debug colour, identical to cuboid.wgsl (both carried it over from
// the since-removed instanced voxel.wgsl, deleted with the legacy mesher, #20).
fn debug_face_color(face_normal: vec3<f32>) -> vec3<f32> {
    let axis_magnitude = abs(face_normal);
    if (axis_magnitude.x > axis_magnitude.y && axis_magnitude.x > axis_magnitude.z) {
        return select(vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), face_normal.x > 0.0);
    } else if (axis_magnitude.y > axis_magnitude.z) {
        return select(vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), face_normal.y > 0.0);
    } else {
        return select(vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), face_normal.z > 0.0);
    }
}

struct VertexInput {
    @location(0) world_position: vec3<f32>,
    @location(1) face_normal: vec3<f32>,
    @location(2) material_id: u32,
    // ADR 0010 E3: the on-face-grid flag is no longer a vertex attribute (it is the
    // per-draw group(2) uniform `draw_overlay`).
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) @interpolate(flat) material_id: u32,
    @location(2) voxel_absolute_position: vec3<f32>,
};

@vertex
fn vertex_main(vertex: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = uniforms.view_projection * vec4<f32>(vertex.world_position, 1.0);
    output.world_normal = vertex.face_normal;
    output.material_id = vertex.material_id;
    output.voxel_absolute_position = vertex.world_position + uniforms.grid_half_extent;
    return output;
}

@fragment
fn fragment_main(
    input: VertexOutput,
    @builtin(front_facing) is_front_facing: bool,
) -> @location(0) vec4<f32> {
    let absolute = input.voxel_absolute_position;

    // --- Face-orientation debug mode (cull-off parity) ---
    if (uniforms.debug_face_mode > 0.5) {
        var debug_color = debug_face_color(input.world_normal);
        if (!is_front_facing) {
            let stripe = step(0.5, fract((input.clip_position.x + input.clip_position.y) * 0.06));
            debug_color = mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 0.0, 0.0), stripe);
        }
        return vec4<f32>(debug_color, 1.0);
    }

    // --- Per-voxel texture slice (per-face UV direction; IDENTICAL to cuboid.wgsl) ---
    // Z-up: the texture's VERTICAL axis (V) is world-Z on every SIDE wall, so a
    // directional texture stands upright on all four walls (±X and ±Y); the horizontal
    // Z faces (top/bottom) tile in XY. Mapping: ±X U=±y V=±z; ±Y U=+x V=±z; ±Z U=±x V=+y.
    // Per-face UV direction (shared `cuboid_face_uv`, in voxels), then per-density.
    let texture_coord = cuboid_face_uv(absolute, input.world_normal) / uniforms.voxels_per_block;

    // Tile the per-voxel slice ourselves with `fract`, then sample the per-face
    // D2Array layer selected from the outward normal. The loaded material's sampler
    // is CLAMP-to-edge (the instanced path's `material_sampler`), and a cuboid merged
    // face spans many voxels (texture_coord runs 0..N/density), so we take `fract`
    // here to repeat the block texture once per voxel — `fract(texture_coord)` is
    // exactly the slice the instanced per-cube `(face_uv + block_local)/density` (in
    // [0,1)) samples, so the loaded cuboid face matches the instanced face texel-wise.
    let layer = face_layer(input.world_normal);
    let tile_uv = fract(texture_coord);
    let sampled = textureSample(material_texture, material_sampler, tile_uv, layer).rgb;

    // Directional + ambient lighting (shared `lambert_lighting`).
    var color = sampled * lambert_lighting(input.world_normal);

    // A loaded VS block renders as a single global material (no per-box modulation),
    // matching the instanced loaded path which disables modulation.

    // --- Position-based grid overlay (BUG 2 parity) ---
    // Per-object (issue #29 S4): master uniform ANDed with this face's flag bit.
    if (on_face_grid_enabled()) {
        // Anchor the overlay to the TRUE world voxel frame (world block lattice), not
        // the render grid's local half-extent frame; `absolute` stays for texture UV.
        let world_voxel = absolute + uniforms.overlay_world_offset;
        // `fwidth` is legal here (uniform control flow). The shared coverage math
        // lives in `grid_overlay_color` (shaders/grid_overlay.wgsl).
        let derivative = fwidth(absolute);
        color = grid_overlay_color(
            color,
            world_voxel,
            input.world_normal,
            derivative,
            uniforms.voxels_per_block,
            uniforms.voxel_line_half_width,
            uniforms.block_line_half_width,
            uniforms.voxel_line_alpha,
            uniforms.block_line_alpha,
            uniforms.voxel_line_color,
            uniforms.block_line_color,
        );
    }

    return vec4<f32>(color, 1.0);
}
