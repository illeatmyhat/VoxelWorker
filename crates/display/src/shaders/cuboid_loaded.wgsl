// Cuboid mesh shader — LOADED VS BLOCK variant (part of #20).
//
// This is the cuboid path's counterpart to the instanced loaded-block path
// (`shaders/voxel.wgsl` with a `MaterialSource::Loaded` 6-layer D2Array). The
// default `cuboid.wgsl` samples a packed PROCEDURAL atlas (Stone/Wood/Plain) and
// cannot show a runtime-loaded block texture. When a VS block is applied, the
// renderer selects THIS pipeline instead: it binds the block's 6-layer D2Array
// (one PNG per cube face) and selects the per-face layer FROM THE FACE NORMAL,
// exactly like the instanced path — so the cuboid path shows the SAME texture the
// instanced path shows (per-face parity).
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
    _band_pad: f32,
    material_base_colors: array<vec4<f32>, 3>,
    material_atlas_rects: array<vec4<f32>, 3>,
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

// Pick the texture-array layer for a cube face from its outward normal — IDENTICAL
// to the CPU `face_layer` in cuboid_mesh.rs (and `CubeFaceSlot`) so per-face textures
// land on the same faces. Z-up: +Z = up (2), -Z = down (3); the four horizontals are
// ±X (east/west) and ±Y (south/north).
fn face_layer(face_normal: vec3<f32>) -> i32 {
    let axis_magnitude = abs(face_normal);
    if (axis_magnitude.z > 0.5) {
        // Vertical axis (Z-up): +Z = up layer 2, -Z = down layer 3.
        return select(3, 2, face_normal.z > 0.0);
    } else if (axis_magnitude.x > 0.5) {
        return select(1, 0, face_normal.x > 0.0);
    } else {
        // ±Y horizontals: -Y = south/front (4), +Y = north/back (5).
        return select(5, 4, face_normal.y < 0.0);
    }
}

// One in-plane UV component (pre-density) for the per-voxel texture slice — copied
// verbatim from cuboid.wgsl's `coord_component` so the loaded slice phase-aligns to
// voxel/block boundaries exactly like the procedural path.
fn coord_component(a: f32, sign: f32) -> f32 {
    let base = floor(a);
    let frac = a - base;
    return base + select(1.0 - frac, frac, sign > 0.0);
}

// Signed-axis debug colour, identical to cuboid.wgsl / voxel.wgsl.
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
    let axis_magnitude = abs(input.world_normal);
    var u_value: f32;
    var v_value: f32;
    if (axis_magnitude.x > 0.5) {
        // X-facing walls (east/west): U follows horizontal Y, V follows up (Z). V sign
        // keys on world_normal.x, mirroring the Y branch's V sign on world_normal.y.
        let v_sign = select(1.0, -1.0, input.world_normal.x > 0.0);
        u_value = coord_component(absolute.y, 1.0);
        v_value = coord_component(absolute.z, v_sign);
    } else if (axis_magnitude.y > 0.5) {
        // Y-facing walls (north/south): U follows +x, V follows up (Z).
        let v_sign = select(1.0, -1.0, input.world_normal.y > 0.0);
        u_value = coord_component(absolute.x, 1.0);
        v_value = coord_component(absolute.z, v_sign);
    } else {
        // Z faces (top/bottom, horizontal): U follows x, V follows +y.
        let u_sign = select(-1.0, 1.0, input.world_normal.z > 0.0);
        u_value = coord_component(absolute.x, u_sign);
        v_value = coord_component(absolute.y, 1.0);
    }
    let texture_coord = vec2<f32>(u_value, v_value) / uniforms.voxels_per_block;

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

    // Directional + ambient lighting — identical constants to cuboid.wgsl / voxel.wgsl.
    let light_direction = normalize(vec3<f32>(0.4, 0.9, 0.5));
    let normal = normalize(input.world_normal);
    let diffuse = max(dot(normal, light_direction), 0.0);
    let ambient = 0.45;
    let lighting = ambient + (1.0 - ambient) * diffuse;
    var color = sampled * lighting;

    // A loaded VS block renders as a single global material (no per-box modulation),
    // matching the instanced loaded path which disables modulation.

    // --- Position-based grid overlay (BUG 2 parity) ---
    // Per-object (issue #29 S4): master uniform ANDed with this face's flag bit.
    if (on_face_grid_enabled()) {
        let in_plane = step(abs(input.world_normal), vec3<f32>(0.5));
        let voxel_distance = abs(absolute - floor(absolute + 0.5));
        let density = uniforms.voxels_per_block;
        let block_distance =
            abs(absolute / density - floor(absolute / density + 0.5)) * density;

        // Screen-space-aware line coverage — identical maths/constants to
        // `cuboid.wgsl` (see the comment there): minimum pixel half-width,
        // ~1-pixel antialias band, per-axis per-tier fade near pixel pitch.
        let derivative = fwidth(absolute);
        let pixel_antialias = max(derivative, vec3<f32>(0.012));
        let voxel_half_width =
            max(vec3<f32>(uniforms.voxel_line_half_width), derivative * 0.6);
        let block_half_width =
            max(vec3<f32>(uniforms.block_line_half_width), derivative * 0.6);
        let voxel_fade =
            vec3<f32>(1.0) - smoothstep(vec3<f32>(0.1), vec3<f32>(0.25), derivative);
        let block_fade = vec3<f32>(1.0)
            - smoothstep(vec3<f32>(0.1), vec3<f32>(0.25), derivative / density);
        let voxel_line = (vec3<f32>(1.0)
            - smoothstep(voxel_half_width, voxel_half_width + pixel_antialias, voxel_distance))
            * voxel_fade * in_plane;
        let block_line = (vec3<f32>(1.0)
            - smoothstep(block_half_width, block_half_width + pixel_antialias, block_distance))
            * block_fade * in_plane;
        let voxel_strength = max(max(voxel_line.x, voxel_line.y), voxel_line.z);
        let block_strength = max(max(block_line.x, block_line.y), block_line.z);

        var blend = voxel_strength * uniforms.voxel_line_alpha;
        var line_color = uniforms.voxel_line_color;
        let block_blend = block_strength * uniforms.block_line_alpha;
        if (block_blend > blend) {
            blend = block_blend;
            line_color = uniforms.block_line_color;
        }
        color = mix(color, line_color, blend);
    }

    return vec4<f32>(color, 1.0);
}
