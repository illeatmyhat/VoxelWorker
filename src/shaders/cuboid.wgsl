// Cuboid mesh shader (ADR 0002 E3b-1, part of #18) — flag-gated alternate path.
//
// Draws the exposed-face triangle mesh built by `cuboid_mesh.rs`: each vertex
// carries a WORLD position, the face's outward normal, and the box's material_id.
// This step renders SHAPE + per-box material colour + basic lighting only — the
// SAME directional+ambient lighting and the SAME per-material base-colour
// modulation the instanced voxel shader uses, so a cuboid render reads as the
// same shaded solid. NO texture sampling / slice / grid overlay / layer clip /
// debug-faces here (those are later E3 sub-steps).

// std140-safe; field order matches `CuboidUniforms` in cuboid_mesh.rs.
struct CuboidUniforms {
    view_projection: mat4x4<f32>,
    // 1 = modulate the lit colour by material_base_colors[material_id], 0 = off.
    // Three trailing scalars pad this to a 16-byte slot (matching the Rust
    // `CuboidUniforms`); a `vec3` here would force 16-byte alignment and a
    // 144-byte struct, mismatching the 128-byte buffer.
    material_modulation_enabled: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    // Per-material base colours ([r,g,b,_pad], LINEAR), relative to the bound
    // texture's average — identical to the instanced path's step-3b array.
    material_base_colors: array<vec4<f32>, 3>,
};

@group(0) @binding(0)
var<uniform> uniforms: CuboidUniforms;

fn material_base_colors_lookup(material_id: u32) -> vec3<f32> {
    let index = min(material_id, 2u);
    return uniforms.material_base_colors[index].rgb;
}

struct VertexInput {
    @location(0) world_position: vec3<f32>,
    @location(1) face_normal: vec3<f32>,
    @location(2) material_id: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) @interpolate(flat) material_id: u32,
};

@vertex
fn vertex_main(vertex: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = uniforms.view_projection * vec4<f32>(vertex.world_position, 1.0);
    output.world_normal = vertex.face_normal;
    output.material_id = vertex.material_id;
    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Flat base colour (no texture in this step): a neutral warm grey that the
    // per-material modulation then tints, so distinct materials read distinct.
    // Matches the "Plain" procedural average closely enough for shape parity.
    let base_surface = vec3<f32>(0.62, 0.55, 0.42);

    // Directional + ambient lighting — identical constants to voxel.wgsl.
    let light_direction = normalize(vec3<f32>(0.4, 0.9, 0.5));
    let normal = normalize(input.world_normal);
    let diffuse = max(dot(normal, light_direction), 0.0);
    let ambient = 0.45;
    let lighting = ambient + (1.0 - ambient) * diffuse;
    var color = base_surface * lighting;

    // Per-box material modulation (ADR 0001 step 3): multiply by the material's
    // relative base colour so distinct boxes render in distinct materials.
    if (uniforms.material_modulation_enabled > 0.5) {
        let base = material_base_colors_lookup(input.material_id);
        color = color * base;
    }

    return vec4<f32>(color, 1.0);
}
