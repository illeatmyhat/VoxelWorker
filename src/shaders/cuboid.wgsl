// Cuboid mesh shader (ADR 0002 E3b-2, part of #18) — flag-gated alternate path.
//
// Draws the exposed-face triangle mesh built by `cuboid_mesh.rs`: each vertex
// carries a WORLD position, the face's outward normal, and the box's material_id.
//
// E3b-1 rendered SHAPE + per-box material colour + lighting only (FLAT). E3b-2
// adds the two per-voxel surface features that make a MERGED box face read as a
// stack of per-voxel cubes, matching the instanced path (`shaders/voxel.wgsl`):
//
//   * PER-VOXEL TEXTURE SLICE — the block texture tiles once per voxel across a
//     merged face. The UV is the fragment's ABSOLUTE voxel position (world +
//     grid half-extent) on the face's two in-plane axes, divided by the density;
//     a `Repeat` sampler then repeats the texture once per voxel. Because the
//     integer part of `absolute / density` is dropped by the Repeat wrap, this is
//     EXACTLY the instanced slice `(face_uv + block_local_coord) / density`: a
//     voxel at absolute index `vi` shows the `vi mod density`-th 1/density slice,
//     phase-aligned to voxel/block boundaries.
//   * POSITION-BASED GRID OVERLAY — the per-voxel + per-block grid lines are
//     derived from the ABSOLUTE voxel position (NOT face UVs — project guard), so
//     they fall on the same boundaries the instanced overlay draws.
//
// The per-face D2Array texture layer is picked from the face normal (same mapping
// as instanced), and the per-box material modulation from E3b-1 still multiplies
// the lit texture colour. Layer-range clip + debug-faces are still later sub-steps.

// std140-safe; field order matches `CuboidUniforms` in cuboid_mesh.rs.
struct CuboidUniforms {
    view_projection: mat4x4<f32>,
    // Half the grid voxel dimensions; `world_position + grid_half_extent` makes
    // voxel boundaries fall on integers (the UV slice + grid overlay both key off
    // this absolute voxel position). The trailing scalar pads the vec3 to 16 bytes
    // and carries the density (voxels per block).
    grid_half_extent: vec3<f32>,
    voxels_per_block: f32,
    // Grid overlay (BUG-2-style position-based lines): the two line colours (each
    // padded by a following scalar so the vec3 never straddles a 16-byte slot).
    voxel_line_color: vec3<f32>,
    grid_overlay_enabled: f32,
    block_line_color: vec3<f32>,
    // 1 = modulate the lit colour by material_base_colors[material_id], 0 = off.
    material_modulation_enabled: f32,
    // Grid-line half-widths (voxel units) and blend alphas. These four floats
    // exactly fill the 16-byte slot, so the band slot below starts 16-aligned.
    voxel_line_half_width: f32,
    block_line_half_width: f32,
    voxel_line_alpha: f32,
    block_line_alpha: f32,
    // --- Layer-range band clip (issue #12 parity) ---
    // The visible band, in voxel Y-layer indices. A fragment is kept when its
    // layer satisfies `band_min <= layer <= band_max` (BOTH ends INCLUSIVE),
    // matching the instanced voxel pass. Full range = band_min 0, band_max huge.
    band_min: f32,
    band_max: f32,
    // Face-orientation debug flag (0 = normal render, 1 = colour-by-normal debug),
    // matching the instanced `debug_face_mode`: colours faces by outward normal,
    // draws a back-face stripe (with culling off), and disables band-clip /
    // texture / material / overlay. A trailing pad fills the 16-byte slot so the
    // array below stays 16-aligned.
    debug_face_mode: f32,
    _band_pad: f32,
    // Per-material base colours ([r,g,b,_pad], LINEAR), relative to the bound
    // texture's average — identical to the instanced path's step-3b array.
    material_base_colors: array<vec4<f32>, 3>,
};

@group(0) @binding(0)
var<uniform> uniforms: CuboidUniforms;

// Same 6-layer face-material array the instanced path binds (one layer per cube
// face). Layer order matches the renderer's CubeFaceSlot: 0 +X(east), 1 -X(west),
// 2 +Y(up), 3 -Y(down), 4 +Z(south), 5 -Z(north). The sampler MUST use a Repeat
// address mode so the per-voxel UV tiling repeats once per voxel.
@group(1) @binding(0)
var material_texture: texture_2d_array<f32>;
@group(1) @binding(1)
var material_sampler: sampler;

// Pick the texture-array layer for a face from its outward normal (identical to
// the instanced `face_layer`).
fn face_layer(face_normal: vec3<f32>) -> i32 {
    let axis_magnitude = abs(face_normal);
    if (axis_magnitude.x > 0.5) {
        return select(1, 0, face_normal.x > 0.0);
    } else if (axis_magnitude.y > 0.5) {
        return select(3, 2, face_normal.y > 0.0);
    } else {
        return select(5, 4, face_normal.z > 0.0);
    }
}

fn material_base_colors_lookup(material_id: u32) -> vec3<f32> {
    let index = min(material_id, 2u);
    return uniforms.material_base_colors[index].rgb;
}

// One in-plane UV component (pre-density) for the per-voxel texture slice, matching
// the instanced path. `sign > 0` runs the UV with the +world axis (→ `a`); `sign
// < 0` mirrors it within each voxel (→ `floor(a) + 1 - fract(a)`), reproducing the
// instanced face's UV direction. The block multiples wash out under the Repeat
// sampler, so `floor(a)` already carries the slice index.
fn coord_component(a: f32, sign: f32) -> f32 {
    let base = floor(a);
    let frac = a - base;
    return base + select(1.0 - frac, frac, sign > 0.0);
}

// Map an outward normal to a signed-axis debug colour — IDENTICAL to the
// instanced `debug_face_color` in voxel.wgsl so the cuboid debug-faces output
// matches the instanced reference:
//   +X red, -X cyan; +Y green, -Y magenta; +Z blue, -Z yellow.
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
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) @interpolate(flat) material_id: u32,
    // Absolute voxel position (world + grid half-extent): voxel boundaries fall on
    // integers. Drives both the per-voxel UV slice and the grid overlay.
    @location(2) voxel_absolute_position: vec3<f32>,
    // The texture-array layer for this face (flat — constant per face).
    @location(3) @interpolate(flat) face_texture_layer: i32,
};

@vertex
fn vertex_main(vertex: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = uniforms.view_projection * vec4<f32>(vertex.world_position, 1.0);
    output.world_normal = vertex.face_normal;
    output.material_id = vertex.material_id;
    // Each box-face vertex sits on an integer voxel plane, so the absolute voxel
    // position interpolates linearly across the face in VOXEL units (a face
    // spanning N voxels runs 0..N). Dividing the in-plane axes by the density in
    // the fragment stage + a Repeat sampler tiles the block texture once per voxel.
    output.voxel_absolute_position = vertex.world_position + uniforms.grid_half_extent;
    output.face_texture_layer = face_layer(vertex.face_normal);
    return output;
}

@fragment
fn fragment_main(
    input: VertexOutput,
    @builtin(front_facing) is_front_facing: bool,
) -> @location(0) vec4<f32> {
    // NOTE on the layer-range band clip (issue #12 parity): unlike the instanced
    // path (which discards per-fragment by voxel layer), the cuboid path clips the
    // band at MESH-BUILD time — the densified region is masked to the band's
    // Y-range before decomposition, so the band's top/bottom voxels expose real CAP
    // faces (a fragment discard on a single merged column would leave the band open-
    // topped, since the merged box's only +Y face is the model's true top). The
    // `band_*` uniforms are therefore unused by the shader and kept only for std140
    // layout/debug parity.
    let absolute = input.voxel_absolute_position;

    // --- Face-orientation debug mode (cull-off parity) ---
    // Colour each fragment by its outward face normal (signed-axis palette),
    // bypassing texture + lighting + material + overlay. Any fragment that is NOT
    // front-facing (a back face that survived because culling is off in the debug
    // pipeline) is flagged with bold black diagonal stripes — identical to the
    // instanced debug shader so the cuboid winding/cull check matches.
    if (uniforms.debug_face_mode > 0.5) {
        var debug_color = debug_face_color(input.world_normal);
        if (!is_front_facing) {
            let stripe = step(0.5, fract((input.clip_position.x + input.clip_position.y) * 0.06));
            debug_color = mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 0.0, 0.0), stripe);
        }
        return vec4<f32>(debug_color, 1.0);
    }

    // --- Per-voxel texture slice (BUG 1 parity) ---
    // Reproduce the instanced per-voxel slice EXACTLY, including the per-face UV
    // direction the instanced cube geometry bakes in (`unit_cube_geometry`), so a
    // non-symmetric texture (wood grain, a loaded VS block) lands texel-for-texel
    // identical — not just "looks like the same noise".
    //
    // Instanced texcoord (pre-/density) along an in-plane axis is
    //   face_uv_component + block_local_coord_component
    // where `block_local_coord` increases with the +world axis and `face_uv`
    // increases along the world axis with a per-face SIGN (table below, from the
    // instanced corner winding). For an absolute coordinate `a`, that component is
    //   floor(a) + (sign > 0 ? fract(a) : 1 - fract(a))
    // (the block multiples wash out under the Repeat sampler, so `floor(a)` already
    // carries the `block_local_coord mod density` slice). `coord_component` returns
    // it; `sign > 0` ⇒ `a`, `sign < 0` ⇒ the mirrored `floor(a) + 1 - fract(a)`.
    // `absolute` (the absolute voxel position) was bound at the top of the fn for
    // the band clip; reuse it here.
    let axis_magnitude = abs(input.world_normal);
    // Per-face (U axis, U sign) and (V axis, V sign), matching the instanced
    // unit-cube face UVs: +X U=+z V=-y; -X U=-z V=-y; +Y U=+x V=-z; -Y U=+x V=+z;
    // +Z U=+x V=+y; -Z U=-x V=+y.
    var u_value: f32;
    var v_value: f32;
    if (axis_magnitude.x > 0.5) {
        // X faces: U follows z, V follows -y. U sign flips with the face dir.
        let u_sign = select(-1.0, 1.0, input.world_normal.x > 0.0);
        u_value = coord_component(absolute.z, u_sign);
        v_value = coord_component(absolute.y, -1.0);
    } else if (axis_magnitude.y > 0.5) {
        // Y faces: U follows +x, V follows z (sign flips with the face dir).
        let v_sign = select(1.0, -1.0, input.world_normal.y > 0.0);
        u_value = coord_component(absolute.x, 1.0);
        v_value = coord_component(absolute.z, v_sign);
    } else {
        // Z faces: U follows x (sign flips with the face dir), V follows +y.
        let u_sign = select(-1.0, 1.0, input.world_normal.z > 0.0);
        u_value = coord_component(absolute.x, u_sign);
        v_value = coord_component(absolute.y, 1.0);
    }
    let texture_coord = vec2<f32>(u_value, v_value) / uniforms.voxels_per_block;

    let sampled = textureSample(
        material_texture,
        material_sampler,
        texture_coord,
        input.face_texture_layer,
    ).rgb;

    // Directional + ambient lighting — identical constants to voxel.wgsl.
    let light_direction = normalize(vec3<f32>(0.4, 0.9, 0.5));
    let normal = normalize(input.world_normal);
    let diffuse = max(dot(normal, light_direction), 0.0);
    let ambient = 0.45;
    let lighting = ambient + (1.0 - ambient) * diffuse;
    var color = sampled * lighting;

    // Per-box material modulation (ADR 0001 step 3): multiply by the material's
    // relative base colour so distinct boxes render in distinct materials.
    if (uniforms.material_modulation_enabled > 0.5) {
        let base = material_base_colors_lookup(input.material_id);
        color = color * base;
    }

    // --- Position-based grid overlay (BUG 2 parity) ---
    // Identical maths/constants to voxel.wgsl: lines from the absolute voxel
    // position (not UVs), with the block line winning over the voxel line.
    if (uniforms.grid_overlay_enabled > 0.5) {
        let in_plane = step(abs(input.world_normal), vec3<f32>(0.5));
        let voxel_distance = abs(absolute - floor(absolute + 0.5));
        let density = uniforms.voxels_per_block;
        let block_distance =
            abs(absolute / density - floor(absolute / density + 0.5)) * density;

        let antialias = 0.012;
        let voxel_half_width = uniforms.voxel_line_half_width;
        let block_half_width = uniforms.block_line_half_width;
        let voxel_line = (vec3<f32>(1.0)
            - smoothstep(vec3<f32>(voxel_half_width), vec3<f32>(voxel_half_width + antialias), voxel_distance))
            * in_plane;
        let block_line = (vec3<f32>(1.0)
            - smoothstep(vec3<f32>(block_half_width), vec3<f32>(block_half_width + antialias), block_distance))
            * in_plane;
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
