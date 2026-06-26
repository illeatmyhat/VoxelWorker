// Instanced unit-cube voxel shader (Milestone 4).
//
// Each instance is one occupied voxel. The unit cube has per-face normals and
// per-face base UVs (24 vertices / 36 indices) so faces shade and texture
// independently.
//
// M4 adds three things on top of the M2 flat shading:
//   * BUG 1 fix — per-voxel texture slice. One procedural texture belongs to the
//     whole BLOCK; the vertex stage offsets the base face UV by the voxel's
//     `block_local_coord` and divides by `voxels_per_block`, so each voxel face
//     shows only its 1/density slice (NOT the whole texture repeated per cube).
//   * BUG 2 fix — grid overlay derived from the fragment's ABSOLUTE voxel
//     position (world position + grid half-extent), not from face UVs. Cube faces
//     flip UV direction, so a UV-derived block boundary lands one voxel off on
//     vertical faces; the world-position form is orientation-independent.
//   * Directional + ambient lighting applied to the sampled texture colour.

// std140-safe uniform block. Each vec3 is followed by a scalar so the vec3 never
// straddles a 16-byte boundary. Field order matches `VoxelUniforms` in renderer.rs.
struct VoxelUniforms {
    view_projection: mat4x4<f32>,
    grid_half_extent: vec3<f32>,
    voxels_per_block: f32,
    voxel_line_color: vec3<f32>,
    grid_overlay_enabled: f32,
    block_line_color: vec3<f32>,
    // Face-orientation debug flag (0 = normal render, 1 = colour-by-normal debug).
    // Reuses the std140 scalar slot that pads the preceding vec3 to 16 bytes.
    debug_face_mode: f32,
    voxel_line_half_width: f32,
    block_line_half_width: f32,
    voxel_line_alpha: f32,
    block_line_alpha: f32,
    // --- Layer-range scrubber (issue #12) ---
    // The visible band, in voxel Y-layer indices. A fragment is kept when its
    // layer satisfies `band_min <= layer <= band_max` (BOTH ends INCLUSIVE).
    // Full range = band_min 0, band_max >= grid_y - 1 (then nothing is clipped).
    // The onion skin is a separate volumetric fog pass (shaders/onion_fog.wgsl),
    // so this opaque pass only needs the band. Two pads keep the std140 16-byte
    // slot (matching `VoxelUniforms` in renderer.rs).
    band_min: f32,
    band_max: f32,
    // Per-voxel material modulation toggle (ADR 0001 step 3): 1 = modulate the
    // lit/textured colour by material_base_colors[material_id], 0 = leave it.
    // Off for debug-faces and for a loaded VS block. (Reuses a former band pad.)
    material_modulation_enabled: f32,
    _band_pad1: f32,
    // Per-material base colours (ADR 0001 step 3), one vec4 per MaterialChoice
    // ([r, g, b, _pad], LINEAR), RELATIVE to the bound texture's average. Indexed
    // by the per-instance material_id; the fragment stage MULTIPLIES the lit
    // texture colour by this so distinct nodes render in distinct materials.
    material_base_colors: array<vec4<f32>, 3>,
};

@group(0) @binding(0)
var<uniform> uniforms: VoxelUniforms;

// Per-object on-face-grid flag bit packed into `material_id` (issue #29 S4).
// MIRRORS `crate::voxel::GRID_OVERLAY_BIT` (= 1 << 15) in `src/voxel.rs` and the
// same const in `cuboid.wgsl` / `cuboid_loaded.wgsl`. The resolver ORs this bit
// into a voxel's `material_id` iff its node enabled the on-face grid; the
// fragment stage gates the grid branch on it (ANDed with the `grid_overlay_enabled`
// master) and masks it OFF before any colour lookup so it never corrupts the colour.
const GRID_OVERLAY_BIT: u32 = 32768u;

// M7: the material is a 6-layer array, one layer per cube face. Layer order
// matches the renderer's CubeFaceSlot: 0 +X(east), 1 -X(west), 2 +Y(up),
// 3 -Y(down), 4 +Z(south), 5 -Z(north). A uniform material has the same image
// on all six layers, so the per-voxel slice + grid overlay below are unchanged.
@group(1) @binding(0)
var material_texture: texture_2d_array<f32>;
@group(1) @binding(1)
var material_sampler: sampler;

// Pick the texture-array layer for a cube face from its outward normal.
fn face_layer(face_normal: vec3<f32>) -> i32 {
    let axis_magnitude = abs(face_normal);
    if (axis_magnitude.x > 0.5) {
        // +X → east (0), -X → west (1).
        return select(1, 0, face_normal.x > 0.0);
    } else if (axis_magnitude.y > 0.5) {
        // +Y → up (2), -Y → down (3).
        return select(3, 2, face_normal.y > 0.0);
    } else {
        // +Z → south (4), -Z → north (5).
        return select(5, 4, face_normal.z > 0.0);
    }
}

// Look up a per-voxel material base colour by id, clamped into range so an
// unexpected id can never index out of bounds (ADR 0001 step 3).
fn material_base_colors_lookup(material_id: u32) -> vec3<f32> {
    // Mask the on-face-grid flag bit (issue #29 S4) OFF before indexing so the
    // flag never corrupts the colour index (it would otherwise push the id far
    // past 2 and clamp every flagged voxel to material 2).
    let index = min(material_id & ~GRID_OVERLAY_BIT, 2u);
    return uniforms.material_base_colors[index].rgb;
}

// Whether this voxel's on-face grid should draw: the per-object flag bit packed
// into `material_id` (issue #29 S4) ANDed with the scene-wide master uniform
// (`grid_overlay_enabled`). Master OFF ⇒ no node draws; master ON ⇒ only voxels
// whose node opted in (bit set) draw — no re-resolve/re-upload to toggle the master.
fn on_face_grid_enabled(material_id: u32) -> bool {
    return uniforms.grid_overlay_enabled > 0.5 && (material_id & GRID_OVERLAY_BIT) != 0u;
}

struct VertexInput {
    @location(0) vertex_position: vec3<f32>,
    @location(1) face_normal: vec3<f32>,
    @location(2) face_uv: vec2<f32>,
};

struct InstanceInput {
    @location(3) world_position: vec3<f32>,
    @location(4) block_local_coord: vec3<f32>,
    // Per-voxel material id (ADR 0001 step 3): indexes material_base_colors.
    @location(5) material_id: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) texture_coord: vec2<f32>,
    // Absolute voxel position: world position of this fragment shifted so that
    // voxel boundaries fall on integers (BUG 2 fix input).
    @location(2) voxel_absolute_position: vec3<f32>,
    // M7: the texture-array layer for this face (flat — constant per face).
    @location(3) @interpolate(flat) face_texture_layer: i32,
    // Issue #12: this voxel's Y layer index, recovered from the instance CENTRE
    // (not the interpolated fragment), so a cube's top/bottom faces share the
    // same layer. `vox_centre_abs.y = world_position.y + grid_half_extent.y`, and
    // because centres sit at integer+0.5, `layer = floor(that)`. Flat so every
    // fragment of the cube reports the voxel's own layer.
    @location(4) @interpolate(flat) voxel_layer: f32,
    // Per-voxel material id (ADR 0001 step 3), flat (constant per instance), used
    // to index material_base_colors in the fragment stage.
    @location(5) @interpolate(flat) material_id: u32,
};

@vertex
fn vertex_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    // Unit cube centred on the voxel centre (half-extent 0.5 in each axis).
    let world_point = instance.world_position + vertex.vertex_position * 0.5;

    // --- BUG 1 fix: per-voxel texture slice within the block ---
    // Pick the two in-plane axes from the face normal, offset the base face UV by
    // the voxel's block-local coordinate, then divide by the density so one
    // block's texture spans all its voxels (each voxel = a 1/density slice).
    let axis_magnitude = abs(vertex.face_normal);
    var voxel_offset: vec2<f32>;
    if (axis_magnitude.x > 0.5) {
        voxel_offset = vec2<f32>(instance.block_local_coord.z, instance.block_local_coord.y);
    } else if (axis_magnitude.y > 0.5) {
        voxel_offset = vec2<f32>(instance.block_local_coord.x, instance.block_local_coord.z);
    } else {
        voxel_offset = vec2<f32>(instance.block_local_coord.x, instance.block_local_coord.y);
    }

    var output: VertexOutput;
    output.clip_position = uniforms.view_projection * vec4<f32>(world_point, 1.0);
    output.world_normal = vertex.face_normal;
    output.texture_coord = (vertex.face_uv + voxel_offset) / uniforms.voxels_per_block;
    output.voxel_absolute_position = world_point + uniforms.grid_half_extent;
    output.face_texture_layer = face_layer(vertex.face_normal);
    // Layer index from the voxel CENTRE (instance.world_position), shifted into
    // absolute (0-based) voxel space. Centres are at integer+0.5 so floor() lands
    // on the voxel's layer index regardless of which face this vertex belongs to.
    output.voxel_layer = floor(instance.world_position.y + uniforms.grid_half_extent.y);
    output.material_id = instance.material_id;
    return output;
}

// Map an outward normal to a signed-axis debug colour:
//   +X red, -X cyan; +Y green, -Y magenta; +Z blue, -Z yellow.
// The dominant axis of the (normalized) normal picks the colour.
fn debug_face_color(face_normal: vec3<f32>) -> vec3<f32> {
    let axis_magnitude = abs(face_normal);
    if (axis_magnitude.x > axis_magnitude.y && axis_magnitude.x > axis_magnitude.z) {
        // +X red, -X cyan.
        return select(vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), face_normal.x > 0.0);
    } else if (axis_magnitude.y > axis_magnitude.z) {
        // +Y green, -Y magenta.
        return select(vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), face_normal.y > 0.0);
    } else {
        // +Z blue, -Z yellow.
        return select(vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), face_normal.z > 0.0);
    }
}

@fragment
fn fragment_main(
    input: VertexOutput,
    @builtin(front_facing) is_front_facing: bool,
) -> @location(0) vec4<f32> {
    // --- Layer-range band clip + onion skin (issue #12) ---
    // The band is INCLUSIVE on both ends: layers [band_min, band_max] are solid.
    // Debug-face mode skips this entirely so the culling regression check always
    // sees the whole model.
    //
    // This opaque voxel pass draws ONLY the displayed band: keep fragments whose
    // layer is in [band_min, band_max] (inclusive) and hard-discard the rest. The
    // onion skin is no longer drawn here — it is a separate fullscreen volumetric
    // SDF-raymarch fog pass (see shaders/onion_fog.wgsl), so the surrounding
    // layers read as smooth fog with no voxel edges.
    if (uniforms.debug_face_mode <= 0.5) {
        let layer = input.voxel_layer;
        let in_band = layer >= uniforms.band_min && layer <= uniforms.band_max;
        if (!in_band) {
            discard;
        }
    }

    // --- Face-orientation debug mode ---
    // Colour each fragment by its outward face normal (signed-axis palette),
    // bypassing texture + lighting. Any fragment that is NOT front-facing (a back
    // face that survived because culling is off in the debug pipeline) is flagged
    // with bold black diagonal stripes so winding/cull bugs are unmistakable.
    if (uniforms.debug_face_mode > 0.5) {
        var debug_color = debug_face_color(input.world_normal);
        if (!is_front_facing) {
            // Diagonal stripes in screen space over a forced-white base.
            let stripe = step(0.5, fract((input.clip_position.x + input.clip_position.y) * 0.06));
            debug_color = mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 0.0, 0.0), stripe);
        }
        return vec4<f32>(debug_color, 1.0);
    }

    // Sampled material colour for this face's layer (sRGB texture → linear via
    // the format). The per-voxel slice in `texture_coord` is unchanged: each face
    // samples its own layer, sliced by block_local_coord exactly as before.
    let sampled = textureSample(
        material_texture,
        material_sampler,
        input.texture_coord,
        input.face_texture_layer,
    ).rgb;

    // Directional + ambient lighting (kept from M2), applied to the texture.
    let light_direction = normalize(vec3<f32>(0.4, 0.9, 0.5));
    let normal = normalize(input.world_normal);
    let diffuse = max(dot(normal, light_direction), 0.0);
    let ambient = 0.45;
    let lighting = ambient + (1.0 - ambient) * diffuse;
    var color = sampled * lighting;

    // --- Per-voxel material modulation (ADR 0001 step 3) ---
    // Multiply by this voxel's material base colour (relative to the bound
    // texture's average) so distinct nodes render in distinct materials from the
    // one shared texture. Off (flag 0) for a loaded VS block, which stays global.
    // Debug-faces mode returns earlier, so it is unaffected.
    if (uniforms.material_modulation_enabled > 0.5) {
        let base = material_base_colors_lookup(input.material_id);
        color = color * base;
    }

    // --- BUG 2 fix: grid overlay from absolute voxel position ---
    // Per-object (issue #29 S4): the master uniform ANDed with this voxel's flag
    // bit. The bold-block-line maths (from the absolute position) is unchanged.
    if (on_face_grid_enabled(input.material_id)) {
        let absolute = input.voxel_absolute_position;
        // 1 on the two in-plane axes, 0 on the face-normal axis.
        let in_plane = step(abs(input.world_normal), vec3<f32>(0.5));
        // Distance to the nearest voxel boundary (per axis).
        let voxel_distance = abs(absolute - floor(absolute + 0.5));
        // Distance to the nearest block boundary (per axis).
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

        // The block line is bolder/darker and wins where it is present.
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
