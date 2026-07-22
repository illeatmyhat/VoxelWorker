// Shared cuboid-face shading math, prepended (like grid_overlay.wgsl) into every
// shader that shades a merged cuboid face: cuboid.wgsl, cuboid_loaded.wgsl and the
// brick raymarch's `shade_cuboid_surface`. These four functions were copy-pasted
// across those three shaders — the exact drift hazard the grid-overlay dedup was
// invented for (the brick copy is a hand "transcription" of cuboid.wgsl's fragment).
// One definition now; the legitimately-differing cores (the mesh path's `fwidth`
// derivative vs the raymarch's analytic one, the atlas vs D2Array sampling) stay in
// each shader.

// One coordinate of the per-face texture UV for an absolute voxel position `a`:
// `floor(a) + (sign_value > 0 ? fract(a) : 1 - fract(a))`. The block multiples wash
// out under the per-voxel `fract`, so `floor(a)` already carries the block-local slice;
// `sign_value` mirrors the axis for the faces whose winding runs against the +world
// direction. (Named `sign_value` to avoid shadowing the `sign` builtin.)
fn coord_component(a: f32, sign_value: f32) -> f32 {
    let base = floor(a);
    let frac = a - base;
    return base + select(1.0 - frac, frac, sign_value > 0.0);
}

// The per-face texture UV (in VOXELS, before the caller's `/ voxels_per_block`) of an
// absolute voxel position on the face with outward `world_normal`. Z-up: the texture's
// vertical axis (V) is world-Z on every side wall, so a directional texture stands
// upright on all four walls (±X, ±Y); the horizontal Z faces (top/bottom) tile in XY.
// Mapping: ±X U=±y V=±z; ±Y U=+x V=±z; ±Z U=±x V=+y.
fn cuboid_face_uv(absolute: vec3<f32>, world_normal: vec3<f32>) -> vec2<f32> {
    let axis_magnitude = abs(world_normal);
    var u_value: f32;
    var v_value: f32;
    if (axis_magnitude.x > 0.5) {
        // X-facing walls: U follows horizontal Y, V follows up (Z); the V sign keys on
        // world_normal.x exactly as the Y branch keys V on world_normal.y.
        let v_sign = select(1.0, -1.0, world_normal.x > 0.0);
        u_value = coord_component(absolute.y, 1.0);
        v_value = coord_component(absolute.z, v_sign);
    } else if (axis_magnitude.y > 0.5) {
        // Y-facing walls: U follows +x, V follows up (Z), sign flips with the face dir.
        let v_sign = select(1.0, -1.0, world_normal.y > 0.0);
        u_value = coord_component(absolute.x, 1.0);
        v_value = coord_component(absolute.z, v_sign);
    } else {
        // Z faces (horizontal): U follows x (sign flips with the face dir), V follows +y.
        let u_sign = select(-1.0, 1.0, world_normal.z > 0.0);
        u_value = coord_component(absolute.x, u_sign);
        v_value = coord_component(absolute.y, 1.0);
    }
    return vec2<f32>(u_value, v_value);
}

// The per-face D2Array texture layer for a loaded VS block, keyed on the outward
// normal (Z-up): +Z=2 up, -Z=3 down, +X=0, -X=1, -Y=4 south/front, +Y=5 north/back.
fn face_layer(face_normal: vec3<f32>) -> i32 {
    let axis_magnitude = abs(face_normal);
    if (axis_magnitude.z > 0.5) {
        return select(3, 2, face_normal.z > 0.0);
    } else if (axis_magnitude.x > 0.5) {
        return select(1, 0, face_normal.x > 0.0);
    } else {
        return select(5, 4, face_normal.y < 0.0);
    }
}

// Directional + ambient Lambert term for a face with outward `world_normal` — the
// constants carried over from the since-removed instanced path (voxel.wgsl, #20).
fn lambert_lighting(world_normal: vec3<f32>) -> f32 {
    let light_direction = normalize(vec3<f32>(0.4, 0.9, 0.5));
    let normal = normalize(world_normal);
    let diffuse = max(dot(normal, light_direction), 0.0);
    let ambient = 0.45;
    return ambient + (1.0 - ambient) * diffuse;
}
