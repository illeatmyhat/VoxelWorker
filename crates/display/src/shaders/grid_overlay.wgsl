// The SINGLE definition of the on-face voxel/block grid overlay coverage math.
//
// This block was copy-pasted across `cuboid.wgsl`, `cuboid_loaded.wgsl` and
// `brick_raymarch.wgsl`, and the same bug had to be fixed three times separately —
// the brick copy was missed once and shipped broken to the live app. It now lives
// here ONCE and is prepended to each of those shader modules at load time by
// `crate::shaders::with_grid_overlay`. WGSL allows module-scope declarations in any
// order, so this function is callable from the entry point below it.
//
// Each shader still computes its OWN `world_voxel` and `derivative` and passes them
// in, because those two legitimately differ per path:
//   * `world_voxel` anchor — cuboid uses `absolute + overlay_world_offset`, the
//     brick path uses `absolute + lattice_shift.xyz` (a different uniform for the
//     same world-lattice anchor).
//   * `derivative` — the cuboid paths take `fwidth(absolute)`; the brick raymarch
//     passes an analytic `screen_derivative`, because `fwidth` is illegal in its
//     non-uniform control flow.
// Everything below (half-widths, per-tier fades, smoothstep bands, voxel/block
// blend) is what stayed identical and kept drifting — so it lives in one place now.
//
// Screen-space-aware line coverage (the `infinite_grid.wgsl` anti-moiré law applied
// to the on-face overlay). `derivative` is voxels per pixel, per axis: each line is
// held at a minimum PIXEL half-width and blended over a ~1-pixel band, and each
// tier's line family fades out per axis as its pitch nears the pixel grid. Constant
// voxel-space widths otherwise undersample into stippled moiré arcs at zoomed-out /
// grazing views. The block line (pitch = density voxels) wins over the voxel line
// where they overlap.
fn grid_overlay_color(
    surface_color: vec3<f32>,
    world_voxel: vec3<f32>,
    world_normal: vec3<f32>,
    derivative: vec3<f32>,
    density: f32,
    voxel_line_half_width: f32,
    block_line_half_width: f32,
    voxel_line_alpha: f32,
    block_line_alpha: f32,
    voxel_line_color: vec3<f32>,
    block_line_color: vec3<f32>,
) -> vec3<f32> {
    let in_plane = step(abs(world_normal), vec3<f32>(0.5));
    let voxel_distance = abs(world_voxel - floor(world_voxel + 0.5));
    let block_distance =
        abs(world_voxel / density - floor(world_voxel / density + 0.5)) * density;

    let pixel_antialias = max(derivative, vec3<f32>(0.012));
    let voxel_half_width =
        max(vec3<f32>(voxel_line_half_width), derivative * 0.6);
    let block_half_width =
        max(vec3<f32>(block_line_half_width), derivative * 0.6);
    // Tier visibility: fully on at >= 10 px per cell (derivative 0.1), gone below
    // ~4 px per cell (derivative 0.25). Block pitch = density voxels.
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

    var blend = voxel_strength * voxel_line_alpha;
    var line_color = voxel_line_color;
    let block_blend = block_strength * block_line_alpha;
    if (block_blend > blend) {
        blend = block_blend;
        line_color = block_line_color;
    }
    return mix(surface_color, line_color, blend);
}
