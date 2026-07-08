// Brick-field raymarch (ADR 0011 G1) — the minimal GPU display sink.
//
// A fullscreen pass whose fragment walks a block-space DDA along the pixel's view
// ray (no clip-map yet — that is G2). At each block it binary-searches the sorted
// resident BrickRecord array (the G0 build, packed world-block key as a (hi, lo)
// u32 pair): a kind-0 COARSE record hits as a solid block-cube; a kind-1 SCULPTED
// record descends to a voxel DDA over the brick's R8 atlas slot; a miss steps on.
//
// **Residency-miss contract (ADR 0011 4a, decided at G1):** a sculpted record whose
// atlas slot is the NON_RESIDENT sentinel renders its COARSE form (the solid
// block-cube) — degraded-but-correct, never skipped. One branch, paid up front, so
// G4's residency rings are a pure eviction policy.
//
// **Depth compositing (grill Q5):** the pass runs INSIDE the existing 4× MSAA voxel
// pass and writes ray-hit depth via `frag_depth`, so the depth-tested overlays
// (scene grid, infinite grid, points, onion fog's depth-stop) and the later passes
// (view cube, egui) composite exactly as they do over the rasterized mesh.
//
// **MSAA parity with the mesh (parity gate clause c):** the fragment runs PER SAMPLE
// (forced by `@builtin(sample_index)`). Each sample casts its own ray — reproducing
// the rasterizer's per-sample coverage and per-sample depth — but SHADES at the
// PIXEL-CENTRE ray's intersection with the hit face's plane, reproducing the
// rasterizer's non-centroid centre evaluation (including extrapolation). Interior
// pixels therefore resolve to exactly the mesh's centre-evaluated colour, and
// silhouette/step edges resolve to the same per-face coverage blend.
//
// ## Frames (ADR 0008)
//
// The march runs in the SHIFTED render frame `sv = world + grid_half_extent +
// lattice_shift`: voxel boundaries sit on integers and BLOCK boundaries on
// multiples of the brick edge (the shift re-aligns a non-block-aligned recentre).
// Absolute quantities are recovered by INTEGER adds carried in the uniforms
// (`voxel_bias`, `block_bias`), never re-derived from floats:
//   absolute voxel  = sv_voxel_cell + voxel_bias
//   absolute block  = sv_block_cell + block_bias
// Shading positions convert back with `p = sv − lattice_shift` (the cuboid
// shader's `voxel_absolute_position` frame) and `world = p − grid_half_extent`.

struct BrickUniforms {
    view_projection: mat4x4<f32>,
    inverse_view_projection: mat4x4<f32>,
    // The central 3D viewport rect in physical pixels: x, y, width, height. The
    // fullscreen triangle is viewport-mapped, so NDC must be derived from the
    // fragment position RELATIVE to this rect (matching the camera's aspect).
    viewport: vec4<f32>,
    // Half the grid voxel dimensions, floored to integers (the cuboid shader's
    // corner-anchoring); `world + grid_half_extent` is the shading-absolute frame.
    grid_half_extent: vec3<f32>,
    voxels_per_block: f32,
    // --- grid-overlay + material shading, mirroring `CuboidUniforms` exactly ---
    voxel_line_color: vec3<f32>,
    grid_overlay_enabled: f32,
    block_line_color: vec3<f32>,
    material_modulation_enabled: f32,
    voxel_line_half_width: f32,
    block_line_half_width: f32,
    voxel_line_alpha: f32,
    block_line_alpha: f32,
    // The single-producer scene's material colour index (G1 engages only for a
    // uniform-material record set) and the resident record count.
    material_id: u32,
    record_count: u32,
    _material_pad0: u32,
    _material_pad1: u32,
    // xyz: the integer lattice shift re-aligning block boundaries in the render
    // frame ((recentre − half_extent) mod edge); w: the brick edge in voxels.
    lattice_shift_and_edge: vec4<i32>,
    // xyz: absolute block = sv block cell + this bias; w: atlas tiles per axis.
    block_bias_and_tiles: vec4<i32>,
    // xyz: absolute voxel = sv voxel cell + this bias; w unused.
    voxel_bias: vec4<i32>,
    // x: first in-band voxel Z (sv frame); y: one-past-last in-band voxel Z (sv
    // frame) — the layer-range band clip, applied at traverse time (the mesh path
    // applies it at build time). zw unused.
    band_voxel_sv: vec4<i32>,
    // The traversal AABB in the sv frame: the resident bricks' bounds intersected
    // with the band slab. Rays outside it never march.
    traversal_lo: vec4<f32>,
    traversal_hi: vec4<f32>,
    material_base_colors: array<vec4<f32>, 3>,
    material_atlas_rects: array<vec4<f32>, 3>,
};

@group(0) @binding(0)
var<uniform> uniforms: BrickUniforms;

// One resident brick, sorted ascending by (key_hi, key_lo) — the G0 packed
// world-block key split for WGSL (no u64). `atlas_slot` == NON_RESIDENT marks a
// sculpted brick whose payload is not resident (the residency-miss contract).
struct BrickGpuRecord {
    key_hi: u32,
    key_lo: u32,
    kind: u32,
    atlas_slot: u32,
};

@group(0) @binding(1)
var<storage, read> brick_records: array<BrickGpuRecord>;

// The sculpted-brick occupancy atlas (R8: 0 empty / 1.0 occupied), read with
// textureLoad — exact, no filtering.
@group(0) @binding(2)
var sculpted_atlas: texture_3d<f32>;

// The SAME procedural material atlas + nearest/clamp sampler the cuboid mesh
// binds, so a brick-path pixel samples the identical texel.
@group(1) @binding(0)
var material_texture: texture_2d<f32>;
@group(1) @binding(1)
var material_sampler: sampler;

// The sentinel marking a sculpted record whose atlas payload is NOT resident;
// must match `NON_RESIDENT_ATLAS_SLOT` in brick_raymarch.rs.
const NON_RESIDENT_ATLAS_SLOT: u32 = 0xffffffffu;

// The 21-bit biased-lane world-block key (must match `pack_world_block_key`):
// biased = coord + 2^20 per axis; packed u64 = z<<42 | y<<21 | x, split (hi, lo).
const WORLD_BLOCK_KEY_BIAS: i32 = 1048576; // 1 << 20

// Block-DDA step budget: no clip-map yet (G2), so a ray crosses at most the
// traversal AABB's block diagonal — generous for every current scene.
const MAX_BLOCK_STEPS: i32 = 1024;
// In-brick voxel-DDA budget: at most 3·edge + 3 voxels per brick (edge ≤ 64).
const MAX_VOXEL_STEPS: i32 = 256;

// The standard 4× MSAA sample locations (identical on D3D12 and Vulkan), in
// pixel-fraction coordinates from the pixel's top-left corner. Each per-sample
// invocation casts its ray through ITS sample position so coverage matches the
// rasterizer's per-sample coverage.
const MSAA_4X_SAMPLE_POSITIONS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(0.375, 0.125),
    vec2<f32>(0.875, 0.375),
    vec2<f32>(0.125, 0.625),
    vec2<f32>(0.625, 0.875),
);

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // One viewport-covering triangle.
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var output: VertexOutput;
    output.clip_position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return output;
}

struct Ray {
    // Origin/direction in the SHIFTED render frame (sv): voxel boundaries on
    // integers, block boundaries on multiples of the brick edge.
    origin: vec3<f32>,
    direction: vec3<f32>,
    // Direction with zero components replaced (no NaN/Inf in the slab math).
    safe_direction: vec3<f32>,
}

// Unproject a framebuffer pixel position through the inverse view-projection into
// an sv-frame ray. Near/far unprojection handles perspective AND orthographic.
fn camera_ray(pixel: vec2<f32>) -> Ray {
    let ndc_x = (pixel.x - uniforms.viewport.x) / uniforms.viewport.z * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel.y - uniforms.viewport.y) / uniforms.viewport.w * 2.0;
    let near_h = uniforms.inverse_view_projection * vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    let far_h = uniforms.inverse_view_projection * vec4<f32>(ndc_x, ndc_y, 1.0, 1.0);
    let near_world = near_h.xyz / near_h.w;
    let far_world = far_h.xyz / far_h.w;
    let direction = normalize(far_world - near_world);
    let shift = vec3<f32>(uniforms.lattice_shift_and_edge.xyz);
    var ray: Ray;
    ray.origin = near_world + uniforms.grid_half_extent + shift;
    ray.direction = direction;
    ray.safe_direction = vec3<f32>(
        select(direction.x, 1e-20, abs(direction.x) < 1e-20),
        select(direction.y, 1e-20, abs(direction.y) < 1e-20),
        select(direction.z, 1e-20, abs(direction.z) < 1e-20),
    );
    return ray;
}

// Binary-search the sorted records for a (hi, lo) key. Returns the index or -1.
fn find_brick_record(key_hi: u32, key_lo: u32) -> i32 {
    var low = 0;
    var high = i32(uniforms.record_count) - 1;
    loop {
        if (low > high) {
            break;
        }
        let mid = (low + high) / 2;
        let record = brick_records[mid];
        if (record.key_hi == key_hi && record.key_lo == key_lo) {
            return mid;
        }
        let record_less = record.key_hi < key_hi
            || (record.key_hi == key_hi && record.key_lo < key_lo);
        if (record_less) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return -1;
}

// Pack an absolute world-block coordinate into the (hi, lo) split of the G0 key:
// three 21-bit biased lanes, z-major (z<<42 | y<<21 | x).
fn pack_world_block_key_split(absolute_block: vec3<i32>) -> vec2<u32> {
    let biased_x = u32(absolute_block.x + WORLD_BLOCK_KEY_BIAS);
    let biased_y = u32(absolute_block.y + WORLD_BLOCK_KEY_BIAS);
    let biased_z = u32(absolute_block.z + WORLD_BLOCK_KEY_BIAS);
    let key_hi = (biased_z << 10u) | (biased_y >> 11u);
    let key_lo = ((biased_y & 0x7ffu) << 21u) | biased_x;
    return vec2<u32>(key_hi, key_lo);
}

// Euclidean floor division (matches Rust div_euclid for positive divisors).
fn floor_div(value: i32, divisor: i32) -> i32 {
    let quotient = value / divisor;
    let remainder = value - quotient * divisor;
    return select(quotient, quotient - 1, remainder != 0 && (remainder < 0) != (divisor < 0));
}

// Is a voxel of a sculpted brick occupied? Exact textureLoad of the R8 atlas.
fn sculpted_voxel_occupied(atlas_slot: u32, brick_local_voxel: vec3<i32>) -> bool {
    let tiles = u32(uniforms.block_bias_and_tiles.w);
    let edge = uniforms.lattice_shift_and_edge.w;
    let tile = vec3<i32>(
        i32(atlas_slot % tiles),
        i32((atlas_slot / tiles) % tiles),
        i32(atlas_slot / (tiles * tiles)),
    );
    let texel = textureLoad(sculpted_atlas, tile * edge + brick_local_voxel, 0).r;
    return texel > 0.5;
}

// A ray-march hit: the entry face (axis + facing sign), the plane's sv-frame
// coordinate on that axis (for the centre-ray shading evaluation), the sample
// ray's hit parameter (for per-sample depth), and the hit voxel cell (sv frame).
struct MarchHit {
    hit: bool,
    entry_axis: i32,
    // +1 when the face's outward normal points along the NEGATIVE ray direction
    // component (the normal is -sign(direction[axis]) on entry_axis).
    normal_sign: f32,
    plane_sv: f32,
    hit_t: f32,
    voxel_cell: vec3<i32>,
}

// Ray/AABB slab entry: max component of the near-face parameters (clamped to 0)
// and its axis. The AABB is the block's box CLAMPED to the traversal bounds, so a
// band cut-plane entry reports axis 2 — the cap face the banded mesher synthesises.
struct SlabEntry {
    t_enter: f32,
    t_exit: f32,
    axis: i32,
}

fn clamped_box_entry(ray: Ray, box_lo: vec3<f32>, box_hi: vec3<f32>) -> SlabEntry {
    let inverse = 1.0 / ray.safe_direction;
    let t_a = (box_lo - ray.origin) * inverse;
    let t_b = (box_hi - ray.origin) * inverse;
    let t_near = min(t_a, t_b);
    let t_far = max(t_a, t_b);
    var entry: SlabEntry;
    entry.t_exit = min(min(t_far.x, t_far.y), t_far.z);
    // The entry face is the LAST near-plane crossed; ties resolve x → y → z,
    // mirrored exactly by the CPU reference march.
    if (t_near.x >= t_near.y && t_near.x >= t_near.z) {
        entry.axis = 0;
        entry.t_enter = t_near.x;
    } else if (t_near.y >= t_near.z) {
        entry.axis = 1;
        entry.t_enter = t_near.y;
    } else {
        entry.axis = 2;
        entry.t_enter = t_near.z;
    }
    entry.t_enter = max(entry.t_enter, 0.0);
    return entry;
}

// March one ray through the brick field. Blocks step by DDA; a resident block
// resolves via the record kinds (coarse cube / sculpted voxel DDA / non-resident
// falls back to the coarse cube). All boxes are clamped to the traversal AABB so
// the band clip yields cap faces, exactly like the banded mesh.
fn march_brick_field(ray: Ray) -> MarchHit {
    var miss: MarchHit;
    miss.hit = false;

    let edge = f32(uniforms.lattice_shift_and_edge.w);
    let edge_i = uniforms.lattice_shift_and_edge.w;
    let bounds_lo = uniforms.traversal_lo.xyz;
    let bounds_hi = uniforms.traversal_hi.xyz;

    let inverse = 1.0 / ray.safe_direction;
    let t_a = (bounds_lo - ray.origin) * inverse;
    let t_b = (bounds_hi - ray.origin) * inverse;
    let t_near = min(t_a, t_b);
    let t_far = max(t_a, t_b);
    let t_enter = max(max(max(t_near.x, t_near.y), t_near.z), 0.0);
    let t_exit = min(min(t_far.x, t_far.y), t_far.z);
    if (t_exit < t_enter) {
        return miss;
    }

    // Seed the block DDA a hair inside the traversal AABB.
    let entry_position = ray.origin + ray.direction * (t_enter + 1e-4);
    var block_cell = vec3<i32>(floor(entry_position / edge));
    let block_step = vec3<i32>(sign(ray.direction));
    let t_delta = abs(vec3<f32>(edge) / ray.safe_direction);
    var t_max = vec3<f32>(
        select(
            (f32(block_cell.x) * edge - entry_position.x) / ray.safe_direction.x,
            (f32(block_cell.x + 1) * edge - entry_position.x) / ray.safe_direction.x,
            block_step.x > 0,
        ) + t_enter,
        select(
            (f32(block_cell.y) * edge - entry_position.y) / ray.safe_direction.y,
            (f32(block_cell.y + 1) * edge - entry_position.y) / ray.safe_direction.y,
            block_step.y > 0,
        ) + t_enter,
        select(
            (f32(block_cell.z) * edge - entry_position.z) / ray.safe_direction.z,
            (f32(block_cell.z + 1) * edge - entry_position.z) / ray.safe_direction.z,
            block_step.z > 0,
        ) + t_enter,
    );
    var t_block_enter = t_enter;

    for (var step = 0; step < MAX_BLOCK_STEPS; step = step + 1) {
        let absolute_block = block_cell + uniforms.block_bias_and_tiles.xyz;
        let key = pack_world_block_key_split(absolute_block);
        let record_index = find_brick_record(key.x, key.y);
        if (record_index >= 0) {
            let record = brick_records[record_index];
            // The block's box, CLAMPED to the traversal bounds (band cut planes
            // become cap faces; a partially-banded block keeps only its slab).
            let block_lo = vec3<f32>(block_cell) * edge;
            let block_hi = block_lo + vec3<f32>(edge);
            let clamped_lo = max(block_lo, bounds_lo);
            let clamped_hi = min(block_hi, bounds_hi);
            if (clamped_lo.x < clamped_hi.x && clamped_lo.y < clamped_hi.y
                && clamped_lo.z < clamped_hi.z) {
                let entry = clamped_box_entry(ray, clamped_lo, clamped_hi);
                if (entry.t_exit >= entry.t_enter) {
                    // Residency-miss contract: a sculpted record with no resident
                    // atlas payload renders its COARSE form.
                    let coarse_form = record.kind == 0u
                        || record.atlas_slot == NON_RESIDENT_ATLAS_SLOT;
                    if (coarse_form) {
                        var hit: MarchHit;
                        hit.hit = true;
                        hit.entry_axis = entry.axis;
                        hit.normal_sign = -sign(ray.direction[entry.axis]);
                        hit.plane_sv = ray.origin[entry.axis]
                            + ray.direction[entry.axis] * entry.t_enter;
                        // Snap the shading plane to the exact clamped face.
                        if (ray.direction[entry.axis] > 0.0) {
                            hit.plane_sv = clamped_lo[entry.axis];
                        } else {
                            hit.plane_sv = clamped_hi[entry.axis];
                        }
                        hit.hit_t = entry.t_enter;
                        let hit_position = ray.origin + ray.direction * (entry.t_enter + 1e-4);
                        let block_min_voxel = block_cell * edge_i;
                        hit.voxel_cell = clamp(
                            vec3<i32>(floor(hit_position)),
                            block_min_voxel,
                            block_min_voxel + vec3<i32>(edge_i - 1),
                        );
                        return hit;
                    }
                    // Sculpted brick: voxel DDA over the atlas slot, bounded to the
                    // in-band voxel range of this block.
                    let voxel_entry_position =
                        ray.origin + ray.direction * (entry.t_enter + 1e-4);
                    var voxel_cell = vec3<i32>(floor(voxel_entry_position));
                    let voxel_step = vec3<i32>(sign(ray.direction));
                    let voxel_t_delta = abs(1.0 / ray.safe_direction);
                    var voxel_t_max = vec3<f32>(
                        select(
                            (f32(voxel_cell.x) - voxel_entry_position.x) / ray.safe_direction.x,
                            (f32(voxel_cell.x + 1) - voxel_entry_position.x) / ray.safe_direction.x,
                            voxel_step.x > 0,
                        ) + entry.t_enter,
                        select(
                            (f32(voxel_cell.y) - voxel_entry_position.y) / ray.safe_direction.y,
                            (f32(voxel_cell.y + 1) - voxel_entry_position.y) / ray.safe_direction.y,
                            voxel_step.y > 0,
                        ) + entry.t_enter,
                        select(
                            (f32(voxel_cell.z) - voxel_entry_position.z) / ray.safe_direction.z,
                            (f32(voxel_cell.z + 1) - voxel_entry_position.z) / ray.safe_direction.z,
                            voxel_step.z > 0,
                        ) + entry.t_enter,
                    );
                    let block_min_voxel = block_cell * edge_i;
                    let block_max_voxel = block_min_voxel + vec3<i32>(edge_i);
                    // The in-band voxel-Z range of this block (the band clip).
                    let band_z_lo = max(block_min_voxel.z, uniforms.band_voxel_sv.x);
                    let band_z_hi = min(block_max_voxel.z, uniforms.band_voxel_sv.y);
                    var voxel_entry_axis = entry.axis;
                    var t_voxel_enter = entry.t_enter;
                    for (var voxel_step_index = 0; voxel_step_index < MAX_VOXEL_STEPS;
                        voxel_step_index = voxel_step_index + 1) {
                        if (voxel_cell.x < block_min_voxel.x || voxel_cell.y < block_min_voxel.y
                            || voxel_cell.z < band_z_lo
                            || voxel_cell.x >= block_max_voxel.x
                            || voxel_cell.y >= block_max_voxel.y
                            || voxel_cell.z >= band_z_hi) {
                            break;
                        }
                        let brick_local = voxel_cell - block_min_voxel;
                        if (sculpted_voxel_occupied(record.atlas_slot, brick_local)) {
                            var hit: MarchHit;
                            hit.hit = true;
                            hit.entry_axis = voxel_entry_axis;
                            hit.normal_sign = -sign(ray.direction[voxel_entry_axis]);
                            // The entered voxel face's exact plane coordinate.
                            if (ray.direction[voxel_entry_axis] > 0.0) {
                                hit.plane_sv = f32(voxel_cell[voxel_entry_axis]);
                            } else {
                                hit.plane_sv = f32(voxel_cell[voxel_entry_axis] + 1);
                            }
                            // A band/traversal cut plane is not a voxel boundary —
                            // snap to the clamped box face instead when the entry
                            // came through it (first voxel only: t == entry.t_enter).
                            if (t_voxel_enter == entry.t_enter) {
                                if (ray.direction[voxel_entry_axis] > 0.0) {
                                    hit.plane_sv = clamped_lo[voxel_entry_axis];
                                } else {
                                    hit.plane_sv = clamped_hi[voxel_entry_axis];
                                }
                            }
                            hit.hit_t = t_voxel_enter;
                            hit.voxel_cell = voxel_cell;
                            return hit;
                        }
                        // Advance; the crossed boundary becomes the next voxel's entry.
                        if (voxel_t_max.x <= voxel_t_max.y && voxel_t_max.x <= voxel_t_max.z) {
                            t_voxel_enter = voxel_t_max.x;
                            voxel_cell.x = voxel_cell.x + voxel_step.x;
                            voxel_t_max.x = voxel_t_max.x + voxel_t_delta.x;
                            voxel_entry_axis = 0;
                        } else if (voxel_t_max.y <= voxel_t_max.z) {
                            t_voxel_enter = voxel_t_max.y;
                            voxel_cell.y = voxel_cell.y + voxel_step.y;
                            voxel_t_max.y = voxel_t_max.y + voxel_t_delta.y;
                            voxel_entry_axis = 1;
                        } else {
                            t_voxel_enter = voxel_t_max.z;
                            voxel_cell.z = voxel_cell.z + voxel_step.z;
                            voxel_t_max.z = voxel_t_max.z + voxel_t_delta.z;
                            voxel_entry_axis = 2;
                        }
                    }
                }
            }
        }

        if (t_block_enter > t_exit) {
            break;
        }
        if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
            block_cell.x = block_cell.x + block_step.x;
            t_block_enter = t_max.x;
            t_max.x = t_max.x + t_delta.x;
        } else if (t_max.y <= t_max.z) {
            block_cell.y = block_cell.y + block_step.y;
            t_block_enter = t_max.y;
            t_max.y = t_max.y + t_delta.y;
        } else {
            block_cell.z = block_cell.z + block_step.z;
            t_block_enter = t_max.z;
            t_max.z = t_max.z + t_delta.z;
        }
    }

    return miss;
}

// ============================================================================
// Shading — a transcription of cuboid.wgsl's fragment (per-voxel texture slice,
// lighting, material modulation, position-based grid overlay), evaluated at an
// explicit position instead of a rasterizer-interpolated varying.
// ============================================================================

fn coord_component(a: f32, sign_value: f32) -> f32 {
    let base = floor(a);
    let frac = a - base;
    return base + select(1.0 - frac, frac, sign_value > 0.0);
}

fn material_base_colors_lookup(material_id: u32) -> vec3<f32> {
    let index = min(material_id, 2u);
    return uniforms.material_base_colors[index].rgb;
}

// `absolute` is the cuboid shader's `voxel_absolute_position` (world +
// grid_half_extent); `world_normal` the face's outward unit normal.
fn shade_cuboid_surface(absolute: vec3<f32>, world_normal: vec3<f32>) -> vec4<f32> {
    let axis_magnitude = abs(world_normal);
    var u_value: f32;
    var v_value: f32;
    if (axis_magnitude.x > 0.5) {
        let v_sign = select(1.0, -1.0, world_normal.x > 0.0);
        u_value = coord_component(absolute.y, 1.0);
        v_value = coord_component(absolute.z, v_sign);
    } else if (axis_magnitude.y > 0.5) {
        let v_sign = select(1.0, -1.0, world_normal.y > 0.0);
        u_value = coord_component(absolute.x, 1.0);
        v_value = coord_component(absolute.z, v_sign);
    } else {
        let u_sign = select(-1.0, 1.0, world_normal.z > 0.0);
        u_value = coord_component(absolute.x, u_sign);
        v_value = coord_component(absolute.y, 1.0);
    }
    let texture_coord = vec2<f32>(u_value, v_value) / uniforms.voxels_per_block;

    let atlas_rect = uniforms.material_atlas_rects[min(uniforms.material_id, 2u)];
    let tile_uv = fract(texture_coord);
    let atlas_uv = atlas_rect.xy + tile_uv * atlas_rect.zw;
    // Level 0 explicitly: no mips + nearest sampler makes this identical to the
    // mesh path's textureSample, and it is legal in non-uniform control flow.
    let sampled = textureSampleLevel(material_texture, material_sampler, atlas_uv, 0.0).rgb;

    let light_direction = normalize(vec3<f32>(0.4, 0.9, 0.5));
    let normal = normalize(world_normal);
    let diffuse = max(dot(normal, light_direction), 0.0);
    let ambient = 0.45;
    let lighting = ambient + (1.0 - ambient) * diffuse;
    var color = sampled * lighting;

    if (uniforms.material_modulation_enabled > 0.5) {
        let base = material_base_colors_lookup(uniforms.material_id);
        color = color * base;
    }

    if (uniforms.grid_overlay_enabled > 0.5) {
        let in_plane = step(abs(world_normal), vec3<f32>(0.5));
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

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fragment_render(
    @builtin(position) position: vec4<f32>,
    @builtin(sample_index) sample_index: u32,
) -> FragmentOutput {
    // Per-sample execution (forced by sample_index): each sample casts its own
    // ray for coverage + depth; shading evaluates at the pixel-centre ray's
    // intersection with the hit face's plane (the rasterizer's centre evaluation).
    let pixel_corner = floor(position.xy);
    let pixel_centre = pixel_corner + vec2<f32>(0.5);
    let sample_offset = MSAA_4X_SAMPLE_POSITIONS[min(sample_index, 3u)];
    let sample_position = pixel_corner + sample_offset;

    let sample_ray = camera_ray(sample_position);
    let hit = march_brick_field(sample_ray);
    if (!hit.hit) {
        discard;
    }

    // Centre-ray evaluation on the hit face's plane (extrapolation allowed —
    // matching non-centroid rasterizer interpolation).
    let centre_ray = camera_ray(pixel_centre);
    let plane_distance = hit.plane_sv - centre_ray.origin[hit.entry_axis];
    let t_centre = plane_distance / centre_ray.safe_direction[hit.entry_axis];
    var evaluation_sv = centre_ray.origin + centre_ray.direction * t_centre;
    // The normal-axis coordinate is exactly on the plane by construction.
    evaluation_sv[hit.entry_axis] = hit.plane_sv;

    let shift = vec3<f32>(uniforms.lattice_shift_and_edge.xyz);
    let absolute = evaluation_sv - shift;

    var world_normal = vec3<f32>(0.0);
    world_normal[hit.entry_axis] = hit.normal_sign;

    // Per-sample depth from the SAMPLE ray's own hit point (the rasterizer
    // interpolates depth per sample).
    let hit_sv = sample_ray.origin + sample_ray.direction * hit.hit_t;
    let hit_world = hit_sv - shift - uniforms.grid_half_extent;
    let clip = uniforms.view_projection * vec4<f32>(hit_world, 1.0);

    var output: FragmentOutput;
    output.color = shade_cuboid_surface(absolute, world_normal);
    output.depth = clamp(clip.z / clip.w, 0.0, 1.0);
    return output;
}

// The parity-harness entry (tests/gpu_parity.rs): a single-sample pass that
// reports the hit voxel's ABSOLUTE coordinate per pixel instead of a colour —
// (hit flag, x, y, z) with the i32 coordinates bitcast into u32 lanes.
@fragment
fn fragment_hit_identity(@builtin(position) position: vec4<f32>) -> @location(0) vec4<u32> {
    let pixel_centre = floor(position.xy) + vec2<f32>(0.5);
    let ray = camera_ray(pixel_centre);
    let hit = march_brick_field(ray);
    if (!hit.hit) {
        return vec4<u32>(0u, 0u, 0u, 0u);
    }
    let absolute_voxel = hit.voxel_cell + uniforms.voxel_bias.xyz;
    return vec4<u32>(
        1u,
        bitcast<u32>(absolute_voxel.x),
        bitcast<u32>(absolute_voxel.y),
        bitcast<u32>(absolute_voxel.z),
    );
}
