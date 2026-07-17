//! The composed ray–volume march: a slab entry, a block-scale [`VoxelDda`] with a
//! hierarchical empty-space skip, a per-block descent to an inner voxel-scale DDA, and
//! the entry-face normal — plus the flat exact-occupancy reference march.
//!
//! Everything domain-specific is injected as a closure: the level-occupancy predicate
//! ([`march_brick_hierarchy`]'s `level_occupied`), the per-block classification
//! ([`BlockContents`], which for a sculpted block carries its own per-voxel occupancy
//! closure), and the exact march's absolute-voxel occupancy. The kernel names nothing
//! about records, atlas bytes, or residency; the app wires those in.
//!
//! ## Literature
//!
//! * **Slab entry**: Kay & Kajiya, "Ray Tracing Complex Scenes" (SIGGRAPH 1986);
//!   Ericson, *Real-Time Collision Detection* (2005) §5.3.3 — the ray's parameter
//!   interval inside an AABB is the intersection of its three per-axis slab intervals.
//!   The traversal-box entry reuses [`substrate::spatial::Ray::intersect_box_slab`].
//! * **DDA**: Amanatides & Woo (1987) — see [`crate::voxel_dda`], stepped at block edge
//!   and at voxel edge 1.
//! * **Hierarchical empty-space skip**: Crassin, Neyret, Lefebvre & Eisemann,
//!   *GigaVoxels* (I3D 2009); Museth, *VDB* (SIGGRAPH 2013) — descend a min-mip
//!   occupancy pyramid coarsest-first and, at the coarsest EMPTY level covering the
//!   current cell, leap the ray to that cell's far face in one stride instead of
//!   stepping block by block. The pyramid's cell-key search is
//!   [`substrate::spatial::min_mip_pyramid`]; the empty-level-means-occupied *policy* stays in
//!   the app's `level_occupied` closure.
//!
//! ## Byte-exact arithmetic (the parity obligation)
//!
//! The WGSL shader is a mirror of this march and `gpu_parity` pins them byte-identical,
//! so the arithmetic here reproduces the shader's exactly: the near-zero direction guard
//! ([`substrate::spatial::SLAB_ZERO_DIRECTION_GUARD`]), the reciprocal `1.0 / safe_direction`
//! shared between the slab entry and every DDA seed, the small `1e-4` entry nudge, the
//! [`CLIPMAP_JUMP_EPSILON`] cell-exit hair, the x → y → z tie order, and the step
//! budgets are all load-bearing. Reproduce the exact ops; never "simplify".

use glam::{IVec3, Vec3};
use substrate::spatial::{Ray, RealAabb};

use crate::voxel_dda::VoxelDda;

/// The hair the hierarchical skip steps PAST a coarse-cell exit face before re-deriving
/// the block cell — larger than the per-block `1e-4` entry nudge so the jump reliably
/// lands in the next cell. MUST match `CLIPMAP_JUMP_EPSILON` in the WGSL mirror
/// (`shaders/brick_raymarch.wgsl`).
pub const CLIPMAP_JUMP_EPSILON: f32 = 1e-3;

/// Block-DDA step budget — the CPU mirror of the shader's `MAX_BLOCK_STEPS`. The pyramid
/// collapses empty space to a handful of strides; this ceiling only bounds the flat
/// fallback (pyramid off) crossing a wide traversal AABB. MUST match the WGSL constant.
pub const MAX_BLOCK_STEPS: u32 = 4096;

/// Per-block inner voxel-DDA step budget — an edge³ sculpted brick is crossed in at most
/// its voxel-diagonal steps; MUST match the shader's inner loop bound.
pub const MAX_VOXEL_STEPS: u32 = 256;

/// The flat exact-occupancy march's step budget — the traversal AABB's voxel diagonal
/// for every gated scene; MUST match the shader's flat march bound.
pub const MAX_EXACT_VOXEL_STEPS: u32 = 4096;

/// The small parameter nudge past a face before flooring to a cell, so a hit exactly on
/// a boundary lands inside the entered cell rather than on its edge. MUST match the
/// shader's `1e-4` entry offsets.
const ENTRY_NUDGE: f32 = 1e-4;

/// A march hit: the hit voxel in ABSOLUTE lattice coordinates (the frame's absolute
/// frame, `cell + bias`), plus the entered face's outward normal as an exact `±1` axis
/// vector (`[i32; 3]`, so `Eq` derives).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarchHit {
    /// The hit voxel's absolute lattice coordinate.
    pub absolute_voxel: [i32; 3],
    /// The entered face's outward normal, an exact `±1` axis vector.
    pub face_normal: [i32; 3],
}

/// What a block cell contains, as the injected classifier reports it. A sculpted block
/// carries its own per-voxel occupancy closure `Fn(brick_local) -> bool` (the app closes
/// over the block's atlas slot); the kernel runs the inner voxel DDA against it.
pub enum BlockContents<VoxelOccupied> {
    /// No record — the ray steps through this block.
    Empty,
    /// A coarse solid block — the first pierced voxel is the hit.
    CoarseSolid,
    /// A sculpted block — descend into the inner voxel DDA, querying `brick_local`
    /// occupancy through the carried closure.
    Sculpted(VoxelOccupied),
}

/// The plain-numeric frame the hierarchical march runs in (everything the shader's
/// uniforms carry that is not a closure). The block/voxel biases convert the shifted
/// march-frame cells to absolute lattice coordinates.
pub struct HierarchicalMarchParams {
    /// The traversal AABB (resident-brick bounds ∩ band slab), in the shifted march frame.
    pub traversal_lo: Vec3,
    pub traversal_hi: Vec3,
    /// The brick edge in voxels (the block-lattice cell edge).
    pub brick_edge_voxels: i32,
    /// absolute block = shifted-frame block cell + this.
    pub block_bias: IVec3,
    /// absolute voxel = shifted-frame voxel cell + this.
    pub voxel_bias: [i32; 3],
    /// `[first_in_band, one_past_last]` voxel-Z in the shifted frame (the band clip on
    /// the inner voxel DDA).
    pub band_voxel_sv: [i32; 2],
    /// The clip-map levels' `blocks_per_cell`, ordered COARSEST → FINEST — the same order
    /// the `level_occupied` closure indexes.
    pub level_blocks_per_cell: Vec<i32>,
}

/// The plain-numeric frame the flat exact-occupancy march runs in.
pub struct ExactMarchParams {
    /// The traversal AABB in the shifted march frame.
    pub traversal_lo: Vec3,
    pub traversal_hi: Vec3,
    /// `[first_in_band, one_past_last]` voxel-Z in the shifted frame (band clip).
    pub band_voxel_sv: [i32; 2],
    /// absolute voxel = shifted-frame voxel cell + this.
    pub voxel_bias: [i32; 3],
}

/// The outward face normal (an exact `±1` axis vector) for a march that ENTERED a box
/// through face `axis`: the normal opposes the ray's motion on that axis (mirrors the
/// shader's `normal_sign = -sign(direction[axis])`).
pub fn entry_face_normal(axis: usize, direction: Vec3) -> [i32; 3] {
    let mut normal = [0i32; 3];
    normal[axis] = if direction[axis] > 0.0 { -1 } else { 1 };
    normal
}

/// The ray direction with any component whose magnitude is below the slab guard nudged
/// to `+guard`, so the reciprocal used for the slab entry and every DDA seed stays finite
/// (the shared arithmetic the shader mirror pins). Identical guard to
/// [`substrate::spatial::Ray::slab_inverse_direction`], applied to the direction itself.
fn guarded_direction(direction: Vec3) -> Vec3 {
    let guard = |component: f32| -> f32 {
        if component.abs() < substrate::spatial::SLAB_ZERO_DIRECTION_GUARD {
            substrate::spatial::SLAB_ZERO_DIRECTION_GUARD
        } else {
            component
        }
    };
    Vec3::new(guard(direction.x), guard(direction.y), guard(direction.z))
}

/// The clamped-box entry: the ray's entry parameter, entry-face axis (x → y → z ties),
/// and exit parameter for the axis-aligned box `[lo, hi]`, using the shared `inverse`
/// reciprocal. Mirrors the shader's `clamped_box_entry`.
fn clamped_box_entry(origin: Vec3, inverse: Vec3, lo: Vec3, hi: Vec3) -> (usize, f32, f32) {
    let t_a = (lo - origin) * inverse;
    let t_b = (hi - origin) * inverse;
    let near = t_a.min(t_b);
    let far = t_a.max(t_b);
    let exit = far.x.min(far.y).min(far.z);
    let (entry_axis, enter) = if near.x >= near.y && near.x >= near.z {
        (0usize, near.x)
    } else if near.y >= near.z {
        (1usize, near.y)
    } else {
        (2usize, near.z)
    };
    (entry_axis, enter.max(0.0), exit)
}

/// The parameter at which the ray exits the clip-map cell (of `blocks_per_cell` blocks
/// per axis) containing `absolute_block`, plus the [`CLIPMAP_JUMP_EPSILON`] hair — the
/// jump target of the hierarchical skip. Mirrors the shader's `clipmap_try_skip` cell
/// bound. `blocks_per_cell` is at least 1.
fn clipmap_cell_exit_t(
    origin: Vec3,
    inverse: Vec3,
    edge: f32,
    block_bias: IVec3,
    absolute_block: IVec3,
    blocks_per_cell: i32,
) -> f32 {
    let cell = IVec3::new(
        absolute_block.x.div_euclid(blocks_per_cell),
        absolute_block.y.div_euclid(blocks_per_cell),
        absolute_block.z.div_euclid(blocks_per_cell),
    );
    let sv_block_lo = cell * blocks_per_cell - block_bias;
    let cell_lo = sv_block_lo.as_vec3() * edge;
    let cell_hi = (sv_block_lo + IVec3::splat(blocks_per_cell)).as_vec3() * edge;
    let ta = (cell_lo - origin) * inverse;
    let tb = (cell_hi - origin) * inverse;
    let tfar = ta.max(tb);
    let cell_exit = tfar.x.min(tfar.y).min(tfar.z);
    cell_exit + CLIPMAP_JUMP_EPSILON
}

/// March one ray through the brick field with the hierarchical empty-space skip — the
/// pure kernel behind `voxel_worker::brick::cpu_march_levels_counted`, a
/// step-for-step mirror of the WGSL `march_brick_field`. Returns the hit voxel (absolute)
/// and the number of block-DDA loop iterations (each iteration is one hierarchical jump
/// OR one per-block step) — the empty-space-skip metric the perf probe reports.
///
/// The domain injects three things: `level_occupied(level_index, absolute_block)` — is
/// the clip-map cell (at `params.level_blocks_per_cell[level_index]`) covering the block
/// occupied, or the level off (empty ⇒ report occupied, disabling that level's skip);
/// `classify_block(absolute_block)` — [`BlockContents`], carrying a sculpted block's
/// per-voxel occupancy closure; and, through the sculpted variant, `brick_local`
/// occupancy. `ray` is the pixel-centre ray in the shifted march frame.
pub fn march_brick_hierarchy<LevelFn, ClassifyFn, VoxelFn>(
    ray: Ray,
    params: &HierarchicalMarchParams,
    level_occupied: LevelFn,
    classify_block: ClassifyFn,
) -> (Option<MarchHit>, u32)
where
    LevelFn: Fn(usize, IVec3) -> bool,
    ClassifyFn: Fn([i32; 3]) -> BlockContents<VoxelFn>,
    VoxelFn: Fn([i32; 3]) -> bool,
{
    let origin = ray.origin;
    let direction = ray.direction;
    let safe = guarded_direction(direction);
    let inverse = 1.0 / safe;
    let edge = params.brick_edge_voxels as f32;
    let edge_i = params.brick_edge_voxels;
    let bounds_lo = params.traversal_lo;
    let bounds_hi = params.traversal_hi;
    let block_bias = params.block_bias;
    let [voxel_bias_x, voxel_bias_y, voxel_bias_z] = params.voxel_bias;

    // Slab entry against the traversal AABB (substrate's Ray primitive; its guarded
    // reciprocal equals `inverse` above, so the entry parameters match the shared-inverse
    // arithmetic the DDA seeds below reuse).
    let traversal_box = RealAabb {
        min: bounds_lo,
        max: bounds_hi,
    };
    let entry = match ray.intersect_box_slab(&traversal_box) {
        Some(interval) => interval,
        None => return (None, 0),
    };
    let t_enter = entry.t_enter;
    let t_exit = entry.t_exit;

    let entry_position = origin + direction * (t_enter + ENTRY_NUDGE);
    let mut block_dda = VoxelDda::seed(direction, safe, entry_position, t_enter, edge, 0);

    let mut steps = 0u32;
    'march: for _ in 0..MAX_BLOCK_STEPS {
        steps += 1;
        let absolute_block_v = block_dda.cell + block_bias;
        // Hierarchical DDA: descend the levels coarsest → finest and skip by the coarsest
        // level whose cell is EMPTY — an empty cell jumps the ray to that cell's exit in
        // ONE stride. A jump that would not advance the block cell falls through to a
        // per-block step (guaranteed progress). Only the coarsest empty level is attempted
        // each step (mirrors the shader's else-if chain).
        let mut jumped = false;
        for (level_index, &blocks_per_cell) in params.level_blocks_per_cell.iter().enumerate() {
            if level_occupied(level_index, absolute_block_v) {
                continue; // occupied (or level off) — try the next finer level
            }
            let jump_t = clipmap_cell_exit_t(
                origin,
                inverse,
                edge,
                block_bias,
                absolute_block_v,
                blocks_per_cell.max(1),
            );
            let jump_position = origin + direction * jump_t;
            let reseeded = VoxelDda::seed(direction, safe, jump_position, jump_t, edge, 0);
            if reseeded.cell != block_dda.cell {
                if jump_t > t_exit {
                    break 'march;
                }
                block_dda = reseeded;
                jumped = true;
            }
            break; // only the coarsest empty level is attempted this step
        }
        if jumped {
            continue 'march;
        }

        let absolute_block = [absolute_block_v.x, absolute_block_v.y, absolute_block_v.z];
        let contents = classify_block(absolute_block);
        if !matches!(contents, BlockContents::Empty) {
            let block_lo = block_dda.cell.as_vec3() * edge;
            let block_hi = block_lo + Vec3::splat(edge);
            let clamped_lo = block_lo.max(bounds_lo);
            let clamped_hi = block_hi.min(bounds_hi);
            if clamped_lo.x < clamped_hi.x
                && clamped_lo.y < clamped_hi.y
                && clamped_lo.z < clamped_hi.z
            {
                let (entry_axis, box_enter, box_exit) =
                    clamped_box_entry(origin, inverse, clamped_lo, clamped_hi);
                if box_exit >= box_enter {
                    match contents {
                        BlockContents::Empty => unreachable!("guarded by matches! above"),
                        BlockContents::CoarseSolid => {
                            let hit_position = origin + direction * (box_enter + ENTRY_NUDGE);
                            let block_min_voxel = block_dda.cell * edge_i;
                            let voxel_cell = hit_position.floor().as_ivec3().clamp(
                                block_min_voxel,
                                block_min_voxel + IVec3::splat(edge_i - 1),
                            );
                            return (
                                Some(MarchHit {
                                    absolute_voxel: [
                                        voxel_cell.x + voxel_bias_x,
                                        voxel_cell.y + voxel_bias_y,
                                        voxel_cell.z + voxel_bias_z,
                                    ],
                                    face_normal: entry_face_normal(entry_axis, direction),
                                }),
                                steps,
                            );
                        }
                        BlockContents::Sculpted(voxel_occupied) => {
                            // Inner voxel DDA (cell edge 1), tracking the per-voxel entry
                            // axis for the hit face's normal, clipped to the block and band.
                            let voxel_entry = origin + direction * (box_enter + ENTRY_NUDGE);
                            let block_min_voxel = block_dda.cell * edge_i;
                            let block_max_voxel = block_min_voxel + IVec3::splat(edge_i);
                            // Seed CLAMPED into the block's voxel range: a grazing ray entering
                            // the block through a MAX face lands `voxel_entry` exactly on that
                            // face, so a plain floor seeds one voxel PAST the block and the bound
                            // check below would break before testing a single voxel — skipping the
                            // block that holds the surface (the grazing-rim bug, 2026-07-17).
                            let mut voxel_dda = VoxelDda::seed_in_box(
                                direction,
                                safe,
                                voxel_entry,
                                box_enter,
                                1.0,
                                entry_axis,
                                block_min_voxel,
                                block_max_voxel - IVec3::ONE,
                            );
                            let band_z_lo = block_min_voxel.z.max(params.band_voxel_sv[0]);
                            let band_z_hi = block_max_voxel.z.min(params.band_voxel_sv[1]);
                            for _ in 0..MAX_VOXEL_STEPS {
                                if voxel_dda.cell.x < block_min_voxel.x
                                    || voxel_dda.cell.y < block_min_voxel.y
                                    || voxel_dda.cell.z < band_z_lo
                                    || voxel_dda.cell.x >= block_max_voxel.x
                                    || voxel_dda.cell.y >= block_max_voxel.y
                                    || voxel_dda.cell.z >= band_z_hi
                                {
                                    break;
                                }
                                let brick_local = voxel_dda.cell - block_min_voxel;
                                if voxel_occupied([brick_local.x, brick_local.y, brick_local.z]) {
                                    return (
                                        Some(MarchHit {
                                            absolute_voxel: [
                                                voxel_dda.cell.x + voxel_bias_x,
                                                voxel_dda.cell.y + voxel_bias_y,
                                                voxel_dda.cell.z + voxel_bias_z,
                                            ],
                                            face_normal: entry_face_normal(
                                                voxel_dda.entry_axis,
                                                direction,
                                            ),
                                        }),
                                        steps,
                                    );
                                }
                                voxel_dda.advance();
                            }
                        }
                    }
                }
            }
        }

        if block_dda.t_cell_enter > t_exit {
            break;
        }
        block_dda.advance();
    }

    (None, steps)
}

/// March one ray over an EXACT occupancy predicate — a flat voxel-level DDA (no blocks,
/// no records) inside the same frame/band, querying `occupied(absolute_voxel)`. The pure
/// kernel behind `voxel_worker::brick::cpu_march_exact_occupancy`, the parity
/// net's INDEPENDENT content oracle: the brick march's hit-voxel set must equal this
/// march's hit-voxel set. `ray` is the pixel-centre ray in the shifted march frame.
pub fn march_exact_occupancy<OccupiedFn>(
    ray: Ray,
    params: &ExactMarchParams,
    occupied: OccupiedFn,
) -> Option<MarchHit>
where
    OccupiedFn: Fn([i64; 3]) -> bool,
{
    let origin = ray.origin;
    let direction = ray.direction;
    let safe = guarded_direction(direction);
    let inverse = 1.0 / safe;
    let bounds_lo = params.traversal_lo;
    let bounds_hi = params.traversal_hi;
    let [voxel_bias_x, voxel_bias_y, voxel_bias_z] = params.voxel_bias;

    // Slab entry, kept inline: the initial entry-face axis is read off the per-axis
    // `t_near` (which the box-interval primitive does not surface), and the same `inverse`
    // feeds the DDA seed.
    let t_a = (bounds_lo - origin) * inverse;
    let t_b = (bounds_hi - origin) * inverse;
    let t_near = t_a.min(t_b);
    let t_far = t_a.max(t_b);
    let t_enter = t_near.x.max(t_near.y).max(t_near.z).max(0.0);
    let t_exit = t_far.x.min(t_far.y).min(t_far.z);
    if t_exit < t_enter {
        return None;
    }

    let entry_position = origin + direction * (t_enter + ENTRY_NUDGE);
    let initial_entry_axis = if t_near.x >= t_near.y && t_near.x >= t_near.z {
        0usize
    } else if t_near.y >= t_near.z {
        1
    } else {
        2
    };
    let mut dda = VoxelDda::seed(direction, safe, entry_position, t_enter, 1.0, initial_entry_axis);

    for _ in 0..MAX_EXACT_VOXEL_STEPS {
        // Band clip per voxel (the traversal AABB already bounds Z; the integer check keeps
        // float-edge voxels honest, mirroring the brick march's bound).
        if dda.cell.z >= params.band_voxel_sv[0] && dda.cell.z < params.band_voxel_sv[1] {
            let absolute = [
                (dda.cell.x + voxel_bias_x) as i64,
                (dda.cell.y + voxel_bias_y) as i64,
                (dda.cell.z + voxel_bias_z) as i64,
            ];
            if occupied(absolute) {
                return Some(MarchHit {
                    absolute_voxel: [
                        dda.cell.x + voxel_bias_x,
                        dda.cell.y + voxel_bias_y,
                        dda.cell.z + voxel_bias_z,
                    ],
                    face_normal: entry_face_normal(dda.entry_axis, direction),
                });
            }
        }
        if dda.t_cell_enter > t_exit {
            break;
        }
        dda.advance();
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single occupied voxel on the exact march: a ray fired straight at it along +x
    /// reports that voxel, with the entry-face normal pointing back down −x.
    #[test]
    fn exact_march_hits_the_single_occupied_voxel() {
        let params = ExactMarchParams {
            traversal_lo: Vec3::new(0.0, 0.0, 0.0),
            traversal_hi: Vec3::new(10.0, 10.0, 10.0),
            band_voxel_sv: [i32::MIN, i32::MAX],
            voxel_bias: [0, 0, 0],
        };
        let ray = Ray::new(Vec3::new(-5.0, 3.5, 3.5), Vec3::new(1.0, 0.0, 0.0));
        let hit = march_exact_occupancy(ray, &params, |v| v == [3, 3, 3]).expect("hit");
        assert_eq!(hit.absolute_voxel, [3, 3, 3]);
        assert_eq!(hit.face_normal, [-1, 0, 0]);
    }

    /// A ray that misses the (empty) traversal box entirely returns nothing.
    #[test]
    fn exact_march_misses_empty_volume() {
        let params = ExactMarchParams {
            traversal_lo: Vec3::ZERO,
            traversal_hi: Vec3::splat(10.0),
            band_voxel_sv: [i32::MIN, i32::MAX],
            voxel_bias: [0, 0, 0],
        };
        let ray = Ray::new(Vec3::new(-5.0, 100.0, 3.5), Vec3::new(1.0, 0.0, 0.0));
        assert!(march_exact_occupancy(ray, &params, |_| true).is_none());
    }

    /// The hierarchical march with the levels OFF (empty ⇒ occupied policy) still finds a
    /// coarse-solid block: the classifier reports one block coarse, and the first pierced
    /// voxel of it is the hit. The voxel bias offsets the absolute coordinate.
    #[test]
    fn hierarchical_march_hits_a_coarse_block_with_levels_off() {
        let edge = 8;
        let params = HierarchicalMarchParams {
            traversal_lo: Vec3::ZERO,
            traversal_hi: Vec3::splat(edge as f32),
            brick_edge_voxels: edge,
            block_bias: IVec3::ZERO,
            voxel_bias: [100, 0, 0],
            band_voxel_sv: [i32::MIN, i32::MAX],
            level_blocks_per_cell: vec![64, 8], // both "off" via the closure below
        };
        let ray = Ray::new(Vec3::new(-5.0, 4.5, 4.5), Vec3::new(1.0, 0.0, 0.0));
        let (hit, _steps) = march_brick_hierarchy(
            ray,
            &params,
            |_level, _block| true, // every level reports occupied → no skip (levels off)
            |block| {
                if block == [0, 0, 0] {
                    BlockContents::<fn([i32; 3]) -> bool>::CoarseSolid
                } else {
                    BlockContents::Empty
                }
            },
        );
        let hit = hit.expect("coarse block hit");
        // First voxel pierced along +x is x=0 in the block; +100 bias.
        assert_eq!(hit.absolute_voxel, [100, 4, 4]);
        assert_eq!(hit.face_normal, [-1, 0, 0]);
    }
}
