//! The Amanatides & Woo fast voxel traversal stepping loop, as a reusable component.
//!
//! A [`VoxelDda`] walks a ray through a uniform lattice of cells one cell at a time,
//! visiting exactly the cells the ray pierces, in order. It is used at two scales in
//! the brick march — once over the block lattice (cell edge = the brick edge) and once
//! over the voxel lattice inside a resident sculpted brick (cell edge = 1) — and again
//! in the flat exact-occupancy reference march; the same seeding and stepping serve all
//! three.
//!
//! ## Literature
//!
//! Amanatides & Woo, *A Fast Voxel Traversal Algorithm for Ray Tracing* (Eurographics
//! 1987). The ray is `p(t) = origin + t · direction`. For each axis the traversal keeps
//! `t_max` — the ray parameter at which it crosses into the next cell along that axis —
//! and `t_delta` — the parameter span of one whole cell along that axis. Each step
//! advances along the axis whose `t_max` is smallest (the nearest upcoming cell
//! boundary), increments that cell coordinate by the sign of the direction, and adds
//! `t_delta` to that axis's `t_max`. The seeds are `t_delta = |cell_edge / direction|`
//! and, per axis, the parameter to the first boundary ahead of the entry point.
//!
//! ## The tie order is load-bearing
//!
//! When two (or three) `t_max` components are equal — the ray crosses a cell corner or
//! edge exactly — the advance MUST pick the axis in the order **x, then y, then z**
//! (`t_max.x <= t_max.y && t_max.x <= t_max.z`, else `t_max.y <= t_max.z`, else z). The
//! WGSL shader mirror and the CPU march are held byte-identical by `gpu_parity`, and
//! this tie order is one of the exact arithmetic details that keeps them so: changing
//! it would move which voxel a corner-grazing ray reports. Do not "simplify" it to a
//! plain `min`.

use glam::{IVec3, Vec3};

/// One Amanatides & Woo voxel-traversal cursor over a uniform cell lattice: the current
/// integer cell, the per-axis step sign, the per-axis distance-to-next-boundary
/// [`t_max`](Self::t_max) and whole-cell span [`t_delta`](Self::t_delta), the parameter
/// at which the ray entered the current cell ([`t_cell_enter`](Self::t_cell_enter)), and
/// the axis of the face last crossed ([`entry_axis`](Self::entry_axis), which the entry-
/// face normal reads).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoxelDda {
    /// The current cell's integer lattice coordinate.
    pub cell: IVec3,
    /// The per-axis step direction (`±1`), the sign of the ray direction on that axis.
    pub step: IVec3,
    /// The per-axis parameter of the next cell boundary ahead along the ray.
    pub t_max: Vec3,
    /// The per-axis parameter span of one whole cell (`|cell_edge / direction|`).
    pub t_delta: Vec3,
    /// The ray parameter at which the ray entered the current cell (seeded to the
    /// entry `t`; updated to the crossed boundary's `t` on each [`advance`](Self::advance)).
    pub t_cell_enter: f32,
    /// The axis (0=x, 1=y, 2=z) of the cell face the ray last crossed to reach the
    /// current cell — the entry-face normal's axis.
    pub entry_axis: usize,
}

impl VoxelDda {
    /// Seed a cursor for a ray entering a lattice of cell edge `cell_edge` at world point
    /// `entry_point`, whose parameter along the ray is `entry_t`.
    ///
    /// `direction` is the ray direction (its per-axis sign gives the step); `safe_direction`
    /// is the same direction with any near-zero component nudged away from zero (see
    /// [`crate::brick_march`]), used as the divisor so the seeds stay finite. The current
    /// cell is `floor(entry_point / cell_edge)`; each axis's `t_max` is the parameter from
    /// the entry point to the first cell boundary ahead of it, offset by `entry_t` so the
    /// cursor's parameters are absolute along the ray. `initial_entry_axis` records the face
    /// the ray entered the lattice through (the block march passes the traversal-box entry
    /// axis; the flat march passes the slab entry axis).
    pub fn seed(
        direction: Vec3,
        safe_direction: Vec3,
        entry_point: Vec3,
        entry_t: f32,
        cell_edge: f32,
        initial_entry_axis: usize,
    ) -> Self {
        let step = IVec3::new(
            direction.x.signum() as i32,
            direction.y.signum() as i32,
            direction.z.signum() as i32,
        );
        let t_delta = (Vec3::splat(cell_edge) / safe_direction).abs();
        let cell = (entry_point / cell_edge).floor().as_ivec3();
        let t_max = Self::seed_t_max(cell, step, entry_point, entry_t, cell_edge, safe_direction);
        VoxelDda {
            cell,
            step,
            t_max,
            t_delta,
            t_cell_enter: entry_t,
            entry_axis: initial_entry_axis,
        }
    }

    /// Like [`seed`](Self::seed), but the entry cell is CLAMPED into the inclusive box
    /// `[cell_min, cell_max]` (with `t_max` recomputed consistently for the clamped cell).
    /// This corrects the classic box-entry hazard: a ray entering a box through one of its
    /// MAX faces lands exactly on that face, so `floor(entry)` falls one cell PAST the box on
    /// that axis. A per-box-confined march (the brick's inner voxel DDA, whose bounds check
    /// then reads the seed as already-exited) would skip the box entirely — the grazing-rim
    /// bug (2026-07-17). Because the ray genuinely occupies the clamped cell at `entry_t`, the
    /// clamp is always sound; `t_max` derives from the clamped cell so an empty seed steps on
    /// correctly. MUST stay mirrored by the WGSL inner voxel-DDA seed (`gpu_parity`).
    #[allow(clippy::too_many_arguments)]
    pub fn seed_in_box(
        direction: Vec3,
        safe_direction: Vec3,
        entry_point: Vec3,
        entry_t: f32,
        cell_edge: f32,
        initial_entry_axis: usize,
        cell_min: IVec3,
        cell_max: IVec3,
    ) -> Self {
        let step = IVec3::new(
            direction.x.signum() as i32,
            direction.y.signum() as i32,
            direction.z.signum() as i32,
        );
        let t_delta = (Vec3::splat(cell_edge) / safe_direction).abs();
        let cell = (entry_point / cell_edge)
            .floor()
            .as_ivec3()
            .clamp(cell_min, cell_max);
        let t_max = Self::seed_t_max(cell, step, entry_point, entry_t, cell_edge, safe_direction);
        VoxelDda {
            cell,
            step,
            t_max,
            t_delta,
            t_cell_enter: entry_t,
            entry_axis: initial_entry_axis,
        }
    }

    /// The per-axis `t_max` seed: the ray parameter from `entry_point` to the first cell
    /// boundary ahead of `cell` — the FAR face (`cell + 1`) when stepping positive, the NEAR
    /// face (`cell`) otherwise — offset by `entry_t` so the cursor's parameters are absolute.
    fn seed_t_max(
        cell: IVec3,
        step: IVec3,
        entry_point: Vec3,
        entry_t: f32,
        cell_edge: f32,
        safe_direction: Vec3,
    ) -> Vec3 {
        let seed_axis = |cell_coord: i32, step_axis: i32, entry_axis: f32, safe_axis: f32| -> f32 {
            if step_axis > 0 {
                ((cell_coord + 1) as f32 * cell_edge - entry_axis) / safe_axis
            } else {
                (cell_coord as f32 * cell_edge - entry_axis) / safe_axis
            }
        };
        Vec3::new(
            seed_axis(cell.x, step.x, entry_point.x, safe_direction.x) + entry_t,
            seed_axis(cell.y, step.y, entry_point.y, safe_direction.y) + entry_t,
            seed_axis(cell.z, step.z, entry_point.z, safe_direction.z) + entry_t,
        )
    }

    /// Step to the next cell the ray pierces: advance along the axis whose `t_max` is
    /// smallest (ties broken x → y → z, see the module docs), move that cell coordinate
    /// by its step, record the crossed boundary's parameter in
    /// [`t_cell_enter`](Self::t_cell_enter) and the axis in [`entry_axis`](Self::entry_axis),
    /// and push that axis's `t_max` forward by one whole cell.
    pub fn advance(&mut self) {
        if self.t_max.x <= self.t_max.y && self.t_max.x <= self.t_max.z {
            self.cell.x += self.step.x;
            self.t_cell_enter = self.t_max.x;
            self.t_max.x += self.t_delta.x;
            self.entry_axis = 0;
        } else if self.t_max.y <= self.t_max.z {
            self.cell.y += self.step.y;
            self.t_cell_enter = self.t_max.y;
            self.t_max.y += self.t_delta.y;
            self.entry_axis = 1;
        } else {
            self.cell.z += self.step.z;
            self.t_cell_enter = self.t_max.z;
            self.t_max.z += self.t_delta.z;
            self.entry_axis = 2;
        }
    }
}

/// Kani bounded-model-checking proofs of the box-entry invariant behind
/// [`VoxelDda::seed_in_box`] (the grazing-rim fix, 2026-07-17). Unlike the differential
/// render / the deterministic sweep (which only catch the bug on the scenes they happen to
/// sample), these verify the postcondition over the WHOLE bounded input space — every finite
/// direction and entry point — so the guarantee does not depend on luck. `#[cfg(kani)]` keeps
/// them out of ordinary builds/tests. Run under WSL: `cargo kani -p raycast`.
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// A finite, magnitude-bounded symbolic `f32` (excludes NaN/±inf and absurd magnitudes,
    /// so the solver reasons over the real geometric domain).
    fn finite_f32(max_abs: f32) -> f32 {
        let value: f32 = kani::any();
        kani::assume(value.is_finite() && value.abs() <= max_abs);
        value
    }

    /// **Safety invariant.** For ANY finite ray and entry point, `seed_in_box` returns a cell
    /// inside the box — the guarantee the per-block-confined inner voxel DDA relies on (a seed
    /// OUTSIDE the box makes its bound check break before testing a voxel, which was the bug).
    #[kani::proof]
    fn seed_in_box_cell_is_always_within_the_box() {
        // A representative unit-cell box `[0, 3]^3` (the property is translation/edge invariant).
        let cell_min = IVec3::splat(0);
        let cell_max = IVec3::splat(3);
        let axis: usize = kani::any();
        kani::assume(axis < 3);
        let dir = Vec3::new(finite_f32(1e3), finite_f32(1e3), finite_f32(1e3));
        // The march guards near-zero components up to a floor before dividing (SLAB guard).
        let g = |d: f32| if d.abs() < 1e-20 { 1e-20 } else { d };
        let safe = Vec3::new(g(dir.x), g(dir.y), g(dir.z));
        let entry = Vec3::new(finite_f32(1e3), finite_f32(1e3), finite_f32(1e3));
        let entry_t = finite_f32(1e6);
        let dda = VoxelDda::seed_in_box(dir, safe, entry, entry_t, 1.0, axis, cell_min, cell_max);
        assert!(dda.cell.x >= cell_min.x && dda.cell.x <= cell_max.x);
        assert!(dda.cell.y >= cell_min.y && dda.cell.y <= cell_max.y);
        assert!(dda.cell.z >= cell_min.z && dda.cell.z <= cell_max.z);
    }

    /// **The fix, directly.** A ray entering the box through its MAX-Z face (entry exactly on
    /// `z = 4`, the integer face a plain `floor` sends one cell PAST to `z = 4`) seeds onto the
    /// LAST in-box layer `z = 3` — for every in-box lateral entry and every descending
    /// direction. This is the grazing-rim staircase's root cause, proved absent.
    #[kani::proof]
    fn seed_in_box_max_face_entry_lands_on_the_last_cell() {
        let cell_min = IVec3::splat(0);
        let cell_max = IVec3::splat(3);
        let ex = finite_f32(3.999);
        let ey = finite_f32(3.999);
        kani::assume(ex >= 0.0 && ey >= 0.0);
        let entry = Vec3::new(ex, ey, 4.0); // exactly on the max-Z face
        let dx = finite_f32(1e3);
        let dy = finite_f32(1e3);
        let dz = finite_f32(1e3);
        kani::assume(dz < -1e-6); // descending into the box through the top
        let safe = Vec3::new(
            if dx.abs() < 1e-20 { 1e-20 } else { dx },
            if dy.abs() < 1e-20 { 1e-20 } else { dy },
            dz,
        );
        let dda = VoxelDda::seed_in_box(Vec3::new(dx, dy, dz), safe, entry, 0.0, 1.0, 2, cell_min, cell_max);
        assert!(dda.cell.z == 3, "max-face entry must land on the last in-box layer, not one past");
        assert!(dda.cell.x >= 0 && dda.cell.x <= 3);
        assert!(dda.cell.y >= 0 && dda.cell.y <= 3);
    }

    /// A symbolic DDA cursor with finite, magnitude-bounded float parameters, a signum step
    /// (`-1..=1`, what [`VoxelDda::seed`] derives from the ray direction), and a lattice cell far
    /// from `i32` saturation — the general pre-state for reasoning about a single
    /// [`advance`](VoxelDda::advance). The bounds keep the solver on the real geometric domain
    /// and away from float/overflow corners no seeded cursor can reach.
    fn arbitrary_cursor() -> VoxelDda {
        let bounded_cell = || -> i32 {
            let value: i32 = kani::any();
            kani::assume(value >= -(1 << 24) && value <= (1 << 24));
            value
        };
        let signum_step = || -> i32 {
            let value: i32 = kani::any();
            kani::assume((-1..=1).contains(&value));
            value
        };
        VoxelDda {
            cell: IVec3::new(bounded_cell(), bounded_cell(), bounded_cell()),
            step: IVec3::new(signum_step(), signum_step(), signum_step()),
            t_max: Vec3::new(finite_f32(1e18), finite_f32(1e18), finite_f32(1e18)),
            t_delta: Vec3::new(finite_f32(1e18), finite_f32(1e18), finite_f32(1e18)),
            t_cell_enter: finite_f32(1e18),
            entry_axis: 0,
        }
    }

    /// **Each `advance` moves exactly one cell axis, by that axis's step, leaving the other two
    /// fixed** — the structural core of "the DDA visits the pierced cells one at a time and never
    /// skips a cell laterally" (Amanatides & Woo). The axis that moved is exactly the one named by
    /// the updated `entry_axis`, so the entry-face normal the shade reads always matches the step
    /// just taken.
    #[kani::proof]
    fn advance_moves_exactly_one_axis_by_its_step() {
        let before = arbitrary_cursor();
        let mut after = before;
        after.advance();
        assert!(after.entry_axis < 3);
        let moved_x = after.cell.x - before.cell.x;
        let moved_y = after.cell.y - before.cell.y;
        let moved_z = after.cell.z - before.cell.z;
        match after.entry_axis {
            0 => assert!(moved_x == before.step.x && moved_y == 0 && moved_z == 0),
            1 => assert!(moved_y == before.step.y && moved_x == 0 && moved_z == 0),
            _ => assert!(moved_z == before.step.z && moved_x == 0 && moved_y == 0),
        }
    }

    /// **`advance` never moves the ray backward, and it preserves the DDA's ordering invariant.**
    /// Precondition — the invariant every seeded cursor satisfies: each axis's next-boundary
    /// parameter is at or ahead of the current cell-entry parameter, and the whole-cell spans are
    /// non-negative. Then after one advance the new cell-entry parameter is ≥ the old one (`t` is
    /// monotone non-decreasing — the march only ever moves forward), AND the same invariant holds
    /// again. Because the step preserves the invariant, induction extends the monotonicity to the
    /// entire walk, however many cells it crosses — the "each pierced cell is entered once, in
    /// increasing `t`" half of Amanatides & Woo, discharged as a one-step obligation.
    #[kani::proof]
    fn advance_is_monotone_in_t_and_preserves_the_invariant() {
        let mut dda = arbitrary_cursor();
        kani::assume(dda.t_delta.x >= 0.0 && dda.t_delta.y >= 0.0 && dda.t_delta.z >= 0.0);
        kani::assume(
            dda.t_max.x >= dda.t_cell_enter
                && dda.t_max.y >= dda.t_cell_enter
                && dda.t_max.z >= dda.t_cell_enter,
        );
        let entry_before = dda.t_cell_enter;
        dda.advance();
        assert!(dda.t_cell_enter >= entry_before);
        assert!(
            dda.t_max.x >= dda.t_cell_enter
                && dda.t_max.y >= dda.t_cell_enter
                && dda.t_max.z >= dda.t_cell_enter
        );
    }

    /// **The advance selects the axis of minimum `t_max`, breaking ties x → y → z, and records
    /// that axis in `entry_axis`.** This pins the load-bearing tie order (module docs) as a
    /// checked contract: the stepped axis's `t_max` is ≤ the other two, and when several are equal
    /// the earliest of x, y, z wins — the exact rule the WGSL mirror shares, so a corner-grazing
    /// ray reports the same voxel on CPU and GPU. (`arbitrary_cursor` seeds finite `t_max`, so the
    /// strict inequalities below are well-defined — no NaN branch.)
    #[kani::proof]
    fn advance_selects_min_t_max_with_x_then_y_then_z_ties() {
        let mut dda = arbitrary_cursor();
        let t = dda.t_max;
        dda.advance();
        match dda.entry_axis {
            0 => assert!(t.x <= t.y && t.x <= t.z),
            1 => assert!(t.y <= t.z && t.y < t.x),
            _ => assert!(t.z < t.x && t.z < t.y),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ray fired straight along +x from just outside the origin cell visits the cells
    /// in ascending x order, one per step, and never moves off the x axis.
    #[test]
    fn axis_aligned_ray_walks_one_cell_per_step() {
        let direction = Vec3::new(1.0, 0.0, 0.0);
        let safe = Vec3::new(1.0, 1e-20, 1e-20);
        let mut dda = VoxelDda::seed(direction, safe, Vec3::new(0.5, 0.5, 0.5), 0.0, 1.0, 0);
        assert_eq!(dda.cell, IVec3::new(0, 0, 0));
        dda.advance();
        assert_eq!(dda.cell, IVec3::new(1, 0, 0));
        assert_eq!(dda.entry_axis, 0);
        dda.advance();
        assert_eq!(dda.cell, IVec3::new(2, 0, 0));
    }

    fn in_box(cell: IVec3, lo: IVec3, hi_inclusive: IVec3) -> bool {
        cell.x >= lo.x
            && cell.y >= lo.y
            && cell.z >= lo.z
            && cell.x <= hi_inclusive.x
            && cell.y <= hi_inclusive.y
            && cell.z <= hi_inclusive.z
    }

    /// **The box-entry invariant behind [`VoxelDda::seed_in_box`]** (the grazing-rim fix,
    /// 2026-07-17). A ray entering a box lands its FIRST in-box voxel at the SAME cell whether
    /// seeded box-confined or traversed by the unconfined flat DDA skipped forward into the box.
    /// A plain `floor(entry)` seed falls one cell PAST a MAX face at grazing (the entry
    /// coordinate sits exactly on the integer face), which a per-box-confined march then reads
    /// as already-exited — the bug that block-stepped the tube rim. This sweeps the failure
    /// class DETERMINISTICALLY: every one of the six faces, entered at grazing incidence from a
    /// dense grid of directions and positions — so the guard does NOT depend on a differential
    /// render happening to sample a lucky scene+camera (which is how the head-on parity case
    /// missed this for months). Kani proves the same postcondition over the bounded input space.
    #[test]
    fn seed_in_box_matches_flat_dda_first_in_box_cell() {
        let edge = 16.0f32;
        let cell_min = IVec3::splat(0);
        let cell_max = IVec3::splat(15); // inclusive: box cells [0,15]^3 for a 16-voxel block
        let guard = |d: f32| if d.abs() < 1e-20 { 1e-20 } else { d };
        let perp_positions = [0.3f32, 2.5, 7.5, 12.5, 15.7];
        let along_small = [0.03f32, 0.2, 1.0, 3.0]; // includes grazing (near-parallel) incidence
        let perp_dirs = [-3.0f32, -1.0, -0.2, 0.2, 1.0, 3.0];
        let mut cases = 0u32;
        for axis in 0..3usize {
            let b = (axis + 1) % 3;
            let c = (axis + 2) % 3;
            // Both faces on this axis: the MIN face (coord 0, ray points +axis) and the MAX
            // face (coord `edge`, ray points −axis, where a plain floor seeds one cell past).
            for &(face_coord, along_sign) in &[(0.0f32, 1.0f32), (edge, -1.0f32)] {
                for &along in &along_small {
                    for &pb in &perp_positions {
                        for &pc in &perp_positions {
                            for &db in &perp_dirs {
                                for &dc in &perp_dirs {
                                    let mut entry = [0.0f32; 3];
                                    entry[axis] = face_coord;
                                    entry[b] = pb;
                                    entry[c] = pc;
                                    let entry = Vec3::from_array(entry);
                                    let mut dir = [0.0f32; 3];
                                    dir[axis] = along_sign * along;
                                    dir[b] = db;
                                    dir[c] = dc;
                                    let dir = Vec3::from_array(dir).normalize();
                                    let safe = Vec3::new(guard(dir.x), guard(dir.y), guard(dir.z));

                                    // Box-confined seed: MUST land inside the box.
                                    let confined =
                                        VoxelDda::seed_in_box(dir, safe, entry, 0.0, 1.0, axis, cell_min, cell_max);
                                    assert!(
                                        in_box(confined.cell, cell_min, cell_max),
                                        "seed_in_box left the box: cell={:?} entry={entry:?} dir={dir:?}",
                                        confined.cell
                                    );

                                    // Reference: the unconfined flat DDA skipped forward to the box.
                                    let mut flat = VoxelDda::seed(dir, safe, entry, 0.0, 1.0, axis);
                                    let mut steps = 0;
                                    while !in_box(flat.cell, cell_min, cell_max) && steps < 64 {
                                        flat.advance();
                                        steps += 1;
                                    }
                                    assert!(
                                        in_box(flat.cell, cell_min, cell_max),
                                        "flat DDA never entered the box: entry={entry:?} dir={dir:?}"
                                    );
                                    assert_eq!(
                                        confined.cell, flat.cell,
                                        "box-confined seed != flat first-in-box cell; axis={axis} entry={entry:?} dir={dir:?}"
                                    );
                                    cases += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(cases >= 500, "sweep unexpectedly small: {cases}");
    }

    /// A ray through the exact cell corner (equal `t_max` on all three axes) advances
    /// x first, then y, then z — the load-bearing tie order the shader mirror shares.
    #[test]
    fn corner_grazing_ray_breaks_ties_x_then_y_then_z() {
        let direction = Vec3::new(1.0, 1.0, 1.0).normalize();
        let safe = direction;
        // Entry at the origin corner: the first boundary on every axis is at the same t.
        let mut dda = VoxelDda::seed(direction, safe, Vec3::new(0.01, 0.01, 0.01), 0.0, 1.0, 0);
        dda.advance();
        assert_eq!(dda.cell, IVec3::new(1, 0, 0), "x wins the first tie");
        assert_eq!(dda.entry_axis, 0);
        dda.advance();
        assert_eq!(dda.cell, IVec3::new(1, 1, 0), "y wins the second tie");
        assert_eq!(dda.entry_axis, 1);
        dda.advance();
        assert_eq!(dda.cell, IVec3::new(1, 1, 1), "z takes the third");
        assert_eq!(dda.entry_axis, 2);
    }

    /// A larger cell edge scales the whole-cell span: stepping across an 8-wide cell
    /// advances the integer cell by one and `t_cell_enter` by ~8 along a unit-x ray.
    #[test]
    fn cell_edge_scales_the_step_span() {
        let direction = Vec3::new(1.0, 0.0, 0.0);
        let safe = Vec3::new(1.0, 1e-20, 1e-20);
        let mut dda = VoxelDda::seed(direction, safe, Vec3::new(0.0, 0.5, 0.5), 0.0, 8.0, 0);
        assert_eq!(dda.cell, IVec3::new(0, 0, 0));
        dda.advance();
        assert_eq!(dda.cell, IVec3::new(1, 0, 0));
        assert!((dda.t_cell_enter - 8.0).abs() < 1e-4);
    }
}
