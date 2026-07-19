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
//! `t_max` — the ray parameter at which it crosses into the next cell along that axis.
//! Each step advances along the axis whose `t_max` is smallest (the nearest upcoming
//! cell boundary) and increments that cell coordinate by the sign of the direction.
//!
//! ## `t_max` is ANCHORED, not accumulated
//!
//! The paper advances `t_max` by adding a whole-cell span `t_delta` per step. We do NOT.
//! `t_max` is recomputed from the ray and the current cell every step:
//!
//! ```text
//! t_max[a] = (boundary_plane(cell[a], step[a]) · cell_edge − origin[a]) · inverse[a]
//! ```
//!
//! Algebraically the two are identical; in f32 they are not. Accumulation drifts — it is
//! a running sum, so error grows with the step count — while the anchored form has the
//! same short lever arm at step 200 as at step 1. That drift was a real bug: the flat
//! reference march seeds ONCE and accumulated ~11 ULP by the time it reached a surface,
//! while the brick march re-seeds at every block entry and barely drifted. On a ray
//! passing within a few ULP of a voxel EDGE the two disagreed about which of two
//! near-equal boundary crossings came first, stepped different axes, and reported hit
//! voxels one apart — `rim_diff` on Linux, where `sin`/`cos` in the camera basis differ
//! by an ULP from the MSVC CRT's and so select different near-degenerate rays.
//!
//! Anchoring makes `t_max` a pure function of `(origin, inverse, cell_edge, cell, step)`,
//! so ANY two cursors over the same ray agree bit-for-bit on the same cell no matter
//! where they were seeded or how many steps it took to get there. That is the property
//! `rim_diff` and `gpu_parity` actually need, and it holds by construction rather than
//! by luck. It is also strictly more accurate: checked in f64, the anchored values are
//! the ones near truth. `t_delta` is deliberately GONE rather than left unused — keeping
//! it is an open invitation to reintroduce the accumulating step.
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
/// [`t_max`](Self::t_max), the parameter at which the ray entered the current cell
/// ([`t_cell_enter`](Self::t_cell_enter)), and the axis of the face last crossed
/// ([`entry_axis`](Self::entry_axis), which the entry-face normal reads).
///
/// The cursor carries the ray it walks ([`origin`](Self::origin), [`inverse`](Self::inverse))
/// and its [`cell_edge`](Self::cell_edge) so [`advance`](Self::advance) can recompute
/// `t_max` from the cell rather than accumulate it — see the module docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoxelDda {
    /// The current cell's integer lattice coordinate.
    pub cell: IVec3,
    /// The per-axis step direction (`±1`), the sign of the ray direction on that axis.
    pub step: IVec3,
    /// The per-axis parameter of the next cell boundary ahead along the ray. A pure
    /// function of the fields below plus [`cell`](Self::cell) — never accumulated.
    pub t_max: Vec3,
    /// The ray parameter at which the ray entered the current cell (seeded to the
    /// entry `t`; updated to the crossed boundary's `t` on each [`advance`](Self::advance)).
    pub t_cell_enter: f32,
    /// The axis (0=x, 1=y, 2=z) of the cell face the ray last crossed to reach the
    /// current cell — the entry-face normal's axis.
    pub entry_axis: usize,
    /// The ray origin every `t_max` is measured from. Held so `t_max` can be re-derived
    /// at any cell instead of stepped forward.
    pub origin: Vec3,
    /// Per-axis reciprocal of the guarded direction (`1.0 / safe_direction`), the same
    /// value the slab entry and `clamped_box_entry` divide by, so the DDA's boundary
    /// parameters and the box-entry parameters are the same arithmetic.
    pub inverse: Vec3,
    /// The lattice cell edge (the brick edge for the block march, `1.0` for a voxel march).
    pub cell_edge: f32,
}

impl VoxelDda {
    /// Seed a cursor for a ray entering a lattice of cell edge `cell_edge` at world point
    /// `entry_point`, whose parameter along the ray is `entry_t`.
    ///
    /// `safe_direction` is the ray direction with any near-zero component nudged away from
    /// zero, sign intact ([`substrate::spatial::guarded_direction`]). It is the ONLY
    /// direction the cursor sees: both the per-axis step sign and the boundary parameters
    /// come from it, so the two cannot disagree about which way an axis runs. Passing the
    /// raw direction alongside it — as this used to — is what allowed that disagreement.
    ///
    /// `entry_point` and `entry_t` locate the ray's ENTRY into this lattice: the starting
    /// cell is `floor(entry_point / cell_edge)` and `t_cell_enter` starts at `entry_t`.
    /// `t_max`, by contrast, is measured from `origin` and does not consult either — see
    /// the module docs. That separation is deliberate and it matters: callers deliberately
    /// nudge `entry_point` forward off a face (`entry_t + ENTRY_NUDGE`) to land the floor
    /// inside the intended cell while passing the UN-nudged `entry_t`. Under the old
    /// entry-relative arithmetic that inconsistency leaked into `t_max` as a systematic
    /// one-nudge skew; anchoring on `origin` makes the seed exact regardless.
    ///
    /// `initial_entry_axis` records the face the ray entered the lattice through (the
    /// block march passes the traversal-box entry axis; the flat march the slab entry axis).
    #[allow(clippy::too_many_arguments)]
    pub fn seed(
        origin: Vec3,
        safe_direction: Vec3,
        entry_point: Vec3,
        entry_t: f32,
        cell_edge: f32,
        initial_entry_axis: usize,
    ) -> Self {
        let step = Self::step_of(safe_direction);
        let cell = (entry_point / cell_edge).floor().as_ivec3();
        Self::at(
            origin,
            safe_direction,
            cell,
            step,
            cell_edge,
            entry_t,
            initial_entry_axis,
        )
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
        origin: Vec3,
        safe_direction: Vec3,
        entry_point: Vec3,
        entry_t: f32,
        cell_edge: f32,
        initial_entry_axis: usize,
        cell_min: IVec3,
        cell_max: IVec3,
    ) -> Self {
        let step = Self::step_of(safe_direction);
        let cell = (entry_point / cell_edge)
            .floor()
            .as_ivec3()
            .clamp(cell_min, cell_max);
        Self::at(
            origin,
            safe_direction,
            cell,
            step,
            cell_edge,
            entry_t,
            initial_entry_axis,
        )
    }

    /// The per-axis step sign, taken from the GUARDED direction.
    ///
    /// Deliberately not the raw direction. `step` and `t_max` must agree about which way
    /// each axis runs, and `t_max` is built from `1 / safe_direction`; deriving `step`
    /// from a different vector is how they can disagree. The guarded direction is never
    /// zero and carries the raw direction's sign (`substrate::spatial::guarded_direction`
    /// preserves it), so `signum` here is always ±1 and always matches the reciprocal —
    /// by construction, for every input, including `±0.0`.
    fn step_of(safe_direction: Vec3) -> IVec3 {
        IVec3::new(
            safe_direction.x.signum() as i32,
            safe_direction.y.signum() as i32,
            safe_direction.z.signum() as i32,
        )
    }

    /// Build a cursor sitting in `cell`, with `t_max` derived from the ray. The ONE place
    /// a `VoxelDda` is constructed, so every seed and every step share one `t_max` rule.
    #[allow(clippy::too_many_arguments)]
    fn at(
        origin: Vec3,
        safe_direction: Vec3,
        cell: IVec3,
        step: IVec3,
        cell_edge: f32,
        t_cell_enter: f32,
        entry_axis: usize,
    ) -> Self {
        let inverse = Vec3::ONE / safe_direction;
        VoxelDda {
            cell,
            step,
            t_max: Self::anchored_t_max(origin, inverse, cell_edge, cell, step),
            t_cell_enter,
            entry_axis,
            origin,
            inverse,
            cell_edge,
        }
    }

    /// The per-axis parameter at which the ray leaves `cell` — the FAR face (`cell + 1`)
    /// when stepping positive, the NEAR face (`cell`) otherwise.
    ///
    /// Depends ONLY on the ray and the cell: no entry point, no running total, no step
    /// count. That is what makes two cursors over one ray agree bit-for-bit on a shared
    /// cell (module docs). Multiplies by `inverse` rather than dividing by
    /// `safe_direction` so it is the same operation `clamped_box_entry` and the slab
    /// entry perform — one arithmetic definition of "where does this ray cross that
    /// plane", not two that agree only by rounding luck.
    fn anchored_t_max(origin: Vec3, inverse: Vec3, cell_edge: f32, cell: IVec3, step: IVec3) -> Vec3 {
        let axis = |cell_coord: i32, step_axis: i32, origin_axis: f32, inverse_axis: f32| -> f32 {
            let boundary = if step_axis > 0 {
                (cell_coord + 1) as f32
            } else {
                cell_coord as f32
            };
            (boundary * cell_edge - origin_axis) * inverse_axis
        };
        Vec3::new(
            axis(cell.x, step.x, origin.x, inverse.x),
            axis(cell.y, step.y, origin.y, inverse.y),
            axis(cell.z, step.z, origin.z, inverse.z),
        )
    }

    /// Step to the next cell the ray pierces: advance along the axis whose `t_max` is
    /// smallest (ties broken x → y → z, see the module docs), move that cell coordinate
    /// by its step, record the crossed boundary's parameter in
    /// [`t_cell_enter`](Self::t_cell_enter) and the axis in [`entry_axis`](Self::entry_axis),
    /// then RE-DERIVE `t_max` for the cell just entered.
    ///
    /// Re-deriving all three axes, not just the stepped one, is deliberate: `t_max` is a
    /// pure function of the cell, and the two unstepped axes' cell coordinates did not
    /// change, so their values are bit-identical either way. Recomputing the whole vector
    /// leaves no path by which a component could carry accumulated error.
    pub fn advance(&mut self) {
        if self.t_max.x <= self.t_max.y && self.t_max.x <= self.t_max.z {
            self.cell.x += self.step.x;
            self.t_cell_enter = self.t_max.x;
            self.entry_axis = 0;
        } else if self.t_max.y <= self.t_max.z {
            self.cell.y += self.step.y;
            self.t_cell_enter = self.t_max.y;
            self.entry_axis = 1;
        } else {
            self.cell.z += self.step.z;
            self.t_cell_enter = self.t_max.z;
            self.entry_axis = 2;
        }
        self.t_max =
            Self::anchored_t_max(self.origin, self.inverse, self.cell_edge, self.cell, self.step);
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
        let safe = substrate::spatial::guarded_direction(dir);
        let entry = Vec3::new(finite_f32(1e3), finite_f32(1e3), finite_f32(1e3));
        let entry_t = finite_f32(1e6);
        let origin = Vec3::new(finite_f32(1e3), finite_f32(1e3), finite_f32(1e3));
        let dda =
            VoxelDda::seed_in_box(origin, safe, entry, entry_t, 1.0, axis, cell_min, cell_max);
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
        let safe = substrate::spatial::guarded_direction(Vec3::new(dx, dy, dz));
        let dda = VoxelDda::seed_in_box(
            entry,
            safe,
            entry,
            0.0,
            1.0,
            2,
            cell_min,
            cell_max,
        );
        assert!(dda.cell.z == 3, "max-face entry must land on the last in-box layer, not one past");
        assert!(dda.cell.x >= 0 && dda.cell.x <= 3);
        assert!(dda.cell.y >= 0 && dda.cell.y <= 3);
    }

    /// A symbolic DDA cursor over a symbolic ray, at a symbolic lattice cell far from `i32`
    /// saturation — the general pre-state for reasoning about a single
    /// [`advance`](VoxelDda::advance).
    ///
    /// Built through [`VoxelDda::at`], NOT by filling the fields independently. Since
    /// `t_max` became a pure function of `(origin, inverse, cell_edge, cell, step)`, a
    /// cursor whose `t_max` is free-floating is not a state any seed or step can produce,
    /// and proving things about it would prove nothing about the real march. Constructing
    /// through `at` also ties `step` to `sign(direction)` and `inverse` to `1/safe`, which
    /// is exactly the relation the monotonicity proof below needs and which independent
    /// symbolic fields silently broke.
    fn arbitrary_cursor() -> VoxelDda {
        let bounded_cell = || -> i32 {
            let value: i32 = kani::any();
            kani::assume(value >= -(1 << 24) && value <= (1 << 24));
            value
        };
        // The REAL guard, not a hand-written model of it. Modelling it here is what let the
        // proof assume a sign-preserving guard while production shipped one that flipped
        // `-0.0` to `+1e-20`; calling the production function makes that impossible.
        let safe = substrate::spatial::guarded_direction(Vec3::new(
            finite_f32(1e3),
            finite_f32(1e3),
            finite_f32(1e3),
        ));
        let origin = Vec3::new(finite_f32(1e3), finite_f32(1e3), finite_f32(1e3));
        let cell_edge = {
            let value = finite_f32(1e3);
            kani::assume(value > 0.0);
            value
        };
        VoxelDda::at(
            origin,
            safe,
            IVec3::new(bounded_cell(), bounded_cell(), bounded_cell()),
            VoxelDda::step_of(safe),
            cell_edge,
            finite_f32(1e18),
            0,
        )
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

    /// **NOT a Kani harness — see [`super::tests::advance_is_monotone_and_preserves_the_invariant`].**
    ///
    /// `advance` re-derives `t_max` from the ray, so any property about its RESULT drags the
    /// anchored float arithmetic into the solver. Kani discharges the two structural
    /// properties above — which axis moves, and which axis is selected — because those turn
    /// on comparisons and integer updates. The forward-progress property additionally needs
    /// float multiply to be order-preserving, and that is the part the bundled bit-blasting
    /// CaDiCaL cannot close: it ran past 36 minutes unterminated, and still exceeded a
    /// 3-minute bound after the per-axis lemma was supplied as a hypothesis (the hypothesis
    /// constrains the value but does not remove the arithmetic).
    ///
    /// Covered by a deterministic sweep instead. Note this is a bound on the SOLVER, not a
    /// gap in the property: the same argument the harness encoded still holds, and the two
    /// structural halves that BMC can check are still checked here.
    fn _monotonicity_is_proved_by_sweep_not_bmc() {}

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
        // Through the real guard, not a hand-written stand-in: the DDA's step is derived
        // from this vector, so a test that fabricates it would not exercise the agreement
        // between `step` and `1 / safe`.
        let safe = substrate::spatial::guarded_direction(Vec3::new(1.0, 0.0, 0.0));
        let mut dda = VoxelDda::seed(
            Vec3::new(0.5, 0.5, 0.5),
            safe,
            Vec3::new(0.5, 0.5, 0.5),
            0.0,
            1.0,
            0,
        );
        assert_eq!(dda.cell, IVec3::new(0, 0, 0));
        dda.advance();
        assert_eq!(dda.cell, IVec3::new(1, 0, 0));
        assert_eq!(dda.entry_axis, 0);
        dda.advance();
        assert_eq!(dda.cell, IVec3::new(2, 0, 0));
    }

    /// **The per-axis lemma the DDA's forward-progress invariant rests on: stepping an axis
    /// never pulls its boundary parameter backwards.**
    ///
    /// `advance` sets `t_cell_enter` to the crossed boundary and re-derives `t_max` for the new
    /// cell. For the walk to move forward, the stepped axis's new `t_max` must be at or ahead
    /// of the one just consumed. That holds because stepping moves the boundary plane one cell
    /// edge FURTHER along the ray, and `(plane − origin) · inverse` is monotone in `plane` —
    /// *provided* `step` and `inverse` share the axis's sign. That proviso is exactly what
    /// [`substrate::spatial::guarded_direction`]'s sign preservation buys: a guard that forced
    /// a positive magnitude broke it for any component in `[-GUARD, 0]`, and with `t_max`
    /// anchored (no `abs` anywhere to re-correct it) such an axis marched backwards forever.
    ///
    /// A sweep rather than a Kani proof because the property is float multiplication over a
    /// symbolic `i32 → f32` conversion, which the available bit-blasting solver cannot
    /// discharge — see `kani_proofs::_anchored_t_max_lemma_is_proved_by_sweep_not_bmc`. The
    /// direction list covers both signs at ordinary magnitude AND both signs BELOW the guard,
    /// including `-0.0`, which is the case that regressed.
    #[test]
    fn anchored_t_max_does_not_decrease_along_the_step() {
        let directions = [
            1.0f32, -1.0, 0.37, -0.37, 1e-3, -1e-3, 1e-30, -1e-30, 0.0, -0.0,
        ];
        let edges = [1.0f32, 8.0, 16.0, 32.0];
        let origins = [-1234.5f32, -7.25, -0.5, 0.0, 0.5, 7.25, 1234.5, 65536.0];
        let cells = [-1048576i32, -4096, -17, -1, 0, 1, 17, 4096, 1048576];
        let mut cases = 0u32;
        for &d in &directions {
            // Through the production guard: the sign agreement under test is ITS job.
            let safe = substrate::spatial::guarded_direction(Vec3::new(d, 1.0, 1.0));
            let step = VoxelDda::step_of(safe);
            // `step` comes from `safe`, so its agreement with `1 / safe` is structural, not
            // something this sweep could falsify — that is the point of deriving it from the
            // guarded vector. Whether `safe` still points the way the CALLER aimed is the
            // guard's own contract, covered by
            // `substrate`'s `guarded_direction_preserves_sign_and_escapes_zero`.
            assert!(step.x == 1 || step.x == -1, "step must be ±1 for d={d}");
            for &edge in &edges {
                for &origin_x in &origins {
                    for &cell_x in &cells {
                        let origin = Vec3::new(origin_x, 0.0, 0.0);
                        let before =
                            VoxelDda::at(origin, safe, IVec3::new(cell_x, 0, 0), step, edge, 0.0, 0);
                        let after = VoxelDda::at(
                            origin,
                            safe,
                            IVec3::new(cell_x + step.x, 0, 0),
                            step,
                            edge,
                            0.0,
                            0,
                        );
                        assert!(
                            after.t_max.x >= before.t_max.x,
                            "stepping pulled t_max backwards: d={d} edge={edge} \
                             origin={origin_x} cell={cell_x} before={} after={}",
                            before.t_max.x,
                            after.t_max.x
                        );
                        cases += 1;
                    }
                }
            }
        }
        assert!(cases >= 2000, "sweep unexpectedly small: {cases}");
    }

    /// **A walk only ever moves forward, and every step re-establishes the DDA's ordering
    /// invariant** (`t_max >= t_cell_enter` on all three axes).
    ///
    /// This is the "each pierced cell is entered once, in increasing `t`" half of Amanatides
    /// & Woo. It is checked over whole WALKS rather than a single step, so a violation that
    /// only appears after the cursor has moved — the failure mode an anchored `t_max` is
    /// meant to rule out, and the one an accumulating `t_max` produced — is in scope.
    ///
    /// A sweep and not a Kani harness: see
    /// `kani_proofs::_monotonicity_is_proved_by_sweep_not_bmc`. Kani still proves the two
    /// structural halves (which axis moves, which axis is selected) symbolically.
    #[test]
    fn advance_is_monotone_and_preserves_the_invariant() {
        let directions = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, -0.0),
            Vec3::new(0.7, -0.5, 0.3),
            Vec3::new(-0.2, 0.9, -0.4),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1e-30, 1.0, -1e-30),
            Vec3::new(-0.0, -0.0, 1.0),
        ];
        let edges = [1.0f32, 16.0];
        let origins = [
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-13.25, 7.5, 101.0),
            Vec3::new(1024.0, -1024.0, 0.0),
        ];
        let mut steps_checked = 0u32;
        for direction in directions {
            let safe = substrate::spatial::guarded_direction(direction);
            for &edge in &edges {
                for origin in origins {
                    let mut dda = VoxelDda::seed(origin, safe, origin, 0.0, edge, 0);
                    // The invariant must hold at the seed, or the walk starts out of contract.
                    assert!(
                        dda.t_max.x >= dda.t_cell_enter
                            && dda.t_max.y >= dda.t_cell_enter
                            && dda.t_max.z >= dda.t_cell_enter,
                        "seed violated the invariant: dir={direction:?} edge={edge} origin={origin:?}"
                    );
                    for _ in 0..64 {
                        let entry_before = dda.t_cell_enter;
                        dda.advance();
                        assert!(
                            dda.t_cell_enter >= entry_before,
                            "walk moved BACKWARD: dir={direction:?} edge={edge} origin={origin:?} \
                             {entry_before} -> {}",
                            dda.t_cell_enter
                        );
                        assert!(
                            dda.t_max.x >= dda.t_cell_enter
                                && dda.t_max.y >= dda.t_cell_enter
                                && dda.t_max.z >= dda.t_cell_enter,
                            "step broke the invariant: dir={direction:?} edge={edge} \
                             origin={origin:?} t_max={:?} enter={}",
                            dda.t_max,
                            dda.t_cell_enter
                        );
                        steps_checked += 1;
                    }
                }
            }
        }
        assert!(steps_checked >= 2000, "sweep unexpectedly small: {steps_checked}");
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
                                    let safe = substrate::spatial::guarded_direction(dir);

                                    // Box-confined seed: MUST land inside the box.
                                    let confined =
                                        VoxelDda::seed_in_box(entry, safe, entry, 0.0, 1.0, axis, cell_min, cell_max);
                                    assert!(
                                        in_box(confined.cell, cell_min, cell_max),
                                        "seed_in_box left the box: cell={:?} entry={entry:?} dir={dir:?}",
                                        confined.cell
                                    );

                                    // Reference: the unconfined flat DDA skipped forward to the box.
                                    let mut flat = VoxelDda::seed(entry, safe, entry, 0.0, 1.0, axis);
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
        let mut dda = VoxelDda::seed(
            Vec3::new(0.01, 0.01, 0.01),
            safe,
            Vec3::new(0.01, 0.01, 0.01),
            0.0,
            1.0,
            0,
        );
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
        // Through the real guard, not a hand-written stand-in: the DDA's step is derived
        // from this vector, so a test that fabricates it would not exercise the agreement
        // between `step` and `1 / safe`.
        let safe = substrate::spatial::guarded_direction(Vec3::new(1.0, 0.0, 0.0));
        let mut dda = VoxelDda::seed(
            Vec3::new(0.0, 0.5, 0.5),
            safe,
            Vec3::new(0.0, 0.5, 0.5),
            0.0,
            8.0,
            0,
        );
        assert_eq!(dda.cell, IVec3::new(0, 0, 0));
        dda.advance();
        assert_eq!(dda.cell, IVec3::new(1, 0, 0));
        assert!((dda.t_cell_enter - 8.0).abs() < 1e-4);
    }
}
