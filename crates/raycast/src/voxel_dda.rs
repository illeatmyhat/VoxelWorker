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
        // Parameter from the entry point to the first cell boundary ahead on each axis:
        // the FAR face (cell + 1) when stepping positive, the NEAR face (cell) otherwise.
        let seed_axis = |cell_coord: i32, step_axis: i32, entry_axis: f32, safe_axis: f32| -> f32 {
            if step_axis > 0 {
                ((cell_coord + 1) as f32 * cell_edge - entry_axis) / safe_axis
            } else {
                (cell_coord as f32 * cell_edge - entry_axis) / safe_axis
            }
        };
        let t_max = Vec3::new(
            seed_axis(cell.x, step.x, entry_point.x, safe_direction.x) + entry_t,
            seed_axis(cell.y, step.y, entry_point.y, safe_direction.y) + entry_t,
            seed_axis(cell.z, step.z, entry_point.z, safe_direction.z) + entry_t,
        );
        VoxelDda {
            cell,
            step,
            t_max,
            t_delta,
            t_cell_enter: entry_t,
            entry_axis: initial_entry_axis,
        }
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
