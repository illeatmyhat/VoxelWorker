use super::*;
use crate::voxel::VoxelProducer;
use voxel_core::voxel::VoxelGrid;

/// The profile's vertices as plain `[f64; 2]` points in its native `(c0, c1)` voxel space —
/// the polygon the [`substrate::geom2d`] predicates consume. Converted ONCE per resolve and
/// reused across every per-voxel sample, so the hot loops never re-allocate.
pub(super) fn to_profile_points(profile: &[SketchPoint]) -> Vec<[f64; 2]> {
    profile
        .iter()
        .map(|point| [point.offset_voxels[0] as f64, point.offset_voxels[1] as f64])
        .collect()
}

/// Whether the centred radial corner box `[a_lo, a_hi] × [b_lo, b_hi]` — over the two
/// radial world axes `(radial_a, radial_b)` in ASCENDING index, matching
/// [`SketchSolid::resolve_revolve`](SketchSolid::resolve_revolve)'s `centred[radial_a]` /
/// `centred[radial_b]` — lies ENTIRELY inside the swept arc `[0, turn_degrees]` (partial
/// coarse-solid condition 2, ADR 0010 Decision 2).
///
/// The resolve keeps a voxel iff its sweep angle `theta = atan2(centred[radial_b],
/// centred[radial_a])` (normalised to `[0, 360)`) satisfies `theta <= turn_degrees`. A cell
/// is coarse-solid only when EVERY sample angle is `<= turn_degrees`; since the passed box is
/// the voxel-INDEX corner box (a superset of the actual sample centres `idx + 0.5 − half`),
/// its angular span over-covers the true samples — never under (CONSERVATIVE-NEVER-NARROW).
///
/// Two configurations are unboundable and return `false` (⇒ BOUNDARY, still exact):
/// - a box CONTAINING or TOUCHING the axis (origin `(0, 0)`) has an ambiguous/unbounded
///   angular span (a cell adjacent to the revolve axis);
/// - a box STRADDLING the `theta = 0` ray (the `+radial_a` axis: `b` crossing 0 while `a`
///   reaches positive) holds samples at `theta → 360⁻`, which — for any partial
///   `turn < 360` — exceed the arc, so the cell genuinely is not fully swept.
///
/// Otherwise the box lies in an open half-plane through the origin (angular width `< 180°`)
/// and does NOT cross the `0/360` seam, so the four corner angles bound the whole span; the
/// MAX corner angle must sit at least `ANGLE_EPS` inside `turn` so the resolve's f32 `atan2`
/// rounding can never push a boundary sample past `turn` after a coarse claim.
pub(super) fn revolve_box_within_sweep_arc(a_lo: f64, a_hi: f64, b_lo: f64, b_hi: f64, turn_degrees: u32) -> bool {
    // Unboundable: the box contains/touches the axis, or straddles the theta=0 ray.
    //
    // The seam of the normalised angle is the `+radial_a` axis alone (`b = 0, a > 0`):
    // approaching from `b > 0` gives `theta → 0⁺`, from `b < 0` gives `theta → 360⁻`. A box
    // that dips to `b < 0` while reaching UP TO OR ABOVE `b = 0` with any `a > 0` therefore
    // holds samples at `theta → 360⁻` (an angle no partial arc `[0, turn < 360]` covers) AND
    // crosses the seam, so its corner angles no longer bound the span. `b_hi >= 0` (not `> 0`)
    // catches the box whose top edge rests exactly on the ray. A box entirely below the ray
    // (`b_hi < 0`) or entirely left of the axis (`a_hi <= 0`) is seam-free and boundable.
    let contains_or_touches_origin = a_lo <= 0.0 && a_hi >= 0.0 && b_lo <= 0.0 && b_hi >= 0.0;
    let straddles_zero_ray = a_hi > 0.0 && b_lo < 0.0 && b_hi >= 0.0;
    if contains_or_touches_origin || straddles_zero_ray {
        return false;
    }
    // Sweep angle of a corner, normalised to [0, 360) exactly as `resolve_revolve` does.
    let sweep_angle = |a: f64, b: f64| -> f64 {
        let mut theta = b.atan2(a).to_degrees();
        if theta < 0.0 {
            theta += 360.0;
        }
        theta
    };
    // With the box in an open half-plane through the origin and off the 0/360 seam, its
    // angular span is contiguous with width < 180°, so the four corner angles bound it and
    // the MAX corner is the span's upper edge. The sweep gate has no lower bound (theta >= 0
    // always), so only the max angle matters.
    let max_angle = sweep_angle(a_lo, b_lo)
        .max(sweep_angle(a_lo, b_hi))
        .max(sweep_angle(a_hi, b_lo))
        .max(sweep_angle(a_hi, b_hi));
    // Widen the acceptance inward so the resolve's f32 angle can't cross `turn` post-claim.
    const ANGLE_EPS: f64 = 1e-2;
    max_angle <= turn_degrees as f64 - ANGLE_EPS
}

impl VoxelProducer for SketchSolid {
    fn resolve(&self, grid: &mut VoxelGrid, voxels_per_block: u32) {
        let [full_x, full_y, full_z] = self.grid_dimensions();
        self.resolve_into(
            grid,
            voxels_per_block,
            voxel_core::spatial_index::VoxelAabb::new(
                [0, 0, 0],
                [full_x as i64, full_y as i64, full_z as i64],
            ),
        );
    }

    fn resolve_into(
        &self,
        grid: &mut VoxelGrid,
        voxels_per_block: u32,
        window_local_voxels: voxel_core::spatial_index::VoxelAabb,
    ) {
        profiling::scope!("sketch_resolve");
        match self.operation {
            Operation::Extrude { height_voxels } => {
                self.resolve_extrude(grid, voxels_per_block, height_voxels, window_local_voxels)
            }
            Operation::Revolve { axis, sweep } => {
                self.resolve_revolve(grid, voxels_per_block, axis, sweep, window_local_voxels)
            }
        }
    }

    /// Conservative field interval over a block cell (ADR 0010 Decision 2), honouring the
    /// interior-elision contract for BOTH extrude and revolve (this FINISHES the
    /// boundary-residency rollout for `SketchSolid` — see ADR 0009 §3–§4 / ADR 0010).
    ///
    /// The occupied set is a SUBSET of the producer's grid AABB `[0, full_dim)`; inside the
    /// AABB the fill is the extruded / revolved polygon. Three verdicts:
    ///
    /// - a cell lying ENTIRELY OUTSIDE the grid AABB is provably AIR — all-positive
    ///   interval `(1, 2)`;
    /// - a cell lying ENTIRELY INSIDE the grid AABB whose whole footprint is PROVABLY solid
    ///   (the operation-specific test below) is COARSE-SOLID — all-negative interval
    ///   `(-2, -1)` (`maximum = −1 <= isolevel` ⇒ [`CoarseSolid`](crate::voxel::FieldClassification::CoarseSolid));
    /// - everything else STRADDLES `(-1, 1)` ⇒ BOUNDARY ⇒ resolved per-voxel by the even-odd
    ///   polygon test. Still exact, just unelided.
    ///
    /// CONSERVATIVE-NEVER-NARROW: coarse-solid is claimed ONLY when provably fully solid; on
    /// any doubt the boundary interval is returned (always correct). A cell that pokes
    /// outside the extent on ANY axis holds clamped-away air (`resolve_into` clamps the
    /// window to `[0, full_dim)`), so it can never be coarse — only a cell wholly inside the
    /// extent is a coarse candidate. The frame is the producer's local voxel-index frame
    /// `[0, full_dim)` (ADR 0008 — carried, never re-derived).
    fn cell_field_interval(
        &self,
        cell_local_voxels: voxel_core::spatial_index::VoxelAabb,
        voxels_per_block: u32,
    ) -> Option<crate::voxel::FieldInterval> {
        let _ = voxels_per_block;
        if cell_local_voxels.is_empty() {
            return None;
        }
        let dimensions = self.grid_dimensions();
        let [full_x, full_y, full_z] = dimensions;
        // A degenerate (empty-occupancy) producer: every cell is AIR.
        let grid_aabb = voxel_core::spatial_index::VoxelAabb::new(
            [0, 0, 0],
            [full_x as i64, full_y as i64, full_z as i64],
        );
        if grid_aabb.is_empty() || !cell_local_voxels.intersects(&grid_aabb) {
            // Wholly outside the producer extent ⇒ provably AIR.
            return Some(crate::voxel::FieldInterval::new(1.0, 2.0));
        }
        // Only a cell wholly inside `[0, full_dim)` can be coarse-solid (an overhang cell
        // has air voxels the resolve clamps away). This also discharges the extrude
        // normal-span condition and the revolve axial/radial extent conditions, since
        // `grid_dimensions()` sizes those axes exactly.
        let fully_inside_extent = (0..3).all(|axis| {
            cell_local_voxels.min[axis] >= 0
                && cell_local_voxels.max[axis] <= dimensions[axis] as i64
        });
        if fully_inside_extent {
            let provably_solid = match self.operation {
                Operation::Extrude { .. } => self.extrude_cell_is_solid(cell_local_voxels),
                Operation::Revolve { axis, sweep } => {
                    self.revolve_cell_is_solid(cell_local_voxels, axis, sweep, dimensions)
                }
            };
            if provably_solid {
                return Some(crate::voxel::FieldInterval::new(-2.0, -1.0));
            }
        }
        // Straddles the surface (or a partial-turn / axis-containing / doubtful cell) ⇒
        // BOUNDARY (per-voxel exact).
        Some(crate::voxel::FieldInterval::new(-1.0, 1.0))
    }

    fn full_dimensions(&self, _voxels_per_block: u32) -> [u32; 3] {
        self.grid_dimensions()
    }
}
