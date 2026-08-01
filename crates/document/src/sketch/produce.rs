#![allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::redundant_closure_for_method_calls,
    clippy::semicolon_if_nothing_returned,
    clippy::wildcard_imports
)]

use super::*;
use crate::voxel::VoxelProducer;
use voxel_core::voxel::VoxelGrid;

/// The profile's vertices as plain `[f64; 2]` points in its native `(c0, c1)` voxel space —
/// the polygon the [`substrate::geom2d`] **predicate** half consumes (the coarse-solid cell
/// classifier, which needs exact orientation signs well past `f32`'s range). Converted ONCE
/// per resolve and reused across every per-voxel sample, so the hot loops never re-allocate.
pub(super) fn to_profile_points(profile: &[SketchPoint]) -> Vec<[f64; 2]> {
    profile_points_as(profile, |voxels, local| voxels as f64 + local as f64)
}

/// Map the profile's points through `cast(offset_voxels, offset_local_voxels)`, one axis
/// at a time. `cast` takes the `i64` + `f32` SOURCE PAIR directly (never a narrowed
/// intermediate) — the same discipline
/// [`SketchPoint::in_plane_measured`](super::SketchPoint::in_plane_measured) keeps on the
/// measurement side: `i64 → f64 → f32` can land a vertex on a different `f32` than
/// `i64 → f32`, reintroducing the CPU/GPU divergence the direct narrowing removes. Each cast
/// widens the integer in ITS OWN width before adding the fraction; a zero fraction adds exactly
/// nothing, so a whole-voxel profile is unaffected.
fn profile_points_as<T>(profile: &[SketchPoint], cast: impl Fn(i64, f32) -> T) -> Vec<[T; 2]> {
    profile
        .iter()
        .map(|point| {
            [
                cast(point.offset_voxels[0], point.offset_local_voxels[0]),
                cast(point.offset_voxels[1], point.offset_local_voxels[1]),
            ]
        })
        .collect()
}

/// The whole tagged region FLATTENED into the **predicate** width — one polygon per loop, keeping
/// each loop's `Fill`/`Hole` role. Converted ONCE per resolve and handed to the region predicates
/// by reference, so the per-cell hot loop neither re-derives faces nor re-allocates.
///
/// This is the one terminal adapter inside the resolve. The coarse cell classifier is exact-`f64`
/// over straight edges (the predicate half), so it wants vertices; everything else in the
/// resolve reads the field and keeps the curve. Pair it with [`to_region_curve_bounds`] —
/// [`substrate::geom2d::rectangle_inside_region`] needs both to stay sound where an arc was cut.
pub(super) fn to_region_points(
    region: &[ProfileLoop],
) -> Vec<(substrate::geom2d::LoopRole, Vec<[f64; 2]>)> {
    region
        .iter()
        .map(|profile_loop| {
            (
                profile_loop.role,
                to_profile_points(&profile_loop.flatten(ARC_SAGITTA_TOLERANCE_VOXELS)),
            )
        })
        .collect()
}

/// The bounds of every CURVED edge in the region, in the predicate width — the guard that keeps
/// [`to_region_points`]'s flattening from ever reaching a claim the classifier makes.
pub(super) fn to_region_curve_bounds(region: &[ProfileLoop]) -> Vec<([f64; 2], [f64; 2])> {
    region
        .iter()
        .flat_map(|profile_loop| profile_loop.edges.iter())
        .filter(|edge| edge.arc.is_some())
        .map(|edge| edge.bounds())
        .collect()
}

/// The whole tagged region in the **measurement** width, arcs intact — what the field folds. Two
/// conversions from one integer truth, for the same reason
/// [`SketchPoint::in_plane_measured`](super::SketchPoint::in_plane_measured) is not a narrowing of
/// [`to_profile_points`].
pub(super) fn to_region_edges_measured(
    region: &[ProfileLoop],
) -> Vec<(
    substrate::geom2d::LoopRole,
    Vec<substrate::geom2d::RegionEdge>,
)> {
    region
        .iter()
        .map(|profile_loop| (profile_loop.role, profile_loop.measured()))
        .collect()
}

/// A lower bound on the distance from every point of the sample box (`center ± half_extent`)
/// to the producer's grid extent `[0, dimensions]`, in `metric`.
///
/// The occupied set is a SUBSET of the extent, so distance-to-the-solid is at least
/// distance-to-the-extent: a sound lower bound, and zero for any box that touches or overlaps
/// the extent (which then contributes nothing and leaves the Lipschitz bound in charge).
///
/// This is what carries real CLEARANCE into an outside cell's interval, and what makes outset
/// sound: a sentinel that answered the same constant for every outside cell however far away it
/// was could not be shifted by an outset without classifying empty space solid.
fn box_clearance(
    center: [f32; 3],
    half_extent: [f32; 3],
    dimensions: [u32; 3],
    metric: substrate::geom2d::Metric,
) -> f32 {
    let mut gaps = [0.0f32; 3];
    for axis in 0..3 {
        let low = center[axis] - half_extent[axis];
        let high = center[axis] + half_extent[axis];
        // Signed gap to `[0, dimension]`, clamped at zero when the box straddles the slab.
        gaps[axis] = (-high).max(low - dimensions[axis] as f32).max(0.0);
    }
    match metric {
        substrate::geom2d::Metric::Chebyshev => gaps.iter().copied().fold(0.0f32, f32::max),
        substrate::geom2d::Metric::Euclidean => {
            gaps.iter().map(|gap| gap * gap).sum::<f32>().sqrt()
        }
    }
}

/// Whether the centered radial corner box `[a_lo, a_hi] × [b_lo, b_hi]` — over the two
/// radial world axes `(radial_a, radial_b)` in ASCENDING index, matching
/// [`SketchSolid::resolve_revolve`](SketchSolid::resolve_revolve)'s `centered[radial_a]` /
/// `centered[radial_b]` — lies ENTIRELY inside the swept arc `[0, turn_degrees]` (partial
/// coarse-solid condition 2).
///
/// The resolve keeps a voxel iff its sweep angle `theta = atan2(centered[radial_b],
/// centered[radial_a])` (normalized to `[0, 360)`) satisfies `theta <= turn_degrees`. A cell
/// is coarse-solid only when EVERY sample angle is `<= turn_degrees`; since the passed box is
/// the voxel-INDEX corner box (a superset of the actual sample centers `idx + 0.5 − half`),
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
pub(super) fn revolve_box_within_sweep_arc(
    a_lo: f64,
    a_hi: f64,
    b_lo: f64,
    b_hi: f64,
    turn_degrees: u32,
) -> bool {
    // Unboundable: the box contains/touches the axis, or straddles the theta=0 ray.
    //
    // The seam of the normalized angle is the `+radial_a` axis alone (`b = 0, a > 0`):
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
    // Sweep angle of a corner, normalized to [0, 360) exactly as `resolve_revolve` does.
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

    /// Conservative field interval over a block cell, honoring the interior-elision contract
    /// for BOTH extrude and revolve.
    ///
    /// **The interval is METRIC: it brackets the true signed distance over the cell.**
    /// Sign-correct but distance-free sentinels — `(1,2)` air, `(-2,-1)` solid, `(-1,1)`
    /// boundary — will not do, because the natural implementation of **outset** shifts the
    /// interval: air `(1,2)` becomes `(1−N, 2−N)` and classifies empty space SOLID for
    /// `N >= 2`, silently and with no type error. A bracket containing the real distance cannot
    /// fail that way — air survives a dilation of `N` exactly when its clearance exceeds `N`,
    /// and the clearance is the number itself.
    ///
    /// The bracket is the Lipschitz bound about the cell's center sample, in the metric the
    /// field is exact in ([`field_metric`](SketchSolid::field_metric)), then NARROWED by the
    /// two structural facts the predicates prove: the occupied set is a SUBSET of the grid
    /// AABB `[0, full_dim)` (giving a real clearance for an outside cell), and a
    /// provably-solid cell has every sample inside (closing the upper bound to zero, which is
    /// what preserves interior elision — a cell exactly filling the body would otherwise
    /// bracket to `[−2r, 0]` and go boundary).
    ///
    /// Being metric STRENGTHENS classification rather than costing it. A sentinel can claim AIR
    /// only for a cell wholly outside the grid AABB, so empty regions *inside* the AABB — the
    /// corners around a revolved cylinder, the notch of a concave L — would always fall back to
    /// a per-voxel resolve. Real distance proves them empty, roughly halving revolve's boundary
    /// cell count.
    ///
    /// CONSERVATIVE-NEVER-NARROW: coarse-solid is claimed ONLY when provably fully solid; on
    /// any doubt the boundary interval is returned (always correct). A cell that pokes
    /// outside the extent on ANY axis holds clamped-away air (`resolve_into` clamps the
    /// window to `[0, full_dim)`), so it can never be coarse — only a cell wholly inside the
    /// extent is a coarse candidate. The frame is the producer's local voxel-index frame
    /// `[0, full_dim)` — carried, never re-derived.
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
        if grid_aabb.is_empty() {
            // No occupancy anywhere: the field is `+inf`, which brackets to AIR at any
            // outset (nothing to dilate).
            return Some(crate::voxel::FieldInterval::new(f32::MAX, f32::MAX));
        }
        // Only a cell wholly inside `[0, full_dim)` can be coarse-solid (an overhang cell
        // has air voxels the resolve clamps away). This also discharges the extrude
        // normal-span condition and the revolve axial/radial extent conditions, since
        // `grid_dimensions()` sizes those axes exactly.
        let fully_inside_extent = (0..3).all(|axis| {
            cell_local_voxels.min[axis] >= 0
                && cell_local_voxels.max[axis] <= dimensions[axis] as i64
        });
        let provably_solid = fully_inside_extent
            && match self.operation {
                Operation::Extrude { .. } => self.extrude_cell_is_solid(cell_local_voxels),
                Operation::Revolve { axis, sweep } => {
                    self.revolve_cell_is_solid(cell_local_voxels, axis, sweep, dimensions)
                }
            };

        // The metric bracket. Occupancy is decided at voxel CENTERS (`index + 0.5`), so the
        // region the bracket must cover is `[min + 0.5, max − 0.5]`, not the whole cell box —
        // tighter, and exactly what `resolve_into` samples. (`center` / `half_extent` are kept
        // for the box-clearance refinement below, so this cannot fold into `metric_cell_bracket`;
        // the drift-prone metric split is still routed through `Metric::cell_circumradius`.)
        let metric = self.field_metric();
        let mut center = [0.0f32; 3];
        let mut half_extent = [0.0f32; 3];
        for axis in 0..3 {
            let low = cell_local_voxels.min[axis] as f32 + 0.5;
            let high = (cell_local_voxels.max[axis] - 1) as f32 + 0.5;
            center[axis] = 0.5 * (low + high);
            half_extent[axis] = 0.5 * (high - low);
        }
        let mut interval = crate::voxel::FieldInterval::from_lipschitz_center(
            SketchSolid::signed_distance(self, center),
            metric.cell_circumradius(half_extent),
        );

        // Refine with the two structural facts the predicates already prove. Both only ever
        // NARROW the bracket, and each is a genuine bound on the true distance, so the
        // CONSERVATIVE-NEVER-NARROW contract survives.
        //
        // (1) The occupied set is a subset of `[0, full_dim)`, so the distance to the solid is
        //     at least the distance to that box — a real positive clearance for an outside
        //     cell, where the Lipschitz bracket alone would only say "somewhere near".
        //     THIS is what makes outset sound: air survives a dilation of `N` exactly when its
        //     clearance exceeds `N`, and the clearance is now carried in the number.
        //     It applies ONLY to a box strictly outside the extent. A positive clearance means
        //     the box is disjoint from the extent, hence from the solid, so every sample is
        //     outside and `d >= clearance` holds. A box that OVERLAPS the extent has zero
        //     clearance and may well be inside the solid, where `d < 0` — raising the lower
        //     bound to zero there would be a false claim, not a refinement.
        let clearance = box_clearance(center, half_extent, dimensions, metric);
        if clearance > 0.0 {
            interval.minimum = interval.minimum.max(clearance);
        }
        // (2) A provably-solid cell has every sample center inside, so the true distance there
        //     is `<= 0` everywhere and the upper bound may close to zero. Without this the
        //     bracket would lose the interior elision the sentinel form had: a cell exactly
        //     filling the body has center distance `−r` and would bracket to `[−2r, 0+]`.
        if provably_solid {
            interval.maximum = interval.maximum.min(0.0);
        }
        Some(interval)
    }

    fn full_dimensions(&self, _voxels_per_block: u32) -> [u32; 3] {
        self.grid_dimensions()
    }

    fn as_field(&self) -> Option<&dyn crate::voxel::Field> {
        Some(self)
    }

    /// Density-independent like the geometry itself (the profile is authored in
    /// voxels outright) — see [`SketchSolid::profile_edge_polylines_local`].
    fn edge_polylines_local(
        &self,
        _voxels_per_block: u32,
        circle_segments: u32,
    ) -> Vec<Vec<[f32; 3]>> {
        self.profile_edge_polylines_local(circle_segments)
    }
}

impl crate::voxel::Field for SketchSolid {
    /// Density-independent: a sketch's geometry is authored in voxels outright, so unlike
    /// `Tube`'s block-authored wall there is nothing here that density could change.
    fn signed_distance(&self, point_local_voxels: [f32; 3], _voxels_per_block: u32) -> f32 {
        SketchSolid::signed_distance(self, point_local_voxels)
    }

    fn metric(&self) -> substrate::geom2d::Metric {
        self.field_metric()
    }
}
