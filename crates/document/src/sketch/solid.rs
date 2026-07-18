use voxel_core::voxel::{Voxel, VoxelGrid, MAX_GRID_VOXELS};
use rayon::prelude::*;
use super::*;
use super::produce::{revolve_box_within_sweep_arc, to_profile_points};

/// A [`Sketch`] paired with an [`Operation`] that turns its 2D profile into a 3D
/// volume — the 2a sketch→volume producer (ADR 0003 §3i, the "Sketch + Operation"
/// model). Added **alongside** `SdfShape`; both implement [`VoxelProducer`](crate::voxel::VoxelProducer) and
/// resolve through the same stamp / `CombineOp` / chunk path. [`Operation::Extrude`] (a
/// prism) and [`Operation::Revolve`] (a solid of revolution) both ship; sweep is the
/// reserved third lift.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SketchSolid {
    /// The closed 2D profile + its plane.
    pub sketch: Sketch,
    /// How the profile is turned into a volume.
    #[serde(default)]
    pub operation: Operation,
}

impl SketchSolid {
    /// A sketch extruded `height_voxels` along its plane normal.
    pub fn extrude(sketch: Sketch, height_voxels: u32) -> Self {
        Self {
            sketch,
            operation: Operation::Extrude { height_voxels },
        }
    }

    /// A sketch revolved around an in-plane `axis` through `turn_degrees`
    /// (`360` = full solid of revolution). See [`Operation::Revolve`] /
    /// [`RevolveAxis`] for the (axial, radial) reinterpretation of the profile.
    pub fn revolve(sketch: Sketch, axis: RevolveAxis, turn_degrees: u32) -> Self {
        Self {
            sketch,
            operation: Operation::Revolve {
                axis,
                sweep: RevolveSweep { turn_degrees },
            },
        }
    }

    /// The profile's 2D bounding box in voxels as `(min, max)` half-open per
    /// in-plane axis, or `None` for a degenerate profile (fewer than 3 points or a
    /// zero-extent span on either in-plane axis). The local in-plane grid is sized
    /// `max − min`; cells are addressed from `min`.
    fn profile_bounds(&self) -> Option<([i64; 2], [i64; 2])> {
        // Per-operation degeneracy: an Extrude with zero height is empty (its prism
        // has no thickness); a Revolve with zero turn is empty (no sweep). Other
        // operations branch here as they are added.
        let operation_is_degenerate = match self.operation {
            Operation::Extrude { height_voxels } => height_voxels == 0,
            Operation::Revolve { sweep, .. } => sweep.turn_degrees == 0,
        };
        if self.sketch.profile.len() < 3 || operation_is_degenerate {
            return None;
        }
        let first = self.sketch.profile[0].offset_voxels;
        let mut min = first;
        let mut max = first;
        for point in &self.sketch.profile {
            for axis in 0..2 {
                min[axis] = min[axis].min(point.offset_voxels[axis]);
                max[axis] = max[axis].max(point.offset_voxels[axis]);
            }
        }
        // A zero-extent span on either in-plane axis is a degenerate (collinear /
        // zero-area) profile: no cell can be inside it.
        if max[0] <= min[0] || max[1] <= min[1] {
            return None;
        }
        Some((min, max))
    }

    /// The resolved grid's voxel dimensions `[x, y, z]` (the prism's AABB), or
    /// `[0, 0, 0]` for a degenerate profile. The two in-plane axes get the
    /// profile's bounding-box span; the normal axis gets `height_voxels`.
    /// The metric this body's field is exact in (ADR 0019 Decision 6).
    ///
    /// **The lift decides it, not the profile.** An extrusion is the product of the profile
    /// region and a slab, and the L∞ norm of a product space is the max of its factors — so a
    /// polygonal profile extrudes to an exactly-Chebyshev field, and outsets square. A
    /// revolve introduces circular cross-sections, whose L∞ distance has no closed form, just
    /// as for the curved primitives — so it is Euclidean, and outsets round.
    ///
    /// This REFINES ADR 0019 Decision 6, whose "boxes and every profile-lifted body outset
    /// square" is too coarse: revolve is profile-lifted and does not.
    pub fn field_metric(&self) -> substrate::geom2d::Metric {
        match self.operation {
            Operation::Extrude { .. } => substrate::geom2d::Metric::Chebyshev,
            Operation::Revolve { .. } => substrate::geom2d::Metric::Euclidean,
        }
    }

    /// Signed distance to the solid at `point_local_voxels`, a point in this producer's own
    /// `[0, full_dim)` voxel frame (ADR 0008 — the frame is carried, never re-derived).
    /// Negative inside, measured in whatever [`field_metric`](Self::field_metric) reports.
    ///
    /// **Extrude is exact.** The prism is the product of the profile region with the slab
    /// `[0, height]` along the plane normal, and under Chebyshev the distance to a product is
    /// the maximum of the per-factor distances — so `max(profile, slab)` IS the distance,
    /// with no correction term. (Under Euclidean the same expression would be exact only
    /// inside and on the faces, needing a `sqrt` term near the rim edge.)
    ///
    /// Consistency with [`resolve_into`] is what the classifier actually requires, and both
    /// read the same profile through the same even-odd rule.
    ///
    /// **On the boundary the predicate is authoritative, not the sign comparison.** A sample
    /// CAN land exactly on an edge — a diagonal between integer vertices passes through
    /// half-integer points, e.g. the edge `(4,3)→(7,6)` contains the voxel centre
    /// `(4.5, 3.5)` — and there the distance is zero with only its SIGN BIT carrying the
    /// even-odd verdict (`-0.0` inside, `+0.0` outside). Occupancy derived from this field
    /// must therefore test [`f32::is_sign_negative`], not `< 0.0`, which is false for `-0.0`.
    ///
    /// This costs the classifier nothing: a cell bracket that straddles zero is Boundary and
    /// falls back to a per-voxel resolve, so the ambiguity is decided by the predicate that
    /// owns it (ADR 0019 — predicates classify, fields measure).
    ///
    /// **Revolve is exact for a full turn, conservative for a partial one.** The map from a
    /// 3D point to its `(axial, radius)` pair is 1-Lipschitz, and for a surface of revolution
    /// the nearest surface point lies in the same meridian half-plane — so the 2D profile
    /// distance evaluated there *is* the 3D distance. A partial turn additionally intersects
    /// a wedge, and `max` of two fields under-estimates distance near the seam while keeping
    /// the sign exact and the field 1-Lipschitz, which is all the classifier consumes (ADR
    /// 0019 Decision 5; ADR 0017 Decision 6 already takes this posture for intersection).
    ///
    /// A degenerate producer — no profile, zero height, zero turn — is empty, so every point
    /// is outside and the distance is `f32::INFINITY`.
    ///
    /// [`resolve_into`]: crate::voxel::VoxelProducer::resolve_into
    pub fn signed_distance(&self, point_local_voxels: [f32; 3]) -> f32 {
        let Some((profile_min, _profile_max)) = self.profile_bounds() else {
            return f32::INFINITY;
        };
        match self.operation {
            Operation::Extrude { height_voxels } => {
                let [in_plane_0, in_plane_1] = self.sketch.plane.in_plane_axes();
                let normal = self.sketch.plane.normal_axis();
                // The resolve tests the polygon at `profile_min + cell + 0.5`; a sample point
                // is already `cell + 0.5`, so profile space is exactly `profile_min + point`.
                let in_profile = [
                    profile_min[0] as f64 + point_local_voxels[in_plane_0] as f64,
                    profile_min[1] as f64 + point_local_voxels[in_plane_1] as f64,
                ];
                let to_profile = substrate::geom2d::signed_distance_to_polygon(
                    &to_profile_points(&self.sketch.profile),
                    in_profile,
                    substrate::geom2d::Metric::Chebyshev,
                );
                // `grid_dimensions` sets `dimensions[normal] = height_voxels`, so the solid
                // spans `[0, height]` along the normal in this frame.
                let along_normal = point_local_voxels[normal] as f64;
                let to_slab = (-along_normal).max(along_normal - height_voxels as f64);
                to_profile.max(to_slab) as f32
            }
            Operation::Revolve { axis, sweep } => {
                let dimensions = self.grid_dimensions();
                let [in_plane_0, in_plane_1] = self.sketch.plane.in_plane_axes();
                let normal = self.sketch.plane.normal_axis();
                // Reinterpret the in-plane axes as (axial, radial), exactly as the resolve
                // does — including the ascending sort that fixes which world axis is which.
                let (axial_world_axis, axial_min, radial_in_plane_axis) = match axis {
                    RevolveAxis::InPlane0 => (in_plane_0, profile_min[0], in_plane_1),
                    RevolveAxis::InPlane1 => (in_plane_1, profile_min[1], in_plane_0),
                };
                let mut radial_world_axes = [radial_in_plane_axis, normal];
                radial_world_axes.sort_unstable();
                let [radial_a, radial_b] = radial_world_axes;

                // The radial axes are CENTRED while the axial one is not — the same
                // asymmetry the resolve carries. Mirror its f32 arithmetic so the two agree
                // bit-for-bit on samples that land near the surface.
                let half_a = dimensions[radial_a] as f32 / 2.0;
                let half_b = dimensions[radial_b] as f32 / 2.0;
                let centred_a = point_local_voxels[radial_a] - half_a;
                let centred_b = point_local_voxels[radial_b] - half_b;
                let radius = (centred_a * centred_a + centred_b * centred_b).sqrt() as f64;
                let profile_axial =
                    axial_min as f64 + point_local_voxels[axial_world_axis] as f64;

                let profile_points = to_profile_points(&self.sketch.profile);
                let distance_at = |signed_radius: f64| {
                    let (sample_0, sample_1) = match axis {
                        RevolveAxis::InPlane0 => (profile_axial, signed_radius),
                        RevolveAxis::InPlane1 => (signed_radius, profile_axial),
                    };
                    substrate::geom2d::signed_distance_to_polygon(
                        &profile_points,
                        [sample_0, sample_1],
                        substrate::geom2d::Metric::Euclidean,
                    )
                };

                // A solid of revolution is symmetric about the axis, so a point is inside if
                // the profile contains it at EITHER sign of radius — a union, hence `min`.
                // Only meaningful when the profile actually reaches across radial 0; for the
                // usual one-sided lathe profile the mirrored query is always outside, and the
                // resolve skips it for the same reason.
                let radial_profile_coord = match axis {
                    RevolveAxis::InPlane0 => 1,
                    RevolveAxis::InPlane1 => 0,
                };
                let profile_straddles_axis = self
                    .sketch
                    .profile
                    .iter()
                    .any(|point| point.offset_voxels[radial_profile_coord] < 0);
                let mut distance = distance_at(radius);
                if profile_straddles_axis {
                    distance = distance.min(distance_at(-radius));
                }

                // PARTIAL turn: intersect with the swept wedge. The resolve keeps cells whose
                // angle is at most `turn`, measuring from the +radial_a axis towards
                // +radial_b. Up to a half turn that region is the INTERSECTION of two
                // half-planes through the origin (`max`); beyond it, their UNION (`min`).
                let turn_degrees = sweep.turn_degrees;
                if turn_degrees < 360 {
                    let turn = (turn_degrees as f64).to_radians();
                    // Inside the first edge (angle 0) is the +radial_b side.
                    let past_first_edge = -(centred_b as f64);
                    // Inside the closing edge is the clockwise side of its direction vector.
                    let past_closing_edge =
                        turn.cos() * centred_b as f64 - turn.sin() * centred_a as f64;
                    let to_wedge = if turn_degrees <= 180 {
                        past_first_edge.max(past_closing_edge)
                    } else {
                        past_first_edge.min(past_closing_edge)
                    };
                    distance = distance.max(to_wedge);
                }
                distance as f32
            }
        }
    }

    pub fn grid_dimensions(&self) -> [u32; 3] {
        let Some((min, max)) = self.profile_bounds() else {
            return [0, 0, 0];
        };
        let [in_plane_0, in_plane_1] = self.sketch.plane.in_plane_axes();
        let normal = self.sketch.plane.normal_axis();
        let mut dimensions = [0u32; 3];
        match self.operation {
            Operation::Extrude { height_voxels } => {
                // Saturating downcast: a profile span exceeding u32::MAX must clamp to a
                // huge dimension (rejected by downstream bounds), never silently wrap.
                dimensions[in_plane_0] = u32::try_from(max[0] - min[0]).unwrap_or(u32::MAX);
                dimensions[in_plane_1] = u32::try_from(max[1] - min[1]).unwrap_or(u32::MAX);
                dimensions[normal] = height_voxels;
            }
            Operation::Revolve { axis, .. } => {
                // Reinterpret the in-plane bbox as (axial, radial) per RevolveAxis. The
                // axial world axis keeps its profile span; each of the two RADIAL world
                // axes (the OTHER in-plane axis + the plane normal) spans the full disc
                // diameter `2 * radial_max`, so the revolve axis sits at the grid centre.
                let (axial_world_axis, axial_span, radial_coord_min, radial_coord_max) = match axis
                {
                    RevolveAxis::InPlane0 => (in_plane_0, max[0] - min[0], min[1], max[1]),
                    RevolveAxis::InPlane1 => (in_plane_1, max[1] - min[1], min[0], max[0]),
                };
                // radial_max folds a straddling profile by abs: the farthest profile
                // vertex from the radial-0 axis, on either side.
                let radial_max = radial_coord_min.abs().max(radial_coord_max.abs());
                let diameter = u64::try_from(radial_max).unwrap_or(u64::MAX) * 2;
                let radial_dimension = u32::try_from(diameter).unwrap_or(u32::MAX);
                // The two radial world axes are the non-axial in-plane axis and the normal.
                let radial_world_axes: [usize; 2] = match axis {
                    RevolveAxis::InPlane0 => [in_plane_1, normal],
                    RevolveAxis::InPlane1 => [in_plane_0, normal],
                };
                dimensions[axial_world_axis] = u32::try_from(axial_span).unwrap_or(u32::MAX);
                dimensions[radial_world_axes[0]] = radial_dimension;
                dimensions[radial_world_axes[1]] = radial_dimension;
            }
        }
        dimensions
    }

    /// Total sampling-grid voxel count (`x · y · z`) as `u64` so it can't overflow.
    pub fn grid_voxel_count(&self) -> u64 {
        let [x, y, z] = self.grid_dimensions();
        x as u64 * y as u64 * z as u64
    }

    /// If the profile is an axis-aligned RECTANGLE — exactly the four corners of its
    /// bounding box (in any winding / starting vertex) — return its in-plane spans
    /// `[width, depth]` in voxels (along the plane's [`in_plane_axes`]); otherwise
    /// `None` (a degenerate or hand-built non-rectangular polygon). This is what the
    /// inspector uses to decide whether to show the editable Width/Depth fields (a
    /// rectangle) versus a read-only "custom profile" note (anything else), so the
    /// editor never clobbers a custom polygon by forcing it to a rectangle.
    ///
    /// [`in_plane_axes`]: PlaneAxis::in_plane_axes
    pub fn rectangle_in_plane_spans(&self) -> Option<[u32; 2]> {
        // Exactly four vertices, spanning a non-degenerate box.
        if self.sketch.profile.len() != 4 {
            return None;
        }
        let (min, max) = self.profile_bounds()?;
        // Every vertex must sit on a corner of the bounding box (each in-plane
        // coordinate is the box min or max), and all four distinct corners must be
        // present — i.e. the four points ARE the rectangle's corners.
        let mut corners_seen = [false; 4];
        for point in &self.sketch.profile {
            let [coord_0, coord_1] = point.offset_voxels;
            let on_0 = if coord_0 == min[0] {
                0
            } else if coord_0 == max[0] {
                1
            } else {
                return None;
            };
            let on_1 = if coord_1 == min[1] {
                0
            } else if coord_1 == max[1] {
                1
            } else {
                return None;
            };
            corners_seen[on_1 * 2 + on_0] = true;
        }
        if corners_seen != [true; 4] {
            return None;
        }
        let width = u32::try_from(max[0] - min[0]).ok()?;
        let depth = u32::try_from(max[1] - min[1]).ok()?;
        Some([width, depth])
    }

    /// Whether the prism's AABB exceeds [`MAX_GRID_VOXELS`] — the same single-shape
    /// sanity cap `SdfShape::exceeds_voxel_cap` applies, so a pathological
    /// profile/height can't blow memory on a lone resolve.
    pub fn exceeds_voxel_cap(&self) -> bool {
        self.grid_voxel_count() > MAX_GRID_VOXELS
    }
}

impl SketchSolid {
    /// Whether an extrude cell (in the producer's local voxel-index frame, PROVEN fully
    /// inside `[0, full_dim)` by the caller) is entirely solid — the coarse-solid test
    /// (ADR 0010). The normal span is already `⊆ [0, height_voxels]` (the caller's
    /// full-inside check + `grid_dimensions()[normal] = height_voxels`), so solidity
    /// reduces to: the cell's in-plane footprint RECTANGLE is entirely inside the profile
    /// polygon. The rectangle is the SAMPLE-CENTRE span, exactly as
    /// [`resolve_extrude`](Self::resolve_extrude) samples occupancy
    /// (`profile = bbox_min + idx + 0.5`): a cell spanning local `[c_lo, c_hi)` maps to
    /// `[min + c_lo + 0.5, min + c_hi − 0.5]`. Testing that (not the voxel corners) elides an
    /// axis-aligned FACE block — fully solid, but with its face lattice line collinear with
    /// the profile edge — while never over-claiming (the edge sits 0.5 beyond the outermost
    /// sample centre).
    pub(super) fn extrude_cell_is_solid(&self, cell: voxel_core::spatial_index::VoxelAabb) -> bool {
        let Some((min, _max)) = self.profile_bounds() else {
            return false;
        };
        let [in_plane_0, in_plane_1] = self.sketch.plane.in_plane_axes();
        let c0_lo = (min[0] + cell.min[in_plane_0]) as f64 + 0.5;
        let c0_hi = (min[0] + cell.max[in_plane_0]) as f64 - 0.5;
        let c1_lo = (min[1] + cell.min[in_plane_1]) as f64 + 0.5;
        let c1_hi = (min[1] + cell.max[in_plane_1]) as f64 - 0.5;
        let profile_points = to_profile_points(&self.sketch.profile);
        substrate::geom2d::rectangle_inside_polygon(&profile_points, [c0_lo, c1_lo], [c0_hi, c1_hi])
    }

    /// Whether a revolve cell (PROVEN fully inside `[0, full_dim)` by the caller) is
    /// entirely solid — the coarse-solid test (ADR 0010 Decision 2). Handles BOTH a full
    /// turn AND a PARTIAL wedge: a partial sweep is coarse-solid only when the cell is
    /// solid in the radial/axial profile AND its ENTIRE angular span lies inside the swept
    /// arc. Any doubt returns `false` (⇒ BOUNDARY, still exact per-voxel).
    ///
    /// The solid-of-revolution occupancy at a voxel is `theta <= turn` (the sweep gate) AND
    /// `point_in_polygon(radius, axial)` (folded by `abs`; the resolve also tests `−radius`
    /// only when the profile straddles the axis, which can only ADD occupancy — see below).
    /// So a cell is coarse-solid iff BOTH hold for its whole footprint:
    ///
    /// 1. RADIAL/AXIAL — the `(radius-range × axial-range)` rectangle is entirely inside the
    ///    profile polygon, mapped into native `(c0, c1)` per [`RevolveAxis`] EXACTLY as
    ///    [`resolve_revolve`](Self::resolve_revolve) maps its per-voxel samples:
    ///    - axial: the SAMPLE-CENTRE span `[axial_min + cell.min + 0.5, axial_min + cell.max − 0.5]`
    ///      (elides the axial END-CAP blocks, whose face is collinear with the profile edge);
    ///    - radius: over the two centred radial world axes (centred = `idx − half`), the
    ///      `[nearest, farthest]` distance from the axis over the cell's voxel-corner box,
    ///      widened by `EPS` so f32/f64 rounding can never SHRINK the tested rectangle below
    ///      the true sample coverage (a wider rectangle only makes "inside" rarer ⇒ never an
    ///      over-claim). Because the `−radius` branch only UNIONS more occupancy, `+radius`
    ///      solidity is SUFFICIENT even for an axis-straddling profile (matching full-turn).
    /// 2. ANGULAR (partial turns only) — the whole cell's sweep angle is inside `[0, turn]`
    ///    (see [`revolve_box_within_sweep_arc`]). At 360° the gate is inert, so a full turn
    ///    needs only condition 1.
    ///
    /// CONSERVATIVE-NEVER-NARROW: the two conditions use the SAME centred corner box the
    /// resolve derives its per-voxel samples from (a superset of the sample centres), so a
    /// coarse claim can never disagree with the per-voxel truth.
    pub(super) fn revolve_cell_is_solid(
        &self,
        cell: voxel_core::spatial_index::VoxelAabb,
        axis: RevolveAxis,
        sweep: RevolveSweep,
        dimensions: [u32; 3],
    ) -> bool {
        let Some((min, _max)) = self.profile_bounds() else {
            return false;
        };
        let [in_plane_0, in_plane_1] = self.sketch.plane.in_plane_axes();
        let normal = self.sketch.plane.normal_axis();
        let (axial_world_axis, axial_min, radial_in_plane_axis) = match axis {
            RevolveAxis::InPlane0 => (in_plane_0, min[0], in_plane_1),
            RevolveAxis::InPlane1 => (in_plane_1, min[1], in_plane_0),
        };
        // The two radial world axes in ASCENDING index, matching `resolve_revolve`.
        let mut radial_world_axes = [radial_in_plane_axis, normal];
        radial_world_axes.sort_unstable();
        let [radial_a, radial_b] = radial_world_axes;

        let half = [
            dimensions[0] as f64 / 2.0,
            dimensions[1] as f64 / 2.0,
            dimensions[2] as f64 / 2.0,
        ];

        // Axial rectangle range in profile-axial coords — the SAMPLE-CENTRE span, matching
        // the resolve's `axial_min + idx + 0.5` sampler exactly (a single-voxel span
        // collapses to a point, handled by `rectangle_inside_polygon`).
        let axial_lo = (axial_min + cell.min[axial_world_axis]) as f64 + 0.5;
        let axial_hi = (axial_min + cell.max[axial_world_axis]) as f64 - 0.5;

        // Centred radial voxel-corner box per radial world axis (centred = idx − half).
        let a_lo = cell.min[radial_a] as f64 - half[radial_a];
        let a_hi = cell.max[radial_a] as f64 - half[radial_a];
        let b_lo = cell.min[radial_b] as f64 - half[radial_b];
        let b_hi = cell.max[radial_b] as f64 - half[radial_b];
        // Nearest coordinate to the axis is 0 when the box straddles 0, else the closer face.
        let nearest = |lo: f64, hi: f64| -> f64 {
            if lo <= 0.0 && hi >= 0.0 {
                0.0
            } else {
                lo.abs().min(hi.abs())
            }
        };
        let farthest = |lo: f64, hi: f64| -> f64 { lo.abs().max(hi.abs()) };
        let r_near = (nearest(a_lo, a_hi).powi(2) + nearest(b_lo, b_hi).powi(2)).sqrt();
        let r_far = (farthest(a_lo, a_hi).powi(2) + farthest(b_lo, b_hi).powi(2)).sqrt();
        const EPS: f64 = 1e-4;
        let r_lo = (r_near - EPS).max(0.0);
        let r_hi = r_far + EPS;

        // Map (radius, axial) into the profile's native (c0, c1) order, matching the
        // resolve's `inside` closure: InPlane0 ⇒ (axial, radius); InPlane1 ⇒ (radius, axial).
        let (c0_lo, c0_hi, c1_lo, c1_hi) = match axis {
            RevolveAxis::InPlane0 => (axial_lo, axial_hi, r_lo, r_hi),
            RevolveAxis::InPlane1 => (r_lo, r_hi, axial_lo, axial_hi),
        };
        let profile_points = to_profile_points(&self.sketch.profile);
        if !substrate::geom2d::rectangle_inside_polygon(&profile_points, [c0_lo, c1_lo], [c0_hi, c1_hi])
        {
            return false;
        }
        // Condition 1 (radial/axial) holds. A full turn needs nothing more (the sweep gate
        // is inert at 360°). A partial turn additionally requires the cell's ENTIRE angular
        // span inside `[0, turn]` — over the SAME centred radial corner box the resolve
        // derives each per-voxel sweep angle from.
        if sweep.turn_degrees >= 360 {
            return true;
        }
        revolve_box_within_sweep_arc(a_lo, a_hi, b_lo, b_hi, sweep.turn_degrees)
    }

    /// The extrude resolve: rasterize the profile once and sweep it across
    /// `height_voxels` layers along the plane normal. Byte-identical to the prior
    /// `SketchExtrude::resolve` (the height now arrives from the matched operation).
    pub(super) fn resolve_extrude(
        &self,
        grid: &mut VoxelGrid,
        voxels_per_block: u32,
        height_voxels: u32,
        window_local_voxels: voxel_core::spatial_index::VoxelAabb,
    ) {
        let dimensions = self.grid_dimensions();
        // FULL dimensions even when only a window is written.
        grid.dimensions = dimensions;
        grid.occupied.clear();

        let Some((min, _max)) = self.profile_bounds() else {
            // Degenerate profile: empty occupancy, no panic (§3i edge case).
            return;
        };

        let [in_plane_0, in_plane_1] = self.sketch.plane.in_plane_axes();
        let normal = self.sketch.plane.normal_axis();
        let in_plane_span_0 = dimensions[in_plane_0];
        let in_plane_span_1 = dimensions[in_plane_1];
        let density = voxels_per_block.max(1);

        // The window is a WORLD-axis box `[0, full_dim)`; map each clamped world-axis
        // range to the producer's (in_plane_0, in_plane_1, normal) frame. The 2D
        // raster's `cell_0` runs along `in_plane_0` and `cell_1` along `in_plane_1`;
        // the layer sweep runs along `normal`. Clamping to full dims makes a
        // full-window call reproduce the historical `0..span` / `0..height` loops.
        let world_bounds = crate::voxel::clamp_window_to_grid(window_local_voxels, dimensions);
        let (cell_0_lo, cell_0_hi) = world_bounds[in_plane_0];
        let (cell_1_lo, cell_1_hi) = world_bounds[in_plane_1];
        let (layer_lo, layer_hi) = world_bounds[normal];
        // `grid_dimensions()` sets `dimensions[normal] = height_voxels`, so the
        // clamped normal range is already `⊆ [0, height_voxels)`.
        let _ = height_voxels;

        // Rasterize the 2D profile ONCE (axis-aligned extrusion ⇒ the same fill on
        // every layer along the normal — §3i, cheap + predictable) over the WINDOWED
        // in-plane range, then sweep it across the WINDOWED `normal` layers. A cell
        // `(cell_0, cell_1)` at local origin `min` is occupied iff its centre
        // `(min + cell + 0.5)` is inside the polygon (even-odd test at the cell
        // centre — §3i). The polygon test is on `min + cell`, which is FULL-derived;
        // only the iterated cell range narrows.
        let _ = (in_plane_span_0, in_plane_span_1);
        let profile_points = to_profile_points(&self.sketch.profile);
        let mut filled_in_plane: Vec<[u32; 2]> = Vec::new();
        for cell_1 in cell_1_lo..cell_1_hi {
            let sample_1 = min[1] as f64 + cell_1 as f64 + 0.5;
            for cell_0 in cell_0_lo..cell_0_hi {
                let sample_0 = min[0] as f64 + cell_0 as f64 + 0.5;
                if substrate::geom2d::point_in_polygon(&profile_points, [sample_0, sample_1]) {
                    filled_in_plane.push([cell_0, cell_1]);
                }
            }
        }

        // The voxel's grid index per world axis, assembled from the in-plane cell
        // and the normal layer, then CORNER-ANCHORED (centre = idx + 0.5) exactly the
        // way `SdfShape::resolve` does, so a rectangle extrude is byte-identical to the
        // matching `Box`. The centre is a half-integer for any grid size → always on
        // the global voxel lattice.
        //
        // The normal-axis LAYERS are order-independent (each layer writes a disjoint
        // set of voxels), so — mirroring `SdfShape::resolve`'s slice parallelism —
        // each layer produces a local `Vec<Voxel>` and the results are concatenated
        // with rayon. The emission ORDER may differ from the serial version, but the
        // SET is identical (consumers recover indices from each voxel's position).
        let profile_axes = [in_plane_0, in_plane_1, normal];
        grid.occupied = (layer_lo..layer_hi)
            .into_par_iter()
            .flat_map_iter(|layer| {
                let [in_plane_0, in_plane_1, normal] = profile_axes;
                filled_in_plane.iter().map(move |&[cell_0, cell_1]| {
                    let mut index = [0u32; 3];
                    index[in_plane_0] = cell_0;
                    index[in_plane_1] = cell_1;
                    index[normal] = layer;
                    Voxel {
                        local_index: [
                            index[0] as i32,
                            index[1] as i32,
                            index[2] as i32,
                        ],
                        block_local_coord: [
                            (index[0] % density) as u8,
                            (index[1] % density) as u8,
                            (index[2] % density) as u8,
                        ],
                        block_id: voxel_core::core_geom::BlockId::DEFAULT,
                        attrs: voxel_core::core_geom::BlockAttrs::DEFAULT,
                        grid_overlay: false,
                    }
                })
            })
            .collect();
    }

    /// The revolve resolve: sweep the profile around an in-plane axis into a solid
    /// of revolution (ADR 0003 §3i). The profile's `(axial, radial)` reinterpretation
    /// (per [`RevolveAxis`]) is sampled at every grid cell:
    ///
    /// - The axial world axis maps the cell to profile-axial space the SAME way the
    ///   extrude rasterizer maps an in-plane span: `axial_min + idx + 0.5` (un-centred
    ///   profile-space mapping), so a rectangle-revolve is exact against a cylinder.
    /// - The two RADIAL world axes (the non-axial in-plane axis + the plane normal)
    ///   are CENTRED exactly like `SdfShape` (`idx + 0.5 − dim/2`); the radius is their
    ///   Euclidean length, so the revolve axis lands at the grid centre.
    /// - A cell is inside iff the even-odd `point_in_polygon` test passes for the
    ///   reconstructed profile point `(+radial folded, profile_axial)` placed back into
    ///   the profile's native `(c0, c1)` slots.
    /// - PARTIAL turn: the swept angle `theta = atan2(centred[radial_b],
    ///   centred[radial_a])` (normalized to `[0, 360)`) gates the cell — kept iff
    ///   `theta <= turn_degrees`. At `turn_degrees == 360` the gate is inert.
    ///
    /// `radial_a` / `radial_b` are the two radial world axes in ASCENDING world-axis
    /// index. With `atan2(b, a)`, theta is measured FROM `radial_a` (the lower-indexed
    /// radial world axis) TOWARD `radial_b` (the higher). The wedge therefore opens
    /// from the lower radial axis. In Z-up terms, for the canonical footprint-revolve
    /// (`PlaneAxis::Z`, axial = X, so radials are Y and Z): theta=0 points along +Y
    /// (away from the viewer / into the scene, since front = −Y) and sweeps up toward
    /// +Z (vertical). The corner-anchored store is IDENTICAL to extrude.
    pub(super) fn resolve_revolve(
        &self,
        grid: &mut VoxelGrid,
        voxels_per_block: u32,
        axis: RevolveAxis,
        sweep: RevolveSweep,
        window_local_voxels: voxel_core::spatial_index::VoxelAabb,
    ) {
        let dimensions = self.grid_dimensions();
        // FULL dimensions even when only a window is written.
        grid.dimensions = dimensions;
        grid.occupied.clear();

        let Some((min, _max)) = self.profile_bounds() else {
            // Degenerate (no profile / zero turn / zero radial extent): empty, no panic.
            return;
        };
        let [in_plane_0, in_plane_1] = self.sketch.plane.in_plane_axes();
        let normal = self.sketch.plane.normal_axis();

        // Reinterpret the in-plane axes as (axial, radial) per RevolveAxis.
        let (axial_world_axis, axial_min, radial_in_plane_axis) = match axis {
            RevolveAxis::InPlane0 => (in_plane_0, min[0], in_plane_1),
            RevolveAxis::InPlane1 => (in_plane_1, min[1], in_plane_0),
        };
        // The two radial world axes (non-axial in-plane axis + normal), taken in
        // ASCENDING world-axis index so radial_a < radial_b deterministically.
        let mut radial_world_axes = [radial_in_plane_axis, normal];
        radial_world_axes.sort_unstable();
        let [radial_a, radial_b] = radial_world_axes;

        let density = voxels_per_block.max(1);
        let turn_degrees = sweep.turn_degrees;
        let is_partial = turn_degrees < 360;
        let turn = turn_degrees as f32;

        let half = [
            dimensions[0] as f32 / 2.0,
            dimensions[1] as f32 / 2.0,
            dimensions[2] as f32 / 2.0,
        ];

        // --- Per-cell-work trims (computed ONCE, before the cell loop) ---
        //
        // (1) STRADDLE flag: a solid of revolution folds both radial signs, so the
        //     general path tests `point_in_polygon` at BOTH +radius and −radius (the
        //     "straddle folded by abs"). But the −radius query can only ever be inside
        //     when some profile vertex has a NEGATIVE radial coordinate (the profile
        //     reaches across radial 0). For the common one-sided lathe profile
        //     (radial >= 0) the −radius query always lands outside, so we skip it —
        //     halving the polygon tests with IDENTICAL output. The radial profile
        //     coordinate is c1 for InPlane0 and c0 for InPlane1 (the NON-axial coord).
        // (2) radial_max: the farthest profile vertex from the radial-0 axis. A cell
        //     whose radius exceeds radial_max can't be inside the polygon (the polygon
        //     does not reach that far), so we skip the test entirely — a cheap compare
        //     before the polygon test, preserving output.
        let radial_profile_coord = match axis {
            RevolveAxis::InPlane0 => 1,
            RevolveAxis::InPlane1 => 0,
        };
        let mut profile_straddles_axis = false;
        let mut radial_max = 0i64;
        for point in &self.sketch.profile {
            let radial_coord = point.offset_voxels[radial_profile_coord];
            if radial_coord < 0 {
                profile_straddles_axis = true;
            }
            radial_max = radial_max.max(radial_coord.abs());
        }
        let radial_max = radial_max as f64;

        let profile_points = to_profile_points(&self.sketch.profile);

        // Clamp the WORLD-axis window to `[0, full_dim)`; all per-cell math (half,
        // radial_max, the centred sample, profile_axial) stays FULL-derived — only
        // the iterated cell range narrows. A full-window call reproduces the
        // historical `0..dimensions[*]` loops exactly.
        let [(win_x_lo, win_x_hi), (win_y_lo, win_y_hi), (win_z_lo, win_z_hi)] =
            crate::voxel::clamp_window_to_grid(window_local_voxels, dimensions);

        // Single-resolve allocation cap ([`MAX_GRID_VOXELS`]) — scoped to the WINDOW,
        // not the full grid. `resolve_into` only materialises the clamped window, so a
        // huge full-grid revolve is fine to resolve one small window at a time (the
        // two-layer/brick path, ADR 0010/0011): a per-chunk window never trips this.
        // The cap still protects a genuine FULL-window dense resolve (`resolve` /
        // the `oracle`-gated whole-region resolvers), where the window IS the full grid,
        // from a blown allocation.
        // The old full-grid `exceeds_voxel_cap()` guard here wrongly returned empty for
        // EVERY window of a large revolve, so large sketches resolved to nothing on the
        // windowed display path — the bug this replaces.
        // `clamp_window_to_grid` guarantees `hi >= lo` per axis, so each span is >= 0.
        let window_voxel_count = (win_x_hi - win_x_lo) as u64
            * (win_y_hi - win_y_lo) as u64
            * (win_z_hi - win_z_lo) as u64;
        if window_voxel_count > MAX_GRID_VOXELS {
            return;
        }

        // Iterate every grid cell. The axial axis uses an un-centred profile-space
        // mapping (matching the extrude rasterizer); the radial axes are centred.
        //
        // The outer `k` slices are order-independent (each samples a disjoint set of
        // voxels), so — mirroring `SdfShape::resolve` — each slice produces a local
        // `Vec<Voxel>` and rayon concatenates them. Emission ORDER may differ from the
        // serial version but the SET is identical. Windowing parallelises over the
        // WINDOWED z range.
        grid.occupied = (win_z_lo..win_z_hi)
            .into_par_iter()
            .flat_map_iter(|k| {
                let mut local = Vec::new();
                for j in win_y_lo..win_y_hi {
                    for i in win_x_lo..win_x_hi {
                        let index = [i, j, k];
                        let centred = [
                            index[0] as f32 + 0.5 - half[0],
                            index[1] as f32 + 0.5 - half[1],
                            index[2] as f32 + 0.5 - half[2],
                        ];
                        let radial =
                            (centred[radial_a].powi(2) + centred[radial_b].powi(2)).sqrt();

                        // PARTIAL turn gate: skip cells outside the swept wedge. Inert at
                        // 360 (theta ∈ [0, 360) is never > 360) — atan2 only on the
                        // partial path.
                        if is_partial {
                            let mut theta =
                                centred[radial_b].atan2(centred[radial_a]).to_degrees();
                            if theta < 0.0 {
                                theta += 360.0;
                            }
                            if theta > turn {
                                continue;
                            }
                        }

                        let radius = radial as f64;
                        // RADIAL EARLY-OUT: a cell beyond the profile's farthest radial
                        // vertex can't be inside the polygon — skip the polygon test.
                        if radius > radial_max {
                            continue;
                        }

                        // Profile-axial coord: un-centred map matching the extrude sampler.
                        let profile_axial =
                            axial_min as f64 + index[axial_world_axis] as f64 + 0.5;
                        // Reconstruct the profile point in its native (c0, c1) order: the
                        // radial-mapped coordinate is the signed radius, the axial-mapped
                        // coordinate is profile_axial, placed per RevolveAxis. A solid of
                        // revolution is symmetric about the axis, so a 3D point is inside
                        // iff the profile contains it at EITHER sign of radius. Only test
                        // −radius when the profile actually straddles radial 0 (a tube
                        // authored on the negative side, or a profile spanning across the
                        // axis); for a one-sided radial>=0 profile the −radius query always
                        // lands outside, so testing +radius alone is IDENTICAL.
                        let inside = |signed_radius: f64| {
                            let (sample_0, sample_1) = match axis {
                                RevolveAxis::InPlane0 => (profile_axial, signed_radius),
                                RevolveAxis::InPlane1 => (signed_radius, profile_axial),
                            };
                            substrate::geom2d::point_in_polygon(&profile_points, [sample_0, sample_1])
                        };
                        let is_inside = if profile_straddles_axis {
                            inside(radius) || inside(-radius)
                        } else {
                            inside(radius)
                        };
                        if !is_inside {
                            continue;
                        }

                        local.push(Voxel {
                            local_index: [
                                index[0] as i32,
                                index[1] as i32,
                                index[2] as i32,
                            ],
                            block_local_coord: [
                                (index[0] % density) as u8,
                                (index[1] % density) as u8,
                                (index[2] % density) as u8,
                            ],
                            block_id: voxel_core::core_geom::BlockId::DEFAULT,
                            attrs: voxel_core::core_geom::BlockAttrs::DEFAULT,
                            grid_overlay: false,
                        });
                    }
                }
                local
            })
            .collect();
    }
}
