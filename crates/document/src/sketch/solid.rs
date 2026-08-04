#![allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::imprecise_flops,
    clippy::items_after_statements,
    clippy::missing_errors_doc,
    clippy::missing_const_for_fn,
    clippy::return_self_not_must_use,
    clippy::suboptimal_flops,
    clippy::too_long_first_doc_paragraph,
    clippy::use_self,
    clippy::wildcard_imports,
    clippy::tuple_array_conversions,
    clippy::similar_names
)]

use super::produce::{revolve_box_within_sweep_arc, to_region_curve_bounds, to_region_points};
use super::*;
use parametric::EvaluationContext;
use rayon::prelude::*;
use std::sync::Arc;
use voxel_core::voxel::{Voxel, VoxelGrid, MAX_GRID_VOXELS, SURFACE_ISOLEVEL};

/// The revolve field, with every per-solid constant hoisted out of the per-voxel loop.
///
/// **This type exists so there is exactly ONE evaluation of the revolve field.** The
/// conservative cell bound (`SketchSolid`'s
/// [`cell_field_interval`](crate::voxel::VoxelProducer::cell_field_interval)) brackets this function,
/// and the resolve decides occupancy by *calling* it — `signed_distance_at(p) <=
/// SURFACE_ISOLEVEL` — rather than re-deciding the same question with independent
/// arithmetic. That is what makes the bound's conservative-never-narrow contract hold by
/// construction instead of by two implementations happening to round alike.
///
/// They do not round alike. Gating the swept wedge on `atan2(b, a).to_degrees() <= turn` and
/// gating it on the half-plane form `cos(turn)·b − sin(turn)·a` describe the same SET and
/// produce different NUMBERS: IEEE-754 mandates correct rounding for `+ − × ÷ √` but
/// explicitly **not** for transcendentals, so glibc and the MSVC CRT are both conformant while
/// disagreeing by an ULP. On a sample lying exactly on the closing edge of a 45° sweep the true
/// value is 0 and the half-plane form gives `+2⁻⁵⁴ = 5.551115e-17` — a hair outside — so a
/// 1×1×1 cell (whose bracket is that single value ±1 ULP) classifies AIR while the resolve
/// counts the voxel occupied, on one platform's libm and not another's. A bound that wrongly
/// says AIR silently drops voxels from export and display, which is a correctness matter and
/// not a test one.
///
/// `SdfShape` has no such split: its resolve is already `signed_distance(..) <=
/// SURFACE_ISOLEVEL` over one field function.
pub(super) struct RevolveField {
    /// The tagged region in the measurement width: a hole in the profile is a hollow in the
    /// lathed body, so the field folds every loop rather than measuring one polygon.
    ///
    /// Owned reference-counted derived view. A prepared evaluator keeps this alive while its
    /// samples borrow the resolved curves without revisiting the sketch memo.
    derived: Arc<super::region_memo::Derived>,
    axis: RevolveAxis,
    turn_degrees: u32,
    /// World axis carrying the profile's AXIAL coordinate (un-centered, profile-space).
    axial_world_axis: usize,
    axial_min: i64,
    /// The two radial world axes, ascending, and their half-extents (the radial axes are
    /// CENTERED; the axial one is not — the asymmetry the resolve has always carried).
    radial_a: usize,
    radial_b: usize,
    half_a: f32,
    half_b: f32,
    /// Whether any profile vertex reaches across radial 0. Only then can the mirrored
    /// `−radius` query be inside, so the one-sided lathe profile skips it.
    profile_straddles_axis: bool,
    /// The farthest profile vertex from the radial-0 axis: a sample beyond it cannot be
    /// inside the profile, which the resolve uses as a cheap conservative reject.
    radial_max: f64,
}

/// Reinterpret the sketch's two in-plane axes as (axial, radial) per `RevolveAxis`,
/// returning `(axial_world_axis, axial_min, radial_a, radial_b)` with the two radial
/// world axes in ASCENDING index (the sort that fixes which world axis is which).
/// `axial_min_by_coord` is the per-in-plane-coord axial minimum — `revolve_field`
/// passes the profile bounds min, `revolve_cell_is_solid` the sample bbox min; both
/// select the same coord from it. One definition so the two stay in lockstep.
pub(super) fn revolve_axes(
    axis: RevolveAxis,
    in_plane_0: usize,
    in_plane_1: usize,
    normal: usize,
    axial_min_by_coord: [i64; 2],
) -> (usize, i64, usize, usize) {
    let (axial_world_axis, axial_min, radial_in_plane_axis) = match axis {
        RevolveAxis::InPlane0 => (in_plane_0, axial_min_by_coord[0], in_plane_1),
        RevolveAxis::InPlane1 => (in_plane_1, axial_min_by_coord[1], in_plane_0),
    };
    let mut radial_world_axes = [radial_in_plane_axis, normal];
    radial_world_axes.sort_unstable();
    let [radial_a, radial_b] = radial_world_axes;
    (axial_world_axis, axial_min, radial_a, radial_b)
}

/// The default-material occupied [`Voxel`] at grid `index`, corner-anchored, with its
/// block-local coord as `index % density`. The leaf struct both `resolve_extrude` and
/// `resolve_revolve` build once they've decided a cell is solid — one definition (their
/// surrounding loop shapes stay distinct: extrude precomputes a 2D fill, revolve tests
/// each cell radially).
fn build_voxel(index: [u32; 3], density: u32) -> Voxel {
    Voxel {
        local_index: [index[0] as i32, index[1] as i32, index[2] as i32],
        block_local_coord: [
            (index[0] % density) as u8,
            (index[1] % density) as u8,
            (index[2] % density) as u8,
        ],
        block_id: voxel_core::core_geom::BlockId::DEFAULT,
        attrs: voxel_core::core_geom::BlockAttrs::DEFAULT,
        grid_overlay: false,
    }
}

impl RevolveField {
    /// The signed distance at a point in the producer's own `[0, full_dim)` voxel frame.
    /// Negative/zero is inside (occupancy is `field <= SURFACE_ISOLEVEL`).
    pub(super) fn signed_distance_at(&self, point_local_voxels: [f32; 3]) -> f32 {
        // f32 throughout — the width of the sample the resolve forms, of the geom2d
        // measurement half, and of the WGSL preview that mirrors this field.
        let centered_a = point_local_voxels[self.radial_a] - self.half_a;
        let centered_b = point_local_voxels[self.radial_b] - self.half_b;
        let radius = (centered_a * centered_a + centered_b * centered_b).sqrt();
        let profile_axial = self.axial_min as f32 + point_local_voxels[self.axial_world_axis];

        let distance_at = |signed_radius: f32| {
            let (sample_0, sample_1) = match self.axis {
                RevolveAxis::InPlane0 => (profile_axial, signed_radius),
                RevolveAxis::InPlane1 => (signed_radius, profile_axial),
            };
            self.derived
                .region_field
                .signed_distance([sample_0, sample_1], substrate::geom2d::Metric::Euclidean)
        };
        // A solid of revolution is symmetric about its axis, so a point is inside if the
        // profile contains it at EITHER sign of radius — a union, hence `min`.
        let mut distance = distance_at(radius);
        if self.profile_straddles_axis {
            distance = distance.min(distance_at(-radius));
        }

        // PARTIAL turn: intersect with the swept wedge. Up to a half turn the swept region
        // is the INTERSECTION of two half-planes through the origin (`max`); beyond it,
        // their UNION (`min`).
        if self.turn_degrees < 360 {
            let turn = (self.turn_degrees as f32).to_radians();
            // Inside the first edge (angle 0) is the +radial_b side.
            let past_first_edge = -centered_b;
            // Inside the closing edge is the clockwise side of its direction vector.
            //
            // The width matters here, and narrowing REPAIRS a seam. At turn = 135°
            // `cos = −sin`, so this collapses to `−k·(centered_a + centered_b)` — exactly
            // zero along the anti-diagonal, where half-integer lattice sites land precisely
            // ON the closing edge. True value 0 ⇒ on-boundary ⇒ occupied. In f64 the
            // libm `cos`/`sin` pair does not cancel and this returns ≈ +4.4e−16, a hair
            // outside, and the voxel is dropped; in f32 the two round to exact negatives of
            // each other and it returns +0.0, keeping the voxel.
            let past_closing_edge = turn.cos() * centered_b - turn.sin() * centered_a;
            let to_wedge = if self.turn_degrees <= 180 {
                past_first_edge.max(past_closing_edge)
            } else {
                past_first_edge.min(past_closing_edge)
            };
            distance = distance.max(to_wedge);
        }
        distance
    }

    /// Cheap conservative reject used by the resolve: a sample farther from the axis than
    /// any profile vertex is outside the profile, so its distance is positive and the
    /// wedge `max` can only keep it positive. Skipping it is output-identical.
    fn beyond_radial_reach(&self, point_local_voxels: [f32; 3]) -> bool {
        let centered_a = point_local_voxels[self.radial_a] - self.half_a;
        let centered_b = point_local_voxels[self.radial_b] - self.half_b;
        let radius = (centered_a * centered_a + centered_b * centered_b).sqrt() as f64;
        radius > self.radial_max
    }
}

/// An immutable, density-resolved sketch field for a single sampling operation.
///
/// It owns the memo's derived view, so its point queries never take the memo lock or resolve a
/// measurement source. Construct it once with `SketchSolid::prepare_field` before a dense walk.
pub(super) struct PreparedSketchField<'a> {
    /// The original producer supplies the established structural interval proof. Keeping this
    /// borrow does not re-resolve curves: `inner` owns the derived region for point queries.
    pub(super) source: &'a SketchSolid,
    pub(super) voxels_per_block: u32,
    pub(super) inner: PreparedSketchFieldKind,
}

pub(super) enum PreparedSketchFieldKind {
    Empty,
    Extrude {
        derived: Arc<super::region_memo::Derived>,
        profile_min: [i64; 2],
        in_plane_0: usize,
        in_plane_1: usize,
        normal: usize,
        height_voxels: u32,
    },
    Revolve(RevolveField),
}

impl PreparedSketchField<'_> {
    /// A legacy producer operation received an invalid zero density. It is intentionally empty:
    /// interpreting zero as density one would evaluate fixed curve sources at invented units.
    pub(super) fn invalid(source: &SketchSolid) -> PreparedSketchField<'_> {
        PreparedSketchField {
            source,
            voxels_per_block: 0,
            inner: PreparedSketchFieldKind::Empty,
        }
    }

    pub(crate) fn signed_distance_at(&self, point_local_voxels: [f32; 3]) -> f32 {
        match &self.inner {
            PreparedSketchFieldKind::Empty => f32::INFINITY,
            PreparedSketchFieldKind::Extrude {
                derived,
                profile_min,
                in_plane_0,
                in_plane_1,
                normal,
                height_voxels,
            } => {
                let in_profile = [
                    profile_min[0] as f32 + point_local_voxels[*in_plane_0],
                    profile_min[1] as f32 + point_local_voxels[*in_plane_1],
                ];
                let to_profile = derived
                    .region_field
                    .signed_distance(in_profile, substrate::geom2d::Metric::Chebyshev);
                let along_normal = point_local_voxels[*normal];
                let to_slab = (-along_normal).max(along_normal - *height_voxels as f32);
                to_profile.max(to_slab)
            }
            PreparedSketchFieldKind::Revolve(field) => field.signed_distance_at(point_local_voxels),
        }
    }
}

/// One independently-invalidatable piece of a profile: a region loop's own footprint, plus the
/// bytes that decide what resolves inside it. Produced by [`SketchSolid::profile_pieces`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfilePiece {
    /// The piece's box in the producer's local voxel grid, half-open `[low, high)` per in-plane
    /// axis, measured from the profile's bbox minimum.
    pub local_box: ([i64; 2], [i64; 2]),
    /// Everything about this piece that changes the voxels inside its box — its boundary, its
    /// Fill/Hole role, and the node-wide settings it resolves under. Two pieces with the same
    /// box and the same fingerprint resolve to the same cells.
    pub fingerprint: String,
}

/// A [`Sketch`] paired with an [`Operation`] that turns its 2D profile into a 3D volume — the
/// sketch→volume producer. It sits **alongside** `SdfShape`; both implement
/// [`VoxelProducer`](crate::voxel::VoxelProducer) and resolve through the same stamp /
/// `CombineOp` / chunk path. [`Operation::Extrude`] produces a prism and
/// [`Operation::Revolve`] a solid of revolution.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SketchSolid {
    /// The closed 2D profile + its plane.
    pub sketch: Box<Sketch>,
    /// How the profile is turned into a volume.
    #[serde(default)]
    pub operation: Operation,
}

impl SketchSolid {
    /// A sketch extruded `height_voxels` along its plane normal.
    pub fn extrude(sketch: Sketch, height_voxels: u32) -> Self {
        Self {
            sketch: Box::new(sketch),
            operation: Operation::Extrude { height_voxels },
        }
    }

    /// A sketch revolved around an in-plane `axis` through `turn_degrees`
    /// (`360` = full solid of revolution). See [`Operation::Revolve`] /
    /// [`RevolveAxis`] for the (axial, radial) reinterpretation of the profile.
    pub fn revolve(sketch: Sketch, axis: RevolveAxis, turn_degrees: u32) -> Self {
        Self {
            sketch: Box::new(sketch),
            operation: Operation::Revolve {
                axis,
                sweep: RevolveSweep { turn_degrees },
            },
        }
    }

    /// Resolve curve geometry once for a dense field walk. The returned evaluator owns the
    /// derived view, so its point queries do not touch the sketch memo.
    pub(super) fn prepare_field(&self, context: EvaluationContext) -> PreparedSketchField<'_> {
        let derived = self.sketch.derived(context);
        let Some((profile_min, _profile_max)) = self.profile_bounds(context) else {
            return PreparedSketchField {
                source: self,
                voxels_per_block: context.voxels_per_block().get(),
                inner: PreparedSketchFieldKind::Empty,
            };
        };
        let inner = match self.operation {
            Operation::Extrude { height_voxels } => {
                let [in_plane_0, in_plane_1] = self.sketch.plane.in_plane_axes();
                PreparedSketchFieldKind::Extrude {
                    derived,
                    profile_min,
                    in_plane_0,
                    in_plane_1,
                    normal: self.sketch.plane.normal_axis(),
                    height_voxels,
                }
            }
            Operation::Revolve { axis, sweep } => {
                self.revolve_field(derived, axis, sweep, context).map_or(
                    PreparedSketchFieldKind::Empty,
                    PreparedSketchFieldKind::Revolve,
                )
            }
        };
        PreparedSketchField {
            source: self,
            voxels_per_block: context.voxels_per_block().get(),
            inner,
        }
    }

    /// The profile's 2D bounding box in voxels as `(min, max)` half-open per
    /// in-plane axis, or `None` for a degenerate profile (fewer than 3 points or a
    /// zero-extent span on either in-plane axis). The local in-plane grid is sized
    /// `max − min`; cells are addressed from `min`.
    pub(super) fn profile_bounds(
        &self,
        context: EvaluationContext,
    ) -> Option<([i64; 2], [i64; 2])> {
        // Per-operation degeneracy: an Extrude with zero height is empty (its prism
        // has no thickness); a Revolve with zero turn is empty (no sweep). Other
        // operations branch here as they are added.
        let operation_is_degenerate = match self.operation {
            Operation::Extrude { height_voxels } => height_voxels == 0,
            Operation::Revolve { sweep, .. } => sweep.turn_degrees == 0,
        };
        if operation_is_degenerate {
            return None;
        }
        let (min, max) = self.filled_extent(context)?;
        // A zero-extent span on either in-plane axis is a degenerate (collinear /
        // zero-area) profile: no cell can be inside it.
        if max[0] <= min[0] || max[1] <= min[1] {
            return None;
        }
        Some((
            [min[0].floor() as i64, min[1].floor() as i64],
            [max[0].ceil() as i64, max[1].ceil() as i64],
        ))
    }

    /// The profile broken into the pieces an edit can dirty INDEPENDENTLY — one per region loop,
    /// each with the box its cells are confined to.
    ///
    /// The edit broadphase records one entry per leaf, and a sketch node is one leaf, so moving a
    /// single vertex reads as "the whole profile changed" and re-resolves every chunk the drawing
    /// covers. That is the difference between a drag costing what the shape costs and a drag
    /// costing what the DRAWING costs — thirty shapes on a plane and every frame pays for all
    /// thirty. A loop's occupancy contribution lives inside the loop's own bounding box, so the
    /// broadphase can carry a box per loop and let the untouched ones cancel in the diff.
    ///
    /// `None` when the profile cannot be split soundly, and the caller must fall back to one
    /// whole-profile entry:
    ///
    /// * a **revolve** — the turn carries a loop's cells around the axis, so they do not stay
    ///   inside the loop's in-plane box;
    /// * a **degenerate or empty** profile, which has no box to divide;
    /// * a loop with no edges, which has no box of its own.
    ///
    /// Boxes are in the producer's local voxel grid, half-open `[low, high)`, measured from the
    /// profile's bbox minimum — the same origin
    /// [`prepare_field`](Self::prepare_field) addresses cells from.
    pub fn profile_pieces(&self, context: EvaluationContext) -> Option<Vec<ProfilePiece>> {
        let Operation::Extrude { height_voxels } = self.operation else {
            return None;
        };
        let (profile_min, _profile_max) = self.profile_bounds(context)?;
        let region = self.sketch.region(context);
        if region.is_empty() {
            return None;
        }
        // Everything about the NODE that changes what its cells resolve to, shared by every
        // piece. The node's world offset is deliberately absent: the entry's AABB already
        // carries where the piece sits, and folding the offset in here would re-fingerprint
        // every untouched loop each time the anchor compensation nudges the node — which is
        // every frame of a drag, and exactly the cancellation this exists to get.
        let shared = format!("{:?}:h={height_voxels}", self.sketch.plane);
        let mut pieces = Vec::with_capacity(region.len());
        for profile_loop in &region {
            let mut extent: Option<([f64; 2], [f64; 2])> = None;
            for edge in &profile_loop.edges {
                let (low, high) = edge.bounds();
                extent = Some(match extent {
                    None => (low, high),
                    Some((min, max)) => (
                        [min[0].min(low[0]), min[1].min(low[1])],
                        [max[0].max(high[0]), max[1].max(high[1])],
                    ),
                });
            }
            let (low, high) = extent?;
            pieces.push(ProfilePiece {
                local_box: (
                    [
                        low[0].floor() as i64 - profile_min[0],
                        low[1].floor() as i64 - profile_min[1],
                    ],
                    [
                        high[0].ceil() as i64 - profile_min[0],
                        high[1].ceil() as i64 - profile_min[1],
                    ],
                ),
                // The loop's ROLE is in the key as well as its geometry: a loop that keeps its
                // shape but flips Fill↔Hole because something moved around it resolves to the
                // opposite occupancy inside the very same box.
                fingerprint: format!("{shared}:{:?}:{:?}", profile_loop.role, profile_loop.edges),
            });
        }
        Some(pieces)
    }

    /// The profile's in-plane bounding-box **minimum** per profile coordinate — `[0, 0]` for an
    /// empty profile. Unlike [`profile_bounds`](Self::profile_bounds) this ignores degeneracy: it
    /// is the authoring anchor the producer re-seats to the node origin, needed while a profile is
    /// still being built (fewer than three points, zero height) and its vertices are being edited.
    pub fn profile_bbox_min(&self, context: EvaluationContext) -> [i64; 2] {
        match self.filled_extent(context) {
            Some((min, _max)) => [min[0].floor() as i64, min[1].floor() as i64],
            None => [0, 0],
        }
    }

    /// The FILLED region's exact continuous in-plane extent — the one measurement both
    /// [`profile_bounds`](Self::profile_bounds) and [`profile_bbox_min`](Self::profile_bbox_min)
    /// read, so the resolve's anchor and the authoring anchor cannot drift apart. `None` when
    /// nothing is filled.
    ///
    /// Taken from each edge's own bounds, so an arc contributes the reach of its BULGE. Measured
    /// off the chords instead, a producer sized from this would clip the curve it was asked to
    /// build by up to the sagitta.
    ///
    /// A hole sits inside a fill and adds no footprint, and an unpicked face on its own is not
    /// occupancy at all.
    fn filled_extent(&self, context: EvaluationContext) -> Option<([f64; 2], [f64; 2])> {
        self.sketch.filled_extent(context)
    }

    /// The node offset that keeps every **un-edited** profile vertex fixed in world after this
    /// producer replaced `previous` at node offset `previous_offset`.
    ///
    /// The resolve re-anchors the profile's bbox-minimum to the node origin
    /// ([`profile_bbox_min`](Self::profile_bbox_min)), so a vertex added, removed, or dragged at
    /// the profile's extreme moves that minimum and would drag the whole profile with it. Absorbing
    /// the bbox-min delta into the node offset — on the plane's two in-plane axes — cancels that,
    /// so only the edited vertex moves and the rest hold still. The normal axis never shifts (the
    /// profile lives on the plane).
    pub fn anchor_preserving_offset(
        &self,
        previous: &SketchSolid,
        previous_offset: [i64; 3],
        context: EvaluationContext,
    ) -> [i64; 3] {
        let old_min = previous.profile_bbox_min(context);
        let new_min = self.profile_bbox_min(context);
        let [in0, in1] = self.sketch.plane.in_plane_axes();
        let mut offset = previous_offset;
        offset[in0] += new_min[0] - old_min[0];
        offset[in1] += new_min[1] - old_min[1];
        offset
    }

    /// This producer with `point` inserted on the segment `seg_id`, splitting it. The two
    /// halves inherit the split segment's `origin`. No-op if `seg_id` is unknown. Pure —
    /// returns a new producer, leaving `self` untouched.
    pub fn with_point_on_segment(&self, seg_id: EntityId, point: SketchPoint) -> SketchSolid {
        let mut next = self.clone();
        next.sketch.split_segment(seg_id, point);
        next
    }

    /// This producer with a free point added at `at` — or, when a point already sits exactly
    /// there, the untouched producer and that point's id (coincidence). Returns the producer
    /// and the id the Line chain continues from. Pure.
    pub fn with_point_placed(&self, at: SketchPoint) -> (SketchSolid, EntityId) {
        if let Some(existing) = self.sketch.point_at(at) {
            return (self.clone(), existing);
        }
        let mut next = self.clone();
        let id = next.sketch.add_free_point(at);
        (next, id)
    }

    /// This producer with a segment joining the existing points `from → to` (the Line tool).
    /// Unchanged for a self-loop, an unknown endpoint, or an already-joined pair
    /// ([`Sketch::connect`]). Pure.
    pub fn with_segment_between(&self, from: EntityId, to: EntityId) -> SketchSolid {
        self.with_segment_between_traced(from, to)
            .map_or_else(|| self.clone(), |(next, _)| next)
    }

    /// This producer with a fresh segment and the segment's stable curve identity.
    pub fn with_segment_between_traced(
        &self,
        from: EntityId,
        to: EntityId,
    ) -> Option<(SketchSolid, SketchCurve)> {
        let mut next = self.clone();
        let id = next.sketch.connect(from, to)?;
        Some((next, SketchCurve::Segment(id)))
    }

    /// Resolve raw midpoint-line input into the exact canonical points that preview and commit
    /// share. The midpoint is a construction input, not an entity: only `endpoint` and
    /// `reflected` are candidates for persistence.
    pub fn midpoint_line_placement(
        &self,
        midpoint: [f64; 2],
        endpoint: [f64; 2],
        endpoint_existing: Option<EntityId>,
    ) -> Result<MidpointLinePlacement, MidpointLineRefusal> {
        if endpoint_existing.is_none() {
            parametric::sketch::midpoint_line_candidate(midpoint, endpoint)
                .map_err(MidpointLineRefusal::Candidate)?;
        }

        let canonical = |point: [f64; 2]| {
            SketchPoint::try_from_continuous(point[0], point[1]).map_err(MidpointLineRefusal::Point)
        };
        let midpoint = canonical(midpoint)?;
        let endpoint = if let Some(id) = endpoint_existing {
            self.sketch
                .points()
                .iter()
                .find(|point| point.id == id)
                .map(|point| point.at)
                .ok_or(MidpointLineRefusal::UnknownEndpoint)?
        } else {
            canonical(endpoint)?
        };
        self.midpoint_line_placement_from_canonical(midpoint, endpoint, endpoint_existing)
    }

    /// Resolve already-canonical midpoint-line input without composing either split-coordinate
    /// point into a large `f64`. This is the drawing-tool adapter: snapped/grabbed positions stay
    /// exact from cursor resolution through preview and commit.
    pub fn midpoint_line_placement_from_canonical(
        &self,
        midpoint: SketchPoint,
        endpoint: SketchPoint,
        endpoint_existing: Option<EntityId>,
    ) -> Result<MidpointLinePlacement, MidpointLineRefusal> {
        let endpoint = if let Some(id) = endpoint_existing {
            self.sketch
                .points()
                .iter()
                .find(|point| point.id == id)
                .map(|point| point.at)
                .ok_or(MidpointLineRefusal::UnknownEndpoint)?
        } else {
            endpoint
        };
        let reflected = midpoint
            .exact_reflection_of(&endpoint)
            .map_err(MidpointLineRefusal::Point)?
            .ok_or(MidpointLineRefusal::CanonicalCollapse)?;

        // The split-coordinate reflection is authoritative. Refuse a self-loop; construction
        // above has already refused any reflection that canonical storage cannot hold exactly.
        if endpoint.coincides(&reflected) {
            return Err(MidpointLineRefusal::CanonicalCollapse);
        }

        Ok(MidpointLinePlacement {
            midpoint,
            endpoint,
            reflected,
        })
    }

    /// Atomically append the one segment defined by a midpoint and one endpoint, returning its
    /// stable segment id. Exact existing coordinates are reused; a supplied clicked endpoint id
    /// is authoritative and must still be live. Every allocation occurs on a clone, so refusal
    /// consumes neither geometry nor ids from `self`.
    pub fn with_midpoint_line(
        &self,
        midpoint: [f64; 2],
        endpoint: [f64; 2],
        endpoint_existing: Option<EntityId>,
    ) -> Result<(SketchSolid, EntityId), MidpointLineRefusal> {
        let placement = self.midpoint_line_placement(midpoint, endpoint, endpoint_existing)?;
        self.with_midpoint_line_placement(&placement, endpoint_existing)
    }

    /// Append a segment from already-canonical midpoint and endpoint inputs. See
    /// [`midpoint_line_placement_from_canonical`](Self::midpoint_line_placement_from_canonical).
    pub fn with_midpoint_line_from_canonical(
        &self,
        midpoint: SketchPoint,
        endpoint: SketchPoint,
        endpoint_existing: Option<EntityId>,
    ) -> Result<(SketchSolid, EntityId), MidpointLineRefusal> {
        let placement =
            self.midpoint_line_placement_from_canonical(midpoint, endpoint, endpoint_existing)?;
        self.with_midpoint_line_placement(&placement, endpoint_existing)
    }

    fn with_midpoint_line_placement(
        &self,
        placement: &MidpointLinePlacement,
        endpoint_existing: Option<EntityId>,
    ) -> Result<(SketchSolid, EntityId), MidpointLineRefusal> {
        let mut next = self.clone();
        let endpoint_id = endpoint_existing.unwrap_or_else(|| {
            next.sketch
                .point_at(placement.endpoint)
                .unwrap_or_else(|| next.sketch.add_free_point(placement.endpoint))
        });
        let reflected_id = next
            .sketch
            .point_at(placement.reflected)
            .unwrap_or_else(|| next.sketch.add_free_point(placement.reflected));
        let segment = next
            .sketch
            .connect(endpoint_id, reflected_id)
            .ok_or(MidpointLineRefusal::DuplicateSegment)?;
        Ok((next, segment))
    }

    /// This producer with a closed axis-aligned rectangle appended between opposite corners
    /// `a` and `b` — the rectangle tool draws a whole loop in one gesture. Corner points
    /// that coincide with existing points reuse their ids; the four edges go through
    /// [`Sketch::connect`], so an edge that already exists is not doubled. Unchanged when the
    /// corners are degenerate (zero span on either in-plane axis — no area to enclose). Pure.
    pub fn with_rectangle(
        &self,
        a: SketchPoint,
        b: SketchPoint,
        context: EvaluationContext,
    ) -> Result<SketchSolid, RectangleRefusal> {
        let placement = self.corner_rectangle_placement(a, b)?;
        let (next, _) =
            self.with_rectangle_placement(&placement, RectangleFrame::AxisAligned, context)?;
        Ok(next)
    }

    /// Resolve an axis-aligned rectangle between opposite corners without allocating entities.
    ///
    /// Shared with [`SketchSolid::with_rectangle`] so the two-click preview draws the very loop
    /// the second click authors, rather than a lookalike built by the shell.
    pub fn corner_rectangle_placement(
        &self,
        a: SketchPoint,
        b: SketchPoint,
    ) -> Result<RectanglePlacement, RectangleRefusal> {
        let (a_pos, b_pos) = (a.in_plane(), b.in_plane());
        if a_pos[0] == b_pos[0] || a_pos[1] == b_pos[1] {
            return Err(RectangleRefusal::Unrepresentable);
        }
        // The two synthesized corners mix one coordinate from each source point,
        // fraction included; they carry no retained expression of their own.
        let mixed = |axis0_of: &SketchPoint, axis1_of: &SketchPoint| SketchPoint {
            offset_voxels: [axis0_of.offset_voxels[0], axis1_of.offset_voxels[1]],
            offset_local_voxels: [
                axis0_of.offset_local_voxels[0],
                axis1_of.offset_local_voxels[1],
            ],
            offset_measurements: None,
        };
        Ok(RectanglePlacement {
            corners: [a, mixed(&b, &a), b, mixed(&a, &b)],
        })
    }

    /// Resolve a center-defined axis-aligned rectangle without allocating entities.
    pub fn center_rectangle_placement(
        &self,
        center: SketchPoint,
        corner: SketchPoint,
    ) -> Result<RectanglePlacement, RectangleRefusal> {
        let candidate =
            parametric::sketch::center_rectangle_candidate(center.in_plane(), corner.in_plane())
                .map_err(RectangleRefusal::Candidate)?;
        canonical_rectangle(candidate)
    }

    /// Resolve an oriented three-point rectangle without allocating entities.
    pub fn three_point_rectangle_placement(
        &self,
        first: SketchPoint,
        second: SketchPoint,
        width_point: SketchPoint,
    ) -> Result<RectanglePlacement, RectangleRefusal> {
        let candidate = parametric::sketch::three_point_rectangle_candidate(
            first.in_plane(),
            second.in_plane(),
            width_point.in_plane(),
        )
        .map_err(RectangleRefusal::Candidate)?;
        canonical_rectangle(candidate)
    }

    /// Atomically append a center-defined axis-aligned rectangle.
    ///
    /// The center is authored, not merely consumed: it persists as a construction point held at
    /// the crossing of two construction diagonals, so dragging it moves the rectangle and the
    /// shape keeps a handle on the thing it was drawn around. The diagonals never bound a region
    /// — that is what [`EntityRole::Construction`] means — so the interior stays one face.
    pub fn with_center_rectangle(
        &self,
        center: SketchPoint,
        corner: SketchPoint,
        context: EvaluationContext,
    ) -> Result<SketchSolid, RectangleRefusal> {
        let placement = self.center_rectangle_placement(center, corner)?;
        let (mut next, ids) =
            self.with_rectangle_placement(&placement, RectangleFrame::AxisAligned, context)?;
        let diagonals = [
            connect_construction(&mut next.sketch, ids[0], ids[2])?,
            connect_construction(&mut next.sketch, ids[1], ids[3])?,
        ];
        let center_id = next
            .sketch
            .point_at(center)
            .unwrap_or_else(|| next.sketch.add_free_point(center));
        // The center belongs to the rectangle, not the author: nothing but the two diagonals
        // refers to it, so it goes when they do.
        next.sketch
            .set_point_lifetime(center_id, PointLifetime::CurveAnchored);
        // Halfway along BOTH diagonals. The second assertion is implied by the first once the
        // sides are square, and the solver keeps it flagged rather than refusing it — the
        // author drew a center rectangle, so both diagonals owning the center is the intent.
        for segment in diagonals {
            assert_rectangle_relation(
                &mut next.sketch,
                ConstraintKind::Midpoint {
                    point: center_id,
                    segment,
                },
                context,
            )?;
        }
        Ok(next)
    }

    /// Atomically append an oriented three-point rectangle.
    pub fn with_three_point_rectangle(
        &self,
        first: SketchPoint,
        second: SketchPoint,
        width_point: SketchPoint,
        context: EvaluationContext,
    ) -> Result<SketchSolid, RectangleRefusal> {
        let placement = self.three_point_rectangle_placement(first, second, width_point)?;
        let (next, _) =
            self.with_rectangle_placement(&placement, RectangleFrame::Oriented, context)?;
        Ok(next)
    }

    /// Append the four corners and their closing loop, returning the corner ids so a caller that
    /// authors more than the boundary — the center construction, say — can name them.
    ///
    /// The frame is passed rather than inferred from the corners. A three-point rectangle the
    /// author happened to draw square to the axes is still an ORIENTED rectangle — inferring
    /// would silently pin its rotation and make the tool's behavior depend on how carefully the
    /// author aimed.
    fn with_rectangle_placement(
        &self,
        placement: &RectanglePlacement,
        frame: RectangleFrame,
        context: EvaluationContext,
    ) -> Result<(SketchSolid, [EntityId; 4]), RectangleRefusal> {
        let mut next = self.clone();
        let ids = placement.corners.map(|corner| {
            next.sketch
                .point_at(corner)
                .unwrap_or_else(|| next.sketch.add_free_point(corner))
        });
        let edges = rectangle_edges(&mut next.sketch, ids)?;
        if next == *self {
            return Err(RectangleRefusal::AlreadyExists);
        }
        match frame {
            RectangleFrame::AxisAligned => constrain_axis_aligned_rectangle(
                &mut next.sketch,
                &placement.corners,
                edges,
                context,
            )?,
            RectangleFrame::Oriented => {
                constrain_oriented_rectangle(&mut next.sketch, edges, context)?;
            }
        }
        Ok((next, ids))
    }

    /// Resolve an inscribed regular polygon without allocation.
    pub fn inscribed_polygon_placement(
        &self,
        center: SketchPoint,
        radius_point: SketchPoint,
        sides: u16,
    ) -> Result<PolygonPlacement, PolygonRefusal> {
        let candidate = parametric::sketch::centered_polygon_candidate(
            parametric::sketch::CenteredPolygonKind::Inscribed,
            center.in_plane(),
            radius_point.in_plane(),
            sides,
        )
        .map_err(PolygonRefusal::Candidate)?;
        canonical_polygon(candidate)
    }

    /// Resolve a circumscribed regular polygon without allocation.
    pub fn circumscribed_polygon_placement(
        &self,
        center: SketchPoint,
        apothem_point: SketchPoint,
        sides: u16,
    ) -> Result<PolygonPlacement, PolygonRefusal> {
        let candidate = parametric::sketch::centered_polygon_candidate(
            parametric::sketch::CenteredPolygonKind::Circumscribed,
            center.in_plane(),
            apothem_point.in_plane(),
            sides,
        )
        .map_err(PolygonRefusal::Candidate)?;
        canonical_polygon(candidate)
    }

    /// Resolve an edge-defined regular polygon without allocation.
    pub fn edge_polygon_placement(
        &self,
        first: SketchPoint,
        second: SketchPoint,
        side_point: SketchPoint,
        sides: u16,
    ) -> Result<PolygonPlacement, PolygonRefusal> {
        let candidate = parametric::sketch::edge_polygon_candidate(
            first.in_plane(),
            second.in_plane(),
            side_point.in_plane(),
            sides,
        )
        .map_err(PolygonRefusal::Candidate)?;
        canonical_polygon(candidate)
    }

    pub fn with_inscribed_polygon(
        &self,
        center: SketchPoint,
        radius_point: SketchPoint,
        sides: u16,
    ) -> Result<SketchSolid, PolygonRefusal> {
        self.inscribed_polygon_placement(center, radius_point, sides)
            .and_then(|placement| self.with_polygon_placement(&placement))
    }

    pub fn with_circumscribed_polygon(
        &self,
        center: SketchPoint,
        apothem_point: SketchPoint,
        sides: u16,
    ) -> Result<SketchSolid, PolygonRefusal> {
        self.circumscribed_polygon_placement(center, apothem_point, sides)
            .and_then(|placement| self.with_polygon_placement(&placement))
    }

    pub fn with_edge_polygon(
        &self,
        first: SketchPoint,
        second: SketchPoint,
        side_point: SketchPoint,
        sides: u16,
    ) -> Result<SketchSolid, PolygonRefusal> {
        self.edge_polygon_placement(first, second, side_point, sides)
            .and_then(|placement| self.with_polygon_placement(&placement))
    }

    fn with_polygon_placement(
        &self,
        placement: &PolygonPlacement,
    ) -> Result<SketchSolid, PolygonRefusal> {
        let mut next = self.clone();
        let ids: Vec<EntityId> = placement
            .vertices
            .iter()
            .map(|&vertex| {
                next.sketch
                    .point_at(vertex)
                    .unwrap_or_else(|| next.sketch.add_free_point(vertex))
            })
            .collect();
        for index in 0..ids.len() {
            next.sketch
                .connect(ids[index], ids[(index + 1) % ids.len()]);
        }
        (next != *self)
            .then_some(next)
            .ok_or(PolygonRefusal::AlreadyExists)
    }

    /// Resolve any linear slot grammar without allocating document entities.
    pub fn linear_slot_placement(
        &self,
        kind: parametric::sketch::LinearSlotKind,
        first: SketchPoint,
        second: SketchPoint,
        width_point: SketchPoint,
    ) -> Result<SlotPlacement, SlotRefusal> {
        canonical_slot(
            parametric::sketch::linear_slot_candidate(
                kind,
                first.in_plane(),
                second.in_plane(),
                width_point.in_plane(),
            )
            .map_err(SlotRefusal::Candidate)?,
        )
    }

    /// Resolve Three Point Arc Slot without allocating document entities.
    pub fn three_point_arc_slot_placement(
        &self,
        start: SketchPoint,
        end: SketchPoint,
        through: SketchPoint,
        width_point: SketchPoint,
    ) -> Result<SlotPlacement, SlotRefusal> {
        canonical_slot(
            parametric::sketch::three_point_arc_slot_candidate(
                start.in_plane(),
                end.in_plane(),
                through.in_plane(),
                width_point.in_plane(),
            )
            .map_err(SlotRefusal::Candidate)?,
        )
    }

    /// Resolve Center Point Arc Slot without allocating document entities.
    pub fn center_arc_slot_placement(
        &self,
        center: SketchPoint,
        start: SketchPoint,
        end_direction: SketchPoint,
        turn: parametric::sketch::ArcTurn,
        width_point: SketchPoint,
    ) -> Result<SlotPlacement, SlotRefusal> {
        canonical_slot(
            parametric::sketch::center_arc_slot_candidate(
                center.in_plane(),
                start.in_plane(),
                end_direction.in_plane(),
                turn,
                width_point.in_plane(),
            )
            .map_err(SlotRefusal::Candidate)?,
        )
    }

    pub fn with_linear_slot(
        &self,
        kind: parametric::sketch::LinearSlotKind,
        first: SketchPoint,
        second: SketchPoint,
        width_point: SketchPoint,
        context: EvaluationContext,
    ) -> Result<SketchSolid, SlotRefusal> {
        let placement = self.linear_slot_placement(kind, first, second, width_point)?;
        self.with_slot_placement(&placement, context)
    }

    pub fn with_three_point_arc_slot(
        &self,
        start: SketchPoint,
        end: SketchPoint,
        through: SketchPoint,
        width_point: SketchPoint,
        context: EvaluationContext,
    ) -> Result<SketchSolid, SlotRefusal> {
        let placement = self.three_point_arc_slot_placement(start, end, through, width_point)?;
        self.with_slot_placement(&placement, context)
    }

    pub fn with_center_arc_slot(
        &self,
        center: SketchPoint,
        start: SketchPoint,
        end_direction: SketchPoint,
        turn: parametric::sketch::ArcTurn,
        width_point: SketchPoint,
        context: EvaluationContext,
    ) -> Result<SketchSolid, SlotRefusal> {
        let placement =
            self.center_arc_slot_placement(center, start, end_direction, turn, width_point)?;
        self.with_slot_placement(&placement, context)
    }

    /// Append a slot's four boundary curves and the five relations that make them one shape:
    /// a tangency at each of the four corners, plus the one that holds the two rails together.
    ///
    /// Without them a placed slot is an inert outline: drag any corner and the caps come away from
    /// the sides. The assertions are what carry the author's intent forward, so a boundary that
    /// stands but cannot be constrained is a refusal, not a partial success — the same
    /// all-or-nothing contract a rectangle keeps.
    ///
    /// What is deliberately NOT asserted is the width. The five relations leave exactly one
    /// freedom, and that freedom is the slot's width — which is why an author changes it by
    /// dragging a rail rather than by editing a stored number.
    fn with_slot_placement(
        &self,
        placement: &SlotPlacement,
        context: EvaluationContext,
    ) -> Result<SketchSolid, SlotRefusal> {
        let mut next = self.clone();
        let mut curves = Vec::with_capacity(4);
        for edge in placement.edges {
            let (from, to) = match edge {
                SlotEdgePlacement::Line { from, to } | SlotEdgePlacement::Arc { from, to, .. } => {
                    (from, to)
                }
            };
            let from = next
                .sketch
                .point_at(from)
                .unwrap_or_else(|| next.sketch.add_free_point(from));
            let to = next
                .sketch
                .point_at(to)
                .unwrap_or_else(|| next.sketch.add_free_point(to));
            // A boundary the drawing already carries is REUSED, not doubled — and it is that
            // standing curve the tangency must name, or the relations would skip whichever side
            // the sketch already had.
            let curve = match edge {
                SlotEdgePlacement::Line { .. } => next
                    .sketch
                    .connect(from, to)
                    .or_else(|| next.sketch.segment_between(from, to))
                    .map(SketchCurve::Segment),
                SlotEdgePlacement::Arc { sweep, .. } => next
                    .sketch
                    .connect_arc(from, to, sweep)
                    .or_else(|| next.sketch.arc_between(from, to, sweep))
                    .map(SketchCurve::Arc),
            };
            curves.push(curve.ok_or(SlotRefusal::Unrepresentable)?);
        }
        if next == *self {
            return Err(SlotRefusal::AlreadyExists);
        }
        // Naming the four curves is what lets the corners be written as literal pairs; walking them
        // by index would put modular arithmetic between the shape and the relations that describe
        // it, for no gain on a boundary that is always exactly four curves long.
        let [first_rail, end_cap, second_rail, start_cap] = curves[..] else {
            return Err(SlotRefusal::Unrepresentable);
        };
        let corners = [
            (first_rail, end_cap),
            (end_cap, second_rail),
            (second_rail, start_cap),
            (start_cap, first_rail),
        ];
        let rails = match (placement.spine.center, first_rail, second_rail) {
            (Some(_), first, second) => ConstraintKind::concentric(first, second),
            (None, SketchCurve::Segment(first), SketchCurve::Segment(second)) => {
                let (first, second) = if first <= second {
                    (first, second)
                } else {
                    (second, first)
                };
                ConstraintKind::Parallel { first, second }
            }
            // A straight-spined slot whose rails came back as anything but segments is not a slot.
            (None, _, _) => return Err(SlotRefusal::Unrepresentable),
        };
        let handles = slot_spine_handles(
            &mut next.sketch,
            placement,
            [start_cap, end_cap, first_rail],
        )?;
        let coincidences = handles
            .iter()
            .map(|&(handle, derived)| ConstraintKind::Coincident {
                first: handle.min(derived),
                second: handle.max(derived),
            });
        let spine_line =
            slot_spine_line(&mut next.sketch, placement, &handles, [start_cap, end_cap])?;
        let tangencies = placement
            .junctions
            .into_iter()
            .zip(corners)
            .map(|(branch, (first, second))| ConstraintKind::tangent(first, second, branch));
        for kind in tangencies
            .chain(std::iter::once(rails))
            .chain(coincidences)
            .chain(spine_line)
        {
            match next.sketch.add_constraint(kind, context) {
                Ok(_) | Err(ConstraintRefusal::AlreadyAsserted { .. }) => {}
                Err(refusal) => return Err(SlotRefusal::Constraint(refusal)),
            }
        }
        Ok(next)
    }

    /// This producer with the point `point_id` deleted, CASCADING to its incident segments:
    /// deleting a point removes its edges and nothing else, and does NOT reclose the loop.
    /// No-op if `point_id` is unknown. Pure — returns a new producer. A loop that
    /// opens (or falls below three vertices) simply resolves to nothing.
    pub fn with_point_deleted(&self, point_id: EntityId) -> SketchSolid {
        let mut next = self.clone();
        next.sketch.delete_point_cascade(point_id);
        next
    }

    /// This producer with the segment `seg_id` deleted, along with each of its ends that nothing
    /// else draws ([`Sketch::delete_segment`]). No-op if unknown. Pure.
    pub fn with_segment_deleted(&self, seg_id: EntityId) -> SketchSolid {
        let mut next = self.clone();
        next.sketch.delete_segment(seg_id);
        next
    }

    /// This producer with an arc of the given signed included angle joining the existing
    /// points `from → to` — the 3-point tool commits through here after consuming its
    /// through-point. Unchanged for a self-loop, an unknown endpoint, a degenerate
    /// bulge, or an already-joined pair ([`Sketch::connect_arc`]). Pure.
    pub fn with_arc_between(
        &self,
        from: EntityId,
        to: EntityId,
        bulge: parametric::units::AngleMeasurement,
    ) -> SketchSolid {
        let mut next = self.clone();
        next.sketch.connect_arc(from, to, bulge);
        next
    }

    /// Resolve Center Point Arc's projected endpoint without allocating document entities.
    /// An existing start id is authoritative; the end click supplies only a direction and is
    /// projected onto the fixed start radius.
    pub fn center_arc_placement(
        &self,
        center: SketchPoint,
        start: SketchPoint,
        start_existing: Option<EntityId>,
        end_direction: SketchPoint,
        turn: parametric::sketch::ArcTurn,
    ) -> Result<CenterArcPlacement, CenterArcRefusal> {
        let start = start_existing.map_or(Ok(start), |id| {
            self.sketch
                .points()
                .iter()
                .find(|point| point.id == id)
                .map(|point| point.at)
                .ok_or(CenterArcRefusal::UnknownStart)
        })?;
        let raw_candidate = parametric::sketch::center_arc_candidate(
            center.in_plane(),
            start.in_plane(),
            end_direction.in_plane(),
            turn,
        )
        .map_err(CenterArcRefusal::Candidate)?;
        let endpoint =
            SketchPoint::try_from_continuous(raw_candidate.endpoint[0], raw_candidate.endpoint[1])
                .map_err(|_| CenterArcRefusal::Unrepresentable)?;
        // Canonical storage narrows the projected endpoint to `i64 + f32`. Recompute the sweep
        // from that durable direction, then expose the center/radius derived from the exact
        // endpoint+sweep representation commit will persist. Preview therefore cannot advertise
        // an ideal f64 circle that writeback changes by a fraction of a voxel.
        // The turn must ride through the narrowing round trip too: re-solving the canonical
        // endpoint without it would take the shortest way round and silently flip an arc the
        // author wound the long way.
        let canonical = parametric::sketch::center_arc_candidate(
            center.in_plane(),
            start.in_plane(),
            endpoint.in_plane(),
            turn,
        )
        .map_err(CenterArcRefusal::Candidate)?;
        let sweep = parametric::units::AngleMeasurement::try_from_degrees_f64(
            canonical.sweep_radians.to_degrees(),
        )
        .map_err(|_| CenterArcRefusal::Unrepresentable)?;
        let (derived_center, radius) = arc_center_radius(
            start.in_plane(),
            endpoint.in_plane(),
            sweep.to_degrees_f64(),
        )
        .ok_or(CenterArcRefusal::Unrepresentable)?;
        let candidate = parametric::sketch::CenterArcCandidate {
            center: derived_center,
            start: start.in_plane(),
            endpoint: endpoint.in_plane(),
            radius,
            sweep_radians: sweep.to_degrees_f64().to_radians(),
        };
        let center = SketchPoint::try_from_continuous(derived_center[0], derived_center[1])
            .map_err(|_| CenterArcRefusal::Unrepresentable)?;
        Ok(CenterArcPlacement {
            center,
            start,
            endpoint,
            candidate,
        })
    }

    /// Atomically append a Center Point Arc. The construction center is reified by the arc as a
    /// derived construction point; only the endpoints and intrinsic sweep are authored freedoms.
    pub fn with_center_arc(
        &self,
        center: SketchPoint,
        start: SketchPoint,
        start_existing: Option<EntityId>,
        end_direction: SketchPoint,
        turn: parametric::sketch::ArcTurn,
    ) -> Result<(SketchSolid, EntityId), CenterArcRefusal> {
        let placement =
            self.center_arc_placement(center, start, start_existing, end_direction, turn)?;
        let sweep = parametric::units::AngleMeasurement::try_from_degrees_f64(
            placement.candidate.sweep_radians.to_degrees(),
        )
        .map_err(|_| CenterArcRefusal::Unrepresentable)?;
        let mut next = self.clone();
        let start_id = start_existing.unwrap_or_else(|| {
            next.sketch
                .point_at(placement.start)
                .unwrap_or_else(|| next.sketch.add_free_point(placement.start))
        });
        let endpoint_id = next
            .sketch
            .point_at(placement.endpoint)
            .unwrap_or_else(|| next.sketch.add_free_point(placement.endpoint));
        let arc = next
            .sketch
            .connect_arc(start_id, endpoint_id, sweep)
            .ok_or(CenterArcRefusal::ArcRefused)?;
        Ok((next, arc))
    }

    /// Append an arc tangent to the live incoming curve at their shared endpoint. Arc creation
    /// and the durable Tangent assertion land together or not at all.
    pub fn with_tangent_arc_between(
        &self,
        incoming: SketchCurve,
        seam: EntityId,
        to: EntityId,
        context: EvaluationContext,
    ) -> Result<(SketchSolid, SketchCurve), TangentArcRefusal> {
        if seam == to {
            return Err(TangentArcRefusal::SelfLoop);
        }
        let seam_at = self
            .sketch
            .points()
            .iter()
            .find(|point| point.id == seam)
            .map(|point| point.at.in_plane())
            .ok_or(TangentArcRefusal::UnknownEndpoint)?;
        let to_at = self
            .sketch
            .points()
            .iter()
            .find(|point| point.id == to)
            .map(|point| point.at.in_plane())
            .ok_or(TangentArcRefusal::UnknownEndpoint)?;
        let candidate = self
            .sketch
            .tangent_arc_candidate(incoming, seam, to_at, context)?;
        let sweep = parametric::units::AngleMeasurement::try_from_degrees_f64(
            candidate.sweep_radians.to_degrees(),
        )
        .map_err(|_| TangentArcRefusal::UnrepresentableSweep)?;

        let mut next = self.clone();
        let arc_id = next
            .sketch
            .connect_arc(seam, to, sweep)
            .ok_or(TangentArcRefusal::ArcRefused)?;
        let arc = SketchCurve::Arc(arc_id);
        let branch = next
            .sketch
            .choose_tangent_branch(incoming, seam_at, arc, seam_at, context)
            .map_err(TangentArcRefusal::Branch)?;
        next.sketch
            .add_constraint(ConstraintKind::tangent(incoming, arc, branch), context)
            .map_err(TangentArcRefusal::Constraint)?;
        Ok((next, arc))
    }

    /// Resolve a standalone Tangent Arc destination without allocating document geometry.
    /// A supplied endpoint id is authoritative, so preview uses the same stored position commit
    /// will connect even when retained measurement provenance or snap policy differs.
    pub fn tangent_arc_placement_to(
        &self,
        incoming: SketchCurve,
        seam: EntityId,
        endpoint: SketchPoint,
        endpoint_existing: Option<EntityId>,
        context: EvaluationContext,
    ) -> Result<TangentArcPlacement, TangentArcRefusal> {
        let point = |id| {
            self.sketch
                .points()
                .iter()
                .find(|point| point.id == id)
                .map(|point| point.at)
        };
        let seam_at = point(seam).ok_or(TangentArcRefusal::UnknownEndpoint)?;
        let endpoint = endpoint_existing.map_or(Ok(endpoint), |id| {
            point(id).ok_or(TangentArcRefusal::UnknownEndpoint)
        })?;
        let candidate =
            self.sketch
                .tangent_arc_candidate(incoming, seam, endpoint.in_plane(), context)?;
        Ok(TangentArcPlacement {
            seam: seam_at,
            endpoint,
            candidate,
        })
    }

    /// Atomically append a standalone Tangent Arc to a canonical destination. A fresh endpoint
    /// is allocated only on a trial clone; every refusal leaves the source and its next id intact.
    pub fn with_tangent_arc_to(
        &self,
        incoming: SketchCurve,
        seam: EntityId,
        endpoint: SketchPoint,
        endpoint_existing: Option<EntityId>,
        context: EvaluationContext,
    ) -> Result<(SketchSolid, SketchCurve), TangentArcRefusal> {
        let placement =
            self.tangent_arc_placement_to(incoming, seam, endpoint, endpoint_existing, context)?;
        let mut trial = self.clone();
        let endpoint_id = endpoint_existing.unwrap_or_else(|| {
            trial
                .sketch
                .point_at(placement.endpoint)
                .unwrap_or_else(|| trial.sketch.add_free_point(placement.endpoint))
        });
        trial.with_tangent_arc_between(incoming, seam, endpoint_id, context)
    }

    /// This producer with the arc `arc_id` deleted, along with each of its ends that nothing else
    /// draws ([`Sketch::delete_arc`]). No-op if unknown. Pure.
    pub fn with_arc_deleted(&self, arc_id: EntityId) -> SketchSolid {
        let mut next = self.clone();
        next.sketch.delete_arc(arc_id);
        next
    }

    /// This producer with a circle centered at `center` and passing through `perimeter`.
    /// Reuses a point already at the center; otherwise the circle owns a construction center.
    /// A zero or non-finite radius leaves the producer unchanged.
    pub fn with_circle_center_diameter(
        &self,
        center: SketchPoint,
        perimeter: SketchPoint,
    ) -> SketchSolid {
        let offset = [
            perimeter.in_plane()[0] - center.in_plane()[0],
            perimeter.in_plane()[1] - center.in_plane()[1],
        ];
        let radius = offset[0].hypot(offset[1]);
        let radius = SketchLength::from_continuous(radius);
        let mut next = self.clone();
        match next.sketch.point_at(center) {
            Some(existing) => {
                next.sketch.circle_about(existing, radius);
            }
            None => {
                next.sketch.add_circle(center, radius);
            }
        }
        next
    }

    /// Resolve a circle whose diameter endpoints are `first` and `second` without allocation.
    pub fn two_point_circle_placement(
        &self,
        first: SketchPoint,
        second: SketchPoint,
    ) -> Result<PointCirclePlacement, PointCircleRefusal> {
        let candidate =
            parametric::sketch::two_point_circle_candidate(first.in_plane(), second.in_plane())
                .map_err(PointCircleRefusal::Candidate)?;
        canonical_point_circle(candidate)
    }

    /// Resolve the unique circle through three circumference points without allocation.
    pub fn three_point_circle_placement(
        &self,
        first: SketchPoint,
        second: SketchPoint,
        third: SketchPoint,
    ) -> Result<PointCirclePlacement, PointCircleRefusal> {
        let candidate = parametric::sketch::three_point_circle_candidate(
            first.in_plane(),
            second.in_plane(),
            third.in_plane(),
        )
        .map_err(PointCircleRefusal::Candidate)?;
        canonical_point_circle(candidate)
    }

    /// Atomically append the circle whose diameter is defined by two points.
    pub fn with_two_point_circle(
        &self,
        first: SketchPoint,
        second: SketchPoint,
    ) -> Result<(SketchSolid, EntityId), PointCircleRefusal> {
        self.two_point_circle_placement(first, second)
            .and_then(|placement| self.with_point_circle_placement(placement))
    }

    /// Atomically append the unique circle through three circumference points.
    pub fn with_three_point_circle(
        &self,
        first: SketchPoint,
        second: SketchPoint,
        third: SketchPoint,
    ) -> Result<(SketchSolid, EntityId), PointCircleRefusal> {
        self.three_point_circle_placement(first, second, third)
            .and_then(|placement| self.with_point_circle_placement(placement))
    }

    fn with_point_circle_placement(
        &self,
        placement: PointCirclePlacement,
    ) -> Result<(SketchSolid, EntityId), PointCircleRefusal> {
        let mut next = self.clone();
        let circle = match next.sketch.point_at(placement.center) {
            Some(center) => next.sketch.circle_about(center, placement.radius),
            None => next.sketch.add_circle(placement.center, placement.radius),
        }
        .ok_or(PointCircleRefusal::CircleRefused)?;
        Ok((next, circle))
    }

    /// Resolve a radius-selected circle tangent to two finite segments.
    pub fn two_tangent_circle_placement(
        &self,
        segments: [EntityId; 2],
        witness: SketchPoint,
    ) -> Result<TangentCirclePlacement, TangentCircleRefusal> {
        let first = self
            .segment_endpoints(segments[0])
            .ok_or(TangentCircleRefusal::UnknownSegment)?;
        let second = self
            .segment_endpoints(segments[1])
            .ok_or(TangentCircleRefusal::UnknownSegment)?;
        canonical_tangent_circle(
            parametric::sketch::two_tangent_circle_candidate(
                [
                    (first.0.in_plane(), first.1.in_plane()),
                    (second.0.in_plane(), second.1.in_plane()),
                ],
                witness.in_plane(),
            )
            .map_err(TangentCircleRefusal::Candidate)?,
        )
    }

    /// Resolve the circle tangent to three finite segments using their click loci for branch
    /// selection.
    pub fn three_tangent_circle_placement(
        &self,
        segments: [(EntityId, SketchPoint); 3],
    ) -> Result<TangentCirclePlacement, TangentCircleRefusal> {
        let line = |(id, locus): (EntityId, SketchPoint)| {
            let (from, to) = self
                .segment_endpoints(id)
                .ok_or(TangentCircleRefusal::UnknownSegment)?;
            Ok((from.in_plane(), to.in_plane(), locus.in_plane()))
        };
        let [first, second, third] = segments.map(line);
        canonical_tangent_circle(
            parametric::sketch::three_tangent_circle_candidate([first?, second?, third?])
                .map_err(TangentCircleRefusal::Candidate)?,
        )
    }

    /// Atomically append a two-line tangent circle and both durable Tangent relations.
    pub fn with_two_tangent_circle(
        &self,
        segments: [EntityId; 2],
        witness: SketchPoint,
        context: EvaluationContext,
    ) -> Result<SketchSolid, TangentCircleRefusal> {
        let placement = self.two_tangent_circle_placement(segments, witness)?;
        self.with_tangent_circle_placement(&placement, &segments, context)
    }

    /// Atomically append a three-line tangent circle and all three durable Tangent relations.
    pub fn with_three_tangent_circle(
        &self,
        segments: [(EntityId, SketchPoint); 3],
        context: EvaluationContext,
    ) -> Result<SketchSolid, TangentCircleRefusal> {
        let placement = self.three_tangent_circle_placement(segments)?;
        let ids = segments.map(|(id, _)| id);
        self.with_tangent_circle_placement(&placement, &ids, context)
    }

    fn with_tangent_circle_placement(
        &self,
        placement: &TangentCirclePlacement,
        segments: &[EntityId],
        context: EvaluationContext,
    ) -> Result<SketchSolid, TangentCircleRefusal> {
        let mut next = self.clone();
        let circle_id = next
            .sketch
            .add_circle(placement.center, placement.radius)
            .ok_or(TangentCircleRefusal::CircleRefused)?;
        let circle = SketchCurve::Circle(circle_id);
        for (&segment_id, contact) in segments.iter().zip(&placement.contacts) {
            let segment = SketchCurve::Segment(segment_id);
            let locus = contact.in_plane();
            let branch = next
                .sketch
                .choose_tangent_branch(segment, locus, circle, locus, context)
                .map_err(TangentCircleRefusal::Branch)?;
            next.sketch
                .add_constraint(ConstraintKind::tangent(segment, circle, branch), context)
                .map_err(TangentCircleRefusal::Constraint)?;
        }
        Ok(next)
    }

    fn segment_endpoints(&self, id: EntityId) -> Option<(SketchPoint, SketchPoint)> {
        let segment = self
            .sketch
            .segments()
            .iter()
            .find(|segment| segment.id == id)?;
        let point = |id| {
            self.sketch
                .points()
                .iter()
                .find(|point| point.id == id)
                .map(|point| point.at)
        };
        Some((point(segment.from)?, point(segment.to)?))
    }

    /// This producer with the circle `circle_id` deleted. Its construction center is pruned only
    /// when no remaining curve names it.
    pub fn with_circle_deleted(&self, circle_id: EntityId) -> SketchSolid {
        let mut next = self.clone();
        next.sketch.delete_circle(circle_id);
        next
    }

    /// This producer with the curve `curve` deleted, whichever store holds it.
    ///
    /// The selection names a curve by its typed identity, so deletion answers in the same
    /// currency rather than making every caller re-split the identity back into a store and an
    /// id. An aggregate leaves through one call even though it draws several spans.
    pub fn with_curve_deleted(&self, curve: SketchCurve) -> SketchSolid {
        let mut next = self.clone();
        match curve {
            SketchCurve::Segment(id) => next.sketch.delete_segment(id),
            SketchCurve::Arc(id) => next.sketch.delete_arc(id),
            SketchCurve::Circle(id) => next.sketch.delete_circle(id),
            SketchCurve::Bezier(id) => next.sketch.delete_bezier(id),
            SketchCurve::Ellipse(id) => next.sketch.delete_ellipse(id),
            SketchCurve::Conic(id) => next.sketch.delete_conic(id),
            SketchCurve::Spline(id) => next.sketch.delete_spline(id),
        }
        next
    }

    /// Toggle every selected geometry entity's real/construction role as one prospective edit.
    ///
    /// `None` means the selection named no toggleable geometry, allowing callers to avoid an
    /// empty undo entry. Each entity toggles independently, matching the command's literal role
    /// semantics for a mixed real/construction selection.
    pub fn with_construction_toggled(
        &self,
        entities: impl IntoIterator<Item = EntityId>,
    ) -> Option<SketchSolid> {
        let mut next = self.clone();
        let mut changed = false;
        for entity in entities {
            changed |= next.sketch.toggle_construction(entity);
        }
        changed.then_some(next)
    }

    /// This producer with a tangent handle standing at each named fit point.
    ///
    /// `None` when none of them took one — a point that is not a fit point of a fit-point spline,
    /// or one that already has a handle, so the verb costs no empty undo entry.
    pub fn with_tangent_handles(
        &self,
        points: impl IntoIterator<Item = EntityId>,
    ) -> Option<SketchSolid> {
        let mut next = self.clone();
        let mut minted = false;
        for point in points {
            let standing = next.sketch.tangent_handle_of(point);
            minted |= next.sketch.add_tangent_handle(point).is_some() && standing.is_none();
        }
        minted.then_some(next)
    }

    /// This producer with `kind` asserted, the drawing moved to where the solve put it, and the
    /// new constraint's id. `Err` leaves nothing changed — a refusal is not a partial edit, so
    /// a caller that discards the `Err` still holds the drawing the author had.
    ///
    /// The solve happens inside [`Sketch::add_constraint`], which trials on a copy and keeps it
    /// only once it converges; solving again here would move nothing and cost a second Jacobian.
    pub fn with_constraint(
        &self,
        kind: ConstraintKind,
        context: EvaluationContext,
    ) -> Result<(SketchSolid, EntityId), ConstraintRefusal> {
        let mut next = self.clone();
        let id = next.sketch.add_constraint(kind, context)?;
        Ok((next, id))
    }

    /// This producer with the constraint `constraint_id` released. The geometry stays where the
    /// last solve left it — dropping an assertion stops re-asserting it, it does not undo it.
    /// No-op if unknown. Pure.
    pub fn with_constraint_deleted(&self, constraint_id: EntityId) -> SketchSolid {
        let mut next = self.clone();
        next.sketch.delete_constraint(constraint_id);
        next
    }

    /// This producer with the derived face `key` picked or unpicked — the only edit that
    /// carves a hole. Geometry is untouched, so the profile bbox and the node
    /// anchor cannot move. Pure.
    pub fn with_face_picked(
        &self,
        key: super::FaceKey,
        picked: bool,
        context: EvaluationContext,
    ) -> SketchSolid {
        let mut next = self.clone();
        next.sketch.set_face_picked(key, picked, context);
        next
    }

    /// The metric this body's field is exact in.
    ///
    /// **The lift decides it, not the profile.** An extrusion is the product of the profile
    /// region and a slab, and the L∞ norm of a product space is the max of its factors — so a
    /// polygonal profile extrudes to an exactly-Chebyshev field, and outsets square. A
    /// revolve introduces circular cross-sections, whose L∞ distance has no closed form, just
    /// as for the curved primitives — so it is Euclidean, and outsets round.
    pub fn field_metric(&self) -> substrate::geom2d::Metric {
        match self.operation {
            Operation::Extrude { .. } => substrate::geom2d::Metric::Chebyshev,
            Operation::Revolve { .. } => substrate::geom2d::Metric::Euclidean,
        }
    }

    /// Build the hoisted revolve field — the ONE evaluation both the bound and the
    /// resolve go through (see [`RevolveField`]). `None` for a degenerate profile, which
    /// is empty everywhere.
    pub(super) fn revolve_field(
        &self,
        derived: Arc<super::region_memo::Derived>,
        axis: RevolveAxis,
        sweep: RevolveSweep,
        context: EvaluationContext,
    ) -> Option<RevolveField> {
        let (profile_min, _profile_max) = self.profile_bounds(context)?;
        // The straddle / reach measurements below are about how far the SOLID reaches from the
        // lathe axis, so they read the filled loops' EXTENT; a hole never extends the body, and a
        // bulge reaches past the chord approximating it.
        let (radial_low, radial_high) = self.filled_extent(context)?;
        let dimensions = self.grid_dimensions(context);
        let [in_plane_0, in_plane_1] = self.sketch.plane.in_plane_axes();
        let normal = self.sketch.plane.normal_axis();
        // Reinterpret the in-plane axes as (axial, radial) per `RevolveAxis` (shared).
        let (axial_world_axis, axial_min, radial_a, radial_b) = revolve_axes(
            axis,
            in_plane_0,
            in_plane_1,
            normal,
            [profile_min[0], profile_min[1]],
        );

        let radial_profile_coord = match axis {
            RevolveAxis::InPlane0 => 1,
            RevolveAxis::InPlane1 => 0,
        };
        let profile_straddles_axis = radial_low[radial_profile_coord] < 0.0;
        let radial_max = radial_low[radial_profile_coord]
            .abs()
            .max(radial_high[radial_profile_coord].abs());

        Some(RevolveField {
            derived,
            axis,
            turn_degrees: sweep.turn_degrees,
            axial_world_axis,
            axial_min,
            radial_a,
            radial_b,
            half_a: dimensions[radial_a] as f32 / 2.0,
            half_b: dimensions[radial_b] as f32 / 2.0,
            profile_straddles_axis,
            radial_max,
        })
    }

    /// Signed distance to the solid at `point_local_voxels`, a point in this producer's own
    /// `[0, full_dim)` voxel frame — the frame is carried, never re-derived. Negative inside,
    /// measured in whatever [`field_metric`](Self::field_metric) reports.
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
    /// half-integer points, e.g. the edge `(4,3)→(7,6)` contains the voxel center
    /// `(4.5, 3.5)` — and there the distance is zero with only its SIGN BIT carrying the
    /// even-odd verdict (`-0.0` inside, `+0.0` outside). Occupancy derived from this field
    /// must therefore test [`f32::is_sign_negative`], not `< 0.0`, which is false for `-0.0`.
    ///
    /// This costs the classifier nothing: a cell bracket that straddles zero is Boundary and
    /// falls back to a per-voxel resolve, so the ambiguity is decided by the predicate that
    /// owns it — predicates classify, fields measure.
    ///
    /// **Revolve is exact for a full turn, conservative for a partial one.** The map from a
    /// 3D point to its `(axial, radius)` pair is 1-Lipschitz, and for a surface of revolution
    /// the nearest surface point lies in the same meridian half-plane — so the 2D profile
    /// distance evaluated there *is* the 3D distance. A partial turn additionally intersects
    /// a wedge, and `max` of two fields under-estimates distance near the seam while keeping
    /// the sign exact and the field 1-Lipschitz, which is all the classifier consumes — the
    /// same posture intersection takes.
    ///
    /// A degenerate producer — no profile, zero height, zero turn — is empty, so every point
    /// is outside and the distance is `f32::INFINITY`.
    ///
    /// [`resolve_into`]: crate::voxel::VoxelProducer::resolve_into
    pub fn signed_distance(&self, point_local_voxels: [f32; 3], context: EvaluationContext) -> f32 {
        let Some((profile_min, _profile_max)) = self.profile_bounds(context) else {
            return f32::INFINITY;
        };
        match self.operation {
            Operation::Extrude { height_voxels } => {
                let [in_plane_0, in_plane_1] = self.sketch.plane.in_plane_axes();
                let normal = self.sketch.plane.normal_axis();
                // The resolve tests the polygon at `profile_min + cell + 0.5`; a sample point
                // is already `cell + 0.5`, so profile space is exactly `profile_min + point`.
                let in_profile = [
                    profile_min[0] as f32 + point_local_voxels[in_plane_0],
                    profile_min[1] as f32 + point_local_voxels[in_plane_1],
                ];
                let to_profile = self
                    .sketch
                    .derived(context)
                    .region_field
                    .signed_distance(in_profile, substrate::geom2d::Metric::Chebyshev);
                // `grid_dimensions` sets `dimensions[normal] = height_voxels`, so the solid
                // spans `[0, height]` along the normal in this frame.
                let along_normal = point_local_voxels[normal];
                let to_slab = (-along_normal).max(along_normal - height_voxels as f32);
                to_profile.max(to_slab)
            }
            Operation::Revolve { axis, sweep } => {
                // ONE evaluation, shared with the resolve — see [`RevolveField`]. The
                // resolve decides occupancy by calling this same function, so the bound
                // brackets exactly what the resolve computed rather than a parallel
                // reimplementation that rounds differently.
                let derived = self.sketch.derived(context);
                match self.revolve_field(derived, axis, sweep, context) {
                    Some(field) => field.signed_distance_at(point_local_voxels),
                    None => f32::INFINITY,
                }
            }
        }
    }

    /// The resolved grid's voxel dimensions `[x, y, z]` (the prism's AABB), or `[0, 0, 0]`
    /// for a degenerate profile. The two in-plane axes get the profile's bounding-box span;
    /// the normal axis gets `height_voxels`.
    pub fn grid_dimensions(&self, context: EvaluationContext) -> [u32; 3] {
        let Some((min, max)) = self.profile_bounds(context) else {
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
                // diameter `2 * radial_max`, so the revolve axis sits at the grid center.
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
    pub fn grid_voxel_count(&self, context: EvaluationContext) -> u64 {
        let [x, y, z] = self.grid_dimensions(context);
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
    pub fn rectangle_in_plane_spans(&self, context: EvaluationContext) -> Option<[u32; 2]> {
        // Exactly four vertices, spanning a non-degenerate box. A fractional vertex
        // disqualifies: the spans are whole voxels, and the inspector's editable
        // Width/Depth would clobber the sub-voxel remainder by rewriting the corners.
        let profile = self.sketch.flattened_loop(context);
        if profile.len() != 4
            || profile
                .iter()
                .any(|point| point.offset_local_voxels != [0.0; 2])
        {
            return None;
        }
        let (min, max) = self.profile_bounds(context)?;
        // Every vertex must sit on a corner of the bounding box (each in-plane
        // coordinate is the box min or max), and all four distinct corners must be
        // present — i.e. the four points ARE the rectangle's corners.
        let mut corners_seen = [false; 4];
        for point in &profile {
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
    pub fn exceeds_voxel_cap(&self, context: EvaluationContext) -> bool {
        self.grid_voxel_count(context) > MAX_GRID_VOXELS
    }
}

fn canonical_point_circle(
    candidate: parametric::sketch::CircleCandidate,
) -> Result<PointCirclePlacement, PointCircleRefusal> {
    let center = SketchPoint::try_from_continuous(candidate.center[0], candidate.center[1])
        .map_err(|_| PointCircleRefusal::Unrepresentable)?;
    const UPPER_EXCLUSIVE: f64 = -(i64::MIN as f64);
    if !(0.0..UPPER_EXCLUSIVE).contains(&candidate.radius) || candidate.radius <= 0.0 {
        return Err(PointCircleRefusal::Unrepresentable);
    }
    let radius = SketchLength::from_continuous(candidate.radius);
    Ok(PointCirclePlacement {
        candidate: parametric::sketch::CircleCandidate {
            center: center.in_plane(),
            radius: radius.value(),
        },
        center,
        radius,
    })
}

fn canonical_tangent_circle(
    candidate: parametric::sketch::TangentCircleCandidate,
) -> Result<TangentCirclePlacement, TangentCircleRefusal> {
    let center = SketchPoint::try_from_continuous(candidate.center[0], candidate.center[1])
        .map_err(|_| TangentCircleRefusal::Unrepresentable)?;
    const UPPER_EXCLUSIVE: f64 = -(i64::MIN as f64);
    if !(0.0..UPPER_EXCLUSIVE).contains(&candidate.radius) || candidate.radius <= 0.0 {
        return Err(TangentCircleRefusal::Unrepresentable);
    }
    let radius = SketchLength::from_continuous(candidate.radius);
    let contacts = candidate
        .contacts
        .into_iter()
        .map(|contact| {
            SketchPoint::try_from_continuous(contact[0], contact[1])
                .map_err(|_| TangentCircleRefusal::Unrepresentable)
        })
        .collect::<Result<_, _>>()?;
    Ok(TangentCirclePlacement {
        center,
        radius,
        contacts,
    })
}

/// Which claim a rectangle tool makes about its own sides.
///
/// Both frames say "this stays a rectangle"; they differ on whether it may turn. Naming the
/// frame keeps the decision at the tool that knows the answer instead of re-deriving it from
/// corner coordinates that only accidentally reveal it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RectangleFrame {
    /// Sides stay on the plane's own axes — the two-point and center constructions.
    AxisAligned,
    /// Corners stay square but the rectangle may rotate — the three-point construction.
    Oriented,
}

/// The slot's spine handles, each paired with the derived center it is to be tied to.
///
/// Each handle is pinned to a center the boundary already derives: the caps' centers are the
/// spine's two ends, and a turning spine's center is the one both rails share (`curves` names the
/// start cap, the end cap, and a rail, in that order). The handle is a REAL point tied by
/// Coincident rather than the derived center itself — dragging a derived center authors the
/// quantity behind it and does not settle, which would make the slot's own center a dead handle.
fn slot_spine_handles(
    sketch: &mut Sketch,
    placement: &SlotPlacement,
    curves: [SketchCurve; 3],
) -> Result<Vec<(EntityId, EntityId)>, SlotRefusal> {
    let [start_cap, end_cap, rail] = curves;
    [
        (placement.spine.start, start_cap),
        (placement.spine.end, end_cap),
    ]
    .into_iter()
    .chain(placement.spine.center.map(|center| (center, rail)))
    .map(|(at, curve)| {
        let derived = sketch
            .center_point_of(curve)
            .ok_or(SlotRefusal::Unrepresentable)?;
        // A handle has to be a point that can be DRAGGED, and the ordinary "reuse whatever is
        // standing here" lookup cannot give one: this spot is already occupied by the very center
        // being tied to, and — where two rails share a center — by its twin as well. Tying to
        // either would assert a coincidence the drawing already keeps and leave the author no
        // handle at all.
        let standing = sketch
            .points()
            .iter()
            .find(|point| !sketch.is_derived_point(point.id) && point.at.coincides(&at))
            .map(|point| point.id);
        // A handle the slot MINTED outlives only the slot: nothing but its coincidence names it,
        // so once the boundary goes the dot has no job and should not be left behind. A handle
        // that reuses a point the author already placed keeps that point's own lifetime — the
        // slot borrowed it, it does not own it.
        let handle = standing.unwrap_or_else(|| {
            let minted = sketch.add_free_point(at);
            sketch.set_point_lifetime(minted, PointLifetime::CurveAnchored);
            minted
        });
        Ok((handle, derived))
    })
    .collect()
}

/// Draw an Overall Slot's middle as a construction line, and return what holds it there.
///
/// An Overall Slot is authored by clicking the two far ends of the finished shape, and those two
/// picks used to be spent: the tool insets each by half a width to find a cap center and throws the
/// extremes away, so all three linear grammars committed an identical drawing. The author's own
/// quantity — how long the slot is END TO END — had no handle on it afterwards.
///
/// So the extremes become points, joined by a construction line down the middle, with the cap
/// centers held on that line and each extreme held on the cap it reaches. Four rows against four
/// new freedoms: the width is still the one thing nothing pins, so it is still dragged rather than
/// typed. Every other slot grammar reports no reach and this does nothing.
fn slot_spine_line(
    sketch: &mut Sketch,
    placement: &SlotPlacement,
    handles: &[(EntityId, EntityId)],
    caps: [SketchCurve; 2],
) -> Result<Vec<ConstraintKind>, SlotRefusal> {
    let Some(reach) = placement.reach else {
        return Ok(Vec::new());
    };
    let ends = reach.map(|at| match sketch.point_at(at) {
        Some(standing) => standing,
        None => {
            let id = sketch.add_free_point(at);
            // The extreme belongs to the slot: it is drawn only because the construction line and
            // the cap name it, and it should go when they do rather than litter a dot.
            sketch.set_point_lifetime(id, PointLifetime::CurveAnchored);
            id
        }
    });
    let [first_end, second_end] = ends;
    let line = sketch
        .connect(first_end, second_end)
        .or_else(|| sketch.segment_between(first_end, second_end))
        .ok_or(SlotRefusal::Unrepresentable)?;
    sketch.set_construction(line);
    let centers_on_the_line =
        handles
            .iter()
            .take(2)
            .map(|&(handle, _)| ConstraintKind::PointOnCurve {
                point: handle,
                curve: SketchCurve::Segment(line),
            });
    let ends_on_their_caps = ends
        .into_iter()
        .zip(caps)
        .map(|(point, curve)| ConstraintKind::PointOnCurve { point, curve });
    Ok(centers_on_the_line.chain(ends_on_their_caps).collect())
}

/// Join two existing points with a CONSTRUCTION segment, reusing a standing edge.
///
/// A diagonal is reference geometry: it exists to give the center something to be the middle of,
/// and it must never bound a region.
fn connect_construction(
    sketch: &mut Sketch,
    from: EntityId,
    to: EntityId,
) -> Result<EntityId, RectangleRefusal> {
    let id = sketch
        .connect(from, to)
        .or_else(|| sketch.segment_between(from, to))
        .ok_or(RectangleRefusal::UnknownSegment)?;
    sketch.set_construction(id);
    Ok(id)
}

/// Close a corner loop into four boundary edges, in the same order as `ids`.
///
/// An edge that already stands is REUSED rather than doubled — [`Sketch::connect`] declines a
/// duplicate, and the edge the rectangle means to constrain is that standing one. Without this
/// the relations would silently skip whichever side the drawing already had.
fn rectangle_edges(
    sketch: &mut Sketch,
    ids: [EntityId; 4],
) -> Result<[EntityId; 4], RectangleRefusal> {
    let mut edges = [0; 4];
    for index in 0..4 {
        let (from, to) = (ids[index], ids[(index + 1) % 4]);
        edges[index] = sketch
            .connect(from, to)
            .or_else(|| sketch.segment_between(from, to))
            .ok_or(RectangleRefusal::UnknownSegment)?;
    }
    Ok(edges)
}

/// Assert one of a rectangle's relations, treating "already asserted" as satisfied.
///
/// A rectangle's corners fuse with coincident existing points and its edges reuse standing
/// segments, so a rectangle drawn against earlier geometry can meet a side that already carries
/// the very relation being asserted. That is the desired end state reached early, not a refusal:
/// re-asserting it is idempotent. Every other refusal aborts the whole command, because a
/// rectangle is the shape AND the assertions that keep it one.
fn assert_rectangle_relation(
    sketch: &mut Sketch,
    kind: ConstraintKind,
    context: EvaluationContext,
) -> Result<(), RectangleRefusal> {
    match sketch.add_constraint(kind, context) {
        Ok(_) | Err(ConstraintRefusal::AlreadyAsserted { .. }) => Ok(()),
        Err(refusal) => Err(RectangleRefusal::Constraint(refusal)),
    }
}

/// Assert that an axis-aligned rectangle stays axis-aligned: each side Horizontal or Vertical.
///
/// Which is which is read from the geometry the tool just placed rather than from the edge's
/// index, so a corner order that differs between the two-point and center constructions cannot
/// silently assert the transpose. Four relations against eight corner freedoms leave exactly the
/// four the shape has — position, width and height.
fn constrain_axis_aligned_rectangle(
    sketch: &mut Sketch,
    corners: &[SketchPoint; 4],
    edges: [EntityId; 4],
    context: EvaluationContext,
) -> Result<(), RectangleRefusal> {
    for index in 0..4 {
        let (from, to) = (
            corners[index].in_plane(),
            corners[(index + 1) % 4].in_plane(),
        );
        let segment = edges[index];
        let kind = if from[1] == to[1] {
            ConstraintKind::Horizontal { segment }
        } else if from[0] == to[0] {
            ConstraintKind::Vertical { segment }
        } else {
            // Not axis-aligned after all: the caller named the wrong frame for this drawing.
            return Err(RectangleRefusal::Unrepresentable);
        };
        assert_rectangle_relation(sketch, kind, context)?;
    }
    Ok(())
}

/// Assert that an ORIENTED rectangle stays a rectangle without pinning which way it faces:
/// both pairs of opposite sides parallel, and one corner square.
///
/// The fourth corner needs no relation of its own — three assertions against eight corner
/// freedoms leave the five an oriented rectangle has (position, rotation, width, height), and a
/// parallelogram with one right angle has four.
fn constrain_oriented_rectangle(
    sketch: &mut Sketch,
    edges: [EntityId; 4],
    context: EvaluationContext,
) -> Result<(), RectangleRefusal> {
    let relations = [
        ConstraintKind::Parallel {
            first: edges[0],
            second: edges[2],
        },
        ConstraintKind::Parallel {
            first: edges[1],
            second: edges[3],
        },
        ConstraintKind::Perpendicular {
            first: edges[0],
            second: edges[1],
        },
    ];
    for kind in relations {
        assert_rectangle_relation(sketch, kind, context)?;
    }
    Ok(())
}

fn canonical_rectangle(
    candidate: parametric::sketch::RectangleCandidate,
) -> Result<RectanglePlacement, RectangleRefusal> {
    let corners = candidate
        .corners
        .map(|corner| SketchPoint::try_from_continuous(corner[0], corner[1]))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| RectangleRefusal::Unrepresentable)?;
    let corners: [SketchPoint; 4] = corners
        .try_into()
        .map_err(|_| RectangleRefusal::Unrepresentable)?;
    if corners
        .iter()
        .enumerate()
        .any(|(index, corner)| corner.coincides(&corners[(index + 1) % 4]))
    {
        return Err(RectangleRefusal::Unrepresentable);
    }
    Ok(RectanglePlacement { corners })
}

fn canonical_polygon(
    candidate: parametric::sketch::PolygonCandidate,
) -> Result<PolygonPlacement, PolygonRefusal> {
    let center = SketchPoint::try_from_continuous(candidate.center[0], candidate.center[1])
        .map_err(|_| PolygonRefusal::Unrepresentable)?;
    let vertices: Vec<SketchPoint> = candidate
        .vertices
        .into_iter()
        .map(|vertex| SketchPoint::try_from_continuous(vertex[0], vertex[1]))
        .collect::<Result<_, _>>()
        .map_err(|_| PolygonRefusal::Unrepresentable)?;
    if vertices.len() < 3
        || vertices
            .iter()
            .enumerate()
            .any(|(index, vertex)| vertex.coincides(&vertices[(index + 1) % vertices.len()]))
    {
        return Err(PolygonRefusal::Unrepresentable);
    }
    Ok(PolygonPlacement { vertices, center })
}

fn canonical_slot(
    candidate: parametric::sketch::SlotCandidate,
) -> Result<SlotPlacement, SlotRefusal> {
    let edges = candidate
        .edges
        .map(|edge| match edge {
            parametric::sketch::SlotEdgeCandidate::Line { from, to } => {
                Ok(SlotEdgePlacement::Line {
                    from: SketchPoint::try_from_continuous(from[0], from[1])
                        .map_err(|_| SlotRefusal::Unrepresentable)?,
                    to: SketchPoint::try_from_continuous(to[0], to[1])
                        .map_err(|_| SlotRefusal::Unrepresentable)?,
                })
            }
            parametric::sketch::SlotEdgeCandidate::Arc {
                from,
                to,
                sweep_degrees,
            } => Ok(SlotEdgePlacement::Arc {
                from: SketchPoint::try_from_continuous(from[0], from[1])
                    .map_err(|_| SlotRefusal::Unrepresentable)?,
                to: SketchPoint::try_from_continuous(to[0], to[1])
                    .map_err(|_| SlotRefusal::Unrepresentable)?,
                sweep: parametric::units::AngleMeasurement::try_from_degrees_f64(sweep_degrees)
                    .map_err(|_| SlotRefusal::Unrepresentable)?,
            }),
        })
        .into_iter()
        .collect::<Result<Vec<_>, SlotRefusal>>()?;
    let edges: [SlotEdgePlacement; 4] =
        edges.try_into().map_err(|_| SlotRefusal::Unrepresentable)?;
    if edges.iter().any(|edge| match edge {
        SlotEdgePlacement::Line { from, to } | SlotEdgePlacement::Arc { from, to, .. } => {
            from.coincides(to)
        }
    }) {
        return Err(SlotRefusal::Unrepresentable);
    }
    let canonical = |at: [f64; 2]| {
        SketchPoint::try_from_continuous(at[0], at[1]).map_err(|_| SlotRefusal::Unrepresentable)
    };
    let spine = SlotSpinePlacement {
        start: canonical(candidate.spine.start)?,
        end: canonical(candidate.spine.end)?,
        center: match candidate.spine.turn {
            parametric::sketch::SlotTurn::Straight => None,
            parametric::sketch::SlotTurn::About(center) => Some(canonical(center)?),
        },
    };
    let reach = match candidate.reach {
        Some([first, second]) => Some([canonical(first)?, canonical(second)?]),
        None => None,
    };
    Ok(SlotPlacement {
        edges,
        junctions: candidate.junctions,
        spine,
        reach,
    })
}

impl SketchSolid {
    /// Whether an extrude cell (in the producer's local voxel-index frame, PROVEN fully
    /// inside `[0, full_dim)` by the caller) is entirely solid — the coarse-solid test. The
    /// normal span is already `⊆ [0, height_voxels]` (the caller's full-inside check +
    /// `grid_dimensions()[normal] = height_voxels`), so solidity
    /// reduces to: the cell's in-plane footprint RECTANGLE is entirely inside the profile
    /// polygon. The rectangle is the SAMPLE-CENTER span, exactly as
    /// [`resolve_extrude`](Self::resolve_extrude) samples occupancy
    /// (`profile = bbox_min + idx + 0.5`): a cell spanning local `[c_lo, c_hi)` maps to
    /// `[min + c_lo + 0.5, min + c_hi − 0.5]`. Testing that (not the voxel corners) elides an
    /// axis-aligned FACE block — fully solid, but with its face lattice line collinear with
    /// the profile edge — while never over-claiming (the edge sits 0.5 beyond the outermost
    /// sample center).
    pub(super) fn extrude_cell_is_solid(
        &self,
        cell: voxel_core::spatial_index::VoxelAabb,
        context: EvaluationContext,
    ) -> bool {
        let Some((min, _max)) = self.profile_bounds(context) else {
            return false;
        };
        let [in_plane_0, in_plane_1] = self.sketch.plane.in_plane_axes();
        let c0_lo = (min[0] + cell.min[in_plane_0]) as f64 + 0.5;
        let c0_hi = (min[0] + cell.max[in_plane_0]) as f64 - 0.5;
        let c1_lo = (min[1] + cell.min[in_plane_1]) as f64 + 0.5;
        let c1_hi = (min[1] + cell.max[in_plane_1]) as f64 - 0.5;
        let region = self.sketch.region(context);
        substrate::geom2d::rectangle_inside_region(
            &to_region_points(&region),
            &to_region_curve_bounds(&region),
            [c0_lo, c1_lo],
            [c0_hi, c1_hi],
        )
    }

    /// Whether a revolve cell (PROVEN fully inside `[0, full_dim)` by the caller) is
    /// entirely solid — the coarse-solid test. Handles BOTH a full turn AND a PARTIAL
    /// wedge: a partial sweep is coarse-solid only when the cell is
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
    ///    - axial: the SAMPLE-CENTER span `[axial_min + cell.min + 0.5, axial_min + cell.max − 0.5]`
    ///      (elides the axial END-CAP blocks, whose face is collinear with the profile edge);
    ///    - radius: over the two centered radial world axes (centered = `idx − half`), the
    ///      `[nearest, farthest]` distance from the axis over the cell's voxel-corner box,
    ///      widened by `EPS` so f32/f64 rounding can never SHRINK the tested rectangle below
    ///      the true sample coverage (a wider rectangle only makes "inside" rarer ⇒ never an
    ///      over-claim). Because the `−radius` branch only UNIONS more occupancy, `+radius`
    ///      solidity is SUFFICIENT even for an axis-straddling profile (matching full-turn).
    /// 2. ANGULAR (partial turns only) — the whole cell's sweep angle is inside `[0, turn]`
    ///    (see [`revolve_box_within_sweep_arc`]). At 360° the gate is inert, so a full turn
    ///    needs only condition 1.
    ///
    /// CONSERVATIVE-NEVER-NARROW: the two conditions use the SAME centered corner box the
    /// resolve derives its per-voxel samples from (a superset of the sample centers), so a
    /// coarse claim can never disagree with the per-voxel truth.
    pub(super) fn revolve_cell_is_solid(
        &self,
        cell: voxel_core::spatial_index::VoxelAabb,
        axis: RevolveAxis,
        sweep: RevolveSweep,
        dimensions: [u32; 3],
        context: EvaluationContext,
    ) -> bool {
        let Some((min, _max)) = self.profile_bounds(context) else {
            return false;
        };
        let [in_plane_0, in_plane_1] = self.sketch.plane.in_plane_axes();
        let normal = self.sketch.plane.normal_axis();
        // (axial, radial) reinterpretation + ascending radial sort (shared, matching
        // `resolve_revolve`).
        let (axial_world_axis, axial_min, radial_a, radial_b) =
            revolve_axes(axis, in_plane_0, in_plane_1, normal, [min[0], min[1]]);

        let half = [
            dimensions[0] as f64 / 2.0,
            dimensions[1] as f64 / 2.0,
            dimensions[2] as f64 / 2.0,
        ];

        // Axial rectangle range in profile-axial coords — the SAMPLE-CENTER span, matching
        // the resolve's `axial_min + idx + 0.5` sampler exactly (a single-voxel span
        // collapses to a point, handled by `rectangle_inside_polygon`).
        let axial_lo = (axial_min + cell.min[axial_world_axis]) as f64 + 0.5;
        let axial_hi = (axial_min + cell.max[axial_world_axis]) as f64 - 0.5;

        // Centered radial voxel-corner box per radial world axis (centered = idx − half).
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
        let region = self.sketch.region(context);
        if !substrate::geom2d::rectangle_inside_region(
            &to_region_points(&region),
            &to_region_curve_bounds(&region),
            [c0_lo, c1_lo],
            [c0_hi, c1_hi],
        ) {
            return false;
        }
        // Condition 1 (radial/axial) holds. A full turn needs nothing more (the sweep gate
        // is inert at 360°). A partial turn additionally requires the cell's ENTIRE angular
        // span inside `[0, turn]` — over the SAME centered radial corner box the resolve
        // derives each per-voxel sweep angle from.
        if sweep.turn_degrees >= 360 {
            return true;
        }
        revolve_box_within_sweep_arc(a_lo, a_hi, b_lo, b_hi, sweep.turn_degrees)
    }

    /// The extrude resolve: rasterize the profile once and sweep it across
    /// `height_voxels` layers along the plane normal.
    pub(super) fn resolve_extrude(
        &self,
        grid: &mut VoxelGrid,
        voxels_per_block: u32,
        height_voxels: u32,
        window_local_voxels: voxel_core::spatial_index::VoxelAabb,
        context: EvaluationContext,
    ) {
        let dimensions = self.grid_dimensions(context);
        // FULL dimensions even when only a window is written.
        grid.dimensions = dimensions;
        grid.occupied.clear();

        let Some((min, _max)) = self.profile_bounds(context) else {
            // Degenerate profile: empty occupancy, no panic.
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
        // full-window call cover the whole grid.
        let world_bounds = crate::voxel::clamp_window_to_grid(window_local_voxels, dimensions);
        let (cell_0_lo, cell_0_hi) = world_bounds[in_plane_0];
        let (cell_1_lo, cell_1_hi) = world_bounds[in_plane_1];
        let (layer_lo, layer_hi) = world_bounds[normal];
        // `grid_dimensions()` sets `dimensions[normal] = height_voxels`, so the
        // clamped normal range is already `⊆ [0, height_voxels)`.
        let _ = height_voxels;

        // Rasterize the 2D profile ONCE (axis-aligned extrusion ⇒ the same fill on every
        // layer along the normal) over the WINDOWED in-plane range, then sweep it across the
        // WINDOWED `normal` layers. A cell `(cell_0, cell_1)` at local origin `min` is
        // occupied iff its center `(min + cell + 0.5)` is inside the REGION — inside some
        // `Fill` loop and no `Hole` loop. The region test is on `min + cell`, which is
        // FULL-derived; only the iterated cell range narrows.
        let _ = (in_plane_span_0, in_plane_span_1);
        let derived = self.sketch.derived(context);
        let region_field = &derived.region_field;
        let mut filled_in_plane: Vec<[u32; 2]> = Vec::new();
        for cell_1 in cell_1_lo..cell_1_hi {
            let sample_1 = min[1] as f32 + cell_1 as f32 + 0.5;
            for cell_0 in cell_0_lo..cell_0_hi {
                let sample_0 = min[0] as f32 + cell_0 as f32 + 0.5;
                if region_field.contains([sample_0, sample_1]) {
                    filled_in_plane.push([cell_0, cell_1]);
                }
            }
        }

        // The voxel's grid index per world axis, assembled from the in-plane cell
        // and the normal layer, then CORNER-ANCHORED (center = idx + 0.5) exactly the
        // way `SdfShape::resolve` does, so a rectangle extrude is byte-identical to the
        // matching `Box`. The center is a half-integer for any grid size → always on
        // the global voxel lattice.
        //
        // The normal-axis layer sweep is SERIAL, deliberately.
        //
        // The layers are order-independent, so it once ran through rayon the way
        // `SdfShape::resolve` slices its work. But the expensive half of this resolve — the
        // in-plane `contains` raster above — is serial regardless, and what the sweep does is
        // copy the same fill up the normal. Parallelizing a memory write bought nothing, and it
        // was NESTED: every boundary block of every dirty chunk opened its own fan-out inside the
        // chunk fan-out that is already saturating the pool. Measured on a real drag frame, the
        // nested split cost 40% at 24 threads and the whole rebuild got SLOWER as cores were
        // added. The chunk is the parallel granularity; below it, do the work.
        let profile_axes = [in_plane_0, in_plane_1, normal];
        grid.occupied = (layer_lo..layer_hi)
            .flat_map(|layer| {
                let [in_plane_0, in_plane_1, normal] = profile_axes;
                filled_in_plane.iter().map(move |&[cell_0, cell_1]| {
                    let mut index = [0u32; 3];
                    index[in_plane_0] = cell_0;
                    index[in_plane_1] = cell_1;
                    index[normal] = layer;
                    build_voxel(index, density)
                })
            })
            .collect();
    }

    /// The revolve resolve: sweep the profile around an in-plane axis into a solid
    /// of revolution. The profile's `(axial, radial)` reinterpretation (per [`RevolveAxis`])
    /// is sampled at every grid cell:
    ///
    /// - The axial world axis maps the cell to profile-axial space the SAME way the
    ///   extrude rasterizer maps an in-plane span: `axial_min + idx + 0.5` (un-centered
    ///   profile-space mapping), so a rectangle-revolve is exact against a cylinder.
    /// - The two RADIAL world axes (the non-axial in-plane axis + the plane normal)
    ///   are CENTERED exactly like `SdfShape` (`idx + 0.5 − dim/2`); the radius is their
    ///   Euclidean length, so the revolve axis lands at the grid center.
    /// - A cell is inside iff the even-odd `point_in_polygon` test passes for the
    ///   reconstructed profile point `(+radial folded, profile_axial)` placed back into
    ///   the profile's native `(c0, c1)` slots.
    /// - PARTIAL turn: the swept angle `theta = atan2(centered[radial_b],
    ///   centered[radial_a])` (normalized to `[0, 360)`) gates the cell — kept iff
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
        context: EvaluationContext,
    ) {
        let dimensions = self.grid_dimensions(context);
        // FULL dimensions even when only a window is written.
        grid.dimensions = dimensions;
        grid.occupied.clear();

        // ONE evaluation, shared with the bound — see [`RevolveField`]. Every per-solid
        // constant (the (axial, radial) reinterpretation, the ascending radial-axis sort,
        // the centered half-extents, the straddle flag and the radial reach) is hoisted
        // into it ONCE here, out of the per-voxel loop; occupancy below is then literally
        // `signed_distance_at(..) <= SURFACE_ISOLEVEL` over that same function. Nothing here
        // re-decides occupancy with its own arithmetic: one set computed two ways rounds two
        // ways, which breaks the bound's conservative-never-narrow contract on samples landing
        // exactly on the surface.
        let derived = self.sketch.derived(context);
        let Some(field) = self.revolve_field(derived, axis, sweep, context) else {
            // Degenerate (no profile / zero turn / zero radial extent): empty, no panic.
            return;
        };
        let density = voxels_per_block.max(1);

        // Clamp the WORLD-axis window to `[0, full_dim)`; all per-cell math (half,
        // radial_max, the centered sample, profile_axial) stays FULL-derived — only
        // the iterated cell range narrows. A full-window call covers the whole grid.
        let [(win_x_lo, win_x_hi), (win_y_lo, win_y_hi), (win_z_lo, win_z_hi)] =
            crate::voxel::clamp_window_to_grid(window_local_voxels, dimensions);

        // Single-resolve allocation cap ([`MAX_GRID_VOXELS`]) — scoped to the WINDOW, not the
        // full grid. `resolve_into` only materializes the clamped window, so a huge full-grid
        // revolve is fine to resolve one small window at a time on the two-layer/brick path:
        // a per-chunk window never trips this. A full-grid cap here would instead return empty
        // for EVERY window of a large revolve. The cap still protects a genuine FULL-window
        // dense resolve (`resolve` / the `oracle`-gated whole-region resolvers), where the
        // window IS the full grid, from a blown allocation.
        // `clamp_window_to_grid` guarantees `hi >= lo` per axis, so each span is >= 0.
        let window_voxel_count = (win_x_hi - win_x_lo) as u64
            * (win_y_hi - win_y_lo) as u64
            * (win_z_hi - win_z_lo) as u64;
        if window_voxel_count > MAX_GRID_VOXELS {
            return;
        }

        // Iterate every grid cell. The axial axis uses an un-centered profile-space
        // mapping (matching the extrude rasterizer); the radial axes are centered.
        //
        // The outer `k` slices are order-independent (each samples a disjoint set of
        // voxels), so — mirroring `SdfShape::resolve` — each slice produces a local
        // `Vec<Voxel>` and rayon concatenates them. Emission ORDER may differ from the
        // serial version but the SET is identical. Windowing parallelizes over the
        // WINDOWED z range.
        grid.occupied = (win_z_lo..win_z_hi)
            .into_par_iter()
            .flat_map_iter(|k| {
                let mut local = Vec::new();
                for j in win_y_lo..win_y_hi {
                    for i in win_x_lo..win_x_hi {
                        let index = [i, j, k];
                        // The sample point in the producer's own [0, full_dim) frame.
                        // `index + 0.5` is exact in f32 for any real grid, so the field
                        // sees precisely the coordinates this loop formed.
                        let point = [
                            index[0] as f32 + 0.5,
                            index[1] as f32 + 0.5,
                            index[2] as f32 + 0.5,
                        ];
                        // RADIAL EARLY-OUT: a sample farther from the axis than any
                        // profile vertex cannot be inside, and the wedge `max` can only
                        // keep its distance positive — so skipping it is output-identical.
                        if field.beyond_radial_reach(point) {
                            continue;
                        }
                        // THE occupancy decision: the shared field, thresholded. Nothing
                        // here re-derives the wedge or the polygon test.
                        if field.signed_distance_at(point) > SURFACE_ISOLEVEL {
                            continue;
                        }

                        local.push(build_voxel(index, density));
                    }
                }
                local
            })
            .collect();
    }
}
