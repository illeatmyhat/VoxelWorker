//! 2D **sketch → extrude → volume** — the sketch-to-volume authoring atom.
//!
//! This is a SECOND [`VoxelProducer`](crate::voxel::VoxelProducer), alongside
//! [`SdfShape`](crate::voxel::SdfShape). It takes a grid-aligned plane plus a closed polygon
//! *profile* of voxel-granular points and extrudes that profile a whole number of voxels along
//! the plane normal, producing a prism. Primitives are sugar over it — a rectangle profile
//! extruded *is* a box, a circle profile extruded *is* a cylinder — so it resolves through the
//! SAME stamp / `CombineOp` / chunk path the SDF producer uses.
//!
//! **Leak-free by construction.** The profile points and the extrude span are integer voxels on
//! the lattice/sub-lattice — there is no implicit center anchor and so no half-block correction.
//! The producer samples CORNER-ANCHORED: the resolve tests the profile at `bbox_min + idx + 0.5`
//! (no `grid/2` centering anywhere — a revolve centers only its two RADIAL axes), and a sketch's
//! footprint is corner-anchored, so the block-lattice shift an implicit-center model would need is
//! identically zero. The resolve path treats a sketch leaf like a VoxelBody — no intrinsic block
//! size, no lattice snap — see `Scene::resolve_*`.
//!
//! Planes are AXIS-ALIGNED: the normal is one of ±X / ±Y / ±Z. The profile is a closed simple
//! polygon (≥3 points); a degenerate profile (fewer than 3 points, or zero area) resolves to
//! nothing rather than panicking.

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
    clippy::manual_midpoint,
    clippy::doc_link_code,
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::needless_collect,
    clippy::needless_pass_by_value,
    clippy::option_if_let_else,
    clippy::return_self_not_must_use,
    clippy::single_match_else,
    clippy::suboptimal_flops,
    clippy::too_long_first_doc_paragraph,
    clippy::unnecessary_wraps,
    clippy::use_self,
    clippy::similar_names
)]

mod constraint;
mod edges;
mod faces;
mod modify;
mod pattern;
mod produce;
mod region_memo;
mod solid;
#[cfg(test)]
mod tests;
mod transform;

pub use constraint::{
    Constraint, ConstraintKind, ConstraintRefusal, InternalContainment, LineSide, SketchCurve,
    SymmetryBranch, TangentBranch,
};
pub use faces::{Face, FaceKey};
pub use modify::{
    BreakPlacement, BreakRefusal, ChamferPlacement, ChamferRefusal, ExtendEndpoint,
    ExtendPlacement, ExtendRefusal, FilletPlacement, FilletRefusal, OffsetPlacement, OffsetRefusal,
    TrimPlacement, TrimRefusal,
};
pub use parametric::sketch::{SolveOutcome, SolveReport};
pub use parametric::{ArcSweep, CircleRadius, CurveParameter, ResolvedLength};
pub use pattern::{
    DerivedPatternCurve, SketchPattern, SketchPatternKind, SketchPatternRefusal, SketchVector,
};
pub use solid::SketchSolid;
pub use substrate::geom2d::LoopRole;
pub use transform::{SketchTransformEntity, SketchTransformRefusal};

use parametric::units::{AngleMeasurement, ExactRational, Measurement};
use std::num::NonZeroU32;

/// An operation reached a fixed measurement source without the document evaluation context that
/// resolves it. This is deliberately an error instead of a density default or stale cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SketchEvaluationError {
    /// A fixed source was encountered without the density needed to resolve it.
    MissingEvaluationContext,
    /// Persisted or programmatically constructed geometry violates the adapter's invariants.
    InvalidDocumentGeometry,
    /// A local solution could not be represented in durable document scalar storage.
    ScalarWritebackFailed,
    /// Standing relations cannot all hold. The ids are individually removable constraints whose
    /// removal restores satisfaction; an empty list means no single removal is sufficient.
    Unsatisfied { conflicts: Vec<EntityId> },
    /// A standing Tangent settled numerically but its derived contact escaped a finite authored
    /// curve or became singular. No candidate coordinates were applied.
    InvalidTangent {
        constraint: EntityId,
        error: parametric::sketch::TangentContactError,
    },
}

/// Why a connected tangent arc could not be appended without a partial document edit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TangentArcRefusal {
    UnsupportedIncoming,
    UnknownIncoming,
    NonIncidentIncoming,
    UnknownEndpoint,
    SelfLoop,
    Candidate(parametric::sketch::TangentArcCandidateError),
    UnrepresentableSweep,
    ArcRefused,
    Branch(parametric::sketch::BranchChoiceError),
    Constraint(ConstraintRefusal),
}

/// Why a center-first arc could not be represented as durable endpoint-and-sweep geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CenterArcRefusal {
    /// The continuous center/start/direction geometry is invalid or degenerate.
    Candidate(parametric::sketch::CenterArcCandidateError),
    /// A supplied stable start-point id no longer exists.
    UnknownStart,
    /// The derived endpoint or sweep cannot be represented in document scalar storage.
    Unrepresentable,
    /// The projected endpoints are already joined or otherwise refuse a new arc.
    ArcRefused,
}

/// Exact document-side geometry shared by Center Point Arc preview and commit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CenterArcPlacement {
    /// The construction center. Arc creation reifies it as a derived construction point; it is
    /// never an independent authored freedom.
    pub center: SketchPoint,
    /// The first on-curve endpoint, authoritative when it came from an existing point id.
    pub start: SketchPoint,
    /// The direction pick projected onto the fixed start radius.
    pub endpoint: SketchPoint,
    /// Continuous circular geometry, including exposed radius and counter-clockwise sweep.
    pub candidate: parametric::sketch::CenterArcCandidate,
}

/// Why a point-defined circle could not become durable center-and-radius geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointCircleRefusal {
    /// The continuous diameter/circumcircle construction is invalid.
    Candidate(parametric::sketch::CircleCandidateError),
    /// The solved center or radius lies outside canonical sketch scalar storage.
    Unrepresentable,
    /// A circle with the same center and radius already exists or storage refused it.
    CircleRefused,
}

/// Exact canonical geometry shared by point-defined circle previews and commits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointCirclePlacement {
    /// Canonical construction center.
    pub center: SketchPoint,
    /// Canonical free radius.
    pub radius: SketchLength,
    /// Continuous view of those same durable values.
    pub candidate: parametric::sketch::CircleCandidate,
}

/// Why a two- or three-line tangent circle could not be authored atomically.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TangentCircleRefusal {
    Candidate(parametric::sketch::TangentCircleCandidateError),
    UnknownSegment,
    Unrepresentable,
    CircleRefused,
    Branch(parametric::sketch::BranchChoiceError),
    Constraint(ConstraintRefusal),
}

/// Canonical tangent-circle geometry shared by preview and atomic commit.
#[derive(Debug, Clone, PartialEq)]
pub struct TangentCirclePlacement {
    pub center: SketchPoint,
    pub radius: SketchLength,
    /// Canonical contact locus corresponding to each selected segment.
    pub contacts: Vec<SketchPoint>,
}

/// Why a point-defined rectangle could not be appended atomically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RectangleRefusal {
    /// The continuous construction is degenerate or non-finite.
    Candidate(parametric::sketch::RectangleCandidateError),
    /// A solved corner cannot be represented distinctly in canonical point storage.
    Unrepresentable,
    /// Every boundary edge already exists, so the command would change nothing.
    AlreadyExists,
}

/// Canonical boundary corners shared by rectangle preview and commit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RectanglePlacement {
    /// Boundary-ordered corners, with the final edge closing index 3 back to index 0.
    pub corners: [SketchPoint; 4],
}

/// Why a regular polygon could not be appended atomically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolygonRefusal {
    /// The continuous construction or side count is invalid.
    Candidate(parametric::sketch::PolygonCandidateError),
    /// A solved vertex cannot be represented distinctly in canonical storage.
    Unrepresentable,
    /// The polygon would add no new boundary edge.
    AlreadyExists,
}

/// Canonical regular-polygon geometry shared by preview and commit.
#[derive(Debug, Clone, PartialEq)]
pub struct PolygonPlacement {
    /// Boundary vertices in traversal order.
    pub vertices: Vec<SketchPoint>,
    /// Canonical geometric center; a construction input rather than a persisted freedom.
    pub center: SketchPoint,
}

/// Why a slot boundary could not be appended atomically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotRefusal {
    /// The continuous construction is invalid or degenerate.
    Candidate(parametric::sketch::SlotCandidateError),
    /// A boundary endpoint or arc sweep cannot be represented in canonical storage.
    Unrepresentable,
    /// The complete boundary already exists.
    AlreadyExists,
}

/// One document-canonical boundary curve of a slot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SlotEdgePlacement {
    /// Straight boundary span.
    Line { from: SketchPoint, to: SketchPoint },
    /// Circular boundary span with its signed included angle.
    Arc {
        from: SketchPoint,
        to: SketchPoint,
        sweep: parametric::units::AngleMeasurement,
    },
}

/// Canonical four-curve slot boundary shared by preview and commit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlotPlacement {
    /// Connected boundary curves in traversal order.
    pub edges: [SlotEdgePlacement; 4],
}

/// The exact document-side geometry shared by standalone Tangent Arc preview and commit.
/// Radius and center stay derived curve data; the persisted arc remains its endpoint ids plus
/// intrinsic sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TangentArcPlacement {
    /// The existing endpoint where the new arc leaves its incoming curve.
    pub seam: SketchPoint,
    /// The canonical destination position commit will persist or reuse by coincidence.
    pub endpoint: SketchPoint,
    /// The derived circular geometry, including the radius exposed to callers and previews.
    pub candidate: parametric::sketch::TangentArcCandidate,
}

/// The document-canonical geometry of a midpoint-defined segment. These are the exact positions
/// preview must draw and commit will persist or reuse by coincidence. A reused point may retain
/// different [`SketchPoint::offset_measurements`] provenance without changing that positional
/// contract. The raw parametric candidate deliberately remains a separate type on the continuous
/// side of the adapter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MidpointLinePlacement {
    /// The transient construction input, canonicalized for preview but never persisted by the
    /// Midpoint Line tool merely because it was clicked.
    pub midpoint: SketchPoint,
    /// The endpoint supplied by the author.
    pub endpoint: SketchPoint,
    /// The endpoint reflected through [`midpoint`](Self::midpoint). Commit may reuse an existing
    /// point that [`SketchPoint::coincides`] with this canonical position.
    pub reflected: SketchPoint,
}

/// Why a midpoint-defined segment could not be appended atomically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MidpointLineRefusal {
    /// Raw continuous construction was nonfinite, overflowed, or exactly collapsed.
    Candidate(parametric::sketch::MidpointLineCandidateError),
    /// A raw coordinate could not be represented by canonical [`SketchPoint`] storage.
    Point(SketchPointConstructionError),
    /// The clicked endpoint id no longer names a point in this sketch.
    UnknownEndpoint,
    /// Canonicalization produced a self-loop or an exactly symmetric reflection cannot be stored.
    CanonicalCollapse,
    /// The two resolved endpoint ids are already joined by a straight segment.
    DuplicateSegment,
}

/// Build the explicit sketch evaluation context at a density-bearing boundary.
///
/// Zero is not a density. Returning `None` keeps a legacy [`crate::voxel::VoxelProducer`] caller from
/// fabricating geometry (especially a fixed curve) at density one; callers must instead take
/// their explicit invalid/non-geometric path.
#[must_use]
pub fn evaluation_context_from_density(
    voxels_per_block: u32,
) -> Option<parametric::EvaluationContext> {
    NonZeroU32::new(voxels_per_block).map(parametric::EvaluationContext::new)
}

fn map_prepare_evaluation_error(error: constraint::PrepareError) -> SketchEvaluationError {
    match error {
        constraint::PrepareError::MissingEvaluationContext => {
            SketchEvaluationError::MissingEvaluationContext
        }
        constraint::PrepareError::InvalidDocumentGeometry
        | constraint::PrepareError::InvalidLocalProblem(_) => {
            SketchEvaluationError::InvalidDocumentGeometry
        }
    }
}

fn validate_prepared_tangent_contacts(
    prepared: &constraint::PreparedProblem,
    solution: &parametric::sketch::Solution,
) -> Result<(), SketchEvaluationError> {
    if let Some(failure) = prepared
        .first_tangent_contact_failure(solution)
        .map_err(map_prepare_evaluation_error)?
    {
        return Err(SketchEvaluationError::InvalidTangent {
            constraint: failure.constraint,
            error: failure.error,
        });
    }
    Ok(())
}

fn validate_prepared_satisfaction(
    prepared: &constraint::PreparedProblem,
    diagnostics: &parametric::sketch::Diagnostics,
) -> Result<(), SketchEvaluationError> {
    if !diagnostics.satisfied {
        return Err(SketchEvaluationError::Unsatisfied {
            conflicts: prepared
                .standing_conflicts()
                .map_err(map_prepare_evaluation_error)?,
        });
    }
    Ok(())
}

fn deserialize_arc_sweep<'de, D>(deserializer: D) -> Result<ArcSweep, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Stored {
        #[serde(default)]
        free: Option<AngleMeasurement>,
        #[serde(default)]
        fixed: Option<AngleMeasurement>,
        #[serde(default)]
        degrees_numerator: Option<i128>,
        #[serde(default)]
        degrees_denominator: Option<i128>,
    }

    let stored = <Stored as serde::Deserialize>::deserialize(deserializer)?;
    match (
        stored.free,
        stored.fixed,
        stored.degrees_numerator,
        stored.degrees_denominator,
    ) {
        (Some(value), None, None, None) => Ok(ArcSweep::free(value)),
        (None, Some(value), None, None) => Ok(ArcSweep::fixed(value)),
        (None, None, Some(numerator), Some(denominator)) => {
            ExactRational::new(numerator, denominator)
                .map(AngleMeasurement::new)
                .map(ArcSweep::free)
                .ok_or_else(|| serde::de::Error::custom("legacy angle has a zero denominator"))
        }
        _ => Err(serde::de::Error::custom(
            "arc sweep must contain exactly one complete authority",
        )),
    }
}

fn deserialize_circle_radius<'de, D>(deserializer: D) -> Result<CircleRadius, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Stored {
        #[serde(default)]
        free: Option<ResolvedLength>,
        #[serde(default)]
        fixed: Option<Measurement>,
        #[serde(default)]
        voxels: Option<i64>,
        #[serde(default)]
        local_voxels: Option<f32>,
        #[serde(default)]
        measurement: Option<Measurement>,
    }

    let stored = <Stored as serde::Deserialize>::deserialize(deserializer)?;
    match (
        stored.free,
        stored.fixed,
        stored.voxels,
        stored.local_voxels,
        stored.measurement,
    ) {
        (Some(value), None, None, None, None) => positive_circle_radius(value)
            .ok_or_else(|| serde::de::Error::custom("circle radius must be strictly positive")),
        (None, Some(source), None, None, None) => Ok(CircleRadius::fixed(source)),
        (None, None, Some(voxels), local_voxels, measurement) => match measurement {
            Some(source) => Ok(CircleRadius::fixed(source)),
            None => {
                ResolvedLength::try_from_f64(voxels as f64 + local_voxels.unwrap_or(0.0) as f64)
                    .map_err(|_| serde::de::Error::custom("legacy radius is not finite"))
                    .and_then(|value| {
                        positive_circle_radius(value).ok_or_else(|| {
                            serde::de::Error::custom(
                                "legacy circle radius must be strictly positive",
                            )
                        })
                    })
            }
        },
        _ => Err(serde::de::Error::custom(
            "circle radius must contain exactly one complete authority",
        )),
    }
}

fn circle_radius_from_sketch_length(value: SketchLength) -> Option<CircleRadius> {
    match value.measurement {
        Some(source) => Some(CircleRadius::fixed(source)),
        None => ResolvedLength::try_from_f64(value.value())
            .ok()
            .and_then(positive_circle_radius),
    }
}

fn positive_circle_radius(value: ResolvedLength) -> Option<CircleRadius> {
    (value.value() > 0.0).then_some(CircleRadius::free(value))
}

/// Which axis the sketch plane's normal points along — i.e. the axis the profile
/// is EXTRUDED along.
///
/// The two in-plane axes (the ones the 2D profile lives in) are the OTHER two
/// world axes, taken in ascending order so the mapping is unambiguous:
///
/// | normal | in-plane axis 0 | in-plane axis 1 |
/// |--------|-----------------|-----------------|
/// | `X`    | Y               | Z               |
/// | `Y`    | X               | Z               |
/// | `Z`    | X               | Y               |
///
/// Sign of the normal does not change the resolved occupancy (an axis-aligned
/// prism is symmetric about its own grid), so only the bare axis is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PlaneAxis {
    /// Profile in the YZ plane, extruded along X.
    X,
    /// Profile in the XZ plane, extruded along Y.
    Y,
    /// Profile in the XY plane, extruded along Z (Z-up: the footprint-extrude-up
    /// default — profile on the XY ground, extruded up along +Z).
    Z,
}

impl PlaneAxis {
    /// The two WORLD axes the 2D profile lives in, in ascending order
    /// (`in_plane_axes()[0]` is profile coordinate 0, `[1]` is profile
    /// coordinate 1). The remaining axis is the extrude/normal axis.
    pub fn in_plane_axes(self) -> [usize; 2] {
        match self {
            PlaneAxis::X => [1, 2], // Y, Z
            PlaneAxis::Y => [0, 2], // X, Z
            PlaneAxis::Z => [0, 1], // X, Y
        }
    }

    /// The WORLD axis the profile is extruded along (the plane normal).
    pub fn normal_axis(self) -> usize {
        match self {
            PlaneAxis::X => 0,
            PlaneAxis::Y => 1,
            PlaneAxis::Z => 2,
        }
    }
}

/// Why a continuous coordinate cannot be represented by canonical [`SketchPoint`] storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SketchPointConstructionError {
    /// At least one supplied coordinate is NaN or infinite.
    NonFinite,
    /// A coordinate lies outside the split `i64 + [0, 1)` representation, including carry
    /// overflow after the fractional part narrows to `f32`.
    OutOfCanonicalRange,
}

/// One vertex of a sketch profile — a 2D point on the plane's in-plane axes (see
/// [`PlaneAxis::in_plane_axes`]), carried as the full node-position representation, mirroring
/// `NodeTransform`: a canonical integer voxel coordinate, a sub-voxel remainder, and an
/// optionally-retained authored [`Measurement`] per axis.
///
/// The in-plane position is `offset_voxels + offset_local_voxels`
/// ([`in_plane`](Self::in_plane) — integer first, then the fraction, the same
/// composition rule as `NodeTransform::world_field_position_voxels`). Coordinates may
/// be negative; the producer normalizes the profile's bounding box (floored) to the
/// local grid origin at resolve, so absolute values only matter relative to the other
/// points.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SketchPoint {
    /// In-plane voxel coordinates `[axis0, axis1]` at the document density `d`.
    pub offset_voxels: [i64; 2],
    /// Sub-voxel remainder per axis, in `[0, 1)` — written by `snap = None`
    /// authoring; a voxel/block snap zeroes it.
    #[serde(default)]
    pub offset_local_voxels: [f32; 2],
    /// The RETAINED authored `Length` expression per axis, or `None` for a plain snapped
    /// point. `SetDensity` re-evaluates a retained expression so a measurement-authored
    /// profile keeps its physical shape across a density re-target; the canonical
    /// `offset_voxels` always wins for geometry.
    #[serde(default)]
    pub offset_measurements: Option<[Measurement; 2]>,
}

impl SketchPoint {
    /// Split one finite continuous coordinate into the document's canonical integer and local
    /// parts. The positive bound is EXCLUSIVE: `i64::MAX as f64` rounds to `2^63`, so accepting
    /// it and casting would silently claim the unrepresentable coordinate is `i64::MAX`.
    fn try_split_continuous(coord: f64) -> Result<(i64, f32), SketchPointConstructionError> {
        const LOWER: f64 = i64::MIN as f64;
        const UPPER_EXCLUSIVE: f64 = -(i64::MIN as f64);

        if !coord.is_finite() {
            return Err(SketchPointConstructionError::NonFinite);
        }
        if !(LOWER..UPPER_EXCLUSIVE).contains(&coord) {
            return Err(SketchPointConstructionError::OutOfCanonicalRange);
        }
        let floor = coord.floor();
        Self::finish_continuous_split(floor as i64, coord - floor)
    }

    /// Finish a split after narrowing the local fraction. A value just below one can round to
    /// `1.0f32`; carrying it keeps the documented `[0, 1)` invariant. Kept separate so the carry
    /// overflow at `i64::MAX` is directly testable even though no `f64` has sub-voxel resolution
    /// that high.
    fn finish_continuous_split(
        voxel: i64,
        fraction: f64,
    ) -> Result<(i64, f32), SketchPointConstructionError> {
        if !fraction.is_finite() || !(0.0..1.0).contains(&fraction) {
            return Err(SketchPointConstructionError::OutOfCanonicalRange);
        }
        let local = fraction as f32;
        if local >= 1.0 {
            return voxel
                .checked_add(1)
                .map(|carried| (carried, 0.0))
                .ok_or(SketchPointConstructionError::OutOfCanonicalRange);
        }
        Ok((voxel, local))
    }

    /// A profile vertex at the given whole-voxel in-plane coordinates (no fraction,
    /// no retained expression).
    pub fn new(axis0: i64, axis1: i64) -> Self {
        Self {
            offset_voxels: [axis0, axis1],
            offset_local_voxels: [0.0; 2],
            offset_measurements: None,
        }
    }

    /// A profile vertex at a CONTINUOUS in-plane coordinate: floor lands in
    /// `offset_voxels`, the fraction in `offset_local_voxels` (the `snap = None`
    /// authoring door). A non-finite coordinate is sanitized to zero:
    /// a `NaN` fraction would poison every position-equality the producer guards
    /// no-op commits with.
    pub fn from_continuous(axis0: f64, axis1: f64) -> Self {
        let split = |coord: f64| -> (i64, f32) {
            if !coord.is_finite() {
                return (0, 0.0);
            }
            let floor = coord.floor();
            Self::finish_continuous_split(floor as i64, coord - floor)
                .unwrap_or((floor as i64, (coord - floor) as f32))
        };
        let (voxels_0, local_0) = split(axis0);
        let (voxels_1, local_1) = split(axis1);
        Self {
            offset_voxels: [voxels_0, voxels_1],
            offset_local_voxels: [local_0, local_1],
            offset_measurements: None,
        }
    }

    /// A profile vertex at the supplied continuous coordinate, refusing input that cannot be
    /// represented as canonical `i64 + f32` parts. Unlike [`from_continuous`](Self::from_continuous),
    /// this programmatic-authoring door never sanitizes or saturates invalid input.
    pub fn try_from_continuous(
        axis0: f64,
        axis1: f64,
    ) -> Result<Self, SketchPointConstructionError> {
        let (voxels_0, local_0) = Self::try_split_continuous(axis0)?;
        let (voxels_1, local_1) = Self::try_split_continuous(axis1)?;
        Ok(Self {
            offset_voxels: [voxels_0, voxels_1],
            offset_local_voxels: [local_0, local_1],
            offset_measurements: None,
        })
    }

    /// The continuous in-plane position: `offset_voxels + offset_local_voxels` per
    /// axis (integer first, then the fraction — exact for the integer part).
    pub fn in_plane(&self) -> [f64; 2] {
        [
            self.offset_voxels[0] as f64 + self.offset_local_voxels[0] as f64,
            self.offset_voxels[1] as f64 + self.offset_local_voxels[1] as f64,
        ]
    }

    /// The same position in the **measurement** width, narrowed from the `i64` source DIRECTLY
    /// rather than by casting [`in_plane`](Self::in_plane).
    ///
    /// `i64 → f64 → f32` can land a vertex on a different `f32` than `i64 → f32` does, and a
    /// double-rounded vertex reintroduces exactly the CPU/GPU divergence the narrowing exists to
    /// remove. Two conversions from one integer truth, not one conversion and a cast.
    pub fn in_plane_measured(&self) -> [f32; 2] {
        [
            self.offset_voxels[0] as f32 + self.offset_local_voxels[0],
            self.offset_voxels[1] as f32 + self.offset_local_voxels[1],
        ]
    }

    /// Whether two points sit at the SAME in-plane position — the coincidence predicate
    /// (coincidence IS shared identity). Position only: a retained measurement is provenance,
    /// not location, so it never splits two coincident points into twins.
    pub fn coincides(&self, other: &SketchPoint) -> bool {
        self.offset_voxels == other.offset_voxels
            && self.offset_local_voxels == other.offset_local_voxels
    }

    /// Whether this point is the exact midpoint of `first` and `second` in canonical document
    /// storage. This avoids composing either endpoint into one large `f64`, where a small midpoint
    /// or sub-voxel remainder could disappear before the equality is asked.
    #[cfg(test)]
    pub(crate) fn is_exact_midpoint_of(&self, first: &SketchPoint, second: &SketchPoint) -> bool {
        matches!(self.exact_reflection_of(first), Ok(Some(reflected)) if reflected.coincides(second))
    }

    /// Reflect `endpoint` through this point using the canonical split representation itself.
    /// `Ok(None)` means the mathematical reflection exists but its local fraction needs more
    /// precision than `f32` can store exactly. Range failure is kept distinct so callers can
    /// report an out-of-document coordinate rather than a rounding collapse.
    #[allow(clippy::float_cmp)]
    pub(crate) fn exact_reflection_of(
        &self,
        endpoint: &SketchPoint,
    ) -> Result<Option<Self>, SketchPointConstructionError> {
        // Knuth's TwoSum returns an exact two-component expansion of a floating-point sum. The
        // error component matters when, for example, a minimum subnormal endpoint is reflected
        // through 0.5: plain f64 arithmetic rounds `1.0 - 2^-149` to `1.0`.
        let two_sum = |a: f64, b: f64| {
            let sum = a + b;
            let b_virtual = sum - a;
            let error = (a - (sum - b_virtual)) + (b - b_virtual);
            (sum, error)
        };
        let reflect_axis = |axis: usize| -> Result<Option<(i64, f32)>, _> {
            let integer =
                2 * i128::from(self.offset_voxels[axis]) - i128::from(endpoint.offset_voxels[axis]);
            let (fraction, fraction_error) = two_sum(
                2.0 * f64::from(self.offset_local_voxels[axis]),
                -f64::from(endpoint.offset_local_voxels[axis]),
            );
            let mut fraction_floor = fraction.floor();
            if fraction == fraction_floor && fraction_error < 0.0 {
                fraction_floor -= 1.0;
            }
            let voxel = integer + fraction_floor as i128;
            let voxel = i64::try_from(voxel)
                .map_err(|_| SketchPointConstructionError::OutOfCanonicalRange)?;
            let (exact_local, local_error) = two_sum(fraction - fraction_floor, fraction_error);
            let local = exact_local as f32;
            if local_error != 0.0 || f64::from(local) != exact_local {
                return Ok(None);
            }
            Ok(Some((voxel, local)))
        };

        let Some((voxel_0, local_0)) = reflect_axis(0)? else {
            return Ok(None);
        };
        let Some((voxel_1, local_1)) = reflect_axis(1)? else {
            return Ok(None);
        };
        Ok(Some(Self {
            offset_voxels: [voxel_0, voxel_1],
            offset_local_voxels: [local_0, local_1],
            offset_measurements: None,
        }))
    }

    /// This point re-targeted from `old_density` to `new_density` — the `SetDensity`
    /// arm. A retained measurement RE-EVALUATES at the new density (lossless block
    /// scaling; a non-dividing axis floors and resynthesizes its retained form, exactly
    /// `NodeTransform::from_measurements`). A plain point rescales its continuous position
    /// so it keeps its physical place, the way a node rescale keeps a non-parametric offset's.
    pub fn retargeted(&self, old_density: u32, new_density: u32) -> Self {
        if let Some(measurements) = self.offset_measurements {
            let resolve_axis = |measurement: Measurement| -> (i64, Measurement) {
                match measurement.to_voxels(new_density) {
                    Ok(voxels) => (voxels, measurement),
                    Err(parametric::units::MeasurementError::BlockTermNotWholeVoxels {
                        nearest_floor_voxels,
                        ..
                    }) => (
                        nearest_floor_voxels,
                        Measurement::from_voxels(nearest_floor_voxels),
                    ),
                    Err(parametric::units::MeasurementError::ZeroDensity) => {
                        let voxels = measurement.voxel_term();
                        (voxels, Measurement::from_voxels(voxels))
                    }
                }
            };
            let (voxels_0, retained_0) = resolve_axis(measurements[0]);
            let (voxels_1, retained_1) = resolve_axis(measurements[1]);
            Self {
                offset_voxels: [voxels_0, voxels_1],
                offset_local_voxels: self.offset_local_voxels,
                offset_measurements: Some([retained_0, retained_1]),
            }
        } else {
            let scale = new_density.max(1) as f64 / old_density.max(1) as f64;
            let [axis0, axis1] = self.in_plane();
            Self::from_continuous(axis0 * scale, axis1 * scale)
        }
    }
}

/// A scalar sketch length — a circle's radius, and whatever else the tool suite dimensions. The
/// one-dimensional twin of [`SketchPoint`], carried the same way for the same reasons: a canonical
/// integer voxel count, a sub-voxel remainder, and an optionally retained authored
/// [`Measurement`].
///
/// It is a separate type rather than a bare `f64` because a radius has to survive a density
/// re-target: `2 blocks` is a different voxel count at `d16` and `d32`, and only the retained
/// expression knows that.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SketchLength {
    /// Whole voxels at the document density `d`.
    pub voxels: i64,
    /// Sub-voxel remainder in `[0, 1)`.
    #[serde(default)]
    pub local_voxels: f32,
    /// The RETAINED authored expression, or `None` for a plain snapped length.
    #[serde(default)]
    pub measurement: Option<Measurement>,
}

impl SketchLength {
    /// A whole-voxel length.
    pub fn new(voxels: i64) -> Self {
        Self {
            voxels,
            local_voxels: 0.0,
            measurement: None,
        }
    }

    /// A voxel-authored length whose unit survives a density retarget. This differs from
    /// [`new`](Self::new): a plain stored length scales with the drawing, while this retained
    /// source continues to mean exactly `voxels` at the new density.
    pub fn retained_voxels(voxels: i64) -> Self {
        Self {
            voxels,
            local_voxels: 0.0,
            measurement: Some(Measurement::from_voxels(voxels)),
        }
    }

    /// A CONTINUOUS length: floor lands in [`voxels`](Self::voxels), the fraction in
    /// [`local_voxels`](Self::local_voxels). A non-finite input sanitizes to zero, the same
    /// `NaN` guard [`SketchPoint::from_continuous`] keeps.
    pub fn from_continuous(voxels: f64) -> Self {
        if !voxels.is_finite() {
            return Self::new(0);
        }
        let floor = voxels.floor();
        Self {
            voxels: floor as i64,
            local_voxels: (voxels - floor) as f32,
            measurement: None,
        }
    }

    /// The continuous value: integer part first, then the fraction.
    pub fn value(&self) -> f64 {
        self.voxels as f64 + self.local_voxels as f64
    }

    /// The same value in the **measurement** width, narrowed from the `i64` source directly
    /// ([`SketchPoint::in_plane_measured`] keeps the same discipline and says why).
    pub fn measured(&self) -> f32 {
        self.voxels as f32 + self.local_voxels
    }

    /// This length re-targeted from `old_density` to `new_density`, exactly as
    /// [`SketchPoint::retargeted`] treats one coordinate.
    pub fn retargeted(&self, old_density: u32, new_density: u32) -> Self {
        let Some(measurement) = self.measurement else {
            let scale = new_density.max(1) as f64 / old_density.max(1) as f64;
            return Self::from_continuous(self.value() * scale);
        };
        let (voxels, retained) = match measurement.to_voxels(new_density) {
            Ok(voxels) => (voxels, measurement),
            Err(parametric::units::MeasurementError::BlockTermNotWholeVoxels {
                nearest_floor_voxels,
                ..
            }) => (
                nearest_floor_voxels,
                Measurement::from_voxels(nearest_floor_voxels),
            ),
            Err(parametric::units::MeasurementError::ZeroDensity) => {
                let voxels = measurement.voxel_term();
                (voxels, Measurement::from_voxels(voxels))
            }
        };
        Self {
            voxels,
            local_voxels: self.local_voxels,
            measurement: Some(retained),
        }
    }
}

/// A stable, monotonically-allocated identifier for a sketch entity (a point or a
/// segment). **Never a `Vec` index** — an index shifts when an entity is deleted, which
/// would silently corrupt every reference; a stable id does not. Ids are handed out
/// once and never reused.
pub type EntityId = u32;

/// Whether an entity is real geometry or a construction/reference line that never bounds
/// a region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum EntityRole {
    /// Real geometry — participates in region derivation.
    #[default]
    Real,
    /// Reference geometry — never bounds a region.
    Construction,
}

impl EntityRole {
    /// The opposite participation role. This is deliberately a role operation rather than a
    /// delete-and-recreate: stable ids, constraints, curve lineage, and face picks all survive a
    /// construction toggle.
    pub const fn toggled(self) -> Self {
        match self {
            Self::Real => Self::Construction,
            Self::Construction => Self::Real,
        }
    }
}

/// One loop of the profile: a closed boundary of [`ProfileEdge`]s plus how it contributes to the
/// region. The unit the 2D CSG folds and the unit the overlay draws.
///
/// The boundary keeps its **curves**. Flattening happens at [`flatten`](Self::flatten), which only
/// the consumers that genuinely produce something discrete call — a voxel grid, a crease polyline,
/// the exact-`f64` cell classifier.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileLoop {
    /// Whether the loop's interior is added or carved out.
    pub role: LoopRole,
    /// The closed boundary, counter-clockwise. The last edge's head is the first edge's tail.
    pub edges: Vec<ProfileEdge>,
}

impl ProfileLoop {
    /// The loop as a closed polygon, each chord's sagitta within `sagitta_tolerance_voxels`.
    ///
    /// **A terminal adapter, not a stage.** Every caller of this is producing something discrete
    /// and has nowhere to put a curve; anything that merely wants to know where the boundary is
    /// asks the field instead.
    pub fn flatten(&self, sagitta_tolerance_voxels: f64) -> Vec<SketchPoint> {
        flatten_edges(&self.edges, sagitta_tolerance_voxels)
    }

    /// The loop's corners — every edge's tail, and nothing an arc passes through in between.
    pub fn corners(&self) -> impl Iterator<Item = SketchPoint> + '_ {
        self.edges.iter().map(|edge| edge.from)
    }

    /// The loop's boundary in the **measurement** width, for the region field.
    pub fn measured(&self) -> Vec<substrate::geom2d::RegionEdge> {
        self.edges.iter().map(ProfileEdge::measured).collect()
    }
}

/// A closed edge loop as a closed polygon, each chord's sagitta within `sagitta_tolerance_voxels`.
///
/// **A terminal adapter, not a stage.** Reach for it only where something discrete is being
/// produced and there is nowhere to put a curve — a crease polyline, a screen-space hit-test
/// polygon, the exact-`f64` cell classifier. Anything that merely wants to know where the boundary
/// is asks the field ([`substrate::geom2d::signed_distance_to_region`]) instead.
pub fn flatten_edges(edges: &[ProfileEdge], sagitta_tolerance_voxels: f64) -> Vec<SketchPoint> {
    let mut points = Vec::with_capacity(edges.len());
    for edge in edges {
        points.push(edge.from);
        points.extend(edge.interior_points(sagitta_tolerance_voxels));
    }
    points
}

/// One boundary edge of a [`ProfileLoop`]: a straight span from `from` to `to`, or — when `arc` is
/// present — the circular arc joining them.
///
/// This is the sketch's half of the contract [`substrate::geom2d::RegionEdge`] states: a curve
/// stays a curve from derivation all the way to the measurement, and no consumer inherits a chord
/// count somebody upstream chose for it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfileEdge {
    /// The tail.
    pub from: SketchPoint,
    /// The head.
    pub to: SketchPoint,
    /// The circle this edge follows, or `None` for a straight span.
    pub arc: Option<ProfileArc>,
    /// The rational Bézier this edge follows, or `None` for a segment/arc. Exactly one of `arc`
    /// and `bezier` may be present.
    pub bezier: Option<substrate::rational_bezier::RationalBezier>,
}

/// The circle a curved [`ProfileEdge`] follows, solved once from the canonical endpoints-plus-bulge
/// form.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfileArc {
    /// The circle's center, in profile voxels.
    pub center: [f64; 2],
    /// The circle's radius, in voxels.
    pub radius: f64,
    /// The bearing of the edge's tail from the center.
    pub start_radians: f64,
    /// The signed angle traveled tail → head; positive counter-clockwise.
    pub sweep_radians: f64,
}

impl ProfileEdge {
    /// A straight span.
    pub fn straight(from: SketchPoint, to: SketchPoint) -> Self {
        ProfileEdge {
            from,
            to,
            arc: None,
            bezier: None,
        }
    }

    /// An arc through the signed `sweep_degrees`, or the plain chord when the sweep is degenerate
    /// — the same fallback [`arc_interior_points`] makes by returning nothing.
    pub fn curved(from: SketchPoint, to: SketchPoint, sweep_degrees: f64) -> Self {
        let Some((center, radius)) =
            arc_center_radius(from.in_plane(), to.in_plane(), sweep_degrees)
        else {
            return ProfileEdge::straight(from, to);
        };
        let tail = from.in_plane();
        ProfileEdge {
            from,
            to,
            arc: Some(ProfileArc {
                center,
                radius,
                start_radians: (tail[1] - center[1]).atan2(tail[0] - center[0]),
                sweep_radians: sweep_degrees.to_radians(),
            }),
            bezier: None,
        }
    }

    /// A whole circle as ONE closed edge: tail and head are the same point, and the arc
    /// sweeps a full turn counter-clockwise about `center`.
    ///
    /// The seam sits at bearing zero — `center + [radius, 0]` — matching
    /// [`substrate::geom2d::RegionEdge`]'s convention so the CPU field and its WGSL mirror cut the
    /// circle in the same place. It is a seam and not a vertex: the document holds no [`Point`]
    /// there, nothing may snap to it, and moving the circle moves it with no trace.
    pub fn circle(center: [f64; 2], radius: f64) -> Self {
        let seam = SketchPoint::from_continuous(center[0] + radius, center[1]);
        ProfileEdge {
            from: seam,
            to: seam,
            arc: Some(ProfileArc {
                center,
                radius,
                start_radians: 0.0,
                sweep_radians: std::f64::consts::TAU,
            }),
            bezier: None,
        }
    }

    /// Whether this edge closes on itself — a whole circle rather than a span between two
    /// distinct points. Such an edge is a loop all by itself.
    pub fn is_closed(&self) -> bool {
        self.arc
            .is_some_and(|arc| arc.sweep_radians.abs() >= std::f64::consts::TAU)
    }

    /// The same edge walked the other way — what a half-edge traversal against the stored direction
    /// gets. An arc keeps its circle and reverses its sweep.
    pub fn reversed(&self) -> Self {
        ProfileEdge {
            from: self.to,
            to: self.from,
            arc: self.arc.map(|arc| ProfileArc {
                start_radians: arc.start_radians + arc.sweep_radians,
                sweep_radians: -arc.sweep_radians,
                ..arc
            }),
            bezier: self.bezier.map(|curve| curve.reversed()),
        }
    }

    /// The direction the edge LEAVES its tail in, as an angle in `(-pi, pi]`. An arc departs along
    /// its tangent — a quarter turn off the radius, on the side it curves toward — which is what
    /// makes two arcs sharing an endpoint order correctly around that vertex.
    pub fn departure_radians(&self) -> f64 {
        if let Some(curve) = self.bezier {
            let tangent = curve.derivative_at(0.0);
            return tangent[1].atan2(tangent[0]);
        }
        match self.arc {
            Some(arc) => {
                let quarter = std::f64::consts::FRAC_PI_2 * arc.sweep_radians.signum();
                let tangent = arc.start_radians + quarter;
                tangent.sin().atan2(tangent.cos())
            }
            None => {
                let (from, to) = (self.from.in_plane(), self.to.in_plane());
                (to[1] - from[1]).atan2(to[0] - from[0])
            }
        }
    }

    /// The edge's contribution to the enclosed signed area, by Green's theorem
    /// `½∮(x dy − y dx)`. **Exact for an arc**: integrating the parameterized circle gives
    /// `½[r²·sweep + cx·Δy − cy·Δx]`, so a bulge contributes the area it really encloses rather
    /// than the area of the chords approximating it.
    pub fn signed_area_term(&self) -> f64 {
        let (from, to) = (self.from.in_plane(), self.to.in_plane());
        if let Some(curve) = self.bezier {
            return curve
                .flatten(1.0e-5)
                .array_windows::<2>()
                .map(|pair| 0.5 * (pair[0][0] * pair[1][1] - pair[1][0] * pair[0][1]))
                .sum();
        }
        match self.arc {
            Some(arc) => {
                0.5 * (arc.radius * arc.radius * arc.sweep_radians
                    + arc.center[0] * (to[1] - from[1])
                    - arc.center[1] * (to[0] - from[0]))
            }
            None => 0.5 * (from[0] * to[1] - to[0] * from[1]),
        }
    }

    /// The edge's tessellated INTERIOR points (both endpoints exclusive), empty for a straight
    /// span. The one place a tolerance enters, reached only through [`ProfileLoop::flatten`].
    ///
    /// It walks the SOLVED circle rather than re-deriving one from the endpoints, which is what
    /// lets a closed curve through at all: a full turn has a zero-length chord, and there is no
    /// circle to be recovered from that.
    pub fn interior_points(&self, sagitta_tolerance_voxels: f64) -> Vec<SketchPoint> {
        if let Some(curve) = self.bezier {
            let points = curve.flatten(sagitta_tolerance_voxels);
            return points
                .iter()
                .skip(1)
                .take(points.len().saturating_sub(2))
                .map(|point| SketchPoint::from_continuous(point[0], point[1]))
                .collect();
        }
        match self.arc {
            Some(arc) => arc_interior_on_circle(arc, sagitta_tolerance_voxels),
            None => Vec::new(),
        }
    }

    /// The edge in the **measurement** width — what the region field folds, on the CPU and in the
    /// wash's WGSL mirror alike.
    ///
    /// Endpoints narrow from the `i64` whole-voxel source directly
    /// ([`SketchPoint::in_plane_measured`]), so a vertex lands on the same `f32` here as it does
    /// everywhere else.
    pub fn measured(&self) -> substrate::geom2d::RegionEdge {
        let start = self.from.in_plane_measured();
        let end = self.to.in_plane_measured();
        if let Some(curve) = self.bezier {
            return substrate::geom2d::RegionEdge::RationalBezier {
                control: curve
                    .control
                    .map(|point| [point[0] as f32, point[1] as f32]),
                weights: curve.weights.map(|weight| weight as f32),
            };
        }
        match self.arc {
            Some(arc) => substrate::geom2d::RegionEdge::Arc {
                start,
                end,
                center: [arc.center[0] as f32, arc.center[1] as f32],
                radius: arc.radius as f32,
                start_radians: arc.start_radians as f32,
                sweep_radians: arc.sweep_radians as f32,
            },
            None => substrate::geom2d::RegionEdge::Segment { start, end },
        }
    }

    /// The TIGHT bounds of the edge in profile voxels — an arc's own extent, which reaches past
    /// its chord at every bulge. What a profile's EXTENT must be measured from.
    pub fn bounds(&self) -> ([f64; 2], [f64; 2]) {
        let (from, to) = (self.from.in_plane(), self.to.in_plane());
        let mut low = [from[0].min(to[0]), from[1].min(to[1])];
        let mut high = [from[0].max(to[0]), from[1].max(to[1])];
        if let Some(arc) = self.arc {
            for quarter in 0..4 {
                let bearing = quarter as f64 * std::f64::consts::FRAC_PI_2;
                let travelled = if arc.sweep_radians < 0.0 {
                    (arc.start_radians - bearing).rem_euclid(std::f64::consts::TAU)
                } else {
                    (bearing - arc.start_radians).rem_euclid(std::f64::consts::TAU)
                };
                if travelled > arc.sweep_radians.abs() {
                    continue;
                }
                let reach = [
                    arc.center[0] + arc.radius * bearing.cos(),
                    arc.center[1] + arc.radius * bearing.sin(),
                ];
                for axis in 0..2 {
                    low[axis] = low[axis].min(reach[axis]);
                    high[axis] = high[axis].max(reach[axis]);
                }
            }
        }
        if let Some(curve) = self.bezier {
            let (curve_low, curve_high) = curve.control_bounds();
            low = [low[0].min(curve_low[0]), low[1].min(curve_low[1])];
            high = [high[0].max(curve_high[0]), high[1].max(curve_high[1])];
        }
        (low, high)
    }
}

/// A point entity: a first-class, independently add/delete-able vertex on the sketch
/// plane, referenced by segments and arcs through its stable [`id`](Self::id). A point
/// with no incident edge is a legal FREE point.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Point {
    /// Stable identity — segments reference this, not the point's `Vec` slot.
    pub id: EntityId,
    /// The point's in-plane position (see [`SketchPoint`]).
    pub at: SketchPoint,
    /// Real vs construction geometry.
    #[serde(default)]
    pub role: EntityRole,
}

/// A line-segment entity joining two [`Point`]s **by id**. Coincidence IS shared
/// identity: two segments meet because they name the same endpoint point, not
/// because a solver forced their coordinates equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Segment {
    /// Stable identity.
    pub id: EntityId,
    /// Endpoint point id (tail).
    pub from: EntityId,
    /// Endpoint point id (head).
    pub to: EntityId,
    /// Lineage id for region identity across edits: a fresh segment's `origin` is its own
    /// `id`; on split, both children inherit the parent's `origin`, so subdividing a loop
    /// edge leaves a face's boundary origin-SET unchanged.
    pub origin: EntityId,
    /// Real vs construction geometry.
    #[serde(default)]
    pub role: EntityRole,
}

/// A circular-arc entity joining two [`Point`]s **by id**. The canonical stored form is
/// the two endpoints plus one included-angle bulge — compact, unambiguous, fully
/// parametric; center and radius are DERIVED. Creation tools (the 3-point tool) compute
/// this form; their extra inputs (the through-point) are consumed at creation, never
/// persisted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Arc {
    /// Stable identity.
    pub id: EntityId,
    /// Endpoint point id (tail).
    pub from: EntityId,
    /// Endpoint point id (head).
    pub to: EntityId,
    /// The signed included angle: the arc sweeps from [`from`](Self::from) to
    /// [`to`](Self::to) **counter-clockwise in the plane's in-plane basis for a positive
    /// angle**, clockwise for a negative one. Magnitude strictly inside `(0, 360)` — zero
    /// and full-turn bulges are degenerate and erased by [`Sketch::repair`].
    #[serde(deserialize_with = "deserialize_arc_sweep")]
    pub bulge: ArcSweep,
    /// The [`Point`] entity standing at the arc's center — a REIFIED derived value. Its
    /// coordinates are recomputed from the endpoints and the bulge by
    /// [`Sketch::sync_arc_centers`] and are never authored directly, but it is a real point
    /// entity with a stable id so it selects, snaps and drags exactly like every other
    /// sketch point. Always [`EntityRole::Construction`]: a center never bounds a region.
    /// `serde(default)` yields [`ABSENT_CENTER`] for a document that names no center, which
    /// [`Sketch::repair`] materializes on load.
    #[serde(default = "absent_center")]
    pub center: EntityId,
    /// Lineage id for region identity across edits, like [`Segment::origin`].
    pub origin: EntityId,
    /// Real vs construction geometry.
    #[serde(default)]
    pub role: EntityRole,
}

/// A whole-circle entity: a center [`Point`] **by id** plus a radius.
///
/// A closed curve is its own loop. There is no on-curve vertex to anchor it to and none is
/// invented — a circle drawn on an empty plane bounds a face immediately, where an arc has to meet
/// something to bound anything. The center is the handle: dragging it moves the circle, and
/// changing [`radius`](Self::radius) resizes it, so the two authored degrees of freedom are exactly
/// the two the shape has.
///
/// The center is always [`EntityRole::Construction`] — a center is not on the boundary, so it never
/// bounds a region, exactly as an [`Arc`]'s center does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Circle {
    /// Stable identity.
    pub id: EntityId,
    /// The [`Point`] entity at the circle's center. Unlike an [`Arc`]'s center this is AUTHORED,
    /// not derived: it is where the author put it, and nothing recomputes it.
    pub center: EntityId,
    /// The one authoritative radius: a free exact solved length or a fixed measurement source.
    #[serde(deserialize_with = "deserialize_circle_radius")]
    pub radius: CircleRadius,
    /// Lineage id for region identity across edits, like [`Segment::origin`].
    pub origin: EntityId,
    /// Real vs construction geometry.
    #[serde(default)]
    pub role: EntityRole,
}

/// One rational cubic Bézier piece whose four controls are stable [`Point`] entities.
///
/// Unit weights describe an ordinary cubic spline piece. Non-unit positive weights also describe
/// exact conics, including the quarter-ellipse pieces emitted by the ellipse tools. Keeping the
/// controls as point references lets adjacent pieces share endpoints and gives programmatic
/// authors the same stable handles as interactive tools.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Bezier {
    /// Stable identity of this curve piece.
    pub id: EntityId,
    /// Endpoint, two tangent controls, and endpoint in parameter order.
    pub controls: [EntityId; 4],
    /// Strictly-positive homogeneous weights paired with [`controls`](Self::controls).
    pub weights: [f64; 4],
    /// Lineage shared by pieces created as one spline, conic, ellipse, or blend operation.
    pub origin: EntityId,
    /// Real vs construction geometry.
    #[serde(default)]
    pub role: EntityRole,
}

/// One closed ellipse authored by a center and two semi-axis gestures.
///
/// The three referenced points are handles, not boundary seams. Four exact rational Bézier
/// quarters are derived only when a geometric consumer asks for them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Ellipse {
    pub id: EntityId,
    pub center: EntityId,
    pub major_endpoint: EntityId,
    pub width_point: EntityId,
    pub origin: EntityId,
    #[serde(default)]
    pub role: EntityRole,
}

/// One endpoint/vertex/rho conic. Rho is exact and dimensionless in durable storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Conic {
    pub id: EntityId,
    pub from: EntityId,
    pub to: EntityId,
    pub vertex: EntityId,
    pub rho: parametric::ResolvedScalar,
    pub origin: EntityId,
    #[serde(default)]
    pub role: EntityRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SplineKind {
    FitPoint,
    ControlPoint,
}

/// One author-visible spline, regardless of how many cubic pieces its evaluator emits.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Spline {
    pub id: EntityId,
    /// Fit points or control-frame points according to [`kind`](Self::kind).
    pub points: Vec<EntityId>,
    pub kind: SplineKind,
    pub closed: bool,
    pub origin: EntityId,
    #[serde(default)]
    pub role: EntityRole,
}

impl Arc {
    pub(crate) fn sweep_degrees(self) -> f64 {
        self.bulge
            .free_value()
            .or_else(|| self.bulge.fixed_source())
            .expect("a curve parameter always has one authority")
            .to_degrees_f64()
    }

    fn replace_free_sweep(&mut self, sweep: AngleMeasurement) -> bool {
        if self.bulge.free_value().is_none() {
            return false;
        }
        self.bulge = ArcSweep::free(sweep);
        true
    }
}

impl Circle {
    pub(crate) fn free_radius_value(&self) -> Option<f64> {
        self.radius.free_value().map(|value| value.value())
    }

    pub(crate) fn resolved_radius(&self, context: parametric::EvaluationContext) -> f64 {
        match (self.radius.free_value(), self.radius.fixed_source()) {
            (Some(value), None) => value.value(),
            (None, Some(source)) => source.to_voxel_rational(context).to_f64(),
            // A malformed in-memory value cannot be constructed through the public parameter
            // doors, but treating it as non-finite lets repair/solver preflight reject it without
            // turning a corrupt document into a process-wide panic.
            _ => f64::NAN,
        }
    }

    fn rescale_free_radius(&mut self, old_density: u32, new_density: u32) {
        let Some(value) = self.radius.free_value().copied() else {
            return;
        };
        let Some(value) = value.scaled_by_ratio(new_density.max(1), old_density.max(1)) else {
            return;
        };
        self.radius = CircleRadius::free(value);
    }
}

/// The `center` of an arc that has no center point yet — a document that names none, or an
/// arc mid-construction. Ids are handed out monotonically from zero and never reused, so the top
/// of the range can never collide with a live entity.
pub const ABSENT_CENTER: EntityId = EntityId::MAX;

fn absent_center() -> EntityId {
    ABSENT_CENTER
}

/// Mutate a compact boxed store without making every [`Sketch`] carry a `Vec`'s spare-capacity
/// word. Higher curves are comparatively rare, while `Sketch` itself sits inside several enums
/// where its inline size matters.
fn boxed_push<T>(store: &mut Box<[T]>, value: T) {
    let mut values = std::mem::take(store).into_vec();
    values.push(value);
    *store = values.into_boxed_slice();
}

fn boxed_retain<T>(store: &mut Box<[T]>, mut keep: impl FnMut(&T) -> bool) {
    let mut values = std::mem::take(store).into_vec();
    values.retain(|value| keep(value));
    *store = values.into_boxed_slice();
}

/// A grid-aligned PLANE plus a collection of sketch ENTITIES — points, segments, arcs and
/// circles. The extrudable **profile is DERIVED** from the closed loops those entities bound
/// (see [`region`](Self::region)); it is never a hand-maintained ordered vertex list.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Sketch {
    /// Which axis the plane normal points along (axis-aligned only).
    pub plane: PlaneAxis,
    /// The point entities (unordered; loop order is derived, never stored).
    points: Vec<Point>,
    /// The segment entities joining points by id.
    segments: Vec<Segment>,
    /// The arc entities joining points by id. `serde(default)` so a document without arcs
    /// loads with none.
    #[serde(default)]
    arcs: Vec<Arc>,
    /// The whole-circle entities. `serde(default)` so a document without circles loads with
    /// none.
    #[serde(default)]
    circles: Vec<Circle>,
    /// Rational cubic curve pieces. `serde(default)` keeps older documents source-compatible.
    #[serde(default)]
    beziers: Box<[Bezier]>,
    /// Closed ellipse entities; absent in older documents.
    #[serde(default)]
    ellipses: Box<[Ellipse]>,
    /// Endpoint/vertex/rho conic entities; absent in older documents.
    #[serde(default)]
    conics: Box<[Conic]>,
    /// Author-visible fit/control splines; absent in older documents.
    #[serde(default)]
    splines: Box<[Spline]>,
    /// Associative operators whose instances are regenerated from authored curves. Generated
    /// curves deliberately have no entity ids of their own: constraints and direct edits continue
    /// to target the sources, so an operator adds no solver freedoms and cannot drift apart.
    #[serde(default, skip_serializing_if = "pattern::pattern_store_is_empty")]
    patterns: Box<[SketchPattern]>,
    /// The faces the author has UNPICKED, each named by a point inside it. Every derived
    /// face is picked by default, so this holds only the exceptions and is usually empty. A
    /// point inside no current face is inert, not an error: it costs nothing and lets an
    /// unpick survive an edit that temporarily breaks its boundary.
    ///
    /// It is a `Vec` and not a set because `f32` is not `Ord`.
    #[serde(default)]
    unpicked_points: Vec<FaceKey>,
    /// The constraint entities. `serde(default)` so a document without constraints loads
    /// with none.
    ///
    /// Deliberately absent from [`region_memo`]'s snapshot: a constraint does not change what the
    /// drawing looks like, only where a SOLVE would move it, and a solve moves points — which the
    /// snapshot already watches.
    #[serde(default)]
    constraints: Vec<Constraint>,
    /// The next id to hand out. Ids are monotonic and never reused, so this only grows.
    next_id: EntityId,
    /// The derived region, remembered between queries — see [`region_memo`]. Not document
    /// state: it is skipped by serde, clones empty, and compares equal, so a sketch is the
    /// same sketch whether or not it has derived itself yet.
    #[serde(skip)]
    region_memo: region_memo::RegionMemo,
}

impl Sketch {
    fn incoming_tangent_at(
        &self,
        curve: SketchCurve,
        seam: EntityId,
        context: parametric::EvaluationContext,
    ) -> Result<[f64; 2], TangentArcRefusal> {
        let point = |id: EntityId| {
            self.points
                .iter()
                .find(|point| point.id == id)
                .map(|point| point.at.in_plane())
        };
        match curve {
            SketchCurve::Circle(_)
            | SketchCurve::Bezier(_)
            | SketchCurve::Ellipse(_)
            | SketchCurve::Conic(_)
            | SketchCurve::Spline(_) => Err(TangentArcRefusal::UnsupportedIncoming),
            SketchCurve::Segment(id) => {
                let segment = self
                    .segments
                    .iter()
                    .find(|segment| segment.id == id)
                    .ok_or(TangentArcRefusal::UnknownIncoming)?;
                let other = if segment.to == seam {
                    segment.from
                } else if segment.from == seam {
                    segment.to
                } else {
                    return Err(TangentArcRefusal::NonIncidentIncoming);
                };
                let other = point(other).ok_or(TangentArcRefusal::UnknownIncoming)?;
                let seam = point(seam).ok_or(TangentArcRefusal::UnknownEndpoint)?;
                Ok([seam[0] - other[0], seam[1] - other[1]])
            }
            SketchCurve::Arc(id) => {
                let arc = self
                    .arcs
                    .iter()
                    .find(|arc| arc.id == id)
                    .ok_or(TangentArcRefusal::UnknownIncoming)?;
                let orientation = if arc.to == seam {
                    arc.sweep_degrees().signum()
                } else if arc.from == seam {
                    -arc.sweep_degrees().signum()
                } else {
                    return Err(TangentArcRefusal::NonIncidentIncoming);
                };
                let seam_at = point(seam).ok_or(TangentArcRefusal::UnknownEndpoint)?;
                let geometry = self
                    .curve_geometry(curve, context)
                    .ok_or(TangentArcRefusal::UnknownIncoming)?;
                let parametric::sketch::CurveGeometry::Circular(circle) = geometry else {
                    return Err(TangentArcRefusal::UnknownIncoming);
                };
                let radius = [seam_at[0] - circle.center[0], seam_at[1] - circle.center[1]];
                Ok([-orientation * radius[1], orientation * radius[0]])
            }
        }
    }

    /// Derive the tangent arc a preview and a commit share without exposing curve orientation.
    pub fn tangent_arc_candidate(
        &self,
        incoming: SketchCurve,
        seam: EntityId,
        target: [f64; 2],
        context: parametric::EvaluationContext,
    ) -> Result<parametric::sketch::TangentArcCandidate, TangentArcRefusal> {
        let tangent = self.incoming_tangent_at(incoming, seam, context)?;
        let seam = self
            .points
            .iter()
            .find(|point| point.id == seam)
            .map(|point| point.at.in_plane())
            .ok_or(TangentArcRefusal::UnknownEndpoint)?;
        parametric::sketch::tangent_arc_candidate(tangent, seam, target)
            .map_err(TangentArcRefusal::Candidate)
    }

    /// Resolve one persisted curve into continuous relation geometry at this evaluation context.
    pub fn curve_geometry(
        &self,
        curve: SketchCurve,
        context: parametric::EvaluationContext,
    ) -> Option<parametric::sketch::CurveGeometry> {
        use parametric::sketch::{ArcDomain, CircularCurve, CurveGeometry};
        let point = |id| {
            self.points
                .iter()
                .find(|point| point.id == id)
                .map(|point| point.at.in_plane())
        };
        match curve {
            SketchCurve::Segment(id) => {
                let edge = self.segments.iter().find(|edge| edge.id == id)?;
                Some(CurveGeometry::Segment {
                    from: point(edge.from)?,
                    to: point(edge.to)?,
                })
            }
            SketchCurve::Circle(id) => {
                let circle = self.circles.iter().find(|circle| circle.id == id)?;
                Some(CurveGeometry::Circular(CircularCurve {
                    center: point(circle.center)?,
                    radius: circle.resolved_radius(context),
                    arc: None,
                }))
            }
            SketchCurve::Bezier(_)
            | SketchCurve::Ellipse(_)
            | SketchCurve::Conic(_)
            | SketchCurve::Spline(_) => None,
            SketchCurve::Arc(id) => {
                let arc = self.arcs.iter().find(|arc| arc.id == id)?;
                let from = point(arc.from)?;
                let to = point(arc.to)?;
                let sweep = arc.sweep_degrees();
                let (center, radius) = arc_center_radius(from, to, sweep)?;
                Some(CurveGeometry::Circular(CircularCurve {
                    center,
                    radius,
                    arc: Some(ArcDomain {
                        from,
                        to,
                        sweep_radians: sweep.to_radians(),
                    }),
                }))
            }
        }
    }

    /// The current center of one authored circular curve. Radius sources are irrelevant to this
    /// query, so overlays can place center-based badges without fabricating evaluation context.
    pub fn circular_curve_center(&self, curve: SketchCurve) -> Option<[f64; 2]> {
        let point = |id: EntityId| {
            self.points
                .iter()
                .find(|point| point.id == id)
                .map(|point| point.at.in_plane())
        };
        match curve {
            SketchCurve::Arc(id) => {
                let arc = self.arcs.iter().find(|arc| arc.id == id)?;
                let (center, _) =
                    arc_center_radius(point(arc.from)?, point(arc.to)?, arc.sweep_degrees())?;
                Some(center)
            }
            SketchCurve::Circle(id) => {
                let circle = self.circles.iter().find(|circle| circle.id == id)?;
                point(circle.center)
            }
            SketchCurve::Segment(_)
            | SketchCurve::Bezier(_)
            | SketchCurve::Ellipse(_)
            | SketchCurve::Conic(_)
            | SketchCurve::Spline(_) => None,
        }
    }

    /// The shared center witness for two satisfied circular curves. The parametric kernel owns
    /// the numerical satisfaction boundary so document and overlay semantics remain identical.
    pub fn concentric_center(&self, first: SketchCurve, second: SketchCurve) -> Option<[f64; 2]> {
        if first.id() == second.id() {
            return None;
        }
        parametric::sketch::concentric_center(
            self.circular_curve_center(first)?,
            self.circular_curve_center(second)?,
        )
    }

    /// Pick the stable Tangent branch from canonical curve/locus pairs.
    pub fn choose_tangent_branch(
        &self,
        first: SketchCurve,
        first_locus: [f64; 2],
        second: SketchCurve,
        second_locus: [f64; 2],
        context: parametric::EvaluationContext,
    ) -> Result<TangentBranch, parametric::sketch::BranchChoiceError> {
        let (first, first_locus, second, second_locus) = if first.id() <= second.id() {
            (first, first_locus, second, second_locus)
        } else {
            (second, second_locus, first, first_locus)
        };
        parametric::sketch::choose_branch(
            self.curve_geometry(first, context)
                .ok_or(parametric::sketch::BranchChoiceError::Degenerate)?,
            first_locus,
            self.curve_geometry(second, context)
                .ok_or(parametric::sketch::BranchChoiceError::Degenerate)?,
            second_locus,
        )
    }

    /// Derive (never persist) a stored Tangent's current finite contact for overlays.
    pub fn tangent_contact(
        &self,
        first: SketchCurve,
        second: SketchCurve,
        branch: TangentBranch,
        context: parametric::EvaluationContext,
    ) -> Result<parametric::sketch::TangentContact, parametric::sketch::TangentContactError> {
        parametric::sketch::tangent_contact(
            self.curve_geometry(first, context)
                .ok_or(parametric::sketch::TangentContactError::InvalidBranch)?,
            self.curve_geometry(second, context)
                .ok_or(parametric::sketch::TangentContactError::InvalidBranch)?,
            branch,
        )
    }

    /// Choose the deterministic persisted Symmetry branch after canonicalizing its subjects.
    pub fn choose_symmetry_branch(
        &self,
        first: SketchCurve,
        second: SketchCurve,
        axis: EntityId,
        context: parametric::EvaluationContext,
    ) -> Result<SymmetryBranch, parametric::sketch::SymmetryError> {
        let identities_are_valid = first.id() != second.id()
            && first.id() != axis
            && second.id() != axis
            && matches!(
                (first, second),
                (SketchCurve::Segment(_), SketchCurve::Segment(_))
                    | (SketchCurve::Arc(_), SketchCurve::Arc(_))
                    | (SketchCurve::Circle(_), SketchCurve::Circle(_))
            );
        if !identities_are_valid {
            return Err(parametric::sketch::SymmetryError::UnsupportedPair);
        }
        let (first, second) = if first.id() <= second.id() {
            (first, second)
        } else {
            (second, first)
        };
        parametric::sketch::choose_symmetry_branch(
            self.curve_geometry(first, context)
                .ok_or(parametric::sketch::SymmetryError::UnsupportedPair)?,
            self.curve_geometry(second, context)
                .ok_or(parametric::sketch::SymmetryError::UnsupportedPair)?,
            self.curve_geometry(SketchCurve::Segment(axis), context)
                .ok_or(parametric::sketch::SymmetryError::DegenerateAxis)?,
        )
    }

    /// Derive one validated badge locus on the stored Symmetry axis.
    pub fn symmetry_badge_locus(
        &self,
        first: SketchCurve,
        second: SketchCurve,
        axis: EntityId,
        branch: SymmetryBranch,
        context: parametric::EvaluationContext,
    ) -> Result<[f64; 2], parametric::sketch::SymmetryError> {
        if !ConstraintKind::symmetry(first, second, axis, branch).symmetry_is_structurally_valid() {
            return Err(parametric::sketch::SymmetryError::InvalidBranch);
        }
        parametric::sketch::symmetry_witness(
            self.curve_geometry(first, context)
                .ok_or(parametric::sketch::SymmetryError::UnsupportedPair)?,
            self.curve_geometry(second, context)
                .ok_or(parametric::sketch::SymmetryError::UnsupportedPair)?,
            self.curve_geometry(SketchCurve::Segment(axis), context)
                .ok_or(parametric::sketch::SymmetryError::DegenerateAxis)?,
            branch,
        )
        .map(|witness| witness.at)
    }
    /// A sketch on `plane` whose entities form ONE closed loop through the given ordered
    /// points — the common case, and the constructor every caller still uses. Builds N
    /// point entities and N segments closing `p[i] → p[i+1]` and `p[last] → p[0]`. A
    /// 0/1-point profile adds no wrap segment (no self-loop); the result is empty or a
    /// lone free point.
    pub fn new(plane: PlaneAxis, profile: Vec<SketchPoint>) -> Self {
        let mut sketch = Self {
            plane,
            points: Vec::with_capacity(profile.len()),
            segments: Vec::with_capacity(profile.len()),
            arcs: Vec::new(),
            circles: Vec::new(),
            beziers: Box::default(),
            ellipses: Box::default(),
            conics: Box::default(),
            splines: Box::default(),
            patterns: Box::default(),
            unpicked_points: Vec::new(),
            constraints: Vec::new(),
            next_id: 0,
            region_memo: region_memo::RegionMemo::default(),
        };
        let ids: Vec<EntityId> = profile.iter().map(|&at| sketch.add_point(at)).collect();
        let n = ids.len();
        if n >= 2 {
            for i in 0..n {
                sketch.add_segment(ids[i], ids[(i + 1) % n]);
            }
        }
        sketch
    }

    /// A rectangle profile spanning `[0, width] × [0, height]` voxels on `plane` — the
    /// degenerate box footprint, and the demonstration that a box IS a rectangle extruded.
    /// The four corners are wound counter-clockwise; winding does not affect the even-odd
    /// rasterizer.
    pub fn rectangle(plane: PlaneAxis, width_voxels: i64, height_voxels: i64) -> Self {
        Self::new(
            plane,
            vec![
                SketchPoint::new(0, 0),
                SketchPoint::new(width_voxels, 0),
                SketchPoint::new(width_voxels, height_voxels),
                SketchPoint::new(0, height_voxels),
            ],
        )
    }

    /// An empty sketch on `plane` — no entities. A totally-empty sketch is first-class: it is
    /// a valid scene object that resolves to nothing, the start state a create-from-scratch
    /// sketch is authored into.
    pub fn empty(plane: PlaneAxis) -> Self {
        Self {
            plane,
            points: Vec::new(),
            segments: Vec::new(),
            arcs: Vec::new(),
            circles: Vec::new(),
            beziers: Box::default(),
            ellipses: Box::default(),
            conics: Box::default(),
            splines: Box::default(),
            patterns: Box::default(),
            unpicked_points: Vec::new(),
            constraints: Vec::new(),
            next_id: 0,
            region_memo: region_memo::RegionMemo::default(),
        }
    }

    /// A sketch on `plane` holding ONE circle of `radius_voxels` about `center` — the circle twin
    /// of [`rectangle`](Self::rectangle), and the shortest path to a profile with no straight edge
    /// in it at all.
    pub fn circle(plane: PlaneAxis, center: SketchPoint, radius_voxels: i64) -> Self {
        let mut sketch = Self::empty(plane);
        sketch.add_circle(center, SketchLength::new(radius_voxels));
        sketch
    }

    /// Read-only view of the point entities.
    pub fn points(&self) -> &[Point] {
        &self.points
    }

    /// Read-only view of the segment entities.
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// Read-only view of the arc entities.
    pub fn arcs(&self) -> &[Arc] {
        &self.arcs
    }

    /// Read-only view of the whole-circle entities.
    pub fn circles(&self) -> &[Circle] {
        &self.circles
    }

    /// Read-only view of rational cubic curve pieces.
    pub fn beziers(&self) -> &[Bezier] {
        &self.beziers
    }

    pub fn ellipses(&self) -> &[Ellipse] {
        &self.ellipses
    }

    pub fn conics(&self) -> &[Conic] {
        &self.conics
    }

    pub fn splines(&self) -> &[Spline] {
        &self.splines
    }

    /// Read-only view of associative mirror and pattern rules.
    pub fn patterns(&self) -> &[SketchPattern] {
        &self.patterns
    }

    /// Flip one geometry entity between real and construction while retaining its stable id.
    ///
    /// Arc and circle centers are structural construction points, not author geometry. They are
    /// refused here so no generic selection action can make a derived center participate as a
    /// profile vertex. Constraint ids and unknown ids are likewise harmless no-ops.
    pub fn toggle_construction(&mut self, id: EntityId) -> bool {
        let is_structural_center = self.arcs.iter().any(|arc| arc.center == id)
            || self.circles.iter().any(|circle| circle.center == id);
        if !is_structural_center {
            if let Some(point) = self.points.iter_mut().find(|point| point.id == id) {
                point.role = point.role.toggled();
                return true;
            }
        }
        if let Some(segment) = self.segments.iter_mut().find(|segment| segment.id == id) {
            segment.role = segment.role.toggled();
            return true;
        }
        if let Some(arc) = self.arcs.iter_mut().find(|arc| arc.id == id) {
            arc.role = arc.role.toggled();
            return true;
        }
        if let Some(circle) = self.circles.iter_mut().find(|circle| circle.id == id) {
            circle.role = circle.role.toggled();
            return true;
        }
        if let Some(bezier) = self.beziers.iter_mut().find(|bezier| bezier.id == id) {
            bezier.role = bezier.role.toggled();
            return true;
        }
        if let Some(ellipse) = self.ellipses.iter_mut().find(|ellipse| ellipse.id == id) {
            ellipse.role = ellipse.role.toggled();
            return true;
        }
        if let Some(conic) = self.conics.iter_mut().find(|conic| conic.id == id) {
            conic.role = conic.role.toggled();
            return true;
        }
        if let Some(spline) = self.splines.iter_mut().find(|spline| spline.id == id) {
            spline.role = spline.role.toggled();
            return true;
        }
        false
    }

    /// Test-only mutable access to the raw segment vector, for constructing the malformed
    /// stores the load-repair path is meant to erase.
    #[cfg(test)]
    pub(crate) fn segments_mut_for_test(&mut self) -> &mut Vec<Segment> {
        &mut self.segments
    }

    /// Test-only mutable access to the raw arc vector — the arc twin of
    /// [`segments_mut_for_test`](Self::segments_mut_for_test).
    #[cfg(test)]
    pub(crate) fn arcs_mut_for_test(&mut self) -> &mut Vec<Arc> {
        &mut self.arcs
    }

    /// Test-only mutable access to the raw circle vector.
    #[cfg(test)]
    pub(crate) fn circles_mut_for_test(&mut self) -> &mut Vec<Circle> {
        &mut self.circles
    }

    /// Test-only mutable access to the raw constraint vector. The public door trial-solves, so
    /// this is the only way to build the dangling constraint `repair` is meant to erase.
    #[cfg(test)]
    pub(crate) fn constraints_mut_for_test(&mut self) -> &mut Vec<Constraint> {
        &mut self.constraints
    }

    #[cfg(test)]
    pub(crate) fn region_memo_is_empty_for_test(&self) -> bool {
        self.region_memo.is_empty_for_test()
    }

    #[cfg(test)]
    pub(crate) fn region_derivation_count_for_test(&self) -> usize {
        self.region_memo.derivation_count_for_test()
    }

    /// Allocate a point entity at `at`, returning its fresh id.
    fn add_point(&mut self, at: SketchPoint) -> EntityId {
        let id = self.alloc_id();
        self.points.push(Point {
            id,
            at,
            role: EntityRole::Real,
        });
        id
    }

    /// Allocate a construction point at `at` — geometry that never bounds a region.
    fn add_construction_point(&mut self, at: SketchPoint) -> EntityId {
        let id = self.alloc_id();
        self.points.push(Point {
            id,
            at,
            role: EntityRole::Construction,
        });
        id
    }

    /// Allocate a segment `from → to`, its `origin` set to its own id (a root of its
    /// lineage), returning its fresh id.
    fn add_segment(&mut self, from: EntityId, to: EntityId) -> EntityId {
        let id = self.alloc_id();
        self.segments.push(Segment {
            id,
            from,
            to,
            origin: id,
            role: EntityRole::Real,
        });
        id
    }

    /// Hand out the next monotonic id.
    fn alloc_id(&mut self) -> EntityId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// The index into [`points`](Self::points) of the point with `id`, if present.
    fn point_index(&self, id: EntityId) -> Option<usize> {
        self.points.iter().position(|point| point.id == id)
    }

    /// The DERIVED bounded faces of the sketch's planar graph, in a deterministic order. Every
    /// face is a candidate region; whether it contributes solid or void is
    /// [`face_is_picked`](Self::face_is_picked).
    pub fn faces(&self, context: parametric::EvaluationContext) -> Vec<Face> {
        faces::derive(self, context)
    }

    /// The region in the **measurement** width — the exact value
    /// [`substrate::geom2d::signed_distance_to_region`] folds, and the exact value the wash's
    /// WGSL mirror is handed.
    ///
    /// One definition of the region, two evaluators of it: the resolve asks it per voxel on the
    /// CPU, the overlay asks it per pixel on the GPU. Curves arrive as curves, so neither is
    /// drawing a polygon the other chose the resolution of.
    pub fn region_field_loops(
        &self,
        context: parametric::EvaluationContext,
    ) -> Vec<(LoopRole, Vec<substrate::geom2d::RegionEdge>)> {
        self.derived(context).region_field_loops.clone()
    }

    /// The `Fill` loops' bounding box in voxels — the profile's FOOTPRINT, and what the producer
    /// sizes its grid from. `None` when nothing is filled.
    pub(super) fn filled_extent(
        &self,
        context: parametric::EvaluationContext,
    ) -> Option<([f64; 2], [f64; 2])> {
        self.derived(context).filled_extent
    }

    /// Whether the face containing this key's point contributes solid. Faces default to PICKED —
    /// the document stores only the unpicked exceptions.
    pub fn face_is_picked(&self, key: &FaceKey, context: parametric::EvaluationContext) -> bool {
        let faces = self.nested_faces(context);
        match innermost_face_at(&faces, key.interior_point) {
            Some(index) => self.pick_flags(&faces)[index],
            None => true,
        }
    }

    /// The identity of the face at `index` in [`faces`](Self::faces), or `None` when the index is
    /// past the end or the face is too thin to hold an interior point.
    ///
    /// The door for a caller holding a face by POSITION — the viewport keeps its hit-test polygons
    /// that way, because minting a key for every face on every frame is the search this whole
    /// arrangement is careful not to run.
    pub fn face_key_at(
        &self,
        index: usize,
        context: parametric::EvaluationContext,
    ) -> Option<FaceKey> {
        let faces = faces::derive(self, context);
        if index >= faces.len() {
            return None;
        }
        let nested: Vec<Face> = faces.iter().rev().cloned().collect();
        let mut keys = faces::identify(&nested);
        keys.reverse();
        keys[index]
    }

    /// The derived faces WITH their identities, in the same order as [`faces`](Self::faces) — for
    /// the callers that have to name a face to something outside the sketch (the viewport's carve
    /// menu, a test). Faces too thin to hold an interior point are dropped.
    ///
    /// This is the expensive door and the other one is not: minting an identity is a search
    /// costing some twenty times the arrangement that produced the face. Use
    /// [`faces`](Self::faces) for anything on a per-voxel or per-frame path, and reach for this
    /// only where a `FaceKey` is genuinely about to be stored or compared.
    pub fn identified_faces(&self, context: parametric::EvaluationContext) -> Vec<(Face, FaceKey)> {
        let faces = faces::derive(self, context);
        // `identify` wants nesting order — smallest first — and `faces()` is largest first, so the
        // reverse IS that order and reversing the answer puts it back.
        let nested: Vec<Face> = faces.iter().rev().cloned().collect();
        let mut keys = faces::identify(&nested);
        keys.reverse();
        faces
            .into_iter()
            .zip(keys)
            .filter_map(|(face, key)| key.map(|key| (face, key)))
            .collect()
    }

    /// Pick or unpick the face containing this key's point, carving or filling a pocket. Storing a
    /// point inside the face rather than its boundary's lineage means the intent survives
    /// re-derivation: a vertex drag, an edge split, and a curve drawn elsewhere all leave the same
    /// ground under the point, while a face that shrinks past it reverts to picked.
    pub fn set_face_picked(
        &mut self,
        key: FaceKey,
        picked: bool,
        context: parametric::EvaluationContext,
    ) {
        let faces = self.nested_faces(context);
        let Some(index) = innermost_face_at(&faces, key.interior_point) else {
            // Nothing is there to carve. An unpick still records the intent — it is inert until
            // an edit puts a face under it — but a pick has nothing to clear.
            if !picked {
                self.unpicked_points.push(key);
            }
            return;
        };
        // Whatever already names this face goes, so a pick clears it and an unpick replaces it
        // with the face's own current deepest point rather than accumulating near-duplicates.
        self.unpicked_points
            .retain(|stored| innermost_face_at(&faces, stored.interior_point) != Some(index));
        if !picked {
            // Store the face's OWN deepest point, not the one the caller happened to name it by —
            // the caller's may be a cursor position a hair from an edge, which the next edit walks
            // out of the face.
            let minted = faces::identify(&faces)[index];
            self.unpicked_points.push(minted.unwrap_or(key));
        }
    }

    /// The points naming the unpicked faces — the whole of the pick state the document carries.
    pub fn unpicked_faces(&self) -> impl Iterator<Item = &FaceKey> {
        self.unpicked_points.iter()
    }

    /// The derived faces in nesting order: smallest area first, so the FIRST face containing a
    /// point is the innermost one that does. [`substrate::geom2d::point_in_region`] takes the
    /// same order for the same reason.
    fn nested_faces(&self, context: parametric::EvaluationContext) -> Vec<Face> {
        let mut faces = faces::derive(self, context);
        // Ties keep `derive`'s deterministic order, so the region is stable across derivations.
        faces.sort_by(|first, second| first.area_voxels.total_cmp(&second.area_voxels));
        faces
    }

    /// Whether each of `faces` (in nesting order) is picked. An unpick point resolves to exactly
    /// one face — the innermost containing it — so an unpick inside a pocket never reads as an
    /// unpick of the shape around it.
    fn pick_flags(&self, faces: &[Face]) -> Vec<bool> {
        let mut picked = vec![true; faces.len()];
        for stored in &self.unpicked_points {
            if let Some(index) = innermost_face_at(faces, stored.interior_point) {
                picked[index] = false;
            }
        }
        picked
    }

    /// The DERIVED profile: one tagged loop per derived face, `Fill` where the face is picked and
    /// `Hole` where it is not, each a closed loop of edges **with its arcs intact**, ordered
    /// SMALLEST-AREA-FIRST.
    ///
    /// That order is [`substrate::geom2d::point_in_region`]'s contract: innermost-first, so each
    /// face decides its own area and nothing nested inside it. A face strictly inside another has
    /// strictly less area, so sorting on area IS the nesting order — no containment analysis
    /// needed. It is what makes carving a region leave a picked region inside it standing: the pick
    /// state of a face governs that face, and a face is the ground its own boundary encloses minus
    /// whatever sits within.
    ///
    /// This is what the producer resolves. The combination is an ordered fold over nesting, never a
    /// global crossing parity, so two fills that touch or share an edge both count where even-odd
    /// would cancel them.
    pub fn region(&self, context: parametric::EvaluationContext) -> Vec<ProfileLoop> {
        self.derived(context).region.clone()
    }

    /// The derived region, its measurement-width twin, and the filled extent, from the cache when
    /// the entity store has not moved — the door every per-voxel path goes through
    /// (see [`region_memo`]).
    pub(super) fn derived(
        &self,
        context: parametric::EvaluationContext,
    ) -> std::sync::Arc<region_memo::Derived> {
        self.region_memo.derived(self, context)
    }

    /// The region derived from scratch. Only [`region_memo`] calls this; everything else asks
    /// [`region`](Self::region) and gets the same answer without re-deriving it.
    fn region_uncached(&self, context: parametric::EvaluationContext) -> Vec<ProfileLoop> {
        let faces = self.nested_faces(context);
        let picked = self.pick_flags(&faces);
        faces
            .into_iter()
            .zip(picked)
            .map(|(face, picked)| ProfileLoop {
                role: if picked {
                    LoopRole::Fill
                } else {
                    LoopRole::Hole
                },
                edges: face.boundary,
            })
            .collect()
    }

    /// The profile's `Fill` loops only — what the region's EXTENT is measured from (a hole adds no
    /// footprint, and an unpicked face with nothing around it is not occupancy).
    pub fn filled_loops(&self, context: parametric::EvaluationContext) -> Vec<ProfileLoop> {
        self.region(context)
            .into_iter()
            .filter(|profile_loop| profile_loop.role == LoopRole::Fill)
            .collect()
    }

    /// The SIMPLE-profile door: the sole boundary when the region is exactly one picked face,
    /// flattened at the default tolerance, and empty otherwise (no face, an unpicked one, or
    /// several — those are questions only [`region`](Self::region) can answer). Callers that reason
    /// about a single closed outline (rectangle detection, most tests) want this; anything that
    /// resolves occupancy wants the region.
    pub fn flattened_loop(&self, context: parametric::EvaluationContext) -> Vec<SketchPoint> {
        let loops = self.region(context);
        match (loops.len(), loops.first().map(|first| first.role)) {
            (1, Some(LoopRole::Fill)) => loops[0].flatten(ARC_SAGITTA_TOLERANCE_VOXELS),
            _ => Vec::new(),
        }
    }

    /// Move the point `id` to `at` and settle the drawing around it — the drag write path.
    /// Reports whether the point exists.
    ///
    /// Dragging an arc's CENTER moves only the center: the endpoints hold still and the arc's
    /// radius follows the cursor ([`resweep_arc_to_center`](Self::resweep_arc_to_center)). Every
    /// other point simply takes `at`, and then the standing constraints are re-solved with it
    /// pinned there — see [`settle_under_the_hand`](Self::settle_under_the_hand). A constraint
    /// that only held at the moment it was asserted is not a constraint; it has to survive the
    /// next drag, which is the first thing the author does to test it.
    pub fn move_point(
        &mut self,
        id: EntityId,
        at: SketchPoint,
        context: parametric::EvaluationContext,
    ) -> Result<bool, SketchEvaluationError> {
        let Some(index) = self.point_index(id) else {
            return Ok(false);
        };
        let before_points = self.points.clone();
        let before_arcs = self.arcs.clone();
        let before_circles = self.circles.clone();
        let result = (|| -> Result<bool, SketchEvaluationError> {
            match self.arcs.iter().position(|arc| arc.center == id) {
                // An arc's center is DERIVED from its ends and its sweep, so there is no pinning it:
                // the resweep is the whole edit and no constraint can hold the result anywhere else.
                Some(arc_index) => {
                    self.resweep_arc_to_center(arc_index, at.in_plane());
                    self.sync_arc_centers();
                    // Center dragging owns only the arc sweep: settling here could "heal" an
                    // invalid resweep by moving its endpoints, breaking the special center-drag
                    // contract. Accept only the authored configuration produced by the resweep.
                    let prepared = constraint::prepare(self, &self.constraints, Some(context))
                        .map_err(map_prepare_evaluation_error)?;
                    let current = prepared.validate_current();
                    if let Some(failure) = current.tangent_failure {
                        let failure = prepared
                            .standing_tangent_failure(failure)
                            .map_err(map_prepare_evaluation_error)?;
                        Err(SketchEvaluationError::InvalidTangent {
                            constraint: failure.constraint,
                            error: failure.error,
                        })
                    } else if !current.satisfied || current.collapsed.is_some() {
                        Ok(false)
                    } else {
                        Ok(true)
                    }
                }
                None => {
                    self.points[index].at = at;
                    self.sync_arc_centers();
                    self.settle_under_the_hand(id, at, context)
                }
            }
        })();
        match result {
            Ok(true) => Ok(true),
            Ok(false) => {
                self.points = before_points;
                self.arcs = before_arcs;
                self.circles = before_circles;
                self.sync_arc_centers();
                Ok(false)
            }
            Err(error) => {
                self.points = before_points;
                self.arcs = before_arcs;
                self.circles = before_circles;
                self.sync_arc_centers();
                Err(error)
            }
        }
    }

    /// Re-solve the standing constraints with the hand pulling `held` toward `at`, writing the
    /// result back only if the standing residuals are met. Reports whether they were.
    ///
    /// The assertions hold DURING the gesture, not merely at the moment they were made.
    ///
    /// **The hand is a PULL, not a demand — two stages.** The drag joins the system as one more
    /// least-squares row and the solve trades it off against everything standing; then the hand
    /// lets go and the standing system alone is re-solved from that answer, which restores it
    /// exactly while moving as little as it can. The grabbed point therefore lands at the nearest
    /// place the drawing allows, and only the standing residuals decide whether the drag stands.
    ///
    /// **Not a hard pin.** A hard pin makes the whole drag all-or-nothing: a point free to slide
    /// along a line but not across it could not be moved AT ALL, because the cursor is essentially
    /// never exactly on that line and the pinned system reads as unsatisfiable. A vertical segment
    /// whose far end is held by an arc that two `Fix`es have already determined has one real
    /// freedom left — its length — and no way to use it. Sliding along the allowed direction is
    /// what the freedom count already promises.
    ///
    /// A drag that IS achievable is unaffected: stage one meets the pull exactly, so stage two
    /// starts at a solution and moves nothing.
    fn settle_under_the_hand(
        &mut self,
        held: EntityId,
        at: SketchPoint,
        context: parametric::EvaluationContext,
    ) -> Result<bool, SketchEvaluationError> {
        if self.constraints.is_empty() {
            return Ok(true);
        }
        let prepared = constraint::prepare(self, &self.constraints, Some(context))
            .map_err(map_prepare_evaluation_error)?;
        let (settled, accepted) = match prepared.drag(held, at.in_plane()) {
            Ok(parametric::sketch::DragOutcome::Accepted(settled)) => (settled, true),
            Ok(parametric::sketch::DragOutcome::Rejected(settled)) => (settled, false),
            Err(_) => return Ok(false),
        };
        validate_prepared_tangent_contacts(&prepared, &settled.solution)?;
        if !accepted {
            return Ok(false);
        }
        let plan = prepared
            .plan_apply(&self.points, &self.arcs, &self.circles, &settled.solution)
            .map_err(|_| SketchEvaluationError::ScalarWritebackFailed)?;
        plan.apply(self);
        self.sync_arc_centers();
        Ok(true)
    }

    /// Re-solve the arc at `arc_index` so its center sits as close to `target` as the canonical
    /// form allows, its endpoints unmoved.
    ///
    /// For a fixed chord a center has ONE degree of freedom, not two: it lives on the chord's
    /// perpendicular bisector, and where it sits along that line IS the sweep — far out for a
    /// shallow arc, on the chord for a half turn, across to the other side for the major one.
    /// So the drag projects onto the bisector and inverts `arc_center_radius`: the signed
    /// apothem `a` and the half-chord `h` give `sweep / 2 = atan2(h, a)`, which covers every
    /// positive sweep in `(0°, 360°)` as `a` runs over the reals. The existing sweep's SIGN is
    /// preserved — it says which way round the arc goes, and a drag of the center is not a
    /// request to reverse it. A degenerate chord or a sweep that quantizes to nothing leaves
    /// the arc alone rather than erasing it.
    fn resweep_arc_to_center(&mut self, arc_index: usize, target: [f64; 2]) {
        let arc = self.arcs[arc_index];
        let (Some(tail), Some(head)) = (self.point_index(arc.from), self.point_index(arc.to))
        else {
            return;
        };
        let (from, to) = (
            self.points[tail].at.in_plane(),
            self.points[head].at.in_plane(),
        );
        let chord = [to[0] - from[0], to[1] - from[1]];
        let chord_length = (chord[0] * chord[0] + chord[1] * chord[1]).sqrt();
        if chord_length <= f64::EPSILON {
            return;
        }
        let mid = [(from[0] + to[0]) / 2.0, (from[1] + to[1]) / 2.0];
        let left = [-chord[1] / chord_length, chord[0] / chord_length];
        let apothem = (target[0] - mid[0]) * left[0] + (target[1] - mid[1]) * left[1];
        let half_sweep = (chord_length / 2.0).atan2(apothem);
        let mut degrees = 2.0 * half_sweep.to_degrees();
        if arc.sweep_degrees() < 0.0 {
            degrees -= 360.0;
        }
        let Ok(bulge) = AngleMeasurement::try_from_degrees_f64(degrees) else {
            return;
        };
        if arc_sweep_is_valid(bulge.to_degrees_f64()) {
            self.arcs[arc_index].replace_free_sweep(bulge);
        }
    }

    /// Delete a point by id and every segment/arc incident to it. The edges' other endpoints
    /// survive as free points. No dangling reference can result: relations do not keep geometry
    /// alive, so their own liveness cascade follows after every geometry cascade.
    /// Deleting an arc's CENTER deletes that arc: the center is the arc's own derived
    /// geometry, so there is no arc left for it to be the center of.
    pub fn delete_point_cascade(&mut self, id: EntityId) {
        self.segments.retain(|seg| seg.from != id && seg.to != id);
        self.arcs
            .retain(|arc| arc.from != id && arc.to != id && arc.center != id);
        // A circle IS its center plus a radius, so deleting the center deletes the circle.
        self.circles.retain(|circle| circle.center != id);
        boxed_retain(&mut self.beziers, |bezier| !bezier.controls.contains(&id));
        boxed_retain(&mut self.ellipses, |ellipse| {
            ellipse.center != id && ellipse.major_endpoint != id && ellipse.width_point != id
        });
        boxed_retain(&mut self.conics, |conic| {
            conic.from != id && conic.to != id && conic.vertex != id
        });
        boxed_retain(&mut self.splines, |spline| !spline.points.contains(&id));
        self.points.retain(|point| point.id != id);
        self.prune_orphan_centers();
        self.drop_dangling_patterns();
        self.drop_dangling_constraints();
    }

    /// Delete the segment with id `seg_id`, **and each of its ends that nothing else draws**.
    /// No-op if `seg_id` is unknown.
    ///
    /// Ends that survived unconditionally as free points would leave two dots behind every
    /// deleted line, dots the author never placed and has no reason to want. A point the author
    /// *did* place stays — it is either an end of some other edge, an arc's center, or a
    /// circle's, and [`point_is_still_drawn`] asks exactly that question.
    ///
    /// **A constraint does not keep a point alive.** An assertion about a point is not a reason
    /// for the point to outlive the geometry it was drawn for, and the cascade takes the
    /// constraint with it — which is what the author asked for when they deleted the line.
    ///
    /// [`point_is_still_drawn`]: Self::point_is_still_drawn
    pub fn delete_segment(&mut self, seg_id: EntityId) {
        let Some(span) = self.segments.iter().find(|seg| seg.id == seg_id).copied() else {
            return;
        };
        self.segments.retain(|seg| seg.id != seg_id);
        self.drop_undrawn_points([span.from, span.to]);
        self.prune_orphan_centers();
        self.drop_dangling_patterns();
        self.drop_dangling_constraints();
    }

    /// Whether any geometry still draws this point — another edge's end, an arc's center, a
    /// circle's. Constraints deliberately do not count; see [`delete_segment`](Self::delete_segment).
    fn point_is_still_drawn(&self, id: EntityId) -> bool {
        self.segments
            .iter()
            .any(|seg| seg.from == id || seg.to == id)
            || self
                .arcs
                .iter()
                .any(|arc| arc.from == id || arc.to == id || arc.center == id)
            || self.circles.iter().any(|circle| circle.center == id)
            || self
                .beziers
                .iter()
                .any(|bezier| bezier.controls.contains(&id))
            || self.ellipses.iter().any(|ellipse| {
                ellipse.center == id || ellipse.major_endpoint == id || ellipse.width_point == id
            })
            || self
                .conics
                .iter()
                .any(|conic| conic.from == id || conic.to == id || conic.vertex == id)
            || self
                .splines
                .iter()
                .any(|spline| spline.points.contains(&id))
    }

    /// Erase each candidate that no geometry draws any more. Asked AFTER the edge has gone, so
    /// "still drawn" is a question about what is left rather than about what was.
    fn drop_undrawn_points(&mut self, candidates: impl IntoIterator<Item = EntityId>) {
        for id in candidates {
            if !self.point_is_still_drawn(id) {
                self.points.retain(|point| point.id != id);
            }
        }
    }

    /// The constraint entities, in the order they were authored.
    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    /// Add one persisted assertion after the parametric kernel has trial-solved it on a copy.
    /// A refusal changes neither geometry nor the stable-id counter: a rejected click must not burn
    /// an id, move even one point, or leave a half-created assertion for undo to discover. Only
    /// after the typed trial has accepted do we allocate, append, and apply its solution together.
    /// Accepted redundancy remains visible because it can express author intent even when rank adds
    /// no new information.
    #[allow(clippy::too_many_lines)]
    pub fn add_constraint(
        &mut self,
        kind: ConstraintKind,
        context: parametric::EvaluationContext,
    ) -> Result<EntityId, ConstraintRefusal> {
        let kind = kind.normalized();
        self.check_names_live_geometry(kind, context)?;
        self.check_is_not_already_asserted(kind)?;
        let prepared = constraint::prepare(self, &self.constraints, Some(context)).map_err(
            |error| match error {
                constraint::PrepareError::MissingEvaluationContext => {
                    ConstraintRefusal::MissingEvaluationContext
                }
                constraint::PrepareError::InvalidDocumentGeometry
                | constraint::PrepareError::InvalidLocalProblem(_) => ConstraintRefusal::Impossible,
            },
        )?;
        let trial = prepared.trial_add(kind).map_err(|error| match error {
            constraint::TrialMapError::UnmappedGeometry
            | constraint::TrialMapError::Request(
                parametric::sketch::RequestError::UnknownPoint
                | parametric::sketch::RequestError::InvalidRelation(
                    parametric::sketch::BuildError::UnknownPoint
                    | parametric::sketch::BuildError::UnknownSegment
                    | parametric::sketch::BuildError::UnknownParameter,
                ),
            ) => ConstraintRefusal::UnknownEntity,
            constraint::TrialMapError::Request(
                parametric::sketch::RequestError::InvalidRelation(
                    parametric::sketch::BuildError::InvalidParameter
                    | parametric::sketch::BuildError::InvalidQuantization,
                ),
            ) => ConstraintRefusal::Impossible,
            constraint::TrialMapError::Request(
                parametric::sketch::RequestError::InvalidRelation(
                    parametric::sketch::BuildError::InvalidTangent,
                ),
            ) => ConstraintRefusal::InvalidTangent {
                constraint: None,
                error: parametric::sketch::TangentContactError::InvalidBranch,
            },
            constraint::TrialMapError::Request(
                parametric::sketch::RequestError::InvalidRelation(
                    parametric::sketch::BuildError::InvalidConcentric,
                ),
            ) => ConstraintRefusal::InvalidConcentric,
            constraint::TrialMapError::Request(
                parametric::sketch::RequestError::InvalidRelation(
                    parametric::sketch::BuildError::InvalidSymmetry,
                ),
            ) => ConstraintRefusal::InvalidSymmetry,
        })?;
        let (settled, redundant) = match trial {
            parametric::sketch::TrialAdd::Accepted { settled, redundant } => (settled, redundant),
            parametric::sketch::TrialAdd::Rejected(
                parametric::sketch::TrialRejection::Unsatisfied { conflicts },
            ) => {
                return Err(ConstraintRefusal::Unsatisfiable {
                    fights: conflicts
                        .into_iter()
                        .filter_map(|id| prepared.constraint(id))
                        .collect(),
                });
            }
            parametric::sketch::TrialAdd::Rejected(
                parametric::sketch::TrialRejection::Collapsed { curve, implicated },
            ) => {
                let Some(entity) = prepared.curve(curve) else {
                    return Err(ConstraintRefusal::UnknownEntity);
                };
                return Err(ConstraintRefusal::WouldCollapse {
                    entity,
                    implicated: implicated
                        .into_iter()
                        .filter_map(|id| prepared.constraint(id))
                        .collect(),
                });
            }
            parametric::sketch::TrialAdd::Rejected(
                parametric::sketch::TrialRejection::InvalidTangent { constraint, error },
            ) => {
                return Err(ConstraintRefusal::InvalidTangent {
                    constraint: prepared.constraint(constraint),
                    error,
                });
            }
        };
        if let Some(failure) = prepared
            .first_tangent_contact_failure(&settled.solution)
            .map_err(|_| ConstraintRefusal::Impossible)?
        {
            return Err(ConstraintRefusal::InvalidTangent {
                constraint: Some(failure.constraint),
                error: failure.error,
            });
        }
        let Ok(plan) =
            prepared.plan_apply(&self.points, &self.arcs, &self.circles, &settled.solution)
        else {
            return Err(ConstraintRefusal::Impossible);
        };
        let id = self.alloc_id();
        self.constraints.push(Constraint {
            id,
            kind,
            redundant,
        });
        plan.apply(self);
        self.sync_arc_centers();
        Ok(id)
    }

    /// One persisted assertion of a kind may name one entity set. This is identity policy, not a
    /// numerical redundancy rule: the stored values deliberately do not participate, so replacing
    /// a `Fix` is delete-then-add rather than two claims that fight about one point.
    fn check_is_not_already_asserted(&self, kind: ConstraintKind) -> Result<(), ConstraintRefusal> {
        match self
            .constraints
            .iter()
            .find(|held| held.kind.is_about_the_same_as(kind))
        {
            Some(held) => Err(ConstraintRefusal::AlreadyAsserted { existing: held.id }),
            None => Ok(()),
        }
    }

    /// Whether every entity `kind` names is live in the store, and its own terms are meetable.
    /// This preflight belongs to the document because the local solver sees only validated handles;
    /// it gives missing geometry and self-contradictory requests distinct author-facing refusals.
    #[allow(clippy::too_many_lines)]
    fn check_names_live_geometry(
        &self,
        kind: ConstraintKind,
        context: parametric::EvaluationContext,
    ) -> Result<(), ConstraintRefusal> {
        let known_point = |id: EntityId| self.points.iter().any(|point| point.id == id);
        let live_segment = |id: EntityId| self.segments.iter().find(|seg| seg.id == id);
        match kind {
            ConstraintKind::Fix { point, .. } => {
                if !known_point(point) {
                    return Err(ConstraintRefusal::UnknownEntity);
                }
            }
            ConstraintKind::Quantize {
                point,
                pitch,
                phase,
            } => {
                if !known_point(point) {
                    return Err(ConstraintRefusal::UnknownEntity);
                }
                if !pitch.value().is_finite() || pitch.value() <= 0.0 || !phase.value().is_finite()
                {
                    return Err(ConstraintRefusal::Impossible);
                }
            }
            ConstraintKind::Horizontal { segment } | ConstraintKind::Vertical { segment } => {
                let Some(seg) = self.segments.iter().find(|seg| seg.id == segment) else {
                    return Err(ConstraintRefusal::UnknownEntity);
                };
                if seg.from == seg.to {
                    return Err(ConstraintRefusal::Impossible);
                }
            }
            ConstraintKind::Distance { from, to, length } => {
                if !known_point(from) || !known_point(to) {
                    return Err(ConstraintRefusal::UnknownEntity);
                }
                // A negative distance is no drawing's distance, and a zero one between two
                // distinct points is Coincident, which asserts one place rather than a span.
                if !length.value().is_finite() || length.value() <= 0.0 || from == to {
                    return Err(ConstraintRefusal::Impossible);
                }
            }
            ConstraintKind::Coincident { first, second } => {
                if !known_point(first) || !known_point(second) {
                    return Err(ConstraintRefusal::UnknownEntity);
                }
                // A point already occupies its own place, so asserting it is a claim with no
                // content rather than a claim that happens to hold.
                if first == second {
                    return Err(ConstraintRefusal::Impossible);
                }
            }
            ConstraintKind::Parallel { first, second }
            | ConstraintKind::Perpendicular { first, second }
            | ConstraintKind::Equal { first, second }
            | ConstraintKind::Collinear { first, second } => {
                let (Some(one), Some(other)) = (live_segment(first), live_segment(second)) else {
                    return Err(ConstraintRefusal::UnknownEntity);
                };
                if one.from == one.to || other.from == other.to {
                    return Err(ConstraintRefusal::Impossible);
                }
                // A segment is trivially parallel to itself and cannot be perpendicular to
                // itself, and neither statement is about the drawing.
                if first == second {
                    return Err(ConstraintRefusal::Impossible);
                }
            }
            ConstraintKind::Midpoint { point, segment } => {
                if !known_point(point) {
                    return Err(ConstraintRefusal::UnknownEntity);
                }
                let Some(span) = live_segment(segment) else {
                    return Err(ConstraintRefusal::UnknownEntity);
                };
                if span.from == span.to {
                    return Err(ConstraintRefusal::Impossible);
                }
                // An endpoint cannot be its own segment's midpoint without collapsing it, and
                // saying so here is a better answer than a solve that squeezes the line to
                // nothing and reports a collapse.
                if point == span.from || point == span.to {
                    return Err(ConstraintRefusal::Impossible);
                }
            }
            ConstraintKind::Tangent { first, second, .. } => {
                if !kind.tangent_is_structurally_valid() {
                    return Err(ConstraintRefusal::InvalidTangent {
                        constraint: None,
                        error: parametric::sketch::TangentContactError::InvalidBranch,
                    });
                }
                let live = |curve: SketchCurve| match curve {
                    SketchCurve::Segment(id) => self
                        .segments
                        .iter()
                        .find(|segment| segment.id == id)
                        .map(|segment| segment.from != segment.to)
                        .unwrap_or(false),
                    SketchCurve::Arc(id) => self.arcs.iter().any(|arc| arc.id == id),
                    SketchCurve::Circle(id) => self.circles.iter().any(|circle| circle.id == id),
                    SketchCurve::Bezier(_)
                    | SketchCurve::Ellipse(_)
                    | SketchCurve::Conic(_)
                    | SketchCurve::Spline(_) => false,
                };
                if !live(first) || !live(second) {
                    return Err(ConstraintRefusal::UnknownEntity);
                }
            }
            ConstraintKind::Concentric { first, second } => {
                if !kind.concentric_is_structurally_valid() {
                    return Err(ConstraintRefusal::InvalidConcentric);
                }
                let live = |curve: SketchCurve| match curve {
                    SketchCurve::Arc(id) => self.arcs.iter().any(|arc| arc.id == id),
                    SketchCurve::Circle(id) => self.circles.iter().any(|circle| circle.id == id),
                    SketchCurve::Segment(_)
                    | SketchCurve::Bezier(_)
                    | SketchCurve::Ellipse(_)
                    | SketchCurve::Conic(_)
                    | SketchCurve::Spline(_) => false,
                };
                if !live(first) || !live(second) {
                    return Err(ConstraintRefusal::UnknownEntity);
                }
            }
            ConstraintKind::Symmetry {
                first,
                second,
                axis,
                ..
            } => {
                if !kind.symmetry_is_structurally_valid() {
                    return Err(ConstraintRefusal::InvalidSymmetry);
                }
                let live = |curve: SketchCurve| match curve {
                    SketchCurve::Segment(id) => self.segments.iter().any(|held| held.id == id),
                    SketchCurve::Arc(id) => self.arcs.iter().any(|held| held.id == id),
                    SketchCurve::Circle(id) => self.circles.iter().any(|held| held.id == id),
                    SketchCurve::Bezier(_)
                    | SketchCurve::Ellipse(_)
                    | SketchCurve::Conic(_)
                    | SketchCurve::Spline(_) => false,
                };
                if !live(first) || !live(second) || live_segment(axis).is_none() {
                    return Err(ConstraintRefusal::UnknownEntity);
                }
                let axis = self
                    .curve_geometry(SketchCurve::Segment(axis), context)
                    .ok_or(ConstraintRefusal::InvalidSymmetry)?;
                if !parametric::sketch::symmetry_axis_is_valid(axis) {
                    return Err(ConstraintRefusal::InvalidSymmetry);
                }
            }
        }
        Ok(())
    }

    /// Delete one constraint by id. The geometry it held stays where the last solve put it —
    /// releasing an assertion does not undo its effect, it only stops re-asserting it.
    pub fn delete_constraint(&mut self, id: EntityId) {
        self.constraints.retain(|constraint| constraint.id != id);
    }

    /// Solve the sketch against its constraints with the document evaluation context that owns
    /// every fixed measurement source.
    ///
    /// `None` when there is nothing to solve. Solved positions are **authored** state, not
    /// `Derived`: they are the solver's input as well as its output, and an under-constrained
    /// sketch has freedoms only the stored position remembers.
    ///
    /// Free curve values are written back atomically with solved points; fixed sources remain
    /// untouched and are re-resolved only for this solve.
    pub fn solve(
        &mut self,
        context: parametric::EvaluationContext,
    ) -> Result<Option<SolveReport>, SketchEvaluationError> {
        let prepared = constraint::prepare(self, &self.constraints, Some(context))
            .map_err(map_prepare_evaluation_error)?;
        let settled = prepared.settle();
        validate_prepared_satisfaction(&prepared, &settled.diagnostics)?;
        validate_prepared_tangent_contacts(&prepared, &settled.solution)?;
        let plan = prepared
            .plan_apply(&self.points, &self.arcs, &self.circles, &settled.solution)
            .map_err(|_| SketchEvaluationError::ScalarWritebackFailed)?;
        plan.apply(self);
        self.sync_arc_centers();
        Ok(settled.diagnostics.report)
    }

    /// What a solve WOULD report, without moving anything.
    ///
    /// Read with the constraints alone — rigidity is a preference and has no place in a rank or a
    /// residual the caller is about to judge the drawing by.
    pub fn solve_report(
        &self,
        context: parametric::EvaluationContext,
    ) -> Result<Option<SolveReport>, SketchEvaluationError> {
        Ok(constraint::prepare(self, &self.constraints, Some(context))
            .map_err(map_prepare_evaluation_error)?
            .analyze()
            .diagnostics
            .report)
    }

    /// How many ways the drawing can still move: `2 × authored points − rank(J)`.
    ///
    /// Zero is a fully-constrained sketch. With no constraints every authored coordinate is free,
    /// which is two per point — the count is read off the store rather than from a solve that has
    /// no residuals to take a rank of.
    ///
    /// **Derived points are not freedoms.** An arc's center cannot be moved except by moving the
    /// arc, so counting its two coordinates would say a sketch is under-constrained in ways
    /// nothing can take up. They occupy parameter slots (which keeps write-back simple) but no
    /// residual reads them, so they contribute zero Jacobian columns and are subtracted here.
    pub fn degrees_of_freedom(
        &self,
        context: parametric::EvaluationContext,
    ) -> Result<usize, SketchEvaluationError> {
        Ok(constraint::prepare(self, &self.constraints, Some(context))
            .map_err(map_prepare_evaluation_error)?
            .analyze()
            .degrees_of_freedom)
    }

    /// Drop constraints naming geometry the store no longer holds. Called by every geometry delete
    /// and by repair, so a constraint never outlives what it constrains or leaves an orphan residual
    /// row in the prepared local problem. A relation is an assertion about a drawing, not ownership
    /// of the drawing it names.
    fn drop_dangling_constraints(&mut self) {
        // Load/programmatic malformed order reaches this same repair door. Normalize before
        // duplicate identity so a reversed internal relation cannot survive as a second claim.
        for constraint in &mut self.constraints {
            constraint.kind = constraint.kind.normalized();
        }
        let point_ids: Vec<EntityId> = self.points.iter().map(|point| point.id).collect();
        let segment_ids: Vec<EntityId> = self.segments.iter().map(|seg| seg.id).collect();
        let arc_ids: Vec<EntityId> = self.arcs.iter().map(|arc| arc.id).collect();
        let circle_ids: Vec<EntityId> = self.circles.iter().map(|circle| circle.id).collect();
        let bezier_ids: Vec<EntityId> = self.beziers.iter().map(|bezier| bezier.id).collect();
        let ellipse_ids: Vec<EntityId> = self.ellipses.iter().map(|ellipse| ellipse.id).collect();
        let conic_ids: Vec<EntityId> = self.conics.iter().map(|conic| conic.id).collect();
        let spline_ids: Vec<EntityId> = self.splines.iter().map(|spline| spline.id).collect();
        let valid_symmetry_axes: Vec<EntityId> = self
            .segments
            .iter()
            .filter_map(|segment| {
                let point = |id| {
                    self.points
                        .iter()
                        .find(|point| point.id == id)
                        .map(|point| point.at.in_plane())
                };
                let (Some(from), Some(to)) = (point(segment.from), point(segment.to)) else {
                    return None;
                };
                parametric::sketch::symmetry_axis_is_valid(
                    parametric::sketch::CurveGeometry::Segment { from, to },
                )
                .then_some(segment.id)
            })
            .collect();
        self.constraints.retain(|constraint| {
            constraint
                .kind
                .points()
                .iter()
                .all(|id| point_ids.contains(id))
                && constraint
                    .kind
                    .segments()
                    .iter()
                    .all(|id| segment_ids.contains(id))
                && constraint.kind.curves().iter().all(|curve| match curve {
                    SketchCurve::Segment(id) => segment_ids.contains(id),
                    SketchCurve::Arc(id) => arc_ids.contains(id),
                    SketchCurve::Circle(id) => circle_ids.contains(id),
                    SketchCurve::Bezier(id) => bezier_ids.contains(id),
                    SketchCurve::Ellipse(id) => ellipse_ids.contains(id),
                    SketchCurve::Conic(id) => conic_ids.contains(id),
                    SketchCurve::Spline(id) => spline_ids.contains(id),
                })
                && constraint.kind.tangent_is_structurally_valid()
                && constraint.kind.concentric_is_structurally_valid()
                && constraint.kind.symmetry_is_structurally_valid()
                && match constraint.kind {
                    ConstraintKind::Symmetry { axis, .. } => valid_symmetry_axes.contains(&axis),
                    _ => true,
                }
        });
        let surviving = self.constraints.clone();
        self.constraints.retain(|constraint| {
            // Stable authored order decides the survivor, matching every other duplicate policy.
            let duplicate = surviving
                .iter()
                .take_while(|held| held.id != constraint.id)
                .any(|held| held.kind.is_about_the_same_as(constraint.kind));
            !duplicate
        });
    }

    /// Split the segment with id `seg_id` by inserting a new point `at` on it. The first
    /// half keeps the segment's id; the new second half inherits its
    /// `origin`, so a bounding face's origin-set is unchanged. No-op if `seg_id` is unknown.
    pub fn split_segment(&mut self, seg_id: EntityId, at: SketchPoint) {
        let Some(index) = self.segments.iter().position(|seg| seg.id == seg_id) else {
            return;
        };
        let new_point = self.add_point(at);
        let origin = self.segments[index].origin;
        let old_to = self.segments[index].to;
        self.segments[index].to = new_point;
        let id = self.alloc_id();
        self.segments.push(Segment {
            id,
            from: new_point,
            to: old_to,
            origin,
            role: EntityRole::Real,
        });
    }

    /// Add a FREE point entity at `at` — no incident segment — returning its fresh id. A free
    /// point is legal geometry; the Line tool places one per click and then connects them.
    /// The public door to [`add_point`](Self::add_point).
    pub fn add_free_point(&mut self, at: SketchPoint) -> EntityId {
        self.add_point(at)
    }

    /// Connect two existing points with a fresh segment, returning its id. Coincidence is
    /// shared point identity, so drawing to an existing point means naming its id here, never
    /// minting a coordinate twin. `None` — and no mutation — for a self-loop, an unknown
    /// endpoint, or a pair a SEGMENT already joins: a straight edge between two points is
    /// unique geometry, so a second one is a duplicate.
    ///
    /// A pair an ARC joins is fine, and is the D-shape (a chord closing a curve): the face
    /// derivation traces that two-edge cycle like any other.
    pub fn connect(&mut self, from: EntityId, to: EntityId) -> Option<EntityId> {
        if from == to
            || self.point_index(from).is_none()
            || self.point_index(to).is_none()
            || self.segment_joins(from, to)
        {
            return None;
        }
        Some(self.add_segment(from, to))
    }

    /// Connect two existing points with a fresh arc of the given signed included angle,
    /// returning its id. `None` — and no mutation — for a self-loop, an unknown
    /// endpoint, a degenerate bulge (zero or a full turn or more), or an arc that would
    /// trace a curve the store already holds.
    ///
    /// A pair already joined by a segment, or by an arc bulging differently, is legal: a
    /// chord plus its arc is a D, and two arcs over one pair are a lens. Both are ordinary
    /// bounded faces to the derivation.
    pub fn connect_arc(
        &mut self,
        from: EntityId,
        to: EntityId,
        bulge: AngleMeasurement,
    ) -> Option<EntityId> {
        let sweep = bulge.to_degrees_f64();
        if from == to
            || self.point_index(from).is_none()
            || self.point_index(to).is_none()
            || !arc_sweep_is_valid(sweep)
            || self.arc_traces(from, to, sweep)
        {
            return None;
        }
        let id = self.alloc_id();
        self.arcs.push(Arc {
            id,
            from,
            to,
            bulge: ArcSweep::free(bulge),
            center: ABSENT_CENTER,
            origin: id,
            role: EntityRole::Real,
        });
        self.sync_arc_centers();
        Some(id)
    }

    /// Draw a circle of `radius` about a FRESH construction center at `at`, returning the
    /// circle's id. `None` — and no mutation — for a non-positive or non-finite radius, which
    /// is not a curve.
    ///
    /// The center is minted here rather than taken as an id because that is what the center-radius
    /// tool does: one click plants the center, the drag sets the radius. Drawing about a point that
    /// already exists is [`circle_about`](Self::circle_about).
    pub fn add_circle(&mut self, at: SketchPoint, radius: SketchLength) -> Option<EntityId> {
        if !circle_radius_is_valid(radius.value()) {
            return None;
        }
        let center = self.add_construction_point(at);
        self.push_circle(center, radius)
    }

    /// Draw a circle of `radius` about the EXISTING point `center`, returning its id. `None` for an
    /// unknown point, an invalid radius, or a circle the store already holds about that center at
    /// that radius — the same curve twice is not two curves.
    ///
    /// Concentric circles of different radii are fine, and are the ring: two faces, the inner one
    /// unpicked.
    pub fn circle_about(&mut self, center: EntityId, radius: SketchLength) -> Option<EntityId> {
        if self.point_index(center).is_none()
            || !circle_radius_is_valid(radius.value())
            || self.circle_traces(center, radius.value())
        {
            return None;
        }
        self.push_circle(center, radius)
    }

    /// Allocate the circle entity itself, its `origin` a root of its own lineage.
    fn push_circle(&mut self, center: EntityId, radius: SketchLength) -> Option<EntityId> {
        let id = self.alloc_id();
        let radius = circle_radius_from_sketch_length(radius)?;
        self.circles.push(Circle {
            id,
            center,
            radius,
            origin: id,
            role: EntityRole::Real,
        });
        Some(id)
    }

    /// Whether a circle of this radius about this center is already stored.
    pub fn circle_traces(&self, center: EntityId, radius_voxels: f64) -> bool {
        self.circles.iter().any(|circle| {
            circle.center == center && circle.free_radius_value() == Some(radius_voxels)
        })
    }

    /// Resize the circle `id` — the radius-drag write path. Reports whether it took: an unknown id
    /// or an invalid radius leaves the store untouched rather than erasing the curve.
    pub fn set_circle_radius(&mut self, id: EntityId, radius: SketchLength) -> bool {
        if !circle_radius_is_valid(radius.value()) {
            return false;
        }
        let Some(index) = self.circles.iter().position(|circle| circle.id == id) else {
            return false;
        };
        let Some(radius) = circle_radius_from_sketch_length(radius) else {
            return false;
        };
        self.circles[index].radius = radius;
        true
    }

    /// Delete just the circle with id `circle_id`. Its center goes with it when nothing else names
    /// it — the center is the circle's own anchor, so there is no circle left for it to center.
    pub fn delete_circle(&mut self, circle_id: EntityId) {
        self.circles.retain(|circle| circle.id != circle_id);
        self.prune_orphan_centers();
        self.drop_dangling_patterns();
        self.drop_dangling_constraints();
    }

    /// Connect four existing control points as one rational cubic curve piece.
    ///
    /// The first and last controls are its topological endpoints. A closed rational curve needs
    /// multiple pieces (as an ellipse does), so equal endpoints are refused here. All weights must
    /// be finite and strictly positive, keeping the projective denominator non-zero throughout.
    pub fn connect_rational_bezier(
        &mut self,
        controls: [EntityId; 4],
        weights: [f64; 4],
    ) -> Option<EntityId> {
        let curve = self.rational_bezier_from(controls, weights)?;
        let reversed_controls = [controls[3], controls[2], controls[1], controls[0]];
        let reversed_weights = [weights[3], weights[2], weights[1], weights[0]];
        if controls[0] == controls[3]
            || curve.control[0] == curve.control[3]
            || self.beziers.iter().any(|held| {
                (held.controls == controls && held.weights == weights)
                    || (held.controls == reversed_controls && held.weights == reversed_weights)
            })
        {
            return None;
        }
        let id = self.alloc_id();
        boxed_push(
            &mut self.beziers,
            Bezier {
                id,
                controls,
                weights,
                origin: id,
                role: EntityRole::Real,
            },
        );
        Some(id)
    }

    /// Draw an ordinary cubic Bézier from four positions, minting visible endpoints and
    /// construction-role tangent controls atomically.
    pub fn add_cubic_bezier(&mut self, control: [SketchPoint; 4]) -> Option<EntityId> {
        let continuous = control.map(|point| point.in_plane());
        let curve = substrate::rational_bezier::RationalBezier::cubic(continuous);
        if !curve.is_valid() || continuous[0] == continuous[3] {
            return None;
        }
        let controls = [
            self.add_point(control[0]),
            self.add_construction_point(control[1]),
            self.add_construction_point(control[2]),
            self.add_point(control[3]),
        ];
        self.connect_rational_bezier(controls, [1.0; 4])
    }

    /// Delete one rational curve piece. Control points shared by another entity survive; private
    /// tangent handles are pruned by the same orphan policy used for circular centers.
    pub fn delete_bezier(&mut self, bezier_id: EntityId) {
        let controls = self
            .beziers
            .iter()
            .find(|bezier| bezier.id == bezier_id)
            .map(|bezier| bezier.controls);
        boxed_retain(&mut self.beziers, |bezier| bezier.id != bezier_id);
        if let Some(controls) = controls {
            self.drop_undrawn_points(controls);
        }
        self.prune_orphan_centers();
        self.drop_dangling_patterns();
        self.drop_dangling_constraints();
    }

    /// Resolve stable control ids into the foundational rational-curve value.
    fn rational_bezier_from(
        &self,
        controls: [EntityId; 4],
        weights: [f64; 4],
    ) -> Option<substrate::rational_bezier::RationalBezier> {
        let position = |id| {
            self.points
                .iter()
                .find(|point| point.id == id)
                .map(|point| point.at.in_plane())
        };
        let curve = substrate::rational_bezier::RationalBezier {
            control: [
                position(controls[0])?,
                position(controls[1])?,
                position(controls[2])?,
                position(controls[3])?,
            ],
            weights,
        };
        curve.is_valid().then_some(curve)
    }

    /// Draw one closed ellipse from center, major-axis endpoint, and width pick.
    pub fn add_ellipse(
        &mut self,
        center: SketchPoint,
        major_endpoint: SketchPoint,
        width_point: SketchPoint,
    ) -> Result<EntityId, parametric::sketch::EllipseCandidateError> {
        parametric::sketch::ellipse_candidate(
            center.in_plane(),
            major_endpoint.in_plane(),
            width_point.in_plane(),
        )?;
        let center = self.add_construction_point(center);
        let major_endpoint = self.add_construction_point(major_endpoint);
        let width_point = self.add_construction_point(width_point);
        let id = self.alloc_id();
        boxed_push(
            &mut self.ellipses,
            Ellipse {
                id,
                center,
                major_endpoint,
                width_point,
                origin: id,
                role: EntityRole::Real,
            },
        );
        Ok(id)
    }

    /// Draw one endpoint/vertex/rho conic with exact dimensionless rho storage.
    pub fn add_conic(
        &mut self,
        from: SketchPoint,
        to: SketchPoint,
        vertex: SketchPoint,
        rho: f64,
    ) -> Result<EntityId, parametric::sketch::ConicCandidateError> {
        parametric::sketch::conic_candidate(
            from.in_plane(),
            to.in_plane(),
            vertex.in_plane(),
            rho,
        )?;
        let rho = parametric::ResolvedScalar::try_from_f64(rho)
            .map_err(|_| parametric::sketch::ConicCandidateError::InvalidRho)?;
        let from = self.add_point(from);
        let to = self.add_point(to);
        let vertex = self.add_point(vertex);
        let id = self.alloc_id();
        boxed_push(
            &mut self.conics,
            Conic {
                id,
                from,
                to,
                vertex,
                rho,
                origin: id,
                role: EntityRole::Real,
            },
        );
        Ok(id)
    }

    fn ellipse_candidate(&self, ellipse: Ellipse) -> Option<parametric::sketch::EllipseCandidate> {
        let position = |id| {
            self.points
                .iter()
                .find(|point| point.id == id)
                .map(|point| point.at.in_plane())
        };
        parametric::sketch::ellipse_candidate(
            position(ellipse.center)?,
            position(ellipse.major_endpoint)?,
            position(ellipse.width_point)?,
        )
        .ok()
    }

    fn conic_candidate(&self, conic: Conic) -> Option<parametric::sketch::ConicCandidate> {
        let position = |id| {
            self.points
                .iter()
                .find(|point| point.id == id)
                .map(|point| point.at.in_plane())
        };
        parametric::sketch::conic_candidate(
            position(conic.from)?,
            position(conic.to)?,
            position(conic.vertex)?,
            conic.rho.value(),
        )
        .ok()
    }

    pub fn delete_ellipse(&mut self, id: EntityId) {
        boxed_retain(&mut self.ellipses, |ellipse| ellipse.id != id);
        self.prune_orphan_centers();
    }

    pub fn delete_conic(&mut self, id: EntityId) {
        let points = self
            .conics
            .iter()
            .find(|conic| conic.id == id)
            .map(|conic| [conic.from, conic.to, conic.vertex]);
        boxed_retain(&mut self.conics, |conic| conic.id != id);
        if let Some(points) = points {
            self.drop_undrawn_points(points);
        }
    }

    pub fn add_fit_point_spline(
        &mut self,
        points: &[SketchPoint],
        closed: bool,
    ) -> Result<EntityId, parametric::sketch::SplineCandidateError> {
        let continuous: Vec<_> = points.iter().map(SketchPoint::in_plane).collect();
        parametric::sketch::fit_point_spline(&continuous, closed)?;
        let points: Vec<_> = points.iter().map(|point| self.add_point(*point)).collect();
        let id = self.alloc_id();
        boxed_push(
            &mut self.splines,
            Spline {
                id,
                points,
                kind: SplineKind::FitPoint,
                closed,
                origin: id,
                role: EntityRole::Real,
            },
        );
        Ok(id)
    }

    pub fn add_control_point_spline(
        &mut self,
        controls: &[SketchPoint],
    ) -> Result<EntityId, parametric::sketch::SplineCandidateError> {
        let continuous: Vec<_> = controls.iter().map(SketchPoint::in_plane).collect();
        parametric::sketch::control_point_spline(&continuous)?;
        let last = controls.len().saturating_sub(1);
        let points: Vec<_> = controls
            .iter()
            .enumerate()
            .map(|(index, point)| {
                if index == 0 || index == last {
                    self.add_point(*point)
                } else {
                    self.add_construction_point(*point)
                }
            })
            .collect();
        let id = self.alloc_id();
        boxed_push(
            &mut self.splines,
            Spline {
                id,
                points,
                kind: SplineKind::ControlPoint,
                closed: false,
                origin: id,
                role: EntityRole::Real,
            },
        );
        Ok(id)
    }

    fn spline_candidate(&self, spline: &Spline) -> Option<parametric::sketch::SplineCandidate> {
        let points: Option<Vec<_>> = spline
            .points
            .iter()
            .map(|id| {
                self.points
                    .iter()
                    .find(|point| point.id == *id)
                    .map(|point| point.at.in_plane())
            })
            .collect();
        let points = points?;
        match spline.kind {
            SplineKind::FitPoint => {
                parametric::sketch::fit_point_spline(&points, spline.closed).ok()
            }
            SplineKind::ControlPoint => parametric::sketch::control_point_spline(&points).ok(),
        }
    }

    pub fn delete_spline(&mut self, id: EntityId) {
        let points = self
            .splines
            .iter()
            .find(|spline| spline.id == id)
            .map(|spline| spline.points.clone());
        boxed_retain(&mut self.splines, |spline| spline.id != id);
        if let Some(points) = points {
            self.drop_undrawn_points(points);
        }
        self.prune_orphan_centers();
    }

    /// Whether the drawing OWNS this point's coordinates — whether it is an arc's center, which
    /// [`sync_arc_centers`](Self::sync_arc_centers) re-derives from the arc's ends and its
    /// sweep. A derived point is selectable, draggable, snappable and **constrainable** like any
    /// other; what it is not is a freedom, which is why
    /// [`degrees_of_freedom`](Self::degrees_of_freedom) does not count it.
    ///
    /// A constraint naming one is met by moving the ARC — see `constraint::position_of`, where the
    /// residual system reads it as the function it is.
    pub fn is_derived_point(&self, id: EntityId) -> bool {
        self.arcs.iter().any(|arc| arc.center == id)
    }

    /// Re-derive every arc's center point from its endpoints and bulge, minting one for any arc
    /// that has none yet. The center is a real [`Point`] so it can be selected, snapped to and
    /// dragged like any other, but its coordinates are OWNED here — every edit that can move an arc
    /// ends by calling this, so a center can never drift out of agreement with the curve it belongs
    /// to. Solver write-back follows the same function rather than trusting the stored center slot.
    /// An arc whose endpoints are missing or coincident is left alone; [`repair`](Self::repair)
    /// erases it.
    pub fn sync_arc_centers(&mut self) {
        for index in 0..self.arcs.len() {
            let arc = self.arcs[index];
            let (Some(tail), Some(head)) = (self.point_index(arc.from), self.point_index(arc.to))
            else {
                continue;
            };
            let Some((center, _radius)) = arc_center_radius(
                self.points[tail].at.in_plane(),
                self.points[head].at.in_plane(),
                arc.sweep_degrees(),
            ) else {
                continue;
            };
            let at = SketchPoint::from_continuous(center[0], center[1]);
            match self.point_index(arc.center) {
                Some(existing) => self.points[existing].at = at,
                None => self.arcs[index].center = self.add_construction_point(at),
            }
        }
    }

    /// Drop every construction point nothing references any more — the center of an arc that
    /// has just been deleted. A center the author has since drawn to (an edge names it) is
    /// referenced, so it survives as ordinary geometry.
    fn prune_orphan_centers(&mut self) {
        let mut referenced = std::collections::BTreeSet::new();
        for arc in &self.arcs {
            referenced.extend([arc.center, arc.from, arc.to]);
        }
        for segment in &self.segments {
            referenced.extend([segment.from, segment.to]);
        }
        for circle in &self.circles {
            referenced.insert(circle.center);
        }
        for bezier in &*self.beziers {
            referenced.extend(bezier.controls);
        }
        for ellipse in &*self.ellipses {
            referenced.extend([ellipse.center, ellipse.major_endpoint, ellipse.width_point]);
        }
        for conic in &*self.conics {
            referenced.extend([conic.from, conic.to, conic.vertex]);
        }
        for spline in &*self.splines {
            referenced.extend(spline.points.iter().copied());
        }
        self.points.retain(|point| {
            point.role != EntityRole::Construction || referenced.contains(&point.id)
        });
    }

    /// Whether a straight segment already joins `a` and `b` in either direction.
    pub fn segment_joins(&self, a: EntityId, b: EntityId) -> bool {
        self.segments
            .iter()
            .any(|seg| (seg.from == a && seg.to == b) || (seg.from == b && seg.to == a))
    }

    /// Whether some stored arc already traces the CURVE `from → to` sweeping `sweep_degrees`.
    /// Reversing an arc's direction mirrors it about the chord unless the sweep's sign flips
    /// too, so the reversed match is against the negated sweep — an arc bulging the other way
    /// over the same pair is a different curve, and legal.
    pub fn arc_traces(&self, from: EntityId, to: EntityId, sweep_degrees: f64) -> bool {
        self.arcs.iter().any(|arc| {
            let stored = arc.sweep_degrees();
            (arc.from == from && arc.to == to && stored == sweep_degrees)
                || (arc.from == to && arc.to == from && stored == -sweep_degrees)
        })
    }

    /// Delete the arc with id `arc_id`, **and each of its ends that nothing else draws** — the
    /// same rule [`delete_segment`](Self::delete_segment) follows, because deleting a curve and
    /// deleting a line are one gesture as far as the author is concerned. Its center goes with it
    /// through [`prune_orphan_centers`](Self::prune_orphan_centers). No-op if `arc_id` is unknown.
    pub fn delete_arc(&mut self, arc_id: EntityId) {
        let Some(curve) = self.arcs.iter().find(|arc| arc.id == arc_id).copied() else {
            return;
        };
        self.arcs.retain(|arc| arc.id != arc_id);
        self.drop_undrawn_points([curve.from, curve.to]);
        self.prune_orphan_centers();
        self.drop_dangling_patterns();
        self.drop_dangling_constraints();
    }

    /// The lowest-id point entity sitting EXACTLY at `at`'s position, if any. The drawing
    /// tools check this after snapping a click, so a click that lands on an existing
    /// point's coordinates reuses its id (coincidence = shared identity) instead of minting
    /// a twin point the region graph would read as a distinct vertex. Position-only
    /// ([`SketchPoint::coincides`]) — a retained measurement never splits coincidence.
    pub fn point_at(&self, at: SketchPoint) -> Option<EntityId> {
        self.points
            .iter()
            .filter(|point| point.at.coincides(&at))
            .map(|point| point.id)
            .min()
    }

    /// The in-plane bbox-minimum over ALL point entities (per coordinate), `[0, 0]` when the
    /// sketch is empty. Unlike [`profile_bbox_min`](SketchSolid::profile_bbox_min) — the loop's
    /// bbox, which the resolve anchors — this covers every point (including free points and the
    /// vertices of an open graph), so the interactive overlay can place a handle on each.
    pub fn points_bbox_min(&self) -> [i64; 2] {
        let mut min = self
            .points
            .first()
            .map(|point| point.at.offset_voxels)
            .unwrap_or([0, 0]);
        for point in &self.points {
            min[0] = min[0].min(point.at.offset_voxels[0]);
            min[1] = min[1].min(point.at.offset_voxels[1]);
        }
        min
    }

    /// Re-target every position and length owned by the sketch from `old_density` to
    /// `new_density` — the `SetDensity` arm. Retained measurements re-evaluate losslessly;
    /// plain stored geometry rescales its continuous value. Constraint targets participate too:
    /// otherwise a later solve would pull a correctly rescaled point back to its old-density fix.
    pub fn retarget_density(&mut self, old_density: u32, new_density: u32) {
        for point in &mut self.points {
            point.at = point.at.retargeted(old_density, new_density);
        }
        // A radius is a length like any other: an authored `2 blocks` must stay two blocks.
        for circle in &mut self.circles {
            circle.rescale_free_radius(old_density, new_density);
        }
        self.retarget_patterns(old_density, new_density);
        for constraint in &mut self.constraints {
            match &mut constraint.kind {
                ConstraintKind::Fix { at, .. } => {
                    *at = at.retargeted(old_density, new_density);
                }
                ConstraintKind::Distance { length, .. } => {
                    *length = length.retargeted(old_density, new_density);
                }
                ConstraintKind::Quantize { pitch, phase, .. } => {
                    *pitch = pitch.retargeted(old_density, new_density);
                    *phase = phase.retargeted(old_density, new_density);
                }
                ConstraintKind::Horizontal { .. }
                | ConstraintKind::Vertical { .. }
                | ConstraintKind::Coincident { .. }
                | ConstraintKind::Parallel { .. }
                | ConstraintKind::Perpendicular { .. }
                | ConstraintKind::Equal { .. }
                | ConstraintKind::Midpoint { .. }
                | ConstraintKind::Collinear { .. }
                | ConstraintKind::Tangent { .. }
                | ConstraintKind::Concentric { .. }
                | ConstraintKind::Symmetry { .. } => {}
            }
        }
        self.sync_arc_centers();
    }

    /// Erase every structurally-invalid segment or arc — one that references a point id not in the
    /// store, a self-loop (`from == to`), or (arcs) a degenerate bulge — returning the number
    /// removed. The load policy is to erase invalid objects rather than fail the load. Points are
    /// never invalid; a point left with no incident edge is a legal free point. The resolve already
    /// tolerates a dangling reference, so this is cleanup + audit rather than a crash guard.
    ///
    /// Repair also cascades dead relations and then settles derived centers: a loaded document can
    /// name neither a usable curve nor its center, but after repair the surviving topology and all
    /// derived center points agree again.
    pub fn repair(&mut self, context: parametric::EvaluationContext) -> usize {
        let point_ids: Vec<EntityId> = self.points.iter().map(|point| point.id).collect();
        let before = self.segments.len()
            + self.arcs.len()
            + self.circles.len()
            + self.beziers.len()
            + self.ellipses.len()
            + self.conics.len()
            + self.splines.len();
        self.segments.retain(|seg| {
            seg.from != seg.to && point_ids.contains(&seg.from) && point_ids.contains(&seg.to)
        });
        // An arc is additionally invalid on a degenerate bulge — a zero sweep is a
        // segment pretending, a full turn or more has no single chord-anchored shape.
        self.arcs.retain(|arc| {
            arc.from != arc.to
                && point_ids.contains(&arc.from)
                && point_ids.contains(&arc.to)
                && arc_sweep_is_valid(arc.sweep_degrees())
        });
        // A circle is invalid on a missing center or a radius that is not a positive finite
        // length — either way there is no curve to draw.
        self.circles.retain(|circle| {
            point_ids.contains(&circle.center)
                && circle_radius_is_valid(circle.resolved_radius(context))
        });
        boxed_retain(&mut self.beziers, |bezier| {
            bezier.controls[0] != bezier.controls[3]
                && bezier.controls.iter().all(|id| point_ids.contains(id))
                && bezier
                    .weights
                    .iter()
                    .all(|weight| weight.is_finite() && *weight > 0.0)
        });
        boxed_retain(&mut self.ellipses, |ellipse| {
            [ellipse.center, ellipse.major_endpoint, ellipse.width_point]
                .iter()
                .all(|id| point_ids.contains(id))
        });
        boxed_retain(&mut self.conics, |conic| {
            [conic.from, conic.to, conic.vertex]
                .iter()
                .all(|id| point_ids.contains(id))
                && (0.0..1.0).contains(&conic.rho.value())
        });
        let valid_splines: Vec<_> = self
            .splines
            .iter()
            .filter(|spline| {
                spline.points.iter().all(|id| point_ids.contains(id))
                    && self.spline_candidate(spline).is_some()
            })
            .map(|spline| spline.id)
            .collect();
        boxed_retain(&mut self.splines, |spline| {
            valid_splines.contains(&spline.id)
        });
        // Geometry-dependent constraint repair must see derived arc centers at their authored
        // positions, not a stale serialized cache.
        self.sync_arc_centers();
        // A constraint naming geometry the store does not hold asserts nothing about anything,
        // and left in place it would keep a row in the residual system for a shape that is gone.
        let before = before + self.constraints.len() + self.patterns.len();
        self.drop_dangling_patterns();
        self.drop_dangling_constraints();
        let dropped = before
            - self.segments.len()
            - self.arcs.len()
            - self.circles.len()
            - self.beziers.len()
            - self.ellipses.len()
            - self.conics.len()
            - self.splines.len()
            - self.patterns.len()
            - self.constraints.len();
        // A document may name no center at all, and a just-erased arc leaves one behind;
        // both are settled here, so a loaded sketch always agrees with its arcs.
        self.prune_orphan_centers();
        self.sync_arc_centers();
        dropped
    }
}

/// Default arc flattening tolerance: the maximum sagitta (chord-to-arc deviation), in voxels,
/// of one chord.
///
/// This is NOT the resolved meaning of a curve — the region carries its arcs, and the field
/// measures them ([`ProfileEdge`]). It is the default a **terminal adapter** flattens at when it
/// has to produce something discrete and has nowhere to put a curve: a crease polyline, the
/// exact-`f64` cell classifier's polygon, a test's outline. Nothing downstream of one of those
/// inherits it, so it is a tuning knob rather than a document-format constant.
pub const ARC_SAGITTA_TOLERANCE_VOXELS: f64 = 1.0 / 16.0;

/// Hard cap on chords per arc, so a huge-radius near-collinear arc cannot degenerate
/// into an unbounded fan.
const ARC_MAX_CHORDS: u32 = 512;

/// Whether a signed sweep is a legal [`Arc`] bulge: finite, non-zero, strictly under a full
/// turn in magnitude.
///
/// The full turn stays excluded ON PURPOSE. A closed curve is a [`Circle`] — a center and a radius
/// — not an arc bulged all the way round: the endpoint-plus-bulge form
/// degenerates there, its chord shrinking to nothing and taking the circle it was supposed to
/// determine with it. Admitting a 360° bulge would put an unsolvable arc in the store to spare a
/// tool one branch.
fn arc_sweep_is_valid(sweep_degrees: f64) -> bool {
    sweep_degrees.is_finite() && sweep_degrees != 0.0 && sweep_degrees.abs() < 360.0
}

/// The index in `faces` of the innermost one containing `point`, or `None` when nothing does.
///
/// `faces` must be in nesting order (smallest area first), which is exactly what makes "innermost"
/// a matter of taking the first hit rather than a containment analysis: a face strictly inside
/// another has strictly less area.
fn innermost_face_at(faces: &[Face], point: [f32; 2]) -> Option<usize> {
    faces.iter().position(|face| face.contains(point))
}

/// Whether a radius is a legal [`Circle`]: finite and strictly positive. A zero radius is a point
/// and a negative one is nothing.
fn circle_radius_is_valid(radius_voxels: f64) -> bool {
    radius_voxels.is_finite() && radius_voxels > 0.0
}

/// The center and radius DERIVED from the canonical arc form: endpoints plus signed
/// sweep, positive sweeping counter-clockwise about the center. `None` for a
/// degenerate chord (coincident endpoints) or an invalid sweep.
pub fn arc_center_radius(
    from: [f64; 2],
    to: [f64; 2],
    sweep_degrees: f64,
) -> Option<([f64; 2], f64)> {
    if !arc_sweep_is_valid(sweep_degrees) {
        return None;
    }
    let chord = [to[0] - from[0], to[1] - from[1]];
    let chord_length = (chord[0] * chord[0] + chord[1] * chord[1]).sqrt();
    if chord_length <= f64::EPSILON {
        return None;
    }
    let half_sweep = sweep_degrees.to_radians() / 2.0;
    let radius = chord_length / (2.0 * half_sweep.sin().abs());
    // The center sits on the chord's perpendicular bisector at the signed apothem: the
    // signed tangent puts it left of `from → to` for a minor CCW sweep and flips it for
    // the major/CW cases, one formula covering all four quadrants (continuous through
    // the 180° apothem-zero).
    let mid = [(from[0] + to[0]) / 2.0, (from[1] + to[1]) / 2.0];
    let left = [-chord[1] / chord_length, chord[0] / chord_length];
    let apothem = (chord_length / 2.0) / half_sweep.tan();
    Some((
        [mid[0] + left[0] * apothem, mid[1] + left[1] * apothem],
        radius,
    ))
}

/// The arc's tessellated INTERIOR vertices from `from` to `to` (both endpoints
/// exclusive), as continuous sub-voxel points, each chord's sagitta within
/// [`ARC_SAGITTA_TOLERANCE_VOXELS`]. Empty when the arc is degenerate — the callers
/// then fall back to the straight chord.
pub fn arc_interior_points(from: [f64; 2], to: [f64; 2], sweep_degrees: f64) -> Vec<SketchPoint> {
    arc_interior_points_within(from, to, sweep_degrees, ARC_SAGITTA_TOLERANCE_VOXELS)
}

/// [`arc_interior_points`] at a caller-chosen sagitta tolerance.
///
/// The default is measured in voxels, so a chord count follows radius-in-voxels and not size on
/// screen: a 15-voxel arc earns nine chords whatever the zoom, which reads as a visible polygon.
/// A screen-space painter that knows what a voxel is currently worth in pixels asks for a
/// tolerance keeping the sagitta under a pixel instead. Neither is the curve's meaning — the
/// region carries its arcs and the field measures them. Every caller here is a
/// terminal adapter, so no tolerance chosen at one reaches anything downstream of it.
pub fn arc_interior_points_within(
    from: [f64; 2],
    to: [f64; 2],
    sweep_degrees: f64,
    sagitta_tolerance_voxels: f64,
) -> Vec<SketchPoint> {
    let Some((center, radius)) = arc_center_radius(from, to, sweep_degrees) else {
        return Vec::new();
    };
    arc_interior_on_circle(
        ProfileArc {
            center,
            radius,
            start_radians: (from[1] - center[1]).atan2(from[0] - center[0]),
            sweep_radians: sweep_degrees.to_radians(),
        },
        sagitta_tolerance_voxels,
    )
}

/// The interior points of an ALREADY-SOLVED arc — the circle walked directly, both endpoints
/// exclusive.
///
/// This is the form the closed case needs. Recovering a circle from endpoints plus a bulge is a
/// chord solve, and a whole turn has no chord; carrying the solved center and radius instead means
/// a circle tessellates by the same rule as every other arc rather than by a special case.
fn arc_interior_on_circle(arc: ProfileArc, sagitta_tolerance_voxels: f64) -> Vec<SketchPoint> {
    let chords = arc_chord_count(
        arc.radius,
        arc.sweep_radians.to_degrees(),
        sagitta_tolerance_voxels,
    );
    let step = arc.sweep_radians / chords as f64;
    (1..chords)
        .map(|chord_index| {
            let angle = arc.start_radians + step * chord_index as f64;
            SketchPoint::from_continuous(
                arc.center[0] + arc.radius * angle.cos(),
                arc.center[1] + arc.radius * angle.sin(),
            )
        })
        .collect()
}

/// How many chords keep each sagitta within tolerance, capped at [`ARC_MAX_CHORDS`].
fn arc_chord_count(radius: f64, sweep_degrees: f64, tolerance: f64) -> u32 {
    // A non-positive or non-finite tolerance would ask for infinite refinement; the chord cap
    // answers it instead of the arithmetic below producing a NaN step.
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return ARC_MAX_CHORDS;
    }
    if 2.0 * radius <= tolerance {
        return 1; // the whole arc deviates less than the tolerance from its chord
    }
    let max_step = 2.0 * (1.0 - tolerance / radius).acos();
    ((sweep_degrees.to_radians().abs() / max_step).ceil() as u32).clamp(1, ARC_MAX_CHORDS)
}

/// Solve the 3-POINT creation: the signed included angle of the arc from `from` to `to`
/// that passes through `through`. The through-point is consumed here — the canonical
/// stored form is endpoints + this angle. `None` when the three points are collinear or
/// coincident (no finite circle).
pub fn included_angle_through_degrees(
    from: [f64; 2],
    to: [f64; 2],
    through: [f64; 2],
) -> Option<f64> {
    // Circumcenter via the perpendicular-bisector determinant.
    let determinant = 2.0
        * (from[0] * (to[1] - through[1])
            + to[0] * (through[1] - from[1])
            + through[0] * (from[1] - to[1]));
    if determinant.abs() <= f64::EPSILON {
        return None;
    }
    let magnitude = |p: [f64; 2]| p[0] * p[0] + p[1] * p[1];
    let center = [
        (magnitude(from) * (to[1] - through[1])
            + magnitude(to) * (through[1] - from[1])
            + magnitude(through) * (from[1] - to[1]))
            / determinant,
        (magnitude(from) * (through[0] - to[0])
            + magnitude(to) * (from[0] - through[0])
            + magnitude(through) * (to[0] - from[0]))
            / determinant,
    ];
    let angle_of = |p: [f64; 2]| (p[1] - center[1]).atan2(p[0] - center[0]).to_degrees();
    let wrap = |a: f64| a.rem_euclid(360.0);
    let ccw_to_end = wrap(angle_of(to) - angle_of(from));
    let ccw_to_through = wrap(angle_of(through) - angle_of(from));
    // `through` on the counter-clockwise leg ⇒ the arc sweeps CCW (positive); otherwise
    // it is the clockwise remainder of the turn.
    Some(if ccw_to_through <= ccw_to_end {
        ccw_to_end
    } else {
        ccw_to_end - 360.0
    })
}

/// The OPERATION that turns a [`Sketch`]'s 2D profile into a 3D volume — the
/// "Sketch + Operation" model. A [`SketchSolid`] pairs a sketch with one of these.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Operation {
    /// Extrude the profile a whole number of voxels along its plane normal,
    /// producing a prism (≥1 for a non-empty prism).
    Extrude {
        /// Extrude span in voxels along the plane normal.
        height_voxels: u32,
    },
    /// Revolve the profile around an in-plane axis, producing a solid of
    /// revolution. The sketch's two in-plane coordinates are
    /// reinterpreted as (axial, radial): one in-plane world axis becomes the
    /// REVOLVE AXIS (selected by [`RevolveAxis`]) and the profile is swept around
    /// it through [`RevolveSweep::turn_degrees`]. A rectangle revolved is a
    /// cylinder; a half-disc revolved is a sphere — revolve is the producer those
    /// primitives are sugar over, the same way extrude subsumes the box.
    Revolve {
        /// Which in-plane world axis is the revolve (axial) axis.
        axis: RevolveAxis,
        /// How far around the axis the profile is swept.
        sweep: RevolveSweep,
    },
}

/// Which of the plane's two in-plane world axes is the REVOLVE (axial) axis — the
/// axis the profile is swept around. The other in-plane axis plus the plane NORMAL
/// become the two RADIAL world axes the swept disc lives in.
///
/// The profile's two coordinates `[c0, c1]` (along [`PlaneAxis::in_plane_axes`]`[0]`
/// and `[1]`) are reinterpreted as (axial, radial):
///
/// | axis        | axial world axis    | axial profile coord | radial profile coord |
/// |-------------|---------------------|---------------------|----------------------|
/// | `InPlane0`  | `in_plane_axes()[0]`| `c0`                | `c1`                 |
/// | `InPlane1`  | `in_plane_axes()[1]`| `c1`                | `c0`                 |
///
/// The revolve axis sits at radial coordinate `= 0`; the profile may sit on one
/// side touching the axis, or straddle it (folded by `abs` into the radius).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RevolveAxis {
    /// Revolve around `in_plane_axes()[0]`; axial profile coord is `c0`, radial is `c1`.
    InPlane0,
    /// Revolve around `in_plane_axes()[1]`; axial profile coord is `c1`, radial is `c0`.
    InPlane1,
}

/// How far the profile is swept around the revolve axis. `360` degrees is a full
/// solid of revolution; a smaller value `(0, 360]` is a partial wedge. `0` is
/// degenerate (empty occupancy).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RevolveSweep {
    /// Sweep angle in whole degrees; `360` = full revolve, `(0, 360]` valid.
    pub turn_degrees: u32,
}

impl Default for Operation {
    /// A degenerate extrude (zero height ⇒ empty occupancy). Used so a document
    /// node missing its operation deserializes to a no-op rather than failing.
    fn default() -> Self {
        Operation::Extrude { height_voxels: 0 }
    }
}
