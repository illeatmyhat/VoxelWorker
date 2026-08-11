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
mod target;
#[cfg(test)]
mod tests;
mod transform;

pub use constraint::{
    AngleArm, AngleCorner, ArcEnd, CoincidentTarget, Constraint, ConstraintKind, ConstraintRefusal,
    Dimension, InPlaneAxis, InternalContainment, LineSide, SketchCurve, SymmetryBranch,
    TangentBranch,
};
pub use faces::{Face, FaceKey};
pub use modify::{
    BreakPlacement, BreakRefusal, ChamferPlacement, ChamferRefusal, ExtendEndpoint,
    ExtendPlacement, ExtendRefusal, FilletPlacement, FilletRefusal, OffsetPlacement, OffsetRefusal,
    TrimPlacement, TrimRefusal,
};
pub use parametric::sketch::{SnapReach, SolveOutcome, SolveReport};
pub use parametric::{CircleRadius, CurveParameter, ResolvedLength};
pub use pattern::{
    DerivedPatternCurve, SketchPattern, SketchPatternKind, SketchPatternRefusal, SketchVector,
};
pub use solid::SketchSolid;
pub use substrate::geom2d::LoopRole;
pub use target::SketchTarget;
pub use transform::{SketchTransformEntity, SketchTransformRefusal};

use parametric::sketch::{HandRole, KeptQuantity};
use parametric::units::{AngleMeasurement, Measurement};
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

impl TangentArcRefusal {
    /// Whether MOVING THE CURSOR could clear this refusal.
    ///
    /// The two halves of this enum answer different questions. A geometric refusal says the arc
    /// through *this* cursor position is impossible — the sweep is absurd, the branch is
    /// unreachable, the endpoint is the seam — and the author fixes it by pointing somewhere else,
    /// so it is worth saying at the cursor. The rest say the HELD incoming curve is unusable, which
    /// no amount of pointing changes; marking those would leave a refusal standing on screen for
    /// the whole gesture and teach the author nothing.
    #[must_use]
    pub const fn is_about_the_cursor(&self) -> bool {
        match self {
            Self::SelfLoop
            | Self::Candidate(_)
            | Self::UnrepresentableSweep
            | Self::ArcRefused
            | Self::Branch(_)
            | Self::Constraint(_) => true,
            Self::UnsupportedIncoming
            | Self::UnknownIncoming
            | Self::NonIncidentIncoming
            | Self::UnknownEndpoint => false,
        }
    }
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
    /// Continuous circular geometry, including exposed radius and signed sweep.
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RectangleRefusal {
    /// The continuous construction is degenerate or non-finite.
    Candidate(parametric::sketch::RectangleCandidateError),
    /// A solved corner cannot be represented distinctly in canonical point storage.
    Unrepresentable,
    /// Every boundary edge already exists, so the command would change nothing.
    AlreadyExists,
    /// A boundary edge the rectangle means to constrain could not be resolved.
    UnknownSegment,
    /// A relation the rectangle asserts was refused. The rectangle is a shape AND the
    /// assertions that keep it one, so a refused relation refuses the whole command rather
    /// than leaving four unconstrained lines behind that merely look rectangular.
    Constraint(ConstraintRefusal),
}

/// Canonical boundary corners shared by rectangle preview and commit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RectanglePlacement {
    /// Boundary-ordered corners, with the final edge closing index 3 back to index 0.
    pub corners: [SketchPoint; 4],
}

/// How far a body drag is allowed to ask for in one solve, in the plane's own units.
///
/// A relation web has a FAMILY of exact answers, not one, and a solve reaches whichever it walks
/// to. Asked for a small motion it walks to the answer beside the drawing, which is the one the
/// author meant; asked for a large one it can cross to a distant member of the same family that
/// satisfies everything equally well and looks nothing like the shape they drew. Measured on a
/// curved slot of radius forty: a quarter-unit pull widened it by exactly a quarter unit, while
/// the same pull delivered as one two-unit jump threw its inner rail twenty-four units inward
/// and then failed a tangency outright.
///
/// So a long displacement is delivered as a run of short ones, each read off the geometry the last
/// one left. This is continuation, and it is the ordinary remedy: every intermediate drawing is a
/// real drawing, so the search never has to cross ground where the answer is ambiguous.
const NUDGE_A_DRAG_WALKS: f64 = 0.25;

/// The ceiling on how many nudges one drag is broken into.
///
/// A drag is answered inside a frame, so the work it can ask for has to be bounded no matter how
/// far the cursor jumped. Past this the nudges simply get longer, which is the old behavior and no
/// worse than it was.
const MOST_NUDGES_A_DRAG_WALKS: usize = 24;

/// Break the displacement from `grabbed` to `to` into the nudges a drag is delivered in, each a
/// (from, to) pair. See [`NUDGE_A_DRAG_WALKS`] for why a drag walks rather than jumps.
///
/// The pairs are struck off the STRAIGHT line the cursor travelled, not off the curve as it moves.
/// Where the drawing carries the curve somewhere else along the way, the next nudge is read
/// against the geometry that arrived, which is what makes the walk follow the shape instead of the
/// plan — see [`Sketch::what_a_body_drag_asks_of`].
fn nudges_a_drag_is_delivered_in(grabbed: [f64; 2], to: [f64; 2]) -> Vec<([f64; 2], [f64; 2])> {
    let by = [to[0] - grabbed[0], to[1] - grabbed[1]];
    let reach = by[0].hypot(by[1]);
    if !reach.is_finite() || reach <= NUDGE_A_DRAG_WALKS {
        return vec![(grabbed, to)];
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a positive finite ratio, and the ceiling below bounds it either way"
    )]
    let count = ((reach / NUDGE_A_DRAG_WALKS).ceil() as usize).clamp(1, MOST_NUDGES_A_DRAG_WALKS);
    let mut walked = Vec::with_capacity(count);
    let mut stood = grabbed;
    for nudge in 1..=count {
        #[expect(
            clippy::cast_precision_loss,
            reason = "bounded by MOST_NUDGES_A_DRAG_WALKS, which is exact in f64"
        )]
        let carried = nudge as f64 / count as f64;
        let next = [
            carried.mul_add(by[0], grabbed[0]),
            carried.mul_add(by[1], grabbed[1]),
        ];
        walked.push((stood, next));
        stood = next;
    }
    walked
}

/// Below this a curve is too small for the direction across it to be read from its own points,
/// and a drag of it would report a wild displacement out of a rounding difference.
const DEGENERATE_CURVE: f64 = 1.0e-9;

/// What a drag answered: whether it stood, and the quantity it was pulled onto if it snapped.
///
/// The snap is the reason this is not just a bool. A drag that keeps a radius puts the hand
/// somewhere slightly other than the cursor, and from the outside that is indistinguishable from
/// a solve that could not reach — the author reported exactly that: "I can't really tell if it's
/// snapping." So the drag says what it kept, and the overlay draws it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DragAnswer {
    /// Whether the drawing stood under the gesture. A drag that did not is rolled back whole.
    pub moved: bool,
    /// The quantity the hand was pulled onto, where one was — see
    /// [`parametric::sketch::KeptQuantity`].
    pub kept: Option<KeptQuantity>,
}

impl DragAnswer {
    /// A gesture that stood or did not, with no quantity to show for it. Every drag but a point's
    /// answers this way: a snap needs a LEAD hand to measure from, and a body drag has none.
    const fn stood(moved: bool) -> Self {
        Self { moved, kept: None }
    }
}

/// One point a drag asserts, on the document's own ids, and what it is doing there.
///
/// The document says the gesture rather than leaving the solver to work it out from the numbers:
/// [`parametric::sketch::HandRole`] carries why. This is the same shape as
/// [`parametric::sketch::Hand`] and is mapped onto it at the solver seam, where entity ids become
/// the solver's own point ids.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Hand {
    /// The point being asserted.
    pub point: EntityId,
    /// Where the gesture puts it, in plane coordinates.
    pub to: [f64; 2],
    /// What it is doing there.
    pub role: HandRole,
}

impl Hand {
    /// A point riding the gesture: the rest of a rigid set, moving by the same motion.
    fn carried(point: EntityId, to: [f64; 2]) -> Self {
        Self {
            point,
            to,
            role: HandRole::Carried,
        }
    }

    /// A point held where it already stands, which is how a reshape names what it turns about.
    fn pin(point: EntityId, at: [f64; 2]) -> Self {
        Self {
            point,
            to: at,
            role: HandRole::Pin,
        }
    }
}

/// What a body drag of a curve asks of the drawing: where the curve's own points are PUT before
/// the solve runs, and where they are PULLED while it does.
///
/// Two lists rather than one, because they mean different things. See
/// [`what_a_body_drag_asks_of`](Sketch::what_a_body_drag_asks_of) for why the difference is the
/// whole of how a drag says "reshape" instead of "travel".
#[derive(Debug, Clone, PartialEq)]
struct BodyDrag {
    seeded: Vec<(EntityId, [f64; 2])>,
    pulled: Vec<Hand>,
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotRefusal {
    /// The continuous construction is invalid or degenerate.
    Candidate(parametric::sketch::SlotCandidateError),
    /// A boundary endpoint or arc sweep cannot be represented in canonical storage.
    Unrepresentable,
    /// The complete boundary already exists.
    AlreadyExists,
    /// The boundary stands but the tangency that makes it a slot could not be asserted.
    Constraint(ConstraintRefusal),
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
    /// Connected boundary curves in traversal order. The two long rails are `[0]` and `[2]`.
    pub edges: [SlotEdgePlacement; 4],
    /// The tangency at each corner: junction `i` joins edge `i` to edge `i + 1` of
    /// [`edges`](Self::edges), wrapping, in that member order. Read off the continuous
    /// construction, never re-derived here.
    pub junctions: [parametric::sketch::TangentBranch; 4],
    /// The centerline, in canonical storage. Its handles become real points on commit and its
    /// turn decides what holds the two rails together.
    pub spine: SlotSpinePlacement,
    /// The two extremes an Overall Slot was authored by, absent for every other grammar. On
    /// commit they become points joined by a construction line down the slot's middle.
    pub reach: Option<[SketchPoint; 2]>,
}

/// A slot's centerline in canonical storage: the handles the drawing will remember it by.
///
/// The boundary does not contain these points — a spine is exactly the curve a slot does not
/// draw — so they are reified and tied to the boundary's own derived centers. That tie is what
/// makes the shape behave the way it was authored: hold the center and the slot translates, hold
/// an end and the slot reshapes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlotSpinePlacement {
    /// Coincides with the center of the cap closing the start.
    pub start: SketchPoint,
    /// Coincides with the center of the cap closing the end.
    pub end: SketchPoint,
    /// Present when the spine turns; coincides with the center both rails share.
    pub center: Option<SketchPoint>,
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

    /// A length the author wrote as an EXPRESSION, retained whole, at the voxel value it lands
    /// on right now.
    ///
    /// The door a measurement field commits through. Both halves are carried because neither is
    /// derivable from the other at the call site: the expression is the truth and survives a
    /// density re-target, while `voxels` is the canonical value the resolve reads today.
    /// [`retained_voxels`](Self::retained_voxels) is the special case where the expression the
    /// author wrote WAS a voxel count.
    pub fn retained(authored: Measurement, voxels: i64) -> Self {
        Self {
            voxels,
            local_voxels: 0.0,
            measurement: Some(authored),
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

/// How long a [`Point`] outlives the things that refer to it.
///
/// This is NOT [`EntityRole`], though a point once carried that type and the confusion cost four
/// bugs. A role is a linetype: how a curve draws and whether the region counts it as a boundary. A
/// point has no linetype and no point bounds anything on its own. What a point has instead is a
/// question no curve has to answer: when the last thing referring to it goes away, is it still
/// there?
///
/// It is also not the same question as [`is_arc_center`](Sketch::is_arc_center), which asks
/// who OWNS a position. An ellipse's width handle is anchored and authored at once: the author
/// placed it, nothing re-derives it, and it still has no business surviving its ellipse.
///
/// And it is not [`point_draws_at_rest`](Sketch::point_draws_at_rest), which asks whether the dot
/// is worth the ink. Three axes, three answers: a rectangle's corner is Freestanding, authored, and
/// silent, because two segment ends already mark it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum PointLifetime {
    /// The author placed it and it stays until the author deletes it, incident geometry or not.
    #[default]
    #[serde(alias = "Real")]
    Freestanding,
    /// It exists to serve the curves that name it. [`prune_orphan_centers`] sweeps it as soon as
    /// none of them do — an ellipse handle, a rectangle's center, a control-point spline's
    /// interior frame.
    ///
    /// [`prune_orphan_centers`]: Sketch::prune_orphan_centers
    #[serde(alias = "Construction")]
    CurveAnchored,
}

impl PointLifetime {
    /// What a point minted to serve a curve of this role should outlive.
    ///
    /// Breaking, filleting or offsetting a curve mints the points its pieces meet at, and those
    /// points inherit from the curve. The two quantities are different, so the inheritance is
    /// written out rather than assumed: reference geometry's junctions are the drawing's and go
    /// when it does; real geometry's junctions are the author's and stay as free points.
    pub const fn serving(role: EntityRole) -> Self {
        match role {
            EntityRole::Real => Self::Freestanding,
            EntityRole::Construction => Self::CurveAnchored,
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
    /// The loop as a closed polygon, each chord's sagitta within `sagitta_tolerance`.
    ///
    /// **A terminal adapter, not a stage.** Every caller of this is producing something discrete
    /// and has nowhere to put a curve; anything that merely wants to know where the boundary is
    /// asks the field instead.
    pub fn flatten(&self, sagitta_tolerance: f64) -> Vec<SketchPoint> {
        flatten_edges(&self.edges, sagitta_tolerance)
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

/// A closed edge loop as a closed polygon, each chord's sagitta within `sagitta_tolerance`.
///
/// **A terminal adapter, not a stage.** Reach for it only where something discrete is being
/// produced and there is nowhere to put a curve — a crease polyline, a screen-space hit-test
/// polygon, the exact-`f64` cell classifier. Anything that merely wants to know where the boundary
/// is asks the field ([`substrate::geom2d::signed_distance_to_region`]) instead.
pub fn flatten_edges(edges: &[ProfileEdge], sagitta_tolerance: f64) -> Vec<SketchPoint> {
    let mut points = Vec::with_capacity(edges.len());
    for edge in edges {
        points.push(edge.from);
        points.extend(edge.interior_points(sagitta_tolerance));
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

/// One [`Arc`]'s three placed points, read as the circle they draw.
///
/// The stored form is three positions and nothing else (ADR 0038). Everything a consumer used to
/// take off the arc — where its center is, how big it is, how far it turns — is this reading, made
/// once by [`Sketch::arc_form`] so that no two places derive it differently.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArcForm {
    /// Where the arc starts.
    pub from: [f64; 2],
    /// Where it ends, having turned counter-clockwise to get there.
    pub to: [f64; 2],
    /// The authored center PROJECTED onto the chord's perpendicular bisector, so the circle
    /// named here passes through both ends exactly. See [`arc_center_on_bisector`].
    pub center: [f64; 2],
    /// The distance from `center` out to either end — they are the same by construction, the
    /// projection having removed the only way they could differ.
    pub radius: f64,
    /// The counter-clockwise turn `from → to`, strictly inside `(0, 360)`.
    pub sweep_degrees: f64,
}

/// Where a curve that TURNS stands: its center, and how far its rim is from it.
///
/// The two things every turning curve has and no straight one does. [`ArcForm`] answers more than
/// this and only for an arc; this answers the part a circle can answer too, so a caller measuring
/// a radius does not have to know first which of the two it is holding — which is the same reason
/// the solver's radius relation is written once.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CircularForm {
    /// The center, in plane coordinates. An arc's is projected onto its chord's bisector, so it is
    /// the center of the circle the arc actually lies on rather than the one the author placed.
    pub center: [f64; 2],
    /// The distance from `center` out to the rim, in voxels.
    pub radius: f64,
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
    pub fn interior_points(&self, sagitta_tolerance: f64) -> Vec<SketchPoint> {
        if let Some(curve) = self.bezier {
            let points = curve.flatten(sagitta_tolerance);
            return points
                .iter()
                .skip(1)
                .take(points.len().saturating_sub(2))
                .map(|point| SketchPoint::from_continuous(point[0], point[1]))
                .collect();
        }
        match self.arc {
            Some(arc) => arc_interior_on_circle(arc, sagitta_tolerance),
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
    /// Whether this point outlives the curves that name it. Reads `role` in documents written
    /// while a point still carried an [`EntityRole`].
    #[serde(default, alias = "role")]
    pub lifetime: PointLifetime,
    /// What a drag that grabs this point is a statement ABOUT.
    #[serde(default)]
    pub handle: PointHandle,
}

/// What a drag that grabs a point is a statement about.
///
/// A shape's hub is DECLARED by the tool that draws the shape, because the tool is the only thing
/// that knows. It used to be inferred — a center was a hub when several curves turned about it —
/// and inference cannot see a shape whose parts do not turn about anything, which is why a
/// straight slot could never be dragged whole while a curved one could.
///
/// Nobody in the field infers this. FreeCAD's sketcher carries a `Group` constraint whose first
/// element is the handle geometry and swaps any grabbed member for it before a drag list is built;
/// D-Cubed, under Fusion, calls the same thing a declared RIGID SET. SolveSpace deletes the
/// question instead — drag a curve's body to translate, drag a point to reshape — which is the
/// answer for every shape that declares no hub.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum PointHandle {
    /// A point of the drawing. Grabbing it is a statement about the curves that meet or turn about
    /// it, and no further.
    #[default]
    Ordinary,
    /// The point a whole shape is dragged BY. Grabbing it translates every curve the shape walk
    /// reaches, and it names no pivot, because moving a shape is not reshaping it.
    ShapeHub,
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
    /// The [`Point`] entity the arc turns about — AUTHORED, exactly as a [`Circle`]'s center is
    /// (ADR 0038). Nothing recomputes it: it is where the author put it, and two arcs are
    /// concentric by naming one id rather than by standing at the same coordinates.
    ///
    /// Always [`EntityRole::Construction`]: a center never bounds a region.
    ///
    /// The arc runs **counter-clockwise from [`from`](Self::from) to [`to`](Self::to)** about it.
    /// There is no stored sweep and no stored direction — the endpoint order IS the direction, and
    /// an arc bent the other way is the same three points with the ends swapped. The swept angle
    /// and the radius are read off the three positions by [`Sketch::arc_form`].
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
    /// The [`Point`] entity at the circle's center. Authored, like an [`Arc`]'s: it is where the
    /// author put it, and nothing recomputes it. The arc's was derived until ADR 0038 placed it,
    /// and the two have read the same ever since.
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

/// One endpoint/control/rho conic. Rho is exact and dimensionless in durable storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Conic {
    pub id: EntityId,
    pub from: EntityId,
    pub to: EntityId,
    /// The construction point the two end tangents meet at — the handle the curve bends toward.
    ///
    /// Stored rather than the on-curve shoulder because this is the point the author placed and
    /// the point they reach for afterwards: moving it drags the curve the way a Bezier handle
    /// does, while the shoulder is a consequence of it and [`rho`](Self::rho).
    pub control: EntityId,
    /// How far out toward [`control`](Self::control) the curve bends, dimensionless in `(0, 1)`.
    ///
    /// The one authored freedom here with no point of its own. It used to be given one — an
    /// on-curve "shoulder" dot at `t = 0.5`, recomputed from the other three every sync — and
    /// ADR 0038 took it away: a point is placed, never computed, and that one was a value wearing
    /// a point's clothes. Rho is re-authored by dragging the conic's BODY, which is where the
    /// author was already aiming.
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

impl SplineKind {
    /// The fewest points this kind of spline still describes a curve with.
    ///
    /// Below it there is no curve left to simplify to, so a point delete that would cross this
    /// floor deletes the spline instead of healing it.
    const fn fewest_points(self, closed: bool) -> usize {
        match self {
            // A closed interpolant needs three points to be a loop rather than a doubled-back
            // pair; open, two points are the curve through both of them.
            Self::FitPoint if closed => 3,
            Self::FitPoint | Self::ControlPoint => 2,
        }
    }
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
    /// The tangent handle standing at each fit point, keyed by the fit point it steers.
    ///
    /// EVERY fit point has one, from the moment the spline is drawn: a handle is furniture the
    /// curve comes with rather than a thing the author adds, and there is no verb to add or remove
    /// one (owner, 2026-08-03). Each is minted where the curve already bends, so a fresh spline
    /// draws exactly the curve it would have drawn with no handles at all.
    ///
    /// Keyed rather than a slot per point, so the map says which point each handle belongs to
    /// instead of a reader having to count. Empty for a control-point spline, whose points steer
    /// the curve by standing off it.
    #[serde(default)]
    pub tangents: std::collections::BTreeMap<EntityId, TangentHandle>,
}

/// One fit point's tangent handle: a double-sided lever whose midpoint IS the fit point.
///
/// # Two ends, one quantity
///
/// The lever is symmetric — equal arms, mirrored about the point they steer (owner, 2026-08-03)
/// — so [`backward`](Self::backward) holds nothing [`forward`](Self::forward) does not. `forward`
/// is the truth; the back arm is restored to the mirror of it by
/// [`Sketch::sync_tangent_arms`] after every edit, and grabbing the back arm steers the front one.
///
/// # So why store the back arm at all
///
/// Because a grabbable thing has to be a point here. The shell resolves a click to an id out of
/// the document's own point list, so an arm drawn without being stored is an arm the author can
/// see and cannot take hold of — a painted affordance that does nothing, which is the one thing
/// this codebase has already been told not to ship. The mirror cannot drift in exchange, because
/// nothing reads it: the curve's derivative is read off `forward` alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TangentHandle {
    /// The arm on the far side of the fit point from the curve's start — the one the tangent is
    /// measured to.
    pub forward: EntityId,
    /// The mirrored arm, behind the fit point. Grabbable, derived, read by nothing.
    pub backward: EntityId,
}

impl TangentHandle {
    /// Both ends, for the cascades that must treat the lever as one object.
    pub fn arms(self) -> [EntityId; 2] {
        [self.forward, self.backward]
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
    /// Endpoint/control/rho conic entities; absent in older documents.
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

/// How close two dots have to be before they are one dot, in plane units.
///
/// A thousandth of a unit is nowhere near drawable and nowhere near authorable — it exists to
/// absorb the residual a solved coincidence leaves behind, not to forgive a near miss. See
/// [`Sketch::point_draws_at_rest`].
const STACKED_DOT_TOLERANCE: f64 = 1.0e-3;

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
                // An arc runs counter-clockwise from its `from` to its `to` (ADR 0038), so
                // which end the joint is at IS the direction of travel through it.
                let orientation = if arc.to == seam {
                    1.0
                } else if arc.from == seam {
                    -1.0
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

    /// Every coincidence whose point has been carried off the DRAWN piece of the curve it names,
    /// paired with the support the author never drew.
    ///
    /// The claim is not broken and the point is not misplaced — a point-on-curve residual reads the
    /// support on purpose, so the point really is on the curve. What misleads is the drawing, which
    /// shows only the piece the author cut and so reads as though the point escaped it. Handing the
    /// undrawn remainder to the shell lets it say the true thing: the point is on this curve, and
    /// here is the part of the curve you did not draw.
    ///
    /// Only a curve TARGET can answer. A coincidence between two points names no extent to be
    /// outside of.
    ///
    /// Read per frame, so it walks the live store and clones nothing.
    #[must_use]
    pub fn undrawn_reaches(
        &self,
        context: parametric::EvaluationContext,
    ) -> Vec<(EntityId, parametric::sketch::UndrawnReach)> {
        // The gate's own tolerance: a point within the piece by any margin at all is simply on it,
        // and a reach the width of a rounding error is a mark that says nothing.
        const MET: f64 = 1.0e-6;
        self.constraints
            .iter()
            .filter_map(|constraint| match constraint.kind {
                ConstraintKind::Coincident {
                    point,
                    onto: CoincidentTarget::Curve(curve),
                } => {
                    let at = self.point_in_plane(point)?;
                    let geometry = self.curve_geometry(curve, context)?;
                    parametric::sketch::undrawn_reach_to(geometry, at, MET)
                        .map(|reach| (point, reach))
                }
                _ => None,
            })
            .collect()
    }

    /// Where on `curve` a pick taken at `at` is standing.
    ///
    /// The pick's own position, moved onto the curve it was taken on. A pick that becomes a stored
    /// point could be left where the cursor was and pulled in by its coincidence, but a pick that
    /// only fixes a radius or an angle has nothing to pull it, and from the author's side the two
    /// are the same gesture: the curve was lit, so the pick is on it. Landing here is what makes
    /// that true of both, and it also means a planted point starts where its own constraint
    /// already wants it instead of being dragged there by the first solve.
    ///
    /// Every curve kind answers, aggregates included. A spline has no one center or direction to
    /// hold a relation to, which is a statement about relations; where the pointer is standing on
    /// it is not in doubt, and the nearest of its spans says so. `None` only when the curve is
    /// gone — the caller keeps the position it had.
    #[must_use]
    pub fn point_on_curve(
        &self,
        curve: SketchCurve,
        at: SketchPoint,
        context: parametric::EvaluationContext,
    ) -> Option<SketchPoint> {
        let in_plane = at.in_plane();
        let landed = self
            .curve_spans(curve, context)
            .into_iter()
            .map(|span| span.point_at(span.nearest_parameter(in_plane)))
            .min_by(|left, right| {
                let away = |from: &[f64; 2]| (from[0] - in_plane[0]).hypot(from[1] - in_plane[1]);
                away(left).total_cmp(&away(right))
            })?;
        Some(SketchPoint::from_continuous(landed[0], landed[1]))
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
            // The aggregates, which have no one center, radius, or direction to report. This arm
            // is what [`SketchCurve::carries_relation_geometry`] states ahead of the click, so a
            // gesture can refuse an aggregate instead of taking it and failing to apply.
            SketchCurve::Bezier(_)
            | SketchCurve::Ellipse(_)
            | SketchCurve::Conic(_)
            | SketchCurve::Spline(_) => None,
            SketchCurve::Arc(id) => {
                let form = self.arc_form_of(id)?;
                Some(CurveGeometry::Circular(CircularCurve {
                    center: form.center,
                    radius: form.radius,
                    arc: Some(ArcDomain {
                        from: form.from,
                        to: form.to,
                        sweep_radians: form.sweep_degrees.to_radians(),
                    }),
                }))
            }
        }
    }

    /// Read one arc's three placed points as the circle they draw, or nothing when they draw no
    /// circle — a point missing from the store, the two ends stacked, an end sitting on the center.
    ///
    /// This is THE reading. An arc stores no sweep, no radius and no direction, so every consumer
    /// that wants one comes through here and they all agree by construction.
    #[must_use]
    pub fn arc_form(&self, arc: &Arc) -> Option<ArcForm> {
        let (from, to) = (self.point_in_plane(arc.from)?, self.point_in_plane(arc.to)?);
        let center = arc_center_on_bisector(from, to, self.point_in_plane(arc.center)?)?;
        let sweep_degrees = counter_clockwise_sweep_degrees(center, from, to)?;
        let radius = (from[0] - center[0]).hypot(from[1] - center[1]);
        radius.is_finite().then_some(ArcForm {
            from,
            to,
            center,
            radius,
            sweep_degrees,
        })
    }

    /// [`arc_form`](Self::arc_form) by arc id, for a caller holding a [`SketchCurve`] rather than
    /// the entity.
    #[must_use]
    pub fn arc_form_of(&self, id: EntityId) -> Option<ArcForm> {
        self.arc_form(self.arcs.iter().find(|arc| arc.id == id)?)
    }

    /// Whether this arc still draws something an author could see and grab.
    ///
    /// Stricter than [`arc_form`](Self::arc_form) reading at all, and the difference is the whole
    /// point: the reading answers arithmetic, and a radius of `4.4e-11` about a place all three
    /// points are stacked on is arithmetically a perfectly good circle. An end within
    /// [`STACKED_DOT_TOLERANCE`] of the center is the same dot as the center, and an arc turning
    /// about a place it stands on has nothing left to turn.
    ///
    /// The two ENDS are asked the same question, and it is the seam a drag crosses rather than a
    /// place it lands: with the ends stacked there is no piece of the circle to prefer, and the
    /// reading answered `sweep = 0.00` with the radius already collapsed from 56.57 to 40 — which
    /// then seeded the next frame and stayed. Both halves are one question, "are these two dots one
    /// dot", so both are asked against [`STACKED_DOT_TOLERANCE`].
    #[must_use]
    fn arc_draws_a_circle(&self, id: EntityId) -> bool {
        self.arcs
            .iter()
            .find(|arc| arc.id == id)
            .and_then(|arc| {
                Some(three_points_draw_a_circle(
                    self.point_in_plane(arc.from)?,
                    self.point_in_plane(arc.to)?,
                    self.point_in_plane(arc.center)?,
                ))
            })
            .unwrap_or(false)
    }

    /// The counter-clockwise sweep from one named end of an arc round to the other, read the way
    /// the drawing reads it — through the SEATED center, so the number is the one that is drawn.
    ///
    /// Asked of two ends by id rather than of an arc, because the caller that wants it
    /// ([`ArcTurnUnderAGesture`]) is watching two particular dots across a gesture that may swap
    /// which of them the arc calls its `from`.
    fn drawn_sweep_between(
        &self,
        center: EntityId,
        first: EntityId,
        second: EntityId,
    ) -> Option<f64> {
        drawn_sweep_of(
            self.point_in_plane(center)?,
            self.point_in_plane(first)?,
            self.point_in_plane(second)?,
        )
    }

    /// How the drawing currently stands `curve` off its own center, or `None` for a curve that
    /// does not turn.
    ///
    /// The one public door to a radius. A circle keeps its radius as an authored parameter and an
    /// arc reads one out of three placed points, and no caller that merely wants to MEASURE a rim
    /// should have to know that.
    pub fn circular_form(
        &self,
        curve: SketchCurve,
        context: parametric::EvaluationContext,
    ) -> Option<CircularForm> {
        match curve {
            SketchCurve::Arc(id) => self.arc_form_of(id).map(|form| CircularForm {
                center: form.center,
                radius: form.radius,
            }),
            SketchCurve::Circle(id) => {
                let circle = self.circles.iter().find(|held| held.id == id)?;
                let center = self.point_in_plane(circle.center)?;
                let radius = circle.resolved_radius(context);
                radius
                    .is_finite()
                    .then_some(CircularForm { center, radius })
            }
            SketchCurve::Segment(_)
            | SketchCurve::Bezier(_)
            | SketchCurve::Ellipse(_)
            | SketchCurve::Conic(_)
            | SketchCurve::Spline(_) => None,
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
            SketchCurve::Arc(id) => Some(self.arc_form_of(id)?.center),
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

    /// Every tangent handle's lever: the spline it steers, and the three points the line runs
    /// through — back arm, fit point, forward arm, in that order.
    ///
    /// Drawn, not stored, exactly as [`control_polygons`](Self::control_polygons) is: the line is
    /// the run of points, and there is nothing to keep. The ARMS are stored, because they are
    /// grabbed; the line between them is not.
    pub fn tangent_handle_legs(&self) -> Vec<(EntityId, [EntityId; 3])> {
        self.splines
            .iter()
            .flat_map(|spline| {
                spline
                    .tangents
                    .iter()
                    .map(|(fit, handle)| (spline.id, [handle.backward, *fit, handle.forward]))
            })
            .collect()
    }

    /// Every control-point spline's frame: the spline's id, and its controls in the order the
    /// legs between them run, closing back to the first control when the spline is closed.
    ///
    /// # The legs are drawn, never stored
    ///
    /// A leg IS the adjacency of the spline's own point list, so there is nothing here to
    /// serialize, retarget, or delete — which is what makes the frame derived and undeletable by
    /// construction rather than by a guard. Reifying the legs as segments would put curve-shaped
    /// things in the store that no face may use and no constraint may name, and every cascade,
    /// trim, and break would have to learn to spare them. A viewer that wants a leg to be
    /// PICKABLE resolves the hit to the spline, the way an arc's rim resolves to the arc;
    /// pickable and stored are separate questions.
    ///
    /// Fit-point splines are absent: their points are ON the curve, so there is no frame standing
    /// off it to draw.
    pub fn control_polygons(&self) -> Vec<(EntityId, Vec<EntityId>)> {
        self.splines
            .iter()
            .filter(|spline| spline.kind == SplineKind::ControlPoint)
            .map(|spline| {
                let mut controls = spline.points.clone();
                if spline.closed {
                    controls.extend(spline.points.first().copied());
                }
                (spline.id, controls)
            })
            .collect()
    }

    /// Put a curve into the construction role, whatever role it held.
    ///
    /// Idempotent where [`toggle_construction`](Self::toggle_construction) is not. A tool that
    /// authors reference geometry knows the role it wants; toggling would flip an entity it
    /// happened to reuse back to real and quietly hand the region a new boundary.
    ///
    /// # Construction is a mode a curve is in, and a point is not a curve
    ///
    /// Construction says how a curve DRAWS and whether the region counts it as a boundary. A point
    /// has neither of those, so a point id is a no-op here. What a point carries in the same field
    /// is a different quantity wearing the same name: its LIFETIME. A Construction point belongs to
    /// the drawing rather than to the author — an ellipse handle, a control-point spline's interior
    /// frame — and [`prune_orphan_centers`](Self::prune_orphan_centers) sweeps it the moment nothing
    /// refers to it. That is bookkeeping, not a mode the author can be offered, so it has its own
    /// door in `set_point_lifetime` and this one refuses it.
    pub fn set_construction(&mut self, id: EntityId) {
        if let Some(curve) = self.curve_named(id) {
            self.set_curve_role(curve, EntityRole::Construction);
        }
    }

    /// Whether `curve` draws as real geometry or as reference, or nothing if the drawing has no
    /// such curve. The question a body drag asks before deciding whether it is an offset or a
    /// translation — see [`translate_curve`](Self::translate_curve).
    pub fn curve_role(&self, curve: SketchCurve) -> Option<EntityRole> {
        self.curve_lineage_and_role(curve).map(|(_, role)| role)
    }

    /// Flip one CURVE between real and construction while retaining its stable id.
    ///
    /// Points, constraint ids and unknown ids are harmless no-ops — see
    /// [`set_construction`](Self::set_construction) for why a point has no construction mode to
    /// flip. Refusing every point is also what keeps a derived arc or circle center out of the
    /// region: no generic selection action can make one participate as a profile vertex.
    pub fn toggle_construction(&mut self, id: EntityId) -> bool {
        let Some(curve) = self.curve_named(id) else {
            return false;
        };
        let Some((_, role)) = self.curve_lineage_and_role(curve) else {
            return false;
        };
        self.set_curve_role(curve, role.toggled());
        true
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
            lifetime: PointLifetime::Freestanding,
            handle: PointHandle::Ordinary,
        });
        id
    }

    /// Allocate a point that serves a curve and is swept once no curve names it.
    fn add_construction_point(&mut self, at: SketchPoint) -> EntityId {
        let id = self.alloc_id();
        self.points.push(Point {
            id,
            at,
            lifetime: PointLifetime::CurveAnchored,
            handle: PointHandle::Ordinary,
        });
        id
    }

    /// Name the point a whole shape is dragged by — see [`PointHandle::ShapeHub`].
    ///
    /// Only the tool that draws a shape may call this, at the moment it draws it. Nothing infers a
    /// hub afterwards, and nothing revokes one: a hub is a fact about how the shape was authored,
    /// so it survives every later edit that keeps the point.
    fn declare_shape_hub(&mut self, point: EntityId) {
        if let Some(index) = self.point_index(point) {
            self.points[index].handle = PointHandle::ShapeHub;
        }
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
    /// Served from [`region_memo`]: deriving the arrangement costs an all-pairs curve
    /// intersection, and the shell asks for these EVERY FRAME to keep its face hit-test polygons.
    /// A committed spline is tens of rational-Bézier pieces, so re-deriving per frame measured
    /// 5–15 ms on nothing but a redraw.
    pub fn faces(&self, context: parametric::EvaluationContext) -> Vec<Face> {
        self.derived(context).faces.clone()
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
        self.derived(context).region_field.to_loops()
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
        let faces = Self::nested_faces(&self.derived(context).faces);
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
        let faces = self.derived(context).faces.clone();
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
        let faces = Self::nested_faces(&self.derived(context).faces);
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

    /// `faces` in nesting order: smallest area first, so the FIRST face containing a point is the
    /// innermost one that does. [`substrate::geom2d::point_in_region`] takes the same order for
    /// the same reason.
    ///
    /// Takes the already-derived faces rather than deriving its own, so the arrangement runs once
    /// per [`region_memo`] miss instead of once per caller.
    fn nested_faces(faces: &[Face]) -> Vec<Face> {
        let mut nested = faces.to_vec();
        // Ties keep `derive`'s deterministic order, so the region is stable across derivations.
        nested.sort_by(|first, second| first.area.total_cmp(&second.area));
        nested
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

    /// The arrangement, derived from scratch. Only [`region_memo`] calls this; everything else
    /// asks [`faces`](Self::faces) and gets the same answer without re-deriving it.
    pub(super) fn faces_uncached(&self, context: parametric::EvaluationContext) -> Vec<Face> {
        faces::derive(self, context)
    }

    /// The region read off an already-derived arrangement. Only [`region_memo`] calls this;
    /// everything else asks [`region`](Self::region).
    pub(super) fn region_from_faces(&self, faces: &[Face]) -> Vec<ProfileLoop> {
        let nested = Self::nested_faces(faces);
        let picked = self.pick_flags(&nested);
        nested
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
            (1, Some(LoopRole::Fill)) => loops[0].flatten(ARC_SAGITTA_TOLERANCE),
            _ => Vec::new(),
        }
    }

    /// Move the point `id` to `at` and settle the drawing around it, reporting only whether the
    /// drawing stood. [`move_point_reporting_its_snap`](Self::move_point_reporting_its_snap) is
    /// the same gesture for a caller that also wants to DRAW what the drag kept — and to say how
    /// far the snap may carry the point, which a caller that cannot draw the ghost has no way of
    /// telling the author about anyway.
    pub fn move_point(
        &mut self,
        id: EntityId,
        at: SketchPoint,
        context: parametric::EvaluationContext,
    ) -> Result<bool, SketchEvaluationError> {
        self.move_point_reporting_its_snap(id, at, context, SnapReach::UNBOUNDED, &mut [])
            .map(|answered| answered.moved)
    }

    /// [`move_point`](Self::move_point), and what the drag was pulled onto — the write path an
    /// overlay uses, because it can show the author the quantity their hand is sliding along.
    ///
    /// One path, for every point (ADR 0038). There is no arm that reads the cursor as a quantity
    /// instead of a position: an arc's center is a placed point like any other, and a conic's rho
    /// is authored by dragging the conic's own body
    /// ([`drag_curve_through`](Self::drag_curve_through)).
    ///
    /// The point takes `at`, and then the standing constraints are re-solved with it
    /// pinned there — see [`settle_under_the_hands`](Self::settle_under_the_hands). A constraint
    /// that only held at the moment it was asserted is not a constraint; it has to survive the
    /// next drag, which is the first thing the author does to test it.
    /// `snap_reach` is how far the snap may carry the point off `at`. The shell computes it from
    /// its camera so the ceiling is a screen distance; a caller without one passes
    /// [`SnapReach::UNBOUNDED`], which is the kernel's own behaviour.
    pub fn move_point_reporting_its_snap(
        &mut self,
        id: EntityId,
        at: SketchPoint,
        context: parametric::EvaluationContext,
        snap_reach: SnapReach,
        carries: &mut [ArcTurnUnderAGesture],
    ) -> Result<DragAnswer, SketchEvaluationError> {
        // Grabbing the BACK arm of a tangent lever steers the FRONT one. The two ends name one
        // quantity, so only one of them can be the thing that moves; the mirror is restored by
        // `sync_tangent_arms` once the drag settles.
        let (id, at) = match self.back_arm_steers(id) {
            Some((fit, forward)) => {
                let Some(anchor) = self.point_in_plane(fit) else {
                    return Ok(DragAnswer::stood(false));
                };
                let grabbed = at.in_plane();
                (
                    forward,
                    SketchPoint::from_continuous(
                        2.0 * anchor[0] - grabbed[0],
                        2.0 * anchor[1] - grabbed[1],
                    ),
                )
            }
            None => (id, at),
        };
        if self.point_index(id).is_none() {
            return Ok(DragAnswer::stood(false));
        }
        self.drag_or_leave_it_alone(|sketch| {
            sketch.point_move_attempt(id, at, context, snap_reach, carries)
        })
    }

    /// Move the whole CURVE `curve` so that it passes under `at`, and settle around it. Reports
    /// whether the drawing accepted the move.
    ///
    /// PERPENDICULAR motion only — radial, on a curve that turns. Dragging a line along itself
    /// produces the same line, so the one thing this gesture can mean is "further from where it
    /// was, or nearer", and stating that as a distance rather than as a displacement makes the
    /// drag ABSOLUTE: the curve goes where the cursor is now, not where a sum of increments left
    /// it. Nothing accumulates and nothing to drift.
    ///
    /// This is the gesture that authors a slot's width, the freedom its relations deliberately
    /// leave open. It widens SYMMETRICALLY: the centerline holds and the far rail mirrors the
    /// grabbed one (owner, 2026-08-04). That is not a rule stated here so much as one the shape
    /// already implies — see
    /// [`handles_a_widening_must_hold`](Self::handles_a_widening_must_hold) for why holding the
    /// spine is the whole of it.
    /// No gesture in the shell reaches this any more: grabbing a curve translates it, and the width
    /// it used to author is now what the relations answer on their own. It stays as a verb because
    /// an explicit offset is a real tool and this is what it would be built on — the objection was
    /// ever only to a drag that GUESSED between offsetting and moving by reading the geometry.
    pub fn move_curve(
        &mut self,
        curve: SketchCurve,
        at: [f64; 2],
        context: parametric::EvaluationContext,
    ) -> Result<bool, SketchEvaluationError> {
        let Some(mut hands) = self.hands_moving_a_curve(curve, at) else {
            return Ok(false);
        };
        hands.extend(self.handles_a_widening_must_hold(curve, &hands));
        self.drag_or_leave_it_alone(|sketch| {
            // Where they stood, read before they are written down. See `settle_under_the_hands`.
            let was: Vec<(EntityId, [f64; 2])> = hands
                .iter()
                .filter_map(|hand| Some((hand.point, sketch.point_in_plane(hand.point)?)))
                .collect();
            for hand in &hands {
                if let Some(index) = sketch.point_index(hand.point) {
                    sketch.points[index].at = SketchPoint::from_continuous(hand.to[0], hand.to[1]);
                }
            }
            sketch.sync_derived_points();
            // No ceiling: a body drag names no lead hand, so there is no snap to bound.
            sketch.settle_under_the_hands(&hands, &was, context, SnapReach::UNBOUNDED, &mut [])
        })
        .map(|answered| answered.moved)
    }

    /// Drag `curve` by the place on it the author actually grabbed: `grabbed` goes to `to`, and
    /// what the rest of the drawing does about it is the relations' answer.
    ///
    /// ONE verb for every curve, and no reading of what a body drag "must have meant" for this
    /// shape. The gesture states the only thing the author expressed — the bit of curve under the
    /// cursor should end up under the cursor — by putting a point there, holding it ON the curve,
    /// and pulling that. Which of the curve's own points give way is then a question the drawing
    /// already has the answer to, and different answers for different shapes come out of the
    /// relations rather than out of a ladder here:
    ///
    /// - a slot's rail has both ends pinned by the tangency web, so nothing can move but the
    ///   radius — the drag WIDENS it, which is the gesture that authors a slot's width;
    /// - that slot's centerline runs between two handles that are free to travel, so carrying it
    ///   is cheaper than reshaping it and the whole slot comes along;
    /// - a line the author drew between two loose points simply travels.
    ///
    /// The grip is temporary in the strictest sense: it is minted, held, and taken away again
    /// inside the gesture, so a drag leaves the drawing with exactly the entities it started with.
    pub fn drag_curve_through(
        &mut self,
        curve: SketchCurve,
        grabbed: [f64; 2],
        to: [f64; 2],
        context: parametric::EvaluationContext,
    ) -> Result<bool, SketchEvaluationError> {
        if !self.holds_curve(curve) {
            return Ok(false);
        }
        // A curve the grip cannot be pinned to would simply not move. That is a gap in what the
        // solver can say, not a decision about what the gesture means, so the meaning is kept and
        // the drawing carries the curve bodily instead — the answer the grip would have produced
        // for a shape with nothing else holding it.
        // A conic's body drag is the one aggregate drag that RESHAPES rather than travels. Rho
        // is the conic's one authored freedom and ADR 0038 took away the on-curve dot that used
        // to carry it, so the body is the handle: pulling it toward the control point sharpens the
        // curve, pushing it back toward the chord flattens it. Translating a conic is what its
        // three placed points are for.
        if let SketchCurve::Conic(id) = curve {
            let Some(index) = self.conics.iter().position(|conic| conic.id == id) else {
                return Ok(false);
            };
            return self
                .drag_or_leave_it_alone(|sketch| {
                    sketch.reshape_conic_toward(index, grabbed, to);
                    sketch
                        .standing_constraints_hold(context)
                        .map(DragAnswer::stood)
                })
                .map(|answered| answered.moved);
        }
        if !curve.carries_relation_geometry() {
            return self.translate_curve(curve, [to[0] - grabbed[0], to[1] - grabbed[1]], context);
        }
        // Only a curve with ENDS can be seeded, which is the whole of what a seed does: put its
        // own points somewhere the relations no longer hold and let the solve repair them. A circle
        // has none, so it keeps the grip below.
        if self.curve_a_body_drag_can_seed(curve) {
            return self
                .drag_or_leave_it_alone(|sketch| {
                    let mut answered = DragAnswer::stood(true);
                    for (from, until) in nudges_a_drag_is_delivered_in(grabbed, to) {
                        let Some(asked) = sketch.what_a_body_drag_asks_of(curve, from, until)
                        else {
                            return Ok(DragAnswer::stood(false));
                        };
                        for (point, at) in &asked.seeded {
                            if let Some(index) = sketch.point_index(*point) {
                                sketch.points[index].at =
                                    SketchPoint::from_continuous(at[0], at[1]);
                            }
                        }
                        sketch.sync_derived_points();
                        answered = sketch.settle_under_the_hands(
                            &asked.pulled,
                            &[],
                            context,
                            SnapReach::UNBOUNDED,
                            &mut [],
                        )?;
                        if !answered.moved {
                            break;
                        }
                    }
                    Ok(answered)
                })
                .map(|answered| answered.moved);
        }
        let grip = self.add_free_point(SketchPoint::from_continuous(grabbed[0], grabbed[1]));
        self.set_point_lifetime(grip, PointLifetime::CurveAnchored);
        let holding = self.alloc_id();
        self.constraints.push(Constraint {
            id: holding,
            kind: ConstraintKind::Coincident {
                point: grip,
                onto: CoincidentTarget::Curve(curve),
            },
            redundant: false,
            anchor: None,
        });
        // A curve that TURNS is dragged by its rim, and a rim drag is about the radius: the center
        // holds and the distance out to it is what the author is changing. That is not a rule for
        // arcs, it is the rule a circle has always had here — dragging its rim grows it — and an
        // arc is a circle with two ends, so it would be strange for the same gesture on the same
        // shape to mean something else. Without it the arithmetic simply carries the whole shape,
        // because travelling costs a least-deformation solve nothing and reshaping costs it
        // something; measured, a slot rail moved the slot and left its width alone.
        //
        // The center is held, not moved, so naming a DERIVED point here is sound where dragging one
        // would not be: it asks the solve to keep the center where it computes to, which the ends
        // and the sweep between them are free to arrange.
        let mut hands = vec![Hand {
            point: grip,
            to,
            role: HandRole::Lead,
        }];
        hands.extend(
            self.center_point_of(curve)
                .and_then(|center| Some(Hand::pin(center, self.point_in_plane(center)?))),
        );
        // Minted BEFORE the rollback point, so a refused drag restores a drawing that still has the
        // grip in it and the same two lines take it away either way.
        let stood = self
            .drag_or_leave_it_alone(|sketch| {
                sketch.settle_under_the_hands(&hands, &[], context, SnapReach::UNBOUNDED, &mut [])
            })
            .map(|answered| answered.moved);
        self.constraints
            .retain(|constraint| constraint.id != holding);
        self.points.retain(|point| point.id != grip);
        self.sync_derived_points();
        stood
    }

    /// Whether `curve` has the ends a body drag needs in order to seed it. A circle has none, and
    /// keeps the grip.
    fn curve_a_body_drag_can_seed(&self, curve: SketchCurve) -> bool {
        match curve {
            SketchCurve::Segment(id) => self.segments.iter().any(|segment| segment.id == id),
            SketchCurve::Arc(id) => self.arcs.iter().any(|arc| arc.id == id),
            SketchCurve::Circle(_)
            | SketchCurve::Bezier(_)
            | SketchCurve::Ellipse(_)
            | SketchCurve::Conic(_)
            | SketchCurve::Spline(_) => false,
        }
    }

    /// What a body drag of `curve` from `grabbed` to `to` asks of the drawing.
    ///
    /// `None` for a curve with no ends to seed — a circle, and the higher curves a body drag
    /// translates whole.
    ///
    /// # Why the seed and the pull are not the same displacement
    ///
    /// A hand is a soft pull, so a solve given nothing but hands answers with the cheapest motion
    /// of the WHOLE drawing, and the cheapest motion is always a rigid slide. Measured that way,
    /// every drag came out a translation and nothing ever changed shape — a rectangle's edge
    /// pulled outward carried the whole rectangle with it.
    ///
    /// What tells the solve that a curve DEFORMED is where the configuration starts. Putting the
    /// curve's own points somewhere the relations no longer hold, and letting them be repaired
    /// from there, is the whole of "this curve moved and what the rest does about it is the
    /// drawing's business". The seed is the intent; the pull only says how far.
    ///
    /// # Which part of a drag can be a deformation
    ///
    /// A curve slid ALONG itself is the same curve, so the along part of a displacement cannot
    /// mean a deformation, and the across part is the only part that can. Nothing here reads what
    /// shape the curve belongs to — the split is a fact about curves, not about slots or
    /// rectangles. So the across part seeds, the whole displacement pulls, and the gap between
    /// them is exactly the travel. Measured, and each of these is the relations' answer rather
    /// than a rule stated here:
    ///
    /// - a rectangle's edge pulled outward moves that edge alone and the rectangle resizes;
    /// - the same edge pulled sideways translates the whole rectangle rigidly;
    /// - a slot's rail pulled across widens it symmetrically, the spine and far rail taking half;
    /// - the same rail pulled along slides the slot with its width EXACTLY unchanged;
    /// - an arc's rim pulled outward grows its radius at fixed sweep, and pulled sideways slides.
    ///
    /// This is also why the earlier reading of a body drag wandered. Seeding the whole
    /// displacement broke tangency along the curve as well as across it, and the repair split the
    /// difference between travelling and reshaping: a slot slid along its own rail fattened by
    /// 0.12, then 0.50, then 0.94 over three steps of the same size.
    fn what_a_body_drag_asks_of(
        &self,
        curve: SketchCurve,
        grabbed: [f64; 2],
        to: [f64; 2],
    ) -> Option<BodyDrag> {
        let ends = match curve {
            SketchCurve::Segment(id) => {
                let segment = self.segments.iter().find(|segment| segment.id == id)?;
                [segment.from, segment.to]
            }
            SketchCurve::Arc(id) => {
                let arc = self.arcs.iter().find(|arc| arc.id == id)?;
                [arc.from, arc.to]
            }
            SketchCurve::Circle(_)
            | SketchCurve::Bezier(_)
            | SketchCurve::Ellipse(_)
            | SketchCurve::Conic(_)
            | SketchCurve::Spline(_) => return None,
        };
        let by = [to[0] - grabbed[0], to[1] - grabbed[1]];
        let hub = self
            .center_point_of(curve)
            .and_then(|id| Some((id, self.point_in_plane(id)?)));
        // A curve that TURNS is dragged by its RIM, and the rim is the same everywhere: the cursor
        // sits on the curve at whatever angle it likes, so the only thing the gesture can be saying
        // is how far out the rim now stands. Read as a DISTANCE FROM THE CENTER that is the whole
        // of it, and the cursor ends up on the curve by construction (owner, 2026-08-07). Read as
        // a projection of the travel it is not: a projection leaves a tangential remainder, the
        // remainder had nowhere to go but the center, and the center slid out from under the shape
        // — grab an arc at (10,0) and pull to (14,6) and the center went to (0,6).
        //
        // This is the rule a circle's grip has always used, and an arc is a circle with two ends,
        // so the same gesture on the same shape has to mean the same thing.
        let (across, reach, along) = match hub {
            Some((_, hub)) => {
                let out = [grabbed[0] - hub[0], grabbed[1] - hub[1]];
                let stood = out[0].hypot(out[1]);
                if stood < DEGENERATE_CURVE {
                    return None;
                }
                let reach = (to[0] - hub[0]).hypot(to[1] - hub[1]) - stood;
                (
                    [reach * out[0] / stood, reach * out[1] / stood],
                    reach,
                    [0.0, 0.0],
                )
            }
            None => {
                let (across, reach) = self.the_part_of_a_drag_that_crosses(curve, grabbed, by)?;
                (across, reach, [by[0] - across[0], by[1] - across[1]])
            }
        };
        // A curve that turns deforms by GROWING about its center, so the across part scales its
        // ends; a straight one has no center and simply moves. Both ends grow by the SAME reach
        // rather than by the projection of a vector onto each, which would be zero at an end
        // standing a quarter turn from the place the author grabbed.
        let deformed = |at: [f64; 2]| match hub {
            Some((_, hub)) => {
                let out = [at[0] - hub[0], at[1] - hub[1]];
                let radius = out[0].hypot(out[1]);
                let grown = if radius > DEGENERATE_CURVE {
                    1.0 + reach / radius
                } else {
                    1.0
                };
                [grown.mul_add(out[0], hub[0]), grown.mul_add(out[1], hub[1])]
            }
            None => [at[0] + across[0], at[1] + across[1]],
        };
        let seeded: Vec<(EntityId, [f64; 2])> = ends
            .into_iter()
            .filter_map(|point| Some((point, deformed(self.point_in_plane(point)?))))
            .collect();
        // The pull is the seed plus the travel, for the curve's center as much as for its ends.
        //
        // The center is never SEEDED, which is the difference between an arc standing on its own
        // and one of an arc slot's two rails. That center is shared — the far rail, the caps and
        // the spine are all concentric with it — so writing it would move their geometry without
        // moving them, and measured, it did: a rail slid along its own sweep threw a point twenty
        // units sideways and failed a tangency outright. A hand states the same wish and lets
        // everything standing on the center come along.
        //
        // A turning curve's center is PINNED rather than carried, which says two things at once
        // and needs both: the center holds, and the radius is the quantity the author is spending,
        // so the carried-radius hold must not reach for it — see
        // [`Problem::quantities_a_carry_holds_still`], which skips an arc whose center is pinned.
        let mut pulled: Vec<Hand> = seeded
            .iter()
            .map(|(point, at)| Hand::carried(*point, [at[0] + along[0], at[1] + along[1]]))
            .collect();
        pulled.extend(hub.map(|(id, at)| match curve {
            SketchCurve::Segment(_) => Hand::carried(id, [at[0] + along[0], at[1] + along[1]]),
            _ => Hand::pin(id, at),
        }));
        (!seeded.is_empty()).then_some(BodyDrag { seeded, pulled })
    }

    /// The part of `by` that crosses `curve` at `grabbed` — across a segment, radially out of a
    /// curve that turns — as a displacement and as the signed reach along that direction.
    ///
    /// Both, because the two forms are wanted in different places: a straight curve moves by the
    /// displacement, while one that turns grows by the reach about a center it keeps.
    ///
    /// `None` where the curve is too small for that direction to mean anything.
    fn the_part_of_a_drag_that_crosses(
        &self,
        curve: SketchCurve,
        grabbed: [f64; 2],
        by: [f64; 2],
    ) -> Option<([f64; 2], f64)> {
        let across = match curve {
            SketchCurve::Segment(id) => {
                let segment = self.segments.iter().find(|segment| segment.id == id)?;
                let tail = self.point_in_plane(segment.from)?;
                let head = self.point_in_plane(segment.to)?;
                let span = [head[0] - tail[0], head[1] - tail[1]];
                let length = span[0].hypot(span[1]);
                if length < DEGENERATE_CURVE {
                    return None;
                }
                [-span[1] / length, span[0] / length]
            }
            SketchCurve::Arc(id) => {
                let arc = self.arcs.iter().find(|arc| arc.id == id)?;
                let center = self.point_in_plane(arc.center)?;
                let out = [grabbed[0] - center[0], grabbed[1] - center[1]];
                let reach = out[0].hypot(out[1]);
                if reach < DEGENERATE_CURVE {
                    return None;
                }
                [out[0] / reach, out[1] / reach]
            }
            SketchCurve::Circle(_)
            | SketchCurve::Bezier(_)
            | SketchCurve::Ellipse(_)
            | SketchCurve::Conic(_)
            | SketchCurve::Spline(_) => return None,
        };
        let reach = by[0].mul_add(across[0], by[1] * across[1]);
        Some(([reach * across[0], reach * across[1]], reach))
    }

    /// Where each conic's SHOULDER stands — the point of the curve at `t = 0.5`, one per conic.
    ///
    /// A reading, never a point. ADR 0038 took the stored shoulder away because it was a value
    /// wearing a point's clothes: it was recomputed from the other three and rho on every sync, so
    /// it could not be placed and dragging it authored a number rather than a position. None of
    /// that is a reason for the author to lose the MARK. The shoulder is where the conic's one
    /// remaining freedom is legible — how hard the curve pulls toward its control point — and it
    /// is the place the author already reaches for, because it is the pick they made to author rho
    /// in the first place (owner, 2026-08-05).
    ///
    /// So it comes back as a derived handle. Nothing has to be added to make it grabbable: the
    /// shoulder lies ON the ink by construction, and a conic's body drag is already a rho drag
    /// ([`drag_curve_through`](Self::drag_curve_through)), so the dot marks a gesture the drawing
    /// already answers rather than introducing one.
    ///
    /// A conic whose points have gone missing, or whose control has collapsed onto its chord, has
    /// no shoulder to report and is skipped.
    #[must_use]
    pub fn conic_shoulders(&self) -> Vec<(EntityId, [f64; 2])> {
        self.conics
            .iter()
            .filter_map(|conic| {
                let at = parametric::sketch::conic_vertex_from_rho(
                    self.point_in_plane(conic.from)?,
                    self.point_in_plane(conic.to)?,
                    self.point_in_plane(conic.control)?,
                    conic.rho.value(),
                )?;
                at.iter()
                    .copied()
                    .all(f64::is_finite)
                    .then_some((conic.id, at))
            })
            .collect()
    }

    /// Whether `curve` is a curve this drawing actually holds.
    ///
    /// The identity is stable but not a guarantee: a pick, a persisted gesture, or a stored
    /// relation can outlive the curve it names, and the answer to "is it still there" is one
    /// lookup per store rather than something a caller should assemble.
    #[must_use]
    pub fn holds_curve(&self, curve: SketchCurve) -> bool {
        match curve {
            SketchCurve::Segment(id) => self.segments.iter().any(|segment| segment.id == id),
            SketchCurve::Arc(id) => self.arcs.iter().any(|arc| arc.id == id),
            SketchCurve::Circle(id) => self.circles.iter().any(|circle| circle.id == id),
            SketchCurve::Bezier(id) => self.beziers.iter().any(|bezier| bezier.id == id),
            SketchCurve::Ellipse(id) => self.ellipses.iter().any(|ellipse| ellipse.id == id),
            SketchCurve::Conic(id) => self.conics.iter().any(|conic| conic.id == id),
            SketchCurve::Spline(id) => self.splines.iter().any(|spline| spline.id == id),
        }
    }

    /// Slide a whole curve by `by`, carrying every point it stands on, and settle around it.
    /// Reports whether the drawing accepted the move.
    ///
    /// A translation, where [`move_curve`](Self::move_curve) is a perpendicular offset. The two
    /// gestures answer different questions, and which one a curve gets is decided by what a body
    /// drag of it could possibly mean:
    ///
    /// - A **boundary** curve bounds a region, so sideways means NEARER OR FURTHER — a rail dragged
    ///   away from its slot's middle is how the author spends the width, and `move_curve` is that.
    /// - A **spline** has no single perpendicular to offset along, so the only motion its aggregate
    ///   has is moving all of it.
    /// - A **construction** curve bounds nothing. There is no region for it to be the near or far
    ///   side of, so an offset says nothing about it, and translating is the whole of what dragging
    ///   its body can mean. This is how a slot's centerline moves the slot: not because a
    ///   centerline is a special kind of curve the drawing knows about, but because the cap centers
    ///   stand on it, and carrying it carries them.
    ///
    /// Which is why this takes the displacement WHOLE and does not project it. Sliding a straight
    /// slot along its own length is a real motion of the shape and the drawing has the freedom for
    /// it; a perpendicular reading throws that component away and the slot only ever moves
    /// sideways (owner, 2026-08-04).
    ///
    /// Nothing about the curve's own shape changes, which is what makes it safe to state as equal
    /// hands: a rigid displacement satisfies every relation that held before it, so the settle has
    /// nothing to trade off and no configuration to split the difference between. What the rest of
    /// the drawing does about it is the constraints' business, not this function's — there is no
    /// walk out to a "whole shape" here, because a coincidence is already the statement that two
    /// points travel together.
    ///
    /// This is a DISPLACEMENT and not an absolute reading, unlike every other sketch drag. The
    /// caller measures it from where the press landed, because "the curve goes where the cursor is"
    /// names no particular place on a curve the author grabbed the middle of.
    pub fn translate_curve(
        &mut self,
        curve: SketchCurve,
        by: [f64; 2],
        context: parametric::EvaluationContext,
    ) -> Result<bool, SketchEvaluationError> {
        // Every point the curve stands on gets a hand, its center included: a center is placed
        // like any other point (ADR 0038), so a translation that left it behind would not be a
        // translation.
        let hands: Vec<_> = self
            .every_point_of(curve)
            .into_iter()
            .filter_map(|point| {
                let stood = self.point_in_plane(point)?;
                Some(Hand::carried(point, [stood[0] + by[0], stood[1] + by[1]]))
            })
            .collect();
        if hands.is_empty() {
            return Ok(false);
        }
        self.drag_or_leave_it_alone(|sketch| {
            // Where they stood, read before they are written down. See `settle_under_the_hands`.
            let was: Vec<(EntityId, [f64; 2])> = hands
                .iter()
                .filter_map(|hand| Some((hand.point, sketch.point_in_plane(hand.point)?)))
                .collect();
            for hand in &hands {
                if let Some(index) = sketch.point_index(hand.point) {
                    sketch.points[index].at = SketchPoint::from_continuous(hand.to[0], hand.to[1]);
                }
            }
            sketch.sync_derived_points();
            sketch.settle_under_the_hands(&hands, &was, context, SnapReach::UNBOUNDED, &mut [])
        })
        .map(|answered| answered.moved)
    }

    /// Every point `curve` is made of: the ones it stands on, and the ones that shape it.
    ///
    /// Wider than [`points_of`](Self::points_of) by exactly the points that steer a curve without
    /// being on it, and two callers need the wider answer for the same underlying reason — a curve
    /// cannot move unless all of them do.
    ///
    /// A TRANSLATION has to name the whole curve or it will leave part of it standing: a higher
    /// curve carries its shape in its points, and a spline's tangent handles are points too, so a
    /// handle left behind would re-aim its tangent by exactly the distance the spline moved, which
    /// is the one thing a rigid translation must not do. A DRAG SCOPE has to name the whole curve
    /// or the solve cannot redraw it, and a spline it cannot redraw is one it will not put in the
    /// problem at all — which silently drops every relation held to that spline.
    fn every_point_of(&self, curve: SketchCurve) -> Vec<EntityId> {
        match curve {
            SketchCurve::Segment(_) | SketchCurve::Arc(_) | SketchCurve::Circle(_) => {
                self.points_of(curve)
            }
            SketchCurve::Bezier(id) => self
                .beziers
                .iter()
                .find(|bezier| bezier.id == id)
                .map(|bezier| bezier.controls.to_vec())
                .unwrap_or_default(),
            SketchCurve::Ellipse(id) => self
                .ellipses
                .iter()
                .find(|ellipse| ellipse.id == id)
                .map(|ellipse| vec![ellipse.center, ellipse.major_endpoint, ellipse.width_point])
                .unwrap_or_default(),
            SketchCurve::Conic(id) => self
                .conics
                .iter()
                .find(|conic| conic.id == id)
                .map(|conic| vec![conic.from, conic.to, conic.control])
                .unwrap_or_default(),
            SketchCurve::Spline(id) => self
                .splines
                .iter()
                .find(|spline| spline.id == id)
                .map(|spline| {
                    spline
                        .points
                        .iter()
                        .copied()
                        .chain(spline.tangents.values().flat_map(|handle| handle.arms()))
                        .collect()
                })
                .unwrap_or_default(),
        }
    }

    /// Run a drag attempt, and put the drawing back exactly as it was unless it stands.
    ///
    /// Every way an attempt can decline — a refusal, an error, a conic left with nothing to shape
    /// — meets ONE rollback here rather than one apiece.
    fn drag_or_leave_it_alone(
        &mut self,
        attempt: impl FnOnce(&mut Self) -> Result<DragAnswer, SketchEvaluationError>,
    ) -> Result<DragAnswer, SketchEvaluationError> {
        let before_points = self.points.clone();
        let before_arcs = self.arcs.clone();
        let before_circles = self.circles.clone();
        let before_conics = self.conics.clone();
        // Any drag can leave a conic with nothing to shape — a hand on its control point, or a
        // settle that walked an endpoint onto it. One check covers them all.
        let result = attempt(self).map(|answered| DragAnswer {
            moved: answered.moved
                && self.every_conic_resolves()
                && self.every_tangent_lever_stands(),
            ..answered
        });
        if matches!(result, Ok(DragAnswer { moved: true, .. })) {
            return result;
        }
        self.points = before_points;
        self.arcs = before_arcs;
        self.circles = before_circles;
        self.conics = before_conics;
        self.sync_derived_points();
        result
    }

    /// The move itself, before [`move_point`](Self::move_point) decides whether to keep it.
    ///
    /// Separate so that every way it can decline — a refusal, an error, a conic left with nothing
    /// to shape — meets ONE rollback rather than one apiece.
    fn point_move_attempt(
        &mut self,
        id: EntityId,
        at: SketchPoint,
        context: parametric::EvaluationContext,
        snap_reach: SnapReach,
        carries: &mut [ArcTurnUnderAGesture],
    ) -> Result<DragAnswer, SketchEvaluationError> {
        // Every point moves the same way now, an arc's center included: ADR 0038 left the
        // drawing with no point whose coordinates are somebody else's arithmetic, so there is no
        // longer a second kind of drag that authors a quantity instead of a position.
        //
        // Read the hand set BEFORE anything moves: everything carried states its own displacement,
        // and that is measured from where the drawing currently stands. Then put the whole set
        // where it is going, so a carried shape starts the settle already standing rather than
        // distorted around the one point that led.
        let before = self.points.clone();
        let hands = self.hands_moving_with(id, at);
        for hand in &hands {
            if let Some(index) = self.point_index(hand.point) {
                self.points[index].at = SketchPoint::from_continuous(hand.to[0], hand.to[1]);
            }
        }
        // The DRAG's own displacement carries the handles, not just the settle's. A solve carries
        // them too, but it measures from here — after the hands have landed — so on a drawing with
        // no standing relation there is no solve at all and the handle would simply be left behind,
        // its offset re-aimed by a gesture that never mentioned it. That is the same re-aiming the
        // solve path guards against, arriving one step earlier.
        self.carry_authored_handles(&before);
        // The tangent arms only. Seating the arc centers here would seat them on the RAW CURSOR,
        // and the cursor is scaffolding — the snap has not had its say yet, and what it lands is
        // what the author asked for.
        //
        // The seat is a projection, so it is lossy in exactly the case that matters. A center off
        // the new bisector is pushed back ALONG the chord, and as an arc's two ends close up that
        // chord shortens until the bisector is nearly parallel to the push: measured on an arc
        // swept to within a chord of 10, three units of cursor error threw the center eleven units,
        // from the origin out to [-10.98, 1.51]. Projecting THAT onto the corrected bisector after
        // the snap cannot get the origin back, so the arc came out at radius 37 instead of 40 and
        // the author watched it deform — "towards the end of the full 360, it tends to deform and
        // the radius won't stay consistent; the center point ends up moving".
        //
        // Left alone, the authored center is already on the bisector once the snap has held the
        // radius, and the seat at the end of the settle moves it not at all.
        self.sync_tangent_arms();
        // The WHOLE drawing as the gesture found it, not only the points under the hand. The
        // pivot a snap measures to is rarely a hand, and by here the hands are written down and the
        // arc centers re-seated on top of them — so a pivot read from the drawing as it now stands
        // is a pivot the gesture has already moved. This is the only record of where it was.
        let was: Vec<(EntityId, [f64; 2])> = before
            .iter()
            .map(|stood| (stood.id, stood.at.in_plane()))
            .collect();
        self.settle_under_the_hands(&hands, &was, context, snap_reach, carries)
    }

    /// Whether the standing constraint system is met by the drawing exactly as it stands, with
    /// nothing moved to make it so. The acceptance test for a derived-point drag, which authors a
    /// quantity rather than a position and so has no freedom left to settle with.
    fn standing_constraints_hold(
        &self,
        context: parametric::EvaluationContext,
    ) -> Result<bool, SketchEvaluationError> {
        let prepared = constraint::prepare(self, &self.constraints, Some(context))
            .map_err(map_prepare_evaluation_error)?;
        let current = prepared.validate_current();
        if let Some(failure) = current.tangent_failure {
            let failure = prepared
                .standing_tangent_failure(failure)
                .map_err(map_prepare_evaluation_error)?;
            return Err(SketchEvaluationError::InvalidTangent {
                constraint: failure.constraint,
                error: failure.error,
            });
        }
        Ok(current.satisfied && current.collapsed.is_none())
    }

    /// Every hand a drag of `id` toward `at` puts on the drawing, the held point's own included.
    ///
    /// Almost always just the one: grabbing a vertex asks the drawing to do whatever it likes so
    /// long as that vertex lands under the cursor, and least motion answers well. The exception is
    /// a handle that names a whole SHAPE rather than a corner of one — a slot's center — where the
    /// author means "move this thing". There, one hand is measurably wrong: the freedoms a slot
    /// keeps on purpose (its width, its radius) are cheaper for least motion to spend than a
    /// translation is, and the pull ends up either reshaping the slot or failing its tangencies
    /// outright. Naming the rest of the spine, each carried by the SAME displacement, says which
    /// motion was meant — and the translated configuration satisfies every standing relation
    /// exactly, because they are all relative.
    ///
    /// The other handles of such a shape get a second hand for the opposite reason: they RESHAPE
    /// it, and a reshape turns about something. Naming the center as a hand that stays put is what
    /// makes it a pivot; without it least motion is free to slide the whole shape a little to meet
    /// the cursor for less, and the author watches the thing they were reshaping wander off.
    ///
    /// This is a drag policy, not a relation. A relation that made the slot rigid would take those
    /// freedoms away for good, and they are the ones the other handles exist to author.
    fn hands_moving_with(&self, id: EntityId, at: SketchPoint) -> Vec<Hand> {
        let mut hands = vec![Hand {
            point: id,
            to: at.in_plane(),
            role: HandRole::Lead,
        }];
        let Some(index) = self.point_index(id) else {
            return hands;
        };
        let was = self.points[index].at.in_plane();
        let now = at.in_plane();
        let delta = [now[0] - was[0], now[1] - was[1]];
        if let Some(pivot) = self.pivot_a_reshape_turns_about(id) {
            if let Some(index) = self.point_index(pivot) {
                hands.push(Hand {
                    point: pivot,
                    to: self.points[index].at.in_plane(),
                    role: HandRole::Pin,
                });
            }
        }
        for carried in self.rest_of_the_shape_held_by(id) {
            let Some(index) = self.point_index(carried) else {
                continue;
            };
            let stood = self.points[index].at.in_plane();
            hands.push(Hand {
                point: carried,
                to: [stood[0] + delta[0], stood[1] + delta[1]],
                role: HandRole::Carried,
            });
        }
        hands
    }

    /// The hands a curve drag puts on the drawing, or nothing if this curve cannot be dragged.
    ///
    /// One signed offset, measured once, applied along each endpoint's OWN outward direction. For
    /// a straight curve those directions agree and the segment slides sideways; for a turning one
    /// they are radial and differ, and applying the same offset to each is what makes the motion a
    /// change of RADIUS rather than a shove. The distinction matters to the solver, not just to the
    /// eye: an equal-offset hand set is one a pure width change satisfies exactly, so the settle
    /// has nothing to trade off. Hands built from a raw cursor displacement would disagree by
    /// bearing, leaving no configuration that meets both and a mushy answer that splits them.
    ///
    /// Only the curves an author can hold as a WHOLE — a segment, an arc. The higher curves carry
    /// their shape in control points, so there is no single offset that means anything for them.
    fn hands_moving_a_curve(&self, curve: SketchCurve, at: [f64; 2]) -> Option<Vec<Hand>> {
        // Below this the direction the offset is measured along stops being meaningful, and the
        // drag would report a wild displacement from a rounding difference.
        const DEGENERATE_SPAN: f64 = 1.0e-9;
        match curve {
            SketchCurve::Segment(id) => {
                let segment = self.segments.iter().find(|segment| segment.id == id)?;
                let tail = self.point_in_plane(segment.from)?;
                let head = self.point_in_plane(segment.to)?;
                let span = [head[0] - tail[0], head[1] - tail[1]];
                let length = span[0].hypot(span[1]);
                if length < DEGENERATE_SPAN {
                    return None;
                }
                let outward = [-span[1] / length, span[0] / length];
                let offset = (at[0] - tail[0]).mul_add(outward[0], (at[1] - tail[1]) * outward[1]);
                let slid = |from: [f64; 2]| {
                    [
                        offset.mul_add(outward[0], from[0]),
                        offset.mul_add(outward[1], from[1]),
                    ]
                };
                Some(vec![
                    Hand::carried(segment.from, slid(tail)),
                    Hand::carried(segment.to, slid(head)),
                ])
            }
            SketchCurve::Arc(id) => {
                let arc = self.arcs.iter().find(|arc| arc.id == id)?;
                let center = self.point_in_plane(arc.center)?;
                let tail = self.point_in_plane(arc.from)?;
                let head = self.point_in_plane(arc.to)?;
                let radius = (tail[0] - center[0]).hypot(tail[1] - center[1]);
                let reach = (at[0] - center[0]).hypot(at[1] - center[1]);
                if radius < DEGENERATE_SPAN || reach < DEGENERATE_SPAN {
                    return None;
                }
                // Both endpoints stand at `radius` by definition of an arc, so scaling each about
                // the center by the same ratio offsets both radially by the same amount.
                let grown = reach / radius;
                let swelled = |from: [f64; 2]| {
                    [
                        grown.mul_add(from[0] - center[0], center[0]),
                        grown.mul_add(from[1] - center[1], center[1]),
                    ]
                };
                Some(vec![
                    Hand::carried(arc.from, swelled(tail)),
                    Hand::carried(arc.to, swelled(head)),
                ])
            }
            SketchCurve::Circle(_)
            | SketchCurve::Bezier(_)
            | SketchCurve::Ellipse(_)
            | SketchCurve::Conic(_)
            | SketchCurve::Spline(_) => None,
        }
    }

    /// The spine points a curve drag has to hold still, each pinned where it already stands.
    ///
    /// # Why a widening needs a second hand at all
    ///
    /// A slot keeps ONE freedom on purpose, its width, and a rail drag is how the author spends
    /// it. But "the width" is not a coordinate the solver can see: what it sees is two rails, two
    /// caps tangent to both, and cap centers standing on the spine. Pull one rail out and that
    /// system has a whole family of answers — move the far rail in, slide the spine, grow the
    /// caps, in any mixture that keeps the tangencies. Least motion picks the cheapest mixture,
    /// which is a bit of each: measured, a 2.0 pull on one rail moved the far rail 0.4 the wrong
    /// way and slid the centerline 0.8. The slot stayed a slot and stopped being the slot the
    /// author drew.
    ///
    /// So the spine is pinned, and that single hand is the whole of the symmetric rule. Nothing
    /// asserts that the rails are equidistant, and nothing needs to: a cap is a circle, its center
    /// is equidistant from its own two ends BY CONSTRUCTION, and that center IS a point of the
    /// spine. Hold the spine and the mirror follows from the tangency web that is already
    /// there. It costs no new relation, and it reads the same for a straight slot and a turning
    /// one — where a `Symmetry` relation could not, since it wants a segment axis and an arc
    /// slot's spine is an arc.
    ///
    /// # What counts as a spine point
    ///
    /// A center a round curve turns about, found by walking out from what the drag already holds.
    /// That is a slot's cap centers and its turning center, and it is not a corner, an endpoint,
    /// or anything in the rest of the drawing. Points the drag is already moving are left alone: a
    /// hand that both moves and holds is not a hand.
    ///
    /// # A spine point standing on the dragged curve is one of those
    ///
    /// The pin says "hold the spine while a RAIL widens", and it only means anything because the
    /// rail and the spine are different curves. Drag the spine itself and the same walk finds the
    /// handles standing on it and pins them where they were — against the very hands carrying the
    /// curve they stand on. Half the hands then say move and half say stay, and least squares does
    /// what it is asked: it splits the difference. Measured on an Overall Slot, a 5.0 pull on the
    /// centerline arrived as 2.25, stretched the slot by 0.8 and grew its half-width by half again.
    ///
    /// A spine point is only free to be pinned if the drag does not already carry it. Standing at one of
    /// the dragged curve's own ends is one way to be carried and is caught by `held`; standing
    /// ANYWHERE on it under a [`Coincident`](ConstraintKind::Coincident) naming that curve is the
    /// other, and is why
    /// this needs to know which curve the hands came from. An Overall Slot is authored by its far
    /// ends, so its centerline runs out past both cap centers and holds them on it that way — the
    /// ends of the curve are not the spine points, and identity alone never sees them.
    fn handles_a_widening_must_hold(&self, curve: SketchCurve, hands: &[Hand]) -> Vec<Hand> {
        let held: Vec<EntityId> = hands.iter().map(|hand| hand.point).collect();
        let carried = |point: EntityId| {
            held.contains(&point)
                || self.constraints.iter().any(|constraint| {
                    matches!(
                        constraint.kind,
                        ConstraintKind::Coincident {
                            point: on,
                            onto: CoincidentTarget::Curve(along),
                        } if on == point && along == curve
                    )
                })
        };
        self.what_a_drag_of_these_can_reach(&held)
            .iter()
            .filter(|point| !carried(**point))
            .filter(|point| self.is_arc_center(**point))
            .filter_map(|point| Some(Hand::pin(*point, self.point_in_plane(*point)?)))
            .collect()
    }

    /// Where the point `id` stands, in plane coordinates.
    fn point_in_plane(&self, id: EntityId) -> Option<[f64; 2]> {
        self.point_index(id)
            .map(|index| self.points[index].at.in_plane())
    }

    /// Every OTHER point a center carries when it is dragged — its RIGID SET.
    ///
    /// A center is rigid with the curves it centers. That is the whole rule, and it is the rule
    /// because a center is not a corner: moving a corner is a statement about that corner, while
    /// moving the place a curve turns about is a statement about the curve. Fusion says the same
    /// thing of the simplest case — "if you drag the center point you will change the position of
    /// the arc like in a circle" — and D-Cubed, the solver underneath it, has a name for the
    /// general case: a RIGID SET, "collections of geometries which 2D DCM solves as if they are
    /// constrained relative to each other", declared rather than inferred from the numbers.
    ///
    /// How far the set reaches is the one question left, and the drawing answers it. Where the
    /// center names a WHOLE shape — a slot's hub, about which both rails and the spine turn — the
    /// set is the shape, walked out through the relations holding it together. Where it names one
    /// curve of a bigger thing — a slot's cap, a lone arc — the set is that curve and its own
    /// points, and no further: a cap center carries its two corners so the cap sweeps as one piece
    /// instead of running ahead of the corners it is supposed to be the middle of, and a lone arc
    /// carries the two ends whose sweep it defines.
    ///
    /// Carrying the whole set is what makes the answer exact rather than merely close: displace all
    /// of it at once and the standing system is satisfied to begin with, so the settle has nothing
    /// left to trade off and no chance to spend the move on a freedom the author did not offer. A
    /// shape held to geometry OUTSIDE it moves alone and the drag is refused, which is the honest
    /// outcome — translating a slot is not permission to drag whatever it was attached to.
    fn rest_of_the_shape_held_by(&self, held: EntityId) -> Vec<EntityId> {
        let seeds = self.curves_centered_on(held);
        if seeds.is_empty() {
            return Vec::new();
        }
        let reach = if self.is_a_shape_hub(held) {
            self.shape_holding(seeds)
        } else {
            seeds
        };
        let mut carried: Vec<EntityId> = Vec::new();
        let mut pending: Vec<EntityId> = reach
            .into_iter()
            .flat_map(|curve| self.points_of(curve))
            .collect();
        while let Some(point) = pending.pop() {
            if point == held || carried.contains(&point) {
                continue;
            }
            carried.push(point);
            // A handle is only reachable through the center it is pinned to, so coincidence is
            // part of the walk rather than a pass over the result.
            pending.extend(self.coincident_partners(point));
        }
        carried
    }

    /// The point a drag of `held` should turn about, or nothing if the drag is not a reshape.
    ///
    /// The mirror image of the translate policy, and it asks the same two questions in the other
    /// order: what `held` belongs to must be ONE end of a shape — not the whole of it — and that
    /// shape must have a center of its own. That center is the pivot.
    ///
    /// A point belongs to a shape two ways, and both count. It can be the center ONE curve turns
    /// about, which is an end cap. Or it can be a corner the curves merely END at, which is the
    /// same gesture arriving on the boundary instead of the spine — a slot's outer corner pulled
    /// round its rail is a reshape by every reading, and without the second way it named no pivot,
    /// so the hub it was supposed to turn about drifted along behind the cursor.
    ///
    /// **A shape need not have a hub for a drag of it to be a reshape**, and asking for one is what
    /// left a STRAIGHT slot unable to lengthen. A hub is DECLARED by the tool that draws the shape
    /// ([`PointHandle::ShapeHub`]), and a straight slot has no one place its parts turn about for a
    /// tool to declare — so no point on it qualified, no pivot was found, and with no pivot the
    /// gesture read as the drawing being moved. The author pulled an end cap and the whole slot
    /// followed. The blind spot is not the slot's: it belongs to every shape drawn as straight runs
    /// between arc caps.
    ///
    /// So the hub is preferred and no longer required. Where the shape has one the answer is
    /// unchanged, which is what keeps a turning slot behaving exactly as it did; where it has none,
    /// the pivot is the FARTHEST other center in the shape, which on anything cap-ended is the
    /// opposite cap — the end a reshape visibly turns about. Farthest rather than first so the
    /// answer is a property of the drawing rather than of the order its curves were stored in.
    fn pivot_a_reshape_turns_about(&self, held: EntityId) -> Option<EntityId> {
        let centered = self.curves_centered_on(held);
        if self.is_a_shape_hub(held) {
            return None;
        }
        let seeds = if centered.is_empty() {
            self.curves_ending_at(held)
        } else {
            centered
        };
        if seeds.is_empty() {
            return None;
        }
        let mut candidates: Vec<EntityId> = Vec::new();
        for point in self
            .shape_holding(seeds)
            .into_iter()
            .flat_map(|curve| self.points_of(curve))
        {
            if point != held && self.is_arc_center(point) && !candidates.contains(&point) {
                candidates.push(point);
            }
        }
        if let Some(hub) = candidates.iter().find(|point| self.is_a_shape_hub(**point)) {
            return Some(*hub);
        }
        let stood = self.point_in_plane(held)?;
        candidates.into_iter().max_by(|first, second| {
            let reach = |point: EntityId| {
                self.point_in_plane(point).map_or(f64::NEG_INFINITY, |at| {
                    (at[0] - stood[0]).hypot(at[1] - stood[1])
                })
            };
            reach(*first).total_cmp(&reach(*second))
        })
    }

    /// Every curve this point is an END of, its center excluded.
    fn curves_ending_at(&self, point: EntityId) -> Vec<SketchCurve> {
        self.arcs
            .iter()
            .filter(|arc| arc.from == point || arc.to == point)
            .map(|arc| SketchCurve::Arc(arc.id))
            .chain(
                self.segments
                    .iter()
                    .filter(|segment| segment.from == point || segment.to == point)
                    .map(|segment| SketchCurve::Segment(segment.id)),
            )
            .collect()
    }

    /// Whether this point was DECLARED the handle of a whole shape — see [`PointHandle::ShapeHub`].
    fn is_a_shape_hub(&self, point: EntityId) -> bool {
        self.point_index(point)
            .is_some_and(|index| self.points[index].handle == PointHandle::ShapeHub)
    }

    /// Every curve reachable from these by the relations that make curves behave as one shape.
    fn shape_holding(&self, seeds: Vec<SketchCurve>) -> Vec<SketchCurve> {
        let mut shape: Vec<SketchCurve> = Vec::new();
        let mut frontier = seeds;
        while let Some(curve) = frontier.pop() {
            if shape.contains(&curve) {
                continue;
            }
            shape.push(curve);
            frontier.extend(self.curves_held_to(curve));
        }
        shape
    }

    /// Every point a curve stands on, its center included where it has one.
    ///
    /// Public for the overlay: a curve the author is touching shows the points it stands on, even
    /// the ones [`point_draws_at_rest`](Self::point_draws_at_rest) keeps quiet.
    ///
    /// A spline answers with the points it is drawn THROUGH and not with their tangent arms, which
    /// is the same line the other kinds draw: an arm steers the curve from beside it rather than
    /// standing on it, and nothing here wants a handle revealed by touching the curve it shapes.
    /// A caller that needs the whole curve — every point that has to move for it to move — wants
    /// [`every_point_of`](Self::every_point_of) instead. The remaining higher curves answer empty
    /// because the solver models no place along them, so nothing walks through one.
    pub fn points_of(&self, curve: SketchCurve) -> Vec<EntityId> {
        match curve {
            SketchCurve::Arc(id) => self
                .arcs
                .iter()
                .find(|arc| arc.id == id)
                .map(|arc| vec![arc.from, arc.to, arc.center])
                .unwrap_or_default(),
            SketchCurve::Segment(id) => self
                .segments
                .iter()
                .find(|segment| segment.id == id)
                .map(|segment| vec![segment.from, segment.to])
                .unwrap_or_default(),
            SketchCurve::Circle(id) => self
                .circles
                .iter()
                .find(|circle| circle.id == id)
                .map(|circle| vec![circle.center])
                .unwrap_or_default(),
            SketchCurve::Spline(id) => self
                .splines
                .iter()
                .find(|spline| spline.id == id)
                .map(|spline| spline.points.clone())
                .unwrap_or_default(),
            SketchCurve::Bezier(_) | SketchCurve::Ellipse(_) | SketchCurve::Conic(_) => Vec::new(),
        }
    }

    /// Every point standing coincident with this one by an authored relation.
    fn coincident_partners(&self, id: EntityId) -> Vec<EntityId> {
        self.constraints
            .iter()
            .filter_map(|constraint| match constraint.kind {
                ConstraintKind::Coincident {
                    point,
                    onto: CoincidentTarget::Point(other),
                } if point == id => Some(other),
                ConstraintKind::Coincident {
                    point,
                    onto: CoincidentTarget::Point(other),
                } if other == id => Some(point),
                _ => None,
            })
            .collect()
    }

    /// Every circular curve turning about this point.
    fn curves_centered_on(&self, center: EntityId) -> Vec<SketchCurve> {
        self.arcs
            .iter()
            .filter(|arc| arc.center == center)
            .map(|arc| SketchCurve::Arc(arc.id))
            .chain(
                self.circles
                    .iter()
                    .filter(|circle| circle.center == center)
                    .map(|circle| SketchCurve::Circle(circle.id)),
            )
            .collect()
    }

    /// Every curve joined to this one by a relation that makes the two behave as one shape.
    fn curves_held_to(&self, curve: SketchCurve) -> Vec<SketchCurve> {
        self.constraints
            .iter()
            .filter_map(|constraint| {
                let (ConstraintKind::Tangent { first, second, .. }
                | ConstraintKind::Concentric { first, second }) = constraint.kind
                else {
                    return None;
                };
                if first == curve {
                    Some(second)
                } else if second == curve {
                    Some(first)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Every point a drag of these could possibly move, the held ones included.
    ///
    /// Closed under two rules, applied until nothing new arrives. A CURVE standing on a reached
    /// point brings the rest of ITSELF — its other ends, and for a spline the tangent arms as well
    /// — because the solver holds every drawn edge's span, and half a curve in the problem is a
    /// different problem than the whole one. A RELATION naming anything reached brings everything
    /// else it names, which is the ordinary sense in which two shapes are one shape.
    ///
    /// Anything left out is unreachable in the strict sense — no relation and no edge connects it
    /// — so no solve could have moved it and leaving it out changes nothing but the price.
    fn what_a_drag_of_these_can_reach(&self, held: &[EntityId]) -> Vec<EntityId> {
        let mut reached: Vec<EntityId> = held.to_vec();
        loop {
            let known = reached.len();
            for curve in self.curves_standing_on_any(&reached) {
                for point in self.every_point_of(curve) {
                    if !reached.contains(&point) {
                        reached.push(point);
                    }
                }
            }
            for constraint in &self.constraints {
                let named = self.points_named_by(constraint);
                if !named.iter().any(|point| reached.contains(point)) {
                    continue;
                }
                for point in named {
                    if !reached.contains(&point) {
                        reached.push(point);
                    }
                }
            }
            if reached.len() == known {
                return reached;
            }
        }
    }

    /// Every curve any of these points belongs to.
    ///
    /// Asked with [`every_point_of`](Self::every_point_of), so a hand on a tangent ARM finds the
    /// spline that arm steers. A hand there reshapes the curve as surely as a hand on a fit point
    /// does, and anything held to the curve has to be in the same problem to hear about it.
    ///
    /// The stores asked are the ones a walk can cross. The remaining higher curves are left out
    /// because the solver names no place along them, so nothing is ever held to one and reaching
    /// one would widen the problem to say nothing.
    fn curves_standing_on_any(&self, points: &[EntityId]) -> Vec<SketchCurve> {
        let stands = |curve: SketchCurve| {
            self.every_point_of(curve)
                .iter()
                .any(|point| points.contains(point))
        };
        self.segments
            .iter()
            .map(|segment| SketchCurve::Segment(segment.id))
            .chain(self.arcs.iter().map(|arc| SketchCurve::Arc(arc.id)))
            .chain(
                self.circles
                    .iter()
                    .map(|circle| SketchCurve::Circle(circle.id)),
            )
            .chain(
                self.splines
                    .iter()
                    .map(|spline| SketchCurve::Spline(spline.id)),
            )
            .filter(|curve| stands(*curve))
            .collect()
    }

    /// Every point a relation reaches, whether it names the point itself or the curve it stands on.
    fn points_named_by(&self, constraint: &Constraint) -> Vec<EntityId> {
        constraint
            .kind
            .named_entities()
            .into_iter()
            .flat_map(|entity| self.points_standing_for(entity))
            .chain(self.span_points_derived_by(constraint.kind))
            .collect()
    }

    /// The points a relation places WITHOUT naming them, because it derives them from a spline.
    ///
    /// Curvature stores its joint and the curve, and reads the rest of the span off the spline at
    /// mapping time, so that inserting a point beside the joint cannot leave it reading a span that
    /// is gone. The cost of that is that `points()` alone understates what the relation reaches,
    /// and two callers care: the drag scope, which would otherwise leave the levers out of the
    /// problem entirely, and [`carry_authored_handles`](Self::carry_authored_handles), which would
    /// see an unclaimed handle and drag it back off by its anchor's displacement. Both ask the same
    /// derivation the residual mapping asks, so all three agree by construction.
    fn span_points_derived_by(&self, kind: ConstraintKind) -> Vec<EntityId> {
        let ConstraintKind::Curvature { joint, .. } = kind else {
            return Vec::new();
        };
        constraint::curvature_span_of(&self.splines, joint)
            .map(|(joint_arm, neighbor, neighbor_arm, _)| vec![joint_arm, neighbor, neighbor_arm])
            .unwrap_or_default()
    }

    /// The points an entity id amounts to: itself if it is a point, the ones its curve stands on
    /// otherwise. Empty for anything the solver does not model, which is the same geometry a
    /// relation naming it would be dropped over.
    fn points_standing_for(&self, entity: EntityId) -> Vec<EntityId> {
        if self.point_index(entity).is_some() {
            return vec![entity];
        }
        self.curve_named(entity)
            .map(|curve| self.points_of(curve))
            .unwrap_or_default()
    }

    /// The curve this id names, if any curve store holds it.
    ///
    /// Asks every store, higher curves included. [`points_of`](Self::points_of) answers for a
    /// spline and empty for the rest of them, so a caller that walks from a curve to its points
    /// reaches exactly the curves the solver can hold something to; a caller that only wants to
    /// know WHAT the id is gets a true answer for all of them instead of `None`.
    fn curve_named(&self, entity: EntityId) -> Option<SketchCurve> {
        if self.segments.iter().any(|segment| segment.id == entity) {
            Some(SketchCurve::Segment(entity))
        } else if self.arcs.iter().any(|arc| arc.id == entity) {
            Some(SketchCurve::Arc(entity))
        } else if self.circles.iter().any(|circle| circle.id == entity) {
            Some(SketchCurve::Circle(entity))
        } else if self.beziers.iter().any(|bezier| bezier.id == entity) {
            Some(SketchCurve::Bezier(entity))
        } else if self.ellipses.iter().any(|ellipse| ellipse.id == entity) {
            Some(SketchCurve::Ellipse(entity))
        } else if self.conics.iter().any(|conic| conic.id == entity) {
            Some(SketchCurve::Conic(entity))
        } else if self.splines.iter().any(|spline| spline.id == entity) {
            Some(SketchCurve::Spline(entity))
        } else {
            None
        }
    }

    /// Whether a relation stands ENTIRELY within these points — the ones a scoped problem can
    /// carry. A relation reaching outside would name geometry the problem does not hold.
    fn constraint_stands_within(&self, constraint: &Constraint, points: &[EntityId]) -> bool {
        let named = self.points_named_by(constraint);
        !named.is_empty() && named.iter().all(|point| points.contains(point))
    }

    /// Re-solve the standing constraints with the gesture's hands pulling, writing the result back
    /// only if the standing residuals are met. Reports whether they were.
    ///
    /// The assertions hold DURING the gesture, not merely at the moment they were made.
    ///
    /// **A hand is a PULL, not a demand — two stages.** The drag joins the system as one more
    /// least-squares row per hand and the solve trades them off against everything standing; then
    /// the hands let go and the standing system alone is re-solved from that answer, which restores
    /// it exactly while moving as little as it can. The grabbed point therefore lands at the nearest
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
    /// `was` names where the hands stood before the caller wrote them down, so the rigidity
    /// preference can be measured on the shape it is trying to keep rather than on the one the
    /// hand has already bent. A caller that has not moved anything yet passes nothing, and every
    /// hand it did not name is read where it currently stands — which, for a caller that moved
    /// nothing, is where it stood. What goes down is therefore always the WHOLE hand set, and the
    /// kernel leans on that: knowing what the hand has hold of is how it tells a corner being
    /// pulled from a whole rail being moved.
    /// Re-solve this frame from a copy of the drawing whose crossing arcs are drawn the other way
    /// round, and take the answer only if it stands.
    ///
    /// One validator reads which way round an arc is drawn: a tangent CONTACT stands on the arc's
    /// DRAWN piece or it does not, and the two readings of an arc whose ends have just crossed are
    /// different pieces. Measured on a segment tangent to an arc's interior, wound through a seam,
    /// the same move is refused as `OutsideFirstDomain` under the reading the last frame left and
    /// accepted under the one this frame is about to be given — and a refusal ends the gesture
    /// outright.
    ///
    /// Re-solved from a relabelled COPY rather than patched in place. The whole frame then comes
    /// from one authority: a prepared problem whose arc record says one order while its sibling
    /// derived state says another is two authorities inside a single frame, which is the disease
    /// this drawing keeps paying for. The solve is order-indifferent — its rows are radii — so the
    /// second answer is the first one within tolerance, which `first` is carried here to assert
    /// rather than assume. The cost is one redundant solve on the one or two frames of a gesture
    /// that cross at all.
    ///
    /// Nothing is written until it stands. A frame the relabelled drawing refuses or stands leaves
    /// this one untouched, carry included, which is what keeps the carry unwrapping over WRITTEN
    /// frames only.
    #[allow(clippy::too_many_arguments)]
    // An arc that was drawing a circle before the frame has to still be drawing one after it.
    //
    // Now that the center stands where the gesture pinned it, a radius is free to go to nothing:
    // pull an end all the way onto the center and the rows are perfectly satisfied by a circle
    // of radius zero, which stacks all three points on one place and leaves no arc for the
    // author to pull back out of. Nothing downstream catches it either: at the collapse the
    // reading answered a radius of 4.4e-11 and a sweep of 180 degrees, which is a circle in
    // every arithmetic sense and a vanished shape on the screen.
    //
    // So the question is asked in the vocabulary that already answers it. An end within
    // [`STACKED_DOT_TOLERANCE`] of the center IS the center, and an arc turning about a place
    // it stands on is not an arc.
    //
    // Refused rather than clamped, and refused whole. A frame the drawing cannot hold leaves
    // the drawing where it stood, which is what every other unanswerable drag does, and what
    // the author sees is an arc that stops at its own limit instead of one that vanishes.
    //
    // Asked of where the drawing STOOD, not of where it stands. By here the caller has already
    // written the raw hands in, so an arc the frame is about to destroy is destroyed already —
    // measured on the crossing frame, the arc excluded itself from its own guard and the
    // collapse went through. `was` is the only record of the drawing the gesture found, which
    // is what the question is about. An arc that was already degenerate before the hand
    // touched it is left out on purpose: it is the one drag that could repair it.
    fn arcs_that_were_still_drawing_circles(
        &self,
        reached: Vec<EntityId>,
        was: &[(EntityId, [f64; 2])],
    ) -> Vec<EntityId> {
        let stood_at = |id: EntityId| {
            was.iter()
                .find(|(named, _)| *named == id)
                .map(|(_, at)| *at)
                .or_else(|| self.point_in_plane(id))
        };
        reached
            .into_iter()
            .filter(|id| {
                self.arcs
                    .iter()
                    .find(|arc| arc.id == *id)
                    .and_then(|arc| {
                        Some(three_points_draw_a_circle(
                            stood_at(arc.from)?,
                            stood_at(arc.to)?,
                            stood_at(arc.center)?,
                        ))
                    })
                    .unwrap_or(false)
            })
            .collect()
    }

    fn settle_again_the_other_way_round(
        &mut self,
        crossing: &[EntityId],
        first: Option<constraint::ApplyPlan>,
        hands: &[Hand],
        was: &[(EntityId, [f64; 2])],
        context: parametric::EvaluationContext,
        snap_reach: SnapReach,
        carries: &mut [ArcTurnUnderAGesture],
    ) -> Result<DragAnswer, SketchEvaluationError> {
        let mut relabelled = self.clone();
        for id in crossing {
            relabelled.reverse_arc(*id);
        }
        let answered =
            relabelled.settle_under_the_hands(hands, was, context, snap_reach, &mut [])?;
        if answered.moved {
            debug_assert!(
                first.is_none_or(|first| first.placed().iter().all(|before| {
                    relabelled.point_in_plane(before.id).is_none_or(|after| {
                        let stood = before.at.in_plane();
                        (after[0] - stood[0]).hypot(after[1] - stood[1]) < 1.0e-6
                    })
                })),
                "re-solving a relabelled arc moved the drawing, so order is an input to the solve"
            );
            *self = relabelled;
            for carry in carries.iter_mut() {
                carry.commit(self);
            }
        }
        Ok(answered)
    }

    fn settle_under_the_hands(
        &mut self,
        hands: &[Hand],
        was: &[(EntityId, [f64; 2])],
        context: parametric::EvaluationContext,
        snap_reach: SnapReach,
        carries: &mut [ArcTurnUnderAGesture],
    ) -> Result<DragAnswer, SketchEvaluationError> {
        // Everything the caller recorded, not just the hands.
        //
        // A snap measures its quantity to a PIVOT — an arc's center, a segment's far end — and the
        // pivot is usually not a hand. Narrowed to the hands, the kernel had to fall back to where
        // the pivot stands NOW, and by then the caller has written the raw cursor into the drawing
        // and re-seated the arc centers on top of it. So the radius was measured to a center the
        // gesture had already dragged: on a bare arc pulled two and a half units, the ghost
        // reported a circle of 38.29 about [1.74, -1.39] while the arc itself settled at 39.87
        // about the origin, and the author saw a ghost that did not lie on the shape it named.
        //
        // Hands the caller did not record are read where they stand, which for a caller that has
        // moved nothing is where they stood.
        let mut stood: Vec<(EntityId, [f64; 2])> = was.to_vec();
        for hand in hands {
            if !stood.iter().any(|(named, _)| *named == hand.point) {
                if let Some(at) = self.point_in_plane(hand.point) {
                    stood.push((hand.point, at));
                }
            }
        }
        let was = stood;
        // Only the part of the drawing the hands can reach takes part. What the rest of the plane
        // holds cannot change the answer, but it does change what the answer COSTS: the kernel
        // prices a solve by how many free coordinates and drawn edges it carries, and its dense
        // algebra over them grows faster than the drawing does. Measured on eight unrelated arc
        // slots, one drag cost 177ms whole-drawing against 1ms for the slot actually held.
        let held: Vec<EntityId> = hands.iter().map(|hand| hand.point).collect();
        let reach = self.what_a_drag_of_these_can_reach(&held);
        let standing: Vec<Constraint> = self
            .constraints
            .iter()
            .filter(|constraint| self.constraint_stands_within(constraint, &reach))
            .copied()
            .collect();
        // Nothing standing AND nothing shaped means nothing to trade the pull off against, so every
        // hand is reachable exactly and the hands ARE the answer. Returning here without writing
        // them would drop the gesture on the floor: measured, a bare arc dragged sideways did not
        // move at all, because no relation touched it and so no solve ran to carry the pull.
        //
        // An ARC is the exception, and it is why the second half of the question is asked at all.
        // Its three points are not three free places: the two ends stand one radius from the
        // center, and the kernel carries that as rows whether or not the author ever asserted
        // anything. Write the hands through and those rows go unheard — the center the gesture
        // pinned holds for exactly one statement before the seat below projects it onto the
        // bisector of the chord the drag just moved, walking a center from [0,40] to
        // [-19.23,36.15] and taking the whole arc with it. Through the solver the same drag is
        // three points, one column and two rows, and least-norm answers it the way the author
        // reads it: the center stands, the dragged end lands under the cursor, and the far end
        // slides out along its own ray to the radius that names.
        //
        // Nothing else needs the detour. A segment holds no shape of its own, a circle keeps its
        // radius as an authored value, and a spline's arms are a mirror restored by the seat.
        let arcs_reached: Vec<EntityId> = self
            .curves_standing_on_any(&reach)
            .into_iter()
            .filter_map(|curve| match curve {
                SketchCurve::Arc(id) => Some(id),
                _ => None,
            })
            .collect();
        if standing.is_empty() && arcs_reached.is_empty() {
            // A snap still applies. It is geometry, not a relation: a bare arc's end stands a
            // radius from its own center whether or not anything is asserted about it, and the
            // author asked for exactly this — "the circle ghost and snapping should apply to any
            // arc-like endpoint". Before it, the one drawing simple enough to skip the solve was
            // also the one where an arc end followed the cursor freely.
            let snapped = constraint::prepare_scoped(self, &standing, Some(context), Some(&reach))
                .ok()
                .and_then(|prepared| {
                    prepared
                        .holding_a_snap_within(snap_reach)
                        .snap_the_hands(hands, &was)
                });
            let (landing, kept) = match snapped {
                Some((onto, kept)) => (onto, Some(kept)),
                None => (hands.to_vec(), None),
            };
            for hand in &landing {
                if let Some(index) = self.point_index(hand.point) {
                    self.points[index].at = SketchPoint::from_continuous(hand.to[0], hand.to[1]);
                }
            }
            self.sync_derived_points();
            for carry in carries.iter_mut() {
                carry.commit(self);
            }
            return Ok(DragAnswer { moved: true, kept });
        }
        let prepared = constraint::prepare_scoped(self, &standing, Some(context), Some(&reach))
            .map_err(map_prepare_evaluation_error)?
            .holding_a_snap_within(snap_reach);
        let (settled, accepted) = match prepared.drag_together(hands, &was) {
            Ok(parametric::sketch::DragOutcome::Accepted(settled)) => (settled, true),
            Ok(parametric::sketch::DragOutcome::Rejected(settled)) => (settled, false),
            Err(_) => return Ok(DragAnswer::stood(false)),
        };
        // Which way round each arc is DRAWN is settled BEFORE the frame is validated, because one
        // validator reads it — see [`settle_again_the_other_way_round`](Self::settle_again_the_other_way_round).
        let crossing: Vec<EntityId> = carries
            .iter()
            .filter_map(|carry| {
                carry.crossing_under(&self.arcs, &|id| {
                    prepared
                        .point(id)
                        .and_then(|point| settled.solution.position(point))
                        .or_else(|| self.point_in_plane(id))
                })
            })
            .collect();
        if !crossing.is_empty() {
            let first = prepared
                .plan_apply(&self.points, &self.circles, &settled.solution)
                .ok();
            return self.settle_again_the_other_way_round(
                &crossing, first, hands, &was, context, snap_reach, carries,
            );
        }
        validate_prepared_tangent_contacts(&prepared, &settled.solution)?;
        if !accepted {
            return Ok(DragAnswer::stood(false));
        }
        let plan = prepared
            .plan_apply(&self.points, &self.circles, &settled.solution)
            .map_err(|_| SketchEvaluationError::ScalarWritebackFailed)?;
        let drawn = self.arcs_that_were_still_drawing_circles(arcs_reached, &was);
        let stood = (self.points.clone(), self.circles.clone());
        plan.apply(self);
        self.carry_authored_handles(&stood.0);
        self.sync_derived_points();
        if !drawn.iter().all(|id| self.arc_draws_a_circle(*id)) {
            self.points = stood.0;
            self.circles = stood.1;
            return Ok(DragAnswer::stood(false));
        }
        // The frame stands, so the carry comes up to it. Last, and only here: every path above
        // that leaves the drawing where it was leaves the carry there too.
        for carry in carries.iter_mut() {
            carry.commit(self);
        }
        Ok(DragAnswer {
            moved: true,
            kept: settled.kept,
        })
    }

    /// Re-solve a conic's rho from a body drag — the post-commit half of the authoring gesture's
    /// last step, and the only way rho is re-authored now that ADR 0038 has taken the shoulder
    /// point away.
    ///
    /// The curve is captive to the track from the chord midpoint out to the control point, and
    /// where it crosses that track IS rho: back at the chord the conic flattens toward its own
    /// straight line, out at the control point it sharpens toward a corner. So the drag projects
    /// onto the track and reads rho off it, the same one definition
    /// ([`parametric::sketch::conic_rho_from_shoulder`]) the gesture uses, clamped short of both
    /// degenerate ends. A conic whose defining points have gone missing is left alone.
    ///
    /// Measured as a DISPLACEMENT from where the author pressed, not as an absolute reading of
    /// where the cursor is. The two agree when the press landed on the shoulder, and they disagree
    /// everywhere else: a press near an end projects to a rho far from the one standing, so an
    /// absolute reading snapped the curve to a new shape before the cursor had moved at all. The
    /// shoulder is a mark on the curve and the rest of the curve is a place to reach it from —
    /// either way, the gesture starts by changing nothing.
    fn reshape_conic_toward(&mut self, conic_index: usize, grabbed: [f64; 2], target: [f64; 2]) {
        let conic = self.conics[conic_index];
        let position = |id| {
            self.points
                .iter()
                .find(|point| point.id == id)
                .map(|point| point.at.in_plane())
        };
        let (Some(from), Some(to), Some(control)) = (
            position(conic.from),
            position(conic.to),
            position(conic.control),
        ) else {
            return;
        };
        let Some(stood) =
            parametric::sketch::conic_vertex_from_rho(from, to, control, conic.rho.value())
        else {
            return;
        };
        let asked = [
            stood[0] + target[0] - grabbed[0],
            stood[1] + target[1] - grabbed[1],
        ];
        let Some(rho) = parametric::sketch::conic_rho_from_shoulder(from, to, control, asked)
        else {
            return;
        };
        if let Ok(rho) = parametric::ResolvedScalar::try_from_f64(rho) {
            self.conics[conic_index].rho = rho;
        }
    }

    /// Delete a point by id and every segment/arc incident to it. The edges' other endpoints
    /// survive as free points. No dangling reference can result: relations do not keep geometry
    /// alive, so their own liveness cascade follows after every geometry cascade.
    /// Deleting an arc's CENTER deletes that arc: the center is the arc's own derived
    /// geometry, so there is no arc left for it to be the center of.
    pub fn delete_point_cascade(&mut self, id: EntityId) {
        // A tangent handle is furniture the curve comes with, not a point the author placed, so it
        // is not theirs to remove — the same standing a control-point spline's frame has, asserted
        // here because a handle IS a real point and would otherwise delete like one.
        if self.is_tangent_handle(id) {
            return;
        }
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
            conic.from != id && conic.to != id && conic.control != id
        });
        self.heal_splines_without(id);
        self.points.retain(|point| point.id != id);
        self.prune_orphan_centers();
        self.drop_dangling_patterns();
        self.drop_dangling_constraints();
    }

    /// Drop `id` out of every spline that names it, SIMPLIFYING the spline instead of deleting it.
    ///
    /// Every other curve is a fixed arity — an arc is three points, a conic is four — so losing one
    /// of them leaves nothing the curve could be, and the cascade deletes it. A spline is the one
    /// curve whose arity is the author's: it is however many points they placed. Removing one is
    /// therefore an edit to the curve, not the end of it. The spline heals down through the degrees
    /// its remaining frame supports — cubic, quadratic, the straight line between two ends — and
    /// only dies when it falls under [`SplineKind::fewest_points`].
    ///
    /// A control-point spline is minted with Real ends and a Construction interior, so when the
    /// deleted point WAS an end the point behind it inherits the job and is promoted. Roles are
    /// only ever promoted here, never demoted: a point that has become interior may still be an
    /// endpoint of some other curve, and Construction on a point is a lifetime — demoting one would
    /// hand it to [`prune_orphan_centers`](Self::prune_orphan_centers) to sweep.
    fn heal_splines_without(&mut self, id: EntityId) {
        let mut healed = Vec::with_capacity(self.splines.len());
        let mut promote = Vec::new();
        for spline in &*self.splines {
            if !spline.points.contains(&id) {
                healed.push(spline.clone());
                continue;
            }
            let mut spline = spline.clone();
            spline.points.retain(|point| *point != id);
            // A fit point that goes takes its handle with it, wherever the spline lands below.
            spline.tangents.remove(&id);
            if spline.points.len() < spline.kind.fewest_points(spline.closed) {
                continue;
            }
            if spline.kind == SplineKind::ControlPoint {
                promote.extend(spline.points.first().copied());
                promote.extend(spline.points.last().copied());
            }
            healed.push(spline);
        }
        self.splines = healed.into_boxed_slice();
        for point in &mut self.points {
            if promote.contains(&point.id) {
                point.lifetime = PointLifetime::Freestanding;
            }
        }
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
                .any(|conic| [conic.from, conic.to, conic.control].contains(&id))
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
    pub fn add_constraint(
        &mut self,
        kind: ConstraintKind,
        context: parametric::EvaluationContext,
    ) -> Result<EntityId, ConstraintRefusal> {
        self.add_constraint_anchored(kind, None, context)
    }

    /// [`add_constraint`](Self::add_constraint) with the place the author dropped the annotation.
    ///
    /// The anchor rides along rather than being asserted: it reaches the stored constraint and
    /// nothing else, so a refusal loses it with everything else the attempt built.
    ///
    /// # Errors
    ///
    /// Every refusal [`add_constraint`](Self::add_constraint) makes, for the same reasons.
    #[allow(clippy::too_many_lines)]
    pub fn add_constraint_anchored(
        &mut self,
        kind: ConstraintKind,
        anchor: Option<[f64; 2]>,
        context: parametric::EvaluationContext,
    ) -> Result<EntityId, ConstraintRefusal> {
        let kind = kind.normalized();
        self.check_names_live_geometry(kind, context)?;
        self.check_names_no_mirrored_arm(kind)?;
        self.check_is_not_already_asserted(kind)?;
        let prepared = constraint::prepare_expecting(self, &self.constraints, Some(context), kind)
            .map_err(|error| match error {
                constraint::PrepareError::MissingEvaluationContext => {
                    ConstraintRefusal::MissingEvaluationContext
                }
                constraint::PrepareError::InvalidDocumentGeometry
                | constraint::PrepareError::InvalidLocalProblem(_) => ConstraintRefusal::Impossible,
            })?;
        let trial = prepared.trial_add(kind).map_err(|error| match error {
            constraint::TrialMapError::UnmappedGeometry
            | constraint::TrialMapError::Request(
                parametric::sketch::RequestError::UnknownPoint
                | parametric::sketch::RequestError::InvalidRelation(
                    parametric::sketch::BuildError::UnknownPoint
                    | parametric::sketch::BuildError::UnknownSegment
                    | parametric::sketch::BuildError::UnknownArc
                    | parametric::sketch::BuildError::UnknownSpline
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
        let Ok(plan) = prepared.plan_apply(&self.points, &self.circles, &settled.solution) else {
            return Err(ConstraintRefusal::Impossible);
        };
        let id = self.alloc_id();
        self.constraints.push(Constraint {
            id,
            kind,
            redundant,
            anchor,
        });
        let before = self.points.clone();
        plan.apply(self);
        self.carry_authored_handles(&before);
        self.sync_derived_points();
        Ok(id)
    }

    /// Move a stored annotation to `anchor`, reporting whether there was one to move.
    ///
    /// **The claim does not move with it.** Which quantity a dimension states was settled by the
    /// gesture that authored it and is written down in the [`Dimension`] member — the width of a
    /// diagonal run, the supplement rather than the angle. Re-reading the drop point here would
    /// let a hand that only wanted the number out of the way restate it as a different number,
    /// which is a thing the author must ASK for by dimensioning again.
    ///
    /// What the anchor still decides is what it always decided: which side of the geometry the
    /// drawing lies on, and for an angle which of the two corners that state the same size the
    /// arc occupies. Those are two views of one claim, so a drag can cross between them freely.
    ///
    /// Nothing is re-solved, because nothing geometric changed.
    pub fn move_annotation(&mut self, constraint: EntityId, anchor: [f64; 2]) -> bool {
        let Some(held) = self
            .constraints
            .iter_mut()
            .find(|held| held.id == constraint)
        else {
            return false;
        };
        held.anchor = Some(anchor);
        true
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

    /// Decline a relation that names the BACK arm of a tangent lever.
    ///
    /// The back arm is not a freedom: [`sync_tangent_arms`](Self::sync_tangent_arms) puts it back
    /// on the mirror of the forward arm after every edit, so a constraint on it would be met by
    /// the solve and then quietly undone by the re-derivation half a step later. An arc's center
    /// is re-derived too and is still constrainable, but only because the residual system reads it
    /// AS the function it is; the back arm has no such reading, and until it does, declining is
    /// the only answer that does not lie.
    fn check_names_no_mirrored_arm(&self, kind: ConstraintKind) -> Result<(), ConstraintRefusal> {
        let named = kind.points();
        if named
            .iter()
            .any(|point| self.is_mirrored_tangent_arm(*point))
        {
            return Err(ConstraintRefusal::MirroredTangentArm);
        }
        Ok(())
    }

    /// Whether `point` is the BACK arm of some lever — the mirrored end, not the authored one.
    pub fn is_mirrored_tangent_arm(&self, point: EntityId) -> bool {
        self.splines.iter().any(|spline| {
            spline
                .tangents
                .values()
                .any(|handle| handle.backward == point)
        })
    }

    /// The FIT POINT whose tangent lever `point` is an arm of, or `None` if it is not an arm.
    ///
    /// An arm has no standing of its own — it is one end of the stick that steers a fit point — so
    /// every question about it (does it draw? may a tool seam onto it?) is really a question about
    /// the point it belongs to, and this is how a caller gets from one to the other.
    pub fn tangent_arm_owner(&self, point: EntityId) -> Option<EntityId> {
        self.splines.iter().find_map(|spline| {
            spline
                .tangents
                .iter()
                .find(|(_, handle)| handle.arms().contains(&point))
                .map(|(fit, _)| *fit)
        })
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
            // A radius is a statement about a curve that TURNS. A segment has no center to
            // measure from and the higher curves have no one radius, so both are refused here
            // rather than left to say nothing in a residual row. A diameter answers to the same
            // three, since it is the same statement doubled.
            ConstraintKind::Dimension(
                Dimension::Radius { curve, length } | Dimension::Diameter { curve, length },
            ) => {
                let turns = match curve {
                    SketchCurve::Arc(id) => self.arcs.iter().any(|arc| arc.id == id),
                    SketchCurve::Circle(id) => self.circles.iter().any(|held| held.id == id),
                    SketchCurve::Segment(_)
                    | SketchCurve::Bezier(_)
                    | SketchCurve::Ellipse(_)
                    | SketchCurve::Conic(_)
                    | SketchCurve::Spline(_) => return Err(ConstraintRefusal::Impossible),
                };
                if !turns {
                    return Err(ConstraintRefusal::UnknownEntity);
                }
                // A curve of no size is not a smaller curve, it is a point.
                if !length.value().is_finite() || length.value() <= 0.0 {
                    return Err(ConstraintRefusal::Impossible);
                }
            }
            // An angle is between two things the drawing gives a direction for, and the refusals
            // are Parallel's because the claim is Parallel's with a number attached: both arms
            // live, neither collapsed to nothing, and not the same arm twice.
            //
            // "The same arm twice" is the whole arm and not merely its entity: the two ENDS of one
            // arc are two different tangents, and the angle between them is what a sweep is.
            // The corner is not checked: both are real corners of any crossing, and which one the
            // author asked about is a fact about the gesture rather than a claim the drawing can
            // refuse.
            ConstraintKind::Dimension(Dimension::Angle {
                first,
                second,
                degrees,
                corner: _,
            }) => {
                let drawn = |arm: AngleArm| match arm {
                    AngleArm::Segment { segment } => live_segment(segment)
                        .map(|held| held.from != held.to)
                        .ok_or(ConstraintRefusal::UnknownEntity),
                    // An arc whose end stands on its own center has no radius there and so no
                    // tangent — the same nothing a collapsed segment gives.
                    AngleArm::ArcEnd { arc, end } => {
                        let held = self
                            .arcs
                            .iter()
                            .find(|held| held.id == arc)
                            .ok_or(ConstraintRefusal::UnknownEntity)?;
                        let standing = match end {
                            ArcEnd::From => held.from,
                            ArcEnd::To => held.to,
                        };
                        Ok(standing != held.center)
                    }
                };
                if !drawn(first)? || !drawn(second)? {
                    return Err(ConstraintRefusal::Impossible);
                }
                if first == second {
                    return Err(ConstraintRefusal::Impossible);
                }
                if !degrees.to_degrees_f64().is_finite() {
                    return Err(ConstraintRefusal::Impossible);
                }
            }
            // One extent is refused for the same three reasons the whole length is. A zero extent
            // is the exception that is NOT refused here but elsewhere: two points level with each
            // other are Horizontal or Vertical, which the author reaches by a different verb.
            ConstraintKind::Dimension(
                Dimension::Span { from, to, length }
                | Dimension::SpanAlong {
                    from, to, length, ..
                },
            ) => {
                if !known_point(from) || !known_point(to) {
                    return Err(ConstraintRefusal::UnknownEntity);
                }
                // A negative distance is no drawing's distance, and a zero one between two
                // distinct points is Coincident, which asserts one place rather than a span.
                if !length.value().is_finite() || length.value() <= 0.0 || from == to {
                    return Err(ConstraintRefusal::Impossible);
                }
            }
            ConstraintKind::Dimension(Dimension::Gap {
                point,
                segment,
                length,
            }) => {
                let (true, Some(line)) = (known_point(point), live_segment(segment)) else {
                    return Err(ConstraintRefusal::UnknownEntity);
                };
                // A run of no length draws no line, so there is no direction to measure across.
                // Read off where the ends STAND rather than which ids they are: two distinct
                // points dropped on one place draw just as little.
                let ends = [line.from, line.to].map(|end| {
                    self.points
                        .iter()
                        .find(|point| point.id == end)
                        .map(|point| point.at.in_plane())
                });
                let [Some(tail), Some(head)] = ends else {
                    return Err(ConstraintRefusal::UnknownEntity);
                };
                if (head[0] - tail[0]).hypot(head[1] - tail[1]) <= f64::EPSILON {
                    return Err(ConstraintRefusal::Impossible);
                }
                // Zero is refused for the reason a zero span is: a point ON the line is
                // `PointOnCurve`, which asserts a place rather than a distance from one.
                if !length.value().is_finite() || length.value() <= 0.0 {
                    return Err(ConstraintRefusal::Impossible);
                }
            }
            ConstraintKind::Dimension(Dimension::RimGap {
                first,
                second,
                length,
            }) => {
                if first.id() == second.id() {
                    return Err(ConstraintRefusal::Impossible);
                }
                // A center is the one thing every rim has and no straight curve does, so asking
                // for it asks both liveness and circularity at once.
                if self.circular_curve_center(first).is_none()
                    || self.circular_curve_center(second).is_none()
                {
                    return Err(ConstraintRefusal::UnknownEntity);
                }
                // Zero is refused for the reason a zero gap is: two rims of one size about one
                // center are one rim, and saying so is `Equal`.
                if !length.value().is_finite() || length.value() <= 0.0 {
                    return Err(ConstraintRefusal::Impossible);
                }
            }
            ConstraintKind::Coincident {
                point,
                onto: CoincidentTarget::Point(other),
            } => {
                if !known_point(point) || !known_point(other) {
                    return Err(ConstraintRefusal::UnknownEntity);
                }
                // A point already occupies its own place, so asserting it is a claim with no
                // content rather than a claim that happens to hold.
                if point == other {
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
            ConstraintKind::Curvature { joint, against } => {
                if !known_point(joint) {
                    return Err(ConstraintRefusal::UnknownEntity);
                }
                // The joint has to BE a joint: the free end of an open fit-point spline, with a
                // lever there and a span behind it to read a curvature from.
                if constraint::curvature_span_of(&self.splines, joint).is_none() {
                    return Err(ConstraintRefusal::CurvatureNeedsAJoint);
                }
                let live = match against {
                    SketchCurve::Segment(id) => self
                        .segments
                        .iter()
                        .find(|segment| segment.id == id)
                        .is_some_and(|segment| segment.from != segment.to),
                    SketchCurve::Arc(id) => self.arcs.iter().any(|arc| arc.id == id),
                    SketchCurve::Circle(id) => self.circles.iter().any(|circle| circle.id == id),
                    // A spline has no curvature the kernel can read off a stored radius, and two
                    // splines meeting is a different relation from a spline meeting a circle.
                    SketchCurve::Bezier(_)
                    | SketchCurve::Ellipse(_)
                    | SketchCurve::Conic(_)
                    | SketchCurve::Spline(_) => false,
                };
                if !live {
                    return Err(ConstraintRefusal::UnknownEntity);
                }
                // And the two have to MEET. The curvature a circle offers at a point is read from
                // that point's direction off the center, so a joint standing away from the curve is
                // handed an answer describing no curve at all.
                let (Some(at), Some(geometry)) = (
                    self.point_in_plane(joint),
                    self.curve_geometry(against, context),
                ) else {
                    return Err(ConstraintRefusal::UnknownEntity);
                };
                if !point_stands_on(at, geometry) {
                    return Err(ConstraintRefusal::CurvatureNeedsAJoint);
                }
            }
            ConstraintKind::Coincident {
                point,
                onto: CoincidentTarget::Curve(curve),
            } => {
                if !known_point(point) {
                    return Err(ConstraintRefusal::UnknownEntity);
                }
                let live = match curve {
                    // A collapsed segment has no line to be on, and the residual would divide by
                    // its length. Refusing says so instead of letting the solve report a collapse.
                    SketchCurve::Segment(id) => self
                        .segments
                        .iter()
                        .find(|segment| segment.id == id)
                        .is_some_and(|segment| segment.from != segment.to),
                    SketchCurve::Arc(id) => self.arcs.iter().any(|arc| arc.id == id),
                    SketchCurve::Circle(id) => self.circles.iter().any(|circle| circle.id == id),
                    // A spline has no support to read a distance from, and does not need one: the
                    // kernel holds a point to it by solving for WHERE along it the point stands.
                    SketchCurve::Spline(id) => self.splines.iter().any(|spline| spline.id == id),
                    // The rest have no support the kernel models — a rational Bézier is neither a
                    // line nor a circle — and no station either, so there is nothing to stand on.
                    SketchCurve::Bezier(_) | SketchCurve::Ellipse(_) | SketchCurve::Conic(_) => {
                        false
                    }
                };
                if !live {
                    return Err(ConstraintRefusal::UnknownEntity);
                }
                // A point that SHAPES a spline cannot also be held to it. A fit point already lies
                // on the curve it draws, so the claim has no content; a control point standing on
                // the curve is the point pulling against itself.
                if let SketchCurve::Spline(id) = curve {
                    if self.splines.iter().any(|spline| {
                        spline.id == id
                            && (spline.points.contains(&point)
                                || spline
                                    .tangents
                                    .values()
                                    .any(|handle| handle.arms().contains(&point)))
                    }) {
                        return Err(ConstraintRefusal::Impossible);
                    }
                }
                // An endpoint is already on its own segment's line, so the assertion is vacuous
                // and would only add a row that can never be violated.
                if let SketchCurve::Segment(id) = curve {
                    if self.segments.iter().any(|segment| {
                        segment.id == id && (segment.from == point || segment.to == point)
                    }) {
                        return Err(ConstraintRefusal::Impossible);
                    }
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
            .plan_apply(&self.points, &self.circles, &settled.solution)
            .map_err(|_| SketchEvaluationError::ScalarWritebackFailed)?;
        let before = self.points.clone();
        plan.apply(self);
        self.carry_authored_handles(&before);
        self.sync_derived_points();
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

    /// Split the segment with id `seg_id` by inserting a new point on it at `at`. The first
    /// half keeps the segment's id; the new second half inherits its
    /// `origin`, so a bounding face's origin-set is unchanged. No-op if `seg_id` is unknown.
    ///
    /// ON it, which is why `at` is projected rather than taken at its word. The caller's point is
    /// a cursor reading, and the snap policy that shaped it answers to the plane's grid, not to
    /// this segment's direction — so on any edge that does not run along the grid the raw point
    /// stands off the line, and inserting it verbatim replaces a straight edge with a dogleg. The
    /// shell already draws the foot as its preview; this makes the commit agree with it.
    ///
    /// A projection landing on either end is a no-op: an end already exists, and the split would
    /// hand the store a zero-length half with no line for a relation to be about.
    pub fn split_segment(&mut self, seg_id: EntityId, at: SketchPoint) {
        const APART: f64 = 1.0e-6;
        let Some(index) = self.segments.iter().position(|seg| seg.id == seg_id) else {
            return;
        };
        let (from, to) = (self.segments[index].from, self.segments[index].to);
        let (Some(tail), Some(head)) = (self.point_in_plane(from), self.point_in_plane(to)) else {
            return;
        };
        let Some(foot) = parametric::sketch::foot_on_span(tail, head, at.in_plane()) else {
            return;
        };
        if (foot[0] - tail[0]).hypot(foot[1] - tail[1]) <= APART
            || (foot[0] - head[0]).hypot(foot[1] - head[1]) <= APART
        {
            return;
        }
        // Standing already on the line, the authored point keeps whatever measurement it carries;
        // only a point that had to MOVE is rewritten, and a rewrite has no measurement to keep.
        let seated = if (foot[0] - at.in_plane()[0]).hypot(foot[1] - at.in_plane()[1]) <= APART {
            at
        } else {
            SketchPoint::from_continuous(foot[0], foot[1])
        };
        let new_point = self.add_point(seated);
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
    ///
    /// The bulge is the CALLER's convenience, not the stored form. An [`Arc`] is three placed
    /// points running counter-clockwise (ADR 0038), so the sweep is inverted once here: it decides
    /// where the center goes, and its SIGN decides which end is `from` — a clockwise bulge is the
    /// same drawn curve with its ends the other way round. Nothing downstream ever sees a negative
    /// sweep again.
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
        let (tail, head) = (self.point_in_plane(from)?, self.point_in_plane(to)?);
        let (center, _radius) = arc_center_radius(tail, head, sweep)?;
        let (from, to) = if sweep < 0.0 { (to, from) } else { (from, to) };
        // The arc's own id comes first so it precedes the center it arrives with, which is the
        // order every other curve-and-its-anchors gesture writes.
        let id = self.alloc_id();
        let center =
            self.add_construction_point(SketchPoint::from_continuous(center[0], center[1]));
        self.arcs.push(Arc {
            id,
            from,
            to,
            center,
            origin: id,
            role: EntityRole::Real,
        });
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

    /// Draw one endpoint/control/rho conic with exact dimensionless rho storage.
    ///
    /// The control point is off the curve, so it is reified as a CONSTRUCTION point — the same
    /// treatment a control-point spline's interior frame gets. On a POINT that role is a lifetime,
    /// not a linetype: it says [`prune_orphan_centers`](Self::prune_orphan_centers) may sweep the
    /// point once no curve names it. A handle draws and hit-tests the same either way, which is
    /// what makes the control point grabbable with no drawing code at all.
    pub fn add_conic(
        &mut self,
        from: SketchPoint,
        to: SketchPoint,
        control: SketchPoint,
        rho: f64,
    ) -> Result<EntityId, parametric::sketch::ConicCandidateError> {
        parametric::sketch::conic_candidate(
            from.in_plane(),
            to.in_plane(),
            control.in_plane(),
            rho,
        )?;
        let rho = parametric::ResolvedScalar::try_from_f64(rho)
            .map_err(|_| parametric::sketch::ConicCandidateError::InvalidRho)?;
        let from = self.add_point(from);
        let to = self.add_point(to);
        let control = self.add_construction_point(control);
        let id = self.alloc_id();
        boxed_push(
            &mut self.conics,
            Conic {
                id,
                from,
                to,
                control,
                rho,
                origin: id,
                role: EntityRole::Real,
            },
        );
        self.sync_derived_points();
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

    /// Whether every conic in the drawing still resolves to a curve.
    ///
    /// A conic collapses when its control point lands on the chord midpoint: the shoulder track has
    /// no length and there is no conic left to shape. The authoring gesture refuses that pick, and
    /// now that the control is a draggable handle the DRAG has to refuse it too — otherwise the
    /// curve silently disappears from the handles, the faces and the patterns while the entity
    /// stays in the document.
    /// Whether every tangent lever still has a direction to give.
    ///
    /// An arm dragged exactly onto its own fit point leaves a zero-length tangent, and
    /// [`spline_tangents`](Self::spline_tangents) reports `None` for it. That reads like a harmless
    /// degeneracy and is not one. An authored tangent is a DIRICHLET ROW in the interpolation, so
    /// every fit point having one is precisely what makes the tridiagonal system the identity and
    /// the spline LOCAL — each span decided by its own two ends and their two arms. A single `None`
    /// puts a genuine C2 row back and couples that span to the whole curve again.
    ///
    /// So the drag refuses, for the reason [`every_conic_resolves`](Self::every_conic_resolves)
    /// refuses: the author cannot see the consequence, and there is no way back but to guess where
    /// the arm stood. Checking it HERE rather than at the grab covers the settle too — a solve is
    /// as able to walk an arm onto its point as a cursor is.
    fn every_tangent_lever_stands(&self) -> bool {
        self.splines.iter().all(|spline| {
            spline.points.iter().all(|fit| {
                let Some(handle) = spline.tangents.get(fit) else {
                    return true;
                };
                let (Some(at), Some(arm)) = (
                    self.point_in_plane(*fit),
                    self.point_in_plane(handle.forward),
                ) else {
                    return true;
                };
                // The same arithmetic `spline_tangents` filters on, so the invariant this keeps is
                // exactly "that function never answers `None`".
                let tangent = [(arm[0] - at[0]) * 3.0, (arm[1] - at[1]) * 3.0];
                tangent[0] != 0.0 || tangent[1] != 0.0
            })
        })
    }

    fn every_conic_resolves(&self) -> bool {
        self.conics
            .iter()
            .all(|conic| self.conic_candidate(*conic).is_some())
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
            position(conic.control)?,
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
            .map(|conic| [conic.from, conic.to, conic.control]);
        boxed_retain(&mut self.conics, |conic| conic.id != id);
        if let Some(points) = points {
            self.drop_undrawn_points(points);
        }
    }

    /// Hold each of `points` to the curve the pick that named it landed on.
    ///
    /// The counterpart of the planting seam for builders that mint their own points: a spline
    /// allocates a point per fit pick rather than resolving one, so the hold cannot be applied
    /// before the build and is applied to what the build produced instead.
    ///
    /// **A fit point and a control point are ordinary points**, so dropping one on a curve means
    /// there what it means for a line's endpoint. Picks that only fix a quantity name no point and
    /// are simply not passed here — a conic's shoulder contributes rho and nothing else, and a
    /// circle's rim picks fix a radius and are gone by the time the circle exists.
    ///
    /// Extra picks or extra points are ignored rather than refused: the zip is a convenience for a
    /// caller that already knows the two lists correspond.
    pub fn hold_points_to_picks(
        &mut self,
        points: &[EntityId],
        picks: &[SketchTarget],
        context: parametric::EvaluationContext,
    ) {
        for (point, pick) in points.iter().zip(picks) {
            let Some(curve) = pick.onto() else {
                continue;
            };
            drop(self.add_constraint(
                ConstraintKind::Coincident {
                    point: *point,
                    onto: CoincidentTarget::Curve(curve),
                },
                context,
            ));
        }
    }

    pub fn add_fit_point_spline(
        &mut self,
        points: &[SketchPoint],
        closed: bool,
    ) -> Result<EntityId, parametric::sketch::SplineCandidateError> {
        let continuous: Vec<_> = points.iter().map(SketchPoint::in_plane).collect();
        // Resolved with no handle anywhere, which is what the curve looks like the instant it is
        // born: every handle is minted below AT the natural tangent, so having them changes
        // nothing until one is dragged.
        parametric::sketch::fit_point_spline(&continuous, &vec![None; continuous.len()], closed)?;
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
                tangents: std::collections::BTreeMap::new(),
            },
        );
        self.mint_tangent_handles(id);
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
                tangents: std::collections::BTreeMap::new(),
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
                let tangents = self.spline_tangents(spline, &points);
                parametric::sketch::fit_point_spline(&points, &tangents, spline.closed).ok()
            }
            SplineKind::ControlPoint => parametric::sketch::control_point_spline(&points).ok(),
        }
    }

    /// The authored derivative at each of `spline`'s fit points, `None` where no handle stands.
    ///
    /// A handle sits on the cubic's own control point, one third of the derivative out from the
    /// fit point, so dragging it lands the curve where the handle is rather than three times
    /// further along — the tangent it authors is therefore three times the offset it holds. A
    /// handle dropped exactly on its fit point names no direction, and reads as absent rather
    /// than collapsing the curve.
    fn spline_tangents(&self, spline: &Spline, points: &[[f64; 2]]) -> Vec<Option<[f64; 2]>> {
        spline
            .points
            .iter()
            .zip(points)
            .map(|(id, at)| {
                let handle = self.point_in_plane(spline.tangents.get(id)?.forward)?;
                let tangent = [(handle[0] - at[0]) * 3.0, (handle[1] - at[1]) * 3.0];
                (tangent[0] != 0.0 || tangent[1] != 0.0).then_some(tangent)
            })
            .collect()
    }

    /// The tangent handle standing at `fit_point`, if one does.
    ///
    /// Answers for the point the handle STEERS, not for the handle itself: asking about a handle
    /// answers `None`, because a handle has no handle.
    pub fn tangent_handle_of(&self, fit_point: EntityId) -> Option<TangentHandle> {
        self.splines
            .iter()
            .find_map(|spline| spline.tangents.get(&fit_point).copied())
    }

    /// The `(fit point, forward arm)` a BACK arm steers, if `point` is one.
    ///
    /// The redirection [`move_point`](Self::move_point) applies: a lever has one authored end, and
    /// a grab on the other end is a grab on that one, mirrored.
    fn back_arm_steers(&self, point: EntityId) -> Option<(EntityId, EntityId)> {
        self.splines.iter().find_map(|spline| {
            spline
                .tangents
                .iter()
                .find(|(_, handle)| handle.backward == point)
                .map(|(fit, handle)| (*fit, handle.forward))
        })
    }

    /// Whether `point` is a tangent handle some spline steers by.
    ///
    /// A handle is not the author's to delete — it is furniture the curve comes with, the way a
    /// control-point spline's frame is — so the delete cascade asks this before it does anything.
    pub fn is_tangent_handle(&self, point: EntityId) -> bool {
        self.splines.iter().any(|spline| {
            spline
                .tangents
                .values()
                .any(|handle| handle.arms().contains(&point))
        })
    }

    /// Give every fit point of `spline` the handle it is born with.
    ///
    /// A fit-point spline has handles on all of its points, always: they are not added and not
    /// removed (owner, 2026-08-03). Each is minted where the curve ALREADY bends, so a spline with
    /// its full set of handles draws exactly the curve it drew without them, and the handles are
    /// there to be dragged rather than there to be arranged for.
    ///
    /// Minting one at a time is safe in any order for exactly that reason: a handle at the natural
    /// position authors the tangent the curve already had, so it moves nothing the next mint reads.
    fn mint_tangent_handles(&mut self, spline: EntityId) {
        let Some(points) = self
            .splines
            .iter()
            .find(|held| held.id == spline && held.kind == SplineKind::FitPoint)
            .map(|held| held.points.clone())
        else {
            return;
        };
        for point in points {
            self.mint_tangent_handle(point);
        }
    }

    /// Mint `fit_point`'s lever, and answer the two arms the curve is now steered by.
    ///
    /// Answers the standing handle if the point already has one; `None` if the point is not a fit
    /// point of a fit-point spline, or if the spline draws no curve to read a tangent off.
    fn mint_tangent_handle(&mut self, fit_point: EntityId) -> Option<TangentHandle> {
        let spline = self
            .splines
            .iter()
            .find(|spline| {
                spline.kind == SplineKind::FitPoint && spline.points.contains(&fit_point)
            })?
            .clone();
        if let Some(standing) = spline.tangents.get(&fit_point) {
            return Some(*standing);
        }
        let anchor = self.point_in_plane(fit_point)?;
        let at = self.natural_handle_position(&spline, fit_point)?;
        let mirrored = at.in_plane();
        let forward = self.add_point(at);
        let backward = self.add_point(SketchPoint::from_continuous(
            2.0 * anchor[0] - mirrored[0],
            2.0 * anchor[1] - mirrored[1],
        ));
        for arm in [forward, backward] {
            self.set_point_lifetime(arm, PointLifetime::CurveAnchored);
        }
        let handle = TangentHandle { forward, backward };
        let index = self.splines.iter().position(|held| held.id == spline.id)?;
        self.splines[index].tangents.insert(fit_point, handle);
        Some(handle)
    }

    /// Put every back arm back on the mirror of its forward arm.
    ///
    /// The arms are symmetric about the fit point between them, and only the forward one is
    /// authored, so this is a re-derivation and not a solve: it runs after every edit that could
    /// have moved either end, out of [`sync_derived_points`](Self::sync_derived_points), which is
    /// already the pass every such edit ends with.
    fn sync_tangent_arms(&mut self) {
        let levers: Vec<_> = self
            .splines
            .iter()
            .flat_map(|spline| {
                spline
                    .tangents
                    .iter()
                    .map(|(fit, handle)| (*fit, *handle))
                    .collect::<Vec<_>>()
            })
            .collect();
        for (fit, handle) in levers {
            let (Some(anchor), Some(forward)) = (
                self.point_in_plane(fit),
                self.point_in_plane(handle.forward),
            ) else {
                continue;
            };
            if let Some(index) = self.point_index(handle.backward) {
                self.points[index].at = SketchPoint::from_continuous(
                    2.0 * anchor[0] - forward[0],
                    2.0 * anchor[1] - forward[1],
                );
            }
        }
    }

    /// Where a fresh handle for `fit_point` goes: the cubic control point beside it, which is the
    /// curve's own reading of the tangent there.
    fn natural_handle_position(&self, spline: &Spline, fit_point: EntityId) -> Option<SketchPoint> {
        // The FORWARD arm's position; the back arm is its mirror through the fit point.
        let index = spline.points.iter().position(|id| *id == fit_point)?;
        let at = self.point_in_plane(fit_point)?;
        let candidate = self.spline_candidate(spline)?;
        // Every piece but the last starts at its own fit point, so the derivative leaving that
        // piece is the tangent there. The final point of an open spline has no piece starting at
        // it, so its tangent is read at the END of the last one — the same slope, from the other
        // side.
        let tangent = match candidate.pieces.get(index) {
            Some(piece) => piece.derivative_at(0.0),
            None => candidate.pieces.last()?.derivative_at(1.0),
        };
        Some(SketchPoint::from_continuous(
            at[0] + tangent[0] / 3.0,
            at[1] + tangent[1] / 3.0,
        ))
    }

    pub fn delete_spline(&mut self, id: EntityId) {
        // The handles go with the fit points they steered: a tangent handle has no meaning apart
        // from the curve it bends.
        let points = self
            .splines
            .iter()
            .find(|spline| spline.id == id)
            .map(|spline| {
                let mut points = spline.points.clone();
                points.extend(spline.tangents.values().flat_map(|handle| handle.arms()));
                points
            });
        boxed_retain(&mut self.splines, |spline| spline.id != id);
        if let Some(points) = points {
            self.drop_undrawn_points(points);
        }
        self.prune_orphan_centers();
    }

    /// Whether some arc turns about this point.
    ///
    /// A STRUCTURAL question, not a provenance one. The center is placed and moves like any other
    /// point (ADR 0038); what this asks is whether it is the thing a round shape is arranged
    /// around, which is what tells a slot's grabbable handle apart from the center it is pinned
    /// to and what decides which of two stacked dots is worth drawing.
    ///
    /// A circle's center is deliberately out. It is the author's handle for the circle already —
    /// there is no second point coincident with it to tell apart — so counting it here would only
    /// hide the one dot a circle has.
    pub fn is_arc_center(&self, id: EntityId) -> bool {
        self.arcs.iter().any(|arc| arc.center == id)
    }

    /// How many curve ENDS meet at `id`: where ink stops, not where a curve merely names the point.
    ///
    /// A center is not an end. Nothing is drawn at an arc's center, so two concentric arcs sharing
    /// one do not "meet" there in any sense the author can see — which is the sense this counts in.
    fn curve_ends_meeting(&self, id: EntityId) -> usize {
        let segments = self
            .segments
            .iter()
            .flat_map(|segment| [segment.from, segment.to]);
        let arcs = self.arcs.iter().flat_map(|arc| [arc.from, arc.to]);
        segments.chain(arcs).filter(|end| *end == id).count()
    }

    /// Whether the point is worth drawing when nothing is hovered and no element is active.
    ///
    /// # A dot is for what the ink cannot say
    ///
    /// Drawing every point at all times is noise, and noise is the thing that hides the one dot
    /// that matters. So the question is not what KIND of point this is; it is whether the drawing
    /// already shows what the point would show.
    ///
    /// Two marks on a spot is the case where it does. A corner where two lines join looks exactly
    /// like a corner, and a dot on it adds nothing — the same is true of a break, a trim or a
    /// fillet junction, which is why none of those needs a case here. So does a point a relation
    /// pins onto a curve that something else already ends on: an Overall Slot's extreme is where
    /// its middle line stops AND where that line crosses a cap, and both facts are drawn.
    ///
    /// - **No mark at all** — a free point the author dropped. There is no ink on it, so the dot is
    ///   the only evidence it exists.
    /// - **One mark** — a loose end, or a point held on a curve with nothing ending there. "This
    ///   line stops here" and "this line joins something here" are the same picture, and the
    ///   difference is exactly what decides whether the profile closes into a region. Two ends that
    ///   merely COINCIDE draw two dots; genuinely joined, they draw none, and the author can see the
    ///   seam without trying to fill it.
    /// - **A fit or control point** — a spline's ink shows its shape and says nothing about its
    ///   parameterization, so a run through five points and a run through seven are one picture.
    /// - **A center** — always, however many marks land on it. A center is the handle a round thing
    ///   is held by, and losing it costs the author the only grip the shape has.
    /// - **A tangent arm** — never. An arm belongs to its lever, and a lever is a manipulator the
    ///   author asks for by selecting the point it steers rather than furniture that is always out.
    ///
    /// # One place, one dot, and it is the one that can be dragged
    ///
    /// Two points at the same spot draw one mark between them, and an ARC CENTER loses to a point
    /// that is nothing else. Every rule above asks what the drawing already shows, and a second dot
    /// on an occupied pixel shows nothing: it is not a fainter version of the answer, it is the same
    /// answer twice.
    ///
    /// Which one goes is not a coin toss. Both are placed points now (ADR 0038), so the tie-break is
    /// about REACH, not provenance: a center is bound to one arc, while the free handle standing on
    /// it is what the author's relations name and what a drag moves the whole shape by. An arc slot
    /// stacks four of these at its middle — the two rails' centers, the centerline's, and the handle
    /// tied to them — so the dot most likely to be under the cursor was the one least able to answer
    /// for the gesture (owner, 2026-08-05).
    pub fn point_draws_at_rest(&self, id: EntityId) -> bool {
        if self.tangent_arm_owner(id).is_some() || self.a_better_dot_stands_here(id) {
            return false;
        }
        if self
            .splines
            .iter()
            .any(|spline| spline.points.contains(&id))
            || self.point_is_a_center(id)
        {
            return true;
        }
        let held_on_curves = self
            .constraints
            .iter()
            .filter(|constraint| {
                matches!(
                    constraint.kind,
                    ConstraintKind::Coincident {
                        point,
                        onto: CoincidentTarget::Curve(_),
                    } if point == id
                )
            })
            .count();
        self.curve_ends_meeting(id).saturating_add(held_on_curves) < 2
    }

    /// Whether `id` is a DERIVED point some other dot already stands on.
    ///
    /// **Asked of every dot about to be drawn, not only of the ones at rest.** Standing under
    /// another dot is a fact about the point, not about the reason it came up: hovering an arc
    /// reveals the points it stands on, and one of those is the center it derives, so a reveal that
    /// skipped this would put the stack straight back the moment the author looked at the shape.
    ///
    /// Ranked so exactly one of a derived stack survives: a real point beats a derived one, and
    /// among derived twins the earliest wins.
    ///
    /// **This is the remedy of last resort, not the first one.** A dot that is the same answer
    /// written twice should not EXIST, and mostly no longer does: arcs turning about one place now
    /// share the center they echo to, so an arc slot's middle holds two points where it once held
    /// four. What is left is the pair that cannot be collapsed — a derived center and the real
    /// handle tied to it by a coincidence. The handle has to be a separate point because it has to
    /// be draggable, and a derived center is not: dragging one authors the quantity behind it. So
    /// the drawing genuinely has two points there, standing in one place on purpose, and only the
    /// one the author can grab is worth drawing.
    ///
    /// **Only ever hides a derived point.** Two REAL points at one spot are the seam case, and
    /// there both dots are the message: "these ends coincide" and "these ends are joined" are the
    /// difference between a profile that closes into a region and one that does not, and the author
    /// can only see it because the drawing declines to merge them.
    ///
    /// Position is compared within [`STACKED_DOT_TOLERANCE`] rather than exactly, because these stacks
    /// are SOLVED rather than authored: a handle is tied to a center by a relation, and a relation
    /// holds to the solver's residual, not to the bit. A three-point arc slot's three centers land
    /// within 3e-11 of each other and are one dot by any reading an author could have of them.
    /// The bound stays far under anything drawable so it can never merge two marks the author meant
    /// to tell apart, and the seam case is out of reach of it regardless — that one is two REAL
    /// points, and this only ever hides a derived one.
    pub fn a_better_dot_stands_here(&self, id: EntityId) -> bool {
        if !self.is_arc_center(id) {
            return false;
        }
        let Some(mine) = self.point_in_plane(id) else {
            return false;
        };
        let rank = |point: EntityId| self.points.iter().position(|stood| stood.id == point);
        self.points.iter().any(|other| {
            let Some(stood) = self.point_in_plane(other.id) else {
                return false;
            };
            other.id != id
                && (stood[0] - mine[0]).hypot(stood[1] - mine[1]) < STACKED_DOT_TOLERANCE
                && (!self.is_arc_center(other.id) || rank(other.id) < rank(id))
        })
    }

    /// Whether `id` is the center a round thing turns about.
    fn point_is_a_center(&self, id: EntityId) -> bool {
        self.circles.iter().any(|circle| circle.center == id)
            || self.arcs.iter().any(|arc| arc.center == id)
            || self.ellipses.iter().any(|ellipse| ellipse.center == id)
    }

    /// Whether some curve's DRAWN PATH runs through `id`.
    ///
    /// The dot takes the value of what it belongs to. Ink through it means the point is part of the
    /// drawing and draws in the drawing's ink; no ink through it means the point is a handle FOR the
    /// drawing — a center, a control point, a conic's off-curve control, a free point — and it draws
    /// recessive, so the shape stays primary over the scaffolding that shapes it. Fusion says the
    /// same thing on a white page by drawing the free one white and the connected one black.
    ///
    /// This is not [`curve_ends_meeting`](Self::curve_ends_meeting), which asks where ink STOPS. A
    /// spline's interior fit point stops nothing and still has ink through it; an arc's center is an
    /// end of nothing and has none. And it is not a lifetime or an ownership question either — see
    /// [`PointLifetime`] for how the three axes divide.
    pub fn point_stands_on_ink(&self, id: EntityId) -> bool {
        let segments = self
            .segments
            .iter()
            .flat_map(|segment| [segment.from, segment.to]);
        let arcs = self.arcs.iter().flat_map(|arc| [arc.from, arc.to]);
        let conics = self.conics.iter().flat_map(|conic| [conic.from, conic.to]);
        let ellipses = self
            .ellipses
            .iter()
            .flat_map(|ellipse| [ellipse.major_endpoint, ellipse.width_point]);
        // Bézier pieces carry their two ends on the curve and their two controls off it, in
        // parameter order — the same on/off split every other curve here draws.
        let beziers = self
            .beziers
            .iter()
            .flat_map(|bezier| [bezier.controls[0], bezier.controls[3]]);
        // A FIT point sits on the run. A CONTROL point steers it from off the curve — except the
        // first and last, which a clamped frame interpolates, so a control-point spline starts and
        // ends on two of its own controls and passes beside all the rest.
        let splines = self.splines.iter().flat_map(|spline| match spline.kind {
            SplineKind::FitPoint => spline.points.clone(),
            SplineKind::ControlPoint => spline
                .points
                .first()
                .into_iter()
                .chain(spline.points.last())
                .copied()
                .collect(),
        });
        segments
            .chain(arcs)
            .chain(conics)
            .chain(ellipses)
            .chain(beziers)
            .chain(splines)
            .any(|on_curve| on_curve == id)
    }

    /// Every point some authored relation reaches — named outright, or standing on a curve the
    /// relation names.
    ///
    /// A point in here is the solver's to place, so anything that moves points on its own
    /// authority ([`carry_authored_handles`](Self::carry_authored_handles)) has to leave it alone.
    ///
    /// Asked with [`every_point_of`](Self::every_point_of) rather than
    /// [`points_of`](Self::points_of), because a relation held to a curve is held to the SHAPE of
    /// it. A spline's tangent arms are ordinarily the drawing's to carry — a handle means its
    /// offset, so it rides its fit point — but the moment something stands on that spline the arms
    /// are load-bearing, and carrying them after the solve would redraw the curve out from under
    /// the very point the solve had just put on it.
    fn constrained_points(&self) -> Vec<EntityId> {
        let mut named = Vec::new();
        for constraint in &self.constraints {
            named.extend(constraint.kind.points());
            named.extend(self.span_points_derived_by(constraint.kind));
            for curve in constraint.kind.curves() {
                named.extend(self.every_point_of(curve));
            }
            // A relation's segment slot can hold an arc or a circle id too (Equal compares radii),
            // and `points_of` finds nothing for a kind the id does not belong to, so asking all
            // three costs a miss rather than a wrong answer.
            for id in constraint.kind.segments() {
                named.extend(self.points_of(SketchCurve::Segment(id)));
                named.extend(self.points_of(SketchCurve::Arc(id)));
                named.extend(self.points_of(SketchCurve::Circle(id)));
            }
        }
        named
    }

    /// Every authored point that STANDS OFF another, as `(anchor, follower)`.
    ///
    /// Both are authored — neither is [`is_arc_center`](Self::is_arc_center) — but the
    /// follower's meaning is its OFFSET from the anchor rather than its position: a spline's
    /// tangent is the vector to its handle, and an ellipse's axes are the vectors to its
    /// endpoints. That is the pairing [`carry_authored_handles`](Self::carry_authored_handles)
    /// preserves across a solve.
    fn authored_followers(&self) -> Vec<(EntityId, EntityId)> {
        self.splines
            .iter()
            .flat_map(|spline| {
                // The forward arm only. The back arm is a mirror, and `sync_tangent_arms` puts it
                // back on the mirror after this pass runs — carrying it here as well would be two
                // answers for one position.
                spline
                    .tangents
                    .iter()
                    .map(|(fit, handle)| (*fit, handle.forward))
                    .collect::<Vec<_>>()
            })
            .chain(self.ellipses.iter().flat_map(|ellipse| {
                [
                    (ellipse.center, ellipse.major_endpoint),
                    (ellipse.center, ellipse.width_point),
                ]
            }))
            .collect()
    }

    /// Carry every authored handle along with the point it stands off, `before` being where the
    /// points were when the solve started.
    ///
    /// # A handle names an offset, and the kernel cannot see that
    ///
    /// To the solver a handle is two loose coordinates in no relation to anything, so a solve that
    /// moves its ANCHOR leaves it standing where it was — and the quantity it names, which is the
    /// offset, silently re-aims. A constraint that never mentioned a spline's tangent would rotate
    /// it as a side effect of moving the fit point under it.
    ///
    /// So the handle follows its anchor's displacement, UNLESS a constraint reached the handle
    /// itself: that constraint decided where it goes, and its answer wins over this one.
    ///
    /// The test is whether a relation NAMES the handle, not whether the solve moved it. A handle
    /// pinned where it already stood does not move while its anchor does, and a
    /// did-it-move test cannot tell that apart from a loose handle — it would carry the pinned one
    /// off its pin, and the next solve would drag the pin's partner back toward it, so moving one
    /// fit point would perturb geometry nobody connected to it.
    fn carry_authored_handles(&mut self, before: &[Point]) {
        let claimed = self.constrained_points();
        let stood = |points: &[Point], id: EntityId| {
            points
                .iter()
                .find(|point| point.id == id)
                .map(|point| point.at)
        };
        for (anchor, follower) in self.authored_followers() {
            if claimed.contains(&follower) {
                continue;
            }
            let (Some(was), Some(now)) = (stood(before, anchor), stood(&self.points, anchor))
            else {
                continue;
            };
            let (was, now) = (was.in_plane(), now.in_plane());
            let delta = [now[0] - was[0], now[1] - was[1]];
            if delta[0] == 0.0 && delta[1] == 0.0 {
                continue;
            }
            let Some(held) = stood(&self.points, follower).map(|point| point.in_plane()) else {
                continue;
            };
            if let Some(index) = self.point_index(follower) {
                self.points[index].at =
                    SketchPoint::from_continuous(held[0] + delta[0], held[1] + delta[1]);
            }
        }
    }

    /// Put every point the drawing owns a COMPONENT of back where the geometry needs it.
    ///
    /// Two things left after ADR 0038, and neither is a derived point. A spline's tangent arms are
    /// one lever with two ends, so the back arm is the mirror of the front. An arc's center has one
    /// freedom and the plane has two, so the leftover direction is seated
    /// ([`seat_arc_centers`](Self::seat_arc_centers)). A conic's shoulder used to be here as well
    /// and is simply gone: rho is a value, and it is authored as one.
    pub fn sync_derived_points(&mut self) {
        self.seat_arc_centers();
        self.sync_tangent_arms();
    }

    /// Draw this arc the other way round its own circle, by swapping the two ends it is drawn
    /// between. Reports whether there was an arc to turn.
    ///
    /// The only way an arc changes direction, because it carries no direction to change. It runs
    /// counter-clockwise by definition (ADR 0038) and the endpoint ORDER is what says which way it
    /// goes — [`counter_clockwise_sweep_degrees`] states it outright: "the caller that wants the
    /// other arc over the same chord asks about the same three points with the ends swapped". So
    /// this is that caller. Nothing moves: the same three dots stand exactly where they stood, and
    /// what changes is which of the two pieces between them is the drawn one.
    ///
    /// **Every reader of the order comes with it.** An [`AngleArm::ArcEnd`] names a physical corner
    /// of the drawing by naming an end, so leaving those tags alone would silently re-aim each one
    /// at the opposite corner — a right angle authored at one end quietly becoming a claim about
    /// the other. The tags on THIS arc are swapped in the same write, which is what keeps the
    /// relation saying what its author said.
    pub fn reverse_arc(&mut self, id: EntityId) -> bool {
        let Some(arc) = self.arcs.iter_mut().find(|arc| arc.id == id) else {
            return false;
        };
        std::mem::swap(&mut arc.from, &mut arc.to);
        for constraint in &mut self.constraints {
            let ConstraintKind::Dimension(Dimension::Angle { first, second, .. }) =
                &mut constraint.kind
            else {
                continue;
            };
            for arm in [first, second] {
                if let AngleArm::ArcEnd { arc: named, end } = arm {
                    if *named == id {
                        *end = match end {
                            ArcEnd::From => ArcEnd::To,
                            ArcEnd::To => ArcEnd::From,
                        };
                    }
                }
            }
        }
        true
    }

    /// Put every UNSHARED arc center back on its chord's perpendicular bisector.
    ///
    /// Not a derivation, and not a walk back from ADR 0038. The center keeps the freedom it
    /// actually has — how far out along the bisector it stands, which is the arc's radius and the
    /// author's to choose. What this takes away is the component running ALONG the chord, and that
    /// is not a freedom of the arc at all: sliding the center that way names no different circle
    /// through the two ends. It only lets the stored dot drift off the curve the author is
    /// looking at, and nobody ever chose the drift.
    ///
    /// A SHARED center — two arcs deliberately tied to one dot by
    /// [`tie_arc_centers`](Self::tie_arc_centers) — is left alone, because it cannot stand on two
    /// bisectors at once. That case belongs to the solver: each arc contributes one equal-radius
    /// row and the settle balances them against each other.
    fn seat_arc_centers(&mut self) {
        for index in 0..self.arcs.len() {
            let arc = self.arcs[index];
            if self
                .arcs
                .iter()
                .filter(|other| other.center == arc.center)
                .count()
                > 1
            {
                continue;
            }
            let (Some(from), Some(to), Some(center)) = (
                self.point_in_plane(arc.from),
                self.point_in_plane(arc.to),
                self.point_in_plane(arc.center),
            ) else {
                continue;
            };
            let Some(seat) = arc_center_on_bisector(from, to, center) else {
                continue;
            };
            if let Some(stood) = self.point_index(arc.center) {
                self.points[stood].at = SketchPoint::from_continuous(seat[0], seat[1]);
            }
        }
    }

    /// Point every arc in `arcs` at the FIRST one's center, so a set of arcs turning about one
    /// place shares one dot instead of holding one each.
    ///
    /// Structural and permanent, which is why the caller must also ASSERT that these arcs are
    /// concentric. A shared dot can only be in one place, so sharing it is a claim that they agree;
    /// backed by a relation the claim stays true through every drag, and backed by nothing it
    /// becomes a lie the first time the shape is resized. Deciding it here rather than by noticing
    /// coincident centers also keeps two arcs that merely pass through a concentric arrangement
    /// from binding to each other forever.
    ///
    /// The centers this orphans are dropped, but ONLY those — a general sweep here would take the
    /// slot's own handles with them. A handle is named by nothing except the coincidence tying it
    /// to a center, and merely being named by a relation is not what keeps a point alive.
    fn tie_arc_centers(&mut self, arcs: &[SketchCurve]) {
        let center = arcs.iter().find_map(|curve| match curve {
            SketchCurve::Arc(id) => self
                .arcs
                .iter()
                .find(|arc| arc.id == *id)
                .map(|arc| arc.center),
            _ => None,
        });
        let Some(center) = center else {
            return;
        };
        let mut replaced = Vec::new();
        for curve in arcs {
            let SketchCurve::Arc(id) = curve else {
                continue;
            };
            if let Some(arc) = self.arcs.iter_mut().find(|arc| arc.id == *id) {
                if arc.center != center {
                    replaced.push(arc.center);
                    arc.center = center;
                }
            }
        }
        let referenced = self.referenced_points();
        self.points
            .retain(|point| !replaced.contains(&point.id) || referenced.contains(&point.id));
    }

    /// Drop every construction point nothing references any more — the center of an arc that
    /// has just been deleted. A center the author has since drawn to (an edge names it) is
    /// referenced, so it survives as ordinary geometry.
    ///
    /// A relation that ANCHORS a point to surviving geometry counts as a reference too. A center
    /// rectangle's center is held at the crossing of its diagonals and is named by no curve at
    /// all; without this it would survive creation and then vanish the first time an unrelated
    /// deletion swept the sketch, taking its assertions with it. Merely being mentioned is not
    /// enough — see [`ConstraintKind::anchored_points`] — so a `Fix`ed circle center still goes
    /// with its circle.
    fn prune_orphan_centers(&mut self) {
        let referenced = self.referenced_points();
        self.points.retain(|point| {
            point.lifetime != PointLifetime::CurveAnchored || referenced.contains(&point.id)
        });
    }

    /// Every point some piece of geometry or anchoring relation names.
    fn referenced_points(&self) -> std::collections::BTreeSet<EntityId> {
        let mut referenced = std::collections::BTreeSet::new();
        for constraint in &self.constraints {
            referenced.extend(constraint.kind.anchored_points());
        }
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
            referenced.extend([conic.from, conic.to, conic.control]);
        }
        for spline in &*self.splines {
            referenced.extend(spline.points.iter().copied());
            referenced.extend(spline.tangents.values().flat_map(|handle| handle.arms()));
        }
        referenced
    }

    /// The straight segment joining `a` and `b` in either direction, if one is held.
    ///
    /// A tool that draws a whole loop needs the id back even when [`connect`](Self::connect)
    /// declined because the edge was already there — the edge it means to constrain is that
    /// standing one, not a duplicate it failed to add.
    pub fn segment_between(&self, a: EntityId, b: EntityId) -> Option<EntityId> {
        self.segments
            .iter()
            .find(|seg| (seg.from == a && seg.to == b) || (seg.from == b && seg.to == a))
            .map(|seg| seg.id)
    }

    /// The arc of exactly this signed sweep joining `a` and `b`, if one is held.
    ///
    /// The sweep is part of the question, unlike [`segment_between`](Self::segment_between): a
    /// pair of points can carry two different arcs at once — a lens — so "the arc between them"
    /// is not on its own a curve. Reading the same direction back the other way is the same arc,
    /// which is why the reversed pair matches the NEGATED sweep.
    pub fn arc_between(
        &self,
        a: EntityId,
        b: EntityId,
        sweep: AngleMeasurement,
    ) -> Option<EntityId> {
        self.arcs
            .iter()
            .find(|arc| self.arc_draws(arc, a, b, sweep.to_degrees_f64()))
            .map(|arc| arc.id)
    }

    /// The point entity holding a circular curve's center, if it has one.
    ///
    /// `None` for every curve that turns without a center to turn about — a segment, a Bézier, a
    /// spline. Asking is how a caller ties something to a curve's center without knowing which
    /// store the curve came out of.
    pub fn center_point_of(&self, curve: SketchCurve) -> Option<EntityId> {
        match curve {
            SketchCurve::Arc(id) => self
                .arcs
                .iter()
                .find(|arc| arc.id == id)
                .map(|arc| arc.center),
            SketchCurve::Circle(id) => self
                .circles
                .iter()
                .find(|circle| circle.id == id)
                .map(|circle| circle.center),
            SketchCurve::Segment(_)
            | SketchCurve::Bezier(_)
            | SketchCurve::Ellipse(_)
            | SketchCurve::Conic(_)
            | SketchCurve::Spline(_) => None,
        }
    }

    /// Whether a straight segment already joins `a` and `b` in either direction.
    pub fn segment_joins(&self, a: EntityId, b: EntityId) -> bool {
        self.segment_between(a, b).is_some()
    }

    /// Whether some stored arc already traces the CURVE `from → to` sweeping `sweep_degrees`.
    /// Reversing an arc's direction mirrors it about the chord unless the sweep's sign flips
    /// too, so the reversed match is against the negated sweep — an arc bulging the other way
    /// over the same pair is a different curve, and legal.
    pub fn arc_traces(&self, from: EntityId, to: EntityId, sweep_degrees: f64) -> bool {
        self.arcs
            .iter()
            .any(|arc| self.arc_draws(arc, from, to, sweep_degrees))
    }

    /// Whether `arc` is the curve running `a → b` through the signed `sweep_degrees`.
    ///
    /// The stored form is counter-clockwise (ADR 0038), so a negative sweep asks about the same
    /// drawn curve with its ends the other way round. The sweep is DERIVED from three placed
    /// points and the ends are stored at `f32` fraction, so it is compared within
    /// [`SAME_ARC_SWEEP_DEGREES`] rather than bit-for-bit: two arcs an author drew alike are the
    /// same curve, and the last few bits of an `atan2` are not the difference between them.
    fn arc_draws(&self, arc: &Arc, a: EntityId, b: EntityId, sweep_degrees: f64) -> bool {
        let (from, to, turn) = if sweep_degrees < 0.0 {
            (b, a, -sweep_degrees)
        } else {
            (a, b, sweep_degrees)
        };
        arc.from == from
            && arc.to == to
            && self
                .arc_form(arc)
                .is_some_and(|form| (form.sweep_degrees - turn).abs() < SAME_ARC_SWEEP_DEGREES)
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
            // The label rides with the drawing. The same voxel numbers mean a different length at
            // a new density, so an anchor left alone would slide off the geometry it annotates.
            if let (Some(anchor), true) = (&mut constraint.anchor, old_density != 0) {
                let scale = f64::from(new_density) / f64::from(old_density);
                anchor[0] *= scale;
                anchor[1] *= scale;
            }
            match &mut constraint.kind {
                ConstraintKind::Fix { at, .. } => {
                    *at = at.retargeted(old_density, new_density);
                }
                ConstraintKind::Dimension(
                    Dimension::Span { length, .. }
                    | Dimension::SpanAlong { length, .. }
                    | Dimension::Gap { length, .. }
                    | Dimension::RimGap { length, .. }
                    | Dimension::Radius { length, .. }
                    | Dimension::Diameter { length, .. },
                ) => {
                    *length = length.retargeted(old_density, new_density);
                }
                ConstraintKind::Quantize { pitch, phase, .. } => {
                    *pitch = pitch.retargeted(old_density, new_density);
                    *phase = phase.retargeted(old_density, new_density);
                }
                // An angle has no block term and no density, so a re-target is not something it
                // can be asked. It stays exactly the number the author wrote.
                ConstraintKind::Dimension(Dimension::Angle { .. })
                | ConstraintKind::Horizontal { .. }
                | ConstraintKind::Vertical { .. }
                | ConstraintKind::Coincident { .. }
                | ConstraintKind::Parallel { .. }
                | ConstraintKind::Perpendicular { .. }
                | ConstraintKind::Equal { .. }
                | ConstraintKind::Midpoint { .. }
                | ConstraintKind::Collinear { .. }
                | ConstraintKind::Curvature { .. }
                | ConstraintKind::Tangent { .. }
                | ConstraintKind::Concentric { .. }
                | ConstraintKind::Symmetry { .. } => {}
            }
        }
        self.sync_derived_points();
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
        // An arc is additionally invalid when its three points do not draw one: a missing
        // center, ends stacked on each other, an end sitting on the center. Each of those is a
        // sweep that cannot be read, which is what `arc_form` returning nothing means.
        let arcs = std::mem::take(&mut self.arcs);
        self.arcs = arcs
            .into_iter()
            .filter(|arc| {
                arc.from != arc.to
                    && point_ids.contains(&arc.from)
                    && point_ids.contains(&arc.to)
                    && point_ids.contains(&arc.center)
                    && self.arc_form(arc).is_some()
            })
            .collect();
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
            [conic.from, conic.to, conic.control]
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
        self.sync_derived_points();
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
        self.sync_derived_points();
        dropped
    }
}

/// Default arc flattening tolerance: the maximum sagitta (chord-to-arc deviation) of one chord, in
/// the sketch plane's own continuous units.
///
/// **Not a screen tolerance.** A chord count derived from this follows the arc's size in the plane
/// and not its size on screen, so the same handful of chords is drawn at every zoom and a
/// magnified curve reads as a visible polygon. A painter derives its own tolerance from the
/// PROJECTED radius instead, and uses this only as the cap.
///
/// This is NOT the resolved meaning of a curve — the region carries its arcs, and the field
/// measures them ([`ProfileEdge`]). It is the default a **terminal adapter** flattens at when it
/// has to produce something discrete and has nowhere to put a curve: a crease polyline, the
/// exact-`f64` cell classifier's polygon, a test's outline. Nothing downstream of one of those
/// inherits it, so it is a tuning knob rather than a document-format constant.
pub const ARC_SAGITTA_TOLERANCE: f64 = 1.0 / 16.0;

/// Hard cap on chords per arc, so a huge-radius near-collinear arc cannot degenerate
/// into an unbounded fan.
const ARC_MAX_CHORDS: u32 = 512;

/// How far one arc's drawn sweep has been carried since a gesture opened, UNWRAPPED — so it keeps
/// counting past a whole turn instead of folding back.
///
/// The one piece of state a gesture that bends an arc needs, and it needs it because orientation is
/// path-dependent. An arc carries no direction: it runs counter-clockwise from
/// [`from`](Arc::from) to [`to`](Arc::to), and the only way it can turn the other way is for those
/// two to swap ([`reverse_arc`](Sketch::reverse_arc)). Deciding when to swap by comparing where the
/// drawing is NOW against where it started cannot work — winding an arc up toward a full circle
/// carries the sweep by up to a whole turn less the one it began with, routinely past a half turn,
/// and any short-way reading answers backwards there, unswapping the arc at exactly the moment the
/// author is most committed to the wind. So the gesture carries the sweep it has drawn.
///
/// The shell's whole-curve translate carrying where it was pressed is the same shape of thing: a
/// gesture whose meaning depends on where it started keeps that where itself, because the drawing
/// does not remember it.
///
/// **Measured from the arc's OWN two ends, not from the hand, and read after the settle.** An arc
/// slot is why: its two rails and its centerline are the same shape three times at three radii
/// about one hub, and a hand on a spine end names the CENTERLINE alone — a spine end IS a cap
/// center, while the rails' ends sit half a width away and are named by nothing. Turning the hand
/// past the far cap swapped the centerline and left both rails going the long way round, 15 degrees
/// against 345. Reading each arc's own ends instead answers for every arc a gesture bends, whatever
/// held it: the rails cross in the same frame the spine does, because their ends are radially
/// aligned, and nothing had to know what a slot is. An arc the gesture never moves has a constant
/// sweep, crosses nothing, and is left alone — so this is opened over the whole drawing rather than
/// over a family somebody has to define.
///
/// **The order is a label the settle ignores.** Its rows are radii; the tangent branch reads
/// centers and radii. So the swap can be applied AFTER the drawing settles, which is the only place
/// the carried arcs can be measured at all. One consumer does run earlier — tangent CONTACT
/// validation asks whether a contact stands on the drawn piece — so a frame that crosses a seam
/// validates under the order the frame before it left. Contacts at arc ends are immune, both
/// complementary readings sharing their endpoints, and an interior contact on an arc crossing its
/// own seam is degenerate that frame whichever way it is read.
///
/// What decides the order is CROSSING PARITY, not the sign of the carry. The two are the same thing
/// only until the drawing has been round once: the ends meet whenever the sweep passes a multiple
/// of a whole circle, so an arc wound a full lap has crossed twice and is drawn the way it started.
/// Reading the sign alone flips it back at the second seam and takes a 345-degree arc down to 15 in
/// one step. Both seams are seams, and they alternate.
///
/// A frame that lands ON one — the ends stacked, no piece of the circle to prefer — is stood rather
/// than written, and **the carry unwraps over WRITTEN frames only**. A stood frame writes nothing,
/// so the next written frame is two frames of motion away from the last one, which is still far
/// under the half turn the unwrapping needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArcTurnUnderAGesture {
    /// The arc being carried.
    arc: EntityId,
    /// The end the arc was drawn FROM when the gesture opened. Held by id, because which field it
    /// sits in is exactly what changes underneath.
    first: EntityId,
    /// The end it was drawn TO.
    second: EntityId,
    /// Degrees, unwrapped: the counter-clockwise sweep from `first` round to `second`, counted on
    /// past a whole turn and on past none. Winding an arc up toward a full circle is a real gesture
    /// and the count has to carry it.
    carried_degrees: f64,
}

impl ArcTurnUnderAGesture {
    /// Every arc in the drawing, as the gesture opens.
    ///
    /// The whole drawing rather than a reach: an arc the gesture does not move cannot cross, so it
    /// costs a subtraction a frame and answers correctly for free. Working out which arcs a gesture
    /// reaches, on the other hand, is a question with a wrong answer in it.
    #[must_use]
    pub fn opening_over(sketch: &Sketch) -> Vec<Self> {
        sketch
            .arcs
            .iter()
            .filter_map(|arc| Self::opening(sketch, arc.id))
            .collect()
    }

    /// Read the sweep one arc already stands at. `None` when its three points cannot be read or do
    /// not draw an arc.
    #[must_use]
    pub fn opening(sketch: &Sketch, arc: EntityId) -> Option<Self> {
        let stood = sketch.arcs.iter().find(|stored| stored.id == arc)?;
        Some(Self {
            arc,
            first: stood.from,
            second: stood.to,
            carried_degrees: sketch.drawn_sweep_between(stood.center, stood.from, stood.to)?,
        })
    }

    /// The carry, advanced by the SHORTEST step from where the arc last stood — which is what
    /// unwraps it.
    fn carried_after(&self, drawn: f64) -> f64 {
        let step = wrapped_into_a_half_turn(drawn - self.carried_degrees);
        if step.is_finite() {
            self.carried_degrees + step
        } else {
            self.carried_degrees
        }
    }

    /// Which end the arc should be drawn FROM at a given carry. Parity, not sign: the ends meet
    /// whenever the carry passes a multiple of a whole circle.
    fn leads_at(&self, carried: f64) -> EntityId {
        if ((carried / 360.0).floor() % 2.0).abs() > 0.5 {
            self.second
        } else {
            self.first
        }
    }

    /// This arc's id, if the frame `at` describes turns it round the other way.
    ///
    /// Asked of a CANDIDATE drawing — the solution the settle has produced and not yet written —
    /// because the answer has to be known before the frame is validated. Tangent CONTACT
    /// validation asks whether a contact stands on the DRAWN piece, and the two readings of a
    /// crossing arc are different pieces: measured, the same move is refused as
    /// `OutsideFirstDomain` under the reading the last frame left, and accepted under the one this
    /// frame is about to be given.
    fn crossing_under(
        &self,
        arcs: &[Arc],
        at: &dyn Fn(EntityId) -> Option<[f64; 2]>,
    ) -> Option<EntityId> {
        let stored = arcs.iter().find(|stored| stored.id == self.arc)?;
        let drawn = drawn_sweep_of(at(stored.center)?, at(self.first)?, at(self.second)?)?;
        (stored.from != self.leads_at(self.carried_after(drawn))).then_some(self.arc)
    }

    /// Take the carry up to where the drawing now stands.
    ///
    /// Called only when the frame WRITES. A stood frame leaves the drawing where it was, so
    /// advancing the carry there would count a crossing the geometry never made, and the next
    /// written frame would count it again.
    fn commit(&mut self, sketch: &Sketch) {
        let Some(stored) = sketch.arcs.iter().find(|stored| stored.id == self.arc) else {
            return;
        };
        if let Some(drawn) = sketch.drawn_sweep_between(stored.center, self.first, self.second) {
            self.carried_degrees = self.carried_after(drawn);
        }
    }
}

/// [`Sketch::drawn_sweep_between`] asked of loose coordinates, so a caller can ask it of a frame
/// the drawing has not been given yet.
fn drawn_sweep_of(center: [f64; 2], first: [f64; 2], second: [f64; 2]) -> Option<f64> {
    counter_clockwise_sweep_degrees(
        arc_center_on_bisector(first, second, center)?,
        first,
        second,
    )
}

/// Whether three placed points draw an arc an author could see and grab.
///
/// The reading's own question ([`Sketch::arc_form`]) asked of loose coordinates, so a caller can
/// ask it of where the drawing STOOD rather than only of where it stands.
fn three_points_draw_a_circle(from: [f64; 2], to: [f64; 2], center: [f64; 2]) -> bool {
    let Some(seat) = arc_center_on_bisector(from, to, center) else {
        return false;
    };
    let radius = (from[0] - seat[0]).hypot(from[1] - seat[1]);
    let apart = (from[0] - to[0]).hypot(from[1] - to[1]);
    counter_clockwise_sweep_degrees(seat, from, to).is_some()
        && radius > STACKED_DOT_TOLERANCE
        && apart > STACKED_DOT_TOLERANCE
}

/// `degrees` brought into `(-180, 180]` — the shortest turn that lands at the same bearing.
fn wrapped_into_a_half_turn(degrees: f64) -> f64 {
    let folded = (180.0 - degrees).rem_euclid(360.0);
    180.0 - folded
}

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

/// How far two derived sweeps may differ and still be the same arc, in degrees.
///
/// A sweep is read off three placed points whose fractions are stored at `f32`, so an arc drawn to
/// a given bulge reads back a hair away from it. The bound sits far under anything an author could
/// have meant to tell apart and far over the drift a round trip through storage costs.
const SAME_ARC_SWEEP_DEGREES: f64 = 1.0e-3;

/// `center` moved onto the perpendicular bisector of `from → to`, so the circle it names passes
/// through BOTH ends exactly. `None` for a degenerate chord, which has no bisector.
///
/// An arc's center is authored (ADR 0038) and nothing stops it standing nearer one end than the
/// other — a drag puts it where the cursor was, and the solver's equal-radius row only pulls it
/// back on the next solve. Display and measurement still have to name one circle, so the component
/// of the center running ALONG the chord is dropped here. That is exactly the motion the
/// equal-radius row removes, so this agrees with the solver rather than hiding a disagreement from
/// it: where the solve has converged the projection moves nothing at all.
///
/// Written as a SUBTRACTION of the along-chord component rather than a rebuild from the midpoint,
/// so a center already on the bisector comes back bit-for-bit. Rebuilding it costs a few ulps every
/// read, and a few ulps is the whole discriminant where an arc meets a line it is nearly tangent
/// to — the reading has to be stable, not merely close.
fn arc_center_on_bisector(from: [f64; 2], to: [f64; 2], center: [f64; 2]) -> Option<[f64; 2]> {
    let chord = [to[0] - from[0], to[1] - from[1]];
    let chord_length = chord[0].hypot(chord[1]);
    // A NaN chord fails both halves, which is the point: a length that cannot be compared is not a
    // length this can divide by.
    if !chord_length.is_finite() || chord_length <= f64::EPSILON {
        return None;
    }
    let along = [chord[0] / chord_length, chord[1] / chord_length];
    let mid = [(from[0] + to[0]) / 2.0, (from[1] + to[1]) / 2.0];
    let drift = (center[0] - mid[0]) * along[0] + (center[1] - mid[1]) * along[1];
    let seat = [center[0] - along[0] * drift, center[1] - along[1] * drift];
    (seat[0].is_finite() && seat[1].is_finite()).then_some(seat)
}

/// The counter-clockwise turn from `from` to `to` about `center`, in degrees strictly inside
/// `(0, 360)`. `None` where there is no turn to read: an end standing on the center, or the two
/// ends at one bearing from it.
///
/// There is no sign to return. An arc runs counter-clockwise by definition (ADR 0038) and the
/// endpoint ORDER is what says which way it goes, so the caller that wants the other arc over the
/// same chord asks about the same three points with the ends swapped.
fn counter_clockwise_sweep_degrees(center: [f64; 2], from: [f64; 2], to: [f64; 2]) -> Option<f64> {
    let bearing = |at: [f64; 2]| {
        let (run, rise) = (at[0] - center[0], at[1] - center[1]);
        (run.hypot(rise) > f64::EPSILON).then(|| rise.atan2(run).to_degrees())
    };
    let turn = (bearing(to)? - bearing(from)?).rem_euclid(360.0);
    arc_sweep_is_valid(turn).then_some(turn)
}

/// Whether `at` stands on the DRAWN piece of `geometry`, closely enough to call them met.
///
/// The drawn piece and not the support, which is the opposite of what
/// [`parametric::sketch::Relation::PointOnCurve`] does, and deliberately. That relation is a
/// RESIDUAL the optimizer walks, so a test that had to report "off the end" would hand it a cliff
/// at the endpoint. This is a yes/no ADMISSION GATE, run once when the author asks for the
/// relation — and there a discontinuity at the endpoint is exactly the answer, because either the
/// author drew the two things touching or they did not. Reading the support here let a curvature
/// row be asserted against a phantom: a joint five hundred voxels past a segment's end but
/// collinear with it, or anywhere on the full circle a twenty-degree arc was cut from.
///
/// The tolerance is tight on purpose. Two curves either meet or they do not; "nearly meets" is the
/// state the author is trying to leave by asserting the relation, and accepting it would let a
/// curvature row be asserted against a curve the joint is merely near — which reads as a solver
/// bug later, not as a mis-pick now. It is tighter than the one a tangent CONTACT is held to, so
/// the shared extent test takes the slack as an argument rather than choosing it.
fn point_stands_on(at: [f64; 2], geometry: parametric::sketch::CurveGeometry) -> bool {
    const MET: f64 = 1.0e-6;
    let on_the_support = match geometry {
        parametric::sketch::CurveGeometry::Segment { from, to } => {
            let along = [to[0] - from[0], to[1] - from[1]];
            let length = along[0].hypot(along[1]);
            if length == 0.0 {
                return false;
            }
            let offset = [at[0] - from[0], at[1] - from[1]];
            (along[0] * offset[1] - along[1] * offset[0]).abs() / length <= MET
        }
        parametric::sketch::CurveGeometry::Circular(circle) => {
            ((at[0] - circle.center[0]).hypot(at[1] - circle.center[1]) - circle.radius).abs()
                <= MET
        }
    };
    on_the_support && parametric::sketch::within_drawn_extent(geometry, at, MET)
}

pub fn arc_interior_points(from: [f64; 2], to: [f64; 2], sweep_degrees: f64) -> Vec<SketchPoint> {
    arc_interior_points_within(from, to, sweep_degrees, ARC_SAGITTA_TOLERANCE)
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
    sagitta_tolerance: f64,
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
        sagitta_tolerance,
    )
}

/// The interior points of an ALREADY-SOLVED arc — the circle walked directly, both endpoints
/// exclusive.
///
/// This is the form the closed case needs. Recovering a circle from endpoints plus a bulge is a
/// chord solve, and a whole turn has no chord; carrying the solved center and radius instead means
/// a circle tessellates by the same rule as every other arc rather than by a special case.
fn arc_interior_on_circle(arc: ProfileArc, sagitta_tolerance: f64) -> Vec<SketchPoint> {
    let chords = arc_chord_count(
        arc.radius,
        arc.sweep_radians.to_degrees(),
        sagitta_tolerance,
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
