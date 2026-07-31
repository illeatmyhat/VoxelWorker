//! 2D **sketch → extrude → volume** — the sketch-to-volume authoring atom
//! (ADR 0003 §3i, Slice 2a).
//!
//! This is a SECOND [`VoxelProducer`](crate::voxel::VoxelProducer), added
//! **alongside** [`SdfShape`](crate::voxel::SdfShape) (NOT replacing it). It takes
//! a grid-aligned plane plus a closed polygon *profile* of voxel-granular points
//! and extrudes that profile a whole number of voxels along the plane normal,
//! producing a prism. It is the engine the §3i build arc reframes primitives as
//! sugar over — a rectangle profile extruded *is* a box, a circle profile extruded
//! *is* a cylinder — so it resolves through the SAME stamp / `CombineOp` / chunk
//! path the SDF producer already uses.
//!
//! **Leak-free by construction (§3i leak-retirement).** The profile points and the
//! extrude span are integer voxels on the lattice/sub-lattice — there is no
//! implicit centre anchor and so no half-block correction. The producer samples
//! CORNER-ANCHORED: the resolve tests the profile at `bbox_min + idx + 0.5` (no
//! `grid/2` centring anywhere — a revolve centres only its two RADIAL axes), and
//! its placement does NOT route through `leaf_lattice_shift_voxels`: a sketch's
//! footprint is corner-anchored, so the block-lattice shift the implicit-centre
//! model needed is identically zero. (The
//! resolve path treats a sketch leaf like a VoxelBody — no intrinsic block size, no
//! lattice snap — see `Scene::resolve_*`.)
//!
//! 2a SCOPE: AXIS-ALIGNED planes only (the normal is one of ±X / ±Y / ±Z). A
//! free-angle sketch plane is the deferred plane-orientation milestone (§3f(a)).
//! The profile is a closed simple polygon (≥3 points); a degenerate profile
//! (fewer than 3 points, or zero area) resolves to nothing rather than panicking.

mod constraint;
mod edges;
mod faces;
mod produce;
mod region_memo;
mod solid;
#[cfg(test)]
mod tests;

pub use constraint::{Constraint, ConstraintKind, ConstraintRefusal};
pub use faces::{Face, FaceKey};
pub use solid::SketchSolid;
pub use substrate::geom2d::LoopRole;
pub use substrate::nonlinear_least_squares::{SolveOutcome, SolveReport};

use parametric::units::{AngleMeasurement, Measurement};

/// Which axis the sketch plane's normal points along — i.e. the axis the profile
/// is EXTRUDED along (ADR 0003 §3i, 2a axis-aligned scope).
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
/// prism is symmetric about its own grid), so 2a stores the bare axis; a signed
/// normal is only meaningful once on-surface sketching (§3i, Slice 2b) needs a
/// facing direction, which is a later concern.
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

/// One vertex of a sketch profile — a 2D point on the plane's in-plane axes (see
/// [`PlaneAxis::in_plane_axes`]), carried as the full node-position representation
/// (#101, mirroring `NodeTransform`, ADR 0027/0029): a canonical integer voxel
/// coordinate, a sub-voxel remainder, and an optionally-retained authored
/// [`Measurement`] per axis.
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
    /// authoring; a voxel/block snap zeroes it (#101).
    #[serde(default)]
    pub offset_local_voxels: [f32; 2],
    /// The RETAINED authored `Length` expression per axis (ADR 0029), or `None` for
    /// a plain snapped point. `SetDensity` re-evaluates a retained expression so a
    /// measurement-authored profile keeps its physical shape across a density
    /// re-target; the canonical `offset_voxels` always wins for geometry.
    #[serde(default)]
    pub offset_measurements: Option<[Measurement; 2]>,
}

impl SketchPoint {
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
    /// `offset_voxels`, the fraction in `offset_local_voxels` (#101 — the
    /// `snap = None` authoring door). A non-finite coordinate is sanitised to zero:
    /// a `NaN` fraction would poison every position-equality the producer guards
    /// no-op commits with.
    pub fn from_continuous(axis0: f64, axis1: f64) -> Self {
        let split = |coord: f64| -> (i64, f32) {
            if !coord.is_finite() {
                return (0, 0.0);
            }
            let floor = coord.floor();
            (floor as i64, (coord - floor) as f32)
        };
        let (voxels_0, local_0) = split(axis0);
        let (voxels_1, local_1) = split(axis1);
        Self {
            offset_voxels: [voxels_0, voxels_1],
            offset_local_voxels: [local_0, local_1],
            offset_measurements: None,
        }
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
    /// remove (#101). Two conversions from one integer truth, not one conversion and a cast.
    pub fn in_plane_measured(&self) -> [f32; 2] {
        [
            self.offset_voxels[0] as f32 + self.offset_local_voxels[0],
            self.offset_voxels[1] as f32 + self.offset_local_voxels[1],
        ]
    }

    /// Whether two points sit at the SAME in-plane position — the coincidence
    /// predicate (coincidence IS shared identity, ADR 0030). Position only: a
    /// retained measurement is provenance, not location, so it never splits two
    /// coincident points into twins.
    pub fn coincides(&self, other: &SketchPoint) -> bool {
        self.offset_voxels == other.offset_voxels
            && self.offset_local_voxels == other.offset_local_voxels
    }

    /// This point re-targeted from `old_density` to `new_density` (#101, the
    /// `SetDensity` arm). A retained measurement RE-EVALUATES at the new density
    /// (lossless block scaling; a non-dividing axis floors and resynthesises its
    /// retained form, exactly `NodeTransform::from_measurements`). A plain point
    /// rescales its continuous position so it keeps its physical place, the way the
    /// legacy node rescale keeps a non-parametric offset's.
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

/// A scalar sketch length — a circle's radius today, whatever else the tool suite dimensions
/// later (ADR 0035 Decision 7). The one-dimensional twin of [`SketchPoint`], carried the same way
/// for the same reasons: a canonical integer voxel count, a sub-voxel remainder, and an optionally
/// retained authored [`Measurement`].
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
    /// The RETAINED authored expression (ADR 0029), or `None` for a plain snapped length.
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

    /// A CONTINUOUS length: floor lands in [`voxels`](Self::voxels), the fraction in
    /// [`local_voxels`](Self::local_voxels). A non-finite input sanitises to zero, the same
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
/// would silently corrupt every reference; a stable id does not (ADR 0030). Ids are
/// handed out once and never reused.
pub type EntityId = u32;

/// Whether an entity is real geometry or a construction/reference line that never bounds
/// a region (ADR 0030). Reserved: the toggle UI is a later slice, but the field rides the
/// document now so it costs no second migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum EntityRole {
    /// Real geometry — participates in region derivation.
    #[default]
    Real,
    /// Reference geometry — never bounds a region.
    Construction,
}

/// One loop of the profile: a closed boundary of [`ProfileEdge`]s plus how it contributes to the
/// region (ADR 0030 §4). The unit the 2D CSG folds and the unit the overlay draws.
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
}

/// The circle a curved [`ProfileEdge`] follows, solved once from the canonical endpoints-plus-bulge
/// form (ADR 0030 §5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfileArc {
    /// The circle's centre, in profile voxels.
    pub centre: [f64; 2],
    /// The circle's radius, in voxels.
    pub radius: f64,
    /// The bearing of the edge's tail from the centre.
    pub start_radians: f64,
    /// The signed angle travelled tail → head; positive counter-clockwise.
    pub sweep_radians: f64,
}

impl ProfileEdge {
    /// A straight span.
    pub fn straight(from: SketchPoint, to: SketchPoint) -> Self {
        ProfileEdge {
            from,
            to,
            arc: None,
        }
    }

    /// An arc through the signed `sweep_degrees`, or the plain chord when the sweep is degenerate
    /// — the same fallback [`arc_interior_points`] makes by returning nothing.
    pub fn curved(from: SketchPoint, to: SketchPoint, sweep_degrees: f64) -> Self {
        let Some((centre, radius)) =
            arc_center_radius(from.in_plane(), to.in_plane(), sweep_degrees)
        else {
            return ProfileEdge::straight(from, to);
        };
        let tail = from.in_plane();
        ProfileEdge {
            from,
            to,
            arc: Some(ProfileArc {
                centre,
                radius,
                start_radians: (tail[1] - centre[1]).atan2(tail[0] - centre[0]),
                sweep_radians: sweep_degrees.to_radians(),
            }),
        }
    }

    /// A whole circle as ONE closed edge (ADR 0035 Decision 7): tail and head are the same point,
    /// and the arc sweeps a full turn counter-clockwise about `centre`.
    ///
    /// The seam sits at bearing zero — `centre + [radius, 0]` — matching
    /// [`substrate::geom2d::RegionEdge`]'s convention so the CPU field and its WGSL mirror cut the
    /// circle in the same place. It is a seam and not a vertex: the document holds no [`Point`]
    /// there, nothing may snap to it, and moving the circle moves it with no trace.
    pub fn circle(centre: [f64; 2], radius: f64) -> Self {
        let seam = SketchPoint::from_continuous(centre[0] + radius, centre[1]);
        ProfileEdge {
            from: seam,
            to: seam,
            arc: Some(ProfileArc {
                centre,
                radius,
                start_radians: 0.0,
                sweep_radians: std::f64::consts::TAU,
            }),
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
        }
    }

    /// The direction the edge LEAVES its tail in, as an angle in `(-pi, pi]`. An arc departs along
    /// its tangent — a quarter turn off the radius, on the side it curves toward — which is what
    /// makes two arcs sharing an endpoint order correctly around that vertex.
    pub fn departure_radians(&self) -> f64 {
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
    /// `½∮(x dy − y dx)`. **Exact for an arc**: integrating the parameterised circle gives
    /// `½[r²·sweep + cx·Δy − cy·Δx]`, so a bulge contributes the area it really encloses rather
    /// than the area of the chords that used to stand in for it.
    pub fn signed_area_term(&self) -> f64 {
        let (from, to) = (self.from.in_plane(), self.to.in_plane());
        match self.arc {
            Some(arc) => {
                0.5 * (arc.radius * arc.radius * arc.sweep_radians
                    + arc.centre[0] * (to[1] - from[1])
                    - arc.centre[1] * (to[0] - from[0]))
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
        match self.arc {
            Some(arc) => substrate::geom2d::RegionEdge::Arc {
                start,
                end,
                centre: [arc.centre[0] as f32, arc.centre[1] as f32],
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
                    arc.centre[0] + arc.radius * bearing.cos(),
                    arc.centre[1] + arc.radius * bearing.sin(),
                ];
                for axis in 0..2 {
                    low[axis] = low[axis].min(reach[axis]);
                    high[axis] = high[axis].max(reach[axis]);
                }
            }
        }
        (low, high)
    }
}

/// A point entity: a first-class, independently add/delete-able vertex on the sketch
/// plane, referenced by segments (and later arcs) through its stable [`id`](Self::id)
/// (ADR 0030). A point with no incident edge is a legal FREE point.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Point {
    /// Stable identity (ADR 0030) — segments reference this, not the point's `Vec` slot.
    pub id: EntityId,
    /// The point's in-plane position (see [`SketchPoint`]).
    pub at: SketchPoint,
    /// Real vs construction geometry (reserved).
    #[serde(default)]
    pub role: EntityRole,
}

/// A line-segment entity joining two [`Point`]s **by id** (ADR 0030). Coincidence IS
/// shared identity: two segments meet because they name the same endpoint point, not
/// because a solver forced their coordinates equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Segment {
    /// Stable identity.
    pub id: EntityId,
    /// Endpoint point id (tail).
    pub from: EntityId,
    /// Endpoint point id (head).
    pub to: EntityId,
    /// Lineage id for region identity across edits (ADR 0030 §3): a fresh segment's
    /// `origin` is its own `id`; on split, both children inherit the parent's `origin`,
    /// so subdividing a loop edge leaves a face's boundary origin-SET unchanged.
    pub origin: EntityId,
    /// Real vs construction geometry (reserved).
    #[serde(default)]
    pub role: EntityRole,
}

/// A circular-arc entity joining two [`Point`]s **by id** (ADR 0030 §5, #102). The
/// canonical stored form is the two endpoints plus one included-angle bulge — compact,
/// unambiguous, fully parametric; centre and radius are DERIVED. Creation tools (the
/// 3-point tool today) compute this form; their extra inputs (the through-point) are
/// consumed at creation, never persisted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Arc {
    /// Stable identity.
    pub id: EntityId,
    /// Endpoint point id (tail).
    pub from: EntityId,
    /// Endpoint point id (head).
    pub to: EntityId,
    /// The signed included angle (ADR 0029's `Angle` kind): the arc sweeps from
    /// [`from`](Self::from) to [`to`](Self::to) **counter-clockwise in the plane's
    /// in-plane basis for a positive angle**, clockwise for a negative one. Magnitude
    /// strictly inside `(0, 360)` — zero and full-turn bulges are degenerate and erased
    /// by [`Sketch::repair`].
    pub bulge: AngleMeasurement,
    /// The [`Point`] entity standing at the arc's centre — a REIFIED derived value. Its
    /// coordinates are recomputed from the endpoints and the bulge by
    /// [`Sketch::sync_arc_centers`] and are never authored directly, but it is a real point
    /// entity with a stable id so it selects, snaps and drags exactly like every other
    /// sketch point. Always [`EntityRole::Construction`]: a centre never bounds a region.
    /// `serde(default)` yields [`ABSENT_CENTER`] for a pre-centre document, which
    /// [`Sketch::repair`] materialises on load.
    #[serde(default = "absent_center")]
    pub center: EntityId,
    /// Lineage id for region identity across edits (ADR 0030 §3), like [`Segment::origin`].
    pub origin: EntityId,
    /// Real vs construction geometry (reserved).
    #[serde(default)]
    pub role: EntityRole,
}

/// A whole-circle entity: a centre [`Point`] **by id** plus a radius (ADR 0035 Decision 7).
///
/// A closed curve is its own loop. There is no on-curve vertex to anchor it to and none is
/// invented — a circle drawn on an empty plane bounds a face immediately, where an arc has to meet
/// something to bound anything. The centre is the handle: dragging it moves the circle, and
/// changing [`radius`](Self::radius) resizes it, so the two authored degrees of freedom are exactly
/// the two the shape has.
///
/// The centre is always [`EntityRole::Construction`] — a centre is not on the boundary, so it never
/// bounds a region, exactly as an [`Arc`]'s centre does not.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Circle {
    /// Stable identity.
    pub id: EntityId,
    /// The [`Point`] entity at the circle's centre. Unlike an [`Arc`]'s centre this is AUTHORED,
    /// not derived: it is where the author put it, and nothing recomputes it.
    pub center: EntityId,
    /// The radius, in voxels, optionally retaining the authored expression.
    pub radius: SketchLength,
    /// Lineage id for region identity across edits (ADR 0030 §3), like [`Segment::origin`].
    pub origin: EntityId,
    /// Real vs construction geometry.
    #[serde(default)]
    pub role: EntityRole,
}

/// The `center` of an arc that has no centre point yet — a pre-centre document, or an arc
/// mid-construction. Ids are handed out monotonically from zero and never reused, so the top
/// of the range can never collide with a live entity.
pub const ABSENT_CENTER: EntityId = EntityId::MAX;

/// A span the drawing needs — a segment's length, an arc's chord or its radius — that closes to
/// less than this (in-plane voxels) has collapsed: the entity is no longer what the store calls
/// it. Far below the 1/256-block granularity a polygon is flattened at (ADR 0019), so nothing an
/// author draws can land under it by accident.
const COLLAPSED_SPAN: f64 = 1e-6;

/// A trial solve whose residuals close to under this (the Euclidean norm, in in-plane voxels) has
/// **met the constraints**, whatever stopped the search.
///
/// The solver's own `Converged` flag is not the test, and reading it as one was a real bug. Its
/// residual tolerance is absolute while its step tolerance is relative to the size of the
/// parameter vector, so on a drawing with enough geometry in it the step test fires first: the
/// search stops with the residuals at, say, 1.7e-10 voxels — satisfied by any measure this
/// document can express — and reports `Stalled`, which read as "unsatisfiable" and refused the
/// constraint. Two unrelated free points elsewhere in the sketch were enough to trigger it, which
/// is to say it fired on nearly every real drawing (owner 2026-07-30).
///
/// So the question asked here is the one that is actually about the answer: are the residuals
/// met? `SolveOutcome` says why the search stopped, which is a fact about the search. The same
/// scale as [`COLLAPSED_SPAN`], and for the same reason — it is four orders below the 1/256-block
/// granularity a profile is flattened at, so a residual under it cannot move a single voxel.
const SATISFIED_RESIDUAL: f64 = 1e-6;

/// One trial solve on a copy of the drawing: what it produced, and whether that is acceptable.
struct Trial {
    points: Vec<Point>,
    verdict: TrialVerdict,
}

/// How a trial solve turned out. Three outcomes rather than two, because "converged" and
/// "acceptable" are different questions (ADR 0035 Decision 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrialVerdict {
    /// A real drawing that meets every assertion.
    Solved,
    /// No solution was reached: the assertions fight.
    Diverged,
    /// A solution WAS reached, and it squeezes this entity to nothing.
    Collapsed(EntityId),
}

fn absent_center() -> EntityId {
    ABSENT_CENTER
}

/// A grid-aligned PLANE plus a collection of sketch ENTITIES — points and segments
/// (arcs, region picks, and sub-voxel/parametric coordinates arrive in later slices,
/// ADR 0030). The extrudable **profile is DERIVED** from the closed loop the segments
/// form (see [`flattened_loop`](Self::flattened_loop)); it is no longer a hand-maintained
/// ordered vertex list.
///
/// **Slice-1 scope (issue #98):** a single closed loop, resolving byte-identical to the
/// former `profile: Vec<SketchPoint>`. Multi-region pick/unpick (#100), sub-voxel /
/// parametric coordinates (#101), and arcs (#102) build on this store.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Sketch {
    /// Which axis the plane normal points along (2a: axis-aligned only).
    pub plane: PlaneAxis,
    /// The point entities (unordered; loop order is derived, never stored).
    points: Vec<Point>,
    /// The segment entities joining points by id.
    segments: Vec<Segment>,
    /// The arc entities joining points by id (#102). `serde(default)` so a pre-arc
    /// document loads with none.
    #[serde(default)]
    arcs: Vec<Arc>,
    /// The whole-circle entities (ADR 0035 Decision 7). `serde(default)` so a pre-circle
    /// document loads with none.
    #[serde(default)]
    circles: Vec<Circle>,
    /// The faces the author has UNPICKED, each named by a point inside it (ADR 0035
    /// Decision 9). Every derived face is picked by default, so this holds only the
    /// exceptions and is usually empty. A point inside no current face is inert, not an
    /// error: it costs nothing and lets an unpick survive an edit that temporarily breaks
    /// its boundary.
    ///
    /// It is a `Vec` and not a set because `f32` is not `Ord`, and the field is renamed from
    /// the origin-set `unpicked` it replaces so a pre-arrangement document loads with every
    /// face picked rather than failing on a key it cannot parse.
    #[serde(default)]
    unpicked_points: Vec<FaceKey>,
    /// The constraint entities (ADR 0035 Decision 3). `serde(default)` so a pre-constraint
    /// document loads with none.
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

    /// A rectangle profile spanning `[0, width] × [0, height]` voxels on `plane`
    /// (the degenerate "box footprint" — proves box = rectangle-extrude sugar,
    /// §3i). The four corners are wound counter-clockwise; winding does not affect
    /// the even-odd rasterizer.
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

    /// An empty sketch on `plane` — no entities. A totally-empty sketch is first-class
    /// (ADR 0030): it is a valid scene object that resolves to nothing, the start state a
    /// create-from-scratch sketch is authored into.
    pub fn empty(plane: PlaneAxis) -> Self {
        Self {
            plane,
            points: Vec::new(),
            segments: Vec::new(),
            arcs: Vec::new(),
            circles: Vec::new(),
            unpicked_points: Vec::new(),
            constraints: Vec::new(),
            next_id: 0,
            region_memo: region_memo::RegionMemo::default(),
        }
    }

    /// A sketch on `plane` holding ONE circle of `radius_voxels` about `centre` — the circle twin
    /// of [`rectangle`](Self::rectangle), and the shortest path to a profile with no straight edge
    /// in it at all.
    pub fn circle(plane: PlaneAxis, centre: SketchPoint, radius_voxels: i64) -> Self {
        let mut sketch = Self::empty(plane);
        sketch.add_circle(centre, SketchLength::new(radius_voxels));
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

    /// Read-only view of the arc entities (#102).
    pub fn arcs(&self) -> &[Arc] {
        &self.arcs
    }

    /// Read-only view of the whole-circle entities (ADR 0035 Decision 7).
    pub fn circles(&self) -> &[Circle] {
        &self.circles
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

    /// The DERIVED bounded faces of the sketch's planar graph (ADR 0030 §2, #100), in a
    /// deterministic order. Every face is a candidate region; whether it contributes solid or
    /// void is [`face_is_picked`](Self::face_is_picked).
    pub fn faces(&self) -> Vec<Face> {
        faces::derive(self)
    }

    /// The region in the **measurement** width — the exact value
    /// [`substrate::geom2d::signed_distance_to_region`] folds, and the exact value the wash's
    /// WGSL mirror is handed (ADR 0030 §3).
    ///
    /// One definition of the region, two evaluators of it: the resolve asks it per voxel on the
    /// CPU, the overlay asks it per pixel on the GPU. Curves arrive as curves, so neither is
    /// drawing a polygon the other chose the resolution of.
    pub fn region_field_loops(&self) -> Vec<(LoopRole, Vec<substrate::geom2d::RegionEdge>)> {
        self.derived().region_field_loops.clone()
    }

    /// The `Fill` loops' bounding box in voxels — the profile's FOOTPRINT, and what the producer
    /// sizes its grid from. `None` when nothing is filled.
    pub(super) fn filled_extent(&self) -> Option<([f64; 2], [f64; 2])> {
        self.derived().filled_extent
    }

    /// Whether the face containing this key's point contributes solid. Faces default to PICKED —
    /// the document stores only the unpicked exceptions (ADR 0030 §3, ADR 0035 Decision 9).
    pub fn face_is_picked(&self, key: &FaceKey) -> bool {
        let faces = self.nested_faces();
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
    pub fn face_key_at(&self, index: usize) -> Option<FaceKey> {
        let faces = faces::derive(self);
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
    pub fn identified_faces(&self) -> Vec<(Face, FaceKey)> {
        let faces = faces::derive(self);
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
    /// ground under the point, while a face that shrinks past it reverts to picked
    /// (ADR 0035 Decision 9).
    pub fn set_face_picked(&mut self, key: FaceKey, picked: bool) {
        let faces = self.nested_faces();
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
    fn nested_faces(&self) -> Vec<Face> {
        let mut faces = faces::derive(self);
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
    /// `Hole` where it is not (ADR 0030 §4), each a closed loop of edges **with its arcs intact**,
    /// ordered SMALLEST-AREA-FIRST.
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
    pub fn region(&self) -> Vec<ProfileLoop> {
        self.derived().region.clone()
    }

    /// The derived region, its measurement-width twin, and the filled extent, from the cache when
    /// the entity store has not moved — the door every per-voxel path goes through
    /// (see [`region_memo`]).
    pub(super) fn derived(&self) -> std::sync::Arc<region_memo::Derived> {
        self.region_memo.derived(self)
    }

    /// The region derived from scratch. Only [`region_memo`] calls this; everything else asks
    /// [`region`](Self::region) and gets the same answer without re-deriving it.
    fn region_uncached(&self) -> Vec<ProfileLoop> {
        let faces = self.nested_faces();
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
    pub fn filled_loops(&self) -> Vec<ProfileLoop> {
        self.region()
            .into_iter()
            .filter(|profile_loop| profile_loop.role == LoopRole::Fill)
            .collect()
    }

    /// The SIMPLE-profile door: the sole boundary when the region is exactly one picked face,
    /// flattened at the default tolerance, and empty otherwise (no face, an unpicked one, or
    /// several — those are questions only [`region`](Self::region) can answer). Callers that reason
    /// about a single closed outline (rectangle detection, most tests) want this; anything that
    /// resolves occupancy wants the region.
    pub fn flattened_loop(&self) -> Vec<SketchPoint> {
        let loops = self.region();
        match (loops.len(), loops.first().map(|first| first.role)) {
            (1, Some(LoopRole::Fill)) => loops[0].flatten(ARC_SAGITTA_TOLERANCE_VOXELS),
            _ => Vec::new(),
        }
    }

    /// Move the point `id` to `at` and settle the drawing around it — the drag write path.
    /// Reports whether the point exists.
    ///
    /// Dragging an arc's CENTRE moves only the centre: the endpoints hold still and the arc's
    /// radius follows the cursor ([`resweep_arc_to_center`](Self::resweep_arc_to_center)). Every
    /// other point simply takes `at`, and then the standing constraints are re-solved with it
    /// pinned there — see [`settle_under_the_hand`](Self::settle_under_the_hand). A constraint
    /// that only held at the moment it was asserted is not a constraint; it has to survive the
    /// next drag, which is the first thing the author does to test it (owner 2026-07-30).
    pub fn move_point(&mut self, id: EntityId, at: SketchPoint) -> bool {
        let Some(index) = self.point_index(id) else {
            return false;
        };
        match self.arcs.iter().position(|arc| arc.center == id) {
            // An arc's centre is DERIVED from its ends and its sweep, so there is no pinning it:
            // the resweep is the whole edit and no constraint can hold the result anywhere else.
            Some(arc_index) => {
                self.resweep_arc_to_center(arc_index, at.in_plane());
                self.sync_arc_centers();
            }
            None => {
                let before = self.points.clone();
                self.points[index].at = at;
                self.sync_arc_centers();
                if !self.settle_under_the_hand(id, at) {
                    self.points = before;
                }
            }
        }
        true
    }

    /// Re-solve the standing constraints with the hand pulling `held` toward `at`, writing the
    /// result back only if the standing residuals are met. Reports whether they were.
    ///
    /// This is the live tier of ADR 0035 Decision 11: the assertions hold DURING the gesture, not
    /// merely at the moment they were made.
    ///
    /// **The hand is a PULL, not a demand — two stages.** The drag joins the system as one more
    /// least-squares row and the solve trades it off against everything standing; then the hand
    /// lets go and the standing system alone is re-solved from that answer, which restores it
    /// exactly while moving as little as it can. The grabbed point therefore lands at the nearest
    /// place the drawing allows, and only the standing residuals decide whether the drag stands.
    ///
    /// It shipped as a hard pin, and that was the bug (owner, 2026-07-31). A hard pin makes the
    /// whole drag all-or-nothing: a point free to slide along a line but not across it could not be
    /// moved AT ALL, because the cursor is essentially never exactly on that line and the pinned
    /// system was refused as unsatisfiable. The reported case was a vertical segment whose far end
    /// was held by an arc that two `Fix`es had already determined — one real freedom left, its
    /// length, and no way to use it. Sliding along the allowed direction is what every CAD tool
    /// does and what the freedom count already promises.
    ///
    /// A drag that IS achievable is unaffected: stage one meets the pull exactly, so stage two
    /// starts at a solution and moves nothing.
    fn settle_under_the_hand(&mut self, held: EntityId, at: SketchPoint) -> bool {
        if self.constraints.is_empty() {
            return true;
        }
        let ends = self.segment_ends();
        let centers = self.arc_centers();
        let mut pulled = self.constraints.clone();
        pulled.push(Constraint {
            id: self.next_id,
            kind: ConstraintKind::Fix { point: held, at },
            redundant: false,
        });
        let mut points = self.points.clone();
        constraint::solve_in_place(&mut points, &ends, &centers, &pulled);
        let report = constraint::solve_in_place(&mut points, &ends, &centers, &self.constraints);
        // Judged on the RESIDUALS, never on why the search stopped — see [`SATISFIED_RESIDUAL`].
        if report.is_some_and(|report| report.residual_norm > SATISFIED_RESIDUAL) {
            return false;
        }
        self.points = points;
        self.sync_arc_centers();
        true
    }

    /// Re-solve the arc at `arc_index` so its centre sits as close to `target` as the canonical
    /// form allows, its endpoints unmoved.
    ///
    /// For a fixed chord a centre has ONE degree of freedom, not two: it lives on the chord's
    /// perpendicular bisector, and where it sits along that line IS the sweep — far out for a
    /// shallow arc, on the chord for a half turn, across to the other side for the major one.
    /// So the drag projects onto the bisector and inverts `arc_center_radius`: the signed
    /// apothem `a` and the half-chord `h` give `sweep / 2 = atan2(h, a)`, which covers every
    /// positive sweep in `(0°, 360°)` as `a` runs over the reals. The existing sweep's SIGN is
    /// preserved — it says which way round the arc goes, and a drag of the centre is not a
    /// request to reverse it. A degenerate chord or a sweep that quantises to nothing leaves
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
        if arc.bulge.to_degrees_f64() < 0.0 {
            degrees -= 360.0;
        }
        let Some(bulge) = AngleMeasurement::from_degrees_f64(degrees) else {
            return;
        };
        if arc_sweep_is_valid(bulge.to_degrees_f64()) {
            self.arcs[arc_index].bulge = bulge;
        }
    }

    /// Delete a point by id and every segment/arc incident to it (ADR 0030 §6). The
    /// edges' other endpoints survive as free points. No dangling reference can result.
    /// Deleting an arc's CENTRE deletes that arc: the centre is the arc's own derived
    /// geometry, so there is no arc left for it to be the centre of.
    pub fn delete_point_cascade(&mut self, id: EntityId) {
        self.segments.retain(|seg| seg.from != id && seg.to != id);
        self.arcs
            .retain(|arc| arc.from != id && arc.to != id && arc.center != id);
        // A circle IS its centre plus a radius, so deleting the centre deletes the circle.
        self.circles.retain(|circle| circle.center != id);
        self.points.retain(|point| point.id != id);
        self.prune_orphan_centers();
        self.drop_dangling_constraints();
    }

    /// Delete the segment with id `seg_id`, **and each of its ends that nothing else draws**.
    /// No-op if `seg_id` is unknown.
    ///
    /// The ends used to survive unconditionally as free points, and that was wrong (owner,
    /// 2026-07-31): a line deleted from a drawing left two dots behind that the author had never
    /// placed and had no reason to want. A point the author *did* place stays — it is either an
    /// end of some other edge, an arc's center, or a circle's, and [`point_is_still_drawn`] asks
    /// exactly that question.
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

    /// The endpoint pairs the residual system needs, as `(segment id, from, to)`.
    fn segment_ends(&self) -> Vec<(EntityId, EntityId, EntityId)> {
        self.segments
            .iter()
            .map(|seg| (seg.id, seg.from, seg.to))
            .collect()
    }

    /// Add a constraint, trial-solving before it is kept (ADR 0035 Decision 4).
    ///
    /// **Unsatisfiable is refused** and nothing changes, so the system is always solvable and every
    /// downstream feature gets to assume it rather than defend against it. **Redundant is accepted
    /// and flagged**: a solution exists but the Jacobian loses rank, and redundancy is sometimes
    /// the intent — symmetry asserted although the geometry already implies it is insurance
    /// against a later edit.
    ///
    /// The trial runs on a copy, so a refusal leaves the drawing exactly where it was rather than
    /// where a failed solve pushed it.
    pub fn add_constraint(&mut self, kind: ConstraintKind) -> Result<EntityId, ConstraintRefusal> {
        self.check_names_live_geometry(kind)?;
        self.check_is_not_already_asserted(kind)?;
        // The id is minted only once the trial has passed, so a refusal leaves the id space
        // untouched rather than burning a number nothing will ever name.
        let candidate = Constraint {
            id: self.next_id,
            kind,
            redundant: false,
        };

        let mut with_candidate = self.constraints.clone();
        with_candidate.push(candidate);
        let trial = self.trial(&with_candidate);
        match trial.verdict {
            TrialVerdict::Diverged => {
                return Err(ConstraintRefusal::Unsatisfiable {
                    fights: self.blame(candidate),
                })
            }
            TrialVerdict::Collapsed(entity) => {
                return Err(ConstraintRefusal::WouldCollapse {
                    entity,
                    implicated: self.constraints_acting_on(entity),
                })
            }
            TrialVerdict::Solved => {}
        }

        // Rank is measured against what the system knew a moment ago: if the new constraint did
        // not raise it, everything it says was already being said. Both readings are taken at the
        // author's PRE-solve drawing (`constraint::witness_rank`) rather than at each system's own
        // solution, which is what keeps a vanishing Jacobian row from reading as redundancy.
        let ends = self.segment_ends();
        let centers = self.arc_centers();
        let witness = |constraints: &[Constraint]| {
            constraint::witness_rank(&self.points, &ends, &centers, constraints)
        };
        let redundant = witness(&with_candidate) <= witness(&self.constraints);
        let id = self.alloc_id();
        self.constraints.push(Constraint {
            id,
            redundant,
            ..candidate
        });
        self.points = trial.points;
        Ok(id)
    }

    /// The standing constraints that act on `entity` — the ones holding the shape that is about
    /// to be squeezed to nothing.
    ///
    /// Asked structurally rather than by experiment because leave-one-out cannot answer it: an
    /// earlier solve has already moved the drawing, and releasing an assertion does not undo its
    /// effect, so dropping the `Horizontal` that levelled a segment leaves the segment level and
    /// `Vertical` still collapses it. "What else is holding this?" is a question about the graph,
    /// and it always has an answer.
    ///
    /// A constraint counts if it names the entity itself or either of its ends.
    fn constraints_acting_on(&self, entity: EntityId) -> Vec<EntityId> {
        let ends: Vec<EntityId> = self
            .segments
            .iter()
            .filter(|seg| seg.id == entity)
            .flat_map(|seg| [seg.from, seg.to])
            .chain(
                self.arcs
                    .iter()
                    .filter(|arc| arc.id == entity)
                    .flat_map(|arc| [arc.from, arc.to, arc.center]),
            )
            .collect();
        self.constraints
            .iter()
            .filter(|held| {
                held.kind.segments().contains(&entity)
                    || held.kind.points().iter().any(|named| ends.contains(named))
            })
            .map(|held| held.id)
            .collect()
    }

    /// Which standing constraints the candidate cannot coexist with — **leave-one-out**.
    ///
    /// Re-run the trial with each standing constraint dropped in turn; any drop that lets the
    /// system succeed names a culprit. That is `n` solves of a system with at most a few dozen
    /// parameters, which at sketch scale is free, and it is an ANSWER rather than an estimate:
    /// the alternative in the literature is a rank heuristic that picks the constraint appearing
    /// in the most dependent groups, and it is known to blame the wrong one.
    ///
    /// An empty result means no SINGLE removal helps — a conflict needing two, or one whose
    /// effect on the geometry outlived the assertion that caused it. Saying nothing is right
    /// there; naming an arbitrary member would send the author to delete something innocent.
    fn blame(&self, candidate: Constraint) -> Vec<EntityId> {
        self.constraints
            .iter()
            .filter(|standing| {
                let mut without: Vec<Constraint> = self
                    .constraints
                    .iter()
                    .filter(|held| held.id != standing.id)
                    .copied()
                    .collect();
                without.push(candidate);
                self.trial(&without).verdict == TrialVerdict::Solved
            })
            .map(|standing| standing.id)
            .collect()
    }

    /// Solve `constraints` on a COPY of the drawing and judge the result. The copy is what lets a
    /// refusal leave the sketch exactly where it was rather than where a failed solve pushed it.
    ///
    /// The judgement is on the RESIDUALS, not on why the search stopped — see
    /// [`SATISFIED_RESIDUAL`], which is where that distinction is argued and where the bug that
    /// came of confusing the two is recorded.
    fn trial(&self, constraints: &[Constraint]) -> Trial {
        let mut points = self.points.clone();
        let report = constraint::solve_in_place(
            &mut points,
            &self.segment_ends(),
            &self.arc_centers(),
            constraints,
        );
        let verdict = match report {
            // Nothing to solve is not a failure: an empty system is met by the drawing as it is.
            None => TrialVerdict::Solved,
            Some(report) if report.residual_norm > SATISFIED_RESIDUAL => TrialVerdict::Diverged,
            Some(_) => match self.collapsed_by(&points) {
                Some(entity) => TrialVerdict::Collapsed(entity),
                None => TrialVerdict::Solved,
            },
        };
        Trial { points, verdict }
    }

    /// **One constraint of a kind per entity set** (ADR 0035 Decision 4).
    ///
    /// Stacking `Horizontal` on a segment that is already asserted horizontal adds a residual that
    /// says exactly what another residual already says. The rank test below would catch it and
    /// flag it `redundant`, but flagging is for redundancy that carries INTENT — symmetry asserted
    /// although the geometry already implies it, kept as insurance against a later edit. A literal
    /// second copy of the same claim carries none: it cannot be told from the first, deleting
    /// either leaves the drawing identically constrained, and two badges would stand on one
    /// anchor saying one thing. So it is refused, and the author already has what they asked for.
    ///
    /// The comparison is on the KIND and the geometry, never the value: a second `Fix` on a fixed
    /// point is refused whether or not it names the same place, because "fix this here, and also
    /// there" is a re-fix — delete the first, assert the second — rather than two live claims.
    fn check_is_not_already_asserted(&self, kind: ConstraintKind) -> Result<(), ConstraintRefusal> {
        let standing = self
            .constraints
            .iter()
            .find(|held| held.kind.is_about_the_same_as(kind));
        match standing {
            Some(held) => Err(ConstraintRefusal::AlreadyAsserted { existing: held.id }),
            None => Ok(()),
        }
    }

    /// WHICH geometry the trial solve collapsed, if it collapsed any — geometry that had extent
    /// before the solve ran and has none after.
    ///
    /// **A singularity solves everything.** Almost every residual in the set is a difference
    /// between two coordinates, so putting every point in one place drives them all to zero: it is
    /// a trivial solution, available to nearly any system, and the solver will happily converge on
    /// it. `Horizontal` and `Vertical` on one segment is the smallest instance — the zero-length
    /// segment satisfies both exactly — but the shape of the failure is general, so the test is
    /// too. A drawing that met its assertions by deleting the thing they name has not met them.
    ///
    /// Checked as a property of the RESULT rather than as a table of forbidden pairs, so it covers
    /// combinations the residual set does not have yet. What must stay open is every span the
    /// drawing needs to still be itself: a segment's two ends, an arc's two ends, and an arc's
    /// radius.
    ///
    /// Geometry that was ALREADY degenerate is not this test's business — an unrelated assertion
    /// elsewhere should not be refused for a collapse that predates it.
    fn collapsed_by(&self, trial: &[Point]) -> Option<EntityId> {
        let span = |points: &[Point], from: EntityId, to: EntityId| -> Option<f64> {
            let at = |id: EntityId| points.iter().find(|point| point.id == id).map(|p| p.at);
            let (a, b) = (at(from)?.in_plane(), at(to)?.in_plane());
            Some(((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt())
        };
        // Every span the drawing needs to still be itself, as `(the entity that needs it, its two
        // ends)`.
        let mut open: Vec<(EntityId, EntityId, EntityId)> = self
            .segments
            .iter()
            .map(|seg| (seg.id, seg.from, seg.to))
            .collect();
        for arc in &self.arcs {
            open.push((arc.id, arc.from, arc.to));
            if arc.center != ABSENT_CENTER {
                open.push((arc.id, arc.center, arc.from));
            }
        }
        open.into_iter()
            .find(|&(_, from, to)| {
                let (Some(before), Some(after)) =
                    (span(&self.points, from, to), span(trial, from, to))
                else {
                    return false;
                };
                before > COLLAPSED_SPAN && after <= COLLAPSED_SPAN
            })
            .map(|(entity, _, _)| entity)
    }

    /// Whether every entity `kind` names is in the store, and its own terms are meetable.
    fn check_names_live_geometry(&self, kind: ConstraintKind) -> Result<(), ConstraintRefusal> {
        let known_point = |id: EntityId| self.points.iter().any(|point| point.id == id);
        let live_segment = |id: EntityId| self.segments.iter().find(|seg| seg.id == id);
        match kind {
            ConstraintKind::Fix { point, .. } => {
                if !known_point(point) {
                    return Err(ConstraintRefusal::UnknownEntity);
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
        }
        Ok(())
    }

    /// Delete one constraint by id. The geometry it held stays where the last solve put it —
    /// releasing an assertion does not undo its effect, it only stops re-asserting it.
    pub fn delete_constraint(&mut self, id: EntityId) {
        self.constraints.retain(|constraint| constraint.id != id);
    }

    /// Solve the sketch against its constraints, writing the solution into the points
    /// (ADR 0035 Decision 2's continuous tier; the integer loop sits above it).
    ///
    /// `None` when there is nothing to solve. Solved positions are **authored** state, not
    /// `Derived` (Decision 3): they are the solver's input as well as its output, and an
    /// under-constrained sketch has freedoms only the stored position remembers.
    pub fn solve(&mut self) -> Option<SolveReport> {
        let ends = self.segment_ends();
        let centers = self.arc_centers();
        constraint::solve_in_place(&mut self.points, &ends, &centers, &self.constraints)
    }

    /// What a solve WOULD report, without moving anything.
    pub fn solve_report(&self) -> Option<SolveReport> {
        let mut trial = self.points.clone();
        constraint::solve_in_place(
            &mut trial,
            &self.segment_ends(),
            &self.arc_centers(),
            &self.constraints,
        )
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
    pub fn degrees_of_freedom(&self) -> usize {
        let derived = self
            .points
            .iter()
            .filter(|point| self.is_derived_point(point.id))
            .count();
        let authored = (self.points.len() - derived) * 2;
        match self.solve_report() {
            Some(report) => report.degrees_of_freedom.saturating_sub(derived * 2),
            None => authored,
        }
    }

    /// Drop constraints naming geometry the store no longer holds. Called by every delete, so a
    /// constraint never outlives what it constrains (ADR 0035 Decision 3's cascade).
    fn drop_dangling_constraints(&mut self) {
        let point_ids: Vec<EntityId> = self.points.iter().map(|point| point.id).collect();
        let segment_ids: Vec<EntityId> = self.segments.iter().map(|seg| seg.id).collect();
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
        });
    }

    /// Split the segment with id `seg_id` by inserting a new point `at` on it (ADR 0030
    /// add-point). The first half keeps the segment's id; the new second half inherits its
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

    /// Add a FREE point entity at `at` — no incident segment — returning its fresh id
    /// (ADR 0030: a free point is legal geometry; #99 polyline places one per click and
    /// then connects them). The public door to [`add_point`](Self::add_point).
    pub fn add_free_point(&mut self, at: SketchPoint) -> EntityId {
        self.add_point(at)
    }

    /// Connect two existing points with a fresh segment, returning its id (ADR 0030 —
    /// coincidence is shared point identity, so drawing to an existing point means naming
    /// its id here, never minting a coordinate twin). `None` — and no mutation — for a
    /// self-loop, an unknown endpoint, or a pair a SEGMENT already joins: a straight edge
    /// between two points is unique geometry, so a second one is a duplicate.
    ///
    /// A pair an ARC joins is fine, and is the D-shape (a chord closing a curve). It was
    /// refused until #100 because the single-loop walk of the time could not orient two
    /// edges over one pair; the face derivation that replaced it traces the two-edge cycle
    /// like any other, so the restriction went with the walk it was protecting.
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

    /// Connect two existing points with a fresh arc of the given signed included angle
    /// (#102), returning its id. `None` — and no mutation — for a self-loop, an unknown
    /// endpoint, a degenerate bulge (zero or a full turn or more), or an arc that would
    /// trace a curve the store already holds.
    ///
    /// A pair already joined by a segment, or by an arc bulging differently, is legal: a
    /// chord plus its arc is a D, and two arcs over one pair are a lens. Both are ordinary
    /// bounded faces to the derivation (see [`connect`](Self::connect) for why this was
    /// once refused).
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
            bulge,
            center: ABSENT_CENTER,
            origin: id,
            role: EntityRole::Real,
        });
        self.sync_arc_centers();
        Some(id)
    }

    /// Draw a circle of `radius` about a FRESH construction centre at `at`, returning the circle's
    /// id (ADR 0035 Decision 7). `None` — and no mutation — for a non-positive or non-finite
    /// radius, which is not a curve.
    ///
    /// The centre is minted here rather than taken as an id because that is what the centre-radius
    /// tool does: one click plants the centre, the drag sets the radius. Drawing about a point that
    /// already exists is [`circle_about`](Self::circle_about).
    pub fn add_circle(&mut self, at: SketchPoint, radius: SketchLength) -> Option<EntityId> {
        if !circle_radius_is_valid(radius.value()) {
            return None;
        }
        let center = self.add_construction_point(at);
        self.push_circle(center, radius)
    }

    /// Draw a circle of `radius` about the EXISTING point `center`, returning its id. `None` for an
    /// unknown point, an invalid radius, or a circle the store already holds about that centre at
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
        self.circles.push(Circle {
            id,
            center,
            radius,
            origin: id,
            role: EntityRole::Real,
        });
        Some(id)
    }

    /// Whether a circle of this radius about this centre is already stored.
    pub fn circle_traces(&self, center: EntityId, radius_voxels: f64) -> bool {
        self.circles
            .iter()
            .any(|circle| circle.center == center && circle.radius.value() == radius_voxels)
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
        self.circles[index].radius = radius;
        true
    }

    /// Delete just the circle with id `circle_id`. Its centre goes with it when nothing else names
    /// it — the centre is the circle's own anchor, so there is no circle left for it to centre.
    pub fn delete_circle(&mut self, circle_id: EntityId) {
        self.circles.retain(|circle| circle.id != circle_id);
        self.prune_orphan_centers();
    }

    /// Whether the drawing OWNS this point's coordinates — today, whether it is an arc's center,
    /// which [`sync_arc_centers`](Self::sync_arc_centers) re-derives from the arc's ends and its
    /// sweep. A derived point is selectable, draggable, snappable and **constrainable** like any
    /// other; what it is not is a freedom, which is why
    /// [`degrees_of_freedom`](Self::degrees_of_freedom) does not count it.
    ///
    /// A constraint naming one is met by moving the ARC — see `constraint::position_of`, where the
    /// residual system reads it as the function it is.
    pub fn is_derived_point(&self, id: EntityId) -> bool {
        self.arcs.iter().any(|arc| arc.center == id)
    }

    /// The derived centers the residual system needs, as `(center, from, to, sweep)`.
    fn arc_centers(&self) -> Vec<constraint::ArcCenter> {
        self.arcs
            .iter()
            .map(|arc| constraint::ArcCenter {
                center: arc.center,
                from: arc.from,
                to: arc.to,
                sweep_degrees: arc.bulge.to_degrees_f64(),
            })
            .collect()
    }

    /// Re-derive every arc's centre point from its endpoints and bulge (ADR 0030 §5), minting
    /// one for any arc that has none yet. The centre is a real [`Point`] so it can be selected,
    /// snapped to and dragged like any other, but its coordinates are OWNED here — every edit
    /// that can move an arc ends by calling this, so a centre can never drift out of agreement
    /// with the curve it belongs to. An arc whose endpoints are missing or coincident is left
    /// alone; [`repair`](Self::repair) erases it.
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
                arc.bulge.to_degrees_f64(),
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

    /// Drop every construction point nothing references any more — the centre of an arc that
    /// has just been deleted. A centre the author has since drawn to (an edge names it) is
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
            let stored = arc.bulge.to_degrees_f64();
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
        self.drop_dangling_constraints();
    }

    /// The lowest-id point entity sitting EXACTLY at `at`'s position, if any. The drawing
    /// tools (#99) check this after snapping a click, so a click that lands on an existing
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

    /// Re-target every point entity from `old_density` to `new_density` (#101 — the
    /// `SetDensity` arm). Per point: a retained measurement re-evaluates losslessly; a
    /// plain point rescales its continuous position ([`SketchPoint::retargeted`]).
    pub fn retarget_density(&mut self, old_density: u32, new_density: u32) {
        for point in &mut self.points {
            point.at = point.at.retargeted(old_density, new_density);
        }
        // A radius is a length like any other: an authored `2 blocks` must stay two blocks.
        for circle in &mut self.circles {
            circle.radius = circle.radius.retargeted(old_density, new_density);
        }
        self.sync_arc_centers();
    }

    /// Erase every structurally-invalid segment or arc — one that references a point id not
    /// in the store, a self-loop (`from == to`), or (arcs) a degenerate bulge — returning
    /// the number removed (ADR 0030 load
    /// policy: erase invalid objects rather than fail the load). Points are never invalid; a
    /// point left with no incident edge is a legal free point. The resolve already tolerates a
    /// dangling reference (the missing vertex is filtered out of the flattened loop), so this
    /// is a cleanup + audit, not a crash guard.
    pub fn repair(&mut self) -> usize {
        let point_ids: Vec<EntityId> = self.points.iter().map(|point| point.id).collect();
        let before = self.segments.len() + self.arcs.len() + self.circles.len();
        self.segments.retain(|seg| {
            seg.from != seg.to && point_ids.contains(&seg.from) && point_ids.contains(&seg.to)
        });
        // An arc is additionally invalid on a degenerate bulge — a zero sweep is a
        // segment pretending, a full turn or more has no single chord-anchored shape.
        self.arcs.retain(|arc| {
            arc.from != arc.to
                && point_ids.contains(&arc.from)
                && point_ids.contains(&arc.to)
                && arc_sweep_is_valid(arc.bulge.to_degrees_f64())
        });
        // A circle is invalid on a missing centre or a radius that is not a positive finite
        // length — either way there is no curve to draw.
        self.circles.retain(|circle| {
            point_ids.contains(&circle.center) && circle_radius_is_valid(circle.radius.value())
        });
        // A constraint naming geometry the store does not hold asserts nothing about anything,
        // and left in place it would keep a row in the residual system for a shape that is gone.
        let before = before + self.constraints.len();
        self.drop_dangling_constraints();
        let dropped = before
            - self.segments.len()
            - self.arcs.len()
            - self.circles.len()
            - self.constraints.len();
        // A pre-centre document names no centre at all, and a just-erased arc leaves one
        // behind; both are settled here, so a loaded sketch always agrees with its arcs.
        self.prune_orphan_centers();
        self.sync_arc_centers();
        dropped
    }
}

/// Default arc flattening tolerance (#102): the maximum sagitta (chord-to-arc deviation), in
/// voxels, of one chord.
///
/// This is no longer the resolved meaning of a curve — the region carries its arcs, and the field
/// measures them ([`ProfileEdge`]). It is the default a **terminal adapter** flattens at when it
/// has to produce something discrete and has nowhere to put a curve: a crease polyline, the
/// exact-`f64` cell classifier's polygon, a test's outline. Nothing downstream of one of those
/// inherits it, so it is a tuning knob again rather than a document-format constant.
pub const ARC_SAGITTA_TOLERANCE_VOXELS: f64 = 1.0 / 16.0;

/// Hard cap on chords per arc, so a huge-radius near-collinear arc cannot degenerate
/// into an unbounded fan.
const ARC_MAX_CHORDS: u32 = 512;

/// Whether a signed sweep is a legal [`Arc`] bulge: finite, non-zero, strictly under a full
/// turn in magnitude.
///
/// The full turn stays excluded ON PURPOSE. A closed curve is a [`Circle`] — a centre and a radius
/// — not an arc bulged all the way round (ADR 0035 Decision 7): the endpoint-plus-bulge form
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

/// The centre and radius DERIVED from the canonical arc form (ADR 0030 §5): endpoints
/// plus signed sweep, positive sweeping counter-clockwise about the centre. `None` for a
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
    // The centre sits on the chord's perpendicular bisector at the signed apothem: the
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
/// exclusive), as continuous sub-voxel points (#101), each chord's sagitta within
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
/// region carries its arcs and the field measures them (ADR 0034). Every caller here is a
/// terminal adapter, so no tolerance chosen at one reaches anything downstream of it.
pub fn arc_interior_points_within(
    from: [f64; 2],
    to: [f64; 2],
    sweep_degrees: f64,
    sagitta_tolerance_voxels: f64,
) -> Vec<SketchPoint> {
    let Some((centre, radius)) = arc_center_radius(from, to, sweep_degrees) else {
        return Vec::new();
    };
    arc_interior_on_circle(
        ProfileArc {
            centre,
            radius,
            start_radians: (from[1] - centre[1]).atan2(from[0] - centre[0]),
            sweep_radians: sweep_degrees.to_radians(),
        },
        sagitta_tolerance_voxels,
    )
}

/// The interior points of an ALREADY-SOLVED arc — the circle walked directly, both endpoints
/// exclusive.
///
/// This is the form the closed case needs. Recovering a circle from endpoints plus a bulge is a
/// chord solve, and a whole turn has no chord; carrying the solved centre and radius instead means
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
                arc.centre[0] + arc.radius * angle.cos(),
                arc.centre[1] + arc.radius * angle.sin(),
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

/// Solve the 3-POINT creation (#102): the signed included angle of the arc from `from`
/// to `to` that passes through `through`. The through-point is consumed here — the
/// canonical stored form is endpoints + this angle (ADR 0030 §5). `None` when the three
/// points are collinear or coincident (no finite circle).
pub fn included_angle_through_degrees(
    from: [f64; 2],
    to: [f64; 2],
    through: [f64; 2],
) -> Option<f64> {
    // Circumcentre via the perpendicular-bisector determinant.
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

/// The OPERATION that turns a [`Sketch`]'s 2D profile into a 3D volume (ADR 0003
/// §3i, the "Sketch + Operation" model). A [`SketchSolid`] pairs a sketch with one
/// of these. Today the only operation is [`Extrude`](Operation::Extrude); revolve
/// and sweep are later commits.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Operation {
    /// Extrude the profile a whole number of voxels along its plane normal,
    /// producing a prism (≥1 for a non-empty prism).
    Extrude {
        /// Extrude span in voxels along the plane normal.
        height_voxels: u32,
    },
    /// Revolve the profile around an in-plane axis, producing a solid of
    /// revolution (ADR 0003 §3i). The sketch's two in-plane coordinates are
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
    // future: Sweep { path }  (added in later commits — leave this comment)
}

/// Which of the plane's two in-plane world axes is the REVOLVE (axial) axis — the
/// axis the profile is swept around (ADR 0003 §3i). The other in-plane axis plus
/// the plane NORMAL become the two RADIAL world axes the swept disc lives in.
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

/// How far the profile is swept around the revolve axis (ADR 0003 §3i). `360`
/// degrees is a full solid of revolution; a smaller value `(0, 360]` is a partial
/// wedge. `0` is degenerate (empty occupancy).
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
