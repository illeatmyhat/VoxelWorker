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

mod edges;
mod faces;
mod produce;
mod solid;
#[cfg(test)]
mod tests;

pub use faces::{Face, FaceKey};
pub use solid::SketchSolid;
pub use substrate::geom2d::LoopRole;

use voxel_core::units::{AngleMeasurement, Measurement};

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
                    Err(voxel_core::units::MeasurementError::BlockTermNotWholeVoxels {
                        nearest_floor_voxels,
                        ..
                    }) => (
                        nearest_floor_voxels,
                        Measurement::from_voxels(nearest_floor_voxels),
                    ),
                    Err(voxel_core::units::MeasurementError::ZeroDensity) => {
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

/// One loop of the flattened profile: a simple closed polygon plus how it contributes to the
/// region (ADR 0030 §4). The unit the 2D CSG folds and the unit the overlay draws.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileLoop {
    /// Whether the loop's interior is added or carved out.
    pub role: LoopRole,
    /// The closed boundary, counter-clockwise, arcs already tessellated.
    pub points: Vec<SketchPoint>,
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

/// The `center` of an arc that has no centre point yet — a pre-centre document, or an arc
/// mid-construction. Ids are handed out monotonically from zero and never reused, so the top
/// of the range can never collide with a live entity.
pub const ABSENT_CENTER: EntityId = EntityId::MAX;

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
    /// The faces the author has UNPICKED, by boundary origin-set key (ADR 0030 §3, #100).
    /// Every derived face is picked by default, so this holds only the exceptions and is
    /// usually empty. A key that matches no current face is inert, not an error: it costs
    /// nothing and lets an unpick survive an edit that temporarily breaks its boundary.
    #[serde(default)]
    unpicked: std::collections::BTreeSet<FaceKey>,
    /// The next id to hand out. Ids are monotonic and never reused, so this only grows.
    next_id: EntityId,
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
            unpicked: std::collections::BTreeSet::new(),
            next_id: 0,
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
            unpicked: std::collections::BTreeSet::new(),
            next_id: 0,
        }
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

    /// The flattened region in the **measurement** width — the exact value
    /// [`substrate::geom2d::signed_distance_to_region`] folds, and the exact value the wash's
    /// WGSL mirror is handed (ADR 0030 §3).
    ///
    /// One definition of the region, two evaluators of it: the resolve asks it per voxel on the
    /// CPU, the overlay asks it per pixel on the GPU. The overlay used to triangulate the faces
    /// instead, which made nesting the overlay's own problem to solve — a fill inside a fill
    /// composited twice — where the region predicate already answers it.
    pub fn region_field_loops(&self) -> Vec<(LoopRole, Vec<[f32; 2]>)> {
        produce::to_region_points_measured(&self.flattened_region())
    }

    /// Whether the face with this boundary key contributes solid. Faces default to PICKED — the
    /// document stores only the unpicked exceptions (ADR 0030 §3).
    pub fn face_is_picked(&self, key: &FaceKey) -> bool {
        !self.unpicked.contains(key)
    }

    /// Pick or unpick the face with this boundary key, carving or filling a pocket. Storing the
    /// key rather than the face means the intent survives re-derivation: a vertex drag leaves the
    /// key untouched, and so does splitting a boundary edge (both children keep the parent's
    /// origin), while restructuring the boundary makes it a different face that reverts to picked.
    pub fn set_face_picked(&mut self, key: FaceKey, picked: bool) {
        if picked {
            self.unpicked.remove(&key);
        } else {
            self.unpicked.insert(key);
        }
    }

    /// The unpicked boundary keys, ascending — the whole of the pick state the document carries.
    pub fn unpicked_faces(&self) -> impl Iterator<Item = &FaceKey> {
        self.unpicked.iter()
    }

    /// The DERIVED flattened profile: one tagged loop per derived face, `Fill` where the face is
    /// picked and `Hole` where it is not (ADR 0030 §4), each a simple closed polygon with its
    /// arcs tessellated into sub-voxel chords ([`ARC_SAGITTA_TOLERANCE_VOXELS`]).
    ///
    /// This is what the producer resolves, and the tessellated polygons ARE the resolved meaning
    /// (ADR 0019). The combination is an explicit 2D boolean — union the fills, subtract the
    /// holes — never a global crossing parity, so two fills that touch or share an edge both
    /// count where even-odd would cancel them.
    pub fn flattened_region(&self) -> Vec<ProfileLoop> {
        self.faces()
            .into_iter()
            .map(|face| ProfileLoop {
                role: if self.face_is_picked(&face.key) {
                    LoopRole::Fill
                } else {
                    LoopRole::Hole
                },
                points: face.boundary,
            })
            .collect()
    }

    /// The flattened profile's `Fill` loops only — what the region's EXTENT is measured from (a
    /// hole adds no footprint, and an unpicked face with nothing around it is not occupancy).
    pub fn filled_loops(&self) -> Vec<Vec<SketchPoint>> {
        self.flattened_region()
            .into_iter()
            .filter(|profile_loop| profile_loop.role == LoopRole::Fill)
            .map(|profile_loop| profile_loop.points)
            .collect()
    }

    /// The SIMPLE-profile door: the sole boundary when the region is exactly one picked face,
    /// and empty otherwise (no face, an unpicked one, or several — those are questions only
    /// [`flattened_region`](Self::flattened_region) can answer). Callers that reason about a
    /// single closed outline (rectangle detection, most tests) want this; anything that resolves
    /// occupancy wants the region.
    pub fn flattened_loop(&self) -> Vec<SketchPoint> {
        let mut loops = self.flattened_region();
        match (loops.len(), loops.first().map(|first| first.role)) {
            (1, Some(LoopRole::Fill)) => loops.remove(0).points,
            _ => Vec::new(),
        }
    }

    /// Move the point `id` to `at` — the drag write path. Reports whether the point exists.
    ///
    /// Dragging an arc's CENTRE moves only the centre: the endpoints hold still and the arc's
    /// radius follows the cursor ([`resweep_arc_to_center`](Self::resweep_arc_to_center)). Every
    /// other point simply takes `at`.
    pub fn move_point(&mut self, id: EntityId, at: SketchPoint) -> bool {
        let Some(index) = self.point_index(id) else {
            return false;
        };
        match self.arcs.iter().position(|arc| arc.center == id) {
            Some(arc_index) => self.resweep_arc_to_center(arc_index, at.in_plane()),
            None => self.points[index].at = at,
        }
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
        self.points.retain(|point| point.id != id);
        self.prune_orphan_centers();
    }

    /// Delete just the segment with id `seg_id` (ADR 0030 — deleting a line removes only the
    /// line). Its endpoint points survive as free points. No-op if `seg_id` is unknown.
    pub fn delete_segment(&mut self, seg_id: EntityId) {
        self.segments.retain(|seg| seg.id != seg_id);
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

    /// Delete just the arc with id `arc_id`, its endpoints left as free points (ADR 0030
    /// §6 — deleting a segment/arc removes only it). No-op if `arc_id` is unknown.
    pub fn delete_arc(&mut self, arc_id: EntityId) {
        self.arcs.retain(|arc| arc.id != arc_id);
        self.prune_orphan_centers();
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
        let before = self.segments.len() + self.arcs.len();
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
        let dropped = before - self.segments.len() - self.arcs.len();
        // A pre-centre document names no centre at all, and a just-erased arc leaves one
        // behind; both are settled here, so a loaded sketch always agrees with its arcs.
        self.prune_orphan_centers();
        self.sync_arc_centers();
        dropped
    }
}

/// Arc tessellation tolerance (#102): the maximum sagitta (chord-to-arc deviation), in
/// voxels, of one tessellated chord. **Versioned**: the flattened polygon is the resolved
/// meaning (ADR 0019), so changing this value changes what an arc-bounded profile
/// occupies — treat an edit like a document-format change, not a tuning knob.
pub const ARC_SAGITTA_TOLERANCE_VOXELS: f64 = 1.0 / 16.0;

/// Hard cap on chords per arc, so a huge-radius near-collinear arc cannot degenerate
/// into an unbounded fan.
const ARC_MAX_CHORDS: u32 = 512;

/// Whether a signed sweep is a legal arc bulge: finite, non-zero, strictly under a full
/// turn in magnitude.
fn arc_sweep_is_valid(sweep_degrees: f64) -> bool {
    sweep_degrees.is_finite() && sweep_degrees != 0.0 && sweep_degrees.abs() < 360.0
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

/// [`arc_interior_points`] at a caller-chosen sagitta tolerance — the DISPLAY door.
///
/// The resolve's tolerance is measured in voxels, so an arc's chord count follows its
/// radius-in-voxels and not its size on screen: a 15-voxel arc earns nine chords whatever the
/// zoom, which reads as a visible polygon. A viewer that knows how many pixels a voxel is
/// currently worth can ask for a tolerance that keeps the sagitta well under a pixel instead.
/// Only the pinned [`ARC_SAGITTA_TOLERANCE_VOXELS`] is the resolved MEANING (ADR 0019); a
/// finer one is a smoother drawing of the same curve, never a different profile.
pub fn arc_interior_points_within(
    from: [f64; 2],
    to: [f64; 2],
    sweep_degrees: f64,
    sagitta_tolerance_voxels: f64,
) -> Vec<SketchPoint> {
    let Some((center, radius)) = arc_center_radius(from, to, sweep_degrees) else {
        return Vec::new();
    };
    let chords = arc_chord_count(radius, sweep_degrees, sagitta_tolerance_voxels);
    let start = (from[1] - center[1]).atan2(from[0] - center[0]);
    let step = sweep_degrees.to_radians() / chords as f64;
    (1..chords)
        .map(|chord_index| {
            let angle = start + step * chord_index as f64;
            SketchPoint::from_continuous(
                center[0] + radius * angle.cos(),
                center[1] + radius * angle.sin(),
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
