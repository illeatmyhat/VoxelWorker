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
//! implicit centre anchor and so no half-block correction. The producer emits its
//! voxels centred on its own origin-centred grid exactly the way `SdfShape` does
//! (centres at `idx + 0.5 − grid/2`), but its placement does NOT route through
//! `leaf_lattice_shift_voxels`: a sketch's footprint is corner-anchored, so the
//! block-lattice shift the implicit-centre model needed is identically zero. (The
//! resolve path treats a sketch leaf like a VoxelBody — no intrinsic block size, no
//! lattice snap — see `Scene::resolve_*`.)
//!
//! 2a SCOPE: AXIS-ALIGNED planes only (the normal is one of ±X / ±Y / ±Z). A
//! free-angle sketch plane is the deferred plane-orientation milestone (§3f(a)).
//! The profile is a closed simple polygon (≥3 points); a degenerate profile
//! (fewer than 3 points, or zero area) resolves to nothing rather than panicking.

mod solid;
mod produce;
#[cfg(test)]
mod tests;

pub use solid::SketchSolid;

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

/// One vertex of a sketch profile — a 2D point, voxel-granular at the document's
/// density `d` (ADR 0003 §3f(0) `offset_voxels` integer-voxel convention, the same
/// representation as `ShapePoint::Inline` and `NodeTransform.offset_voxels`).
///
/// The two coordinates are in the plane's in-plane axes (see
/// [`PlaneAxis::in_plane_axes`]). They may be negative; the producer normalizes the
/// profile's bounding box to the local grid origin at resolve, so absolute values
/// only matter relative to the other points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SketchPoint {
    /// In-plane voxel coordinates `[axis0, axis1]` at the document density `d`.
    pub offset_voxels: [i64; 2],
}

impl SketchPoint {
    /// A profile vertex at the given in-plane voxel coordinates.
    pub fn new(axis0: i64, axis1: i64) -> Self {
        Self {
            offset_voxels: [axis0, axis1],
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

/// A point entity: a first-class, independently add/delete-able vertex on the sketch
/// plane, referenced by segments (and later arcs) through its stable [`id`](Self::id)
/// (ADR 0030). A point with no incident edge is a legal FREE point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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

/// A grid-aligned PLANE plus a collection of sketch ENTITIES — points and segments
/// (arcs, region picks, and sub-voxel/parametric coordinates arrive in later slices,
/// ADR 0030). The extrudable **profile is DERIVED** from the closed loop the segments
/// form (see [`flattened_loop`](Self::flattened_loop)); it is no longer a hand-maintained
/// ordered vertex list.
///
/// **Slice-1 scope (issue #98):** a single closed loop, resolving byte-identical to the
/// former `profile: Vec<SketchPoint>`. Multi-region pick/unpick (#100), sub-voxel /
/// parametric coordinates (#101), and arcs (#102) build on this store.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Sketch {
    /// Which axis the plane normal points along (2a: axis-aligned only).
    pub plane: PlaneAxis,
    /// The point entities (unordered; loop order is derived, never stored).
    points: Vec<Point>,
    /// The segment entities joining points by id.
    segments: Vec<Segment>,
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
        Self { plane, points: Vec::new(), segments: Vec::new(), next_id: 0 }
    }

    /// Read-only view of the point entities.
    pub fn points(&self) -> &[Point] {
        &self.points
    }

    /// Read-only view of the segment entities.
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// Test-only mutable access to the raw segment vector, for constructing the malformed
    /// stores the load-repair path is meant to erase.
    #[cfg(test)]
    pub(crate) fn segments_mut_for_test(&mut self) -> &mut Vec<Segment> {
        &mut self.segments
    }

    /// Allocate a point entity at `at`, returning its fresh id.
    fn add_point(&mut self, at: SketchPoint) -> EntityId {
        let id = self.alloc_id();
        self.points.push(Point { id, at, role: EntityRole::Real });
        id
    }

    /// Allocate a segment `from → to`, its `origin` set to its own id (a root of its
    /// lineage), returning its fresh id.
    fn add_segment(&mut self, from: EntityId, to: EntityId) -> EntityId {
        let id = self.alloc_id();
        self.segments.push(Segment { id, from, to, origin: id, role: EntityRole::Real });
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

    /// The DERIVED single closed loop, as point ids in traversal order (ADR 0030). A
    /// deterministic walk of the point-segment graph: start at the incident point of
    /// lowest id, then at each step follow the lowest-id neighbour that is not the one
    /// just left. Empty if there are no segments. Slice-1 assumes one simple loop; the
    /// general planar-face derivation is #100.
    fn loop_order(&self) -> Vec<EntityId> {
        if self.points.is_empty() {
            return Vec::new();
        }
        // Neighbour point-ids per point, in ascending-id order for determinism.
        let neighbours = |point_id: EntityId| -> Vec<EntityId> {
            let mut ns: Vec<EntityId> = self
                .segments
                .iter()
                .filter_map(|seg| {
                    if seg.from == point_id {
                        Some(seg.to)
                    } else if seg.to == point_id {
                        Some(seg.from)
                    } else {
                        None
                    }
                })
                .collect();
            ns.sort_unstable();
            ns
        };
        // Start at the lowest-id point that has at least one incident segment.
        let Some(start) = self
            .points
            .iter()
            .map(|point| point.id)
            .filter(|&id| !neighbours(id).is_empty())
            .min()
        else {
            return Vec::new();
        };
        let mut order = vec![start];
        let mut previous = None;
        let mut current = start;
        // A simple loop has one unvisited neighbour per step; cap the walk at the point
        // count so a malformed graph cannot spin forever.
        for _ in 0..self.points.len() {
            let ns = neighbours(current);
            let next = ns
                .iter()
                .copied()
                .find(|&n| Some(n) != previous)
                .or_else(|| ns.first().copied());
            let Some(next) = next else { break };
            if next == start {
                break;
            }
            order.push(next);
            previous = Some(current);
            current = next;
        }
        order
    }

    /// The DERIVED flattened profile: the closed loop's vertices in traversal order (ADR
    /// 0030). This is what the producer resolves; slice 1 yields exactly one loop, so the
    /// occupancy is byte-identical to the former `profile` vector (`point_in_polygon` is
    /// winding-agnostic, so the traversal direction does not matter).
    pub fn flattened_loop(&self) -> Vec<SketchPoint> {
        self.loop_order()
            .into_iter()
            .filter_map(|id| self.point_index(id).map(|i| self.points[i].at))
            .collect()
    }

    /// The point ids of the flattened loop, in the SAME order as
    /// [`flattened_loop`](Self::flattened_loop) — so the UI can map a loop-vertex index
    /// back to the entity it must mutate (drag / add-point / delete).
    pub fn flattened_loop_ids(&self) -> Vec<EntityId> {
        self.loop_order()
    }

    /// Mutable access to a point's position by id (the drag write path).
    pub fn point_position_mut(&mut self, id: EntityId) -> Option<&mut SketchPoint> {
        self.point_index(id).map(|i| &mut self.points[i].at)
    }

    /// Split the loop edge between flattened positions `after` and `after + 1` by
    /// inserting a new point `at` on it (ADR 0030 add-point, #95's insert generalized).
    /// The split segment keeps its id for the first half and its `origin` is inherited by
    /// the new second half, so the bounding face's origin-set is unchanged. `after` past
    /// the loop end appends onto the last→first closing edge. A no-op on an empty loop.
    fn insert_point_on_loop_edge(&mut self, after: usize, at: SketchPoint) {
        let order = self.loop_order();
        if order.is_empty() {
            // No loop yet: just add the point as a free vertex.
            self.add_point(at);
            return;
        }
        let a = order[after.min(order.len() - 1)];
        let b = order[(after + 1) % order.len()];
        let new_id = self.add_point(at);
        // Find the segment joining a and b (either direction) and split it.
        if let Some(seg_index) = self.segments.iter().position(|seg| {
            (seg.from == a && seg.to == b) || (seg.from == b && seg.to == a)
        }) {
            let origin = self.segments[seg_index].origin;
            let old_to = self.segments[seg_index].to;
            // First half keeps the id: `... → new`. Second half inherits the origin.
            self.segments[seg_index].to = new_id;
            let id = self.alloc_id();
            self.segments.push(Segment { id, from: new_id, to: old_to, origin, role: EntityRole::Real });
        } else {
            // a and b are not directly joined (an open authoring state): connect a → new.
            self.add_segment(a, new_id);
        }
    }

    /// Delete the loop vertex at flattened position `index`, CASCADING to its incident
    /// segments (ADR 0030 §6 — deleting a point removes its edges; it does NOT reclose the
    /// loop, superseding #95's loop-specific reclose). Out-of-range is a no-op.
    fn delete_loop_vertex(&mut self, index: usize) {
        let order = self.loop_order();
        let Some(&id) = order.get(index) else { return };
        self.delete_point_cascade(id);
    }

    /// Delete a point by id and every segment incident to it (ADR 0030 §6). Segments'
    /// other endpoints survive as free points. No dangling reference can result.
    pub fn delete_point_cascade(&mut self, id: EntityId) {
        self.segments.retain(|seg| seg.from != id && seg.to != id);
        self.points.retain(|point| point.id != id);
    }

    /// Erase every structurally-invalid segment — one that references a point id not in the
    /// store, or a self-loop (`from == to`) — returning the number removed (ADR 0030 load
    /// policy: erase invalid objects rather than fail the load). Points are never invalid; a
    /// point left with no incident edge is a legal free point. The resolve already tolerates a
    /// dangling reference (the missing vertex is filtered out of the flattened loop), so this
    /// is a cleanup + audit, not a crash guard.
    pub fn repair(&mut self) -> usize {
        let point_ids: Vec<EntityId> = self.points.iter().map(|point| point.id).collect();
        let before = self.segments.len();
        self.segments.retain(|seg| {
            seg.from != seg.to && point_ids.contains(&seg.from) && point_ids.contains(&seg.to)
        });
        before - self.segments.len()
    }
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
