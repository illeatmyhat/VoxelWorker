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
//! resolve path treats a sketch leaf like a Part — no intrinsic block size, no
//! lattice snap — see `Scene::resolve_*`.)
//!
//! 2a SCOPE: AXIS-ALIGNED planes only (the normal is one of ±X / ±Y / ±Z). A
//! free-angle sketch plane is the deferred plane-orientation milestone (§3f(a)).
//! The profile is a closed simple polygon (≥3 points); a degenerate profile
//! (fewer than 3 points, or zero area) resolves to nothing rather than panicking.

use crate::voxel::{Voxel, VoxelGrid, VoxelProducer, MAX_GRID_VOXELS};
use rayon::prelude::*;

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

/// A grid-aligned PLANE plus a closed POLYGON PROFILE of ordered points (ADR 0003
/// §3i). The profile is a closed simple polygon: the last vertex connects back to
/// the first, so the points list does NOT repeat the start vertex.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Sketch {
    /// Which axis the plane normal points along (2a: axis-aligned only).
    pub plane: PlaneAxis,
    /// The ordered profile vertices (≥3 for a non-degenerate polygon).
    pub profile: Vec<SketchPoint>,
}

impl Sketch {
    /// A sketch on `plane` with the given ordered profile.
    pub fn new(plane: PlaneAxis, profile: Vec<SketchPoint>) -> Self {
        Self { plane, profile }
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

/// A [`Sketch`] paired with an [`Operation`] that turns its 2D profile into a 3D
/// volume — the 2a sketch→volume producer (ADR 0003 §3i, the "Sketch + Operation"
/// model). Added **alongside** `SdfShape`; both implement [`VoxelProducer`] and
/// resolve through the same stamp / `CombineOp` / chunk path. The only operation
/// today is [`Operation::Extrude`] (a prism); revolve / sweep are later commits.
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

/// Even-odd (ray-crossing) point-in-polygon test for the cell whose centre sits at
/// `(sample_0, sample_1)` in the profile's own (un-normalized) coordinate space.
///
/// The classic crossing-number test: count how many polygon edges a ray cast in
/// the +axis1 direction from the sample point crosses. An odd count is inside.
/// Edges are taken between consecutive vertices, closing the last → first. Cell
/// centres sit at half-integer positions (`min + i + 0.5`), which never coincide
/// with the integer-coordinate vertices/edges, so there are no on-boundary
/// ambiguities — the rasterization is exact and deterministic.
fn point_in_polygon(profile: &[SketchPoint], sample_0: f64, sample_1: f64) -> bool {
    let mut inside = false;
    let count = profile.len();
    let mut previous = count - 1;
    for current in 0..count {
        let current_point = profile[current].offset_voxels;
        let previous_point = profile[previous].offset_voxels;
        let current_0 = current_point[0] as f64;
        let current_1 = current_point[1] as f64;
        let previous_0 = previous_point[0] as f64;
        let previous_1 = previous_point[1] as f64;
        // Does a horizontal-in-axis1 ray from the sample cross this edge?
        let straddles = (current_1 > sample_1) != (previous_1 > sample_1);
        if straddles {
            // X (axis0) of the edge at the sample's axis1 height.
            let crossing_0 = (previous_0 - current_0) * (sample_1 - current_1)
                / (previous_1 - current_1)
                + current_0;
            if sample_0 < crossing_0 {
                inside = !inside;
            }
        }
        previous = current;
    }
    inside
}

/// Whether the CLOSED axis-aligned rectangle `[c0_lo, c0_hi] × [c1_lo, c1_hi]` lies
/// ENTIRELY inside the profile polygon, in the profile's native `(c0, c1)` space (the SAME
/// space [`point_in_polygon`] samples). The coarse-solid interior-elision test (ADR 0010).
///
/// Callers pass the SAMPLE-CENTRE rectangle — the span of a block's per-voxel sample
/// centres (`min + idx + 0.5`), NOT the voxel corners — so the polygon boundary that runs
/// along a block face sits 0.5 beyond the outermost centre and no longer "crosses" (an
/// axis-aligned face block IS fully solid and elides).
///
/// EXACT by connectedness: if no polygon edge crosses the closed rectangle then it contains
/// no piece of the polygon boundary, so it is wholly inside or wholly outside — one interior
/// sample (the centre, which is not on any edge given no crossing) decides. So the rectangle
/// is inside iff **no polygon edge intersects it AND its centre is inside**; every sample
/// centre then lies inside ⇒ every voxel solid. Conservative: a rectangle whose edge grazes
/// a polygon edge counts as crossing ⇒ BOUNDARY (still exact). A DEGENERATE rectangle (a span
/// that collapses to a single voxel ⇒ `hi == lo`, i.e. a segment or a point) is handled
/// directly: the edge tests run against the degenerate box and the centre reduces to the
/// point/segment-midpoint — for a single-voxel block this is exactly `point_in_polygon` at
/// that voxel's own centre, matching the per-voxel resolve. Correct for convex, concave (the
/// L reflex corner), and rectangle profiles alike.
fn rectangle_inside_polygon(
    profile: &[SketchPoint],
    c0_lo: f64,
    c0_hi: f64,
    c1_lo: f64,
    c1_hi: f64,
) -> bool {
    let count = profile.len();
    // Allow a degenerate (single-voxel) span (`hi == lo`); only a truly inverted box is
    // rejected. The sample-centre span guarantees `hi >= lo` (width = voxels − 1 >= 0).
    if count < 3 || c0_hi < c0_lo || c1_hi < c1_lo {
        return false;
    }
    let rect_min = [c0_lo, c1_lo];
    let rect_max = [c0_hi, c1_hi];
    let mut previous = count - 1;
    for current in 0..count {
        let a = profile[current].offset_voxels;
        let b = profile[previous].offset_voxels;
        let a = [a[0] as f64, a[1] as f64];
        let b = [b[0] as f64, b[1] as f64];
        if segment_intersects_rect(a, b, rect_min, rect_max) {
            return false;
        }
        previous = current;
    }
    point_in_polygon(profile, (c0_lo + c0_hi) * 0.5, (c1_lo + c1_hi) * 0.5)
}

/// Whether segment `a→b` intersects the CLOSED axis-aligned rectangle
/// `[rect_min, rect_max]` (component-wise min <= max). True iff an endpoint is inside the
/// rectangle OR the segment crosses one of the four rectangle edges — complete for a
/// convex box. Points are `[coord0, coord1]` in the profile's native space.
fn segment_intersects_rect(a: [f64; 2], b: [f64; 2], rect_min: [f64; 2], rect_max: [f64; 2]) -> bool {
    let inside = |p: [f64; 2]| {
        p[0] >= rect_min[0] && p[0] <= rect_max[0] && p[1] >= rect_min[1] && p[1] <= rect_max[1]
    };
    if inside(a) || inside(b) {
        return true;
    }
    let corners = [
        [rect_min[0], rect_min[1]],
        [rect_max[0], rect_min[1]],
        [rect_max[0], rect_max[1]],
        [rect_min[0], rect_max[1]],
    ];
    (0..4).any(|edge| segments_intersect(a, b, corners[edge], corners[(edge + 1) % 4]))
}

/// Robust segment–segment intersection (proper crossings AND collinear / endpoint
/// touches), via orientation signs. Used only for the exact rectangle-inside-polygon test.
fn segments_intersect(p0: [f64; 2], p1: [f64; 2], q0: [f64; 2], q1: [f64; 2]) -> bool {
    let orient = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| -> i32 {
        let value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        if value > 0.0 {
            1
        } else if value < 0.0 {
            -1
        } else {
            0
        }
    };
    // `c` (collinear with `a→b`) lies within `a→b`'s bounding box.
    let on_segment = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| -> bool {
        c[0] >= a[0].min(b[0])
            && c[0] <= a[0].max(b[0])
            && c[1] >= a[1].min(b[1])
            && c[1] <= a[1].max(b[1])
    };
    let d1 = orient(q0, q1, p0);
    let d2 = orient(q0, q1, p1);
    let d3 = orient(p0, p1, q0);
    let d4 = orient(p0, p1, q1);
    if d1 != d2 && d3 != d4 {
        return true;
    }
    (d1 == 0 && on_segment(q0, q1, p0))
        || (d2 == 0 && on_segment(q0, q1, p1))
        || (d3 == 0 && on_segment(p0, p1, q0))
        || (d4 == 0 && on_segment(p0, p1, q1))
}

impl VoxelProducer for SketchSolid {
    fn resolve(&self, grid: &mut VoxelGrid, voxels_per_block: u32) {
        let [full_x, full_y, full_z] = self.grid_dimensions();
        self.resolve_into(
            grid,
            voxels_per_block,
            crate::spatial_index::VoxelAabb::new(
                [0, 0, 0],
                [full_x as i64, full_y as i64, full_z as i64],
            ),
        );
    }

    fn resolve_into(
        &self,
        grid: &mut VoxelGrid,
        voxels_per_block: u32,
        window_local_voxels: crate::spatial_index::VoxelAabb,
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
        cell_local_voxels: crate::spatial_index::VoxelAabb,
        voxels_per_block: u32,
    ) -> Option<crate::voxel::FieldInterval> {
        let _ = voxels_per_block;
        if cell_local_voxels.is_empty() {
            return None;
        }
        let dimensions = self.grid_dimensions();
        let [full_x, full_y, full_z] = dimensions;
        // A degenerate (empty-occupancy) producer: every cell is AIR.
        let grid_aabb = crate::spatial_index::VoxelAabb::new(
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
    fn extrude_cell_is_solid(&self, cell: crate::spatial_index::VoxelAabb) -> bool {
        let Some((min, _max)) = self.profile_bounds() else {
            return false;
        };
        let [in_plane_0, in_plane_1] = self.sketch.plane.in_plane_axes();
        let c0_lo = (min[0] + cell.min[in_plane_0]) as f64 + 0.5;
        let c0_hi = (min[0] + cell.max[in_plane_0]) as f64 - 0.5;
        let c1_lo = (min[1] + cell.min[in_plane_1]) as f64 + 0.5;
        let c1_hi = (min[1] + cell.max[in_plane_1]) as f64 - 0.5;
        rectangle_inside_polygon(&self.sketch.profile, c0_lo, c0_hi, c1_lo, c1_hi)
    }

    /// Whether a revolve cell (PROVEN fully inside `[0, full_dim)` by the caller) is
    /// entirely solid — the coarse-solid test (ADR 0010). Full-turn only: a PARTIAL wedge
    /// is not cleanly boundable per cell here, so it returns `false` (⇒ BOUNDARY, still
    /// exact) — a documented deferral covered by the parity fuzz.
    ///
    /// For a full turn the solid-of-revolution occupancy at a cell is
    /// `point_in_polygon(radius, axial)` (folded by `abs`; the resolve also tests `−radius`
    /// only when the profile straddles the axis, which can only ADD occupancy). So the cell
    /// is solid iff the `(radius-range × axial-range)` rectangle is entirely inside the
    /// profile polygon, mapped into native `(c0, c1)` per [`RevolveAxis`] EXACTLY as
    /// [`resolve_revolve`](Self::resolve_revolve) maps its per-voxel samples:
    /// - axial: the SAMPLE-CENTRE span `[axial_min + cell.min + 0.5, axial_min + cell.max − 0.5]`
    ///   (elides the axial END-CAP blocks, whose face is collinear with the profile edge);
    /// - radius: over the two centred radial world axes (centred = `idx − half`), the
    ///   `[nearest, farthest]` distance from the axis over the cell's voxel-corner box,
    ///   widened by `EPS` so f32/f64 rounding can never SHRINK the tested rectangle below
    ///   the true sample coverage (a wider rectangle only makes "inside" rarer ⇒ never an
    ///   over-claim). The two are INDEPENDENT: axial uses the centre span, radius the
    ///   conservative corner box + `EPS`.
    fn revolve_cell_is_solid(
        &self,
        cell: crate::spatial_index::VoxelAabb,
        axis: RevolveAxis,
        sweep: RevolveSweep,
        dimensions: [u32; 3],
    ) -> bool {
        // Only full turns are elided; a partial wedge falls back to BOUNDARY.
        if sweep.turn_degrees < 360 {
            return false;
        }
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
        rectangle_inside_polygon(&self.sketch.profile, c0_lo, c0_hi, c1_lo, c1_hi)
    }

    /// The extrude resolve: rasterize the profile once and sweep it across
    /// `height_voxels` layers along the plane normal. Byte-identical to the prior
    /// `SketchExtrude::resolve` (the height now arrives from the matched operation).
    fn resolve_extrude(
        &self,
        grid: &mut VoxelGrid,
        voxels_per_block: u32,
        height_voxels: u32,
        window_local_voxels: crate::spatial_index::VoxelAabb,
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
        let mut filled_in_plane: Vec<[u32; 2]> = Vec::new();
        for cell_1 in cell_1_lo..cell_1_hi {
            let sample_1 = min[1] as f64 + cell_1 as f64 + 0.5;
            for cell_0 in cell_0_lo..cell_0_hi {
                let sample_0 = min[0] as f64 + cell_0 as f64 + 0.5;
                if point_in_polygon(&self.sketch.profile, sample_0, sample_1) {
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
                        block_id: crate::core_geom::BlockId::DEFAULT,
                        attrs: crate::core_geom::BlockAttrs::DEFAULT,
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
    fn resolve_revolve(
        &self,
        grid: &mut VoxelGrid,
        voxels_per_block: u32,
        axis: RevolveAxis,
        sweep: RevolveSweep,
        window_local_voxels: crate::spatial_index::VoxelAabb,
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

        let profile = &self.sketch.profile;

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
        // `resolve_scene`), where the window IS the full grid, from a blown allocation.
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
                            point_in_polygon(profile, sample_0, sample_1)
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
                            block_id: crate::core_geom::BlockId::DEFAULT,
                            attrs: crate::core_geom::BlockAttrs::DEFAULT,
                            grid_overlay: false,
                        });
                    }
                }
                local
            })
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::{SdfShape, ShapeKind, VoxelProducer};
    use std::collections::BTreeSet;

    /// Collect a producer's occupied voxels as a sorted set of
    /// `(world_position_bits, block_local_coord, material_id)` so two producers can
    /// be compared for SET equality independent of emission order. World positions
    /// are integer + 0.5, so the f32 bit pattern is exact and hashable.
    fn occupancy_set(
        producer: &dyn VoxelProducer,
        density: u32,
    ) -> BTreeSet<([i32; 3], [u8; 3], u16)> {
        let mut grid = VoxelGrid::default();
        producer.resolve(&mut grid, density);
        grid.occupied
            .iter()
            .map(|voxel| {
                let position = voxel.world_position();
                (
                    [
                        (position[0] * 2.0).round() as i32,
                        (position[1] * 2.0).round() as i32,
                        (position[2] * 2.0).round() as i32,
                    ],
                    voxel.block_local_coord,
                    voxel.color_index(),
                )
            })
            .collect()
    }

    /// LOAD-BEARING: a rectangle-profile extrude produces EXACTLY the same occupied
    /// voxel set (positions, block-local coords, materials) as the axis-aligned
    /// `Box` `SdfShape` of the same size/placement/density. This is the "box =
    /// rectangle-extrude sugar" proof (§3i). Covered for several sizes including an
    /// odd size and density 16.
    #[test]
    fn rectangle_extrude_equals_box() {
        // (size_blocks, density). An odd size (3) and density 16 are included.
        let cases = [
            ([2u32, 2, 2], 4u32),
            ([3, 1, 5], 4),
            ([3, 3, 3], 16),
            ([4, 2, 6], 16),
        ];
        for (size_blocks, density) in cases {
            let box_shape = SdfShape::from_blocks(ShapeKind::Box, size_blocks, 1, density);
            let grid_x = (size_blocks[0] * density) as i64;
            let grid_y = (size_blocks[1] * density) as i64;
            let grid_z = (size_blocks[2] * density) as i64;
            // Plane Y: profile in XZ (width = X span, height = Z span), extruded
            // grid_y voxels along Y — matches the box's [x, y, z] grid exactly.
            let extrude = SketchSolid::extrude(
                Sketch::rectangle(PlaneAxis::Y, grid_x, grid_z),
                grid_y as u32,
            );
            assert_eq!(
                extrude.grid_dimensions(),
                box_shape.grid_dimensions(density),
                "grid dims must match for size {size_blocks:?} @ d{density}"
            );
            assert_eq!(
                occupancy_set(&extrude, density),
                occupancy_set(&box_shape, density),
                "rectangle extrude must equal Box for size {size_blocks:?} @ d{density}"
            );
        }
    }

    /// A rectangle extrude on EACH of the three axis-aligned plane orientations
    /// equals the matching `Box` — proves the plane mapping is correct for X, Y, Z.
    #[test]
    fn rectangle_extrude_each_plane_equals_box() {
        let density = 4u32;
        let size_blocks = [2u32, 3, 4];
        let box_shape = SdfShape::from_blocks(ShapeKind::Box, size_blocks, 1, density);
        let dims = box_shape.grid_dimensions(density);
        let box_set = occupancy_set(&box_shape, density);
        for plane in [PlaneAxis::X, PlaneAxis::Y, PlaneAxis::Z] {
            let [in_plane_0, in_plane_1] = plane.in_plane_axes();
            let normal = plane.normal_axis();
            let extrude = SketchSolid::extrude(
                Sketch::rectangle(plane, dims[in_plane_0] as i64, dims[in_plane_1] as i64),
                dims[normal],
            );
            assert_eq!(
                extrude.grid_dimensions(),
                dims,
                "plane {plane:?} grid dims must match the box AABB"
            );
            assert_eq!(
                occupancy_set(&extrude, density),
                box_set,
                "plane {plane:?} rectangle extrude must equal the same Box"
            );
        }
    }

    /// CONCAVITY / added value: an L-shaped (non-convex) hexagon profile extrudes to
    /// the correct occupancy. A box CANNOT make this; the reflex vertex exercises the
    /// rasterizer. The L is a 4×4 square with its top-right 2×2 quadrant removed:
    ///
    /// ```text
    /// axis1
    ///  3 | X X . .
    ///  2 | X X . .
    ///  1 | X X X X
    ///  0 | X X X X
    ///      0 1 2 3  axis0
    /// ```
    #[test]
    fn l_shape_extrude_occupancy() {
        // Profile (CCW) of the L: outer rectangle 0..4 × 0..2, plus left column
        // 0..2 × 2..4. Six vertices, one reflex corner at (2, 2).
        let profile = vec![
            SketchPoint::new(0, 0),
            SketchPoint::new(4, 0),
            SketchPoint::new(4, 2),
            SketchPoint::new(2, 2), // reflex vertex
            SketchPoint::new(2, 4),
            SketchPoint::new(0, 4),
        ];
        let extrude = SketchSolid::extrude(Sketch::new(PlaneAxis::Y, profile), 1);
        let mut grid = VoxelGrid::default();
        extrude.resolve(&mut grid, 4);
        assert_eq!(grid.dimensions, [4, 1, 4], "L AABB is 4×1×4");

        // Recover the in-plane cell of each voxel (plane Y ⇒ axes X, Z). Corner-
        // anchored: centres are `idx + 0.5`, so the cell index is `world − 0.5`.
        let mut cells: BTreeSet<(i64, i64)> = BTreeSet::new();
        for voxel in &grid.occupied {
            let position = voxel.world_position();
            let cell_x = (position[0] - 0.5).round() as i64;
            let cell_z = (position[2] - 0.5).round() as i64;
            cells.insert((cell_x, cell_z));
        }
        // The L occupies the full bottom two rows (z 0..2, all x) and the left two
        // columns of the top two rows (x 0..2, z 2..4) — 8 + 4 = 12 cells.
        let mut expected: BTreeSet<(i64, i64)> = BTreeSet::new();
        for x in 0..4 {
            for z in 0..2 {
                expected.insert((x, z));
            }
        }
        for x in 0..2 {
            for z in 2..4 {
                expected.insert((x, z));
            }
        }
        assert_eq!(cells, expected, "L footprint occupancy is wrong");
        // Spot-check specific in/out cells: filled bottom-right corner, EMPTY
        // top-right quadrant (the removed 2×2 a box could not exclude).
        assert!(cells.contains(&(3, 0)), "(3,0) inside the L");
        assert!(cells.contains(&(0, 3)), "(0,3) inside the L left column");
        assert!(!cells.contains(&(3, 3)), "(3,3) is the removed quadrant");
        assert!(!cells.contains(&(2, 2)), "(2,2) is outside the reflex corner");
    }

    /// EDGE CASE: degenerate profiles resolve to empty occupancy without panicking —
    /// fewer than 3 points, collinear (zero-area) points, and a zero height.
    #[test]
    fn degenerate_profiles_are_empty() {
        let empty = |producer: &SketchSolid| {
            let mut grid = VoxelGrid::default();
            producer.resolve(&mut grid, 4);
            assert!(grid.occupied.is_empty());
            assert_eq!(grid.dimensions, [0, 0, 0]);
        };
        // < 3 points.
        empty(&SketchSolid::extrude(
            Sketch::new(PlaneAxis::Y, vec![SketchPoint::new(0, 0), SketchPoint::new(4, 0)]),
            2,
        ));
        // Collinear (zero-area) — three points on one line.
        empty(&SketchSolid::extrude(
            Sketch::new(
                PlaneAxis::Y,
                vec![
                    SketchPoint::new(0, 0),
                    SketchPoint::new(2, 0),
                    SketchPoint::new(4, 0),
                ],
            ),
            2,
        ));
        // Zero height.
        empty(&SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Y, 4, 4), 0));
    }

    /// EDGE CASE: a sub-block-precise profile at d=16 (a vertex NOT on a block
    /// boundary) rasterizes correctly. The profile is a 20×20-voxel square (1.25
    /// blocks per side at d16) whose extent is not a whole number of blocks; the fill
    /// is exactly the 20×20 cell set on every layer.
    #[test]
    fn sub_block_precise_profile_at_d16() {
        let density = 16u32;
        // 20 voxels = 1 block + 4 voxels — a sub-block extent on a non-block boundary.
        let extrude = SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Y, 20, 20), 3);
        assert_eq!(extrude.grid_dimensions(), [20, 3, 20]);
        let mut grid = VoxelGrid::default();
        extrude.resolve(&mut grid, density);
        // A full 20×3×20 rectangular prism.
        assert_eq!(grid.occupied.len(), 20 * 3 * 20);
        // block_local_coord wraps at the density: a cell at in-plane index 17 has
        // block-local X = 17 % 16 = 1 (proves the sub-block fraction is carried).
        let has_local_one = grid.occupied.iter().any(|voxel| {
            // Corner-anchored: cell index = world − 0.5.
            let cell_x = (voxel.world_position()[0] - 0.5).round() as i64;
            cell_x == 17 && voxel.block_local_coord[0] == 1
        });
        assert!(has_local_one, "sub-block block_local_coord must wrap at d=16");
    }

    /// The rectangle-detection helper the inspector uses to choose editable
    /// Width/Depth vs. a read-only custom-profile note: a `rectangle` profile is
    /// detected (returning its in-plane spans) regardless of plane; an L-shape and a
    /// degenerate (triangle) profile are not; and a four-point profile whose corners
    /// are NOT the bounding-box corners (a non-axis-aligned quad) is rejected.
    #[test]
    fn rectangle_in_plane_spans_detection() {
        // A genuine rectangle on each plane reports its [width, depth] spans.
        for plane in [PlaneAxis::X, PlaneAxis::Y, PlaneAxis::Z] {
            let extrude = SketchSolid::extrude(Sketch::rectangle(plane, 6, 4), 3);
            assert_eq!(
                extrude.rectangle_in_plane_spans(),
                Some([6, 4]),
                "plane {plane:?} rectangle must report its spans"
            );
        }
        // An L-shape (six vertices) is not a rectangle.
        let l_profile = vec![
            SketchPoint::new(0, 0),
            SketchPoint::new(4, 0),
            SketchPoint::new(4, 2),
            SketchPoint::new(2, 2),
            SketchPoint::new(2, 4),
            SketchPoint::new(0, 4),
        ];
        assert_eq!(
            SketchSolid::extrude(Sketch::new(PlaneAxis::Z, l_profile), 1)
                .rectangle_in_plane_spans(),
            None,
            "an L-shape is not a rectangle"
        );
        // A four-point quad whose corners are NOT the bounding-box corners (a
        // diamond) must be rejected — its vertices lie on edge midpoints, not corners.
        let diamond = vec![
            SketchPoint::new(2, 0),
            SketchPoint::new(4, 2),
            SketchPoint::new(2, 4),
            SketchPoint::new(0, 2),
        ];
        assert_eq!(
            SketchSolid::extrude(Sketch::new(PlaneAxis::Z, diamond), 1)
                .rectangle_in_plane_spans(),
            None,
            "a diamond quad is not an axis-aligned rectangle"
        );
        // A degenerate (triangle / <4 point) profile is not a rectangle.
        let triangle = vec![
            SketchPoint::new(0, 0),
            SketchPoint::new(4, 0),
            SketchPoint::new(0, 4),
        ];
        assert_eq!(
            SketchSolid::extrude(Sketch::new(PlaneAxis::Z, triangle), 1)
                .rectangle_in_plane_spans(),
            None,
            "a triangle is not a rectangle"
        );
    }

    /// A non-rectangular extrude still matches between `grid_dimensions` and the
    /// resolved grid's `dimensions`, and respects the voxel cap predicate.
    #[test]
    fn grid_dimensions_consistent_and_cap() {
        let extrude = SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, 6, 4), 5);
        let mut grid = VoxelGrid::default();
        extrude.resolve(&mut grid, 16);
        assert_eq!(grid.dimensions, extrude.grid_dimensions());
        assert!(!extrude.exceeds_voxel_cap());
    }

    // ----- Revolve operation (ADR 0003 §3i, the solid-of-revolution producer) -----

    use crate::sketch::RevolveAxis;

    /// Collect a producer's occupied voxels as a set of corner-anchored CELL indices
    /// `(i, j, k)` (`world − 0.5`). Used for IoU / overlap comparisons against an
    /// `SdfShape` of the same grid dimensions.
    fn cell_set(producer: &dyn VoxelProducer, density: u32) -> BTreeSet<[i64; 3]> {
        let mut grid = VoxelGrid::default();
        producer.resolve(&mut grid, density);
        grid.occupied
            .iter()
            .map(|voxel| {
                let position = voxel.world_position();
                [
                    (position[0] - 0.5).round() as i64,
                    (position[1] - 0.5).round() as i64,
                    (position[2] - 0.5).round() as i64,
                ]
            })
            .collect()
    }

    /// THE LOCK: a rectangle profile (radial 0..R, axial 0..H) revolved a full 360°
    /// about an axis oriented so the AXIAL world axis is Z must match the `Cylinder`
    /// `SdfShape` of diameter 2R, height H.
    ///
    /// Orientation: `PlaneAxis::X` + `RevolveAxis::InPlane1` ⇒ axial world axis = Z
    /// (the cylinder's Z-up vertical axis), radial world axes = {X, Y} (the circular
    /// cross-section). The profile coord (c0, c1) = (radial, axial), so
    /// `Sketch::rectangle(PlaneAxis::X, R, H)` is exactly the radial×axial rectangle.
    ///
    /// EXACT occupancy-set equality. The revolve rasterizes the rim with
    /// `radial = sqrt(x²+y²) <= R` via the polygon edge at radial R, while the SDF
    /// rasterizes `(|p_xy| − R_semi) <= 0`. Both compare the SAME centred radius to the
    /// SAME R (R = grid/2 = semi-axis), so the rim cells agree cell-for-cell and the
    /// equality holds EXACTLY (measured symmetric difference = 0 for both R parities).
    /// Covered for an EVEN and an ODD radial extent at density 16 (parity).
    #[test]
    fn rectangle_revolve_equals_cylinder() {
        let density = 16u32;
        // (radial_extent R in voxels, axial height H in voxels). 2R is the diameter.
        // R=32 → even diameter 64; R=33 → odd radial extent (diameter 66) — exercise
        // both R parities at d16.
        let cases = [(32i64, 48i64), (33, 47)];
        for (radial, axial) in cases {
            let revolve = SketchSolid::revolve(
                Sketch::rectangle(PlaneAxis::X, radial, axial),
                RevolveAxis::InPlane1,
                360,
            );
            // Cylinder of diameter 2R (X, Y) and height H (Z), pure-voxel size.
            let cylinder = SdfShape::from_voxels(
                ShapeKind::Cylinder,
                [(2 * radial) as u32, (2 * radial) as u32, axial as u32],
                1,
            );
            assert_eq!(
                revolve.grid_dimensions(),
                cylinder.grid_dimensions(density),
                "grid dims must match for radial {radial}, axial {axial}"
            );
            assert_eq!(
                cell_set(&revolve, density),
                cell_set(&cylinder, density),
                "rectangle revolve must EXACTLY equal Cylinder for radial {radial}, axial {axial}"
            );
        }
    }

    /// FOLD CORRECTNESS: a solid of revolution is symmetric about its axis, so the
    /// resolve folds both radial signs. (a) A rectangle authored entirely on the
    /// NEGATIVE radial side ([−30, −20]) revolves to the SAME tube as the same
    /// rectangle mirrored to the positive side ([20, 30]). (b) A profile STRADDLING
    /// the axis fills the union of both sides — the |radial|-folded region — and is
    /// non-empty. Without two-sided folding the negative-side rectangle would resolve
    /// to nothing and a straddling profile would lose its negative half.
    #[test]
    fn revolve_negative_and_straddling_radial_fold() {
        let density = 8u32;
        let axial = 24i64;

        // (a) Negative-side rectangle == positive mirror (same tube).
        // Profile (radial, axial) = (c0, c1); radial spans [−30, −20] vs [20, 30].
        let negative_side = SketchSolid::revolve(
            Sketch::new(
                PlaneAxis::X,
                vec![
                    SketchPoint::new(-30, 0),
                    SketchPoint::new(-20, 0),
                    SketchPoint::new(-20, axial),
                    SketchPoint::new(-30, axial),
                ],
            ),
            RevolveAxis::InPlane1,
            360,
        );
        let positive_mirror = SketchSolid::revolve(
            Sketch::new(
                PlaneAxis::X,
                vec![
                    SketchPoint::new(20, 0),
                    SketchPoint::new(30, 0),
                    SketchPoint::new(30, axial),
                    SketchPoint::new(20, axial),
                ],
            ),
            RevolveAxis::InPlane1,
            360,
        );
        assert_eq!(
            negative_side.grid_dimensions(),
            positive_mirror.grid_dimensions(),
            "negative-side and positive-mirror tubes must share grid dims (radial_max folds by abs)"
        );
        let negative_cells = cell_set(&negative_side, density);
        assert!(!negative_cells.is_empty(), "negative-side rectangle must NOT be empty");
        assert_eq!(
            negative_cells,
            cell_set(&positive_mirror, density),
            "negative-side rectangle must revolve to the same tube as its positive mirror"
        );

        // (b) Straddling profile fills the |radial|-folded region. A rectangle radial
        // [−15, 25] straddles the axis; its |radial| union covers [0, 25] (the larger
        // side dominates). Equivalent to a SOLID disc rectangle radial [0, 25] revolved
        // — the straddle's smaller (−15) side is wholly subsumed by the +25 side.
        let straddling = SketchSolid::revolve(
            Sketch::new(
                PlaneAxis::X,
                vec![
                    SketchPoint::new(-15, 0),
                    SketchPoint::new(25, 0),
                    SketchPoint::new(25, axial),
                    SketchPoint::new(-15, axial),
                ],
            ),
            RevolveAxis::InPlane1,
            360,
        );
        let straddling_cells = cell_set(&straddling, density);
        assert!(!straddling_cells.is_empty(), "straddling profile must NOT be empty");
        // The folded region is the disc of radius max(|−15|, 25) = 25 — a solid disc,
        // because the profile spans through radial 0 (no inner hole). Compare against a
        // one-sided solid rectangle radial [0, 25]: same axial extent, same outer radius,
        // both solid to the axis ⇒ identical folded occupancy.
        let solid_disc = SketchSolid::revolve(
            Sketch::rectangle(PlaneAxis::X, 25, axial),
            RevolveAxis::InPlane1,
            360,
        );
        assert_eq!(
            straddling.grid_dimensions(),
            solid_disc.grid_dimensions(),
            "straddling profile diameter is 2·max(|radial|) = 50, matching the solid disc"
        );
        assert_eq!(
            straddling_cells,
            cell_set(&solid_disc, density),
            "straddling profile must fill the |radial|-folded solid disc"
        );
    }

    /// A polygon approximating a half-disc (radial profile of a semicircle) revolved
    /// 360° about Z approximates a `Sphere` — IoU overlap >= 0.97. This is a TOLERANCE
    /// assertion, not exact: the profile is a many-segment polygon approximating the
    /// circular arc, and the SDF sphere uses its own ellipsoid isolevel, so the two
    /// rims never coincide exactly (the polygon under/over-shoots the arc per segment).
    #[test]
    fn half_disc_revolve_approximates_sphere() {
        let density = 16u32;
        let radius = 40i64; // sphere radius in voxels ⇒ diameter 80
        // Half-disc profile in (radial, axial) = (c0, c1): the radial extent is the
        // sphere radius at each axial height. Axial runs 0..2R; radial(axial) =
        // sqrt(R² − (axial − R)²). Many segments ⇒ a close polygon arc. The flat side
        // (radial = 0) is the revolve axis, so revolving the half-disc gives a sphere.
        let segments = 64;
        let mut profile = vec![SketchPoint::new(0, 0)]; // bottom pole on the axis
        for step in 0..=segments {
            let axial = (2 * radius) * step / segments;
            let dz = (axial - radius) as f64;
            let r = ((radius * radius) as f64 - dz * dz).max(0.0).sqrt();
            profile.push(SketchPoint::new(r.round() as i64, axial));
        }
        profile.push(SketchPoint::new(0, 2 * radius)); // top pole on the axis
        let revolve =
            SketchSolid::revolve(Sketch::new(PlaneAxis::X, profile), RevolveAxis::InPlane1, 360);
        let sphere = SdfShape::from_voxels(
            ShapeKind::Sphere,
            [(2 * radius) as u32, (2 * radius) as u32, (2 * radius) as u32],
            1,
        );
        assert_eq!(revolve.grid_dimensions(), sphere.grid_dimensions(density));
        let revolve_set = cell_set(&revolve, density);
        let sphere_set = cell_set(&sphere, density);
        let intersection = revolve_set.intersection(&sphere_set).count();
        let union = revolve_set.union(&sphere_set).count();
        let iou = intersection as f64 / union as f64;
        assert!(iou >= 0.97, "half-disc revolve IoU vs sphere {iou} < 0.97");
    }

    /// EDGE CASE: degenerate revolve profiles resolve to empty without panicking —
    /// fewer than 3 points, zero radial extent, and a zero turn. (Mirror of
    /// `degenerate_profiles_are_empty` for the revolve arm.)
    #[test]
    fn revolve_degenerate_profiles_are_empty() {
        let empty = |producer: &SketchSolid| {
            let mut grid = VoxelGrid::default();
            producer.resolve(&mut grid, 4);
            assert!(grid.occupied.is_empty());
            assert_eq!(grid.dimensions, [0, 0, 0]);
        };
        // < 3 points.
        empty(&SketchSolid::revolve(
            Sketch::new(PlaneAxis::X, vec![SketchPoint::new(0, 0), SketchPoint::new(4, 0)]),
            RevolveAxis::InPlane1,
            360,
        ));
        // Zero radial extent: a profile collinear on the radial axis (radial coord all
        // 0) — profile_bounds rejects the zero-span axis ⇒ empty.
        empty(&SketchSolid::revolve(
            Sketch::new(
                PlaneAxis::X,
                vec![
                    SketchPoint::new(0, 0),
                    SketchPoint::new(0, 4),
                    SketchPoint::new(0, 8),
                ],
            ),
            RevolveAxis::InPlane1,
            360,
        ));
        // Zero turn.
        empty(&SketchSolid::revolve(
            Sketch::rectangle(PlaneAxis::X, 8, 8),
            RevolveAxis::InPlane1,
            0,
        ));
    }

    /// PARITY: a rectangle revolved on EACH `RevolveAxis` at even and odd diameters is
    /// corner-anchored with NO straddle — occupancy spans exactly `[0, dim)` per axis,
    /// and the disc is symmetric about the grid centre on the two radial axes.
    #[test]
    fn revolve_parity_axis_placement() {
        let density = 8u32;
        // (radial R, axial H): R=10 ⇒ even diameter 20; R=11 ⇒ odd-extent, diameter 22.
        for (radial, axial) in [(10i64, 12i64), (11, 13)] {
            for plane in [PlaneAxis::X, PlaneAxis::Y, PlaneAxis::Z] {
                for revolve_axis in [RevolveAxis::InPlane0, RevolveAxis::InPlane1] {
                    // Place radial on c0, axial on c1 for InPlane1; swapped for InPlane0.
                    let sketch = match revolve_axis {
                        RevolveAxis::InPlane1 => Sketch::rectangle(plane, radial, axial),
                        RevolveAxis::InPlane0 => Sketch::rectangle(plane, axial, radial),
                    };
                    let revolve = SketchSolid::revolve(sketch, revolve_axis, 360);
                    let dims = revolve.grid_dimensions();
                    let mut grid = VoxelGrid::default();
                    revolve.resolve(&mut grid, density);
                    assert!(!grid.occupied.is_empty(), "{plane:?}/{revolve_axis:?} empty");

                    // No straddle: every cell index is within [0, dim) per axis, and the
                    // occupancy touches 0 and dim-1 on the radial axes (the disc spans
                    // the full diameter symmetric about the centre).
                    let mut min_cell = [i64::MAX; 3];
                    let mut max_cell = [i64::MIN; 3];
                    for voxel in &grid.occupied {
                        let position = voxel.world_position();
                        for axis in 0..3 {
                            let cell = (position[axis] - 0.5).round() as i64;
                            assert!(
                                cell >= 0 && (cell as u32) < dims[axis],
                                "{plane:?}/{revolve_axis:?}: cell {cell} out of [0,{}) on axis {axis}",
                                dims[axis]
                            );
                            min_cell[axis] = min_cell[axis].min(cell);
                            max_cell[axis] = max_cell[axis].max(cell);
                        }
                    }
                    // Identify the two radial world axes (full-diameter, symmetric span).
                    let [ip0, ip1] = plane.in_plane_axes();
                    let normal = plane.normal_axis();
                    let radial_axes: [usize; 2] = match revolve_axis {
                        RevolveAxis::InPlane0 => {
                            let mut a = [ip1, normal];
                            a.sort_unstable();
                            a
                        }
                        RevolveAxis::InPlane1 => {
                            let mut a = [ip0, normal];
                            a.sort_unstable();
                            a
                        }
                    };
                    for &axis in &radial_axes {
                        // The widest slice (through the rectangle's full radial extent)
                        // spans the whole diameter, touching both ends ⇒ no straddle and
                        // symmetric about the centre.
                        assert_eq!(
                            min_cell[axis], 0,
                            "{plane:?}/{revolve_axis:?}: radial axis {axis} does not start at 0"
                        );
                        assert_eq!(
                            max_cell[axis] as u32,
                            dims[axis] - 1,
                            "{plane:?}/{revolve_axis:?}: radial axis {axis} does not reach dim-1"
                        );
                    }
                }
            }
        }
    }

    /// PARTIAL turn is inert at 360: a 360° revolve is byte-identical (occupancy SET
    /// equal) to one built without the partial path engaging — proves the atan2 gate
    /// never fires at a full turn (theta ∈ [0,360) is never > 360).
    #[test]
    fn partial_revolve_360_equals_full() {
        let density = 16u32;
        let full_a = SketchSolid::revolve(
            Sketch::rectangle(PlaneAxis::X, 32, 48),
            RevolveAxis::InPlane1,
            360,
        );
        // The "full" reference is the same operation; this asserts determinism AND that
        // the 360 gate produces the same occupancy as the cylinder lock above relies on.
        let full_b = SketchSolid::revolve(
            Sketch::rectangle(PlaneAxis::X, 32, 48),
            RevolveAxis::InPlane1,
            360,
        );
        assert_eq!(
            cell_set(&full_a, density),
            cell_set(&full_b, density),
            "360° revolve must be deterministic / gate-inert"
        );
        // And it equals the matching cylinder (the partial gate did not eat any cells).
        let cylinder = SdfShape::from_voxels(ShapeKind::Cylinder, [64, 64, 48], 1);
        let diff = cell_set(&full_a, density)
            .symmetric_difference(&cell_set(&cylinder, density))
            .count();
        let total = cell_set(&cylinder, density).len();
        assert!(diff * 100 <= total, "360 revolve diff from cylinder {diff} > 1%");
    }

    /// PARTIAL turn 180° is roughly half a 360° revolve, and one angular half of the
    /// disc is empty (structural). The angle is measured from radial_a toward radial_b
    /// (ascending world-axis index); for PlaneAxis::X + InPlane1, radial_a=X, radial_b=Y,
    /// so the kept wedge is theta ∈ [0,180] ⇒ the centred-Y < 0 half is empty.
    #[test]
    fn partial_revolve_180_is_half() {
        let density = 8u32;
        let full = SketchSolid::revolve(
            Sketch::rectangle(PlaneAxis::X, 24, 32),
            RevolveAxis::InPlane1,
            360,
        );
        let half = SketchSolid::revolve(
            Sketch::rectangle(PlaneAxis::X, 24, 32),
            RevolveAxis::InPlane1,
            180,
        );
        let full_count = cell_set(&full, density).len();
        let half_count = cell_set(&half, density).len();
        let ratio = half_count as f64 / full_count as f64;
        assert!(
            (0.40..=0.60).contains(&ratio),
            "180° revolve count ratio {ratio} not ~0.5"
        );
        // Structural: the half with theta > 180 is empty. For PlaneAxis::X + InPlane1,
        // radial_a = X (idx 0), radial_b = Y (idx 1); theta > 180 ⇔ centred-Y < 0. The
        // grid is [48,48,32]; centred-Y < 0 means cell-Y < 24.
        let mut grid = VoxelGrid::default();
        half.resolve(&mut grid, density);
        let dim_y = grid.dimensions[1];
        let any_in_lower_half = grid.occupied.iter().any(|voxel| {
            let cell_y = (voxel.world_position()[1] - 0.5).round() as i64;
            // centred-Y = cell_y + 0.5 - dim_y/2 < 0
            (cell_y as f32 + 0.5 - dim_y as f32 / 2.0) < 0.0
        });
        assert!(!any_in_lower_half, "180° revolve leaked into the theta>180 half");
    }
}
