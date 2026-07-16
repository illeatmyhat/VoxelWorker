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
