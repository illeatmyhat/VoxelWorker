//! The derived region, remembered between queries.
//!
//! [`SketchSolid::signed_distance`](super::solid::SketchSolid::signed_distance) is asked once per
//! voxel sample. Running the whole arrangement per ask — twice, since the profile bounds derive it
//! too — is a 650× tax on the composite fold, which measures a field rather than rasterizing a
//! producer.
//!
//! The cache validates by **comparing the entity store it was derived from**, not by a flag a
//! mutator has to remember to clear. A missed invalidation is a stale-geometry bug that does not
//! look like a cache bug, so the invariant is enforced by construction instead: nothing can go
//! stale that the comparison would not catch. A `NaN` coordinate simply never compares equal and
//! forces a miss, which is slow and correct rather than fast and wrong.
//!
//! The value is handed out behind an [`Arc`] so no lock is held while a caller uses it — the
//! callers re-enter (`signed_distance` asks for the bounds and then the region), and a recursive
//! read lock is free to deadlock.

use std::sync::{Arc, RwLock};

use substrate::geom2d::{LoopRole, RegionEdge};

use super::{Arc as ArcEntity, Circle, FaceKey, PlaneAxis, Point, ProfileLoop, Segment, Sketch};

/// Everything the per-sample paths ask of the entity store, derived once.
pub(crate) struct Derived {
    /// The tagged loops — what [`Sketch::region`] hands out.
    pub region: Vec<ProfileLoop>,
    /// The same region in the measurement width, which the field folds per sample.
    pub region_field_loops: Vec<(LoopRole, Vec<RegionEdge>)>,
    /// The `Fill` loops' bounding box, the profile's footprint.
    pub filled_extent: Option<([f64; 2], [f64; 2])>,
}

impl Derived {
    fn of(sketch: &Sketch) -> Self {
        let region = sketch.region_uncached();
        let region_field_loops = super::produce::to_region_edges_measured(&region);
        let filled_extent = filled_extent(&region);
        Self {
            region,
            region_field_loops,
            filled_extent,
        }
    }
}

/// The `Fill` loops' bounding box. A hole adds no footprint, and an unpicked face with nothing
/// around it is not occupancy.
fn filled_extent(region: &[ProfileLoop]) -> Option<([f64; 2], [f64; 2])> {
    let mut extent: Option<([f64; 2], [f64; 2])> = None;
    for profile_loop in region.iter().filter(|loop_| loop_.role == LoopRole::Fill) {
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
    }
    extent
}

/// The entity store as it stood when the [`Derived`] beside it was computed.
struct Snapshot {
    plane: PlaneAxis,
    points: Vec<Point>,
    segments: Vec<Segment>,
    arcs: Vec<ArcEntity>,
    circles: Vec<Circle>,
    unpicked_points: Vec<FaceKey>,
}

impl Snapshot {
    fn of(sketch: &Sketch) -> Self {
        Self {
            plane: sketch.plane,
            points: sketch.points.clone(),
            segments: sketch.segments.clone(),
            arcs: sketch.arcs.clone(),
            circles: sketch.circles.clone(),
            unpicked_points: sketch.unpicked_points.clone(),
        }
    }

    /// Whether `sketch` would derive to the same region. Length mismatches short-circuit, so the
    /// common miss — an entity just added — costs two integer comparisons.
    fn matches(&self, sketch: &Sketch) -> bool {
        self.plane == sketch.plane
            && self.points == sketch.points
            && self.segments == sketch.segments
            && self.arcs == sketch.arcs
            && self.circles == sketch.circles
            && self.unpicked_points == sketch.unpicked_points
    }
}

/// A derivation and the store it came from, kept together so neither can be read without the
/// other.
struct Remembered {
    snapshot: Snapshot,
    derived: Arc<Derived>,
}

/// The cache cell on [`Sketch`]. It clones EMPTY and compares EQUAL: a cache is not identity, and
/// a copy of a sketch is the same sketch whether or not it has derived itself yet. That is what
/// lets `Sketch` keep its derived `Clone`/`PartialEq` with the cell inside it.
///
/// The contents are BOXED so the cell costs a sketch two words. A `Sketch` is a variant of two
/// scene enums, and a cache is not worth widening every node in the document by the size of a
/// snapshot it usually does not hold.
#[derive(Default)]
pub(super) struct RegionMemo(RwLock<Option<Box<Remembered>>>);

impl RegionMemo {
    /// The region derived from `sketch`, from the cache when the store has not moved since.
    ///
    /// A miss derives OUTSIDE the write lock, so two threads racing a miss both compute and the
    /// later one wins — wasteful once, never wrong. A poisoned lock degrades to deriving every
    /// time rather than propagating a panic into geometry.
    pub(super) fn derived(&self, sketch: &Sketch) -> Arc<Derived> {
        if let Ok(guard) = self.0.read() {
            if let Some(held) = guard.as_deref() {
                if held.snapshot.matches(sketch) {
                    return Arc::clone(&held.derived);
                }
            }
        }
        let fresh = Arc::new(Derived::of(sketch));
        if let Ok(mut guard) = self.0.write() {
            *guard = Some(Box::new(Remembered {
                snapshot: Snapshot::of(sketch),
                derived: Arc::clone(&fresh),
            }));
        }
        fresh
    }
}

impl Clone for RegionMemo {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl PartialEq for RegionMemo {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl std::fmt::Debug for RegionMemo {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("RegionMemo")
    }
}
