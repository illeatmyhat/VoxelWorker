//! Planar-face derivation for a sketch (ADR 0030 §2/§3, issue #100).
//!
//! A region is a **bounded face of the planar graph** whose nodes are points and whose edges are
//! segments and arcs — not a face of the full geometric arrangement. Two edges that visually cross
//! without a shared point bound nothing; snapping a point at the crossing is what creates the face,
//! so derivation stays a deterministic graph walk rather than a continuous intersection solver.
//!
//! The walk is the textbook DCEL one: each edge becomes two half-edges, and the successor of a
//! half-edge arriving at a vertex is the edge **immediately clockwise** from the one it came in on.
//! That traces every bounded face counter-clockwise and each connected component's unbounded face
//! clockwise, so the signed area's sign is what tells them apart.
//!
//! Nesting is deliberately NOT computed here. Two disjoint loops derive as two faces, and a hole
//! appears only because the author unpicked the inner one — the 2D CSG in
//! [`substrate::geom2d::signed_distance_to_region`] then subtracts it from whatever contains it.

use std::collections::{BTreeSet, HashMap};

use super::{arc_interior_points, EntityId, EntityRole, Sketch, SketchPoint};

/// A derived face's identity: the sorted set of the `origin` ids of its boundary edges (ADR 0030
/// §3). It survives the edits that leave a face the same face — dragging a vertex touches no
/// origin, and splitting an edge gives both children the parent's origin — and changes when the
/// boundary genuinely does, which is exactly when a pick should reset.
///
/// Ordered so the document's `unpicked` set is stable on disk and comparisons are cheap.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct FaceKey(Vec<EntityId>);

impl FaceKey {
    /// The key of a face bounded by the edges with these `origin` ids. Sorted and deduplicated:
    /// the key is a SET, so traversal order and a repeated origin (an edge split into children
    /// that both bound this face) cannot change it.
    pub fn from_origins(origins: impl IntoIterator<Item = EntityId>) -> Self {
        let set: BTreeSet<EntityId> = origins.into_iter().collect();
        FaceKey(set.into_iter().collect())
    }

    /// The boundary origin ids, ascending.
    pub fn origins(&self) -> &[EntityId] {
        &self.0
    }
}

/// One derived bounded face of the sketch graph.
#[derive(Debug, Clone, PartialEq)]
pub struct Face {
    /// The face's identity across re-derivation (ADR 0030 §3).
    pub key: FaceKey,
    /// The boundary as a closed simple polygon, counter-clockwise, with arcs tessellated —
    /// exactly what the region CSG and the overlay consume.
    pub boundary: Vec<SketchPoint>,
    /// The enclosed area in square voxels. Positive by construction (a clockwise cycle is an
    /// unbounded face and never becomes a `Face`); used to order nested faces smallest-first so
    /// a click inside a pocket picks the pocket, not the shape around it.
    pub area_voxels: f64,
}

/// One connected piece of the RESOLVED 2D material region: an outer boundary with the voids inside
/// it. The pieces never overlap and never nest, so a translucent wash over them composites once —
/// which is the difference between a fill and a stack of fills (`substrate::geom2d::point_in_region`
/// is the truth being mirrored: material means inside a `Fill` loop and inside no `Hole` loop).
#[derive(Debug, Clone, PartialEq)]
pub struct MaterialComponent {
    /// The outer boundary, counter-clockwise, arcs tessellated.
    pub outer: Vec<SketchPoint>,
    /// The voids inside it — the unpicked faces this piece encloses, shallowest ones only.
    pub holes: Vec<Vec<SketchPoint>>,
}

/// Resolve the derived faces into non-overlapping material pieces.
///
/// A face's polygon is its own cycle, so a face that encloses another CONTAINS it — the areas
/// overlap even though the faces are distinct. Nesting is therefore what turns a face list into a
/// region: a piece of material is a top-level PICKED face, and its voids are the shallowest unpicked
/// faces beneath it. Nothing deeper needs enumerating, because a `Hole` vetoes everything inside it
/// (an island of picked faces inside an unpicked one is not material — the resolve says so, so the
/// wash must too).
pub fn material_components(sketch: &Sketch) -> Vec<MaterialComponent> {
    let faces = derive(sketch);
    let parents = nesting_parents(&faces);
    let picked: Vec<bool> = faces
        .iter()
        .map(|face| sketch.face_is_picked(&face.key))
        .collect();
    faces
        .iter()
        .enumerate()
        .filter(|&(index, _)| parents[index].is_none() && picked[index])
        .map(|(root, face)| MaterialComponent {
            outer: face.boundary.clone(),
            holes: (0..faces.len())
                .filter(|&index| shallowest_void_under(root, index, &parents, &picked))
                .map(|index| faces[index].boundary.clone())
                .collect(),
        })
        .collect()
}

/// Whether face `candidate` is one of `root`'s voids: unpicked, inside `root`, and with every face
/// between the two picked (a void inside a void is already void).
fn shallowest_void_under(
    root: usize,
    candidate: usize,
    parents: &[Option<usize>],
    picked: &[bool],
) -> bool {
    if picked[candidate] {
        return false;
    }
    let mut walk = parents[candidate];
    while let Some(ancestor) = walk {
        if !picked[ancestor] {
            return false;
        }
        if ancestor == root {
            return true;
        }
        walk = parents[ancestor];
    }
    false
}

/// Each face's parent — the SMALLEST face strictly containing it, or `None` at the top level.
///
/// Faces of a planar graph never partially overlap, so one interior point decides containment. The
/// point comes from the face's own triangle fan rather than its centroid, which a concave face can
/// place outside itself.
fn nesting_parents(faces: &[Face]) -> Vec<Option<usize>> {
    let outlines: Vec<Vec<[f64; 2]>> = faces
        .iter()
        .map(|face| face.boundary.iter().map(|point| point.in_plane()).collect())
        .collect();
    let interiors: Vec<[f64; 2]> = outlines
        .iter()
        .map(|outline| interior_of(outline))
        .collect();
    (0..faces.len())
        .map(|inner| {
            (0..faces.len())
                .filter(|&outer| {
                    outer != inner
                        && faces[outer].area_voxels > faces[inner].area_voxels
                        && substrate::geom2d::point_in_polygon(
                            &to_f32(&outlines[outer]),
                            [interiors[inner][0] as f32, interiors[inner][1] as f32],
                        )
                })
                .min_by(|&a, &b| faces[a].area_voxels.total_cmp(&faces[b].area_voxels))
        })
        .collect()
}

/// A point strictly inside `outline` — the centroid of the first triangle of its fan, which is
/// inside however concave the outline is. Falls back to the first vertex for a degenerate outline.
fn interior_of(outline: &[[f64; 2]]) -> [f64; 2] {
    let fan = substrate::geom2d::triangulate_simple_polygon(outline);
    match fan.first() {
        Some(&[a, b, c]) => [
            (outline[a][0] + outline[b][0] + outline[c][0]) / 3.0,
            (outline[a][1] + outline[b][1] + outline[c][1]) / 3.0,
        ],
        None => outline.first().copied().unwrap_or([0.0, 0.0]),
    }
}

/// The `f32` narrowing `point_in_polygon` takes (see that module's width note).
fn to_f32(outline: &[[f64; 2]]) -> Vec<[f32; 2]> {
    outline
        .iter()
        .map(|point| [point[0] as f32, point[1] as f32])
        .collect()
}

/// One directed traversal of an edge, tagged with the lineage the face key is built from.
struct HalfEdge {
    from: EntityId,
    /// The entity id of the edge itself — half-edges are keyed by `(edge, forward)`.
    edge: EntityId,
    origin: EntityId,
    /// The outgoing direction at `from`, as an angle in `(-pi, pi]`. An arc leaves along its
    /// TANGENT, not its chord, so two arcs sharing endpoints still order correctly.
    departure: f64,
}

/// Every bounded face of the sketch's planar graph, deterministically ordered by key.
///
/// Construction geometry is skipped: a construction edge never bounds a region (ADR 0030 §1).
pub fn derive(sketch: &Sketch) -> Vec<Face> {
    let position = |id: EntityId| {
        sketch
            .points
            .iter()
            .find(|point| point.id == id)
            .map(|point| point.at)
    };
    // Half-edges, both directions per real edge, with each end's departure direction.
    let mut half_edges: Vec<HalfEdge> = Vec::new();
    let mut interiors: HashMap<EntityId, Vec<SketchPoint>> = HashMap::new();
    for segment in sketch
        .segments
        .iter()
        .filter(|s| s.role == EntityRole::Real)
    {
        let (Some(from), Some(to)) = (position(segment.from), position(segment.to)) else {
            continue;
        };
        push_half_edges(
            &mut half_edges,
            segment.id,
            segment.origin,
            (segment.from, from.in_plane()),
            (segment.to, to.in_plane()),
            &[],
        );
    }
    for arc in sketch.arcs.iter().filter(|a| a.role == EntityRole::Real) {
        let (Some(from), Some(to)) = (position(arc.from), position(arc.to)) else {
            continue;
        };
        let interior =
            arc_interior_points(from.in_plane(), to.in_plane(), arc.bulge.to_degrees_f64());
        push_half_edges(
            &mut half_edges,
            arc.id,
            arc.origin,
            (arc.from, from.in_plane()),
            (arc.to, to.in_plane()),
            &interior,
        );
        interiors.insert(arc.id, interior);
    }
    if half_edges.is_empty() {
        return Vec::new();
    }

    // The cyclic order of departures around each vertex, counter-clockwise. Ties (two edges
    // leaving in the same direction) break by half-edge index so the order is total.
    let mut around: HashMap<EntityId, Vec<usize>> = HashMap::new();
    for (index, half) in half_edges.iter().enumerate() {
        around.entry(half.from).or_default().push(index);
    }
    for fan in around.values_mut() {
        fan.sort_by(|&a, &b| {
            half_edges[a]
                .departure
                .partial_cmp(&half_edges[b].departure)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
    }
    // Where each half-edge sits in its own vertex's fan, so `next` is an index step.
    let mut slot: HashMap<usize, usize> = HashMap::new();
    for fan in around.values() {
        for (position_in_fan, &index) in fan.iter().enumerate() {
            slot.insert(index, position_in_fan);
        }
    }
    let mut visited = vec![false; half_edges.len()];
    let mut faces: Vec<Face> = Vec::new();
    for start in 0..half_edges.len() {
        if visited[start] {
            continue;
        }
        let mut cycle: Vec<usize> = Vec::new();
        let mut current = start;
        let mut closed = false;
        for _ in 0..half_edges.len() {
            if visited[current] {
                // Every half-edge belongs to exactly one face cycle, so this is unreachable for
                // a well-formed graph; drop the partial trace rather than emit a torn face.
                break;
            }
            visited[current] = true;
            cycle.push(current);
            // Arrive along `current`; leave along the edge immediately CLOCKWISE from the way
            // back. That keeps the face's interior on the left, so a bounded face comes out
            // counter-clockwise and its component's unbounded face clockwise.
            let back = current ^ 1;
            let fan = &around[&half_edges[back].from];
            let here = slot[&back];
            current = fan[(here + fan.len() - 1) % fan.len()];
            if current == start {
                closed = true;
                break;
            }
        }
        if !closed {
            continue;
        }
        if let Some(face) = face_from_cycle(sketch, &half_edges, &interiors, &cycle) {
            faces.push(face);
        }
    }
    faces.sort_by(|a, b| {
        a.key
            .cmp(&b.key)
            .then(b.area_voxels.total_cmp(&a.area_voxels))
    });
    faces
}

/// Append the two half-edges of one edge — ALWAYS as a pair, so twins are neighbours and
/// `index ^ 1` is the twin. `interior` is the arc's chord fan from tail to head
/// (empty for a segment); the departure angle at each end points at the first thing along the
/// edge, which for an arc is its tangent rather than its chord.
fn push_half_edges(
    into: &mut Vec<HalfEdge>,
    edge: EntityId,
    origin: EntityId,
    tail: (EntityId, [f64; 2]),
    head: (EntityId, [f64; 2]),
    interior: &[SketchPoint],
) {
    let first = interior.first().map(|p| p.in_plane()).unwrap_or(head.1);
    let last = interior.last().map(|p| p.in_plane()).unwrap_or(tail.1);
    into.push(HalfEdge {
        from: tail.0,
        edge,
        origin,
        departure: (first[1] - tail.1[1]).atan2(first[0] - tail.1[0]),
    });
    into.push(HalfEdge {
        from: head.0,
        edge,
        origin,
        departure: (last[1] - head.1[1]).atan2(last[0] - head.1[0]),
    });
}

/// Turn a traced half-edge cycle into a bounded face, or `None` when the cycle is a component's
/// unbounded face (clockwise ⇒ non-positive area) or degenerate (a whisker walked out and back,
/// which encloses nothing).
fn face_from_cycle(
    sketch: &Sketch,
    half_edges: &[HalfEdge],
    interiors: &HashMap<EntityId, Vec<SketchPoint>>,
    cycle: &[usize],
) -> Option<Face> {
    let position = |id: EntityId| {
        sketch
            .points
            .iter()
            .find(|point| point.id == id)
            .map(|point| point.at)
    };
    let mut boundary: Vec<SketchPoint> = Vec::with_capacity(cycle.len());
    for &index in cycle {
        let half = &half_edges[index];
        boundary.push(position(half.from)?);
        if let Some(interior) = interiors.get(&half.edge) {
            // The fan is stored tail→head; walked the other way it is the same points reversed.
            let forward = sketch
                .arcs
                .iter()
                .any(|arc| arc.id == half.edge && arc.from == half.from);
            if forward {
                boundary.extend(interior.iter().copied());
            } else {
                boundary.extend(interior.iter().rev().copied());
            }
        }
    }
    let area = signed_area(&boundary);
    if area <= AREA_EPSILON_SQUARE_VOXELS {
        return None;
    }
    Some(Face {
        key: FaceKey::from_origins(cycle.iter().map(|&index| half_edges[index].origin)),
        boundary,
        area_voxels: area,
    })
}

/// Below this a traced cycle encloses nothing worth resolving — a whisker walked out and back, or
/// a loop collapsed onto a line. Sub-voxel coordinates make an exact zero unreliable, so the
/// threshold is a hundredth of a voxel of area rather than `0.0`.
const AREA_EPSILON_SQUARE_VOXELS: f64 = 1.0e-2;

/// Twice-the-shoelace, halved: positive counter-clockwise, negative clockwise.
fn signed_area(boundary: &[SketchPoint]) -> f64 {
    let count = boundary.len();
    if count < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut previous = boundary[count - 1].in_plane();
    for point in boundary {
        let current = point.in_plane();
        sum += previous[0] * current[1] - current[0] * previous[1];
        previous = current;
    }
    sum * 0.5
}
