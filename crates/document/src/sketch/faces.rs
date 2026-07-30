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
//! A [`Circle`](super::Circle) skips the walk entirely: a closed curve is already a loop, so it
//! contributes one face directly (ADR 0035 Decision 7). It needs no vertex to hang from, which is
//! why a circle drawn on an empty plane bounds a region where a lone arc bounds nothing.
//!
//! Nesting is deliberately NOT computed here. Two disjoint loops derive as two faces, and a hole
//! appears only because the author unpicked the inner one — the 2D CSG in
//! [`substrate::geom2d::signed_distance_to_region`] then subtracts it from whatever contains it.

use std::collections::{BTreeSet, HashMap};

use super::{EntityId, EntityRole, ProfileEdge, Sketch};

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
    /// The boundary as a closed loop of edges, counter-clockwise, **with its arcs intact** —
    /// exactly what the region CSG and the overlay consume.
    pub boundary: Vec<ProfileEdge>,
    /// The enclosed area in square voxels. Positive by construction (a clockwise cycle is an
    /// unbounded face and never becomes a `Face`); used to order nested faces smallest-first so
    /// a click inside a pocket picks the pocket, not the shape around it.
    pub area_voxels: f64,
}

/// One directed traversal of an edge, tagged with the lineage the face key is built from.
struct HalfEdge {
    from: EntityId,
    origin: EntityId,
    /// The outgoing direction at `from`, as an angle in `(-pi, pi]`. An arc leaves along its
    /// TANGENT, not its chord, so two arcs sharing endpoints still order correctly.
    departure: f64,
    /// The edge's geometry, oriented tail-first for THIS traversal.
    geometry: ProfileEdge,
}

/// Every bounded face of the sketch's planar graph, deterministically ordered by key, **with its
/// arcs intact** — the faces that are the profile's meaning.
///
/// There is no tolerance here, and no variant of this that takes one. A face's boundary is a loop
/// of [`ProfileEdge`]s, so the walk, the area and everything downstream read the curve itself;
/// flattening is something a consumer does at its own edge ([`super::ProfileLoop::flatten`]).
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
            segment.origin,
            (segment.from, segment.to),
            ProfileEdge::straight(from, to),
        );
    }
    for arc in sketch.arcs.iter().filter(|a| a.role == EntityRole::Real) {
        let (Some(from), Some(to)) = (position(arc.from), position(arc.to)) else {
            continue;
        };
        push_half_edges(
            &mut half_edges,
            arc.origin,
            (arc.from, arc.to),
            ProfileEdge::curved(from, to, arc.bulge.to_degrees_f64()),
        );
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
        if let Some(face) = face_from_cycle(&half_edges, &cycle) {
            faces.push(face);
        }
    }
    // A circle needs no walk: it closes on itself, so it IS a face (ADR 0035 Decision 7). It joins
    // the graph's faces here as a peer, and the region's ordered fold decides nesting from area the
    // same way it does for two disjoint squares.
    for circle in sketch.circles.iter().filter(|c| c.role == EntityRole::Real) {
        let Some(centre) = position(circle.center) else {
            continue;
        };
        let edge = ProfileEdge::circle(centre.in_plane(), circle.radius.value());
        let area = edge.signed_area_term();
        if area <= AREA_EPSILON_SQUARE_VOXELS {
            continue;
        }
        faces.push(Face {
            key: FaceKey::from_origins([circle.origin]),
            boundary: vec![edge],
            area_voxels: area,
        });
    }
    faces.sort_by(|a, b| {
        a.key
            .cmp(&b.key)
            .then(b.area_voxels.total_cmp(&a.area_voxels))
    });
    faces
}

/// Append the two half-edges of one edge — ALWAYS as a pair, so twins are neighbours and
/// `index ^ 1` is the twin. The departure angle at each end is the edge's own outgoing tangent
/// there ([`ProfileEdge::departure_radians`]), taken analytically: it used to be read off the
/// arc's first tessellated chord, which made the vertex ordering depend on how finely the arc had
/// been cut.
fn push_half_edges(
    into: &mut Vec<HalfEdge>,
    origin: EntityId,
    ends: (EntityId, EntityId),
    geometry: ProfileEdge,
) {
    let backward = geometry.reversed();
    into.push(HalfEdge {
        from: ends.0,
        origin,
        departure: geometry.departure_radians(),
        geometry,
    });
    into.push(HalfEdge {
        from: ends.1,
        origin,
        departure: backward.departure_radians(),
        geometry: backward,
    });
}

/// Turn a traced half-edge cycle into a bounded face, or `None` when the cycle is a component's
/// unbounded face (clockwise ⇒ non-positive area) or degenerate (a whisker walked out and back,
/// which encloses nothing).
fn face_from_cycle(half_edges: &[HalfEdge], cycle: &[usize]) -> Option<Face> {
    let boundary: Vec<ProfileEdge> = cycle
        .iter()
        .map(|&index| half_edges[index].geometry)
        .collect();
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

/// The enclosed signed area by Green's theorem: positive counter-clockwise, negative clockwise.
///
/// Each edge contributes `½∮(x dy − y dx)` over itself ([`ProfileEdge::signed_area_term`]), which
/// is the shoelace term for a straight span and the exact circular integral for an arc. A
/// two-edge cycle (an arc and its chord, say) genuinely encloses area, so there is no
/// minimum-vertex floor here — [`AREA_EPSILON_SQUARE_VOXELS`] is the only filter.
fn signed_area(boundary: &[ProfileEdge]) -> f64 {
    boundary.iter().map(ProfileEdge::signed_area_term).sum()
}
