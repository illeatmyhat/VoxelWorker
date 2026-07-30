//! Planar-face derivation for a sketch (ADR 0030 §2/§3, ADR 0035 Decisions 8 and 9, issue #100).
//!
//! A region is a bounded face of the **geometric arrangement** of the sketch's curves: every curve
//! is cut at every intersection with every other curve, and the pieces form the graph whose bounded
//! faces are the regions. A crossing needs no shared point, so two overlapping circles are three
//! regions and a rectangle struck through by a line is two — neither of which the author has to
//! prepare by snapping a vertex at the crossing first (ADR 0035 Decision 8, retiring ADR 0030 §2's
//! shared-point requirement).
//!
//! The cutting is [`substrate::curve_intersection::cut_at_crossings`]; a piece of an arc is still an
//! arc of the same circle, so nothing is approximated by being split. What arrives here is a bag of
//! curve pieces, and the work is to weld their endpoints into arrangement vertices and walk.
//!
//! The walk is the textbook DCEL one: each piece becomes two half-edges, and the successor of a
//! half-edge arriving at a vertex is the edge **immediately clockwise** from the one it came in on.
//! That traces every bounded face counter-clockwise and each connected component's unbounded face
//! clockwise, so the signed area's sign is what tells them apart. A whole circle nobody crosses is a
//! self-loop at its own seam vertex and walks correctly with no special case: its forward half-edge
//! traces the disc, its reverse traces the unbounded face outside it.
//!
//! Nesting is deliberately NOT computed here. Two disjoint loops derive as two faces, and a hole
//! appears only because the author unpicked the inner one — the 2D CSG in
//! [`substrate::geom2d::signed_distance_to_region`] then subtracts it from whatever contains it.

use substrate::curve_intersection::{cut_at_crossings, PlanarCurve};
use substrate::geom2d::deepest_interior_point;

use super::{EntityId, EntityRole, ProfileArc, ProfileEdge, Sketch, SketchPoint};

/// A derived face's identity: **one point strictly inside it** (ADR 0035 Decision 9). A re-derived
/// face *is* that face when it still contains the stored point.
///
/// # Why not the boundary's lineage
///
/// The key used to be the set of `origin` ids of the boundary edges, which worked while a face was
/// a cycle of drawn entities. In an arrangement it is not: a face can be bounded by pieces of
/// curves the author never drew as separate things, and drawing one new line across a shape
/// renumbers the lineage of every face it touches. A point does not care. It survives a vertex
/// drag, an edge split, a curve added elsewhere, and the arrangement re-cutting the same face into
/// the same ground under a different set of pieces.
///
/// The point is the face's DEEPEST interior point ([`deepest_interior_point`]), the one with the
/// most room to survive an edit rather than merely the first one found.
///
/// Two failure modes are accepted rather than defended against. A face that shrinks past its own
/// sample point resets to picked — the author's carve is forgotten by an edit that made the face
/// substantially smaller. And a sample point that ends up inside a NEIGHBOURING face migrates the
/// unpick there. Both are visible immediately and undoable; the alternative is a lineage that
/// pretends to know more than it does.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FaceKey {
    /// A point strictly inside the face, in profile voxels.
    pub interior_point: [f32; 2],
}

impl FaceKey {
    /// The key naming the face that contains this point.
    pub fn at(interior_point: [f32; 2]) -> Self {
        FaceKey { interior_point }
    }
}

/// One derived bounded face of the sketch arrangement.
#[derive(Debug, Clone, PartialEq)]
pub struct Face {
    /// The face's identity across re-derivation (ADR 0035 Decision 9).
    pub key: FaceKey,
    /// The boundary as a closed loop of edges, counter-clockwise, **with its arcs intact** —
    /// exactly what the region CSG and the overlay consume.
    pub boundary: Vec<ProfileEdge>,
    /// The enclosed area in square voxels. Positive by construction (a clockwise cycle is an
    /// unbounded face and never becomes a `Face`); used to order nested faces smallest-first so
    /// a click inside a pocket picks the pocket, not the shape around it.
    pub area_voxels: f64,
}

impl Face {
    /// Whether `point` lies inside this face's own boundary loop. The containment test
    /// [`FaceKey`] identity rests on.
    pub fn contains(&self, point: [f32; 2]) -> bool {
        let boundary: Vec<substrate::geom2d::RegionEdge> =
            self.boundary.iter().map(ProfileEdge::measured).collect();
        substrate::geom2d::point_in_edge_loop(&boundary, point)
    }
}

/// One directed traversal of an arrangement piece.
struct HalfEdge {
    /// The arrangement vertex this traversal leaves.
    from: usize,
    /// The outgoing direction at `from`, as an angle in `(-pi, pi]`. An arc leaves along its
    /// TANGENT, not its chord, so two arcs sharing endpoints still order correctly.
    departure: f64,
    /// The piece's geometry, oriented tail-first for THIS traversal.
    geometry: ProfileEdge,
}

/// Every bounded face of the sketch's arrangement, deterministically ordered, **with its arcs
/// intact** — the faces that are the profile's meaning.
///
/// There is no tolerance here in the flattening sense, and no variant of this that takes one. A
/// face's boundary is a loop of [`ProfileEdge`]s, so the walk, the area and everything downstream
/// read the curve itself; flattening is something a consumer does at its own edge
/// ([`super::ProfileLoop::flatten`]).
///
/// Construction geometry is skipped: a construction edge never bounds a region (ADR 0030 §1).
pub fn derive(sketch: &Sketch) -> Vec<Face> {
    let drawn = drawn_curves(sketch);
    if drawn.is_empty() {
        return Vec::new();
    }
    let curves: Vec<PlanarCurve> = drawn.iter().map(|(_, curve)| *curve).collect();
    let mut vertices: Vec<[f64; 2]> = Vec::new();
    let mut half_edges: Vec<HalfEdge> = Vec::new();
    for pieces in cut_at_crossings(&curves) {
        for piece in pieces {
            let from = weld(&mut vertices, piece.start());
            let to = weld(&mut vertices, piece.end());
            // A piece whose ends welded together without closing on itself encloses nothing and
            // would seat a bogus self-loop in the fan.
            if from == to && !piece.is_closed() {
                continue;
            }
            push_half_edges(
                &mut half_edges,
                (from, to),
                profile_edge(piece, vertices[from], vertices[to]),
            );
        }
    }
    // The cyclic order of departures around each vertex, counter-clockwise. Ties (two edges
    // leaving in the same direction) break by half-edge index so the order is total.
    let mut around: Vec<Vec<usize>> = vec![Vec::new(); vertices.len()];
    for (index, half) in half_edges.iter().enumerate() {
        around[half.from].push(index);
    }
    for fan in &mut around {
        fan.sort_by(|&a, &b| {
            half_edges[a]
                .departure
                .total_cmp(&half_edges[b].departure)
                .then(a.cmp(&b))
        });
    }
    // Where each half-edge sits in its own vertex's fan, so `next` is an index step.
    let mut slot = vec![0usize; half_edges.len()];
    for fan in &around {
        for (position_in_fan, &index) in fan.iter().enumerate() {
            slot[index] = position_in_fan;
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
            let fan = &around[half_edges[back].from];
            let here = slot[back];
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
    key_by_own_ground(&mut faces);
    faces.sort_by(|a, b| {
        a.key.interior_point[0]
            .total_cmp(&b.key.interior_point[0])
            .then(a.key.interior_point[1].total_cmp(&b.key.interior_point[1]))
            .then(b.area_voxels.total_cmp(&a.area_voxels))
    });
    faces
}

/// Re-key every face that has other faces nested inside it, so its point sits in the ground it
/// actually governs.
///
/// A face's boundary is its OUTER loop alone — nesting is not in it — so the deepest point of that
/// loop can easily land inside a face nested within, which would make the key name the wrong face.
/// A ring's identity has to be in the ring. The nested faces are the ones whose (provisional)
/// point is inside this loop, and they enter the search as `Hole`s.
///
/// Two passes rather than one because deciding what is nested inside what needs a point known to
/// be inside each face, which is the very thing the first pass produces. Only faces that turn out
/// to have something nested inside them pay for the second search.
fn key_by_own_ground(faces: &mut [Face]) {
    faces.sort_by(|first, second| first.area_voxels.total_cmp(&second.area_voxels));
    for index in 0..faces.len() {
        // Innermost-first, which for `point_in_region` means area-ascending — and only faces
        // SMALLER than this one can be nested inside it.
        let mut loops: Vec<(
            substrate::geom2d::LoopRole,
            Vec<substrate::geom2d::RegionEdge>,
        )> = faces[..index]
            .iter()
            .filter(|inner| faces[index].contains(inner.key.interior_point))
            .map(|inner| {
                (
                    substrate::geom2d::LoopRole::Hole,
                    inner.boundary.iter().map(ProfileEdge::measured).collect(),
                )
            })
            .collect();
        if loops.is_empty() {
            continue;
        }
        loops.push((
            substrate::geom2d::LoopRole::Fill,
            faces[index]
                .boundary
                .iter()
                .map(ProfileEdge::measured)
                .collect(),
        ));
        if let Some((point, _)) = deepest_interior_point(&loops, INTERIOR_POINT_PRECISION_VOXELS) {
            faces[index].key = FaceKey::at(point);
        }
    }
}

/// The real (non-construction) curves the author has drawn, each as a
/// [`PlanarCurve`] the arrangement can cut, paired with the entity it came from.
///
/// A segment is its span, an arc is the circle its endpoints-plus-bulge form already solves for,
/// and a [`Circle`](super::Circle) is the whole turn — closed, and therefore a curve like any
/// other here rather than a face that skips the walk (ADR 0035 Decision 7).
fn drawn_curves(sketch: &Sketch) -> Vec<(EntityId, PlanarCurve)> {
    let position = |id: EntityId| {
        sketch
            .points
            .iter()
            .find(|point| point.id == id)
            .map(|point| point.at)
    };
    let mut curves: Vec<(EntityId, PlanarCurve)> = Vec::new();
    for segment in sketch
        .segments
        .iter()
        .filter(|s| s.role == EntityRole::Real)
    {
        let (Some(from), Some(to)) = (position(segment.from), position(segment.to)) else {
            continue;
        };
        curves.push((
            segment.origin,
            PlanarCurve::Segment {
                start: from.in_plane(),
                end: to.in_plane(),
            },
        ));
    }
    for arc in sketch.arcs.iter().filter(|a| a.role == EntityRole::Real) {
        let (Some(from), Some(to)) = (position(arc.from), position(arc.to)) else {
            continue;
        };
        // The bulge solve already lives in `ProfileEdge`; read the circle back off it rather than
        // keeping a second copy of that trigonometry here.
        let Some(solved) = ProfileEdge::curved(from, to, arc.bulge.to_degrees_f64()).arc else {
            continue;
        };
        curves.push((
            arc.origin,
            PlanarCurve::Arc {
                centre: solved.centre,
                radius: solved.radius,
                start_radians: solved.start_radians,
                sweep_radians: solved.sweep_radians,
            },
        ));
    }
    for circle in sketch.circles.iter().filter(|c| c.role == EntityRole::Real) {
        let Some(centre) = position(circle.center) else {
            continue;
        };
        curves.push((
            circle.origin,
            PlanarCurve::circle(centre.in_plane(), circle.radius.value()),
        ));
    }
    curves
}

/// How far apart two piece endpoints may be and still be ONE arrangement vertex, in voxels.
///
/// A crossing is solved independently on each of the two curves through it, so the two copies of
/// that point differ by rounding; they have to weld or the graph tears at every crossing. It is
/// well above [`substrate::curve_intersection::CROSSING_EPSILON`] and far below anything an author
/// can draw, so it never welds two points that were meant to be distinct.
const VERTEX_WELD_VOXELS: f64 = 1.0e-6;

/// The index of the arrangement vertex at `point`, appending one if nothing is already there.
fn weld(vertices: &mut Vec<[f64; 2]>, point: [f64; 2]) -> usize {
    let found = vertices.iter().position(|at| {
        (at[0] - point[0]).abs() <= VERTEX_WELD_VOXELS
            && (at[1] - point[1]).abs() <= VERTEX_WELD_VOXELS
    });
    match found {
        Some(index) => index,
        None => {
            vertices.push(point);
            vertices.len() - 1
        }
    }
}

/// One arrangement piece as a profile edge, its ends snapped to the CANONICAL vertex positions.
///
/// Both sides of a welded vertex must be the same value or the crossing parity there can count
/// twice or not at all ([`substrate::geom2d::RegionEdge`] states the same contract). The arc's
/// circle is untouched by the snap — a piece of an arc is a piece of the same circle.
fn profile_edge(piece: PlanarCurve, from: [f64; 2], to: [f64; 2]) -> ProfileEdge {
    let ends = (
        SketchPoint::from_continuous(from[0], from[1]),
        SketchPoint::from_continuous(to[0], to[1]),
    );
    match piece {
        PlanarCurve::Segment { .. } => ProfileEdge::straight(ends.0, ends.1),
        PlanarCurve::Arc {
            centre,
            radius,
            start_radians,
            sweep_radians,
        } => ProfileEdge {
            from: ends.0,
            to: ends.1,
            arc: Some(ProfileArc {
                centre,
                radius,
                start_radians,
                sweep_radians,
            }),
        },
    }
}

/// Append the two half-edges of one piece — ALWAYS as a pair, so twins are neighbours and
/// `index ^ 1` is the twin. The departure angle at each end is the edge's own outgoing tangent
/// there ([`ProfileEdge::departure_radians`]), taken analytically: it used to be read off the
/// arc's first tessellated chord, which made the vertex ordering depend on how finely the arc had
/// been cut.
fn push_half_edges(into: &mut Vec<HalfEdge>, ends: (usize, usize), geometry: ProfileEdge) {
    let backward = geometry.reversed();
    into.push(HalfEdge {
        from: ends.0,
        departure: geometry.departure_radians(),
        geometry,
    });
    into.push(HalfEdge {
        from: ends.1,
        departure: backward.departure_radians(),
        geometry: backward,
    });
}

/// Turn a traced half-edge cycle into a bounded face, or `None` when the cycle is a component's
/// unbounded face (clockwise ⇒ non-positive area), degenerate (a whisker walked out and back,
/// which encloses nothing), or too thin to hold an interior point to name it by.
fn face_from_cycle(half_edges: &[HalfEdge], cycle: &[usize]) -> Option<Face> {
    let boundary: Vec<ProfileEdge> = cycle
        .iter()
        .map(|&index| half_edges[index].geometry)
        .collect();
    let area = signed_area(&boundary);
    if area <= AREA_EPSILON_SQUARE_VOXELS {
        return None;
    }
    let measured: Vec<substrate::geom2d::RegionEdge> =
        boundary.iter().map(ProfileEdge::measured).collect();
    let (interior_point, _) = deepest_interior_point(
        &[(substrate::geom2d::LoopRole::Fill, measured)],
        INTERIOR_POINT_PRECISION_VOXELS,
    )?;
    Some(Face {
        key: FaceKey::at(interior_point),
        boundary,
        area_voxels: area,
    })
}

/// Below this a traced cycle encloses nothing worth resolving — a whisker walked out and back, or
/// a loop collapsed onto a line. Sub-voxel coordinates make an exact zero unreliable, so the
/// threshold is a hundredth of a voxel of area rather than `0.0`.
const AREA_EPSILON_SQUARE_VOXELS: f64 = 1.0e-2;

/// How close to the true deepest point the search must get, in voxels.
///
/// It buys nothing but cost. The point is an IDENTITY: what it has to be is strictly inside the
/// face, deep enough to survive an edit, and identical across two derivations of the same sketch
/// — and the last of those is exact whatever this is set to, because the search is deterministic.
/// Half a voxel is well inside any face big enough to be worth naming, and the search is on the
/// per-voxel resolve path, so buying decimal places here is paid for on every sample. It does NOT
/// cross a layer boundary: nothing downstream is handed this number or asked to match it.
const INTERIOR_POINT_PRECISION_VOXELS: f32 = 0.5;

/// The enclosed signed area by Green's theorem: positive counter-clockwise, negative clockwise.
///
/// Each edge contributes `½∮(x dy − y dx)` over itself ([`ProfileEdge::signed_area_term`]), which
/// is the shoelace term for a straight span and the exact circular integral for an arc. A
/// two-edge cycle (an arc and its chord, say) genuinely encloses area, so there is no
/// minimum-vertex floor here — [`AREA_EPSILON_SQUARE_VOXELS`] is the only filter.
fn signed_area(boundary: &[ProfileEdge]) -> f64 {
    boundary.iter().map(ProfileEdge::signed_area_term).sum()
}
