//! Multi-region derivation and face pick/unpick.

use super::ctx;
use super::*;
use crate::sketch::FaceKey;

/// A `size × size` square starting at `(origin, origin)`, added to `sketch` as four points and
/// four segments. Returns the corner point ids counter-clockwise from the low corner.
fn square(sketch: &mut Sketch, origin: i64, size: i64) -> [EntityId; 4] {
    let corners = [
        SketchPoint::new(origin, origin),
        SketchPoint::new(origin + size, origin),
        SketchPoint::new(origin + size, origin + size),
        SketchPoint::new(origin, origin + size),
    ];
    let ids = corners.map(|at| sketch.add_point(at));
    for index in 0..4 {
        sketch.connect(ids[index], ids[(index + 1) % 4]);
    }
    ids
}

/// A 12×12 square with a 4×4 square inside it — two faces, the inner one nested.
fn nested_squares() -> Sketch {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    square(&mut sketch, 0, 12);
    square(&mut sketch, 4, 4);
    sketch
}

/// The face of `sketch` whose area is smallest — the inner one in every fixture here.
fn innermost(sketch: &Sketch) -> FaceKey {
    let mut faces = sketch.identified_faces(ctx(16));
    faces.sort_by(|a, b| a.0.area_voxels.total_cmp(&b.0.area_voxels));
    faces.first().expect("a face").1
}

/// Derivation enumerates every bounded face, and only bounded ones: a component's unbounded
/// face is traced clockwise and dropped, so two nested squares are two faces, not three.
#[test]
fn derivation_enumerates_the_bounded_faces_and_nothing_else() {
    let sketch = nested_squares();
    let faces = sketch.faces(ctx(16));
    assert_eq!(faces.len(), 2, "one face per square, no unbounded face");
    let mut areas: Vec<f64> = faces.iter().map(|face| face.area_voxels).collect();
    areas.sort_by(f64::total_cmp);
    assert_eq!(areas, vec![16.0, 144.0]);
    assert!(
        faces.iter().all(|face| face.boundary.len() == 4),
        "each face's boundary is its own four corners"
    );
}

/// A face split in two by a chord is TWO faces sharing that chord — the DCEL walk turns at the
/// shared vertices rather than running around the outside.
#[test]
fn a_chord_splits_one_face_into_two() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let corners = square(&mut sketch, 0, 4);
    assert_eq!(sketch.faces(ctx(16)).len(), 1);
    sketch.connect(corners[0], corners[2]);
    let faces = sketch.faces(ctx(16));
    assert_eq!(faces.len(), 2, "the diagonal cuts the square in half");
    for face in &faces {
        assert!((face.area_voxels - 8.0).abs() < 1e-9, "each half is half");
    }
}

/// Faces default to picked, and unpicking one carves it — the donut, then the tube.
#[test]
fn unpicking_the_inner_face_carves_a_hole_through_the_extrude() {
    let mut sketch = nested_squares();
    assert!(
        sketch
            .identified_faces(ctx(16))
            .iter()
            .all(|f| sketch.face_is_picked(&f.1, ctx(16))),
        "every derived face starts picked"
    );
    let solid = SketchSolid::extrude(sketch.clone(), 2);
    let full = occupancy_set(&solid, 8).len();
    assert_eq!(full, 12 * 12 * 2, "picked-everything is the whole square");

    sketch.set_face_picked(innermost(&sketch), false, ctx(16));
    let holed = SketchSolid::extrude(sketch, 2);
    assert_eq!(
        occupancy_set(&holed, 8).len(),
        (12 * 12 - 4 * 4) * 2,
        "the pocket is gone from every layer — a tube"
    );
    assert_eq!(
        holed.grid_dimensions(ctx(16)),
        [12, 12, 2],
        "a hole changes no extent"
    );
}

/// A hole survives into the revolve too — the hollow vase.
#[test]
fn a_revolve_lifts_the_hole_as_well() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    square(&mut sketch, 2, 10);
    square(&mut sketch, 5, 4);
    let solid_wall = SketchSolid::revolve(sketch.clone(), RevolveAxis::InPlane0, 360);
    sketch.set_face_picked(innermost(&sketch), false, ctx(16));
    let hollow = SketchSolid::revolve(sketch, RevolveAxis::InPlane0, 360);
    let full = occupancy_set(&solid_wall, 8).len();
    let carved = occupancy_set(&hollow, 8).len();
    assert!(full > 0 && carved > 0, "both lathe to something");
    assert!(
        carved < full,
        "the unpicked pocket lathes an annular void: {carved} < {full}"
    );
}

/// The point of the interior-point key: an unpick survives the edits that leave the same ground
/// under the point. Dragging a vertex and splitting a boundary edge both do — neither moves the
/// pocket out from under its own deepest point.
#[test]
fn an_unpick_survives_a_vertex_drag_and_an_edge_split() {
    let mut sketch = nested_squares();
    sketch.set_face_picked(innermost(&sketch), false, ctx(16));

    let moved = sketch.points().last().expect("a point").id;
    assert!(
        sketch
            .move_point(moved, SketchPoint::new(5, 5), ctx(16))
            .expect("evaluation context"),
        "the point"
    );
    assert!(
        !sketch.face_is_picked(&innermost(&sketch), ctx(16)),
        "a drag leaves the pocket under its own point"
    );

    let inner_corners = inner_corner_ids(&sketch);
    let inner_edge = sketch
        .segments()
        .iter()
        .find(|segment| {
            inner_corners.contains(&segment.from) && inner_corners.contains(&segment.to)
        })
        .expect("a boundary edge of the unpicked face")
        .id;
    sketch.split_segment(inner_edge, SketchPoint::new(6, 4));
    assert!(
        !sketch.face_is_picked(&innermost(&sketch), ctx(16)),
        "the hole is still a hole"
    );
}

/// The corner point ids of the inner square in [`nested_squares`], however it has been nudged.
fn inner_corner_ids(sketch: &Sketch) -> Vec<EntityId> {
    sketch
        .points()
        .iter()
        .filter(|point| point.at.offset_voxels.iter().all(|&c| (4..=8).contains(&c)))
        .map(|point| point.id)
        .collect()
}

/// The other half of the contract, and the failure mode the interior-point key ACCEPTS: cutting
/// the pocket in two does not reset both halves to picked, it migrates the unpick into whichever
/// half still holds the stored point. Exactly one hole survives: the half the point is in.
#[test]
fn cutting_an_unpicked_face_in_two_migrates_the_unpick() {
    let mut sketch = nested_squares();
    let carved = innermost(&sketch);
    sketch.set_face_picked(carved, false, ctx(16));

    // A chord across the pocket, well off its center so the stored point is unambiguously on one
    // side. Its ends are free points on the boundary — the arrangement cuts at the crossings.
    let low = sketch.add_free_point(SketchPoint::new(4, 5));
    let high = sketch.add_free_point(SketchPoint::new(8, 5));
    sketch.connect(low, high).expect("the chord");

    let mut faces = sketch.identified_faces(ctx(16));
    faces.sort_by(|a, b| a.0.area_voxels.total_cmp(&b.0.area_voxels));
    let holes: Vec<&(Face, FaceKey)> = faces
        .iter()
        .filter(|(_, key)| !sketch.face_is_picked(key, ctx(16)))
        .collect();
    assert_eq!(holes.len(), 1, "the unpick names ONE face, not both halves");
    assert!(
        holes[0].0.contains(carved.interior_point),
        "and it is the half the stored point landed in"
    );
    assert!(
        holes[0].0.area_voxels < 16.0,
        "which is smaller than the pocket it was cut from: {}",
        holes[0].0.area_voxels
    );
}

/// A crossing needs NO shared point: the bowtie's two segments cross in mid-air, and the
/// arrangement cuts both there, so the two triangles are two regions without the author snapping a
/// vertex at the crossing first.
#[test]
fn a_crossing_bounds_faces_with_no_snapped_point() {
    let bowtie = Sketch::new(
        PlaneAxis::Z,
        vec![
            SketchPoint::new(0, 0),
            SketchPoint::new(6, 6),
            SketchPoint::new(0, 6),
            SketchPoint::new(6, 0),
        ],
    );
    let faces = bowtie.faces(ctx(16));
    assert_eq!(faces.len(), 2, "one triangle either side of the crossing");
    for face in &faces {
        assert!(
            (face.area_voxels - 9.0).abs() < 1e-6,
            "each is half the 6x6 square's diagonal split: {}",
            face.area_voxels
        );
    }
    assert_eq!(
        bowtie.points().len(),
        4,
        "and no vertex was minted at the crossing"
    );
}

/// The unpicked set is document state: it round-trips, and a document written before face picking
/// loads with every face picked, without a migration.
#[test]
fn the_pick_state_round_trips_and_an_older_document_loads_picked() {
    let mut sketch = nested_squares();
    sketch.set_face_picked(innermost(&sketch), false, ctx(16));
    let json = serde_json::to_string(&sketch).expect("serialize");
    let restored: Sketch = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored, sketch);
    assert!(!restored.face_is_picked(&innermost(&restored), ctx(16)));

    let mut value: serde_json::Value = serde_json::from_str(&json).expect("json");
    value
        .as_object_mut()
        .expect("object")
        .remove("unpicked_points")
        .expect("the key was written");
    let older: Sketch = serde_json::from_value(value).expect("a document with no pick state");
    assert!(
        older
            .identified_faces(ctx(16))
            .iter()
            .all(|f| older.face_is_picked(&f.1, ctx(16))),
        "no unpicked list means everything is picked"
    );
}

/// Coarse-solid stays CONSERVATIVE across a hole: it may decline a cell that is in fact solid,
/// but it must never claim one the pocket touches (over-claiming fills a cell without ever
/// sampling it, which would silently plug the hole).
#[test]
fn the_coarse_claim_never_over_claims_across_a_hole() {
    let mut sketch = nested_squares();
    sketch.set_face_picked(innermost(&sketch), false, ctx(16));
    let solid = SketchSolid::extrude(sketch, 2);
    let occupied = occupancy_set(&solid, 8);
    for cell_0 in 0..12u32 {
        for cell_1 in 0..12u32 {
            let cell = voxel_core::spatial_index::VoxelAabb {
                min: [cell_0 as i64, cell_1 as i64, 0],
                max: [cell_0 as i64 + 1, cell_1 as i64 + 1, 2],
            };
            if !solid.extrude_cell_is_solid(cell, ctx(16)) {
                continue;
            }
            for layer in 0..2u32 {
                let claimed = (
                    [
                        (cell_0 * 2 + 1) as i32,
                        (cell_1 * 2 + 1) as i32,
                        (layer * 2 + 1) as i32,
                    ],
                    [(cell_0 % 8) as u8, (cell_1 % 8) as u8, (layer % 8) as u8],
                    0u16,
                );
                assert!(
                    occupied.contains(&claimed),
                    "cell ({cell_0}, {cell_1}) claimed solid but the resolve left it air"
                );
            }
        }
    }
}

/// The wash asks the SAME field the resolve does ([`Sketch::region_field_loops`]), so what an
/// overlay covers is decided by `point_in_region` and nesting is never the overlay's problem. Two
/// nested PICKED faces claim the outer square once — the inner adds nothing the outer does not
/// already claim, where two triangulated fills would composite their alpha twice.
#[test]
fn nested_picked_faces_claim_the_region_once() {
    let sketch = nested_squares();
    assert_eq!(sketch.faces(ctx(16)).len(), 2, "still two faces");
    let field = sketch.region_field_loops(ctx(16));
    assert!(substrate::geom2d::point_in_region(&field, [6.0, 6.0]));
    assert!(substrate::geom2d::point_in_region(&field, [2.0, 2.0]));
    assert!(!substrate::geom2d::point_in_region(&field, [13.0, 6.0]));
    // Inside, so negative, and no deeper for being doubly enclosed: the union is a `min` over the
    // Fill loops, so the field is the distance to the NEAREST boundary either way.
    let inside = substrate::geom2d::signed_distance_to_region(
        &field,
        [6.0, 6.0],
        substrate::geom2d::Metric::Euclidean,
    );
    assert!(inside < 0.0, "inside the material: {inside}");
}

/// Unpick the inner face and the same two faces read as a donut — the void carries no wash, because
/// a `Hole` loop vetoes the point outright.
#[test]
fn an_unpicked_inner_face_reads_as_a_void() {
    let mut sketch = nested_squares();
    sketch.set_face_picked(innermost(&sketch), false, ctx(16));
    let field = sketch.region_field_loops(ctx(16));
    assert!(
        substrate::geom2d::point_in_region(&field, [2.0, 2.0]),
        "the ring"
    );
    assert!(
        !substrate::geom2d::point_in_region(&field, [6.0, 6.0]),
        "the pocket"
    );
}

/// Carving a region does NOT carve what is nested inside it: a picked face inside an unpicked one
/// stands as an island. Each face's pick state governs its OWN area, so "carve this region" carves
/// that region and nothing else — the region fold is ordered innermost-first, not a global veto.
#[test]
fn a_picked_island_inside_a_void_survives_the_carve() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    square(&mut sketch, 0, 20);
    square(&mut sketch, 4, 12);
    square(&mut sketch, 8, 4);
    let middle = {
        let mut faces = sketch.identified_faces(ctx(16));
        faces.sort_by(|a, b| a.0.area_voxels.total_cmp(&b.0.area_voxels));
        faces[1].1
    };
    sketch.set_face_picked(middle, false, ctx(16));
    let field = sketch.region_field_loops(ctx(16));
    assert!(
        substrate::geom2d::point_in_region(&field, [2.0, 2.0]),
        "the outermost ring is material"
    );
    assert!(
        !substrate::geom2d::point_in_region(&field, [6.0, 6.0]),
        "the carved middle is not"
    );
    assert!(
        substrate::geom2d::point_in_region(&field, [10.0, 10.0]),
        "the island inside the carve still is"
    );
    let occupied = occupancy_set(&SketchSolid::extrude(sketch, 1), 8).len();
    assert_eq!(
        occupied,
        20 * 20 - 12 * 12 + 4 * 4,
        "the resolve agrees: the ring and the island, not the pocket between them"
    );
}

/// Faces that merely share an edge are both material in full — the chord case, where neither half
/// contains the other and the shared edge is interior to the region.
#[test]
fn faces_sharing_an_edge_are_both_material() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let corners = square(&mut sketch, 0, 4);
    sketch.connect(corners[0], corners[2]);
    let field = sketch.region_field_loops(ctx(16));
    assert_eq!(field.len(), 2);
    assert!(substrate::geom2d::point_in_region(&field, [1.0, 2.0]));
    assert!(substrate::geom2d::point_in_region(&field, [3.0, 2.0]));
}
