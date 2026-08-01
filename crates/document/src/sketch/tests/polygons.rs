use super::*;

fn source() -> SketchSolid {
    SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 4)
}

#[test]
fn polygon_placements_and_commits_share_canonical_vertices() {
    let source = source();
    let center = SketchPoint::new(0, 0);
    let radius = SketchPoint::new(8, 0);
    for sides in [3, 5, 8, 128] {
        let placement = source
            .inscribed_polygon_placement(center, radius, sides)
            .unwrap();
        let made = source
            .with_inscribed_polygon(center, radius, sides)
            .unwrap();
        assert_eq!(placement.vertices.len(), usize::from(sides));
        assert_eq!(made.sketch.segments().len(), usize::from(sides));
        for vertex in placement.vertices {
            assert!(made.sketch.point_at(vertex).is_some());
        }
    }
}

#[test]
fn circumscribed_and_edge_polygons_preserve_their_authored_loci() {
    let source = source();
    let circumscribed = source
        .circumscribed_polygon_placement(SketchPoint::new(0, 0), SketchPoint::new(4, 0), 4)
        .unwrap();
    let [first, second, ..] = circumscribed.vertices.as_slice() else {
        panic!("square has four vertices")
    };
    assert_eq!(first.in_plane()[0].midpoint(second.in_plane()[0]), 4.0);
    assert_eq!(first.in_plane()[1].midpoint(second.in_plane()[1]), 0.0);

    let edge = source
        .edge_polygon_placement(
            SketchPoint::new(0, 0),
            SketchPoint::new(6, 0),
            SketchPoint::new(3, 4),
            6,
        )
        .unwrap();
    assert_eq!(edge.vertices[0], SketchPoint::new(0, 0));
    assert_eq!(edge.vertices[1], SketchPoint::new(6, 0));
    assert!(edge.center.in_plane()[1] > 0.0);
}

#[test]
fn invalid_and_duplicate_polygons_are_refused_without_mutating_the_source() {
    let source = source();
    assert!(source
        .with_inscribed_polygon(SketchPoint::new(0, 0), SketchPoint::new(3, 0), 2)
        .is_err());
    assert!(source
        .with_inscribed_polygon(SketchPoint::new(0, 0), SketchPoint::new(3, 0), 129)
        .is_err());
    assert!(source.sketch.points().is_empty());

    let made = source
        .with_edge_polygon(
            SketchPoint::new(0, 0),
            SketchPoint::new(4, 0),
            SketchPoint::new(2, 3),
            4,
        )
        .unwrap();
    assert!(made
        .with_edge_polygon(
            SketchPoint::new(0, 0),
            SketchPoint::new(4, 0),
            SketchPoint::new(2, 3),
            4,
        )
        .is_err());
}
