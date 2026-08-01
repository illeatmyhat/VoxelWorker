use super::ctx;
use super::*;

/// The L profile: 6 vertices, every one a true corner.
fn l_profile() -> Vec<SketchPoint> {
    vec![
        SketchPoint::new(0, 0),
        SketchPoint::new(4, 0),
        SketchPoint::new(4, 2),
        SketchPoint::new(2, 2),
        SketchPoint::new(2, 4),
        SketchPoint::new(0, 4),
    ]
}

/// An L extrude catalogs 2 cap outlines + one lateral edge per corner: an L has
/// 6 corners, so 8 polylines. Laterals span the full height at the corner's
/// bbox-anchored in-plane position.
#[test]
fn l_extrude_catalogs_caps_and_six_laterals() {
    let solid = SketchSolid::extrude(Sketch::new(PlaneAxis::Z, l_profile()), 3);
    let polylines = solid.profile_edge_polylines_local(96, ctx(16));
    assert_eq!(polylines.len(), 8);
    let caps: Vec<_> = polylines.iter().filter(|p| p.len() == 7).collect();
    let laterals: Vec<_> = polylines.iter().filter(|p| p.len() == 2).collect();
    assert_eq!(caps.len(), 2, "closed 6-vertex outline at each cap");
    assert_eq!(laterals.len(), 6);
    for cap in &caps {
        assert_eq!(cap.first(), cap.last(), "cap outline closes");
        let z = cap[0][2];
        assert!(z == 0.0 || z == 3.0);
        assert!(cap.iter().all(|point| point[2] == z));
    }
    for lateral in &laterals {
        assert_eq!(lateral[0][2], 0.0);
        assert_eq!(lateral[1][2], 3.0);
        assert_eq!(lateral[0][..2], lateral[1][..2]);
    }
}

/// A `split_segment` midpoint is collinear-same-direction — tangent — so it adds a
/// cap vertex but NO lateral edge: a split rectangle still has exactly 4 laterals.
#[test]
fn split_segment_midpoint_is_tangent_and_adds_no_lateral() {
    let mut sketch = Sketch::rectangle(PlaneAxis::Z, 4, 4);
    let seg_id = sketch.segments[0].id;
    let from = sketch.segments[0].from;
    let to = sketch.segments[0].to;
    let midpoint = {
        let at = |id| {
            sketch
                .points
                .iter()
                .find(|point| point.id == id)
                .unwrap()
                .at
                .offset_voxels
        };
        let (a, b) = (at(from), at(to));
        SketchPoint::new((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    };
    sketch.split_segment(seg_id, midpoint);
    let solid = SketchSolid::extrude(sketch, 2);
    let polylines = solid.profile_edge_polylines_local(96, ctx(16));
    let laterals = polylines.iter().filter(|p| p.len() == 2).count();
    let caps = polylines.iter().filter(|p| p.len() == 6).count();
    assert_eq!(laterals, 4, "the split vertex creases nothing");
    assert_eq!(caps, 2, "caps carry the split vertex (5 + closing point)");
}

/// A full-turn revolve of an off-axis rectangle: the two on-axis vertices are poles
/// (nothing), the two off-axis vertices each catalog a closed latitude circle at
/// their axial height, centered on the radial grid center; no meridian outlines.
#[test]
fn full_revolve_catalogs_latitude_circles_only() {
    // Plane Z, axis InPlane0: axial = coord 0 (world X), radial = coord 1;
    // radial world axes = {Y, Z} → dims[1] = dims[2] = 2 * radial_max = 6.
    let solid = SketchSolid::revolve(
        Sketch::rectangle(PlaneAxis::Z, 4, 3),
        RevolveAxis::InPlane0,
        360,
    );
    let polylines = solid.profile_edge_polylines_local(96, ctx(16));
    assert_eq!(polylines.len(), 2, "two off-axis corners, two circles");
    for circle in &polylines {
        assert_eq!(circle.len(), 97, "96 segments, closed");
        assert_eq!(circle.first(), circle.last());
        let axial = circle[0][0];
        assert!(axial == 0.0 || axial == 4.0, "circles sit at the caps");
        assert!(circle.iter().all(|point| point[0] == axial));
        for point in circle.iter() {
            let radius = ((point[1] - 3.0).powi(2) + (point[2] - 3.0).powi(2)).sqrt();
            assert!(
                (radius - 3.0).abs() < 1e-4,
                "latitude radius is the vertex's |radial|, got {radius}"
            );
        }
    }
}

/// A quarter-turn revolve keeps the latitude arcs' angular density (24 steps of the
/// full 96) and adds the profile outline at both sweep ends.
#[test]
fn partial_revolve_catalogs_arcs_and_end_meridians() {
    let solid = SketchSolid::revolve(
        Sketch::rectangle(PlaneAxis::Z, 4, 3),
        RevolveAxis::InPlane0,
        90,
    );
    let polylines = solid.profile_edge_polylines_local(96, ctx(16));
    let arcs: Vec<_> = polylines.iter().filter(|p| p.len() == 25).collect();
    let meridians: Vec<_> = polylines.iter().filter(|p| p.len() == 5).collect();
    assert_eq!(arcs.len(), 2, "quarter arcs at 24 + 1 points");
    assert_eq!(meridians.len(), 2, "profile outline at each sweep end");
    assert_eq!(polylines.len(), 4);
    for arc in &arcs {
        let start = arc.first().unwrap();
        let end = arc.last().unwrap();
        // Angle 0 lies along +radial_a (world Y), the quarter end along +radial_b (Z).
        assert!((start[1] - 6.0).abs() < 1e-4 && (start[2] - 3.0).abs() < 1e-4);
        assert!((end[1] - 3.0).abs() < 1e-4 && (end[2] - 6.0).abs() < 1e-4);
    }
    for meridian in &meridians {
        assert_eq!(meridian.first(), meridian.last(), "outline closes");
    }
}

/// A straddling profile folds at the axis: each sweep-end outline gains one
/// interpolated axis-crossing point per sign-changing edge.
#[test]
fn straddling_partial_revolve_outline_folds_at_the_axis() {
    let profile = vec![
        SketchPoint::new(0, -2),
        SketchPoint::new(4, -2),
        SketchPoint::new(4, 2),
        SketchPoint::new(0, 2),
    ];
    let solid = SketchSolid::revolve(
        Sketch::new(PlaneAxis::Z, profile),
        RevolveAxis::InPlane0,
        90,
    );
    let polylines = solid.profile_edge_polylines_local(96, ctx(16));
    // All four vertices sit at |radial| = 2: four arcs.
    let arcs = polylines.iter().filter(|p| p.len() == 25).count();
    assert_eq!(arcs, 4);
    // Two sign-changing edges → 4 vertices + 2 crossings + closing point = 7.
    let meridians: Vec<_> = polylines.iter().filter(|p| p.len() == 7).collect();
    assert_eq!(meridians.len(), 2);
    for meridian in &meridians {
        let on_axis = meridian
            .iter()
            .filter(|point| (point[1] - 2.0).abs() < 1e-4 && (point[2] - 2.0).abs() < 1e-4)
            .count();
        assert_eq!(on_axis, 2, "the outline touches the axis at both crossings");
    }
}

/// Degenerate producers catalog nothing: a zero-height extrude and a zero-turn
/// revolve have no body, hence no edges.
#[test]
fn degenerate_solids_catalog_nothing() {
    let flat = SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, 4, 4), 0);
    assert!(flat.profile_edge_polylines_local(96, ctx(16)).is_empty());
    let unswept = SketchSolid::revolve(
        Sketch::rectangle(PlaneAxis::Z, 4, 3),
        RevolveAxis::InPlane0,
        0,
    );
    assert!(unswept.profile_edge_polylines_local(96, ctx(16)).is_empty());
}

/// An arc reaches the boundary as a run of tessellation samples. Those are steps around a
/// smooth curve, so they crease nothing: the rounded-bottom profile catalogs laterals at
/// its four AUTHORED corners and not one per chord, however finely the arc is tessellated.
#[test]
fn an_arc_creases_only_at_its_authored_ends() {
    let mut sketch = Sketch::new(
        PlaneAxis::Z,
        vec![
            SketchPoint::new(0, 0),
            SketchPoint::new(0, 3),
            SketchPoint::new(4, 3),
            SketchPoint::new(4, 0),
        ],
    );
    let bottom = sketch
        .segments()
        .iter()
        .find(|seg| {
            let ends = [seg.from, seg.to];
            ends.contains(&sketch.points()[3].id) && ends.contains(&sketch.points()[0].id)
        })
        .expect("the bottom edge")
        .id;
    sketch.delete_segment(bottom);
    sketch
        .connect_arc(
            sketch.points()[3].id,
            sketch.points()[0].id,
            ::parametric::units::AngleMeasurement::from_degrees(180),
        )
        .expect("a half turn under the box");

    let solid = SketchSolid::extrude(sketch, 3);
    let polylines = solid.profile_edge_polylines_local(96, ctx(16));
    let laterals = polylines.iter().filter(|p| p.len() == 2).count();
    assert_eq!(laterals, 4, "one per authored corner, none per chord");
    // The caps still trace the FULL tessellated outline — the curve is drawn, just not creased.
    let caps: Vec<_> = polylines.iter().filter(|p| p.len() > 2).collect();
    assert_eq!(caps.len(), 2);
    assert!(
        caps[0].len() > 8,
        "the cap outline follows the arc, not its chord"
    );
}
