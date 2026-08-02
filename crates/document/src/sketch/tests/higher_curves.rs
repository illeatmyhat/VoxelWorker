use super::*;

#[test]
fn ellipse_is_one_closed_profile_without_boundary_points() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let ellipse = sketch
        .add_ellipse(
            SketchPoint::new(2, 3),
            SketchPoint::new(8, 3),
            SketchPoint::new(2, 7),
        )
        .expect("valid ellipse");

    assert_eq!(sketch.ellipses()[0].id, ellipse);
    assert_eq!(sketch.points().len(), 3);
    assert!(sketch
        .points()
        .iter()
        .all(|point| point.role == EntityRole::Construction));
    assert_eq!(sketch.faces(ctx(16)).len(), 1);

    let restored: Sketch =
        serde_json::from_str(&serde_json::to_string(&sketch).expect("ellipse serializes"))
            .expect("ellipse deserializes");
    assert_eq!(restored, sketch);
}

#[test]
fn conic_and_its_chord_bound_a_profile_and_preserve_exact_rho() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let conic = sketch
        .add_conic(
            SketchPoint::new(-4, 0),
            SketchPoint::new(4, 0),
            SketchPoint::new(0, 4),
            0.5,
        )
        .expect("valid conic");
    let held = sketch.conics()[0];
    sketch.connect(held.to, held.from).expect("closing chord");

    assert_eq!(held.id, conic);
    assert_eq!(held.rho.rational(), ExactRational::new(1, 2).unwrap());
    assert_eq!(sketch.faces(ctx(16)).len(), 1);
}

#[test]
fn higher_curve_handles_retarget_with_density_but_rho_does_not() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_conic(
            SketchPoint::new(0, 0),
            SketchPoint::new(4, 0),
            SketchPoint::new(2, 2),
            0.75,
        )
        .expect("valid conic");
    let rho = sketch.conics()[0].rho;

    sketch.retarget_density(16, 32);

    assert_eq!(sketch.conics()[0].rho, rho);
    let positions: Vec<_> = sketch
        .points()
        .iter()
        .map(|point| point.at.in_plane())
        .collect();
    assert_eq!(positions, vec![[0.0, 0.0], [8.0, 0.0], [4.0, 4.0]]);
}

#[test]
fn closed_fit_spline_is_one_profile_with_only_authored_fit_points() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let spline = sketch
        .add_fit_point_spline(
            &[
                SketchPoint::new(-4, -2),
                SketchPoint::new(4, -2),
                SketchPoint::new(0, 5),
            ],
            true,
        )
        .expect("three distinct points make a closed fit spline");

    assert_eq!(sketch.splines()[0].id, spline);
    assert_eq!(sketch.splines()[0].kind, SplineKind::FitPoint);
    assert!(sketch.splines()[0].closed);
    assert_eq!(sketch.points().len(), 3);
    assert!(sketch
        .points()
        .iter()
        .all(|point| point.role == EntityRole::Real));
    assert_eq!(sketch.faces(ctx(16)).len(), 1);

    let restored: Sketch =
        serde_json::from_str(&serde_json::to_string(&sketch).expect("fit spline serializes"))
            .expect("fit spline deserializes");
    assert_eq!(restored, sketch);
}

#[test]
fn control_point_spline_keeps_only_its_endpoints_visible() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_control_point_spline(&[
            SketchPoint::new(0, 0),
            SketchPoint::new(2, 5),
            SketchPoint::new(6, 5),
            SketchPoint::new(8, 0),
        ])
        .expect("four controls make one cubic span");

    let spline = &sketch.splines()[0];
    assert_eq!(spline.kind, SplineKind::ControlPoint);
    assert!(!spline.closed);
    let roles: Vec<_> = spline
        .points
        .iter()
        .map(|id| {
            sketch
                .points()
                .iter()
                .find(|point| point.id == *id)
                .expect("control exists")
                .role
        })
        .collect();
    assert_eq!(
        roles,
        vec![
            EntityRole::Real,
            EntityRole::Construction,
            EntityRole::Construction,
            EntityRole::Real,
        ]
    );
}

#[test]
fn spline_points_retarget_and_invalid_loaded_splines_repair_atomically() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_fit_point_spline(
            &[
                SketchPoint::new(1, 2),
                SketchPoint::new(4, 6),
                SketchPoint::new(8, 2),
            ],
            false,
        )
        .expect("valid open fit spline");
    sketch.retarget_density(16, 32);
    let positions: Vec<_> = sketch
        .points()
        .iter()
        .map(|point| point.at.in_plane())
        .collect();
    assert_eq!(positions, vec![[2.0, 4.0], [8.0, 12.0], [16.0, 4.0]]);

    let mut raw = serde_json::to_value(&sketch).expect("spline serializes");
    raw["splines"][0]["points"][1] = serde_json::json!(EntityId::MAX);
    let mut loaded: Sketch = serde_json::from_value(raw).expect("structural load succeeds");
    assert_eq!(loaded.repair(ctx(32)), 1);
    assert!(loaded.splines().is_empty());
}
