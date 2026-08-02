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
