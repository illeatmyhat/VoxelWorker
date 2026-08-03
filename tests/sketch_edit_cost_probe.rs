//! What a SKETCH node costs on the real edit path, by how its boundary is represented.
//!
//! The field probe in `document` times one `signed_distance` call. This one times what the
//! author actually waits for: `AppCore::rebuild`, which classifies every chunk the node covers.
//! The two answer different questions — a per-sample cost that looks fine can still be
//! intolerable once it is multiplied by the footprint, and only this probe knows the multiplier.
//!
//! Reported per profile radius, because the interesting number is not one measurement but where
//! the spline row crosses the frame budget the analytic row sits comfortably under.
//!
//! Run: `cargo test --release --test sketch_edit_cost_probe -- --ignored --nocapture`

#![allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use std::time::Instant;

use camera::OrbitCamera;
use document::scene::{Node, NodeContent, Scene};
use document::sketch::{PlaneAxis, Sketch, SketchLength, SketchPoint, SketchSolid};
use voxel_core::core_geom::MaterialChoice;
use voxel_worker::AppCore;

const DENSITY: u32 = 16;
const DEPTH_VOXELS: u32 = 32;

const fn context() -> parametric::EvaluationContext {
    parametric::EvaluationContext::new(
        std::num::NonZeroU32::new(DENSITY).expect("probe density is non-zero"),
    )
}

fn circle_profile(radius_voxels: i64) -> Sketch {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_circle(SketchPoint::new(0, 0), SketchLength::new(radius_voxels))
        .expect("a circle of non-zero radius");
    sketch
}

/// A closed fit-point spline around the same circle, so the only thing that changes between
/// rows is how the boundary is REPRESENTED.
fn spline_profile(radius_voxels: i64, count: i64) -> Sketch {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let points: Vec<SketchPoint> = (0..count)
        .map(|index| {
            let angle = index as f64 / count as f64 * std::f64::consts::TAU;
            SketchPoint::from_continuous(
                angle.cos() * radius_voxels as f64,
                angle.sin() * radius_voxels as f64,
            )
        })
        .collect();
    sketch
        .add_fit_point_spline(&points, true)
        .expect("a closed spline through distinct points");
    sketch
}

fn scene_of(sketch: Sketch) -> Scene {
    Scene::from_nodes(vec![Node::new(
        "Profile",
        NodeContent::SketchTool {
            producer: SketchSolid::extrude(sketch, DEPTH_VOXELS),
            material: MaterialChoice::Stone,
        },
    )])
}

fn report(label: &str, sketch: Sketch) {
    let boundary_edges: usize = sketch
        .region_field_loops(context())
        .iter()
        .map(|(_, edges)| edges.len())
        .sum();
    let scene = scene_of(sketch);
    let mut app_core = AppCore::new(OrbitCamera::default());
    let started = Instant::now();
    let _outcome = app_core.rebuild(&scene, DENSITY);
    let cold_ms = started.elapsed().as_secs_f64() * 1000.0;
    println!("{label:<28} {boundary_edges:>6} {cold_ms:>12.1}");
}

#[test]
#[ignore = "perf probe — run in release with --ignored --nocapture"]
fn sketch_rebuild_cost_by_boundary_kind() {
    for radius in [16_i64, 32, 64] {
        println!(
            "\nradius {radius} voxels\n{:<28} {:>6} {:>12}",
            "profile", "edges", "build (ms)"
        );
        println!("{}", "-".repeat(48));
        report("analytic circle", circle_profile(radius));
        for count in [8_i64, 16, 32] {
            report(
                &format!("{count}-point closed spline"),
                spline_profile(radius, count),
            );
        }
    }
}
