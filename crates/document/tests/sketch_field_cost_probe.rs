//! What one voxel sample of a sketch profile costs, by the kind of curve on its boundary.
//!
//! Resolving a `SketchSolid` asks `signed_distance` once per voxel in the profile's footprint,
//! and every ask folds the WHOLE boundary — a distance and a crossing count per edge. An analytic
//! edge answers in closed form; a rational curve is walked in uniform steps. So the number that
//! decides whether a profile is usable is not how many entities it has but how expensive ONE of
//! its edges is to measure, multiplied by how many survive the bounding-box prune.
//!
//! Reported per boundary kind and per spline point count, because the interesting question is
//! whether the cost grows with the author's point count or stays flat.
//!
//! Run: `cargo test --release -p document --test sketch_field_cost_probe -- --ignored --nocapture`

#![allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::missing_const_for_fn
)]

use document::sketch::{PlaneAxis, Sketch, SketchLength, SketchPoint, SketchSolid};

/// The extrude depth and the half-width of the sampled square, in voxels.
const DEPTH_VOXELS: i64 = 8;
const HALF_EXTENT_VOXELS: i64 = 24;
const PROFILE_RADIUS_VOXELS: f64 = 20.0;

fn context() -> parametric::EvaluationContext {
    parametric::EvaluationContext::new(
        std::num::NonZeroU32::new(16).expect("probe density is non-zero"),
    )
}

/// A closed fit-point spline around the same circle every other fixture uses, so the only thing
/// that changes between rows is how the boundary is REPRESENTED.
fn spline_profile(count: i64) -> Sketch {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let points: Vec<SketchPoint> = (0..count)
        .map(|index| {
            let angle = index as f64 / count as f64 * std::f64::consts::TAU;
            SketchPoint::from_continuous(
                angle.cos() * PROFILE_RADIUS_VOXELS,
                angle.sin() * PROFILE_RADIUS_VOXELS,
            )
        })
        .collect();
    sketch
        .add_fit_point_spline(&points, true)
        .expect("a closed spline through distinct points");
    sketch
}

fn circle_profile() -> Sketch {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_circle(
            SketchPoint::new(0, 0),
            SketchLength::new(PROFILE_RADIUS_VOXELS as i64),
        )
        .expect("a circle of non-zero radius");
    sketch
}

fn report(name: &str, sketch: Sketch) {
    let boundary_edges: usize = sketch
        .region_field_loops(context())
        .iter()
        .map(|(_, edges)| edges.len())
        .sum();
    let solid = SketchSolid::extrude(sketch, DEPTH_VOXELS.try_into().unwrap_or(1));
    // Warm the arrangement so this times the FOLD and not the derivation behind it.
    drop(solid.sketch.region_field_loops(context()));

    let mut samples = 0_u32;
    let start = std::time::Instant::now();
    for depth in 0..DEPTH_VOXELS {
        for axis1 in -HALF_EXTENT_VOXELS..HALF_EXTENT_VOXELS {
            for axis0 in -HALF_EXTENT_VOXELS..HALF_EXTENT_VOXELS {
                std::hint::black_box(
                    solid.signed_distance([axis0 as f32, axis1 as f32, depth as f32], context()),
                );
                samples += 1;
            }
        }
    }
    let elapsed = start.elapsed();
    println!(
        "{name}: {boundary_edges} boundary edges, {samples} samples in {elapsed:?} \
         ({:?}/sample)",
        elapsed / samples.max(1)
    );
}

#[test]
#[ignore = "perf probe — run in release with --ignored --nocapture"]
fn signed_distance_cost_by_boundary_kind() {
    report("analytic circle", circle_profile());
    for count in [8, 16, 24, 32] {
        report(
            &format!("{count}-point closed spline"),
            spline_profile(count),
        );
    }
}
