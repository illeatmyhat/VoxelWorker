//! What a drag frame costs on a REAL drawing, loaded from an F9 repro dump.
//!
//! The synthetic probe next door builds a population of identical slots and reports a slope. That
//! answers "what grows", but it cannot answer "why is the owner's file slow" — a real drawing is
//! one profile of mixed curves, not thirty-two disjoint copies of one shape, and the two have
//! nothing in common but the word "sketch". This one replays the actual scene.
//!
//! Point it at a dump with `VOXELWORKER_REPRO`, defaulting to the path F9 writes.
//!
//! Run: `cargo test --release --test live_scene_cost_probe -- --ignored --nocapture`

#![allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::suboptimal_flops,
    clippy::too_many_lines
)]

use std::time::Instant;

use camera::OrbitCamera;
use document::scene::{NodeContent, Scene};
use document::sketch::SketchSolid;
use voxel_worker::AppCore;

fn milliseconds(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn repro_path() -> std::path::PathBuf {
    std::env::var_os("VOXELWORKER_REPRO").map_or_else(
        || std::env::temp_dir().join("voxelworker-repro.json"),
        std::path::PathBuf::from,
    )
}

/// Every sketch node in the scene, with the node offset it sits at.
fn sketch_nodes(scene: &Scene) -> Vec<(u64, String, SketchSolid, [i64; 3])> {
    let mut found = Vec::new();
    for (_, id, _) in scene.tree_rows() {
        let Some(node) = scene.node_by_id(id) else {
            continue;
        };
        if let NodeContent::SketchTool { producer, .. } = &node.content {
            found.push((
                id.0,
                node.name.clone(),
                producer.clone(),
                node.transform.offset_voxels,
            ));
        }
    }
    found
}

#[test]
#[ignore = "perf probe — needs an F9 dump; run in release with --ignored --nocapture"]
fn live_scene_drag_frame_cost() {
    let path = repro_path();
    let config = voxel_worker::AppConfig::load_from(&path)
        .unwrap_or_else(|error| panic!("load {}: {error}", path.display()));
    let restored = config.to_panel_state();
    let scene = restored.scene;
    let density = restored.geometry.voxels_per_block;
    let context = parametric::EvaluationContext::new(
        std::num::NonZeroU32::new(density).expect("a non-zero density"),
    );
    println!("\ndump: {}", path.display());
    println!("density: {density} voxels/block");

    for (id, name, producer, offset) in sketch_nodes(&scene) {
        let sketch = &producer.sketch;
        println!(
            "\nnode {id} \"{name}\" at {offset:?}: {} points, {} constraints, {} arcs, \
             {} segments, {} conics, {} splines, {} beziers, {} ellipses, {} circles",
            sketch.points().len(),
            sketch.constraints().len(),
            sketch.arcs().len(),
            sketch.segments().len(),
            sketch.conics().len(),
            sketch.splines().len(),
            sketch.beziers().len(),
            sketch.ellipses().len(),
            sketch.circles().len(),
        );

        // A FRESH clone each time: the region memo is per-value, and every path that matters
        // here — the preview clone, the leaf producer the resolve builds — starts with an empty
        // one. Timing a warm memo would measure nothing the app ever pays.
        let started = Instant::now();
        let faces = producer.sketch.clone().faces(context).len();
        let faces_ms = milliseconds(started);

        let cold = producer.sketch.clone();
        let started = Instant::now();
        let region = cold.region(context);
        let region_ms = milliseconds(started);

        let started = Instant::now();
        let field_loops = producer.sketch.clone().region_field_loops(context);
        let field_ms = milliseconds(started);
        let edges: usize = field_loops.iter().map(|(_, edges)| edges.len()).sum();
        let widest = field_loops
            .iter()
            .map(|(_, edges)| edges.len())
            .max()
            .unwrap_or(0);
        println!(
            "  arrangement {faces_ms:.2}ms ({faces} faces) | region {region_ms:.2}ms \
             ({} loops) | field {field_ms:.2}ms ({edges} edges, widest loop {widest})",
            region.len()
        );

        // What ONE voxel sample of this profile costs. The resolve's whole bill is this number
        // times the samples below, so it decides whether the answer is "make the field cheaper"
        // or "sample fewer voxels".
        let probe_grid = producer.grid_dimensions(context);
        let mut sink = 0.0_f32;
        let samples = 200_000_u32;
        let started = Instant::now();
        for step in 0..samples {
            let x = (step % 449) as f32 * 0.61;
            let y = (step / 449) as f32 * 0.37;
            sink += producer.signed_distance([x, y, 1.5], context);
        }
        let loose_ns = started.elapsed().as_secs_f64() * 1.0e9 / f64::from(samples);
        assert!(sink.is_finite() || sink.is_infinite(), "the field answered");

        // The same field through the PREPARED evaluator the resolve actually uses, which resolves
        // the region once and holds it. The gap between the two is what a per-sample memo lookup
        // costs — a read lock plus a full comparison of the entity store, per voxel.
        let prepared = document::voxel::Field::prepare(&producer, density);
        let mut sink = 0.0_f32;
        let started = Instant::now();
        for step in 0..samples {
            let x = (step % 449) as f32 * 0.61;
            let y = (step / 449) as f32 * 0.37;
            sink += prepared.signed_distance([x, y, 1.5]);
        }
        let prepared_ns = started.elapsed().as_secs_f64() * 1.0e9 / f64::from(samples);
        assert!(sink.is_finite() || sink.is_infinite(), "the field answered");
        // Split the field's two halves. The dense extrude rasterize asks only CONTAINMENT; the
        // classifier's interval bracket asks DISTANCE. They cost very different things — a
        // crossing test walks the boundary once, a distance measures every curve it cannot
        // reject — so one number for "the field" hides which one to fix.
        let bounded = substrate::geom2d::BoundedRegion::new(field_loops.clone());
        let mut kinds = (0_u32, 0_u32, 0_u32);
        for (_, edges) in &field_loops {
            for edge in edges {
                match edge {
                    substrate::geom2d::RegionEdge::Segment { .. } => kinds.0 += 1,
                    substrate::geom2d::RegionEdge::Arc { .. } => kinds.1 += 1,
                    substrate::geom2d::RegionEdge::RationalBezier { .. } => kinds.2 += 1,
                }
            }
        }
        // Sample INSIDE the profile's own box. Points far outside are rejected by a loop's
        // bounding box before any curve is touched, so a sweep that mostly misses the drawing
        // reports the cost of the reject rather than the cost of the field — and the resolve only
        // ever asks about voxels the profile covers.
        let (low, high) = bounded.bounds().unwrap_or(([0.0, 0.0], [1.0, 1.0]));
        let across = 449_u32;
        let step_x = (high[0] - low[0]) / across as f32;
        let step_y = (high[1] - low[1]) / (samples / across).max(1) as f32;
        let at = |step: u32| {
            [
                (step % across) as f32 * step_x + low[0],
                (step / across) as f32 * step_y + low[1],
            ]
        };
        let mut hits = 0_u32;
        let started = Instant::now();
        for step in 0..samples {
            hits += u32::from(bounded.contains(at(step)));
        }
        let contains_ns = started.elapsed().as_secs_f64() * 1.0e9 / f64::from(samples);
        let mut sink = 0.0_f32;
        let started = Instant::now();
        for step in 0..samples {
            sink += bounded.signed_distance(at(step), substrate::geom2d::Metric::Chebyshev);
        }
        let distance_ns = started.elapsed().as_secs_f64() * 1.0e9 / f64::from(samples);
        assert!(sink.is_finite() || sink.is_infinite(), "the field answered");
        println!(
            "  edges: {} segments, {} arcs, {} rational Beziers | contains {contains_ns:.0}ns \
             ({hits} inside) | distance {distance_ns:.0}ns",
            kinds.0, kinds.1, kinds.2
        );

        // Which KIND of edge the distance is spent in. The same sweep against the profile with one
        // kind dropped — not a real region, but the drop tells which curve to make cheaper.
        for (label, keep) in [
            ("segments only", 0_u8),
            ("arcs only", 1),
            ("rational Beziers only", 2),
        ] {
            let only: Vec<_> = field_loops
                .iter()
                .map(|(role, edges)| {
                    let kept: Vec<_> = edges
                        .iter()
                        .filter(|edge| {
                            keep == match edge {
                                substrate::geom2d::RegionEdge::Segment { .. } => 0,
                                substrate::geom2d::RegionEdge::Arc { .. } => 1,
                                substrate::geom2d::RegionEdge::RationalBezier { .. } => 2,
                            }
                        })
                        .copied()
                        .collect();
                    (*role, kept)
                })
                .collect();
            let count: usize = only.iter().map(|(_, edges)| edges.len()).sum();
            if count == 0 {
                continue;
            }
            let partial = substrate::geom2d::BoundedRegion::new(only);
            let mut sink = 0.0_f32;
            let started = Instant::now();
            for step in 0..samples {
                sink += partial.signed_distance(at(step), substrate::geom2d::Metric::Chebyshev);
            }
            let each = started.elapsed().as_secs_f64() * 1.0e9 / f64::from(samples);
            assert!(sink.is_finite() || sink.is_infinite(), "the field answered");
            println!("    {label}: {count} edges, {each:.0}ns");
        }

        let grid_samples =
            f64::from(probe_grid[0]) * f64::from(probe_grid[1]) * f64::from(probe_grid[2]);
        println!(
            "  field {loose_ns:.0}ns per sample unprepared, {prepared_ns:.0}ns prepared \
             -> {:.1}ms / {:.1}ms for the whole grid",
            loose_ns * grid_samples / 1.0e6,
            prepared_ns * grid_samples / 1.0e6
        );

        // The grid the resolve fills — the profile's footprint swept along the normal, and the
        // sample count every per-voxel cost above is multiplied by.
        let grid = producer.grid_dimensions(context);
        println!(
            "  grid {}x{}x{} voxels = {} samples, {:?}",
            grid[0],
            grid[1],
            grid[2],
            u64::from(grid[0]) * u64::from(grid[1]) * u64::from(grid[2]),
            producer.operation
        );
    }

    // The drag frame itself, warm — the shell always has a previous leaf index, so this is the
    // targeted-invalidation arm and not the wholesale clear a cold `AppCore` would take.
    let mut app_core = AppCore::new(OrbitCamera::default());
    let started = Instant::now();
    drop(app_core.rebuild(&scene, density));
    let cold_ms = milliseconds(started);
    let started = Instant::now();
    let outcome = app_core.rebuild(&scene, density);
    let idle_ms = milliseconds(started);
    let chunks = match &outcome {
        voxel_worker::RebuildOutcome::Built(output) => output.two_layer_chunks.len(),
        voxel_worker::RebuildOutcome::DensityRejected { .. } => 0,
    };
    println!(
        "\nwhole scene: first rebuild {cold_ms:.2}ms, unchanged rebuild {idle_ms:.2}ms, \
         {chunks} resident chunks"
    );

    // One frame of an actual drag on this drawing, in the shell's order: clone the pre-drag
    // snapshot, settle the moved point, compensate the anchor, rebuild.
    for (id, name, producer, offset) in sketch_nodes(&scene) {
        let points = producer.sketch.points();
        // Mid-drawing, so the profile's bounding box does not move and the anchor compensation
        // has nothing to absorb — an author's ordinary grab.
        let Some(grabbed) = points.get(points.len() / 2) else {
            continue;
        };
        let was = grabbed.at.in_plane();

        // A RUN of frames, not one. The chunk build is a rayon fan-out whose wall time is the
        // slowest chunk on a machine that is also running everything else, and a single sample of
        // it swings wider than the changes being measured. Each frame moves the point somewhere
        // new so the incremental invalidation has real work every time.
        let mut frames: Vec<(f64, f64, f64, f64)> = Vec::new();
        let mut last = None;
        for step in 0..9_i32 {
            let nudge = f64::from(step % 3 + 1);
            let to = document::sketch::SketchPoint::from_continuous(was[0] + nudge, was[1] + nudge);

            let started = Instant::now();
            let mut preview = producer.clone();
            let clone_ms = milliseconds(started);

            let started = Instant::now();
            let moved = preview.sketch.move_point(grabbed.id, to, context);
            let settle_ms = milliseconds(started);

            // The fresh preview's memo starts EMPTY, so the frame pays a whole arrangement before
            // anything is resolved. This is the cost the region memo cannot help with, because the
            // value it is attached to is new every frame.
            let started = Instant::now();
            let derive_ms_loops = preview.sketch.region(context).len();
            let derive_ms = milliseconds(started);

            let compensated = preview.anchor_preserving_offset(&producer, offset, context);
            let mut edited = scene.clone();
            if let Some(node) = edited.node_by_id(document::scene::NodeId(id)).cloned() {
                if let Some(target) = edited.node_by_id_mut(document::scene::NodeId(id)) {
                    target.content = match node.content {
                        NodeContent::SketchTool { material, .. } => NodeContent::SketchTool {
                            producer: preview.clone(),
                            material,
                        },
                        other => other,
                    };
                    target.transform.offset_voxels = compensated;
                }
            }

            let started = Instant::now();
            let outcome = app_core.rebuild(&edited, density);
            let rebuild_ms = milliseconds(started);
            let dirty = match &outcome {
                voxel_worker::RebuildOutcome::Built(output) => output
                    .incremental_dirty_chunks
                    .as_ref()
                    .map_or_else(|| "all".to_owned(), |dirty| dirty.len().to_string()),
                voxel_worker::RebuildOutcome::DensityRejected { .. } => "rejected".to_owned(),
            };
            frames.push((clone_ms, settle_ms, derive_ms, rebuild_ms));
            last = Some((moved, derive_ms_loops, dirty));
        }

        let Some((moved, loops, dirty)) = last else {
            continue;
        };
        // The MINIMUM of the run, per part. A frame can only be slowed by what else the machine is
        // doing, so the floor is the honest cost and the median carries the noise with it.
        let best = |part: fn(&(f64, f64, f64, f64)) -> f64| {
            frames.iter().map(part).fold(f64::INFINITY, f64::min)
        };
        let (clone_ms, settle_ms, derive_ms, rebuild_ms) =
            (best(|f| f.0), best(|f| f.1), best(|f| f.2), best(|f| f.3));
        let mut totals: Vec<f64> = frames.iter().map(|f| f.0 + f.1 + f.2 + f.3).collect();
        totals.sort_by(f64::total_cmp);
        println!(
            "\ndrag one vertex of \"{name}\" over {} frames (settled: {moved:?}, {loops} loops)\n  \
             best-of clone {clone_ms:.2}ms | settle {settle_ms:.2}ms | region {derive_ms:.2}ms | \
             rebuild {rebuild_ms:.2}ms | {dirty}/{chunks} chunks dirty\n  \
             FRAME TOTAL best {:.2}ms, median {:.2}ms, worst {:.2}ms",
            frames.len(),
            totals.first().copied().unwrap_or_default(),
            totals[totals.len() / 2],
            totals.last().copied().unwrap_or_default(),
        );
    }
}
