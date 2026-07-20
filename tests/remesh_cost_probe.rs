//! **The other half of what one edit costs.** `tests/edit_cost_probe.rs` times
//! [`AppCore::rebuild`] — the headless resolve — and its third probe closes by refuting its
//! own hypothesis: an origin-shifting drag is not more expensive to RESOLVE, because forcing
//! `incremental_dirty_chunks` to `None` is a branch, not work. The expense a `None` buys is
//! entirely DOWNSTREAM, at the shell, and `rebuild` returns before any of it happens. Every
//! number in that file is therefore a lower bound. This file measures the surplus.
//!
//! The contrast, on one scene, from one rebuild's output:
//!
//! * **incremental** — `rebuild` returned `Some(dirty)`, so the shell calls
//!   [`CuboidMeshRenderer::incremental_rebuild_from_two_layer_chunks`]: re-mesh + re-upload
//!   only `dirty ∪ 26-neighbourhood(dirty) ∩ resident`, keep every other chunk's buffers.
//! * **wholesale** — `rebuild` returned `None`, so the shell calls
//!   [`CuboidMeshRenderer::new_from_two_layer_chunks`]: re-mesh + re-upload every resident
//!   chunk, and rebuild the renderer whole.
//!
//! Scene size is exactly what should make these diverge: the incremental's work is bounded by
//! the edit's footprint, the wholesale's by the covering set.
//!
//! **What this measures: CPU meshing AND GPU buffer creation together, not separately.** The
//! pure-CPU mesher ([`display::mesh::two_layer::build_two_layer_chunk_meshes_filtered`]) is
//! `pub(crate)` to the `display` crate, so an integration test cannot reach it and no public
//! CPU-only two-layer seam exists. Rather than change library visibility to get a prettier
//! number, this probe times the two PUBLIC entry points the shell actually calls, each of
//! which meshes and then uploads. Both are dominated by CPU work either way — the uploads go
//! through `create_buffer_init`, whose cost is an allocation plus a host memcpy on the calling
//! thread — but the split is not attributed here, and this doc comment is the only honest
//! place to say so.
//!
//! **Three further things the numbers do not cover, all of which inflate `wholesale`:**
//!
//! 1. `new_from_two_layer_chunks` rebuilds the whole renderer, so it also compiles nothing but
//!    does construct every pipeline, bind group, layout and uniform buffer from scratch. That
//!    is a FIXED cost, independent of scene size, and it is not what "wholesale re-mesh" is
//!    supposed to mean. The `setup` column measures it directly (the same builder over an
//!    EMPTY chunk set) so a reader can subtract it instead of taking this probe's word.
//! 2. Each timed wholesale build is dropped at the end of the sample, so wgpu resource
//!    teardown lands somewhere in the loop — outside the timed span, but on the same thread.
//! 3. Nothing here waits on the GPU. A timing ends when the CPU has handed the work over, not
//!    when the device is done with it. For a mesh path whose per-edit cost is meshing and
//!    buffer creation that is the right span, but it is not a frame time.
//!
//! The probe runs the incremental FIRST and the wholesale SECOND within each sample, because
//! the incremental has to mutate the persistent renderer for the next sample to be a real
//! incremental. Any warm-cache advantage from that ordering therefore favours the wholesale
//! column, i.e. it cannot manufacture the divergence the probe is looking for.
//!
//! Run: `cargo test --release --test remesh_cost_probe -- --ignored --nocapture --test-threads=1`

use std::time::Instant;

use camera::OrbitCamera;
use document::intent::{Intent, NodeSpec};
use document::scene::{NodeId, Scene};
use document::voxel::{GeometryParams, SdfShape};
use voxel_core::core_geom::MaterialChoice;
use voxel_core::units::Measurement;
use voxel_core::voxel::ShapeKind;
use voxel_worker::{
    AppCore, CuboidMeshRenderer, GpuContext, RebuildOutcome, COLOR_TARGET_FORMAT,
};

/// Voxels per block — the document default, and the density the edit-cost probes use, so the
/// two files' scene columns name the same scenes.
const DENSITY: u32 = 16;
/// How many edits to time per scene; the median is reported so one unlucky allocation does not
/// decide an interaction model.
const SAMPLES: usize = 9;
/// The blocks-extent of the node a manipulator actually drags: small, and entirely inside the
/// backdrop it sits in. Copied from `edit_cost_probe.rs` rather than imported — integration
/// test binaries share no code — so the two files stay independently editable.
const DRAGGED_BLOCKS: [u32; 3] = [2, 2, 2];

/// A one-node backdrop scene of `blocks` extent, as the panel would seed it.
fn scene_of(kind: ShapeKind, blocks: [u32; 3]) -> Scene {
    let shape = SdfShape::from_blocks(kind, blocks, 1, DENSITY);
    Scene::from_geometry(
        GeometryParams {
            shape: kind,
            size_voxels: shape.size_voxels,
            size_measurements: None,
            voxels_per_block: DENSITY,
            wall_blocks: 1,
        },
        MaterialChoice::Stone,
    )
}

/// The median of a set of millisecond timings.
fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
    samples[samples.len() / 2]
}

/// Add the small dragged Tool node through the same intent the add-flow emits, and return its
/// minted id (`add_node` selects what it adds, so `scene.active` names it).
fn add_dragged_node(scene: &mut Scene, app_core: &mut AppCore) -> NodeId {
    let shape = SdfShape::from_blocks(ShapeKind::Box, DRAGGED_BLOCKS, 1, DENSITY);
    app_core.apply_intent(
        scene,
        Intent::AddNode {
            content: NodeSpec::Tool {
                shape,
                material: MaterialChoice::Stone,
            },
        },
    );
    scene.active.expect("add_node selects the node it mints")
}

/// Park the dragged node at `offset_voxels` — the intent half of a drag step, untimed.
fn move_dragged_node(
    scene: &mut Scene,
    app_core: &mut AppCore,
    target: NodeId,
    offset_voxels: [i64; 3],
) {
    app_core.apply_intent(
        scene,
        Intent::SetOffset {
            target,
            offset_measurements: offset_voxels.map(Measurement::from_voxels),
        },
    );
}

/// **What a `None` from `rebuild` actually costs the user.** Both columns come off the SAME
/// rebuild output on the SAME scene at the SAME drag step, so the contrast is the re-mesh
/// strategy and nothing else — the resolve, the covering set, the frame and the dirty set are
/// all held fixed and shared between them.
///
/// Read the two columns as the two arms of the branch at `rebuild.rs`'s `buffers_reframed`: an
/// ordinary in-extent drag pays the `incr` column, and a drag that shifts the floating origin
/// (or changes density) pays the `whole` column INSTEAD. The gap between them is the true price
/// of a reframe, and it is the number `edit_cost_probe.rs` could not see.
///
/// `setup` is the control, and it is load-bearing: the wholesale builder does not merely
/// re-mesh, it reconstructs the entire renderer. `setup` is the same builder run over an EMPTY
/// chunk set, so `whole - setup` is the part that is genuinely proportional to the scene while
/// `setup` is the fixed renderer-construction toll. Do not quote `whole` without it.
///
/// **The expectation holds, and it holds hard.** Unlike the resolve, the re-mesh diverges
/// exactly as the shape of the two calls predicts: wholesale tracks the resident chunk count
/// almost linearly (~0.13 ms per chunk once `setup` is subtracted) while incremental tracks the
/// edit's footprint and is FLAT in scene size. At 3125 chunks that is ~390 ms against ~1.3 ms —
/// a ratio near 300×, and a number that is not a dropped frame but a visible stall. So the
/// answer `edit_cost_probe.rs` left open is settled: what makes a big scene expensive to edit
/// is not resolving it, it is re-meshing it, and a `None` from `rebuild` is the whole bill.
///
/// **The incremental column is not monotonic, and that is the more interesting half.** It peaks
/// on the MEDIUM backdrop and then falls as the scene grows. Nothing is anomalous about it: the
/// incremental re-meshes the dirty set dilated by its 26-neighbourhood, so its cost is the cost
/// of ~27 chunks of geometry, and how expensive a chunk is depends on how much surface runs
/// through it. The medium cylinder is small enough that the dragged node's neighbourhood is a
/// large slice of a dense body; the large and huge backdrops are thin-walled Tubes whose
/// interior — where the node is parked — is mostly empty, so the same 27 chunks carry far fewer
/// faces. This is the same lesson the locality probe reached from the other side: at this layer
/// cost tracks the neighbourhood's OCCUPANCY, not the scene's extent.
///
/// The probe deliberately keeps the dragged node INSIDE the backdrop so every rebuild localises
/// and the incremental arm is legitimate; it then calls the wholesale builder anyway, on that
/// same localised state, to price the counterfactual. That is the only way to hold everything
/// but the strategy constant — a real origin-shifting drag would change the frame too, and the
/// comparison would no longer be clean.
#[test]
#[ignore = "perf probe — run in release with --ignored --nocapture"]
fn remesh_cost_incremental_versus_wholesale_by_scene_size() {
    if !voxel_worker::gpu::adapter_available() {
        println!(
            "\nSKIPPED: no GPU adapter. Both entry points under test take a `wgpu::Device`, and\n\
             the two-layer CPU mesher is `pub(crate)` to `display`, so there is nothing to\n\
             measure without one. A software rasteriser would answer, but its buffer-creation\n\
             cost is not this machine's, so the probe declines rather than report a fiction.\n"
        );
        return;
    }
    let gpu = pollster::block_on(GpuContext::new(None));

    println!(
        "\n{:<22} {:>10} {:>8} {:>10} {:>11} {:>11} {:>8}",
        "backdrop", "voxels", "chunks", "dirty", "incr (ms)", "whole (ms)", "setup"
    );
    println!("{}", "-".repeat(86));

    for (label, kind, blocks) in [
        ("small  5x1x5", ShapeKind::Cylinder, [5u32, 1, 5]),
        ("medium 20x8x20", ShapeKind::Cylinder, [20, 8, 20]),
        ("large  50x10x50", ShapeKind::Tube, [50, 10, 50]),
        ("huge   100x20x100", ShapeKind::Tube, [100, 20, 100]),
    ] {
        let mut scene = scene_of(kind, blocks);
        let mut app_core = AppCore::new(OrbitCamera::default());
        let dragged = add_dragged_node(&mut scene, &mut app_core);

        // Park the dragged node near the backdrop's centre, so the whole gesture happens inside
        // existing geometry: the extent never grows, the origin never shifts, and every rebuild
        // below hands back a real `Some(dirty)`.
        let centre = blocks.map(|b| (b as i64 / 2) * DENSITY as i64);
        move_dragged_node(&mut scene, &mut app_core, dragged, centre);

        // The cold build: one wholesale mesh, which also seeds the persistent renderer the
        // incremental arm will keep mutating. Untimed — it is the first-build cost, already
        // reported by `edit_cost_probe.rs`, and not what a drag pays.
        let mut renderer = match app_core.rebuild(&scene, DENSITY) {
            RebuildOutcome::Built(output) => CuboidMeshRenderer::new_from_two_layer_chunks(
                &gpu.device,
                &gpu.queue,
                COLOR_TARGET_FORMAT,
                &output.two_layer_chunks,
                output.region_dimensions,
                output.recentre_voxels,
                DENSITY,
            ),
            RebuildOutcome::DensityRejected { .. } => panic!("the probe's density is in bounds"),
        };

        let mut incremental_samples = Vec::with_capacity(SAMPLES);
        let mut wholesale_samples = Vec::with_capacity(SAMPLES);
        let mut setup_samples = Vec::with_capacity(SAMPLES);
        let mut resident_chunks = 0usize;
        let mut dirty_chunks = 0usize;

        for step in 1..=SAMPLES as i64 {
            move_dragged_node(
                &mut scene,
                &mut app_core,
                dragged,
                [centre[0] + step, centre[1], centre[2]],
            );
            let RebuildOutcome::Built(output) = app_core.rebuild(&scene, DENSITY) else {
                panic!("the probe's density is in bounds");
            };
            let dirty = output
                .incremental_dirty_chunks
                .as_ref()
                .expect("an in-extent drag step must localise, or the contrast is meaningless");
            resident_chunks = output.two_layer_chunks.len();
            dirty_chunks = dirty.len();

            // (a) The incremental arm — and the one that must run first, because the renderer
            // it mutates is the state the NEXT step's incremental depends on.
            let started = Instant::now();
            renderer.incremental_rebuild_from_two_layer_chunks(
                &gpu.device,
                &output.two_layer_chunks,
                output.region_dimensions,
                output.recentre_voxels,
                DENSITY,
                dirty,
            );
            incremental_samples.push(started.elapsed().as_secs_f64() * 1000.0);

            // (b) The wholesale arm, on the very same output: what the shell would have done
            // had this step shifted the origin. Built and immediately dropped — it is priced,
            // not kept, so the incremental arm's renderer stays the live one.
            let started = Instant::now();
            let counterfactual = CuboidMeshRenderer::new_from_two_layer_chunks(
                &gpu.device,
                &gpu.queue,
                COLOR_TARGET_FORMAT,
                &output.two_layer_chunks,
                output.region_dimensions,
                output.recentre_voxels,
                DENSITY,
            );
            wholesale_samples.push(started.elapsed().as_secs_f64() * 1000.0);

            // (c) The control: the SAME builder over no chunks at all. Whatever this costs is
            // renderer construction, not meshing, and it is inside every (b) above.
            let started = Instant::now();
            let empty = CuboidMeshRenderer::new_from_two_layer_chunks(
                &gpu.device,
                &gpu.queue,
                COLOR_TARGET_FORMAT,
                &[],
                output.region_dimensions,
                output.recentre_voxels,
                DENSITY,
            );
            setup_samples.push(started.elapsed().as_secs_f64() * 1000.0);

            drop(counterfactual);
            drop(empty);
        }

        let voxels: u64 = blocks.iter().map(|b| (*b * DENSITY) as u64).product();
        println!(
            "{label:<22} {voxels:>10} {resident_chunks:>8} {dirty_chunks:>10} {:>11.2} {:>11.2} {:>8.2}",
            median(incremental_samples),
            median(wholesale_samples),
            median(setup_samples),
        );
    }

    println!(
        "\n'chunks' = the resident covering set the wholesale arm re-meshes in full.\n\
         'dirty'  = the chunks the edit evicted; the incremental arm re-meshes those dilated by\n\
         their 26-neighbourhood and intersected with the resident set, so its real footprint is\n\
         larger than this column and smaller than 'chunks'.\n\
         'incr'   = re-mesh + re-upload the dirty-dilated subset, keeping every other buffer.\n\
         'whole'  = re-mesh + re-upload the whole resident set, reconstructing the renderer.\n\
         'setup'  = the wholesale builder over an EMPTY chunk set: fixed renderer construction\n\
         (pipelines, layouts, bind groups, uniform buffer), which is included in 'whole'. The\n\
         scene-proportional part of a wholesale re-mesh is 'whole' MINUS 'setup'.\n\
         \n\
         Both columns time CPU meshing AND wgpu buffer creation together — the two-layer CPU\n\
         mesher is crate-private to `display`, so no public seam splits them, and this probe\n\
         does not guess at the ratio. Neither column waits on the GPU: a timing ends when the\n\
         CPU has handed the work over. Neither is a frame time.\n\
         \n\
         All drag steps here stay INSIDE the backdrop, so every rebuild genuinely localised;\n\
         the wholesale column is the priced counterfactual for the same step, not a separate\n\
         gesture. That is what makes the gap attributable to the strategy alone.\n"
    );
}
