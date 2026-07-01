//! GPU view-resolve P1 spike — the CPU↔GPU A/B equivalence net (ADR 0007 §5/§6).
//!
//! Resolves a producer on the CPU through the REAL scene path (so the reference grid
//! is exactly what the app feeds the fog — a single node, recentred by
//! `resolve_region`), buckets it into apron'd per-chunk fog volumes via the SHIPPED
//! `build_per_chunk_fog_occupancy`, then GPU-evaluates the SAME chunks' occupancy and
//! asserts they are **byte-identical**. This is the spike that answers ADR 0007's
//! central open question: does the Rust↔WGSL float eval agree at the occupancy
//! boundary? Exact is the target; a measured divergence is a finding to REPORT
//! (count + where), never silently tolerate (ADR 0007 §6).
//!
//! - SDF tier: f32 both sides → expected exact.
//! - Sketch tier: CPU does the polygon test in **f64**, the GPU in **f32** (no portable
//!   f64 in WGSL). Extrude samples are half-integer over integer vertices (a wide gap
//!   from any edge → expected exact); revolve samples include an irrational radius
//!   (the genuine divergence surface). The matrix measures all of it.
//!
//! Run: `cargo test --features gpu --test gpu_parity`
#![cfg(feature = "gpu")]
// This IS the A/B net for the (deprecated) CPU fog densify — referencing it is the point.
#![allow(deprecated)]

use voxel_worker::gpu_resolve::GpuResolver;
use voxel_worker::renderer::{build_per_chunk_fog_occupancy, PerChunkFogOccupancy, MAX_FOG_CHUNKS};
use voxel_worker::voxel::{signed_distance, GeometryParams, SdfShape, ShapeKind, VoxelGrid, VoxelProducer};
use voxel_worker::{
    DebugCloudField, GpuContext, MaterialChoice, Node, NodeContent, Part, PlaneAxis, RevolveAxis,
    Scene, Sketch, SketchPoint, SketchSolid,
};

/// A divergent apron cell, located for the report.
struct Mismatch {
    chunk_coord: [i32; 3],
    apron: [usize; 3],
    cpu: u8,
    gpu: u8,
}

/// Walk the CPU reference vs the GPU occupancy and collect every byte that differs.
/// (Asserts the per-chunk lengths line up first.)
fn collect_mismatches(
    case: &str,
    reference: &PerChunkFogOccupancy,
    gpu_occupancy: &[Vec<u8>],
    pad: usize,
) -> Vec<Mismatch> {
    let mut mismatches = Vec::new();
    for (volume, gpu_cells) in reference.volumes.iter().zip(gpu_occupancy) {
        assert_eq!(
            volume.occupancy.len(),
            gpu_cells.len(),
            "{case}: chunk {:?} length mismatch (CPU {} vs GPU {})",
            volume.chunk_coord,
            volume.occupancy.len(),
            gpu_cells.len()
        );
        for (idx, (&cpu, &gpu)) in volume.occupancy.iter().zip(gpu_cells).enumerate() {
            if cpu != gpu {
                mismatches.push(Mismatch {
                    chunk_coord: volume.chunk_coord,
                    apron: [idx % pad, (idx / pad) % pad, idx / (pad * pad)],
                    cpu,
                    gpu,
                });
            }
        }
    }
    mismatches
}

/// The recentred-frame fog-global voxel coordinate of an apron cell (== the producer
/// local voxel index for a lone producer, `local_offset == 0` — see the shader).
fn voxel_index_of(chunk_coord: [i32; 3], apron: [usize; 3], chunk_extent: i64) -> [i64; 3] {
    [
        chunk_coord[0] as i64 * chunk_extent + apron[0] as i64 - 1,
        chunk_coord[1] as i64 * chunk_extent + apron[1] as i64 - 1,
        chunk_coord[2] as i64 * chunk_extent + apron[2] as i64 - 1,
    ]
}

/// Format the first few mismatches into a one-line-each report via a per-cell
/// diagnostic that recomputes the CPU predicate at the differing voxel.
fn report(case: &str, total: usize, mismatches: &[Mismatch], diagnose: impl Fn(&Mismatch) -> String) -> String {
    let mut lines = vec![format!("{case}: {}/{total} cells differ", mismatches.len())];
    for m in mismatches.iter().take(6) {
        lines.push(format!("    {}", diagnose(m)));
    }
    lines.join("\n")
}

// ===========================================================================
// SDF tier
// ===========================================================================

struct SdfCase {
    name: &'static str,
    kind: ShapeKind,
    size_voxels: [u32; 3],
    wall_blocks: u32,
    voxels_per_block: u32,
}

const SDF_CASES: &[SdfCase] = &[
    SdfCase { name: "cylinder-80-16-80-d16", kind: ShapeKind::Cylinder, size_voxels: [80, 16, 80], wall_blocks: 1, voxels_per_block: 16 },
    SdfCase { name: "box-80-16-80-d16", kind: ShapeKind::Box, size_voxels: [80, 16, 80], wall_blocks: 1, voxels_per_block: 16 },
    SdfCase { name: "sphere-80-80-80-d16", kind: ShapeKind::Sphere, size_voxels: [80, 80, 80], wall_blocks: 1, voxels_per_block: 16 },
    SdfCase { name: "torus-128-32-128-d16", kind: ShapeKind::Torus, size_voxels: [128, 32, 128], wall_blocks: 1, voxels_per_block: 16 },
    SdfCase { name: "tube-80-16-80-w1-d16", kind: ShapeKind::Tube, size_voxels: [80, 16, 80], wall_blocks: 1, voxels_per_block: 16 },
    SdfCase { name: "sphere-33-33-33-d4", kind: ShapeKind::Sphere, size_voxels: [33, 33, 33], wall_blocks: 1, voxels_per_block: 4 },
    SdfCase { name: "box-31-17-49-d4", kind: ShapeKind::Box, size_voxels: [31, 17, 49], wall_blocks: 1, voxels_per_block: 4 },
    SdfCase { name: "cylinder-45-21-45-d4", kind: ShapeKind::Cylinder, size_voxels: [45, 21, 45], wall_blocks: 1, voxels_per_block: 4 },
    SdfCase { name: "torus-49-13-49-d4", kind: ShapeKind::Torus, size_voxels: [49, 13, 49], wall_blocks: 1, voxels_per_block: 4 },
    SdfCase { name: "tube-50-20-50-w2-d4", kind: ShapeKind::Tube, size_voxels: [50, 20, 50], wall_blocks: 2, voxels_per_block: 4 },
];

#[test]
fn gpu_sdf_occupancy_matches_per_chunk_fog_exactly() {
    let gpu = pollster::block_on(GpuContext::new(None));
    let resolver = GpuResolver::new(&gpu.device);
    let mut failures: Vec<String> = Vec::new();

    for case in SDF_CASES {
        let vpb = case.voxels_per_block;
        let shape = SdfShape::from_voxels(case.kind, case.size_voxels, case.wall_blocks);
        let geometry = GeometryParams {
            shape: case.kind,
            size_voxels: case.size_voxels,
            size_measurements: None,
            voxels_per_block: vpb,
            wall_blocks: case.wall_blocks,
        };
        let scene = Scene::from_geometry(geometry, MaterialChoice::default());
        let grid = scene.resolve_region(scene.full_extent_blocks(vpb), vpb, 0);
        let reference = build_per_chunk_fog_occupancy(&grid, vpb);
        let chunk_coords: Vec<[i32; 3]> = reference.volumes.iter().map(|v| v.chunk_coord).collect();
        if chunk_coords.is_empty() {
            failures.push(format!("{}: CPU produced zero chunk volumes", case.name));
            continue;
        }

        let gpu_occupancy = resolver.resolve_sdf_occupancy(&gpu.device, &gpu.queue, &shape, vpb, &chunk_coords);
        let pad = (reference.chunk_extent + 2) as usize;
        let chunk_extent = reference.chunk_extent as i64;
        let mismatches = collect_mismatches(case.name, &reference, &gpu_occupancy, pad);
        if mismatches.is_empty() {
            continue;
        }
        let total: usize = reference.volumes.iter().map(|v| v.occupancy.len()).sum();
        let semi = glam::Vec3::new(
            grid.dimensions[0] as f32 / 2.0,
            grid.dimensions[1] as f32 / 2.0,
            grid.dimensions[2] as f32 / 2.0,
        );
        let wall_voxels = (case.wall_blocks * vpb.max(1)) as f32;
        failures.push(report(case.name, total, &mismatches, |m| {
            let vi = voxel_index_of(m.chunk_coord, m.apron, chunk_extent);
            let sample = glam::Vec3::new(
                vi[0] as f32 + 0.5 - semi.x,
                vi[1] as f32 + 0.5 - semi.y,
                vi[2] as f32 + 0.5 - semi.z,
            );
            let sdf = signed_distance(case.kind, sample, semi, wall_voxels);
            format!("vi={vi:?} cpu_sdf={sdf:+.6e} cpu={} gpu={}", m.cpu, m.gpu)
        }));
    }

    assert!(
        failures.is_empty(),
        "GPU↔CPU SDF occupancy diverged (ADR 0007 §6 — exact is the target):\n{}",
        failures.join("\n")
    );
}

// ===========================================================================
// Sketch tier (extrude + revolve)
// ===========================================================================

/// How the sketch case builds its producer; the test wraps it in a one-node scene.
enum SketchKind {
    Extrude { height_blocks: i64 },
    Revolve { axis: RevolveAxis, turn_degrees: u32 },
}

struct SketchCase {
    name: &'static str,
    plane: PlaneAxis,
    /// Profile vertices in BLOCKS (scaled by density at build time → voxel coords).
    profile_blocks: &'static [[i64; 2]],
    kind: SketchKind,
    voxels_per_block: u32,
}

const SKETCH_CASES: &[SketchCase] = &[
    // Rectangle extrude == box (exact reference for the extrude path).
    SketchCase { name: "extrude-rect-4x2x3-d4", plane: PlaneAxis::Z, profile_blocks: &[[0, 0], [4, 0], [4, 2], [0, 2]], kind: SketchKind::Extrude { height_blocks: 3 }, voxels_per_block: 4 },
    // The demo L (concave, reflex vertex) extruded up, multi-chunk at d8.
    SketchCase { name: "extrude-L-d8", plane: PlaneAxis::Z, profile_blocks: &[[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]], kind: SketchKind::Extrude { height_blocks: 3 }, voxels_per_block: 8 },
    // A triangle (odd, non-axis-aligned edges) extruded — slanted crossings.
    SketchCase { name: "extrude-tri-d4", plane: PlaneAxis::Y, profile_blocks: &[[0, 0], [7, 1], [3, 6]], kind: SketchKind::Extrude { height_blocks: 2 }, voxels_per_block: 4 },
    // Rectangle revolve == cylinder, full turn (one-sided radial).
    SketchCase { name: "revolve-rect-cyl-d4", plane: PlaneAxis::X, profile_blocks: &[[0, 0], [5, 0], [5, 4], [0, 4]], kind: SketchKind::Revolve { axis: RevolveAxis::InPlane1, turn_degrees: 360 }, voxels_per_block: 4 },
    // The demo vase (stepped silhouette) revolved 360° — the headline revolve shape.
    SketchCase { name: "revolve-vase-d4", plane: PlaneAxis::X, profile_blocks: &[[0, 0], [4, 0], [4, 1], [2, 3], [2, 5], [4, 6], [3, 8], [0, 8]], kind: SketchKind::Revolve { axis: RevolveAxis::InPlane1, turn_degrees: 360 }, voxels_per_block: 4 },
    // A half-disc-ish profile revolved → rounded body (curved radial boundary).
    SketchCase { name: "revolve-bowl-d8", plane: PlaneAxis::X, profile_blocks: &[[0, 0], [6, 0], [6, 1], [1, 6], [0, 6]], kind: SketchKind::Revolve { axis: RevolveAxis::InPlane1, turn_degrees: 360 }, voxels_per_block: 8 },
    // Partial turn (180°) — exercises the atan2 theta gate (transcendental divergence).
    SketchCase { name: "revolve-rect-half-d4", plane: PlaneAxis::X, profile_blocks: &[[0, 0], [5, 0], [5, 4], [0, 4]], kind: SketchKind::Revolve { axis: RevolveAxis::InPlane1, turn_degrees: 180 }, voxels_per_block: 4 },
    // Straddling profile (radial coords cross 0) — exercises the +radius/−radius fold.
    SketchCase { name: "revolve-straddle-d4", plane: PlaneAxis::X, profile_blocks: &[[-3, 0], [3, 0], [3, 4], [-3, 4]], kind: SketchKind::Revolve { axis: RevolveAxis::InPlane1, turn_degrees: 360 }, voxels_per_block: 4 },
];

impl SketchCase {
    fn build(&self) -> SketchSolid {
        let d = self.voxels_per_block as i64;
        let profile: Vec<SketchPoint> = self
            .profile_blocks
            .iter()
            .map(|&[a, b]| SketchPoint::new(a * d, b * d))
            .collect();
        let sketch = Sketch::new(self.plane, profile);
        match self.kind {
            SketchKind::Extrude { height_blocks } => {
                SketchSolid::extrude(sketch, (height_blocks * d) as u32)
            }
            SketchKind::Revolve { axis, turn_degrees } => {
                SketchSolid::revolve(sketch, axis, turn_degrees)
            }
        }
    }
}

#[test]
fn gpu_sketch_occupancy_matches_per_chunk_fog_exactly() {
    let gpu = pollster::block_on(GpuContext::new(None));
    let resolver = GpuResolver::new(&gpu.device);
    let mut failures: Vec<String> = Vec::new();

    for case in SKETCH_CASES {
        let vpb = case.voxels_per_block;
        let producer = case.build();
        let node = Node::new(
            "Sketch",
            NodeContent::SketchTool {
                producer: producer.clone(),
                material: MaterialChoice::default(),
            },
        );
        let mut scene = Scene::single_node(node);
        scene.voxels_per_block = vpb;
        let grid = scene.resolve_region(scene.full_extent_blocks(vpb), vpb, 0);
        let reference = build_per_chunk_fog_occupancy(&grid, vpb);
        let chunk_coords: Vec<[i32; 3]> = reference.volumes.iter().map(|v| v.chunk_coord).collect();
        if chunk_coords.is_empty() {
            failures.push(format!("{}: CPU produced zero chunk volumes", case.name));
            continue;
        }

        let gpu_occupancy = resolver.resolve_sketch_occupancy(&gpu.device, &gpu.queue, &producer, vpb, &chunk_coords);
        let pad = (reference.chunk_extent + 2) as usize;
        let chunk_extent = reference.chunk_extent as i64;
        let mismatches = collect_mismatches(case.name, &reference, &gpu_occupancy, pad);
        if mismatches.is_empty() {
            continue;
        }
        let total: usize = reference.volumes.iter().map(|v| v.occupancy.len()).sum();
        let dims = grid.dimensions;
        failures.push(report(case.name, total, &mismatches, |m| {
            let vi = voxel_index_of(m.chunk_coord, m.apron, chunk_extent);
            // Centred radius (revolve diagnostic) — the f32 value both sides start from.
            let centred = [
                vi[0] as f32 + 0.5 - dims[0] as f32 / 2.0,
                vi[1] as f32 + 0.5 - dims[1] as f32 / 2.0,
                vi[2] as f32 + 0.5 - dims[2] as f32 / 2.0,
            ];
            format!("vi={vi:?} centred={centred:?} cpu={} gpu={}", m.cpu, m.gpu)
        }));
    }

    assert!(
        failures.is_empty(),
        "GPU↔CPU sketch occupancy diverged (ADR 0007 §6 — measure, don't silently tolerate):\n{}",
        failures.join("\n")
    );
}

// ===========================================================================
// DebugClouds tier (Perlin fBm) — the §6 noise-parity question
// ===========================================================================

/// `(name, dimensions, seed, voxels_per_block)`. Densities kept low so the chunk
/// count stays under the single-dimension workgroup limit.
const CLOUD_CASES: &[(&str, [u32; 3], u32, u32)] = &[
    ("clouds-48-d4-s1", [48, 48, 48], 1, 4),
    ("clouds-48-d4-s7", [48, 48, 48], 7, 4),
    ("clouds-64-32-64-d4-s3", [64, 32, 64], 3, 4),
];

#[test]
fn gpu_clouds_occupancy_matches_per_chunk_fog_exactly() {
    let gpu = pollster::block_on(GpuContext::new(None));
    let resolver = GpuResolver::new(&gpu.device);
    let mut failures: Vec<String> = Vec::new();

    for &(name, dims, seed, vpb) in CLOUD_CASES {
        let field = DebugCloudField { dimensions: dims, seed };
        // A bare cloud resolve is corner-anchored at `[0, dim)` with `recentre = [0,0,0]`
        // (the real Part-only frame, ADR 0008). The fog decodes it correctly via the
        // carried recentre, so the GPU evaluates at `local_offset = 0` (fog-global ==
        // producer-local index) — no manual recentre needed.
        let mut grid = VoxelGrid::new(dims);
        field.resolve(&mut grid, vpb);

        let reference = build_per_chunk_fog_occupancy(&grid, vpb);
        let chunk_coords: Vec<[i32; 3]> = reference.volumes.iter().map(|v| v.chunk_coord).collect();
        if chunk_coords.is_empty() {
            failures.push(format!("{name}: CPU produced zero chunk volumes"));
            continue;
        }

        let gpu_occupancy = resolver.resolve_clouds_occupancy(&gpu.device, &gpu.queue, &field, vpb, &chunk_coords);
        let pad = (reference.chunk_extent + 2) as usize;
        let chunk_extent = reference.chunk_extent as i64;
        let mismatches = collect_mismatches(name, &reference, &gpu_occupancy, pad);
        if mismatches.is_empty() {
            continue;
        }
        let total: usize = reference.volumes.iter().map(|v| v.occupancy.len()).sum();
        failures.push(report(name, total, &mismatches, |m| {
            let vi = voxel_index_of(m.chunk_coord, m.apron, chunk_extent);
            format!("vi={vi:?} cpu={} gpu={}", m.cpu, m.gpu)
        }));
    }

    assert!(
        failures.is_empty(),
        "GPU↔CPU cloud occupancy diverged (ADR 0007 §6 — Perlin fBm must match):\n{}",
        failures.join("\n")
    );
}

// ===========================================================================
// Atlas packing tier — the production texture-write mechanic
// ===========================================================================

/// Replicate `upload_grid_per_chunk`'s atlas packing on the CPU from the reference
/// volumes, so the GPU-produced R8 atlas can be asserted byte-identical to it. Returns
/// the `atlas_dim³` bytes plus the tile geometry.
fn cpu_atlas(reference: &PerChunkFogOccupancy) -> (Vec<u8>, u32, u32) {
    let pad = reference.chunk_extent as usize + 2;
    let chunk_count = reference.volumes.len();
    let tiles_per_axis = ((chunk_count as f64).cbrt().ceil() as u32).max(1);
    let atlas_dim = tiles_per_axis * pad as u32;
    let atlas_dim_usize = atlas_dim as usize;
    let mut atlas = vec![0u8; atlas_dim_usize.pow(3)];
    for (tile_index, volume) in reference.volumes.iter().enumerate() {
        let tx = (tile_index as u32) % tiles_per_axis;
        let ty = ((tile_index as u32) / tiles_per_axis) % tiles_per_axis;
        let tz = (tile_index as u32) / (tiles_per_axis * tiles_per_axis);
        let base = [tx as usize * pad, ty as usize * pad, tz as usize * pad];
        for lz in 0..pad {
            for ly in 0..pad {
                for lx in 0..pad {
                    let src = (lz * pad + ly) * pad + lx;
                    let ax = base[0] + lx;
                    let ay = base[1] + ly;
                    let az = base[2] + lz;
                    let dst = (az * atlas_dim_usize + ay) * atlas_dim_usize + ax;
                    atlas[dst] = volume.occupancy[src];
                }
            }
        }
    }
    (atlas, atlas_dim, tiles_per_axis)
}

/// Compare a GPU `AtlasResult` against the CPU-packed atlas; `None` on byte-identical.
fn compare_atlas(
    case: &str,
    cpu: &[u8],
    cpu_dim: u32,
    cpu_tiles: u32,
    gpu_atlas: &[u8],
    gpu_dim: u32,
    gpu_tiles: u32,
) -> Option<String> {
    if (cpu_dim, cpu_tiles) != (gpu_dim, gpu_tiles) {
        return Some(format!(
            "{case}: geometry mismatch — CPU dim={cpu_dim} tiles={cpu_tiles} vs GPU dim={gpu_dim} tiles={gpu_tiles}"
        ));
    }
    let differing = cpu.iter().zip(gpu_atlas).filter(|(a, b)| a != b).count();
    if differing == 0 {
        None
    } else {
        Some(format!("{case}: {differing}/{} atlas bytes differ", cpu.len()))
    }
}

#[test]
fn gpu_atlas_matches_cpu_upload_packing() {
    let gpu = pollster::block_on(GpuContext::new(None));
    let resolver = GpuResolver::new(&gpu.device);
    let mut failures: Vec<String> = Vec::new();

    // SDF: sphere@d16 (8 chunks → tiles_per_axis 2, full) and box@d4 (12 chunks →
    // tiles_per_axis 3, so 27 tile slots with 15 EMPTY — exercises the zero-fill).
    for case in [&SDF_CASES[2], &SDF_CASES[6]] {
        let vpb = case.voxels_per_block;
        let shape = SdfShape::from_voxels(case.kind, case.size_voxels, case.wall_blocks);
        let geometry = GeometryParams {
            shape: case.kind,
            size_voxels: case.size_voxels,
            size_measurements: None,
            voxels_per_block: vpb,
            wall_blocks: case.wall_blocks,
        };
        let scene = Scene::from_geometry(geometry, MaterialChoice::default());
        let grid = scene.resolve_region(scene.full_extent_blocks(vpb), vpb, 0);
        let reference = build_per_chunk_fog_occupancy(&grid, vpb);
        let chunk_coords: Vec<[i32; 3]> = reference.volumes.iter().map(|v| v.chunk_coord).collect();
        let (cpu, dim, tiles) = cpu_atlas(&reference);
        let result = resolver.resolve_sdf_atlas(&gpu.device, &gpu.queue, &shape, vpb, &chunk_coords);
        if let Some(f) = compare_atlas(case.name, &cpu, dim, tiles, &result.atlas, result.atlas_dim, result.tiles_per_axis) {
            failures.push(f);
        }
    }

    // Sketch: the concave L extrude and the revolved vase.
    for case in [&SKETCH_CASES[1], &SKETCH_CASES[4]] {
        let vpb = case.voxels_per_block;
        let producer = case.build();
        let node = Node::new(
            "Sketch",
            NodeContent::SketchTool { producer: producer.clone(), material: MaterialChoice::default() },
        );
        let mut scene = Scene::single_node(node);
        scene.voxels_per_block = vpb;
        let grid = scene.resolve_region(scene.full_extent_blocks(vpb), vpb, 0);
        let reference = build_per_chunk_fog_occupancy(&grid, vpb);
        let chunk_coords: Vec<[i32; 3]> = reference.volumes.iter().map(|v| v.chunk_coord).collect();
        let (cpu, dim, tiles) = cpu_atlas(&reference);
        let result = resolver.resolve_sketch_atlas(&gpu.device, &gpu.queue, &producer, vpb, &chunk_coords);
        if let Some(f) = compare_atlas(case.name, &cpu, dim, tiles, &result.atlas, result.atlas_dim, result.tiles_per_axis) {
            failures.push(f);
        }
    }

    assert!(
        failures.is_empty(),
        "GPU atlas != CPU upload_grid_per_chunk packing:\n{}",
        failures.join("\n")
    );
}


// ===========================================================================
// Compaction tier (ADR 0007 option C) — drop empty-interior covering tiles
// ===========================================================================

/// `resolve_single_producer_fog_atlas` must COMPACT the covering set down to exactly the
/// CPU non-empty chunk set, so a dense producer whose covering tiles overflow the atlas
/// budget still fits the GPU path (the dense-`DebugClouds` case from ADR 0007 finding #2).
/// This is the `debug-clouds` golden's scene (128³ @ d2): 4096 covering tiles, but only
/// ~679 non-empty — the full covering set exceeds MAX_FOG_CHUNKS while the compacted set
/// fits. The fog shader is world_origin-keyed, so a dropped empty tile renders identically
/// to the zeroed C′ tile it replaces (the goldens guard that render equivalence).
#[test]
fn gpu_atlas_compaction_drops_empty_interior_tiles() {
    let gpu = pollster::block_on(GpuContext::new(None));
    let resolver = GpuResolver::new(&gpu.device);

    let dims = [128u32, 128, 128];
    let vpb = 2u32;
    let seed = 0u32;

    // CPU reference: resolve the cloud field corner-anchored (recentre [0,0,0], the real
    // Part-only frame) and bucket into the non-empty per-chunk set.
    let field = DebugCloudField { dimensions: dims, seed };
    let mut grid = VoxelGrid::new(dims);
    field.resolve(&mut grid, vpb);
    let reference = build_per_chunk_fog_occupancy(&grid, vpb);
    let chunk_extent = reference.chunk_extent as i64;

    // The covering set the resolver enumerates BEFORE compaction.
    let ceil = |d: i64| (d + chunk_extent - 1) / chunk_extent;
    let covering = (ceil(dims[0] as i64) * ceil(dims[1] as i64) * ceil(dims[2] as i64)) as usize;

    // GPU compacting resolve through the real single-producer path.
    let scene = Scene::single_node(Node::new(
        "Clouds",
        NodeContent::Part(Part::DebugClouds { seed }),
    ));
    let producer = scene.single_producer().expect("DebugClouds is a single producer");
    let atlas = resolver
        .resolve_single_producer_fog_atlas(&gpu.device, &gpu.queue, &producer, dims, [0, 0, 0], vpb)
        .expect("the clouds dispatch fits and the producer has interior voxels");

    // The scene must actually have empty-interior covering tiles to drop, else the test
    // proves nothing.
    assert!(
        reference.volumes.len() < covering,
        "test scene has no empty covering tiles to drop (covering={covering}, nonempty={})",
        reference.volumes.len()
    );

    // Compaction shrank the covering set to EXACTLY the CPU non-empty set...
    assert_eq!(
        atlas.world_origins.len(),
        reference.volumes.len(),
        "compacted tile count must equal the CPU non-empty chunk count"
    );

    // ...and the surviving tiles ARE the CPU non-empty chunks (compared as a SET — the
    // covering enumeration order differs from the CPU's coord sort, but the fog shader is
    // world_origin-keyed so tile order is irrelevant). world_origin is an exact multiple of
    // the (small) chunk extent here, so the f32 → i64 key is lossless.
    let to_key = |o: [f32; 3]| [o[0] as i64, o[1] as i64, o[2] as i64];
    let gpu_set: std::collections::HashSet<[i64; 3]> =
        atlas.world_origins.iter().map(|&o| to_key(o)).collect();
    let cpu_set: std::collections::HashSet<[i64; 3]> =
        reference.volumes.iter().map(|v| to_key(v.world_origin)).collect();
    assert_eq!(gpu_set, cpu_set, "compacted tiles must be exactly the CPU non-empty set");

    // The headline: the full covering set overflows the budget but the compacted set fits,
    // so the GPU path now COVERS this scene instead of falling back to the CPU densify.
    assert!(
        covering > MAX_FOG_CHUNKS,
        "precondition: covering ({covering}) must overflow MAX_FOG_CHUNKS ({MAX_FOG_CHUNKS})"
    );
    assert!(
        atlas.world_origins.len() <= MAX_FOG_CHUNKS,
        "compacted count ({}) must fit MAX_FOG_CHUNKS ({MAX_FOG_CHUNKS})",
        atlas.world_origins.len()
    );

    // The atlas is sized to the COMPACT count, not the covering count.
    let expected_tiles = ((atlas.world_origins.len() as f64).cbrt().ceil() as u32).max(1);
    assert_eq!(atlas.tiles_per_axis, expected_tiles, "atlas tiles sized to compact count");
    assert_eq!(atlas.atlas_dim, expected_tiles * atlas.pad);
}

// ===========================================================================
// Multi-dimensional dispatch tier (#56) — large scenes stay on the GPU path
// ===========================================================================

/// The single-dimension workgroup limit the fix routes AROUND. A scene whose
/// `pad³ · num_chunks / 64` exceeds this used to bail `resolve_single_producer_fog_atlas`
/// to `None` → the 26s CPU densify; the 2-D dispatch now covers it. wgpu guarantees at
/// least this on every backend (the real device limit is usually exactly this).
const SINGLE_DIM_WORKGROUP_LIMIT: usize = 65_535;

/// A solid box big enough that its covering-chunk dispatch REQUIRES a 2-D workgroup grid
/// (`pad³ · num_chunks / 64 > 65_535`), so this case actually exercises the #56 fix. At
/// d16 the chunk extent is 64 voxels (pad 66, 66³ = 287_496 cells/chunk), so a 256×256×64
/// box covers 4×4×1 = 16 chunks → 16·287_496/64 ≈ 71_874 workgroups, well over the limit,
/// yet only 16 non-empty chunks (< MAX_FOG_CHUNKS) so the scene stays on the GPU. A solid
/// box has every covering chunk occupied, so no compaction hides the large dispatch.
#[test]
fn gpu_multidim_dispatch_matches_cpu_and_trips_single_dim_limit() {
    let gpu = pollster::block_on(GpuContext::new(None));
    let resolver = GpuResolver::new(&gpu.device);

    let vpb = 16u32;
    let size_voxels = [256u32, 256, 64];
    let kind = ShapeKind::Box;
    let wall_blocks = 1u32;

    let shape = SdfShape::from_voxels(kind, size_voxels, wall_blocks);
    let geometry = GeometryParams {
        shape: kind,
        size_voxels,
        size_measurements: None,
        voxels_per_block: vpb,
        wall_blocks,
    };
    let scene = Scene::from_geometry(geometry, MaterialChoice::default());
    let grid = scene.resolve_region(scene.full_extent_blocks(vpb), vpb, 0);
    let reference = build_per_chunk_fog_occupancy(&grid, vpb);
    let chunk_coords: Vec<[i32; 3]> = reference.volumes.iter().map(|v| v.chunk_coord).collect();
    assert!(!chunk_coords.is_empty(), "CPU produced zero chunk volumes");

    // Precondition: the dispatch MUST need a 2-D grid, else the fix is unexercised.
    let pad = (reference.chunk_extent + 2) as usize;
    let cells = pad * pad * pad * chunk_coords.len();
    let workgroups = cells.div_ceil(64);
    assert!(
        workgroups > SINGLE_DIM_WORKGROUP_LIMIT,
        "multi-dim precondition: {workgroups} workgroups ({} chunks × {pad}³) must exceed the \
         {SINGLE_DIM_WORKGROUP_LIMIT} single-dimension limit",
        chunk_coords.len()
    );
    assert!(
        chunk_coords.len() <= MAX_FOG_CHUNKS,
        "the scene must fit the atlas budget so it stays on the GPU (#56)"
    );

    // (a) A/B `main` pass (binding 2 = per-cell u32) over the multi-dim dispatch.
    let gpu_occupancy = resolver.resolve_sdf_occupancy(&gpu.device, &gpu.queue, &shape, vpb, &chunk_coords);
    let mismatches = collect_mismatches("multidim-box", &reference, &gpu_occupancy, pad);
    assert!(
        mismatches.is_empty(),
        "GPU↔CPU occupancy diverged on the multi-dim `main` dispatch: {}/{} cells differ (first {:?})",
        mismatches.len(),
        cells,
        mismatches.first().map(|m| (m.chunk_coord, m.apron, m.cpu, m.gpu))
    );

    // (b) The full single-producer atlas path (`main_flags` + `main_atlas`), which is the
    // one that used to bail to `None`. It must now return `Some` and pack every covering
    // chunk (a solid box has no empty interior, so no compaction).
    let producer = scene.single_producer().expect("a from-geometry scene is a single producer");
    let atlas = resolver
        .resolve_single_producer_fog_atlas(
            &gpu.device,
            &gpu.queue,
            &producer,
            grid.dimensions,
            grid.recentre_voxels,
            vpb,
        )
        .expect("large single-producer scene must stay on the GPU path (#56), not fall back");
    assert_eq!(
        atlas.world_origins.len(),
        reference.volumes.len(),
        "the solid box's covering set is fully occupied, so the GPU atlas keeps every chunk"
    );
}
