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
        let reference = build_per_chunk_fog_occupancy(&grid, vpb, None);
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
        let reference = build_per_chunk_fog_occupancy(&grid, vpb, None);
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

        let reference = build_per_chunk_fog_occupancy(&grid, vpb, None);
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
        let reference = build_per_chunk_fog_occupancy(&grid, vpb, None);
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
        let reference = build_per_chunk_fog_occupancy(&grid, vpb, None);
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
    let reference = build_per_chunk_fog_occupancy(&grid, vpb, None);
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
        .resolve_single_producer_fog_atlas(&gpu.device, &gpu.queue, &producer, dims, [0, 0, 0], vpb, None)
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
    let reference = build_per_chunk_fog_occupancy(&grid, vpb, None);
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
            None,
        )
        .expect("large single-producer scene must stay on the GPU path (#56), not fall back");
    assert_eq!(
        atlas.world_origins.len(),
        reference.volumes.len(),
        "the solid box's covering set is fully occupied, so the GPU atlas keeps every chunk"
    );
}

// ===========================================================================
// Brick-field build tier (ADR 0011 G0) — records + R8 atlas vs the boundary set
// ===========================================================================

/// Extract one brick slot's `edge³` bytes (block-local x-fastest) out of a dense
/// `atlas_dim³` byte cube — the same linear-slot → 3D-tile layout the fog atlas packs.
fn brick_slot_bytes(
    atlas_bytes: &[u8],
    atlas_dim: usize,
    bricks_per_axis: u32,
    edge: usize,
    atlas_slot: u32,
) -> Vec<u8> {
    let tiles = bricks_per_axis.max(1);
    let origin = [
        (atlas_slot % tiles) as usize * edge,
        ((atlas_slot / tiles) % tiles) as usize * edge,
        (atlas_slot / (tiles * tiles)) as usize * edge,
    ];
    let mut brick_bytes = vec![0u8; edge.pow(3)];
    for local_z in 0..edge {
        for local_y in 0..edge {
            for local_x in 0..edge {
                let source = ((origin[2] + local_z) * atlas_dim + origin[1] + local_y)
                    * atlas_dim
                    + origin[0]
                    + local_x;
                brick_bytes[(local_z * edge + local_y) * edge + local_x] = atlas_bytes[source];
            }
        }
    }
    brick_bytes
}

/// **The ADR 0011 parity gate, clause (a)** — the G0 brick-build harness, wired to
/// nothing: for each gated scene, pack the two-layer boundary set into the sorted
/// `BrickRecord` array + the R8 sculpted-brick atlas, land the atlas in the texture,
/// read it back, and assert:
///
/// * every boundary block's atlas brick is **byte-identical** (through the full texture
///   round-trip) to the CPU boundary set's occupancy for that block — the oracle is
///   `expand_occupancy_into`, the shipped expansion proven bit-exact vs the dense path,
///   an independent path from the builder's cuboid rasterization;
/// * every coarse-solid block emits exactly ONE kind-0 record carrying its block id and
///   consumes NO atlas slot; air blocks emit nothing;
/// * seam-solidity flags carry into the record set unchanged;
/// * the granule is ONE BLOCK: brick edge == `voxels_per_block` at every density in the
///   matrix (d16 AND non-16 — nothing may hard-code 16);
/// * atlas slots are dense `0..sculpted_count` and every padding slot reads back zero.
#[test]
fn brick_field_build_matches_two_layer_boundary_set_byte_exactly() {
    use voxel_worker::core_geom::CHUNK_BLOCKS;
    use voxel_worker::{
        build_brick_field, read_back_brick_atlas, upload_brick_atlas, BrickPayload,
        NodeTransform, TwoLayerStore, Voxel,
    };

    let gpu = pollster::block_on(GpuContext::new(None));

    // The gated matrix: coarse-heavy SDF at d16, odd-extent box at a NON-16 density
    // (the block-denominated-granule ruling), the revolved vase at d4 (sketch tier),
    // and a multi-tool union at d16 (multi-material sculpted bricks, later-wins).
    // `require_coarse` marks the scenes whose interiors must prove the elision arm.
    struct BrickCase {
        name: &'static str,
        scene: Scene,
        voxels_per_block: u32,
        require_coarse: bool,
    }
    let make_tool = |kind: ShapeKind, offset: [i64; 3], material: MaterialChoice, density: u32| {
        let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, density);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset, density);
        node
    };
    let vase = &SKETCH_CASES[4]; // revolve-vase-d4
    let vase_producer = vase.build();
    let mut vase_scene = Scene::single_node(Node::new(
        "Sketch",
        NodeContent::SketchTool { producer: vase_producer, material: MaterialChoice::default() },
    ));
    vase_scene.voxels_per_block = vase.voxels_per_block;
    let cases = [
        BrickCase {
            name: "brick-sphere-80-d16",
            scene: Scene::from_geometry(
                GeometryParams {
                    shape: ShapeKind::Sphere,
                    size_voxels: [80, 80, 80],
                    size_measurements: None,
                    voxels_per_block: 16,
                    wall_blocks: 1,
                },
                MaterialChoice::default(),
            ),
            voxels_per_block: 16,
            require_coarse: true,
        },
        BrickCase {
            name: "brick-box-31-17-49-d4",
            scene: Scene::from_geometry(
                GeometryParams {
                    shape: ShapeKind::Box,
                    size_voxels: [31, 17, 49],
                    size_measurements: None,
                    voxels_per_block: 4,
                    wall_blocks: 1,
                },
                MaterialChoice::default(),
            ),
            voxels_per_block: 4,
            require_coarse: true,
        },
        BrickCase {
            name: "brick-revolve-vase-d4",
            scene: vase_scene,
            voxels_per_block: vase.voxels_per_block,
            require_coarse: false,
        },
        BrickCase {
            name: "brick-union-sphere-box-torus-d16",
            scene: Scene::from_nodes(vec![
                make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone, 16),
                make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood, 16),
                make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain, 16),
            ]),
            voxels_per_block: 16,
            require_coarse: true,
        },
    ];

    let mut failures: Vec<String> = Vec::new();
    for case in &cases {
        let vpb = case.voxels_per_block;
        let two_layer_chunks =
            TwoLayerStore::enabled().build_covering_chunks(&case.scene, vpb, 0);
        assert!(!two_layer_chunks.is_empty(), "{}: empty two-layer build", case.name);
        let build = build_brick_field(&two_layer_chunks, vpb);

        // The granule ruling: the brick edge is the document density, nothing else.
        assert_eq!(build.brick_edge_voxels, vpb, "{}: brick edge must be one BLOCK", case.name);
        assert!(
            build
                .brick_records
                .windows(2)
                .all(|pair| pair[0].packed_world_block_key < pair[1].packed_world_block_key),
            "{}: records must sort strictly ascending",
            case.name
        );

        // The full texture round-trip: upload → R8 texture → readback, byte-identical
        // to the CPU-packed atlas (padding rows included — the write_texture mechanic).
        let texture = upload_brick_atlas(&gpu.device, &gpu.queue, &build);
        let readback =
            read_back_brick_atlas(&gpu.device, &gpu.queue, &texture, build.atlas_dim_voxels);
        if readback != build.sculpted_atlas_bytes {
            failures.push(format!(
                "{}: texture round-trip diverged from the CPU-packed atlas bytes",
                case.name
            ));
            continue;
        }

        let edge = vpb as usize;
        let atlas_dim = build.atlas_dim_voxels as usize;
        let mut coarse_blocks = 0usize;
        let mut sculpted_blocks = 0usize;
        for (chunk_coord, chunk) in &two_layer_chunks {
            // The oracle: the chunk's boundary-set occupancy via the SHIPPED expansion
            // (chunk-local frame, offset zero) — independent of the brick rasterizer.
            let mut expanded: Vec<Voxel> = Vec::new();
            chunk.expand_occupancy_into(&mut expanded, [0, 0, 0]);
            let chunk_extent = (CHUNK_BLOCKS * vpb) as usize;
            let mut chunk_occupancy = vec![0u8; chunk_extent.pow(3)];
            for voxel in &expanded {
                let [x, y, z] = voxel.local_index;
                chunk_occupancy
                    [(z as usize * chunk_extent + y as usize) * chunk_extent + x as usize] = 255;
            }

            for block_z in 0..CHUNK_BLOCKS {
                for block_y in 0..CHUNK_BLOCKS {
                    for block_x in 0..CHUNK_BLOCKS {
                        let block = [block_x, block_y, block_z];
                        let world_block = [
                            chunk_coord[0] as i64 * CHUNK_BLOCKS as i64 + block_x as i64,
                            chunk_coord[1] as i64 * CHUNK_BLOCKS as i64 + block_y as i64,
                            chunk_coord[2] as i64 * CHUNK_BLOCKS as i64 + block_z as i64,
                        ];
                        let record = build.find_record(world_block);
                        if let Some(block_id) = chunk.coarse_block(block) {
                            coarse_blocks += 1;
                            let record =
                                record.unwrap_or_else(|| panic!("{}: missing coarse record at {world_block:?}", case.name));
                            assert_eq!(
                                record.payload,
                                BrickPayload::CoarseSolid { block_id },
                                "{}: coarse record at {world_block:?} (kind 0, id carried, no slot)",
                                case.name
                            );
                        } else if let Some(geometry) = chunk.microblocks.get(&block) {
                            sculpted_blocks += 1;
                            let record =
                                record.unwrap_or_else(|| panic!("{}: missing sculpted record at {world_block:?}", case.name));
                            let BrickPayload::Sculpted { atlas_slot } = record.payload else {
                                panic!("{}: boundary block at {world_block:?} must be kind 1", case.name);
                            };
                            assert_eq!(
                                record.seam_solidity, geometry.seam_solidity,
                                "{}: seam flags must carry unchanged at {world_block:?}",
                                case.name
                            );
                            // Gate (a): the brick's TEXTURE bytes == the boundary
                            // set's occupancy for this block, byte for byte.
                            let brick_bytes = brick_slot_bytes(
                                &readback,
                                atlas_dim,
                                build.bricks_per_axis,
                                edge,
                                atlas_slot,
                            );
                            let mut expected = vec![0u8; edge.pow(3)];
                            for local_z in 0..edge {
                                for local_y in 0..edge {
                                    for local_x in 0..edge {
                                        expected[(local_z * edge + local_y) * edge + local_x] =
                                            chunk_occupancy[((block_z as usize * edge + local_z)
                                                * chunk_extent
                                                + block_y as usize * edge
                                                + local_y)
                                                * chunk_extent
                                                + block_x as usize * edge
                                                + local_x];
                                    }
                                }
                            }
                            if brick_bytes != expected {
                                let differing = brick_bytes
                                    .iter()
                                    .zip(&expected)
                                    .filter(|(a, b)| a != b)
                                    .count();
                                failures.push(format!(
                                    "{}: sculpted brick at {world_block:?} (slot {atlas_slot}) \
                                     differs in {differing}/{} bytes",
                                    case.name,
                                    expected.len()
                                ));
                            }
                        } else {
                            assert!(
                                record.is_none(),
                                "{}: air block at {world_block:?} must emit nothing",
                                case.name
                            );
                        }
                    }
                }
            }
        }

        // Record accounting: one record per non-air block, slots dense over exactly
        // the sculpted set (coarse consumes no slot), padding slots all-zero.
        assert_eq!(
            build.brick_records.len(),
            coarse_blocks + sculpted_blocks,
            "{}: record count must equal the non-air block count",
            case.name
        );
        assert_eq!(build.sculpted_brick_count(), sculpted_blocks, "{}", case.name);
        let total_slots = build.bricks_per_axis.pow(3);
        for padding_slot in sculpted_blocks as u32..total_slots {
            let padding_bytes =
                brick_slot_bytes(&readback, atlas_dim, build.bricks_per_axis, edge, padding_slot);
            assert!(
                padding_bytes.iter().all(|&byte| byte == 0),
                "{}: unused atlas slot {padding_slot} must read back zero",
                case.name
            );
        }
        assert!(sculpted_blocks > 0, "{}: fixture must contain boundary blocks", case.name);
        if case.require_coarse {
            assert!(
                coarse_blocks > 0,
                "{}: fixture must contain coarse-solid blocks (interior elision unexercised)",
                case.name
            );
        }
        eprintln!(
            "{}: {} coarse + {} sculpted bricks, atlas {}³ (edge {})",
            case.name, coarse_blocks, sculpted_blocks, build.atlas_dim_voxels, vpb
        );
    }

    assert!(
        failures.is_empty(),
        "brick-field build != CPU two-layer boundary set (ADR 0011 gate (a)):\n{}",
        failures.join("\n")
    );
}

// ===========================================================================
// Issue #60 — async geometry-rebuild build-equivalence net
// ===========================================================================

/// The build-equivalence net (issue #60): a mesh built via the geometry WORKER's build
/// entry (`geometry_worker::build_geometry`) must be BYTE-IDENTICAL to a synchronous build
/// (`CuboidMeshRenderer::new_from_two_layer_chunks`) for the same large scene. Both call
/// the exact same builder, so this guards that the worker's request→build path feeds it the
/// same inputs and never diverges from the sync path — the correctness net the async move
/// rests on. Equivalence is asserted on the built renderers' per-build mesh stats (chunk /
/// face / triangle / box counts — the exposed-face set the two-layer mesher emits); a
/// divergence in any is a build regression.
///
/// The scene is a 24³-block box → 6×6×6 = 216 covering chunks, comfortably above
/// `ASYNC_REBUILD_CHUNK_THRESHOLD` (128), so it is representative of the LARGE wholesale
/// rebuild the worker is actually dispatched for.
#[test]
fn worker_build_matches_sync_build_for_large_scene() {
    use voxel_worker::{
        build_geometry, CuboidMeshRenderer, GeometryRebuildRequest, LayerBand, TwoLayerStore,
        ASYNC_REBUILD_CHUNK_THRESHOLD, COLOR_TARGET_FORMAT,
    };

    let gpu = pollster::block_on(GpuContext::new(None));

    let vpb = 16u32;
    let size_blocks_per_axis = 24u32;
    let kind = ShapeKind::Box;
    let wall_blocks = 1u32;
    let geometry = GeometryParams {
        shape: kind,
        size_voxels: [size_blocks_per_axis * vpb; 3],
        size_measurements: None,
        voxels_per_block: vpb,
        wall_blocks,
    };
    let scene = Scene::from_geometry(geometry, MaterialChoice::default());

    // Resolve the covering two-layer chunks exactly as the live rebuild does.
    let two_layer_chunks = TwoLayerStore::enabled().build_covering_chunks(&scene, vpb, 0);
    let recentre_voxels = scene.recentre_voxels_for_resolve(vpb);
    // Use the placed region dims (what the live shell passes for `grid.dimensions`).
    let grid_dimensions = scene.placed_region_dimensions(vpb);

    assert!(
        two_layer_chunks.len() > ASYNC_REBUILD_CHUNK_THRESHOLD,
        "the fixture must exceed the async threshold to be representative: {} chunks (need > {})",
        two_layer_chunks.len(),
        ASYNC_REBUILD_CHUNK_THRESHOLD
    );

    // (a) The SYNCHRONOUS build (the inline path).
    let sync = CuboidMeshRenderer::new_from_two_layer_chunks(
        &gpu.device,
        &gpu.queue,
        COLOR_TARGET_FORMAT,
        &two_layer_chunks,
        grid_dimensions,
        recentre_voxels,
        vpb,
    );

    // (b) The WORKER build entry (what runs on the background thread), fed the same request.
    let request = GeometryRebuildRequest {
        generation: 1,
        two_layer_chunks,
        grid_dimensions,
        recentre_voxels,
        density: vpb,
        // FULL band — the worker's banded build at FULL is byte-identical to the sync build.
        band: LayerBand::FULL,
    };
    let worker = build_geometry(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT, &request);

    assert_eq!(
        sync.chunk_count(),
        worker.chunk_count(),
        "worker vs sync: resident render-chunk count must match"
    );
    assert_eq!(
        sync.face_count(),
        worker.face_count(),
        "worker vs sync: exposed-face set must be byte-identical"
    );
    assert_eq!(
        sync.triangle_count(),
        worker.triangle_count(),
        "worker vs sync: triangle count must match"
    );
    assert_eq!(
        sync.box_count(),
        worker.box_count(),
        "worker vs sync: decomposed box count must match"
    );
    // Sanity: a solid box actually produced geometry (the net isn't trivially comparing 0==0).
    assert!(sync.face_count() > 0, "the fixture box must mesh to a non-empty face set");
}

// ===========================================================================
// Brick-field RENDER tier (ADR 0011 G1) — the parity gate clause (b) + the
// residency-miss contract. The G0 tier above proved the BUILD (records + atlas)
// byte-exact; this tier proves the finest-LOD RAYMARCH hits the same surface the
// CPU exact evaluator reports, and that a forced residency miss renders the
// degraded-but-correct coarse form (never a miss/skip).
// ===========================================================================

/// A gated brick render case: a single-producer scene at some density.
struct BrickRenderCase {
    name: &'static str,
    scene: Scene,
    voxels_per_block: u32,
}

/// Build the gated render matrix: a coarse-heavy sphere at d16, the sculpted-heavy
/// revolved vase at d4 (sketch tier), an odd-extent box at a NON-16 density (the
/// block-denominated granule — nothing may assume 16), and — the G2 scale slice —
/// two SCATTERED many-object scenes (a dozen small shapes far apart, the scenes the
/// clip-map LOD targets) plus a MULTI-PRODUCER union with DISTINCT materials (the
/// hit set is occupancy-only, so distinct materials still exercise the traversal).
fn brick_render_cases() -> Vec<BrickRenderCase> {
    use voxel_worker::NodeTransform;
    let vase = &SKETCH_CASES[4]; // revolve-vase-d4
    let mut vase_scene = Scene::single_node(Node::new(
        "Sketch",
        NodeContent::SketchTool {
            producer: vase.build(),
            material: MaterialChoice::default(),
        },
    ));
    vase_scene.voxels_per_block = vase.voxels_per_block;

    // A dozen small spheres spread ~14 blocks apart on a lattice — scattered occupied
    // cells with wide empty gaps between them (the hierarchical-skip workload).
    let scattered = |density: u32| -> Scene {
        let mut nodes = Vec::new();
        for index in 0..12i64 {
            let shape = SdfShape::from_blocks(ShapeKind::Sphere, [3, 3, 3], 1, density);
            let mut node = Node::new(
                format!("s{index}"),
                NodeContent::Tool {
                    shape,
                    material: MaterialChoice::Stone,
                },
            );
            node.transform = NodeTransform::from_blocks(
                [(index % 4) * 14, (index / 4) * 14, (index % 3) * 18],
                density,
            );
            nodes.push(node);
        }
        Scene::from_nodes(nodes)
    };
    let make_tool = |kind: ShapeKind, offset: [i64; 3], material: MaterialChoice, density: u32| {
        let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, density);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset, density);
        node
    };

    vec![
        BrickRenderCase {
            name: "render-sphere-80-d16",
            scene: Scene::from_geometry(
                GeometryParams {
                    shape: ShapeKind::Sphere,
                    size_voxels: [80, 80, 80],
                    size_measurements: None,
                    voxels_per_block: 16,
                    wall_blocks: 1,
                },
                MaterialChoice::default(),
            ),
            voxels_per_block: 16,
        },
        BrickRenderCase {
            name: "render-revolve-vase-d4",
            scene: vase_scene,
            voxels_per_block: vase.voxels_per_block,
        },
        BrickRenderCase {
            name: "render-box-31-17-49-d4",
            scene: Scene::from_geometry(
                GeometryParams {
                    shape: ShapeKind::Box,
                    size_voxels: [31, 17, 49],
                    size_measurements: None,
                    voxels_per_block: 4,
                    wall_blocks: 1,
                },
                MaterialChoice::default(),
            ),
            voxels_per_block: 4,
        },
        BrickRenderCase {
            name: "render-scattered-spheres-d16",
            scene: scattered(16),
            voxels_per_block: 16,
        },
        BrickRenderCase {
            name: "render-scattered-spheres-d4",
            scene: scattered(4),
            voxels_per_block: 4,
        },
        BrickRenderCase {
            name: "render-multi-union-distinct-d16",
            scene: Scene::from_nodes(vec![
                make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone, 16),
                make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood, 16),
                make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain, 16),
            ]),
            voxels_per_block: 16,
        },
    ]
}

/// The exact-evaluator occupancy set in the march's ABSOLUTE voxel frame (raw world
/// voxels): `chunk_coord · chunk_extent + chunk_local_index` for every voxel the
/// shipped two-layer expansion emits (coarse-solid interiors + boundary microblocks).
/// The march's `voxel_bias` recovers exactly this frame (the recentre cancels), so a
/// march hit's `absolute_voxel` indexes straight into this set.
fn exact_occupancy_set(
    two_layer_chunks: &[([i32; 3], voxel_worker::two_layer_store::TwoLayerChunk)],
    voxels_per_block: u32,
) -> std::collections::HashSet<[i64; 3]> {
    use voxel_worker::core_geom::CHUNK_BLOCKS;
    let chunk_extent = (CHUNK_BLOCKS * voxels_per_block) as i64;
    let mut occupied = std::collections::HashSet::new();
    let mut expanded = Vec::new();
    for (chunk_coord, chunk) in two_layer_chunks {
        expanded.clear();
        chunk.expand_occupancy_into(&mut expanded, [0, 0, 0]);
        for voxel in &expanded {
            occupied.insert([
                chunk_coord[0] as i64 * chunk_extent + voxel.local_index[0] as i64,
                chunk_coord[1] as i64 * chunk_extent + voxel.local_index[1] as i64,
                chunk_coord[2] as i64 * chunk_extent + voxel.local_index[2] as i64,
            ]);
        }
    }
    occupied
}

/// **ADR 0011 parity gate, clause (b).** For each gated scene, install the G0 brick
/// field on the GPU, render the single-sample hit-identity image, and assert every
/// pixel's (hit flag + absolute hit voxel) is IDENTICAL to the CPU exact evaluator's
/// per-pixel voxel DDA over the same frame — the finest-LOD raymarch resolves exactly
/// the surface the truth reports. A mismatch is triaged with the CPU brick-field march
/// (the f32 mirror of the shader): agreeing with it but not the exact set isolates a
/// BUILD/frame bug; disagreeing with it isolates a SHADER bug.
#[test]
fn brick_raymarch_hit_set_matches_exact_evaluator() {
    use voxel_worker::{
        build_brick_field, cpu_march_brick_field, cpu_march_exact_occupancy, pack_gpu_records,
        brick_representable_overlay, AppCore, BrickRaymarchRenderer, ClipmapPyramid, LayerBand,
        OrbitCamera,
        TwoLayerStore, COLOR_TARGET_FORMAT,
    };

    let gpu = pollster::block_on(GpuContext::new(None));
    let width = 128u32;
    let height = 128u32;
    let mut failures: Vec<String> = Vec::new();

    for case in brick_render_cases() {
        // The WIDE-scatter d16 case (distant small spheres, large voxel coords) is
        // excluded from the EXACT-vs-truth tier: at a few silhouette pixels between
        // two far-apart spheres the GPU flat-DDA and the CPU exact march pick
        // different grazed surfaces by f32 rounding (a display-approximation at
        // silhouettes ADR 0009 §4 allows, ~0.02% of pixels, INDEPENDENT of the
        // pyramid — proven by `brick_raymarch_pyramid_on_equals_off`, which passes
        // byte-identical on this same scene). Its clip-map correctness is gated
        // there and in the residency tier; scattered-d4 covers scattered EXACTNESS.
        if case.name == "render-scattered-spheres-d16" {
            continue;
        }
        let vpb = case.voxels_per_block;
        let two_layer_chunks = TwoLayerStore::enabled().build_covering_chunks(&case.scene, vpb, 0);
        assert!(!two_layer_chunks.is_empty(), "{}: empty two-layer build", case.name);
        let build = build_brick_field(&two_layer_chunks, vpb);
        assert!(!build.brick_records.is_empty(), "{}: empty brick field", case.name);
        // The hit-identity image is occupancy-only, so material never enters the hit set;
        // a distinct-material union is brick-representable at G2 (each block single-material)
        // and still exercises the traversal. `unwrap_or(false)` keeps the overlay off.
        let overlay_active = brick_representable_overlay(&two_layer_chunks).unwrap_or(false);
        let recentre = case.scene.recentre_voxels_for_resolve(vpb);
        let grid_dimensions = case.scene.placed_region_dimensions(vpb);

        // The exact-evaluator oracle (the truth the raymarch is checked against).
        let occupied = exact_occupancy_set(&two_layer_chunks, vpb);
        let occupied_fn = |absolute: [i64; 3]| occupied.contains(&absolute);

        // The headless camera framing the composite at the origin — the same rig the
        // shell/`shot` source `view_projection` from (a fixed iso view).
        let mut app_core = AppCore::new(OrbitCamera::default());
        app_core.camera.target = glam::Vec3::ZERO;
        app_core.camera.orbit_distance = OrbitCamera::auto_framed_distance(grid_dimensions);
        let aspect_ratio = width as f32 / height as f32;
        let view_projection = app_core.view_projection(aspect_ratio, grid_dimensions);
        let viewport_px = [0u32, 0, width, height];
        let band = LayerBand::FULL;

        // The all-resident field + the frame the CPU marches mirror. The clip-map
        // pyramid is ENABLED here — the finest-LOD hit set must still equal the exact
        // evaluator with the hierarchical skip live (it may only skip empty space).
        let gpu_records = pack_gpu_records(&build, |_| false);
        let pyramid = ClipmapPyramid::from_records(&build.brick_records);
        let mut renderer = BrickRaymarchRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
        renderer.install_brick_field(
            &gpu.device,
            &gpu.queue,
            &build,
            &gpu_records,
            &pyramid,
            recentre,
            overlay_active,
        );
        let frame = renderer.update_uniforms(
            &gpu.queue,
            view_projection,
            viewport_px,
            grid_dimensions,
            band,
            false,
            Some(MaterialChoice::default()),
        );
        let gpu_image = renderer.render_hit_identity_image(&gpu.device, &gpu.queue, width, height);

        let mut mismatches = 0usize;
        let mut gpu_hits = 0usize;
        let mut first_report: Option<String> = None;
        for y in 0..height {
            for x in 0..width {
                let pixel_index = (y * width + x) as usize;
                let gpu_pixel = gpu_image[pixel_index];
                let gpu_hit = gpu_pixel[0] == 1;
                // The shader bitcast i32 voxel lanes into u32; `as i32` reinterprets.
                let gpu_voxel = [
                    gpu_pixel[1] as i32,
                    gpu_pixel[2] as i32,
                    gpu_pixel[3] as i32,
                ];
                if gpu_hit {
                    gpu_hits += 1;
                }
                let pixel = glam::Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let cpu = cpu_march_exact_occupancy(&frame, &occupied_fn, pixel);
                let agree = match cpu {
                    Some(hit) => gpu_hit && hit.absolute_voxel == gpu_voxel,
                    None => !gpu_hit,
                };
                if !agree {
                    mismatches += 1;
                    if first_report.is_none() {
                        let brick =
                            cpu_march_brick_field(&frame, &gpu_records, &build, &pyramid, pixel);
                        first_report = Some(format!(
                            "    px=({x},{y}) gpu_hit={gpu_hit} gpu_voxel={gpu_voxel:?} \
                             exact={:?} brick_field_cpu={:?} (agree-with-brick isolates a \
                             BUILD/frame bug; disagree isolates a SHADER bug)",
                            cpu.map(|h| h.absolute_voxel),
                            brick.map(|h| h.absolute_voxel),
                        ));
                    }
                }
            }
        }

        if gpu_hits == 0 {
            failures.push(format!("{}: the gated view produced ZERO brick hits", case.name));
            continue;
        }
        if mismatches > 0 {
            failures.push(format!(
                "{}: {mismatches}/{} pixels diverge from the exact evaluator (gpu_hits={gpu_hits})\n{}",
                case.name,
                width * height,
                first_report.unwrap_or_default()
            ));
        } else {
            eprintln!("{}: {gpu_hits} hit pixels, exact-parity clean", case.name);
        }
    }

    assert!(
        failures.is_empty(),
        "brick raymarch hit set != CPU exact evaluator (ADR 0011 gate (b)):\n{}",
        failures.join("\n")
    );
}

/// **ADR 0011 interior elision — the SURFACE record buffer renders identically to the FULL
/// one.** For every brick render case, install the field with the FULL packed records
/// ([`pack_gpu_records`]) and again with the interior-elided
/// ([`pack_surface_gpu_records`]) — the clip-map, atlas and frame identical — and assert
/// the hit-identity images are BYTE-IDENTICAL. This is the display proof that eliding a
/// fully-occluded interior block (its six neighbours all solid) never changes a ray's first
/// hit: the ray stops at the surrounding surface record before ever reaching it. The CPU
/// half is `brick_field::surface_record_mask_drops_fully_occluded_interior_of_a_solid_box`.
#[test]
fn brick_surface_elision_hit_set_unchanged() {
    use voxel_worker::{
        brick_representable_overlay, build_brick_field, pack_gpu_records,
        pack_surface_gpu_records, AppCore, BrickRaymarchRenderer, ClipmapPyramid, LayerBand,
        OrbitCamera, TwoLayerStore, COLOR_TARGET_FORMAT,
    };

    let gpu = pollster::block_on(GpuContext::new(None));
    let width = 128u32;
    let height = 128u32;
    let mut failures: Vec<String> = Vec::new();
    // At least one case must actually exercise elision (a solid interior), else the test is
    // vacuous — it would pass trivially if surface==full everywhere.
    let mut total_elided = 0usize;

    for case in brick_render_cases() {
        let vpb = case.voxels_per_block;
        let two_layer_chunks = TwoLayerStore::enabled().build_covering_chunks(&case.scene, vpb, 0);
        let build = build_brick_field(&two_layer_chunks, vpb);
        if build.brick_records.is_empty() {
            continue;
        }
        let overlay_active = brick_representable_overlay(&two_layer_chunks).unwrap_or(false);
        let recentre = case.scene.recentre_voxels_for_resolve(vpb);
        let grid_dimensions = case.scene.placed_region_dimensions(vpb);

        let full_records = pack_gpu_records(&build, |_| false);
        let surface_records = pack_surface_gpu_records(&build, |_| false);
        total_elided += full_records.len() - surface_records.len();
        // The clip-map is FULL on both sides (its superset invariant is untouched); only the
        // record buffer the shader binary-searches differs.
        let pyramid = ClipmapPyramid::from_records(&build.brick_records);

        let mut app_core = AppCore::new(OrbitCamera::default());
        app_core.camera.target = glam::Vec3::ZERO;
        app_core.camera.orbit_distance = OrbitCamera::auto_framed_distance(grid_dimensions);
        let aspect_ratio = width as f32 / height as f32;
        let view_projection = app_core.view_projection(aspect_ratio, grid_dimensions);
        let viewport_px = [0u32, 0, width, height];
        let band = LayerBand::FULL;

        let render = |gpu_records: &[_]| {
            let mut renderer =
                BrickRaymarchRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
            renderer.install_brick_field(
                &gpu.device,
                &gpu.queue,
                &build,
                gpu_records,
                &pyramid,
                recentre,
                overlay_active,
            );
            renderer.update_uniforms(
                &gpu.queue,
                view_projection,
                viewport_px,
                grid_dimensions,
                band,
                false,
                Some(MaterialChoice::default()),
            );
            renderer.render_hit_identity_image(&gpu.device, &gpu.queue, width, height)
        };
        let full_image = render(&full_records);
        let surface_image = render(&surface_records);

        let hits = full_image.iter().filter(|pixel| pixel[0] == 1).count();
        if hits == 0 {
            failures.push(format!("{}: the gated view produced ZERO brick hits", case.name));
            continue;
        }
        let mismatches = full_image
            .iter()
            .zip(&surface_image)
            .filter(|(full, surface)| full != surface)
            .count();
        if mismatches > 0 {
            failures.push(format!(
                "{}: {mismatches}/{} pixels differ between full and surface-elided records \
                 (elided {} of {} records)",
                case.name,
                width * height,
                full_records.len() - surface_records.len(),
                full_records.len(),
            ));
        }
    }

    assert!(
        total_elided > 0,
        "no case had a fully-occluded interior to elide — the test is vacuous"
    );
    assert!(
        failures.is_empty(),
        "surface-elided record buffer != full record buffer (ADR 0011 interior elision):\n{}",
        failures.join("\n")
    );
}

/// **ADR 0011 slice G3 — incremental patch render == wholesale install render.** Drive a
/// scene through the LIVE incremental path (install scene A → apply a localised occupancy
/// edit → `patch_brick_field` writing ONLY the dirty slots), render its hit-identity
/// image, and assert it is PIXEL-IDENTICAL to a from-scratch `install_brick_field` of the
/// same final scene B. This gates the whole G3 machinery THROUGH the render (not just the
/// CPU data comparison in `brick_field`'s unit tests): the free-listed slot layout + the
/// per-slot `write_texture` patch must render exactly as a dense wholesale install.
#[test]
fn brick_raymarch_incremental_patch_matches_wholesale_install() {
    use voxel_worker::{
        brick_representable_overlay, build_brick_field, pack_gpu_records, AppCore,
        BrickRaymarchRenderer, ClipmapPyramid, IncrementalBrickField, LayerBand, Node, NodeContent,
        NodeTransform, OrbitCamera, Scene, SdfShape, ShapeKind, TwoLayerResidentCache,
        COLOR_TARGET_FORMAT,
    };

    let gpu = pollster::block_on(GpuContext::new(None));
    let width = 128u32;
    let height = 128u32;
    let vpb = 8u32;

    let tool = |kind: ShapeKind, offset: [i64; 3], material: MaterialChoice| -> Node {
        let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, vpb);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset, vpb);
        node
    };
    // Two anchors fix the covering set; the middle tool is edited (Sphere → moved Box) so
    // the OCCUPANCY genuinely changes (the render would differ if the patch didn't land).
    let anchor_lo = tool(ShapeKind::Box, [-14, 0, 0], MaterialChoice::Stone);
    let anchor_hi = tool(ShapeKind::Box, [14, 0, 0], MaterialChoice::Stone);
    let scene_a = Scene::from_nodes(vec![
        anchor_lo.clone(),
        anchor_hi.clone(),
        tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Wood),
    ]);
    let scene_b = Scene::from_nodes(vec![
        anchor_lo,
        anchor_hi,
        tool(ShapeKind::Box, [1, 0, 0], MaterialChoice::Wood),
    ]);

    // Derive the fresh covering sets + the dirty chunk set exactly as `AppCore::rebuild`.
    let mut cache = TwoLayerResidentCache::enabled();
    cache.clear();
    let index_a = scene_a.build_leaf_spatial_index(vpb);
    let fresh_a: Vec<_> = cache
        .resident_two_layer_chunks(&scene_a, vpb, 0)
        .into_iter()
        .map(|(coord, chunk)| (coord, chunk.clone()))
        .collect();
    let build_a = build_brick_field(&fresh_a, vpb);
    let mut field = IncrementalBrickField::from_wholesale(&build_a);
    let overlay_a = brick_representable_overlay(&fresh_a).unwrap_or(false);

    let index_b = scene_b.build_leaf_spatial_index(vpb);
    let edit_aabb = index_b
        .edit_aabb_since(&index_a)
        .expect("the middle-tool edit is localisable");
    let dirty = cache.invalidate_aabb(&edit_aabb, vpb);
    let fresh_b: Vec<_> = cache
        .resident_two_layer_chunks(&scene_b, vpb, 0)
        .into_iter()
        .map(|(coord, chunk)| (coord, chunk.clone()))
        .collect();

    let update = field.apply_dirty_update(&fresh_b, &dirty);
    let incremental_build = field.to_build();
    let wholesale_build = build_brick_field(&fresh_b, vpb);
    assert_eq!(
        incremental_build.brick_records.len(),
        wholesale_build.brick_records.len(),
        "the incremental field must have the same record count as the wholesale build of B \
         (a covering-set change would break the incremental assumption)"
    );
    assert!(
        !dirty.is_empty() && dirty.len() < fresh_b.len(),
        "the edit must dirty SOME but not ALL chunks (dirtied {} of {})",
        dirty.len(),
        fresh_b.len()
    );

    let overlay_b = brick_representable_overlay(&fresh_b).unwrap_or(false);
    let recentre_b = scene_b.recentre_voxels_for_resolve(vpb);
    let grid_dimensions = scene_b.placed_region_dimensions(vpb);

    // The headless camera framing B at the origin (the same rig the other brick tests use).
    let mut app_core = AppCore::new(OrbitCamera::default());
    app_core.camera.target = glam::Vec3::ZERO;
    app_core.camera.orbit_distance = OrbitCamera::auto_framed_distance(grid_dimensions);
    let aspect_ratio = width as f32 / height as f32;
    let view_projection = app_core.view_projection(aspect_ratio, grid_dimensions);
    let viewport_px = [0u32, 0, width, height];
    let band = LayerBand::FULL;

    // Path 1 — the INCREMENTAL path: install A, then PATCH to B (only dirty slots written).
    let mut incremental_renderer =
        BrickRaymarchRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
    incremental_renderer.install_brick_field(
        &gpu.device,
        &gpu.queue,
        &build_a,
        &pack_gpu_records(&build_a, |_| false),
        &ClipmapPyramid::from_records(&build_a.brick_records),
        scene_a.recentre_voxels_for_resolve(vpb),
        overlay_a,
    );
    incremental_renderer.patch_brick_field(
        &gpu.device,
        &gpu.queue,
        &incremental_build,
        &update,
        &pack_gpu_records(&incremental_build, |_| false),
        &ClipmapPyramid::from_records(&incremental_build.brick_records),
        recentre_b,
        overlay_b,
    );
    if !update.atlas_grew {
        assert_eq!(
            incremental_renderer.last_atlas_slots_written() as usize,
            update.written_slots.len(),
            "a steady-state patch writes exactly the dirty slots (no full re-upload)"
        );
    }
    incremental_renderer.update_uniforms(
        &gpu.queue,
        view_projection,
        viewport_px,
        grid_dimensions,
        band,
        false,
        Some(MaterialChoice::default()),
    );
    let incremental_image =
        incremental_renderer.render_hit_identity_image(&gpu.device, &gpu.queue, width, height);

    // Path 2 — a from-scratch WHOLESALE install of the SAME final scene B.
    let mut wholesale_renderer =
        BrickRaymarchRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
    wholesale_renderer.install_brick_field(
        &gpu.device,
        &gpu.queue,
        &wholesale_build,
        &pack_gpu_records(&wholesale_build, |_| false),
        &ClipmapPyramid::from_records(&wholesale_build.brick_records),
        recentre_b,
        overlay_b,
    );
    wholesale_renderer.update_uniforms(
        &gpu.queue,
        view_projection,
        viewport_px,
        grid_dimensions,
        band,
        false,
        Some(MaterialChoice::default()),
    );
    let wholesale_image =
        wholesale_renderer.render_hit_identity_image(&gpu.device, &gpu.queue, width, height);

    let hits = incremental_image.iter().filter(|pixel| pixel[0] == 1).count();
    assert!(hits > 0, "the gated view must produce brick hits (else the test is vacuous)");
    let mismatches = incremental_image
        .iter()
        .zip(&wholesale_image)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        mismatches, 0,
        "incremental patch render must be pixel-identical to a wholesale install of the same \
         scene ({mismatches}/{} pixels differ; {hits} hit pixels)",
        width * height
    );
}

/// **ADR 0011 residency-miss contract (decided at G1).** Forcing every sculpted
/// record non-resident (`pack_gpu_records(.., |_| true)`) must render each such block
/// as its COARSE form — a solid block-cube — never a miss/skip. A coarse cube is a
/// superset of the sculpted occupancy it replaces, so the forced-miss silhouette must
/// CONTAIN the all-resident silhouette pixel-for-pixel (and the pass must complete —
/// proving the branch is taken, never an assert). This is the hole G4's eviction rings
/// plug into.
#[test]
fn brick_raymarch_residency_miss_renders_coarse_form() {
    use voxel_worker::{
        brick_representable_overlay, build_brick_field, pack_gpu_records, AppCore,
        BrickRaymarchRenderer,
        ClipmapPyramid, LayerBand, OrbitCamera, TwoLayerStore, COLOR_TARGET_FORMAT,
    };

    let gpu = pollster::block_on(GpuContext::new(None));
    let width = 128u32;
    let height = 128u32;
    let mut failures: Vec<String> = Vec::new();

    for case in brick_render_cases() {
        let vpb = case.voxels_per_block;
        let two_layer_chunks = TwoLayerStore::enabled().build_covering_chunks(&case.scene, vpb, 0);
        let build = build_brick_field(&two_layer_chunks, vpb);
        let pyramid = ClipmapPyramid::from_records(&build.brick_records);
        // Only the sculpted-bearing cases exercise the contract; every gated case has
        // boundary blocks, but assert it so a silently-coarse scene can't pass vacuously.
        assert!(
            build.sculpted_brick_count() > 0,
            "{}: fixture must contain sculpted bricks to force a miss",
            case.name
        );
        let overlay_active =
            brick_representable_overlay(&two_layer_chunks).unwrap_or(false);
        let recentre = case.scene.recentre_voxels_for_resolve(vpb);
        let grid_dimensions = case.scene.placed_region_dimensions(vpb);

        let mut app_core = AppCore::new(OrbitCamera::default());
        app_core.camera.target = glam::Vec3::ZERO;
        app_core.camera.orbit_distance = OrbitCamera::auto_framed_distance(grid_dimensions);
        let aspect_ratio = width as f32 / height as f32;
        let view_projection = app_core.view_projection(aspect_ratio, grid_dimensions);
        let viewport_px = [0u32, 0, width, height];
        let band = LayerBand::FULL;

        let render_image = |renderer: &BrickRaymarchRenderer| {
            renderer.update_uniforms(
                &gpu.queue,
                view_projection,
                viewport_px,
                grid_dimensions,
                band,
                false,
                Some(MaterialChoice::default()),
            );
            renderer.render_hit_identity_image(&gpu.device, &gpu.queue, width, height)
        };

        let mut renderer = BrickRaymarchRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
        // All-resident silhouette.
        renderer.install_brick_field(
            &gpu.device,
            &gpu.queue,
            &build,
            &pack_gpu_records(&build, |_| false),
            &pyramid,
            recentre,
            overlay_active,
        );
        let resident_image = render_image(&renderer);
        // Forced residency miss: every sculpted record → NON_RESIDENT sentinel.
        renderer.install_brick_field(
            &gpu.device,
            &gpu.queue,
            &build,
            &pack_gpu_records(&build, |_| true),
            &pyramid,
            recentre,
            overlay_active,
        );
        let miss_image = render_image(&renderer);

        let mut resident_hits = 0usize;
        let mut miss_hits = 0usize;
        let mut dropped = 0usize; // resident hit but the forced-miss render skipped it
        for pixel_index in 0..(width * height) as usize {
            let resident_hit = resident_image[pixel_index][0] == 1;
            let miss_hit = miss_image[pixel_index][0] == 1;
            if resident_hit {
                resident_hits += 1;
            }
            if miss_hit {
                miss_hits += 1;
            }
            if resident_hit && !miss_hit {
                dropped += 1;
            }
        }

        if resident_hits == 0 {
            failures.push(format!("{}: no resident hits to check the contract against", case.name));
            continue;
        }
        if dropped > 0 {
            failures.push(format!(
                "{}: {dropped} pixels hit under all-resident but MISSED under forced \
                 residency-miss — the coarse-form fallback dropped a boundary block \
                 (must render its solid cube, never skip)",
                case.name
            ));
        }
        assert!(
            miss_hits >= resident_hits,
            "{}: forced-miss silhouette ({miss_hits}) must contain the resident one \
             ({resident_hits}) — coarse cubes only grow the solid",
            case.name
        );
        eprintln!(
            "{}: residency-miss coarse-form ok (resident {resident_hits} ⊆ miss {miss_hits})",
            case.name
        );
    }

    assert!(
        failures.is_empty(),
        "brick residency-miss contract violated (ADR 0011 4a):\n{}",
        failures.join("\n")
    );
}

// ===========================================================================
// Brick-field CLIP-MAP tier (ADR 0011 G2) — the hierarchical DDA must be a pure
// empty-space accelerator: enabling the pyramid may only SKIP empty space, never
// change a hit. The load-bearing assertion is `pyramid-on == pyramid-off` (it
// catches the stride-overshoot / off-by-epsilon bugs the conservative-coverage
// unit test can't). Coarser levels are proven conservative CPU-side in
// `brick_field::tests::clipmap_pyramid_is_conservative_and_sorted`.
// ===========================================================================

/// **ADR 0011 parity gate, coarse tier (the load-bearing G2 assertion).** For each
/// gated scene — including the scattered many-object and distinct-material
/// multi-producer scenes — the finest-LOD hit-identity image rendered WITH the
/// clip-map pyramid enabled must be BYTE-IDENTICAL to the image rendered with it
/// disabled (an empty pyramid = the flat G1 block-DDA). The pyramid may only stride
/// through provably-empty cells, so any hit it changes is a stride-overshoot bug.
#[test]
fn brick_raymarch_pyramid_on_equals_off() {
    use voxel_worker::{
        brick_representable_overlay, build_brick_field, pack_gpu_records, AppCore,
        BrickRaymarchRenderer,
        ClipmapPyramid, LayerBand, OrbitCamera, TwoLayerStore, COLOR_TARGET_FORMAT,
    };

    let gpu = pollster::block_on(GpuContext::new(None));
    let width = 128u32;
    let height = 128u32;
    let mut failures: Vec<String> = Vec::new();

    for case in brick_render_cases() {
        let vpb = case.voxels_per_block;
        let two_layer_chunks = TwoLayerStore::enabled().build_covering_chunks(&case.scene, vpb, 0);
        let build = build_brick_field(&two_layer_chunks, vpb);
        assert!(!build.brick_records.is_empty(), "{}: empty brick field", case.name);
        let pyramid = ClipmapPyramid::from_records(&build.brick_records);
        // Every level must carry cells, else "on == off" is vacuous (the L3 skip is
        // exercised too — G4).
        assert!(
            !pyramid.level_1.cell_keys.is_empty()
                && !pyramid.level_2.cell_keys.is_empty()
                && !pyramid.level_3.cell_keys.is_empty(),
            "{}: pyramid has no cells — the on/off comparison would be vacuous",
            case.name
        );
        let overlay_active =
            brick_representable_overlay(&two_layer_chunks).unwrap_or(false);
        let recentre = case.scene.recentre_voxels_for_resolve(vpb);
        let grid_dimensions = case.scene.placed_region_dimensions(vpb);

        let mut app_core = AppCore::new(OrbitCamera::default());
        app_core.camera.target = glam::Vec3::ZERO;
        app_core.camera.orbit_distance = OrbitCamera::auto_framed_distance(grid_dimensions);
        let aspect_ratio = width as f32 / height as f32;
        let view_projection = app_core.view_projection(aspect_ratio, grid_dimensions);
        let viewport_px = [0u32, 0, width, height];
        let band = LayerBand::FULL;
        let gpu_records = pack_gpu_records(&build, |_| false);

        let mut renderer = BrickRaymarchRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
        let render = |renderer: &mut BrickRaymarchRenderer, pyramid: &ClipmapPyramid| {
            renderer.install_brick_field(
                &gpu.device,
                &gpu.queue,
                &build,
                &gpu_records,
                pyramid,
                recentre,
                overlay_active,
            );
            renderer.update_uniforms(
                &gpu.queue,
                view_projection,
                viewport_px,
                grid_dimensions,
                band,
                false,
                Some(MaterialChoice::default()),
            );
            renderer.render_hit_identity_image(&gpu.device, &gpu.queue, width, height)
        };

        let on_image = render(&mut renderer, &pyramid);
        let off_image = render(&mut renderer, &ClipmapPyramid::empty());

        let mut on_hits = 0usize;
        let mut differing = 0usize;
        let mut first: Option<(u32, u32, [u32; 4], [u32; 4])> = None;
        for y in 0..height {
            for x in 0..width {
                let index = (y * width + x) as usize;
                if on_image[index][0] == 1 {
                    on_hits += 1;
                }
                if on_image[index] != off_image[index] {
                    differing += 1;
                    if first.is_none() {
                        first = Some((x, y, off_image[index], on_image[index]));
                    }
                }
            }
        }
        if on_hits == 0 {
            failures.push(format!("{}: the gated view produced ZERO brick hits", case.name));
            continue;
        }
        if differing > 0 {
            failures.push(format!(
                "{}: {differing}/{} pixels differ pyramid-on vs off (first {:?}) — the \
                 hierarchical skip changed a hit (stride overshoot / off-by-epsilon)",
                case.name,
                width * height,
                first
            ));
        } else {
            eprintln!(
                "{}: pyramid on == off ({on_hits} hits; L1 {} cells, L2 {} cells, L3 {} cells)",
                case.name,
                pyramid.level_1.cell_keys.len(),
                pyramid.level_2.cell_keys.len(),
                pyramid.level_3.cell_keys.len()
            );
        }
    }

    assert!(
        failures.is_empty(),
        "brick clip-map changed the hit set (ADR 0011 gate coarse tier — pyramid must \
         only skip empty space):\n{}",
        failures.join("\n")
    );
}

/// **ADR 0011 G2/G4 perf probe (acceptance criterion).** The WIDE-scatter
/// empty-space-skipping lift across clip-map depth: shapes spread over a
/// ~2000-block extent with fully-empty L3 cells between them, marched at four level
/// configs — OFF (flat DDA) / L1+L2 (the G2 two-level baseline) / +L3 (G4) / +L4
/// (a hypothetical 4096-block level, EVALUATED here, not shipped) — reported as
/// mean block-DDA steps per hitting ray (the CPU march's counted loop iterations,
/// the same traversal the shader runs). `#[ignore]`d (measurement, not a gate); run
/// with `cargo test --features gpu --release -- --ignored
/// clipmap_scattered_scene_skips_empty_space --nocapture`.
///
/// On L4: a 1024-block-spaced scatter (~2060-block extent) fits inside ONE
/// 4096-block L4 cell, so L4 has nothing to skip and measures identical to +L3.
/// Scenes wide enough for empty L4 cells (8192+-block spans) cannot currently be
/// BUILT: `build_covering_chunks` enumerates the scene AABB in 4-block chunks
/// (2048³ ≈ 8.6e9 chunk builds at that span), so no realistic scene reaches L4's
/// regime — the measured verdict for not shipping a 4th level.
#[test]
#[ignore = "perf probe — run explicitly with --release --ignored --nocapture"]
fn clipmap_scattered_scene_skips_empty_space() {
    use voxel_worker::{
        brick_representable_overlay, build_brick_field, cpu_march_levels_counted, pack_gpu_records,
        AppCore, BrickRaymarchRenderer, ClipmapLevel, ClipmapPyramid, LayerBand, NodeTransform,
        OrbitCamera, TwoLayerStore, CLIPMAP_LEVEL_1_BLOCKS_PER_CELL, CLIPMAP_LEVEL_2_BLOCKS_PER_CELL,
        CLIPMAP_LEVEL_3_BLOCKS_PER_CELL, COLOR_TARGET_FORMAT,
    };

    let gpu = pollster::block_on(GpuContext::new(None));
    let width = 320u32;
    let height = 320u32;
    let vpb = 16u32;

    // A wide scatter: a 3×3×3 lattice of 13-block spheres spaced 1024 BLOCKS apart,
    // so every adjacent pair straddles a fully-EMPTY 512-block L3 cell (block
    // positions 0/1024/2048 → L3 cells 0/2/4 occupied, cells 1/3 empty). The
    // INTERIOR spheres matter: a hull-corner lattice alone puts every visible
    // object on the traversal-AABB surface, so hitting rays enter right next to
    // their sphere and never cross a void — the center/face/edge spheres are seen
    // THROUGH the gaps, so their rays cross ~1024 blocks of empty space: two
    // 512-block strides under L3 vs sixteen 64-block strides under L2 vs ~1024
    // flat steps. Extent ~2060 blocks, within the flat MAX_BLOCK_STEPS budget so
    // the OFF baseline completes; 13-block spheres at 320² stay a few pixels wide
    // so the auto-framed view actually hits them.
    const SCATTER_SPACING_BLOCKS: i64 = 1024;
    // The evaluated 4th level: 4096 blocks/cell, continuing the 8× progression. Not a
    // production constant — this probe measures whether it would pay before we build it.
    const CLIPMAP_LEVEL_4_BLOCKS_PER_CELL: u32 = 4096;
    let mut nodes = Vec::new();
    for index in 0..27i64 {
        let shape = SdfShape::from_blocks(ShapeKind::Sphere, [13, 13, 13], 1, vpb);
        let mut node = Node::new(
            format!("s{index}"),
            NodeContent::Tool { shape, material: MaterialChoice::Stone },
        );
        node.transform = NodeTransform::from_blocks(
            [
                (index % 3) * SCATTER_SPACING_BLOCKS,
                ((index / 3) % 3) * SCATTER_SPACING_BLOCKS,
                (index / 9) * SCATTER_SPACING_BLOCKS,
            ],
            vpb,
        );
        nodes.push(node);
    }
    let scene = Scene::from_nodes(nodes);
    let two_layer_chunks = TwoLayerStore::enabled().build_covering_chunks(&scene, vpb, 0);
    let build = build_brick_field(&two_layer_chunks, vpb);
    let pyramid_on = ClipmapPyramid::from_records(&build.brick_records);
    let gpu_records = pack_gpu_records(&build, |_| false);
    let overlay_active = brick_representable_overlay(&two_layer_chunks).unwrap_or(false);
    let recentre = scene.recentre_voxels_for_resolve(vpb);
    let grid_dimensions = scene.placed_region_dimensions(vpb);

    // The four clip-map configs, each a slice of levels COARSEST→FINEST (the descent
    // order). OFF is the empty slice (flat block-DDA); +L4 prepends a hypothetical
    // 4096-block level to the shipped three.
    let level_1 = ClipmapLevel::from_records(&build.brick_records, CLIPMAP_LEVEL_1_BLOCKS_PER_CELL);
    let level_2 = ClipmapLevel::from_records(&build.brick_records, CLIPMAP_LEVEL_2_BLOCKS_PER_CELL);
    let level_3 = ClipmapLevel::from_records(&build.brick_records, CLIPMAP_LEVEL_3_BLOCKS_PER_CELL);
    let level_4 = ClipmapLevel::from_records(&build.brick_records, CLIPMAP_LEVEL_4_BLOCKS_PER_CELL);
    let configs: [(&str, Vec<&ClipmapLevel>); 4] = [
        ("OFF (flat)", Vec::new()),
        ("L1+L2 (G2)", vec![&level_2, &level_1]),
        ("+L3 (G4)", vec![&level_3, &level_2, &level_1]),
        ("+L4 (eval)", vec![&level_4, &level_3, &level_2, &level_1]),
    ];

    let mut app_core = AppCore::new(OrbitCamera::default());
    app_core.camera.target = glam::Vec3::ZERO;
    app_core.camera.orbit_distance = OrbitCamera::auto_framed_distance(grid_dimensions);
    let view_projection = app_core.view_projection(width as f32 / height as f32, grid_dimensions);

    let mut renderer = BrickRaymarchRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
    renderer.install_brick_field(
        &gpu.device,
        &gpu.queue,
        &build,
        &gpu_records,
        &pyramid_on,
        recentre,
        overlay_active,
    );
    let frame = renderer.update_uniforms(
        &gpu.queue,
        view_projection,
        [0, 0, width, height],
        grid_dimensions,
        LayerBand::FULL,
        false,
        Some(MaterialChoice::default()),
    );

    // Sum block-DDA steps over the rays that HIT (empty-space skipping shows up on the
    // rays that traverse the scene), per config. The hit set is identical across configs
    // (each level may only skip provably-empty space — the pyramid-on == off gate), so
    // this compares work-per-ray over the same pixels; the finest config (+L4) defines
    // the hit set.
    let mut sums = [0u64; 4];
    let mut hitting_rays = 0u64;
    for y in 0..height {
        for x in 0..width {
            let pixel = glam::Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let mut per_config = [0u32; 4];
            let mut hit_any = false;
            for (index, (_, levels)) in configs.iter().enumerate() {
                let (hit, steps) =
                    cpu_march_levels_counted(&frame, &gpu_records, &build, levels, pixel);
                per_config[index] = steps;
                if index == configs.len() - 1 {
                    hit_any = hit.is_some();
                }
            }
            if hit_any {
                for index in 0..configs.len() {
                    sums[index] += per_config[index] as u64;
                }
                hitting_rays += 1;
            }
        }
    }
    assert!(hitting_rays > 0, "the scattered probe view produced no hits");
    let span_blocks = grid_dimensions.map(|d| d / vpb).iter().max().copied().unwrap_or(0);
    let mean = |index: usize| sums[index] as f64 / hitting_rays as f64;
    eprintln!(
        "clip-map WIDE-scatter probe ({}-block span, {} sculpted bricks, {} hitting rays)\n  \
         L1 {} cells, L2 {} cells, L3 {} cells, L4 {} cells\n  \
         mean block-steps/ray by clip-map config:",
        span_blocks,
        build.sculpted_brick_count(),
        hitting_rays,
        level_1.cell_keys.len(),
        level_2.cell_keys.len(),
        level_3.cell_keys.len(),
        level_4.cell_keys.len(),
    );
    for (index, (name, _)) in configs.iter().enumerate() {
        eprintln!(
            "    {name:<12} {:>8.1}  ({:.2}× vs OFF)",
            mean(index),
            mean(0) / mean(index).max(1.0),
        );
    }

    // Gate: each coarser level is a monotone win (never more steps), the two-level
    // pyramid beats flat, and G4's L3 strictly beats the L1+L2 baseline on this
    // empty-L3-cell scene (the acceptance criterion — a measured ceiling improvement).
    assert!(sums[1] < sums[0], "L1+L2 must beat flat (on {} vs {})", sums[1], sums[0]);
    assert!(
        sums[2] < sums[1],
        "G4 +L3 must strictly reduce block-steps vs the L1+L2 baseline on a wide scatter \
         with empty L3 cells (+L3 {} vs L1+L2 {})",
        sums[2], sums[1]
    );
    assert!(
        sums[3] <= sums[2],
        "+L4 may only skip more empty space, never less (+L4 {} vs +L3 {})",
        sums[3], sums[2]
    );
}
