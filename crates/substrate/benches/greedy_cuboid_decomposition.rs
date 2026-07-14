//! Domain question: what does decomposing one boundary block into cuboids cost
//! across the range of shapes the boundary actually presents — a solid block, a
//! hollow shell, and the pathological checkerboard? (The decomposition runs per
//! edit per boundary block on the evaluator's output — see
//! `docs/architecture/02-evaluation.md`, boundary-set to geometry.)
//!
//! Edge 16 is the block density. The three fixtures bracket the cost: a solid
//! cube collapses to ONE cuboid (best case), a hollow shell is the realistic
//! boundary block, and a checkerboard forces one cuboid per cell (the worst case
//! greedy growth can never merge).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use substrate::{CellGrid, GreedyCuboidDecomposition};

/// The block edge in cells (Vintage Story block density).
const EDGE: u32 = 16;

/// Build a `u16`-labeled grid of edge³ from a per-cell closure.
fn grid_from_fn<F: Fn(u32, u32, u32) -> Option<u16>>(edge: u32, f: F) -> CellGrid<u16> {
    let mut grid = CellGrid::new_empty([edge, edge, edge]);
    for z in 0..edge {
        for y in 0..edge {
            for x in 0..edge {
                grid.set(x, y, z, f(x, y, z));
            }
        }
    }
    grid
}

fn bench_decompose(c: &mut Criterion) {
    let mut group = c.benchmark_group("greedy_cuboid_decomposition/decompose_edge16");
    group.throughput(Throughput::Elements((EDGE * EDGE * EDGE) as u64));

    let solid = grid_from_fn(EDGE, |_, _, _| Some(1));
    // Checkerboard: every labeled cell is isolated — greedy growth merges nothing,
    // so this emits one cuboid per set cell (the worst case).
    let checkerboard = grid_from_fn(EDGE, |x, y, z| {
        if (x + y + z) % 2 == 0 {
            Some(1)
        } else {
            None
        }
    });
    // Hollow shell: only the six faces are solid, the interior is air — the
    // realistic boundary block.
    let hollow_shell = grid_from_fn(EDGE, |x, y, z| {
        let last = EDGE - 1;
        if x == 0 || y == 0 || z == 0 || x == last || y == last || z == last {
            Some(1)
        } else {
            None
        }
    });

    for (name, grid) in [
        ("solid_cube", &solid),
        ("checkerboard", &checkerboard),
        ("hollow_shell", &hollow_shell),
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(name), grid, |b, grid| {
            b.iter(|| black_box(GreedyCuboidDecomposition::decompose(black_box(grid))));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_decompose);
criterion_main!(benches);
