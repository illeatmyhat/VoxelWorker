//! Domain question: how cheap are the brick-occupancy-tile bit ops the display
//! pipeline runs to rasterize, popcount, and unpack an occupancy brick?
//! (`BitCube` is the CS core of the brick occupancy tile — see
//! `docs/architecture/03-display.md`, the brick pipeline and its atlas.)
//!
//! Two edges bracket the domain: 16 (Vintage Story block density) and 64 (the
//! density bound, one voxel row = one machine word). We bench the run-set writer
//! (short runs vs whole rows), the popcount reduction, and the dense↔packed
//! byte round-trip an atlas upload/readback pays.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use substrate::occupancy::BitCube;

/// The two edges the domain actually uses: VS density and the 1..=64 bound.
const EDGES: [u32; 2] = [16, 64];

/// The "occupied" byte the expand injects — an arbitrary distinctive nonzero.
const SET_BYTE: u8 = 0xFF;

/// A cube whose every row carries a short centred X-run (roughly the middle
/// third) — the sparse-ish brick a boundary rasterization tends to produce.
fn cube_with_short_runs(edge: u32) -> BitCube {
    let mut cube = BitCube::empty(edge);
    let lo = edge / 3;
    let hi = (2 * edge / 3).min(edge - 1).max(lo);
    for z in 0..edge {
        for y in 0..edge {
            cube.set_x_run(y, z, lo, hi);
        }
    }
    cube
}

/// A fully solid cube (every row is a whole-width run) — the dense-brick case.
fn cube_solid(edge: u32) -> BitCube {
    let mut cube = BitCube::empty(edge);
    for z in 0..edge {
        for y in 0..edge {
            cube.set_x_run(y, z, 0, edge - 1);
        }
    }
    cube
}

fn bench_set_x_run(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_cube/set_x_run");
    for &edge in &EDGES {
        // edge² rows written per full-cube fill.
        group.throughput(Throughput::Elements((edge * edge) as u64));
        let short_lo = edge / 3;
        let short_hi = (2 * edge / 3).min(edge - 1).max(short_lo);
        group.bench_with_input(BenchmarkId::new("short_run", edge), &edge, |b, &edge| {
            b.iter(|| {
                let mut cube = BitCube::empty(edge);
                for z in 0..edge {
                    for y in 0..edge {
                        cube.set_x_run(y, z, short_lo, short_hi);
                    }
                }
                black_box(cube.popcount())
            });
        });
        group.bench_with_input(BenchmarkId::new("full_row", edge), &edge, |b, &edge| {
            b.iter(|| {
                let mut cube = BitCube::empty(edge);
                for z in 0..edge {
                    for y in 0..edge {
                        cube.set_x_run(y, z, 0, edge - 1);
                    }
                }
                black_box(cube.popcount())
            });
        });
    }
    group.finish();
}

fn bench_popcount(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_cube/popcount");
    for &edge in &EDGES {
        group.throughput(Throughput::Elements((edge * edge) as u64));
        let solid = cube_solid(edge);
        group.bench_with_input(BenchmarkId::from_parameter(edge), &solid, |b, cube| {
            b.iter(|| black_box(cube.popcount()));
        });
    }
    group.finish();
}

fn bench_expand_from_bytes_round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_cube/expand_from_bytes_round_trip");
    for &edge in &EDGES {
        // edge³ bytes cross the expand/pack boundary.
        group.throughput(Throughput::Bytes((edge * edge * edge) as u64));
        let cube = cube_with_short_runs(edge);
        group.bench_with_input(BenchmarkId::from_parameter(edge), &cube, |b, cube| {
            b.iter(|| {
                let bytes = cube.expand_to_bytes(SET_BYTE);
                black_box(BitCube::from_bytes(edge, &bytes))
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_set_x_run,
    bench_popcount,
    bench_expand_from_bytes_round_trip
);
criterion_main!(benches);
