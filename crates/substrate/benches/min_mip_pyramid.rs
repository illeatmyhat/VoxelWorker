//! Domain question: is folding the occupancy key set into the clip-map pyramid cheap enough to
//! rebuild per edit? The domain builds three levels (edges 8, 64, 512) over the occupied-block key
//! set every time the boundary set changes, so this bench measures the fold + parallel sort + dedup
//! at two occupancy scales. (The pyramid keys the sparse hierarchy the display raymarch descends —
//! see `docs/architecture/03-display.md`, the brick-field clip-map.)
//!
//! We bench `SparseMinMipPyramid::from_keys` over a deterministic scattered key set at the domain's
//! real edge progression, at 10k and 100k keys — the per-edit occupancy counts the fold meets.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use substrate::spatial::lattice_key::pack_lattice_key;
use substrate::spatial::SparseMinMipPyramid;

/// The domain's three clip-map cell edges (blocks per cell): 8 → 64 → 512.
const CELL_EDGES: [u32; 3] = [8, 64, 512];

/// A deterministic LCG (Numerical Recipes constants) — reproducible, no `rand`.
struct Lcg(u64);
impl Lcg {
    fn next_axis(&mut self) -> i64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // A ±few-thousand-block span, so the fold collapses many keys into shared coarse cells
        // (the scattered scene the LOD targets), staying well inside the packing lane.
        ((self.0 >> 33) as i64).rem_euclid(1 << 13) - (1 << 12)
    }
}

fn packed_keys(count: usize) -> Vec<u64> {
    let mut lcg = Lcg(0x0f0f_0f0f_dead_beef);
    (0..count)
        .map(|_| pack_lattice_key([lcg.next_axis(), lcg.next_axis(), lcg.next_axis()]))
        .collect()
}

fn bench_from_keys(c: &mut Criterion) {
    let mut group = c.benchmark_group("min_mip_pyramid/from_keys");
    for &count in &[10_000usize, 100_000] {
        let keys = packed_keys(count);
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &keys, |b, keys| {
            b.iter(|| black_box(SparseMinMipPyramid::from_keys(black_box(keys), &CELL_EDGES)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_from_keys);
criterion_main!(benches);
