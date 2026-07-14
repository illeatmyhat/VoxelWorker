//! Domain question: what does the per-edit broadphase cost at the 10k-producer
//! target — both the rebuild-from-scratch and the overlap query the edit runs
//! against it? (The `Bvh` is the edit broadphase; the scene-scale target and the
//! "interface never waits" budget are in `docs/architecture/04-work.md`.)
//!
//! The hierarchy is rebuilt per use, so build speed is the cost that matters; we
//! bench build at N = 1_000 and 10_000 AABBs, then two query shapes on the 10k
//! build — a small local edit box (the common case, prunes almost everything) and
//! a scene-spanning box (the worst case, visits the whole tree).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use substrate::{Aabb, Bvh};

/// A deterministic LCG (Numerical Recipes constants) — the same generator the
/// substrate tests use, so bench populations are reproducible with no `rand` dep.
struct Lcg(u64);
impl Lcg {
    fn next_in(&mut self, range: i64) -> i64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as i64).rem_euclid(range)
    }
}

/// `count` pseudo-random non-empty boxes scattered across a ±`span` cube, each
/// with a small positive extent (so none is empty and the broadphase keeps them).
fn scatter_boxes(count: usize, span: i64) -> Vec<Aabb> {
    let mut lcg = Lcg(0x1234_5678_9abc_def0);
    let mut boxes = Vec::with_capacity(count);
    for _ in 0..count {
        let min = [
            lcg.next_in(2 * span) - span,
            lcg.next_in(2 * span) - span,
            lcg.next_in(2 * span) - span,
        ];
        let extent = [
            1 + lcg.next_in(32),
            1 + lcg.next_in(32),
            1 + lcg.next_in(32),
        ];
        boxes.push(Aabb::new(
            min,
            [min[0] + extent[0], min[1] + extent[1], min[2] + extent[2]],
        ));
    }
    boxes
}

fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("bvh/build");
    for &count in &[1_000usize, 10_000] {
        let boxes = scatter_boxes(count, 4_000);
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &boxes, |b, boxes| {
            b.iter(|| black_box(Bvh::build(black_box(boxes))));
        });
    }
    group.finish();
}

fn bench_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("bvh/query_10k");
    let boxes = scatter_boxes(10_000, 4_000);
    let bvh = Bvh::build(&boxes);
    // A small local edit box near the origin — the common case, prunes almost all.
    let small = Aabb::new([-64, -64, -64], [64, 64, 64]);
    // The whole scene — the worst case, visits every node.
    let spanning = Aabb::new([-4_100, -4_100, -4_100], [4_100, 4_100, 4_100]);
    group.bench_function("small_box", |b| {
        b.iter(|| black_box(bvh.overlapping_input_indices(black_box(&small))));
    });
    group.bench_function("scene_spanning_box", |b| {
        b.iter(|| black_box(bvh.overlapping_input_indices(black_box(&spanning))));
    });
    group.finish();
}

criterion_group!(benches, bench_build, bench_query);
criterion_main!(benches);
