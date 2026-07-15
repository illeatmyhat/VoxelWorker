//! Domain question: what does maintaining the run-list cost as the streamed
//! band sweep feeds it — the common ascending-arrival case versus the
//! out-of-order merge fallback? (The `DisjointIntervalSet` is the run-list inside
//! the widest-run band sweep — see `docs/architecture/02-evaluation.md`, the
//! boundary-set evaluation the sweep serves.)
//!
//! Ascending arrivals hit the O(1) append/extend fast path (the dominant case);
//! out-of-order arrivals with overlaps take the linear splice-merge. We bench
//! both at 10k inserts so the fast path's win over the fallback is legible.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use std::hint::black_box;
use substrate::interval::DisjointIntervalSet;

/// Number of intervals inserted per iteration — the sweep's per-band scale.
const INTERVAL_COUNT: i64 = 10_000;

/// A deterministic LCG (Numerical Recipes constants) — reproducible, no `rand`.
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

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("disjoint_interval_set/insert_10k");
    group.throughput(Throughput::Elements(INTERVAL_COUNT as u64));

    // Fast path: strictly ascending, gapped intervals — every insert is an O(1)
    // push after the last run.
    group.bench_function("ascending_append", |b| {
        b.iter(|| {
            let mut set = DisjointIntervalSet::new();
            for i in 0..INTERVAL_COUNT {
                let lo = i * 4;
                set.insert(lo, lo + 2);
            }
            black_box(set.widest_span())
        });
    });

    // Fallback: intervals arrive in pseudo-random order across a bounded span,
    // wide enough to overlap and force the splice-merge path each time.
    let mut lcg = Lcg(0xfeed_face_dead_beef);
    let span = INTERVAL_COUNT * 2;
    let random_intervals: Vec<(i64, i64)> = (0..INTERVAL_COUNT)
        .map(|_| {
            let lo = lcg.next_in(span);
            (lo, lo + 1 + lcg.next_in(8))
        })
        .collect();
    group.bench_function("random_order_merge", |b| {
        b.iter(|| {
            let mut set = DisjointIntervalSet::new();
            for &(lo, hi) in &random_intervals {
                set.insert(lo, hi);
            }
            black_box(set.widest_span())
        });
    });

    group.finish();
}

criterion_group!(benches, bench_insert);
criterion_main!(benches);
