//! Domain question: is the lattice-key pack/unpack negligible? It runs per record
//! per edit (keying every brick/record into the sorted, binary-searchable set),
//! so the point of this bench is to CONFIRM it stays in the noise. (The key codec
//! keys the sorted record set the display pipeline binary-searches — see
//! `docs/architecture/03-display.md`, the brick-field record layout.)
//!
//! We bench the full pack → hi/lo split → unpack round-trip over a batch of
//! deterministic coordinates: the per-record work an edit multiplies by its
//! record count.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use std::hint::black_box;
use substrate::spatial::lattice_key::{pack_lattice_key, split_key_hi_lo, unpack_lattice_key};

/// Records touched per edit at scale — the multiplier this per-record cost meets.
const RECORD_COUNT: usize = 100_000;

/// A deterministic LCG (Numerical Recipes constants) — reproducible, no `rand`.
struct Lcg(u64);
impl Lcg {
    fn next_axis(&mut self) -> i64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Keep inside the ±2^19 lane the tests exercise (well within the bias).
        ((self.0 >> 33) as i64).rem_euclid(1 << 20) - (1 << 19)
    }
}

fn bench_round_trip(c: &mut Criterion) {
    let mut lcg = Lcg(0x0123_4567_89ab_cdef);
    let coordinates: Vec<[i64; 3]> = (0..RECORD_COUNT)
        .map(|_| [lcg.next_axis(), lcg.next_axis(), lcg.next_axis()])
        .collect();

    let mut group = c.benchmark_group("lattice_key/pack_unpack_round_trip");
    group.throughput(Throughput::Elements(RECORD_COUNT as u64));
    group.bench_function("100k_records", |b| {
        b.iter(|| {
            let mut acc = 0i64;
            for &coordinate in &coordinates {
                let key = pack_lattice_key(black_box(coordinate));
                let [hi, lo] = split_key_hi_lo(key);
                let unpacked = unpack_lattice_key(((hi as u64) << 32) | lo as u64);
                acc = acc.wrapping_add(unpacked[0] ^ unpacked[1] ^ unpacked[2]);
            }
            black_box(acc)
        });
    });
    group.finish();
}

criterion_group!(benches, bench_round_trip);
criterion_main!(benches);
