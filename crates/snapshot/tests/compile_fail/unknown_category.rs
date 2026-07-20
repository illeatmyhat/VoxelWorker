// A plausible-sounding category that does not exist. Worth a fixture because the
// failure mode is a developer inventing a word that reads correctly ("cache", "ui",
// "temporary") and needing to be told the closed set rather than guessing again.

use snapshot::Snapshot;

#[derive(Snapshot)]
struct CacheState {
    #[snapshot(cache)]
    resolved_chunk_count: usize,
}

fn main() {}
