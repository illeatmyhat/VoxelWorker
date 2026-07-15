//! Procedural-generation kit: a deterministic small RNG ([`SmallRng`], an LCG with
//! a Fisher–Yates shuffle) and improved Perlin gradient noise with fBm
//! ([`PerlinNoise`]). Deterministic from a seed — the same seed always yields the
//! same field — so it backs reproducible content and golden-image tests. Each
//! submodule carries its own literature citations.

pub mod perlin;
pub mod rng;

pub use perlin::PerlinNoise;
pub use rng::SmallRng;
