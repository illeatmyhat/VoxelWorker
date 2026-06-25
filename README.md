# Chisel Bench

A native desktop planning tool for **Vintage Story** chiseling. It defines a parametric shape
in a block-sized box, samples it onto a voxel grid (VS's 16×16×16 microblock grid by default),
and renders the result as hard, individually-textured cubes — so you can see the exact
stair-stepped quantization *before* you chisel in-game. It also reads your real Vintage Story
install to populate a palette of chiselable block textures.

Native **Rust** reimplementation of a three.js browser prototype (`chisel-bench-reference.html`),
built with **wgpu + egui + winit**. The native port exists primarily to read the VS asset folder
directly (the browser File System Access sandbox blocks `%APPDATA%`).

## Status

In active autonomous development — see [`PROGRESS.md`](PROGRESS.md) and the GitHub issues.

## Design docs

- [`HANDOFF.md`](HANDOFF.md) — what this is, why these tech choices, build order, don'ts.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — SDF math, shader logic, the two fixed bugs, camera rig.
- [`DATA.md`](DATA.md) — units model, chiselable block list, VS install paths, texture/JSON layout.
- [`docs/DEV_NOTES.md`](docs/DEV_NOTES.md) — verified crate API signatures for the pinned versions.

## Build & run

```sh
cargo run                       # windowed app
cargo run --bin shot -- --help  # headless screenshot capture (no window)
```

## Two non-negotiable invariants (regression guards)

1. **One texture per block**, sliced per voxel by `block_local_coord / voxels_per_block` — never
   one full texture repeated per cube.
2. **Grid overlay computed from world position, not face UVs** (UVs flip per face → off-by-one).
