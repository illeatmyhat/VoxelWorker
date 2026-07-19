# VoxelWorker

A native desktop planning tool for **Vintage Story** chiselling. You compose a shape out of
solids and boolean cuts, sample it onto the voxel lattice VS actually chisels on (16³
microblocks per block by default), and see the exact stair-stepped result — with real block
textures — *before* you spend the in-game hours. It reads your Vintage Story install directly
to populate a palette of chiselable block textures, and exports `.vox`.

Built in **Rust** with **wgpu + egui + winit**. The world is **Z-up** (ground plane is XY).

## Build & run

```sh
cargo run                    # the windowed app
cargo run -p shot -- --help  # headless offscreen capture (no window, no GPU surface)
cargo test --workspace       # the full gate
```

No feature flags are needed for any of the above. Two optional ones exist: `tracy` (opt-in CPU
profiling, pulls a C/C++ dependency — see `docs/profiling.md`) and per-crate `test-support`.

## What it does

- **Composition, not a single shape.** The document is an ordered fold of parts and tools —
  union / subtract / intersect, sealed scopes, reusable definitions placed as linked instances,
  and half-space cutters. The operation stack is the truth; the runtime builds no dense grid.
- **Fields.** Outset dilates any node's composed body; emboss moves accumulated surface within
  a cutter's footprint; noise is a bounded field operation. Authored measurements are retained
  (`"1/4 block"` survives a density change as intent, not as a stale derived number).
- **Two display paths, one truth.** A cuboid mesher and a GPU brick-field raymarch render the
  same evaluated boundary set, held comparable by a golden-image suite.
- **Sparse evaluation.** A boundary-aware two-layer store keeps un-chiselled regions as coarse
  block ids and only chiselled boundary blocks as sub-block cuboids — the same split Vintage
  Story itself uses. Solid interiors are never voxelised.
- **Viewer modes.** Normal, onion (region-scoped clip slabs for reading interior layers), and
  show-booleans x-ray ghosts of operands.

## Architecture

The layers are cut into workspace crates so the dependency direction is compile-enforced and
flows strictly downward (`docs/adr/0016`):

```
substrate · camera · raycast   pure CS/math, and wgpu-free graphics math
voxel_core                     the value vocabulary (voxel, units, palette id, frames)
document                       authored truth: scene graph, producers, sketches, intents
evaluation                     the one evaluator: residency, chunk resolve, two-layer store
assets                         pure-CPU block-texture sourcing from a VS install
display / interchange          the sinks: GPU pipelines (the only wgpu) / headless .vox export
work                           async workers + the engagement state machine
ui · root crate                the shell: egui panels, winit, app wiring
shot                           the golden reference tool (its own package)
```

A crate cannot name a crate it does not depend on, so the layering is enforced by Cargo rather
than by convention.

## Documentation

- [`CONTEXT.md`](CONTEXT.md) — the glossary. Canonical meaning of every term (block vs voxel,
  chunk, boundary residency, brick, fold). Start here.
- [`docs/architecture/`](docs/architecture/) — the *timeless* shape of the system, one file per
  layer. Describes what is, not how it got there.
- [`docs/adr/`](docs/adr/) — append-only decision records (0001–0021). The reasoning behind
  every non-obvious choice lives here.
- [`docs/design/`](docs/design/) — working notes: extraction maps, prior-art studies, the
  refactoring map, the Signal chrome spec.
- [`docs/DEV_NOTES.md`](docs/DEV_NOTES.md) — verified crate API signatures for the pinned
  versions, and the golden-image workflow.

## Regression guards

Two invariants that were expensive to learn and are cheap to break. Both are enforced in the
shaders and annotated there:

1. **One texture per block**, sliced per voxel by `block_local_coord / voxels_per_block` — never
   one full texture repeated per cube. A block face carries a whole block texture; chiselling
   subdivides the *geometry*, not the texture addressing.
2. **The grid overlay is computed from world position, not face UVs.** UVs flip per face, which
   produces an off-by-one on half of them. `cuboid.wgsl` marks this "NOT face UVs — project
   guard".

## Verification

The golden-image suite renders canonical scenes through the real `shot` binary and
tolerance-compares against committed references — the safety net for renderer work:

```sh
cargo test -p shot --test golden                     # compare
UPDATE_GOLDENS=1 cargo test -p shot --test golden    # re-bless after an INTENDED visual change
```

Beyond the test suite, `verification/` carries on-demand proof tiers (Kani bounded model
checking, Verus, Lean) for the wgpu-free crates. They are not part of the cargo gate;
`verification/run-all.sh --quick` takes seconds and is safe per-commit.

## Provenance

The tool concept appears as "Chisel Bench" in the earliest design notes. It began as a Rust port
of a three.js browser prototype (`chisel-bench-reference.html`, kept for reference) — the native
port existed to read the VS asset folder directly, which the browser File System Access sandbox
blocks for `%APPDATA%`. The engine has since diverged completely from that prototype.
