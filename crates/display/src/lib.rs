//! # display — the one crate that links wgpu (classified boundary set → pixels)
//!
//! This crate is the whole system's window onto the GPU, and the only place a wgpu type is
//! named. Everything above it authors intent and evaluates it to occupancy on the CPU;
//! this crate takes that occupancy — the evaluation layer's classified two-layer chunks —
//! and turns it into pixels. That is Law 4 ("the CPU owns truth; the GPU owns the frame"):
//! evaluation, classification, export, and measurement are correct without any GPU present,
//! and this crate receives derived display caches it is free to render fast and absent (a
//! headless build renders the same voxels through the same mesher). Nothing here is truth;
//! every sink here reads the one evaluator's output and never re-evaluates the scene, so two
//! sinks can never drift (Law 6, "classified once, consumed everywhere").
//!
//! ## The boundary law
//!
//! A component belongs here if and only if it **consumes the classified boundary set and
//! produces pixels** — it names a wgpu device, queue, pipeline, buffer, or shader. Every
//! GPU sink lives here: the cuboid fallback mesher, the brick field build + its GPU record
//! pack, the brick raymarch pipeline, the material texture atlas, the loaded block
//! material, the asset-pack decode/registry, and the render pipelines (view
//! cube, grids, gizmo, points). The device and queue are handed **in** from the shell as
//! parameters; this crate never creates a device, opens a surface, or touches a window — no
//! winit, no UI toolkit, no event loop. Windowing, input, the UI-facing palette state, and the
//! frame loop are the shell's; the display-state machine that OWNS the async workers (the
//! engagement orchestrator/routing) is work-layer, not here, and lands at the work-crate cut.
//!
//! The dependency edge is one-way: `evaluation ← display ← {work, shell}`, compile-enforced
//! — an upward `use` (orchestrator, routing, workers, app_core, panel, settings, vox_export,
//! gpu) fails to build. The dependencies are `evaluation` (the two-layer chunks it renders),
//! `document` (the Scene the resolve oracle densifies in tests + the scene-graph nouns the
//! brick-field tests build fixtures from), `voxel_core` (the value vocabulary + block/cell
//! codec), `substrate` (interval arithmetic, cuboid decomposition, `GenerationTracker`),
//! `camera` + `raycast` (the wgpu-free viewing + traversal mathematics the shaders mirror),
//! `assets` (the pure-CPU block-texture loader `block_texture` builds materials from), plus
//! `wgpu`/`glam`/`bytemuck`/`rayon`/`profiling`. Its tests hold the mesh + brick paths against the dense `Scene::resolve_region`
//! oracle (document's `oracle` feature), compile-gated out of production builds.
//!
//! ## The chapter it serves
//!
//! These are the nouns and verbs of the architecture's display layer — see
//! `docs/architecture/03-display.md` (the CPU/GPU truth boundary, the two display pipelines,
//! the brick field + raymarch, the onion ghost) for the timeless statement, and
//! `docs/design/per-layer-crates-extraction-map.md` (the display row) for the dated
//! provenance of each module.
//!
//! ## Modules
//!
//! * [`renderer`] — the render-pipeline surface: the view cube, the infinite/scene grids,
//!   the transform gizmo, the points/axes, the material source + layer band, and the depth /
//!   MSAA target helpers.
//! * [`mesh`] — the CPU box-decomposed fallback mesher ([`mesh::CuboidMeshRenderer`]):
//!   the always-present, no-GPU-capable voxel render path + the incremental re-mesh.
//! * [`brick`] — the brick display path: the brick-field BUILD (two-layer boundary set →
//!   sorted brick records + the sculpted-brick + cell-key atlases + the L1–L3 clipmap
//!   pyramid) and the raymarch display sink ([`brick::BrickRaymarchRenderer`]): block DDA +
//!   record binary search + sculpted voxel DDA, and the CPU march mirror.
//! * [`texture_atlas`] — the packed material atlas ([`texture_atlas::MaterialAtlas`]) the sinks sample.
//! * [`block_texture`] — the runtime-loaded scene block material + its bind-group layout (pure wgpu).
//! * [`assets`] — the asset-pack decode + registry (custom packs, VS packs, face textures).

// A public item's doc may link to a private helper to explain how the two relate; that
// cross-reference stays a navigable link under `--document-private-items`. The CI doc gate
// denies broken and redundant links but permits these.
#![allow(rustdoc::private_intra_doc_links)]

pub mod block_texture;
pub mod brick;
pub mod mesh;
pub mod renderer;
pub mod texture_atlas;
