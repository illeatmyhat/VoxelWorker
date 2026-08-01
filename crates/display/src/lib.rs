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
//! — an upward `use` (orchestrator, routing, workers, `app_core`, panel, settings, `vox_export`,
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
//! the brick field + raymarch, the onion ghost) for the statement over the whole layer.
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
//! * [`assets`] — the asset-pack decode + registry (custom packs, game packs, face textures).
//!   A sibling dependency crate, not a module of this one; it earns its place in this list
//!   because `block_texture` above builds its materials out of it.

// A public item's doc may link to a private helper to explain how the two relate; that
// cross-reference stays a navigable link under `--document-private-items`. The CI doc gate
// denies broken and redundant links but permits these.
#![allow(rustdoc::private_intra_doc_links)]
#![allow(
    clippy::bool_to_int_with_if,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::map_unwrap_or,
    clippy::needless_continue,
    clippy::needless_pass_by_value,
    clippy::option_if_let_else,
    clippy::or_fun_call,
    clippy::redundant_closure_for_method_calls,
    clippy::redundant_pub_crate,
    clippy::single_option_map,
    clippy::suboptimal_flops,
    clippy::trivially_copy_pass_by_ref,
    clippy::tuple_array_conversions,
    clippy::use_self,
    clippy::useless_let_if_seq,
    clippy::wildcard_imports,
    clippy::while_float
)]

#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::redundant_pub_crate
)]
pub mod block_texture;
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::missing_const_for_fn,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::redundant_pub_crate,
    clippy::too_many_lines
)]
pub mod brick;
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::redundant_pub_crate,
    clippy::too_many_lines
)]
pub mod mesh;
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::redundant_pub_crate,
    clippy::too_many_lines
)]
pub mod renderer;
mod shaders;
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::redundant_pub_crate
)]
pub mod texture_atlas;

/// A renderer that records one draw call into a frame phase.
///
/// The shell groups scene draws into ordered phases — background, over-model ghosts, scaffold, on-top —
/// and records each phase's slice in turn into the single viewport pass. The solid model and
/// the view cube are NOT scene draws (they need the material bind group / their own sub-pass);
/// everything else drawn in the viewport is.
pub trait SceneDraw {
    /// Record this draw into the in-progress viewport render pass. Self-gating: an empty batch
    /// (nothing to show this frame) records nothing.
    fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>);
}

/// Delegate `SceneDraw::draw` to each renderer's inherent `draw` (inherent methods win method
/// resolution, so this is a plain forward, not recursion). The draw logic stays in each
/// renderer's own file; this is only the roster of what the shell may put in a phase.
macro_rules! impl_scene_draw {
    ($($ty:ty),+ $(,)?) => {
        $(impl SceneDraw for $ty {
            fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>) {
                self.draw(render_pass)
            }
        })+
    };
}
impl_scene_draw!(
    renderer::BackgroundGradientRenderer,
    renderer::SceneGridRenderer,
    renderer::InfiniteGridRenderer,
    renderer::PointsRenderer,
    renderer::TransformGizmoRenderer,
    renderer::PlacementGhostRenderer,
    renderer::SketchRegionRenderer,
    mesh::SelectedOperandGhostRenderer,
);
