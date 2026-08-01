//! `VoxelWorker` — a voxel chiseling planner.
//!
//! This crate is the rendering foundation shared by both the windowed application
//! (`src/main.rs`) and the headless screenshot harness, which is now its own package
//! at `crates/shot/`:
//!
//!   * A render-target-agnostic frame function ([`frame::render::render_frame`]) that paints
//!     into any [`wgpu::TextureView`]. It knows nothing about winit or surfaces,
//!     so the same code draws the on-screen surface texture and the offscreen
//!     capture texture — guaranteeing the screenshot matches the window.
//!   * A single egui panel builder ([`build_panel`]) used by both paths so the
//!     captured frame is identical to the live one.
//!   * The warm-dark "workshop" color identity.

// A public item's doc may link to a private helper to explain how the two relate; that
// cross-reference is deliberate and stays a navigable link under `--document-private-items`.
// The CI doc gate denies broken and redundant links but permits these.
#![allow(rustdoc::private_intra_doc_links)]
// Colors live in `ui::theme::color_palette`; a raw `Color32::from_*` elsewhere is an error.
#![deny(clippy::disallowed_methods)]
#![allow(
    clippy::assigning_clones,
    clippy::bool_to_int_with_if,
    clippy::derive_partial_eq_without_eq,
    clippy::explicit_iter_loop,
    clippy::imprecise_flops,
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::manual_midpoint,
    clippy::map_unwrap_or,
    clippy::match_same_arms,
    clippy::match_wildcard_for_single_variants,
    clippy::needless_pass_by_ref_mut,
    clippy::needless_pass_by_value,
    clippy::option_if_let_else,
    clippy::redundant_closure_for_method_calls,
    clippy::redundant_clone,
    clippy::similar_names,
    clippy::single_match_else,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::unused_self,
    clippy::use_self,
    clippy::wildcard_imports
)]

// The headless orchestrator (scene + store + camera).
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
    clippy::too_many_lines
)]
pub mod app_core;
// The shell's palette GPU host (`PaletteHost`): it owns the wgpu backing the UI-facing
// palette cannot name (the `crate::thumbnail::ThumbnailRenderer`, the texture
// keep-alives, the scanned `BlockGroup`s) and keeps them index-aligned with the
// `ui::palette::BlockPalette` tiles it renders + registers into egui. The egui-facing
// palette state and the inspector panel live in the `ui` crate.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate
)]
pub mod block_palette;
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate
)]
pub mod gpu;
// The persistence artifacts and the exhaustive captures that carry
// classified state into them. Separate from `settings` on purpose: that module holds the
// classified state record, this one holds where it goes and enforces that it gets there.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate
)]
pub mod artifacts;
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate
)]
pub mod settings;
// The palette PREVIEW thumbnail renderer: a shell-side GPU sink that draws the UI's
// 45° cube tiles (NOT the scene), reaching down into `display` only for the shared
// block-texture bind-group layout. Kept out of the `display` scene-view crate.
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
    clippy::too_many_lines
)]
pub mod thumbnail;
// The windowed application (the default binary's logic): `WindowedState` + `App` + the winit
// `ApplicationHandler` + per-frame render + async-worker poll seams. Carved out of `src/main.rs`
// into a shell LIB module tree so the bin is a thin `windowed::run` entry point; the lib
// already carries the winit/egui/wgpu deps this needs.
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
    clippy::too_many_lines
)]
pub mod windowed;
// The engagement state machine + the async worker pool live in the `work` crate
// (`{display, interchange} <- work <- shell`); their types are re-exported flat below so the
// shell's `voxel_worker::<Name>` uses keep resolving.

#[cfg(test)]
mod windowed_resolve_tests;

// The standalone exactness parity for the conservative cell-interval
// bound primitive (VoxelProducer::cell_field_interval) + the CSG interval composition.
#[cfg(test)]
mod cell_interval_parity_tests;

pub use app_core::{
    default_replay_seed_scene, replay_intent_script, AppCore, MeshClip, PickFrame, RebuildOutcome,
    RebuildOutput, SelectedOperandGhost,
};
pub use artifacts::{DocumentArtifact, Dump, SettingsArtifact, ViewArtifact};
pub use assets::{CubeFaceSlot, FaceProvenance, FaceTextures};
pub use camera::{
    adjacent_face, chrome_zone_left_click_action, classify_cube_point, nearest_equivalent_theta,
    ArrowDir, ChromeClickAction, CubeChromeZone, CubeFace, CubeRect, HomeView, OrbitCamera,
    OrbitType, ProjectionMode, RollDir, SnapTween, ViewCubeElement, CUBE_FACES, POLE_EPSILON,
};
pub use display::brick::{
    build_brick_field, build_brick_field_all_blocks, build_brick_field_with_tiles,
    pack_clipmap_level_keys, pack_world_block_key, read_back_brick_atlas, unpack_world_block_key,
    upload_brick_atlas, upload_brick_cell_key_atlas, BrickCellKeyTile, BrickFieldBuild,
    BrickFieldUpdate, BrickPayload, BrickRecord, ClipmapLevel, ClipmapPyramid,
    IncrementalBrickField, SculptedAtlasGeometry, SculptedAtlasPayload,
    SculptedCellKeyAtlasGeometry, SculptedCellKeyAtlasPayload, CELL_KEY_TEXEL_BYTES,
    CLIPMAP_LEVEL_1_BLOCKS_PER_CELL, CLIPMAP_LEVEL_2_BLOCKS_PER_CELL,
    CLIPMAP_LEVEL_3_BLOCKS_PER_CELL,
};
pub use display::brick::{
    cpu_brick_hit_material, cpu_march_brick_field, cpu_march_brick_field_counted,
    cpu_march_exact_occupancy, cpu_march_levels_counted, pack_gpu_records, BrickGpuRecord,
    BrickMarchFrame, BrickRaymarchRenderer, CpuMarchHit, NON_RESIDENT_ATLAS_SLOT,
};
pub use display::mesh::{
    build_cuboid_mesh, CuboidMesh, CuboidMeshRenderer, SelectedOperandGhostBody,
    SelectedOperandGhostRenderer,
};
pub use display::renderer::procedural_material_average_color;
pub use display::renderer::{
    create_depth_view, create_msaa_color_view, view_cube_corner, InfiniteGridRenderer, LayerBand,
    MaterialSource, OnionFogParams, PlacementGhostRenderer, PointsRenderer, RegionClip, RegionRole,
    SceneGridRenderer, TransformGizmoRenderer, ViewCubeRenderer, DEPTH_FORMAT, MSAA_SAMPLE_COUNT,
    PLACEMENT_GHOST_TINT, VIEW_CUBE_VIEWPORT_PIXELS,
};
pub use display::texture_atlas::{AtlasSubRect, MaterialAtlas};
pub use document::debug_clouds::DebugCloudField;
pub use document::intent::{Intent, IntentEffect, NodeSpec};
pub use document::scene::{
    AssemblyDef, CombineOp, DefId, Node, NodeBuilder, NodeContent, NodeId, NodePath, NodeTransform,
    Point, RegionBlocks, Scene, VoxelBody, ROOT_NODE_ID,
};
pub use evaluation::chunk_storage::{compress, decompress, CompressedChunk, Occupancy, SparseCell};
pub use evaluation::disk_chunk_store::{DiskChunkStore, DiskChunkStoreStats};
pub use evaluation::store::{ChunkCacheKey, ChunkResolveCache, Store};
pub use evaluation::two_layer_store::{
    stream_vox_occupancy, streamed_widest_run_in_band, BlockClassification, MicroblockGeometry,
    SeamSolidity, TwoLayerChunk, TwoLayerResidentCache, TwoLayerStore,
};
pub use gpu::GpuContext;
pub use settings::AppConfig;
pub use ui::panel::{
    build_add_shape_dialog, build_panel, build_signal_stack, cube_right_inset_points, ArmedTool,
    ExportPanelState, LayerRange, PanelResponse, PanelState, PlacementGhost, Selection,
    SelectionTarget, SignalStackState, ViewMode,
};
pub use voxel_core::core_geom::MaterialChoice;
pub use work::engagement::orchestrator::{DisplayOrchestrator, DisplayRefreshContext};
pub use work::engagement::routing::{
    brick_display_handover, brick_patch_in_place, route_brick_rebuild, route_geometry_rebuild,
    route_mesh_build, BrickDisplayHandover, BrickRebuildAction, EditShape, GenerationTracker,
    MeshBuildRoute, RebuildRoute, ASYNC_REBUILD_CHUNK_THRESHOLD,
};
pub use work::workers::brick::{
    build_brick_rebuild, spawn_brick_worker, BrickDisplayInstall, BrickRebuildOutcome,
    BrickRebuildRequest, BrickRebuildResult, BrickWorker,
};
pub use work::workers::diameter::{
    spawn_diameter_worker, DiameterRequest, DiameterResult, DiameterWorker,
};
pub use work::workers::export::{
    spawn_vox_export_worker, VoxExportRequest, VoxExportResult, VoxExportSummary, VoxExportWorker,
};
pub use work::workers::geometry::{
    build_geometry, spawn_geometry_worker, GeometryRebuildRequest, GeometryRebuildResult,
    GeometryWorker,
};
pub use work::workers::Worker;
// The dense whole-region resolve oracle is compile-gated out of production builds
// (see the proof chapter's "Oracles" section, `docs/architecture/05-proof.md`).
// `cfg(test)` only: this crate has no `oracle` feature since `shot` became its own
// package, and `shot` reaches the resolver through `evaluation` directly rather than
// through this re-export. The dev-dependency on `evaluation` supplies the feature.
pub use document::sketch::{
    Circle, Operation, PlaneAxis, RevolveAxis, Sketch, SketchLength, SketchPoint, SketchSolid,
};
#[cfg(test)]
pub use evaluation::two_layer_store::resolve_region_two_layer;
pub use voxel_core::spatial_index::{LeafEntry, LeafFingerprint, LeafSpatialIndex, VoxelAabb};
// The headless `.vox` export sink lives in the `interchange` crate; re-exported flat so `voxel_worker::VoxExport` / `VoxExportBuilder` keep resolving.
pub use interchange::vox_export::{VoxExport, VoxExportBuilder};
// Value vocabulary lives in the voxel_core crate; the producer half in the document crate.
// Both are re-exported flat so `voxel_worker::Voxel`, `voxel_worker::SdfShape`, etc. keep
// resolving for the bins and integration tests.
pub use document::voxel::{GeometryParams, SdfShape, VoxelProducer};
pub use voxel_core::voxel::{
    widest_run_in_band_over_chunks, RecenterVoxels, ShapeKind, Voxel, VoxelGrid,
};

/// Surface / offscreen color format used everywhere in the project.
///
/// Using the same sRGB format for the windowed surface and the headless capture
/// texture keeps the screenshot identical to the window.
pub const COLOR_TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// The warm-dark "workshop" clear color.
///
/// These are *linear* component values handed to wgpu; with an sRGB render
/// target the GPU encodes them back to sRGB on write, so the perceived color is
/// a warm near-black with a faint copper cast.
pub const WORKSHOP_CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.030,
    g: 0.024,
    b: 0.018,
    a: 1.0,
};

// The per-frame pipeline: egui pass ([`frame::egui_frame`]) + GPU viewport pass
// ([`frame::render`]). A public module, NOT re-exported flat: callers name
// `voxel_worker::frame::render::render_frame` etc. directly.
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
    clippy::too_many_lines
)]
pub mod frame;
