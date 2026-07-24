//! VoxelWorker — native Rust port of the Vintage Story chiseling planner.
//!
//! This crate is the rendering foundation shared by both the windowed application
//! (`src/main.rs`) and the headless screenshot harness, which is now its own package
//! at `crates/shot/`:
//!
//!   * A render-target-agnostic frame function ([`render_frame`]) that paints
//!     into any [`wgpu::TextureView`]. It knows nothing about winit or surfaces,
//!     so the same code draws the on-screen surface texture and the offscreen
//!     capture texture — guaranteeing the screenshot matches the window.
//!   * A single egui panel builder ([`build_panel`]) used by both paths so the
//!     captured frame is identical to the live one.
//!   * The warm-dark "workshop" colour identity (`docs/design/colour-vocabulary.md`).

// A public item's doc may link to a private helper to explain how the two relate; that
// cross-reference is deliberate and stays a navigable link under `--document-private-items`.
// The CI doc gate denies broken and redundant links but permits these.
#![allow(rustdoc::private_intra_doc_links)]
// Colours live in `ui::theme::color_palette`; a raw `Color32::from_*` elsewhere is an error.
#![deny(clippy::disallowed_methods)]

// ADR 0003 keystone: headless orchestrator (scene + store + camera). See app_core.rs.
pub mod app_core;
// The shell's palette GPU host (`PaletteHost`): it owns the wgpu backing the UI-facing
// palette cannot name (the `crate::thumbnail::ThumbnailRenderer`, the texture
// keep-alives, the scanned `BlockGroup`s) and keeps them index-aligned with the
// `ui::palette::BlockPalette` tiles it renders + registers into egui (ADR 0016 Phase 8b
// — the egui-facing palette state + the inspector panel moved to the `ui` crate).
pub mod block_palette;
pub mod gpu;
// The three persistence artifacts (ADR 0022) and the exhaustive captures that carry
// classified state into them. Separate from `settings` on purpose: that module holds the
// classified state record, this one holds where it goes and enforces that it gets there.
pub mod artifacts;
pub mod settings;
// The palette PREVIEW thumbnail renderer: a shell-side GPU sink that draws the UI's
// 45° cube tiles (NOT the scene), reaching down into `display` only for the shared
// block-texture bind-group layout. Kept out of the `display` scene-view crate.
pub mod thumbnail;
// The windowed application (the default binary's logic): `WindowedState` + `App` + the winit
// `ApplicationHandler` + per-frame render + async-worker poll seams. Carved out of `src/main.rs`
// into a shell LIB module tree (ADR 0016) so the bin is a thin `windowed::run` entry point; the
// lib already carries the winit/egui/wgpu deps this needs.
pub mod windowed;
// The engagement state machine + the async worker pool moved to the `work` crate at the ADR 0016
// Phase 6 cut (`{display, interchange} <- work <- shell`); their types are re-exported flat below
// so the shell's `voxel_worker::<Name>` uses keep resolving.

#[cfg(test)]
mod windowed_resolve_tests;

// ADR 0010 E1: the standalone exactness parity for the conservative cell-interval
// bound primitive (VoxelProducer::cell_field_interval) + the CSG interval composition.
#[cfg(test)]
mod cell_interval_parity_tests;

pub use app_core::{
    default_replay_seed_scene, replay_intent_script, AppCore, MeshClip, PickFrame, RebuildOutcome,
    RebuildOutput, SelectedOperandGhost,
};
pub use evaluation::store::{ChunkCacheKey, ChunkResolveCache, Store};
pub use display::brick::{
    build_brick_field, build_brick_field_all_blocks, build_brick_field_with_tiles,
    pack_clipmap_level_keys, pack_world_block_key,
    read_back_brick_atlas, unpack_world_block_key, upload_brick_atlas,
    upload_brick_cell_key_atlas, BrickCellKeyTile, BrickFieldBuild,
    BrickFieldUpdate, BrickPayload, BrickRecord, ClipmapLevel, ClipmapPyramid,
    IncrementalBrickField, SculptedAtlasGeometry, SculptedAtlasPayload,
    SculptedCellKeyAtlasGeometry, SculptedCellKeyAtlasPayload,
    CELL_KEY_TEXEL_BYTES, CLIPMAP_LEVEL_1_BLOCKS_PER_CELL, CLIPMAP_LEVEL_2_BLOCKS_PER_CELL,
    CLIPMAP_LEVEL_3_BLOCKS_PER_CELL,
};
pub use display::brick::{
    cpu_brick_hit_material, cpu_march_brick_field, cpu_march_brick_field_counted,
    cpu_march_levels_counted, cpu_march_exact_occupancy,
    pack_gpu_records, BrickGpuRecord,
    BrickMarchFrame, BrickRaymarchRenderer, CpuMarchHit, NON_RESIDENT_ATLAS_SLOT,
};
pub use work::workers::brick::{
    build_brick_rebuild, spawn_brick_worker, BrickDisplayInstall, BrickRebuildOutcome,
    BrickRebuildRequest, BrickRebuildResult, BrickWorker,
};
pub use work::engagement::orchestrator::{DisplayOrchestrator, DisplayRefreshContext};
pub use work::engagement::routing::{
    brick_display_handover, brick_patch_in_place, route_brick_rebuild, route_geometry_rebuild,
    route_mesh_build, BrickDisplayHandover, BrickRebuildAction, EditShape, GenerationTracker,
    MeshBuildRoute, RebuildRoute, ASYNC_REBUILD_CHUNK_THRESHOLD,
};
pub use evaluation::chunk_storage::{compress, decompress, CompressedChunk, Occupancy, SparseCell};
pub use evaluation::disk_chunk_store::{DiskChunkStore, DiskChunkStoreStats};
pub use display::mesh::{
    build_cuboid_mesh, CuboidMesh, CuboidMeshRenderer, SelectedOperandGhostBody,
    SelectedOperandGhostRenderer,
};
pub use work::workers::geometry::{
    build_geometry, spawn_geometry_worker, GeometryRebuildRequest, GeometryRebuildResult,
    GeometryWorker,
};
pub use work::workers::diameter::{
    spawn_diameter_worker, DiameterRequest, DiameterResult, DiameterWorker,
};
pub use work::workers::export::{
    spawn_vox_export_worker, VoxExportRequest, VoxExportResult, VoxExportSummary, VoxExportWorker,
};
pub use work::workers::Worker;
pub use display::texture_atlas::{AtlasSubRect, MaterialAtlas};
pub use document::debug_clouds::DebugCloudField;
pub use camera::{
    adjacent_face, chrome_zone_left_click_action, classify_cube_point,
    nearest_equivalent_theta, ArrowDir, ChromeClickAction, CubeChromeZone, CubeFace, CubeRect,
    HomeView, OrbitCamera, ProjectionMode,
    RollDir, SnapTween, ViewCubeElement, CUBE_FACES, POLE_EPSILON,
};
pub use gpu::GpuContext;
pub use document::intent::{Intent, IntentEffect, NodeSpec};
pub use voxel_core::core_geom::MaterialChoice;
pub use ui::panel::{
    build_add_shape_dialog, build_panel, build_signal_stack, cube_right_inset_points,
    ExportPanelState, LayerRange, PanelResponse, PanelState, PlacementGhost, SignalStackState,
    ViewMode,
};
pub use assets::{CubeFaceSlot, FaceProvenance, FaceTextures};
pub use display::renderer::{
    create_depth_view, create_msaa_color_view, view_cube_corner, InfiniteGridRenderer, LayerBand,
    MaterialSource, OnionFogParams, PlacementGhostRenderer, PointsRenderer, RegionClip, RegionRole,
    SceneGridRenderer, TransformGizmoRenderer, ViewCubeRenderer, DEPTH_FORMAT, MSAA_SAMPLE_COUNT,
    PLACEMENT_GHOST_TINT, VIEW_CUBE_VIEWPORT_PIXELS,
};
pub use display::renderer::procedural_material_average_color;
pub use document::scene::{
    AssemblyDef, CombineOp, DefId, Node, NodeBuilder, NodeContent, NodeId, NodePath, NodeTransform,
    VoxelBody, Point, RegionBlocks, Scene, ROOT_NODE_ID,
};
pub use artifacts::{DocumentArtifact, Dump, SettingsArtifact, ViewArtifact};
pub use settings::AppConfig;
pub use evaluation::two_layer_store::{
    stream_vox_occupancy, streamed_widest_run_in_band, BlockClassification, MicroblockGeometry,
    SeamSolidity, TwoLayerChunk, TwoLayerResidentCache, TwoLayerStore,
};
// The dense whole-region resolve oracle is compile-gated out of production builds
// (see the proof chapter's "Oracles" section, `docs/architecture/05-proof.md`).
// `cfg(test)` only: this crate has no `oracle` feature since `shot` became its own
// package, and `shot` reaches the resolver through `evaluation` directly rather than
// through this re-export. The dev-dependency on `evaluation` supplies the feature.
#[cfg(test)]
pub use evaluation::two_layer_store::resolve_region_two_layer;
pub use document::sketch::{Operation, PlaneAxis, RevolveAxis, Sketch, SketchPoint, SketchSolid};
pub use voxel_core::spatial_index::{LeafEntry, LeafFingerprint, LeafSpatialIndex, VoxelAabb};
// The headless `.vox` export sink now lives in the `interchange` crate (ADR 0016 Phase 5);
// re-exported flat so `voxel_worker::VoxExport` / `VoxExportBuilder` keep resolving.
pub use interchange::vox_export::{VoxExport, VoxExportBuilder};
// Value vocabulary lives in the voxel_core crate; the producer half now lives in the
// document crate. Both are re-exported flat so `voxel_worker::Voxel`, `voxel_worker::SdfShape`,
// etc. keep resolving for the bins and integration tests.
pub use document::voxel::{GeometryParams, SdfShape, VoxelProducer};
pub use voxel_core::voxel::{
    widest_run_in_band_over_chunks, RecentreVoxels, ShapeKind, Voxel, VoxelGrid,
};

/// Surface / offscreen colour format used everywhere in the project.
///
/// Using the same sRGB format for the windowed surface and the headless capture
/// texture keeps the screenshot identical to the window (Hard requirement #9).
pub const COLOR_TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// The warm-dark "workshop" clear colour (`docs/design/colour-vocabulary.md`).
///
/// These are *linear* component values handed to wgpu; with an sRGB render
/// target the GPU encodes them back to sRGB on write, so the perceived colour is
/// a warm near-black with a faint copper cast.
pub const WORKSHOP_CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.030,
    g: 0.024,
    b: 0.018,
    a: 1.0,
};

// The per-frame pipeline (ADR 0031): egui pass + GPU viewport pass. Split out of this root so
// the two responsibilities stop sharing a file; re-exported flat so `voxel_worker::<name>` uses
// keep resolving for the bins, the shot harness, and the tests.
mod frame;
pub use frame::egui_frame::{
    run_egui_frame, EguiPaintBridge, PreparedEguiFrame, ViewCubeMenuRequest,
};
pub use frame::render::{render_frame, FramePhases};
