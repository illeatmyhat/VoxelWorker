//! Config persistence (Milestone 8) — the **classified state record**.
//!
//! [`AppConfig`] is a flat, self-contained mirror of everything the application
//! persists, captured from the live render-coupled `PanelState` rather than derived on
//! it. That indirection is what keeps internal struct churn away from anything durable,
//! and it is why every field can carry a `#[snapshot(...)]` category (ADR 0022) in one
//! readable column.
//!
//! What this type is *not*, since 2026-07-20, is an on-disk format. It has no serde
//! derive at all. The artifacts it is carried into — the document, the settings, and the
//! dump that is their superset — live in [`crate::artifacts`], and every one of them
//! destructures this struct exhaustively. That is the whole mechanism: the category on a
//! field records where it should go, and the destructuring next door refuses to compile
//! if it does not get there. A struct that both classified its fields and serialized
//! itself would have no place for that second check to live, which is how `orbit_target`
//! went missing from the F9 dump in the first place.
//!
//! Loading still never panics: a missing file, an unreadable file, or invalid JSON all
//! yield `None`, and the caller uses its built-in defaults.

use camera::{HomeView, OrbitCamera, ProjectionMode};
use voxel_core::core_geom::MaterialChoice;
use voxel_core::voxel::ShapeKind;
use ui::panel::{
    LayerRange, PanelState, PlacementGhost, PlacementSnap, SignalStackState, SketchTool, ViewMode,
};
use document::scene::{NodeContent, NodeId, Scene};
use document::voxel::{GeometryParams, SdfShape};

/// The serde-able mirror of the armed-tool [`PlacementGhost`] (ADR 0022), carried in the
/// session dump so a repro taken mid-gesture replays the pending drop.
///
/// [`PlacementGhost`] lives in the `ui` crate, which links no serde (ADR 0016's crate
/// law), so — like [`ViewMode`] and the Signal stack — it is persisted from out here. It
/// stores the armed primitive's kind/size/wall plus the ABSOLUTE corner-anchored offset
/// the node would take (`Intent::PlaceNode`'s frame); the render-frame centre is re-derived
/// from the live recentre at draw time, never stored.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PlacementGhostConfig {
    /// The armed primitive kind (`ShapeKind` carries its own serde).
    pub shape_kind: ShapeKind,
    /// Bounding-box size in voxels at the document density.
    pub size_voxels: [u32; 3],
    /// Tube wall thickness in whole blocks (Tube only).
    #[serde(default = "default_wall_blocks")]
    pub wall_blocks: u32,
    /// Absolute, corner-anchored voxel offset the node would drop at.
    pub offset_voxels: [i64; 3],
}

/// Default wall thickness for a partial config missing the key (mirrors `SdfShape`'s).
fn default_wall_blocks() -> u32 {
    1
}

impl PlacementGhostConfig {
    /// Capture the config mirror from the live [`PlacementGhost`].
    pub fn from_ghost(ghost: &PlacementGhost) -> Self {
        Self {
            shape_kind: ghost.shape.kind,
            size_voxels: ghost.shape.size_voxels,
            wall_blocks: ghost.shape.wall_blocks,
            offset_voxels: ghost.offset_voxels,
        }
    }

    /// Rebuild the runtime [`PlacementGhost`] this config describes. The shape is rebuilt
    /// from pure voxels (a ghost is a geometry preview; the authored size expression is
    /// not needed to trace its surface).
    pub fn to_ghost(&self) -> PlacementGhost {
        PlacementGhost {
            shape: SdfShape::from_voxels(self.shape_kind, self.size_voxels, self.wall_blocks),
            offset_voxels: self.offset_voxels,
            // The persisted ghost config does not carry the sub-voxel remainder or rotation (ADR
            // 0027); an F9 repro of an armed tilted / off-block ghost previews it upright and
            // voxel-aligned. The placed nodes it captures keep their full transform in the scene
            // tree — only the transient armed-ghost tilt/fraction is dropped.
            offset_local: [0.0, 0.0, 0.0],
            rotation: glam::Quat::IDENTITY,
        }
    }
}

use crate::artifacts::{
    default_density, default_distance, default_onion_depth, default_phi, default_theta,
    default_window_size, Dump,
};

/// The whole persisted application state, one field per decision, each classified.
///
/// It used to be the single persistence artifact as well — config file, project and debug
/// repro all at once — which is the arrangement ADR 0022 unpicked. It is now the *source*
/// the artifacts are captured from and the *target* a loaded dump is restored into, which
/// leaves it doing exactly one job: being the complete, reviewable list of what the
/// application considers durable state.
#[derive(Debug, Clone, PartialEq, snapshot::Snapshot)]
pub struct AppConfig {
    // --- scene (ADR 0001 step 8: full scene persistence) ---
    // The whole assembly (node tree + reusable definitions + the active
    // selection) is persisted here. It is the ONE `#[snapshot(document)]` field, so
    // `DocumentArtifact` is built from it alone. An absent `scene` key on disk
    // deserialises to `None`, which loads the default seed scene in `to_panel_state`
    // (the same one a brand-new config gets). A malformed/partial `scene` value can
    // never reach this field as garbage: serde tolerates missing inner fields (every
    // scene field is `#[serde(default)]`), and an outright unparseable config is
    // rejected wholesale by `load()` → defaults. Density
    // (`voxels_per_block`) is now a document-level attribute on the `scene` (ADR 0003
    // §3f(0)); the app-level field below persists the inspector slider's transient
    // mirror value, kept in sync with `scene.voxels_per_block` via `SetDensity`.
    //
    // issue #32: the flat `shape` / `size_blocks` / `wall_blocks` geometry mirror
    // fields were deleted (no config back-compat — see #31). They previously built a
    // one-Tool-node scene when `scene` was absent, but the current build always
    // writes a `scene`, so they were dead for live configs. An OLD config still
    // carrying those keys (plus the old `debug_clouds`) loads fine: there is no
    // `deny_unknown_fields`, so serde ignores the now-unknown keys, and a scene-less
    // config just loads the default seed scene.
    //
    // regional export: deferred to the chunking milestone (ADR 0001 step 8's
    // "regional/streamed .vox export" sub-part — meaningless until chunking; the
    // current full-grid `.vox` export already covers bounded scenes).
    #[snapshot(document)]
    pub scene: Option<Scene>,

    // --- density (the inspector slider's persisted mirror; the document truth is
    // `scene.voxels_per_block`, ADR 0003 §3f(0)) ---
    // View, not document: the truth is `scene.voxels_per_block`, and a mirror of document
    // truth is where you are working rather than what the model is.
    #[snapshot(view)]
    pub voxels_per_block: u32,

    // --- display / material ---
    // `ProjectionMode` lives in the wgpu-free `camera` crate, which carries no serde
    // dependency (its boundary law keeps it to glam + substrate), so it is persisted
    // through the `ProjectionModeConfig` remote-derive shim in `crate::artifacts`.
    #[snapshot(settings)]
    pub projection_mode: ProjectionMode,
    #[snapshot(settings)]
    pub material: MaterialChoice,
    // Issue #31: the three legacy grid `show_*` mirror fields
    // (`show_grid_overlay` / `show_block_lattice` / `show_floor_grid`) were deleted.
    // The grid masters now persist as the single source of truth on the `scene`
    // field (`scene.master_voxel_grid` / `master_block_lattice` / `master_floor_grid`).
    // No `deny_unknown_fields`, so an OLD config still carrying those keys loads fine
    // (serde ignores the now-unknown keys); the scene's own masters are authoritative.
    #[snapshot(settings)]
    pub show_view_cube: bool,
    /// Whether the Points' axes draw on top of the model vs occluded (ADR 0031). ON by
    /// default; the same display-preference footing as `show_view_cube`.
    #[snapshot(settings)]
    pub axes_on_top: bool,
    // NOTE: the legacy `show_origin_gizmo` field was removed in the issue #29 S6
    // cleanup. The old origin-gizmo Display toggle was replaced by the
    // selection-driven transform gizmo, so the field drove nothing. There is no
    // `deny_unknown_fields`, so an OLD config still carrying `"show_origin_gizmo"`
    // continues to deserialize cleanly (serde ignores the now-unknown key).
    /// Best-effort applied-block label (re-applied lazily; see module docs).
    #[snapshot(settings)]
    pub applied_block_label: Option<String>,

    // --- layer-range scrubber (issue #12) ---
    // The bounds themselves depend on the live grid_z (Z-up: layers are Z-slices), so
    // they are NOT persisted
    // (they always re-derive to the full range on load); only the sticky control
    // preferences are saved here.
    // The flattened mirror of `PanelState::layer_range`, classified there as one view
    // object; view here too so the two agree. A collaborator should not inherit the band
    // someone else was working in.
    #[snapshot(view)]
    pub snap_to_blocks: bool,
    #[snapshot(view)]
    pub onion_skin: bool,
    #[snapshot(view)]
    pub onion_depth: u32,

    // --- camera ---
    // The live camera pose: view state. `orbit_target` below is the field that went
    // missing from the dump, and the reason this scheme exists.
    #[snapshot(view)]
    pub orbit_theta: f32,
    #[snapshot(view)]
    pub orbit_phi: f32,
    #[snapshot(view)]
    pub orbit_distance: f32,
    /// The orbit TARGET (the world point the camera looks at / orbits). Panning moves it off
    /// the origin, so without it a repro reframes on the origin and misses a panned view (the
    /// F9 `--from-config` flow). A dump written without the key restores `[0,0,0]`, matching
    /// the pre-field behaviour (target defaulted to the origin).
    #[snapshot(view)]
    pub orbit_target: [f32; 3],

    // --- view-cube home view (#13) ---
    // The Home button's saved view. A dump missing these keys loads the camera
    // defaults — the artifact's per-field default fns derive from
    // `OrbitCamera::default()`, so a persisted default can never drift from the live one.
    // Settings, unlike the orbit fields above: a Home view is a viewpoint the user chose
    // to KEEP, so it outlives any one project rather than describing this session.
    #[snapshot(settings)]
    pub home_theta: f32,
    #[snapshot(settings)]
    pub home_phi: f32,
    #[snapshot(settings)]
    pub home_distance: f32,
    /// #13 Step 6.4: was the home view explicitly captured by the user? When
    /// `false` (the default), the Home button re-frames the model instead of using
    /// `home_distance`, so a default home never zooms in too close.
    #[snapshot(settings)]
    pub home_explicit: bool,

    // --- window ---
    #[snapshot(settings)]
    pub window_size: [u32; 2],

    // --- session: how the workspace was left (ADR 0024) ---
    // These four were classified on `PanelState` as reaching the dump and reached
    // nothing: `to_panel_state` hard-coded every one of them to a default, and no field
    // here carried them. That is the pan-target bug exactly — a decision recorded at the
    // field and never honoured by a capture — and it survived because the
    // `PanelState` -> `AppConfig` seam is hand-written, which is the gap ADR 0022's third
    // amendment recorded. These fields are that seam being closed.
    //
    // Session rather than view: they say how the workspace was arranged, not where the
    // camera was. Session rather than settings: nobody chose them, they are merely where
    // the user stopped, and a preference is something you would want honoured in every
    // project.
    /// The viewer's exclusive rendering mode. ADR 0018 decision 3 ruled it out of the
    /// document, which stands; ADR 0024 supersedes the part where that was implemented as
    /// out of persistence altogether.
    #[snapshot(session)]
    pub view_mode: ViewMode,
    /// The Signal display stack's fold + per-section open state (issue #88), carried
    /// whole rather than as a hand-picked subset.
    #[snapshot(session)]
    pub stack: SignalStackState,
    /// Face-orientation debug shading.
    #[snapshot(session)]
    pub debug_face_orientation: bool,
    /// The brick-raymarch grazing-rim diagnostic. A repro of a rendering fault that drops
    /// the diagnostic the fault was visible under is not a repro.
    #[snapshot(session)]
    pub debug_brick_faces: bool,
    /// The armed-tool placement ghost (ADR 0022), `None` when no tool is armed. Session
    /// state on the same footing as [`view_mode`](Self::view_mode): an armed drop is how
    /// the workspace was left, so a mid-gesture dump replays it. Named `placement_ghost` to
    /// match the [`PanelState`] field it routes to (the ADR 0024 seam guard keys on the
    /// name).
    #[snapshot(session)]
    pub placement_ghost: Option<PlacementGhostConfig>,

    /// The armed-tool placement snap settings (owner ruling 2026-07-21): position (no snap /
    /// block / voxel) and orientation (no snap / surface). Session state — durable across adds
    /// and relaunch. The `SessionArtifact` serde default degrades to the finished defaults
    /// (voxel + surface) for a dump written before the field existed.
    #[snapshot(session)]
    pub placement_snap: PlacementSnap,

    /// The sketch node being edited in sketch mode (ADR 0028), `None` in the normal chrome.
    /// Session state on the same footing as [`placement_ghost`](Self::placement_ghost): the
    /// mode is how the workspace was left, so a mid-edit dump (an F9 repro) re-enters the same
    /// sketch. Named `sketch_mode` to match the [`PanelState`] field it routes to. `NodeId` is
    /// serde-able, so it needs no `Config` mirror. The `SessionArtifact` serde default
    /// degrades a pre-field dump to `None` (the normal chrome).
    #[snapshot(session)]
    pub sketch_mode: Option<NodeId>,

    /// The armed sketch-mode tool (ADR 0028, #95), restored so a mid-edit dump re-enters the
    /// mode with the same tool in hand. Session state alongside [`sketch_mode`](Self::sketch_mode);
    /// `SketchTool` lives in the serde-free `ui` crate, so `SessionArtifact` persists it through a
    /// `SketchToolConfig` remote shim exactly as it does the viewer mode. Named `sketch_tool` to
    /// match the [`PanelState`] field it routes to.
    #[snapshot(session)]
    pub sketch_tool: SketchTool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            scene: None,
            voxels_per_block: default_density(),
            projection_mode: ProjectionMode::default(),
            material: MaterialChoice::default(),
            show_view_cube: true,
            axes_on_top: true,
            applied_block_label: None,
            snap_to_blocks: true,
            onion_skin: false,
            onion_depth: default_onion_depth(),
            orbit_theta: default_theta(),
            orbit_phi: default_phi(),
            orbit_distance: default_distance(),
            orbit_target: [0.0, 0.0, 0.0],
            home_theta: default_theta(),
            home_phi: default_phi(),
            home_distance: default_distance(),
            home_explicit: false,
            window_size: default_window_size(),
            view_mode: ViewMode::default(),
            placement_snap: PlacementSnap::default(),
            stack: SignalStackState::default(),
            debug_face_orientation: false,
            debug_brick_faces: false,
            placement_ghost: None,
            sketch_mode: None,
            sketch_tool: SketchTool::default(),
        }
    }
}

impl AppConfig {
    /// Capture the persisted fields from the live [`PanelState`], [`OrbitCamera`],
    /// the saved [`HomeView`] (#13) and the current window size.
    pub fn capture(
        panel: &PanelState,
        camera: &OrbitCamera,
        home_view: HomeView,
        window_size: [u32; 2],
    ) -> Self {
        Self {
            // step 8: persist the whole scene (node tree + definitions + active
            // selection). issue #32: the legacy flat geometry mirror fields are gone;
            // only the app-level density rides alongside the scene.
            scene: Some(panel.scene.clone()),
            voxels_per_block: panel.geometry.voxels_per_block,
            projection_mode: panel.projection_mode,
            material: panel.material,
            // Issue #31: the three grid masters persist as the single source of
            // truth on the `scene` field above (`scene.master_*`). The legacy
            // `show_grid_overlay` / `show_block_lattice` / `show_floor_grid` mirror
            // fields were deleted, so there is no stale mirror to drift out of sync.
            show_view_cube: panel.show_view_cube,
            axes_on_top: panel.axes_on_top,
            applied_block_label: panel.applied_block_label.clone(),
            snap_to_blocks: panel.layer_range.snap_to_blocks,
            onion_skin: panel.layer_range.onion_skin,
            onion_depth: panel.layer_range.onion_depth,
            orbit_theta: camera.orbit_theta,
            orbit_phi: camera.orbit_phi,
            orbit_distance: camera.orbit_distance,
            orbit_target: camera.target.to_array(),
            home_theta: home_view.theta,
            home_phi: home_view.phi,
            home_distance: home_view.distance,
            home_explicit: home_view.explicitly_set,
            window_size,
            // ADR 0024: the session fields. Read straight off the panel, which is all
            // they ever needed — the omission was never subtle, it was simply never
            // forced to be noticed.
            view_mode: panel.view_mode,
            placement_snap: panel.placement_snap,
            stack: panel.stack,
            debug_face_orientation: panel.debug_face_orientation,
            debug_brick_faces: panel.debug_brick_faces,
            // ADR 0022: the armed placement ghost, captured as its serde-able mirror so a
            // mid-gesture dump replays the pending drop.
            placement_ghost: panel.placement_ghost.as_ref().map(PlacementGhostConfig::from_ghost),
            sketch_tool: panel.sketch_tool,
            // ADR 0028: the sketch node under edit, so a mid-edit dump re-enters sketch mode.
            sketch_mode: panel.sketch_mode,
        }
    }

    /// The persisted [`HomeView`] (#13) — the saved Home-button view restored on
    /// load. An EXPLICIT home (the user pressed "set home") is honoured verbatim; an
    /// IMPLICIT home (never captured) always tracks the CURRENT code default instead
    /// of whatever angles a prior session happened to persist, so changing the
    /// default Home angle takes effect even on an existing config (no stale TOP view
    /// lingering from a pre-change save).
    pub fn home_view(&self) -> HomeView {
        if self.home_explicit {
            HomeView {
                theta: self.home_theta,
                phi: self.home_phi,
                distance: self.home_distance,
                explicitly_set: true,
            }
        } else {
            HomeView::default()
        }
    }

    /// Build the [`PanelState`] this config describes.
    ///
    /// step 8 (ADR 0001): the full scene (node tree + definitions + active
    /// selection) is restored from [`scene`](Self::scene) when present. When it is
    /// absent — an OLD config that predates scene persistence (issue #32 deleted the
    /// flat geometry mirror fields) — the default seed scene is loaded, the same one
    /// a brand-new config gets, via [`PanelState::seed_scene_from_geometry`]. A
    /// restored scene that resolves to no nodes (a malformed/empty persisted scene)
    /// also falls back to that seed, so loading never yields an empty document and
    /// never panics. Only the app-level density carries over from the config.
    pub fn to_panel_state(&self) -> PanelState {
        let mut state = PanelState {
            // issue #32: the flat geometry mirror fields are gone. The inspector
            // mirror starts at its defaults, overridden only by the persisted
            // app-level density; it is re-synced from the active node after the seed.
            // Size is voxel-granular (ADR 0003 §3f(0)): the default seed is 5×1×5
            // BLOCKS, so build its canonical voxels at the PERSISTED density (not the
            // Default impl's hardcoded d16) so a config at d20 still seeds a 5-block
            // shape, matching the old block-granular seed.
            geometry: GeometryParams {
                size_voxels: [
                    5 * self.voxels_per_block.max(1),
                    self.voxels_per_block.max(1),
                    5 * self.voxels_per_block.max(1),
                ],
                voxels_per_block: self.voxels_per_block,
                ..GeometryParams::default()
            },
            projection_mode: self.projection_mode,
            material: self.material,
            show_view_cube: self.show_view_cube,
            axes_on_top: self.axes_on_top,
            // ADR 0024: the debug verification modes are session state and are restored,
            // not reset. They used to be hard-coded to `false` here while classified as
            // reaching the dump — a category promising one thing and the code doing
            // another, which is the failure mode the classification exists to make
            // visible rather than one it is allowed to have.
            debug_face_orientation: self.debug_face_orientation,
            debug_brick_faces: self.debug_brick_faces,
            voxel_cap_warning_millions: None,
            // Re-applied lazily/best-effort: only the label is restored (for the
            // panel readout); the material itself reverts to procedural.
            applied_block_label: self.applied_block_label.clone(),
            // Issue #12: only the sticky control prefs persist; the band bounds
            // are re-derived to the full range against the live grid_z on load.
            layer_range: LayerRange {
                lower: 0,
                upper: 0, // rescaled to grid_z by the caller after the grid resolves.
                snap_to_blocks: self.snap_to_blocks,
                onion_skin: self.onion_skin,
                onion_depth: self.onion_depth.clamp(1, 8),
            },
            // ADR 0024, superseding ADR 0018 decision 3: the viewer mode stays out of the
            // document and is restored across relaunch. Decision 3 said "not saved with
            // the scene"; this always honoured that, and the reset was the wider claim
            // nobody made.
            view_mode: self.view_mode,
            placement_snap: self.placement_snap,
            // Issue #88's stack state travels with it, for the same reason and whole.
            stack: self.stack,
            // Restored just below: the persisted full scene, or — for an old
            // config without one — the default seed scene.
            scene: Scene::default(),
            // Issue #29 S5: refreshed each frame from the camera target by the windowed
            // caller; defaults to the world origin (the headless harness keeps it 0).
            point_add_position_blocks: [0, 0, 0],
            // ADR 0022: restore the armed placement ghost (session state), so a mid-gesture
            // repro replays the pending drop rather than resetting to no armed tool.
            placement_ghost: self.placement_ghost.as_ref().map(PlacementGhostConfig::to_ghost),
            // ADR 0028: re-enter sketch mode on the same node a mid-edit dump was taken in.
            // Cleared to `None` below if the id no longer resolves in the restored scene, so a
            // stale sketch node can never trap the mode.
            sketch_mode: self.sketch_mode,
            // ADR 0028 (#95): restore the armed sketch tool, so a mid-edit repro re-enters with
            // the same verb in hand. Latent until sketch mode is active.
            sketch_tool: self.sketch_tool,
            // ADR 0030: the sketch selection is transient in-mode state; a fresh load starts with
            // nothing picked (the config does not persist it — re-entering a sketch clears it anyway).
            sketch_selection: ui::panel::SketchSelection::default(),
        };
        // step 8: restore the persisted full scene when present and non-empty;
        // otherwise (a scene-less old config, or a `Some(scene)` with no nodes — a
        // malformed/empty persisted scene) load the default seed scene, the same one
        // a brand-new config gets, so the seed always produces a usable document.
        match &self.scene {
            Some(scene) if !scene.roots.is_empty() => {
                state.scene = scene.clone();
            }
            _ => state.seed_scene_from_geometry(),
        }
        // issue #29 (grid rework S1): every loaded scene gains exactly one Origin
        // Point (idempotent — a scene that already carries one is untouched).
        //
        // issue #31: the grid masters are no longer migrated from legacy `show_*`
        // config fields (those mirrors were deleted). The scene's own `master_*`
        // fields are the single source of truth: a persisted scene carries them
        // directly, and a fresh/legacy config with no scene falls back to the
        // default seed scene whose `Scene::default()` masters all default ON.
        state.scene.ensure_origin_point();
        // ADR 0003 Phase B3: selection and the edit ops key on a stable `NodeId`, and
        // `mint_node_id` trusts `next_node_id` to already sit past every live id. A
        // restored scene may carry unminted nodes (`NodeId(0)` sentinel) and/or a stale
        // counter, so mint ids and advance the counter here — idempotent, and uniform
        // with the seed branch and `shot.rs`. Runs after `ensure_origin_point` so the
        // origin point it may have just appended also receives an id.
        state.scene.ensure_node_ids();
        // ADR 0030 load policy: erase structurally-invalid sketch entities (a segment
        // referencing a missing point, a self-loop) rather than fail the load, warning on the
        // CLI. A clean scene drops nothing.
        for (node_name, dropped) in state.scene.repair_sketches() {
            eprintln!(
                "warning: dropped {dropped} invalid sketch segment(s) from \"{node_name}\" on load"
            );
        }
        // ADR 0028: a restored sketch mode must point at a live sketch node. Drop it if the
        // id no longer resolves to a `SketchTool` in the loaded scene (a scene-less config, a
        // deleted node, or a node that is no longer a sketch), so a stale id cannot trap the
        // mode with no way out.
        if let Some(id) = state.sketch_mode {
            let live = matches!(
                state.scene.node_by_id(id).map(|node| &node.content),
                Some(NodeContent::SketchTool { .. })
            );
            if !live {
                state.sketch_mode = None;
            }
        }
        state
    }

    /// Apply this config's camera fields to an [`OrbitCamera`] (keeps its target).
    pub fn apply_camera(&self, camera: &mut OrbitCamera) {
        camera.orbit_theta = self.orbit_theta;
        camera.orbit_phi = self.orbit_phi;
        camera.orbit_distance = self.orbit_distance;
        camera.target = glam::Vec3::from_array(self.orbit_target);
        camera.projection_mode = self.projection_mode;
    }

    /// The config file path: `%APPDATA%\VoxelWorker\config.json` on Windows,
    /// falling back to `$XDG_CONFIG_HOME`/`$HOME/.config` elsewhere. Returns
    /// `None` if no suitable directory env var is set.
    pub fn config_path() -> Option<std::path::PathBuf> {
        let base = if cfg!(windows) {
            std::env::var_os("APPDATA").map(std::path::PathBuf::from)
        } else {
            std::env::var_os("XDG_CONFIG_HOME")
                .map(std::path::PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home).join(".config"))
                })
        }?;
        Some(base.join("VoxelWorker").join("config.json"))
    }

    /// Load the config from the platform path, returning `None` on any failure
    /// (missing file, unreadable, or invalid JSON) so the caller falls back to
    /// defaults. NEVER panics.
    pub fn load() -> Option<Self> {
        let path = Self::config_path()?;
        let text = std::fs::read_to_string(&path).ok()?;
        match Self::from_dump_json(&text) {
            Ok(config) => Some(config),
            Err(error) => {
                eprintln!("config: ignoring invalid {}: {error}", path.display());
                None
            }
        }
    }

    /// Load a config from an EXPLICIT path (the `shot --from-config` repro flow / the app's
    /// F9 `export_repro` dump). Unlike [`load`](Self::load) this is fallible-loud: an unreadable
    /// or malformed file returns the parse/IO error so the harness can exit with a clear message
    /// (a headless repro must not silently fall back to a different scene).
    pub fn load_from(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        Self::from_dump_json(&text)
            .map_err(|e| format!("invalid config {}: {e}", path.display()))
    }

    /// Restore the state a dump's JSON describes.
    ///
    /// Both on-disk artifacts are dumps, so both loads land here. Keeping the parse in one
    /// place is what stops the F9 file and the config file from drifting into two formats
    /// that happen to look alike — the failure this ADR's whole split is about.
    pub fn from_dump_json(text: &str) -> Result<Self, serde_json::Error> {
        Dump::from_json(text).map(Dump::into_state)
    }

    /// Serialize this state as a **dump** — the superset artifact, from which a scene must
    /// be completely reproducible.
    ///
    /// It is what F9 writes and what exit writes, for the same reason: restoring a session
    /// needs the scene, the preferences and the camera pose together, which is the dump's
    /// field set and not the document's. The document and settings projections
    /// ([`DocumentArtifact`](crate::artifacts::DocumentArtifact),
    /// [`SettingsArtifact`](crate::artifacts::SettingsArtifact)) are the parts this is
    /// composed of; giving either its own file would be a save/open workflow, which is a
    /// product decision and not this one.
    pub fn to_dump_json(&self) -> Result<String, String> {
        Dump::from_state(self).to_json()
    }

    /// Save the state to the platform path as a dump (pretty JSON), creating parent dirs.
    /// Errors are reported but not fatal — a failed save must not crash exit.
    pub fn save(&self) {
        let Some(path) = Self::config_path() else {
            eprintln!("config: no platform config dir; not saving");
            return;
        };
        if let Some(parent) = path.parent() {
            if let Err(error) = std::fs::create_dir_all(parent) {
                eprintln!("config: could not create {}: {error}", parent.display());
                return;
            }
        }
        match self.to_dump_json() {
            Ok(json) => {
                if let Err(error) = std::fs::write(&path, json) {
                    eprintln!("config: could not write {}: {error}", path.display());
                }
            }
            Err(error) => eprintln!("config: could not serialise: {error}"),
        }
    }
}

#[cfg(test)]
mod tests;
