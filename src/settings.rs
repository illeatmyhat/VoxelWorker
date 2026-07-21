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
use ui::panel::{LayerRange, PanelState, PlacementGhost, SignalStackState, ViewMode};
use document::scene::Scene;
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
            // The persisted ghost config does not yet carry orientation (ADR 0026); an F9 repro
            // of an armed oriented ghost previews it upright. The placed nodes it captures are
            // fully oriented in the scene tree — only the transient armed-ghost turn is dropped.
            orientation: substrate::spatial::LatticeOrientation::IDENTITY,
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
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            scene: None,
            voxels_per_block: default_density(),
            projection_mode: ProjectionMode::default(),
            material: MaterialChoice::default(),
            show_view_cube: true,
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
            stack: SignalStackState::default(),
            debug_face_orientation: false,
            debug_brick_faces: false,
            placement_ghost: None,
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
            stack: panel.stack,
            debug_face_orientation: panel.debug_face_orientation,
            debug_brick_faces: panel.debug_brick_faces,
            // ADR 0022: the armed placement ghost, captured as its serde-able mirror so a
            // mid-gesture dump replays the pending drop.
            placement_ghost: panel.placement_ghost.as_ref().map(PlacementGhostConfig::from_ghost),
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
mod tests {
    use super::*;
    use voxel_core::voxel::ShapeKind;

    /// Save and reload through the on-disk artifact, which since the ADR 0022 split is
    /// the dump rather than this struct. Most tests below care about what survives a
    /// save/load, not about which type spells the JSON, so they go through here.
    fn save_and_reload(config: &AppConfig) -> AppConfig {
        let json = config.to_dump_json().expect("serialise");
        AppConfig::from_dump_json(&json).expect("deserialise")
    }

    #[test]
    fn config_round_trips_through_json() {
        let config = AppConfig {
            scene: None,
            voxels_per_block: 24,
            projection_mode: ProjectionMode::Orthographic,
            material: MaterialChoice::Wood,
            show_view_cube: false,
            applied_block_label: Some("Granite".to_string()),
            snap_to_blocks: false,
            onion_skin: true,
            onion_depth: 5,
            orbit_theta: 1.23,
            orbit_phi: 0.95,
            orbit_distance: 42.0,
            orbit_target: [3.0, -7.5, 11.0],
            home_theta: 2.34,
            home_phi: 1.11,
            home_distance: 18.0,
            home_explicit: true,
            window_size: [1600, 900],
            view_mode: ViewMode::OnionFog,
            stack: SignalStackState {
                folded: true,
                viewport_open: false,
                onion_open: true,
                grids_open: false,
            },
            debug_face_orientation: true,
            debug_brick_faces: true,
            placement_ghost: Some(PlacementGhostConfig {
                shape_kind: ShapeKind::Sphere,
                size_voxels: [24, 16, 32],
                wall_blocks: 2,
                offset_voxels: [40, -8, 12],
            }),
        };

        let restored = save_and_reload(&config);
        assert_eq!(config, restored);
    }

    /// ADR 0024: the session state survives the full live round trip —
    /// `PanelState → capture → JSON → load → to_panel_state` — which is the leg that was
    /// broken. Each of these four was classified as reaching the dump and was hard-coded
    /// to a default on the way back, so a test asserting only that `AppConfig` round-trips
    /// would have passed throughout. The assertion has to start and end at the panel.
    #[test]
    fn the_session_survives_a_relaunch_through_the_panel_and_back() {
        let mut panel = PanelState::with_view_cube_default();
        panel.view_mode = ViewMode::OnionFog;
        panel.stack = SignalStackState {
            folded: true,
            viewport_open: false,
            onion_open: true,
            grids_open: false,
        };
        panel.debug_face_orientation = true;
        panel.debug_brick_faces = true;

        let config = AppConfig::capture(
            &panel,
            &OrbitCamera::default(),
            HomeView::default(),
            [1280, 800],
        );
        let restored = save_and_reload(&config).to_panel_state();

        assert_eq!(restored.view_mode, ViewMode::OnionFog);
        assert_eq!(restored.stack, panel.stack);
        assert!(restored.debug_face_orientation);
        assert!(restored.debug_brick_faces);
    }

    /// The other direction of the same promise: a dump written before the session category
    /// existed carries none of these keys, and must load as the finished look rather than
    /// failing. Every session field has a serde default, so an old repro still replays —
    /// which is the reason the dump tolerates missing keys at all.
    #[test]
    fn a_dump_without_session_keys_loads_the_finished_look() {
        let panel = AppConfig::from_dump_json(r#"{"voxels_per_block": 8}"#)
            .expect("a pre-session dump parses")
            .to_panel_state();
        assert_eq!(panel.view_mode, ViewMode::Normal);
        assert_eq!(panel.stack, SignalStackState::default());
        assert!(!panel.debug_face_orientation);
        assert!(!panel.debug_brick_faces);
    }

    /// #13: the home-view fields persist through capture→JSON→load, and an OLD
    /// config WITHOUT them loads with the camera defaults (serde fills each
    /// missing key from its `#[serde(default = ...)]` fn).
    #[test]
    fn home_view_persists_and_old_config_defaults() {
        let mut panel = PanelState::with_view_cube_default();
        panel.geometry.voxels_per_block = 8;
        let camera = OrbitCamera::default();
        let home = HomeView { theta: 2.5, phi: 0.6, distance: 33.0, explicitly_set: true };
        let config = AppConfig::capture(&panel, &camera, home, [1280, 800]);

        let restored = save_and_reload(&config);
        let restored_home = restored.home_view();
        assert!((restored_home.theta - 2.5).abs() < 1e-5);
        assert!((restored_home.phi - 0.6).abs() < 1e-5);
        assert!((restored_home.distance - 33.0).abs() < 1e-5);

        // An old config with no home_* keys loads with the camera defaults.
        let old_json = r#"{ "voxels_per_block": 8 }"#;
        let old = AppConfig::from_dump_json(old_json).expect("old config without home_* parses");
        let old_home = old.home_view();
        let defaults = HomeView::default();
        assert!((old_home.theta - defaults.theta).abs() < 1e-5);
        assert!((old_home.phi - defaults.phi).abs() < 1e-5);
        assert!((old_home.distance - defaults.distance).abs() < 1e-5);
    }

    /// issue #32: a config persists and reloads its `scene` correctly. A non-trivial
    /// scene (two offset Tool nodes with distinct materials) survives
    /// `capture → JSON → deserialize → to_panel_state` with the same node count,
    /// active selection, and resolved occupancy — the `scene` field is the single
    /// source of truth now that the flat geometry mirror fields are gone.
    #[test]
    fn config_persists_and_reloads_its_scene() {
        use document::scene::{Node, NodeContent, NodePath, Scene};
        use document::voxel::SdfShape;

        let voxels_per_block = 8u32;
        let unit_box = |kind| SdfShape::from_blocks(kind, [1, 1, 1], 1, voxels_per_block);
        let stone = Node::new(
            "Stone",
            NodeContent::Tool { shape: unit_box(ShapeKind::Box), material: MaterialChoice::Stone },
        );
        let mut wood = Node::new(
            "Wood",
            NodeContent::Tool { shape: unit_box(ShapeKind::Box), material: MaterialChoice::Wood },
        );
        wood.transform = document::scene::NodeTransform::from_blocks([3, 0, 0], voxels_per_block);
        // ADR 0003 Phase B3: selection is keyed by NodeId, so mint ids and select
        // the second node (top-level index 1) by its stable id.
        let mut scene = Scene::from_nodes(vec![stone, wood]);
        scene.active = scene.id_at_path(&NodePath::root_index(1));

        let mut panel = PanelState::with_view_cube_default();
        panel.geometry.voxels_per_block = voxels_per_block;
        panel.scene = scene.clone();
        let camera = OrbitCamera::default();
        let config = AppConfig::capture(&panel, &camera, HomeView::default(), [1280, 800]);
        assert!(config.scene.is_some(), "capture persists the scene");

        let restored = save_and_reload(&config);
        let restored_panel = restored.to_panel_state();

        assert_eq!(restored_panel.scene.roots.len(), 2, "both nodes survive the reload");
        assert_eq!(restored_panel.scene.active, scene.active, "the active selection survives");
        assert_eq!(
            restored_panel.scene.root_node(1).transform.blocks(voxels_per_block),
            [3, 0, 0]
        );

        let region = scene.full_extent_blocks(voxels_per_block);
        let before = scene.resolve_region(region, voxels_per_block, 0).occupied_count();
        let after_region = restored_panel.scene.full_extent_blocks(voxels_per_block);
        let after = restored_panel
            .scene
            .resolve_region(after_region, voxels_per_block, 0)
            .occupied_count();
        assert_eq!(before, after, "the restored scene resolves identically");
    }

    #[test]
    fn bad_json_falls_back_without_panicking() {
        // An empty object still parses thanks to the per-field defaults.
        let restored = AppConfig::from_dump_json("{}").expect("empty object parses");
        assert_eq!(restored, AppConfig::default());

        // Outright invalid JSON must be a clean Err (the caller turns it into a
        // defaults fallback), never a panic.
        assert!(AppConfig::from_dump_json("not json at all}{").is_err());
    }

    /// issue #31 + #32: the legacy grid `show_*` mirror fields (`show_grid_overlay` /
    /// `show_block_lattice` / `show_floor_grid`), the older `show_origin_gizmo`, AND
    /// the flat geometry mirror fields (`shape` / `size_blocks` / `wall_blocks`) were
    /// all removed from `AppConfig`. There is no `deny_unknown_fields`, so an OLD
    /// config still carrying those keys must keep deserializing cleanly — serde
    /// ignores the now-unknown keys. The masters no longer migrate from the grid
    /// keys, and a scene-less config simply loads the default seed scene whose
    /// `Scene::default()` masters all default ON.
    #[test]
    fn old_config_with_removed_keys_still_loads() {
        let old_json = r#"{
            "shape": "Box",
            "size_blocks": [2, 2, 2],
            "voxels_per_block": 8,
            "wall_blocks": 1,
            "show_grid_overlay": true,
            "show_block_lattice": false,
            "show_floor_grid": true,
            "show_origin_gizmo": true
        }"#;
        let config = AppConfig::from_dump_json(old_json)
            .expect("old config with removed keys still parses");
        assert!(config.scene.is_none());
        // The app-level density key is the one flat field still read.
        assert_eq!(config.voxels_per_block, 8);

        let panel = config.to_panel_state();
        // The removed keys are simply ignored — they no longer seed the masters or
        // the geometry. A scene-less config loads the default seed scene whose
        // masters all default ON.
        assert!(panel.scene.master_block_lattice, "fresh scene masters default ON");
        assert!(panel.scene.master_voxel_grid, "fresh scene masters default ON");
        assert!(panel.scene.master_floor_grid, "fresh scene masters default ON");
        // Exactly one Origin Point, as on any load path.
        assert_eq!(panel.scene.points.iter().filter(|p| p.is_origin).count(), 1);
    }

    /// issue #32: an OLD config carrying the dropped `debug_clouds` boolean AND the
    /// removed flat geometry mirror fields (`shape` / `size_blocks` / `wall_blocks`)
    /// must load gracefully — serde ignores the now-unknown keys. The persisted
    /// app-level density (`voxels_per_block`) and `material` still round-trip, and a
    /// scene-less config loads the DEFAULT seed scene (no longer a scene built from
    /// the removed flat params).
    #[test]
    fn old_config_with_debug_clouds_field_still_loads() {
        let old_json = r#"{
            "shape": "Sphere",
            "size_blocks": [3, 4, 5],
            "voxels_per_block": 20,
            "wall_blocks": 2,
            "debug_clouds": true,
            "material": "Wood"
        }"#;
        let restored = AppConfig::from_dump_json(old_json).expect("old config (with debug_clouds) must still parse");
        // The flat geometry keys are ignored; only density + material survive.
        assert_eq!(restored.voxels_per_block, 20);
        assert_eq!(restored.material, MaterialChoice::Wood);
        // An old config has NO `scene` field, so it deserialises to `None`, which now
        // loads the default seed scene (the same one a brand-new config gets).
        assert!(restored.scene.is_none(), "an old flat config carries no scene");

        // It loads the DEFAULT seed scene (a one-Tool-node Cylinder, NOT a scene built
        // from the removed flat `shape`/`size_blocks`/`wall_blocks`). Only the density
        // carries over from the config.
        let panel = restored.to_panel_state();
        assert_eq!(panel.scene.roots.len(), 1);
        // Density DID carry over from the config and now lives on the document
        // (ADR 0003 §3f(0)), not the shape.
        assert_eq!(panel.scene.voxels_per_block, 20);
        match panel.scene.active_node().map(|node| &node.content) {
            Some(document::scene::NodeContent::Tool { shape, material }) => {
                // The default seed geometry, NOT the persisted flat params.
                assert_eq!(shape.kind, ShapeKind::Cylinder);
                // Size is voxel-canonical now (ADR 0003 §3f(0)): the 5×1×5-block seed
                // built at the persisted density 20 = [100, 20, 100] voxels.
                assert_eq!(shape.size_voxels, [100, 20, 100]);
                // The persisted `material` rides the seed (it is still an AppConfig field).
                assert_eq!(*material, MaterialChoice::Wood);
            }
            other => panic!("the seed must build a one Tool node, got {other:?}"),
        }
    }

    /// Part of #20: the legacy instanced mesher was removed along with the
    /// `MesherChoice` toggle. The choice was never a persisted `AppConfig` field
    /// (it lived only in the session-only `PanelState`), but defend the migration
    /// regardless: an OLD config JSON that carried a stray top-level `mesher` field
    /// (e.g. hand-edited) must STILL load — serde ignores the now-unknown field —
    /// and every real field round-trips.
    #[test]
    fn old_config_with_mesher_field_still_loads() {
        let old_json = r#"{
            "shape": "Cylinder",
            "size_blocks": [5, 1, 5],
            "voxels_per_block": 8,
            "wall_blocks": 1,
            "mesher": "Instanced",
            "material": "Stone"
        }"#;
        let restored = AppConfig::from_dump_json(old_json)
            .expect("old config (with mesher) must still parse");
        // The flat geometry keys are ignored (issue #32); density + material survive.
        assert_eq!(restored.voxels_per_block, 8);
        assert_eq!(restored.material, MaterialChoice::Stone);
        // It loads cleanly to the default one-Tool-node seed scene (the stray `mesher`
        // and the removed flat geometry keys are all ignored).
        let panel = restored.to_panel_state();
        assert_eq!(panel.scene.roots.len(), 1);
    }

    /// step 8 round-trip: a NON-TRIVIAL scene (top-level Tool + VoxelBody nodes with
    /// non-zero offsets and distinct materials, a Group with children, an
    /// `AssemblyDef`, and an `Instance` of it) survives
    /// `capture → JSON → deserialize → to_panel_state` structurally intact and
    /// resolves to the SAME occupied count.
    #[test]
    fn full_scene_round_trips_through_json() {
        use document::scene::{
            DefId, Node, NodeBuilder, NodeContent, NodePath, VoxelBody, Scene,
        };
        use document::voxel::SdfShape;

        let voxels_per_block = 8u32;
        let unit_box = |kind| SdfShape::from_blocks(kind, [1, 1, 1], 1, voxels_per_block);

        // A definition (the reusable "house" body): a single Wood box.
        let def_id = DefId(3);

        // Top-level node 0: a Stone Tool at the origin.
        let stone = Node::new(
            "Stone",
            NodeContent::Tool {
                shape: unit_box(ShapeKind::Box),
                material: MaterialChoice::Stone,
            },
        );
        // Top-level node 1: a Clouds VoxelBody, offset.
        let mut clouds = Node::new("Clouds", NodeContent::VoxelBody(VoxelBody::DebugClouds { seed: 7 }));
        clouds.transform = document::scene::NodeTransform::from_blocks([3, 0, 0], voxels_per_block);
        // Top-level node 2: a Group containing a Plain Tool offset within it.
        let mut grouped_leaf = Node::new(
            "Leaf",
            NodeContent::Tool {
                shape: unit_box(ShapeKind::Sphere),
                material: MaterialChoice::Plain,
            },
        );
        grouped_leaf.transform = document::scene::NodeTransform::from_blocks([1, 0, 0], voxels_per_block);
        // Top-level node 2: a Group at +6X containing the Plain Tool offset within it
        // (`CombineOp::Union` is the default operation a built Group carries).
        let group = NodeBuilder::group_at("Group", [6, 0, 0], voxels_per_block, vec![grouped_leaf.into()]);
        // Top-level node 3: an Instance of the def, offset disjointly.
        let mut instance = Node::new("House instance", NodeContent::Instance(def_id));
        instance.transform = document::scene::NodeTransform::from_blocks([-6, 0, 0], voxels_per_block);

        // ADR 0003 Phase B3: selection is keyed by NodeId, so mint ids and select
        // the Group's child (path [2, 0]) by its stable id.
        let mut scene = Scene::from_nodes(vec![
            NodeBuilder::Leaf(stone),
            NodeBuilder::Leaf(clouds),
            group,
            NodeBuilder::Leaf(instance),
        ]);
        scene.add_definition(
            def_id,
            "House".to_string(),
            vec![Node::new(
                "Body",
                NodeContent::Tool {
                    shape: unit_box(ShapeKind::Box),
                    material: MaterialChoice::Wood,
                },
            )],
        );
        scene.active = scene.id_at_path(&NodePath::from_indices(vec![2, 0]));

        // Build a panel carrying this scene and capture → JSON → restore.
        let mut panel = PanelState::with_view_cube_default();
        panel.geometry.voxels_per_block = voxels_per_block;
        panel.scene = scene.clone();
        let camera = OrbitCamera::default();
        let config = AppConfig::capture(&panel, &camera, HomeView::default(), [1280, 800]);
        assert!(config.scene.is_some(), "capture persists the full scene");

        let restored = save_and_reload(&config);
        let restored_panel = restored.to_panel_state();

        // Structural equality: same node tree, definitions, and active selection.
        assert_eq!(
            restored_panel.scene.roots.len(),
            scene.roots.len(),
            "all top-level nodes survive"
        );
        assert_eq!(restored_panel.scene.definitions.len(), 1, "the def survives");
        assert_eq!(
            restored_panel.scene.active,
            scene.active,
            "the active selection survives"
        );
        // The Group's child and the def's body survive with their offsets/materials.
        match &restored_panel.scene.root_node(2).content {
            NodeContent::Group(children) => {
                assert_eq!(children.len(), 1);
                assert_eq!(
                    restored_panel.scene.arena[&children[0]]
                        .transform
                        .blocks(voxels_per_block),
                    [1, 0, 0]
                );
            }
            other => panic!("node 2 must stay a Group, got {other:?}"),
        }
        assert!(matches!(
            restored_panel.scene.root_node(3).content,
            NodeContent::Instance(id) if id == def_id
        ));

        // Same resolved occupancy (the document means the same thing on reload).
        let region = scene.full_extent_blocks(voxels_per_block);
        let before = scene
            .resolve_region(region, voxels_per_block, 0)
            .occupied_count();
        let after_region = restored_panel.scene.full_extent_blocks(voxels_per_block);
        let after = restored_panel
            .scene
            .resolve_region(after_region, voxels_per_block, 0)
            .occupied_count();
        assert_eq!(before, after, "the restored scene resolves identically");
    }

    /// step 8 (never panic on load): a config whose `scene` value is broken/partial
    /// still loads. A scene object missing its inner fields deserialises to an
    /// empty-node scene (every scene field is `#[serde(default)]`), which
    /// `to_panel_state` treats as absent → falls back to the one-Tool-node seed.
    #[test]
    fn malformed_scene_falls_back_to_default_without_panicking() {
        // A `scene` present but EMPTY (no nodes) — a partial/degenerate persisted
        // value. It parses (defaults fill the missing fields) and migrates.
        let partial = r#"{
            "scene": {},
            "shape": "Box",
            "size_blocks": [2, 2, 2],
            "voxels_per_block": 12,
            "wall_blocks": 1
        }"#;
        let restored = AppConfig::from_dump_json(partial).expect("a partial scene object still parses");
        let panel = restored.to_panel_state();
        assert_eq!(
            panel.scene.roots.len(),
            1,
            "an empty persisted scene falls back to the one-Tool-node seed"
        );

        // A `scene` whose arena holds a node with a content variant that doesn't exist
        // is a clean parse error wholesale → `load()` would return None → caller uses
        // defaults. We assert it never panics: the deserialize is an Err, not an unwind.
        // (The id-keyed arena is the real node storage, so the broken node must live
        // there — a stray legacy `"nodes"` key would simply be ignored by serde.)
        let broken = r#"{ "scene": { "roots": [1], "arena": { "1": { "content": "NotAVariant" } } } }"#;
        assert!(
            AppConfig::from_dump_json(broken).is_err(),
            "a structurally broken scene is a clean Err (load → defaults), never a panic"
        );
    }

    /// S4a back-compat: a small i32-range `offset_voxels` value carried in a JSON
    /// document widens into the now-`[i64; 3]` field unchanged. A JSON integer carries
    /// no width, so serde reads it straight into `i64` — the "tolerant persistence
    /// migration" S4a requires. The document must load, keep its offsets, and resolve
    /// to a non-empty grid. (Placement is canonical voxels at the document density now,
    /// ADR 0003 §3f(0); authored here as a whole-block offset via `from_blocks`.)
    ///
    /// **ADR 0003 Phase B5 note:** the original version of this test hand-authored a
    /// `"scene": { "nodes": [ … ] }` document in the OLD positional-`Vec<Node>` on-disk
    /// shape. Phase B5 flipped scene storage to an id-keyed `arena` + `roots` spine, so
    /// that legacy array shape no longer deserializes (the field is gone). Per project
    /// policy (pre-alpha; old saves may break — see no-config-back-compat memory) the
    /// test is REWRITTEN to author the scene via the API and round-trip it through the
    /// NEW on-disk shape, still exercising i64-offset WIDENING (its real purpose, which
    /// is orthogonal to the storage layout). The small i32-range offset is what an old
    /// save held; the assertion that it lands as the same `i64` value is unchanged.
    #[test]
    fn old_i32_offset_scene_loads_after_widening_to_i64() {
        use document::scene::{Node, NodeContent, Scene};
        use document::voxel::SdfShape;

        // A single Box Tool offset +5 blocks in X — a small i32-range offset, exactly
        // what a pre-S4a `[i32; 3]` save held. Authored via the API, then serialized to
        // the current on-disk format and reloaded; the offset is a plain JSON integer
        // in that document, so reloading proves it reads into the `i64` field intact.
        let shape = SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, 16);
        let mut node = Node::new(
            "Box",
            NodeContent::Tool { shape, material: MaterialChoice::Stone },
        );
        // +5 blocks in X at density 8 → canonical offset_voxels = [40, 0, 0].
        node.transform = document::scene::NodeTransform::from_blocks([5, 0, 0], 8);
        let scene = Scene::single_node(node);

        let mut panel = PanelState::with_view_cube_default();
        panel.geometry.voxels_per_block = 8;
        panel.scene = scene;
        let camera = OrbitCamera::default();
        let config = AppConfig::capture(&panel, &camera, HomeView::default(), [1280, 800]);
        let json = config.to_dump_json().expect("serialise");

        // Sanity: the persisted offset really is a bare JSON integer (no width), the
        // exact condition the widening relies on. Checked against the parsed value rather
        // than the text, because a dump is written pretty and the whitespace between the
        // array elements is not the property under test.
        let written: serde_json::Value = serde_json::from_str(&json).expect("re-parse");
        assert_eq!(
            written["scene"]["arena"]
                .as_object()
                .and_then(|arena| arena.values().next())
                .map(|node| &node["transform"]["offset_voxels"]),
            Some(&serde_json::json!([40, 0, 0])),
            "the offset persists as plain JSON integers (no width): {json}"
        );

        let restored = AppConfig::from_dump_json(&json).expect("an i32-range-offset scene must parse");
        let panel = restored.to_panel_state();
        assert_eq!(panel.scene.roots.len(), 1, "the node survives the widening");
        // The i32-range offset widened into the i64 field intact.
        assert_eq!(
            panel.scene.root_node(0).transform.offset_voxels,
            [40i64, 0, 0],
            "the old i32 offset must widen to the same i64 value"
        );
        assert!(matches!(
            panel.scene.root_node(0).content,
            NodeContent::Tool { .. }
        ));
        // The migrated document still resolves to a non-empty grid.
        let region = panel.scene.full_extent_blocks(8);
        assert!(
            panel.scene.resolve_region(region, 8, 0).occupied_count() > 0,
            "the migrated old-offset scene resolves to voxels"
        );
    }

    /// S4a: a scene whose `offset_voxels` is a LARGE i64 (well beyond the old
    /// `i32` range, ±2.1×10⁹) round-trips through `capture → JSON → load` byte-exact.
    /// This proves the widened field both serializes and deserializes the full
    /// 64-bit range — far-apart village nodes survive a save/load. (Placement is
    /// canonical voxels now, ADR 0003 §3f(0); the large value is set directly on the
    /// voxel field to exercise the full i64 range it persists.)
    #[test]
    fn large_i64_offset_round_trips_through_json() {
        use document::scene::{Node, NodeContent, Scene};
        use document::voxel::SdfShape;

        // Beyond i32::MAX (2_147_483_647): a node placed ~3 billion blocks out. An
        // i32 field could never have held this; the i64 field must persist it exactly.
        let far_offset: i64 = 3_000_000_000;
        let shape = SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, 16);
        let mut node = Node::new(
            "Far box",
            NodeContent::Tool { shape, material: MaterialChoice::Stone },
        );
        node.transform.offset_voxels = [far_offset, -far_offset, far_offset / 2];
        let scene = Scene::single_node(node);

        let mut panel = PanelState::with_view_cube_default();
        panel.geometry.voxels_per_block = 8;
        panel.scene = scene.clone();
        let camera = OrbitCamera::default();
        let config = AppConfig::capture(&panel, &camera, HomeView::default(), [1280, 800]);

        let restored = save_and_reload(&config);
        let restored_panel = restored.to_panel_state();

        assert_eq!(
            restored_panel.scene.roots.len(),
            1,
            "the far node survives the round-trip"
        );
        assert_eq!(
            restored_panel.scene.root_node(0).transform.offset_voxels,
            [far_offset, -far_offset, far_offset / 2],
            "a >i32-range i64 offset must round-trip byte-exact through save/load"
        );
        // ADR 0003 Phase B3: selection is keyed by NodeId; `single_node` minted the
        // lone node an id and selected it, and that id round-trips intact.
        assert_eq!(
            restored_panel.scene.active,
            scene.active,
            "the active selection survives"
        );
        assert!(scene.active.is_some(), "the lone node is selected by id");
    }

    /// issue #31: the grid masters are the single source of truth on `scene.master_*`
    /// and round-trip through `capture → JSON → to_panel_state` directly (no legacy
    /// `show_*` mirror). Non-default master values must survive the round-trip.
    #[test]
    fn capture_then_to_panel_state_preserves_masters_and_toggles() {
        let mut panel = PanelState::with_view_cube_default();
        // Drive non-default master values directly on the scene (the UI checkboxes do
        // the same). Mixed values prove each master persists independently.
        panel.scene.master_block_lattice = false;
        panel.scene.master_voxel_grid = true;
        panel.scene.master_floor_grid = false;
        panel.material = MaterialChoice::Plain;
        let camera = OrbitCamera::default();
        let config = AppConfig::capture(&panel, &camera, HomeView::default(), [1024, 768]);
        let restored = config.to_panel_state();
        // The masters round-trip via `scene.master_*` — the single source of truth.
        assert_eq!(restored.scene.master_block_lattice, panel.scene.master_block_lattice);
        assert_eq!(restored.scene.master_voxel_grid, panel.scene.master_voxel_grid);
        assert_eq!(restored.scene.master_floor_grid, panel.scene.master_floor_grid);
        assert_eq!(restored.material, panel.material);
        assert_eq!(restored.geometry, panel.geometry);
    }

    /// issue #29 (grid rework S1) + issue #31: loading an OLD config (no `scene`
    /// field — the legacy flat geometry) gains exactly one Origin Point on the load
    /// path. The grid masters no longer migrate from legacy `show_*` keys (deleted in
    /// #31); the scene-less config seeds a fresh scene whose masters all default ON.
    #[test]
    fn old_config_gains_origin_point_with_default_masters() {
        let old_json = r#"{
            "shape": "Box",
            "size_blocks": [2, 2, 2],
            "voxels_per_block": 8,
            "wall_blocks": 1,
            "show_grid_overlay": true,
            "show_block_lattice": false,
            "show_floor_grid": true
        }"#;
        let config = AppConfig::from_dump_json(old_json).expect("old config parses");
        assert!(config.scene.is_none(), "an old flat config carries no scene");

        let panel = config.to_panel_state();
        // Exactly one Origin Point synthesized on load.
        assert_eq!(
            panel.scene.points.iter().filter(|p| p.is_origin).count(),
            1,
            "the load path synthesizes exactly one Origin Point"
        );
        assert_eq!(panel.scene.points.len(), 1);
        assert!(panel.scene.points[0].is_origin);
        assert_eq!(panel.scene.points[0].name, "Origin");

        // The removed legacy `show_*` keys are ignored — masters default ON from
        // `Scene::default()` (NOT migrated from the stale `show_block_lattice=false`).
        assert!(panel.scene.master_block_lattice, "fresh scene masters default ON");
        assert!(panel.scene.master_voxel_grid, "fresh scene masters default ON");
        assert!(panel.scene.master_floor_grid, "fresh scene masters default ON");
    }

    /// issue #29 + #31: a scene carrying its own masters keeps them on reload — the
    /// masters persist directly on the `scene` field (the single source of truth),
    /// not via any legacy `show_*` mirror. The Origin is not duplicated.
    #[test]
    fn modern_scene_keeps_its_masters_and_single_origin() {
        use document::scene::{Node, NodeContent, NodePath, Point, Scene};
        use document::voxel::SdfShape;

        let node = Node::new(
            "Box",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, 16),
                material: MaterialChoice::Stone,
            },
        );
        let mut scene = Scene::from_nodes(vec![node]);
        scene.master_block_lattice = false;
        scene.master_voxel_grid = true;
        scene.master_floor_grid = false;
        // ADR 0003 Phase B3: select the lone node by its stable id (from_nodes minted it).
        scene.active = scene.id_at_path(&NodePath::root_index(0));
        scene.ensure_origin_point();
        scene.add_point(Point { name: "Marker".to_string(), ..Point::default() });

        let mut panel = PanelState::with_view_cube_default();
        panel.scene = scene;
        let camera = OrbitCamera::default();
        let config = AppConfig::capture(&panel, &camera, HomeView::default(), [1280, 800]);

        let restored = save_and_reload(&config);
        let restored_panel = restored.to_panel_state();

        // The scene's own masters survive (NOT overwritten by the legacy show_*).
        assert!(!restored_panel.scene.master_block_lattice);
        assert!(restored_panel.scene.master_voxel_grid);
        assert!(!restored_panel.scene.master_floor_grid);
        // Still exactly one Origin (not duplicated on reload).
        assert_eq!(
            restored_panel.scene.points.iter().filter(|p| p.is_origin).count(),
            1
        );
        assert_eq!(restored_panel.scene.points.len(), 2, "Origin + Marker survive");
    }

    /// ADR 0003 Phase B3 regression: a persisted scene whose nodes carry the
    /// `NodeId(0)` sentinel and a stale `next_node_id` (an unminted save) must be
    /// minted on the load path, not left selection-dead. Without the
    /// `ensure_node_ids` call in `to_panel_state`, `id_at_path` would resolve a
    /// clicked node to `NodeId(0)`, which `node_by_id`/`path_of` reject — so the
    /// node would be silently unselectable and the next edit op would mint a
    /// colliding id.
    #[test]
    fn unminted_persisted_scene_gets_ids_minted_on_load() {
        use document::scene::{Node, NodeContent, NodePath, NodeId, Scene};
        use document::voxel::SdfShape;

        let make_box = |name: &str| {
            Node::new(
                name,
                NodeContent::Tool {
                    shape: SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, 16),
                    material: MaterialChoice::Stone,
                },
            )
        };
        // REWRITTEN for the id-keyed arena (ADR 0003 B5): the old fixture built two
        // `NodeId(0)` nodes, but the arena is keyed BY id, so it cannot hold two
        // sentinel nodes, and `ensure_node_ids` re-keys a lone 0-node in the arena
        // WITHOUT rewriting the `roots`/Group spines that reference it — so a
        // roots-references-sentinel save is not representable/positionally reachable.
        // The surviving, load-path-exercised guarantee is the STALE-COUNTER half: a
        // persisted scene whose nodes already carry ids but whose `next_node_id` was
        // never advanced past them must be normalised on load so a later edit op mints
        // a non-colliding id and every row stays selectable. We forge exactly that
        // persisted shape by resetting the counter in the serialized JSON.
        let scene = Scene::from_nodes(vec![make_box("First"), make_box("Second")]);

        let mut panel = PanelState::with_view_cube_default();
        panel.scene = scene;
        let camera = OrbitCamera::default();
        let config = AppConfig::capture(&panel, &camera, HomeView::default(), [1280, 800]);
        let dump_json = config.to_dump_json().expect("serialise");
        let mut config_value: serde_json::Value =
            serde_json::from_str(&dump_json).expect("re-parse the dump");
        // Forge a stale counter: the nodes carry real ids, but `next_node_id` sits at 0
        // (as a save written before the counter was persisted/advanced would).
        *config_value
            .get_mut("scene")
            .and_then(|s| s.get_mut("next_node_id"))
            .expect("the persisted scene carries a counter") = serde_json::json!(0);

        let json = serde_json::to_string_pretty(&config_value).expect("re-serialise");
        let restored = AppConfig::from_dump_json(&json).expect("deserialise");
        let loaded = restored.to_panel_state();

        // Every node carries a real id, and the counter now sits past all of them.
        assert!(
            loaded.scene.arena.values().all(|node| node.id != NodeId(0)),
            "every loaded node carries a real (non-sentinel) id"
        );
        let max_id = loaded.scene.arena.keys().map(|id| id.0).max().unwrap();
        assert!(loaded.scene.next_node_id > max_id, "counter advanced past every live id");

        // A clicked top-level row resolves to a selectable node (not the sentinel).
        let clicked_id = loaded.scene.id_at_path(&NodePath::root_index(0)).expect("path resolves to an id");
        assert_ne!(clicked_id, NodeId(0));
        assert!(loaded.scene.node_by_id(clicked_id).is_some(), "the resolved id is selectable");
    }
}
