//! Config persistence (Milestone 8).
//!
//! Serialises the user-facing state — geometry, projection, material choice, the
//! display toggles, the applied-block label, the camera orbit + projection, and
//! the window size — to a JSON file under the platform config dir. On Windows
//! that is `%APPDATA%\VoxelWorker\config.json`.
//!
//! Design notes:
//!   * [`AppConfig`] is a *flat, self-contained* mirror of the persisted fields,
//!     not a `#[derive(Serialize)]` on the live render-coupled `PanelState`. This
//!     keeps the on-disk format stable and decoupled from internal struct churn,
//!     and lets every field be `#[serde(default)]` so an older/newer config never
//!     fails to parse (a missing field falls back to its default).
//!   * Loading never panics: a missing file, an unreadable file, or invalid JSON
//!     all yield `None`, and the caller uses its built-in defaults.
//!   * The applied VS block is persisted only as its *label* (a string). Re-
//!     resolving its texture on load is heavy (needs a folder scan + JSON
//!     resolution), so the label is restored best-effort for display and the
//!     material reverts to procedural until the user re-applies. Documented here
//!     because it is an intentional, lazy re-apply.

use serde::{Deserialize, Serialize};

use crate::camera::{HomeView, OrbitCamera, ProjectionMode};
use crate::panel::{GeometryParams, LayerRange, MaterialChoice, PanelState};
use crate::scene::Scene;

/// The whole persisted configuration. Every field is `#[serde(default)]` so a
/// partial or older config still loads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppConfig {
    // --- scene (ADR 0001 step 8: full scene persistence) ---
    // The whole assembly (node tree + reusable definitions + the active
    // selection) is persisted here. `#[serde(default)]` means an OLD config with
    // no `scene` field deserialises to `None`, which loads the default seed scene
    // in `to_panel_state` (the same one a brand-new config gets). A malformed/partial
    // `scene` value can never reach this field as garbage: serde tolerates missing
    // inner fields (every scene field is `#[serde(default)]`), and an outright
    // unparseable config is rejected wholesale by `load()` → defaults. Density
    // (`voxels_per_block`) stays an app-level field below; the scene reads it at
    // resolve time (ADR 0001 "Density").
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
    #[serde(default)]
    pub scene: Option<Scene>,

    // --- density (app-level; the scene reads it at resolve time, ADR 0001) ---
    #[serde(default = "default_density")]
    pub voxels_per_block: u32,

    // --- display / material ---
    #[serde(default)]
    pub projection_mode: ProjectionMode,
    #[serde(default)]
    pub material: MaterialChoice,
    // Issue #31: the three legacy grid `show_*` mirror fields
    // (`show_grid_overlay` / `show_block_lattice` / `show_floor_grid`) were deleted.
    // The grid masters now persist as the single source of truth on the `scene`
    // field (`scene.master_voxel_grid` / `master_block_lattice` / `master_floor_grid`).
    // No `deny_unknown_fields`, so an OLD config still carrying those keys loads fine
    // (serde ignores the now-unknown keys); the scene's own masters are authoritative.
    #[serde(default = "default_true")]
    pub show_view_cube: bool,
    // NOTE: the legacy `show_origin_gizmo` field was removed in the issue #29 S6
    // cleanup. The old origin-gizmo Display toggle was replaced by the
    // selection-driven transform gizmo, so the field drove nothing. There is no
    // `deny_unknown_fields`, so an OLD config still carrying `"show_origin_gizmo"`
    // continues to deserialize cleanly (serde ignores the now-unknown key).
    /// Best-effort applied-block label (re-applied lazily; see module docs).
    #[serde(default)]
    pub applied_block_label: Option<String>,

    // --- layer-range scrubber (issue #12) ---
    // The bounds themselves depend on the live grid_y, so they are NOT persisted
    // (they always re-derive to the full range on load); only the sticky control
    // preferences are saved here.
    #[serde(default = "default_true")]
    pub snap_to_blocks: bool,
    #[serde(default)]
    pub onion_skin: bool,
    #[serde(default = "default_onion_depth")]
    pub onion_depth: u32,

    // --- camera ---
    #[serde(default = "default_theta")]
    pub orbit_theta: f32,
    #[serde(default = "default_phi")]
    pub orbit_phi: f32,
    #[serde(default = "default_distance")]
    pub orbit_distance: f32,

    // --- view-cube home view (#13) ---
    // The Home button's saved view. New fields: `#[serde(default)]` so an OLD
    // config without them loads with the camera defaults (theta≈0.7/phi≈1.05/
    // distance 10) — serde fills each missing key from its default fn.
    #[serde(default = "default_theta")]
    pub home_theta: f32,
    #[serde(default = "default_phi")]
    pub home_phi: f32,
    #[serde(default = "default_distance")]
    pub home_distance: f32,
    /// #13 Step 6.4: was the home view explicitly captured by the user? When
    /// `false` (the default), the Home button re-frames the model instead of using
    /// `home_distance`, so a default home never zooms in too close.
    #[serde(default)]
    pub home_explicit: bool,

    // --- window ---
    #[serde(default = "default_window_size")]
    pub window_size: [u32; 2],
}

fn default_density() -> u32 {
    16
}
fn default_true() -> bool {
    true
}
fn default_theta() -> f32 {
    0.7
}
fn default_phi() -> f32 {
    1.05
}
fn default_distance() -> f32 {
    10.0
}
fn default_window_size() -> [u32; 2] {
    [1280, 800]
}
fn default_onion_depth() -> u32 {
    2
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
            home_theta: default_theta(),
            home_phi: default_phi(),
            home_distance: default_distance(),
            home_explicit: false,
            window_size: default_window_size(),
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
            home_theta: home_view.theta,
            home_phi: home_view.phi,
            home_distance: home_view.distance,
            home_explicit: home_view.explicitly_set,
            window_size,
        }
    }

    /// The persisted [`HomeView`] (#13) — the saved Home-button view restored on
    /// load. Defaults to the camera defaults for an old config without the keys.
    pub fn home_view(&self) -> HomeView {
        HomeView {
            theta: self.home_theta,
            phi: self.home_phi,
            distance: self.home_distance,
            explicitly_set: self.home_explicit,
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
            geometry: GeometryParams {
                voxels_per_block: self.voxels_per_block,
                ..GeometryParams::default()
            },
            projection_mode: self.projection_mode,
            material: self.material,
            show_view_cube: self.show_view_cube,
            // Face-orientation debug is a transient verification mode; it is not
            // persisted, so it always starts off.
            debug_face_orientation: false,
            voxel_cap_warning_millions: None,
            // Re-applied lazily/best-effort: only the label is restored (for the
            // panel readout); the material itself reverts to procedural.
            applied_block_label: self.applied_block_label.clone(),
            // Issue #12: only the sticky control prefs persist; the band bounds
            // are re-derived to the full range against the live grid_y on load.
            layer_range: LayerRange {
                lower: 0,
                upper: 0, // rescaled to grid_y by the caller after the grid resolves.
                snap_to_blocks: self.snap_to_blocks,
                onion_skin: self.onion_skin,
                onion_depth: self.onion_depth.clamp(1, 8),
            },
            // Restored just below: the persisted full scene, or — for an old
            // config without one — the default seed scene.
            scene: Scene::default(),
            // Issue #29 S5: refreshed each frame from the camera target by the windowed
            // caller; defaults to the world origin (the headless harness keeps it 0).
            point_add_position_blocks: [0, 0, 0],
        };
        // step 8: restore the persisted full scene when present and non-empty;
        // otherwise (a scene-less old config, or a `Some(scene)` with no nodes — a
        // malformed/empty persisted scene) load the default seed scene, the same one
        // a brand-new config gets, so the seed always produces a usable document.
        match &self.scene {
            Some(scene) if !scene.nodes.is_empty() => {
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
        state
    }

    /// Apply this config's camera fields to an [`OrbitCamera`] (keeps its target).
    pub fn apply_camera(&self, camera: &mut OrbitCamera) {
        camera.orbit_theta = self.orbit_theta;
        camera.orbit_phi = self.orbit_phi;
        camera.orbit_distance = self.orbit_distance;
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
        match serde_json::from_str::<Self>(&text) {
            Ok(config) => Some(config),
            Err(error) => {
                eprintln!("config: ignoring invalid {}: {error}", path.display());
                None
            }
        }
    }

    /// Save the config to the platform path (pretty JSON), creating parent dirs.
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
        match serde_json::to_string_pretty(self) {
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
    use crate::voxel::ShapeKind;

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
            home_theta: 2.34,
            home_phi: 1.11,
            home_distance: 18.0,
            home_explicit: true,
            window_size: [1600, 900],
        };

        let json = serde_json::to_string_pretty(&config).expect("serialise");
        let restored: AppConfig = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(config, restored);
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

        let json = serde_json::to_string_pretty(&config).expect("serialise");
        let restored: AppConfig = serde_json::from_str(&json).expect("deserialise");
        let restored_home = restored.home_view();
        assert!((restored_home.theta - 2.5).abs() < 1e-5);
        assert!((restored_home.phi - 0.6).abs() < 1e-5);
        assert!((restored_home.distance - 33.0).abs() < 1e-5);

        // An old config with no home_* keys loads with the camera defaults.
        let old_json = r#"{ "voxels_per_block": 8 }"#;
        let old: AppConfig =
            serde_json::from_str(old_json).expect("old config without home_* parses");
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
        use crate::scene::{Node, NodeContent, NodePath, Scene};
        use crate::voxel::SdfShape;

        let voxels_per_block = 8u32;
        let unit_box = |kind| SdfShape {
            kind,
            size_blocks: [1, 1, 1],
            voxels_per_block,
            wall_blocks: 1,
        };
        let stone = Node::new(
            "Stone",
            NodeContent::Tool { shape: unit_box(ShapeKind::Box), material: MaterialChoice::Stone },
        );
        let mut wood = Node::new(
            "Wood",
            NodeContent::Tool { shape: unit_box(ShapeKind::Box), material: MaterialChoice::Wood },
        );
        wood.transform.offset_blocks = [3, 0, 0];
        let scene = Scene {
            nodes: vec![stone, wood],
            active: Some(NodePath::root_index(1)),
            ..Scene::default()
        };

        let mut panel = PanelState::with_view_cube_default();
        panel.geometry.voxels_per_block = voxels_per_block;
        panel.scene = scene.clone();
        let camera = OrbitCamera::default();
        let config = AppConfig::capture(&panel, &camera, HomeView::default(), [1280, 800]);
        assert!(config.scene.is_some(), "capture persists the scene");

        let json = serde_json::to_string_pretty(&config).expect("serialise");
        let restored: AppConfig = serde_json::from_str(&json).expect("deserialise");
        let restored_panel = restored.to_panel_state();

        assert_eq!(restored_panel.scene.nodes.len(), 2, "both nodes survive the reload");
        assert_eq!(restored_panel.scene.active, scene.active, "the active selection survives");
        assert_eq!(restored_panel.scene.nodes[1].transform.offset_blocks, [3, 0, 0]);

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
        let restored: AppConfig = serde_json::from_str("{}").expect("empty object parses");
        assert_eq!(restored, AppConfig::default());

        // Outright invalid JSON must be a clean Err (the caller turns it into a
        // defaults fallback), never a panic.
        assert!(serde_json::from_str::<AppConfig>("not json at all}{").is_err());
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
        let config: AppConfig = serde_json::from_str(old_json)
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
        let restored: AppConfig =
            serde_json::from_str(old_json).expect("old config (with debug_clouds) must still parse");
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
        assert_eq!(panel.scene.nodes.len(), 1);
        match panel.scene.active_node().map(|node| &node.content) {
            Some(crate::scene::NodeContent::Tool { shape, material }) => {
                // The default seed geometry, NOT the persisted flat params.
                assert_eq!(shape.kind, ShapeKind::Cylinder);
                assert_eq!(shape.size_blocks, [5, 1, 5]);
                // Density DID carry over from the config (app-level field).
                assert_eq!(shape.voxels_per_block, 20);
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
        let restored: AppConfig = serde_json::from_str(old_json)
            .expect("old config (with mesher) must still parse");
        // The flat geometry keys are ignored (issue #32); density + material survive.
        assert_eq!(restored.voxels_per_block, 8);
        assert_eq!(restored.material, MaterialChoice::Stone);
        // It loads cleanly to the default one-Tool-node seed scene (the stray `mesher`
        // and the removed flat geometry keys are all ignored).
        let panel = restored.to_panel_state();
        assert_eq!(panel.scene.nodes.len(), 1);
    }

    /// step 8 round-trip: a NON-TRIVIAL scene (top-level Tool + Part nodes with
    /// non-zero offsets and distinct materials, a Group with children, an
    /// `AssemblyDef`, and an `Instance` of it) survives
    /// `capture → JSON → deserialize → to_panel_state` structurally intact and
    /// resolves to the SAME occupied count.
    #[test]
    fn full_scene_round_trips_through_json() {
        use crate::scene::{
            AssemblyDef, CombineOp, DefId, Node, NodeContent, NodePath, Part, Scene,
        };
        use crate::voxel::SdfShape;

        let voxels_per_block = 8u32;
        let unit_box = |kind| SdfShape {
            kind,
            size_blocks: [1, 1, 1],
            voxels_per_block,
            wall_blocks: 1,
        };

        // A definition (the reusable "house" body): a single Wood box.
        let def_id = DefId(3);
        let def = AssemblyDef {
            id: def_id,
            name: "House".to_string(),
            children: vec![Node::new(
                "Body",
                NodeContent::Tool {
                    shape: unit_box(ShapeKind::Box),
                    material: MaterialChoice::Wood,
                },
            )],
        };

        // Top-level node 0: a Stone Tool at the origin.
        let stone = Node::new(
            "Stone",
            NodeContent::Tool {
                shape: unit_box(ShapeKind::Box),
                material: MaterialChoice::Stone,
            },
        );
        // Top-level node 1: a Clouds Part, offset.
        let mut clouds = Node::new("Clouds", NodeContent::Part(Part::DebugClouds { seed: 7 }));
        clouds.transform.offset_blocks = [3, 0, 0];
        // Top-level node 2: a Group containing a Plain Tool offset within it.
        let mut grouped_leaf = Node::new(
            "Leaf",
            NodeContent::Tool {
                shape: unit_box(ShapeKind::Sphere),
                material: MaterialChoice::Plain,
            },
        );
        grouped_leaf.transform.offset_blocks = [1, 0, 0];
        let mut group = Node::new("Group", NodeContent::Group(vec![grouped_leaf]));
        group.transform.offset_blocks = [6, 0, 0];
        group.operation = CombineOp::Union;
        // Top-level node 3: an Instance of the def, offset disjointly.
        let mut instance = Node::new("House instance", NodeContent::Instance(def_id));
        instance.transform.offset_blocks = [-6, 0, 0];

        let scene = Scene {
            nodes: vec![stone, clouds, group, instance],
            definitions: vec![def],
            active: Some(NodePath::from_indices(vec![2, 0])),
            ..Scene::default()
        };

        // Build a panel carrying this scene and capture → JSON → restore.
        let mut panel = PanelState::with_view_cube_default();
        panel.geometry.voxels_per_block = voxels_per_block;
        panel.scene = scene.clone();
        let camera = OrbitCamera::default();
        let config = AppConfig::capture(&panel, &camera, HomeView::default(), [1280, 800]);
        assert!(config.scene.is_some(), "capture persists the full scene");

        let json = serde_json::to_string_pretty(&config).expect("serialise");
        let restored: AppConfig = serde_json::from_str(&json).expect("deserialise");
        let restored_panel = restored.to_panel_state();

        // Structural equality: same node tree, definitions, and active selection.
        assert_eq!(
            restored_panel.scene.nodes.len(),
            scene.nodes.len(),
            "all top-level nodes survive"
        );
        assert_eq!(restored_panel.scene.definitions.len(), 1, "the def survives");
        assert_eq!(
            restored_panel.scene.active,
            scene.active,
            "the active selection survives"
        );
        // The Group's child and the def's body survive with their offsets/materials.
        match &restored_panel.scene.nodes[2].content {
            NodeContent::Group(children) => {
                assert_eq!(children.len(), 1);
                assert_eq!(children[0].transform.offset_blocks, [1, 0, 0]);
            }
            other => panic!("node 2 must stay a Group, got {other:?}"),
        }
        assert!(matches!(
            restored_panel.scene.nodes[3].content,
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
        let restored: AppConfig =
            serde_json::from_str(partial).expect("a partial scene object still parses");
        let panel = restored.to_panel_state();
        assert_eq!(
            panel.scene.nodes.len(),
            1,
            "an empty persisted scene falls back to the one-Tool-node seed"
        );

        // A `scene` whose nodes contain a content variant missing fields is a clean
        // parse error wholesale → `load()` would return None → caller uses defaults.
        // We assert it never panics: the deserialize is an Err, not an unwind.
        let broken = r#"{ "scene": { "nodes": [ { "content": "NotAVariant" } ] } }"#;
        assert!(
            serde_json::from_str::<AppConfig>(broken).is_err(),
            "a structurally broken scene is a clean Err (load → defaults), never a panic"
        );
    }

    /// S4a back-compat: a scene serialized by an OLDER build (when `offset_blocks`
    /// was `[i32; 3]`) loads into the now-`[i64; 3]` field unchanged. A JSON integer
    /// carries no width, so serde widens the old numbers transparently — this is the
    /// "tolerant persistence migration" the S4a task requires, proven by a
    /// hand-authored document that predates the widening (note the small i32-range
    /// offsets, exactly what an old save held). The document must load, keep its
    /// offsets, and resolve to a non-empty grid.
    #[test]
    fn old_i32_offset_scene_loads_after_widening_to_i64() {
        use crate::scene::NodeContent;

        // An old-style document: a single Box Tool offset +5 blocks in X. The
        // `offset_blocks` numbers are plain JSON integers, exactly as the pre-S4a
        // `[i32; 3]` field serialized them.
        let old_json = r#"{
            "scene": {
                "nodes": [
                    {
                        "name": "Box",
                        "transform": { "offset_blocks": [5, 0, 0] },
                        "operation": "Union",
                        "visible": true,
                        "content": {
                            "Tool": {
                                "shape": {
                                    "kind": "Box",
                                    "size_blocks": [2, 2, 2],
                                    "voxels_per_block": 8,
                                    "wall_blocks": 1
                                },
                                "material": "Stone"
                            }
                        }
                    }
                ],
                "active": { "indices": [0] }
            },
            "shape": "Box",
            "size_blocks": [2, 2, 2],
            "voxels_per_block": 8,
            "wall_blocks": 1
        }"#;

        let restored: AppConfig =
            serde_json::from_str(old_json).expect("an old i32-offset scene must still parse");
        let panel = restored.to_panel_state();
        assert_eq!(panel.scene.nodes.len(), 1, "the old node survives the widening");
        // The i32-range offset widened into the i64 field intact.
        assert_eq!(
            panel.scene.nodes[0].transform.offset_blocks,
            [5i64, 0, 0],
            "the old i32 offset must widen to the same i64 value"
        );
        assert!(matches!(
            panel.scene.nodes[0].content,
            NodeContent::Tool { .. }
        ));
        // The migrated document still resolves to a non-empty grid.
        let region = panel.scene.full_extent_blocks(8);
        assert!(
            panel.scene.resolve_region(region, 8, 0).occupied_count() > 0,
            "the migrated old-offset scene resolves to voxels"
        );
    }

    /// S4a: a scene whose `offset_blocks` is a LARGE i64 (well beyond the old
    /// `i32` range, ±2.1×10⁹) round-trips through `capture → JSON → load` byte-exact.
    /// This proves the widened field both serializes and deserializes the full
    /// 64-bit range — far-apart village nodes survive a save/load.
    #[test]
    fn large_i64_offset_round_trips_through_json() {
        use crate::scene::{Node, NodeContent, NodePath, Scene};
        use crate::voxel::SdfShape;

        // Beyond i32::MAX (2_147_483_647): a node placed ~3 billion blocks out. An
        // i32 field could never have held this; the i64 field must persist it exactly.
        let far_offset: i64 = 3_000_000_000;
        let shape = SdfShape {
            kind: ShapeKind::Box,
            size_blocks: [2, 2, 2],
            voxels_per_block: 8,
            wall_blocks: 1,
        };
        let mut node = Node::new(
            "Far box",
            NodeContent::Tool { shape, material: MaterialChoice::Stone },
        );
        node.transform.offset_blocks = [far_offset, -far_offset, far_offset / 2];
        let scene = Scene::single_node(node);

        let mut panel = PanelState::with_view_cube_default();
        panel.geometry.voxels_per_block = 8;
        panel.scene = scene.clone();
        let camera = OrbitCamera::default();
        let config = AppConfig::capture(&panel, &camera, HomeView::default(), [1280, 800]);

        let json = serde_json::to_string_pretty(&config).expect("serialise");
        let restored: AppConfig = serde_json::from_str(&json).expect("deserialise");
        let restored_panel = restored.to_panel_state();

        assert_eq!(
            restored_panel.scene.nodes.len(),
            1,
            "the far node survives the round-trip"
        );
        assert_eq!(
            restored_panel.scene.nodes[0].transform.offset_blocks,
            [far_offset, -far_offset, far_offset / 2],
            "a >i32-range i64 offset must round-trip byte-exact through save/load"
        );
        assert_eq!(
            restored_panel.scene.active,
            Some(NodePath::root_index(0)),
            "the active selection survives"
        );
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
        let config: AppConfig = serde_json::from_str(old_json).expect("old config parses");
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
        use crate::scene::{Node, NodeContent, NodePath, Point, Scene};
        use crate::voxel::SdfShape;

        let node = Node::new(
            "Box",
            NodeContent::Tool {
                shape: SdfShape { kind: ShapeKind::Box, size_blocks: [2, 2, 2], voxels_per_block: 8, wall_blocks: 1 },
                material: MaterialChoice::Stone,
            },
        );
        let mut scene = Scene {
            nodes: vec![node],
            active: Some(NodePath::root_index(0)),
            master_block_lattice: false,
            master_voxel_grid: true,
            master_floor_grid: false,
            ..Scene::default()
        };
        scene.ensure_origin_point();
        scene.add_point(Point { name: "Marker".to_string(), ..Point::default() });

        let mut panel = PanelState::with_view_cube_default();
        panel.scene = scene;
        let camera = OrbitCamera::default();
        let config = AppConfig::capture(&panel, &camera, HomeView::default(), [1280, 800]);

        let json = serde_json::to_string_pretty(&config).expect("serialise");
        let restored: AppConfig = serde_json::from_str(&json).expect("deserialise");
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
}
