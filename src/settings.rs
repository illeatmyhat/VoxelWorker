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

use crate::camera::{OrbitCamera, ProjectionMode};
use crate::panel::{GeometryParams, LayerRange, MaterialChoice, PanelState};
use crate::scene::Scene;
use crate::voxel::ShapeKind;

/// The whole persisted configuration. Every field is `#[serde(default)]` so a
/// partial or older config still loads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppConfig {
    // --- scene (ADR 0001 step 8: full scene persistence) ---
    // The whole assembly (node tree + reusable definitions + the active
    // selection) is persisted here. `#[serde(default)]` means an OLD config with
    // no `scene` field deserialises to `None`, which triggers the migration path
    // in `to_panel_state` (the flat `shape/size_blocks/...` fields below build a
    // one-Tool-node scene). A malformed/partial `scene` value can never reach this
    // field as garbage: serde tolerates missing inner fields (every scene field is
    // `#[serde(default)]`), and an outright unparseable config is rejected wholesale
    // by `load()` → defaults. Density (`voxels_per_block`) stays an app-level field
    // below; the scene reads it at resolve time (ADR 0001 "Density").
    //
    // regional export: deferred to the chunking milestone (ADR 0001 step 8's
    // "regional/streamed .vox export" sub-part — meaningless until chunking; the
    // current full-grid `.vox` export already covers bounded scenes).
    #[serde(default)]
    pub scene: Option<Scene>,

    // --- geometry (legacy single-Tool mirror; kept for back-compat migration) ---
    // Before step 8 the config persisted only the active/first Tool node's
    // geometry. These flat fields are still written (so a NEW config also opens in
    // an OLDER build that only reads them) and are the SOLE source when `scene` is
    // absent: an old config with no `scene` field migrates to a one-Tool-node scene
    // built from these. The old `debug_clouds: bool` field was dropped; an old
    // config carrying it still loads (serde ignores the now-unknown field).
    #[serde(default = "default_shape")]
    pub shape: ShapeKind,
    #[serde(default = "default_size")]
    pub size_blocks: [u32; 3],
    #[serde(default = "default_density")]
    pub voxels_per_block: u32,
    #[serde(default = "default_wall")]
    pub wall_blocks: u32,

    // --- display / material ---
    #[serde(default)]
    pub projection_mode: ProjectionMode,
    #[serde(default)]
    pub material: MaterialChoice,
    #[serde(default)]
    pub show_grid_overlay: bool,
    #[serde(default = "default_true")]
    pub show_block_lattice: bool,
    #[serde(default)]
    pub show_floor_grid: bool,
    #[serde(default = "default_true")]
    pub show_view_cube: bool,
    #[serde(default)]
    pub show_origin_gizmo: bool,
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

    // --- window ---
    #[serde(default = "default_window_size")]
    pub window_size: [u32; 2],
}

fn default_shape() -> ShapeKind {
    ShapeKind::Cylinder
}
fn default_size() -> [u32; 3] {
    [5, 1, 5]
}
fn default_density() -> u32 {
    16
}
fn default_wall() -> u32 {
    1
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
            shape: default_shape(),
            size_blocks: default_size(),
            voxels_per_block: default_density(),
            wall_blocks: default_wall(),
            projection_mode: ProjectionMode::default(),
            material: MaterialChoice::default(),
            show_grid_overlay: false,
            show_block_lattice: true,
            show_floor_grid: false,
            show_view_cube: true,
            show_origin_gizmo: false,
            applied_block_label: None,
            snap_to_blocks: true,
            onion_skin: false,
            onion_depth: default_onion_depth(),
            orbit_theta: default_theta(),
            orbit_phi: default_phi(),
            orbit_distance: default_distance(),
            window_size: default_window_size(),
        }
    }
}

impl AppConfig {
    /// Capture the persisted fields from the live [`PanelState`], [`OrbitCamera`]
    /// and the current window size.
    pub fn capture(panel: &PanelState, camera: &OrbitCamera, window_size: [u32; 2]) -> Self {
        Self {
            // step 8: persist the whole scene (node tree + definitions + active
            // selection). The legacy flat geometry fields below keep mirroring the
            // active-Tool inspector so a NEW config still opens in an older build.
            scene: Some(panel.scene.clone()),
            shape: panel.geometry.shape,
            size_blocks: panel.geometry.size_blocks,
            voxels_per_block: panel.geometry.voxels_per_block,
            wall_blocks: panel.geometry.wall_blocks,
            projection_mode: panel.projection_mode,
            material: panel.material,
            show_grid_overlay: panel.show_grid_overlay,
            show_block_lattice: panel.show_block_lattice,
            show_floor_grid: panel.show_floor_grid,
            show_view_cube: panel.show_view_cube,
            show_origin_gizmo: panel.show_origin_gizmo,
            applied_block_label: panel.applied_block_label.clone(),
            snap_to_blocks: panel.layer_range.snap_to_blocks,
            onion_skin: panel.layer_range.onion_skin,
            onion_depth: panel.layer_range.onion_depth,
            orbit_theta: camera.orbit_theta,
            orbit_phi: camera.orbit_phi,
            orbit_distance: camera.orbit_distance,
            window_size,
        }
    }

    /// Build the [`PanelState`] this config describes.
    ///
    /// step 8 (ADR 0001): the full scene (node tree + definitions + active
    /// selection) is restored from [`scene`](Self::scene) when present. When it is
    /// absent — an OLD config that predates scene persistence — the flat
    /// `shape/size_blocks/...` fields **migrate** to a one-Tool-node scene via
    /// [`PanelState::seed_scene_from_geometry`]. A restored scene that resolves to
    /// no nodes (a malformed/empty persisted scene) also falls back to that seed,
    /// so loading never yields an empty document and never panics.
    pub fn to_panel_state(&self) -> PanelState {
        let mut state = PanelState {
            geometry: GeometryParams {
                shape: self.shape,
                size_blocks: self.size_blocks,
                voxels_per_block: self.voxels_per_block,
                wall_blocks: self.wall_blocks,
            },
            projection_mode: self.projection_mode,
            material: self.material,
            show_grid_overlay: self.show_grid_overlay,
            show_block_lattice: self.show_block_lattice,
            show_floor_grid: self.show_floor_grid,
            show_view_cube: self.show_view_cube,
            show_origin_gizmo: self.show_origin_gizmo,
            // Face-orientation debug is a transient verification mode; it is not
            // persisted, so it always starts off.
            debug_face_orientation: false,
            // The mesher choice is a session-only toggle (ADR 0002, part of #18);
            // not persisted, so it always starts at the default (now cuboid).
            mesher: crate::panel::MesherChoice::default(),
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
            // config without one — a one-Tool-node scene seeded from the geometry.
            scene: Scene::default(),
        };
        // step 8: restore the persisted full scene when present and non-empty;
        // otherwise migrate the legacy single geometry into a one-Tool-node scene.
        // A `Some(scene)` with no nodes (a malformed/empty persisted scene) is
        // treated as absent, so the seed always produces a usable document.
        match &self.scene {
            Some(scene) if !scene.nodes.is_empty() => {
                state.scene = scene.clone();
            }
            _ => state.seed_scene_from_geometry(),
        }
        // issue #29 (grid rework S1): every loaded scene gains exactly one Origin
        // Point (idempotent — a scene that already carries one is untouched), and
        // the scene-wide grid MASTERS migrate from the legacy `show_*` config
        // fields so an existing user's prefs carry over. We only seed the masters
        // when the persisted scene predates them (an old config with no `scene`,
        // or a scene whose masters are still at their struct default) — otherwise
        // a scene saved by a #29-aware build keeps its own masters.
        //
        // NOTE: S1 does NOT rewire any renderer. The masters are data only; the
        // existing `PanelState.show_*` toggles (set above) still drive the live
        // renderers unchanged. Master→renderer wiring is S3/S4.
        let scene_predates_points = match &self.scene {
            Some(scene) => scene.points.is_empty() && !scene.nodes.is_empty(),
            None => true,
        };
        if scene_predates_points {
            state.scene.master_block_lattice = self.show_block_lattice;
            state.scene.master_voxel_grid = self.show_grid_overlay;
            state.scene.master_floor_grid = self.show_floor_grid;
        }
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

    #[test]
    fn config_round_trips_through_json() {
        let config = AppConfig {
            scene: None,
            shape: ShapeKind::Torus,
            size_blocks: [7, 3, 9],
            voxels_per_block: 24,
            wall_blocks: 2,
            projection_mode: ProjectionMode::Orthographic,
            material: MaterialChoice::Wood,
            show_grid_overlay: true,
            show_block_lattice: false,
            show_floor_grid: true,
            show_view_cube: false,
            show_origin_gizmo: true,
            applied_block_label: Some("Granite".to_string()),
            snap_to_blocks: false,
            onion_skin: true,
            onion_depth: 5,
            orbit_theta: 1.23,
            orbit_phi: 0.95,
            orbit_distance: 42.0,
            window_size: [1600, 900],
        };

        let json = serde_json::to_string_pretty(&config).expect("serialise");
        let restored: AppConfig = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(config, restored);
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

    /// step 2 migration: an OLD config that still carries the dropped
    /// `debug_clouds` boolean must load gracefully — serde ignores the now-unknown
    /// field, and every other field round-trips to its persisted value.
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
        assert_eq!(restored.shape, ShapeKind::Sphere);
        assert_eq!(restored.size_blocks, [3, 4, 5]);
        assert_eq!(restored.voxels_per_block, 20);
        assert_eq!(restored.material, MaterialChoice::Wood);
        // step 8: an old config has NO `scene` field, so it deserialises to `None`,
        // which is exactly what triggers the migration to a one-Tool-node scene.
        assert!(restored.scene.is_none(), "an old flat config carries no scene");

        // It migrates to a one-Tool-node scene built from the flat geometry (no
        // Clouds Part — the boolean is dropped; Clouds is now an Add-a-Part action),
        // and that node's shape matches the persisted flat params.
        let panel = restored.to_panel_state();
        assert_eq!(panel.scene.nodes.len(), 1);
        match panel.scene.active_node().map(|node| &node.content) {
            Some(crate::scene::NodeContent::Tool { shape, material }) => {
                assert_eq!(shape.kind, ShapeKind::Sphere);
                assert_eq!(shape.size_blocks, [3, 4, 5]);
                assert_eq!(shape.voxels_per_block, 20);
                assert_eq!(shape.wall_blocks, 2);
                assert_eq!(*material, MaterialChoice::Wood);
            }
            other => panic!("migration must build a one Tool node, got {other:?}"),
        }
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
        let config = AppConfig::capture(&panel, &camera, [1280, 800]);
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
        let config = AppConfig::capture(&panel, &camera, [1280, 800]);

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

    #[test]
    fn capture_then_to_panel_state_preserves_toggles() {
        let mut panel = PanelState::with_view_cube_default();
        panel.show_floor_grid = true;
        panel.material = MaterialChoice::Plain;
        let camera = OrbitCamera::default();
        let config = AppConfig::capture(&panel, &camera, [1024, 768]);
        let restored = config.to_panel_state();
        assert_eq!(restored.show_block_lattice, panel.show_block_lattice);
        assert_eq!(restored.show_floor_grid, panel.show_floor_grid);
        assert_eq!(restored.material, panel.material);
        assert_eq!(restored.geometry, panel.geometry);
    }

    /// issue #29 (grid rework S1): loading an OLD config (no `scene` field — the
    /// legacy flat geometry) gains exactly one Origin Point on the load path, and
    /// the scene-wide grid masters MIGRATE from the legacy `show_*` fields so an
    /// existing user's prefs carry over.
    #[test]
    fn old_config_gains_origin_point_and_migrates_masters() {
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

        // Masters migrated from the legacy show_* fields.
        assert!(!panel.scene.master_block_lattice, "migrated from show_block_lattice=false");
        assert!(panel.scene.master_voxel_grid, "migrated from show_grid_overlay=true");
        assert!(panel.scene.master_floor_grid, "migrated from show_floor_grid=true");

        // S1 does NOT rewire renderers: the legacy PanelState.show_* toggles still
        // mirror the config exactly (they keep driving the live renderers).
        assert!(!panel.show_block_lattice);
        assert!(panel.show_grid_overlay);
        assert!(panel.show_floor_grid);
    }

    /// issue #29: a scene saved by a #29-aware build (already carrying an Origin
    /// Point and its own masters) keeps its masters on reload — the legacy `show_*`
    /// migration only seeds masters for a scene that predates Points. The Origin is
    /// not duplicated.
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
        // The legacy config show_* are the OPPOSITE of the scene masters, to prove
        // they do NOT overwrite a modern scene's masters.
        panel.show_block_lattice = true;
        panel.show_grid_overlay = false;
        panel.show_floor_grid = true;
        let camera = OrbitCamera::default();
        let config = AppConfig::capture(&panel, &camera, [1280, 800]);

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
