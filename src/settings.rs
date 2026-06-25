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
use crate::panel::{GeometryParams, MaterialChoice, PanelState};
use crate::voxel::ShapeKind;

/// The whole persisted configuration. Every field is `#[serde(default)]` so a
/// partial or older config still loads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppConfig {
    // --- geometry ---
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

impl Default for AppConfig {
    fn default() -> Self {
        Self {
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
            orbit_theta: camera.orbit_theta,
            orbit_phi: camera.orbit_phi,
            orbit_distance: camera.orbit_distance,
            window_size,
        }
    }

    /// Build the [`PanelState`] this config describes.
    pub fn to_panel_state(&self) -> PanelState {
        PanelState {
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
            voxel_cap_warning_millions: None,
            // Re-applied lazily/best-effort: only the label is restored (for the
            // panel readout); the material itself reverts to procedural.
            applied_block_label: self.applied_block_label.clone(),
        }
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
}
