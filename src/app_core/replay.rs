//! Intent-script replay — the testable core of `shot --replay` (ADR 0003 Phase C, C3).

use camera::OrbitCamera;
use document::intent::Intent;
use document::scene::Scene;

use super::AppCore;


/// The **default seed scene** the windowed app starts from (ADR 0003 Phase C, slice
/// C3 — the base a `shot --replay` script is applied against). A single Tool node
/// from the default geometry/material, the Origin Point synthesized, stable
/// [`NodeId`](document::scene::NodeId)s minted — i.e. exactly `PanelState::with_view_cube_default().scene`
/// (which runs `Scene::from_geometry(default)` + `ensure_origin_point` +
/// `ensure_node_ids`). Kept here so both `bin/shot` and the lib tests build the
/// replay base the same way.
pub fn default_replay_seed_scene() -> Scene {
    ui::panel::PanelState::with_view_cube_default().scene
}

/// Replay a **newline-delimited-JSON Intent script** into a [`Scene`] (ADR 0003
/// Phase C, slice C3 — the testable core of `shot --replay`).
///
/// The `script` is one [`Intent`] per line: each non-empty line is parsed with
/// `serde_json::from_str::<Intent>` and applied IN ORDER, via
/// [`AppCore::apply_intent`], to the [`default_replay_seed_scene`]. Blank /
/// whitespace-only lines are skipped. Returns the post-replay scene.
///
/// On a JSON parse error on any line, returns `Err` with a message naming the
/// 1-based line number and the offending line (no panic) — the caller prints it and
/// exits non-zero. `bin/shot` reads the file then calls this; the lib tests feed a
/// string directly (keeping the GPU render out of the unit test).
pub fn replay_intent_script(script: &str) -> Result<Scene, String> {
    let mut scene = default_replay_seed_scene();
    let mut app_core = AppCore::new(OrbitCamera::default());
    for (line_index, raw_line) in script.lines().enumerate() {
        let line_number = line_index + 1;
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let intent: Intent = serde_json::from_str(trimmed).map_err(|error| {
            format!("parse error on line {line_number}: {error}\n  line: {trimmed}")
        })?;
        app_core.apply_intent(&mut scene, intent);
    }
    Ok(scene)
}
