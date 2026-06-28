//! Regression guards for the C4a panel→`Intent` routing (ADR 0003 Phase C).
//!
//! Slice C4a flips the windowed panel from MUTATING `PanelState.scene` directly to
//! DESCRIBING each mutation as an [`Intent`] in [`PanelResponse::intents`], which the
//! loop applies through `AppCore::apply_intent`. These tests drive the SHARED
//! [`voxel_worker::build_panel`] headlessly (a bare egui `Context` — `build_panel`
//! itself needs no GPU; only palette tiles do, and we pass an empty palette) and
//! assert the routing invariants that protect the goldens + the live app:
//!
//!   * A frame with NO pointer input emits ZERO intents and never moves the scene —
//!     the headless render path (`shot`, the goldens) stays mutation-free.
//!   * The panel never mutates `state.scene` itself during a frame: the scene a frame
//!     STARTED with is byte-identical to the one it ENDS with (the loop, not the
//!     panel, applies the emitted intents).

use egui::{Context, RawInput, Rect, pos2, vec2};

use voxel_worker::block_palette::BlockPalette;
use voxel_worker::{build_panel, PanelState};

/// Run one headless `build_panel` frame over a 1280×720 surface with an empty
/// palette, returning the `PanelResponse`. No pointer events are injected unless the
/// caller pre-loads them into `raw_input`. The response is captured out of the egui
/// closure via an outer binding (egui's `run` does not surface the closure's value).
fn run_frame(state: &mut PanelState, raw_input: RawInput) -> voxel_worker::PanelResponse {
    let context = Context::default();
    let palette = BlockPalette::default();
    let mut response = voxel_worker::PanelResponse::default();
    // Mirror `run_egui_frame`: `build_panel` shows its own side + bottom panels inside
    // the root `ui` handed to `run_ui` (grid_y / measured_diameter are readout-only).
    let _ = context.run_ui(raw_input, |ui| {
        response = build_panel(ui, state, 16, 0, &palette);
    });
    response
}

/// A windowed-default panel state (one Tool node + the Origin Point, ids minted),
/// exactly what the live app starts from.
fn windowed_state() -> PanelState {
    PanelState::with_view_cube_default()
}

#[test]
fn idle_frame_emits_no_intents() {
    let mut state = windowed_state();
    let raw_input = RawInput {
        screen_rect: Some(Rect::from_min_size(pos2(0.0, 0.0), vec2(1280.0, 720.0))),
        ..Default::default()
    };
    let response = run_frame(&mut state, raw_input);
    assert!(
        response.intents.is_empty(),
        "a frame with no input must emit zero intents, got {:?}",
        response.intents
    );
    assert!(
        !response.frame_after_apply,
        "a no-input frame must not request an auto-frame"
    );
}

#[test]
fn panel_never_mutates_scene_in_place() {
    // The panel DESCRIBES mutations; it must not apply them itself. So the scene a
    // frame starts with equals the scene it ends with, regardless of input — the loop
    // is the only mutator (via `apply_intent`).
    let mut state = windowed_state();
    let scene_before = state.scene.clone();
    let raw_input = RawInput {
        screen_rect: Some(Rect::from_min_size(pos2(0.0, 0.0), vec2(1280.0, 720.0))),
        ..Default::default()
    };
    let _ = run_frame(&mut state, raw_input);
    assert_eq!(
        state.scene, scene_before,
        "build_panel must not mutate state.scene in place (C4a routes through intents)"
    );
}
