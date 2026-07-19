//! Regression guard for the windowed palette-click apply path (BUG 2).
//!
//! This whole test requires a real wgpu device (it builds GPU-backed palette
//! tiles), so it probes for an adapter at runtime and skips loudly without one.
//!
//! We cannot click the live window from a test, but egui's interaction is
//! deterministic given a `RawInput` event stream. This test populates a real
//! `BlockPalette` with GPU-rendered thumbnail tiles, runs the SHARED
//! [`voxel_worker::build_panel`] (the exact function the window uses), injects a
//! synthetic press+release pointer sequence over a tile, and asserts the
//! returned `PanelResponse::clicked_palette_tile` reports a tile index — i.e.
//! the click propagates out of `build_panel` to the index the caller applies as
//! the active material.
//!
//! The tile rect is discovered by sweeping candidate points over the bottom-left
//! palette dock (the dock height + tile size are layout details that should not
//! be hard-coded), so the test stays robust to small layout tweaks.

use egui::{pos2, vec2, Event, PointerButton, Pos2, RawInput, Rect};

use assets::BlockGroup;
use voxel_worker::block_palette::PaletteHost;
use voxel_worker::{build_panel, EguiPaintBridge, GpuContext, PanelState};

/// A tiny solid-colour decoded RGBA image to stand in for a block texture.
fn dummy_decoded() -> (u32, u32, Vec<u8>) {
    let size = 4u32;
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);
    for _ in 0..(size * size) {
        pixels.extend_from_slice(&[0xb0, 0x90, 0x60, 0xff]);
    }
    (size, size, pixels)
}

/// Build a `PaletteHost` holding `count` real GPU-backed tiles.
fn build_palette(gpu: &GpuContext, bridge: &mut EguiPaintBridge, count: usize) -> PaletteHost {
    let mut palette = PaletteHost::new(&gpu.device, &gpu.queue, String::new());
    let decoded = dummy_decoded();
    for index in 0..count {
        let group = BlockGroup {
            label: format!("tile{index}"),
            key: format!("test/tile{index}"),
            variants: vec![std::path::PathBuf::from(format!("tile{index}.png"))],
        };
        palette.add_group(&gpu.device, &gpu.queue, &mut bridge.renderer, group, &decoded);
    }
    palette
}

#[test]
fn windowed_palette_tile_click_reaches_apply_path() {
    if skip_without_gpu("windowed_palette_tile_click_reaches_apply_path") {
        return;
    }
    let gpu = pollster::block_on(GpuContext::new(None));
    let mut bridge = EguiPaintBridge::new(&gpu.device, voxel_worker::COLOR_TARGET_FORMAT);

    let palette = build_palette(&gpu, &mut bridge, 3);
    assert_eq!(palette.ui.tiles.len(), 3, "three GPU tiles should be registered");

    let mut panel_state = PanelState::with_view_cube_default();
    let screen = Rect::from_min_size(pos2(0.0, 0.0), vec2(1280.0, 800.0));

    let mut run = |raw_input: RawInput, palette: &PaletteHost, state: &mut PanelState| {
        let mut response = None;
        let _ = bridge.context.run_ui(raw_input, |ui| {
            response = Some(build_panel(
                ui,
                state,
                voxel_worker::ExportPanelState::default(),
                &palette.ui,
            ));
        });
        response.unwrap()
    };

    let click_at = |run: &mut dyn FnMut(RawInput, &PaletteHost, &mut PanelState) -> _,
                    target: Pos2,
                    palette: &PaletteHost,
                    state: &mut PanelState| {
        run(
            RawInput {
                screen_rect: Some(screen),
                events: vec![
                    Event::PointerMoved(target),
                    Event::PointerButton {
                        pos: target,
                        button: PointerButton::Primary,
                        pressed: true,
                        modifiers: Default::default(),
                    },
                    Event::PointerButton {
                        pos: target,
                        button: PointerButton::Primary,
                        pressed: false,
                        modifiers: Default::default(),
                    },
                ],
                ..Default::default()
            },
            palette,
            state,
        )
    };

    // Sweep the bottom-left dock region for the first point that lands on a tile.
    // (The dock occupies the bottom strip of the window; tiles start at the left
    // edge below the "Blocks" header.)
    let mut hit: Option<usize> = None;
    'sweep: for y in (560..=799).step_by(6) {
        for x in (4..=320).step_by(6) {
            // A fresh layout frame before each probe clears prior hover state.
            let _ = run(
                RawInput {
                    screen_rect: Some(screen),
                    ..Default::default()
                },
                &palette,
                &mut panel_state,
            );
            let response = click_at(&mut run, pos2(x as f32, y as f32), &palette, &mut panel_state);
            if let Some(index) = response.clicked_palette_tile {
                hit = Some(index);
                break 'sweep;
            }
        }
    }

    assert!(
        hit.is_some(),
        "a click on the palette dock must report a tile index out of build_panel \
         (the windowed apply path consumes `clicked_palette_tile`); none found",
    );
}

/// Runtime GPU-availability probe — the replacement for the deleted `gpu` Cargo feature.
///
/// These tests used to be compiled out entirely behind `#![cfg(feature = "gpu")]`, which
/// meant a GPU-less machine did not skip them, it LOST them (and forgetting the flag made
/// the suite pass vacuously). Now they always compile and skip loudly here instead.
fn skip_without_gpu(test: &str) -> bool {
    static ADAPTER: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ADAPTER.get_or_init(voxel_worker::gpu::adapter_available) {
        return false;
    }
    eprintln!("skipping {test}: no GPU adapter on this machine");
    true
}
