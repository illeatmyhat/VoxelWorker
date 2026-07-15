//! The Layers section (issue #12): the layer-range scrubber widget.

use super::{LayerRange, PanelState};

/// The Layers section (issue #12): the layer-range scrubber that subsumes the old
/// 2D mid-vertical slice map. Z-up: layers are Z-slices. A video-clip-style track
/// over `0..grid_z` with two trim handles (lower/upper), the selected band
/// highlighted, block-boundary ticks, the layers/blocks readout, the snap + onion
/// controls, and the measured-diameter stat line (widest occupied run in the band).
pub(super) fn build_layers_section(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    grid_z: u32,
    measured_diameter: u32,
) {
    ui.add_space(8.0);
    ui.strong("Layers");

    let voxels_per_block = state.geometry.voxels_per_block.max(1);
    // The scrubber edits `state.layer_range` in place; the bounds are kept valid
    // (clamped to grid_z, lower <= upper, snapped if requested) by the widget.
    layer_scrubber(ui, &mut state.layer_range, grid_z, voxels_per_block);

    let range = state.layer_range;
    // Readout: "layers L–U of N · blocks b0–b1".
    let block_lower = range.lower / voxels_per_block;
    let block_upper = range.upper.saturating_sub(1).max(range.lower) / voxels_per_block;
    ui.label(
        egui::RichText::new(format!(
            "layers {}–{} of {grid_z} · blocks {block_lower}–{block_upper}",
            range.lower, range.upper
        ))
        .small()
        .weak(),
    );

    ui.add_space(4.0);
    ui.checkbox(&mut state.layer_range.snap_to_blocks, "Snap to blocks");
    if state.layer_range.snap_to_blocks {
        // Re-snap the current handles immediately so toggling snap on tidies them.
        state.layer_range.lower =
            LayerRange::snap_value(state.layer_range.lower, voxels_per_block, grid_z);
        state.layer_range.upper =
            LayerRange::snap_value(state.layer_range.upper, voxels_per_block, grid_z);
        if state.layer_range.lower > state.layer_range.upper {
            std::mem::swap(&mut state.layer_range.lower, &mut state.layer_range.upper);
        }
    }
    ui.checkbox(&mut state.layer_range.onion_skin, "Onion skin");
    if state.layer_range.onion_skin {
        let mut depth = state.layer_range.onion_depth.clamp(1, 8);
        if ui
            .add(egui::Slider::new(&mut depth, 1..=8).text("onion depth"))
            .changed()
        {
            state.layer_range.onion_depth = depth;
        }
    }

    // Measured-diameter stat line: the widest occupied voxel run in the active
    // band (the chisel-diameter readout the old 2D slice carried).
    let blocks = measured_diameter as f32 / voxels_per_block as f32;
    ui.label(
        egui::RichText::new(format!("Ø {measured_diameter} vx · {blocks:.2} bl"))
            .small()
            .weak(),
    );
    ui.separator();
}

/// Custom range-scrubber widget (issue #12). Z-up: layers are Z-slices, so it paints
/// a track spanning `0..grid_z` with block-boundary ticks, the selected band
/// highlighted, and two draggable trim handles (lower/upper). Drag is handled via
/// `ui.interact` + the pointer: the nearer handle to the press grabs, then follows
/// the pointer (snapped to block boundaries when `snap_to_blocks` is on). Keeps
/// `lower <= upper` by swapping when the handles cross. Edits `range` in place.
fn layer_scrubber(
    ui: &mut egui::Ui,
    range: &mut LayerRange,
    grid_z: u32,
    voxels_per_block: u32,
) {
    let grid_z = grid_z.max(1);
    let track_height = 26.0;
    let handle_half_width = 5.0;
    let desired = egui::vec2(ui.available_width(), track_height + 14.0);
    let (rect, response) = ui.allocate_exact_size(desired, egui::Sense::click_and_drag());

    // The track is inset so the handles have room at both ends.
    let track_left = rect.left() + handle_half_width + 2.0;
    let track_right = rect.right() - handle_half_width - 2.0;
    let track_width = (track_right - track_left).max(1.0);
    let track_top = rect.top() + 4.0;
    let track_bottom = track_top + track_height;
    let track_rect = egui::Rect::from_min_max(
        egui::pos2(track_left, track_top),
        egui::pos2(track_right, track_bottom),
    );

    // Map a layer index <-> an x pixel on the track.
    let layer_to_x = |layer: u32| -> f32 {
        track_left + (layer as f32 / grid_z as f32) * track_width
    };
    let x_to_layer = |x: f32| -> u32 {
        let t = ((x - track_left) / track_width).clamp(0.0, 1.0);
        (t * grid_z as f32).round() as u32
    };

    let painter = ui.painter_at(rect);
    let visuals = ui.visuals();

    // Track background.
    painter.rect_filled(track_rect, 3.0, egui::Color32::from_rgb(0x1b, 0x17, 0x12));

    // Block-boundary tick marks every `voxels_per_block` layers (the snap points).
    let mut boundary = 0u32;
    while boundary <= grid_z {
        let x = layer_to_x(boundary);
        painter.line_segment(
            [egui::pos2(x, track_top), egui::pos2(x, track_bottom)],
            egui::Stroke::new(1.0, egui::Color32::from_rgb(0x3a, 0x5f, 0x57)),
        );
        if boundary == grid_z {
            break;
        }
        boundary = (boundary + voxels_per_block).min(grid_z);
        if boundary == grid_z {
            // Draw the final endpoint tick then stop.
            let x = layer_to_x(grid_z);
            painter.line_segment(
                [egui::pos2(x, track_top), egui::pos2(x, track_bottom)],
                egui::Stroke::new(1.0, egui::Color32::from_rgb(0x3a, 0x5f, 0x57)),
            );
            break;
        }
    }

    // Selected band highlight between the handles.
    let lower_x = layer_to_x(range.lower);
    let upper_x = layer_to_x(range.upper);
    let band_rect = egui::Rect::from_min_max(
        egui::pos2(lower_x.min(upper_x), track_top),
        egui::pos2(lower_x.max(upper_x), track_bottom),
    );
    painter.rect_filled(band_rect, 0.0, egui::Color32::from_rgba_unmultiplied(0x5f, 0xb8, 0xa4, 70));

    // Drag handling: on press, grab whichever handle is nearer the pointer; while
    // dragging, that handle follows the pointer.
    if response.drag_started() || (response.clicked() && response.hover_pos().is_some()) {
        if let Some(pos) = response.interact_pointer_pos() {
            let dist_lower = (pos.x - lower_x).abs();
            let dist_upper = (pos.x - upper_x).abs();
            // Stash which handle is active in egui temp memory keyed by widget id.
            let active_upper = dist_upper < dist_lower;
            ui.memory_mut(|m| m.data.insert_temp(response.id, active_upper));
        }
    }
    if response.dragged() || response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let active_upper = ui
                .memory(|m| m.data.get_temp::<bool>(response.id))
                .unwrap_or_else(|| {
                    (pos.x - upper_x).abs() < (pos.x - lower_x).abs()
                });
            let mut value = x_to_layer(pos.x);
            if range.snap_to_blocks {
                value = LayerRange::snap_value(value, voxels_per_block, grid_z);
            }
            if active_upper {
                range.upper = value;
            } else {
                range.lower = value;
            }
            if range.lower > range.upper {
                std::mem::swap(&mut range.lower, &mut range.upper);
                // The active handle effectively swapped sides; update the memory so
                // continued dragging keeps tracking the same pointer.
                ui.memory_mut(|m| m.data.insert_temp(response.id, !active_upper));
            }
        }
    }

    // Draw the two handles last so they sit on top of the band.
    let handle_color = visuals.widgets.active.fg_stroke.color;
    for layer in [range.lower, range.upper] {
        let x = layer_to_x(layer);
        let handle_rect = egui::Rect::from_min_max(
            egui::pos2(x - handle_half_width, track_top - 3.0),
            egui::pos2(x + handle_half_width, track_bottom + 3.0),
        );
        painter.rect_filled(handle_rect, 2.0, handle_color);
        painter.rect_stroke(
            handle_rect,
            2.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(0x10, 0x0c, 0x08)),
            egui::StrokeKind::Inside,
        );
    }
}
