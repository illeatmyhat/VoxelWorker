//! The egui half of the per-frame pipeline (ADR 0031 — split out of the former monolithic
//! `lib.rs`): build the panel + tessellate the UI ([`run_egui_frame`]), the persistent
//! [`EguiPaintBridge`], the [`PreparedEguiFrame`] it produces, and the view-cube context-menu
//! request. The GPU pass recording is the sibling [`render`](super::render).

use crate::*;

/// Everything needed to translate egui output into wgpu draw calls, plus the
/// persistent egui context. Lives for the whole program; reused every frame.
pub struct EguiPaintBridge {
    pub context: egui::Context,
    pub renderer: egui_wgpu::Renderer,
}

impl EguiPaintBridge {
    /// Build the bridge for a given render-target format.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let renderer = egui_wgpu::Renderer::new(
            device,
            target_format,
            egui_wgpu::RendererOptions {
                // egui feathers its own AA at 1 sample. M4 splits the frame into a
                // 4× MSAA 3D pass (resolved) followed by a separate egui pass that
                // loads the resolved single-sample target — so egui's pipeline
                // needs neither MSAA nor a depth attachment.
                msaa_samples: 1,
                depth_stencil_format: None,
                dithering: true,
                predictable_texture_filtering: false,
            },
        );
        Self {
            context: egui::Context::default(),
            renderer,
        }
    }
}

/// A ViewCube right-click context-menu item the user chose this frame (#13
/// Step 3). The windowed caller executes it after `run_egui_frame` returns; egui
/// draws the menu and swallows its own clicks, so these never leak to the
/// left-click snap path. `OrthographicToggle` is handled INSIDE `run_egui_frame`
/// (it just flips `panel_state.projection_mode`, the same field the side panel
/// binds, keeping the two in sync), so it is not surfaced here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewCubeMenuRequest {
    /// "Home" — snap to the saved home view.
    Home,
    /// "Fit" — frame the model.
    Fit,
    /// "Set current as home" — capture the live camera as the home view.
    SetHome,
}

/// The fully-prepared egui draw data for one frame.
///
/// Produced by [`run_egui_frame`] and consumed by [`render_frame`](super::render::render_frame). Keeping it
/// in a struct lets the windowed path interleave winit-specific work (feeding
/// `platform_output` back to the window) between the two steps.
pub struct PreparedEguiFrame {
    pub paint_jobs: Vec<egui::ClippedPrimitive>,
    pub screen_descriptor: egui_wgpu::ScreenDescriptor,
    pub textures_to_free: Vec<egui::TextureId>,
    pub platform_output: egui::PlatformOutput,
    /// What the user changed in the panel this frame (M3): drives the geometry
    /// rebuild + camera auto-frame in the caller.
    pub panel_response: PanelResponse,
    /// The central 3D viewport rect in PHYSICAL PIXELS (issue #25): `[x, y, w, h]`
    /// = the window/target area LEFT of the right side panel and ABOVE the bottom
    /// palette dock. Derived from egui's post-panel `available_rect` × the frame's
    /// `pixels_per_point`, then clamped into the target. The caller computes the
    /// camera aspect from `w/h` and confines the 3D pass (voxels, gizmo, fog, view
    /// cube) to this rect, so the model is centred in the VISIBLE 3D area instead
    /// of the whole window (which the panels would otherwise cover).
    pub viewport_px: [u32; 4],
    /// The ViewCube context-menu item chosen this frame (#13 Step 3), if any. The
    /// caller runs Home/Fit/SetHome; the ortho toggle is applied in-place to
    /// `panel_state.projection_mode` and is not reported here.
    pub cube_menu_request: Option<ViewCubeMenuRequest>,
    /// The Signal icon rail's Home / Fit click this frame (ADR 0018 Decision 8), if any,
    /// pre-mapped onto the SAME [`ChromeClickAction`] the retired cube badges dispatched
    /// so the caller runs it through the shell's existing `run_chrome_action` (no forked
    /// logic). The rail's viewport-mode-cycle button is applied IN PLACE to
    /// `panel_state.view_mode` (pure display state), like the ortho toggle, so it is not
    /// reported here. `None` on the headless `shot` path (the rail is never clicked).
    pub rail_action: Option<ChromeClickAction>,
    /// Signal (issue #88): the horizontal inset (PHYSICAL PIXELS) from the central
    /// viewport's RIGHT edge to the view cube's right edge, so the cube + rail slide left
    /// of the floating display stack and track its fold state. The caller feeds it to
    /// [`view_cube_corner`] (the GPU cube draw) and caches it for the cube hit-testing, so
    /// the drawn cube, its pick rect and the egui rail share one anchor.
    pub view_cube_right_inset_px: u32,
    /// The Signal chrome's hit-rects in PHYSICAL PIXELS (`[x, y, w, h]`): the floating
    /// display stack plus the icon rail. The windowed shell gates camera input
    /// (orbit / pan / wheel-zoom) OFF inside these, the same way `position_in_view_cube`
    /// reserves the cube region — the stack no longer allocates in the root ui (the #88
    /// full-width dead-band regression), so egui's own "over egui" heuristic no longer
    /// covers this chrome and the shell must.
    pub chrome_rects_px: Vec<[f32; 4]>,
}

/// Run the egui pass for one frame: build the panel, upload changed textures to
/// the GPU, and tessellate the UI into paint jobs.
///
/// This is the render-target-agnostic half of egui integration. Both binaries
/// call it; the windowed binary supplies `raw_input` from `egui_winit`, the
/// headless binary builds `raw_input` by hand.
#[allow(clippy::too_many_arguments)]
pub fn run_egui_frame(
    bridge: &mut EguiPaintBridge,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    panel_state: &mut PanelState,
    grid_z: u32,
    measured_diameter: u32,
    export: ui::panel::ExportPanelState,
    palette: &crate::block_palette::BlockPalette,
    raw_input: egui::RawInput,
    size_in_pixels: [u32; 2],
    pixels_per_point: f32,
    // #13 Step 3: position (in egui points) of an open ViewCube right-click
    // context menu, or `None`. Drawn inside the egui pass so egui swallows the
    // menu's clicks. The menu clears this (`= None`) on selection or click-away.
    // The headless `shot` path passes `&mut None` (no menu).
    cube_context_menu_at: &mut Option<egui::Pos2>,
    // The general **viewport** right-click context menu's open position (PHYSICAL window pixels,
    // like `cube_context_menu_at` — divided by `pixels_per_point` here), or `None`. Drawn inside
    // the egui pass so egui swallows the menu's clicks; a mode-dispatched Delete acts on the sketch
    // selection (sketch mode) or the active node (normal mode). The headless `shot` path passes
    // `&mut None` (no menu).
    viewport_menu_at: &mut Option<egui::Pos2>,
    // Signal (#86): the hovered view-cube zone's name (e.g. `TOP·FRONT`), drawn as a
    // faint readout line under the cube. `None` when nothing is hovered — and always
    // `None` on the headless `shot` path, so the goldens stay pure cube geometry.
    view_cube_zone_readout: Option<&str>,
    // Owner ruling 2026-07-21: the armed primitive's kind, or `None` when nothing is armed.
    // `Some` draws the floating `Add <shape>` dialog with the placement-snap sliders.
    armed_shape: Option<voxel_core::voxel::ShapeKind>,
    // ADR 0028 (#94): the sketch profile's vertex handles for THIS frame — each already
    // projected to a screen position (egui points) with its interaction state. Empty unless
    // a sketch is being edited. Drawn as a foreground overlay + registered as chrome so a
    // handle drag never orbits the camera. The shell owns projection / hit-test / drag.
    sketch_handles: &[(egui::Pos2, ui::gizmos::HandleState)],
    // ADR 0030: the sketch's committed segment lines for THIS frame — each a pair of already-
    // projected endpoints (egui points) plus its interaction state. Drawn UNDER the vertex handles
    // so the profile reads as connected edges (an open sketch resolves to nothing, so these are
    // the only shape cue); the hovered segment draws brighter (Select) or warn-red with a `✕`
    // (Delete). Empty unless a sketch is being edited, always empty on the headless `shot` path.
    sketch_segment_lines: &[(egui::Pos2, egui::Pos2, ui::gizmos::HandleState)],
    // ADR 0028 (#95): the add-point insert-preview marker for THIS frame (egui points), or
    // `None` when the add-point tool is idle / no edge is hovered. Drawn as a diamond on the
    // hovered profile edge. Always `None` on the headless `shot` path.
    sketch_insert_preview: Option<egui::Pos2>,
) -> PreparedEguiFrame {
    let mut panel_response = PanelResponse::default();
    let mut cube_menu_request: Option<ViewCubeMenuRequest> = None;
    // Signal (ADR 0018 Decision 8): the icon rail's Home/Fit click, pre-mapped onto the
    // shell's `ChromeClickAction`; a mode-cycle click mutates `panel_state.view_mode` in
    // place inside the closure (never surfaced), like the ortho toggle.
    let mut rail_action: Option<ChromeClickAction> = None;
    // Signal (issue #88): the cube's right inset (physical px) = the display stack's current
    // width, computed inside the closure once the central rect + fold state are known.
    let mut view_cube_right_inset_px: u32 = 0;
    // The Signal chrome hit-rects (egui points; converted to px after the frame): the
    // stack + the rail — the shell's camera gate reads these (see `chrome_rects_px`).
    let mut chrome_rects_points: Vec<egui::Rect> = Vec::new();
    // Issue #25: the central 3D viewport rect, in egui points. `build_panel` shows
    // the right side panel + bottom palette dock INSIDE `ui`; whatever room those
    // panels leave is the central area where the 3D scene should be centred. We
    // read it AFTER the panels are laid out (`available_rect`), so a resized panel
    // moves the viewport with it.
    let mut central_rect_points = egui::Rect::from_min_size(
        egui::pos2(0.0, 0.0),
        egui::vec2(size_in_pixels[0] as f32, size_in_pixels[1] as f32),
    );
    // Signal (issue #89): dress the WHOLE app in the Signal instrument-panel skin — the
    // right sidebar + bottom palette dock inherit the near-black fills, hairlines,
    // monospace type and the one accent from `theme`. Applied to both the dark and
    // light context styles so it holds regardless of theme; the floating DISPLAY stack
    // re-scopes its own variant, and the chrome painters (cube/rail/status) are
    // style-immune (explicit colours), so both stay byte-stable.
    bridge
        .context
        .all_styles_mut(ui::theme::apply_app_style);
    let full_output = bridge.context.run_ui(raw_input, |ui| {
        panel_response = ui::workspace::build_workspace(ui, panel_state, export, palette);
        // After both panels have been shown inside the root ui, the remaining
        // space is the central viewport.
        central_rect_points = ui.available_rect_before_wrap();

        // #13 Step 3: the ViewCube right-click context menu. Drawn as a floating
        // egui Area at the press position when open. egui owns its hit-testing, so
        // its buttons swallow the click (no leak to the snap path). A click on an
        // item runs the action and closes the menu; a click anywhere OUTSIDE the
        // menu (detected via the area response) closes it without acting.
        if let Some(menu_pos_px) = *cube_context_menu_at {
            // `cube_context_menu_at` is stored in PHYSICAL pixels (the winit cursor
            // space); egui positions in points, so divide by pixels_per_point.
            let menu_pos = egui::pos2(
                menu_pos_px.x / pixels_per_point,
                menu_pos_px.y / pixels_per_point,
            );
            let context = ui.ctx().clone();
            let area = egui::Area::new(egui::Id::new("view_cube_context_menu"))
                .order(egui::Order::Foreground)
                .fixed_pos(menu_pos)
                .show(&context, |ui| {
                    egui::Frame::menu(ui.style()).show(ui, |ui| {
                        ui.set_min_width(180.0);
                        if ui.button("Home").clicked() {
                            cube_menu_request = Some(ViewCubeMenuRequest::Home);
                        }
                        if ui.button("Fit").clicked() {
                            cube_menu_request = Some(ViewCubeMenuRequest::Fit);
                        }
                        // Ortho ↔ Perspective: toggle the SAME field the side panel
                        // binds, so the menu and the panel stay in sync.
                        let projection_label = match panel_state.projection_mode {
                            ProjectionMode::Perspective => "Orthographic",
                            ProjectionMode::Orthographic => "Perspective",
                        };
                        if ui.button(projection_label).clicked() {
                            panel_state.projection_mode = match panel_state.projection_mode {
                                ProjectionMode::Perspective => ProjectionMode::Orthographic,
                                ProjectionMode::Orthographic => ProjectionMode::Perspective,
                            };
                            *cube_context_menu_at = None;
                        }
                        ui.separator();
                        if ui.button("Set current as home").clicked() {
                            cube_menu_request = Some(ViewCubeMenuRequest::SetHome);
                        }
                    });
                });
            // Close on selection (an item set a request or toggled projection).
            if cube_menu_request.is_some() {
                *cube_context_menu_at = None;
            }
            // Click-away: only a PRIMARY (left) click that lands OUTSIDE the menu's
            // rect closes it. #13 Step 6.5: the previous `any_click()` also fired on
            // the SECONDARY (right) click that OPENS the menu — and on the open frame
            // egui's `interact_pos` is the cursor at the menu's very corner, which the
            // freshly-laid-out rect didn't yet count as "inside", so the menu closed
            // the same frame it appeared (the flicker). Restricting the close to a
            // primary click leaves the opening right-click alone, so the menu stays up
            // until the user picks an item or left-clicks elsewhere.
            let pointer = &context.input(|i| i.pointer.clone());
            if pointer.primary_clicked() {
                let clicked_in_menu = pointer
                    .interact_pos()
                    .map(|p| area.response.rect.contains(p))
                    .unwrap_or(false);
                if !clicked_in_menu {
                    *cube_context_menu_at = None;
                }
            }
        }

        // The general VIEWPORT context menu (docs/design/tool-modes-and-navigation.md): a
        // mode-dispatched right-click menu. Delete (a warn-red ✕, the one destructive glyph) acts
        // on the sketch selection in sketch mode and the active node in normal mode. An egui Area,
        // so egui owns its hit-testing and its click never leaks to the viewport.
        if let Some(menu_pos_px) = *viewport_menu_at {
            let menu_pos = egui::pos2(
                menu_pos_px.x / pixels_per_point,
                menu_pos_px.y / pixels_per_point,
            );
            let context = ui.ctx().clone();
            let in_sketch = panel_state.sketch_mode.is_some();
            // Delete is enabled only when there is something to remove: a non-empty sketch
            // selection, or (normal mode) an active node.
            let delete_enabled = if in_sketch {
                !panel_state.sketch_selection.is_empty()
            } else {
                panel_state.scene.active_node().is_some()
            };
            let mut close = false;
            let area = egui::Area::new(egui::Id::new("viewport_context_menu"))
                .order(egui::Order::Foreground)
                .fixed_pos(menu_pos)
                .show(&context, |ui| {
                    egui::Frame::menu(ui.style()).show(ui, |ui| {
                        ui.set_min_width(160.0);
                        // A menu row: the warn ✕ is drawn by the egui icon painter
                        // (`Icon::Cancel`), NEVER a font character — a unicode glyph renders as
                        // tofu in egui's font. Manual allocate-and-paint so the icon is real
                        // graphics, not text.
                        let (rect, response) = ui.allocate_exact_size(
                            egui::vec2(ui.available_width().max(150.0), 22.0),
                            egui::Sense::click(),
                        );
                        let color = if delete_enabled {
                            ui::theme::WARN
                        } else {
                            ui.visuals().weak_text_color()
                        };
                        if delete_enabled && response.hovered() {
                            ui.painter().rect_filled(
                                rect,
                                3.0,
                                ui.visuals().widgets.hovered.bg_fill,
                            );
                        }
                        let icon = 13.0;
                        let icon_rect = egui::Rect::from_center_size(
                            egui::pos2(rect.left() + 6.0 + icon / 2.0, rect.center().y),
                            egui::vec2(icon, icon),
                        );
                        ui::icons::Icon::Cancel.draw(ui.painter(), icon_rect, color);
                        ui.painter().text(
                            egui::pos2(icon_rect.right() + 8.0, rect.center().y),
                            egui::Align2::LEFT_CENTER,
                            "Delete",
                            egui::TextStyle::Button.resolve(ui.style()),
                            color,
                        );
                        if delete_enabled && response.clicked() {
                            if in_sketch {
                                // The shell owns the selection + the sketch commit path.
                                panel_response.delete_sketch_selection = true;
                            } else if let Some(id) =
                                panel_state.scene.active_node().map(|node| node.id)
                            {
                                panel_response.frame_after_apply = true;
                                panel_response
                                    .intents
                                    .push(document::intent::Intent::RemoveNode { target: id });
                            }
                            close = true;
                        }
                    });
                });
            if close {
                *viewport_menu_at = None;
            }
            // Click-away: a PRIMARY click outside the menu closes it (mirrors the cube menu — the
            // opening right-click is left alone so the menu does not flicker shut the same frame).
            let pointer = context.input(|i| i.pointer.clone());
            if pointer.primary_clicked() {
                let clicked_in_menu = pointer
                    .interact_pos()
                    .map(|p| area.response.rect.contains(p))
                    .unwrap_or(false);
                if !clicked_in_menu {
                    *viewport_menu_at = None;
                }
            }
        }

        // Signal (issue #88): the floating DISPLAY stack, anchored to the top-right of the
        // central viewport (the cube + rail slide to its left). Drawn on the SAME single
        // frame the side panel is (an absolute-rect immediate-mode child, not an Area) so it
        // renders on the headless `shot` capture. It mutates `panel_state` (fold / section
        // toggles, projection, layer band) and appends any `SetGridMasters` to the response.
        // Capture the fold state as DRAWN this frame (a fold/expand click takes effect next
        // frame), so the cube slide matches the panel width actually painted.
        let stack_folded_drawn = panel_state.stack.folded;
        let stack_rect_points = build_signal_stack(
            ui,
            panel_state,
            central_rect_points,
            grid_z,
            measured_diameter,
            &mut panel_response,
        );
        chrome_rects_points.push(stack_rect_points);

        // Owner ruling 2026-07-21: the armed-tool `Add <shape>` dialog, pinned top-left of the
        // central viewport while a primitive is armed. Same absolute-child idiom as the stack,
        // so it renders on the shot capture and counts as chrome (its clicks don't orbit).
        if let Some(kind) = armed_shape {
            let dialog_rect =
                build_add_shape_dialog(ui, panel_state, central_rect_points, kind);
            chrome_rects_points.push(dialog_rect);
        }

        // Signal (ADR 0018 Decision 8): the cube's on-screen anchors in egui points
        // (shared by the readout, icon rail, and status line so they track the cube as
        // the side panel resizes AND slide left of the display stack). The cube's right
        // inset from the central edge is the stack's current width (issue #88); `cube_fits`
        // mirrors `view_cube_corner`'s minimum-size rule (viewport ≥ inset + cube wide, ≥
        // margin + cube tall) — below it the cube isn't drawn, so the rail hides too.
        let cube_margin = display::renderer::VIEW_CUBE_VIEWPORT_MARGIN as f32 / pixels_per_point;
        let cube_size = VIEW_CUBE_VIEWPORT_PIXELS as f32 / pixels_per_point;
        let cube_right_inset = cube_right_inset_points(stack_folded_drawn);
        let cube_left = central_rect_points.right() - cube_right_inset - cube_size;
        let cube_bottom = central_rect_points.top() + cube_margin + cube_size;
        let cube_right_inset_px = (cube_right_inset * pixels_per_point).round() as u32;
        view_cube_right_inset_px = cube_right_inset_px;
        let cube_fits = central_rect_points.width() * pixels_per_point
            >= cube_right_inset_px as f32 + VIEW_CUBE_VIEWPORT_PIXELS as f32
            && central_rect_points.height() * pixels_per_point
                >= (display::renderer::VIEW_CUBE_VIEWPORT_MARGIN + VIEW_CUBE_VIEWPORT_PIXELS) as f32;

        // Signal: the icon rail directly under the cube (Home / Fit / viewport-mode
        // cycle). Home/Fit reuse the shell's `ChromeClickAction`; a mode-cycle click
        // steps `view_mode` in place (pure display state — the shell re-derives overlays
        // at its existing mode-change seam). Hidden when the cube can't fit or is toggled
        // off. Rendered here (inside `run_egui_frame`) so it draws on BOTH the windowed
        // surface and the `shot` capture.
        if cube_fits {
            chrome_rects_points.push(ui::chrome::rail_rect(cube_left, cube_bottom, cube_size));
            if let Some(click) = ui::chrome::icon_rail(
                ui,
                cube_left,
                cube_bottom,
                cube_size,
                panel_state.view_mode,
            ) {
                match click {
                    ui::chrome::RailClick::Home => rail_action = Some(ChromeClickAction::Home),
                    ui::chrome::RailClick::Fit => rail_action = Some(ChromeClickAction::Fit),
                    ui::chrome::RailClick::CycleMode => {
                        panel_state.view_mode = panel_state.view_mode.next();
                    }
                }
            }
        }

        // Signal: the persistent bottom-left status line (mode · selection · dims ·
        // density). Draws on BOTH paths. Selection name + scene dims + density are read
        // from the panel's scene each frame.
        {
            let density = panel_state.scene.voxels_per_block;
            let dims = panel_state.scene.placed_region_dimensions(density);
            let selection = panel_state
                .scene
                .active_node()
                .map(|node| node.name.as_str())
                .filter(|name| !name.is_empty());
            ui::chrome::status_line(
                ui,
                central_rect_points,
                panel_state.view_mode,
                selection,
                dims,
                density,
            );
        }

        // ADR 0028: while a sketch is being edited, the immersive accent viewport border + the
        // floating CANCEL | FINISH SKETCH control (the two mode signals the owner review kept,
        // besides the rail swap). Draws on BOTH paths so the mode chrome is verifiable by the
        // headless `shot` capture. A click routes onto the response as `exit_sketch`; the
        // button rects register as chrome so they never leak to the camera orbit.
        if panel_state.sketch_mode.is_some() {
            if let Some(exit) = ui::chrome::sketch_exit_control(
                ui,
                central_rect_points,
                &mut chrome_rects_points,
            ) {
                panel_response.exit_sketch = Some(exit);
            }
            // ADR 0030: the committed segment lines, drawn FIRST so the vertex dots sit on top.
            // Not chrome — a segment press is handled by the shell's hit-test, and these are a
            // passive under-layer.
            ui::chrome::sketch_segment_lines(ui, sketch_segment_lines);
            // ADR 0028 (#94): the draggable profile-vertex handles, drawn at the shell's
            // projected screen positions and registered as chrome (a handle press drags the
            // vertex, never orbits).
            ui::chrome::sketch_vertex_handles(ui, sketch_handles, &mut chrome_rects_points);
            // ADR 0028 (#95): the add-point insert preview — a diamond on the hovered edge. NOT
            // chrome (a passive marker), so a click passes through to the stationary-release insert.
            if let Some(center) = sketch_insert_preview {
                ui::chrome::sketch_insert_marker(ui, center);
            }
        }

        // Signal (#86): the faint zone-name readout, centred under the cube but BELOW the
        // icon rail (so the two never overlap). Anchored off the post-panel central rect
        // so it tracks the cube as the side panel resizes. Non-interactive (a pure label);
        // windowed-only (the `shot` path passes `None`).
        if let Some(label) = view_cube_zone_readout {
            let readout_top = ui::chrome::rail_top(cube_bottom) + ui::chrome::rail_height() + 4.0;
            let context = ui.ctx().clone();
            egui::Area::new(egui::Id::new("view_cube_zone_readout"))
                .order(egui::Order::Foreground)
                .interactable(false)
                .fixed_pos(egui::pos2(cube_left, readout_top))
                .show(&context, |ui| {
                    ui.allocate_ui_with_layout(
                        egui::vec2(cube_size, 0.0),
                        egui::Layout::top_down(egui::Align::Center),
                        |ui| {
                            ui.label(
                                egui::RichText::new(label)
                                    .monospace()
                                    .size(10.0)
                                    // Signal "text — faint" readout.
                                    .color(ui::theme::TEXT_FAINT),
                            );
                        },
                    );
                });
        }
    });

    // Convert the central rect from egui points to physical pixels, then clamp it
    // inside the target so the viewport/scissor below are always valid.
    let viewport_px = {
        let to_px = |value: f32| (value * pixels_per_point).round();
        let left = to_px(central_rect_points.min.x).max(0.0) as u32;
        let top = to_px(central_rect_points.min.y).max(0.0) as u32;
        let right = to_px(central_rect_points.max.x).max(0.0) as u32;
        let bottom = to_px(central_rect_points.max.y).max(0.0) as u32;
        let x = left.min(size_in_pixels[0]);
        let y = top.min(size_in_pixels[1]);
        // Always leave at least a 1×1 viewport so set_viewport never gets 0 dims.
        let width = right.min(size_in_pixels[0]).saturating_sub(x).max(1);
        let height = bottom.min(size_in_pixels[1]).saturating_sub(y).max(1);
        [x, y, width, height]
    };

    // The chrome hit-rects, points → physical pixels (same conversion as `viewport_px`).
    let chrome_rects_px: Vec<[f32; 4]> = chrome_rects_points
        .iter()
        .map(|rect| {
            [
                rect.min.x * pixels_per_point,
                rect.min.y * pixels_per_point,
                rect.width() * pixels_per_point,
                rect.height() * pixels_per_point,
            ]
        })
        .collect();

    for (texture_id, image_delta) in &full_output.textures_delta.set {
        bridge
            .renderer
            .update_texture(device, queue, *texture_id, image_delta);
    }

    let paint_jobs = bridge
        .context
        .tessellate(full_output.shapes, pixels_per_point);

    PreparedEguiFrame {
        paint_jobs,
        screen_descriptor: egui_wgpu::ScreenDescriptor {
            size_in_pixels,
            pixels_per_point,
        },
        textures_to_free: full_output.textures_delta.free,
        platform_output: full_output.platform_output,
        panel_response,
        viewport_px,
        cube_menu_request,
        rail_action,
        view_cube_right_inset_px,
        chrome_rects_px,
    }
}
