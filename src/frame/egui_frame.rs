//! The egui half of the per-frame pipeline (ADR 0031 — split out of the former monolithic
//! `lib.rs`): build the panel + tessellate the UI ([`run_egui_frame`]), the persistent
//! [`EguiPaintBridge`], the [`PreparedEguiFrame`] it produces, and the view-cube context-menu
//! request. The GPU pass recording is the sibling [`render`](super::render).

use crate::*;

/// Width (egui points) of the rail's orbit-type menu. The menu opens LEFTWARD off its button, so
/// this is also how far left of the rail it reaches.
const MENU_WIDTH: f32 = 160.0;

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
    /// The orbit TYPE picked from the rail's type menu this frame, if any. The pick has already
    /// been written to `panel_state.default_orbit_type` — this reports it so the shell can close
    /// the Free Orbit seam and start the animated re-level when the pick is Constrained
    /// (`docs/design/tool-modes-and-navigation.md`). `None` on the headless `shot` path.
    pub orbit_type_picked: Option<OrbitType>,
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

/// One row of a viewport context menu: an [`Icon`](ui::icons::Icon) painter, the label, then the
/// row's keyboard shortcut flushed to the right edge.
///
/// Manual allocate-and-paint rather than an [`egui::Button`], because the mark has to be real
/// graphics: a unicode character renders as tofu in egui's font, so every glyph in this app is
/// drawn by an icon painter. That is also why the rows share one helper — a row that reached for
/// a text glyph because the icon path was inconvenient is exactly the regression to prevent.
///
/// The row names its [`ShortcutCommand`](ui::shortcuts::ShortcutCommand) and the binding is looked
/// up in the [`Shortcuts`](ui::shortcuts::Shortcuts) settings — there is deliberately no way to
/// pass the text, so a hardcoded "Esc" is a type error rather than a review note. An unbound
/// command leaves the column empty rather than inventing a binding, which is what lets the menu
/// double as the honest list of what IS bound.
///
/// `enabled == false` greys the row and swallows the click, which is how Delete shows that there
/// is nothing picked to delete without vanishing and making the menu twitch between opens.
fn context_menu_row(
    ui: &mut egui::Ui,
    shortcuts: &ui::shortcuts::Shortcuts,
    icon: ui::icons::Icon,
    label: &str,
    command: ui::shortcuts::ShortcutCommand,
    enabled: bool,
    tint: Option<egui::Color32>,
) -> bool {
    const ROW_HEIGHT: f32 = 22.0;
    const ICON_SIZE: f32 = 13.0;
    const ICON_INSET: f32 = 6.0;
    const LABEL_GAP: f32 = 8.0;
    const SHORTCUT_INSET: f32 = 8.0;
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width().max(150.0), ROW_HEIGHT),
        egui::Sense::click(),
    );
    let color = match (enabled, tint) {
        (false, _) => ui.visuals().weak_text_color(),
        (true, Some(tint)) => tint,
        (true, None) => ui.visuals().text_color(),
    };
    if enabled && response.hovered() {
        ui.painter()
            .rect_filled(rect, 3.0, ui.visuals().widgets.hovered.bg_fill);
    }
    let icon_rect = egui::Rect::from_center_size(
        egui::pos2(rect.left() + ICON_INSET + ICON_SIZE / 2.0, rect.center().y),
        egui::vec2(ICON_SIZE, ICON_SIZE),
    );
    icon.draw(ui.painter(), icon_rect, color);
    let font = egui::TextStyle::Button.resolve(ui.style());
    ui.painter().text(
        egui::pos2(icon_rect.right() + LABEL_GAP, rect.center().y),
        egui::Align2::LEFT_CENTER,
        label,
        font.clone(),
        color,
    );
    if let Some(shortcut) = shortcuts.display(ui.ctx(), command) {
        // Always the weak tone, even on the warn-tinted row: the binding is a reminder, not a
        // second thing to act on, and matching the label's colour would make it read as one.
        ui.painter().text(
            egui::pos2(rect.right() - SHORTCUT_INSET, rect.center().y),
            egui::Align2::RIGHT_CENTER,
            shortcut,
            font,
            ui.visuals().weak_text_color(),
        );
    }
    enabled && response.clicked()
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
    // Whether the icon rail's orbit-TYPE menu is open. Unlike the two context menus this carries
    // no position: the menu anchors to the rail button it belongs to, whose rect is computed
    // inside this pass. The headless `shot` path passes `&mut false` (no menu).
    orbit_type_menu_open: &mut bool,
    // Signal (#86): the hovered view-cube zone's name (e.g. `TOP·FRONT`), drawn as a
    // faint readout line under the cube. `None` when nothing is hovered — and always
    // `None` on the headless `shot` path, so the goldens stay pure cube geometry.
    view_cube_zone_readout: Option<&str>,
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
    // #99: the drawing tools' dashed preview polyline (egui points) — the polyline rubber
    // line to the snapped cursor, or the rectangle ghost's five closing corners. Empty when
    // no drawing gesture is live, and always empty on the headless `shot` path.
    sketch_draw_preview: &[egui::Pos2],
    // Sketch-selection slice 3: the marquee rubber band (egui points) and whether it is a
    // window (`true`, solid + stronger fill) or crossing (`false`, dashed + lighter) box, or
    // `None` when no marquee is live. Always `None` on the headless `shot` path.
    sketch_marquee: Option<(egui::Rect, bool)>,
    // ADR 0032: the orbit center's projected position (egui points) plus whether a placement is
    // armed, or `None` when the pivot should not be drawn — it shows while a placement rides the
    // cursor and while Shift+MMB is turning about it, and is hidden otherwise. Registers no chrome
    // rect: the pivot is moved by the context menu, never by dragging it. Always `None` on the
    // headless `shot` path.
    orbit_center: Option<(egui::Pos2, bool)>,
    // ADR 0032: whether the explicit orbit mode's targeting reticle draws this frame. It fills
    // the central viewport rect — computed inside this pass, so no position travels with the
    // flag — and the shell clears it while a TURN is in flight, so the model comes round against
    // an unobstructed view. Always `false` on the headless `shot` path.
    orbit_reticle: bool,
) -> PreparedEguiFrame {
    let mut panel_response = PanelResponse::default();
    let mut cube_menu_request: Option<ViewCubeMenuRequest> = None;
    // Signal (ADR 0018 Decision 8): the icon rail's Home/Fit click, pre-mapped onto the
    // shell's `ChromeClickAction`; a mode-cycle click mutates `panel_state.view_mode` in
    // place inside the closure (never surfaced), like the ortho toggle.
    let mut rail_action: Option<ChromeClickAction> = None;
    // The rail's type menu pick, applied in place to `panel_state.default_orbit_type` and
    // reported so the shell can re-level.
    let mut orbit_type_picked: Option<OrbitType> = None;
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
    bridge.context.all_styles_mut(ui::theme::apply_app_style);
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
            // Delete is enabled only when there is something to remove: an entity picked inside
            // the OPEN sketch, or (normal mode) a picked node. Asking per-sketch, not "is the
            // selection non-empty", so a node picked outside the mode never arms it (ADR 0032).
            let delete_enabled = match panel_state.sketch_mode {
                Some(sketch) => panel_state.selection.holds_sketch_entities(sketch),
                None => panel_state.selected_node().is_some(),
            };
            let mut close = false;
            // Cloned out of the state the rows also mutate — the bindings are read-only here.
            let shortcuts = panel_state.shortcuts.clone();
            let area = egui::Area::new(egui::Id::new("viewport_context_menu"))
                .order(egui::Order::Foreground)
                .fixed_pos(menu_pos)
                .show(&context, |ui| {
                    egui::Frame::menu(ui.style()).show(ui, |ui| {
                        ui.set_min_width(160.0);
                        // A running MODAL COMMAND replaces the whole menu with OK / Cancel. This
                        // is the general viewport variant, not an orbit-mode special case: while
                        // a command is up there is no third choice, because a menu offering
                        // unrelated verbs mid-command would be offering to start a second one.
                        // The keyboard shortcuts are the universal pair — Return accepts, Escape
                        // cancels — handled by the shell, which also decides what each MEANS. For
                        // the explicit orbit mode both simply end it: navigating IS the result and
                        // it has already happened, so there is nothing left to discard.
                        //
                        // The unrelated verbs a running command still wants to reach are planned
                        // to live in a Fusion/Maya-style PIE MENU above this list, not in it —
                        // which is what keeps this list exactly two rows.
                        if panel_state.orbit_mode.is_on() {
                            if context_menu_row(
                                ui,
                                &shortcuts,
                                ui::icons::Icon::Commit,
                                "OK",
                                ui::shortcuts::ShortcutCommand::AcceptCommand,
                                true,
                                None,
                            ) {
                                panel_response.mode_command = Some(ui::panel::ModeCommand::Accept);
                                close = true;
                            }
                            if context_menu_row(
                                ui,
                                &shortcuts,
                                ui::icons::Icon::Cancel,
                                "Cancel",
                                ui::shortcuts::ShortcutCommand::CancelCommand,
                                true,
                                None,
                            ) {
                                panel_response.mode_command = Some(ui::panel::ModeCommand::Cancel);
                                close = true;
                            }
                            return;
                        }
                        // Delete is the one row that carries a colour of its own: removal is the
                        // only warn-valent act in the menu.
                        if context_menu_row(
                            ui,
                            &shortcuts,
                            ui::icons::Icon::Cancel,
                            "Delete",
                            ui::shortcuts::ShortcutCommand::DeleteSelection,
                            delete_enabled,
                            Some(ui::theme::WARN),
                        ) {
                            // One flag, not two branches: whether "delete" means sketch entities
                            // or a node is the shell's call, and it has to be, because the
                            // keyboard binding for the same command arrives with no menu at all.
                            panel_response.delete_selection = true;
                            close = true;
                        }

                        // The ORBIT CENTER rows (docs/design/tool-modes-and-navigation.md).
                        // These two are the ONLY things that move the Shift+MMB pivot — pan,
                        // zoom, the view cube and the explicit orbit mode all work on
                        // `camera.target` instead, which is what lets a pan slide the view
                        // while the feature you are inspecting stays the one you turn around.
                        // "Place" ARMS a placement rather than placing on the spot: the center
                        // then follows the cursor as its own gizmo until a click commits it,
                        // so you watch it land instead of finding out after the menu closed.
                        ui.separator();
                        // Two marks authored for these rows, sharing a silhouette: place IS the
                        // pivot gizmo, so the row shows what will appear on screen, and reset is
                        // the same mark with its ring opened into a revert arrow.
                        if context_menu_row(
                            ui,
                            &shortcuts,
                            ui::icons::Icon::OrbitCenterPlace,
                            "Place orbit center",
                            ui::shortcuts::ShortcutCommand::PlaceOrbitCenter,
                            true,
                            None,
                        ) {
                            panel_response.orbit_center_request =
                                Some(ui::panel::OrbitCenterRequest::Place);
                            close = true;
                        }
                        // Always enabled: resetting an unplaced center is a harmless no-op,
                        // and greying it out would only make the menu twitch between opens.
                        if context_menu_row(
                            ui,
                            &shortcuts,
                            ui::icons::Icon::OrbitCenterReset,
                            "Reset orbit center",
                            ui::shortcuts::ShortcutCommand::ResetOrbitCenter,
                            true,
                            None,
                        ) {
                            panel_response.orbit_center_request =
                                Some(ui::panel::OrbitCenterRequest::Reset);
                            close = true;
                        }

                        // The explicit ORBIT MODE, entered by NAMING a type
                        // (docs/design/tool-modes-and-navigation.md, the entry-path table). This
                        // is the one entry that names one, and naming here does NOT write the
                        // default: invoking a tool has never meant "make this the default", and
                        // the override lives exactly as long as the mode does.
                        //
                        // Only Constrained is offered. Free Orbit lives in the rail's dropdown
                        // and nowhere else — it is the type you SET, not the one you reach for on
                        // a particular object.
                        //
                        // Entry only. Leaving is the OK / Cancel variant above, which is where
                        // every modal command ends — a per-command "leave" row would be a second
                        // exit for one command and no exit for the rest.
                        ui.separator();
                        if context_menu_row(
                            ui,
                            &shortcuts,
                            ui::icons::Icon::OrbitConstrained,
                            "Constrained Orbit",
                            ui::shortcuts::ShortcutCommand::EnterConstrainedOrbit,
                            true,
                            None,
                        ) {
                            panel_state.orbit_mode =
                                ui::panel::OrbitMode::Named(OrbitType::Constrained);
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
        // The armed kind is read off the armed tool itself — the one reader path the rail's
        // shape cells and this dialog share.
        if let Some(kind) = panel_state.armed_shape() {
            let dialog_rect = build_add_shape_dialog(ui, panel_state, central_rect_points, kind);
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
                >= (display::renderer::VIEW_CUBE_VIEWPORT_MARGIN + VIEW_CUBE_VIEWPORT_PIXELS)
                    as f32;

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
                panel_state.default_orbit_type,
                panel_state.orbit_mode,
            ) {
                match click {
                    ui::chrome::RailClick::Home => rail_action = Some(ChromeClickAction::Home),
                    ui::chrome::RailClick::Fit => rail_action = Some(ChromeClickAction::Fit),
                    ui::chrome::RailClick::CycleMode => {
                        panel_state.view_mode = panel_state.view_mode.next();
                    }
                    // The FACE toggles the explicit orbit mode, entering it as the DEFAULT type —
                    // a split button's face starts what it shows, and it shows the default. It
                    // never names a type, so re-entering from here drops any override the last
                    // session was carrying.
                    ui::chrome::RailClick::OrbitType => {
                        panel_state.orbit_mode = if panel_state.orbit_mode.is_on() {
                            ui::panel::OrbitMode::Off
                        } else {
                            ui::panel::OrbitMode::UsingDefault
                        };
                    }
                    ui::chrome::RailClick::OrbitTypeMenu => {
                        *orbit_type_menu_open = !*orbit_type_menu_open;
                    }
                }
            }

            // The orbit-TYPE menu, anchored to its own rail button and opening to the LEFT (the
            // rail sits against the viewport's right edge). This is the one control that writes
            // the DEFAULT orbit type; every other entry into an orbit either uses the default or
            // overrides it without changing it.
            if *orbit_type_menu_open {
                let button = ui::chrome::orbit_type_button_rect(cube_left, cube_bottom, cube_size);
                let context = ui.ctx().clone();
                let area = egui::Area::new(egui::Id::new("orbit_type_menu"))
                    .order(egui::Order::Foreground)
                    .fixed_pos(egui::pos2(button.left() - MENU_WIDTH, button.top()))
                    .show(&context, |ui| {
                        egui::Frame::menu(ui.style()).show(ui, |ui| {
                            ui.set_min_width(MENU_WIDTH);
                            for (orbit_type, label) in [
                                (OrbitType::Constrained, "Constrained orbit"),
                                (OrbitType::Free, "Free orbit"),
                            ] {
                                let selected = panel_state.default_orbit_type == orbit_type;
                                if ui.selectable_label(selected, label).clicked() {
                                    panel_state.default_orbit_type = orbit_type;
                                    orbit_type_picked = Some(orbit_type);
                                }
                            }
                        });
                    });
                if orbit_type_picked.is_some() {
                    *orbit_type_menu_open = false;
                }
                // Click-away, the same rule the cube menu uses: a PRIMARY click outside the menu
                // closes it. The rail button's own click is consumed by `icon_rail` above (it
                // toggles), so a second click on the button closes rather than re-opening.
                let pointer = context.input(|input| input.pointer.clone());
                if pointer.primary_clicked() {
                    let inside = pointer
                        .interact_pos()
                        .map(|position| {
                            area.response.rect.contains(position) || button.contains(position)
                        })
                        .unwrap_or(false);
                    if !inside {
                        *orbit_type_menu_open = false;
                    }
                }
            }
        }

        // Signal: the persistent bottom-left status line (mode · dims · density).
        // Draws on BOTH paths; dims + density are read from the panel's scene each frame.
        {
            let density = panel_state.scene.voxels_per_block;
            let dims = panel_state.scene.placed_region_dimensions(density);
            ui::chrome::status_line(
                ui,
                central_rect_points,
                panel_state.view_mode,
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
            if let Some(exit) =
                ui::chrome::sketch_exit_control(ui, central_rect_points, &mut chrome_rects_points)
            {
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
            // #99: the drawing tools' dashed preview — the uncommitted polyline rubber line /
            // rectangle ghost. Passive like the insert marker.
            if !sketch_draw_preview.is_empty() {
                ui::chrome::sketch_draw_preview(ui, sketch_draw_preview);
            }
            // Slice 3: the marquee rubber band — solid window / dashed crossing. Passive.
            if let Some((rect, window)) = sketch_marquee {
                ui::chrome::sketch_marquee_band(ui, rect, window);
            }
        }

        // ADR 0032: the orbit center, on its own foreground layer so it reads over the scene in
        // every mode — sketch or not.
        if let Some((center, placing)) = orbit_center {
            ui::gizmos::orbit_center_overlay(ui, center, placing);
        }

        // ADR 0032: the explicit orbit mode's targeting reticle, filling the central viewport.
        // Drawn whenever the mode runs and the button is up — it is what says the left button now
        // turns and re-centres, so hiding it between gestures would leave the flipped verb
        // invisible for most of the mode.
        if orbit_reticle {
            ui::gizmos::orbit_reticle_overlay(ui, central_rect_points);
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
        orbit_type_picked,
        view_cube_right_inset_px,
        chrome_rects_px,
    }
}
