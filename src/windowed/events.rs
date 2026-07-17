//! The winit event pump: `ApplicationHandler for App` — lazy window/GPU creation on `resumed`,
//! then the window-event router that drives orbit/pan/zoom, the ViewCube click/hover, the
//! deferred-close data-loss guard, and the per-frame redraw. Split out of `windowed/mod.rs`
//! (ADR 0016).

use super::*;

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            self.state = Some(WindowedState::new(event_loop));
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        // Let egui consume the event first; if it did, don't also use it to
        // drive the camera (so dragging on the panel doesn't orbit the scene).
        let response = state
            .egui_winit_state
            .on_window_event(&state.window, &event);
        let egui_consumed = response.consumed;

        match event {
            WindowEvent::CloseRequested => {
                if state.export_outstanding && !state.close_requested_while_exporting {
                    // Data-loss guard: a `.vox` export is in flight on the detached worker.
                    // Exiting now would kill it mid-build/mid-write, so DEFER the close — the
                    // `RedrawRequested` seam exits once the result lands (poll clears
                    // `export_outstanding`). Frames keep pumping meanwhile because
                    // `poll_vox_export_worker` requests a redraw while an export is in flight.
                    state.close_requested_while_exporting = true;
                    state.export_status = Some("Finishing export before closing…".to_string());
                    state.window.request_redraw();
                } else {
                    // No export outstanding, OR a SECOND close request while already deferring
                    // (the user insisting) — exit immediately. The atomic `.vox` write bounds
                    // the worst case to "no file", never a truncated one.
                    // M8: persist UI + camera + window size before exiting.
                    state.shutdown(event_loop);
                }
            }
            WindowEvent::Resized(new_size) => {
                state.resize(new_size.width, new_size.height);
            }
            WindowEvent::MouseInput {
                state: button_state,
                button: MouseButton::Left,
                ..
            } => {
                if button_state == ElementState::Pressed {
                    let position = state.last_cursor_position;
                    let in_cube = state.panel_state.show_view_cube
                        && position
                            .map(|(x, y)| state.position_in_view_cube(x, y))
                            .unwrap_or(false);
                    state.press_position = position;
                    state.press_in_view_cube = in_cube;
                    state.view_cube_drag_active = false;
                    // Pressing on the view cube does NOT start a scene-path orbit
                    // (`left_button_held`): a press on the cube either becomes a
                    // cube-drag orbit (handled in CursorMoved) or, if it stays put,
                    // snaps on release. So the scene orbit path is reserved for
                    // presses that started outside the cube, outside the Signal
                    // chrome (stack + rail — egui's heuristic no longer covers them
                    // now the stack doesn't allocate in the root ui), and not on egui.
                    let in_chrome = position
                        .map(|(x, y)| state.position_in_signal_chrome(x, y))
                        .unwrap_or(false);
                    state.left_button_held = !egui_consumed && !in_cube && !in_chrome;
                } else {
                    // Release: a press that started in the cube and DIDN'T become a
                    // drag (stayed within the threshold) selects the picked hot-zone
                    // element and snaps to it (prototype pointerup). A cube-drag has
                    // already orbited the camera, so it snaps nothing.
                    if state.press_in_view_cube && !state.view_cube_drag_active {
                        if let (Some((down_x, down_y)), Some((up_x, up_y))) =
                            (state.press_position, state.last_cursor_position)
                        {
                            let stationary = (up_x - down_x).abs()
                                < VIEW_CUBE_DRAG_THRESHOLD_PIXELS
                                && (up_y - down_y).abs() < VIEW_CUBE_DRAG_THRESHOLD_PIXELS;
                            if stationary && state.position_in_view_cube(up_x, up_y) {
                                // #13 Step 3: classify the stationary release into a
                                // chrome zone (rotate / roll / Home / Fit /
                                // cube body). The body region delegates to the same
                                // raycast picker as before, so a body click still
                                // resolves to an Element snap; the gutters/badges map
                                // to their actions. A drag-orbit never reaches here
                                // (it sets `view_cube_drag_active`, gated above), so
                                // orbiting still wins over a click.
                                let rect = state.cube_rect();
                                let zone = classify_cube_point(
                                    rect,
                                    up_x as f32,
                                    up_y as f32,
                                    || state.pick_view_cube_element(up_x, up_y),
                                );
                                // #13 Step 6.6: a rotate-arrow click only acts when the
                                // view is face-constrained (the arrows are hidden
                                // otherwise, so a stray gutter click is a no-op).
                                let rotate_disabled = matches!(
                                    zone,
                                    Some(CubeChromeZone::RotateArrow(_))
                                ) && !state.app_core.camera.is_face_constrained();
                                if let (Some(zone), false) = (zone, rotate_disabled) {
                                    let action = chrome_zone_left_click_action(
                                        zone,
                                        &state.app_core.camera,
                                    );
                                    state.run_chrome_action(action);
                                }
                            }
                        }
                    }
                    state.left_button_held = false;
                    state.last_cursor_position = None;
                    state.press_in_view_cube = false;
                    state.view_cube_drag_active = false;
                }
            }
            WindowEvent::MouseInput {
                state: button_state,
                button: MouseButton::Middle,
                ..
            } => {
                // Middle-drag pans the camera (explicit camera action). A press
                // that egui consumed (over the side panel / dock) or on the Signal
                // chrome doesn't grab the scene, mirroring the left-orbit gate. The
                // view cube doesn't take middle clicks, so no cube gating is needed here.
                let in_chrome = state
                    .last_cursor_position
                    .map(|(x, y)| state.position_in_signal_chrome(x, y))
                    .unwrap_or(false);
                state.middle_button_held =
                    button_state == ElementState::Pressed && !egui_consumed && !in_chrome;
            }
            WindowEvent::MouseInput {
                state: button_state,
                button: MouseButton::Right,
                ..
            } => {
                // #13 Step 3: a right-press inside the cube rect (not on egui) opens
                // the ViewCube context menu at the cursor. The menu itself is drawn
                // by egui in `run_egui_frame`; egui swallows its own clicks, so the
                // menu items never leak to the left-click snap path. Any other
                // right-press closes a menu that was open.
                if button_state == ElementState::Pressed && !egui_consumed {
                    let position = state.last_cursor_position;
                    let in_cube = state.panel_state.show_view_cube
                        && position
                            .map(|(x, y)| state.position_in_view_cube(x, y))
                            .unwrap_or(false);
                    state.context_menu_open_at = if in_cube {
                        position.map(|(x, y)| egui::pos2(x as f32, y as f32))
                    } else {
                        None
                    };
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let current = (position.x, position.y);

                // A press that started on the view cube becomes an orbit drag once
                // it moves past the threshold. This routes the SAME delta into
                // `orbit_by_drag` as a scene drag (no double-application: the cube
                // press never sets `left_button_held`, so only one path fires).
                if state.press_in_view_cube && !state.view_cube_drag_active {
                    if let Some((down_x, down_y)) = state.press_position {
                        let moved = (current.0 - down_x).abs() >= VIEW_CUBE_DRAG_THRESHOLD_PIXELS
                            || (current.1 - down_y).abs() >= VIEW_CUBE_DRAG_THRESHOLD_PIXELS;
                        if moved {
                            state.view_cube_drag_active = true;
                            // Promote to an orbit drag: cancel any in-progress snap.
                            state.snap_tween = None;
                        }
                    }
                }

                let orbiting = state.left_button_held || state.view_cube_drag_active;
                if orbiting {
                    if let Some((previous_x, previous_y)) = state.last_cursor_position {
                        let mut delta_x = (current.0 - previous_x) as f32;
                        let delta_y = (current.1 - previous_y) as f32;
                        // #13 Step 6.1: a cube drag GRABS the cube and turns it with
                        // the cursor, so the camera must orbit the OPPOSITE way round
                        // the model from a scene drag (dragging the cube's right edge
                        // leftward spins the model to show its right face). The scene
                        // drag keeps its existing sign; only the cube-drag path flips
                        // the horizontal component.
                        if state.view_cube_drag_active {
                            delta_x = -delta_x;
                        }
                        if delta_x != 0.0 || delta_y != 0.0 {
                            // A manual orbit cancels any in-progress snap tween.
                            state.snap_tween = None;
                            state.app_core.camera.orbit_by_drag(delta_x, delta_y);
                        }
                    }
                }

                // Middle-drag pans the target in the view plane (independent of the
                // orbit path, so the cursor can never both orbit and pan in one
                // move). Like orbit, a manual pan cancels any in-progress snap tween.
                if state.middle_button_held {
                    if let Some((previous_x, previous_y)) = state.last_cursor_position {
                        let delta_x = (current.0 - previous_x) as f32;
                        let delta_y = (current.1 - previous_y) as f32;
                        if delta_x != 0.0 || delta_y != 0.0 {
                            state.snap_tween = None;
                            // The 3D viewport height (cached each frame) makes the
                            // pan cursor-locked: a pixel of drag == a pixel of scene.
                            let viewport_height_px = state.last_viewport_px[3] as f32;
                            state
                                .app_core
                                .camera
                                .pan_by_drag(delta_x, delta_y, viewport_height_px);
                        }
                    }
                }
                state.last_cursor_position = Some(current);

                // #13 Step 4: live hover highlight for the chrome arrows. This runs
                // on every move, so keep it cheap: the chrome zones are pure
                // screen-rect tests, and we DELIBERATELY pass a `None` body picker so
                // the expensive cube raycast never fires for hover — a body-region
                // hover resolves to `None` (the body doesn't highlight anyway). Hover
                // stays `None` while orbiting/dragging, when egui ate the move, when
                // the cube is hidden, or when the cursor is outside the cube rect, so
                // it never interferes with drag-orbit, the click dispatch, or the
                // scene input.
                state.hovered_cube_zone = if orbiting
                    || egui_consumed
                    || !state.panel_state.show_view_cube
                    || !state.position_in_view_cube(current.0, current.1)
                {
                    None
                } else {
                    match classify_cube_point(
                        state.cube_rect(),
                        current.0 as f32,
                        current.1 as f32,
                        || state.pick_view_cube_element(current.0, current.1),
                    ) {
                        // #13 Step 6.6: rotate arrows are a face-relative affordance —
                        // only offer them when the view is constrained to a face
                        // (Fusion behaviour). Off-face hovers over a rotate gutter
                        // don't light up.
                        Some(CubeChromeZone::RotateArrow(_))
                            if !state.app_core.camera.is_face_constrained() =>
                        {
                            None
                        }
                        // #13 Step 6.2: faces/edges/corners DO highlight on hover now
                        // (the body picker resolves the hovered element); arrows and
                        // badges highlight as before.
                        Some(zone) => Some(zone),
                        None => None,
                    }
                };
            }
            WindowEvent::MouseWheel { delta, .. } if !egui_consumed => {
                // Wheel over the Signal chrome (stack + rail) belongs to the chrome,
                // not the camera — mirroring the orbit/pan gates.
                let in_chrome = state
                    .last_cursor_position
                    .map(|(x, y)| state.position_in_signal_chrome(x, y))
                    .unwrap_or(false);
                if !in_chrome {
                    let scroll_lines = match delta {
                        MouseScrollDelta::LineDelta(_, vertical) => vertical,
                        MouseScrollDelta::PixelDelta(position) => position.y as f32,
                    };
                    state.app_core.camera.zoom_by_wheel(scroll_lines);
                }
            }
            WindowEvent::RedrawRequested => {
                // Finding #0 (data-loss guard): poll the export worker and honour a pending
                // deferred close BEFORE `render()`. `render()` early-returns before it can
                // poll anything when the surface isn't presentable (window minimized /
                // occluded), which would otherwise hang the deferred close FOREVER — the
                // export result would never be observed and the app would never exit. This
                // poll and the exit check need no presentable surface, so they run here.
                state.poll_vox_export_worker();
                if state.close_requested_while_exporting && !state.export_outstanding {
                    // The export we were waiting on landed successfully (a failure clears
                    // the deferral in the poll above), so honour the pending close.
                    state.shutdown(event_loop);
                } else {
                    state.render();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }

    /// Loop is exiting (e.g. OS-initiated): persist config as a safety net in
    /// case the exit didn't go through `CloseRequested` (M8).
    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
            state.save_config();
        }
    }
}
