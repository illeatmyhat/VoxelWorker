//! The winit event pump: `ApplicationHandler` for `App` — lazy window/GPU creation on `resumed`,
//! then the window-event router that drives orbit/pan/zoom, the `ViewCube` click/hover, the
//! deferred-close data-loss guard, and the per-frame redraw.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SketchPointerRoute {
    Select,
    StationaryEdit,
    LineClickOrArcDrag,
}

/// Classify the pointer grammar once. Midpoint Line deliberately shares the ordinary stationary
/// edit route and can never latch Line's tangent-arc drag state.
const fn sketch_pointer_route(tool: ui::panel::SketchTool) -> SketchPointerRoute {
    match tool {
        ui::panel::SketchTool::Select => SketchPointerRoute::Select,
        ui::panel::SketchTool::AddPoint
        | ui::panel::SketchTool::MidpointLine
        | ui::panel::SketchTool::ArcCenterEndpoints
        | ui::panel::SketchTool::ArcTangent
        | ui::panel::SketchTool::ThreePointArc
        | ui::panel::SketchTool::CircleCenterDiameter
        | ui::panel::SketchTool::Circle2Point
        | ui::panel::SketchTool::Circle3Point
        | ui::panel::SketchTool::Circle2Tangent
        | ui::panel::SketchTool::Circle3Tangent
        | ui::panel::SketchTool::Rectangle3Point
        | ui::panel::SketchTool::Rectangle
        | ui::panel::SketchTool::RectangleCenterCorner
        | ui::panel::SketchTool::PolygonInscribed
        | ui::panel::SketchTool::PolygonCircumscribed
        | ui::panel::SketchTool::PolygonEdge
        | ui::panel::SketchTool::SlotCenterToCenter
        | ui::panel::SketchTool::SlotOverall
        | ui::panel::SketchTool::SlotCenterPoint
        | ui::panel::SketchTool::SlotCenterPointArc
        | ui::panel::SketchTool::Slot3PointArc
        | ui::panel::SketchTool::Ellipse
        | ui::panel::SketchTool::Conic
        | ui::panel::SketchTool::FitPointSpline
        | ui::panel::SketchTool::ControlPointSpline
        | ui::panel::SketchTool::BreakCurve
        | ui::panel::SketchTool::Trim
        | ui::panel::SketchTool::Extend
        | ui::panel::SketchTool::Fillet
        | ui::panel::SketchTool::ChamferEqual
        | ui::panel::SketchTool::ChamferDistanceAngle
        | ui::panel::SketchTool::ChamferTwoDistance
        | ui::panel::SketchTool::Offset
        | ui::panel::SketchTool::MoveCopy
        | ui::panel::SketchTool::Scale
        | ui::panel::SketchTool::Mirror
        | ui::panel::SketchTool::RectangularPattern
        | ui::panel::SketchTool::CircularPattern
        | ui::panel::SketchTool::FillRegion
        | ui::panel::SketchTool::CarveRegion => SketchPointerRoute::StationaryEdit,
        ui::panel::SketchTool::Line => SketchPointerRoute::LineClickOrArcDrag,
    }
}

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
            // No keyboard arm here. Presses are read out of egui's own input after the pass
            // (`run_shortcut_commands`), so a focused text field eats its keys first and the
            // dispatch never names a `KeyCode` — see `ui::shortcuts`.
            WindowEvent::MouseInput {
                state: button_state,
                button: MouseButton::Left,
                ..
            } => {
                if button_state == ElementState::Pressed {
                    let position = state.pointer.at();
                    let in_cube = position
                        .map(|(x, y)| state.position_in_view_cube(x, y))
                        .unwrap_or(false);
                    state.pointer.press();
                    state.press_in_view_cube = in_cube;
                    state.view_cube_drag_active = false;
                    // A press on the view cube either becomes a cube-drag orbit (handled in
                    // CursorMoved) or, if it stays put, snaps on release. The cube is the one
                    // affordance where LEFT-drag still turns the camera; the scene left-drag
                    // orbit is gone (tool-modes-and-navigation.md — left's default verb is
                    // Select, orbit moved to Shift+MMB).
                    let in_chrome = position
                        .map(|(x, y)| state.position_in_signal_chrome(x, y))
                        .unwrap_or(false);
                    // This press begins placement when a tool
                    // is armed and it landed on the live viewport (not egui / cube /
                    // chrome). Only a stationary release drops the node — a drag no longer
                    // orbits, but the threshold still keeps a twitchy click from placing.
                    // An armed orbit-center placement takes the click: the center is already
                    // under the cursor, so committing is just dropping the restore point. Done
                    // before every other consumer so the same press cannot also select or place.
                    let committed_orbit_center = !egui_consumed
                        && !in_cube
                        && !in_chrome
                        && state.commit_orbit_center_placement();
                    // Orbit mode flips the left button's verb — a drag
                    // turns the camera about `camera.target` and a stationary click re-centers
                    // the view on the surface it hits. It outranks selection and the sketch
                    // paths, which is the whole point of a mode, but NOT an armed orbit-center
                    // placement: that is a transient overlay the user is mid-way through.
                    let in_orbit_mode = !committed_orbit_center
                        && !egui_consumed
                        && !in_cube
                        && !in_chrome
                        && state.panel_state.orbit_mode.is_on();
                    state.orbiting_in_orbit_mode = in_orbit_mode;
                    state.orbit_mode_recenter_press = in_orbit_mode;
                    if in_orbit_mode {
                        // The TYPE is latched at press for the same reason Shift+MMB latches it:
                        // the two types are two representations of one orientation, exact only
                        // as a single point, so a conversion must never land inside a trajectory.
                        state.active_orbit_type = state.panel_state.active_orbit_type();
                    }
                    state.armed_press = !committed_orbit_center
                        && !in_orbit_mode
                        && state.panel_state.armed_tool.is_some()
                        && !egui_consumed
                        && !in_cube
                        && !in_chrome;
                    // A plain viewport press arms a node-selection resolve — but only
                    // when every other left-click consumer declined. An armed tool keeps its click
                    // for the placement drop, and sketch mode keeps its three paths; selecting a
                    // node from inside a sketch would leave the mode's own selection behind.
                    state.viewport_select_press = !committed_orbit_center
                        && !in_orbit_mode
                        && !egui_consumed
                        && !in_cube
                        && !in_chrome
                        && state.panel_state.armed_tool.is_none()
                        && state.panel_state.sketch_mode.is_none();
                    // A sketch-mode press on the live viewport (not egui / cube). Select grabs a
                    // vertex handle; Add Point latches for a generic stationary-release edit;
                    // Line owns a typed click/tangent-arc press. The view stays freely rotatable
                    // throughout via Shift+MMB, which is gated on neither sketch mode nor the
                    // armed tool.
                    if state.panel_state.sketch_mode.is_some()
                        && !in_orbit_mode
                        && !egui_consumed
                        && !in_cube
                    {
                        if let Some((cursor_x, cursor_y)) = position {
                            // An armed constraint overrides the drawing
                            // tool for the duration of its gesture. It hit-tests the same
                            // entities Select does but answers a different question, and letting
                            // the two run together would draw geometry mid-assertion.
                            if state.panel_state.armed_constraint.is_some() {
                                state.sketch_constraint_press = true;
                            } else {
                                match sketch_pointer_route(state.panel_state.sketch_tool) {
                                    SketchPointerRoute::Select => {
                                        // A viewport Select press arms a selection resolve on the
                                        // stationary release (this arm only runs under `!egui_consumed`,
                                        // so a press on the context menu never arms it).
                                        state.sketch_select_press = true;
                                        // A new gesture starts owing nothing: the last drag's snap
                                        // circle belonged to the hand that has already let go, and
                                        // a drag still standing here never saw its release (a lost
                                        // focus can eat one). Put it back rather than overwrite it
                                        // — the field is the only handle on the pre-drag drawing,
                                        // and the preview has already written past it.
                                        state.cancel_the_vertex_drag();
                                        state.sketch_drag =
                                            state.begin_sketch_vertex_drag(cursor_x, cursor_y);
                                        // An EMPTY-SPACE press may become a marquee past the click
                                        // threshold; a press on a vertex or edge never does (it
                                        // clicks / drags that entity instead).
                                        state.sketch_marquee_anchor = (state.sketch_drag.is_none()
                                            && state
                                                .nearest_sketch_edge(cursor_x, cursor_y)
                                                .is_none())
                                        .then_some((cursor_x, cursor_y));
                                    }
                                    // Generic click tools latch their press here; a stationary
                                    // release dispatches the selected edit below.
                                    SketchPointerRoute::StationaryEdit => {
                                        state.sketch_edit_press = true;
                                    }
                                    // Line owns a typed click-or-tangent-arc press path.
                                    SketchPointerRoute::LineClickOrArcDrag => {
                                        state.begin_line_press(cursor_x, cursor_y);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // Release: a press that started in the cube and DIDN'T become a
                    // drag (stayed within the threshold) selects the picked hot-zone
                    // element and snaps to it. A cube-drag has
                    // already orbited the camera, so it snaps nothing.
                    if state.press_in_view_cube && !state.view_cube_drag_active {
                        if let Some((down_x, down_y, up_x, up_y)) = state.pointer.press_and_now() {
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
                                let zone =
                                    classify_cube_point(rect, up_x as f32, up_y as f32, || {
                                        state.pick_view_cube_element(up_x, up_y)
                                    });
                                // #13 Step 6.6: a rotate-arrow click only acts when the
                                // view is face-constrained (the arrows are hidden
                                // otherwise, so a stray gutter click is a no-op).
                                let rotate_disabled =
                                    matches!(zone, Some(CubeChromeZone::RotateArrow(_)))
                                        && !state.app_core.camera.is_face_constrained();
                                if let (Some(zone), false) = (zone, rotate_disabled) {
                                    // The cube is chart-native: close the Free Orbit seam before
                                    // reading angles off the camera, or the tween would start
                                    // from the stale chart and jump.
                                    state.settle_to_constrained();
                                    let action =
                                        chrome_zone_left_click_action(zone, &state.app_core.camera);
                                    state.run_chrome_action(action);
                                }
                            }
                        }
                    }
                    // A stationary armed release drops the
                    // pending node. A drag no longer orbits, but the threshold stays: it is
                    // what keeps a twitchy press from placing, and it becomes the
                    // click-vs-marquee discriminator once the marquee lands. The tool STAYS armed so several
                    // can be placed; the ghost keeps following. NoSurface/TooFar left
                    // `pending_placement` None, so a click there does nothing.
                    if state.armed_press && state.pending_placement.is_some() {
                        if let Some((down_x, down_y, up_x, up_y)) = state.pointer.press_and_now() {
                            let stationary = (up_x - down_x).abs()
                                < VIEW_CUBE_DRAG_THRESHOLD_PIXELS
                                && (up_y - down_y).abs() < VIEW_CUBE_DRAG_THRESHOLD_PIXELS;
                            if stationary {
                                if let Some(intent) = state.pending_placement.take() {
                                    state.viewport_intents.push(intent);
                                }
                            }
                        }
                    }
                    // A stationary release with a constraint armed offers
                    // the entity under the cursor to the slot that is waiting. Same
                    // click-vs-drag threshold every other sketch release uses, so a press that
                    // turned into a camera drag never picks. Runs BEFORE the press ends below,
                    // since the hit-test needs the pair the press and the release make.
                    if state.sketch_constraint_press {
                        if let Some((down_x, down_y, up_x, up_y)) = state.pointer.press_and_now() {
                            let stationary = (up_x - down_x).abs()
                                < VIEW_CUBE_DRAG_THRESHOLD_PIXELS
                                && (up_y - down_y).abs() < VIEW_CUBE_DRAG_THRESHOLD_PIXELS;
                            if stationary {
                                state.resolve_sketch_constraint_click(up_x, up_y);
                            }
                        }
                    }
                    // Line owns both click and non-stationary release: a drag may be a tangent
                    // arc, so it resolves before the generic stationary drawing-tool door.
                    if state.panel_state.sketch_tool == ui::panel::SketchTool::Line
                        && state.line_press_is_live()
                    {
                        if let Some((down_x, down_y, up_x, up_y)) = state.pointer.press_and_now() {
                            let stationary = (up_x - down_x).abs()
                                < VIEW_CUBE_DRAG_THRESHOLD_PIXELS
                                && (up_y - down_y).abs() < VIEW_CUBE_DRAG_THRESHOLD_PIXELS;
                            if state.line_arc_is_latched() {
                                state.sketch_line_arc_release(up_x, up_y);
                            } else if stationary {
                                state.sketch_line_click(up_x, up_y);
                            }
                        }
                        state.end_line_press();
                    }
                    // A stationary release with a sketch edit armed
                    // performs it (the same click-vs-drag threshold placement uses; a drag
                    // no longer orbits, but a twitchy press must still not edit).
                    // Runs BEFORE the press ends below, since the hit-tests need the pair the
                    // press and the release make. The tool stays armed.
                    if state.sketch_edit_press {
                        if let Some((down_x, down_y, up_x, up_y)) = state.pointer.press_and_now() {
                            let stationary = (up_x - down_x).abs()
                                < VIEW_CUBE_DRAG_THRESHOLD_PIXELS
                                && (up_y - down_y).abs() < VIEW_CUBE_DRAG_THRESHOLD_PIXELS;
                            if let (true, Some(target)) =
                                (stationary, state.panel_state.sketch_mode)
                            {
                                match state.panel_state.sketch_tool {
                                    ui::panel::SketchTool::AddPoint => {
                                        if let Some(producer) = state.sketch_insert_at(up_x, up_y) {
                                            state.commit_sketch_profile_edit(target, producer);
                                        }
                                    }
                                    // Endpoint, endpoint, then the through-point that
                                    // solves the bulge; commits internally.
                                    ui::panel::SketchTool::ThreePointArc => {
                                        state.sketch_arc_click(up_x, up_y);
                                    }
                                    ui::panel::SketchTool::CircleCenterDiameter => {
                                        state.sketch_circle_click(up_x, up_y);
                                    }
                                    ui::panel::SketchTool::Circle2Point => {
                                        state.sketch_point_circle_click(
                                            point_circle::PointCircleKind::TwoPoint,
                                            up_x,
                                            up_y,
                                        );
                                    }
                                    ui::panel::SketchTool::Circle3Point => {
                                        state.sketch_point_circle_click(
                                            point_circle::PointCircleKind::ThreePoint,
                                            up_x,
                                            up_y,
                                        );
                                    }
                                    ui::panel::SketchTool::Circle2Tangent
                                    | ui::panel::SketchTool::Circle3Tangent => {
                                        state.sketch_tangent_circle_click(up_x, up_y);
                                    }
                                    ui::panel::SketchTool::Rectangle3Point => {
                                        state.sketch_three_point_rectangle_click(up_x, up_y);
                                    }
                                    ui::panel::SketchTool::PolygonInscribed
                                    | ui::panel::SketchTool::PolygonCircumscribed
                                    | ui::panel::SketchTool::PolygonEdge => {
                                        state.sketch_polygon_click(up_x, up_y);
                                    }
                                    ui::panel::SketchTool::SlotCenterToCenter
                                    | ui::panel::SketchTool::SlotOverall
                                    | ui::panel::SketchTool::SlotCenterPoint
                                    | ui::panel::SketchTool::SlotCenterPointArc
                                    | ui::panel::SketchTool::Slot3PointArc => {
                                        state.sketch_slot_click(up_x, up_y);
                                    }
                                    ui::panel::SketchTool::Ellipse
                                    | ui::panel::SketchTool::Conic
                                    | ui::panel::SketchTool::FitPointSpline
                                    | ui::panel::SketchTool::ControlPointSpline => {
                                        state.sketch_higher_curve_click(up_x, up_y);
                                    }
                                    ui::panel::SketchTool::MidpointLine => {
                                        state.sketch_midpoint_line_click(up_x, up_y);
                                    }
                                    ui::panel::SketchTool::ArcCenterEndpoints => {
                                        state.sketch_center_arc_click(up_x, up_y);
                                    }
                                    ui::panel::SketchTool::ArcTangent => {
                                        state.sketch_tangent_arc_click(up_x, up_y);
                                    }
                                    ui::panel::SketchTool::BreakCurve => {
                                        state.sketch_break_click(up_x, up_y);
                                    }
                                    ui::panel::SketchTool::Trim => {
                                        state.sketch_trim_click(up_x, up_y);
                                    }
                                    ui::panel::SketchTool::Extend => {
                                        state.sketch_extend_click(up_x, up_y);
                                    }
                                    ui::panel::SketchTool::Fillet => {
                                        state.sketch_fillet_click(up_x, up_y);
                                    }
                                    ui::panel::SketchTool::ChamferEqual
                                    | ui::panel::SketchTool::ChamferDistanceAngle
                                    | ui::panel::SketchTool::ChamferTwoDistance => {
                                        state.sketch_chamfer_click(up_x, up_y);
                                    }
                                    ui::panel::SketchTool::Offset => {
                                        state.sketch_offset_click(up_x, up_y);
                                    }
                                    ui::panel::SketchTool::MoveCopy => {
                                        state.sketch_move_copy_click(up_x, up_y);
                                    }
                                    ui::panel::SketchTool::Scale => {
                                        state.sketch_scale_click(up_x, up_y);
                                    }
                                    ui::panel::SketchTool::Mirror => {
                                        state.sketch_mirror_click(up_x, up_y);
                                    }
                                    ui::panel::SketchTool::RectangularPattern => {
                                        state.sketch_rectangular_pattern_click(up_x, up_y);
                                    }
                                    ui::panel::SketchTool::CircularPattern => {
                                        state.sketch_circular_pattern_click(up_x, up_y);
                                    }
                                    ui::panel::SketchTool::FillRegion => {
                                        state.sketch_set_face_picked(up_x, up_y, true);
                                    }
                                    ui::panel::SketchTool::CarveRegion => {
                                        state.sketch_set_face_picked(up_x, up_y, false);
                                    }
                                    ui::panel::SketchTool::Rectangle
                                    | ui::panel::SketchTool::RectangleCenterCorner => {
                                        state.sketch_corner_rectangle_click(up_x, up_y);
                                    }
                                    ui::panel::SketchTool::Select | ui::panel::SketchTool::Line => {
                                    }
                                }
                            }
                        }
                    }
                    // A stationary release of a viewport Select press resolves the sketch
                    // selection (a drag moved a vertex instead — the same click-vs-drag threshold
                    // placement / add-point use). Gated on `sketch_select_press` so a click on the
                    // context menu (egui-consumed, never armed) can't be read as click-empty-clear.
                    // Runs BEFORE the press ends below (the hit-test needs the press's own end).
                    if state.sketch_select_press {
                        if let Some((down_x, down_y, up_x, up_y)) = state.pointer.press_and_now() {
                            let stationary = (up_x - down_x).abs()
                                < VIEW_CUBE_DRAG_THRESHOLD_PIXELS
                                && (up_y - down_y).abs() < VIEW_CUBE_DRAG_THRESHOLD_PIXELS;
                            if stationary {
                                state.resolve_sketch_selection_click(up_x, up_y);
                                // The click selected; a SECOND one on the same number says the
                                // author means to change it, and opens the box over it. Here
                                // rather than beside the constraint release because THIS is the
                                // click a number receives: with a constraint armed the click is
                                // feeding that gesture's slots, and with a drawing tool armed it
                                // is placing geometry. Select is the only state in which clicking
                                // a number means the number.
                                state.open_measurement_editor_on_double_click(up_x, up_y);
                            } else {
                                // Slice 3: a DRAGGED release of an empty-space press resolves
                                // the directional marquee (left→right window, right→left
                                // crossing). A drag from an entity moved the vertex instead
                                // (the anchor never armed).
                                state.resolve_sketch_marquee(up_x, up_y);
                            }
                        }
                    }
                    state.sketch_marquee_anchor = None;
                    // A stationary release of a plain viewport press picks the node
                    // under the cursor (the same click-vs-drag threshold every other release path
                    // uses; it survives the orbit rebind as the future marquee discriminator).
                    // Runs BEFORE the press ends below, which the raycast needs.
                    if state.viewport_select_press {
                        if let Some((down_x, down_y, up_x, up_y)) = state.pointer.press_and_now() {
                            let stationary = (up_x - down_x).abs()
                                < VIEW_CUBE_DRAG_THRESHOLD_PIXELS
                                && (up_y - down_y).abs() < VIEW_CUBE_DRAG_THRESHOLD_PIXELS;
                            if stationary {
                                state.pending_viewport_select =
                                    state.resolve_viewport_selection_click(up_x, up_y);
                            }
                        }
                    }
                    // In orbit mode, a stationary release re-centers the view — the surface
                    // under the cursor becomes `camera.target`, so the next turn happens about
                    // what the user just pointed at. A miss (sky, no plane) is a REFUSAL, not a
                    // fallback: the view keeps the target it had. This never touches the orbit
                    // CENTER, which only the context menu moves.
                    //
                    // It ANIMATES, through the same eased tween every camera snap uses: a cut
                    // straight to the new center gives no cue which way the view went, and the
                    // point of aiming at a feature is to keep hold of it while the frame comes.
                    if state.orbit_mode_recenter_press {
                        if let Some((down_x, down_y, up_x, up_y)) = state.pointer.press_and_now() {
                            let stationary = (up_x - down_x).abs()
                                < VIEW_CUBE_DRAG_THRESHOLD_PIXELS
                                && (up_y - down_y).abs() < VIEW_CUBE_DRAG_THRESHOLD_PIXELS;
                            if stationary {
                                if let Some(point) = state.surface_point_at(Some((up_x, up_y))) {
                                    state.snap_tween = Some(camera::SnapTween::recenter(
                                        &state.app_core.camera,
                                        point,
                                    ));
                                }
                            }
                        }
                    }
                    state.orbiting_in_orbit_mode = false;
                    state.orbit_mode_recenter_press = false;
                    state.viewport_select_press = false;
                    state.sketch_select_press = false;
                    state.sketch_edit_press = false;
                    state.sketch_constraint_press = false;
                    state.armed_press = false;
                    state.pointer.end_press();
                    state.press_in_view_cube = false;
                    state.view_cube_drag_active = false;
                    // Commit the vertex drag synchronously here (a no-op if no
                    // drag is in progress). It restores the pre-drag state and queues the final
                    // position as intents the next `render` applies as one group edit. Doing it
                    // inline — not via a flag the next render reads — closes the window where a
                    // second press between release and commit could orphan the un-recorded
                    // preview. A press that never moved commits nothing.
                    state.commit_sketch_vertex_drag();
                }
            }
            WindowEvent::MouseInput {
                state: button_state,
                button: MouseButton::Middle,
                ..
            } => {
                // The middle button carries BOTH camera verbs, chosen by Shift at press:
                // plain MMB pans, Shift+MMB orbits about the ORBIT CENTER
                // (tool-modes-and-navigation.md). The center is not resolved here — the
                // camera holds it, and the whole point of it being placed is that a gesture
                // never picks it. A press that egui consumed (over the side panel / dock) or
                // on the Signal chrome doesn't grab the scene; the view cube takes no middle
                // clicks, so no cube gating is needed here.
                //
                // The verb is LATCHED at press and the two flags are mutually exclusive, so
                // releasing Shift mid-drag cannot flip an orbit into a pan halfway through.
                let in_chrome = state
                    .pointer
                    .at()
                    .map(|(x, y)| state.position_in_signal_chrome(x, y))
                    .unwrap_or(false);
                let grabbed = button_state == ElementState::Pressed && !egui_consumed && !in_chrome;
                state.orbiting_about_center = grabbed && state.shift_held;
                state.middle_button_held = grabbed && !state.orbiting_about_center;
                if state.orbiting_about_center {
                    // The TYPE is latched with the verb, and for a stronger reason: the two types
                    // are two representations of one orientation, exact only as a single point.
                    // Converting mid-drag would put a conversion inside a trajectory.
                    state.active_orbit_type = state.panel_state.default_orbit_type;
                }
            }
            WindowEvent::MouseInput {
                state: button_state,
                button: MouseButton::Right,
                ..
            } => {
                // A right-press while a tool is armed disarms it
                // (the ghost vanishes) instead of opening the cube menu — done first so
                // the cube-menu logic below never runs during placement.
                if button_state == ElementState::Pressed
                    && !egui_consumed
                    && state.cancel_orbit_center_placement()
                {
                    return;
                }
                if button_state == ElementState::Pressed
                    && !egui_consumed
                    && state.panel_state.armed_tool.is_some()
                {
                    state.disarm_placement();
                    return;
                }
                // #13 Step 3: a right-press inside the cube rect (not on egui) opens
                // the ViewCube context menu at the cursor. The menu itself is drawn
                // by egui in `run_egui_frame`; egui swallows its own clicks, so the
                // menu items never leak to the left-click snap path. Any other
                // right-press closes a menu that was open.
                if button_state == ElementState::Pressed && !egui_consumed {
                    let position = state.pointer.at();
                    let in_cube = position
                        .map(|(x, y)| state.position_in_view_cube(x, y))
                        .unwrap_or(false);
                    // A right-click on a sketch ENTITY opens the menu even though the vertex handle
                    // registers as chrome (a handle rect is in the chrome set) — the entity
                    // hit-test tells a sketch handle from the real Signal chrome, so the same menu
                    // comes up on an entity as in empty sketch space.
                    let on_sketch_entity = state.panel_state.sketch_mode.is_some()
                        && position
                            .map(|(x, y)| state.cursor_over_sketch_entity(x, y))
                            .unwrap_or(false);
                    let in_chrome = !on_sketch_entity
                        && position
                            .map(|(x, y)| state.position_in_signal_chrome(x, y))
                            .unwrap_or(false);
                    let at = position.map(|(x, y)| egui::pos2(x as f32, y as f32));
                    // A cube right-press opens the cube's own menu; a right-press anywhere else in
                    // the live viewport (not the Signal chrome) opens the general viewport menu.
                    // The two are mutually exclusive so only one is ever up.
                    // #100: the region the menu will act on — resolved at the PRESS, from the
                    // overlay the frame just projected, so the verb cannot drift to another face
                    // while the menu is up.
                    state.sketch_menu_face = (!in_cube && !in_chrome)
                        .then(|| position.and_then(|(x, y)| state.sketch_face_at(x, y)))
                        .flatten();
                    if in_cube {
                        state.context_menu_open_at = at;
                        state.viewport_menu_at = None;
                    } else if !in_chrome {
                        // Right-clicking a sketch entity selects it first, so the menu's
                        // Delete acts on it (an already-selected entity keeps the whole set).
                        if on_sketch_entity {
                            if let Some((x, y)) = position {
                                state.right_click_select_sketch_entity(x, y);
                            }
                        }
                        state.viewport_menu_at = at;
                        state.context_menu_open_at = None;
                    } else {
                        state.context_menu_open_at = None;
                        state.viewport_menu_at = None;
                    }
                }
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                // Track Shift so the sketch selection resolve can toggle/accumulate (Shift-click)
                // rather than replace. A pure modifier update; drives nothing else here.
                state.shift_held = modifiers.state().shift_key();
            }
            WindowEvent::CursorMoved { position, .. } => {
                let current = (position.x, position.y);

                if state.panel_state.sketch_tool == ui::panel::SketchTool::Line {
                    if let Some((down_x, down_y)) = state.pointer.pressed_at() {
                        state.update_line_drag((down_x, down_y), current);
                    }
                }

                // A press that started on the view cube becomes an orbit drag once it
                // moves past the threshold — the cube's own affordance, kept when orbit
                // left the left button everywhere else.
                if state.press_in_view_cube && !state.view_cube_drag_active {
                    if let Some((down_x, down_y)) = state.pointer.pressed_at() {
                        let moved = (current.0 - down_x).abs() >= VIEW_CUBE_DRAG_THRESHOLD_PIXELS
                            || (current.1 - down_y).abs() >= VIEW_CUBE_DRAG_THRESHOLD_PIXELS;
                        if moved {
                            state.view_cube_drag_active = true;
                            // Promote to an orbit drag: cancel any in-progress snap.
                            state.snap_tween = None;
                        }
                    }
                }

                // The view cube is the one place a LEFT-drag still turns the camera, about
                // `camera.target` like every non-Shift+MMB mechanism.
                let orbiting = state.view_cube_drag_active;
                if orbiting {
                    if let Some((previous_x, previous_y)) = state.pointer.at() {
                        // #13 Step 6.1: a cube drag GRABS the cube and turns it with the
                        // cursor, so the camera must orbit the OPPOSITE way round the model
                        // than the cursor moves (dragging the cube's right edge leftward spins
                        // the model to show its right face) — hence the flipped horizontal.
                        let delta_x = -((current.0 - previous_x) as f32);
                        let delta_y = (current.1 - previous_y) as f32;
                        if delta_x != 0.0 || delta_y != 0.0 {
                            // A manual orbit cancels any in-progress snap tween.
                            state.snap_tween = None;
                            // Always Constrained, whatever the default type is: the cube is a
                            // chart-native surface, and a drag on it is the same turntable the
                            // face snaps land on.
                            state.app_core.camera.orbit_by_drag_as(
                                OrbitType::Constrained,
                                delta_x,
                                delta_y,
                            );
                        }
                    }
                }

                // Shift+MMB orbits about the ORBIT CENTER, read fresh each move rather than
                // latched: it cannot change mid-drag, because the only things that write it
                // are the context menu's place/reset. Mutually exclusive with the pan below
                // (the press arm sets exactly one), so the cursor can never both orbit and
                // pan in a move.
                let orbiting = orbiting || state.orbiting_about_center;
                if state.orbiting_about_center {
                    if let Some((previous_x, previous_y)) = state.pointer.at() {
                        let delta_x = (current.0 - previous_x) as f32;
                        let delta_y = (current.1 - previous_y) as f32;
                        if delta_x != 0.0 || delta_y != 0.0 {
                            state.snap_tween = None;
                            let center = state.app_core.camera.orbit_center;
                            state.app_core.camera.orbit_about_point_as(
                                state.active_orbit_type,
                                center,
                                delta_x,
                                delta_y,
                            );
                        }
                    }
                }

                // Under orbit mode the left button turns the camera about
                // `camera.target` — the same turn the cube drag performs, but at the latched
                // ACTIVE type (the mode may be running a per-session override of the default).
                // The first move past the threshold also spends the press: one press is either a
                // turn or a re-centering click, never both.
                let orbiting = orbiting || state.orbiting_in_orbit_mode;
                if state.orbiting_in_orbit_mode {
                    if let Some((previous_x, previous_y)) = state.pointer.at() {
                        let delta_x = (current.0 - previous_x) as f32;
                        let delta_y = (current.1 - previous_y) as f32;
                        if let Some((down_x, down_y)) = state.pointer.pressed_at() {
                            let moved = (current.0 - down_x).abs()
                                >= VIEW_CUBE_DRAG_THRESHOLD_PIXELS
                                || (current.1 - down_y).abs() >= VIEW_CUBE_DRAG_THRESHOLD_PIXELS;
                            if moved {
                                state.orbit_mode_recenter_press = false;
                            }
                        }
                        if delta_x != 0.0 || delta_y != 0.0 {
                            state.snap_tween = None;
                            state.app_core.camera.orbit_by_drag_as(
                                state.active_orbit_type,
                                delta_x,
                                delta_y,
                            );
                        }
                    }
                }

                // Middle-drag pans the target in the view plane (independent of the
                // orbit path, so the cursor can never both orbit and pan in one
                // move). Like orbit, a manual pan cancels any in-progress snap tween.
                if state.middle_button_held {
                    if let Some((previous_x, previous_y)) = state.pointer.at() {
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
                state.pointer.see(current);

                // Nothing to re-aim for an armed orbit-center placement: the gizmo draws at the
                // cursor, and the ray is cast once, at the click.

                // #13 Step 4: live hover highlight for the chrome arrows. This runs
                // on every move, so keep it cheap: the chrome zones are pure
                // screen-rect tests, and we DELIBERATELY pass a `None` body picker so
                // the expensive cube raycast never fires for hover — a body-region
                // hover resolves to `None` (the body doesn't highlight anyway). Hover
                // stays `None` while orbiting/dragging, when egui ate the move, or when
                // the cursor is outside the cube rect, so it never interferes with
                // drag-orbit, the click dispatch, or the scene input.
                state.hovered_cube_zone = if orbiting
                    || egui_consumed
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
                        // Rotate arrows are a face-relative affordance — only offer
                        // them when the view is constrained to a face. Off-face hovers
                        // over a rotate gutter don't light up.
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
            WindowEvent::CursorLeft { .. } => {
                // The pointer is no longer anywhere in this window, so the place it was is not a
                // place any more. Kept apart from the release path on purpose: a release lets go
                // of the button while the pointer stays put, and this is the one event that does
                // move it away.
                state.pointer.left();
            }
            WindowEvent::MouseWheel { delta, .. } if !egui_consumed => {
                // Wheel over the Signal chrome (stack + rail) belongs to the chrome,
                // not the camera — mirroring the orbit/pan gates.
                let in_chrome = state
                    .pointer
                    .at()
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
                // Finding #0 (data-loss guard): poll the export worker and honor a pending
                // deferred close BEFORE `render()`. `render()` early-returns before it can
                // poll anything when the surface isn't presentable (window minimized /
                // occluded), which would otherwise hang the deferred close FOREVER — the
                // export result would never be observed and the app would never exit. This
                // poll and the exit check need no presentable surface, so they run here.
                state.poll_vox_export_worker();
                if state.close_requested_while_exporting && !state.export_outstanding {
                    // The export we were waiting on landed successfully (a failure clears
                    // the deferral in the poll above), so honor the pending close.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn midpoint_line_uses_only_the_stationary_edit_route() {
        assert_eq!(
            sketch_pointer_route(ui::panel::SketchTool::MidpointLine),
            SketchPointerRoute::StationaryEdit
        );
        assert_eq!(
            sketch_pointer_route(ui::panel::SketchTool::ArcCenterEndpoints),
            SketchPointerRoute::StationaryEdit
        );
        assert_eq!(
            sketch_pointer_route(ui::panel::SketchTool::ArcTangent),
            SketchPointerRoute::StationaryEdit
        );
        assert_eq!(
            sketch_pointer_route(ui::panel::SketchTool::Circle2Point),
            SketchPointerRoute::StationaryEdit
        );
        assert_eq!(
            sketch_pointer_route(ui::panel::SketchTool::Circle3Point),
            SketchPointerRoute::StationaryEdit
        );
        assert_eq!(
            sketch_pointer_route(ui::panel::SketchTool::Rectangle3Point),
            SketchPointerRoute::StationaryEdit
        );
        for tool in [
            ui::panel::SketchTool::PolygonInscribed,
            ui::panel::SketchTool::PolygonCircumscribed,
            ui::panel::SketchTool::PolygonEdge,
            ui::panel::SketchTool::Circle2Tangent,
            ui::panel::SketchTool::Circle3Tangent,
            ui::panel::SketchTool::SlotCenterToCenter,
            ui::panel::SketchTool::SlotOverall,
            ui::panel::SketchTool::SlotCenterPoint,
            ui::panel::SketchTool::SlotCenterPointArc,
            ui::panel::SketchTool::Slot3PointArc,
            ui::panel::SketchTool::BreakCurve,
            ui::panel::SketchTool::Trim,
            ui::panel::SketchTool::Extend,
            ui::panel::SketchTool::Fillet,
            ui::panel::SketchTool::ChamferEqual,
            ui::panel::SketchTool::ChamferDistanceAngle,
            ui::panel::SketchTool::ChamferTwoDistance,
            ui::panel::SketchTool::Offset,
            ui::panel::SketchTool::MoveCopy,
            ui::panel::SketchTool::Scale,
            // Both corner rectangles are two-click grammars, not drags.
            ui::panel::SketchTool::Rectangle,
            ui::panel::SketchTool::RectangleCenterCorner,
        ] {
            assert_eq!(
                sketch_pointer_route(tool),
                SketchPointerRoute::StationaryEdit
            );
        }
        assert_eq!(
            sketch_pointer_route(ui::panel::SketchTool::Line),
            SketchPointerRoute::LineClickOrArcDrag
        );
    }
}
