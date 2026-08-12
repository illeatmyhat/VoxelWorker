//! The shell's per-frame render seam: acquire the surface texture, poll the display/measurement
//! workers, run the egui frame, apply this frame's Intents + view actions, upload every
//! renderer's uniforms, and submit the shared [`render_frame`].

use super::*;

/// How far (physical px) a drawn arc chord may sag from the true curve. A quarter pixel is under
/// the width of the thinnest stroke the gizmo family draws, so the tessellation is invisible at
/// any zoom; it only ever refines the DRAWING, never the resolved profile.
const ARC_SCREEN_SAGITTA_PX: f64 = 0.25;

/// How much closer (logical px) a DERIVED vertex must be before it outranks an authored one under
/// the cursor. Sub-pixel on purpose: it settles the stacked case and nothing else.
const SKETCH_STACKED_HANDLE_BIAS: f32 = 0.5;

/// The shape a click is about to author — every drawing tool's ordinary preview mark.
fn preview_outline(chords: Vec<egui::Pos2>) -> ui::chrome::SketchPreviewMark {
    ui::chrome::SketchPreviewMark::Polyline {
        chords,
        line: ui::chrome::SketchPreviewLine::Outline,
        strength: 1.0,
    }
}

/// The datum a preview rests on — a polygon's base circle, a slot's spine. Drawn under the
/// outline, at the lighter guide weight, and never authored.
fn preview_guide(chords: Vec<egui::Pos2>) -> ui::chrome::SketchPreviewMark {
    fading_guide(chords, 1.0)
}

/// The same datum, drawn at the strength it is actually in force at. A guide that STANDS passes
/// one through [`preview_guide`]; a snap's circle passes how hard the quantity is being held.
fn fading_guide(chords: Vec<egui::Pos2>, strength: f32) -> ui::chrome::SketchPreviewMark {
    ui::chrome::SketchPreviewMark::Polyline {
        chords,
        line: ui::chrome::SketchPreviewLine::Guide,
        strength,
    }
}

impl WindowedState {
    pub(super) fn render(&mut self) {
        profiling::scope!("render");
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            // Surface lost / outdated: reconfigure and skip this frame.
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface
                    .configure(&self.gpu.device, &self.surface_config);
                return;
            }
            // Transient conditions: skip this frame, try again next redraw.
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return;
            }
            other => {
                eprintln!("surface acquisition failed: {other:?}");
                return;
            }
        };

        let target_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Issue #60: poll the geometry worker — swap in a finished (non-stale) wholesale
        // mesh rebuild before drawing so it shows this frame (stale-while-rebuilding). The
        // orchestrator is window-free, so the SHELL requests the redraw when it installed one.
        if self.display.poll_geometry_worker() {
            self.window.request_redraw();
        }
        // Perf follow-up to epic #64: install a finished (non-stale) wholesale brick
        // rebuild — mirror + display field — before drawing (stale-while-rebuilding).
        {
            let clip = self.current_mesh_clip(self.region_dimensions[2]);
            let context = Self::make_refresh_context(
                &self.panel_state,
                &mut self.app_core.two_layer_cache,
                self.region_dimensions,
                self.recenter_voxels,
                clip.band,
                clip.region,
            );
            if self.display.poll_brick_worker(context) {
                self.window.request_redraw();
            }
        }
        // Accept a finished, non-stale diameter measurement.
        self.poll_diameter_worker();

        // M6: drain the background scan channel and turn any new groups into
        // palette tiles (GPU thumbnail + egui texture registration on this thread).
        self.poll_scan();

        let raw_input = self.egui_winit_state.take_egui_input(&self.window);
        let pixels_per_point = self.egui_winit_state.egui_ctx().pixels_per_point();
        self.last_pixels_per_point = pixels_per_point;

        // Issue #12/#20 S6c-1: the layer scrubber's vertical extent comes from the
        // SCENE's region dimensions, not the assembled grid object — identical to
        // `self.region_dimensions[2]` for a chunkable scene. Z-up: layers are Z-slices,
        // so the track spans the Z dimension (index 2).
        let grid_z = AppCore::region_dimensions_for(
            &self.panel_state.scene,
            self.panel_state.geometry.voxels_per_block,
        )[2];
        let current_band = (
            self.panel_state.layer_range.lower,
            self.panel_state.layer_range.upper,
        );
        if current_band != self.measured_band {
            // Re-measure the diameter asynchronously. The streamed
            // cacheless query (a coarse block contributes its run block-granular, boundary
            // per-voxel — the same value as the dense query) is
            // O(total blocks): sub-second on a huge solid but not free, and it must never
            // block the event-loop thread. Dispatch it to the `DiameterWorker`; the shell
            // keeps showing the previous `measured_diameter` until the result lands
            // (`poll_diameter_worker`). Record `current_band` as dispatched so we don't
            // re-dispatch every frame; a later scrub or a grid edit (which resets
            // `measured_band` to `(u32::MAX, u32::MAX)`) supersedes it via the generation.
            let density = self.panel_state.geometry.voxels_per_block;
            let generation = self.diameter_generation.next_generation();
            self.diameter_worker.dispatch(DiameterRequest {
                generation,
                scene: self.panel_state.scene.clone(),
                density,
                band: current_band,
            });
            self.measured_band = current_band;
        }

        // Issue #29 S5: tell the panel where **+ Add Point** should drop a new Point —
        // the camera target, converted from the recentered render frame back to whole
        // world blocks (`(target_voxels + recenter) / density`), so a new Point lands
        // where the user is looking.
        {
            let density = self.panel_state.geometry.voxels_per_block.max(1) as i64;
            let recenter = self
                .panel_state
                .scene
                .recenter_voxels_for_resolve(self.panel_state.geometry.voxels_per_block)
                .voxels();
            let target = self.app_core.camera.target;
            self.panel_state.point_add_position_blocks = [
                ((target.x.round() as i64) + recenter[0]).div_euclid(density),
                ((target.y.round() as i64) + recenter[1]).div_euclid(density),
                ((target.z.round() as i64) + recenter[2]).div_euclid(density),
            ];
        }

        // Slow-paths item 2: the export section's live line. While an export is in flight
        // show the per-chunk progress (plus the large-export warning, if any, that was
        // stashed in `export_status` at dispatch); otherwise show the last completion /
        // failure message. Owned here so it outlives the borrow into `run_egui_frame`.
        let export_status_line = if self.export_outstanding {
            let progress = self.export_progress.as_ref().map(|(counter, total)| {
                let done = counter.load(std::sync::atomic::Ordering::Relaxed);
                if *total > 0 {
                    format!("Exporting… {done}/{total} chunks")
                } else {
                    format!("Exporting… {done} chunks")
                }
            });
            match (self.export_status.as_deref(), progress) {
                (Some(warning), Some(progress)) => Some(format!("{warning}\n{progress}")),
                (Some(warning), None) => Some(warning.to_string()),
                (None, progress) => progress,
            }
        } else {
            self.export_status.clone()
        };
        let export_panel = crate::ExportPanelState {
            in_flight: self.export_outstanding,
            status_line: export_status_line.as_deref(),
        };

        // The layer scrubber's track spans the selected object's Z
        // extent in Onion-fog mode (else the whole scene). Read it from the shared clip
        // (a no-op walk outside Onion-fog mode, where it returns the scene `grid_z`).
        let layer_track_len = self.current_mesh_clip(grid_z).track_len;
        // Read before the call: `run_egui_frame` borrows `self` mutably.
        let orbit_center_marker = self.orbit_center_marker(pixels_per_point);
        let orbit_reticle = self.orbit_reticle_visible();
        let sketch_face_at_menu = self.sketch_menu_face_is_picked();
        // Registered on the first frame and reused; the cube's square never resizes.
        let cube_texture = self
            .egui_bridge
            .view_cube_texture(&self.gpu.device, self.view_cube_renderer.standing_texture());
        let mut prepared = {
            profiling::scope!("egui_frame");
            run_egui_frame(
                &mut self.egui_bridge,
                &self.gpu.device,
                &self.gpu.queue,
                &mut self.panel_state,
                layer_track_len,
                self.measured_diameter,
                export_panel,
                &self.palette.ui,
                raw_input,
                [self.surface_config.width, self.surface_config.height],
                pixels_per_point,
                &mut self.context_menu_open_at,
                // The general viewport right-click menu (mode-dispatched Delete).
                &mut self.viewport_menu_at,
                // The icon rail's orbit-type menu.
                &mut self.orbit_type_menu_open,
                // Signal (#86): the hovered cube zone's readout name, or None.
                self.hovered_cube_zone
                    .and_then(camera::view_cube_zone_readout)
                    .as_deref(),
                // The sketch vertex handles, projected LAST frame (the
                // viewport + camera the projection needs are only known after this call).
                // A one-frame lag is imperceptible for handle chrome and self-corrects; the
                // cache is refreshed at the end of `render` below.
                &self.sketch_overlay_points,
                // The committed segment lines, projected last frame — drawn under the
                // vertex dots so the profile reads as connected edges.
                &self.sketch_segment_lines,
                // The committed arc curves, projected last frame — the same
                // under-layer as the straight edges.
                &self.sketch_arc_lines,
                // The constraint badges, projected last frame — each
                // asserted relation's glyph beside the geometry it names.
                &self.sketch_constraint_badges,
                &self.sketch_dimension_gizmos,
                // #100: the pick state of the region the open menu was raised inside, so the
                // menu can label its row "carve" or "fill".
                sketch_face_at_menu,
                // Whether the selection holds a fit point with no tangent handle yet — the one
                // question that decides whether the menu offers to mint one.
                // The add-point insert preview, projected last frame.
                self.sketch_insert_preview,
                // The snapping mark, projected last frame: where a drawing tool's pick will land
                // when it lands on a curve.
                self.sketch_snap_marker,
                // #99: the drawing tools' dashed preview, projected last frame.
                &self.sketch_draw_preview,
                // Slice 3: the marquee rubber band, computed last frame.
                self.sketch_marquee_band,
                // The orbit-center marker — live under the cursor while a placement is
                // armed, projected-last-frame while Shift+MMB turns about it.
                orbit_center_marker,
                // Whether the orbit mode's targeting reticle draws — it fills the
                // viewport rect the frame computes, so no position travels with the flag.
                orbit_reticle,
                cube_texture,
            )
        };

        // Issue #25: cache the central 3D viewport rect so the view-cube
        // hit-testing (run later, in mouse events) can offset the cube corner.
        self.last_viewport_px = prepared.viewport_px;
        // Issue #88: cache the cube's stack-derived right inset for the hit-testing.
        self.last_cube_right_inset = prepared.view_cube_right_inset_px;
        // Cache the Signal chrome hit-rects (stack + rail) for the camera gate
        // (`position_in_signal_chrome`, run in mouse events like the cube hit-test).
        self.last_chrome_rects_px = prepared.chrome_rects_px.clone();

        // #13 Step 3: execute a context-menu selection (egui drew + closed the
        // menu; the ortho toggle already mutated `panel_state.projection_mode`).
        // Home / Fit / Set-home are chart-native like the cube itself (Home SAVES and restores
        // `theta`/`phi`), so the Free Orbit seam closes before any of them reads or writes an
        // angle.
        if prepared.cube_menu_request.is_some() {
            self.settle_to_constrained();
        }
        match prepared.cube_menu_request {
            Some(ViewCubeMenuRequest::Home) => {
                self.snap_tween = Some(self.home_snap_tween());
            }
            Some(ViewCubeMenuRequest::Fit) => self.fit_to_view(),
            Some(ViewCubeMenuRequest::SetHome) => self.set_home_to_current(),
            None => {}
        }

        // The icon rail's Home/Fit click, pre-mapped onto a
        // `ChromeClickAction`, runs through the same `run_chrome_action` as the
        // cube badges used — no forked framing logic. A rail mode-cycle already mutated
        // `panel_state.view_mode` inside `run_egui_frame`, so it needs nothing here (the
        // overlay re-derivation below keys on the mode change, like a panel-driven one).
        if let Some(action) = prepared.rail_action {
            self.settle_to_constrained();
            self.run_chrome_action(action);
        }

        // Picking Constrained from the rail's type menu is the one act that is a TYPE SWITCH and
        // nothing else, so it is where the owner's animated re-level actually plays: nothing is
        // orbiting to fight the tween. Picking Free needs no eager conversion — the next drag
        // seeds the trackball from whatever the chart holds then.
        if prepared.orbit_type_picked == Some(OrbitType::Constrained) {
            self.settle_to_constrained();
        }

        // Camera UX change: right-click a node row → "Focus" frames that node. This
        // is the ONLY edit-tree action that moves the camera. Set the orbit target to
        // the node's recentered world center and fit the distance to its AABB (same fit
        // math as Fit, scoped to the node). The orbit ANGLES are held (Focus moves the
        // pivot + distance only). A node with no resolvable extent is a no-op.
        if let Some(focus_id) = prepared.panel_response.focus_node {
            if let Some((pivot, extent)) = AppCore::gizmo_placement_for_id(
                &self.panel_state.scene,
                focus_id,
                self.panel_state.geometry.voxels_per_block,
            ) {
                let (target, distance) =
                    OrbitCamera::focus_target_and_distance(glam::Vec3::from_array(pivot), extent);
                self.app_core.camera.target = target;
                self.app_core.camera.orbit_distance = distance;
            }
        }

        // M6: react to palette interactions (apply a block, connect a folder,
        // revert to a procedural material).
        self.handle_palette_response(&prepared.panel_response);

        // Advance an in-progress view-cube snap tween (eased over ~380ms).
        let now = std::time::Instant::now();
        let delta_seconds = (now - self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;
        if let Some(tween) = self.snap_tween.as_mut() {
            if tween.advance(&mut self.app_core.camera, delta_seconds) {
                self.snap_tween = None;
            }
        }

        // Feed egui's platform output (cursor icon, clipboard, …) back to winit.
        self.egui_winit_state
            .handle_platform_output(&self.window, prepared.platform_output.clone());

        // The panel does not mutate the scene directly — it
        // DESCRIBES this frame's mutations as a `Vec<Intent>`. Apply each through the
        // single `AppCore::apply_intent` door (in order), merging the returned typed
        // `IntentEffect`s, then fold them into the loop's existing decisions:
        //   * `scene_changed`     → re-resolve the grid (the old `geometry_changed` /
        //                           `scene_changed` rebuild).
        //   * `selection_changed` → re-sync the inspector mirror (the gizmo + node
        //                           highlight are recomputed every frame below from the
        //                           workspace `Selection`, so they already track it —
        //                           a pure selection click must NOT force a re-resolve).
        //   * `points_changed`    → the Points overlay is rebuilt every frame anyway
        //                           (camera-relative), so no extra work is needed.
        // Camera UX change: edits NO LONGER auto-frame the camera. The camera orbits
        // a FIXED/floating target (the world origin by default) and never jumps when
        // the user adds/moves/deletes/edits nodes. The panel's `frame_after_apply`
        // hint is intentionally IGNORED here — only the EXPLICIT view controls move
        // the camera now (startup fit, the ViewCube Home/Fit buttons, and the
        // right-click "Focus" action below). Take the intents out of `prepared`
        // (leaving it otherwise intact for the `render_frame` call below).
        // Live placement: adopt a tool the panel armed this frame (a VIEW
        // action carried on the response, like `focus_node`, not a document Intent).
        // Freshly armed = no pending drop yet; the arm pass below resolves one.
        if let Some(spec) = prepared.panel_response.arm_tool.take() {
            self.panel_state.armed_tool = Some(ui::panel::ArmedTool {
                spec,
                pending_drop: None,
            });
        }
        // The rail's armed-cell second click. Handled AFTER the adoption above so an explicit
        // disarm wins if a future second source ever sets both in one frame; the rail itself
        // emits the two mutually exclusively. Full disarm (ghost + press latch), same as Esc.
        if prepared.panel_response.disarm_tool {
            self.disarm_placement();
        }
        // A clicked row's selection change — a VIEW action on the response, like
        // `armed_tool`. The shell is the single place a click lands on the workspace
        // selection; the effect below re-syncs the inspector mirror and the operand ghost
        // exactly as a selection-only Intent used to.
        let selection_effect = match prepared
            .panel_response
            .select
            .take()
            .or_else(|| self.pending_viewport_select.take())
        {
            Some(request) => {
                // The admission rule, asserted rather than typed: a sketch entity may only be
                // picked while its own sketch is the open mode. It holds by construction — the
                // only minters of a sketch target are the in-mode click handlers, which read
                // `sketch_mode` for the id — so a violation is a shell bug, not a user state.
                debug_assert!(
                    match request {
                        ui::panel::SelectionRequest::Only(target)
                        | ui::panel::SelectionRequest::Toggle(target) => target
                            .owning_sketch()
                            .is_none_or(|sketch| self.panel_state.sketch_mode == Some(sketch)),
                        ui::panel::SelectionRequest::Clear => true,
                    },
                    "a sketch entity was selected outside its own open sketch"
                );
                self.panel_state.selection.apply_request(request);
                crate::IntentEffect::selection()
            }
            None => crate::IntentEffect::none(),
        };
        // Enter / leave sketch mode — a VIEW action on the response (entering a mode
        // mutates no document state), like `armed_tool`. Entering scopes the mode to the
        // requested node, disarms any placement tool (non-sketch ops withdraw in the mode), and
        // OPENS the undo group (§4). Finish commits the session as one main-history entry;
        // Cancel rolls it back to the enter-state (which re-resolves) — both drop the mode. The
        // group-close effect folds into `merged_effect` below so a Cancel rebuilds like an edit.
        let mut sketch_effect = crate::IntentEffect::none();
        if let Some(node) = prepared.panel_response.enter_sketch.take() {
            self.reset_sketch_gestures();
            self.sketch_chamfer_pending = None;
            self.sketch_offset_pending = None;
            self.sketch_move_copy_pending = None;
            self.sketch_scale_pending = None;
            self.sketch_rectangular_pattern_pending = None;
            self.panel_state.sketch_mode = Some(node);
            self.disarm_placement();
            self.panel_state.selection.clear_sketch_entities();
            self.app_core.begin_sketch_group();
        }
        if let Some(exit) = prepared.panel_response.exit_sketch.take() {
            self.reset_sketch_gestures();
            self.sketch_chamfer_pending = None;
            self.sketch_offset_pending = None;
            self.sketch_move_copy_pending = None;
            self.sketch_scale_pending = None;
            self.sketch_rectangular_pattern_pending = None;
            sketch_effect = match exit {
                ui::panel::SketchExit::Finish => self.app_core.finish_sketch_group(),
                ui::panel::SketchExit::Cancel => self.app_core.cancel_sketch_group(
                    &mut self.panel_state.scene,
                    &mut self.panel_state.selection,
                ),
            };
            self.panel_state.sketch_mode = None;
            self.panel_state.selection.clear_sketch_entities();
        }
        // The context menu's Delete. Its keyboard binding lands on the same method from
        // `run_shortcut_commands` below; both queue through `viewport_intents`, gathered just
        // after.
        if prepared.panel_response.delete_selection {
            self.delete_selection();
        }
        // #100: the context menu's carve / fill row, acting on the region the press resolved.
        if prepared.panel_response.toggle_sketch_face {
            self.toggle_sketch_menu_face();
        }
        if prepared.panel_response.close_sketch_loop {
            self.close_sketch_line_loop();
        }
        if prepared.panel_response.toggle_sketch_construction {
            self.toggle_sketch_selection_construction();
        }
        if let Some((constraint, restated)) = prepared.panel_response.restate_sketch_dimension {
            self.restate_sketch_dimension(constraint, restated);
        }
        // The context menu's orbit-center rows. Not an `Intent` and not undoable — the camera
        // is not the document (this is view state).
        match prepared.panel_response.orbit_center_request {
            // "Place" ARMS rather than places: the center then follows the cursor, visibly,
            // until a click commits it. Placing straight onto the right-clicked point would
            // put it somewhere the user only sees after the menu has already closed.
            Some(ui::panel::OrbitCenterRequest::Place) => {
                self.begin_orbit_center_placement();
            }
            Some(ui::panel::OrbitCenterRequest::Reset) => {
                self.cancel_orbit_center_placement();
                self.app_core.camera.reset_orbit_center();
            }
            None => {}
        }
        // The viewport menu's OK / Cancel variant, which every running modal command ends
        // through. The same door Return / Escape use, so a command can never be left half-out.
        if let Some(command) = prepared.panel_response.mode_command {
            self.end_modal_command(command);
        }
        // The keyboard half of the same doors. Read AFTER the pass, so a focused text field has
        // already eaten its own keys. Returns the Undo/Redo effect, folded into the merge below.
        let shortcut_effect = self.run_shortcut_commands();
        // Advance an in-progress sketch vertex drag — a live preview that
        // re-resolves the volume and records ONE coalesced command in the open group. Uses
        // this frame's viewport (from `prepared`) to build the cursor→plane ray; its effect
        // folds into `merged_effect` below so the display re-resolves like any other edit.
        let drag_effect = {
            let [_, _, drag_vw, drag_vh] = prepared.viewport_px;
            let drag_aspect = drag_vw as f32 / drag_vh.max(1) as f32;
            // The cursor→plane INVERSE map wants the ray-frame matrix (wide-baseline precise);
            // the forward handle projection below keeps the full VP.
            let drag_ray_unprojection = self
                .app_core
                .scene_matrices(drag_aspect, self.region_dimensions)
                .ray_unprojection;
            self.update_sketch_vertex_drag(drag_ray_unprojection, prepared.viewport_px)
        };
        let mut intents = std::mem::take(&mut prepared.panel_response.intents);
        // Live placement: a viewport click's drop intent is applied through the
        // SAME door as the panel's edits (taken BEFORE the borrow of `prepared` ends), so
        // a placement re-resolves + rebuilds identically to a panel-driven add.
        intents.extend(std::mem::take(&mut self.viewport_intents));
        let mut merged_effect = sketch_effect
            .merged_with(drag_effect)
            .merged_with(selection_effect)
            .merged_with(shortcut_effect);
        for intent in intents {
            let effect = self.app_core.apply_intent(
                &mut self.panel_state.scene,
                &mut self.panel_state.selection,
                intent,
            );
            merged_effect = merged_effect.merged_with(effect);
        }
        // Batched intents that must land as ONE undo step: a multi-node Delete, and
        // every sketch commit — one authoring act, one press of Ctrl+Z.
        for transaction in std::mem::take(&mut self.viewport_transactions) {
            let effect = self.app_core.apply_transaction(
                &mut self.panel_state.scene,
                &mut self.panel_state.selection,
                transaction,
            );
            merged_effect = merged_effect.merged_with(effect);
        }
        // Coordinate-limit warning (authoring-time only): latch a rejected edit into the
        // inspector warning, and clear it on the next accepted geometry edit.
        if merged_effect.coordinate_limit_rejected {
            self.panel_state.coordinate_limit_warning = true;
        } else if merged_effect.scene_changed {
            self.panel_state.coordinate_limit_warning = false;
        }
        if merged_effect.selection_changed || merged_effect.scene_changed {
            // Re-sync the inspector mirror to the active node. The OLD panel called
            // `sync_mirror_from_active` after EVERY structural action (add / group /
            // make-definition / add-instance / delete — each of which changes the
            // active node) AND on a row select; we reproduce that by syncing on a
            // `selection_changed` (a pure selection click) OR a `scene_changed` (a
            // structural edit may have moved the active selection to a freshly-added /
            // re-derived node). Syncing after an inspector `SetShape`/`SetDensity` is a
            // harmless no-op (the node now equals the buffer it was written from). The
            // transform gizmo + row highlight read the workspace `Selection` live each
            // frame, so a pure selection click updates them WITHOUT a re-resolve.
            self.panel_state.sync_mirror_from_active();
        }
        if merged_effect.scene_changed {
            // A structural / node-field / global-density edit re-resolves the grid.
            // Camera UX change: this NEVER auto-frames any more — `false` keeps the
            // camera target + distance fixed across every edit. Re-framing is now only
            // via explicit controls (Home/Fit/Focus) and the startup fit.
            self.rebuild_geometry();
        }
        // Re-derive the boolean-operand ghost on selection /
        // geometry / MODE change ONLY (never per frame). A selection click marks it dirty
        // without a scene re-resolve; the derivation is bounded by the ghosted operands'
        // covering chunks (`AppCore::boolean_operand_ghost`), so this stays cheap even in
        // a huge scene. The selection / mode comparisons are belt-and-braces for any
        // selection or mode writer that bypassed the Intent effects. The ghost is
        // populated only in Show-booleans mode; Normal / Onion-fog derive nothing.
        if merged_effect.selection_changed || merged_effect.scene_changed {
            self.selected_ghost_dirty = true;
        }
        // The selection outline+wash re-derives on the SAME seam (selection /
        // geometry change), for ALL view modes and the WHOLE pick-ordered node list —
        // reading the shared dirty flag BEFORE the operand-ghost block below clears it.
        // Bounded per node by its own covering chunks (`AppCore::selected_body_cel`).
        {
            let cel_nodes: Vec<crate::NodeId> = self.panel_state.selection.nodes().collect();
            if self.selected_ghost_dirty || self.selected_cel_nodes != cel_nodes {
                let cel = AppCore::selected_body_cel(
                    &self.panel_state.scene,
                    &cel_nodes,
                    self.panel_state.geometry.voxels_per_block,
                );
                self.selected_cel_nodes = cel_nodes;
                match cel {
                    Some(cel) => self.selection_outline_renderer.rebuild(
                        &self.gpu.device,
                        &cel.bodies,
                        &cel.edge_segments,
                        cel.grid_dimensions,
                        cel.recenter,
                        cel.density,
                    ),
                    None => self.selection_outline_renderer.clear(),
                }
            }
        }
        if self.selected_ghost_dirty
            || self.selected_ghost_selection != self.panel_state.selection.primary_node_id()
            || self.selected_ghost_view_mode != self.panel_state.view_mode
        {
            self.selected_ghost_dirty = false;
            self.selected_ghost_selection = self.panel_state.selection.primary_node_id();
            self.selected_ghost_view_mode = self.panel_state.view_mode;
            let ghost = self
                .panel_state
                .selection
                .primary_node_id()
                .filter(|_| self.panel_state.view_mode == crate::ViewMode::ShowBooleans)
                .and_then(|target| {
                    AppCore::boolean_operand_ghost(
                        &self.panel_state.scene,
                        target,
                        self.panel_state.geometry.voxels_per_block,
                    )
                });
            match ghost {
                Some(ghost) => self.selected_operand_ghost_renderer.rebuild(
                    &self.gpu.device,
                    &ghost.bodies,
                    ghost.grid_dimensions,
                    ghost.recenter,
                    ghost.density,
                ),
                None => self.selected_operand_ghost_renderer.clear(),
            }
        }
        // Brick-display perf follow-up to epic #64: a debug-face toggle or a loaded-material
        // change are PURE display flags (they never `scene_changed`, so no rebuild fires) that
        // can turn OFF brick engagement — making the SKIPPED fallback mesh the display. Rebuild
        // it here the frame it is next needed, so a stale/empty mesh is never drawn. A no-op
        // unless the mesh is stale AND about to be shown.
        {
            let clip = self.current_mesh_clip(self.region_dimensions[2]);
            let context = Self::make_refresh_context(
                &self.panel_state,
                &mut self.app_core.two_layer_cache,
                self.region_dimensions,
                self.recenter_voxels,
                clip.band,
                clip.region,
            );
            self.display.ensure_display_mesh_current(context);
        }

        // Projection is a display-only param: apply it to the camera each frame
        // (no rebuild).
        self.app_core.camera.projection_mode = self.panel_state.projection_mode;

        // Upload the per-frame uniforms before drawing: camera matrix, grid
        // half-extent + density (per-voxel slice + overlay), and the overlay
        // toggle. The grid dims are the current geometry's voxel-space size.
        // Issue #25: the camera aspect comes from the CENTRAL 3D viewport rect (the
        // window minus the side panel + bottom dock), not the whole window, so the
        // model is centered in the visible 3D area instead of partly hidden behind
        // the side panel. `prepared.viewport_px` = [x, y, w, h] in physical pixels.
        let [_, _, viewport_width, viewport_height] = prepared.viewport_px;
        let aspect_ratio = viewport_width as f32 / viewport_height.max(1) as f32;
        let geometry = self.panel_state.geometry.clone();
        // The grid dims come from the ACTUALLY resolved scene grid (the composited
        // region's extent), not the active node's geometry — with several nodes the
        // region is the per-axis max of their sizes.
        let grid_dimensions = self.region_dimensions;
        let scene_matrices = self.app_core.scene_matrices(aspect_ratio, grid_dimensions);
        let view_projection = scene_matrices.view_projection;
        // Refresh the sketch vertex-handle overlay from the CURRENT geometry
        // (post-rebuild recenter) and camera, caching the projected handles for NEXT frame's
        // draw (in `run_egui_frame`) and the press hit-test (in `events`). A one-frame lag on
        // the handles is imperceptible and self-corrects.
        self.refresh_sketch_overlay(view_projection, prepared.viewport_px, pixels_per_point);
        // The orbit-center marker, projected for NEXT frame's draw with the same one-frame lag.
        self.refresh_orbit_center_overlay(view_projection, prepared.viewport_px, pixels_per_point);
        // #95: cache the ray-frame matrix so the release handler (in `events`) can invert a
        // cursor into a profile coordinate for an add-point insert, using the SAME frame the
        // overlay saw and without the wide-baseline `/w` melt of the full-VP inverse.
        self.last_ray_unprojection = Some(scene_matrices.ray_unprojection);
        // Issue #12: translate the layer-range scrubber into the shader band. The
        // band is inclusive on both ends; the upper handle is a layer index, so a
        // single-layer band is `lower == upper`. A full range draws everything.
        // Z-up: layers are Z-slices, so the band is a Z-layer range (index 2). The band
        // is computed by the shared `current_layer_band` helper (issue #60 M2) so the async
        // worker builds the mesh at the SAME band the render path applies here.
        let layer_range = self.panel_state.layer_range;
        // The region-scoped clip (band + onion-fog region). The
        // band bites only in Onion-fog mode with a selection; the region confines it to the
        // selected object's AABB. BOTH display paths honor the region — the cuboid mesh path
        // (geometry) and the brick raymarch (per-frame uniforms, #85).
        let clip = self.current_mesh_clip(grid_dimensions[2]);
        let band = clip.band;
        // Part of #20: the cuboid mesh path is the sole voxel renderer. Upload its
        // per-frame uniforms (camera + per-material base colors + band + region clip). A
        // loaded VS block textures it per-face (its 6-layer D2Array is bound at DRAW
        // time in `render_frame`, selecting the loaded pipeline); `bound = None` then
        // just disables the procedural per-box modulation/atlas, which the loaded
        // pipeline ignores.
        let bound = match &self.loaded_material {
            Some(_) => None,
            None => Some(self.panel_state.material),
        };
        // The onion GHOST replaces the volumetric fog. Active when onion skin is on
        // and the band is a real slab (`current_layer_band` sets a non-zero `onion_depth` exactly
        // then; debug-face mode forces FULL → 0). The engaged display path draws the ghost after
        // its solid pass (`render_frame`); a band scrub is a pure uniform update on the brick path,
        // a thin-slab re-mesh on the cuboid path — never the fog atlas rebuild.
        let onion_ghost_active = band.onion_depth > 0;
        // Voxel-model uniforms, shared with `shot`: the cuboid mesh + (when engaged) the
        // brick raymarch that replaces its draw. One call keeps the two paths pixel-comparable (the
        // gpu_parity premise). Interactive-only brick flags (loaded-material shade, the brick-faces
        // diagnostic) travel as params; the on-face-grid MASTER (#29 S4) is `scene.master_voxel_grid`.
        let grid_overlay_master = self.panel_state.scene.master_voxel_grid;
        let debug_face_orientation = self.panel_state.debug_face_orientation;
        let loaded_material_active = self.loaded_material.is_some();
        let debug_brick_faces = u32::from(self.panel_state.debug_brick_faces);
        let voxels_per_block = geometry.voxels_per_block;
        let region = clip.region;
        // A live brick field AND no mesh-only mode ⇒ the brick draw replaces the mesh this frame.
        let brick_engaged = self.display.brick_display_engaged(debug_face_orientation);
        let (cuboid, brick_present) = self.display.voxel_renderers_mut();
        let brick = if brick_engaged { brick_present } else { None };
        let brick_raymarch_engaged = crate::frame::render::upload_voxel_uniforms(
            &self.gpu.device,
            &self.gpu.queue,
            scene_matrices,
            prepared.viewport_px,
            grid_dimensions,
            voxels_per_block,
            band,
            region,
            grid_overlay_master,
            bound,
            debug_face_orientation,
            loaded_material_active,
            debug_brick_faces,
            cuboid,
            brick,
        );
        // Transform gizmo (issue #29 S2): it FOLLOWS the selected node — `None` (nothing
        // selected, or selection has no extent) hides it. Its camera upload rides the shared
        // overlay-uniforms call below; here we only resolve WHETHER it is placed (the phase
        // assembly gates its draw on this).
        let gizmo_placement = self
            .panel_state
            .selection
            .primary_node_id()
            .and_then(|target| {
                AppCore::gizmo_placement_for_id(
                    &self.panel_state.scene,
                    target,
                    self.panel_state.geometry.voxels_per_block,
                )
            });
        // Per-object block lattice + floor grid (issue #29 S3): rebuild this frame's
        // line batch from the scene — for every node whose grids are enabled (the
        // scene master ANDed with the node's own toggle), its enclosing-block lattice
        // / base-plane floor lines. Empty when no node enables a grid (the new
        // default — per-object grids are OFF until the user turns them on).
        // Scene scaffold uniforms (per-object scene grid + world-reference Points + analytic
        // infinite grid), shared with `shot` through one orchestration point. The
        // shell always draws the Points; `axes_on_top` skips the depth-tested Points instance.
        let axes_through = self.panel_state.axes_on_top;
        let overlay_vp = self.app_core.points_overlay_view_projection(
            aspect_ratio,
            &self.panel_state.scene,
            self.panel_state.geometry.voxels_per_block,
        );
        crate::frame::render::upload_scene_scaffold(
            &self.gpu.device,
            &self.gpu.queue,
            &self.panel_state.scene,
            self.panel_state.geometry.voxels_per_block,
            &self.app_core.camera,
            scene_matrices,
            overlay_vp,
            true,
            axes_through,
            &mut self.scene_grid_renderer,
            &mut self.points_renderer,
            &mut self.points_overlay_renderer,
            &mut self.infinite_grid_renderer,
        );
        // Live placement: while a tool is armed and the cursor is over the
        // viewport, resolve where it would drop (via the headless `place_primitive`) and
        // write the pending drop + the pending click intent onto the armed tool. Armed
        // with NO cursor keeps a restored drop untouched (an F9 repro replays it until
        // the first motion re-resolves); a non-Tool spec clears it, so a stale preview
        // never lingers.
        // NO resident-geometry guard: `place_primitive`'s tier 1 (`pick_voxel`) returns
        // `None` on an empty scene and falls through to the world-plane tier, which needs
        // no chunks — so the ghost must preview on an empty scene (the ground plane), not
        // only once something is built. This runs before the ghost's uniform upload below,
        // which reads the armed tool's pending drop.
        let armed_spec = self
            .panel_state
            .armed_tool
            .as_ref()
            .map(|armed| armed.spec.clone());
        match (armed_spec, self.last_cursor_position) {
            (Some(NodeSpec::Tool { shape, material }), Some((cursor_x, cursor_y))) => {
                let vp = prepared.viewport_px;
                // Same physical-pixel viewport/cursor space `pick_voxel` marches in.
                let viewport = [vp[0] as f32, vp[1] as f32, vp[2] as f32, vp[3] as f32];
                let cursor = [cursor_x as f32, cursor_y as f32];
                let frame = crate::PickFrame {
                    region_dimensions: self.region_dimensions,
                    recenter_voxels: self.recenter_voxels.voxels(),
                    density: self.panel_state.geometry.voxels_per_block,
                    chunks: &self.resident_chunks,
                    band: self.last_pick_band,
                };
                let outcome = self.app_core.place_primitive(
                    cursor,
                    viewport,
                    &frame,
                    &self.panel_state.scene,
                    shape.clone(),
                    material,
                    self.panel_state.scene.master_floor_grid,
                    self.panel_state.placement_snap,
                );
                self.pending_placement = outcome.intent.clone();
                let pending_drop = match &outcome.intent {
                    Some(crate::Intent::PlaceNode {
                        offset_voxels,
                        offset_local,
                        rotation_quaternion,
                        ..
                    }) => {
                        // The ghost previews the node as it will land — tilted to the
                        // surface normal AND at the exact sub-voxel offset — so carry the same
                        // continuous rotation AND `offset_local` the intent would apply (placement
                        // writes the whole tilt into the quaternion, so a `None` is an upright drop;
                        // `offset_local` is the sub-voxel remainder a `NoSnap` drop keeps).
                        Some(crate::PlacementGhost {
                            shape,
                            offset_voxels: *offset_voxels,
                            offset_local: *offset_local,
                            rotation: rotation_quaternion
                                .map(glam::Quat::from_array)
                                .unwrap_or(glam::Quat::IDENTITY),
                        })
                    }
                    // NoSurface / TooFar carry no intent → no ghost, and a click there
                    // does nothing (the pending intent is None).
                    _ => None,
                };
                if let Some(armed) = &mut self.panel_state.armed_tool {
                    armed.pending_drop = pending_drop;
                }
            }
            // Armed, cursor not over the viewport yet: keep a restored pending drop for
            // display, but no click can land it (the pending intent stays cleared).
            (Some(NodeSpec::Tool { .. }), None) => {
                self.pending_placement = None;
            }
            // A non-Tool spec has no drop resolve; clear any stale one.
            (Some(_), _) => {
                self.pending_placement = None;
                if let Some(armed) = &mut self.panel_state.armed_tool {
                    armed.pending_drop = None;
                }
            }
            (None, _) => {
                self.pending_placement = None;
            }
        }
        // The armed-tool placement ghost. Arm it from the armed tool's pending
        // drop (resolved live above, or restored from a loaded config F9 repro), resolving
        // the render-frame field center from THIS rebuild's recenter so the ghost sits in
        // the exact frame the solid voxels are drawn in. Disarmed → no-op.
        if let Some(ghost) = self.panel_state.placement_ghost() {
            let voxels_per_block = self.panel_state.geometry.voxels_per_block;
            let recenter = self.recenter_voxels.voxels();
            self.placement_ghost_renderer.update_uniforms(
                &self.gpu.queue,
                view_projection,
                scene_matrices.ray_unprojection.inverse(),
                scene_matrices.ray_eye,
                prepared.viewport_px,
                glam::Vec3::from_array(ghost.center_world(recenter, voxels_per_block)),
                ghost.shape.kind,
                glam::Vec3::from_array(ghost.semi_axes(voxels_per_block)),
                ghost.wall_voxels(voxels_per_block),
                crate::PLACEMENT_GHOST_TINT,
                ghost.rotation_inverse_columns(),
            );
        } else {
            self.placement_ghost_renderer.disarm();
        }
        // The picked region's wash, as a FIELD over the sketch plane. The plane
        // basis comes from the one forward map the vertex handles use, so the wash cannot land on a
        // different plane than the handles do.
        let sketch_region = self.panel_state.sketch_mode.and_then(|target| {
            let handles = self
                .panel_state
                .scene
                .sketch_handles(target, self.panel_state.geometry.voxels_per_block)?;
            let node = self.panel_state.scene.node_by_id(target)?;
            let document::scene::NodeContent::SketchTool { producer, .. } = &node.content else {
                return None;
            };
            // No tolerance and no screen-scale heuristic: the region carries its arcs and the wash
            // measures the curve, so there is nothing here for a zoom level to be right about.
            //
            // Derived from the sketch IN PLACE, never from a copy. `RegionMemo` clones EMPTY, so a
            // cloned sketch re-runs the whole arrangement every frame and throws the result away
            // with the copy.
            let region = self
                .sketch_evaluation_context()
                .map(|context| producer.sketch.region_field_loops(context));
            Some((handles, region))
        });
        if let Some((handles, region)) = sketch_region {
            let Some(region) = region else {
                self.sketch_region_renderer.disarm();
                return;
            };
            let origin = handles.profile_to_render([0.0, 0.0]);
            let axis = |coord: [f64; 2]| {
                let tip = handles.profile_to_render(coord);
                [tip[0] - origin[0], tip[1] - origin[1], tip[2] - origin[2]]
            };
            self.sketch_region_renderer.update(
                &self.gpu.device,
                &self.gpu.queue,
                scene_matrices.ray_unprojection.inverse(),
                scene_matrices.ray_eye,
                prepared.viewport_px,
                display::renderer::SketchPlaneFrame {
                    origin,
                    axis0: axis([1.0, 0.0]),
                    axis1: axis([0.0, 1.0]),
                    normal: handles.plane_normal,
                },
                &region,
                ui::theme::linear_rgba(ui::theme::color_palette::SKETCH_REGION_FILL),
            );
        } else {
            self.sketch_region_renderer.disarm();
        }
        // Overlay uniforms shared with `shot`: the selection-follow gizmo, the
        // boolean-operand x-ray ghost, and the corner view cube — one orchestration point so the
        // two paths cannot drift.
        // The outline's target-sized depth map (a cheap no-op unless the target resized).
        self.selection_outline_renderer.prepare(
            &self.gpu.device,
            self.surface_config.width,
            self.surface_config.height,
            &self.depth_view,
        );
        crate::frame::render::upload_overlay_uniforms(
            &self.gpu.queue,
            &self.app_core.camera,
            aspect_ratio,
            view_projection,
            scene_matrices.ndc_depth,
            gizmo_placement,
            &self.transform_gizmo_renderer,
            &self.selected_operand_ghost_renderer,
            &self.selection_outline_renderer,
            &self.view_cube_renderer,
        );

        // Onion context draws as the
        // display paths' ghost pass (prepared above: the brick slabs in `update_ghost_uniforms`,
        // the cuboid slabs in `update_uniforms` → `rebuild_for_band`; drawn in `render_frame`
        // when `onion_ghost_active`).
        let _ = layer_range;

        // The ordered frame phases. Each renderer self-gates (empty batch → no
        // draw), so an always-included draw is a cheap no-op; only the gizmo (a fixed unit
        // gizmo, always non-empty) is gated on there being a selection.
        let background: [&dyn display::SceneDraw; 1] = [&self.background_gradient_renderer];
        let mut over_model: Vec<&dyn display::SceneDraw> = Vec::new();
        // The operand x-ray — suppressed in debug-faces mode; self-gates when
        // empty. (The selection feedback is no longer an over-model draw: it is
        // the screen-space outline+wash composite, wired below as `selection_outline`.)
        if !self.panel_state.debug_face_orientation {
            over_model.push(&self.selected_operand_ghost_renderer);
        }
        // The armed-tool placement ghost self-gates on a pending drop.
        if self.panel_state.placement_ghost().is_some() {
            over_model.push(&self.placement_ghost_renderer);
        }
        // The sketch region wash self-gates on a sketch being open.
        over_model.push(&self.sketch_region_renderer);
        // Behind-model: the occluded axes' paint-order pass (depth-off overlay), drawn before the
        // model so geometry paints over it — the invariant part that never clips.
        let mut behind_model: Vec<&dyn display::SceneDraw> = Vec::new();
        if !axes_through {
            behind_model.push(&self.points_overlay_renderer);
        }
        // Scaffold: per-object grids + the analytic infinite grid (Points' planes) — each
        // self-gates. The Points' axes' depth-tested pass joins it only when occluded.
        let mut scaffold: Vec<&dyn display::SceneDraw> =
            vec![&self.scene_grid_renderer, &self.infinite_grid_renderer];
        if !axes_through {
            scaffold.push(&self.points_renderer);
        }
        // On-top: the Points' axes (overlay instance, when the on-top setting is on) then the gizmo.
        let mut on_top: Vec<&dyn display::SceneDraw> = Vec::new();
        if axes_through {
            on_top.push(&self.points_overlay_renderer);
        }
        if gizmo_placement.is_some() {
            on_top.push(&self.transform_gizmo_renderer);
        }
        let phases = FramePhases {
            background: &background,
            behind_model: &behind_model,
            over_model: &over_model,
            scaffold: &scaffold,
            on_top: &on_top,
            cuboid_mesh: self.display.cuboid_mesh_renderer(),
            // When engaged, the brick raymarch replaces the cuboid-mesh DRAW for
            // this frame; the mesh stays built as the fallback + A/B reference.
            brick_raymarch: if brick_raymarch_engaged {
                self.display.brick_raymarch_renderer()
            } else {
                None
            },
            // Ghost the onion slabs after the solid draw (uniforms/geometry prepared above).
            onion_ghost_active,
            // The selection outline+wash — suppressed in debug-faces mode (like
            // the operand x-ray); self-gates on an empty selection.
            selection_outline: if self.panel_state.debug_face_orientation {
                None
            } else {
                Some(&self.selection_outline_renderer)
            },
            // The view cube is always drawn.
            view_cube: Some(&self.view_cube_renderer),
            // #13 Step 4: live hover — the chrome zone under the cursor so the hovered arrow brightens.
            cube_hovered_zone: self.hovered_cube_zone,
            // #13 Step 6 follow-up: the rotate arrows are a standing affordance whenever the view
            // is face-constrained (not hover-gated).
            cube_rotate_arrows_visible: self.app_core.camera.is_face_constrained(),
        };

        // M6: an applied VS block overrides the procedural material selection.
        let material = match &self.loaded_material {
            Some(loaded) => MaterialSource::Loaded(&loaded.bind_group),
            None => MaterialSource::Procedural(self.panel_state.material),
        };

        {
            profiling::scope!("render_submit");
            render_frame(
                &mut self.egui_bridge,
                &self.gpu.device,
                &self.gpu.queue,
                &target_view,
                &self.msaa_color_view,
                &self.depth_view,
                material,
                &phases,
                &prepared,
            );

            surface_texture.present();
        }

        // One frame mark per rendered frame (not per event). No-op unless a
        // profiling backend is enabled; under `--features tracy` this delimits the
        // frame on the Tracy timeline.
        profiling::finish_frame!();
    }

    /// Advance an in-progress sketch vertex drag by one frame — a LIVE PREVIEW.
    /// The gesture is COMMITTED synchronously by [`commit_sketch_vertex_drag`], called from the
    /// `events` release handler (NOT deferred to a render flag: deferring left a window where a
    /// second press between release and the commit frame could orphan the un-recorded preview).
    /// Returns the effect to merge (a `scene_changed` drives the live re-resolve).
    ///
    /// [`commit_sketch_vertex_drag`]: Self::commit_sketch_vertex_drag
    fn update_sketch_vertex_drag(
        &mut self,
        ray_unprojection: glam::Mat4,
        viewport_px: [u32; 4],
    ) -> crate::IntentEffect {
        self.preview_sketch_vertex_drag(ray_unprojection, viewport_px)
    }

    /// The live-preview half: project the cursor onto the sketch plane, grid-snap the profile
    /// coordinate (grid density = voxel density ⇒ round to the nearest whole voxel), compensate
    /// the node offset by the bbox-min shift so the NON-dragged vertices stay put in world (the
    /// producer re-anchors its bbox-min to the node origin, so without this the grabbed
    /// min-vertex would pin and the rest would lurch), and direct-mutate the node for a LIVE
    /// re-resolve — no command recorded. `none` when nothing changed this frame.
    fn preview_sketch_vertex_drag(
        &mut self,
        ray_unprojection: glam::Mat4,
        viewport_px: [u32; 4],
    ) -> crate::IntentEffect {
        use crate::IntentEffect;
        let Some((held, original_min, original_offset)) = self
            .sketch_drag
            .as_ref()
            .map(|drag| (drag.held, drag.original_min, drag.original_offset))
        else {
            return IntentEffect::none();
        };
        let Some(target) = self.panel_state.sketch_mode else {
            self.end_the_vertex_drag();
            return IntentEffect::none();
        };
        let Some((cursor_x, cursor_y)) = self.last_cursor_position else {
            return IntentEffect::none();
        };
        // Recompute the handles from the CURRENT scene (not last frame's cache): a mid-drag
        // move can shift the composite recenter / profile bbox, and the forward projection and
        // the inverse plane-hit map must share ONE frame or the vertex jitters.
        let Some(handles) = self
            .panel_state
            .scene
            .sketch_handles(target, self.panel_state.geometry.voxels_per_block)
        else {
            return IntentEffect::none();
        };

        // Cursor → the continuous profile coordinate under it, then quantize by the position
        // snap (#96). The ray/plane math is shared with the add-point insert.
        let Some(profile_coord) = self.cursor_to_profile_coord(
            cursor_x,
            cursor_y,
            ray_unprojection,
            viewport_px,
            &handles,
        ) else {
            return IntentEffect::none();
        };
        let snapped = apply_sketch_snap(
            profile_coord,
            self.panel_state.sketch_snap,
            self.panel_state.geometry.voxels_per_block,
        );
        // The snap's ceiling, stated in screen points and converted here because only the shell has
        // a camera. Measured by asking the SAME cursor-to-plane map one pixel right and one pixel
        // down, so it is exact under perspective and on a tilted plane and cannot drift out of step
        // with the map the drag itself used. The larger of the two steps: a foreshortened plane
        // should err toward letting the snap hold rather than clipping it.
        let snap_reach = self
            .cursor_to_profile_coord(
                cursor_x + 1.0,
                cursor_y,
                ray_unprojection,
                viewport_px,
                &handles,
            )
            .zip(self.cursor_to_profile_coord(
                cursor_x,
                cursor_y + 1.0,
                ray_unprojection,
                viewport_px,
                &handles,
            ))
            .map_or(document::sketch::SnapReach::UNBOUNDED, |(right, down)| {
                let step =
                    |to: [f64; 2]| (to[0] - profile_coord[0]).hypot(to[1] - profile_coord[1]);
                document::sketch::SnapReach::of_length(
                    f64::from(ui::chrome::SKETCH_SNAP_REACH)
                        * self.window.scale_factor()
                        * step(right).max(step(down)),
                )
            });

        // Build the preview from the pre-drag producer with ONLY the dragged vertex moved, then
        // compensate the offset by the bbox-min shift so the rest of the profile holds still.
        let Some(drag) = self.sketch_drag.as_ref() else {
            return IntentEffect::none();
        };
        let mut preview = drag.original.clone();
        // Mutate the grabbed point ENTITY directly by its stable id — no loop index.
        // The snap policy re-authors the whole position (#96/#101): a snapped drag zeroes
        // the fraction, NoSnap carries it; either way a stale retained expression drops.
        let Some(context) = self.sketch_evaluation_context() else {
            self.end_the_vertex_drag();
            return IntentEffect::none();
        };
        // A CLICK IS NOT A TINY DRAG. Every arm below is gated on the gesture having begun, and a
        // press that has not left its own start by the drag threshold has not: it is on its way to
        // being a click, and a click makes things active without making them move.
        let began = drag.began || self.pointer_left_the_press();
        // The gesture's own memory of how far it has turned each arc, lent to the drawing for the
        // frame and taken back after. Which way round an arc is drawn is path-dependent, and the
        // preview is rebuilt from the pre-drag producer every frame, so the drawing cannot know it.
        let mut carried_arcs = drag.arc_turns.clone();
        let moved = match held {
            SketchGrab::Point(id) if began => preview
                .sketch
                .move_point_reporting_its_snap(id, snapped, context, snap_reach, &mut carried_arcs)
                .map(|answered| {
                    // The ghost is the drag's, not the frame's: a step that did not snap says so
                    // by clearing it, so the circle appears and disappears with the hand rather
                    // than lingering once the author has pulled off it.
                    self.sketch_snap_ghost = answered.kept;
                    answered.moved
                }),
            // Measured from the press, and the preview is rebuilt from the pre-drag producer each
            // frame, so the displacement is applied to where the point STOOD rather than summed
            // frame over frame — and it is measured from the PRESS rather than from where the
            // threshold was crossed, so the geometry sits under the cursor instead of trailing it
            // by the threshold.
            SketchGrab::TranslateLever {
                fit,
                from: Some(from),
            } if began => {
                let (from, now) = (from.in_plane(), snapped.in_plane());
                let stood = preview
                    .sketch
                    .points()
                    .iter()
                    .find(|point| point.id == fit)
                    .map(|point| point.at.in_plane());
                match stood {
                    Some(stood) => preview.sketch.move_point(
                        fit,
                        document::sketch::SketchPoint::from_continuous(
                            stood[0] + now[0] - from[0],
                            stood[1] + now[1] - from[1],
                        ),
                        context,
                    ),
                    None => Ok(false),
                }
            }
            // The UNSNAPPED coordinate: a label is not geometry, and rounding it onto the voxel
            // lattice would make the number jump between grid cells while the hand moved smoothly.
            SketchGrab::Annotation { constraint } if began => {
                Ok(preview.sketch.move_annotation(constraint, profile_coord))
            }
            // Absolute, not summed: the place the author pressed goes where the cursor is now.
            SketchGrab::Translate {
                curve,
                from: Some(from),
            } if began => preview.sketch.drag_curve_through(
                curve,
                from.in_plane(),
                snapped.in_plane(),
                context,
            ),
            // The press could not read a profile coordinate, so the first frame that can records
            // where the gesture started and moves nothing.
            SketchGrab::TranslateLever { fit, from: None } => {
                if let Some(drag) = self.sketch_drag.as_mut() {
                    drag.began = began;
                    drag.held = SketchGrab::TranslateLever {
                        fit,
                        from: Some(snapped),
                    };
                }
                return IntentEffect::none();
            }
            SketchGrab::Translate { curve, from: None } => {
                if let Some(drag) = self.sketch_drag.as_mut() {
                    drag.began = began;
                    drag.held = SketchGrab::Translate {
                        curve,
                        from: Some(snapped),
                    };
                }
                return IntentEffect::none();
            }
            // Still a click so far. Nothing is touched, and nothing is torn down either: the very
            // next frame past the threshold picks the gesture up where the press left it.
            _ => return IntentEffect::none(),
        };
        if let Some(drag) = self.sketch_drag.as_mut() {
            drag.began = true;
            drag.arc_turns = carried_arcs;
        }
        let Ok(moved) = moved else {
            self.end_the_vertex_drag();
            return IntentEffect::none();
        };
        // A frame the drawing would not stand under is DROPPED, not the end of the gesture. The
        // hand crossing an arc's far end lands on a frame whose two ends are one dot, which is no
        // arc and is written nowhere — and it is the single frame the author is most committed to
        // the wind. Ending there would strand the gesture mid-crossing. Nothing was written, so
        // the next frame picks up from exactly where this one started.
        if !moved {
            return IntentEffect::none();
        }
        let Some(new_min) = self.profile_bbox_min(&preview) else {
            self.end_the_vertex_drag();
            return IntentEffect::none();
        };
        let [in0, in1] = preview.sketch.plane.in_plane_axes();
        let mut new_offset = original_offset;
        new_offset[in0] += new_min[0] - original_min[0];
        new_offset[in1] += new_min[1] - original_min[1];

        // Skip a redundant re-resolve when the node already shows exactly this (a stationary
        // cursor still inside the same voxel).
        if self.sketch_node_matches(target, &preview, new_offset) {
            return IntentEffect::none();
        }
        self.set_sketch_node(target, preview, new_offset);
        IntentEffect::scene()
    }

    /// Whether the cursor has left the press by the general drag threshold — the one question that
    /// separates a click from a drag, asked the same way the view cube and the marquee ask it.
    fn pointer_left_the_press(&self) -> bool {
        pointer_left_the_press(self.press_position, self.last_cursor_position)
    }

    /// End the vertex drag, and with it the snap ghost — the circle means "your hand is sliding
    /// along this", which stops being true the moment the hand lets go.
    fn end_the_vertex_drag(&mut self) {
        self.sketch_drag = None;
        self.forget_the_snap_ghost();
    }

    /// Drop the snap circle. A gesture ending and a gesture starting both owe this: the circle
    /// says "your hand is sliding along this", which is true of exactly one live hand.
    pub(super) fn forget_the_snap_ghost(&mut self) {
        self.sketch_snap_ghost = None;
    }

    /// Commit an in-progress vertex drag — called SYNCHRONOUSLY from the `events` release handler
    /// (not deferred to a render flag: a deferred commit left a window where a second press could
    /// orphan the un-recorded preview). Reads the final previewed producer + offset off the node,
    /// restores the pre-drag state, then queues the final state as intents so the next `render`
    /// applies them through `apply_intent` and they record in the open group — ONE `SetSketch`,
    /// plus a `SetOffset` only when the anchor compensation actually moved the node. A gesture
    /// that ended where it began records nothing (the restored original is left in place).
    pub(super) fn commit_sketch_vertex_drag(&mut self) {
        self.forget_the_snap_ghost();
        let Some(drag) = self.sketch_drag.take() else {
            return;
        };
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        let Some((final_producer, final_offset)) = self.sketch_node_state(target) else {
            return;
        };
        // Restore the pre-drag state so `record()` captures original → final for the inverse.
        self.set_sketch_node(target, drag.original.clone(), drag.original_offset);

        if final_producer == drag.original && final_offset == drag.original_offset {
            return; // nothing moved — leave the restored original in place
        }

        // Queue the final state through the intent door (drained + applied by the next render's
        // loop, the same door as any placement drop) so it lands in the open group. ONE
        // transaction, so the drag is ONE in-mode undo step even when the anchor compensation
        // moved the node too — the `SetOffset` is emitted only when it actually did.
        let mut transaction = vec![crate::Intent::SetSketch {
            target,
            producer: final_producer,
        }];
        if final_offset != drag.original_offset {
            transaction.push(crate::Intent::SetOffset {
                target,
                offset_measurements: [
                    parametric::units::Measurement::from_voxels(final_offset[0]),
                    parametric::units::Measurement::from_voxels(final_offset[1]),
                    parametric::units::Measurement::from_voxels(final_offset[2]),
                ],
            });
        }
        self.viewport_transactions.push(transaction);
    }

    /// Whether the sketch node `target` currently holds exactly `producer` + `offset_voxels` —
    /// the no-op check the preview uses to skip a redundant re-resolve, comparing by reference
    /// (no clone).
    fn sketch_node_matches(
        &self,
        target: document::scene::NodeId,
        producer: &document::sketch::SketchSolid,
        offset_voxels: [i64; 3],
    ) -> bool {
        let Some(node) = self.panel_state.scene.node_by_id(target) else {
            return false;
        };
        let document::scene::NodeContent::SketchTool {
            producer: current, ..
        } = &node.content
        else {
            return false;
        };
        current == producer && node.transform.offset_voxels == offset_voxels
    }

    /// The sketch node `target`'s current producer + world voxel offset, or `None` if it is not
    /// an enabled sketch node — the final previewed state the commit captures.
    fn sketch_node_state(
        &self,
        target: document::scene::NodeId,
    ) -> Option<(document::sketch::SketchSolid, [i64; 3])> {
        let node = self.panel_state.scene.node_by_id(target)?;
        let document::scene::NodeContent::SketchTool { producer, .. } = &node.content else {
            return None;
        };
        Some((producer.clone(), node.transform.offset_voxels))
    }

    /// Every construction run a spline in `target` draws, as the spline's aggregate identity and
    /// the points the run passes through: the control frame, and each tangent handle's line.
    ///
    /// Ids only, and deliberately not the producer: this runs every frame, and
    /// [`sketch_node_state`](Self::sketch_node_state) clones the whole solid to hand one back.
    fn control_polygons(
        &self,
        target: document::scene::NodeId,
    ) -> Vec<(
        document::sketch::SketchCurve,
        Vec<document::sketch::EntityId>,
    )> {
        let Some(node) = self.panel_state.scene.node_by_id(target) else {
            return Vec::new();
        };
        let document::scene::NodeContent::SketchTool { producer, .. } = &node.content else {
            return Vec::new();
        };
        producer
            .sketch
            .control_polygons()
            .into_iter()
            .map(|(spline, controls)| (document::sketch::SketchCurve::Spline(spline), controls))
            .collect()
    }

    /// Every fit point's tangent lever as ids: the FIT POINT whose handle it is, and the run
    /// back-arm → fit point → forward-arm. Ids only, for the same reason
    /// [`control_polygons`](Self::control_polygons) is.
    ///
    /// The spline is dropped on the way through. A control frame's leg has no meaning but the
    /// spline it steers; a lever's does. It belongs to ONE fit point, and answering as that point
    /// is what keeps a hover from lighting every lever on the curve.
    ///
    /// Only the levers the author has ASKED FOR are here, and that is the one seam the rule needs:
    /// hover, the grab hit-test and the paint all read this cache, so a lever that is not in it is
    /// invisible and ungrabbable at once, with no second rule to keep in step.
    fn tangent_levers(
        &self,
        target: document::scene::NodeId,
    ) -> Vec<(document::sketch::EntityId, Vec<document::sketch::EntityId>)> {
        let Some(node) = self.panel_state.scene.node_by_id(target) else {
            return Vec::new();
        };
        let document::scene::NodeContent::SketchTool { producer, .. } = &node.content else {
            return Vec::new();
        };
        let asked_for = self.levers_the_author_asked_for(target);
        producer
            .sketch
            .tangent_handle_legs()
            .into_iter()
            .map(|(_, lever)| (lever[1], lever.to_vec()))
            .filter(|(fit, _)| asked_for.contains(fit))
            .collect()
    }

    /// The fit points whose tangent lever should be out.
    ///
    /// A lever is a manipulator, not part of the drawing, so it is not furniture that is always on
    /// display: the author asks for it by SELECTING the point it steers or the spline that point
    /// belongs to (owner, 2026-08-04). A lever mid-drag counts as asked for whatever the selection
    /// says, so a gesture never pulls its own handle out from under itself.
    fn levers_the_author_asked_for(
        &self,
        target: document::scene::NodeId,
    ) -> std::collections::BTreeSet<document::sketch::EntityId> {
        let Some(node) = self.panel_state.scene.node_by_id(target) else {
            return std::collections::BTreeSet::new();
        };
        let document::scene::NodeContent::SketchTool { producer, .. } = &node.content else {
            return std::collections::BTreeSet::new();
        };
        let sketch = &producer.sketch;
        // An arm answers for the point it steers, so grabbing or picking one keeps the stick out.
        let as_fit = |id| sketch.tangent_arm_owner(id).unwrap_or(id);
        let held = match self.sketch_drag.as_ref().map(|drag| drag.held) {
            Some(SketchGrab::Point(id)) => Some(as_fit(id)),
            Some(SketchGrab::TranslateLever { fit, .. }) => Some(fit),
            _ => None,
        };
        sketch
            .splines()
            .iter()
            .flat_map(|spline| {
                let curve = document::sketch::SketchCurve::Spline(spline.id);
                let whole = self.panel_state.selection.contains(
                    ui::panel::SelectionTarget::SketchHigherCurve {
                        sketch: target,
                        curve,
                    },
                );
                spline.tangents.keys().copied().filter(move |fit| {
                    whole
                        || held == Some(*fit)
                        || self.panel_state.selection.contains(
                            ui::panel::SelectionTarget::SketchPoint {
                                sketch: target,
                                entity: *fit,
                            },
                        )
                })
            })
            .collect()
    }

    /// Every conic's shoulder in this sketch, in profile space. See
    /// [`conic_shoulders`](document::sketch::Sketch::conic_shoulders) for why it is a reading.
    fn conic_shoulders_in_profile(
        &self,
        target: document::scene::NodeId,
    ) -> Vec<(document::sketch::EntityId, [f64; 2])> {
        let Some(node) = self.panel_state.scene.node_by_id(target) else {
            return Vec::new();
        };
        let document::scene::NodeContent::SketchTool { producer, .. } = &node.content else {
            return Vec::new();
        };
        producer.sketch.conic_shoulders()
    }

    /// The points the drawing already accounts for, and so may draw with nothing hovered.
    ///
    /// [`point_draws_at_rest`](document::sketch::Sketch::point_draws_at_rest) is the rule; this
    /// gathers it, and joins the points of every SELECTED curve — selecting a line is a way of
    /// saying "this one", and the corners it runs between are part of the answer.
    ///
    /// The rule holds under EVERY tool. Arming one used to show every point in the drawing, on the
    /// reasoning that a tool reaching for a point must be able to see it — but the rest-rule is
    /// already about which dots the drawing owes the author, and a tool does not change that
    /// answer (owner, 2026-08-04). Hovering still reveals, which is what a tool actually reaches
    /// through.
    fn points_the_drawing_shows_by_itself(
        &self,
        target: document::scene::NodeId,
    ) -> std::collections::BTreeSet<document::sketch::EntityId> {
        let Some(node) = self.panel_state.scene.node_by_id(target) else {
            return std::collections::BTreeSet::new();
        };
        let document::scene::NodeContent::SketchTool { producer, .. } = &node.content else {
            return std::collections::BTreeSet::new();
        };
        let sketch = &producer.sketch;
        let mut shown: std::collections::BTreeSet<document::sketch::EntityId> = sketch
            .points()
            .iter()
            .filter(|point| sketch.point_draws_at_rest(point.id))
            .map(|point| point.id)
            .collect();
        for curve in self.selected_sketch_curves(target) {
            shown.extend(sketch.points_of(curve));
        }
        shown
    }

    /// Every point of this sketch that another dot already stands on, and so never draws.
    ///
    /// [`a_better_dot_stands_here`](document::sketch::Sketch::a_better_dot_stands_here) is the
    /// rule; this gathers it once per frame rather than per revealed dot.
    fn dots_standing_under_another(
        &self,
        target: document::scene::NodeId,
    ) -> std::collections::BTreeSet<document::sketch::EntityId> {
        let Some(node) = self.panel_state.scene.node_by_id(target) else {
            return std::collections::BTreeSet::new();
        };
        let document::scene::NodeContent::SketchTool { producer, .. } = &node.content else {
            return std::collections::BTreeSet::new();
        };
        producer
            .sketch
            .points()
            .iter()
            .filter(|point| producer.sketch.a_better_dot_stands_here(point.id))
            .map(|point| point.id)
            .collect()
    }

    /// The points the curve behind an edge hit stands on.
    fn points_of_edge_hit(
        &self,
        target: document::scene::NodeId,
        hit: SketchEdgeHit,
    ) -> Vec<document::sketch::EntityId> {
        let curve = match hit {
            SketchEdgeHit::Segment(id) => document::sketch::SketchCurve::Segment(id),
            SketchEdgeHit::Arc(id) => document::sketch::SketchCurve::Arc(id),
            SketchEdgeHit::Circle(id) => document::sketch::SketchCurve::Circle(id),
            SketchEdgeHit::HigherCurve(curve) => curve,
        };
        let Some(node) = self.panel_state.scene.node_by_id(target) else {
            return Vec::new();
        };
        let document::scene::NodeContent::SketchTool { producer, .. } = &node.content else {
            return Vec::new();
        };
        producer.sketch.points_of(curve)
    }

    /// Every curve of `target` the selection holds.
    fn selected_sketch_curves(
        &self,
        target: document::scene::NodeId,
    ) -> Vec<document::sketch::SketchCurve> {
        self.panel_state
            .selection
            .targets()
            .filter_map(|picked| match picked {
                ui::panel::SelectionTarget::SketchSegment { sketch, entity }
                    if sketch == target =>
                {
                    Some(document::sketch::SketchCurve::Segment(entity))
                }
                ui::panel::SelectionTarget::SketchArc { sketch, entity } if sketch == target => {
                    Some(document::sketch::SketchCurve::Arc(entity))
                }
                ui::panel::SelectionTarget::SketchCircle { sketch, entity } if sketch == target => {
                    Some(document::sketch::SketchCurve::Circle(entity))
                }
                ui::panel::SelectionTarget::SketchHigherCurve { sketch, curve }
                    if sketch == target =>
                {
                    Some(curve)
                }
                _ => None,
            })
            .collect()
    }

    /// Every point some curve is drawn THROUGH, gathered once for the frame rather than asked per
    /// dot — the predicate walks every curve store, and the overlay asks it of every vertex.
    fn points_standing_on_ink(
        &self,
        target: document::scene::NodeId,
    ) -> std::collections::BTreeSet<document::sketch::EntityId> {
        let Some(node) = self.panel_state.scene.node_by_id(target) else {
            return std::collections::BTreeSet::new();
        };
        let document::scene::NodeContent::SketchTool { producer, .. } = &node.content else {
            return std::collections::BTreeSet::new();
        };
        producer
            .sketch
            .points()
            .iter()
            .filter(|point| producer.sketch.point_stands_on_ink(point.id))
            .map(|point| point.id)
            .collect()
    }

    /// The two arms of the lever standing at `fit`, or nothing if it has no handle.
    fn tangent_arms_of(
        &self,
        target: document::scene::NodeId,
        fit: document::sketch::EntityId,
    ) -> Vec<document::sketch::EntityId> {
        let Some(node) = self.panel_state.scene.node_by_id(target) else {
            return Vec::new();
        };
        let document::scene::NodeContent::SketchTool { producer, .. } = &node.content else {
            return Vec::new();
        };
        producer
            .sketch
            .splines()
            .iter()
            .filter_map(|spline| spline.tangents.get(&fit))
            .flat_map(|handle| handle.arms())
            .collect()
    }

    /// Every point that is one end of a tangent lever, for the frame that has to paint them green.
    fn tangent_arm_points(
        &self,
        target: document::scene::NodeId,
    ) -> std::collections::BTreeSet<document::sketch::EntityId> {
        let Some(node) = self.panel_state.scene.node_by_id(target) else {
            return std::collections::BTreeSet::new();
        };
        let document::scene::NodeContent::SketchTool { producer, .. } = &node.content else {
            return std::collections::BTreeSet::new();
        };
        producer
            .sketch
            .splines()
            .iter()
            .flat_map(|spline| spline.tangents.values().flat_map(|handle| handle.arms()))
            .collect()
    }

    /// Direct-mutate the sketch node `target`'s producer + world voxel offset — the transient
    /// live-drag preview / restore. Always reconciled through `apply_intent` on release, so the
    /// command stack stays the single source of truth for undo.
    fn set_sketch_node(
        &mut self,
        target: document::scene::NodeId,
        producer: document::sketch::SketchSolid,
        offset_voxels: [i64; 3],
    ) {
        if let Some(node) = self.panel_state.scene.node_by_id_mut(target) {
            if let document::scene::NodeContent::SketchTool { producer: slot, .. } =
                &mut node.content
            {
                *slot = producer;
            }
            node.transform.offset_voxels = offset_voxels;
        }
    }

    /// Drop the pending picks of every multi-click drawing gesture.
    ///
    /// One roster, called everywhere a half-drawn gesture must not survive: entering and leaving
    /// the mode, Escape, undo and redo, the tool disarming, and the sketch or its handles going
    /// away. The list was hand-copied at eight sites and rotted — `higher_curve_gesture` was
    /// absent from undo and redo, so a pending spline point outlived Ctrl+Z.
    fn reset_sketch_gestures(&mut self) {
        self.line_gesture.reset();
        self.midpoint_line_gesture.reset();
        self.tangent_arc_gesture.reset();
        self.center_arc_gesture.reset();
        self.point_circle_gesture.reset();
        self.higher_curve_gesture.reset();
        self.three_point_rectangle_gesture.reset();
        self.corner_rectangle_gesture.reset();
        self.polygon_gesture.reset();
        self.slot_gesture.reset();
        self.tangent_circle_gesture.reset();
    }

    /// The in-plane bbox-minimum (per profile coordinate) of a sketch producer's profile — the
    /// anchor the drag compensation measures its bbox-min shift against.
    fn sketch_evaluation_context(&self) -> Option<parametric::EvaluationContext> {
        document::sketch::evaluation_context_from_density(
            self.panel_state.geometry.voxels_per_block,
        )
    }

    fn profile_bbox_min(&self, producer: &document::sketch::SketchSolid) -> Option<[i64; 2]> {
        self.sketch_evaluation_context()
            .map(|context| producer.profile_bbox_min(context))
    }

    /// Cursor (physical px) → the CONTINUOUS profile coordinate `(c0, c1)` under it on the
    /// sketch node's plane, using `handles` for the plane + inverse map. Shared by
    /// the vertex-drag preview (#94) and the add-point insert (#95) so the frame math lives once.
    ///
    /// Casts from the EYE under perspective — the near-plane ray origin is unreliable at close
    /// zoom and can sit past the target plane (placement casts from the eye for the same reason);
    /// orthographic keeps the near-plane point (parallel rays have no single eye). `None` when the
    /// unprojection fails, the ray is parallel to the plane, or the plane is behind the viewer.
    ///
    /// `ray_unprojection` is the RAY-FRAME matrix (`SceneMatrices::ray_unprojection`), not the full
    /// scene VP: under perspective the full inverse melts the `/w` divide at a wide-baseline
    /// recenter (a06d215), so we unproject the DIRECTION through the camera-relative bracket and
    /// take the origin from the eye. Ortho keeps the plain frame (`ray_unprojection == view_projection`).
    fn cursor_to_profile_coord(
        &self,
        cursor_x: f64,
        cursor_y: f64,
        ray_unprojection: glam::Mat4,
        viewport_px: [u32; 4],
        handles: &document::scene::SketchHandles,
    ) -> Option<[f64; 2]> {
        let [vx, vy, vw, vh] = viewport_px;
        let ndc_x = (cursor_x as f32 - vx as f32) / vw.max(1) as f32 * 2.0 - 1.0;
        let ndc_y = 1.0 - (cursor_y as f32 - vy as f32) / vh.max(1) as f32 * 2.0;
        let ray = camera::unproject_screen_point_to_ray(ray_unprojection, ndc_x, ndc_y)?;
        let ray_origin = match self.app_core.camera.projection_mode {
            camera::ProjectionMode::Perspective => self.app_core.camera.eye(),
            camera::ProjectionMode::Orthographic => ray.origin,
        };
        let normal = glam::Vec3::from_array(handles.plane_normal);
        let plane_point = glam::Vec3::from_array(handles.plane_point);
        let denom = ray.direction.dot(normal);
        if denom.abs() < 1e-6 {
            return None;
        }
        let t = (plane_point - ray_origin).dot(normal) / denom;
        if t <= 0.0 {
            return None;
        }
        let hit = ray_origin + ray.direction * t;
        Some(handles.render_hit_to_profile(hit.to_array()))
    }

    /// The profile-vertex index under the cursor (physical px), the nearest within the handle
    /// grab radius, or `None`. Reads the profile-order [`sketch_vertex_px`](Self::sketch_vertex_px)
    /// cache, so it shares the exact projection the overlay drew. Used by the vertex-drag grab
    /// (#94) and the selection click resolve.
    ///
    /// A DERIVED point loses a near-tie. Some shapes deliberately stack an authored handle on top
    /// of a point the drawing derives — a slot's center is a real, draggable point pinned to the
    /// center its rails turn about — and the two then project to the same pixel, so distance alone
    /// decides the grab by whatever order the store happens to be in. Landing on the derived twin
    /// is the dead half of that coin: dragging it authors a radius rather than moving the shape.
    /// The bias is under a pixel, so it can only ever settle a tie the author could not see anyway.
    /// The GEOMETRY point under the cursor: like
    /// [`sketch_vertex_at`](Self::sketch_vertex_at), but blind to a tangent lever's arms.
    ///
    /// An arm is a manipulator, not a place in the drawing. A tool that seams onto an existing
    /// point — a line chain, an arc, a pattern center — must not seam onto one, or the author
    /// draws geometry anchored to a thing whose whole job is to move when they steer the curve.
    /// The DRAG path deliberately does not go through here: grabbing an arm is the point of it.
    fn sketch_geometry_point_at(
        &self,
        target: document::scene::NodeId,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<document::sketch::EntityId> {
        let point = self
            .sketch_vertex_at(cursor_x, cursor_y)
            .and_then(|index| self.sketch_point_ids.get(index).copied())?;
        (!self.tangent_arm_points(target).contains(&point)).then_some(point)
    }

    /// The lever arms that are not out this frame.
    ///
    /// A hidden point stays hit-testable everywhere else — that is how hovering brings a quiet
    /// corner back — but an arm is the exception, because its lever is what makes it legible and
    /// the lever is not there. Reaching into empty space and finding a tangent by accident is not
    /// a reveal, it is a trap.
    fn hidden_tangent_arms(&self) -> std::collections::BTreeSet<document::sketch::EntityId> {
        let Some(target) = self.panel_state.sketch_mode else {
            return std::collections::BTreeSet::new();
        };
        let out: std::collections::BTreeSet<_> = self
            .sketch_tangent_levers
            .iter()
            .flat_map(|(fit, _)| self.tangent_arms_of(target, *fit))
            .collect();
        self.tangent_arm_points(target)
            .difference(&out)
            .copied()
            .collect()
    }

    fn sketch_vertex_at(&self, cursor_x: f64, cursor_y: f64) -> Option<usize> {
        let scale = self.window.scale_factor() as f32;
        let grab_px = (ui::chrome::SKETCH_HANDLE_HALF + ui::chrome::SKETCH_HANDLE_GRAB_PAD) * scale;
        let stacked_px = SKETCH_STACKED_HANDLE_BIAS * scale;
        let hidden_arms = self.hidden_tangent_arms();
        let mut nearest: Option<(usize, f32)> = None;
        for (index, center) in self.sketch_vertex_px.iter().enumerate() {
            let Some(center) = center else { continue };
            if self
                .sketch_point_ids
                .get(index)
                .is_some_and(|id| hidden_arms.contains(id))
            {
                continue;
            }
            let distance = (cursor_x as f32 - center.x).hypot(cursor_y as f32 - center.y);
            if distance > grab_px {
                continue;
            }
            let ranked = if self
                .sketch_point_derived
                .get(index)
                .copied()
                .unwrap_or(false)
            {
                distance + stacked_px
            } else {
                distance
            };
            if nearest.map(|(_, best)| ranked < best).unwrap_or(true) {
                nearest = Some((index, ranked));
            }
        }
        nearest.map(|(index, _)| index)
    }

    /// The sketch SEGMENT under the cursor (physical px) as `(segment id, endpoint a px,
    /// endpoint b px)`, the nearest within the grab pad — iterated over the actual segment
    /// ENTITIES, not consecutive vertices, so it is correct for an open or
    /// multi-loop graph. `None` when no edge is close enough or an endpoint is culled.
    pub(super) fn nearest_sketch_segment(
        &self,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<(document::sketch::EntityId, egui::Pos2, egui::Pos2)> {
        let pad_px = ui::chrome::SKETCH_SEGMENT_GRAB_PAD * self.window.scale_factor() as f32;
        let cursor = egui::Pos2::new(cursor_x as f32, cursor_y as f32);
        let mut nearest: Option<(document::sketch::EntityId, egui::Pos2, egui::Pos2, f32)> = None;
        for segment in &self.sketch_segments {
            let (Some(&Some(a)), Some(&Some(b))) = (
                self.sketch_vertex_px.get(segment.from),
                self.sketch_vertex_px.get(segment.to),
            ) else {
                continue;
            };
            let distance = point_to_segment_distance(cursor, a, b);
            if distance <= pad_px
                && nearest
                    .map(|(_, _, _, best)| distance < best)
                    .unwrap_or(true)
            {
                nearest = Some((segment.entity, a, b, distance));
            }
        }
        nearest.map(|(seg_id, a, b, _)| (seg_id, a, b))
    }

    /// The sketch ARC under the cursor (physical px) as `(arc id, distance)`, measured against
    /// the arc's drawn chord polyline so the pick follows the curve rather than its chord
    /// (#102). `None` when no arc is within the grab pad.
    fn nearest_sketch_arc(
        &self,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<(document::sketch::EntityId, f32)> {
        let pad_px = ui::chrome::SKETCH_SEGMENT_GRAB_PAD * self.window.scale_factor() as f32;
        let cursor = egui::Pos2::new(cursor_x as f32, cursor_y as f32);
        let mut nearest: Option<(document::sketch::EntityId, f32)> = None;
        for (arc_id, chords) in &self.sketch_arc_chords {
            let Some(distance) = chords
                .array_windows::<2>()
                .map(|pair| point_to_segment_distance(cursor, pair[0], pair[1]))
                .min_by(|a, b| a.total_cmp(b))
            else {
                continue;
            };
            if distance <= pad_px && nearest.map(|(_, best)| distance < best).unwrap_or(true) {
                nearest = Some((*arc_id, distance));
            }
        }
        nearest
    }

    /// The sketch CIRCLE under the cursor (physical px), measured against its projected ring.
    fn nearest_sketch_circle(
        &self,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<(document::sketch::EntityId, f32)> {
        let pad_px = ui::chrome::SKETCH_SEGMENT_GRAB_PAD * self.window.scale_factor() as f32;
        let cursor = egui::Pos2::new(cursor_x as f32, cursor_y as f32);
        let mut nearest: Option<(document::sketch::EntityId, f32)> = None;
        for (circle_id, ring) in &self.sketch_circle_chords {
            let Some(distance) = ring
                .array_windows::<2>()
                .map(|pair| point_to_segment_distance(cursor, pair[0], pair[1]))
                .min_by(|a, b| a.total_cmp(b))
            else {
                continue;
            };
            if distance <= pad_px && nearest.map(|(_, best)| distance < best).unwrap_or(true) {
                nearest = Some((*circle_id, distance));
            }
        }
        nearest
    }

    /// The higher-order AGGREGATE under the cursor (physical px), measured against every span it
    /// draws.
    ///
    /// The spans compete as one candidate rather than several: an ellipse is four rational
    /// quarters on screen and one object to the author, so the distance that answers is the
    /// minimum over all of them and the identity that answers is the aggregate's. This is the
    /// hit-test half of the rule the selection states — every span of an ellipse or spline
    /// selects the same object.
    fn nearest_sketch_higher_curve(
        &self,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<(document::sketch::SketchCurve, f32)> {
        let pad_px = ui::chrome::SKETCH_SEGMENT_GRAB_PAD * self.window.scale_factor() as f32;
        let cursor = egui::Pos2::new(cursor_x as f32, cursor_y as f32);
        let mut nearest: Option<(document::sketch::SketchCurve, f32)> = None;
        // A control frame's legs compete as spans of the spline they steer. They are drawn, not
        // stored, so there is no entity for a leg hit to mean — and the spline is what the author
        // is reaching for when they grab one.
        //
        // A TANGENT lever's leg is NOT here, and that is the difference between the two kinds of
        // leg: a lever belongs to one fit point, so its hit has somewhere better to land than the
        // whole curve. See [`tangent_lever_at`](Self::tangent_lever_at).
        for (curve, chords) in self
            .sketch_higher_curve_chords
            .iter()
            .chain(&self.sketch_spline_polygons)
        {
            let Some(distance) = chords
                .array_windows::<2>()
                .map(|pair| point_to_segment_distance(cursor, pair[0], pair[1]))
                .min_by(|a, b| a.total_cmp(b))
            else {
                continue;
            };
            if distance <= pad_px && nearest.map(|(_, best)| distance < best).unwrap_or(true) {
                nearest = Some((*curve, distance));
            }
        }
        nearest
    }

    /// The FIT POINT whose tangent lever is under the cursor, if one is.
    ///
    /// A lever is that point's manipulator, drawn through it and out to both arms, so every way of
    /// reaching for it resolves to the same place: hover lights that lever alone, a click selects
    /// the point, and a drag moves the point — carrying the handle at the angle and length the
    /// author left it, which is what dragging the point has always done.
    ///
    /// The arms are excluded by ordering rather than by a filter: every caller asks for a VERTEX
    /// first, and an arm is a vertex. So grabbing the dot at the end of a lever still steers the
    /// tangent, and only the stick between the dots answers here.
    fn tangent_lever_at(&self, cursor_x: f64, cursor_y: f64) -> Option<document::sketch::EntityId> {
        nearest_tangent_lever(
            &self.sketch_tangent_levers,
            egui::Pos2::new(cursor_x as f32, cursor_y as f32),
            ui::chrome::SKETCH_SEGMENT_GRAB_PAD * self.window.scale_factor() as f32,
        )
    }

    /// The sketch EDGE under the cursor — the nearest segment or curve. One resolution so hover
    /// feedback and the click that follows it can never disagree about which edge the cursor is
    /// on.
    pub(super) fn nearest_sketch_edge(
        &self,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<SketchEdgeHit> {
        let cursor = egui::Pos2::new(cursor_x as f32, cursor_y as f32);
        let segment = self
            .nearest_sketch_segment(cursor_x, cursor_y)
            .map(|(id, a, b)| (id, point_to_segment_distance(cursor, a, b)));
        let arc = self
            .nearest_sketch_arc(cursor_x, cursor_y)
            .map(|(id, distance)| (SketchEdgeHit::Arc(id), distance));
        let circle = self
            .nearest_sketch_circle(cursor_x, cursor_y)
            .map(|(id, distance)| (SketchEdgeHit::Circle(id), distance));
        let higher = self
            .nearest_sketch_higher_curve(cursor_x, cursor_y)
            .map(|(curve, distance)| (SketchEdgeHit::HigherCurve(curve), distance));
        nearest_sketch_edge_from_candidates([
            segment.map(|(id, distance)| (SketchEdgeHit::Segment(id), distance)),
            arc,
            circle,
            higher,
        ])
    }

    /// The nearest sketch edge with an addressable endpoint. Tangent Arc begins at a seam, so a
    /// closed circle must not mask a line or arc that is equally close to the cursor.
    fn nearest_open_sketch_edge(&self, cursor_x: f64, cursor_y: f64) -> Option<SketchEdgeHit> {
        let cursor = egui::Pos2::new(cursor_x as f32, cursor_y as f32);
        let segment = self
            .nearest_sketch_segment(cursor_x, cursor_y)
            .map(|(id, a, b)| {
                (
                    SketchEdgeHit::Segment(id),
                    point_to_segment_distance(cursor, a, b),
                )
            });
        let arc = self
            .nearest_sketch_arc(cursor_x, cursor_y)
            .map(|(id, distance)| (SketchEdgeHit::Arc(id), distance));
        nearest_sketch_edge_from_candidates([segment, arc])
    }

    /// The id of the sketch SEGMENT under the cursor (physical px), for add-point — the click
    /// splits the named segment. `None` when no edge is close enough.
    fn sketch_segment_at(
        &self,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<document::sketch::EntityId> {
        self.nearest_sketch_segment(cursor_x, cursor_y)
            .map(|(seg_id, _, _)| seg_id)
    }

    /// The add-point producer for a click at the cursor (physical px) — the
    /// current sketch with a new grid-snapped vertex inserted into the segment under the cursor,
    /// splitting that edge. `None` when no segment is under the cursor, the cursor cannot be
    /// projected onto the plane, or `target` is not an enabled sketch node. The caller routes the
    /// returned producer through [`commit_sketch_profile_edit`](Self::commit_sketch_profile_edit).
    pub(super) fn sketch_insert_at(
        &self,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<document::sketch::SketchSolid> {
        let target = self.panel_state.sketch_mode?;
        let seg_id = self.sketch_segment_at(cursor_x, cursor_y)?;
        let handles = self
            .panel_state
            .scene
            .sketch_handles(target, self.panel_state.geometry.voxels_per_block)?;
        let coord = self.cursor_to_profile_coord(
            cursor_x,
            cursor_y,
            self.last_ray_unprojection?,
            self.last_viewport_px,
            &handles,
        )?;
        let (producer, _) = self.sketch_node_state(target)?;
        // Split the segment under the cursor with a policy-snapped point.
        let point = apply_sketch_snap(
            coord,
            self.panel_state.sketch_snap,
            self.panel_state.geometry.voxels_per_block,
        );
        Some(producer.with_point_on_segment(seg_id, point))
    }

    /// Break the native curve under the cursor at all of its intersections with the rest of the
    /// sketch. A refusal is a no-op, so a miss or endpoint-only contact creates no undo entry.
    pub(super) fn sketch_break_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        let Some(hit) = self.nearest_sketch_edge(cursor_x, cursor_y) else {
            return;
        };
        let Some(context) = document::sketch::evaluation_context_from_density(
            self.panel_state.geometry.voxels_per_block,
        ) else {
            return;
        };
        if let Ok(next) = producer.with_curve_broken(sketch_curve_from_hit(hit), context) {
            self.commit_sketch_profile_edit(target, next);
        }
    }

    /// Trim the native curve interval nearest the unsnapped cursor witness. With no intersection
    /// the document operation deletes the whole curve, matching the standard Trim contract.
    pub(super) fn sketch_trim_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        let (Some(hit), Some(witness), Some(context)) = (
            self.nearest_sketch_edge(cursor_x, cursor_y),
            self.sketch_unsnapped_profile_coord(cursor_x, cursor_y),
            document::sketch::evaluation_context_from_density(
                self.panel_state.geometry.voxels_per_block,
            ),
        ) else {
            return;
        };
        if let Ok(next) = producer.with_curve_trimmed(sketch_curve_from_hit(hit), witness, context)
        {
            self.commit_sketch_profile_edit(target, next);
        }
    }

    /// Extend the native open curve under the cursor from its nearer endpoint to the next finite
    /// authored intersection. Placement and commit share the same unsnapped profile witness.
    pub(super) fn sketch_extend_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        let (Some(hit), Some(witness), Some(context)) = (
            self.nearest_sketch_edge(cursor_x, cursor_y),
            self.sketch_unsnapped_profile_coord(cursor_x, cursor_y),
            document::sketch::evaluation_context_from_density(
                self.panel_state.geometry.voxels_per_block,
            ),
        ) else {
            return;
        };
        if let Ok(next) = producer.with_curve_extended(sketch_curve_from_hit(hit), witness, context)
        {
            self.commit_sketch_profile_edit(target, next);
        }
    }

    /// Replace the two-line corner nearest the clicked leg endpoint by the exact native fillet
    /// previewed at the cursor-selected tangent distance.
    pub(super) fn sketch_fillet_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        let (Some((segment, _, _)), Some(witness), Some(context)) = (
            self.nearest_sketch_segment(cursor_x, cursor_y),
            self.sketch_unsnapped_profile_coord(cursor_x, cursor_y),
            document::sketch::evaluation_context_from_density(
                self.panel_state.geometry.voxels_per_block,
            ),
        ) else {
            return;
        };
        if let Ok(next) = producer.with_corner_filleted(
            document::sketch::SketchCurve::Segment(segment),
            witness,
            context,
        ) {
            self.commit_sketch_profile_edit(target, next);
        }
    }

    /// Advance one of the three Chamfer grammars. Equal Distance commits from one witness; the
    /// other two retain the first leg until the author clicks the adjacent leg, then commit the
    /// shared canonical two-distance placement atomically.
    pub(super) fn sketch_chamfer_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        let tool = self.panel_state.sketch_tool;
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        let (Some((segment, _, _)), Some(witness), Some(context)) = (
            self.nearest_sketch_segment(cursor_x, cursor_y),
            self.sketch_unsnapped_profile_coord(cursor_x, cursor_y),
            document::sketch::evaluation_context_from_density(
                self.panel_state.geometry.voxels_per_block,
            ),
        ) else {
            return;
        };
        if tool == ui::panel::SketchTool::ChamferEqual {
            if let Ok(next) = producer.with_corner_chamfered(
                document::sketch::SketchCurve::Segment(segment),
                witness,
                None,
                context,
            ) {
                self.commit_sketch_profile_edit(target, next);
            }
            return;
        }
        let Some(pending) = self.sketch_chamfer_pending else {
            if let Ok(placement) = producer.chamfer_placement(
                document::sketch::SketchCurve::Segment(segment),
                witness,
                None,
                context,
            ) {
                let document::sketch::SketchCurve::Segment(second) = placement.second else {
                    return;
                };
                self.sketch_chamfer_pending = Some(PendingChamfer {
                    target,
                    tool,
                    source: segment,
                    second,
                    first_witness: witness,
                });
            }
            return;
        };
        if pending.target != target || pending.tool != tool || pending.second != segment {
            return;
        }
        self.sketch_chamfer_pending = None;
        if let Ok(next) = producer.with_corner_chamfered(
            document::sketch::SketchCurve::Segment(pending.source),
            pending.first_witness,
            Some(witness),
            context,
        ) {
            self.commit_sketch_profile_edit(target, next);
        }
    }

    /// Offset is a two-click command: name one authored curve, then use any point on the sketch
    /// plane to choose the signed line distance or circular radius of its native copy.
    pub(super) fn sketch_offset_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        if let Some(pending) = self.sketch_offset_pending {
            if pending.target != target {
                self.sketch_offset_pending = None;
                return;
            }
            let (Some(witness), Some(context)) = (
                self.sketch_unsnapped_profile_coord(cursor_x, cursor_y),
                document::sketch::evaluation_context_from_density(
                    self.panel_state.geometry.voxels_per_block,
                ),
            ) else {
                return;
            };
            self.sketch_offset_pending = None;
            if let Ok(next) = producer.with_curve_offset(pending.source, witness, context) {
                self.commit_sketch_profile_edit(target, next);
            }
            return;
        }
        let Some(hit) = self.nearest_sketch_edge(cursor_x, cursor_y) else {
            return;
        };
        self.sketch_offset_pending = Some(PendingOffset {
            target,
            source: sketch_curve_from_hit(hit),
        });
    }

    /// Begin or finish a rigid translation of the current typed selection. Shift is sampled on
    /// the destination click so Copy is an explicit completion modifier, not latched accidentally
    /// when the base point was chosen.
    pub(super) fn sketch_move_copy_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        let Some(witness) = self.sketch_unsnapped_profile_coord(cursor_x, cursor_y) else {
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        if let Some(pending) = self.sketch_move_copy_pending.take() {
            if pending.target != target {
                return;
            }
            let delta = [
                witness[0] - pending.anchor[0],
                witness[1] - pending.anchor[1],
            ];
            if let Ok(next) =
                producer.with_entities_translated(&pending.entities, delta, self.shift_held)
            {
                self.commit_sketch_profile_edit(target, next);
            }
            return;
        }
        let entities = self.sketch_transform_selection(target);
        if !entities.is_empty() {
            self.sketch_move_copy_pending = Some(PendingMoveCopy {
                target,
                entities,
                anchor: witness,
            });
        }
    }

    /// Begin or finish a uniform Scale. The first click is the center; the selected geometry's
    /// original spatial radius becomes factor 1, and the second click chooses the new radius.
    pub(super) fn sketch_scale_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        let Some(witness) = self.sketch_unsnapped_profile_coord(cursor_x, cursor_y) else {
            return;
        };
        let (Some((producer, _)), Some(context)) = (
            self.sketch_node_state(target),
            document::sketch::evaluation_context_from_density(
                self.panel_state.geometry.voxels_per_block,
            ),
        ) else {
            return;
        };
        if let Some(pending) = self.sketch_scale_pending.take() {
            if pending.target != target {
                return;
            }
            let radius = (witness[0] - pending.center[0]).hypot(witness[1] - pending.center[1]);
            let factor = radius / pending.base_radius;
            if let Ok(next) =
                producer.with_entities_scaled(&pending.entities, pending.center, factor)
            {
                self.commit_sketch_profile_edit(target, next);
            }
            return;
        }
        let entities = self.sketch_transform_selection(target);
        if let Ok(base_radius) = producer.selection_scale_radius(&entities, witness, context) {
            self.sketch_scale_pending = Some(PendingScale {
                target,
                entities,
                center: witness,
                base_radius,
            });
        }
    }

    /// Mirror the current typed curve selection across the authored line under the cursor. The
    /// result is one persisted generator, so later source or axis edits regenerate the instance.
    pub(super) fn sketch_mirror_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        let Some((axis, _, _)) = self.nearest_sketch_segment(cursor_x, cursor_y) else {
            return;
        };
        let sources = self.sketch_curve_selection(target);
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        if let Ok(next) = producer.with_mirror_pattern(sources, axis) {
            self.commit_sketch_profile_edit(target, next);
        }
    }

    /// Advance rectangular-pattern input. The first click establishes a common anchor, the
    /// second defines X spacing, and a third defines Y spacing only when its count exceeds one.
    pub(super) fn sketch_rectangular_pattern_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        let Some(witness) = self.sketch_unsnapped_profile_coord(cursor_x, cursor_y) else {
            return;
        };
        let counts = self
            .panel_state
            .sketch_pattern_counts
            .map(|count| u32::from(count.clamp(1, 128)));
        let Some(mut pending) = self.sketch_rectangular_pattern_pending.take() else {
            let sources = self.sketch_curve_selection(target);
            if !sources.is_empty() {
                self.sketch_rectangular_pattern_pending = Some(PendingRectangularPattern {
                    target,
                    sources,
                    anchor: witness,
                    first_step: None,
                });
            }
            return;
        };
        if pending.target != target {
            return;
        }
        let step = [
            witness[0] - pending.anchor[0],
            witness[1] - pending.anchor[1],
        ];
        let steps = match pending.first_step {
            None if counts[1] > 1 => {
                pending.first_step = Some(step);
                self.sketch_rectangular_pattern_pending = Some(pending);
                return;
            }
            None => [step, [0.0, 0.0]],
            Some(first) => [first, step],
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        if let Ok(next) = producer.with_rectangular_pattern(
            pending.sources,
            counts,
            steps.map(|step| document::sketch::SketchVector::from_continuous(step[0], step[1])),
        ) {
            self.commit_sketch_profile_edit(target, next);
        }
    }

    /// Array selected curves through one full turn around the authored point under the cursor.
    pub(super) fn sketch_circular_pattern_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        let Some(center) = self.sketch_geometry_point_at(target, cursor_x, cursor_y) else {
            return;
        };
        let sources = self.sketch_curve_selection(target);
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        let Ok(full_turn) = parametric::units::AngleMeasurement::try_from_degrees_f64(360.0) else {
            return;
        };
        let count = u32::from(self.panel_state.sketch_circular_pattern_count.clamp(2, 128));
        if let Ok(next) = producer.with_circular_pattern(sources, center, count, full_turn) {
            self.commit_sketch_profile_edit(target, next);
        }
    }

    /// The selected authored curves in stable pick order. Points and constraints are deliberately
    /// ignored: operator instances are regenerated curves, while their source points remain owned
    /// by the selected curves and the solver.
    fn sketch_curve_selection(
        &self,
        target: document::scene::NodeId,
    ) -> Vec<document::sketch::SketchCurve> {
        self.panel_state
            .selection
            .sketch_segments(target)
            .map(document::sketch::SketchCurve::Segment)
            .chain(
                self.panel_state
                    .selection
                    .sketch_arcs(target)
                    .map(document::sketch::SketchCurve::Arc),
            )
            .chain(
                self.panel_state
                    .selection
                    .sketch_circles(target)
                    .map(document::sketch::SketchCurve::Circle),
            )
            // Aggregates arrive already typed: the selection named the whole ellipse or spline,
            // not one of its spans, so there is nothing to reassemble here.
            .chain(self.panel_state.selection.sketch_higher_curves(target))
            .collect()
    }

    fn sketch_transform_selection(
        &self,
        target: document::scene::NodeId,
    ) -> Vec<document::sketch::SketchTransformEntity> {
        self.panel_state
            .selection
            .sketch_points(target)
            .map(document::sketch::SketchTransformEntity::Point)
            .chain(
                self.panel_state
                    .selection
                    .sketch_segments(target)
                    .map(|id| {
                        document::sketch::SketchTransformEntity::Curve(
                            document::sketch::SketchCurve::Segment(id),
                        )
                    }),
            )
            .chain(self.panel_state.selection.sketch_arcs(target).map(|id| {
                document::sketch::SketchTransformEntity::Curve(document::sketch::SketchCurve::Arc(
                    id,
                ))
            }))
            .chain(self.panel_state.selection.sketch_circles(target).map(|id| {
                document::sketch::SketchTransformEntity::Curve(
                    document::sketch::SketchCurve::Circle(id),
                )
            }))
            .chain(
                self.panel_state
                    .selection
                    .sketch_higher_curves(target)
                    .map(document::sketch::SketchTransformEntity::Curve),
            )
            .collect()
    }

    /// The snap-policy profile point under the cursor (physical px), through the cached
    /// ray frame — the shared entry the drawing tools (#99) and vertex edits resolve a press
    /// or release with, quantized by [`PanelState::sketch_snap`] (#96). `None` when the
    /// cursor misses the plane or no sketch is being edited.
    pub(super) fn sketch_snapped_point_at(
        &self,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<document::sketch::SketchPoint> {
        let target = self.panel_state.sketch_mode?;
        let handles = self
            .panel_state
            .scene
            .sketch_handles(target, self.panel_state.geometry.voxels_per_block)?;
        let coord = self.cursor_to_profile_coord(
            cursor_x,
            cursor_y,
            self.last_ray_unprojection?,
            self.last_viewport_px,
            &handles,
        )?;
        Some(apply_sketch_snap(
            coord,
            self.panel_state.sketch_snap,
            self.panel_state.geometry.voxels_per_block,
        ))
    }

    /// The continuous, unsnapped profile-plane location under the cursor. Constraint branch
    /// choice stores this only in the live gesture; snapping it would change the chosen branch.
    fn sketch_unsnapped_profile_coord(&self, cursor_x: f64, cursor_y: f64) -> Option<[f64; 2]> {
        let target = self.panel_state.sketch_mode?;
        let handles = self
            .panel_state
            .scene
            .sketch_handles(target, self.panel_state.geometry.voxels_per_block)?;
        self.cursor_to_profile_coord(
            cursor_x,
            cursor_y,
            self.last_ray_unprojection?,
            self.last_viewport_px,
            &handles,
        )
    }

    /// Resolve the exact target shared by sketch drawing previews and commits. A grabbed existing
    /// vertex wins over snap policy, so an off-grid target cannot drift on release.
    ///
    /// The hovered curve comes from the same nearest-edge resolution the overlay highlights, so a
    /// point planted on an edge is held to the edge the author was looking at.
    fn sketch_target_at(
        &self,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<document::sketch::SketchTarget> {
        let target = self.panel_state.sketch_mode?;
        let (producer, _) = self.sketch_node_state(target)?;
        let existing = self.sketch_geometry_point_at(target, cursor_x, cursor_y);
        // Every kind travels, aggregates included. A pick can land on a spline even though no
        // relation can be held to one: the hold is attempted and refused at the seam that plants,
        // and the landing does not depend on it.
        let hovered = self
            .nearest_sketch_edge(cursor_x, cursor_y)
            .map(sketch_curve_from_hit);
        sketch_target::resolve_target(
            &producer,
            existing,
            self.sketch_snapped_point_at(cursor_x, cursor_y),
            hovered,
            self.sketch_evaluation_context()?,
        )
    }

    /// Resolve the endpoint and hovered open curve that form Tangent Arc's first semantic pick.
    /// The edge comes from the same nearest-edge cache the overlay highlights, so a junction's
    /// deterministic choice cannot disagree with what the author saw under the cursor.
    fn sketch_tangent_arc_source_at(
        &self,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<tangent_arc::TangentArcSource> {
        let target = self.panel_state.sketch_mode?;
        let (producer, _) = self.sketch_node_state(target)?;
        let seam = self.sketch_geometry_point_at(target, cursor_x, cursor_y)?;
        let curve = match self.nearest_open_sketch_edge(cursor_x, cursor_y)? {
            SketchEdgeHit::Segment(id) => document::sketch::SketchCurve::Segment(id),
            SketchEdgeHit::Arc(id) => document::sketch::SketchCurve::Arc(id),
            // Neither answers a seam: a circle has no endpoint, and an aggregate's endpoint
            // semantics are undecided while a spline's spans are re-derived from its fit points.
            SketchEdgeHit::Circle(_) | SketchEdgeHit::HigherCurve(_) => return None,
        };
        tangent_arc::resolve_source(&producer, curve, seam)
    }

    fn validate_line_gesture(&mut self, target: document::scene::NodeId) {
        let Some((producer, _)) = self.sketch_node_state(target) else {
            self.line_gesture.reset();
            return;
        };
        self.line_gesture.retain_if_live(
            target,
            |id| producer.sketch.points().iter().any(|point| point.id == id),
            |curve| match curve {
                document::sketch::SketchCurve::Segment(id) => producer
                    .sketch
                    .segments()
                    .iter()
                    .any(|segment| segment.id == id),
                document::sketch::SketchCurve::Arc(id) => {
                    producer.sketch.arcs().iter().any(|arc| arc.id == id)
                }
                document::sketch::SketchCurve::Circle(_) => false,
                document::sketch::SketchCurve::Bezier(_)
                | document::sketch::SketchCurve::Ellipse(_)
                | document::sketch::SketchCurve::Conic(_)
                | document::sketch::SketchCurve::Spline(_) => false,
            },
        );
    }

    pub(super) fn begin_line_press(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(target) = self.panel_state.sketch_mode else {
            self.line_gesture.reset();
            return;
        };
        self.validate_line_gesture(target);
        let hit = self.sketch_geometry_point_at(target, cursor_x, cursor_y);
        let hit_live_end = self
            .line_gesture
            .chain()
            .is_some_and(|chain| chain.owner == target && hit == Some(chain.end));
        self.line_gesture.begin_press(hit_live_end);
    }

    pub(super) fn update_line_drag(&mut self, down: (f64, f64), current: (f64, f64)) {
        self.line_gesture
            .update_drag(down, current, VIEW_CUBE_DRAG_THRESHOLD_PIXELS);
    }

    pub(super) fn line_arc_is_latched(&self) -> bool {
        self.line_gesture.arc_is_latched()
    }

    pub(super) fn line_press_is_live(&self) -> bool {
        self.line_gesture.press_is_live()
    }

    /// Commit one ordinary Line click. A point and its segment are produced on one local clone and
    /// reach history together; a refused segment leaves both the document and chain untouched.
    pub(super) fn sketch_line_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        self.validate_line_gesture(target);
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        let Some(resolved) = self.sketch_target_at(cursor_x, cursor_y) else {
            return;
        };
        let Some(context) = self.sketch_evaluation_context() else {
            return;
        };
        if let line::LineEdit::Document(next) = self
            .line_gesture
            .click(target, &producer, resolved, context)
        {
            self.commit_sketch_profile_edit(target, next);
        }
    }

    /// Commit a latched tangent arc through the document's candidate/atomic append seam.
    /// This explicit gesture owns its Tangent assertion; Shift inference is not consulted.
    pub(super) fn sketch_line_arc_release(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        self.validate_line_gesture(target);
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        let Some(resolved) = self.sketch_target_at(cursor_x, cursor_y) else {
            return;
        };
        let Some(context) = self.sketch_evaluation_context() else {
            return;
        };
        let Ok(next) = self
            .line_gesture
            .append_tangent_arc(&producer, resolved, context)
        else {
            return;
        };
        self.commit_sketch_profile_edit(target, next);
    }

    pub(super) fn end_line_press(&mut self) {
        self.line_gesture.end_press();
    }

    pub(super) fn accept_sketch_gesture(&mut self) -> bool {
        if self.panel_state.armed_constraint.is_none()
            && matches!(
                self.panel_state.sketch_tool,
                ui::panel::SketchTool::BreakCurve
                    | ui::panel::SketchTool::Trim
                    | ui::panel::SketchTool::Extend
                    | ui::panel::SketchTool::Fillet
                    | ui::panel::SketchTool::ChamferEqual
                    | ui::panel::SketchTool::ChamferDistanceAngle
                    | ui::panel::SketchTool::ChamferTwoDistance
                    | ui::panel::SketchTool::Offset
                    | ui::panel::SketchTool::MoveCopy
                    | ui::panel::SketchTool::Scale
            )
        {
            self.panel_state.sketch_tool = ui::panel::SketchTool::Select;
            return true;
        }
        self.midpoint_line_gesture.retain_for_context(
            self.panel_state.sketch_tool == ui::panel::SketchTool::MidpointLine,
            self.panel_state.armed_constraint.is_some(),
            self.panel_state.sketch_mode,
        );
        if self.midpoint_line_gesture.blocks_enter(
            self.panel_state.sketch_mode.is_some()
                && self.panel_state.sketch_tool == ui::panel::SketchTool::MidpointLine,
            self.panel_state.armed_constraint.is_some(),
        ) {
            return true;
        }
        let center_arc_producer = self
            .panel_state
            .sketch_mode
            .and_then(|owner| self.sketch_node_state(owner).map(|(producer, _)| producer));
        self.center_arc_gesture.retain_for_context(
            self.panel_state.sketch_tool == ui::panel::SketchTool::ArcCenterEndpoints,
            self.panel_state.armed_constraint.is_some(),
            self.panel_state.sketch_mode,
            center_arc_producer.as_ref(),
        );
        if self.center_arc_gesture.blocks_enter(
            self.panel_state.sketch_mode.is_some()
                && self.panel_state.sketch_tool == ui::panel::SketchTool::ArcCenterEndpoints,
            self.panel_state.armed_constraint.is_some(),
        ) {
            return true;
        }
        let point_circle_kind = point_circle_kind(self.panel_state.sketch_tool);
        self.point_circle_gesture.retain_for_context(
            point_circle_kind,
            self.panel_state.armed_constraint.is_some(),
            self.panel_state.sketch_mode,
        );
        if self.point_circle_gesture.blocks_enter(
            point_circle_kind,
            self.panel_state.armed_constraint.is_some(),
        ) {
            return true;
        }
        self.three_point_rectangle_gesture.retain_for_context(
            self.panel_state.sketch_tool == ui::panel::SketchTool::Rectangle3Point,
            self.panel_state.armed_constraint.is_some(),
            self.panel_state.sketch_mode,
        );
        if self.three_point_rectangle_gesture.blocks_enter(
            self.panel_state.sketch_mode.is_some()
                && self.panel_state.sketch_tool == ui::panel::SketchTool::Rectangle3Point,
            self.panel_state.armed_constraint.is_some(),
        ) {
            return true;
        }
        let corner_rectangle_kind = corner_rectangle_kind(self.panel_state.sketch_tool);
        self.corner_rectangle_gesture.retain_for_context(
            corner_rectangle_kind,
            self.panel_state.armed_constraint.is_some(),
            self.panel_state.sketch_mode,
        );
        if self.corner_rectangle_gesture.blocks_enter(
            corner_rectangle_kind,
            self.panel_state.armed_constraint.is_some(),
        ) {
            return true;
        }
        let polygon_kind = polygon_kind(self.panel_state.sketch_tool);
        self.polygon_gesture.retain_for_context(
            polygon_kind,
            self.panel_state.armed_constraint.is_some(),
            self.panel_state.sketch_mode,
        );
        if self
            .polygon_gesture
            .blocks_enter(polygon_kind, self.panel_state.armed_constraint.is_some())
        {
            return true;
        }
        let slot_kind = slot_kind(self.panel_state.sketch_tool);
        self.slot_gesture.retain_for_context(
            slot_kind,
            self.panel_state.armed_constraint.is_some(),
            self.panel_state.sketch_mode,
        );
        if self
            .slot_gesture
            .blocks_enter(slot_kind, self.panel_state.armed_constraint.is_some())
        {
            return true;
        }
        let tangent_circle_kind = tangent_circle_kind(self.panel_state.sketch_tool);
        self.tangent_circle_gesture.retain_for_context(
            tangent_circle_kind,
            self.panel_state.armed_constraint.is_some(),
            self.panel_state.sketch_mode,
        );
        if self.tangent_circle_gesture.blocks_enter(
            tangent_circle_kind,
            self.panel_state.armed_constraint.is_some(),
        ) {
            return true;
        }
        let tangent_producer = self
            .panel_state
            .sketch_mode
            .and_then(|owner| self.sketch_node_state(owner).map(|(producer, _)| producer));
        self.tangent_arc_gesture.retain_for_context(
            self.panel_state.sketch_tool == ui::panel::SketchTool::ArcTangent,
            self.panel_state.armed_constraint.is_some(),
            self.panel_state.sketch_mode,
            tangent_producer.as_ref(),
        );
        if self.tangent_arc_gesture.blocks_enter(
            self.panel_state.sketch_mode.is_some()
                && self.panel_state.sketch_tool == ui::panel::SketchTool::ArcTangent,
            self.panel_state.armed_constraint.is_some(),
        ) {
            return true;
        }
        let higher_kind = higher_curve_kind(self.panel_state.sketch_tool);
        self.higher_curve_gesture.retain_for_context(
            higher_kind,
            self.panel_state.armed_constraint.is_some(),
            self.panel_state.sketch_mode,
        );
        if self
            .higher_curve_gesture
            .blocks_enter(higher_kind, self.panel_state.armed_constraint.is_some())
        {
            if matches!(
                higher_kind,
                Some(
                    higher_curve::HigherCurveKind::FitPointSpline
                        | higher_curve::HigherCurveKind::ControlPointSpline
                )
            ) {
                if let Some((target, producer)) = self.panel_state.sketch_mode.and_then(|target| {
                    self.sketch_node_state(target)
                        .map(|(producer, _)| (target, producer))
                }) {
                    if let Some(context) = self.sketch_evaluation_context() {
                        let edit = self.higher_curve_gesture.finish(&producer, context);
                        if let higher_curve::HigherCurveEdit::Document(next) = edit {
                            self.commit_sketch_profile_edit(target, next);
                        }
                    }
                }
            }
            return true;
        }
        self.line_gesture.accept_for_enter(
            self.panel_state.sketch_mode.is_some()
                && self.panel_state.sketch_tool == ui::panel::SketchTool::Line,
            self.panel_state.armed_constraint.is_some(),
        )
    }

    /// Advance Midpoint Line on the ordinary stationary-click path. A missing/refused second
    /// target still reaches the gesture so its pending midpoint is consumed without history.
    pub(super) fn sketch_midpoint_line_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(target) = self.panel_state.sketch_mode else {
            self.midpoint_line_gesture.reset();
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            self.midpoint_line_gesture.reset();
            return;
        };
        let Some(context) = self.sketch_evaluation_context() else {
            self.midpoint_line_gesture.reset();
            return;
        };
        let resolved = self.sketch_target_at(cursor_x, cursor_y);
        if let midpoint_line::MidpointLineEdit::Document(next) = self
            .midpoint_line_gesture
            .click(target, &producer, resolved, context)
        {
            self.commit_sketch_profile_edit(target, next);
        }
    }

    /// Advance Center Point Arc on stationary clicks. Center and start are transient inputs; only
    /// a valid end-direction click crosses the document/undo boundary.
    pub(super) fn sketch_center_arc_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(target) = self.panel_state.sketch_mode else {
            self.center_arc_gesture.reset();
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            self.center_arc_gesture.reset();
            return;
        };
        let Some(context) = self.sketch_evaluation_context() else {
            self.center_arc_gesture.reset();
            return;
        };
        let resolved = self.sketch_target_at(cursor_x, cursor_y);
        if let center_arc::CenterArcEdit::Document(next) = self
            .center_arc_gesture
            .click(target, &producer, resolved, context)
        {
            self.commit_sketch_profile_edit(target, next);
        }
    }

    /// Advance either point-defined circle grammar. All circumference picks stay transient until
    /// the final click creates the center/radius circle in one edit.
    pub(super) fn sketch_point_circle_click(
        &mut self,
        kind: point_circle::PointCircleKind,
        cursor_x: f64,
        cursor_y: f64,
    ) {
        let Some(target) = self.panel_state.sketch_mode else {
            self.point_circle_gesture.reset();
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            self.point_circle_gesture.reset();
            return;
        };
        let resolved = self.sketch_target_at(cursor_x, cursor_y);
        if let point_circle::PointCircleEdit::Document(next) = self
            .point_circle_gesture
            .click(target, kind, &producer, resolved)
        {
            self.commit_sketch_profile_edit(target, next);
        }
    }

    /// Advance ellipse, conic, or a repeated-point spline grammar. Picks remain transient until
    /// the aggregate constructor accepts the complete curve, so a cancelled gesture has no
    /// document cleanup path.
    pub(super) fn sketch_higher_curve_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(kind) = higher_curve_kind(self.panel_state.sketch_tool) else {
            self.higher_curve_gesture.reset();
            return;
        };
        let Some(target) = self.panel_state.sketch_mode else {
            self.higher_curve_gesture.reset();
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            self.higher_curve_gesture.reset();
            return;
        };
        let Some(context) = self.sketch_evaluation_context() else {
            self.higher_curve_gesture.reset();
            return;
        };
        let resolved = self.sketch_target_at(cursor_x, cursor_y);
        let edit = self
            .higher_curve_gesture
            .click(target, kind, &producer, resolved, context);
        if let higher_curve::HigherCurveEdit::Document(next) = edit {
            self.commit_sketch_profile_edit(target, next);
        }
    }

    pub(super) fn sketch_three_point_rectangle_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(target) = self.panel_state.sketch_mode else {
            self.three_point_rectangle_gesture.reset();
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            self.three_point_rectangle_gesture.reset();
            return;
        };
        let resolved = self.sketch_target_at(cursor_x, cursor_y);
        let Some(context) = self.sketch_evaluation_context() else {
            self.three_point_rectangle_gesture.reset();
            return;
        };
        if let three_point_rectangle::ThreePointRectangleEdit::Document(next) = self
            .three_point_rectangle_gesture
            .click(target, &producer, resolved, context)
        {
            self.commit_sketch_profile_edit(target, next);
        }
    }

    /// Advance the active regular-polygon grammar. Its construction picks remain transient;
    /// the final click authors the complete closed loop as one undoable edit.
    pub(super) fn sketch_polygon_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(kind) = polygon_kind(self.panel_state.sketch_tool) else {
            self.polygon_gesture.reset();
            return;
        };
        let Some(target) = self.panel_state.sketch_mode else {
            self.polygon_gesture.reset();
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            self.polygon_gesture.reset();
            return;
        };
        let resolved = self.sketch_target_at(cursor_x, cursor_y);
        let sides = normalized_polygon_sides(self.panel_state.sketch_polygon_sides);
        if let polygon::PolygonEdit::Document(next) = self
            .polygon_gesture
            .click(target, kind, &producer, resolved, sides)
        {
            self.commit_sketch_profile_edit(target, next);
        }
    }

    /// Advance the active slot grammar; only its final width pick commits the native line/arc
    /// boundary and enters undo history.
    pub(super) fn sketch_slot_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(kind) = slot_kind(self.panel_state.sketch_tool) else {
            self.slot_gesture.reset();
            return;
        };
        let Some(target) = self.panel_state.sketch_mode else {
            self.slot_gesture.reset();
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            self.slot_gesture.reset();
            return;
        };
        let Some(context) = self.sketch_evaluation_context() else {
            self.slot_gesture.reset();
            return;
        };
        let resolved = self.sketch_target_at(cursor_x, cursor_y);
        if let slot::SlotEdit::Document(next) = self
            .slot_gesture
            .click(target, kind, &producer, resolved, context)
        {
            self.commit_sketch_profile_edit(target, next);
        }
    }

    /// Advance a tangent-circle grammar from line-only picks. Two-Tangent consumes a final radius
    /// witness; Three-Tangent completes on its third distinct line.
    pub(super) fn sketch_tangent_circle_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(kind) = tangent_circle_kind(self.panel_state.sketch_tool) else {
            self.tangent_circle_gesture.reset();
            return;
        };
        let Some(target) = self.panel_state.sketch_mode else {
            self.tangent_circle_gesture.reset();
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            self.tangent_circle_gesture.reset();
            return;
        };
        let Some(context) = document::sketch::evaluation_context_from_density(
            self.panel_state.geometry.voxels_per_block,
        ) else {
            self.tangent_circle_gesture.reset();
            return;
        };
        let cursor = self.sketch_snapped_point_at(cursor_x, cursor_y);
        let line = self.sketch_segment_at(cursor_x, cursor_y).zip(cursor);
        if let tangent_circle::TangentCircleEdit::Document(next) = self
            .tangent_circle_gesture
            .click(target, kind, &producer, line, cursor, context)
        {
            self.commit_sketch_profile_edit(target, next);
        }
    }

    /// Advance standalone Tangent Arc on stationary clicks. The first click captures only a
    /// supported incoming endpoint; the second is the sole document/undo boundary.
    pub(super) fn sketch_tangent_arc_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(target) = self.panel_state.sketch_mode else {
            self.tangent_arc_gesture.reset();
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            self.tangent_arc_gesture.reset();
            return;
        };
        if !self.tangent_arc_gesture.is_pending() {
            if let Some(source) = self.sketch_tangent_arc_source_at(cursor_x, cursor_y) {
                self.tangent_arc_gesture.begin(target, source);
            }
            return;
        }
        let endpoint = self.sketch_target_at(cursor_x, cursor_y);
        let Some(context) = self.sketch_evaluation_context() else {
            self.tangent_arc_gesture.reset();
            return;
        };
        if let tangent_arc::TangentArcEdit::Document(next) = self
            .tangent_arc_gesture
            .complete(target, &producer, endpoint, context)
        {
            self.commit_sketch_profile_edit(target, next);
        }
    }

    /// #102: one 3-point-arc click. Click 1 picks the start endpoint, click 2 the
    /// end endpoint — each an existing vertex under the cursor or a fresh snapped free point —
    /// and click 3 names a coordinate the curve passes through, which the included angle is
    /// solved from and then discarded (a through-point is an input, never an entity). Picking the
    /// start twice drops the gesture rather than storing a degenerate arc. Each click that changes the store commits one entry in the open sketch undo group.
    pub(super) fn sketch_arc_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        // An endpoint deleted mid-gesture (Delete key, undo) leaves a dangling id — drop the
        // gesture rather than arc to a ghost.
        let alive = |id| producer.sketch.points().iter().any(|point| point.id == id);
        if let Some((start, end)) = self.sketch_arc_gesture {
            if !alive(start) || end.is_some_and(|id| !alive(id)) {
                self.sketch_arc_gesture = None;
            }
        }
        // The third click is a coordinate, not a vertex: solve the sweep and store the arc.
        if let Some((start, Some(end))) = self.sketch_arc_gesture {
            self.sketch_arc_gesture = None;
            let Some(through) = self.sketch_snapped_point_at(cursor_x, cursor_y) else {
                return;
            };
            let coord = |id| {
                producer
                    .sketch
                    .points()
                    .iter()
                    .find(|point| point.id == id)
                    .map(|point| point.at.in_plane())
            };
            let (Some(from), Some(to)) = (coord(start), coord(end)) else {
                return;
            };
            let Some(degrees) =
                document::sketch::included_angle_through_degrees(from, to, through.in_plane())
            else {
                return;
            };
            let Ok(bulge) = parametric::units::AngleMeasurement::try_from_degrees_f64(degrees)
            else {
                return;
            };
            let next = producer.with_arc_between(start, end, bulge);
            if next != producer {
                self.commit_sketch_profile_edit(target, next);
            }
            return;
        }

        // Clicks one and two are endpoints the author points at, so they go through the shared
        // seam and are held to whatever they land on. Click three above is not: it is a coordinate
        // the sweep is read from and then discarded, and it mints no point to hold.
        let Some(resolved) = self.sketch_target_at(cursor_x, cursor_y) else {
            return;
        };
        let Some(context) = self.sketch_evaluation_context() else {
            return;
        };
        let (next, clicked) = match resolved.existing() {
            Some(id) => (producer.clone(), id),
            None => producer.with_target_point(resolved, context),
        };
        self.sketch_arc_gesture = match self.sketch_arc_gesture {
            None => Some((clicked, None)),
            // A zero-length arc cannot be held. An already-joined pair CAN: the
            // curve is not known until the third click, and arcing over a chord (or over another
            // arc that bulges elsewhere) is legal geometry. A true duplicate is refused at the
            // commit, where the sweep exists to compare.
            Some((start, _)) if start == clicked => None,
            Some((start, _)) => Some((start, Some(clicked))),
        };
        if next != producer {
            self.commit_sketch_profile_edit(target, next);
        }
    }

    /// One Circle Center-Diameter click. The first captures the center without writing; the
    /// second supplies only the radius and commits the whole circle as one profile edit.
    pub(super) fn sketch_circle_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        let Some(point) = self.sketch_snapped_point_at(cursor_x, cursor_y) else {
            return;
        };
        let Some((center, perimeter)) =
            advance_circle_center_diameter_gesture(&mut self.sketch_circle_center, point)
        else {
            self.sketch_circle_target = Some(target);
            return;
        };
        self.sketch_circle_target = None;
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        if let Some(next) = complete_circle_center_diameter(&producer, center, perimeter) {
            self.commit_sketch_profile_edit(target, next);
        }
    }

    /// Advance whichever corner-rectangle grammar is armed. The first click pins the anchor
    /// (a corner, or the center); the second appends the closed loop as one undo entry. A
    /// degenerate or off-plane second click authors nothing and leaves the anchor standing,
    /// so the gesture can simply be finished somewhere else.
    pub(super) fn sketch_corner_rectangle_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(kind) = corner_rectangle_kind(self.panel_state.sketch_tool) else {
            return;
        };
        let Some(target) = self.panel_state.sketch_mode else {
            self.corner_rectangle_gesture.reset();
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            self.corner_rectangle_gesture.reset();
            return;
        };
        let Some(context) = self.sketch_evaluation_context() else {
            self.corner_rectangle_gesture.reset();
            return;
        };
        let resolved = self.sketch_target_at(cursor_x, cursor_y);
        if let corner_rectangle::CornerRectangleEdit::Document(next) = self
            .corner_rectangle_gesture
            .click(target, kind, &producer, resolved, context)
        {
            self.commit_sketch_profile_edit(target, next);
        }
    }

    /// Resolve a stationary Select-tool click into the sketch selection. A vertex under
    /// the cursor takes priority (it already answers as a handle), then a segment, else empty
    /// space. Plain click **replaces** the selection with that one entity; `shift` **toggles** it
    /// in/out (accumulate). A plain click on empty space **clears**; a Shift-click on empty space
    /// keeps the selection (Fusion). Reuses the same hit-tests the drag runs, so what you
    /// click is what you pick. Pure selection-state mutation — records no document edit.
    pub(super) fn resolve_sketch_selection_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(sketch) = self.panel_state.sketch_mode else {
            return;
        };
        let shift = self.shift_held;
        match self.sketch_entity_target_at(sketch, cursor_x, cursor_y) {
            Some(target) if shift => self.panel_state.selection.toggle(target),
            Some(target) => self.panel_state.selection.select_only(target),
            // Empty space: a plain click clears; a Shift-click leaves the set alone (Fusion).
            // Only the sketch side goes — what is picked outside the mode is not this click's
            // business.
            None if !shift => self.panel_state.selection.clear_sketch_entities(),
            None => {}
        }
    }

    /// The marquee's box in physical pixels, **clamped to the viewport**.
    ///
    /// One minting serves both readings of the gesture — the band that is drawn and the box that
    /// selects — because they are the same rectangle and had drifted into being two. The overlay
    /// paints nothing outside the viewport, and the sketch overlay's own law is that what is
    /// clickable is exactly what is drawn; a box reaching under the side panel and selecting the
    /// geometry hidden there would be a mutation the author had no way to see coming.
    ///
    /// The DIRECTION is read from the raw cursor, never from this rectangle: which way the hand
    /// travelled is what picks window-vs-crossing, and clamping must not be able to change it.
    fn sketch_marquee_box_px(&self, from: (f64, f64), to: (f64, f64)) -> egui::Rect {
        marquee_box_px(self.last_viewport_px, from, to)
    }

    /// Sketch-selection slice 3: resolve a DRAGGED empty-space Select release into the box
    /// selection. Direction picks the semantic (Fusion): left→right = **window** — points inside
    /// the box, segments with ≥1 endpoint inside; right→left = **crossing** — any entity the box
    /// touches, so a segment passing through with both endpoints outside still selects. Shift
    /// accumulates into the set; a plain marquee replaces the sketch-entity selection (an empty
    /// box therefore clears, like a plain empty click). A behind-camera endpoint culls its
    /// entity, matching the overlay cull. Pure selection-state mutation — no document edit.
    ///
    /// Constraint badges are swept too, on the point rule: a badge is a small square mark, so
    /// window takes it when its CENTER is inside and crossing when the box touches its box. A
    /// constraint has no position of its own, but the badge does — and the badge is the whole of
    /// how a constraint is on screen, so a box drawn around it has named it as plainly as a box
    /// drawn around a vertex names that.
    pub(super) fn resolve_sketch_marquee(&mut self, up_x: f64, up_y: f64) {
        let Some((down_x, down_y)) = self.sketch_marquee_anchor.take() else {
            return;
        };
        let Some(sketch) = self.panel_state.sketch_mode else {
            return;
        };
        let window = up_x >= down_x;
        let rect = self.sketch_marquee_box_px((down_x, down_y), (up_x, up_y));
        let mut picked: Vec<ui::panel::SelectionTarget> = Vec::new();
        let arms = self.tangent_arm_points(sketch);
        for (index, vertex) in self.sketch_vertex_px.iter().enumerate() {
            let inside = vertex.map(|px| rect.contains(px)).unwrap_or(false);
            if let (true, Some(&entity)) = (inside, self.sketch_point_ids.get(index)) {
                // A lever's arms are swept up by any box drawn over the spline they steer, and
                // they are manipulators rather than geometry: selecting them lights chrome the
                // author did not reach for, and hands the next verb a set it cannot act on.
                // A deliberate click still picks one — this is about the sweep.
                if arms.contains(&entity) {
                    continue;
                }
                picked.push(ui::panel::SelectionTarget::SketchPoint { sketch, entity });
            }
        }
        for segment in &self.sketch_segments {
            if let (Some(Some(a)), Some(Some(b))) = (
                self.sketch_vertex_px.get(segment.from),
                self.sketch_vertex_px.get(segment.to),
            ) {
                let hit = if window {
                    rect.contains(*a) || rect.contains(*b)
                } else {
                    segment_touches_rect(*a, *b, rect)
                };
                if hit {
                    picked.push(ui::panel::SelectionTarget::SketchSegment {
                        sketch,
                        entity: segment.entity,
                    });
                }
            }
        }
        // #102: an arc answers the box through its drawn chords — window takes it when an
        // ENDPOINT is inside (the segment rule, so a bulge crossing the box edge doesn't count),
        // crossing when any chord touches.
        for (entity, chords) in &self.sketch_arc_chords {
            let hit = match (window, chords.first(), chords.last()) {
                (true, Some(a), Some(b)) => rect.contains(*a) || rect.contains(*b),
                (false, _, _) => chords
                    .array_windows::<2>()
                    .any(|pair| segment_touches_rect(pair[0], pair[1], rect)),
                (true, _, _) => false,
            };
            if hit {
                picked.push(ui::panel::SelectionTarget::SketchArc {
                    sketch,
                    entity: *entity,
                });
            }
        }
        for (entity, ring) in &self.sketch_circle_chords {
            let hit = circle_marquee_hit(ring, rect, window);
            if hit {
                picked.push(ui::panel::SelectionTarget::SketchCircle {
                    sketch,
                    entity: *entity,
                });
            }
        }
        for curve in aggregate_marquee_picks(&self.sketch_higher_curve_chords, rect, window) {
            picked.push(ui::panel::SelectionTarget::SketchHigherCurve { sketch, curve });
        }
        // The badges are laid out in egui points; the box is in physical pixels, like every other
        // array here.
        let scale = self.last_pixels_per_point;
        let half = ui::chrome::SKETCH_CONSTRAINT_BADGE * 0.5 * scale;
        for gizmo in &self.sketch_dimension_gizmos {
            // A ghost is not in the drawing yet, so a rubber band cannot catch it.
            let Some(entity) = gizmo.constraint else {
                continue;
            };
            let hit = gizmo.drawing.label_boxes().into_iter().any(|box_px| {
                let box_px = egui::Rect::from_min_max(
                    egui::Pos2::new(box_px.min.x * scale, box_px.min.y * scale),
                    egui::Pos2::new(box_px.max.x * scale, box_px.max.y * scale),
                );
                if window {
                    rect.contains_rect(box_px)
                } else {
                    rect.intersects(box_px)
                }
            });
            if hit {
                picked.push(ui::panel::SelectionTarget::SketchConstraint { sketch, entity });
            }
        }
        for badge in &self.sketch_constraint_badges {
            let center = egui::Pos2::new(badge.center.x * scale, badge.center.y * scale);
            let hit = if window {
                rect.contains(center)
            } else {
                rect.intersects(egui::Rect::from_center_size(
                    center,
                    egui::Vec2::splat(half * 2.0),
                ))
            };
            if hit {
                picked.push(ui::panel::SelectionTarget::SketchConstraint {
                    sketch,
                    entity: badge.constraint,
                });
            }
        }
        if !self.shift_held {
            self.panel_state.selection.clear_sketch_entities();
        }
        for target in picked {
            if !self.panel_state.selection.contains(target) {
                self.panel_state.selection.toggle(target);
            }
        }
    }

    /// Resolve a stationary viewport click into a node selection change, or `None`
    /// when the click asks for nothing.
    ///
    /// The raycast names the solid absolute voxel under the cursor (CPU truth over the resident
    /// chunks — never the GPU brick field, so what is selected is what the document says is
    /// there), and the document says which node owns it. Plain click **replaces** the selection;
    /// Shift **toggles** that node in/out. A plain click on empty space **clears**; a Shift-click
    /// on empty space keeps the selection (Fusion, matching the sketch rule).
    ///
    /// A voxel that resolves to no node is treated as empty space: it means the raycast and the
    /// document's fold disagree, and clearing is the honest answer to "you clicked nothing I can
    /// name".
    pub(super) fn resolve_viewport_selection_click(
        &self,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<ui::panel::SelectionRequest> {
        let density = self.panel_state.geometry.voxels_per_block;
        let [vx, vy, vw, vh] = self.last_viewport_px;
        let frame = crate::PickFrame {
            region_dimensions: self.region_dimensions,
            recenter_voxels: self.recenter_voxels.voxels(),
            density,
            chunks: &self.resident_chunks,
            band: self.last_pick_band,
        };
        // `pick_voxel` answers in the scene's ABSOLUTE voxel frame, which is exactly the frame
        // `picked_node_at_voxel` reads — no recenter to undo (the frame is carried).
        let picked = self
            .app_core
            .pick_voxel(
                [cursor_x as f32, cursor_y as f32],
                [vx as f32, vy as f32, vw as f32, vh as f32],
                &frame,
            )
            .and_then(|pick| {
                self.panel_state
                    .scene
                    .picked_node_at_voxel(pick.absolute_voxel, density)
            });
        match (picked, self.shift_held) {
            (Some(node), true) => Some(ui::panel::SelectionRequest::Toggle(
                ui::panel::SelectionTarget::Node(node),
            )),
            (Some(node), false) => Some(ui::panel::SelectionRequest::Only(
                ui::panel::SelectionTarget::Node(node),
            )),
            (None, false) => Some(ui::panel::SelectionRequest::Clear),
            (None, true) => None,
        }
    }

    /// Arm an orbit-center placement: from here the gizmo rides the cursor until a click commits
    /// it or Esc / a right-click drops it.
    pub(super) fn begin_orbit_center_placement(&mut self) {
        self.placing_orbit_center = true;
    }

    /// Commit the armed placement. Returns whether the click was CONSUMED — true while a
    /// placement is armed at all, so a click over nothing keeps the placement armed rather
    /// than falling through to select something. Only a click that HITS a surface places.
    ///
    /// The ray is cast here and nowhere else. It runs on the stationary-click path so a
    /// preview point in step, which put a full CPU pick between the mouse and the frame that
    /// showed it — the reason the gizmo trailed the cursor on a large scene. It answers one
    /// question, asked once, at the moment the answer is needed.
    pub(super) fn commit_orbit_center_placement(&mut self) -> bool {
        if !self.placing_orbit_center {
            return false;
        }
        if let Some(point) = self.surface_point_at(self.last_cursor_position) {
            self.app_core.camera.place_orbit_center(point);
            self.placing_orbit_center = false;
        }
        true
    }

    /// Drop the armed placement, leaving the committed orbit center untouched — nothing to
    /// restore, because arming never wrote to it. Returns whether anything was armed.
    pub(super) fn cancel_orbit_center_placement(&mut self) -> bool {
        std::mem::take(&mut self.placing_orbit_center)
    }

    /// Escape's first sketch rung: drop whatever half-finished gesture the armed tool is holding
    /// — the Line chain, construction inputs, a rectangle's first corner, the marquee's anchor,
    /// or an arc's pending picks. Reports whether anything was actually put back, so the cancel chain can fall
    /// through when there was nothing mid-stroke. The tool stays armed: dropping a stroke is not
    /// the same act as putting the tool down.
    pub(super) fn cancel_sketch_gesture(&mut self) -> bool {
        if self.panel_state.sketch_mode.is_none() {
            self.reset_sketch_gestures();
            self.sketch_edit_press = false;
            return false;
        }
        // A constraint holding picks is a half-finished gesture like any other, and Escape puts
        // the picks back without putting the constraint down — the same rung, the same rule.
        let constraint_picks = self
            .panel_state
            .armed_constraint
            .as_ref()
            .is_some_and(|armed| !armed.picked().is_empty());
        if constraint_picks {
            let verb = self
                .panel_state
                .armed_constraint
                .as_ref()
                .map(ui::panel::ArmedConstraint::verb);
            self.panel_state.armed_constraint = verb.map(ui::panel::ArmedConstraint::new);
            self.panel_state.selection.clear_sketch_entities();
            self.panel_state.sketch_constraint_refusal = None;
        }
        let line_live = self.line_gesture.cancel_for_escape(
            self.panel_state.sketch_tool == ui::panel::SketchTool::Line,
            self.panel_state.armed_constraint.is_some(),
        );
        let midpoint_line_live = self.midpoint_line_gesture.cancel_for_escape(
            self.panel_state.sketch_tool == ui::panel::SketchTool::MidpointLine,
            self.panel_state.armed_constraint.is_some(),
        );
        let stationary_gesture_press = self.sketch_edit_press
            && matches!(
                self.panel_state.sketch_tool,
                ui::panel::SketchTool::MidpointLine
                    | ui::panel::SketchTool::ArcCenterEndpoints
                    | ui::panel::SketchTool::ArcTangent
                    | ui::panel::SketchTool::Circle2Point
                    | ui::panel::SketchTool::Circle3Point
                    | ui::panel::SketchTool::Circle2Tangent
                    | ui::panel::SketchTool::Circle3Tangent
                    | ui::panel::SketchTool::Rectangle3Point
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
            )
            && self.panel_state.armed_constraint.is_none();
        let center_arc_live = self.center_arc_gesture.cancel_for_escape(
            self.panel_state.sketch_tool == ui::panel::SketchTool::ArcCenterEndpoints,
            self.panel_state.armed_constraint.is_some(),
        );
        let tangent_arc_live = self.tangent_arc_gesture.cancel_for_escape(
            self.panel_state.sketch_tool == ui::panel::SketchTool::ArcTangent,
            self.panel_state.armed_constraint.is_some(),
        );
        let point_circle_live = self.point_circle_gesture.cancel_for_escape(
            point_circle_kind(self.panel_state.sketch_tool),
            self.panel_state.armed_constraint.is_some(),
        );
        let higher_curve_live = self.higher_curve_gesture.cancel_for_escape(
            higher_curve_kind(self.panel_state.sketch_tool),
            self.panel_state.armed_constraint.is_some(),
        );
        let three_point_rectangle_live = self.three_point_rectangle_gesture.cancel_for_escape(
            self.panel_state.sketch_tool == ui::panel::SketchTool::Rectangle3Point,
            self.panel_state.armed_constraint.is_some(),
        );
        let corner_rectangle_live = self.corner_rectangle_gesture.cancel_for_escape(
            corner_rectangle_kind(self.panel_state.sketch_tool),
            self.panel_state.armed_constraint.is_some(),
        );
        let polygon_live = self.polygon_gesture.cancel_for_escape(
            polygon_kind(self.panel_state.sketch_tool),
            self.panel_state.armed_constraint.is_some(),
        );
        let slot_live = self.slot_gesture.cancel_for_escape(
            slot_kind(self.panel_state.sketch_tool),
            self.panel_state.armed_constraint.is_some(),
        );
        let tangent_circle_live = self.tangent_circle_gesture.cancel_for_escape(
            tangent_circle_kind(self.panel_state.sketch_tool),
            self.panel_state.armed_constraint.is_some(),
        );
        let chamfer_live = self.sketch_chamfer_pending.take().is_some();
        let offset_live = self.sketch_offset_pending.take().is_some();
        let move_copy_live = self.sketch_move_copy_pending.take().is_some();
        let scale_live = self.sketch_scale_pending.take().is_some();
        let rectangular_pattern_live = self.sketch_rectangular_pattern_pending.take().is_some();
        let live = constraint_picks
            || line_live
            || midpoint_line_live
            || center_arc_live
            || tangent_arc_live
            || point_circle_live
            || higher_curve_live
            || three_point_rectangle_live
            || corner_rectangle_live
            || polygon_live
            || slot_live
            || tangent_circle_live
            || chamfer_live
            || offset_live
            || move_copy_live
            || scale_live
            || rectangular_pattern_live
            || stationary_gesture_press
            || self.sketch_marquee_anchor.is_some()
            || self.sketch_arc_gesture.is_some()
            || self.sketch_circle_center.is_some();
        self.sketch_marquee_anchor = None;
        self.sketch_arc_gesture = None;
        self.sketch_circle_center = None;
        self.sketch_circle_target = None;
        self.sketch_edit_press = false;
        live
    }

    /// Escape's second sketch rung: put the armed sketch tool down, back to Select — the arrow is
    /// the mode's rest state, the way no-tool-armed is the viewport's. Reports whether a tool was
    /// actually armed, so Escape on the bare Select tool falls through to the rest of the chain
    /// rather than swallowing the key.
    pub(super) fn disarm_sketch_tool(&mut self) -> bool {
        if self.panel_state.sketch_mode.is_none() {
            return false;
        }
        // A constraint is put down FIRST, and on its own: it overrides the drawing tool while it
        // runs, so the tool underneath is not what the author is trying to escape from.
        if self.panel_state.armed_constraint.take().is_some() {
            self.panel_state.sketch_constraint_refusal = None;
            return true;
        }
        if self.panel_state.sketch_tool == ui::panel::SketchTool::Select {
            return false;
        }
        self.panel_state.sketch_tool = ui::panel::SketchTool::Select;
        self.reset_sketch_gestures();
        self.sketch_chamfer_pending = None;
        self.sketch_offset_pending = None;
        self.sketch_move_copy_pending = None;
        self.sketch_scale_pending = None;
        self.sketch_rectangular_pattern_pending = None;
        true
    }

    /// End the running **modal command** — the OK / Cancel pair the viewport menu offers and
    /// Return / Escape drive. Returns whether a command was running, so Escape can fall through
    /// to the tool ghost when none was.
    ///
    /// This is the one exit every modal command reports through. Today the only one is the
    /// explicit orbit mode, and `Accept` and `Cancel` do the same thing for it: navigating IS the
    /// result and it has already happened, so there is nothing pending to discard. A future
    /// command with an uncommitted edit is where the two diverge, and it diverges HERE — the
    /// panel never learns what either word means.
    ///
    /// Any in-flight gesture dies with the mode rather than being stranded in a mode that no
    /// longer exists, and a per-session type override dies with it too: the DEFAULT type is never
    /// written on the way out.
    pub(super) fn end_modal_command(&mut self, _command: ui::panel::ModeCommand) -> bool {
        if !self.panel_state.orbit_mode.is_on() {
            return false;
        }
        self.panel_state.orbit_mode = ui::panel::OrbitMode::Off;
        self.orbiting_in_orbit_mode = false;
        self.orbit_mode_recenter_press = false;
        true
    }

    /// Run whatever the frame's key presses were bound to (`ui::shortcuts`).
    ///
    /// No handler here names a key. The settings say which command each binding holds, egui's
    /// `consume_shortcut` says which of them fired, and this match says what each one DOES — so a
    /// rebind moves one settings entry and both the menu's right-hand column and this dispatch
    /// follow it. Called after the egui pass, which is what makes a focused text field swallow its
    /// own Escape instead of canceling the running viewport command (and its own Ctrl+Z instead
    /// of undoing a document edit). Returns the Undo/Redo [`crate::IntentEffect`], which the
    /// caller folds into the frame's merged effect so the display rebuilds like any other edit.
    fn run_shortcut_commands(&mut self) -> crate::IntentEffect {
        let mut effect = crate::IntentEffect::none();
        for command in self
            .panel_state
            .shortcuts
            .clone()
            .consume(&self.egui_bridge.context)
        {
            match command {
                // The document history. `AppCore::undo`/`redo` route into an open sketch
                // group's fine-grained session stacks by themselves.
                ui::shortcuts::ShortcutCommand::Undo => {
                    self.reset_sketch_gestures();
                    effect = effect.merged_with(
                        self.app_core
                            .undo(&mut self.panel_state.scene, &mut self.panel_state.selection),
                    );
                }
                ui::shortcuts::ShortcutCommand::Redo => {
                    self.reset_sketch_gestures();
                    effect = effect.merged_with(
                        self.app_core
                            .redo(&mut self.panel_state.scene, &mut self.panel_state.selection),
                    );
                }
                // Dump the scene + LIVE camera to the repro file (`shot --from-config`), so an
                // exact live-view bug reproduces headlessly.
                ui::shortcuts::ShortcutCommand::ExportRepro => self.export_repro(),
                // Cancel is a priority chain, not one act. An armed orbit-center
                // placement outranks the tool ghost — it is what the cursor is carrying, so it
                // goes back first and leaves any armed tool alone. With nothing to put back it
                // CANCELS the running modal command (the same act the viewport menu's Cancel row
                // performs); with no command running it disarms the tool ghost.
                // Leaving never writes the DEFAULT orbit type: a session override dies with the
                // mode rather than outliving it.
                //
                // Inside a sketch the chain gains two rungs, innermost first: a half-drawn Line
                // chain, pending Midpoint Line, rectangle press, marquee, arc/circle gesture, or
                // constraint pick-set goes back before anything else the mode is holding. An
                // armed sketch TOOL then falls back to Select before the placement ghost is
                // touched. Escape never leaves sketch mode; that is the mode's own Cancel button's
                // job.
                ui::shortcuts::ShortcutCommand::CancelCommand => {
                    if !self.cancel_orbit_center_placement()
                        && !self.cancel_sketch_gesture()
                        && !self.end_modal_command(ui::panel::ModeCommand::Cancel)
                        && !self.disarm_sketch_tool()
                    {
                        self.disarm_placement();
                    }
                }
                // The other half of the universal pair. Line finishes an open chain; Midpoint
                // Line explicitly keeps its pending midpoint because Enter cannot name the second
                // endpoint. Other sketch gestures do nothing here. Outside those cases Accept is
                // not a general viewport verb.
                ui::shortcuts::ShortcutCommand::AcceptCommand => {
                    if !self.accept_sketch_gesture() {
                        self.end_modal_command(ui::panel::ModeCommand::Accept);
                    }
                }
                // The same door the menu's Delete row goes through.
                ui::shortcuts::ShortcutCommand::DeleteSelection => self.delete_selection(),
                // Listed in the settings so they CAN be bound, but reachable only from the
                // viewport menu today. Unbound by default, so these are unreachable until
                // somebody binds one — at which point this is the missing half, not dead code.
                ui::shortcuts::ShortcutCommand::PlaceOrbitCenter
                | ui::shortcuts::ShortcutCommand::ResetOrbitCenter
                | ui::shortcuts::ShortcutCommand::EnterConstrainedOrbit => {}
                // #100: the carve / fill verb needs the region the RIGHT-PRESS resolved, so a
                // keyboard binding has nothing to act on until the menu has been raised.
                ui::shortcuts::ShortcutCommand::ToggleSketchFace => self.toggle_sketch_menu_face(),
                ui::shortcuts::ShortcutCommand::ToggleSketchConstruction => {
                    self.toggle_sketch_selection_construction();
                }
            }
        }
        effect
    }

    /// Where the orbit-center gizmo draws in WORLD space, and whether it draws at all: the
    /// committed center while a Shift+MMB orbit turns about it. `None` the rest of the time — the
    /// center is a pivot, not permanent furniture.
    ///
    /// An armed placement has no world point to answer with, by design: it draws at the cursor
    /// ([`orbit_center_marker`](Self::orbit_center_marker)) and resolves a point only at the
    /// click.
    pub(super) fn visible_orbit_center(&self) -> Option<glam::Vec3> {
        (!self.placing_orbit_center && self.orbiting_about_center)
            .then_some(self.app_core.camera.orbit_center)
    }

    /// Project the ORBITING marker for NEXT frame's draw, the same one-frame lag the sketch
    /// overlay takes. Culled behind the camera.
    ///
    /// The armed-placement marker does not come through here — see
    /// [`orbit_center_marker`](Self::orbit_center_marker) for why it must not.
    fn refresh_orbit_center_overlay(
        &mut self,
        view_projection: glam::Mat4,
        viewport_px: [u32; 4],
        pixels_per_point: f32,
    ) {
        self.orbit_center_overlay = None;
        if self.placing_orbit_center {
            return;
        }
        let Some(center) = self.visible_orbit_center() else {
            return;
        };
        self.orbit_center_overlay =
            project_to_screen(center, view_projection, viewport_px, pixels_per_point)
                .map(|position| (position, false));
    }

    /// Whether the explicit orbit mode's reticle draws this frame.
    ///
    /// Nothing is projected: the camera looks AT its target, so the target is the viewport's
    /// center by construction and the reticle is laid out against the rect egui itself just
    /// measured. That also means it cannot lag the camera by a frame the way a cached
    /// projection would.
    ///
    /// It hides while a TURN is in flight — the mark spans most of the frame, and watching the
    /// model come round is exactly when you need it out of the way. A press that has not crossed
    /// the drag threshold keeps it: that press is still a candidate for the re-centering click,
    /// which aims *at* the reticle, so blanking on mouse-down would hide the sight the moment
    /// you took the shot.
    pub(super) fn orbit_reticle_visible(&self) -> bool {
        let turning = self.orbiting_in_orbit_mode && !self.orbit_mode_recenter_press;
        self.panel_state.orbit_mode.is_on() && !turning
    }

    /// Where the orbit-center gizmo draws THIS frame, and whether a placement is armed.
    ///
    /// An armed placement IS the cursor. The gizmo draws at the cursor position and nothing else
    /// is consulted — no projection cache, no ray, so nothing can put a frame or a CPU pick
    /// between the mouse and the mark that follows it. What is under the cursor when the click
    /// lands is the placement's business, not the gizmo's: a click over nothing simply does not
    /// commit, and the placement stays armed.
    ///
    /// A stationary-click gate keeps the gizmo responsive and visible over
    /// sky. Lag is the worse failure — a mark that trails the cursor stops reading as "this is
    /// what you are carrying".
    ///
    /// The orbiting marker keeps the cached projection, where the lag is invisible —
    /// `orbit_about_point` holds the pivot screen-fixed for the whole drag by construction.
    fn orbit_center_marker(&self, pixels_per_point: f32) -> Option<(egui::Pos2, bool)> {
        if !self.placing_orbit_center {
            return self.orbit_center_overlay;
        }
        let (cursor_x, cursor_y) = self.last_cursor_position?;
        Some((
            egui::Pos2::new(
                cursor_x as f32 / pixels_per_point,
                cursor_y as f32 / pixels_per_point,
            ),
            true,
        ))
    }

    /// The surface point at `cursor_px` in the camera's render frame, or `None` when the ray
    /// finds nothing — where an armed orbit-center placement would land.
    ///
    /// Both tiers of the armed-tool drop, asked in the same order: geometry first via the
    /// selection click's own ray, so the point the camera will turn about is the point a click
    /// there would have selected, then the visible built-in world planes via
    /// [`world_plane_target`](crate::AppCore::world_plane_target) — one shared implementation,
    /// so a click cannot drop a node on the ground and place a pivot somewhere else.
    ///
    /// A miss is a REFUSAL, not a fallback. This used to answer `camera.target` when the ray
    /// found nothing, which made placing over sky or an empty scene silently equivalent to not
    /// placing at all — the failure was invisible precisely because the fallback was plausible.
    /// The gizmo simply does not draw on a miss, and the click does not commit.
    ///
    /// The point is CONTINUOUS, not a voxel center: a pivot is a camera quantity with no lattice
    /// meaning, and a snapped one visibly jumps a whole cell at a time under the cursor.
    pub(super) fn surface_point_at(&self, cursor_px: Option<(f64, f64)>) -> Option<glam::Vec3> {
        let (cursor_x, cursor_y) = cursor_px?;
        let density = self.panel_state.geometry.voxels_per_block;
        let [vx, vy, vw, vh] = self.last_viewport_px;
        let recenter = self.recenter_voxels.voxels();
        let frame = crate::PickFrame {
            region_dimensions: self.region_dimensions,
            recenter_voxels: recenter,
            density,
            chunks: &self.resident_chunks,
            band: self.last_pick_band,
        };
        let cursor = [cursor_x as f32, cursor_y as f32];
        let viewport = [vx as f32, vy as f32, vw as f32, vh as f32];
        // Both tiers answer in ABSOLUTE voxels; the camera lives in the RECENTERED render frame,
        // so the point rebases once here (the recenter is carried, this is the only
        // conversion).
        let absolute = self.app_core.surface_point_absolute(
            cursor,
            viewport,
            &frame,
            &self.panel_state.scene,
            self.panel_state.scene.master_floor_grid,
        )?;
        Some(absolute - glam::Vec3::new(recenter[0] as f32, recenter[1] as f32, recenter[2] as f32))
    }

    /// The [`SelectionTarget`](ui::panel::SelectionTarget) under the cursor
    /// (physical px) inside `sketch`, or `None` over empty space. The order is dot, then badge,
    /// then lever, then edge — vertices take priority over segments as everywhere, and the badge's
    /// place in it is argued at the branch. The ONE place a sketch target is minted, which is what
    /// makes the shell's admission `debug_assert` hold by construction.
    fn sketch_entity_target_at(
        &self,
        sketch: document::scene::NodeId,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<ui::panel::SelectionTarget> {
        // A DOT beats a badge, and everything else loses to one.
        //
        // Badges paint last, so the rule was that a badge wins whatever it covers — but an
        // unpicked badge draws its glyph and nothing else, no plate and no fill, so a 32-point box
        // is mostly see-through and a dot inside it is fully visible. Handing that pixel to the
        // badge picks something the cursor is demonstrably NOT on top of, which is the failure the
        // paint-order rule was written to prevent, in the other direction.
        //
        // A dot is allowed to take the bite because a dot is BOUNDED: 20 points across, out of a
        // box of 32, and a badge floats 30 points off the point it names — so a badge never loses
        // ground to its own anchor beyond a sliver at one corner, and stays reachable everywhere
        // else. A lever's stick and an edge are not bounded; they would run clean across a badge
        // and cut it in two, so they stay below it.
        if let Some(index) = self.sketch_vertex_at(cursor_x, cursor_y) {
            if let Some(&entity) = self.sketch_point_ids.get(index) {
                return Some(ui::panel::SelectionTarget::SketchPoint { sketch, entity });
            }
        }
        if let Some(entity) = self.sketch_constraint_at(cursor_x, cursor_y) {
            return Some(ui::panel::SelectionTarget::SketchConstraint { sketch, entity });
        }
        // A lever answers as its fit point, and beats the curve underneath it — it is drawn over
        // that curve precisely because it is the thing being reached for.
        if let Some(entity) = self.tangent_lever_at(cursor_x, cursor_y) {
            return Some(ui::panel::SelectionTarget::SketchPoint { sketch, entity });
        }
        self.nearest_sketch_edge(cursor_x, cursor_y)
            .map(|hit| match hit {
                SketchEdgeHit::Segment(entity) => {
                    ui::panel::SelectionTarget::SketchSegment { sketch, entity }
                }
                SketchEdgeHit::Arc(entity) => {
                    ui::panel::SelectionTarget::SketchArc { sketch, entity }
                }
                SketchEdgeHit::Circle(entity) => {
                    ui::panel::SelectionTarget::SketchCircle { sketch, entity }
                }
                SketchEdgeHit::HigherCurve(curve) => {
                    ui::panel::SelectionTarget::SketchHigherCurve { sketch, curve }
                }
            })
    }

    /// Is the cursor (physical px) over a sketch entity — a vertex or a segment? Used by
    /// the right-click handler to tell a sketch handle (which registers as chrome so a LEFT press
    /// drags it) from the real Signal chrome, so a right-click on an entity opens the context menu
    /// even though the handle sits in the chrome hit-set.
    pub(super) fn cursor_over_sketch_entity(&self, cursor_x: f64, cursor_y: f64) -> bool {
        self.sketch_vertex_at(cursor_x, cursor_y).is_some()
            || self.nearest_sketch_edge(cursor_x, cursor_y).is_some()
            || self.tangent_lever_at(cursor_x, cursor_y).is_some()
            || self.sketch_constraint_at(cursor_x, cursor_y).is_some()
    }

    /// A right-click over a sketch entity selects it (Fusion: right-clicking an entity
    /// acts on it). If the entity is already in the selection the whole set is kept — so
    /// right-clicking one of several selected entities deletes them all — otherwise the selection is
    /// replaced with just that entity. Vertices take priority over segments, as everywhere.
    pub(super) fn right_click_select_sketch_entity(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(sketch) = self.panel_state.sketch_mode else {
            return;
        };
        if let Some(target) = self.sketch_entity_target_at(sketch, cursor_x, cursor_y) {
            if !self.panel_state.selection.contains(target) {
                self.panel_state.selection.select_only(target);
            }
        }
    }

    /// Remove what is picked — the one implementation of the Delete command, reached from the
    /// viewport menu's row and from its keyboard binding alike.
    ///
    /// What "picked" means depends on where you are, and deciding that HERE is the point: inside a
    /// sketch it is the picked entities, outside one it is the picked nodes. The panel cannot make
    /// that call for the keyboard path, which arrives with no menu to have been built in a mode.
    /// A no-op when nothing is picked, so the binding is safe to press at any time.
    ///
    /// A multi-selection deletes whole subtrees, filtered to its **selection roots**: a node
    /// whose ancestor is also picked is skipped, because removing the ancestor takes it anyway
    /// and a second `RemoveNode` on the dead id would ride the transaction as a no-op. The batch
    /// is ONE undo step, mirroring the sketch multi-delete.
    pub(super) fn delete_selection(&mut self) {
        if self.panel_state.sketch_mode.is_some() {
            self.delete_sketch_selection();
            return;
        }
        let picked: std::collections::BTreeSet<document::scene::NodeId> =
            self.panel_state.selection.nodes().collect();
        let scene = &self.panel_state.scene;
        let has_picked_ancestor = |id: document::scene::NodeId| {
            let mut current = id;
            while let Some((Some(parent), _)) = scene.parent_and_index_of(current) {
                if picked.contains(&parent) {
                    return true;
                }
                current = parent;
            }
            false
        };
        let intents: Vec<crate::Intent> = self
            .panel_state
            .selection
            .nodes()
            .filter(|id| !has_picked_ancestor(*id))
            .map(|target| crate::Intent::RemoveNode { target })
            .collect();
        if !intents.is_empty() {
            self.viewport_transactions.push(intents);
        }
    }

    /// Delete every entity in the sketch selection as ONE edit — each selected point
    /// (cascading its incident segments and arcs) then each selected segment and arc (a no-op if
    /// a cascade already took it), committed through the same anchor-preserving path a single
    /// delete uses
    /// ([`commit_sketch_profile_edit`](Self::commit_sketch_profile_edit)), then the selection is
    /// cleared. No-op when nothing is picked or no sketch is being edited. Invoked by the general
    /// viewport context menu's Delete.
    ///
    /// Constraints go LAST, and that ordering is the whole subtlety: deleting a point cascades
    /// into the constraints that named it, so a constraint picked alongside its own geometry is
    /// already gone by the time its turn comes. `with_constraint_deleted` is a no-op on an id that
    /// no longer resolves, which is what lets one pass cover both cases without asking which
    /// happened.
    pub(super) fn delete_sketch_selection(&mut self) {
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        if !self.panel_state.selection.holds_sketch_entities(target) {
            return;
        }
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        let points: Vec<_> = self.panel_state.selection.sketch_points(target).collect();
        let segments: Vec<_> = self.panel_state.selection.sketch_segments(target).collect();
        let arcs: Vec<_> = self.panel_state.selection.sketch_arcs(target).collect();
        let circles: Vec<_> = self.panel_state.selection.sketch_circles(target).collect();
        let higher_curves: Vec<_> = self
            .panel_state
            .selection
            .sketch_higher_curves(target)
            .collect();
        let constraints: Vec<_> = self
            .panel_state
            .selection
            .sketch_constraints(target)
            .collect();
        let mut next = producer;
        for point_id in points {
            next = next.with_point_deleted(point_id);
        }
        for seg_id in segments {
            next = next.with_segment_deleted(seg_id);
        }
        for arc_id in arcs {
            next = next.with_arc_deleted(arc_id);
        }
        for circle_id in circles {
            next = next.with_circle_deleted(circle_id);
        }
        for curve in higher_curves {
            next = next.with_curve_deleted(curve);
        }
        for constraint_id in constraints {
            next = next.with_constraint_deleted(constraint_id);
        }
        self.commit_sketch_profile_edit(target, next);
        self.panel_state.selection.clear_sketch_entities();
    }

    /// Flip the selected sketch geometry between real and construction as one undoable edit.
    /// Constraints are selection entities too but are intentionally excluded; structural arc and
    /// circle centers are filtered again by the document invariant.
    /// Restate a picked dimension's value, and keep the author's hold on it.
    ///
    /// The restatement releases the old assertion and makes the new one, so the constraint comes
    /// back with a NEW id — and the selection is re-pointed at it, because the rail field the
    /// author just typed in is drawn only while its dimension is picked. Losing the selection on
    /// commit would make the field vanish out from under the value it had just accepted.
    ///
    /// A refused number leaves the drawing and the selection exactly as they were, and says why
    /// on the viewport notice — the same place a refused constraint says it.
    fn restate_sketch_dimension(
        &mut self,
        constraint: document::sketch::EntityId,
        restated: document::sketch::Dimension,
    ) {
        let (Some(target), Some(context)) = (
            self.panel_state.sketch_mode,
            self.sketch_evaluation_context(),
        ) else {
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        match producer.with_dimension_restated(constraint, restated, context) {
            Ok((next, id)) => {
                self.commit_sketch_profile_edit(target, next);
                self.panel_state.selection.clear_sketch_entities();
                self.panel_state
                    .selection
                    .toggle(ui::panel::SelectionTarget::SketchConstraint {
                        sketch: target,
                        entity: id,
                    });
            }
            Err(refusal) => {
                self.panel_state.sketch_constraint_refusal = Some(refusal_text(&refusal));
                select_sketch_constraint_refusal_culprits(
                    &mut self.panel_state.selection,
                    target,
                    &refusal,
                );
            }
        }
    }

    pub(super) fn toggle_sketch_selection_construction(&mut self) {
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        // Points are absent deliberately: construction is a mode a CURVE is in, and offering it on
        // a selected point would flip a lifetime flag the author never asked about.
        let entities: Vec<_> = self
            .panel_state
            .selection
            .sketch_segments(target)
            .chain(self.panel_state.selection.sketch_arcs(target))
            .chain(self.panel_state.selection.sketch_circles(target))
            // An aggregate toggles as one entity: the role lives on the authored curve, not on
            // the spans it happens to resolve to.
            .chain(
                self.panel_state
                    .selection
                    .sketch_higher_curves(target)
                    .map(document::sketch::SketchCurve::id),
            )
            .collect();
        let Some(next) = producer.with_construction_toggled(entities) else {
            return;
        };
        self.commit_sketch_profile_edit(target, next);
    }

    /// The dimension gizmos to draw next frame: one measured mark per authored quantity.
    ///
    /// A dimension is the one relation that does NOT get a badge — the number is the mark, and a
    /// glyph beside it would say the same thing twice. This is the "instead".
    ///
    /// Everything is projected through the same plane-to-screen path the arcs are tessellated
    /// with, so a gizmo tracks its geometry through every camera move. A dimension whose geometry
    /// went behind the camera simply has no gizmo, which is the rule the badges and the drawing
    /// both already follow.
    fn refresh_sketch_dimension_gizmos(
        &mut self,
        target: document::scene::NodeId,
        view_projection: glam::Mat4,
        viewport_px: [u32; 4],
        pixels_per_point: f32,
    ) {
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        let density = self.panel_state.geometry.voxels_per_block;
        let (Some(context), Some(handles)) = (
            self.sketch_evaluation_context(),
            self.panel_state.scene.sketch_handles(target, density),
        ) else {
            return;
        };
        let [vx, vy, vw, vh] = viewport_px.map(|value| value as f32);
        let clip_of = |coord: [f64; 2]| {
            let vertex = handles.profile_to_render(coord);
            view_projection * glam::Vec4::new(vertex[0], vertex[1], vertex[2], 1.0)
        };
        // The plane's whole projection as one matrix — see [`a_sketch_planes_frame`]. A dimension
        // needs more of it than a point does: not only where a plane coordinate lands but which way
        // the plane runs at any PIXEL and how far a plane unit reaches there, and only the matrix
        // has those.
        let Some(plane) = a_sketch_planes_frame(&clip_of, [vx, vy, vw, vh], pixels_per_point)
        else {
            return;
        };
        // The plane-to-screen door, in EGUI POINTS — the gizmo modules lay out in the same units
        // egui paints in, so the conversion belongs here rather than at every call. Read off the
        // frame rather than struck again beside it: two expressions of one projection is two
        // things to keep agreeing, and the value's own layout would be the one that drifted.
        let to_px = |coord: [f64; 2]| plane.at(coord);
        let in_plane = |id: document::sketch::EntityId| {
            producer
                .sketch
                .points()
                .iter()
                .find(|point| point.id == id)
                .map(|point| point.at.in_plane())
        };
        // Which way a PLANE direction runs on screen. The step is taken in the plane and only then
        // projected, so what comes back is the image of a plane line rather than a direction the
        // screen invented for itself. The distinction is the whole of this pass, and
        // [`a_plane_direction_on_screen`] is where it is argued.
        let along_the_plane =
            |at: [f64; 2], step: [f64; 2]| a_plane_direction_on_screen(at, step, &to_px);
        let the_line_through = |at: [f64; 2], toward: [f64; 2], anchor: egui::Pos2| {
            a_dimension_lines_direction(at, toward, anchor, &to_px)
        };
        // Everything a dimension needs of a rim: where it stands on screen, how much of itself it
        // draws, and the one nominal radius the layout reasons in. Asked once per drawing, because
        // projecting the ring is the expensive part and all three answers come off it.
        let dimensioned_rim = |curve| {
            let form = producer.sketch.circular_form(curve, context)?;
            let standing = ProjectedRim::project(form.center, form.radius, &to_px)?;
            let (from, turn) = drawn_turn(&producer.sketch, curve, standing.center, &to_px)?;
            let radius_px = standing.mean_reach();
            Some((standing, from, turn, radius_px))
        };
        let ends_in_plane = |id: document::sketch::EntityId| {
            let held = producer
                .sketch
                .segments()
                .iter()
                .find(|held| held.id == id)?;
            Some((in_plane(held.from)?, in_plane(held.to)?))
        };
        // What an angle's arm draws, as two plane points a line can be struck through. A straight
        // arm is its own two ends; an arc arm is the TANGENT at the end it names, given the arc's
        // own radius as a length so the leg reaches about as far as the curve does.
        let arm_line = |arm: document::sketch::AngleArm| match arm {
            document::sketch::AngleArm::Segment { segment } => ends_in_plane(segment),
            document::sketch::AngleArm::ArcEnd { arc, end } => {
                let held = producer.sketch.arcs().iter().find(|held| held.id == arc)?;
                let standing = match end {
                    document::sketch::ArcEnd::From => held.from,
                    document::sketch::ArcEnd::To => held.to,
                };
                let (at, center) = (in_plane(standing)?, in_plane(held.center)?);
                let radius = [at[0] - center[0], at[1] - center[1]];
                let reach = radius[0].hypot(radius[1]);
                (reach > f64::EPSILON).then(|| (at, [at[0] - radius[1], at[1] + radius[0]]))
            }
        };
        let voxels = |length: document::sketch::SketchLength| {
            parametric::units::format(
                length.value().round() as i64,
                density,
                parametric::units::DisplayUnit::BlocksAndVoxels,
            )
        };

        // How one dimension lays itself out, given where its annotation sits. Shared by the
        // committed ones and by the ghost the author is dragging, because a preview drawn any
        // other way is a preview that can disagree with what the click will actually produce.
        //
        // `placed` is `None` for a dimension authored before the annotation had a place of its
        // own; each member then falls back to the position the renderer used to invent, so an old
        // drawing opens looking exactly as it did.
        let lay_out = |dimension: document::sketch::Dimension,
                       placed: Option<egui::Pos2>|
         -> Option<ui::gizmos::dimension::Drawing> {
            // Every authored dimension DRIVES: the family exists for quantities the author states
            // and the solver honours. Reference rank is reserved for a derived readout, which is a
            // different thing the drawing does not yet offer.
            let rank = ui::gizmos::dimension::Rank::Driving;
            let drawing = match dimension {
                document::sketch::Dimension::Span { from, to, length } => {
                    let (Some(tail), Some(head)) = (in_plane(from), in_plane(to)) else {
                        return None;
                    };
                    let (Some(from), Some(to)) = (to_px(tail), to_px(head)) else {
                        return None;
                    };
                    let run = to - from;
                    // Square to the run IN THE PLANE, then projected — the direction the extension
                    // lines stand along and the one the drawing steps off by. Asked at EACH end,
                    // because a projection that divides carries one plane direction to a different
                    // screen direction at every point.
                    let square = [tail[1] - head[1], head[0] - tail[0]];
                    let (Some(at_tail), Some(at_head)) =
                        (along_the_plane(tail, square), along_the_plane(head, square))
                    else {
                        return None;
                    };
                    if a_plane_too_edge_on_to_dimension(run, [at_tail, at_head]) {
                        return None;
                    }
                    // Where the author put the text IS the placement, in both directions at once:
                    // how far off the run the dimension line sits, and how far along it the value
                    // rides. Unplaced it stands off ABOVE the span on screen whichever way the run
                    // is drawn, so two spans on one drawing do not sit on opposite sides of
                    // geometry that merely happens to have been drawn the other way round.
                    let anchor = placed.unwrap_or_else(|| {
                        let side = if at_tail.y > 0.0 { -1.0 } else { 1.0 };
                        // Two halves of one convention, and they answer to different things. WHICH
                        // SIDE is chrome and stays the screen's, as the comment above says. HOW FAR
                        // ALONG names a place on the drawing, so it is the plane's middle.
                        plane.along(from, to, 0.5) + at_tail * side * DIMENSION_STANDOFF_PX
                    });
                    let along =
                        the_line_through(tail, [head[0] - tail[0], head[1] - tail[1]], anchor)?;
                    ui::gizmos::dimension::axis_span(
                        from,
                        to,
                        along,
                        [at_tail, at_head],
                        anchor,
                        plane,
                        &voxels(length),
                        rank,
                    )
                }
                document::sketch::Dimension::SpanAlong {
                    from,
                    to,
                    axis,
                    length,
                } => {
                    let (Some(tail), Some(head)) = (in_plane(from), in_plane(to)) else {
                        return None;
                    };
                    // The direction measured in is the PLANE's axis, not the screen's: a sketch
                    // drawn on a plane the camera is not square to still measures its width across
                    // the plane, and a screen-axis line would be measuring the camera.
                    let mut run = [0.0; 2];
                    run[axis.coordinate()] = 1.0;
                    // The plane's OTHER direction, which is what square to this one means here —
                    // the two coordinates of a plane are perpendicular in it whatever the camera
                    // makes of them. Asked at each end, because a dividing projection answers a
                    // plane direction differently at every point.
                    let mut square = [0.0; 2];
                    square[1 - axis.coordinate()] = 1.0;
                    let (Some(from), Some(to)) = (to_px(tail), to_px(head)) else {
                        return None;
                    };
                    let (Some(run_at_tail), Some(at_tail), Some(at_head)) = (
                        along_the_plane(tail, run),
                        along_the_plane(tail, square),
                        along_the_plane(head, square),
                    ) else {
                        return None;
                    };
                    if a_plane_too_edge_on_to_dimension(run_at_tail, [at_tail, at_head]) {
                        return None;
                    }
                    let anchor = placed.unwrap_or_else(|| plane.along(from, to, 0.5));
                    let along = the_line_through(tail, run, anchor)?;
                    ui::gizmos::dimension::axis_span(
                        from,
                        to,
                        along,
                        [at_tail, at_head],
                        anchor,
                        plane,
                        &voxels(length),
                        rank,
                    )
                }
                document::sketch::Dimension::Gap {
                    point,
                    segment,
                    length,
                } => {
                    let (Some(standing), Some((tail, head))) =
                        (in_plane(point), ends_in_plane(segment))
                    else {
                        return None;
                    };
                    let run = [head[0] - tail[0], head[1] - tail[1]];
                    let reach = run[0].hypot(run[1]);
                    if reach <= f64::EPSILON {
                        return None;
                    }
                    // The point's own place on the line, so the drawing hangs off the nearest part
                    // of the run rather than off whichever end the author happened to draw first.
                    // Dropped IN THE PLANE: where a perpendicular lands is a fact about the
                    // drawing, and finding it after the projection would make it one about the
                    // camera — the two disagree by the tilt.
                    let carried = ((standing[0] - tail[0]) * run[0]
                        + (standing[1] - tail[1]) * run[1])
                        / (reach * reach);
                    let stands = [tail[0] + run[0] * carried, tail[1] + run[1] * carried];
                    let (Some(stood), Some(foot)) = (to_px(standing), to_px(stands)) else {
                        return None;
                    };
                    // The dimension line lies ACROSS the line, which is the one direction the
                    // distance is measured in. Each end then reaches it by running parallel to the
                    // line — which for the line's own end is the line carried on, and for the point
                    // is the same rule read the other way. Both directions are the plane's, not the
                    // screen's.
                    let square = [-run[1], run[0]];
                    let (Some(square_at_point), Some(at_point), Some(at_foot)) = (
                        along_the_plane(standing, square),
                        along_the_plane(standing, run),
                        along_the_plane(stands, run),
                    ) else {
                        return None;
                    };
                    if a_plane_too_edge_on_to_dimension(square_at_point, [at_point, at_foot]) {
                        return None;
                    }
                    let anchor = placed.unwrap_or_else(|| {
                        stood + (foot - stood) / 2.0 + at_point * DIMENSION_STANDOFF_PX
                    });
                    let measured = the_line_through(standing, square, anchor)?;
                    ui::gizmos::dimension::axis_span(
                        stood,
                        foot,
                        measured,
                        [at_point, at_foot],
                        anchor,
                        plane,
                        &voxels(length),
                        rank,
                    )
                }
                document::sketch::Dimension::RimGap {
                    first,
                    second,
                    length,
                } => {
                    let (first_standing, first_from, first_turn, first_px) =
                        dimensioned_rim(first)?;
                    let (second_standing, second_from, second_turn, second_px) =
                        dimensioned_rim(second)?;
                    let first_rim = ui::gizmos::dimension::Rim {
                        from: first_from,
                        turn: first_turn,
                        at: &|bearing| first_standing.touch(bearing),
                    };
                    let second_rim = ui::gizmos::dimension::Rim {
                        from: second_from,
                        turn: second_turn,
                        at: &|bearing| second_standing.touch(bearing),
                    };
                    let center = first_standing.center;
                    let anchor = placed
                        .unwrap_or_else(|| default_rim_anchor(center, first_px.max(second_px)));
                    // The bearing the annotation was dropped at is the radius the gap is measured
                    // out along, clamped to where each rim is actually drawn — asked of each in
                    // turn, so a bearing NEITHER reaches settles on the second's end and the
                    // drawing hangs off a rim rather than floating past where anything is drawn.
                    let reach = anchor - center;
                    if reach.length() <= f32::EPSILON {
                        return None;
                    }
                    let bearing =
                        second_rim.nearest_drawn(first_rim.nearest_drawn(reach.y.atan2(reach.x)));
                    let out = egui::vec2(bearing.cos(), bearing.sin());
                    // The dimension line runs ALONG that radius, so each extension line lies on
                    // the tangent at the rim it leaves — the same drawing a gap across a line
                    // makes, read on a curve. Each end is asked of its OWN rim: on a plane the
                    // camera is not square to, a screen radius is right in one direction only.
                    // Each extension leaves its rim along the TANGENT there — the projected
                    // plane tangent, which the rim answers directly, rather than the screen's
                    // square of the radius.
                    let facing = [first_rim.tangent(bearing), second_rim.tangent(bearing)];
                    ui::gizmos::dimension::axis_span(
                        first_rim.touch(bearing),
                        second_rim.touch(bearing),
                        out,
                        facing,
                        anchor,
                        plane,
                        &voxels(length),
                        rank,
                    )
                }
                document::sketch::Dimension::Radius { curve, length } => {
                    let (standing, from, turn, radius_px) = dimensioned_rim(curve)?;
                    let center = standing.center;
                    let anchor = placed.unwrap_or_else(|| default_rim_anchor(center, radius_px));
                    ui::gizmos::dimension::radius(
                        center,
                        radius_px,
                        anchor,
                        ui::gizmos::dimension::Rim {
                            from,
                            turn,
                            at: &|bearing| standing.touch(bearing),
                        },
                        plane,
                        &voxels(length),
                        rank,
                    )
                }
                document::sketch::Dimension::Diameter { curve, length } => {
                    let (standing, from, turn, radius_px) = dimensioned_rim(curve)?;
                    let center = standing.center;
                    let anchor = placed.unwrap_or_else(|| default_rim_anchor(center, radius_px));
                    ui::gizmos::dimension::diameter(
                        center,
                        radius_px,
                        anchor,
                        ui::gizmos::dimension::Rim {
                            from,
                            turn,
                            at: &|bearing| standing.touch(bearing),
                        },
                        plane,
                        &voxels(length),
                        rank,
                    )
                }
                document::sketch::Dimension::Angle {
                    first,
                    second,
                    degrees,
                    corner,
                } => {
                    let (Some(first), Some(second)) = (arm_line(first), arm_line(second)) else {
                        return None;
                    };
                    // The vertex is found in PLANE coordinates and only then projected: a virtual
                    // intersection is a fact about the drawing, and finding it after a perspective
                    // divide would make it a fact about the camera.
                    let corner_at =
                        substrate::geom2d::line_intersection(first.0, first.1, second.0, second.1)?;
                    let vertex = to_px(corner_at)?;
                    let (from, to, legs) =
                        angle_legs(vertex, &to_px, first, second, corner, placed)?;
                    // The shorter leg sets the default arc: it is the one that decides whether an
                    // extension line is needed at all.
                    let reach = legs[0].furthest.min(legs[1].furthest);
                    let radius = angle_arc_radius(vertex, placed, reach);
                    // The arc is struck IN THE PLANE and projected, so it comes out the ellipse the
                    // plane actually draws rather than a circle about the vertex's image. The
                    // radius is asked for in screen points, so it is first turned into the plane
                    // length that projects to about that much, measured off a unit ring.
                    //
                    // Along ONE bearing, not the ring's mean: a projection reaches a different
                    // distance in every direction, and the bearing that matters is the one the
                    // screen radius was asked at — the anchor's own, so the arc still runs under
                    // the text the author dropped. Unplaced there is no such bearing and the mean
                    // stands in.
                    let unit = ProjectedRim::project(corner_at, 1.0, &to_px)?;
                    let per_unit = placed
                        .map(|at| at - vertex)
                        .filter(|reach| reach.length() > f32::EPSILON)
                        .map_or_else(
                            || unit.mean_reach(),
                            |reach| unit.center.distance(unit.touch(reach.y.atan2(reach.x))),
                        );
                    if per_unit <= f32::EPSILON {
                        return None;
                    }
                    let struck =
                        ProjectedRim::project(corner_at, f64::from(radius / per_unit), &to_px)?;
                    let value = format!("{}\u{b0}", trim_number(degrees.to_degrees_f64()));
                    ui::gizmos::dimension::angle(
                        vertex,
                        from,
                        to,
                        radius,
                        ui::gizmos::dimension::Rim {
                            from: 0.0,
                            turn: std::f32::consts::TAU,
                            at: &|bearing| struck.touch(bearing),
                        },
                        legs,
                        plane,
                        &value,
                        rank,
                    )
                }
            };
            Some(drawing)
        };

        for constraint in producer.sketch.constraints() {
            let document::sketch::ConstraintKind::Dimension(dimension) = constraint.kind else {
                continue;
            };
            let Some(drawing) = lay_out(dimension, constraint.anchor.and_then(&to_px)) else {
                continue;
            };
            self.sketch_dimension_gizmos
                .push(ui::chrome::DimensionGizmo {
                    drawing,
                    constraint: Some(constraint.id),
                    picked: self.panel_state.selection.contains(
                        ui::panel::SelectionTarget::SketchConstraint {
                            sketch: target,
                            entity: constraint.id,
                        },
                    ),
                });
        }

        // The one being placed right now, following the cursor. It goes through `lay_out` like
        // every committed dimension, so what the author is dragging IS what the click will make —
        // and it carries no id, so it cannot be picked or deleted while it is still a question.
        let hovering = self
            .last_cursor_position
            .and_then(|(x, y)| self.sketch_unsnapped_profile_coord(x, y));
        let ghost = self
            .panel_state
            .armed_constraint
            .as_ref()
            .filter(|armed| armed.is_placing())
            .zip(hovering)
            .and_then(|(armed, at)| {
                let document::sketch::ConstraintKind::Dimension(dimension) = armed
                    .dimension_dropped_at(at, &producer.sketch, context)
                    .ok()?
                else {
                    return None;
                };
                lay_out(dimension, to_px(at))
            });
        if let Some(drawing) = ghost {
            self.sketch_dimension_gizmos
                .push(ui::chrome::DimensionGizmo {
                    drawing,
                    constraint: None,
                    picked: false,
                });
        }
    }

    /// The constraint badges to draw next frame: one glyph per asserted
    /// relation, anchored on the geometry the relation NAMES.
    ///
    /// The anchor comes from the constraint's entity ids resolved through the same projected
    /// arrays the handles and lines use, so a badge cannot drift from its entity — it is placed
    /// by the entity graph, not beside it. A segment's badge sits off the midpoint along the
    /// edge normal (perpendicular is the only offset that reads as "about this line" at every
    /// angle); a point's sits up and to the right, where a lock hangs in every CAD tool.
    ///
    /// **Every one of those directions is read IN THE SKETCH PLANE.** A constraint symbol is
    /// notation on the drawing the way a dimension's number is, so square-to-a-segment is the
    /// plane's square and not the screen's — 31 degrees apart at a three-quarter view — and the
    /// point convention is the plane's own up-and-right rather than a fixed screen diagonal that
    /// slides off the paper as the camera turns. The offset LENGTH stays in pixels: direction
    /// from the plane, clearance from the screen, which is the same split the value of a
    /// dimension makes when it lifts off its line.
    ///
    /// Several badges on one anchor step further along that same offset rather than overprinting.
    /// A constraint naming geometry that is off-screen or behind the camera simply has no badge:
    /// the drawing is what carries them.
    fn refresh_sketch_constraint_badges(
        &mut self,
        target: document::scene::NodeId,
        view_projection: glam::Mat4,
        viewport_px: [u32; 4],
        pixels_per_point: f32,
    ) {
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        let context = self.sketch_evaluation_context();
        let handles = self
            .panel_state
            .scene
            .sketch_handles(target, self.panel_state.geometry.voxels_per_block);
        let at = |index: usize| -> Option<egui::Pos2> {
            let px = (*self.sketch_vertex_px.get(index)?)?;
            Some(egui::Pos2::new(
                px.x / pixels_per_point,
                px.y / pixels_per_point,
            ))
        };
        let point_index = |id: document::sketch::EntityId| {
            self.sketch_point_ids.iter().position(|held| *held == id)
        };
        // The plane's whole projection as one matrix, so a badge can ask it for a direction at
        // any point in the viewport — including the standoff positions, which are pixel offsets
        // and so have no sketch coordinate of their own. `facing` where the sketch has no
        // placement yet or the plane is edge-on: the flat-page reading, which is what every one
        // of these conventions used to be unconditionally.
        let [vx, vy, vw, vh] = viewport_px.map(|value| value as f32);
        let plane = handles
            .as_ref()
            .and_then(|handles| {
                let clip_of = |coord: [f64; 2]| {
                    let vertex = handles.profile_to_render(coord);
                    view_projection * glam::Vec4::new(vertex[0], vertex[1], vertex[2], 1.0)
                };
                a_sketch_planes_frame(&clip_of, [vx, vy, vw, vh], pixels_per_point)
            })
            .unwrap_or_else(ui::gizmos::dimension::PlaneFrame::facing);
        // The plane's own up-and-right at a point: the image of its 45 degree diagonal, which
        // square-on is exactly the screen diagonal this replaced.
        let out_of_the_corner = move |here: egui::Pos2| {
            let reading = plane.reading_at(here);
            let diagonal = reading + plane.square_to(reading, here);
            if diagonal.length() > f32::EPSILON {
                diagonal.normalized()
            } else {
                egui::vec2(0.707, -0.707)
            }
        };
        // How many badges already stand on this anchor, so the next one steps clear of them.
        let mut stacked: std::collections::HashMap<[u32; 2], f32> =
            std::collections::HashMap::new();

        // A badge stands at a segment's midpoint, offset along that segment's normal.
        let beside_segment = |segment: document::sketch::EntityId| {
            let held = self
                .sketch_segments
                .iter()
                .find(|held| held.entity == segment)?;
            let (a, b) = (at(held.from)?, at(held.to)?);
            let along = b - a;
            let length = along.length();
            if length < f32::EPSILON {
                return None;
            }
            // The middle of the SEGMENT, not the middle of its image. Those are different
            // points under a projection that divides, by 28 pixels at an ordinary three-quarter
            // view against a badge 32 wide — see `PlaneFrame::along`.
            let middle = plane.along(a, b, 0.5);
            // Square to the segment IN THE PLANE. The sign holds the side the badge has always
            // stood on: `square_to` agrees with the OTHER perpendicular, so it is turned back.
            let square = -plane.square_to(along / length, middle);
            Some((
                middle,
                if square.length() > f32::EPSILON {
                    square.normalized()
                } else {
                    egui::vec2(-along.y, along.x) / length
                },
            ))
        };
        // A badge on a point sits up and to the right of it — there is no geometry to take a
        // normal from, so the direction is a convention, read in the plane the point lies in.
        let beside_point = |id: document::sketch::EntityId| {
            let here = at(point_index(id)?)?;
            Some((here, out_of_the_corner(here)))
        };
        let ends_of = |segment: document::sketch::EntityId| {
            self.sketch_segments
                .iter()
                .find(|held| held.entity == segment)
                .map(|held| (held.from, held.to))
        };
        // Where two segments MEET, offset into the angle they make — the square that the mark
        // asserts is the one the badge is sitting in. `None` when they share no endpoint.
        let inside_the_corner = |first, second| {
            let (first_from, first_to) = ends_of(first)?;
            let (second_from, second_to) = ends_of(second)?;
            let corner = [first_from, first_to]
                .into_iter()
                .find(|end| *end == second_from || *end == second_to)?;
            let here = at(corner)?;
            let arm = |(from, to): (usize, usize)| {
                let away = at(if from == corner { to } else { from })? - here;
                let length = away.length();
                (length > f32::EPSILON).then_some(away / length)
            };
            let (first_arm, second_arm) =
                (arm((first_from, first_to))?, arm((second_from, second_to))?);
            // Bisected IN THE PLANE: the halfway direction between two projected plane lines is
            // not the halfway direction on screen, because the angle between them is not the
            // angle the plane holds them at.
            // Doubling back means the two arms run along one line, which nothing perpendicular
            // can look like — fall back to the point convention rather than to a zero vector.
            Some((
                here,
                plane
                    .bisector_of(first_arm, second_arm, here)
                    .unwrap_or_else(|| out_of_the_corner(here)),
            ))
        };

        for constraint in producer.sketch.constraints() {
            // **A relation with no locus gets a badge on EVERY member.** One badge on a two-segment relation
            // would read as belonging to whichever segment it stood beside, and the whole job of
            // the mark is to say which geometry is bound to which — a single mark on one member
            // leaves the other looking free. They share the constraint id, so a click on either
            // picks the one relation.
            let placements: Vec<(egui::Pos2, egui::Vec2)> = match constraint.kind {
                document::sketch::ConstraintKind::Horizontal { segment }
                | document::sketch::ConstraintKind::Vertical { segment } => {
                    beside_segment(segment).into_iter().collect()
                }
                document::sketch::ConstraintKind::Fix { point, .. } => {
                    beside_point(point).into_iter().collect()
                }
                document::sketch::ConstraintKind::Quantize { point, .. } => {
                    beside_point(point).into_iter().collect()
                }
                // Both mark the POINT, which is the thing being placed — and for a coincidence
                // between two points that also settles where the one badge goes, since the pair
                // ends up in one place and a mark on each would overprint.
                document::sketch::ConstraintKind::Midpoint { point, .. }
                | document::sketch::ConstraintKind::Coincident { point, .. } => {
                    beside_point(point).into_iter().collect()
                }
                // Curvature marks the JOINT and not the neighbour curve: the joint is the one
                // place the claim is about, and the curve it runs out of may be long enough that a
                // badge on it would land nowhere near the smoothness it is describing.
                document::sketch::ConstraintKind::Curvature { joint, .. } => {
                    beside_point(joint).into_iter().collect()
                }
                // Perpendicular has a LOCUS, and the rule above does not apply to it: two lines
                // meeting square make ONE right angle, and the mark belongs in it. Two badges at
                // two midpoints say the same thing twice and neither says where the corner is
                // (owner 2026-07-31). Segments that never meet keep the per-member placement —
                // there is no angle to stand in, and the relation still binds both.
                document::sketch::ConstraintKind::Perpendicular { first, second } => {
                    match inside_the_corner(first, second) {
                        Some(corner) => vec![corner],
                        None => [first, second]
                            .into_iter()
                            .filter_map(beside_segment)
                            .collect(),
                    }
                }
                document::sketch::ConstraintKind::Parallel { first, second }
                | document::sketch::ConstraintKind::Equal { first, second }
                | document::sketch::ConstraintKind::Collinear { first, second } => [first, second]
                    .into_iter()
                    .filter_map(beside_segment)
                    .collect(),
                // A dimension draws as a dimension gizmo, not a badge — the number IS the mark,
                // and a glyph beside it would say the same thing twice. True of every member of
                // the family, so this asks the family and not its members.
                document::sketch::ConstraintKind::Dimension(_) => Vec::new(),
                // Tangent has one derived, finite-domain-validated contact, so it gets one badge
                // at that locus rather than one duplicate mark per member curve.
                document::sketch::ConstraintKind::Tangent {
                    first,
                    second,
                    branch,
                } => tangent_badge_anchor(
                    &producer.sketch,
                    first,
                    second,
                    branch,
                    context,
                    handles
                        .as_ref()
                        .map(|handles| |coord| handles.profile_to_render(coord)),
                    (view_projection, viewport_px, pixels_per_point),
                )
                .map(|at| (at, out_of_the_corner(at)))
                .into_iter()
                .collect(),
                // Concentric has one semantic locus: the shared center. Radius and evaluation
                // context do not participate in that placement.
                document::sketch::ConstraintKind::Concentric { first, second } => {
                    concentric_badge_anchor(
                        &producer.sketch,
                        first,
                        second,
                        handles
                            .as_ref()
                            .map(|handles| |coord| handles.profile_to_render(coord)),
                        (view_projection, viewport_px, pixels_per_point),
                    )
                    .map(|at| (at, out_of_the_corner(at)))
                    .into_iter()
                    .collect()
                }
                document::sketch::ConstraintKind::Symmetry {
                    first,
                    second,
                    axis,
                    branch,
                } => symmetry_badge_anchor(
                    &producer.sketch,
                    first,
                    second,
                    axis,
                    branch,
                    context,
                    handles
                        .as_ref()
                        .map(|handles| |coord| handles.profile_to_render(coord)),
                    (view_projection, viewport_px, pixels_per_point),
                )
                .map(|at| (at, out_of_the_corner(at)))
                .into_iter()
                .collect(),
            };
            for (anchor, direction) in placements {
                // Anchors are keyed by their rounded bits so two constraints on the same midpoint
                // share a stack; f32 has no Hash, and exact equality is what "same anchor" means.
                let key = [anchor.x.round().to_bits(), anchor.y.round().to_bits()];
                let step = stacked.entry(key).or_insert(1.0);
                let center =
                    anchor + direction * (ui::chrome::SKETCH_CONSTRAINT_BADGE_OFFSET * *step);
                *step += 1.0;
                // Read where the badge ENDS UP, not at its anchor: the standoff is far enough
                // that a projection which divides answers the plane's level differently there.
                let reading = plane.reading_at(center);
                self.sketch_constraint_badges
                    .push(ui::chrome::ConstraintBadge {
                        center,
                        reading,
                        square: plane.square_to(reading, center),
                        icon: ui::panel::constraint_icon(constraint.kind),
                        constraint: constraint.id,
                        picked: self.panel_state.selection.contains(
                            ui::panel::SelectionTarget::SketchConstraint {
                                sketch: target,
                                entity: constraint.id,
                            },
                        ),
                    });
            }
        }
    }

    /// The constraint whose badge is under the cursor, in PHYSICAL pixels — the shell's hit-test
    /// for the one sketch entity that has no geometry to hit.
    ///
    /// It reads the badges the last overlay refresh laid out rather than recomputing anchors, so
    /// what is clickable is exactly what is drawn: a constraint whose geometry went off-screen has
    /// no badge and therefore no target, which is the same rule the drawing itself follows.
    fn sketch_constraint_at(
        &self,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<document::sketch::EntityId> {
        let cursor = egui::Pos2::new(cursor_x as f32, cursor_y as f32);
        sketch_constraint_badge_at(
            &self.sketch_constraint_badges,
            cursor,
            self.last_pixels_per_point,
        )
        .or_else(|| {
            sketch_dimension_value_at(
                &self.sketch_dimension_gizmos,
                cursor,
                self.last_pixels_per_point,
            )
        })
    }

    /// Feed the entity under the cursor to the armed constraint.
    ///
    /// A pick the waiting slot cannot take is refused and the gesture keeps running — a click on
    /// the wrong kind of thing is a mis-click, not a decision to abandon the command. A pick that
    /// fills the last slot applies the constraint, commits the solved drawing through the same
    /// anchor-preserving door every sketch edit uses, and DISARMS: there is nothing left to ask.
    ///
    /// Taken picks are mirrored into the selection so the entities the gesture is holding light
    /// up through the shipped highlight path rather than a second one that could disagree with it.
    pub(super) fn resolve_sketch_constraint_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        let Some(mut armed) = self.panel_state.armed_constraint.clone() else {
            return;
        };
        if armed.restart_if_invalid(&producer.sketch) {
            self.panel_state.selection.clear_sketch_entities();
            self.panel_state.armed_constraint = Some(armed.clone());
        }
        // A dimension with all its picks in is asking WHERE, not WHAT — but a lone line finishes
        // that way while still welcoming a second one, so the click has to be read against the
        // drawing before it can be read as a place. Geometry the gesture would still take wins:
        // clicking another line turns a length into an angle, and there is nowhere else to say so.
        let placing = armed.is_placing();
        // The gesture's own question decides what the click can resolve to — see
        // `sketch_entity_for_requirement`. Anything else with every slot filled cannot reach here
        // (a completed offer disarms), so both slots answering `None` means nothing is being asked.
        let Some(slot) = armed.wants().or_else(|| armed.would_also_take()) else {
            if placing {
                self.place_armed_dimension(target, cursor_x, cursor_y);
            }
            return;
        };
        let candidate = match self.sketch_entity_for_requirement(slot, cursor_x, cursor_y) {
            // Dropping the annotation ON the line being measured is the ordinary thing to do, and
            // the gesture already holds that pick — so it is a place rather than a second one.
            Some(candidate) if placing && armed.holds_pick(candidate) => {
                self.place_armed_dimension(target, cursor_x, cursor_y);
                return;
            }
            Some(candidate) => candidate,
            None if placing => {
                self.place_armed_dimension(target, cursor_x, cursor_y);
                return;
            }
            // A miss reports the active requirement so the gesture remains observable.
            None => {
                self.panel_state.sketch_constraint_refusal = Some(slot.nothing_under_the_cursor());
                return;
            }
        };

        let locus = if armed.verb().reads_its_loci() {
            let Some(locus) = self.sketch_unsnapped_profile_coord(cursor_x, cursor_y) else {
                self.panel_state.sketch_constraint_refusal =
                    Some("cursor is not on the sketch plane");
                return;
            };
            locus
        } else {
            [0.0, 0.0]
        };
        match armed.offer_at(candidate, locus, &producer.sketch) {
            ui::panel::Offer::Refused(why) => {
                self.panel_state.sketch_constraint_refusal = Some(why);
            }
            ui::panel::Offer::Taken => {
                self.panel_state.sketch_constraint_refusal = None;
                // `toggle` only ever ADDS here: arming cleared the sketch selection, and `offer`
                // refuses an entity the gesture already holds, so the remove branch is
                // unreachable for as long as both of those hold.
                self.panel_state
                    .selection
                    .toggle(selection_target(target, candidate));
                self.panel_state.armed_constraint = Some(armed);
            }
            // Every entity is in and the annotation still needs somewhere to go. The picks stay
            // lit, because they are what the author is about to measure.
            ui::panel::Offer::Placing => {
                self.panel_state.sketch_constraint_refusal = None;
                self.panel_state
                    .selection
                    .toggle(selection_target(target, candidate));
                self.panel_state.armed_constraint = Some(armed);
            }
            ui::panel::Offer::Complete => {
                self.panel_state.sketch_constraint_refusal = None;
                let Some(context) = self.sketch_evaluation_context() else {
                    self.panel_state.sketch_constraint_refusal =
                        Some(reset_refused_sketch_constraint_completion(
                            &mut self.panel_state.armed_constraint,
                            &mut self.panel_state.selection,
                            armed.verb(),
                            &document::sketch::ConstraintRefusal::MissingEvaluationContext,
                        ));
                    return;
                };
                let kind = match armed.kind_at_context(&producer.sketch, context) {
                    Ok(kind) => kind,
                    Err(why) => {
                        self.panel_state.sketch_constraint_refusal = Some(why);
                        reset_failed_sketch_constraint_completion(
                            &mut self.panel_state.armed_constraint,
                            &mut self.panel_state.selection,
                            armed.verb(),
                        );
                        return;
                    }
                };
                match producer.with_constraint(kind, context) {
                    Ok((constrained, _)) => {
                        self.commit_sketch_profile_edit(target, constrained);
                        // The gesture's picks were scaffolding for the question, not a selection
                        // the author made; leaving them lit would make the next Delete act on
                        // geometry they only pointed at.
                        self.panel_state.selection.clear_sketch_entities();
                        self.panel_state.armed_constraint = None;
                    }
                    // The only refusal that reaches here is geometric — `offer` screened the
                    // clerical ones. The gesture stays armed but gives its picks BACK, because a
                    // full slot list answers every further click with "already complete": the
                    // author would be holding a command that can no longer be told anything.
                    //
                    // In their place go the constraints the refusal BLAMES, selected. A reason
                    // sentence in the top bar cannot say *which* of twenty assertions is the
                    // problem; a lit badge can, and it lands the culprit in the selection so
                    // Delete is the next key rather than the next search.
                    Err(why) => {
                        self.panel_state.sketch_constraint_refusal =
                            Some(reset_refused_sketch_constraint_completion(
                                &mut self.panel_state.armed_constraint,
                                &mut self.panel_state.selection,
                                armed.verb(),
                                &why,
                            ));
                        select_sketch_constraint_refusal_culprits(
                            &mut self.panel_state.selection,
                            target,
                            &why,
                        );
                    }
                }
            }
        }
    }

    /// Drop the armed dimension's annotation where the cursor is, and commit it.
    ///
    /// The second half of a dimension gesture. The picks said WHAT is measured; this says where
    /// the author wants to read it, which — once the region rules land — is also part of what is
    /// being asked. The anchor is carried into the document rather than re-derived, so the label
    /// stays where it was put.
    ///
    /// A refusal behaves exactly as it does for any other completion: the gesture gives its picks
    /// back and names the culprits, because a full slot list can be told nothing more.
    fn place_armed_dimension(
        &mut self,
        target: document::scene::NodeId,
        cursor_x: f64,
        cursor_y: f64,
    ) {
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        let Some(armed) = self.panel_state.armed_constraint.clone() else {
            return;
        };
        let (Some(context), Some(anchor)) = (
            self.sketch_evaluation_context(),
            self.sketch_unsnapped_profile_coord(cursor_x, cursor_y),
        ) else {
            self.panel_state.sketch_constraint_refusal =
                Some("point somewhere on the sketch plane");
            return;
        };
        let kind = match armed.dimension_dropped_at(anchor, &producer.sketch, context) {
            Ok(kind) => kind,
            Err(why) => {
                self.panel_state.sketch_constraint_refusal = Some(why);
                reset_failed_sketch_constraint_completion(
                    &mut self.panel_state.armed_constraint,
                    &mut self.panel_state.selection,
                    armed.verb(),
                );
                return;
            }
        };
        match producer.with_constraint_anchored(kind, Some(anchor), context) {
            Ok((constrained, _)) => {
                self.commit_sketch_profile_edit(target, constrained);
                self.panel_state.selection.clear_sketch_entities();
                self.panel_state.armed_constraint = None;
            }
            Err(why) => {
                self.panel_state.sketch_constraint_refusal =
                    Some(reset_refused_sketch_constraint_completion(
                        &mut self.panel_state.armed_constraint,
                        &mut self.panel_state.selection,
                        armed.verb(),
                        &why,
                    ));
                select_sketch_constraint_refusal_culprits(
                    &mut self.panel_state.selection,
                    target,
                    &why,
                );
            }
        }
    }

    /// The sketch entity of the kind `slot` accepts under the physical-px cursor, or `None`.
    ///
    /// Resolve directly against the required kind. Filtering a general hit afterwards lets a
    /// closer wrong-kind entity hide a valid candidate under the same cursor.
    ///
    /// Segment slots intentionally ignore arcs and circles. Curve slots preserve all three edge
    /// identities for Tangent; circular-curve slots retain only arcs and circles for Concentric.
    fn sketch_entity_for_requirement(
        &self,
        slot: ui::panel::PickRequirement,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<ui::panel::SketchEntity> {
        // A vertex under the cursor beats an edge under the same cursor — the priority the rest of
        // sketch picking already uses, since the vertex is the more specific thing. So the
        // either-kind slot asks for a point first and only falls through to the curves on a miss.
        if matches!(
            slot,
            ui::panel::PickRequirement::Point | ui::panel::PickRequirement::PointOrCurve
        ) {
            let vertex = self
                .sketch_vertex_at(cursor_x, cursor_y)
                .and_then(|index| self.sketch_point_ids.get(index).copied())
                .map(ui::panel::SketchEntity::Point);
            if vertex.is_some() || slot == ui::panel::PickRequirement::Point {
                return vertex;
            }
        }
        let cursor = egui::Pos2::new(cursor_x as f32, cursor_y as f32);
        let segment = self
            .nearest_sketch_segment(cursor_x, cursor_y)
            .map(|(id, from, to)| {
                (
                    SketchEdgeHit::Segment(id),
                    point_to_segment_distance(cursor, from, to),
                )
            });
        nearest_sketch_edge_for_requirement(
            slot,
            segment,
            self.nearest_sketch_arc(cursor_x, cursor_y)
                .map(|(id, distance)| (SketchEdgeHit::Arc(id), distance)),
            self.nearest_sketch_circle(cursor_x, cursor_y)
                .map(|(id, distance)| (SketchEdgeHit::Circle(id), distance)),
        )
        .map(|hit| ui::panel::SketchEntity::Curve(sketch_curve_from_hit(hit)))
    }

    /// The derived region under the physical-px cursor (#100), or `None`. The SMALLEST containing
    /// face wins, so a click inside a pocket carves the pocket rather than the shape around it —
    /// the same "most specific thing under the cursor" rule the vertex-over-edge priority uses.
    pub(super) fn sketch_face_at(
        &self,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<document::sketch::FaceKey> {
        let cursor = egui::Pos2::new(cursor_x as f32, cursor_y as f32);
        let index = self
            .sketch_face_polygons
            .iter()
            .filter(|(_, boundary)| point_in_screen_polygon(boundary, cursor))
            .min_by(|(_, a), (_, b)| {
                polygon_double_area(a)
                    .abs()
                    .total_cmp(&polygon_double_area(b).abs())
            })
            .map(|(index, _)| *index)?;
        // The hit-test polygons are kept by POSITION and the identity is minted only here, on the
        // click that is about to store one — the search is far too dear to run every frame for
        // faces nobody points at.
        let target = self.panel_state.sketch_mode?;
        let (producer, _) = self.sketch_node_state(target)?;
        producer
            .sketch
            .face_key_at(index, self.sketch_evaluation_context()?)
    }

    /// Whether the region the open viewport menu is acting on is picked (#100), or `None` when the
    /// menu has no region under it — what decides whether the menu offers "carve" or "fill".
    pub(super) fn sketch_menu_face_is_picked(&self) -> Option<bool> {
        let target = self.panel_state.sketch_mode?;
        let key = self.sketch_menu_face.as_ref()?;
        let (producer, _) = self.sketch_node_state(target)?;
        Some(
            producer
                .sketch
                .face_is_picked(key, self.sketch_evaluation_context()?),
        )
    }

    /// Flip the pick state of the region the viewport menu is acting on (#100) — the one edit that
    /// carves a hole, committed through the same profile-edit path every other sketch edit uses so
    /// it coalesces into the open undo group. No-op when no region is under the menu.
    pub(super) fn toggle_sketch_menu_face(&mut self) {
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        let Some(key) = self.sketch_menu_face.take() else {
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        let Some(context) = self.sketch_evaluation_context() else {
            return;
        };
        let picked = producer.sketch.face_is_picked(&key, context);
        let next = producer.with_face_picked(key, !picked, context);
        self.commit_sketch_profile_edit(target, next);
    }

    /// Close the active connected-Line chain with one native segment. The existing start identity
    /// is reused, so no coincident twin or hidden constraint is introduced; refusal merely ends
    /// the session chain and leaves the document untouched.
    fn close_sketch_line_loop(&mut self) {
        let Some(chain) = self.line_gesture.chain() else {
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(chain.owner) else {
            return;
        };
        if let line::LineEdit::Document(next) = self.line_gesture.close(chain.owner, &producer) {
            self.commit_sketch_profile_edit(chain.owner, next);
        }
    }

    /// Apply an explicit face role from the rail rather than toggling it. Repeated Fill or Carve
    /// clicks are idempotent, which is important for a modal tool: the same verb should never undo
    /// itself merely because the author clicked twice.
    pub(super) fn sketch_set_face_picked(&mut self, cursor_x: f64, cursor_y: f64, picked: bool) {
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        let Some(key) = self.sketch_face_at(cursor_x, cursor_y) else {
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        let Some(context) = self.sketch_evaluation_context() else {
            return;
        };
        if producer.sketch.face_is_picked(&key, context) == picked {
            return;
        }
        self.commit_sketch_profile_edit(target, producer.with_face_picked(key, picked, context));
    }

    /// Queue an add/delete profile edit as ONE entry in the open sketch undo
    /// group. Recomputes the bbox-min anchor compensation exactly like the vertex drag — the
    /// producer re-anchors its bbox-min to the node origin, so a vertex inserted or removed at
    /// the bbox extreme would shift the whole profile in world unless the node offset absorbs the
    /// bbox-min delta — then pushes `SetSketch` (+ `SetOffset` when the anchor moved) through the
    /// viewport-intent door so the next `render` records it through `apply_intent`. A single
    /// click therefore coalesces to one in-mode undo step, the same discipline the drag uses.
    pub(super) fn commit_sketch_profile_edit(
        &mut self,
        target: document::scene::NodeId,
        new_producer: document::sketch::SketchSolid,
    ) {
        let Some((old_producer, old_offset)) = self.sketch_node_state(target) else {
            return;
        };
        let Some(context) = self.sketch_evaluation_context() else {
            return;
        };
        let new_offset = new_producer.anchor_preserving_offset(&old_producer, old_offset, context);
        self.viewport_transactions
            .push(sketch_profile_edit_transaction(
                target,
                new_producer,
                old_offset,
                new_offset,
            ));
    }

    /// If the cursor (physical px) is over a profile handle, build the [`SketchVertexDrag`] that
    /// grabs it — the nearest vertex within the grab radius, or failing that the edge under the
    /// cursor, with the current producer snapshotted so the whole gesture coalesces to one
    /// command. `None` when nothing grabbable is under the cursor (the press falls through to the
    /// normal camera/placement path). Called from the `events` press handler, only under Select.
    ///
    /// Vertices win over edges, and by more than proximity: every edge passes through its own
    /// endpoints, so an endpoint grab would be unreachable if the nearer edge could take it.
    ///
    /// **A badge takes a click; it never takes a drag.** A constraint has no position, so a badge
    /// is not something a drag could move — and a badge is a 32-point box floating 30 points off
    /// the geometry it labels, so on a drawing that carries many relations the badges cover the
    /// curves. Refusing the whole gesture there made the geometry under them unreachable. It is
    /// not asked at all now: the grabs below are all positional by construction, and the badge
    /// keeps its click regardless, because the press arms the selection resolve independently of
    /// the drag and a release that never left the press still resolves it.
    pub(super) fn begin_sketch_vertex_drag(
        &self,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<SketchVertexDrag> {
        let target = self.panel_state.sketch_mode?;
        let held = self
            .sketch_vertex_at(cursor_x, cursor_y)
            .and_then(|index| self.sketch_point_ids.get(index).copied())
            .map(SketchGrab::Point)
            // A dimension's number is reached for the same way a badge is and answers a different
            // gesture: a badge takes a click and never a drag (ADR 0046) because it could sit
            // anywhere without saying anything different, and a number cannot — where it sits is
            // part of what the author said. So the one is draggable and the other is not, and they
            // sit at the same height in the order for the same reason: both are small bounded
            // marks floating clear of the geometry they name.
            .or_else(|| {
                sketch_dimension_value_at(
                    &self.sketch_dimension_gizmos,
                    egui::Pos2::new(cursor_x as f32, cursor_y as f32),
                    self.last_pixels_per_point,
                )
                .map(|constraint| SketchGrab::Annotation { constraint })
            })
            // A lever's stick moves the point it belongs to, never the spline it steers. The
            // handle rides along at the angle and length it was left at, which is exactly what a
            // fit-point drag already does — so this is that same motion, reached for by the
            // manipulator instead of by the dot, and measured as a displacement because a press
            // anywhere along the stick must not teleport the point out to meet it.
            .or_else(|| {
                self.tangent_lever_at(cursor_x, cursor_y)
                    .map(|fit| SketchGrab::TranslateLever { fit, from: None })
            })
            .or_else(|| self.grabbable_sketch_curve_at(cursor_x, cursor_y))?;
        let node = self.panel_state.scene.node_by_id(target)?;
        let document::scene::NodeContent::SketchTool { producer, .. } = &node.content else {
            return None;
        };
        Some(SketchVertexDrag {
            held,
            original: producer.clone(),
            original_offset: node.transform.offset_voxels,
            original_min: self.profile_bbox_min(producer)?,
            began: false,
            arc_turns: document::sketch::ArcTurnUnderAGesture::opening_over(&producer.sketch),
        })
    }

    /// The curve under the cursor that a drag can move as a WHOLE, if there is one.
    ///
    /// Grabbing the body of a curve MOVES it, whatever kind of curve it is.
    ///
    /// One verb, and no per-shape reading of what the author must have meant. What happens to the
    /// rest of the drawing is the constraints' answer: a slot's centerline carries the whole slot
    /// because the cap centers stand on it, a rail widens the slot because the far rail is not
    /// asked to move and the tangency web lets the caps grow, and a line the author drew between
    /// two loose points goes alone. None of those three is written down anywhere — they are what
    /// the relations already say, read back by a solve that would rather move than deform.
    ///
    /// A rail widens the slot about the FAR RAIL, not about its centerline: holding one rail still
    /// is less change than moving both, so the centerline travels to the new middle. That is not
    /// the symmetric widening ratified on 2026-08-04, and it is not a regression against it — the
    /// symmetry was a property of the old dedicated offset verb, which this no longer calls, and
    /// the drawing does not assert it. A slot that should stay symmetric needs to SAY so, with a
    /// relation, which is the whole point of removing the branch.
    ///
    /// That branch was a perpendicular OFFSET for boundary segments and arcs and a translation for
    /// splines and construction curves. It is the shape of defect the `FreeCAD` drag code documents —
    /// a ladder of per-geometry cases, each right about the shape that prompted it and a guess
    /// about everything else — and it is only ever needed when the solve's objective prefers
    /// stretching to moving. Ours no longer does (owner, 2026-08-05).
    ///
    /// A CONIC is here for the opposite reason to everything else: its body drag does not move it,
    /// it reshapes it. Rho is the conic's one authored freedom and ADR 0038 took away the point
    /// that used to carry it, so the body IS the handle, and the shoulder mark
    /// ([`Sketch::conic_shoulders`](document::sketch::Sketch::conic_shoulders)) is where the author
    /// reaches for it. Without this arm that mark is a dot that does nothing: the drawing has
    /// answered a conic body drag as a rho drag since the shoulder was removed, and the press
    /// simply never arrived.
    ///
    /// A CIRCLE is offered for the same reason a conic is: the gesture already means something the
    /// press never arrived to deliver. Dragging a rim grows the shape about a center it holds, and
    /// since 2026-08-07 an arc reads its rim exactly the way a circle's grip always has — as the
    /// distance out from the center — so the two shapes now answer the same gesture the same way
    /// and there is nothing left to tell the author apart. It was withheld while they differed.
    ///
    /// The remaining higher curves are fixed-arity handles the author moves one at a time.
    fn grabbable_sketch_curve_at(&self, cursor_x: f64, cursor_y: f64) -> Option<SketchGrab> {
        let curve = match self.nearest_sketch_edge(cursor_x, cursor_y)? {
            SketchEdgeHit::Segment(id) => document::sketch::SketchCurve::Segment(id),
            SketchEdgeHit::Arc(id) => document::sketch::SketchCurve::Arc(id),
            SketchEdgeHit::Circle(id) => document::sketch::SketchCurve::Circle(id),
            SketchEdgeHit::HigherCurve(
                curve @ (document::sketch::SketchCurve::Spline(_)
                | document::sketch::SketchCurve::Conic(_)),
            ) => curve,
            SketchEdgeHit::HigherCurve(_) => return None,
        };
        Some(SketchGrab::Translate { curve, from: None })
    }

    /// Recompute the sketch overlay for the NEXT frame. Projects
    /// each profile vertex (render frame) to screen, storing the egui-point handles + their
    /// interaction state for drawing, and the physical-pixel centers **in profile order** for the
    /// press hit-tests (a culled behind-camera vertex is `None`, keeping the indices aligned so
    /// segments can pair adjacent vertices). Also derives the add-point **insert-preview**
    /// marker from the armed tool. Clears everything outside sketch mode.
    fn refresh_sketch_overlay(
        &mut self,
        view_projection: glam::Mat4,
        viewport_px: [u32; 4],
        pixels_per_point: f32,
    ) {
        self.sketch_overlay_points.clear();
        self.sketch_vertex_px.clear();
        self.sketch_point_ids.clear();
        self.sketch_point_derived.clear();
        self.sketch_segments.clear();
        self.sketch_segment_lines.clear();
        self.sketch_arc_lines.clear();
        self.sketch_arc_chords.clear();
        self.sketch_circle_chords.clear();
        self.sketch_higher_curve_chords.clear();
        self.sketch_spline_polygons.clear();
        self.sketch_tangent_levers.clear();
        self.sketch_face_polygons.clear();
        self.sketch_constraint_badges.clear();
        self.sketch_dimension_gizmos.clear();
        self.sketch_insert_preview = None;
        self.sketch_snap_marker = None;
        self.sketch_draw_preview.clear();
        self.sketch_marquee_band = None;

        let Some(target) = self.panel_state.sketch_mode else {
            // #99 / slice 3 / #102: a drawing or marquee gesture dies with the mode.
            self.reset_sketch_gestures();
            self.sketch_chamfer_pending = None;
            self.sketch_offset_pending = None;
            self.sketch_move_copy_pending = None;
            self.sketch_scale_pending = None;
            self.sketch_rectangular_pattern_pending = None;
            self.sketch_marquee_anchor = None;
            self.sketch_arc_gesture = None;
            self.sketch_circle_center = None;
            self.sketch_circle_target = None;
            // #100: so does the region a closed menu was acting on.
            self.sketch_menu_face = None;
            return;
        };
        let Some(handles) = self
            .panel_state
            .scene
            .sketch_handles(target, self.panel_state.geometry.voxels_per_block)
        else {
            self.reset_sketch_gestures();
            self.sketch_chamfer_pending = None;
            self.sketch_offset_pending = None;
            self.sketch_move_copy_pending = None;
            self.sketch_scale_pending = None;
            self.sketch_rectangular_pattern_pending = None;
            return;
        };

        let tool = self.panel_state.sketch_tool;
        let chamfer_context_is_live = matches!(
            tool,
            ui::panel::SketchTool::ChamferDistanceAngle | ui::panel::SketchTool::ChamferTwoDistance
        ) && self.panel_state.armed_constraint.is_none()
            && self
                .sketch_chamfer_pending
                .is_none_or(|pending| pending.target == target && pending.tool == tool);
        if !chamfer_context_is_live {
            self.sketch_chamfer_pending = None;
        }
        let offset_context_is_live = tool == ui::panel::SketchTool::Offset
            && self.panel_state.armed_constraint.is_none()
            && self
                .sketch_offset_pending
                .is_none_or(|pending| pending.target == target);
        if !offset_context_is_live {
            self.sketch_offset_pending = None;
        }
        let move_copy_context_is_live = tool == ui::panel::SketchTool::MoveCopy
            && self.panel_state.armed_constraint.is_none()
            && self
                .sketch_move_copy_pending
                .as_ref()
                .is_none_or(|pending| pending.target == target);
        if !move_copy_context_is_live {
            self.sketch_move_copy_pending = None;
        }
        let scale_context_is_live = tool == ui::panel::SketchTool::Scale
            && self.panel_state.armed_constraint.is_none()
            && self
                .sketch_scale_pending
                .as_ref()
                .is_none_or(|pending| pending.target == target);
        if !scale_context_is_live {
            self.sketch_scale_pending = None;
        }
        let rectangular_pattern_context_is_live = tool == ui::panel::SketchTool::RectangularPattern
            && self.panel_state.armed_constraint.is_none()
            && self
                .sketch_rectangular_pattern_pending
                .as_ref()
                .is_none_or(|pending| pending.target == target);
        if !rectangular_pattern_context_is_live {
            self.sketch_rectangular_pattern_pending = None;
        }
        // #99: a chain / rectangle anchor belongs to its tool — switching away drops it.
        self.line_gesture.retain_for_context(
            tool == ui::panel::SketchTool::Line,
            self.panel_state.armed_constraint.is_some(),
            Some(target),
        );
        self.higher_curve_gesture.retain_for_context(
            higher_curve_kind(tool),
            self.panel_state.armed_constraint.is_some(),
            Some(target),
        );
        self.midpoint_line_gesture.retain_for_context(
            tool == ui::panel::SketchTool::MidpointLine,
            self.panel_state.armed_constraint.is_some(),
            Some(target),
        );
        let center_arc_producer = self.sketch_node_state(target).map(|(producer, _)| producer);
        self.center_arc_gesture.retain_for_context(
            tool == ui::panel::SketchTool::ArcCenterEndpoints,
            self.panel_state.armed_constraint.is_some(),
            Some(target),
            center_arc_producer.as_ref(),
        );
        self.point_circle_gesture.retain_for_context(
            point_circle_kind(tool),
            self.panel_state.armed_constraint.is_some(),
            Some(target),
        );
        self.three_point_rectangle_gesture.retain_for_context(
            tool == ui::panel::SketchTool::Rectangle3Point,
            self.panel_state.armed_constraint.is_some(),
            Some(target),
        );
        self.polygon_gesture.retain_for_context(
            polygon_kind(tool),
            self.panel_state.armed_constraint.is_some(),
            Some(target),
        );
        self.slot_gesture.retain_for_context(
            slot_kind(tool),
            self.panel_state.armed_constraint.is_some(),
            Some(target),
        );
        self.tangent_circle_gesture.retain_for_context(
            tangent_circle_kind(tool),
            self.panel_state.armed_constraint.is_some(),
            Some(target),
        );
        self.corner_rectangle_gesture.retain_for_context(
            corner_rectangle_kind(tool),
            self.panel_state.armed_constraint.is_some(),
            Some(target),
        );
        let tangent_producer = self.sketch_node_state(target).map(|(producer, _)| producer);
        self.tangent_arc_gesture.retain_for_context(
            tool == ui::panel::SketchTool::ArcTangent,
            self.panel_state.armed_constraint.is_some(),
            Some(target),
            tangent_producer.as_ref(),
        );
        if tool == ui::panel::SketchTool::Line && self.panel_state.armed_constraint.is_none() {
            self.validate_line_gesture(target);
        }
        if tool != ui::panel::SketchTool::Select {
            self.sketch_marquee_anchor = None;
        }
        if tool != ui::panel::SketchTool::ThreePointArc {
            self.sketch_arc_gesture = None;
        }
        if !circle_gesture_is_current(tool, target, self.sketch_circle_target) {
            self.sketch_circle_center = None;
            self.sketch_circle_target = None;
        }
        let [vx, vy, vw, vh] = viewport_px.map(|component| component as f32);
        let tangent_arms = self.tangent_arm_points(target);
        let on_ink = self.points_standing_on_ink(target);
        let dragging_point = self.sketch_drag.as_ref().and_then(|drag| drag.held.point());
        let mut revealed = self.points_the_drawing_shows_by_itself(target);
        // A forgiving grab radius (physical px) so a hover reads as "draggable" near the thumb.
        let hover_radius_px = (ui::chrome::SKETCH_HANDLE_HALF + ui::chrome::SKETCH_HANDLE_GRAB_PAD)
            * pixels_per_point;
        let mut pending_points: Vec<(
            Option<document::sketch::EntityId>,
            ui::chrome::SketchVertexHandle,
        )> = Vec::with_capacity(handles.vertices.len());
        for (index, vertex) in handles.vertices.iter().enumerate() {
            let clip = view_projection * glam::Vec4::new(vertex[0], vertex[1], vertex[2], 1.0);
            if clip.w <= 0.0 {
                // Behind the camera: hold the index with `None` so segment adjacency survives.
                self.sketch_vertex_px.push(None);
                continue;
            }
            let ndc_x = clip.x / clip.w;
            let ndc_y = clip.y / clip.w;
            let px = vx + (ndc_x * 0.5 + 0.5) * vw;
            let py = vy + (1.0 - (ndc_y * 0.5 + 0.5)) * vh;
            let center_px = egui::Pos2::new(px, py);

            let hovered = self
                .last_cursor_position
                .map(|(cx, cy)| (cx as f32 - px).hypot(cy as f32 - py) <= hover_radius_px)
                .unwrap_or(false);
            let point_id = handles.point_ids.get(index).copied();
            let selected = point_id
                .map(|entity| {
                    let picked = ui::panel::SelectionTarget::SketchPoint {
                        sketch: target,
                        entity,
                    };
                    self.panel_state.selection.contains(picked)
                })
                .unwrap_or(false);
            // Precedence: dragged > selected > hover > idle. A selected vertex stays
            // filled-accent even under the cursor, matching the segment rule so a point and an
            // edge read alike.
            let state = if dragging_point == point_id {
                ui::gizmos::HandleState::Snapped
            } else if selected {
                ui::gizmos::HandleState::Selected
            } else if hovered {
                ui::gizmos::HandleState::Hover
            } else {
                ui::gizmos::HandleState::Idle
            };

            let center_pt = egui::Pos2::new(px / pixels_per_point, py / pixels_per_point);
            // Held back rather than pushed: whether this dot draws can depend on the curve under
            // the cursor, and that is not resolved until the chord caches below are built. A dot
            // the author is already touching answers for itself and is revealed here.
            if hovered || selected || dragging_point == point_id {
                revealed.extend(point_id);
            }
            pending_points.push((
                point_id,
                ui::chrome::SketchVertexHandle {
                    at: center_pt,
                    state,
                    ink: match point_id {
                        Some(id) if tangent_arms.contains(&id) => {
                            ui::chrome::SketchVertexInk::TangentArm
                        }
                        Some(id) if on_ink.contains(&id) => ui::chrome::SketchVertexInk::OnInk,
                        // A vertex with no id is a PREVIEW dot the tool is placing: it belongs to
                        // the mark being drawn, so it reads as drawing.
                        None => ui::chrome::SketchVertexInk::OnInk,
                        Some(_) => ui::chrome::SketchVertexInk::OffInk,
                    },
                },
            ));
            self.sketch_vertex_px.push(Some(center_px));
        }

        // The stable point id + segment connectivity for THIS frame, aligned with
        // `sketch_vertex_px` — the press hit-tests (in `events`) read these to resolve a click to
        // the entity it targets.
        self.sketch_point_ids = handles.point_ids.clone();
        self.sketch_point_derived = handles.derived.clone();
        self.sketch_segments = handles.segments.clone();

        // Arc chord polylines in PHYSICAL px (#102), tessellated for the SCREEN. A behind-camera
        // chord vertex culls the whole arc, matching the segment rule: a partially-projected curve
        // would fold across the viewport.
        let to_viewport_px = |coord: [f64; 2]| {
            let vertex = handles.profile_to_render(coord);
            let clip = view_projection * glam::Vec4::new(vertex[0], vertex[1], vertex[2], 1.0);
            (clip.w > 0.0).then(|| {
                egui::Pos2::new(
                    vx + (clip.x / clip.w * 0.5 + 0.5) * vw,
                    vy + (1.0 - (clip.y / clip.w * 0.5 + 0.5)) * vh,
                )
            })
        };
        // The one rule every turn on this page is flattened by. A tolerance in the plane's own
        // units earns a chord count from the arc's size in the PLANE, so the same handful of
        // chords is drawn at every zoom and a magnified curve reads as a visible polygon. The
        // projected radius says what a plane unit is currently worth in pixels — one number
        // already carrying the zoom, the foreshortening and the plane's tilt — and the tolerance
        // follows from it. Never coarser than the resolve tolerance, which is the profile's own
        // meaning; this is the same curve, drawn smoothly.
        let screen_chord_tolerance = |center: [f64; 2], on_rim: [f64; 2], radius: f64| {
            to_viewport_px(center)
                .zip(to_viewport_px(on_rim))
                .map(|(center_px, rim_px)| f64::from(center_px.distance(rim_px)))
                .filter(|radius_px| *radius_px > 1.0)
                .map_or(document::sketch::ARC_SAGITTA_TOLERANCE, |radius_px| {
                    radius * ARC_SCREEN_SAGITTA_PX / radius_px
                })
                .min(document::sketch::ARC_SAGITTA_TOLERANCE)
        };
        for arc in &handles.arcs {
            let (arc_id, from, to, sweep) = (arc.entity, arc.from, arc.to, arc.sweep_degrees);
            let tolerance = document::sketch::arc_center_radius(from, to, sweep).map_or(
                document::sketch::ARC_SAGITTA_TOLERANCE,
                |(center, radius)| screen_chord_tolerance(center, from, radius),
            );
            let mut profile = vec![from];
            profile.extend(
                document::sketch::arc_interior_points_within(from, to, sweep, tolerance)
                    .iter()
                    .map(|point| point.in_plane()),
            );
            profile.push(to);
            let projected: Option<Vec<egui::Pos2>> =
                profile.into_iter().map(&to_viewport_px).collect();
            if let Some(projected) = projected {
                self.sketch_arc_chords.push((arc_id, projected));
            }
        }
        for circle in &handles.circles {
            let circle_id = circle.entity;
            let center = circle.center;
            let radius = circle.radius;
            let tolerance = screen_chord_tolerance(center, [center[0] + radius, center[1]], radius);
            let profile = circle_ring(center, radius, tolerance);
            let projected: Option<Vec<egui::Pos2>> =
                profile.into_iter().map(&to_viewport_px).collect();
            if let Some(projected) = projected {
                self.sketch_circle_chords.push((circle_id, projected));
            }
        }
        for curve in &handles.higher_curves {
            for piece in &curve.pieces {
                let projected: Option<Vec<egui::Pos2>> = break_piece_points(piece)
                    .into_iter()
                    .map(&to_viewport_px)
                    .collect();
                if let Some(projected) = projected {
                    self.sketch_higher_curve_chords
                        .push((curve.entity, projected));
                }
            }
        }
        // The control frames, projected through the vertex cache their controls already went
        // through — so a leg meets its two dots exactly, rather than by two projections agreeing.
        // One behind-camera control culls the whole frame, as a behind-camera endpoint culls a
        // segment line.
        for (spline, controls) in self.control_polygons(target) {
            let legs: Option<Vec<egui::Pos2>> = controls
                .iter()
                .map(|id| {
                    let index = self.sketch_point_ids.iter().position(|held| held == id)?;
                    *self.sketch_vertex_px.get(index)?
                })
                .collect();
            if let Some(legs) = legs {
                self.sketch_spline_polygons.push((spline, legs));
            }
        }
        // The tangent levers, through the same vertex cache and for the same reason: a lever must
        // meet its two green arms exactly, not by two projections agreeing.
        for (fit, lever) in self.tangent_levers(target) {
            let run: Option<Vec<egui::Pos2>> = lever
                .iter()
                .map(|id| {
                    let index = self.sketch_point_ids.iter().position(|held| held == id)?;
                    *self.sketch_vertex_px.get(index)?
                })
                .collect();
            if let Some(run) = run {
                self.sketch_tangent_levers.push((fit, run));
            }
        }

        // Which curved entities are construction, joined from the handles while they are still in
        // hand. The chord caches above answer "where is it on screen" for the hit-test and are
        // deliberately left free of linetype; the draw loops below join back to these.
        let construction =
            |role: document::sketch::EntityRole| role == document::sketch::EntityRole::Construction;
        let construction_arcs: std::collections::BTreeSet<document::sketch::EntityId> = handles
            .arcs
            .iter()
            .filter(|arc| construction(arc.role))
            .map(|arc| arc.entity)
            .collect();
        let construction_circles: std::collections::BTreeSet<document::sketch::EntityId> = handles
            .circles
            .iter()
            .filter(|circle| construction(circle.role))
            .map(|circle| circle.entity)
            .collect();
        let construction_higher: std::collections::BTreeSet<document::sketch::SketchCurve> =
            handles
                .higher_curves
                .iter()
                .filter(|curve| construction(curve.role))
                .map(|curve| curve.entity)
                .collect();

        // The pick a drawing tool would take right now, resolved once. The lit curve below and the
        // snap mark are two readings of the same answer, and working them out separately is how
        // they get to disagree.
        let picked_target = (tool.curve_under_pointer() == ui::panel::CurveUnderPointer::PickedOn)
            .then_some(self.last_cursor_position)
            .flatten()
            .and_then(|(cursor_x, cursor_y)| self.sketch_target_at(cursor_x, cursor_y));

        // The segment under the cursor and the state it should draw in. A vertex under the cursor
        // takes priority — it already answers with its own handle state — so a segment lights up
        // only when no vertex is hit, the SAME decision the vertex-grab makes. Reusing that
        // hit-test keeps the feedback exactly aligned with what a click acts on.
        //
        // A tool that acts on a curve and a tool that takes its picks on one light it the same
        // way: the highlight answers "the pointer is on that curve", which is the same question
        // either way. What the tool then does with the answer is [`CurveUnderPointer`]'s to say.
        let hovered_edge: Option<(SketchEdgeHit, ui::gizmos::HandleState)> = match tool
            .curve_under_pointer()
        {
            ui::panel::CurveUnderPointer::ActedOn | ui::panel::CurveUnderPointer::PickedOn => {
                Some(ui::gizmos::HandleState::Hover)
            }
            ui::panel::CurveUnderPointer::Ignored => None,
        }
        .and_then(|state| {
            self.last_cursor_position.and_then(|(cx, cy)| {
                if tool == ui::panel::SketchTool::ArcTangent {
                    if self.tangent_arc_gesture.is_pending() {
                        return None;
                    }
                    return self
                        .sketch_tangent_arc_source_at(cx, cy)
                        .and_then(|source| match source.curve {
                            document::sketch::SketchCurve::Segment(id) => {
                                Some(SketchEdgeHit::Segment(id))
                            }
                            document::sketch::SketchCurve::Arc(id) => Some(SketchEdgeHit::Arc(id)),
                            document::sketch::SketchCurve::Circle(id) => {
                                Some(SketchEdgeHit::Circle(id))
                            }
                            document::sketch::SketchCurve::Bezier(_)
                            | document::sketch::SketchCurve::Ellipse(_)
                            | document::sketch::SketchCurve::Conic(_)
                            | document::sketch::SketchCurve::Spline(_) => None,
                        })
                        .map(|hit| (hit, state));
                }
                // The one seam again: a drawing tool's highlight comes from the same resolution its
                // click will run, so a point under the cursor puts the highlight out exactly when
                // it takes the pick — the author never sees a curve lit for a click that will
                // land on a vertex instead.
                if tool.curve_under_pointer() == ui::panel::CurveUnderPointer::PickedOn {
                    return picked_target
                        .and_then(document::sketch::SketchTarget::onto)
                        .map(|curve| (sketch_edge_hit_from_curve(curve), state));
                }
                if matches!(
                    tool,
                    ui::panel::SketchTool::Circle2Tangent | ui::panel::SketchTool::Circle3Tangent
                ) {
                    return self
                        .nearest_sketch_segment(cx, cy)
                        .map(|(id, _, _)| (SketchEdgeHit::Segment(id), state));
                }
                if matches!(
                    tool,
                    ui::panel::SketchTool::BreakCurve
                        | ui::panel::SketchTool::Trim
                        | ui::panel::SketchTool::Extend
                        | ui::panel::SketchTool::Offset
                ) {
                    return self.nearest_sketch_edge(cx, cy).map(|hit| (hit, state));
                }
                if matches!(
                    tool,
                    ui::panel::SketchTool::Fillet
                        | ui::panel::SketchTool::ChamferEqual
                        | ui::panel::SketchTool::ChamferDistanceAngle
                        | ui::panel::SketchTool::ChamferTwoDistance
                ) {
                    return self
                        .nearest_sketch_segment(cx, cy)
                        .map(|(id, _, _)| (SketchEdgeHit::Segment(id), state));
                }
                // A lever suppresses the edge under it for the same reason a vertex does: the
                // press will act on the lever, and hover has to agree with the press. Near a fit
                // point a lever and the curve it steers overlap, so without this the whole spline
                // lights for a cursor that is plainly on the handle.
                if self.sketch_vertex_at(cx, cy).is_some()
                    || self.tangent_lever_at(cx, cy).is_some()
                {
                    None
                } else {
                    self.nearest_sketch_edge(cx, cy).map(|hit| (hit, state))
                }
            })
        });

        // The snapping mark, standing where the pick will actually land. The lit curve says which
        // curve is under the pointer; this says the pick has left the grid and is riding that
        // curve, which is the part a lit curve on its own leaves the author to guess at. It is the
        // tick-cross the dragged handle wears for the same reason — "engaged", in the one
        // vocabulary the chrome already has for it.
        self.sketch_snap_marker = picked_target
            .filter(|pick| pick.onto().is_some())
            .and_then(|pick| to_viewport_px(pick.at().in_plane()))
            .map(|px| egui::Pos2::new(px.x / pixels_per_point, px.y / pixels_per_point));

        // Now the dots. The curve under the cursor shows the points it stands on, which is the
        // last thing `revealed` was waiting for — hovering a line has to bring up the corners it
        // runs between, or the author cannot tell a joined corner from a seam without clicking.
        if let Some((hit, _)) = hovered_edge {
            revealed.extend(self.points_of_edge_hit(target, hit));
        }
        // An arm shows with its LEVER and never on its own, in every tool: a green dot with no
        // stick under it is a manipulator the author cannot read. `sketch_tangent_levers` already
        // holds exactly the levers that are out, so the two cannot disagree.
        let arms_out: std::collections::BTreeSet<document::sketch::EntityId> = self
            .sketch_tangent_levers
            .iter()
            .flat_map(|(fit, _)| self.tangent_arms_of(target, *fit))
            .collect();
        revealed.retain(|id| !tangent_arms.contains(id) || arms_out.contains(id));
        revealed.extend(arms_out);
        // A dot standing under another dot never draws, whatever revealed it. Hovering an arc
        // brings up the points it stands on and one of those is the center it derives, so without
        // this the stack the rest-rule collapsed comes straight back the moment the author looks
        // at the shape.
        let stacked = self.dots_standing_under_another(target);
        for (point_id, handle) in pending_points {
            if point_id.is_none_or(|id| revealed.contains(&id) && !stacked.contains(&id)) {
                self.sketch_overlay_points.push(handle);
            }
        }
        // A conic's shoulder, which is a reading rather than a point and so has no id to be
        // revealed by. It draws unconditionally: rho is the conic's one authored freedom and this
        // is the only mark that shows it, where every other dot here is answering the question of
        // whether the ink has already said the same thing.
        //
        // It reads as ON the ink because it is, and it needs no grab of its own for the same
        // reason — the press under it lands on the conic, whose body drag is already the rho drag.
        for (_, at) in self.conic_shoulders_in_profile(target) {
            let Some(px) = to_viewport_px(at) else {
                continue;
            };
            let hovered = self
                .last_cursor_position
                .map(|(cx, cy)| (cx as f32 - px.x).hypot(cy as f32 - px.y) <= hover_radius_px)
                .unwrap_or(false);
            self.sketch_overlay_points
                .push(ui::chrome::SketchVertexHandle {
                    at: egui::Pos2::new(px.x / pixels_per_point, px.y / pixels_per_point),
                    state: if hovered {
                        ui::gizmos::HandleState::Hover
                    } else {
                        ui::gizmos::HandleState::Idle
                    },
                    ink: ui::chrome::SketchVertexInk::OnInk,
                });
        }

        self.refresh_sketch_constraint_badges(
            target,
            view_projection,
            viewport_px,
            pixels_per_point,
        );
        self.refresh_sketch_dimension_gizmos(
            target,
            view_projection,
            viewport_px,
            pixels_per_point,
        );

        // The segment LINES to draw next frame: each committed edge between its two projected
        // endpoints, in egui points — an open sketch resolves to nothing, so the edges
        // are the only thing that shows the profile is connected). A behind-camera endpoint
        // (`None` in `sketch_vertex_px`) culls its line, matching the vertex-dot cull. The one
        // hovered segment carries its Hover/Marked state; the rest are Idle.
        for segment in &self.sketch_segments {
            if let (Some(Some(a_px)), Some(Some(b_px))) = (
                self.sketch_vertex_px.get(segment.from),
                self.sketch_vertex_px.get(segment.to),
            ) {
                let a = egui::Pos2::new(a_px.x / pixels_per_point, a_px.y / pixels_per_point);
                let b = egui::Pos2::new(b_px.x / pixels_per_point, b_px.y / pixels_per_point);
                // Precedence: Selected > plain Hover > Idle. A selected edge stays bold even
                // under the cursor (Select hover never shrinks it).
                let picked = ui::panel::SelectionTarget::SketchSegment {
                    sketch: target,
                    entity: segment.entity,
                };
                let selected = self.panel_state.selection.contains(picked);
                let state = match hovered_edge {
                    _ if selected => ui::gizmos::HandleState::Selected,
                    Some((SketchEdgeHit::Segment(id), state)) if id == segment.entity => state,
                    _ => ui::gizmos::HandleState::Idle,
                };
                self.sketch_segment_lines.push(ui::chrome::SketchEdgeLine {
                    a,
                    b,
                    state,
                    construction: segment.role == document::sketch::EntityRole::Construction,
                });
            }
        }

        // The arc curves to draw next frame, in egui points — the same precedence the
        // segments use, so a picked arc and a picked segment read identically (#102).
        for (arc_id, chords) in &self.sketch_arc_chords {
            let picked = ui::panel::SelectionTarget::SketchArc {
                sketch: target,
                entity: *arc_id,
            };
            let selected = self.panel_state.selection.contains(picked);
            let state = match hovered_edge {
                _ if selected => ui::gizmos::HandleState::Selected,
                Some((SketchEdgeHit::Arc(id), state)) if id == *arc_id => state,
                _ => ui::gizmos::HandleState::Idle,
            };
            let chords = chords
                .iter()
                .map(|px| egui::Pos2::new(px.x / pixels_per_point, px.y / pixels_per_point))
                .collect();
            self.sketch_arc_lines.push(ui::chrome::SketchCurveLine {
                chords,
                state,
                ink: curve_ink(construction_arcs.contains(arc_id)),
            });
        }
        for (circle_id, ring) in &self.sketch_circle_chords {
            let picked = ui::panel::SelectionTarget::SketchCircle {
                sketch: target,
                entity: *circle_id,
            };
            let selected = self.panel_state.selection.contains(picked);
            let state = match hovered_edge {
                _ if selected => ui::gizmos::HandleState::Selected,
                Some((SketchEdgeHit::Circle(id), state)) if id == *circle_id => state,
                _ => ui::gizmos::HandleState::Idle,
            };
            let chords = ring
                .iter()
                .map(|px| egui::Pos2::new(px.x / pixels_per_point, px.y / pixels_per_point))
                .collect();
            self.sketch_arc_lines.push(ui::chrome::SketchCurveLine {
                chords,
                state,
                ink: curve_ink(construction_circles.contains(circle_id)),
            });
        }
        // Every span of an aggregate reads the SAME state, resolved from the aggregate identity —
        // so selecting an ellipse lights all four quarters and hovering one span lights the
        // whole spline. Anything per-span here would let one object draw in two states at once.
        for (entity, chords) in &self.sketch_higher_curve_chords {
            let picked = ui::panel::SelectionTarget::SketchHigherCurve {
                sketch: target,
                curve: *entity,
            };
            let selected = self.panel_state.selection.contains(picked);
            let state = match hovered_edge {
                _ if selected => ui::gizmos::HandleState::Selected,
                Some((SketchEdgeHit::HigherCurve(curve), state)) if curve == *entity => state,
                _ => ui::gizmos::HandleState::Idle,
            };
            let chords = chords
                .iter()
                .map(|px| egui::Pos2::new(px.x / pixels_per_point, px.y / pixels_per_point))
                .collect();
            self.sketch_arc_lines.push(ui::chrome::SketchCurveLine {
                chords,
                state,
                ink: curve_ink(construction_higher.contains(entity)),
            });
        }

        // A control-point spline's frame, in construction ink and under the spline's own state:
        // hovering a leg lights the curve it steers, because that is what the leg resolves to.
        for (spline, legs) in &self.sketch_spline_polygons {
            let picked = ui::panel::SelectionTarget::SketchHigherCurve {
                sketch: target,
                curve: *spline,
            };
            let selected = self.panel_state.selection.contains(picked);
            let state = match hovered_edge {
                _ if selected => ui::gizmos::HandleState::Selected,
                Some((SketchEdgeHit::HigherCurve(curve), state)) if curve == *spline => state,
                _ => ui::gizmos::HandleState::Idle,
            };
            let chords = legs
                .iter()
                .map(|px| egui::Pos2::new(px.x / pixels_per_point, px.y / pixels_per_point))
                .collect();
            self.sketch_arc_lines.push(ui::chrome::SketchCurveLine {
                chords,
                state,
                ink: ui::chrome::SketchCurveInk::Construction,
            });
        }

        // The tangent levers, in their own teal ink, each under the state of the FIT POINT it
        // belongs to. Last of the curve pushes, so a lever draws OVER the curve it steers rather
        // than under it: the handle is what the author is reaching for, and it is the thinner mark
        // of the two.
        //
        // Reading its own point rather than the spline is what makes one lever light instead of
        // all of them. A spline carries a lever per fit point; taking the curve's state would
        // paint every one of them for a cursor that is over exactly one.
        // Under Select alone: a lever is a manipulator, and every other tool is reaching past it
        // for geometry. Not read off `hovered_edge`, which a lever deliberately suppresses.
        let hovered_lever = (tool == ui::panel::SketchTool::Select)
            .then(|| {
                self.last_cursor_position
                    .and_then(|(cx, cy)| self.tangent_lever_at(cx, cy))
            })
            .flatten();
        for (fit, run) in &self.sketch_tangent_levers {
            let picked = ui::panel::SelectionTarget::SketchPoint {
                sketch: target,
                entity: *fit,
            };
            let state = if self.panel_state.selection.contains(picked) {
                ui::gizmos::HandleState::Selected
            } else if hovered_lever == Some(*fit) {
                ui::gizmos::HandleState::Hover
            } else {
                ui::gizmos::HandleState::Idle
            };
            let chords = run
                .iter()
                .map(|px| egui::Pos2::new(px.x / pixels_per_point, px.y / pixels_per_point))
                .collect();
            self.sketch_arc_lines.push(ui::chrome::SketchCurveLine {
                chords,
                state,
                ink: ui::chrome::SketchCurveInk::TangentLever,
            });
        }

        // Generated operator instances draw from their regenerated curves, but never enter the
        // authored hit-test caches above. Selecting or constraining an instance would imply an
        // entity id and an independent solver coordinate it intentionally does not own.
        if let (Some((producer, _)), Some(context)) = (
            self.sketch_node_state(target),
            self.sketch_evaluation_context(),
        ) {
            for derived in producer.sketch.derived_pattern_curves(context) {
                let points = break_piece_points(&derived.geometry);
                let projected: Option<Vec<egui::Pos2>> = points
                    .into_iter()
                    .map(|point| {
                        to_viewport_px(point).map(|px| {
                            egui::Pos2::new(px.x / pixels_per_point, px.y / pixels_per_point)
                        })
                    })
                    .collect();
                if let Some(projected) = projected {
                    // An instance of a construction source is construction too — the pattern
                    // copies the geometry's role along with its shape.
                    self.sketch_arc_lines.push(ui::chrome::SketchCurveLine {
                        chords: projected,
                        state: ui::gizmos::HandleState::Idle,
                        ink: curve_ink(construction(derived.role)),
                    });
                }
            }
        }

        // The support under a point the solve carried past its curve's own end. A point-on-curve
        // residual reads the support on purpose, so the point IS on the curve and the drawing is
        // what misleads: it shows only the piece the author cut. Drawing the rest says the true
        // thing. Asked of the sketch in place, like the faces below and for the same reason.
        if let (Some((producer, _)), Some(context)) = (
            self.sketch_node_state(target),
            self.sketch_evaluation_context(),
        ) {
            for (_, reach) in producer.sketch.undrawn_reaches(context) {
                let points = match reach {
                    parametric::sketch::UndrawnReach::Span { from, to } => vec![from, to],
                    parametric::sketch::UndrawnReach::Sweep {
                        center,
                        radius,
                        from_radians,
                        sweep_radians,
                    } => {
                        // A reach is named by a turn and `arc_interior_points_within` walks between
                        // two stored ends, so the turn's ends are struck on the rim first. From
                        // there it is the same flattening every other curve on the page gets,
                        // against the same screen tolerance — a reach is a piece of its curve, and
                        // drawing it to a coarser standard would show its host as a polygon.
                        let rim = |bearing: f64| {
                            [
                                radius.mul_add(bearing.cos(), center[0]),
                                radius.mul_add(bearing.sin(), center[1]),
                            ]
                        };
                        let (from, to) = (rim(from_radians), rim(from_radians + sweep_radians));
                        let mut points = vec![from];
                        points.extend(
                            document::sketch::arc_interior_points_within(
                                from,
                                to,
                                sweep_radians.to_degrees(),
                                screen_chord_tolerance(center, from, radius),
                            )
                            .iter()
                            .map(document::sketch::SketchPoint::in_plane),
                        );
                        points.push(to);
                        points
                    }
                };
                let projected: Option<Vec<egui::Pos2>> = points
                    .into_iter()
                    .map(|point| {
                        to_viewport_px(point).map(|px| {
                            egui::Pos2::new(px.x / pixels_per_point, px.y / pixels_per_point)
                        })
                    })
                    .collect();
                if let Some(chords) = projected {
                    self.sketch_arc_lines.push(ui::chrome::SketchCurveLine {
                        chords,
                        state: ui::gizmos::HandleState::Idle,
                        ink: ui::chrome::SketchCurveInk::UndrawnReach,
                    });
                }
            }
        }

        // The derived faces (#100), in physical px for the right-press hit-test. Derivation is a
        // graph walk over the sketch's own entities, and it is asked for here once per frame. The
        // WASH is not projected here at all: it is a GPU pass over the plane
        // (`SketchRegionRenderer`), so no boundary is projected for it and nesting is the region
        // field's business, not the overlay's.
        //
        // Asked of the sketch IN PLACE, never of a copy. `RegionMemo` clones EMPTY — a copy of a
        // sketch is the same sketch, cache or no cache — so a cloned sketch re-runs the whole
        // arrangement every frame and throws the result away with the copy. On a document holding
        // a couple of splines that alone is a frame's entire budget.
        // A behind-camera boundary vertex culls the whole outline, as it culls an arc.
        let project = |boundary: &[document::sketch::SketchPoint]| -> Option<Vec<egui::Pos2>> {
            boundary
                .iter()
                .map(|point| {
                    let render = handles.profile_to_render(point.in_plane());
                    let clip =
                        view_projection * glam::Vec4::new(render[0], render[1], render[2], 1.0);
                    (clip.w > 0.0).then(|| {
                        egui::Pos2::new(
                            vx + (clip.x / clip.w * 0.5 + 0.5) * vw,
                            vy + (1.0 - (clip.y / clip.w * 0.5 + 0.5)) * vh,
                        )
                    })
                })
                .collect()
        };
        if let Some(context) = self.sketch_evaluation_context() {
            let projected_faces: Vec<(usize, Vec<egui::Pos2>)> = self
                .panel_state
                .scene
                .node_by_id(target)
                .and_then(|node| match &node.content {
                    document::scene::NodeContent::SketchTool { producer, .. } => {
                        Some(&producer.sketch)
                    }
                    _ => None,
                })
                .map(|sketch| {
                    sketch
                        .faces(context)
                        .into_iter()
                        .enumerate()
                        .filter_map(|(index, face)| {
                            // A hit-test polygon IS discrete, so this is a terminal adapter: it
                            // flattens here rather than asking the region for a coarser boundary.
                            let boundary = document::sketch::flatten_edges(
                                &face.boundary,
                                document::sketch::ARC_SAGITTA_TOLERANCE,
                            );
                            project(&boundary).map(|projected| (index, projected))
                        })
                        .collect()
                })
                .unwrap_or_default();
            self.sketch_face_polygons.extend(projected_faces);
        }

        // Add-point insert preview: the point on the hovered segment nearest the cursor (physical
        // px), in egui points — "a vertex lands here on this edge". Drawn as a diamond next frame.
        if tool == ui::panel::SketchTool::AddPoint {
            if let Some((cursor_x, cursor_y)) = self.last_cursor_position {
                if let Some((_, a, b)) = self.nearest_sketch_segment(cursor_x, cursor_y) {
                    let cursor = egui::Pos2::new(cursor_x as f32, cursor_y as f32);
                    let foot = closest_point_on_segment(cursor, a, b);
                    self.sketch_insert_preview = Some(egui::Pos2::new(
                        foot.x / pixels_per_point,
                        foot.y / pixels_per_point,
                    ));
                }
            }
        }

        // Drawing-tool previews (#99): the uncommitted geometry, snapped exactly as the click
        // will commit it (#96: through the same policy), so the dashed line never lies about
        // where the point lands.
        let snapped_screen = |coord: [f64; 2]| {
            let render = handles.profile_to_render(coord);
            project_to_screen(
                glam::Vec3::from_array(render),
                view_projection,
                viewport_px,
                pixels_per_point,
            )
        };
        // The quantity a live drag is being pulled onto, under everything else at the guide
        // weight — the same linetype a polygon's base circle takes, and for the same reason: it is
        // the thing the shape is being derived FROM, and it is never authored.
        if let Some(kept) = self.sketch_snap_ghost {
            let ring = circle_ring(
                kept.about,
                kept.radius,
                document::sketch::ARC_SAGITTA_TOLERANCE,
            );
            let projected: Vec<egui::Pos2> =
                ring.iter().copied().filter_map(snapped_screen).collect();
            if !ring.is_empty() && projected.len() == ring.len() {
                // At the strength that says how much room is LEFT — see `ghost_ink` for why that
                // is not the same as how hard the quantity is being held, and for the two
                // measurements that chose between them.
                self.sketch_draw_preview
                    .push(fading_guide(projected, kept.ghost_ink() as f32));
            }
        }
        match tool {
            ui::panel::SketchTool::Line => {
                if let (Some(chain), Some((cursor_x, cursor_y))) =
                    (self.line_gesture.chain(), self.last_cursor_position)
                {
                    let profile_of = |id| {
                        self.sketch_node_state(target).and_then(|(producer, _)| {
                            producer
                                .sketch
                                .points()
                                .iter()
                                .find(|point| point.id == id)
                                .map(|point| point.at.in_plane())
                        })
                    };
                    if let (Some(from), Some(point)) = (
                        profile_of(chain.end),
                        self.sketch_target_at(cursor_x, cursor_y),
                    ) {
                        let to = point.at().in_plane();
                        let profile = if self.line_gesture.arc_is_latched() {
                            chain.incoming.and_then(|_| {
                                let (producer, _) = self.sketch_node_state(target)?;
                                let context = self.sketch_evaluation_context()?;
                                let candidate = self
                                    .line_gesture
                                    .tangent_arc_candidate(&producer, to, context)
                                    .ok()?;
                                let mut points = vec![from];
                                points.extend(
                                    document::sketch::arc_interior_points(
                                        from,
                                        to,
                                        candidate.sweep_radians.to_degrees(),
                                    )
                                    .iter()
                                    .map(|point| point.in_plane()),
                                );
                                points.push(to);
                                Some(points)
                            })
                        } else {
                            Some(vec![from, to])
                        };
                        if let Some(profile) = profile {
                            let projected: Vec<egui::Pos2> =
                                profile.iter().copied().filter_map(snapped_screen).collect();
                            if projected.len() == profile.len() {
                                self.sketch_draw_preview = vec![preview_outline(projected)];
                            }
                        }
                    }
                }
            }
            ui::panel::SketchTool::MidpointLine => {
                if let (Some((producer, _)), Some((cursor_x, cursor_y))) =
                    (self.sketch_node_state(target), self.last_cursor_position)
                {
                    let endpoint = self.sketch_target_at(cursor_x, cursor_y);
                    if let Some(placement) = endpoint.and_then(|endpoint| {
                        self.midpoint_line_gesture
                            .placement(target, &producer, endpoint)
                    }) {
                        let profile = [
                            placement.reflected.in_plane(),
                            placement.midpoint.in_plane(),
                            placement.endpoint.in_plane(),
                        ];
                        let projected: Vec<egui::Pos2> =
                            profile.iter().copied().filter_map(snapped_screen).collect();
                        if projected.len() == profile.len() {
                            self.sketch_draw_preview = vec![preview_outline(projected)];
                        }
                    }
                }
            }
            ui::panel::SketchTool::ArcTangent => {
                if let (Some((producer, _)), Some((cursor_x, cursor_y)), Some(context)) = (
                    self.sketch_node_state(target),
                    self.last_cursor_position,
                    self.sketch_evaluation_context(),
                ) {
                    if self.tangent_arc_gesture.is_pending() {
                        let endpoint = self.sketch_target_at(cursor_x, cursor_y);
                        let attempt = endpoint.map(|endpoint| {
                            self.tangent_arc_gesture
                                .placement(target, &producer, endpoint, context)
                        });
                        match attempt {
                            Some(Ok(placement)) => {
                                let from = placement.seam.in_plane();
                                let to = placement.endpoint.in_plane();
                                let mut profile = vec![from];
                                profile.extend(
                                    document::sketch::arc_interior_points(
                                        from,
                                        to,
                                        placement.candidate.sweep_radians.to_degrees(),
                                    )
                                    .iter()
                                    .map(|point| point.in_plane()),
                                );
                                profile.push(to);
                                let projected: Vec<egui::Pos2> =
                                    profile.iter().copied().filter_map(snapped_screen).collect();
                                if projected.len() == profile.len() {
                                    self.sketch_draw_preview = vec![preview_outline(projected)];
                                }
                            }
                            // The refusal was swallowed here, so a tangent arc the tool cannot
                            // build just showed nothing — indistinguishable from a dead tool. It
                            // is said AT the cursor because moving the cursor is the fix.
                            Some(Err(refusal)) if refusal.is_about_the_cursor() => {
                                if let Some(at) = endpoint
                                    .and_then(|endpoint| snapped_screen(endpoint.at().in_plane()))
                                {
                                    self.sketch_draw_preview =
                                        vec![ui::chrome::SketchPreviewMark::Refused { at }];
                                }
                            }
                            Some(Err(_)) | None => {}
                        }
                    }
                }
            }
            ui::panel::SketchTool::ArcCenterEndpoints => {
                if let (Some((producer, _)), Some((cursor_x, cursor_y))) =
                    (self.sketch_node_state(target), self.last_cursor_position)
                {
                    if let (Some(center), Some(direction)) = (
                        self.center_arc_gesture.center(target),
                        self.sketch_target_at(cursor_x, cursor_y),
                    ) {
                        if self.center_arc_gesture.start(target).is_some() {
                            // Which way round the arc goes is in the ROUTE the cursor took, so the
                            // reading has to be folded in before the preview asks for a direction.
                            self.center_arc_gesture.track_cursor(target, direction.at());
                            if let Some(placement) = self
                                .center_arc_gesture
                                .placement(target, &producer, direction)
                            {
                                let mut profile = vec![placement.start.in_plane()];
                                profile.extend(
                                    document::sketch::arc_interior_points(
                                        placement.start.in_plane(),
                                        placement.endpoint.in_plane(),
                                        placement.candidate.sweep_radians.to_degrees(),
                                    )
                                    .iter()
                                    .map(|point| point.in_plane()),
                                );
                                profile.push(placement.endpoint.in_plane());
                                let projected: Vec<egui::Pos2> =
                                    profile.iter().copied().filter_map(snapped_screen).collect();
                                if projected.len() == profile.len() {
                                    self.sketch_draw_preview = vec![preview_outline(projected)];
                                }
                            }
                        } else {
                            let profile = [center.in_plane(), direction.at().in_plane()];
                            let projected: Vec<egui::Pos2> =
                                profile.iter().copied().filter_map(snapped_screen).collect();
                            if projected.len() == profile.len() {
                                self.sketch_draw_preview = vec![preview_outline(projected)];
                            }
                        }
                    }
                }
            }
            // Both corner grammars preview through the gesture's own placement, so the loop
            // drawn under the cursor is the loop the second click authors.
            ui::panel::SketchTool::Rectangle | ui::panel::SketchTool::RectangleCenterCorner => {
                if let (Some(kind), Some((producer, _)), Some(cursor)) = (
                    corner_rectangle_kind(tool),
                    self.sketch_node_state(target),
                    self.last_cursor_position
                        .and_then(|(x, y)| self.sketch_target_at(x, y)),
                ) {
                    if let Some(placement) = self
                        .corner_rectangle_gesture
                        .placement(target, kind, &producer, cursor)
                    {
                        let mut ring: Vec<[f64; 2]> = placement
                            .corners
                            .iter()
                            .map(document::sketch::SketchPoint::in_plane)
                            .collect();
                        ring.push(placement.corners[0].in_plane());
                        let projected: Vec<egui::Pos2> =
                            ring.iter().copied().filter_map(snapped_screen).collect();
                        // A behind-camera corner culls the whole preview rather than
                        // drawing a broken ring.
                        if projected.len() == ring.len() {
                            self.sketch_draw_preview = vec![preview_outline(projected)];
                        }
                    }
                }
            }
            ui::panel::SketchTool::Rectangle3Point => {
                if let (Some((producer, _)), Some((cursor_x, cursor_y)), Some(cursor)) = (
                    self.sketch_node_state(target),
                    self.last_cursor_position,
                    self.last_cursor_position
                        .and_then(|(x, y)| self.sketch_target_at(x, y)),
                ) {
                    let ring = if let Some(placement) = self
                        .three_point_rectangle_gesture
                        .placement(target, &producer, cursor)
                    {
                        let mut ring: Vec<[f64; 2]> = placement
                            .corners
                            .iter()
                            .map(document::sketch::SketchPoint::in_plane)
                            .collect();
                        ring.push(placement.corners[0].in_plane());
                        Some(ring)
                    } else {
                        self.three_point_rectangle_gesture.guide(target).and_then(
                            |(first, second)| {
                                second.is_none().then_some(vec![
                                    first.in_plane(),
                                    self.sketch_target_at(cursor_x, cursor_y)?.at().in_plane(),
                                ])
                            },
                        )
                    };
                    if let Some(ring) = ring {
                        let projected: Vec<egui::Pos2> =
                            ring.iter().copied().filter_map(snapped_screen).collect();
                        if projected.len() == ring.len() {
                            self.sketch_draw_preview = vec![preview_outline(projected)];
                        }
                    }
                }
            }
            ui::panel::SketchTool::Select => {
                // Slice 3: the marquee rubber band, once the press travels past the click
                // threshold. Direction is read LIVE from the cursor, so the style flips
                // mid-drag with the semantic (solid window / dashed crossing).
                if let (Some((down_x, down_y)), Some((cursor_x, cursor_y))) =
                    (self.sketch_marquee_anchor, self.last_cursor_position)
                {
                    let past_threshold = (cursor_x - down_x).abs()
                        >= VIEW_CUBE_DRAG_THRESHOLD_PIXELS
                        || (cursor_y - down_y).abs() >= VIEW_CUBE_DRAG_THRESHOLD_PIXELS;
                    if past_threshold {
                        // The same box the release will select with, in points rather than pixels
                        // — see `sketch_marquee_box_px`. Drawn from the clamped rectangle so the
                        // band stops where the selection stops.
                        let box_px =
                            self.sketch_marquee_box_px((down_x, down_y), (cursor_x, cursor_y));
                        let rect = egui::Rect::from_min_max(
                            (box_px.min.to_vec2() / pixels_per_point).to_pos2(),
                            (box_px.max.to_vec2() / pixels_per_point).to_pos2(),
                        );
                        self.sketch_marquee_band = Some((rect, cursor_x >= down_x));
                    }
                }
            }
            ui::panel::SketchTool::ThreePointArc => {
                // #102: with one endpoint down, the dashed chord to the snapped cursor (the
                // curve is not determined yet); with both down, the arc the third click would
                // commit, tessellated through the cursor as its through-point.
                if let (Some((start, end)), Some((cursor_x, cursor_y))) =
                    (self.sketch_arc_gesture, self.last_cursor_position)
                {
                    let profile_of = |id| {
                        self.sketch_node_state(target).and_then(|(producer, _)| {
                            producer
                                .sketch
                                .points()
                                .iter()
                                .find(|point| point.id == id)
                                .map(|point| point.at.in_plane())
                        })
                    };
                    let cursor = self
                        .sketch_snapped_point_at(cursor_x, cursor_y)
                        .map(|point| point.in_plane());
                    let ring: Option<Vec<[f64; 2]>> = match (profile_of(start), end, cursor) {
                        (Some(from), None, Some(cursor)) => Some(vec![from, cursor]),
                        (Some(from), Some(end), Some(through)) => profile_of(end).map(|to| {
                            let sweep =
                                document::sketch::included_angle_through_degrees(from, to, through)
                                    .unwrap_or(0.0);
                            let mut ring = vec![from];
                            ring.extend(
                                document::sketch::arc_interior_points(from, to, sweep)
                                    .iter()
                                    .map(|point| point.in_plane()),
                            );
                            ring.push(to);
                            ring
                        }),
                        _ => None,
                    };
                    if let Some(ring) = ring {
                        let projected: Vec<egui::Pos2> =
                            ring.iter().copied().filter_map(snapped_screen).collect();
                        // A behind-camera vertex culls the whole preview, as the rectangle's does.
                        if projected.len() == ring.len() {
                            self.sketch_draw_preview = vec![preview_outline(projected)];
                        }
                    }
                }
            }
            ui::panel::SketchTool::CircleCenterDiameter => {
                if let (Some(center), Some((cursor_x, cursor_y))) =
                    (self.sketch_circle_center, self.last_cursor_position)
                {
                    if let Some(perimeter) = self.sketch_snapped_point_at(cursor_x, cursor_y) {
                        let center = center.in_plane();
                        let perimeter = perimeter.in_plane();
                        let radius = (perimeter[0] - center[0]).hypot(perimeter[1] - center[1]);
                        let ring =
                            circle_ring(center, radius, document::sketch::ARC_SAGITTA_TOLERANCE);
                        let projected: Vec<egui::Pos2> =
                            ring.iter().copied().filter_map(snapped_screen).collect();
                        if projected.len() == ring.len() {
                            self.sketch_draw_preview = vec![preview_outline(projected)];
                        }
                    }
                }
            }
            ui::panel::SketchTool::Circle2Point | ui::panel::SketchTool::Circle3Point => {
                if let (Some(kind), Some((producer, _)), Some((cursor_x, cursor_y))) = (
                    point_circle_kind(tool),
                    self.sketch_node_state(target),
                    self.last_cursor_position,
                ) {
                    let placement = self
                        .sketch_target_at(cursor_x, cursor_y)
                        .and_then(|cursor| {
                            self.point_circle_gesture
                                .placement(target, kind, &producer, cursor)
                        });
                    if let Some(placement) = placement {
                        let ring = circle_ring(
                            placement.candidate.center,
                            placement.candidate.radius,
                            document::sketch::ARC_SAGITTA_TOLERANCE,
                        );
                        let projected: Vec<egui::Pos2> =
                            ring.iter().copied().filter_map(snapped_screen).collect();
                        if projected.len() == ring.len() {
                            self.sketch_draw_preview = vec![preview_outline(projected)];
                        }
                    }
                }
            }
            ui::panel::SketchTool::Circle2Tangent | ui::panel::SketchTool::Circle3Tangent => {
                if let (Some(kind), Some((producer, _)), Some((cursor_x, cursor_y))) = (
                    tangent_circle_kind(tool),
                    self.sketch_node_state(target),
                    self.last_cursor_position,
                ) {
                    let cursor = self.sketch_snapped_point_at(cursor_x, cursor_y);
                    let hovered = self.sketch_segment_at(cursor_x, cursor_y);
                    let placement = cursor.and_then(|cursor| {
                        self.tangent_circle_gesture
                            .placement(target, kind, &producer, cursor, hovered)
                    });
                    if let Some(placement) = placement {
                        let ring = circle_ring(
                            placement.center.in_plane(),
                            placement.radius.value(),
                            document::sketch::ARC_SAGITTA_TOLERANCE,
                        );
                        let projected: Vec<egui::Pos2> =
                            ring.iter().copied().filter_map(snapped_screen).collect();
                        if projected.len() == ring.len() {
                            self.sketch_draw_preview = vec![preview_outline(projected)];
                        }
                    }
                }
            }
            ui::panel::SketchTool::PolygonInscribed
            | ui::panel::SketchTool::PolygonCircumscribed
            | ui::panel::SketchTool::PolygonEdge => {
                if let (Some(kind), Some((producer, _)), Some((cursor_x, cursor_y))) = (
                    polygon_kind(tool),
                    self.sketch_node_state(target),
                    self.last_cursor_position,
                ) {
                    let cursor = self.sketch_target_at(cursor_x, cursor_y);
                    let sides = normalized_polygon_sides(self.panel_state.sketch_polygon_sides);
                    let placement = cursor.and_then(|cursor| {
                        self.polygon_gesture
                            .placement(target, kind, &producer, cursor, sides)
                    });
                    let ring = placement
                        .as_ref()
                        .map(|placement| {
                            let mut ring: Vec<[f64; 2]> = placement
                                .vertices
                                .iter()
                                .map(document::sketch::SketchPoint::in_plane)
                                .collect();
                            ring.push(placement.vertices[0].in_plane());
                            ring
                        })
                        .or_else(|| {
                            let cursor = cursor?;
                            self.polygon_gesture
                                .guide(target, kind)
                                .and_then(|(first, second)| {
                                    second
                                        .is_none()
                                        .then_some(vec![first.in_plane(), cursor.at().in_plane()])
                                })
                        });
                    if let Some(ring) = ring {
                        let projected: Vec<egui::Pos2> =
                            ring.iter().copied().filter_map(snapped_screen).collect();
                        if projected.len() == ring.len() {
                            self.sketch_draw_preview = vec![preview_outline(projected)];
                        }
                    }
                    // The circle the polygon is BEING FITTED TO — the whole meaning of
                    // "inscribed" versus "circumscribed", which the vertex ring alone cannot
                    // show. Inscribed touches it at the vertices, circumscribed at the edge
                    // midpoints; an edge-defined polygon rests on no circle, so it gets none.
                    if let Some(base) = placement
                        .as_ref()
                        .and_then(|placement| polygon_base_circle(kind, placement))
                    {
                        let projected: Vec<egui::Pos2> =
                            base.iter().copied().filter_map(snapped_screen).collect();
                        if projected.len() == base.len() {
                            self.sketch_draw_preview.push(preview_guide(projected));
                        }
                    }
                }
            }
            ui::panel::SketchTool::SlotCenterToCenter
            | ui::panel::SketchTool::SlotOverall
            | ui::panel::SketchTool::SlotCenterPoint
            | ui::panel::SketchTool::SlotCenterPointArc
            | ui::panel::SketchTool::Slot3PointArc => {
                if let (Some(kind), Some((producer, _)), Some((cursor_x, cursor_y))) = (
                    slot_kind(tool),
                    self.sketch_node_state(target),
                    self.last_cursor_position,
                ) {
                    let cursor = self.sketch_target_at(cursor_x, cursor_y);
                    // Which way round a center-arc spine goes is in the ROUTE the cursor took, so
                    // the reading has to be folded in before the preview asks for a direction.
                    if let Some(cursor) = cursor {
                        self.slot_gesture.track_cursor(target, kind, cursor.at());
                    }
                    // An arc slot's CENTERLINE, from the moment its picks settle an arc.
                    let spine = cursor
                        .and_then(|cursor| self.slot_gesture.spine(target, kind, cursor.at()))
                        .map(|spine| arc_spine_points(&spine));
                    let ring = cursor.and_then(|cursor| {
                        self.slot_gesture
                            .placement(target, kind, &producer, cursor)
                            .map(|placement| slot_ring(&placement))
                            .or_else(|| {
                                // A straight run through the picks is all a PARTIAL grammar can
                                // say, and it looks nothing like the arc the slot is about to be
                                // swept around. Once the arc exists it is the better answer, and
                                // drawing both leaves the polyline underneath it.
                                if spine.is_some() {
                                    return None;
                                }
                                self.slot_gesture.guide(target, kind).map(|mut guide| {
                                    guide.push(cursor.at());
                                    guide.into_iter().map(|point| point.in_plane()).collect()
                                })
                            })
                    });
                    if let Some(ring) = ring {
                        let projected: Vec<egui::Pos2> =
                            ring.iter().copied().filter_map(snapped_screen).collect();
                        if projected.len() == ring.len() {
                            self.sketch_draw_preview = vec![preview_outline(projected)];
                        }
                    }
                    if let Some(spine) = spine {
                        let projected: Vec<egui::Pos2> =
                            spine.iter().copied().filter_map(snapped_screen).collect();
                        if projected.len() == spine.len() {
                            self.sketch_draw_preview.push(preview_guide(projected));
                        }
                    }
                }
            }
            ui::panel::SketchTool::Ellipse
            | ui::panel::SketchTool::Conic
            | ui::panel::SketchTool::FitPointSpline
            | ui::panel::SketchTool::ControlPointSpline => {
                if let (Some(kind), Some((cursor_x, cursor_y))) =
                    (higher_curve_kind(tool), self.last_cursor_position)
                {
                    if let Some(cursor) = self.sketch_snapped_point_at(cursor_x, cursor_y) {
                        let profile = self.higher_curve_gesture.preview(target, kind, cursor);
                        let projected: Vec<egui::Pos2> =
                            profile.iter().copied().filter_map(snapped_screen).collect();
                        if projected.len() == profile.len() {
                            self.sketch_draw_preview = vec![preview_outline(projected)];
                        }
                        // The conic's last step is a gizmo, not a free pick: the shoulder slides on
                        // the track from the chord's midpoint out to the control point, and where
                        // it sits IS how hard the control point pulls. Drawing the track is what
                        // makes that a thing to grab rather than a number.
                        if let Some((track, shoulder)) = self
                            .higher_curve_gesture
                            .conic_shoulder_gizmo(target, kind, cursor)
                        {
                            let rail: Vec<egui::Pos2> =
                                track.iter().copied().filter_map(snapped_screen).collect();
                            if rail.len() == track.len() {
                                self.sketch_draw_preview.push(preview_guide(rail));
                            }
                            if let Some(at) = snapped_screen(shoulder) {
                                self.sketch_draw_preview
                                    .push(ui::chrome::SketchPreviewMark::Point { at });
                            }
                        }
                        // A refused control point still draws its pick polyline, which reads
                        // exactly like a gesture mid-flight. The mark is what separates "keep
                        // going" from "this click buys nothing"; it sits at the cursor because
                        // moving the cursor is the fix.
                        if self
                            .higher_curve_gesture
                            .refuses_cursor(target, kind, cursor)
                        {
                            if let Some(at) = snapped_screen(cursor.in_plane()) {
                                self.sketch_draw_preview
                                    .push(ui::chrome::SketchPreviewMark::Refused { at });
                            }
                        }
                    }
                }
            }
            ui::panel::SketchTool::BreakCurve => {
                if let (Some((producer, _)), Some((cursor_x, cursor_y)), Some(context)) = (
                    self.sketch_node_state(target),
                    self.last_cursor_position,
                    document::sketch::evaluation_context_from_density(
                        self.panel_state.geometry.voxels_per_block,
                    ),
                ) {
                    let placement = self
                        .nearest_sketch_edge(cursor_x, cursor_y)
                        .and_then(|hit| {
                            producer
                                .break_placement(sketch_curve_from_hit(hit), context)
                                .ok()
                        });
                    if let Some(placement) = placement {
                        let ring: Vec<[f64; 2]> = placement
                            .pieces
                            .iter()
                            .flat_map(break_piece_points)
                            .collect();
                        let projected: Vec<egui::Pos2> =
                            ring.iter().copied().filter_map(snapped_screen).collect();
                        if projected.len() == ring.len() {
                            self.sketch_draw_preview = vec![preview_outline(projected)];
                        }
                    }
                }
            }
            ui::panel::SketchTool::Trim => {
                if let (Some((producer, _)), Some((cursor_x, cursor_y)), Some(context)) = (
                    self.sketch_node_state(target),
                    self.last_cursor_position,
                    document::sketch::evaluation_context_from_density(
                        self.panel_state.geometry.voxels_per_block,
                    ),
                ) {
                    let placement = self
                        .nearest_sketch_edge(cursor_x, cursor_y)
                        .zip(self.sketch_unsnapped_profile_coord(cursor_x, cursor_y))
                        .and_then(|(hit, witness)| {
                            producer
                                .trim_placement(sketch_curve_from_hit(hit), witness, context)
                                .ok()
                        });
                    if let Some(placement) = placement {
                        let ring: Vec<[f64; 2]> =
                            placement.kept.iter().flat_map(break_piece_points).collect();
                        let projected: Vec<egui::Pos2> =
                            ring.iter().copied().filter_map(snapped_screen).collect();
                        if projected.len() == ring.len() {
                            self.sketch_draw_preview = vec![preview_outline(projected)];
                        }
                    }
                }
            }
            ui::panel::SketchTool::Extend => {
                if let (Some((producer, _)), Some((cursor_x, cursor_y)), Some(context)) = (
                    self.sketch_node_state(target),
                    self.last_cursor_position,
                    document::sketch::evaluation_context_from_density(
                        self.panel_state.geometry.voxels_per_block,
                    ),
                ) {
                    let placement = self
                        .nearest_sketch_edge(cursor_x, cursor_y)
                        .zip(self.sketch_unsnapped_profile_coord(cursor_x, cursor_y))
                        .and_then(|(hit, witness)| {
                            producer
                                .extend_placement(sketch_curve_from_hit(hit), witness, context)
                                .ok()
                        });
                    if let Some(placement) = placement {
                        let ring = break_piece_points(&placement.extended);
                        let projected: Vec<egui::Pos2> =
                            ring.iter().copied().filter_map(snapped_screen).collect();
                        if projected.len() == ring.len() {
                            self.sketch_draw_preview = vec![preview_outline(projected)];
                        }
                    }
                }
            }
            ui::panel::SketchTool::Fillet => {
                if let (Some((producer, _)), Some((cursor_x, cursor_y)), Some(context)) = (
                    self.sketch_node_state(target),
                    self.last_cursor_position,
                    document::sketch::evaluation_context_from_density(
                        self.panel_state.geometry.voxels_per_block,
                    ),
                ) {
                    let placement = self
                        .nearest_sketch_segment(cursor_x, cursor_y)
                        .map(|(id, _, _)| id)
                        .zip(self.sketch_unsnapped_profile_coord(cursor_x, cursor_y))
                        .and_then(|(id, witness)| {
                            producer
                                .fillet_placement(
                                    document::sketch::SketchCurve::Segment(id),
                                    witness,
                                    context,
                                )
                                .ok()
                        });
                    if let Some(placement) = placement {
                        let ring: Vec<[f64; 2]> = [
                            &placement.shortened_first,
                            &placement.arc,
                            &placement.shortened_second,
                        ]
                        .into_iter()
                        .flat_map(break_piece_points)
                        .collect();
                        let projected: Vec<egui::Pos2> =
                            ring.iter().copied().filter_map(snapped_screen).collect();
                        if projected.len() == ring.len() {
                            self.sketch_draw_preview = vec![preview_outline(projected)];
                        }
                    }
                }
            }
            ui::panel::SketchTool::ChamferEqual
            | ui::panel::SketchTool::ChamferDistanceAngle
            | ui::panel::SketchTool::ChamferTwoDistance => {
                if let (Some((producer, _)), Some((cursor_x, cursor_y)), Some(context)) = (
                    self.sketch_node_state(target),
                    self.last_cursor_position,
                    document::sketch::evaluation_context_from_density(
                        self.panel_state.geometry.voxels_per_block,
                    ),
                ) {
                    let hovered = self.nearest_sketch_segment(cursor_x, cursor_y);
                    let witness = self.sketch_unsnapped_profile_coord(cursor_x, cursor_y);
                    let placement = match (self.sketch_chamfer_pending, hovered, witness) {
                        (Some(pending), Some((segment, _, _)), Some(second_witness))
                            if pending.target == target
                                && pending.tool == tool
                                && pending.second == segment =>
                        {
                            producer
                                .chamfer_placement(
                                    document::sketch::SketchCurve::Segment(pending.source),
                                    pending.first_witness,
                                    Some(second_witness),
                                    context,
                                )
                                .ok()
                        }
                        (None, Some((segment, _, _)), Some(first_witness)) => producer
                            .chamfer_placement(
                                document::sketch::SketchCurve::Segment(segment),
                                first_witness,
                                None,
                                context,
                            )
                            .ok(),
                        _ => None,
                    };
                    if let Some(placement) = placement {
                        let ring: Vec<[f64; 2]> = [
                            &placement.shortened_first,
                            &placement.connector,
                            &placement.shortened_second,
                        ]
                        .into_iter()
                        .flat_map(break_piece_points)
                        .collect();
                        let projected: Vec<egui::Pos2> =
                            ring.iter().copied().filter_map(snapped_screen).collect();
                        if projected.len() == ring.len() {
                            self.sketch_draw_preview = vec![preview_outline(projected)];
                        }
                    }
                }
            }
            ui::panel::SketchTool::Offset => {
                if let (
                    Some((producer, _)),
                    Some(pending),
                    Some((cursor_x, cursor_y)),
                    Some(context),
                ) = (
                    self.sketch_node_state(target),
                    self.sketch_offset_pending,
                    self.last_cursor_position,
                    document::sketch::evaluation_context_from_density(
                        self.panel_state.geometry.voxels_per_block,
                    ),
                ) {
                    let placement = self
                        .sketch_unsnapped_profile_coord(cursor_x, cursor_y)
                        .and_then(|witness| {
                            producer
                                .offset_placement(pending.source, witness, context)
                                .ok()
                        });
                    if let Some(placement) = placement {
                        let ring = break_piece_points(&placement.offset);
                        let projected: Vec<egui::Pos2> =
                            ring.iter().copied().filter_map(snapped_screen).collect();
                        if projected.len() == ring.len() {
                            self.sketch_draw_preview = vec![preview_outline(projected)];
                        }
                    }
                }
            }
            ui::panel::SketchTool::MoveCopy => {
                if let (
                    Some((producer, _)),
                    Some(pending),
                    Some((cursor_x, cursor_y)),
                    Some(context),
                ) = (
                    self.sketch_node_state(target),
                    self.sketch_move_copy_pending.as_ref(),
                    self.last_cursor_position,
                    document::sketch::evaluation_context_from_density(
                        self.panel_state.geometry.voxels_per_block,
                    ),
                ) {
                    if pending.target == target {
                        let placement = self
                            .sketch_unsnapped_profile_coord(cursor_x, cursor_y)
                            .and_then(|witness| {
                                let delta = [
                                    witness[0] - pending.anchor[0],
                                    witness[1] - pending.anchor[1],
                                ];
                                producer
                                    .translated_curve_preview(
                                        &pending.entities,
                                        delta,
                                        self.shift_held,
                                        context,
                                    )
                                    .ok()
                            });
                        if let Some(curves) = placement {
                            let ring: Vec<[f64; 2]> =
                                curves.iter().flat_map(break_piece_points).collect();
                            let projected: Vec<egui::Pos2> =
                                ring.iter().copied().filter_map(snapped_screen).collect();
                            if projected.len() == ring.len() {
                                self.sketch_draw_preview = vec![preview_outline(projected)];
                            }
                        }
                    }
                }
            }
            ui::panel::SketchTool::Scale => {
                if let (
                    Some((producer, _)),
                    Some(pending),
                    Some((cursor_x, cursor_y)),
                    Some(context),
                ) = (
                    self.sketch_node_state(target),
                    self.sketch_scale_pending.as_ref(),
                    self.last_cursor_position,
                    document::sketch::evaluation_context_from_density(
                        self.panel_state.geometry.voxels_per_block,
                    ),
                ) {
                    if pending.target == target {
                        let placement = self
                            .sketch_unsnapped_profile_coord(cursor_x, cursor_y)
                            .and_then(|witness| {
                                let radius = (witness[0] - pending.center[0])
                                    .hypot(witness[1] - pending.center[1]);
                                producer
                                    .scaled_curve_preview(
                                        &pending.entities,
                                        pending.center,
                                        radius / pending.base_radius,
                                        context,
                                    )
                                    .ok()
                            });
                        if let Some(curves) = placement {
                            let ring: Vec<[f64; 2]> =
                                curves.iter().flat_map(break_piece_points).collect();
                            let projected: Vec<egui::Pos2> =
                                ring.iter().copied().filter_map(snapped_screen).collect();
                            if projected.len() == ring.len() {
                                self.sketch_draw_preview = vec![preview_outline(projected)];
                            }
                        }
                    }
                }
            }
            ui::panel::SketchTool::Mirror => {
                if let (Some((producer, _)), Some((cursor_x, cursor_y)), Some(context)) = (
                    self.sketch_node_state(target),
                    self.last_cursor_position,
                    self.sketch_evaluation_context(),
                ) {
                    let preview = self
                        .nearest_sketch_segment(cursor_x, cursor_y)
                        .and_then(|(axis, _, _)| {
                            producer
                                .with_mirror_pattern(self.sketch_curve_selection(target), axis)
                                .ok()
                        })
                        .map(|next| newest_pattern_curves(&next, context));
                    if let Some(curves) = preview {
                        let ring: Vec<_> = curves.iter().flat_map(break_piece_points).collect();
                        let projected: Vec<_> =
                            ring.iter().copied().filter_map(snapped_screen).collect();
                        if projected.len() == ring.len() {
                            self.sketch_draw_preview = vec![preview_outline(projected)];
                        }
                    }
                }
            }
            ui::panel::SketchTool::RectangularPattern => {
                if let (
                    Some((producer, _)),
                    Some(pending),
                    Some((cursor_x, cursor_y)),
                    Some(context),
                ) = (
                    self.sketch_node_state(target),
                    self.sketch_rectangular_pattern_pending.as_ref(),
                    self.last_cursor_position,
                    self.sketch_evaluation_context(),
                ) {
                    if pending.target == target {
                        let witness = self
                            .sketch_unsnapped_profile_coord(cursor_x, cursor_y)
                            .unwrap_or(pending.anchor);
                        let cursor_step = [
                            witness[0] - pending.anchor[0],
                            witness[1] - pending.anchor[1],
                        ];
                        let configured = self
                            .panel_state
                            .sketch_pattern_counts
                            .map(|count| u32::from(count.clamp(1, 128)));
                        let (counts, steps) = match pending.first_step {
                            Some(first) => (configured, [first, cursor_step]),
                            None => ([configured[0], 1], [cursor_step, [0.0, 0.0]]),
                        };
                        let preview = producer
                            .with_rectangular_pattern(
                                pending.sources.iter().copied(),
                                counts,
                                steps.map(|step| {
                                    document::sketch::SketchVector::from_continuous(
                                        step[0], step[1],
                                    )
                                }),
                            )
                            .ok()
                            .map(|next| newest_pattern_curves(&next, context));
                        if let Some(curves) = preview {
                            let ring: Vec<_> = curves.iter().flat_map(break_piece_points).collect();
                            let projected: Vec<_> =
                                ring.iter().copied().filter_map(snapped_screen).collect();
                            if projected.len() == ring.len() {
                                self.sketch_draw_preview = vec![preview_outline(projected)];
                            }
                        }
                    }
                }
            }
            ui::panel::SketchTool::CircularPattern => {
                if let (Some((producer, _)), Some((cursor_x, cursor_y)), Some(context)) = (
                    self.sketch_node_state(target),
                    self.last_cursor_position,
                    self.sketch_evaluation_context(),
                ) {
                    let preview = self
                        .sketch_vertex_at(cursor_x, cursor_y)
                        .and_then(|index| self.sketch_point_ids.get(index).copied())
                        .and_then(|center| {
                            let angle =
                                parametric::units::AngleMeasurement::try_from_degrees_f64(360.0)
                                    .ok()?;
                            producer
                                .with_circular_pattern(
                                    self.sketch_curve_selection(target),
                                    center,
                                    u32::from(
                                        self.panel_state
                                            .sketch_circular_pattern_count
                                            .clamp(2, 128),
                                    ),
                                    angle,
                                )
                                .ok()
                        })
                        .map(|next| newest_pattern_curves(&next, context));
                    if let Some(curves) = preview {
                        let ring: Vec<_> = curves.iter().flat_map(break_piece_points).collect();
                        let projected: Vec<_> =
                            ring.iter().copied().filter_map(snapped_screen).collect();
                        if projected.len() == ring.len() {
                            self.sketch_draw_preview = vec![preview_outline(projected)];
                        }
                    }
                }
            }
            ui::panel::SketchTool::AddPoint
            | ui::panel::SketchTool::FillRegion
            | ui::panel::SketchTool::CarveRegion => {}
        }

        // Every multi-step tool shows the points it has already taken, in ONE place rather than
        // per tool. A tool that consumes clicks and draws nothing until it has a whole shape reads
        // as a tool that is broken — the three-point circle showed nothing at all until the second
        // click. Whichever gesture is live answers; the rest hold no pending for this sketch.
        let taken = [
            self.point_circle_gesture.placed_points(target),
            self.polygon_gesture.placed_points(target),
            self.three_point_rectangle_gesture.placed_points(target),
            self.corner_rectangle_gesture.placed_points(target),
            self.center_arc_gesture.placed_points(target),
            self.midpoint_line_gesture.placed_points(target),
            self.slot_gesture.placed_points(target),
            self.higher_curve_gesture.placed_points(target),
            self.tangent_circle_gesture.placed_points(target),
        ];
        for at in taken
            .into_iter()
            .flatten()
            .filter_map(|point| snapped_screen(point.in_plane()))
        {
            self.sketch_draw_preview
                .push(ui::chrome::SketchPreviewMark::Point { at });
        }
    }
}

fn newest_pattern_curves(
    producer: &document::sketch::SketchSolid,
    context: parametric::EvaluationContext,
) -> Vec<substrate::curve_intersection::PlanarCurve> {
    let Some(pattern) = producer.sketch.patterns().last() else {
        return Vec::new();
    };
    producer
        .sketch
        .derived_pattern_curves(context)
        .into_iter()
        .filter(|curve| curve.pattern == pattern.id)
        .map(|curve| curve.geometry)
        .collect()
}

/// Which kind of sketch EDGE a cursor resolved to (#102) — the two entity stores share an id
/// space but not a vector, so the kind travels with the id rather than being re-derived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SketchEdgeHit {
    /// A straight segment.
    Segment(document::sketch::EntityId),
    /// An arc.
    Arc(document::sketch::EntityId),
    /// A circle.
    Circle(document::sketch::EntityId),
    /// A higher-order aggregate — Bézier, ellipse, conic or spline — named by the identity the
    /// author sees rather than by whichever span the cursor happened to land on.
    HigherCurve(document::sketch::SketchCurve),
}

fn break_piece_points(piece: &substrate::curve_intersection::PlanarCurve) -> Vec<[f64; 2]> {
    match *piece {
        substrate::curve_intersection::PlanarCurve::Segment { start, end } => vec![start, end],
        substrate::curve_intersection::PlanarCurve::Arc { sweep_radians, .. } => {
            let from = piece.start();
            let to = piece.end();
            let mut points = vec![from];
            points.extend(
                document::sketch::arc_interior_points_within(
                    from,
                    to,
                    sweep_radians.to_degrees(),
                    document::sketch::ARC_SAGITTA_TOLERANCE,
                )
                .iter()
                .map(document::sketch::SketchPoint::in_plane),
            );
            points.push(to);
            points
        }
        substrate::curve_intersection::PlanarCurve::RationalBezier(curve) => {
            curve.flatten(document::sketch::ARC_SAGITTA_TOLERANCE)
        }
    }
}

/// An arc slot's centerline, tessellated to chords the projection can take.
fn arc_spine_points(spine: &parametric::sketch::ArcSlotSpine) -> Vec<[f64; 2]> {
    let mut points = vec![spine.start];
    points.extend(
        document::sketch::arc_interior_points(
            spine.start,
            spine.end,
            spine.sweep_radians.to_degrees(),
        )
        .iter()
        .map(document::sketch::SketchPoint::in_plane),
    );
    points.push(spine.end);
    points
}

/// The circle a regular polygon is fitted to, as a projected-ready ring — or `None` for a grammar
/// that rests on no circle.
///
/// The two circle grammars differ ONLY in where the polygon touches the circle, so the circle is
/// the one mark that tells them apart: inscribed touches at the vertices, circumscribed at the
/// edge midpoints. An edge-defined polygon is built from a side, not a radius.
fn polygon_base_circle(
    kind: polygon::PolygonKind,
    placement: &document::sketch::PolygonPlacement,
) -> Option<Vec<[f64; 2]>> {
    let center = placement.center.in_plane();
    let first = placement.vertices.first()?.in_plane();
    let touch = match kind {
        polygon::PolygonKind::Inscribed => first,
        polygon::PolygonKind::Circumscribed => {
            let next = placement.vertices.get(1)?.in_plane();
            [(first[0] + next[0]) * 0.5, (first[1] + next[1]) * 0.5]
        }
        polygon::PolygonKind::Edge => return None,
    };
    let radius = (touch[0] - center[0]).hypot(touch[1] - center[1]);
    let ring = circle_ring(center, radius, document::sketch::ARC_SAGITTA_TOLERANCE);
    (!ring.is_empty()).then_some(ring)
}

fn circle_ring(center: [f64; 2], radius: f64, tolerance: f64) -> Vec<[f64; 2]> {
    if !radius.is_finite() || radius <= 0.0 {
        return Vec::new();
    }
    let mut ring = Vec::new();
    for quarter in 0..4 {
        let angle = std::f64::consts::FRAC_PI_2 * f64::from(quarter);
        let next = angle + std::f64::consts::FRAC_PI_2;
        let from = [
            center[0] + radius * angle.cos(),
            center[1] + radius * angle.sin(),
        ];
        let to = [
            center[0] + radius * next.cos(),
            center[1] + radius * next.sin(),
        ];
        ring.push(from);
        ring.extend(
            document::sketch::arc_interior_points_within(from, to, 90.0, tolerance)
                .iter()
                .map(|point| point.in_plane()),
        );
    }
    if let Some(first) = ring.first().copied() {
        ring.push(first);
    }
    ring
}

/// Advance the transient two-click circle gesture. The first point remains transient; the second
/// returns a complete center/perimeter pair and clears the pending center.
fn advance_circle_center_diameter_gesture(
    center: &mut Option<document::sketch::SketchPoint>,
    point: document::sketch::SketchPoint,
) -> Option<(document::sketch::SketchPoint, document::sketch::SketchPoint)> {
    match center.take() {
        Some(center) => Some((center, point)),
        None => {
            *center = Some(point);
            None
        }
    }
}

/// Produce the circle edit only when a complete gesture changes the sketch. `None` means no
/// transaction must reach history, including zero-radius and duplicate circles.
fn complete_circle_center_diameter(
    producer: &document::sketch::SketchSolid,
    center: document::sketch::SketchPoint,
    perimeter: document::sketch::SketchPoint,
) -> Option<document::sketch::SketchSolid> {
    let next = producer.with_circle_center_diameter(center, perimeter);
    (next != *producer).then_some(next)
}

/// Whether the pending circle center still belongs to the active tool and sketch.
fn circle_gesture_is_current(
    tool: ui::panel::SketchTool,
    target: document::scene::NodeId,
    pending_target: Option<document::scene::NodeId>,
) -> bool {
    tool == ui::panel::SketchTool::CircleCenterDiameter && pending_target == Some(target)
}

const fn higher_curve_kind(tool: ui::panel::SketchTool) -> Option<higher_curve::HigherCurveKind> {
    match tool {
        ui::panel::SketchTool::Ellipse => Some(higher_curve::HigherCurveKind::Ellipse),
        ui::panel::SketchTool::Conic => Some(higher_curve::HigherCurveKind::Conic),
        ui::panel::SketchTool::FitPointSpline => {
            Some(higher_curve::HigherCurveKind::FitPointSpline)
        }
        ui::panel::SketchTool::ControlPointSpline => {
            Some(higher_curve::HigherCurveKind::ControlPointSpline)
        }
        _ => None,
    }
}

const fn corner_rectangle_kind(
    tool: ui::panel::SketchTool,
) -> Option<corner_rectangle::CornerRectangleKind> {
    match tool {
        ui::panel::SketchTool::Rectangle => Some(corner_rectangle::CornerRectangleKind::TwoPoint),
        ui::panel::SketchTool::RectangleCenterCorner => {
            Some(corner_rectangle::CornerRectangleKind::CenterCorner)
        }
        _ => None,
    }
}

fn point_circle_kind(tool: ui::panel::SketchTool) -> Option<point_circle::PointCircleKind> {
    match tool {
        ui::panel::SketchTool::Circle2Point => Some(point_circle::PointCircleKind::TwoPoint),
        ui::panel::SketchTool::Circle3Point => Some(point_circle::PointCircleKind::ThreePoint),
        ui::panel::SketchTool::Select
        | ui::panel::SketchTool::AddPoint
        | ui::panel::SketchTool::Line
        | ui::panel::SketchTool::MidpointLine
        | ui::panel::SketchTool::Rectangle
        | ui::panel::SketchTool::Rectangle3Point
        | ui::panel::SketchTool::RectangleCenterCorner
        | ui::panel::SketchTool::ThreePointArc
        | ui::panel::SketchTool::ArcCenterEndpoints
        | ui::panel::SketchTool::ArcTangent
        | ui::panel::SketchTool::CircleCenterDiameter
        | ui::panel::SketchTool::Circle2Tangent
        | ui::panel::SketchTool::Circle3Tangent
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
        | ui::panel::SketchTool::CarveRegion => None,
    }
}

const fn polygon_kind(tool: ui::panel::SketchTool) -> Option<polygon::PolygonKind> {
    match tool {
        ui::panel::SketchTool::PolygonInscribed => Some(polygon::PolygonKind::Inscribed),
        ui::panel::SketchTool::PolygonCircumscribed => Some(polygon::PolygonKind::Circumscribed),
        ui::panel::SketchTool::PolygonEdge => Some(polygon::PolygonKind::Edge),
        ui::panel::SketchTool::Select
        | ui::panel::SketchTool::AddPoint
        | ui::panel::SketchTool::Line
        | ui::panel::SketchTool::MidpointLine
        | ui::panel::SketchTool::Rectangle
        | ui::panel::SketchTool::Rectangle3Point
        | ui::panel::SketchTool::RectangleCenterCorner
        | ui::panel::SketchTool::ThreePointArc
        | ui::panel::SketchTool::ArcCenterEndpoints
        | ui::panel::SketchTool::ArcTangent
        | ui::panel::SketchTool::CircleCenterDiameter
        | ui::panel::SketchTool::Circle2Point
        | ui::panel::SketchTool::Circle3Point
        | ui::panel::SketchTool::Circle2Tangent
        | ui::panel::SketchTool::Circle3Tangent
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
        | ui::panel::SketchTool::CarveRegion => None,
    }
}

const fn slot_kind(tool: ui::panel::SketchTool) -> Option<slot::SlotKind> {
    match tool {
        ui::panel::SketchTool::SlotCenterToCenter => Some(slot::SlotKind::CenterToCenter),
        ui::panel::SketchTool::SlotOverall => Some(slot::SlotKind::Overall),
        ui::panel::SketchTool::SlotCenterPoint => Some(slot::SlotKind::CenterPoint),
        ui::panel::SketchTool::SlotCenterPointArc => Some(slot::SlotKind::CenterPointArc),
        ui::panel::SketchTool::Slot3PointArc => Some(slot::SlotKind::ThreePointArc),
        ui::panel::SketchTool::Select
        | ui::panel::SketchTool::AddPoint
        | ui::panel::SketchTool::Line
        | ui::panel::SketchTool::MidpointLine
        | ui::panel::SketchTool::Rectangle
        | ui::panel::SketchTool::Rectangle3Point
        | ui::panel::SketchTool::RectangleCenterCorner
        | ui::panel::SketchTool::ThreePointArc
        | ui::panel::SketchTool::ArcCenterEndpoints
        | ui::panel::SketchTool::ArcTangent
        | ui::panel::SketchTool::CircleCenterDiameter
        | ui::panel::SketchTool::Circle2Point
        | ui::panel::SketchTool::Circle3Point
        | ui::panel::SketchTool::Circle2Tangent
        | ui::panel::SketchTool::Circle3Tangent
        | ui::panel::SketchTool::PolygonInscribed
        | ui::panel::SketchTool::PolygonCircumscribed
        | ui::panel::SketchTool::PolygonEdge
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
        | ui::panel::SketchTool::CarveRegion => None,
    }
}

const fn tangent_circle_kind(
    tool: ui::panel::SketchTool,
) -> Option<tangent_circle::TangentCircleKind> {
    match tool {
        ui::panel::SketchTool::Circle2Tangent => Some(tangent_circle::TangentCircleKind::Two),
        ui::panel::SketchTool::Circle3Tangent => Some(tangent_circle::TangentCircleKind::Three),
        ui::panel::SketchTool::Select
        | ui::panel::SketchTool::AddPoint
        | ui::panel::SketchTool::Line
        | ui::panel::SketchTool::MidpointLine
        | ui::panel::SketchTool::Rectangle
        | ui::panel::SketchTool::Rectangle3Point
        | ui::panel::SketchTool::RectangleCenterCorner
        | ui::panel::SketchTool::ThreePointArc
        | ui::panel::SketchTool::ArcCenterEndpoints
        | ui::panel::SketchTool::ArcTangent
        | ui::panel::SketchTool::CircleCenterDiameter
        | ui::panel::SketchTool::Circle2Point
        | ui::panel::SketchTool::Circle3Point
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
        | ui::panel::SketchTool::CarveRegion => None,
    }
}

fn slot_ring(placement: &document::sketch::SlotPlacement) -> Vec<[f64; 2]> {
    let mut ring = Vec::new();
    for edge in placement.edges {
        match edge {
            document::sketch::SlotEdgePlacement::Line { from, .. } => {
                ring.push(from.in_plane());
            }
            document::sketch::SlotEdgePlacement::Arc { from, to, sweep } => {
                ring.push(from.in_plane());
                ring.extend(
                    document::sketch::arc_interior_points(
                        from.in_plane(),
                        to.in_plane(),
                        sweep.to_degrees_f64(),
                    )
                    .iter()
                    .map(document::sketch::SketchPoint::in_plane),
                );
            }
        }
    }
    if let Some(first) = ring.first().copied() {
        ring.push(first);
    }
    ring
}

const fn normalized_polygon_sides(sides: u16) -> u16 {
    if sides >= 3 && sides <= 128 {
        sides
    } else {
        6
    }
}

/// The directional marquee predicate for the projected closed circle ring.
/// The marquee's box in physical pixels, clamped to `viewport_px` (`[x, y, w, h]`).
///
/// Free of the shell so it can be measured on its own; see
/// [`WindowedApp::sketch_marquee_box_px`](struct.WindowedApp.html) for why one rectangle serves
/// both the band and the selection.
fn marquee_box_px(viewport_px: [u32; 4], from: (f64, f64), to: (f64, f64)) -> egui::Rect {
    let [left, top, width, height] = viewport_px;
    let viewport = egui::Rect::from_min_size(
        egui::Pos2::new(left as f32, top as f32),
        egui::Vec2::new(width as f32, height as f32),
    );
    egui::Rect::from_two_pos(
        egui::Pos2::new(from.0 as f32, from.1 as f32),
        egui::Pos2::new(to.0 as f32, to.1 as f32),
    )
    .intersect(viewport)
}

fn circle_marquee_hit(ring: &[egui::Pos2], rect: egui::Rect, window: bool) -> bool {
    if window {
        ring.iter().all(|point| rect.contains(*point))
    } else {
        ring.array_windows::<2>()
            .any(|pair| segment_touches_rect(pair[0], pair[1], rect))
    }
}

/// The aggregates the marquee picks, each named once however many spans it drew.
///
/// A higher-order curve answers the box as ONE object, so its spans are folded before the
/// verdict: window needs every span enclosed, crossing needs only one touched. Each span uses
/// the closed-ring predicate, so window measures the DRAWN points rather than the span endpoints
/// the arc rule uses — an ellipse's quarter seams are interior to the shape and invisible to the
/// author, and a box holding only those four encloses nothing anybody can see. Pick follows the
/// drawn curve here for the same reason it does in the hit test (#102).
fn aggregate_marquee_picks(
    chords: &[(document::sketch::SketchCurve, Vec<egui::Pos2>)],
    rect: egui::Rect,
    window: bool,
) -> Vec<document::sketch::SketchCurve> {
    let mut verdicts: Vec<(document::sketch::SketchCurve, bool)> = Vec::new();
    for (curve, span) in chords {
        let hit = circle_marquee_hit(span, rect, window);
        match verdicts.iter_mut().find(|(seen, _)| seen == curve) {
            Some((_, verdict)) if window => *verdict &= hit,
            Some((_, verdict)) => *verdict |= hit,
            None => verdicts.push((*curve, hit)),
        }
    }
    verdicts
        .into_iter()
        .filter_map(|(curve, hit)| hit.then_some(curve))
        .collect()
}

/// Whether the cursor has left the press far enough that the gesture is a DRAG and not a click.
///
/// Free of the renderer so the rule can be tested without a window. Answered on either axis rather
/// than by distance, the way the view cube and the marquee answer it, so every gesture in the app
/// gives up being a click at the same reach.
fn pointer_left_the_press(press: Option<(f64, f64)>, now: Option<(f64, f64)>) -> bool {
    let (Some((down_x, down_y)), Some((now_x, now_y))) = (press, now) else {
        return false;
    };
    (now_x - down_x).abs() >= super::VIEW_CUBE_DRAG_THRESHOLD_PIXELS
        || (now_y - down_y).abs() >= super::VIEW_CUBE_DRAG_THRESHOLD_PIXELS
}

/// The fit point of the nearest tangent lever within `pad_px` of `cursor`, if any.
///
/// Free of the renderer so the rule can be tested without a window: the pad and the projected runs
/// are the whole of the input. See [`tangent_lever_at`](WindowedState::tangent_lever_at) for why the
/// answer is a fit point rather than the spline the lever steers.
fn nearest_tangent_lever(
    levers: &[(document::sketch::EntityId, Vec<egui::Pos2>)],
    cursor: egui::Pos2,
    pad_px: f32,
) -> Option<document::sketch::EntityId> {
    levers
        .iter()
        .filter_map(|(fit, run)| {
            let distance = run
                .array_windows::<2>()
                .map(|pair| point_to_segment_distance(cursor, pair[0], pair[1]))
                .min_by(|a, b| a.total_cmp(b))?;
            (distance <= pad_px).then_some((*fit, distance))
        })
        .min_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(fit, _)| fit)
}

/// Resolve the nearest already-qualified edge candidate. Equal distances keep candidate order,
/// with segments before arcs, then circles, then higher-order aggregates at the call site — so a
/// simple primitive wins an exact tie against the curve that merely passes through it.
fn nearest_sketch_edge_from_candidates<const N: usize>(
    candidates: [Option<(SketchEdgeHit, f32)>; N],
) -> Option<SketchEdgeHit> {
    candidates
        .into_iter()
        .flatten()
        .min_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(hit, _)| hit)
}

/// Resolve only candidates admitted by the current constraint question, then compare distance.
fn nearest_sketch_edge_for_requirement(
    requirement: ui::panel::PickRequirement,
    segment: Option<(SketchEdgeHit, f32)>,
    arc: Option<(SketchEdgeHit, f32)>,
    circle: Option<(SketchEdgeHit, f32)>,
) -> Option<SketchEdgeHit> {
    match requirement {
        ui::panel::PickRequirement::Segment | ui::panel::PickRequirement::PointOrLine => {
            segment.map(|(hit, _)| hit)
        }
        // The narrowed second slot filters by KIND rather than by an arm per curve kind, so a
        // curve kind added to the document needs nothing here and cannot silently fall through to
        // the general comparison.
        ui::panel::PickRequirement::MatchingCurve(like) => {
            nearest_sketch_edge_from_candidates([segment, arc, circle].map(|candidate| {
                candidate.filter(|(hit, _)| sketch_curve_from_hit(*hit).same_kind_as(like))
            }))
        }
        ui::panel::PickRequirement::Curve | ui::panel::PickRequirement::PointOrCurve => {
            nearest_sketch_edge_from_candidates([segment, arc, circle])
        }
        ui::panel::PickRequirement::CircularCurve => {
            nearest_sketch_edge_from_candidates([arc, circle])
        }
        // A circle is left out on purpose: it has no end, and an angle to a curve that turns is
        // read at one.
        ui::panel::PickRequirement::DirectedCurve => {
            nearest_sketch_edge_from_candidates([segment, arc])
        }
        ui::panel::PickRequirement::Point => None,
    }
}

/// Quantize a continuous in-plane profile coordinate by the sketch position snap (#96):
/// `NoSnap` carries the sub-voxel fraction on the point (#101), `Voxel` rounds to the plane's
/// own voxel grid, `Block` rounds to block boundaries. Every sketch vertex edit — drag,
/// add-point split, Line, rectangle — resolves through this one policy.
fn apply_sketch_snap(
    coord: [f64; 2],
    snap: ui::panel::PositionSnap,
    voxels_per_block: u32,
) -> document::sketch::SketchPoint {
    match snap {
        ui::panel::PositionSnap::NoSnap => {
            document::sketch::SketchPoint::from_continuous(coord[0], coord[1])
        }
        ui::panel::PositionSnap::Voxel => {
            document::sketch::SketchPoint::new(coord[0].round() as i64, coord[1].round() as i64)
        }
        ui::panel::PositionSnap::Block => {
            let block = voxels_per_block.max(1) as f64;
            document::sketch::SketchPoint::new(
                ((coord[0] / block).round() * block) as i64,
                ((coord[1] / block).round() * block) as i64,
            )
        }
    }
}

/// Project a point in the camera's render frame to egui points inside `viewport_px`, or `None`
/// when it sits behind the camera. The two pivot overlays share it so a marker and a reticle
/// cannot disagree about where a world point lands on screen.
fn project_to_screen(
    point: glam::Vec3,
    view_projection: glam::Mat4,
    viewport_px: [u32; 4],
    pixels_per_point: f32,
) -> Option<egui::Pos2> {
    let clip = view_projection * glam::Vec4::new(point.x, point.y, point.z, 1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let [vx, vy, vw, vh] = viewport_px.map(|component| component as f32);
    let px = vx + (clip.x / clip.w * 0.5 + 0.5) * vw;
    let py = vy + (1.0 - (clip.y / clip.w * 0.5 + 0.5)) * vh;
    Some(egui::Pos2::new(
        px / pixels_per_point,
        py / pixels_per_point,
    ))
}

/// Whether segment `a→b` touches `rect` at all — the crossing-marquee predicate: an endpoint
/// inside, or an intersection with any rect edge (touching counts).
fn segment_touches_rect(a: egui::Pos2, b: egui::Pos2, rect: egui::Rect) -> bool {
    if rect.contains(a) || rect.contains(b) {
        return true;
    }
    let corners = [
        rect.left_top(),
        rect.right_top(),
        rect.right_bottom(),
        rect.left_bottom(),
    ];
    (0..4).any(|edge| segments_intersect(a, b, corners[edge], corners[(edge + 1) % 4]))
}

/// Whether segments `a→b` and `c→d` intersect, inclusive of touching endpoints.
fn segments_intersect(a: egui::Pos2, b: egui::Pos2, c: egui::Pos2, d: egui::Pos2) -> bool {
    let orient = |p: egui::Pos2, q: egui::Pos2, r: egui::Pos2| {
        (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x)
    };
    let (o1, o2) = (orient(a, b, c), orient(a, b, d));
    let (o3, o4) = (orient(c, d, a), orient(c, d, b));
    if o1 == 0.0 && o2 == 0.0 && o3 == 0.0 && o4 == 0.0 {
        // Fully collinear: intersect iff the 1D shadows overlap on both axes.
        let overlaps =
            |lo_a: f32, hi_a: f32, lo_b: f32, hi_b: f32| lo_a.max(lo_b) <= hi_a.min(hi_b);
        return overlaps(a.x.min(b.x), a.x.max(b.x), c.x.min(d.x), c.x.max(d.x))
            && overlaps(a.y.min(b.y), a.y.max(b.y), c.y.min(d.y), c.y.max(d.y));
    }
    o1 * o2 <= 0.0 && o3 * o4 <= 0.0
}

/// The closest point on segment `a→b` to `p` (all in the same 2D space) — the foot of the
/// perpendicular, clamped to the segment ends. The add-point insert preview sits here.
fn closest_point_on_segment(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> egui::Pos2 {
    let ab = b - a;
    let length_squared = ab.length_sq();
    if length_squared <= f32::EPSILON {
        return a; // degenerate segment (coincident endpoints)
    }
    let t = ((p - a).dot(ab) / length_squared).clamp(0.0, 1.0);
    a + ab * t
}

/// The distance from `p` to segment `a→b` — the add-point segment hit-test's metric.
fn point_to_segment_distance(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    (p - closest_point_on_segment(p, a, b)).length()
}

/// Twice the signed area of the closed polygon through `boundary` — the shoelace, unhalved,
/// because the region hit-test only ever compares magnitudes.
fn polygon_double_area(boundary: &[egui::Pos2]) -> f32 {
    let Some(&last) = boundary.last() else {
        return 0.0;
    };
    let mut previous = last;
    let mut sum = 0.0;
    for &point in boundary {
        sum += previous.x * point.y - point.x * previous.y;
        previous = point;
    }
    sum
}

/// The picked entity as the SELECTION names it, so a constraint's in-progress picks light up
/// through the shipped sketch highlight path rather than a second one that could disagree.
fn selection_target(
    sketch: document::scene::NodeId,
    entity: ui::panel::SketchEntity,
) -> ui::panel::SelectionTarget {
    use document::sketch::SketchCurve;
    match entity {
        ui::panel::SketchEntity::Point(id) => {
            ui::panel::SelectionTarget::SketchPoint { sketch, entity: id }
        }
        ui::panel::SketchEntity::Curve(SketchCurve::Segment(id)) => {
            ui::panel::SelectionTarget::SketchSegment { sketch, entity: id }
        }
        ui::panel::SketchEntity::Curve(SketchCurve::Arc(id)) => {
            ui::panel::SelectionTarget::SketchArc { sketch, entity: id }
        }
        ui::panel::SketchEntity::Curve(SketchCurve::Circle(id)) => {
            ui::panel::SelectionTarget::SketchCircle { sketch, entity: id }
        }
        // No verb takes an aggregate — the pick slots refuse one — but the selection has a place
        // for it, so a pick that ever does reach here lights up rather than vanishing.
        ui::panel::SketchEntity::Curve(curve) => {
            ui::panel::SelectionTarget::SketchHigherCurve { sketch, curve }
        }
    }
}

/// The hit that names `curve`, for a caller that resolved the curve first and needs to light it.
///
/// A higher curve carries its own identity across, the same way [`sketch_curve_from_hit`] carries
/// it back — the two are inverses, and neither invents a kind the other cannot spell.
const fn sketch_edge_hit_from_curve(curve: document::sketch::SketchCurve) -> SketchEdgeHit {
    match curve {
        document::sketch::SketchCurve::Segment(id) => SketchEdgeHit::Segment(id),
        document::sketch::SketchCurve::Arc(id) => SketchEdgeHit::Arc(id),
        document::sketch::SketchCurve::Circle(id) => SketchEdgeHit::Circle(id),
        document::sketch::SketchCurve::Bezier(_)
        | document::sketch::SketchCurve::Ellipse(_)
        | document::sketch::SketchCurve::Conic(_)
        | document::sketch::SketchCurve::Spline(_) => SketchEdgeHit::HigherCurve(curve),
    }
}

const fn sketch_curve_from_hit(hit: SketchEdgeHit) -> document::sketch::SketchCurve {
    match hit {
        SketchEdgeHit::Segment(id) => document::sketch::SketchCurve::Segment(id),
        SketchEdgeHit::Arc(id) => document::sketch::SketchCurve::Arc(id),
        SketchEdgeHit::Circle(id) => document::sketch::SketchCurve::Circle(id),
        // The aggregate already IS a `SketchCurve`; the hit carried the author's identity.
        SketchEdgeHit::HigherCurve(curve) => curve,
    }
}

/// Build the one undoable edit emitted for a completed sketch command. A Tangent reaches this
/// same intent door as every other profile edit; anchor compensation remains inseparable from
/// the authored change.
pub(super) fn sketch_profile_edit_transaction(
    target: document::scene::NodeId,
    producer: document::sketch::SketchSolid,
    old_offset: [i64; 3],
    new_offset: [i64; 3],
) -> Vec<crate::Intent> {
    let mut transaction = vec![crate::Intent::SetSketch { target, producer }];
    if new_offset != old_offset {
        transaction.push(crate::Intent::SetOffset {
            target,
            offset_measurements: [
                parametric::units::Measurement::from_voxels(new_offset[0]),
                parametric::units::Measurement::from_voxels(new_offset[1]),
                parametric::units::Measurement::from_voxels(new_offset[2]),
            ],
        });
    }
    transaction
}

/// A failed completion gives its temporary geometry picks back but keeps the same command armed,
/// so the next click begins a fresh question instead of colliding with a full stale slot list.
fn reset_failed_sketch_constraint_completion(
    armed: &mut Option<ui::panel::ArmedConstraint>,
    selection: &mut ui::panel::Selection,
    verb: ui::panel::ConstraintVerb,
) {
    *armed = Some(ui::panel::ArmedConstraint::new(verb));
    selection.clear_sketch_entities();
}

fn reset_refused_sketch_constraint_completion(
    armed: &mut Option<ui::panel::ArmedConstraint>,
    selection: &mut ui::panel::Selection,
    verb: ui::panel::ConstraintVerb,
    refusal: &document::sketch::ConstraintRefusal,
) -> &'static str {
    reset_failed_sketch_constraint_completion(armed, selection, verb);
    refusal_text(refusal)
}

fn select_sketch_constraint_refusal_culprits(
    selection: &mut ui::panel::Selection,
    sketch: document::scene::NodeId,
    refusal: &document::sketch::ConstraintRefusal,
) {
    for entity in refusal.culprits() {
        selection.toggle(ui::panel::SelectionTarget::SketchConstraint { sketch, entity });
    }
}

/// Derive a Tangent's one finite contact and carry it through the exact profile→render→screen
/// path used by the overlay. `None` deliberately covers missing evaluation/render context as
/// well as an invalid current branch: without all three, there is no honest badge locus.
fn tangent_badge_anchor<F>(
    sketch: &document::sketch::Sketch,
    first: document::sketch::SketchCurve,
    second: document::sketch::SketchCurve,
    branch: document::sketch::TangentBranch,
    context: Option<parametric::EvaluationContext>,
    profile_to_render: Option<F>,
    projection: (glam::Mat4, [u32; 4], f32),
) -> Option<egui::Pos2>
where
    F: FnOnce([f64; 2]) -> [f32; 3],
{
    let contact = sketch
        .tangent_contact(first, second, branch, context?)
        .ok()?;
    let render = profile_to_render?(contact.at);
    let (view_projection, viewport_px, pixels_per_point) = projection;
    project_to_screen(
        glam::Vec3::from_array(render),
        view_projection,
        viewport_px,
        pixels_per_point,
    )
}

/// Project a Concentric relation's one shared center. Malformed or unsatisfied pairs
/// have no honest common locus and therefore no badge.
fn concentric_badge_anchor<F>(
    sketch: &document::sketch::Sketch,
    first: document::sketch::SketchCurve,
    second: document::sketch::SketchCurve,
    profile_to_render: Option<F>,
    projection: (glam::Mat4, [u32; 4], f32),
) -> Option<egui::Pos2>
where
    F: FnOnce([f64; 2]) -> [f32; 3],
{
    let center = sketch.concentric_center(first, second)?;
    let render = profile_to_render?(center);
    let (view_projection, viewport_px, pixels_per_point) = projection;
    project_to_screen(
        glam::Vec3::from_array(render),
        view_projection,
        viewport_px,
        pixels_per_point,
    )
}

/// Project the document-owned witness for one satisfied Symmetry relation.
#[allow(clippy::too_many_arguments)]
fn symmetry_badge_anchor<F>(
    sketch: &document::sketch::Sketch,
    first: document::sketch::SketchCurve,
    second: document::sketch::SketchCurve,
    axis: document::sketch::EntityId,
    branch: document::sketch::SymmetryBranch,
    context: Option<parametric::EvaluationContext>,
    profile_to_render: Option<F>,
    projection: (glam::Mat4, [u32; 4], f32),
) -> Option<egui::Pos2>
where
    F: FnOnce([f64; 2]) -> [f32; 3],
{
    let locus = sketch
        .symmetry_badge_locus(first, second, axis, branch, context?)
        .ok()?;
    let render = profile_to_render?(locus);
    let (view_projection, viewport_px, pixels_per_point) = projection;
    project_to_screen(
        glam::Vec3::from_array(render),
        view_projection,
        viewport_px,
        pixels_per_point,
    )
}

/// How far a dimension stands off the geometry it measures, in egui points.
///
/// One number for all three members, so a span's dimension line, a radius leader's elbow and an
/// angle's arc all sit the same distance out and read as one drawing rather than three.
const DIMENSION_STANDOFF_PX: f32 = 26.0;

/// How nearly parallel a dimension's two projected directions may come before it declines to draw,
/// as the sine of the angle between them.
///
/// A linear dimension is built from two plane directions — the one it measures along and the one
/// its extension lines stand on. Seen edge-on, every direction in the plane projects onto one
/// screen line and those two collapse together: the meeting point of the dimension line and an
/// extension line stops being defined, and any answer is the camera's rather than the drawing's.
/// Declining there is the drawing's own answer, the same one a corner too square to read gives.
const A_PLANE_TOO_EDGE_ON_TO_DIMENSION: f32 = 0.1;

/// Below this sine between one plane direction struck at two places, the projection is flat enough
/// that the plane's parallels stay parallel and there is no vanishing point to aim at.
///
/// It is a floor on the divide rather than a judgement: at this angle the meeting point already
/// stands ten thousand pixels off, so the direction to it and the direction at the feature agree to
/// far better than a pixel and the cheaper of the two is the right answer.
const A_PROJECTION_TOO_FLAT_TO_CONVERGE: f32 = 1.0e-4;

/// Which way a step IN THE PLANE runs on screen, taken AT A POINT.
///
/// Plane to screen is a homography, so the screen perpendicular of a projected direction is the
/// image of some OTHER plane direction — at a three-quarter view, one 31 degrees away. Every
/// direction a dimension is built from has to come through here or the drawing reads as leaning out
/// of the sketch it annotates.
///
/// Taken at a point because the map DIVIDES: one plane direction runs a different way on screen at
/// every place, by 13 degrees across a strongly perspective view.
///
/// The step is taken at unit plane length however long the one handed in is, and the reason is the
/// CULL rather than accuracy — a plane line images as a straight line, so both samples lie on it and
/// the chord IS the direction at any step length. A step as long as the whole span can land behind
/// the camera and answer nothing where the short one beside it answers fine.
/// The sketch plane's whole projection, as the one three-by-three that states it exactly.
///
/// A plane coordinate reaches the world by an affine and the world reaches clip space by a matrix,
/// so a plane coordinate reaches CLIP linearly — three strikes are the entire map. What makes the
/// composition a homography rather than an affine is the last step, the divide into pixels, and a
/// homography is what a dimension has to reason in: it carries lines to lines but neither angles
/// nor lengths, so the plane's own right angle and the plane's own unit both have to be asked for
/// at the place they are wanted.
///
/// The two strikes are taken a long way apart and divided back down. A sketch can sit far from the
/// render origin, and a one-unit difference between two `f32` clip values out there keeps only a
/// few significant digits; sixty-four units keeps six more.
///
/// The viewport's own affine is folded into the first two rows, so the matrix answers in EGUI
/// POINTS and needs nothing downstream. The last row is left as the projection's `w` untouched,
/// because its SIGN is the in-front-of-the-camera test every caller of
/// [`PlaneFrame::at`](ui::gizmos::dimension::PlaneFrame::at) relies on.
fn a_sketch_planes_frame<F>(
    clip_of: &F,
    viewport: [f32; 4],
    pixels_per_point: f32,
) -> Option<ui::gizmos::dimension::PlaneFrame>
where
    F: Fn([f64; 2]) -> glam::Vec4,
{
    const STRIDE: f32 = 64.0;
    let origin = clip_of([0.0, 0.0]);
    let across = (clip_of([f64::from(STRIDE), 0.0]) - origin) / STRIDE;
    let down = (clip_of([0.0, f64::from(STRIDE)]) - origin) / STRIDE;
    let row = |read: fn(glam::Vec4) -> f32| {
        [
            f64::from(read(across)),
            f64::from(read(down)),
            f64::from(read(origin)),
        ]
    };
    let (to_x, to_y, to_w) = (row(|clip| clip.x), row(|clip| clip.y), row(|clip| clip.w));

    let [left, top, wide, tall] = viewport.map(f64::from);
    let ratio = f64::from(pixels_per_point);
    // `x = (left + (X/W * 0.5 + 0.5) * wide) / ratio`, cleared of the divide; `y` the same with the
    // vertical flip, which is the one minus sign.
    let mix = |gain: f64, axis: [f64; 3], bias: f64| {
        [0, 1, 2].map(|index| gain.mul_add(axis[index], bias * to_w[index]))
    };
    ui::gizmos::dimension::PlaneFrame::from_plane_to_screen([
        mix(0.5 * wide / ratio, to_x, (left + 0.5 * wide) / ratio),
        mix(-0.5 * tall / ratio, to_y, (top + 0.5 * tall) / ratio),
        to_w,
    ])
}

fn a_plane_direction_on_screen<F>(at: [f64; 2], step: [f64; 2], to_px: &F) -> Option<egui::Vec2>
where
    F: Fn([f64; 2]) -> Option<egui::Pos2>,
{
    let span = step[0].hypot(step[1]);
    if span <= f64::EPSILON {
        return None;
    }
    let (Some(here), Some(there)) = (
        to_px(at),
        to_px([at[0] + step[0] / span, at[1] + step[1] / span]),
    ) else {
        return None;
    };
    let reach = there - here;
    (reach.length() > f32::EPSILON).then(|| reach / reach.length())
}

/// Which way a DIMENSION LINE runs on screen: the plane line in `toward`'s direction that passes
/// through where the author dropped the annotation.
///
/// Not the same question as [`a_plane_direction_on_screen`] at the feature, and the difference is
/// not small — 3.5 degrees on a strongly perspective view, which reads as a dimension measuring
/// some direction other than the one it names. Parallel plane lines CONVERGE on screen, so carrying
/// the run's direction over to the anchor lands on a line that is the image of nothing. Every line
/// of the family meets at one vanishing point instead, found by striking the direction at two
/// places the plane keeps a unit apart; aiming the anchor at that point is the whole construction.
///
/// The anchor is asked for as a SCREEN point on purpose. Every screen point has exactly one line of
/// the family through it whether or not it stands on the plane at all — which an unplaced drawing's
/// invented anchor, stepped off by a pixel standoff, does not.
///
/// Without a divide the two strikes stay parallel, no such point exists, and the direction at the
/// feature is already the exact answer.
fn a_dimension_lines_direction<F>(
    at: [f64; 2],
    toward: [f64; 2],
    anchor: egui::Pos2,
    to_px: &F,
) -> Option<egui::Vec2>
where
    F: Fn([f64; 2]) -> Option<egui::Pos2>,
{
    let here = a_plane_direction_on_screen(at, toward, to_px)?;
    let span = toward[0].hypot(toward[1]);
    let beside = [at[0] - toward[1] / span, at[1] + toward[0] / span];
    let (Some(from), Some(there), Some(also)) = (
        to_px(at),
        to_px(beside),
        a_plane_direction_on_screen(beside, toward, to_px),
    ) else {
        return Some(here);
    };
    let apart = here.x.mul_add(also.y, -(here.y * also.x));
    if apart.abs() < A_PROJECTION_TOO_FLAT_TO_CONVERGE {
        return Some(here);
    }
    let gap = there - from;
    let vanishing = from + here * (gap.x.mul_add(also.y, -(gap.y * also.x)) / apart);
    let reach = vanishing - anchor;
    // The sign is free: a dimension line is a line, and everything downstream re-derives its
    // direction from the two feet the line is met at.
    Some(if reach.length() > f32::EPSILON {
        reach / reach.length()
    } else {
        here
    })
}

/// Whether a dimension's projected directions have collapsed together at EITHER END — see
/// [`A_PLANE_TOO_EDGE_ON_TO_DIMENSION`].
///
/// Both ends, because a projection that divides answers one plane direction differently at each of
/// them — by up to 13 degrees on a strongly perspective view, which is more than twice this band.
/// A run receding toward the horizon can stand square at the tail and near-parallel at the head, and
/// the head's foot is then the meeting of two lines that barely cross: the drawing passes the guard
/// and puts its far extension somewhere the geometry never went.
///
/// A degenerate direction is NOT edge-on and does not decline here: a span whose two points the
/// solver has driven together still wants its number on screen so the author can see what to fix,
/// and the gizmo already draws that case finitely.
fn a_plane_too_edge_on_to_dimension(along: egui::Vec2, across: [egui::Vec2; 2]) -> bool {
    let along = along.normalized();
    if along.length() <= f32::EPSILON {
        return false;
    }
    across.into_iter().any(|across| {
        let across = across.normalized();
        across.length() > f32::EPSILON
            && across.x.mul_add(-along.y, along.x * across.y).abs()
                < A_PLANE_TOO_EDGE_ON_TO_DIMENSION
    })
}

/// Where a rim dimension's leader ends when the author never said — out and up from the center,
/// the direction a badge on a point already goes.
///
/// Only reached by a dimension authored before the annotation carried a place of its own. A placed
/// one uses what the author dropped, which is the whole point of storing it.
fn default_rim_anchor(center: egui::Pos2, radius_px: f32) -> egui::Pos2 {
    center + egui::vec2(0.707, -0.707) * (radius_px + DIMENSION_STANDOFF_PX)
}

/// How big an angle's arc is struck, given where the author pulled its text and how far the legs
/// themselves run.
///
/// **There is no ceiling.** An arc drawn past the ends of the legs is the ordinary case that
/// extension lines exist for, and a label that stopped following the cursor at the end of the
/// geometry would be the drawing refusing a placement the author is entitled to make — the same
/// freedom a span's offset and a rim's leader already have. The floor stays: an arc inside its own
/// arrowheads is not a smaller dimension, it is an unreadable one.
fn angle_arc_radius(vertex: egui::Pos2, placed: Option<egui::Pos2>, reach: f32) -> f32 {
    placed
        .map_or(reach * 0.55, |placed| placed.distance(vertex))
        .max(DIMENSION_STANDOFF_PX)
}

/// A number for a dimension label: no trailing zeros, and no decimal point when it is whole.
///
/// An angle authored as 30 should read `30`, not `30.00`. Two places is where a sketch angle stops
/// being a number the author recognises as the one they typed.
fn trim_number(value: f64) -> String {
    let text = format!("{value:.2}");
    match text.trim_end_matches('0').trim_end_matches('.') {
        // A value that rounds away to nothing IS zero, and nobody writes that as "-0".
        "" | "-" | "-0" => "0".to_string(),
        trimmed => trimmed.to_string(),
    }
}

/// The signed turn from `from` onto `to`, folded into `(-π, π]` — the short way round.
fn signed_turn(from: egui::Vec2, to: egui::Vec2) -> f32 {
    let cross = from.x * to.y - from.y * to.x;
    cross.atan2(from.dot(to))
}

/// Whether `at` lies in the corner swept from `first` onto `second` the short way round.
fn corner_holds(first: egui::Vec2, second: egui::Vec2, at: egui::Vec2) -> bool {
    let sweep = signed_turn(first, second);
    let toward = signed_turn(first, at);
    if sweep >= 0.0 {
        (0.0..=sweep).contains(&toward)
    } else {
        (sweep..=0.0).contains(&toward)
    }
}

/// Where a rim STANDS on screen, sampled once round its own circle.
///
/// A circle drawn in a sketch plane is not a circle on screen: `to_px` is a projection, so unless
/// the plane faces the camera the drawing is an ellipse. Everything a dimension puts ON that
/// drawing — an arrowhead, an extension carried round to a leader — asks this instead of stepping
/// out along a screen radius, which is right only in the one direction it was measured.
struct ProjectedRim {
    center: egui::Pos2,
    /// One whole turn, evenly spaced in the PLANE and projected point by point, so the ring is the
    /// same curve the overlay draws rather than an approximation of it.
    ring: Vec<egui::Pos2>,
}

impl ProjectedRim {
    /// A whole rim's circle, projected. `None` when the center or any of the ring leaves the view.
    fn project(
        plane_center: [f64; 2],
        radius: f64,
        to_px: &dyn Fn([f64; 2]) -> Option<egui::Pos2>,
    ) -> Option<Self> {
        const STEPS: usize = 72;
        let center = to_px(plane_center)?;
        let ring = (0..STEPS)
            .map(|step| {
                #[allow(clippy::cast_precision_loss)]
                let turn = std::f64::consts::TAU * step as f64 / STEPS as f64;
                to_px([
                    radius.mul_add(turn.cos(), plane_center[0]),
                    radius.mul_add(turn.sin(), plane_center[1]),
                ])
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self { center, ring })
    }

    /// The ring's own mean reach from its center, in screen points.
    ///
    /// The one nominal radius a layout reasons in — how much room an arc has for its arrows and its
    /// value. Not one sample along the plane's first axis: on the ellipse a tilted plane draws,
    /// that sample is right in one direction and wrong in every other, and this is a length weighed
    /// against text widths rather than a distance anything steps out by.
    fn mean_reach(&self) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let count = self.ring.len().max(1) as f32;
        self.ring
            .iter()
            .map(|at| self.center.distance(*at))
            .sum::<f32>()
            / count
    }

    /// Where the rim stands at a screen bearing, by striking the ray out of the center against the
    /// ring — no inverse of the projection needed, and exact at every point the ring was sampled.
    fn touch(&self, bearing: f32) -> egui::Pos2 {
        let out = egui::vec2(bearing.cos(), bearing.sin());
        let across = egui::vec2(out.y, -out.x);
        let steps = self.ring.len();
        for index in 0..steps {
            let (here, next) = (self.ring[index], self.ring[(index + 1) % steps]);
            let (side, other) = (
                (here - self.center).dot(across),
                (next - self.center).dot(across),
            );
            if (side <= 0.0) == (other <= 0.0) {
                continue;
            }
            let hit = here + (next - here) * (side / (side - other));
            // The ray's LINE meets the ring twice; the half the bearing points along is the one.
            if (hit - self.center).dot(out) >= 0.0 {
                return hit;
            }
        }
        // Unreachable while the center is inside its own ring, which a projected circle's is.
        self.center + out
    }
}

/// How much of its own circle `curve` actually draws, seen from `center` on screen — a whole turn
/// for a closed rim, which draws all of it.
///
/// The turn is read on SCREEN and its direction is found by projecting the curve's own midpoint,
/// not by carrying the plane's counter-clockwise sense across: a sketch seen from behind its plane
/// runs the other way round, and an extension drawn the long way round a rim it should have met in
/// a few degrees is the failure that would say so.
fn drawn_turn(
    sketch: &document::sketch::Sketch,
    curve: document::sketch::SketchCurve,
    center: egui::Pos2,
    to_px: &dyn Fn([f64; 2]) -> Option<egui::Pos2>,
) -> Option<(f32, f32)> {
    let document::sketch::SketchCurve::Arc(arc) = curve else {
        // A closed rim reaches every bearing, so it never falls short of a leader.
        return Some((0.0, std::f32::consts::TAU));
    };
    let form = sketch.arc_form_of(arc)?;
    // Halfway along the curve, found by turning the start point half the sweep about the center —
    // a point ON the drawing, so whichever way round the screen shows it is the way it is drawn.
    let (sine, cosine) = (form.sweep_degrees / 2.0).to_radians().sin_cos();
    let reach = [form.from[0] - form.center[0], form.from[1] - form.center[1]];
    let middle = [
        form.center[0] + reach[0] * cosine - reach[1] * sine,
        form.center[1] + reach[0] * sine + reach[1] * cosine,
    ];
    let bearing = |at: [f64; 2]| {
        let out = to_px(at)? - center;
        (out.length() > f32::EPSILON).then_some(out)
    };
    let (from, middle, to) = (bearing(form.from)?, bearing(middle)?, bearing(form.to)?);
    let direction = if signed_turn(from, middle) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let from = from.y.atan2(from.x);
    let round = ((to.y.atan2(to.x) - from) * direction).rem_euclid(std::f32::consts::TAU);
    Some((from, direction * round))
}

/// The two bearings an angular dimension is struck between, and how far each leg reaches THAT WAY.
///
/// **Two lines make four corners and the annotation's own place picks one of them.** Two of the
/// four are the turn between the arms and two are its supplement, so `corner` — which is stored,
/// because it is the claim — narrows the four to two, and the anchor chooses between the pair that
/// remains. Anything less than this and an author cannot dimension the corner they are pointing at.
///
/// Each leg is answered as the INTERVAL its own line occupies along the ray the arc is struck on,
/// separately from the other. Two lines that never touch cross at a point neither of them reaches,
/// so an arc struck near the vertex sits in a gap and the dogleg runs inward to meet it; a line the
/// corner points away from has an interval entirely behind the vertex and is carried forward
/// through it. Both are the same rule — the dogleg spans whatever the line does not.
///
/// Without an anchor — a dimension authored before annotations had a place — the arms are read the
/// way they were drawn, which is the pair that agrees with the number `corner` names.
fn angle_legs(
    vertex: egui::Pos2,
    to_px: &dyn Fn([f64; 2]) -> Option<egui::Pos2>,
    first: ([f64; 2], [f64; 2]),
    second: ([f64; 2], [f64; 2]),
    corner: document::sketch::AngleCorner,
    placed: Option<egui::Pos2>,
) -> Option<(f32, f32, [ui::gizmos::dimension::Leg; 2])> {
    let line = |(from, to): ([f64; 2], [f64; 2])| {
        let (from, to) = (to_px(from)?, to_px(to)?);
        let along = to - from;
        let length = along.length();
        (length > f32::EPSILON).then(|| (along / length, [from, to]))
    };
    let ((first_along, first_ends), (second_along, second_ends)) = (line(first)?, line(second)?);
    // The supplement is the same two lines with one arm read the other way, which is what makes it
    // a different number rather than a different picture of the same one.
    let facing = match corner {
        document::sketch::AngleCorner::Between => 1.0,
        document::sketch::AngleCorner::Supplementary => -1.0,
    };
    let canonical = (first_along, second_along * facing);
    let opposite = (-canonical.0, -canonical.1);
    // The two remaining corners are opposite each other, so at most one holds the anchor.
    let (first_ray, second_ray) = match placed {
        Some(placed) if corner_holds(opposite.0, opposite.1, placed - vertex) => opposite,
        _ => canonical,
    };

    // Where this line starts and stops along the ray its arc is struck on. Both ends negative
    // means the line lies entirely the other way, which the gizmo draws by carrying it forward
    // through the vertex rather than by a case of its own.
    let leg = |ends: [egui::Pos2; 2], ray: egui::Vec2| {
        let (one, other) = ((ends[0] - vertex).dot(ray), (ends[1] - vertex).dot(ray));
        ui::gizmos::dimension::Leg {
            nearest: one.min(other),
            furthest: one.max(other),
        }
    };
    let from = first_ray.y.atan2(first_ray.x);
    Some((
        from,
        from + signed_turn(first_ray, second_ray),
        [leg(first_ends, first_ray), leg(second_ends, second_ray)],
    ))
}

/// Return the topmost generic constraint badge under a physical-pixel cursor. The badge keeps
/// the constraint id beside its position, so every caller picks the authored relation directly.
fn sketch_constraint_badge_at(
    badges: &[ui::chrome::ConstraintBadge],
    cursor_px: egui::Pos2,
    pixels_per_point: f32,
) -> Option<document::sketch::EntityId> {
    let cursor = cursor_px / pixels_per_point;
    let half = ui::chrome::SKETCH_CONSTRAINT_BADGE * 0.5;
    // Last drawn wins: badges stack along one offset, and the later ones paint over the earlier,
    // so the pick must agree with what is on top.
    badges
        .iter()
        .rev()
        .find(|badge| {
            egui::Rect::from_center_size(badge.center, egui::Vec2::splat(half * 2.0))
                .contains(cursor)
        })
        .map(|badge| badge.constraint)
}

/// The constraint whose dimension VALUE is under a physical-pixel cursor.
///
/// The number is a dimension's only mark and therefore its only target. Later gizmos win a tie,
/// matching the badge rule: what was drawn last is what is on top.
fn sketch_dimension_value_at(
    gizmos: &[ui::chrome::DimensionGizmo],
    cursor: egui::Pos2,
    pixels_per_point: f32,
) -> Option<document::sketch::EntityId> {
    let cursor = egui::Pos2::new(cursor.x / pixels_per_point, cursor.y / pixels_per_point);
    gizmos
        .iter()
        .rev()
        .find(|gizmo| {
            gizmo
                .drawing
                .label_boxes()
                .into_iter()
                .any(|box_px| box_px.expand(3.0).contains(cursor))
        })
        .and_then(|gizmo| gizmo.constraint)
}

/// What the top bar says about a refused constraint. `offer` screens the clerical
/// refusals before the producer sees them, so in practice the author reads the third — but a
/// message per variant is what keeps that claim checkable rather than assumed.
fn refusal_text(why: &document::sketch::ConstraintRefusal) -> &'static str {
    use document::sketch::ConstraintRefusal;
    match why {
        ConstraintRefusal::UnknownEntity => "names geometry that is gone",
        ConstraintRefusal::CurvatureNeedsAJoint => {
            "curvature wants a spline's free end standing on that curve"
        }
        ConstraintRefusal::MirroredTangentArm => {
            "that end of the handle is a mirror — relate the other one"
        }
        ConstraintRefusal::Impossible => "no drawing can meet it",
        // Whether a culprit was isolated changes what the sentence can honestly promise: with one
        // selected, "this" points at a lit badge; without, the author is on their own and the
        // message must not imply otherwise.
        ConstraintRefusal::Unsatisfiable { fights } if fights.is_empty() => {
            "fights a constraint already set"
        }
        ConstraintRefusal::Unsatisfiable { .. } => "fights the selected constraint",
        ConstraintRefusal::WouldCollapse { implicated, .. } if implicated.is_empty() => {
            "would squeeze that shape to nothing"
        }
        ConstraintRefusal::WouldCollapse { .. } => {
            "would squeeze that shape to nothing — see the selected constraint"
        }
        ConstraintRefusal::AlreadyAsserted { .. } => "already asserted — it is selected",
        ConstraintRefusal::MissingEvaluationContext => {
            "needs the document density to resolve its fixed curve"
        }
        ConstraintRefusal::InvalidTangent { .. } => {
            "that tangent branch has no finite contact on both curves"
        }
        ConstraintRefusal::InvalidConcentric => "pick two distinct arcs or circles",
        ConstraintRefusal::InvalidSymmetry => {
            "pick two matching curves and a distinct nonzero line axis"
        }
    }
}

/// Whether `point` lies inside the closed polygon through `boundary` — the even-odd crossing
/// count. Screen space, so the f32 the projection already produced is the right width; the
/// f64 predicates in `substrate::geom2d` guard voxel-space classification, not a cursor test.
fn point_in_screen_polygon(boundary: &[egui::Pos2], point: egui::Pos2) -> bool {
    let mut inside = false;
    let count = boundary.len();
    for index in 0..count {
        let a = boundary[index];
        let b = boundary[(index + 1) % count];
        if (a.y > point.y) != (b.y > point.y) {
            let x = a.x + (point.y - a.y) / (b.y - a.y) * (b.x - a.x);
            if point.x < x {
                inside = !inside;
            }
        }
    }
    inside
}

/// A curved mark's ink from the one bit the caches carry: whether it is construction.
///
/// The third answer, [`TangentLever`](ui::chrome::SketchCurveInk::TangentLever), is never reached
/// through here — a lever is not an entity with a role, so it names its ink at its own push site.
fn curve_ink(construction: bool) -> ui::chrome::SketchCurveInk {
    if construction {
        ui::chrome::SketchCurveInk::Construction
    } else {
        ui::chrome::SketchCurveInk::Real
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::{
        a_dimension_lines_direction, a_plane_direction_on_screen, a_plane_too_edge_on_to_dimension,
        advance_circle_center_diameter_gesture, aggregate_marquee_picks, angle_arc_radius,
        angle_legs, apply_sketch_snap, circle_gesture_is_current, circle_marquee_hit, circle_ring,
        closest_point_on_segment, complete_circle_center_diameter, concentric_badge_anchor,
        marquee_box_px, nearest_sketch_edge_for_requirement, nearest_sketch_edge_from_candidates,
        nearest_tangent_lever, point_in_screen_polygon, point_to_segment_distance,
        pointer_left_the_press, polygon_double_area, reset_failed_sketch_constraint_completion,
        reset_refused_sketch_constraint_completion, segment_touches_rect, segments_intersect,
        select_sketch_constraint_refusal_culprits, sketch_constraint_badge_at,
        sketch_curve_from_hit, sketch_profile_edit_transaction, symmetry_badge_anchor,
        tangent_badge_anchor, trim_number, SketchEdgeHit, DIMENSION_STANDOFF_PX,
    };
    use document::sketch::{
        ConstraintKind, LineSide, PlaneAxis, Sketch, SketchCurve, SketchLength, SketchPoint,
        SketchSolid, SymmetryBranch, TangentBranch,
    };
    use egui::{pos2, Rect};
    use std::num::NonZeroU32;
    use ui::panel::PositionSnap;

    /// A sketch plane seen from three-quarters, spelled as the bare homography `to_px` composes to.
    ///
    /// `depth` is the divide. Zero is orthographic and the plane's parallels stay parallel on
    /// screen; the value used below puts the far end of a forty-unit drawing at about two thirds
    /// the scale of the near one, which is a strong perspective rather than a token one.
    #[allow(clippy::cast_possible_truncation)]
    fn a_tilted_camera(elevation: f64, depth: f64) -> impl Fn([f64; 2]) -> Option<egui::Pos2> {
        move |at| {
            let (turn, up) = (std::f64::consts::FRAC_PI_4.sin_cos(), elevation.sin());
            let across = at[1].mul_add(turn.1, -(at[0] * turn.0));
            let into = at[0].mul_add(turn.1, at[1] * turn.0);
            let divide = depth.mul_add(into, 1.0);
            // What stands behind the eye is not on screen, and `to_px` answers nothing for it. A
            // fixture without this cull can be handed cameras the shell would never reach, which
            // makes any failure it finds unfalsifiable.
            (divide > 0.0).then(|| {
                egui::pos2(
                    3.0f64.mul_add(across / divide, 300.0) as f32,
                    (3.0 * up).mul_add(into / divide, 200.0) as f32,
                )
            })
        }
    }

    /// How far apart two screen directions stand, in degrees, whichever way round each points.
    fn apart_in_degrees(one: egui::Vec2, other: egui::Vec2) -> f32 {
        one.x
            .mul_add(other.y, -(one.y * other.x))
            .abs()
            .min(1.0)
            .asin()
            .to_degrees()
    }

    /// **A dimension line runs the way the plane does WHERE THE ANNOTATION WAS DROPPED.**
    ///
    /// Under a projection that divides, the plane's parallels converge on screen — so the run's
    /// direction at the feature and its direction at the anchor are two different screen
    /// directions. Carrying the first one over and drawing through the anchor lands on a line that
    /// is the image of no plane line at all, and the dimension reads as measuring some direction
    /// other than the one it names. Every line of the family passes through one vanishing point
    /// instead, which is what [`a_dimension_lines_direction`] aims at.
    ///
    /// The camera is a bare homography because that is exactly what `to_px` composes to, and a
    /// fixture without a divide is structurally unable to go red here — so the orthographic reading
    /// is checked too, for the opposite claim: no vanishing point exists and the direction at the
    /// feature is already the answer.
    ///
    /// The construction is judged against the DEFINITION rather than against itself: the anchor
    /// used here is the image of a known plane point, so the way the plane line runs there can be
    /// said outright — which is the thing the vanishing point has to reproduce without being told
    /// the anchor stands on the plane at all.
    #[test]
    fn a_dimension_line_runs_the_way_the_plane_does_where_it_was_dropped() {
        let (run, feature, dropped) = ([1.0, 0.0], [0.0, 0.0], [20.0, 12.0]);

        let flat = a_tilted_camera(std::f64::consts::FRAC_PI_6, 0.0);
        let anchor = flat(dropped).expect("the anchor projects");
        let at_the_feature =
            a_plane_direction_on_screen(feature, run, &flat).expect("the run projects");
        let struck = a_dimension_lines_direction(feature, run, anchor, &flat)
            .expect("a flat projection still answers");
        assert!(
            apart_in_degrees(struck, at_the_feature) < 0.01,
            "with nothing to converge on, the answer is the direction at the feature"
        );

        let deep = a_tilted_camera(std::f64::consts::FRAC_PI_6, 0.008);
        let anchor = deep(dropped).expect("the anchor projects");
        let at_the_feature =
            a_plane_direction_on_screen(feature, run, &deep).expect("the run projects");
        let at_the_anchor =
            a_plane_direction_on_screen(dropped, run, &deep).expect("the run projects there too");
        let apart = apart_in_degrees(at_the_feature, at_the_anchor);
        assert!(
            apart > 1.0,
            "this camera barely divides ({apart} degrees), so nothing below would mean anything"
        );

        let struck = a_dimension_lines_direction(feature, run, anchor, &deep)
            .expect("a dividing projection has a vanishing point");
        let off = apart_in_degrees(struck, at_the_anchor);
        assert!(
            off < 0.01,
            "the dimension line runs {off} degrees off the plane line through the anchor"
        );
    }

    /// **A dimension declines when EITHER of its ends has gone edge-on, not just the first.**
    ///
    /// The guard exists because two projected directions that have collapsed together have no
    /// well-conditioned meeting point, and a foot struck between them lands wherever the arithmetic
    /// takes it. Under a projection that divides, the two ends of one run answer the plane's square
    /// differently — by more than twice this guard's whole band — so a run receding toward the
    /// horizon can stand square at the tail and near-parallel at the head. Reading the tail alone
    /// passes that drawing and puts its far extension line somewhere the geometry never went.
    ///
    /// The camera here is a real one in the sense that matters: every point it is asked about stands
    /// in front of the eye, so it is a view the shell can actually be in rather than an arithmetic
    /// curiosity behind the near plane.
    #[test]
    fn a_dimension_declines_when_either_end_has_gone_edge_on() {
        let (tail, head, square) = ([0.0, 0.0], [40.0, 0.0], [0.0, 40.0]);
        // Steep, and diving hard enough that the head sits just short of the horizon.
        let camera = a_tilted_camera(std::f64::consts::FRAC_PI_3, -0.034);
        let (from, to) = (
            camera(tail).expect("the tail stands in front of the eye"),
            camera(head).expect("and so does the head, or this is not a view the shell can be in"),
        );
        let at_tail = a_plane_direction_on_screen(tail, square, &camera).expect("the square shows");
        let at_head = a_plane_direction_on_screen(head, square, &camera).expect("and at the head");

        let run = (to - from).normalized();
        let (standing, collapsed) = (
            apart_in_degrees(run, at_tail),
            apart_in_degrees(run, at_head),
        );
        assert!(
            standing > 30.0 && collapsed < 5.0,
            "this view has to stand square at one end and collapse at the other to say anything: \
             {standing} degrees at the tail, {collapsed} at the head"
        );

        assert!(
            !a_plane_too_edge_on_to_dimension(run, [at_tail, at_tail]),
            "read at the tail alone, this drawing looks perfectly square"
        );
        assert!(
            a_plane_too_edge_on_to_dimension(run, [at_tail, at_head]),
            "the head has collapsed, so the drawing has no far foot worth striking"
        );
    }

    /// **A marquee cannot select what the viewport does not show.**
    ///
    /// The band is drawn clipped to the viewport like every other sketch mark, and this file's own
    /// law is that what is clickable is exactly what is drawn. A box the author drags out past the
    /// side panel would otherwise sweep up the geometry hidden under it — a selection with no
    /// visible cause, which is the worst kind. One clamped rectangle is minted here and read by
    /// both the band and the release.
    ///
    /// **Seen red**: with the `.intersect(viewport)` dropped, the box came back
    /// `[[100 100] - [900 700]]` and swept the point at `(850, 400)` that sits under the panel.
    #[test]
    fn a_marquee_box_stops_at_the_edge_of_the_viewport() {
        // A 1200×800 window with a 400-point side panel on the right and a 60-point top bar.
        let viewport = [0_u32, 60, 800, 740];
        let hidden = pos2(850.0, 400.0);
        let shown = pos2(400.0, 400.0);

        let dragged_past_the_panel = marquee_box_px(viewport, (100.0, 100.0), (900.0, 700.0));
        assert!(
            !dragged_past_the_panel.contains(hidden),
            "the box reached under the side panel: {dragged_past_the_panel:?}"
        );
        assert!(
            dragged_past_the_panel.contains(shown),
            "the box lost what the author could see: {dragged_past_the_panel:?}"
        );

        // Above the viewport's top edge, which the top bar occupies, clamps the same way.
        let dragged_into_the_top_bar = marquee_box_px(viewport, (100.0, 10.0), (300.0, 300.0));
        assert_eq!(dragged_into_the_top_bar.min.y, 60.0);

        // A box wholly inside is untouched — the clamp must not cost the ordinary gesture
        // anything.
        let ordinary = marquee_box_px(viewport, (100.0, 100.0), (300.0, 300.0));
        assert_eq!(
            ordinary,
            Rect::from_min_max(pos2(100.0, 100.0), pos2(300.0, 300.0))
        );
    }

    fn tangent_ready_sketch() -> (
        Sketch,
        document::sketch::EntityId,
        document::sketch::EntityId,
    ) {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let from = sketch.add_free_point(SketchPoint::new(0, 0));
        let to = sketch.add_free_point(SketchPoint::new(10, 0));
        let segment = sketch.connect(from, to).expect("segment");
        let circle = sketch
            .add_circle(SketchPoint::new(5, 4), SketchLength::new(4))
            .expect("circle");
        (sketch, segment, circle)
    }

    fn context() -> parametric::EvaluationContext {
        parametric::EvaluationContext::new(NonZeroU32::new(16).expect("non-zero density"))
    }

    #[test]
    fn the_snap_policy_quantizes_a_profile_coordinate() {
        // Voxel rounds to the plane's own grid; Block to multiples of the density; NoSnap
        // carries the exact position as integer + fraction (#96/#101).
        assert_eq!(
            apply_sketch_snap([2.6, -1.4], PositionSnap::Voxel, 8).in_plane(),
            [3.0, -1.0]
        );
        assert_eq!(
            apply_sketch_snap([11.0, -5.0], PositionSnap::Block, 8).in_plane(),
            [8.0, -8.0]
        );
        let free = apply_sketch_snap([2.6, -1.4], PositionSnap::NoSnap, 8).in_plane();
        assert!(
            (free[0] - 2.6).abs() < 1e-6 && (free[1] + 1.4).abs() < 1e-6,
            "NoSnap carries the position to f32-fraction precision, got {free:?}"
        );
    }

    #[test]
    fn foot_falls_inside_the_span_for_a_perpendicular_drop() {
        // A cursor above the middle of a horizontal edge projects to that midpoint, so the
        // insert preview and the hit distance are the perpendicular offset.
        let a = pos2(0.0, 0.0);
        let b = pos2(10.0, 0.0);
        let foot = closest_point_on_segment(pos2(4.0, 3.0), a, b);
        assert!(
            (foot.x - 4.0).abs() < 1e-4 && foot.y.abs() < 1e-4,
            "foot at (4, 0), got {foot:?}"
        );
        assert!((point_to_segment_distance(pos2(4.0, 3.0), a, b) - 3.0).abs() < 1e-4);
    }

    /// **An annotation follows the cursor however far out it goes.** The angle's arc used to be
    /// capped at the length of the legs, so past their ends the label stopped moving and the
    /// gesture read as jammed. The gizmo already draws extension lines out to an arc beyond the
    /// geometry — the cap was the only thing preventing one.
    #[test]
    fn an_angles_arc_follows_the_cursor_past_the_ends_of_its_legs() {
        let vertex = pos2(100.0, 100.0);
        let reach = 60.0_f32;
        for pulled in [70.0_f32, 200.0, 5000.0] {
            let placed = pos2(vertex.x + pulled, vertex.y);
            assert!(
                (angle_arc_radius(vertex, Some(placed), reach) - pulled).abs() < 1e-4,
                "pulled {pulled} past legs of {reach} and the arc did not follow"
            );
        }
        // The floor stands: an arc inside its own arrowheads reads as noise, not as a dimension.
        assert!(
            (angle_arc_radius(vertex, Some(pos2(vertex.x + 2.0, vertex.y)), reach)
                - DIMENSION_STANDOFF_PX)
                .abs()
                < 1e-4
        );
        // And a dimension authored before annotations had a place still opens inside its legs.
        assert!((angle_arc_radius(vertex, None, reach) - 33.0).abs() < 1e-4);
    }

    #[test]
    fn foot_clamps_to_the_nearer_end_past_the_segment() {
        // A cursor beyond an endpoint clamps to that endpoint — the distance is to the vertex,
        // NOT to the infinite line, so a click off the end of an edge does not falsely hit it.
        let a = pos2(0.0, 0.0);
        let b = pos2(10.0, 0.0);
        assert_eq!(
            closest_point_on_segment(pos2(-5.0, 0.0), a, b),
            a,
            "clamps to the start"
        );
        assert!(
            (point_to_segment_distance(pos2(15.0, 0.0), a, b) - 5.0).abs() < 1e-4,
            "distance is to the end vertex (5), not 0 on the extended line"
        );
    }

    #[test]
    fn a_degenerate_segment_reduces_to_its_endpoint() {
        // Coincident endpoints (a culled/collapsed edge) must not divide by zero.
        let a = pos2(3.0, 3.0);
        assert_eq!(closest_point_on_segment(pos2(9.0, 9.0), a, a), a);
    }

    #[test]
    fn a_segment_through_the_box_touches_without_an_endpoint_inside() {
        // The crossing-only case the spec calls out: both endpoints outside, the run passes
        // through — crossing selects it, window (≥1 endpoint inside) would not.
        let rect = Rect::from_min_max(pos2(10.0, 10.0), pos2(20.0, 20.0));
        assert!(segment_touches_rect(
            pos2(0.0, 15.0),
            pos2(30.0, 15.0),
            rect
        ));
        assert!(
            !(rect.contains(pos2(0.0, 15.0)) || rect.contains(pos2(30.0, 15.0))),
            "the window predicate misses it"
        );
    }

    #[test]
    fn a_circle_ring_is_closed_and_answers_marquee_edges() {
        let ring = circle_ring([0.0, 0.0], 5.0, 1.0 / 16.0);
        assert!(ring.len() > 4, "the ring carries curve samples");
        assert_eq!(
            ring.first(),
            ring.last(),
            "the final chord closes the circle"
        );

        let projected: Vec<_> = ring
            .iter()
            .map(|point| pos2(point[0] as f32, point[1] as f32))
            .collect();
        let rim = Rect::from_min_max(pos2(4.5, -1.0), pos2(6.0, 1.0));
        assert!(
            circle_marquee_hit(&projected, rim, false),
            "a crossing marquee reaches the actual rim"
        );
        let interior = Rect::from_min_max(pos2(-1.0, -1.0), pos2(1.0, 1.0));
        assert!(
            !circle_marquee_hit(&projected, interior, false),
            "the empty center is not a circle edge"
        );
        let containing = Rect::from_min_max(pos2(-6.0, -6.0), pos2(6.0, 6.0));
        assert!(circle_marquee_hit(&projected, containing, true));
        assert!(!circle_marquee_hit(&projected, rim, true));
    }

    #[test]
    fn circle_gesture_holds_one_transient_center_then_clears_on_completion() {
        let mut pending = None;
        let center = SketchPoint::new(2, 3);
        let perimeter = SketchPoint::new(7, 3);

        assert_eq!(
            advance_circle_center_diameter_gesture(&mut pending, center),
            None,
            "the first click writes no document geometry"
        );
        assert_eq!(pending, Some(center));
        assert_eq!(
            advance_circle_center_diameter_gesture(&mut pending, perimeter),
            Some((center, perimeter))
        );
        assert_eq!(pending, None, "a completion clears transient state");
    }

    #[test]
    fn a_zero_or_duplicate_circle_completion_skips_history() {
        let empty = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
        let center = SketchPoint::new(2, 3);
        let perimeter = SketchPoint::new(7, 3);
        let circle = complete_circle_center_diameter(&empty, center, perimeter);
        assert!(circle.is_some(), "a nonzero circle produces one edit");
        let Some(circle) = circle else {
            return;
        };
        assert!(
            complete_circle_center_diameter(&circle, center, center).is_none(),
            "a zero-radius completion emits no transaction"
        );
        assert!(
            complete_circle_center_diameter(&circle, center, perimeter).is_none(),
            "an identical circle emits no transaction"
        );
    }

    #[test]
    fn pending_circle_center_dies_when_its_tool_or_sketch_changes() {
        let first = document::scene::NodeId(7);
        let second = document::scene::NodeId(8);
        assert!(circle_gesture_is_current(
            ui::panel::SketchTool::CircleCenterDiameter,
            first,
            Some(first)
        ));
        assert!(!circle_gesture_is_current(
            ui::panel::SketchTool::Select,
            first,
            Some(first)
        ));
        assert!(!circle_gesture_is_current(
            ui::panel::SketchTool::CircleCenterDiameter,
            second,
            Some(first)
        ));
    }

    #[test]
    fn circle_hit_joins_the_existing_nearest_edge_priority() {
        assert_eq!(
            nearest_sketch_edge_from_candidates([
                Some((SketchEdgeHit::Segment(1), 4.0)),
                Some((SketchEdgeHit::Arc(2), 3.0)),
                Some((SketchEdgeHit::Circle(3), 2.0)),
            ]),
            Some(SketchEdgeHit::Circle(3))
        );
        assert_eq!(
            nearest_sketch_edge_from_candidates([
                Some((SketchEdgeHit::Segment(1), 2.0)),
                Some((SketchEdgeHit::Arc(2), 2.0)),
                Some((SketchEdgeHit::Circle(3), 2.0)),
            ]),
            Some(SketchEdgeHit::Segment(1)),
            "the established segment tie break remains intact"
        );
        assert_eq!(
            nearest_sketch_edge_from_candidates([
                Some((SketchEdgeHit::Segment(1), 3.0)),
                Some((SketchEdgeHit::Arc(2), 2.0)),
                Some((SketchEdgeHit::Circle(3), 4.0)),
            ]),
            Some(SketchEdgeHit::Arc(2)),
            "an arc remains a real curve hit between segment and circle cases"
        );
    }

    #[test]
    fn typed_edge_slots_ignore_closer_overlapping_wrong_kinds() {
        for (requirement, segment, arc, circle, expected) in [
            (
                ui::panel::PickRequirement::Segment,
                Some((SketchEdgeHit::Segment(1), 3.0)),
                Some((SketchEdgeHit::Arc(2), 1.0)),
                Some((SketchEdgeHit::Circle(3), 2.0)),
                SketchEdgeHit::Segment(1),
            ),
            (
                ui::panel::PickRequirement::MatchingCurve(document::sketch::SketchCurve::Arc(9)),
                Some((SketchEdgeHit::Segment(1), 1.0)),
                Some((SketchEdgeHit::Arc(2), 3.0)),
                Some((SketchEdgeHit::Circle(3), 2.0)),
                SketchEdgeHit::Arc(2),
            ),
            (
                ui::panel::PickRequirement::MatchingCurve(document::sketch::SketchCurve::Circle(9)),
                Some((SketchEdgeHit::Segment(1), 1.0)),
                Some((SketchEdgeHit::Arc(2), 2.0)),
                Some((SketchEdgeHit::Circle(3), 3.0)),
                SketchEdgeHit::Circle(3),
            ),
        ] {
            assert_eq!(
                nearest_sketch_edge_for_requirement(requirement, segment, arc, circle),
                Some(expected),
                "wrong-kind distance never hides the requested variant"
            );
        }
    }

    #[test]
    fn restored_dead_symmetry_pick_restarts_before_requirement_dispatch() {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let from = sketch.add_free_point(SketchPoint::new(0, 0));
        let to = sketch.add_free_point(SketchPoint::new(4, 0));
        let segment = sketch.connect(from, to).expect("segment");
        let mut armed = ui::panel::ArmedConstraint::from_parts(
            ui::panel::ConstraintVerb::Symmetry,
            vec![ui::panel::SketchEntity::Curve(
                document::sketch::SketchCurve::Segment(segment),
            )],
        );
        assert_eq!(
            armed.wants(),
            Some(ui::panel::PickRequirement::MatchingCurve(
                document::sketch::SketchCurve::Segment(segment)
            ))
        );
        sketch.delete_segment(segment);
        assert!(armed.restart_if_invalid(&sketch));
        assert!(armed.picked().is_empty());
        assert_eq!(armed.wants(), Some(ui::panel::PickRequirement::Curve));
    }

    /// Every edge hit keeps the author's own curve identity, the aggregate included. The shell has
    /// exactly one spelling of a curve and does not decide which kinds a constraint may name —
    /// that is the slot's answer, given ahead of the click.
    #[test]
    fn a_curve_hit_keeps_the_authors_identity_for_every_kind() {
        for (hit, curve) in [
            (SketchEdgeHit::Segment(11), SketchCurve::Segment(11)),
            (SketchEdgeHit::Arc(12), SketchCurve::Arc(12)),
            (SketchEdgeHit::Circle(13), SketchCurve::Circle(13)),
            (
                SketchEdgeHit::HigherCurve(SketchCurve::Ellipse(14)),
                SketchCurve::Ellipse(14),
            ),
        ] {
            assert_eq!(sketch_curve_from_hit(hit), curve);
        }
    }

    /// A lever answers as the ONE fit point it belongs to, so a cursor on one handle cannot mean
    /// the spline — which would light every other handle the spline carries along with the curve.
    #[test]
    fn a_tangent_lever_answers_for_its_own_fit_point_alone() {
        // Two levers on one spline, far apart: fit point 5 near the origin, fit point 9 off right.
        let levers = vec![
            (
                5,
                vec![
                    egui::pos2(0.0, 10.0),
                    egui::pos2(0.0, 0.0),
                    egui::pos2(0.0, -10.0),
                ],
            ),
            (
                9,
                vec![
                    egui::pos2(200.0, 10.0),
                    egui::pos2(200.0, 0.0),
                    egui::pos2(200.0, -10.0),
                ],
            ),
        ];
        assert_eq!(
            nearest_tangent_lever(&levers, egui::pos2(2.0, 5.0), 6.0),
            Some(5),
            "the near lever answers, and it answers as its own point"
        );
        assert_eq!(
            nearest_tangent_lever(&levers, egui::pos2(198.0, -5.0), 6.0),
            Some(9),
            "the far lever answers for itself, not for the one that was hit first"
        );
        assert_eq!(
            nearest_tangent_lever(&levers, egui::pos2(100.0, 0.0), 6.0),
            None,
            "between the two handles there is no handle"
        );
    }

    /// A click is not a tiny drag. Nothing downstream of this runs until it says so, which is what
    /// stops a press on a point from re-authoring the point's position to the snapped cursor.
    #[test]
    fn a_press_that_has_not_travelled_is_still_a_click() {
        let press = Some((100.0, 100.0));
        assert!(
            !pointer_left_the_press(press, press),
            "a press and a release in the same place moved nothing"
        );
        assert!(
            !pointer_left_the_press(press, Some((103.0, 104.0))),
            "a hand that shakes a few pixels is still clicking"
        );
        assert!(
            pointer_left_the_press(press, Some((100.0, 105.0))),
            "one axis reaching the threshold is a drag, the way the view cube reads it"
        );
        assert!(
            !pointer_left_the_press(None, Some((999.0, 999.0))),
            "no press to measure from is no drag"
        );
    }

    /// Two spans of one ellipse are one marquee answer, and window means the WHOLE curve —
    /// enclosing a single quarter is not enclosing the ellipse.
    #[test]
    fn the_marquee_folds_an_aggregate_into_one_pick() {
        let ellipse = SketchCurve::Ellipse(7);
        let inside = vec![egui::pos2(10.0, 10.0), egui::pos2(20.0, 20.0)];
        let outside = vec![egui::pos2(400.0, 400.0), egui::pos2(500.0, 500.0)];
        let box_over_the_inside_span =
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(100.0, 100.0));
        let spans = vec![(ellipse, inside.clone()), (ellipse, outside)];

        // Crossing: touching one span is touching the ellipse, and it is named once.
        assert_eq!(
            aggregate_marquee_picks(&spans, box_over_the_inside_span, false),
            vec![ellipse]
        );
        // Window: one span outside the box means the ellipse is not enclosed.
        assert!(aggregate_marquee_picks(&spans, box_over_the_inside_span, true).is_empty());
        // Window: every span inside encloses it.
        assert_eq!(
            aggregate_marquee_picks(&[(ellipse, inside)], box_over_the_inside_span, true),
            vec![ellipse]
        );
    }

    /// An aggregate is offered last, so a segment lying exactly under a spline still wins the
    /// click — the simpler primitive is what the author means by an exact tie.
    #[test]
    fn an_exact_tie_prefers_the_simpler_primitive_over_an_aggregate() {
        assert_eq!(
            nearest_sketch_edge_from_candidates([
                Some((SketchEdgeHit::Segment(1), 4.0)),
                Some((SketchEdgeHit::HigherCurve(SketchCurve::Spline(2)), 4.0)),
            ]),
            Some(SketchEdgeHit::Segment(1))
        );
        // A genuinely nearer aggregate still wins.
        assert_eq!(
            nearest_sketch_edge_from_candidates([
                Some((SketchEdgeHit::Segment(1), 4.0)),
                Some((SketchEdgeHit::HigherCurve(SketchCurve::Spline(2)), 1.0)),
            ]),
            Some(SketchEdgeHit::HigherCurve(SketchCurve::Spline(2)))
        );
    }

    #[test]
    fn tangent_badge_has_one_projected_contact_anchor_or_none() {
        let (sketch, segment, circle) = tangent_ready_sketch();
        let anchor = tangent_badge_anchor(
            &sketch,
            SketchCurve::Segment(segment),
            SketchCurve::Circle(circle),
            TangentBranch::Line(LineSide::Left),
            Some(context()),
            Some(|at: [f64; 2]| [at[0] as f32 / 10.0, at[1] as f32 / 10.0, 0.0]),
            (glam::Mat4::IDENTITY, [0, 0, 200, 100], 2.0),
        );
        let anchors: Vec<_> = anchor.into_iter().collect();
        assert_eq!(anchors, vec![pos2(75.0, 25.0)]);

        let mut off_domain = Sketch::empty(PlaneAxis::Z);
        let from = off_domain.add_free_point(SketchPoint::new(0, 0));
        let to = off_domain.add_free_point(SketchPoint::new(1, 0));
        let short_segment = off_domain.connect(from, to).expect("short segment");
        let off_domain_circle = off_domain
            .add_circle(SketchPoint::new(5, 4), SketchLength::new(4))
            .expect("circle");
        for (sketch, first, second, evaluation) in [
            (
                &off_domain,
                SketchCurve::Segment(short_segment),
                SketchCurve::Circle(off_domain_circle),
                Some(context()),
            ),
            (
                &sketch,
                SketchCurve::Segment(segment),
                SketchCurve::Circle(circle),
                None,
            ),
        ] {
            assert!(
                tangent_badge_anchor(
                    sketch,
                    first,
                    second,
                    TangentBranch::Line(LineSide::Left),
                    evaluation,
                    Some(|at: [f64; 2]| [at[0] as f32, at[1] as f32, 0.0]),
                    (glam::Mat4::IDENTITY, [0, 0, 200, 100], 1.0),
                )
                .is_none(),
                "off-domain or missing-context Tangents have no badge anchor"
            );
        }
    }

    #[test]
    fn tangent_badge_hit_selects_and_deletes_its_constraint_id() {
        let (sketch, segment, circle) = tangent_ready_sketch();
        let (producer, constraint) = SketchSolid::extrude(sketch, 3)
            .with_constraint(
                ConstraintKind::tangent(
                    SketchCurve::Segment(segment),
                    SketchCurve::Circle(circle),
                    TangentBranch::Line(LineSide::Left),
                ),
                context(),
            )
            .expect("valid tangent");
        let badges = [ui::chrome::ConstraintBadge {
            // The hit rect is the constant screen square whatever the plane does, so
            // these say the flat reading and the test stays about the pick.
            reading: egui::Vec2::X,
            square: egui::vec2(0.0, -1.0),
            center: pos2(40.0, 20.0),
            icon: ui::icons::Icon::ConstraintTangent,
            constraint,
            picked: false,
        }];
        let hit = sketch_constraint_badge_at(&badges, pos2(80.0, 40.0), 2.0)
            .expect("the generic badge returns its constraint id");
        assert_eq!(hit, constraint);

        let owner = document::scene::NodeId(77);
        let mut selection = ui::panel::Selection::default();
        selection.toggle(ui::panel::SelectionTarget::SketchConstraint {
            sketch: owner,
            entity: hit,
        });
        assert_eq!(
            selection.sketch_constraints(owner).collect::<Vec<_>>(),
            vec![hit]
        );
        let deleted = producer.with_constraint_deleted(hit);
        assert!(deleted.sketch.constraints().is_empty());
    }

    #[test]
    fn concentric_badge_projects_shared_center_and_uses_generic_delete_path() {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let first = sketch
            .add_circle(SketchPoint::new(2, 3), SketchLength::new(2))
            .expect("first circle");
        let center = sketch.circles()[0].center;
        let second = sketch
            .circle_about(center, SketchLength::new(6))
            .expect("second circle");
        let anchor = concentric_badge_anchor(
            &sketch,
            SketchCurve::Circle(first),
            SketchCurve::Circle(second),
            Some(|at: [f64; 2]| [at[0] as f32 / 10.0, at[1] as f32 / 10.0, 0.0]),
            (glam::Mat4::IDENTITY, [0, 0, 200, 100], 2.0),
        );
        let anchor = anchor.expect("projected anchor");
        assert!((anchor.x - 60.0).abs() < 1e-5 && (anchor.y - 17.5).abs() < 1e-5);

        let (producer, constraint) = SketchSolid::extrude(sketch, 3)
            .with_constraint(
                ConstraintKind::concentric(SketchCurve::Circle(first), SketchCurve::Circle(second)),
                context(),
            )
            .expect("concentric");
        let badges = [ui::chrome::ConstraintBadge {
            // The hit rect is the constant screen square whatever the plane does, so
            // these say the flat reading and the test stays about the pick.
            reading: egui::Vec2::X,
            square: egui::vec2(0.0, -1.0),
            center: anchor,
            icon: ui::icons::Icon::ConstraintConcentric,
            constraint,
            picked: false,
        }];
        assert_eq!(
            sketch_constraint_badge_at(&badges, pos2(120.0, 35.0), 2.0),
            Some(constraint)
        );
        assert!(producer
            .with_constraint_deleted(constraint)
            .sketch
            .constraints()
            .is_empty());

        let mut invalid = Sketch::empty(PlaneAxis::Z);
        let first = invalid
            .add_circle(SketchPoint::new(0, 0), SketchLength::new(2))
            .expect("first");
        let second = invalid
            .add_circle(SketchPoint::new(4, 0), SketchLength::new(3))
            .expect("second");
        assert!(concentric_badge_anchor(
            &invalid,
            SketchCurve::Circle(first),
            SketchCurve::Circle(second),
            Some(|at: [f64; 2]| [at[0] as f32, at[1] as f32, 0.0]),
            (glam::Mat4::IDENTITY, [0, 0, 200, 100], 1.0),
        )
        .is_none());
    }

    #[test]
    fn symmetry_badge_projects_document_witness_and_uses_generic_delete_path() {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let axis_from = sketch.add_free_point(SketchPoint::new(-4, -4));
        let axis_to = sketch.add_free_point(SketchPoint::new(4, 4));
        let axis = sketch.connect(axis_from, axis_to).expect("axis");
        let a0 = sketch.add_free_point(SketchPoint::new(-2, 0));
        let a1 = sketch.add_free_point(SketchPoint::new(-1, 1));
        let b0 = sketch.add_free_point(SketchPoint::new(0, -2));
        let b1 = sketch.add_free_point(SketchPoint::new(1, -1));
        let first = sketch.connect(a0, a1).expect("first");
        let second = sketch.connect(b0, b1).expect("second");
        let anchor = symmetry_badge_anchor(
            &sketch,
            SketchCurve::Segment(first),
            SketchCurve::Segment(second),
            axis,
            SymmetryBranch::Direct,
            Some(context()),
            Some(|at: [f64; 2]| [at[0] as f32 / 10.0, at[1] as f32 / 10.0, 0.0]),
            (glam::Mat4::IDENTITY, [0, 0, 200, 100], 2.0),
        )
        .expect("projected witness");
        let (producer, constraint) = SketchSolid::extrude(sketch, 3)
            .with_constraint(
                ConstraintKind::symmetry(
                    SketchCurve::Segment(first),
                    SketchCurve::Segment(second),
                    axis,
                    SymmetryBranch::Direct,
                ),
                context(),
            )
            .expect("symmetry");
        let badges = [ui::chrome::ConstraintBadge {
            // The hit rect is the constant screen square whatever the plane does, so
            // these say the flat reading and the test stays about the pick.
            reading: egui::Vec2::X,
            square: egui::vec2(0.0, -1.0),
            center: anchor,
            icon: ui::icons::Icon::ConstraintSymmetry,
            constraint,
            picked: false,
        }];
        assert_eq!(
            sketch_constraint_badge_at(&badges, anchor * 2.0, 2.0),
            Some(constraint)
        );
        assert!(producer
            .with_constraint_deleted(constraint)
            .sketch
            .constraints()
            .is_empty());
    }

    #[test]
    fn completed_tangent_queues_its_canonical_branch_through_set_sketch() {
        let (sketch, segment, circle) = tangent_ready_sketch();
        let mut armed = ui::panel::ArmedConstraint::new(ui::panel::ConstraintVerb::Tangent);
        assert_eq!(
            armed.offer_at(
                ui::panel::SketchEntity::Curve(document::sketch::SketchCurve::Circle(circle)),
                [5.0, 0.0],
                &sketch
            ),
            ui::panel::Offer::Taken
        );
        assert_eq!(
            armed.offer_at(
                ui::panel::SketchEntity::Curve(document::sketch::SketchCurve::Segment(segment)),
                [5.0, 0.0],
                &sketch
            ),
            ui::panel::Offer::Complete
        );
        let kind = armed.kind_at_context(&sketch, context()).expect("branch");
        assert!(matches!(
            kind,
            ConstraintKind::Tangent {
                branch: TangentBranch::Line(LineSide::Left),
                ..
            }
        ));
        let (constrained, _) = SketchSolid::extrude(sketch, 3)
            .with_constraint(kind, context())
            .expect("tangent completion");
        let target = document::scene::NodeId(78);
        let transaction =
            sketch_profile_edit_transaction(target, constrained, [0, 0, 0], [0, 0, 0]);
        assert_eq!(transaction.len(), 1, "one completed constraint edit");
        let Some(crate::Intent::SetSketch {
            target: queued_target,
            producer,
        }) = transaction.first()
        else {
            return;
        };
        assert_eq!(*queued_target, target);
        assert!(matches!(
            producer
                .sketch
                .constraints()
                .first()
                .map(|constraint| constraint.kind),
            Some(ConstraintKind::Tangent {
                branch: TangentBranch::Line(LineSide::Left),
                ..
            })
        ));
    }

    #[test]
    fn completed_concentric_queues_set_sketch_and_refusal_restarts_cleanly() {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let first = sketch
            .add_circle(SketchPoint::new(0, 0), SketchLength::new(2))
            .expect("first");
        let second = sketch
            .add_circle(SketchPoint::new(8, 4), SketchLength::new(5))
            .expect("second");
        let mut armed = ui::panel::ArmedConstraint::new(ui::panel::ConstraintVerb::Concentric);
        assert_eq!(
            armed.offer(
                ui::panel::SketchEntity::Curve(document::sketch::SketchCurve::Circle(second)),
                &sketch
            ),
            ui::panel::Offer::Taken
        );
        assert_eq!(
            armed.offer(
                ui::panel::SketchEntity::Curve(document::sketch::SketchCurve::Circle(first)),
                &sketch
            ),
            ui::panel::Offer::Complete
        );
        let kind = armed.kind_at_context(&sketch, context()).expect("complete");
        let (constrained, constraint) = SketchSolid::extrude(sketch, 3)
            .with_constraint(kind, context())
            .expect("concentric completion");
        let target = document::scene::NodeId(82);
        let transaction =
            sketch_profile_edit_transaction(target, constrained.clone(), [0, 0, 0], [0, 0, 0]);
        assert!(matches!(
            transaction.as_slice(),
            [crate::Intent::SetSketch { target: queued, producer }]
                if *queued == target && producer.sketch.constraints()[0].id == constraint
        ));

        let mut selection = ui::panel::Selection::from_targets([
            ui::panel::SelectionTarget::SketchCircle {
                sketch: target,
                entity: first,
            },
            ui::panel::SelectionTarget::SketchCircle {
                sketch: target,
                entity: second,
            },
        ]);
        let mut armed = Some(armed);
        let refusal = reset_refused_sketch_constraint_completion(
            &mut armed,
            &mut selection,
            ui::panel::ConstraintVerb::Concentric,
            &document::sketch::ConstraintRefusal::AlreadyAsserted {
                existing: constraint,
            },
        );
        select_sketch_constraint_refusal_culprits(
            &mut selection,
            target,
            &document::sketch::ConstraintRefusal::AlreadyAsserted {
                existing: constraint,
            },
        );
        assert_eq!(refusal, "already asserted — it is selected");
        assert_eq!(
            selection.sketch_constraints(target).collect::<Vec<_>>(),
            vec![constraint]
        );
        assert!(armed.is_some_and(|armed| {
            armed.verb() == ui::panel::ConstraintVerb::Concentric && armed.picked().is_empty()
        }));
    }

    #[test]
    fn completed_symmetry_queues_one_set_sketch_and_refusal_restarts_cleanly() {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let axis_from = sketch.add_free_point(SketchPoint::new(0, -10));
        let axis_to = sketch.add_free_point(SketchPoint::new(0, 10));
        let axis = sketch.connect(axis_from, axis_to).expect("axis");
        let a0 = sketch.add_free_point(SketchPoint::new(-4, 0));
        let a1 = sketch.add_free_point(SketchPoint::new(-4, 4));
        let b0 = sketch.add_free_point(SketchPoint::new(4, 0));
        let b1 = sketch.add_free_point(SketchPoint::new(4, 4));
        let first = sketch.connect(a0, a1).expect("first");
        let second = sketch.connect(b0, b1).expect("second");
        let mut armed = ui::panel::ArmedConstraint::new(ui::panel::ConstraintVerb::Symmetry);
        assert_eq!(
            armed.offer(
                ui::panel::SketchEntity::Curve(document::sketch::SketchCurve::Segment(second)),
                &sketch
            ),
            ui::panel::Offer::Taken
        );
        assert_eq!(
            armed.offer(
                ui::panel::SketchEntity::Curve(document::sketch::SketchCurve::Segment(first)),
                &sketch
            ),
            ui::panel::Offer::Taken
        );
        assert_eq!(
            armed.offer(
                ui::panel::SketchEntity::Curve(document::sketch::SketchCurve::Segment(axis)),
                &sketch
            ),
            ui::panel::Offer::Complete
        );
        let kind = armed.kind_at_context(&sketch, context()).expect("branch");
        let (constrained, constraint) = SketchSolid::extrude(sketch, 3)
            .with_constraint(kind, context())
            .expect("symmetry completion");
        let target = document::scene::NodeId(83);
        let transaction =
            sketch_profile_edit_transaction(target, constrained, [0, 0, 0], [0, 0, 0]);
        assert!(matches!(
            transaction.as_slice(),
            [crate::Intent::SetSketch { target: queued, producer }]
                if *queued == target && producer.sketch.constraints()[0].id == constraint
        ));
        let mut selection = ui::panel::Selection::from_targets([
            ui::panel::SelectionTarget::SketchSegment {
                sketch: target,
                entity: first,
            },
            ui::panel::SelectionTarget::SketchSegment {
                sketch: target,
                entity: second,
            },
            ui::panel::SelectionTarget::SketchSegment {
                sketch: target,
                entity: axis,
            },
        ]);
        let mut held = Some(armed);
        let refusal = reset_refused_sketch_constraint_completion(
            &mut held,
            &mut selection,
            ui::panel::ConstraintVerb::Symmetry,
            &document::sketch::ConstraintRefusal::AlreadyAsserted {
                existing: constraint,
            },
        );
        assert_eq!(refusal, "already asserted — it is selected");
        assert!(selection.is_empty());
        assert!(held.is_some_and(|held| {
            held.verb() == ui::panel::ConstraintVerb::Symmetry && held.picked().is_empty()
        }));
    }

    #[test]
    fn failed_tangent_branch_choice_restarts_empty_and_clears_temporary_selection() {
        let (sketch, segment, circle) = tangent_ready_sketch();
        let mut attempted = ui::panel::ArmedConstraint::new(ui::panel::ConstraintVerb::Tangent);
        assert_eq!(
            attempted.offer_at(
                ui::panel::SketchEntity::Curve(document::sketch::SketchCurve::Segment(segment)),
                [f64::NAN, 0.0],
                &sketch
            ),
            ui::panel::Offer::Taken
        );
        assert_eq!(
            attempted.offer_at(
                ui::panel::SketchEntity::Curve(document::sketch::SketchCurve::Circle(circle)),
                [5.0, 0.0],
                &sketch
            ),
            ui::panel::Offer::Complete
        );
        assert_eq!(
            attempted.kind_at_context(&sketch, context()),
            Err("cannot choose a tangent branch here")
        );

        let owner = document::scene::NodeId(79);
        let mut selection = ui::panel::Selection::default();
        selection.toggle(ui::panel::SelectionTarget::SketchSegment {
            sketch: owner,
            entity: segment,
        });
        selection.toggle(ui::panel::SelectionTarget::SketchCircle {
            sketch: owner,
            entity: circle,
        });
        let mut armed = Some(attempted);
        reset_failed_sketch_constraint_completion(
            &mut armed,
            &mut selection,
            ui::panel::ConstraintVerb::Tangent,
        );
        assert!(selection.is_empty());
        assert_eq!(
            armed.as_ref().map(ui::panel::ArmedConstraint::verb),
            Some(ui::panel::ConstraintVerb::Tangent)
        );
        assert!(armed.is_some_and(|armed| armed.picked().is_empty()));
    }

    #[test]
    fn completed_constraint_without_context_reports_and_restarts_exactly() {
        let (sketch, segment, circle) = tangent_ready_sketch();
        let mut attempted = ui::panel::ArmedConstraint::new(ui::panel::ConstraintVerb::Tangent);
        assert_eq!(
            attempted.offer_at(
                ui::panel::SketchEntity::Curve(document::sketch::SketchCurve::Segment(segment)),
                [5.0, 0.0],
                &sketch
            ),
            ui::panel::Offer::Taken
        );
        assert_eq!(
            attempted.offer_at(
                ui::panel::SketchEntity::Curve(document::sketch::SketchCurve::Circle(circle)),
                [5.0, 0.0],
                &sketch
            ),
            ui::panel::Offer::Complete
        );
        let owner = document::scene::NodeId(80);
        let mut selection = ui::panel::Selection::from_targets([
            ui::panel::SelectionTarget::SketchSegment {
                sketch: owner,
                entity: segment,
            },
            ui::panel::SelectionTarget::SketchCircle {
                sketch: owner,
                entity: circle,
            },
        ]);
        let mut armed = Some(attempted);

        let refusal = reset_refused_sketch_constraint_completion(
            &mut armed,
            &mut selection,
            ui::panel::ConstraintVerb::Tangent,
            &document::sketch::ConstraintRefusal::MissingEvaluationContext,
        );

        assert_eq!(
            refusal,
            "needs the document density to resolve its fixed curve"
        );
        assert!(selection.is_empty());
        assert!(armed.is_some_and(|armed| {
            armed.verb() == ui::panel::ConstraintVerb::Tangent && armed.picked().is_empty()
        }));
    }

    #[test]
    fn invalid_standing_tangent_refusal_selects_its_badge() {
        let owner = document::scene::NodeId(81);
        let culprit = 912;
        let refusal = document::sketch::ConstraintRefusal::InvalidTangent {
            constraint: Some(culprit),
            error: parametric::sketch::TangentContactError::OutsideSecondDomain,
        };
        let mut selection = ui::panel::Selection::default();

        select_sketch_constraint_refusal_culprits(&mut selection, owner, &refusal);

        assert!(
            selection.contains(ui::panel::SelectionTarget::SketchConstraint {
                sketch: owner,
                entity: culprit,
            })
        );
    }

    #[test]
    fn a_segment_beside_the_box_touches_nothing() {
        let rect = Rect::from_min_max(pos2(10.0, 10.0), pos2(20.0, 20.0));
        assert!(!segment_touches_rect(pos2(0.0, 5.0), pos2(30.0, 5.0), rect));
        // A diagonal skimming PAST the corner (outside) also misses.
        assert!(!segment_touches_rect(pos2(0.0, 8.0), pos2(8.0, 0.0), rect));
    }

    #[test]
    fn an_endpoint_inside_the_box_touches() {
        let rect = Rect::from_min_max(pos2(10.0, 10.0), pos2(20.0, 20.0));
        assert!(segment_touches_rect(
            pos2(15.0, 15.0),
            pos2(40.0, 40.0),
            rect
        ));
    }

    #[test]
    fn collinear_disjoint_segments_do_not_intersect() {
        // All four orientations are zero, but the shadows do not overlap — the naive sign test
        // would call this an intersection.
        assert!(!segments_intersect(
            pos2(0.0, 10.0),
            pos2(5.0, 10.0),
            pos2(10.0, 10.0),
            pos2(20.0, 10.0)
        ));
        assert!(segments_intersect(
            pos2(0.0, 10.0),
            pos2(12.0, 10.0),
            pos2(10.0, 10.0),
            pos2(20.0, 10.0)
        ));
    }

    #[test]
    fn crossing_segments_intersect() {
        assert!(segments_intersect(
            pos2(0.0, 0.0),
            pos2(10.0, 10.0),
            pos2(0.0, 10.0),
            pos2(10.0, 0.0)
        ));
        assert!(!segments_intersect(
            pos2(0.0, 0.0),
            pos2(1.0, 1.0),
            pos2(0.0, 10.0),
            pos2(10.0, 0.0)
        ));
    }

    /// The region hit-test's two primitives, on a concave face — the shape the badge and the
    /// smallest-wins rule both have to survive (#100). An L is the cheapest concave polygon.
    #[test]
    fn the_region_hit_test_handles_a_concave_face() {
        let ell = [
            pos2(0.0, 0.0),
            pos2(6.0, 0.0),
            pos2(6.0, 2.0),
            pos2(2.0, 2.0),
            pos2(2.0, 6.0),
            pos2(0.0, 6.0),
        ];
        assert!(
            point_in_screen_polygon(&ell, pos2(1.0, 5.0)),
            "inside the tall arm"
        );
        assert!(
            point_in_screen_polygon(&ell, pos2(5.0, 1.0)),
            "inside the wide arm"
        );
        assert!(
            !point_in_screen_polygon(&ell, pos2(5.0, 5.0)),
            "the notch is outside"
        );
        // Winding must not change the answer: a face traced the other way is the same face.
        let reversed: Vec<_> = ell.iter().rev().copied().collect();
        assert!(point_in_screen_polygon(&reversed, pos2(1.0, 5.0)));
        assert_eq!(
            polygon_double_area(&ell).abs(),
            polygon_double_area(&reversed).abs()
        );
    }
    /// A dimension label shows the number the author typed, not a float's idea of it.
    #[test]
    fn a_dimension_value_drops_the_zeros_it_does_not_need() {
        assert_eq!(trim_number(30.0), "30");
        assert_eq!(trim_number(22.5), "22.5");
        // Rust formats half-way values to EVEN, so this is 0.12 and not 0.13.
        assert_eq!(trim_number(0.125), "0.12");
        assert_eq!(trim_number(-0.001), "0");
        assert_eq!(trim_number(0.0), "0");
    }

    /// **Two lines make four corners and the annotation picks one of them.**
    ///
    /// The stored corner narrows the four to two — a size is a claim and the supplement is a
    /// different number — and the anchor chooses between the two of that size, which are opposite
    /// each other and say the same thing. Read the arms by their authored direction alone and an
    /// author can only ever dimension one of the four.
    #[test]
    fn the_corner_an_angle_is_struck_in_is_the_one_the_text_was_dropped_in() {
        let flat = |coord: [f64; 2]| Some(egui::Pos2::new(coord[0] as f32, coord[1] as f32));
        let vertex = egui::Pos2::new(0.0, 0.0);
        let quarter = std::f32::consts::FRAC_PI_2;
        // Along +x and +y. Screen y runs DOWN, so this reads as a quarter turn clockwise.
        let across = ([0.0, 0.0], [10.0, 0.0]);
        let up = ([0.0, 0.0], [0.0, 4.0]);
        let struck = |corner, at: Option<[f32; 2]>| {
            angle_legs(
                vertex,
                &flat,
                across,
                up,
                corner,
                at.map(|at| egui::Pos2::new(at[0], at[1])),
            )
            .expect("two live arms off one vertex")
        };

        // Dropped inside the corner the two arms bound: the arc is struck there.
        let (from, to, legs) = struck(document::sketch::AngleCorner::Between, Some([3.0, 3.0]));
        assert!(from.abs() < 1e-5, "the first arm bears along +x: {from}");
        assert!((to - quarter).abs() < 1e-5, "a quarter turn away: {to}");
        assert!((legs[0].furthest - 10.0).abs() < 1e-5);
        assert!((legs[1].furthest - 4.0).abs() < 1e-5);

        // Dropped in the corner OPPOSITE it: the same number, struck the other side of the vertex,
        // and now each arm reaches BACKWARD so the whole of both legs is a dogleg.
        let (from, to, legs) = struck(document::sketch::AngleCorner::Between, Some([-3.0, -3.0]));
        assert!(
            (from.abs() - std::f32::consts::PI).abs() < 1e-5,
            "the first arm now bears along -x: {from}"
        );
        assert!(
            (to - from).abs() - quarter < 1e-5,
            "still a quarter: {}",
            to - from
        );
        assert!(
            legs[0].furthest <= 0.0 && legs[1].furthest <= 0.0,
            "both arms lie behind the corner: {legs:?}"
        );

        // Dropped in a corner only ONE arm bounds: the supplement, and the arms are counter-run.
        let (from, to, _) = struck(
            document::sketch::AngleCorner::Supplementary,
            Some([3.0, -3.0]),
        );
        assert!(from.abs() < 1e-5, "{from}");
        assert!(
            ((to - from).abs() - quarter).abs() < 1e-5,
            "a right angle's supplement is also a right angle: {}",
            to - from
        );
        assert!(
            to < from,
            "and it is struck the other way round: {to} from {from}"
        );

        // With no anchor the arms are read the way they were drawn, which is the pair that agrees
        // with the number the corner names.
        let (from, to, _) = struck(document::sketch::AngleCorner::Between, None);
        assert!(from.abs() < 1e-5 && (to - quarter).abs() < 1e-5);

        // An arm of no length has no direction to give, and the gizmo is skipped rather than
        // struck about an arbitrary one.
        assert!(angle_legs(
            vertex,
            &flat,
            ([0.0, 0.0], [0.0, 0.0]),
            up,
            document::sketch::AngleCorner::Between,
            None,
        )
        .is_none());
    }

    /// **The dogleg spans whatever the line does not**, which for two lines that never touch means
    /// it can have to run INWARD, toward a vertex neither of them reaches.
    #[test]
    fn two_lines_that_never_touch_still_state_the_angle_between_them() {
        let flat = |coord: [f64; 2]| Some(egui::Pos2::new(coord[0] as f32, coord[1] as f32));
        let vertex = egui::Pos2::new(0.0, 0.0);
        // Neither line contains the origin they cross at: one starts 2 out along +x, the other 3
        // out along +y.
        let (_, _, legs) = angle_legs(
            vertex,
            &flat,
            ([2.0, 0.0], [10.0, 0.0]),
            ([0.0, 3.0], [0.0, 9.0]),
            document::sketch::AngleCorner::Between,
            Some(egui::Pos2::new(4.0, 4.0)),
        )
        .expect("two lines that cross nowhere on themselves still cross");
        assert!(
            (legs[0].nearest - 2.0).abs() < 1e-5 && (legs[0].furthest - 10.0).abs() < 1e-5,
            "the first line runs 2 to 10 along its ray: {:?}",
            legs[0]
        );
        assert!(
            (legs[1].nearest - 3.0).abs() < 1e-5 && (legs[1].furthest - 9.0).abs() < 1e-5,
            "{:?}",
            legs[1]
        );
    }
}
