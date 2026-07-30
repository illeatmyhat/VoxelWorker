//! The shell's per-frame render seam: acquire the surface texture, poll the display/measurement
//! workers, run the egui frame, apply this frame's Intents + view actions, upload every
//! renderer's uniforms, and submit the shared [`render_frame`]. Split out of `windowed/mod.rs`
//! (ADR 0016).

use super::*;

/// How far (physical px) a drawn arc chord may sag from the true curve. A quarter pixel is under
/// the width of the thinnest stroke the gizmo family draws, so the tessellation is invisible at
/// any zoom; it only ever refines the DRAWING, never the resolved profile.
const ARC_SCREEN_SAGITTA_PX: f64 = 0.25;

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
                self.recentre_voxels,
                clip.band,
                clip.region,
            );
            if self.display.poll_brick_worker(context) {
                self.window.request_redraw();
            }
        }
        // ADR 0010 E5 follow-up: accept a finished (non-stale) diameter measurement.
        self.poll_diameter_worker();

        // M6: drain the background scan channel and turn any new groups into
        // palette tiles (GPU thumbnail + egui texture registration on this thread).
        self.poll_scan();

        let raw_input = self.egui_winit_state.take_egui_input(&self.window);
        let pixels_per_point = self.egui_winit_state.egui_ctx().pixels_per_point();

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
            // ADR 0010 E5 follow-up: re-measure the diameter ASYNCHRONOUSLY. The streamed
            // cacheless query (a coarse block contributes its run block-granular, boundary
            // per-voxel — the SAME value the retired dense `widest_run_in_band` returns) is
            // O(total blocks): sub-second on a huge solid but not free, and it must never
            // block the event-loop thread. Dispatch it to the `DiameterWorker`; the shell
            // keeps showing the previous (stale) `measured_diameter` until the result lands
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
        // the camera target, converted from the recentred render frame back to whole
        // world blocks (`(target_voxels + recentre) / density`), so a new Point lands
        // where the user is looking.
        {
            let density = self.panel_state.geometry.voxels_per_block.max(1) as i64;
            let recentre = self
                .panel_state
                .scene
                .recentre_voxels_for_resolve(self.panel_state.geometry.voxels_per_block)
                .voxels();
            let target = self.app_core.camera.target;
            self.panel_state.point_add_position_blocks = [
                ((target.x.round() as i64) + recentre[0]).div_euclid(density),
                ((target.y.round() as i64) + recentre[1]).div_euclid(density),
                ((target.z.round() as i64) + recentre[2]).div_euclid(density),
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

        // ADR 0018 Decision 5: the layer scrubber's track spans the SELECTED object's Z
        // extent in Onion-fog mode (else the whole scene). Read it from the shared clip
        // (a no-op walk outside Onion-fog mode, where it returns the scene `grid_z`).
        let layer_track_len = self.current_mesh_clip(grid_z).track_len;
        // Read before the call: `run_egui_frame` borrows `self` mutably.
        let orbit_center_marker = self.orbit_center_marker(pixels_per_point);
        let orbit_reticle = self.orbit_reticle_visible();
        let sketch_face_at_menu = self.sketch_menu_face_is_picked();
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
                // ADR 0028 (#94): the sketch vertex handles, projected LAST frame (the
                // viewport + camera the projection needs are only known after this call).
                // A one-frame lag is imperceptible for handle chrome and self-corrects; the
                // cache is refreshed at the end of `render` below.
                &self.sketch_overlay_points,
                // ADR 0030: the committed segment lines, projected last frame — drawn under the
                // vertex dots so the profile reads as connected edges.
                &self.sketch_segment_lines,
                // ADR 0030 §5 (#102): the committed arc curves, projected last frame — the same
                // under-layer as the straight edges.
                &self.sketch_arc_lines,
                // ADR 0030 §3 (#100): the picked regions' wash, projected last frame.
                &self.sketch_face_washes,
                // #100: the pick state of the region the open menu was raised inside, so the
                // menu can label its row "carve" or "fill".
                sketch_face_at_menu,
                // ADR 0028 (#95): the add-point insert preview, projected last frame.
                self.sketch_insert_preview,
                // #99: the drawing tools' dashed preview, projected last frame.
                &self.sketch_draw_preview,
                // Slice 3: the marquee rubber band, computed last frame.
                self.sketch_marquee_band,
                // ADR 0032: the orbit-center marker — live under the cursor while a placement is
                // armed, projected-last-frame while Shift+MMB turns about it.
                orbit_center_marker,
                // ADR 0032: whether the orbit mode's targeting reticle draws — it fills the
                // viewport rect the frame computes, so no position travels with the flag.
                orbit_reticle,
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

        // Signal (ADR 0018 Decision 8): the icon rail's Home/Fit click, pre-mapped onto a
        // `ChromeClickAction`, runs through the SAME `run_chrome_action` the (now retired)
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
        // the node's recentred world centre and fit the distance to its AABB (same fit
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

        // ADR 0003 Phase C C4a: the panel no longer mutates the scene directly — it
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
        // ADR 0022 live placement: adopt a tool the panel armed this frame (a VIEW
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
        // ADR 0032: a clicked row's selection change — a VIEW action on the response, like
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
        // ADR 0028: enter / leave sketch mode — a VIEW action on the response (entering a mode
        // mutates no document state), like `armed_tool`. Entering scopes the mode to the
        // requested node, disarms any placement tool (non-sketch ops withdraw in the mode), and
        // OPENS the undo group (§4). Finish commits the session as one main-history entry;
        // Cancel rolls it back to the enter-state (which re-resolves) — both drop the mode. The
        // group-close effect folds into `merged_effect` below so a Cancel rebuilds like an edit.
        let mut sketch_effect = crate::IntentEffect::none();
        if let Some(node) = prepared.panel_response.enter_sketch.take() {
            self.panel_state.sketch_mode = Some(node);
            self.disarm_placement();
            self.panel_state.selection.clear_sketch_entities();
            self.app_core.begin_sketch_group();
        }
        if let Some(exit) = prepared.panel_response.exit_sketch.take() {
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
        // The context menu's orbit-center rows. Not an `Intent` and not undoable — the camera
        // is not the document (ADR 0022's classification: this is view state).
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
        // ADR 0028 (#94): advance an in-progress sketch vertex drag — a live preview that
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
        // ADR 0022 live placement: a viewport click's drop intent is applied through the
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
        // Batched intents that must land as ONE undo step: a multi-node Delete (ADR 0033), and
        // every sketch commit — one authoring act, one press of Ctrl+Z (owner 2026-07-29).
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
        // ADR 0018 Decision 6: re-derive the boolean-operand ghost on selection /
        // geometry / MODE change ONLY (never per frame). A selection click marks it dirty
        // without a scene re-resolve; the derivation is bounded by the ghosted operands'
        // covering chunks (`AppCore::boolean_operand_ghost`), so this stays cheap even in
        // a huge scene. The selection / mode comparisons are belt-and-braces for any
        // selection or mode writer that bypassed the Intent effects. The ghost is
        // populated only in Show-booleans mode; Normal / Onion-fog derive nothing.
        if merged_effect.selection_changed || merged_effect.scene_changed {
            self.selected_ghost_dirty = true;
        }
        // ADR 0032: the selection outline+wash re-derives on the SAME seam (selection /
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
                        cel.recentre,
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
                    ghost.recentre,
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
                self.recentre_voxels,
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
        // model is centred in the visible 3D area instead of partly hidden behind
        // the side panel. `prepared.viewport_px` = [x, y, w, h] in physical pixels.
        let [_, _, viewport_width, viewport_height] = prepared.viewport_px;
        let aspect_ratio = viewport_width as f32 / viewport_height.max(1) as f32;
        let geometry = self.panel_state.geometry.clone();
        // The grid dims come from the ACTUALLY resolved scene grid (the composited
        // region's extent), not the active node's geometry — with several nodes the
        // region is the per-axis max of their sizes (ADR 0001 step 2).
        let grid_dimensions = self.region_dimensions;
        let scene_matrices = self.app_core.scene_matrices(aspect_ratio, grid_dimensions);
        let view_projection = scene_matrices.view_projection;
        // ADR 0028 (#94): refresh the sketch vertex-handle overlay from the CURRENT geometry
        // (post-rebuild recentre) and camera, caching the projected handles for NEXT frame's
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
        // ADR 0018 Decisions 4–5: the region-scoped clip (band + onion-fog region). The
        // band bites only in Onion-fog mode with a selection; the region confines it to the
        // selected object's AABB. BOTH display paths honour the region — the cuboid mesh path
        // (geometry) and the brick raymarch (per-frame uniforms, #85).
        let clip = self.current_mesh_clip(grid_dimensions[2]);
        let band = clip.band;
        // Part of #20: the cuboid mesh path is the sole voxel renderer. Upload its
        // per-frame uniforms (camera + per-material base colours + band + region clip). A
        // loaded VS block textures it per-face (its 6-layer D2Array is bound at DRAW
        // time in `render_frame`, selecting the loaded pipeline); `bound = None` then
        // just disables the procedural per-box modulation/atlas, which the loaded
        // pipeline ignores.
        let bound = match &self.loaded_material {
            Some(_) => None,
            None => Some(self.panel_state.material),
        };
        // ADR 0012 (H1): the onion GHOST replaces the volumetric fog. Active when onion skin is on
        // and the band is a real slab (`current_layer_band` sets a non-zero `onion_depth` exactly
        // then; debug-face mode forces FULL → 0). The engaged display path draws the ghost after
        // its solid pass (`render_frame`); a band scrub is a pure uniform update on the brick path,
        // a thin-slab re-mesh on the cuboid path — never the fog atlas rebuild.
        let onion_ghost_active = band.onion_depth > 0;
        // Voxel-model uniforms, shared with `shot` (ADR 0031): the cuboid mesh + (when engaged) the
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
        // infinite grid), shared with `shot` through one orchestration point (ADR 0031). The
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
        // ADR 0022 live placement: while a tool is armed and the cursor is over the
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
                    recentre_voxels: self.recentre_voxels.voxels(),
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
                        // ADR 0027: the ghost previews the node as it WILL land — tilted to the
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
        // ADR 0022: the armed-tool placement ghost. Arm it from the armed tool's pending
        // drop (resolved live above, or restored from a loaded config F9 repro), resolving
        // the render-frame field centre from THIS rebuild's recentre so the ghost sits in
        // the exact frame the solid voxels are drawn in (ADR 0008). Disarmed → no-op.
        if let Some(ghost) = self.panel_state.placement_ghost() {
            let voxels_per_block = self.panel_state.geometry.voxels_per_block;
            let recentre = self.recentre_voxels.voxels();
            self.placement_ghost_renderer.update_uniforms(
                &self.gpu.queue,
                view_projection,
                scene_matrices.ray_unprojection.inverse(),
                scene_matrices.ray_eye,
                prepared.viewport_px,
                glam::Vec3::from_array(ghost.center_world(recentre, voxels_per_block)),
                ghost.shape.kind,
                glam::Vec3::from_array(ghost.semi_axes(voxels_per_block)),
                ghost.wall_voxels(voxels_per_block),
                crate::PLACEMENT_GHOST_TINT,
                ghost.rotation_inverse_columns(),
            );
        } else {
            self.placement_ghost_renderer.disarm();
        }
        // Overlay uniforms shared with `shot` (ADR 0031): the selection-follow gizmo, the
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

        // ADR 0012: the onion-skin VOLUMETRIC FOG is retired. Onion context draws as the
        // display paths' ghost pass (prepared above: the brick slabs in `update_ghost_uniforms`,
        // the cuboid slabs in `update_uniforms` → `rebuild_for_band`; drawn in `render_frame`
        // when `onion_ghost_active`).
        let _ = layer_range;

        // The ordered frame phases (ADR 0031). Each renderer self-gates (empty batch → no
        // draw), so an always-included draw is a cheap no-op; only the gizmo (a fixed unit
        // gizmo, always non-empty) is gated on there being a selection.
        let background: [&dyn display::SceneDraw; 1] = [&self.background_gradient_renderer];
        let mut over_model: Vec<&dyn display::SceneDraw> = Vec::new();
        // ADR 0018 D6: the operand x-ray — suppressed in debug-faces mode; self-gates when
        // empty. (The ADR 0032 selection feedback is no longer an over-model draw: it is
        // the screen-space outline+wash composite, wired below as `selection_outline`.)
        if !self.panel_state.debug_face_orientation {
            over_model.push(&self.selected_operand_ghost_renderer);
        }
        // ADR 0022: the armed-tool placement ghost self-gates on a pending drop.
        if self.panel_state.placement_ghost().is_some() {
            over_model.push(&self.placement_ghost_renderer);
        }
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
            // ADR 0011 G1: when engaged, the brick raymarch replaces the cuboid-mesh DRAW for
            // this frame; the mesh stays built as the fallback + A/B reference.
            brick_raymarch: if brick_raymarch_engaged {
                self.display.brick_raymarch_renderer()
            } else {
                None
            },
            // ADR 0012: ghost the onion slabs after the solid draw (uniforms/geometry prepared above).
            onion_ghost_active,
            // ADR 0032: the selection outline+wash — suppressed in debug-faces mode (like
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
            // Signal (issue #88): slide the cube left of the floating display stack.
            view_cube_right_inset_px: prepared.view_cube_right_inset_px,
            target_width: self.surface_config.width,
            target_height: self.surface_config.height,
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

    /// ADR 0028 (#94): advance an in-progress sketch vertex drag by one frame — a LIVE PREVIEW.
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
        let Some((point_id, original_min, original_offset)) = self
            .sketch_drag
            .as_ref()
            .map(|drag| (drag.point_id, drag.original_min, drag.original_offset))
        else {
            return IntentEffect::none();
        };
        let Some(target) = self.panel_state.sketch_mode else {
            self.sketch_drag = None;
            return IntentEffect::none();
        };
        let Some((cursor_x, cursor_y)) = self.last_cursor_position else {
            return IntentEffect::none();
        };
        // Recompute the handles from the CURRENT scene (not last frame's cache): a mid-drag
        // move can shift the composite recentre / profile bbox, and the forward projection and
        // the inverse plane-hit map must share ONE frame or the vertex jitters (ADR 0008).
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

        // Build the preview from the pre-drag producer with ONLY the dragged vertex moved, then
        // compensate the offset by the bbox-min shift so the rest of the profile holds still.
        let Some(drag) = self.sketch_drag.as_ref() else {
            return IntentEffect::none();
        };
        let mut preview = drag.original.clone();
        // Mutate the grabbed point ENTITY directly by its stable id (ADR 0030 — no loop index).
        // The snap policy re-authors the whole position (#96/#101): a snapped drag zeroes
        // the fraction, NoSnap carries it; either way a stale retained expression drops.
        if !preview.sketch.move_point(point_id, snapped) {
            self.sketch_drag = None;
            return IntentEffect::none();
        }
        let new_min = Self::profile_bbox_min(&preview);
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

    /// Commit an in-progress vertex drag — called SYNCHRONOUSLY from the `events` release handler
    /// (not deferred to a render flag: a deferred commit left a window where a second press could
    /// orphan the un-recorded preview). Reads the final previewed producer + offset off the node,
    /// restores the pre-drag state, then queues the final state as intents so the next `render`
    /// applies them through `apply_intent` and they record in the open group — ONE `SetSketch`,
    /// plus a `SetOffset` only when the anchor compensation actually moved the node. A gesture
    /// that ended where it began records nothing (the restored original is left in place).
    pub(super) fn commit_sketch_vertex_drag(&mut self) {
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
                    voxel_core::units::Measurement::from_voxels(final_offset[0]),
                    voxel_core::units::Measurement::from_voxels(final_offset[1]),
                    voxel_core::units::Measurement::from_voxels(final_offset[2]),
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

    /// The in-plane bbox-minimum (per profile coordinate) of a sketch producer's profile — the
    /// anchor the drag compensation measures its bbox-min shift against.
    fn profile_bbox_min(producer: &document::sketch::SketchSolid) -> [i64; 2] {
        producer.profile_bbox_min()
    }

    /// Cursor (physical px) → the CONTINUOUS profile coordinate `(c0, c1)` under it on the
    /// sketch node's plane, using `handles` for the plane + inverse map (ADR 0028). Shared by
    /// the vertex-drag preview (#94) and the add-point insert (#95) so the frame math lives once.
    ///
    /// Casts from the EYE under perspective — the near-plane ray origin is unreliable at close
    /// zoom and can sit past the target plane (placement casts from the eye for the same reason);
    /// orthographic keeps the near-plane point (parallel rays have no single eye). `None` when the
    /// unprojection fails, the ray is parallel to the plane, or the plane is behind the viewer.
    ///
    /// `ray_unprojection` is the RAY-FRAME matrix (`SceneMatrices::ray_unprojection`), not the full
    /// scene VP: under perspective the full inverse melts the `/w` divide at a wide-baseline
    /// recentre (a06d215), so we unproject the DIRECTION through the camera-relative bracket and
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
    /// (#94) and the selection click resolve (ADR 0030).
    fn sketch_vertex_at(&self, cursor_x: f64, cursor_y: f64) -> Option<usize> {
        let grab_px = (ui::chrome::SKETCH_HANDLE_HALF + ui::chrome::SKETCH_HANDLE_GRAB_PAD)
            * self.window.scale_factor() as f32;
        let mut nearest: Option<(usize, f32)> = None;
        for (index, center) in self.sketch_vertex_px.iter().enumerate() {
            let Some(center) = center else { continue };
            let distance = (cursor_x as f32 - center.x).hypot(cursor_y as f32 - center.y);
            if distance <= grab_px && nearest.map(|(_, best)| distance < best).unwrap_or(true) {
                nearest = Some((index, distance));
            }
        }
        nearest.map(|(index, _)| index)
    }

    /// The sketch SEGMENT under the cursor (physical px) as `(segment id, endpoint a px,
    /// endpoint b px)`, the nearest within the grab pad — iterated over the actual segment
    /// ENTITIES (ADR 0030), not consecutive vertices, so it is correct for an open or
    /// multi-loop graph. `None` when no edge is close enough or an endpoint is culled.
    pub(super) fn nearest_sketch_segment(
        &self,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<(document::sketch::EntityId, egui::Pos2, egui::Pos2)> {
        let pad_px = ui::chrome::SKETCH_SEGMENT_GRAB_PAD * self.window.scale_factor() as f32;
        let cursor = egui::Pos2::new(cursor_x as f32, cursor_y as f32);
        let mut nearest: Option<(document::sketch::EntityId, egui::Pos2, egui::Pos2, f32)> = None;
        for &(seg_id, a_idx, b_idx) in &self.sketch_segments {
            let (Some(&Some(a)), Some(&Some(b))) = (
                self.sketch_vertex_px.get(a_idx),
                self.sketch_vertex_px.get(b_idx),
            ) else {
                continue;
            };
            let distance = point_to_segment_distance(cursor, a, b);
            if distance <= pad_px
                && nearest
                    .map(|(_, _, _, best)| distance < best)
                    .unwrap_or(true)
            {
                nearest = Some((seg_id, a, b, distance));
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
                .windows(2)
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

    /// The sketch EDGE under the cursor — the nearer of the closest segment and the closest arc
    /// (#102). One resolution so hover feedback and the click that follows it can never
    /// disagree about which edge the cursor is on.
    pub(super) fn nearest_sketch_edge(
        &self,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<SketchEdgeHit> {
        let cursor = egui::Pos2::new(cursor_x as f32, cursor_y as f32);
        let segment = self
            .nearest_sketch_segment(cursor_x, cursor_y)
            .map(|(id, a, b)| (id, point_to_segment_distance(cursor, a, b)));
        let arc = self.nearest_sketch_arc(cursor_x, cursor_y);
        match (segment, arc) {
            (Some((seg_id, seg_d)), Some((arc_id, arc_d))) => Some(if arc_d < seg_d {
                SketchEdgeHit::Arc(arc_id)
            } else {
                SketchEdgeHit::Segment(seg_id)
            }),
            (Some((seg_id, _)), None) => Some(SketchEdgeHit::Segment(seg_id)),
            (None, Some((arc_id, _))) => Some(SketchEdgeHit::Arc(arc_id)),
            (None, None) => None,
        }
    }

    /// The id of the sketch SEGMENT under the cursor (physical px), for add-point — the click
    /// splits the named segment (ADR 0030). `None` when no edge is close enough.
    fn sketch_segment_at(
        &self,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<document::sketch::EntityId> {
        self.nearest_sketch_segment(cursor_x, cursor_y)
            .map(|(seg_id, _, _)| seg_id)
    }

    /// ADR 0028 (#95): the add-point producer for a click at the cursor (physical px) — the
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
        // Split the segment under the cursor with a policy-snapped point (ADR 0030, #96).
        let point = apply_sketch_snap(
            coord,
            self.panel_state.sketch_snap,
            self.panel_state.geometry.voxels_per_block,
        );
        Some(producer.with_point_on_segment(seg_id, point))
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

    /// #99: one polyline click. Resolves the cursor to a point — an existing vertex under it
    /// (coincidence, by screen grab radius) or a fresh grid-snapped free point — then chains:
    /// no open chain starts one at that point; an open chain connects `last → clicked` and
    /// advances; clicking the chain's FIRST point closes the loop and ends the chain; clicking
    /// its LAST point again ends it open. Each click that changes the store commits as one
    /// entry in the open sketch undo group.
    pub(super) fn sketch_polyline_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        // A chain endpoint deleted mid-gesture (Delete key, undo) leaves a dangling id —
        // drop the chain rather than connect to a ghost.
        if let Some((start, last)) = self.sketch_chain {
            let alive = |id| producer.sketch.points().iter().any(|point| point.id == id);
            if !alive(start) || !alive(last) {
                self.sketch_chain = None;
            }
        }
        let existing = self
            .sketch_vertex_at(cursor_x, cursor_y)
            .and_then(|index| self.sketch_point_ids.get(index).copied());
        let (mut next, clicked) = match existing {
            Some(id) => (producer.clone(), id),
            None => {
                let Some(snapped) = self.sketch_snapped_point_at(cursor_x, cursor_y) else {
                    return;
                };
                producer.with_point_placed(snapped)
            }
        };
        self.sketch_chain = match self.sketch_chain {
            None => Some((clicked, clicked)),
            Some((_, last)) if clicked == last => None,
            Some((start, last)) => {
                next = next.with_segment_between(last, clicked);
                (clicked != start).then_some((start, clicked))
            }
        };
        if next != producer {
            self.commit_sketch_profile_edit(target, next);
        }
    }

    /// #102: one 3-point-arc click (ADR 0030 §5). Click 1 picks the start endpoint, click 2 the
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
            let Some(bulge) = voxel_core::units::AngleMeasurement::from_degrees_f64(degrees) else {
                return;
            };
            let next = producer.with_arc_between(start, end, bulge);
            if next != producer {
                self.commit_sketch_profile_edit(target, next);
            }
            return;
        }

        let existing = self
            .sketch_vertex_at(cursor_x, cursor_y)
            .and_then(|index| self.sketch_point_ids.get(index).copied());
        let (next, clicked) = match existing {
            Some(id) => (producer.clone(), id),
            None => {
                let Some(snapped) = self.sketch_snapped_point_at(cursor_x, cursor_y) else {
                    return;
                };
                producer.with_point_placed(snapped)
            }
        };
        self.sketch_arc_gesture = match self.sketch_arc_gesture {
            None => Some((clicked, None)),
            // A zero-length arc cannot be held (ADR 0030 §5). An already-joined pair CAN: the
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

    /// #99: the rectangle tool's release. Takes the press-time anchor corner; a release whose
    /// snapped opposite corner spans both in-plane axes appends the closed four-segment loop
    /// as one undo entry — a degenerate (zero-span) or off-plane release draws nothing. Either
    /// way the anchor is consumed: each rectangle is one press-drag-release gesture.
    pub(super) fn sketch_rectangle_release(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(anchor) = self.sketch_rect_anchor.take() else {
            return;
        };
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        let Some(corner) = self.sketch_snapped_point_at(cursor_x, cursor_y) else {
            return;
        };
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        let next = producer.with_rectangle(anchor, corner);
        if next != producer {
            self.commit_sketch_profile_edit(target, next);
        }
    }

    /// ADR 0030: resolve a stationary Select-tool click into the sketch selection. A vertex under
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
            // business (ADR 0032).
            None if !shift => self.panel_state.selection.clear_sketch_entities(),
            None => {}
        }
    }

    /// Sketch-selection slice 3: resolve a DRAGGED empty-space Select release into the box
    /// selection. Direction picks the semantic (Fusion): left→right = **window** — points inside
    /// the box, segments with ≥1 endpoint inside; right→left = **crossing** — any entity the box
    /// touches, so a segment passing through with both endpoints outside still selects. Shift
    /// accumulates into the set; a plain marquee replaces the sketch-entity selection (an empty
    /// box therefore clears, like a plain empty click). A behind-camera endpoint culls its
    /// entity, matching the overlay cull. Pure selection-state mutation — no document edit.
    pub(super) fn resolve_sketch_marquee(&mut self, up_x: f64, up_y: f64) {
        let Some((down_x, down_y)) = self.sketch_marquee_anchor.take() else {
            return;
        };
        let Some(sketch) = self.panel_state.sketch_mode else {
            return;
        };
        let window = up_x >= down_x;
        let rect = egui::Rect::from_two_pos(
            egui::Pos2::new(down_x as f32, down_y as f32),
            egui::Pos2::new(up_x as f32, up_y as f32),
        );
        let mut picked: Vec<ui::panel::SelectionTarget> = Vec::new();
        for (index, vertex) in self.sketch_vertex_px.iter().enumerate() {
            let inside = vertex.map(|px| rect.contains(px)).unwrap_or(false);
            if let (true, Some(&entity)) = (inside, self.sketch_point_ids.get(index)) {
                picked.push(ui::panel::SelectionTarget::SketchPoint { sketch, entity });
            }
        }
        for &(entity, a_index, b_index) in &self.sketch_segments {
            if let (Some(Some(a)), Some(Some(b))) = (
                self.sketch_vertex_px.get(a_index),
                self.sketch_vertex_px.get(b_index),
            ) {
                let hit = if window {
                    rect.contains(*a) || rect.contains(*b)
                } else {
                    segment_touches_rect(*a, *b, rect)
                };
                if hit {
                    picked.push(ui::panel::SelectionTarget::SketchSegment { sketch, entity });
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
                    .windows(2)
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
        if !self.shift_held {
            self.panel_state.selection.clear_sketch_entities();
        }
        for target in picked {
            if !self.panel_state.selection.contains(target) {
                self.panel_state.selection.toggle(target);
            }
        }
    }

    /// ADR 0032: resolve a stationary viewport click into a node selection change, or `None`
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
            recentre_voxels: self.recentre_voxels.voxels(),
            density,
            chunks: &self.resident_chunks,
            band: self.last_pick_band,
        };
        // `pick_voxel` answers in the scene's ABSOLUTE voxel frame, which is exactly the frame
        // `picked_node_at_voxel` reads — no recentre to undo (ADR 0008: the frame is carried).
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
    /// The ray is cast here and nowhere else. It used to run on every `CursorMoved` to keep a
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
    /// — the polyline chain, the rectangle's press corner, the marquee's anchor, the arc's
    /// endpoints. Reports whether anything was actually put back, so the cancel chain can fall
    /// through when there was nothing mid-stroke. The tool stays armed: dropping a stroke is not
    /// the same act as putting the tool down.
    pub(super) fn cancel_sketch_gesture(&mut self) -> bool {
        if self.panel_state.sketch_mode.is_none() {
            return false;
        }
        let live = self.sketch_chain.is_some()
            || self.sketch_rect_anchor.is_some()
            || self.sketch_marquee_anchor.is_some()
            || self.sketch_arc_gesture.is_some();
        self.sketch_chain = None;
        self.sketch_rect_anchor = None;
        self.sketch_marquee_anchor = None;
        self.sketch_arc_gesture = None;
        live
    }

    /// Escape's second sketch rung: put the armed sketch tool down, back to Select — the arrow is
    /// the mode's rest state, the way no-tool-armed is the viewport's. Reports whether a tool was
    /// actually armed, so Escape on the bare Select tool falls through to the rest of the chain
    /// rather than swallowing the key.
    pub(super) fn disarm_sketch_tool(&mut self) -> bool {
        if self.panel_state.sketch_mode.is_none()
            || self.panel_state.sketch_tool == ui::panel::SketchTool::Select
        {
            return false;
        }
        self.panel_state.sketch_tool = ui::panel::SketchTool::Select;
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
    /// own Escape instead of cancelling the running viewport command (and its own Ctrl+Z instead
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
                // group's fine-grained session stacks by themselves (ADR 0028 §4).
                ui::shortcuts::ShortcutCommand::Undo => {
                    effect = effect.merged_with(
                        self.app_core
                            .undo(&mut self.panel_state.scene, &mut self.panel_state.selection),
                    );
                }
                ui::shortcuts::ShortcutCommand::Redo => {
                    effect = effect.merged_with(
                        self.app_core
                            .redo(&mut self.panel_state.scene, &mut self.panel_state.selection),
                    );
                }
                // Dump the scene + LIVE camera to the repro file (`shot --from-config`), so an
                // exact live-view bug reproduces headlessly.
                ui::shortcuts::ShortcutCommand::ExportRepro => self.export_repro(),
                // ADR 0032: Cancel is a priority chain, not one act. An armed orbit-center
                // placement outranks the tool ghost — it is what the cursor is carrying, so it
                // goes back first and leaves any armed tool alone. With nothing to put back it
                // CANCELS the running modal command (the same act the viewport menu's Cancel row
                // performs); with no command running it disarms the tool ghost (ADR 0022).
                // Leaving never writes the DEFAULT orbit type: a session override dies with the
                // mode rather than outliving it.
                //
                // Inside a sketch the chain gains two rungs, innermost first (owner
                // 2026-07-29): a half-drawn polyline / rectangle / arc goes back before
                // anything else the mode is holding, and an armed sketch TOOL falls back to
                // Select before the placement ghost is touched. Escape never leaves sketch
                // mode — that is what the mode's own Cancel button is for.
                ui::shortcuts::ShortcutCommand::CancelCommand => {
                    if !self.cancel_orbit_center_placement()
                        && !self.cancel_sketch_gesture()
                        && !self.end_modal_command(ui::panel::ModeCommand::Cancel)
                        && !self.disarm_sketch_tool()
                    {
                        self.disarm_placement();
                    }
                }
                // The other half of the universal pair. It does nothing when no command is
                // running — Accept is not a general viewport verb.
                ui::shortcuts::ShortcutCommand::AcceptCommand => {
                    self.end_modal_command(ui::panel::ModeCommand::Accept);
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
    /// centre by construction and the reticle is laid out against the rect egui itself just
    /// measured. That also means it cannot lag the camera by a frame the way a cached
    /// projection would.
    ///
    /// It hides while a TURN is in flight — the mark spans most of the frame, and watching the
    /// model come round is exactly when you need it out of the way. A press that has not crossed
    /// the drag threshold keeps it: that press is still a candidate for the re-centring click,
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
    /// This used to gate on a per-move raycast, which made the gizmo both lag and blink out over
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
    /// The point is CONTINUOUS, not a voxel centre: a pivot is a camera quantity with no lattice
    /// meaning, and a snapped one visibly jumps a whole cell at a time under the cursor.
    pub(super) fn surface_point_at(&self, cursor_px: Option<(f64, f64)>) -> Option<glam::Vec3> {
        let (cursor_x, cursor_y) = cursor_px?;
        let density = self.panel_state.geometry.voxels_per_block;
        let [vx, vy, vw, vh] = self.last_viewport_px;
        let recentre = self.recentre_voxels.voxels();
        let frame = crate::PickFrame {
            region_dimensions: self.region_dimensions,
            recentre_voxels: recentre,
            density,
            chunks: &self.resident_chunks,
            band: self.last_pick_band,
        };
        let cursor = [cursor_x as f32, cursor_y as f32];
        let viewport = [vx as f32, vy as f32, vw as f32, vh as f32];
        // Both tiers answer in ABSOLUTE voxels; the camera lives in the RECENTRED render frame,
        // so the point rebases once here (ADR 0008 — the recentre is carried, this is the only
        // conversion).
        let absolute = self.app_core.surface_point_absolute(
            cursor,
            viewport,
            &frame,
            &self.panel_state.scene,
            self.panel_state.scene.master_floor_grid,
        )?;
        Some(absolute - glam::Vec3::new(recentre[0] as f32, recentre[1] as f32, recentre[2] as f32))
    }

    /// ADR 0030/0032: the [`SelectionTarget`](ui::panel::SelectionTarget) under the cursor
    /// (physical px) inside `sketch`, or `None` over empty space. Vertices take priority over
    /// segments, as everywhere. The ONE place a sketch target is minted, which is what makes
    /// the shell's admission `debug_assert` hold by construction.
    fn sketch_entity_target_at(
        &self,
        sketch: document::scene::NodeId,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<ui::panel::SelectionTarget> {
        if let Some(index) = self.sketch_vertex_at(cursor_x, cursor_y) {
            if let Some(&entity) = self.sketch_point_ids.get(index) {
                return Some(ui::panel::SelectionTarget::SketchPoint { sketch, entity });
            }
        }
        self.nearest_sketch_edge(cursor_x, cursor_y)
            .map(|hit| match hit {
                SketchEdgeHit::Segment(entity) => {
                    ui::panel::SelectionTarget::SketchSegment { sketch, entity }
                }
                SketchEdgeHit::Arc(entity) => {
                    ui::panel::SelectionTarget::SketchArc { sketch, entity }
                }
            })
    }

    /// ADR 0030: is the cursor (physical px) over a sketch entity — a vertex or a segment? Used by
    /// the right-click handler to tell a sketch handle (which registers as chrome so a LEFT press
    /// drags it) from the real Signal chrome, so a right-click on an entity opens the context menu
    /// even though the handle sits in the chrome hit-set.
    pub(super) fn cursor_over_sketch_entity(&self, cursor_x: f64, cursor_y: f64) -> bool {
        self.sketch_vertex_at(cursor_x, cursor_y).is_some()
            || self.nearest_sketch_edge(cursor_x, cursor_y).is_some()
    }

    /// ADR 0030: a right-click over a sketch entity selects it (Fusion: right-clicking an entity
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
    /// A multi-selection deletes whole (ADR 0033), filtered to its **selection roots**: a node
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

    /// ADR 0030: delete every entity in the sketch selection as ONE edit — each selected point
    /// (cascading its incident segments and arcs) then each selected segment and arc (a no-op if
    /// a cascade already took it), committed through the same anchor-preserving path a single
    /// delete uses
    /// ([`commit_sketch_profile_edit`](Self::commit_sketch_profile_edit)), then the selection is
    /// cleared. No-op when nothing is picked or no sketch is being edited. Invoked by the general
    /// viewport context menu's Delete.
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
        self.commit_sketch_profile_edit(target, next);
        self.panel_state.selection.clear_sketch_entities();
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
        self.sketch_face_polygons
            .iter()
            .filter(|(_, boundary)| point_in_screen_polygon(boundary, cursor))
            .min_by(|(_, a), (_, b)| {
                polygon_double_area(a)
                    .abs()
                    .total_cmp(&polygon_double_area(b).abs())
            })
            .map(|(key, _)| key.clone())
    }

    /// Whether the region the open viewport menu is acting on is picked (#100), or `None` when the
    /// menu has no region under it — what decides whether the menu offers "carve" or "fill".
    pub(super) fn sketch_menu_face_is_picked(&self) -> Option<bool> {
        let target = self.panel_state.sketch_mode?;
        let key = self.sketch_menu_face.as_ref()?;
        let (producer, _) = self.sketch_node_state(target)?;
        Some(producer.sketch.face_is_picked(key))
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
        let picked = producer.sketch.face_is_picked(&key);
        let next = producer.with_face_picked(key, !picked);
        self.commit_sketch_profile_edit(target, next);
    }

    /// ADR 0028 (#95): queue an add/delete profile edit as ONE entry in the open sketch undo
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
        let new_offset = new_producer.anchor_preserving_offset(&old_producer, old_offset);

        // ONE transaction: an authoring act is one in-mode undo step, and the anchor
        // compensation is part of the act rather than an edit of its own (owner 2026-07-29).
        // Undoing half of it would leave the profile somewhere the author never put it.
        let mut transaction = vec![crate::Intent::SetSketch {
            target,
            producer: new_producer,
        }];
        if new_offset != old_offset {
            transaction.push(crate::Intent::SetOffset {
                target,
                offset_measurements: [
                    voxel_core::units::Measurement::from_voxels(new_offset[0]),
                    voxel_core::units::Measurement::from_voxels(new_offset[1]),
                    voxel_core::units::Measurement::from_voxels(new_offset[2]),
                ],
            });
        }
        self.viewport_transactions.push(transaction);
    }

    /// ADR 0028 (#94): if the cursor (physical px) is over a profile-vertex handle, build the
    /// [`SketchVertexDrag`] that grabs it — the nearest handle within the grab radius, with the
    /// current producer snapshotted so the whole gesture coalesces to one command. `None` when
    /// no handle is under the cursor (the press falls through to the normal camera/placement
    /// path). Called from the `events` press handler, only under the Select tool.
    pub(super) fn begin_sketch_vertex_drag(
        &self,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<SketchVertexDrag> {
        let target = self.panel_state.sketch_mode?;
        let index = self.sketch_vertex_at(cursor_x, cursor_y)?;
        let point_id = *self.sketch_point_ids.get(index)?;
        let node = self.panel_state.scene.node_by_id(target)?;
        let document::scene::NodeContent::SketchTool { producer, .. } = &node.content else {
            return None;
        };
        Some(SketchVertexDrag {
            point_id,
            original: producer.clone(),
            original_offset: node.transform.offset_voxels,
            original_min: Self::profile_bbox_min(producer),
        })
    }

    /// ADR 0028 (#94, extended #95): recompute the sketch overlay for the NEXT frame. Projects
    /// each profile vertex (render frame) to screen, storing the egui-point handles + their
    /// interaction state for drawing, and the physical-pixel centres **in profile order** for the
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
        self.sketch_segments.clear();
        self.sketch_segment_lines.clear();
        self.sketch_arc_lines.clear();
        self.sketch_arc_chords.clear();
        self.sketch_face_polygons.clear();
        self.sketch_face_washes.clear();
        self.sketch_insert_preview = None;
        self.sketch_draw_preview.clear();
        self.sketch_marquee_band = None;

        let Some(target) = self.panel_state.sketch_mode else {
            // #99 / slice 3 / #102: a drawing or marquee gesture dies with the mode.
            self.sketch_chain = None;
            self.sketch_rect_anchor = None;
            self.sketch_marquee_anchor = None;
            self.sketch_arc_gesture = None;
            // #100: so does the region a closed menu was acting on.
            self.sketch_menu_face = None;
            return;
        };
        let Some(handles) = self
            .panel_state
            .scene
            .sketch_handles(target, self.panel_state.geometry.voxels_per_block)
        else {
            return;
        };

        let tool = self.panel_state.sketch_tool;
        // #99: a chain / rectangle anchor belongs to its tool — switching away drops it.
        if tool != ui::panel::SketchTool::Polyline {
            self.sketch_chain = None;
        }
        if tool != ui::panel::SketchTool::Rectangle {
            self.sketch_rect_anchor = None;
        }
        if tool != ui::panel::SketchTool::Select {
            self.sketch_marquee_anchor = None;
        }
        if tool != ui::panel::SketchTool::ThreePointArc {
            self.sketch_arc_gesture = None;
        }
        let [vx, vy, vw, vh] = viewport_px.map(|component| component as f32);
        let dragging_point = self.sketch_drag.as_ref().map(|drag| drag.point_id);
        // A forgiving grab radius (physical px) so a hover reads as "draggable" near the thumb.
        let hover_radius_px = (ui::chrome::SKETCH_HANDLE_HALF + ui::chrome::SKETCH_HANDLE_GRAB_PAD)
            * pixels_per_point;
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
            // edge read alike (ADR 0030).
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
            self.sketch_overlay_points.push((center_pt, state));
            self.sketch_vertex_px.push(Some(center_px));
        }

        // The stable point id + segment connectivity for THIS frame, aligned with
        // `sketch_vertex_px` — the press hit-tests (in `events`) read these to resolve a click to
        // the entity it targets (ADR 0030).
        self.sketch_point_ids = handles.point_ids.clone();
        self.sketch_segments = handles.segments.clone();

        // Arc chord polylines in PHYSICAL px (#102), tessellated for the SCREEN. The resolve's
        // sagitta tolerance is measured in VOXELS, so an arc earns the same handful of chords at
        // every zoom and draws as a visible polygon once a voxel is worth more than a few pixels.
        // The projected radius says what a voxel is currently worth — one number that already
        // carries the zoom, the foreshortening and the plane's tilt — and the tolerance follows
        // from it. Only the pinned resolve tolerance is the profile's MEANING (ADR 0019); this is
        // the same curve, drawn smoothly. A behind-camera chord vertex culls the whole arc,
        // matching the segment rule: a partially-projected curve would fold across the viewport.
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
        for &(arc_id, from, to, sweep) in &handles.arcs {
            let tolerance = document::sketch::arc_center_radius(from, to, sweep)
                .and_then(|(center, radius)| {
                    let radius_px = to_viewport_px(center)?.distance(to_viewport_px(from)?);
                    (radius_px > 1.0).then(|| radius * ARC_SCREEN_SAGITTA_PX / f64::from(radius_px))
                })
                .unwrap_or(document::sketch::ARC_SAGITTA_TOLERANCE_VOXELS)
                .min(document::sketch::ARC_SAGITTA_TOLERANCE_VOXELS);
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

        // The segment under the cursor and the state it should draw in. A vertex under the cursor
        // takes priority — it already answers with its own handle state — so a segment lights up
        // only when no vertex is hit, the SAME decision the vertex-grab makes. Reusing that
        // hit-test keeps the feedback exactly aligned with what a click acts on. Select → Hover
        // (brighter, "you can pick this edge"); Add-point has its own insert diamond, so
        // segments stay Idle.
        let hovered_edge: Option<(SketchEdgeHit, ui::gizmos::HandleState)> = match tool {
            ui::panel::SketchTool::Select => Some(ui::gizmos::HandleState::Hover),
            // Add-point has its own insert diamond; the drawing tools (#99, #102) target
            // points and empty plane, never an edge.
            ui::panel::SketchTool::AddPoint
            | ui::panel::SketchTool::Polyline
            | ui::panel::SketchTool::Rectangle
            | ui::panel::SketchTool::ThreePointArc => None,
        }
        .and_then(|state| {
            self.last_cursor_position.and_then(|(cx, cy)| {
                if self.sketch_vertex_at(cx, cy).is_some() {
                    None
                } else {
                    self.nearest_sketch_edge(cx, cy).map(|hit| (hit, state))
                }
            })
        });

        // The segment LINES to draw next frame: each committed edge between its two projected
        // endpoints, in egui points (ADR 0030 — an open sketch resolves to nothing, so the edges
        // are the only thing that shows the profile is connected). A behind-camera endpoint
        // (`None` in `sketch_vertex_px`) culls its line, matching the vertex-dot cull. The one
        // hovered segment carries its Hover/Marked state; the rest are Idle.
        for &(seg_id, a_idx, b_idx) in &self.sketch_segments {
            if let (Some(Some(a_px)), Some(Some(b_px))) = (
                self.sketch_vertex_px.get(a_idx),
                self.sketch_vertex_px.get(b_idx),
            ) {
                let a = egui::Pos2::new(a_px.x / pixels_per_point, a_px.y / pixels_per_point);
                let b = egui::Pos2::new(b_px.x / pixels_per_point, b_px.y / pixels_per_point);
                // Precedence: Selected > plain Hover > Idle. A selected edge stays bold even
                // under the cursor (Select hover never shrinks it).
                let picked = ui::panel::SelectionTarget::SketchSegment {
                    sketch: target,
                    entity: seg_id,
                };
                let selected = self.panel_state.selection.contains(picked);
                let state = match hovered_edge {
                    _ if selected => ui::gizmos::HandleState::Selected,
                    Some((SketchEdgeHit::Segment(id), state)) if id == seg_id => state,
                    _ => ui::gizmos::HandleState::Idle,
                };
                self.sketch_segment_lines.push((a, b, state));
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
            let curve = chords
                .iter()
                .map(|px| egui::Pos2::new(px.x / pixels_per_point, px.y / pixels_per_point))
                .collect();
            self.sketch_arc_lines.push((curve, state));
        }

        // The derived regions (#100). Derivation is a graph walk over the sketch's own entities, so
        // it re-runs here with the rest of the overlay rather than being cached against an edit
        // counter. Two consumers, two shapes: every FACE in physical px for the right-press
        // hit-test, and the resolved MATERIAL pieces for the wash. They are not the same list — a
        // face nested in another face is its own pick target but no extra material, and washing
        // both would composite the same place twice.
        let sketch = self
            .panel_state
            .scene
            .node_by_id(target)
            .and_then(|node| match &node.content {
                document::scene::NodeContent::SketchTool { producer, .. } => Some(&producer.sketch),
                _ => None,
            })
            .cloned();
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
        let to_points = |projected: &[egui::Pos2]| -> Vec<egui::Pos2> {
            projected
                .iter()
                .map(|point| {
                    egui::Pos2::new(point.x / pixels_per_point, point.y / pixels_per_point)
                })
                .collect()
        };
        if let Some(sketch) = sketch {
            for face in sketch.faces() {
                if let Some(projected) = project(&face.boundary) {
                    self.sketch_face_polygons.push((face.key, projected));
                }
            }
            for piece in sketch.material_components() {
                let Some(outer) = project(&piece.outer) else {
                    continue;
                };
                let holes = piece
                    .holes
                    .iter()
                    .filter_map(|hole| project(hole).map(|hole| to_points(&hole)))
                    .collect();
                self.sketch_face_washes.push((to_points(&outer), holes));
            }
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
        match tool {
            ui::panel::SketchTool::Polyline => {
                // Rubber line: the chain's live end to the snapped cursor.
                if let (Some((_, last)), Some((cursor_x, cursor_y))) =
                    (self.sketch_chain, self.last_cursor_position)
                {
                    let chain_end = self
                        .sketch_point_ids
                        .iter()
                        .position(|&id| id == last)
                        .and_then(|idx| self.sketch_vertex_px.get(idx).copied().flatten())
                        .map(|px| {
                            egui::Pos2::new(px.x / pixels_per_point, px.y / pixels_per_point)
                        });
                    let cursor = self
                        .sketch_snapped_point_at(cursor_x, cursor_y)
                        .and_then(|point| snapped_screen(point.in_plane()));
                    if let (Some(a), Some(b)) = (chain_end, cursor) {
                        self.sketch_draw_preview.extend([a, b]);
                    }
                }
            }
            ui::panel::SketchTool::Rectangle => {
                // The four edges the release will commit, closed back to the anchor.
                if let (Some(anchor), Some((cursor_x, cursor_y))) =
                    (self.sketch_rect_anchor, self.last_cursor_position)
                {
                    if let Some(corner) = self.sketch_snapped_point_at(cursor_x, cursor_y) {
                        let (a, c) = (anchor.in_plane(), corner.in_plane());
                        let ring = [a, [c[0], a[1]], c, [a[0], c[1]], a];
                        let projected: Vec<egui::Pos2> =
                            ring.iter().copied().filter_map(snapped_screen).collect();
                        // A behind-camera corner culls the whole preview rather than
                        // drawing a broken ring.
                        if projected.len() == ring.len() {
                            self.sketch_draw_preview = projected;
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
                        let rect = egui::Rect::from_two_pos(
                            egui::Pos2::new(
                                down_x as f32 / pixels_per_point,
                                down_y as f32 / pixels_per_point,
                            ),
                            egui::Pos2::new(
                                cursor_x as f32 / pixels_per_point,
                                cursor_y as f32 / pixels_per_point,
                            ),
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
                            self.sketch_draw_preview = projected;
                        }
                    }
                }
            }
            ui::panel::SketchTool::AddPoint => {}
        }
    }
}

/// Which kind of sketch EDGE a cursor resolved to (#102) — the two entity stores share an id
/// space but not a vector, so the kind travels with the id rather than being re-derived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SketchEdgeHit {
    /// A straight segment.
    Segment(document::sketch::EntityId),
    /// An arc.
    Arc(document::sketch::EntityId),
}

/// Quantize a continuous in-plane profile coordinate by the sketch position snap (#96):
/// `NoSnap` carries the sub-voxel fraction on the point (#101), `Voxel` rounds to the plane's
/// own voxel grid, `Block` rounds to block boundaries. Every sketch vertex edit — drag,
/// add-point split, polyline, rectangle — resolves through this one policy.
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

#[cfg(test)]
mod tests {
    use super::{
        apply_sketch_snap, closest_point_on_segment, point_in_screen_polygon,
        point_to_segment_distance, polygon_double_area, segment_touches_rect, segments_intersect,
    };
    use egui::{pos2, Rect};
    use ui::panel::PositionSnap;

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
}
