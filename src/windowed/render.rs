//! The shell's per-frame render seam: acquire the surface texture, poll the display/measurement
//! workers, run the egui frame, apply this frame's Intents + view actions, upload every
//! renderer's uniforms, and submit the shared [`render_frame`]. Split out of `windowed/mod.rs`
//! (ADR 0016).

use super::*;

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
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded => {
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
        let current_band = (self.panel_state.layer_range.lower, self.panel_state.layer_range.upper);
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
                // Signal (#86): the hovered cube zone's readout name, or None.
                self.hovered_cube_zone
                    .and_then(camera::view_cube_zone_readout)
                    .as_deref(),
                // The armed primitive's kind → the "Add <shape>" dialog (owner 2026-07-21).
                self.armed_tool.as_ref().and_then(|spec| match spec {
                    document::intent::NodeSpec::Tool { shape, .. } => Some(shape.kind),
                    _ => None,
                }),
                // ADR 0028 (#94): the sketch vertex handles, projected LAST frame (the
                // viewport + camera the projection needs are only known after this call).
                // A one-frame lag is imperceptible for handle chrome and self-corrects; the
                // cache is refreshed at the end of `render` below.
                &self.sketch_overlay_points,
                // ADR 0030: the committed segment lines, projected last frame — drawn under the
                // vertex dots so the profile reads as connected edges.
                &self.sketch_segment_lines,
                // ADR 0028 (#95): the add-point insert preview, projected last frame.
                self.sketch_insert_preview,
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
            self.run_chrome_action(action);
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
                let (target, distance) = OrbitCamera::focus_target_and_distance(
                    glam::Vec3::from_array(pivot),
                    extent,
                );
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
        //                           highlight are recomputed every frame below from
        //                           `scene.active`, so they already track selection —
        //                           a pure `SelectNode` must NOT force a re-resolve).
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
        if let Some(spec) = prepared.panel_response.armed_tool.take() {
            self.armed_tool = Some(spec);
        }
        // ADR 0028: enter / leave sketch mode — a VIEW action on the response (entering a mode
        // mutates no document state), like `armed_tool`. Entering scopes the mode to the
        // requested node, disarms any placement tool (non-sketch ops withdraw in the mode), and
        // OPENS the undo group (§4). Finish commits the session as one main-history entry;
        // Cancel rolls it back to the enter-state (which re-resolves) — both drop the mode. The
        // group-close effect folds into `merged_effect` below so a Cancel rebuilds like an edit.
        let mut sketch_effect = crate::IntentEffect::none();
        if let Some(node) = prepared.panel_response.enter_sketch.take() {
            self.panel_state.sketch_mode = Some(node);
            self.armed_tool = None;
            self.panel_state.sketch_selection.clear();
            self.app_core.begin_sketch_group();
        }
        if let Some(exit) = prepared.panel_response.exit_sketch.take() {
            sketch_effect = match exit {
                ui::panel::SketchExit::Finish => self.app_core.finish_sketch_group(),
                ui::panel::SketchExit::Cancel => {
                    self.app_core.cancel_sketch_group(&mut self.panel_state.scene)
                }
            };
            self.panel_state.sketch_mode = None;
            self.panel_state.sketch_selection.clear();
        }
        // ADR 0030: a context-menu Delete in sketch mode removes the current selection as one edit
        // (queues a SetSketch through `viewport_intents`, gathered just below).
        if prepared.panel_response.delete_sketch_selection {
            self.delete_sketch_selection();
        }
        // ADR 0028 (#94): advance an in-progress sketch vertex drag — a live preview that
        // re-resolves the volume and records ONE coalesced command in the open group. Uses
        // this frame's viewport (from `prepared`) to build the cursor→plane ray; its effect
        // folds into `merged_effect` below so the display re-resolves like any other edit.
        let drag_effect = {
            let [_, _, drag_vw, drag_vh] = prepared.viewport_px;
            let drag_aspect = drag_vw as f32 / drag_vh.max(1) as f32;
            let drag_view_projection =
                self.app_core.view_projection(drag_aspect, self.region_dimensions);
            self.update_sketch_vertex_drag(drag_view_projection, prepared.viewport_px)
        };
        let mut intents = std::mem::take(&mut prepared.panel_response.intents);
        // ADR 0022 live placement: a viewport click's drop intent is applied through the
        // SAME door as the panel's edits (taken BEFORE the borrow of `prepared` ends), so
        // a placement re-resolves + rebuilds identically to a panel-driven add.
        intents.extend(std::mem::take(&mut self.viewport_intents));
        let mut merged_effect = sketch_effect.merged_with(drag_effect);
        for intent in intents {
            let effect = self
                .app_core
                .apply_intent(&mut self.panel_state.scene, intent);
            merged_effect = merged_effect.merged_with(effect);
        }
        if merged_effect.selection_changed || merged_effect.scene_changed {
            // Re-sync the inspector mirror to the active node. The OLD panel called
            // `sync_mirror_from_active` after EVERY structural action (add / group /
            // make-definition / add-instance / delete — each of which changes the
            // active node) AND on a row select; we reproduce that by syncing on a
            // `selection_changed` (a pure `SelectNode`) OR a `scene_changed` (a
            // structural edit may have moved the active selection to a freshly-added /
            // re-derived node). Syncing after an inspector `SetShape`/`SetDensity` is a
            // harmless no-op (the node now equals the buffer it was written from). The
            // transform gizmo + row highlight read `scene.active` live each frame, so a
            // pure `SelectNode` updates them WITHOUT a re-resolve (the efficiency win).
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
        // geometry / MODE change ONLY (never per frame). A `SelectNode` marks it dirty
        // without a scene re-resolve; the derivation is bounded by the ghosted operands'
        // covering chunks (`AppCore::boolean_operand_ghost`), so this stays cheap even in
        // a huge scene. The `active` / mode comparisons are belt-and-braces for any
        // selection or mode writer that bypassed the Intent effects. The ghost is
        // populated only in Show-booleans mode; Normal / Onion-fog derive nothing.
        if merged_effect.selection_changed || merged_effect.scene_changed {
            self.selected_ghost_dirty = true;
        }
        if self.selected_ghost_dirty
            || self.selected_ghost_selection != self.panel_state.scene.active
            || self.selected_ghost_view_mode != self.panel_state.view_mode
        {
            self.selected_ghost_dirty = false;
            self.selected_ghost_selection = self.panel_state.scene.active;
            self.selected_ghost_view_mode = self.panel_state.view_mode;
            let ghost = (self.panel_state.view_mode == crate::ViewMode::ShowBooleans)
                .then(|| {
                    AppCore::boolean_operand_ghost(
                        &self.panel_state.scene,
                        self.panel_state.geometry.voxels_per_block,
                    )
                })
                .flatten();
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
        let view_projection = self.app_core.view_projection(aspect_ratio, grid_dimensions);
        // ADR 0028 (#94): refresh the sketch vertex-handle overlay from the CURRENT geometry
        // (post-rebuild recentre) and camera, caching the projected handles for NEXT frame's
        // draw (in `run_egui_frame`) and the press hit-test (in `events`). A one-frame lag on
        // the handles is imperceptible and self-corrects.
        self.refresh_sketch_overlay(view_projection, prepared.viewport_px, pixels_per_point);
        // #95: cache the projection so the release handler (in `events`) can invert a cursor
        // into a profile coordinate for an add-point insert, using the SAME frame the overlay saw.
        self.last_view_projection = Some(view_projection);
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
        self.display.cuboid_mesh_renderer_mut().update_uniforms(
            &self.gpu.device,
            &self.gpu.queue,
            view_projection,
            grid_dimensions,
            geometry.voxels_per_block,
            // Issue #29 S4: the on-face-grid MASTER (Display checkbox →
            // `scene.master_voxel_grid`). The shader ANDs it with each voxel's
            // per-object flag bit packed into `material_id`.
            self.panel_state.scene.master_voxel_grid,
            bound,
            band,
            clip.region,
            self.panel_state.debug_face_orientation,
        );
        // ADR 0011 G1: the brick raymarch takes THIS frame's voxel-model draw when a
        // field is installed and no mesh-only display mode is active — debug-faces
        // and a loaded VS material are per-frame toggles that never rebuild geometry,
        // so the draw decision is per-frame (the field stays installed). Its uniforms
        // mirror the cuboid upload above (camera, viewport, band, overlay master,
        // bound material) so the two paths render pixel-comparable.
        // Shared engagement gate (term-identical to `ensure_display_mesh_current`): a live brick
        // field AND no mesh-only mode. When engaged, upload the raymarch uniforms (mirroring the
        // cuboid upload above) so the brick draw replaces the mesh draw this frame.
        // ADR 0012 (H1): the onion GHOST replaces the volumetric fog. Active when onion
        // skin is on and the band is a real slab (`current_layer_band` sets a non-zero
        // `onion_depth` exactly then; debug-face mode forces FULL → 0). The engaged display
        // path draws the ghost after its solid pass (`render_frame`); a band scrub is a pure
        // uniform update on the brick path, a thin-slab re-mesh on the cuboid path — never
        // the fog atlas rebuild.
        let onion_ghost_active = band.onion_depth > 0;
        let brick_raymarch_engaged = if self
            .display
            .brick_display_engaged(self.panel_state.debug_face_orientation)
        {
            let has_loaded_material = self.loaded_material.is_some();
            let renderer = self
                .display
                .brick_raymarch_renderer_mut()
                .expect("brick_display_engaged ⇒ renderer holds a live field");
            // ADR 0011 G2: mirror the applied-block state into the shader so solid hits
            // shade from the block's D2Array (its bind group is passed to `draw`).
            renderer.set_loaded_material_active(has_loaded_material);
            // Grazing-rim diagnostic (Display → "Debug: brick faces"): face-axis colour +
            // UV checkerboard in place of the material shade. Per-frame toggle, no rebuild.
            renderer.set_debug_mode(u32::from(self.panel_state.debug_brick_faces));
            renderer.update_uniforms(
                &self.gpu.queue,
                view_projection,
                prepared.viewport_px,
                grid_dimensions,
                band,
                // ADR 0018 Decision 5 (S5): the region-scoped clip on the brick path too — the
                // band bites only inside the selected object's AABB (`None` ⇒ whole scene).
                clip.region,
                self.panel_state.scene.master_voxel_grid,
                bound,
            );
            // Prepare the two onion ghost slab uniforms (slots 1 + 2). Self-gates on
            // `band.onion_depth == 0`, so this is a cheap no-op when onion is off.
            renderer.update_ghost_uniforms(
                &self.gpu.queue,
                view_projection,
                prepared.viewport_px,
                grid_dimensions,
                band,
                clip.region,
            );
            true
        } else {
            false
        };
        // Transform gizmo (issue #29 S2): it FOLLOWS the selected node. Size it to
        // the selected node's own extent and bake its recentred pivot into the
        // camera matrix. `None` (nothing selected, or selection has no extent) hides
        // it — visibility is selection-driven, no longer a Display toggle.
        let gizmo_placement = AppCore::gizmo_placement(
            &self.panel_state.scene,
            self.panel_state.geometry.voxels_per_block,
        );
        if let Some((pivot, _extent)) = gizmo_placement {
            let pivot = glam::Vec3::from_array(pivot);
            let fraction = display::renderer::GIZMO_SCREEN_FRACTION;
            let model = self.app_core.camera.screen_stable_model(pivot, fraction);
            // The gizmo draws depth-OFF with a generous overlay near/far (the scene's tight
            // window, sized to the model, would clip the screen-stable gizmo when zoomed far).
            let gizmo_vp = self
                .app_core
                .camera
                .overlay_view_projection(aspect_ratio, pivot);
            self.transform_gizmo_renderer
                .update_uniforms(&self.gpu.queue, gizmo_vp, model);
        }
        // Per-object block lattice + floor grid (issue #29 S3): rebuild this frame's
        // line batch from the scene — for every node whose grids are enabled (the
        // scene master ANDed with the node's own toggle), its enclosing-block lattice
        // / base-plane floor lines. Empty when no node enables a grid (the new
        // default — per-object grids are OFF until the user turns them on).
        self.scene_grid_renderer.rebuild_from_scene(
            &self.gpu.device,
            &self.gpu.queue,
            &self.panel_state.scene,
            self.panel_state.geometry.voxels_per_block,
        );
        self.scene_grid_renderer
            .update_uniforms(&self.gpu.queue, view_projection);
        // World reference AXES (issue #29 S5): rebuild the visible Points' axis lines,
        // screen-stable-sized from the camera (ADR 0031). On-top (depth off) reads as a nav
        // marker through the model with a generous overlay near/far so the screen-stable axes
        // never clip; occluded uses the scene matrix + the depth-tested scaffold phase.
        let axes_on_top = self.panel_state.axes_on_top;
        self.points_renderer.rebuild_from_scene(
            &self.gpu.device,
            &self.gpu.queue,
            &self.panel_state.scene,
            self.panel_state.geometry.voxels_per_block,
            &self.app_core.camera,
            axes_on_top,
        );
        let points_vp = if axes_on_top {
            self.app_core
                .camera
                .overlay_view_projection(aspect_ratio, glam::Vec3::ZERO)
        } else {
            view_projection
        };
        self.points_renderer
            .update_uniforms(&self.gpu.queue, points_vp);
        // Analytic infinite reference grid (issue #29 Points fast-follow): rebuild the
        // visible Points' enabled PLANES with the camera matrices (recentred frame) so
        // the fullscreen ray-plane shader intersects each pixel's ray with the plane —
        // the grid extends to the horizon with no finite edge, fading with distance.
        self.infinite_grid_renderer.rebuild_from_scene(
            &self.gpu.queue,
            &self.panel_state.scene,
            self.panel_state.geometry.voxels_per_block,
            view_projection,
            self.app_core.camera.eye().to_array(),
        );
        // ADR 0022 live placement: while a tool is armed and the cursor is over the
        // viewport, resolve where it would drop (via the headless `place_primitive`) and
        // arm the ghost + the pending click intent. Anything else — nothing armed, a
        // non-Tool spec, or no cursor — clears both, so a stale preview never lingers.
        // NO resident-geometry guard: `place_primitive`'s tier 1 (`pick_voxel`) returns
        // `None` on an empty scene and falls through to the world-plane tier, which needs
        // no chunks — so the ghost must preview on an empty scene (the ground plane), not
        // only once something is built. This runs before the ghost's uniform upload below,
        // which reads `self.panel_state.placement_ghost`.
        match (&self.armed_tool, self.last_cursor_position) {
            (Some(NodeSpec::Tool { shape, material }), Some((cursor_x, cursor_y))) => {
                let shape = shape.clone();
                let material = *material;
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
                let outcome =
                    self.app_core
                        .place_primitive(
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
                self.panel_state.placement_ghost = match &outcome.intent {
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
            }
            _ => {
                self.pending_placement = None;
                self.panel_state.placement_ghost = None;
            }
        }
        // ADR 0022: the armed-tool placement ghost. Arm it from `PanelState::placement_ghost`
        // (populated live above, or from a loaded config F9 repro), resolving the
        // render-frame field centre from THIS rebuild's recentre so the ghost sits in the
        // exact frame the solid voxels are drawn in (ADR 0008). Disarmed → the pass is a no-op.
        if let Some(ghost) = &self.panel_state.placement_ghost {
            let voxels_per_block = self.panel_state.geometry.voxels_per_block;
            let recentre = self.recentre_voxels.voxels();
            self.placement_ghost_renderer.update_uniforms(
                &self.gpu.queue,
                view_projection,
                view_projection.inverse(),
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
        // ADR 0018 Decision 6: the boolean-operand ghost's per-frame camera + tint upload
        // (the meshes were derived at the selection/geometry/mode seam above, never here).
        self.selected_operand_ghost_renderer
            .update_uniforms(&self.gpu.queue, view_projection);
        self.view_cube_renderer
            .update_uniforms(&self.gpu.queue, self.app_core.camera.view_cube_view_projection());

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
        // ADR 0018 Decision 6: the operand x-ray, suppressed in debug-faces mode; self-gates
        // on an empty ghost otherwise.
        if !self.panel_state.debug_face_orientation {
            over_model.push(&self.selected_operand_ghost_renderer);
        }
        // ADR 0022: the armed-tool placement ghost self-gates on being armed.
        if self.panel_state.placement_ghost.is_some() {
            over_model.push(&self.placement_ghost_renderer);
        }
        // Scaffold: per-object grids + the analytic infinite grid (Points' planes) — each
        // self-gates. The Points' axes join the scaffold only when NOT on-top (occluded).
        let mut scaffold: Vec<&dyn display::SceneDraw> =
            vec![&self.scene_grid_renderer, &self.infinite_grid_renderer];
        if !axes_on_top {
            scaffold.push(&self.points_renderer);
        }
        // On-top: the Points' axes (when on-top, the default) then the manipulator gizmo.
        let mut on_top: Vec<&dyn display::SceneDraw> = Vec::new();
        if axes_on_top {
            on_top.push(&self.points_renderer);
        }
        if gizmo_placement.is_some() {
            on_top.push(&self.transform_gizmo_renderer);
        }
        let phases = FramePhases {
            background: &background,
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
            view_cube: if self.panel_state.show_view_cube {
                Some(&self.view_cube_renderer)
            } else {
                None
            },
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
        view_projection: glam::Mat4,
        viewport_px: [u32; 4],
    ) -> crate::IntentEffect {
        self.preview_sketch_vertex_drag(view_projection, viewport_px)
    }

    /// The live-preview half: project the cursor onto the sketch plane, grid-snap the profile
    /// coordinate (grid density = voxel density ⇒ round to the nearest whole voxel), compensate
    /// the node offset by the bbox-min shift so the NON-dragged vertices stay put in world (the
    /// producer re-anchors its bbox-min to the node origin, so without this the grabbed
    /// min-vertex would pin and the rest would lurch), and direct-mutate the node for a LIVE
    /// re-resolve — no command recorded. `none` when nothing changed this frame.
    fn preview_sketch_vertex_drag(
        &mut self,
        view_projection: glam::Mat4,
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

        // Cursor → the continuous profile coordinate under it, then grid-snap (round to the
        // nearest voxel). The ray/plane math is shared with the add-point insert.
        let Some(profile_coord) =
            self.cursor_to_profile_coord(cursor_x, cursor_y, view_projection, viewport_px, &handles)
        else {
            return IntentEffect::none();
        };
        let snapped = [profile_coord[0].round() as i64, profile_coord[1].round() as i64];

        // Build the preview from the pre-drag producer with ONLY the dragged vertex moved, then
        // compensate the offset by the bbox-min shift so the rest of the profile holds still.
        let Some(drag) = self.sketch_drag.as_ref() else {
            return IntentEffect::none();
        };
        let mut preview = drag.original.clone();
        // Mutate the grabbed point ENTITY directly by its stable id (ADR 0030 — no loop index).
        let Some(point) = preview.sketch.point_position_mut(point_id) else {
            self.sketch_drag = None;
            return IntentEffect::none();
        };
        point.offset_voxels = snapped;
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
        // loop, the same door as any placement drop) so it lands in the open group. The
        // `SetOffset` is emitted only when the anchor compensation actually moved the node.
        self.viewport_intents.push(crate::Intent::SetSketch {
            target,
            producer: final_producer,
        });
        if final_offset != drag.original_offset {
            self.viewport_intents.push(crate::Intent::SetOffset {
                target,
                offset_measurements: [
                    voxel_core::units::Measurement::from_voxels(final_offset[0]),
                    voxel_core::units::Measurement::from_voxels(final_offset[1]),
                    voxel_core::units::Measurement::from_voxels(final_offset[2]),
                ],
            });
        }
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
        let document::scene::NodeContent::SketchTool { producer: current, .. } = &node.content
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
    fn cursor_to_profile_coord(
        &self,
        cursor_x: f64,
        cursor_y: f64,
        view_projection: glam::Mat4,
        viewport_px: [u32; 4],
        handles: &document::scene::SketchHandles,
    ) -> Option<[f64; 2]> {
        let [vx, vy, vw, vh] = viewport_px;
        let ndc_x = (cursor_x as f32 - vx as f32) / vw.max(1) as f32 * 2.0 - 1.0;
        let ndc_y = 1.0 - (cursor_y as f32 - vy as f32) / vh.max(1) as f32 * 2.0;
        let ray = camera::unproject_screen_point_to_ray(view_projection, ndc_x, ndc_y)?;
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
    /// (#94) and the delete hit-test (#95).
    fn sketch_vertex_at(&self, cursor_x: f64, cursor_y: f64) -> Option<usize> {
        let grab_px = (ui::chrome::SKETCH_HANDLE_HALF
            + ui::chrome::SKETCH_HANDLE_GRAB_PAD)
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
    fn nearest_sketch_segment(
        &self,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<(document::sketch::EntityId, egui::Pos2, egui::Pos2)> {
        let pad_px = ui::chrome::SKETCH_SEGMENT_GRAB_PAD * self.window.scale_factor() as f32;
        let cursor = egui::Pos2::new(cursor_x as f32, cursor_y as f32);
        let mut nearest: Option<(document::sketch::EntityId, egui::Pos2, egui::Pos2, f32)> = None;
        for &(seg_id, a_idx, b_idx) in &self.sketch_segments {
            let (Some(&Some(a)), Some(&Some(b))) =
                (self.sketch_vertex_px.get(a_idx), self.sketch_vertex_px.get(b_idx))
            else {
                continue;
            };
            let distance = point_to_segment_distance(cursor, a, b);
            if distance <= pad_px && nearest.map(|(_, _, _, best)| distance < best).unwrap_or(true) {
                nearest = Some((seg_id, a, b, distance));
            }
        }
        nearest.map(|(seg_id, a, b, _)| (seg_id, a, b))
    }

    /// The id of the sketch SEGMENT under the cursor (physical px), for add-point — the click
    /// splits the named segment (ADR 0030). `None` when no edge is close enough.
    fn sketch_segment_at(&self, cursor_x: f64, cursor_y: f64) -> Option<document::sketch::EntityId> {
        self.nearest_sketch_segment(cursor_x, cursor_y).map(|(seg_id, _, _)| seg_id)
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
            self.last_view_projection?,
            self.last_viewport_px,
            &handles,
        )?;
        let (producer, _) = self.sketch_node_state(target)?;
        // Split the segment under the cursor with a grid-snapped point (ADR 0030).
        let point = document::sketch::SketchPoint::new(coord[0].round() as i64, coord[1].round() as i64);
        Some(producer.with_point_on_segment(seg_id, point))
    }

    /// ADR 0030: the delete producer for a click at the cursor (physical px). A POINT under the
    /// cursor is deleted, cascading its incident segments (delete a point → remove its edges and
    /// nothing else); otherwise a SEGMENT under the cursor is deleted on its own, its endpoints
    /// left as free points (delete a line → remove only the line). `None` when neither is under
    /// the cursor or `target` is not an enabled sketch node. The caller routes the returned
    /// producer through [`commit_sketch_profile_edit`](Self::commit_sketch_profile_edit).
    pub(super) fn sketch_delete_at(
        &self,
        cursor_x: f64,
        cursor_y: f64,
    ) -> Option<document::sketch::SketchSolid> {
        let target = self.panel_state.sketch_mode?;
        let (producer, _) = self.sketch_node_state(target)?;
        // Prefer a vertex hit (delete the point + its segments); fall back to a segment hit
        // (delete just that line). ADR 0030 — delete any entity, the clicked one only.
        if let Some(index) = self.sketch_vertex_at(cursor_x, cursor_y) {
            if let Some(&point_id) = self.sketch_point_ids.get(index) {
                return Some(producer.with_point_deleted(point_id));
            }
        }
        let seg_id = self.sketch_segment_at(cursor_x, cursor_y)?;
        Some(producer.with_segment_deleted(seg_id))
    }

    /// ADR 0030: resolve a stationary Select-tool click into the sketch selection. A vertex under
    /// the cursor takes priority (it already answers as a handle), then a segment, else empty
    /// space. Plain click **replaces** the selection with that one entity; `shift` **toggles** it
    /// in/out (accumulate). A plain click on empty space **clears**; a Shift-click on empty space
    /// keeps the selection (Fusion). Reuses the same hit-tests the drag and delete run, so what you
    /// click is what you pick. Pure selection-state mutation — records no document edit.
    pub(super) fn resolve_sketch_selection_click(&mut self, cursor_x: f64, cursor_y: f64) {
        let shift = self.shift_held;
        if let Some(index) = self.sketch_vertex_at(cursor_x, cursor_y) {
            if let Some(&point_id) = self.sketch_point_ids.get(index) {
                if shift {
                    self.panel_state.sketch_selection.toggle_point(point_id);
                } else {
                    self.panel_state.sketch_selection.select_point(point_id);
                }
                return;
            }
        }
        if let Some(seg_id) = self.sketch_segment_at(cursor_x, cursor_y) {
            if shift {
                self.panel_state.sketch_selection.toggle_segment(seg_id);
            } else {
                self.panel_state.sketch_selection.select_segment(seg_id);
            }
            return;
        }
        // Empty space: a plain click clears; a Shift-click leaves the set alone (Fusion).
        if !shift {
            self.panel_state.sketch_selection.clear();
        }
    }

    /// ADR 0030: is the cursor (physical px) over a sketch entity — a vertex or a segment? Used by
    /// the right-click handler to tell a sketch handle (which registers as chrome so a LEFT press
    /// drags it) from the real Signal chrome, so a right-click on an entity opens the context menu
    /// even though the handle sits in the chrome hit-set.
    pub(super) fn cursor_over_sketch_entity(&self, cursor_x: f64, cursor_y: f64) -> bool {
        self.sketch_vertex_at(cursor_x, cursor_y).is_some()
            || self.sketch_segment_at(cursor_x, cursor_y).is_some()
    }

    /// ADR 0030: a right-click over a sketch entity selects it (Fusion: right-clicking an entity
    /// acts on it). If the entity is already in the selection the whole set is kept — so
    /// right-clicking one of several selected entities deletes them all — otherwise the selection is
    /// replaced with just that entity. Vertices take priority over segments, as everywhere.
    pub(super) fn right_click_select_sketch_entity(&mut self, cursor_x: f64, cursor_y: f64) {
        if let Some(index) = self.sketch_vertex_at(cursor_x, cursor_y) {
            if let Some(&point_id) = self.sketch_point_ids.get(index) {
                if !self.panel_state.sketch_selection.contains_point(point_id) {
                    self.panel_state.sketch_selection.select_point(point_id);
                }
                return;
            }
        }
        if let Some(seg_id) = self.sketch_segment_at(cursor_x, cursor_y) {
            if !self.panel_state.sketch_selection.contains_segment(seg_id) {
                self.panel_state.sketch_selection.select_segment(seg_id);
            }
        }
    }

    /// ADR 0030: delete every entity in the sketch selection as ONE edit — each selected point
    /// (cascading its incident segments) then each selected segment (a no-op if a cascade already
    /// took it), committed through the same anchor-preserving path a single delete uses
    /// ([`commit_sketch_profile_edit`](Self::commit_sketch_profile_edit)), then the selection is
    /// cleared. No-op when nothing is picked or no sketch is being edited. Invoked by the general
    /// viewport context menu's Delete.
    pub(super) fn delete_sketch_selection(&mut self) {
        let Some(target) = self.panel_state.sketch_mode else {
            return;
        };
        if self.panel_state.sketch_selection.is_empty() {
            return;
        }
        let Some((producer, _)) = self.sketch_node_state(target) else {
            return;
        };
        let points: Vec<_> = self.panel_state.sketch_selection.points().collect();
        let segments: Vec<_> = self.panel_state.sketch_selection.segments().collect();
        let mut next = producer;
        for point_id in points {
            next = next.with_point_deleted(point_id);
        }
        for seg_id in segments {
            next = next.with_segment_deleted(seg_id);
        }
        self.commit_sketch_profile_edit(target, next);
        self.panel_state.sketch_selection.clear();
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

        self.viewport_intents.push(crate::Intent::SetSketch {
            target,
            producer: new_producer,
        });
        if new_offset != old_offset {
            self.viewport_intents.push(crate::Intent::SetOffset {
                target,
                offset_measurements: [
                    voxel_core::units::Measurement::from_voxels(new_offset[0]),
                    voxel_core::units::Measurement::from_voxels(new_offset[1]),
                    voxel_core::units::Measurement::from_voxels(new_offset[2]),
                ],
            });
        }
    }

    /// ADR 0028 (#94): if the cursor (physical px) is over a profile-vertex handle, build the
    /// [`SketchVertexDrag`] that grabs it — the nearest handle within the grab radius, with the
    /// current producer snapshotted so the whole gesture coalesces to one command. `None` when
    /// no handle is under the cursor (the press falls through to the normal camera/placement
    /// path). Called from the `events` press handler, only under the Select tool.
    pub(super) fn begin_sketch_vertex_drag(&self, cursor_x: f64, cursor_y: f64) -> Option<SketchVertexDrag> {
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
    /// segments can pair adjacent vertices). Also derives the delete-hover **Marked** state and
    /// the add-point **insert-preview** marker from the armed tool. Clears everything outside
    /// sketch mode.
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
        self.sketch_insert_preview = None;

        let Some(target) = self.panel_state.sketch_mode else {
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
        let [vx, vy, vw, vh] = viewport_px.map(|component| component as f32);
        let dragging_point = self.sketch_drag.as_ref().map(|drag| drag.point_id);
        // A forgiving grab radius (physical px) so a hover reads as "draggable" near the thumb.
        let hover_radius_px =
            (ui::chrome::SKETCH_HANDLE_HALF + ui::chrome::SKETCH_HANDLE_GRAB_PAD)
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
                .map(|(cx, cy)| {
                    (cx as f32 - px).hypot(cy as f32 - py) <= hover_radius_px
                })
                .unwrap_or(false);
            let point_id = handles.point_ids.get(index).copied();
            let selected = point_id
                .map(|id| self.panel_state.sketch_selection.contains_point(id))
                .unwrap_or(false);
            // Precedence: dragged > delete-hover (warn ✕) > selected > hover > idle. A selected
            // vertex stays filled-accent even under the cursor (only the destructive delete-hover
            // overrides it), matching the segment rule so a point and an edge read alike (ADR 0030).
            let state = if dragging_point == point_id {
                ui::gizmos::HandleState::Snapped
            } else if hovered && tool == ui::panel::SketchTool::Delete {
                ui::gizmos::HandleState::Marked
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

        // The segment under the cursor and the state it should draw in. A vertex under the cursor
        // takes priority — it already answers with its own handle state — so a segment lights up
        // only when no vertex is hit, the SAME decision the vertex-grab and `sketch_delete_at`
        // make. Reusing those two hit-tests keeps the feedback exactly aligned with what a click
        // acts on. Select → Hover (brighter, "you can pick this edge"); Delete → Marked (warn +
        // `✕`, "this edge goes"); Add-point has its own insert diamond, so segments stay Idle.
        let hovered_segment: Option<(document::sketch::EntityId, ui::gizmos::HandleState)> =
            match tool {
                ui::panel::SketchTool::Select => Some(ui::gizmos::HandleState::Hover),
                ui::panel::SketchTool::Delete => Some(ui::gizmos::HandleState::Marked),
                ui::panel::SketchTool::AddPoint => None,
            }
            .and_then(|state| {
                self.last_cursor_position.and_then(|(cx, cy)| {
                    if self.sketch_vertex_at(cx, cy).is_some() {
                        None
                    } else {
                        self.nearest_sketch_segment(cx, cy)
                            .map(|(seg_id, _, _)| (seg_id, state))
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
                // Precedence: delete-hover (Marked ✕) > Selected > plain Hover > Idle. A selected
                // edge stays bold even under the cursor (Select hover never shrinks it); only the
                // destructive delete-hover overrides it.
                let selected = self.panel_state.sketch_selection.contains_segment(seg_id);
                let state = match hovered_segment {
                    Some((id, ui::gizmos::HandleState::Marked)) if id == seg_id => {
                        ui::gizmos::HandleState::Marked
                    }
                    _ if selected => ui::gizmos::HandleState::Selected,
                    Some((id, state)) if id == seg_id => state,
                    _ => ui::gizmos::HandleState::Idle,
                };
                self.sketch_segment_lines.push((a, b, state));
            }
        }

        // Add-point insert preview: the point on the hovered segment nearest the cursor (physical
        // px), in egui points — "a vertex lands here on this edge". Drawn as a diamond next frame.
        if tool == ui::panel::SketchTool::AddPoint {
            if let Some((cursor_x, cursor_y)) = self.last_cursor_position {
                if let Some((_, a, b)) = self.nearest_sketch_segment(cursor_x, cursor_y) {
                    let cursor = egui::Pos2::new(cursor_x as f32, cursor_y as f32);
                    let foot = closest_point_on_segment(cursor, a, b);
                    self.sketch_insert_preview =
                        Some(egui::Pos2::new(foot.x / pixels_per_point, foot.y / pixels_per_point));
                }
            }
        }
    }
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

#[cfg(test)]
mod tests {
    use super::{closest_point_on_segment, point_to_segment_distance};
    use egui::pos2;

    #[test]
    fn foot_falls_inside_the_span_for_a_perpendicular_drop() {
        // A cursor above the middle of a horizontal edge projects to that midpoint, so the
        // insert preview and the hit distance are the perpendicular offset.
        let a = pos2(0.0, 0.0);
        let b = pos2(10.0, 0.0);
        let foot = closest_point_on_segment(pos2(4.0, 3.0), a, b);
        assert!((foot.x - 4.0).abs() < 1e-4 && foot.y.abs() < 1e-4, "foot at (4, 0), got {foot:?}");
        assert!((point_to_segment_distance(pos2(4.0, 3.0), a, b) - 3.0).abs() < 1e-4);
    }

    #[test]
    fn foot_clamps_to_the_nearer_end_past_the_segment() {
        // A cursor beyond an endpoint clamps to that endpoint — the distance is to the vertex,
        // NOT to the infinite line, so a click off the end of an edge does not falsely hit it.
        let a = pos2(0.0, 0.0);
        let b = pos2(10.0, 0.0);
        assert_eq!(closest_point_on_segment(pos2(-5.0, 0.0), a, b), a, "clamps to the start");
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
}
