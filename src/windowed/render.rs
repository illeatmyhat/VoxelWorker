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
            let band = self.current_layer_band(self.region_dimensions[2]);
            let context = Self::make_refresh_context(
                &self.panel_state,
                &mut self.app_core.two_layer_cache,
                self.region_dimensions,
                self.recentre_voxels,
                band,
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

        let mut prepared = {
            profiling::scope!("egui_frame");
            run_egui_frame(
                &mut self.egui_bridge,
            &self.gpu.device,
            &self.gpu.queue,
            &mut self.panel_state,
            grid_z,
            self.measured_diameter,
            export_panel,
            &self.palette.ui,
            raw_input,
            [self.surface_config.width, self.surface_config.height],
                pixels_per_point,
                &mut self.context_menu_open_at,
            )
        };

        // Issue #25: cache the central 3D viewport rect so the view-cube
        // hit-testing (run later, in mouse events) can offset the cube corner.
        self.last_viewport_px = prepared.viewport_px;

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
        let intents = std::mem::take(&mut prepared.panel_response.intents);
        let mut merged_effect = crate::IntentEffect::none();
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
        // Issue #78: re-derive the selected-operand ghost on selection/geometry change
        // ONLY (never per frame). A `SelectNode` marks it dirty without a scene
        // re-resolve; the derivation itself is bounded by the SELECTED subtree's
        // covering chunks (`AppCore::selected_operand_ghost`), so this stays cheap even
        // in a huge scene. The `active` comparison is belt-and-braces for any selection
        // writer that bypassed the Intent effects.
        // Issue #79: the persistent child-boolean ghost re-derives at the SAME seam —
        // the two overlays share every trigger (the selection matters to both via the
        // cross-overlay dedupe rule), plus the checkbox toggle's dedicated
        // `operand_ghosts_changed` effect, which re-derives WITHOUT the scene
        // re-resolve above (the #79 acceptance bound).
        if merged_effect.selection_changed
            || merged_effect.scene_changed
            || merged_effect.operand_ghosts_changed
        {
            self.selected_ghost_dirty = true;
        }
        if self.selected_ghost_dirty
            || self.selected_ghost_selection != self.panel_state.scene.active
        {
            self.selected_ghost_dirty = false;
            self.selected_ghost_selection = self.panel_state.scene.active;
            match AppCore::selected_operand_ghost(
                &self.panel_state.scene,
                self.panel_state.geometry.voxels_per_block,
            ) {
                Some(ghost) => self.selected_operand_ghost_renderer.rebuild(
                    &self.gpu.device,
                    &ghost.bodies,
                    ghost.grid_dimensions,
                    ghost.recentre,
                    ghost.density,
                ),
                None => self.selected_operand_ghost_renderer.clear(),
            }
            // Issue #79: the persistent set, through the same bounded evaluation.
            match AppCore::child_boolean_ghost(
                &self.panel_state.scene,
                self.panel_state.geometry.voxels_per_block,
            ) {
                Some(ghost) => self.child_boolean_ghost_renderer.rebuild(
                    &self.gpu.device,
                    &ghost.bodies,
                    ghost.grid_dimensions,
                    ghost.recentre,
                    ghost.density,
                ),
                None => self.child_boolean_ghost_renderer.clear(),
            }
        }
        // Brick-display perf follow-up to epic #64: a debug-face toggle or a loaded-material
        // change are PURE display flags (they never `scene_changed`, so no rebuild fires) that
        // can turn OFF brick engagement — making the SKIPPED fallback mesh the display. Rebuild
        // it here the frame it is next needed, so a stale/empty mesh is never drawn. A no-op
        // unless the mesh is stale AND about to be shown.
        {
            let band = self.current_layer_band(self.region_dimensions[2]);
            let context = Self::make_refresh_context(
                &self.panel_state,
                &mut self.app_core.two_layer_cache,
                self.region_dimensions,
                self.recentre_voxels,
                band,
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
        // Issue #12: translate the layer-range scrubber into the shader band. The
        // band is inclusive on both ends; the upper handle is a layer index, so a
        // single-layer band is `lower == upper`. A full range draws everything.
        // Z-up: layers are Z-slices, so the band is a Z-layer range (index 2). The band
        // is computed by the shared `current_layer_band` helper (issue #60 M2) so the async
        // worker builds the mesh at the SAME band the render path applies here.
        let layer_range = self.panel_state.layer_range;
        let band = self.current_layer_band(grid_dimensions[2]);
        // Part of #20: the cuboid mesh path is the sole voxel renderer. Upload its
        // per-frame uniforms (camera + per-material base colours + band clip). A
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
            renderer.update_uniforms(
                &self.gpu.queue,
                view_projection,
                prepared.viewport_px,
                grid_dimensions,
                band,
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
        if let Some((pivot, extent)) = gizmo_placement {
            let extent_dims = [
                extent[0].round().max(0.0) as u32,
                extent[1].round().max(0.0) as u32,
                extent[2].round().max(0.0) as u32,
            ];
            self.transform_gizmo_renderer
                .rebuild(&self.gpu.device, &self.gpu.queue, extent_dims);
            self.transform_gizmo_renderer.update_uniforms(
                &self.gpu.queue,
                view_projection,
                glam::Vec3::from_array(pivot),
            );
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
        // World reference AXES (issue #29 S5): rebuild the visible Points' axis lines.
        self.points_renderer.rebuild_from_scene(
            &self.gpu.device,
            &self.gpu.queue,
            &self.panel_state.scene,
            self.panel_state.geometry.voxels_per_block,
        );
        self.points_renderer
            .update_uniforms(&self.gpu.queue, view_projection);
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
        // Issue #78/#79: the operand ghosts' per-frame camera + tint uploads (the
        // meshes were derived at the selection/geometry seam above, never here).
        self.selected_operand_ghost_renderer
            .update_uniforms(&self.gpu.queue, view_projection);
        self.child_boolean_ghost_renderer
            .update_uniforms(&self.gpu.queue, view_projection);
        self.view_cube_renderer
            .update_uniforms(&self.gpu.queue, self.app_core.camera.view_cube_view_projection());

        // ADR 0012: the onion-skin VOLUMETRIC FOG is retired. Onion context draws as the
        // display paths' ghost pass (prepared above: the brick slabs in `update_ghost_uniforms`,
        // the cuboid slabs in `update_uniforms` → `rebuild_for_band`; drawn in `render_frame`
        // when `onion_ghost_active`).
        let _ = layer_range;

        let overlays = FrameOverlays {
            gizmo: gizmo_placement
                .is_some()
                .then_some(&self.transform_gizmo_renderer),
            view_cube: if self.panel_state.show_view_cube {
                Some(&self.view_cube_renderer)
            } else {
                None
            },
            // #13 Step 4: live hover — the chrome zone under the cursor (computed
            // cheaply in `CursorMoved`) so the hovered rotate/roll arrow brightens.
            // `None` when nothing's hovered or while orbiting/dragging.
            cube_hovered_zone: self.hovered_cube_zone,
            // #13 Step 6 follow-up: the four rotate arrows are a standing affordance
            // whenever the view is constrained to a face (not hover-gated), with the
            // hovered one brightened. Off-face views show none.
            cube_rotate_arrows_visible: self.app_core.camera.is_face_constrained(),
            scene_grid: Some(&self.scene_grid_renderer),
            // Issue #29 S5: the windowed app always shows the Points (the Origin's
            // ground+axes are on by default); the batch self-gates on hidden/off.
            points: Some(&self.points_renderer),
            // Issue #29 Points fast-follow: the analytic infinite grid (Points' planes);
            // self-gates on no enabled plane.
            infinite_grid: Some(&self.infinite_grid_renderer),
            // ADR 0012: draw the onion GHOST pass this frame (the engaged display path
            // ghosts the onion slabs after its solid draw). Its uniforms/geometry were
            // prepared by the renderers' `update_uniforms` / `update_ghost_uniforms` above.
            onion_ghost_active,
            // Issue #78: the selected-operand ghost draws over BOTH display paths.
            // Suppressed in debug-faces mode (a diagnostic render — every ghost is off
            // there, matching the onion ghost's forced-FULL band); self-gates on an
            // empty selection.
            selected_operand_ghost: (!self.panel_state.debug_face_orientation)
                .then_some(&self.selected_operand_ghost_renderer),
            // Issue #79: the persistent child-boolean ghost draws UNDER the selection
            // ghost (same suppression rules); the persistent set excludes the active
            // node's body, so the two overlays never double an alpha.
            child_boolean_ghost: (!self.panel_state.debug_face_orientation)
                .then_some(&self.child_boolean_ghost_renderer),
            cuboid_mesh: self.display.cuboid_mesh_renderer(),
            // ADR 0011 G1: when engaged (field installed, no mesh-only mode), the
            // brick raymarch replaces the cuboid-mesh DRAW for this frame; the mesh
            // stays built as the fallback + A/B reference (ADR 0011 Decision 6).
            brick_raymarch: if brick_raymarch_engaged {
                self.display.brick_raymarch_renderer()
            } else {
                None
            },
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
                &overlays,
                &prepared,
            );

            surface_texture.present();
        }

        // One frame mark per rendered frame (not per event). No-op unless a
        // profiling backend is enabled; under `--features tracy` this delimits the
        // frame on the Tracy timeline.
        profiling::finish_frame!();
    }
}
