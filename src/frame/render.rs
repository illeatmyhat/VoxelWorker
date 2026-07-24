//! The GPU half of the per-frame pipeline (ADR 0031): [`render_frame`] records the viewport
//! MSAA pass as ordered [`FramePhases`] (background → model → over-model → scaffold → on-top),
//! then the view cube in its own corner pass. The egui pass is the sibling
//! [`egui_frame`](super::egui_frame).

use crate::*;
use super::egui_frame::{EguiPaintBridge, PreparedEguiFrame};


/// Render a complete frame into `target_view`.
///
/// This is the render-target-agnostic core (Hard requirement #2): it accepts a
/// resolved single-sample colour [`wgpu::TextureView`] plus the prepared egui
/// data and has no knowledge of winit or surfaces. The windowed binary passes
/// the surface texture's view; the headless binary passes the offscreen capture
/// texture's view.
///
/// Milestone 4 restructures the frame into two passes:
///   1. **3D MSAA pass** — the instanced voxel cubes are drawn into a 4-sample
///      colour texture (`msaa_color_view`) with a 4-sample depth attachment
///      (`depth_view`) and resolved into `target_view` (the single-sample
///      surface / capture texture). `material` selects the bound texture and
///      `grid_overlay_enabled` was already folded into the uniforms by the
///      caller.
///   2. **egui pass** — egui renders at 1 sample directly onto the RESOLVED
///      `target_view` with `LoadOp::Load`, compositing the panel on top.
///
/// `msaa_color_view` and `depth_view` are render-target-agnostic: the window and
/// the headless capture pass their own 4-sample textures sized to the same target.
/// The viewport render as ordered frame phases (ADR 0031). [`render_frame`] records these into
/// the single MSAA pass in a FIXED order — background → model → over-model → scaffold → on-top —
/// then the view cube in its own corner pass. Each phase slice is a caller-filled list of
/// [`SceneDraw`](display::SceneDraw)s (self-gating, so an empty batch draws nothing); the model
/// and the cube are special (material bind / own sub-pass) and stay named fields.
pub struct FramePhases<'a> {
    /// Fullscreen, pre-solid, depth off (the Signal background gradient, issue #91).
    pub background: &'a [&'a dyn display::SceneDraw],
    /// Depth-off draws recorded BEFORE the model so opaque geometry paints over them —
    /// paint-order occlusion (ADR 0031). The far-distance reference-axes fallback lives here:
    /// invariant (never clips) yet still occluded by geometry, where depth-testing can't survive
    /// the collapsed near/far.
    pub behind_model: &'a [&'a dyn display::SceneDraw],
    /// Translucent ghosts blended over the solid, depth-tested no-write (the operand x-ray,
    /// the placement ghost). Drawn after the model so both display paths' depth is final.
    pub over_model: &'a [&'a dyn display::SceneDraw],
    /// Depth-tested reference lines the model occludes (block/floor grids, point axes,
    /// analytic infinite grid) — scaffold behind/under the model.
    pub scaffold: &'a [&'a dyn display::SceneDraw],
    /// Depth off, drawn through the model (the manipulator gizmos).
    pub on_top: &'a [&'a dyn display::SceneDraw],
    /// The cuboid mesh renderer — the CPU voxel render path (part of #20; the legacy
    /// instanced mesher was removed). Draws the voxels as a box-decomposed mesh; its
    /// uniforms must already be uploaded via `CuboidMeshRenderer::update_uniforms`.
    /// Kept PERMANENTLY as the headless/no-GPU fallback + A/B reference (ADR 0011
    /// Decision 6) even when the brick path below takes the frame.
    pub cuboid_mesh: &'a display::mesh::CuboidMeshRenderer,
    /// ADR 0011 G1: the brick raymarch display sink. `Some` replaces the cuboid
    /// mesh DRAW for this frame (single ported-producer scenes on the GPU path) —
    /// the pass runs in the same MSAA pass and writes ray-hit depth, so every
    /// phase after it composites unchanged. `None` keeps the mesh path (multi-producer,
    /// loaded materials, debug modes, no-GPU builds).
    pub brick_raymarch: Option<&'a display::brick::BrickRaymarchRenderer>,
    /// ADR 0012: draw the onion GHOST this frame. When `true`, immediately after the solid
    /// voxel draw (inside the model phase), the engaged display path (brick raymarch when
    /// present, else the cuboid mesh) draws its translucent ghost of the voxels in the onion
    /// slabs. Depth-tested `Less` + alpha-blended, depth WRITE ON so only the NEAREST ghost
    /// surface shows; the solid, drawn first, still occludes it. The ghost uniforms/geometry
    /// must already be prepared by the renderers' `update_uniforms`.
    pub onion_ghost_active: bool,
    /// The corner view cube (its own scissored pass). `None` when its Display toggle is off.
    pub view_cube: Option<&'a display::renderer::ViewCubeRenderer>,
    /// The ViewCube chrome zone under the cursor (#13 Step 2). Drives which hover
    /// arrows the cube draws and which glyph is highlighted. `None` = nothing hovered.
    pub cube_hovered_zone: Option<camera::CubeChromeZone>,
    /// #13 Step 6 follow-up: draw all four ViewCube rotate arrows PERSISTENTLY (set
    /// when the view is face-constrained), with the hovered one brightened. `false`
    /// (off-face view) draws no rotate arrows.
    pub cube_rotate_arrows_visible: bool,
    /// Signal (issue #88): the view cube's right inset (physical px) = the floating display
    /// stack's current width, so the GPU-drawn cube slides left of the stack. From
    /// `PreparedEguiFrame`.
    pub view_cube_right_inset_px: u32,
    /// Target dimensions (needed to place the view-cube corner viewport).
    pub target_width: u32,
    pub target_height: u32,
}

/// Upload the per-frame **scene scaffold** uniforms shared by the windowed shell and `shot`
/// (ADR 0031): the per-object scene grid, the world-reference Points (screen-stable axes +
/// planes), and the analytic infinite grid. Both paths previously drove these renderers with
/// byte-identical orchestration inline — the drift that let the overlay matrix diverge between
/// them (a Point far from the render origin clipped in one path but not the other). Centralising
/// it here makes that divergence unrepresentable: one place computes the sequence, both call it.
///
/// `scene_matrices` bundles the scene matrix, its camera-relative companion (the infinite
/// grid unprojects per fragment and melts on the full matrix at wide-baseline coordinates),
/// and the eye; `overlay_view_projection` is the depth-off axes matrix (all from the shared
/// [`AppCore`](crate::AppCore) getters). The scene grid always uploads.
/// `show_points` gates the Points + infinite grid (the shell always draws them; `shot` only under
/// `--points`). `axes_through` ("axes on top") skips the depth-tested Points instance, leaving
/// only the depth-off overlay instance.
#[allow(clippy::too_many_arguments)]
pub fn upload_scene_scaffold(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &Scene,
    density: u32,
    camera: &camera::OrbitCamera,
    scene_matrices: camera::SceneMatrices,
    overlay_view_projection: glam::Mat4,
    show_points: bool,
    axes_through: bool,
    scene_grid: &mut SceneGridRenderer,
    points: &mut PointsRenderer,
    points_overlay: &mut PointsRenderer,
    infinite_grid: &mut InfiniteGridRenderer,
) {
    scene_grid.rebuild_from_scene(device, queue, scene, density);
    scene_grid.update_uniforms(queue, scene_matrices.view_projection);
    if !show_points {
        return;
    }
    // Depth-off overlay instance (the on-top / paint-order axes) — always drawn.
    points_overlay.rebuild_from_scene(device, queue, scene, density, camera, false);
    points_overlay.update_uniforms(queue, overlay_view_projection);
    // Depth-tested instance for crisp near occlusion — only when NOT drawing axes on top.
    if !axes_through {
        points.rebuild_from_scene(device, queue, scene, density, camera, true);
        points.update_uniforms(queue, scene_matrices.view_projection);
    }
    infinite_grid.rebuild_from_scene(queue, scene, density, scene_matrices);
}

#[allow(clippy::too_many_arguments)]
pub fn render_frame(
    bridge: &mut EguiPaintBridge,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target_view: &wgpu::TextureView,
    msaa_color_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    material: display::renderer::MaterialSource,
    phases: &FramePhases,
    prepared: &PreparedEguiFrame,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-worker frame encoder"),
    });

    // egui's buffer upload happens on the same encoder; the returned command
    // buffers must be submitted before (or alongside) the main encoder.
    let egui_upload_commands = bridge.renderer.update_buffers(
        device,
        queue,
        &mut encoder,
        &prepared.paint_jobs,
        &prepared.screen_descriptor,
    );

    // === Pass 1: 3D voxel pass at 4× MSAA, resolved into the single-sample target.
    {
        let mut voxel_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("voxel-worker 3D msaa pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: msaa_color_view,
                // Resolve the multisampled colour into the single-sample target.
                resolve_target: Some(target_view),
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(WORKSHOP_CLEAR_COLOR),
                    // The multisampled texture is transient; we only keep the
                    // resolved result. Discarding it is the cheaper store.
                    store: wgpu::StoreOp::Discard,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    // Stored (not discarded) so the onion-skin fog pass can sample
                    // this MSAA depth to stop its raymarch at opaque surfaces.
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        // Issue #25: confine the 3D geometry to the central viewport rect (the
        // window minus the side panel + bottom dock). The MSAA target was still
        // CLEARED to the workshop colour across the WHOLE target above, so any
        // sliver not covered by egui isn't garbage; only the draws are scissored.
        let [viewport_x, viewport_y, viewport_width, viewport_height] = prepared.viewport_px;
        voxel_pass.set_viewport(
            viewport_x as f32,
            viewport_y as f32,
            viewport_width as f32,
            viewport_height as f32,
            0.0,
            1.0,
        );
        voxel_pass.set_scissor_rect(viewport_x, viewport_y, viewport_width, viewport_height);

        // Background phase (ADR 0031): fullscreen, pre-solid, depth off — the Signal
        // background gradient — so every voxel + phase below composites over it.
        for draw in phases.background {
            draw.draw(&mut voxel_pass);
        }

        // Behind-model phase (ADR 0031): depth-off draws recorded before the model, so the opaque
        // model paints over them (paint-order occlusion) — the far-distance reference-axes fallback.
        for draw in phases.behind_model {
            draw.draw(&mut voxel_pass);
        }

        // The voxel model: the brick raymarch (ADR 0011 G1) when engaged, else the
        // cuboid mesh path. When a VS block is applied the mesh path binds the
        // block's 6-layer D2Array so it textures per-face; no applied block →
        // `None` keeps the procedural-atlas path. The brick pass writes ray-hit
        // depth into this same MSAA depth attachment, so the depth-tested overlays
        // below (and the fog's depth-stop) composite identically on both paths.
        let loaded_material = match material {
            display::renderer::MaterialSource::Loaded(bind_group) => Some(bind_group),
            display::renderer::MaterialSource::Procedural(_) => None,
        };
        if let Some(brick_raymarch) = phases.brick_raymarch {
            // ADR 0011 G2: a loaded VS block now textures the raymarch too — bind the
            // block's 6-layer D2Array at group(2) so solid hits shade per-face by the
            // owner's lattice rule (the brick renderer's `loaded_material_active` flag,
            // set alongside its uniforms, selects that branch). `None` binds the dummy.
            brick_raymarch.draw(&mut voxel_pass, loaded_material);
        } else {
            phases.cuboid_mesh.draw(&mut voxel_pass, loaded_material);
        }

        // ADR 0012 (H1) — the onion GHOST pass. Immediately after the SOLID band draw,
        // in the SAME MSAA pass, the engaged display path ghosts the voxels in the onion
        // slabs (recentred-Z outside the band, within ±onion_depth). Depth-tested
        // `Less` + alpha-blended, with depth WRITE ON so only the nearest ghost surface
        // shows (a builder-independent render); the just-drawn solid still occludes it. The
        // brick ghost is two per-slab raymarches; the mesh ghost is two thin per-slab
        // meshes — both shaded flat translucent (the retired fog haze's hue). This
        // REPLACES the former volumetric fog pass (Pass 1a below, now always `None`).
        if phases.onion_ghost_active {
            if let Some(brick_raymarch) = phases.brick_raymarch {
                brick_raymarch.draw_ghost(&mut voxel_pass);
            } else {
                phases.cuboid_mesh.draw_ghost(&mut voxel_pass);
            }
        }

        // Over-model phase (ADR 0031): translucent ghosts blended over the solid, depth-tested
        // no-write — the operand x-ray (#78 / ADR 0018 D6) and the armed-tool placement ghost
        // (ADR 0022). After the solid + onion ghost so both display paths' depth is final;
        // before the scaffold, which they cannot occlude (they write no depth).
        for draw in phases.over_model {
            draw.draw(&mut voxel_pass);
        }

        // Scaffold phase (ADR 0031): depth-tested reference lines the solid model occludes —
        // per-object block/floor grids (#29 S3), the analytic infinite reference grid (#29
        // Points fast-follow, depth-tested via `frag_depth`), and the visible Points' axes
        // (#29 S5). Scaffold behind/under the model, not an overlay on top.
        for draw in phases.scaffold {
            draw.draw(&mut voxel_pass);
        }

        // On-top phase (ADR 0031): depth-test OFF, drawn through the solid model — the
        // manipulator gizmos.
        for draw in phases.on_top {
            draw.draw(&mut voxel_pass);
        }
    }

    // === Pass 1b: view cube into a scissored top-left corner (its own depth).
    // Drawn after the 3D resolve, before egui (render layering).
    if let Some(view_cube) = phases.view_cube {
        view_cube.draw(
            device,
            queue,
            &mut encoder,
            target_view,
            phases.target_width,
            phases.target_height,
            prepared.viewport_px,
            phases.view_cube_right_inset_px,
            phases.cube_hovered_zone,
            phases.cube_rotate_arrows_visible,
        );
    }

    // === Pass 2: egui at 1 sample onto the RESOLVED target (load, don't clear).
    {
        let egui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("voxel-worker egui pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        // egui wants a RenderPass<'static>; forget_lifetime converts it.
        bridge.renderer.render(
            &mut egui_pass.forget_lifetime(),
            &prepared.paint_jobs,
            &prepared.screen_descriptor,
        );
    }

    queue.submit(egui_upload_commands.into_iter().chain(std::iter::once(encoder.finish())));

    for texture_id in &prepared.textures_to_free {
        bridge.renderer.free_texture(texture_id);
    }
}
