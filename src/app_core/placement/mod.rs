//! **Cursor → placed node** — turning a screen position into the `PlaceNode` intent
//! an armed tool would drop (`docs/design/direct-manipulation.md`, the placed point;
//! `crates/raycast/src/placement.rs`, the three-world-plane model).
//!
//! [`AppCore::place_primitive`] is the logic core of placement, one step above
//! [`pick_voxel`](AppCore::pick_voxel): pick answers *what voxel is under the cursor*;
//! this answers *where would a new node land, and what intent places it there*. It is
//! the headless half a live click will call — the viewport click handler
//! (`src/windowed/events.rs`) is a later slice that only forwards the cursor here and
//! drains the returned intent.
//!
//! **Two tiers, matching [`resolve_placement`].** A geometry hit is unambiguous, so
//! tier 1 is [`pick_voxel`](AppCore::pick_voxel) and the node lands on the OUTER side
//! of the entered face (`absolute_voxel + face_normal` — the empty Minecraft-style
//! neighbour). Missing geometry falls to tier 3, the built-in world planes, via
//! [`resolve_placement`] fed a ray rebased into the absolute voxel frame. (Tier 2,
//! user-created planes, is not wired yet.)
//!
//! **The frame (ADR 0008).** [`pick_voxel`] already returns `absolute_voxel` in the
//! absolute lattice, and a node's `offset_voxels` IS that absolute frame (a producer
//! emits `[offset, offset + grid)` corner-anchored — verified by the end-to-end tests
//! below, which drop a node and confirm its occupancy lands where the cursor pointed),
//! so the geometry tier needs no frame math beyond the face step. The empty-space tier
//! forms the absolute ray by shifting the render-frame cursor ray by `recentre_voxels`
//! (`absolute = render + recentre`), the same frame chain `pick_voxel` documents.

use camera::{unproject_screen_point_to_ray, ProjectionMode};
use glam::{Quat, Vec3};
use raycast::{resolve_placement, select_world_plane, world_plane_hit, PlacementTarget};
use substrate::spatial::Ray;

use document::intent::{Intent, NodeSpec};
use document::scene::{LeafProducer, Scene};
use document::voxel::SdfShape;
use ui::panel::{AngleSnap, PlacementPivot, PlacementSnap, PositionSnap};
use voxel_core::core_geom::MaterialChoice;

use super::picking::PickFrame;
use super::AppCore;

/// Snap a corner-anchored voxel offset to the grain the armed tool requests (owner ruling
/// 2026-07-21): whole **voxels** (the finest, no change) or whole **blocks** (offset a
/// multiple of the density, for clean inter-part mating). `NoSnap` is voxel-granular for now —
/// the freest placement the voxel document can store.
fn snap_offset(offset: [i64; 3], position: PositionSnap, density: u32) -> [i64; 3] {
    match position {
        PositionSnap::NoSnap | PositionSnap::Voxel => offset,
        PositionSnap::Block => {
            let d = density.max(1) as i64;
            // Round each axis to the nearest block boundary (round-half-up, symmetric enough
            // for placement; div_euclid keeps it consistent across the sign boundary).
            offset.map(|c| (c + d / 2).div_euclid(d) * d)
        }
    }
}

/// The corner-anchored world offset a node lands at when seated at `contact` with its local +Z
/// turned to `seat_normal` — the ONE map from a surface contact to the placed corner, shared by
/// the seat itself and the 15° joint solve's scoring so the two never drift.
///
/// `seat_centre_at` lands the object's local centre (`full/2`) at a target centroid; the pivot
/// choice is only *where that centroid goes*: [`PlacementPivot::Base`] pushes it half the local
/// height out along the normal (the base rests on the contact), [`PlacementPivot::VolumetricCenter`]
/// puts the centroid on the contact (the object straddles the surface).
fn seated_world_offset(contact: Vec3, seat_normal: Vec3, full: Vec3, pivot: PlacementPivot) -> Vec3 {
    let rotation = Quat::from_rotation_arc(Vec3::Z, seat_normal);
    let centre = match pivot {
        PlacementPivot::Base => contact + seat_normal * (full.z * 0.5),
        PlacementPivot::VolumetricCenter => contact,
    };
    evaluation::seat_centre_at(rotation, full, centre)
}

/// The position lattice granule, in voxels, a [`PositionSnap`] quantizes the placed corner to:
/// one voxel, one block (`density` voxels), or none (`0.0`, no position constraint).
fn position_lattice_step(position: PositionSnap, density: u32) -> f32 {
    match position {
        PositionSnap::NoSnap => 0.0,
        PositionSnap::Voxel => 1.0,
        PositionSnap::Block => density.max(1) as f32,
    }
}

/// How many alternating-projection rounds the 15° joint solve walks. Each round slides the contact
/// toward its quantized normal, then re-snaps it toward the position lattice; a handful is ample
/// because Newton on a true distance field converges in one step and the walk only samples the
/// tradeoff frontier for the scorer.
const JOINT_SOLVE_ROUNDS: usize = 4;

/// Seat a 15° angle-snapped drop (ADR 0027 §2): find the surface contact whose quantized seat
/// minimizes the **combined** position + angle error, and return `(contact, quantized_normal)` to
/// hand to the seat.
///
/// Position and angle are two views of one degree of freedom — *where the contact sits on the
/// surface* — so on a curved surface the constant-normal contour (a curve) and the position lattice
/// (a grid) generically do not intersect: no contact satisfies both exactly. Per the owner ruling
/// (2026-07-22) the solve therefore **minimizes the combined error** rather than favouring one
/// constraint. The two errors are made commensurable without a magic weight by charging the angular
/// error at the object's rim: an angle error `δ` displaces the rim by `≈ rim · δ`, so both terms are
/// world (voxel) distances.
///
/// It walks an alternating projection from the raw `hit` — slide toward the quantized normal, then
/// re-snap toward the position lattice — scoring every visited contact and keeping the best. With
/// `NoSnap` position (no lattice) the position error is zero, so it degrades to a pure slide onto
/// the nearest reachable 15° normal.
fn solve_seated_15deg(
    hit: Vec3,
    full: Vec3,
    snap: PlacementSnap,
    lattice_step: f32,
    field: &impl Fn(Vec3) -> f32,
) -> (Vec3, Vec3) {
    // An angle error of `δ` radians swings the object's rim through `≈ rim · δ`, converting the
    // angular error into the voxel units the position error already lives in.
    let rim = 0.5 * full.max_element();

    // Score a candidate contact by its quantized seat: how far the placed corner lands from the
    // position lattice, plus the rim-weighted gap between the true normal and its quantization.
    // Returns the score (smaller is better) and the quantized normal to seat with.
    let score = |contact: Vec3| -> (f32, Vec3) {
        let normal = raycast::gradient_normal(contact, field);
        let quantized = raycast::quantize_normal_to_15deg(normal);
        let position_error = if lattice_step > 0.0 {
            let offset = seated_world_offset(contact, quantized, full, snap.pivot);
            let rounded = (offset / lattice_step).round() * lattice_step;
            (offset - rounded).length()
        } else {
            0.0
        };
        let angle_error = normal.dot(quantized).clamp(-1.0, 1.0).acos();
        (position_error + rim * angle_error, quantized)
    };

    let (mut best_score, mut best_normal) = score(hit);
    let mut best_contact = hit;
    let mut current = hit;
    for _ in 0..JOINT_SOLVE_ROUNDS {
        // Angle projection: slide the contact along the surface until its normal matches the
        // quantized target (a no-op on a flat face, where only one normal is reachable).
        let target = raycast::quantize_normal_to_15deg(raycast::gradient_normal(current, field));
        let slid = raycast::snap_slide_to_normal(current, target, field);
        let (slid_score, slid_normal) = score(slid);
        if slid_score < best_score {
            best_score = slid_score;
            best_contact = slid;
            best_normal = slid_normal;
        }

        // Position projection: pull the slid contact toward the lattice and re-seat it. With no
        // position lattice this is the slid contact unchanged.
        let snapped = if lattice_step > 0.0 {
            raycast::snap_to_lattice_then_reproject(slid, lattice_step, field)
        } else {
            slid
        };
        let (snapped_score, snapped_normal) = score(snapped);
        if snapped_score < best_score {
            best_score = snapped_score;
            best_contact = snapped;
            best_normal = snapped_normal;
        }

        if (snapped - current).length() < 1.0e-6 {
            break; // the walk has settled at a fixed point
        }
        current = snapped;
    }
    (best_contact, best_normal)
}

/// The grazing threshold [`resolve_placement`] selects a world plane by:
/// `sin(20°) ≈ 0.342`, the smallest `|ray·plane_normal|` at which a plane is still
/// worth placing on. Must stay `≤ 1/√3` — the bound under which the three fixed
/// normals are guaranteed to keep one plane well-faced for any view direction (see
/// `raycast::select_world_plane`).
const MIN_GROUND_FACING: f32 = 0.342_f32;

/// The result of [`AppCore::place_primitive`]: the resolved [`PlacementTarget`] AND
/// the intent that places a node there, if any.
///
/// Both are carried because a viewport needs both: the [`intent`](Self::intent) is
/// `Some` only when the cursor resolved to a real placement (a geometry face or a
/// world plane), while the [`target`](Self::target) is always present so the chrome
/// can render the NEGATIVE answers too — the "point toward the ground"
/// ([`PlacementTarget::NoSurface`]) and "zoom in" ([`PlacementTarget::TooFar`])
/// feedback a bare `Option<Intent>` could not distinguish.
#[derive(Debug, Clone, PartialEq)]
pub struct PlacementOutcome {
    /// Where the cursor resolved to, or why it could not — the full four-answer
    /// [`PlacementTarget`], for the preview/affordance and the negative-state chrome.
    pub target: PlacementTarget,
    /// The `PlaceNode` intent that drops the primitive at the resolved point —
    /// `Some` for [`OnSurface`](PlacementTarget::OnSurface) /
    /// [`OnWorldPlane`](PlacementTarget::OnWorldPlane), `None` for the two negative
    /// answers (there is nowhere to place).
    pub intent: Option<Intent>,
}

impl AppCore {
    /// The cursor's pick ray, resolved into the ABSOLUTE voxel frame — the shared front half of
    /// BOTH placement tiers (the geometry SDF raymarch and the world-plane hit). Returns the
    /// unprojected render-frame ray, the `recentre` vector that rebases it to absolute, and the
    /// unit direction; each tier derives its own origin via [`cursor_ray_origin_absolute`]. `None`
    /// on a degenerate viewport or a failed unprojection.
    ///
    /// [`cursor_ray_origin_absolute`]: AppCore::cursor_ray_origin_absolute
    fn cursor_pick_ray(
        &self,
        cursor: [f32; 2],
        viewport: [f32; 4],
        frame: &PickFrame,
    ) -> Option<(Ray, Vec3, Vec3)> {
        if viewport[2] <= 0.0 || viewport[3] <= 0.0 {
            return None;
        }
        let aspect_ratio = viewport[2] / viewport[3];
        let view_projection = self.view_projection(aspect_ratio, frame.region_dimensions);
        let normalized_x = (cursor[0] - viewport[0]) / viewport[2] * 2.0 - 1.0;
        let normalized_y = 1.0 - (cursor[1] - viewport[1]) / viewport[3] * 2.0;
        let render_ray = unproject_screen_point_to_ray(view_projection, normalized_x, normalized_y)?;
        let recentre = frame.recentre_voxels;
        let recentre_vec = Vec3::new(recentre[0] as f32, recentre[1] as f32, recentre[2] as f32);
        let unit_direction = render_ray.direction.normalize();
        Some((render_ray, recentre_vec, unit_direction))
    }

    /// The absolute-frame origin the cursor ray is CAST from for surface intersection: the **eye**
    /// under perspective (the near-plane origin is unreliable at far zoom — it can dip below the
    /// ground), the pixel's **near-plane point** under orthographic (rays are parallel, there is no
    /// single eye). Both tiers cast from here so geometry and the world planes see the same ray.
    fn cursor_ray_origin_absolute(&self, render_ray: &Ray, recentre_vec: Vec3) -> Vec3 {
        match self.camera.projection_mode {
            ProjectionMode::Perspective => self.camera.eye() + recentre_vec,
            ProjectionMode::Orthographic => render_ray.origin + recentre_vec,
        }
    }

    /// Resolve where an armed primitive would drop for a cursor position, and the
    /// [`Intent`] that places it there (`docs/design/direct-manipulation.md`).
    ///
    /// `cursor` / `viewport` are the same physical-pixel space
    /// [`pick_voxel`](AppCore::pick_voxel) takes (`[x, y, width, height]`); `frame` is
    /// the last rebuild's [`PickFrame`]; `shape` / `material` are the armed tool's
    /// primitive. Tier 1 tries geometry via `pick_voxel` and places on the outer side
    /// of the entered face; on a miss, tier 3 intersects the cursor ray with the
    /// built-in world planes ([`resolve_placement`]). The negative answers
    /// ([`NoSurface`](PlacementTarget::NoSurface) /
    /// [`TooFar`](PlacementTarget::TooFar)) carry no intent.
    #[allow(clippy::too_many_arguments)] // a cohesive placement entry point: cursor + viewport +
    // frame + the armed tool (shape/material) + the two environment facts (ground visibility,
    // snap). Bundling them would just relocate the list without clarifying it.
    pub fn place_primitive(
        &self,
        cursor: [f32; 2],
        viewport: [f32; 4],
        frame: &PickFrame<'_>,
        scene: &Scene,
        shape: SdfShape,
        material: MaterialChoice,
        ground_plane_visible: bool,
        snap: PlacementSnap,
    ) -> PlacementOutcome {
        // A drop's rotation is written as a CONTINUOUS quaternion (ADR 0027) — surface placement
        // tilts the node's local +Z to the true gradient normal; a world-plane / upright drop
        // leaves it `None`. The whole tilt lives in the quaternion, which the classifier resolves
        // for any angle (a tube on a cylinder's curved side seats to the radial normal, not the
        // nearest of the 24 turns). Set per-tier.
        let place_node =
            |offset_voxels: [i64; 3], offset_local: [f32; 3], rotation_quaternion: Option<[f32; 4]>| {
                Intent::PlaceNode {
                    content: NodeSpec::Tool {
                        shape: shape.clone(),
                        material,
                    },
                    offset_voxels,
                    offset_local,
                    rotation_quaternion,
                }
            };

        // Seat a node at `contact` with its local +Z turned to `surface_normal` — the ONE seating
        // definition, shared by the geometry tier and the world-plane tier (owner ruling
        // 2026-07-21: there is NO upright mode; every drop orients to the surface it lands on). The
        // authoring PIVOT (where the object's centroid goes relative to the contact) is `snap.pivot`
        // via `seated_world_offset`. The pivot is CONTINUOUS: a `NoSnap` drop keeps the sub-voxel
        // remainder (origin integer part in `offset_voxels`, pivot fraction in `offset_local`),
        // while Voxel/Block snap quantizes the corner to the lattice and drops the fraction. Returns
        // the integer offset, the sub-voxel remainder, and the rotation, so the caller carries all
        // three into the intent.
        // The armed tool's producer-local extent, in voxels — the object's full size, shared by the
        // seat and the 15° joint solve.
        let full_size = Vec3::new(
            shape.size_voxels[0] as f32,
            shape.size_voxels[1] as f32,
            shape.size_voxels[2] as f32,
        );
        let seat_at = |contact: Vec3, surface_normal: Vec3| -> ([i64; 3], [f32; 3], Quat) {
            let rotation = Quat::from_rotation_arc(Vec3::Z, surface_normal);
            let world_offset = seated_world_offset(contact, surface_normal, full_size, snap.pivot);
            let (offset_voxels, offset_local) = match snap.position {
                // Continuous placement (ADR 0027): keep the pivot exactly under the cursor by
                // carrying its sub-voxel fraction. The integer floor is the far-world-safe origin;
                // the remainder is always in `[0, 1)` per axis.
                PositionSnap::NoSnap => {
                    let floor = world_offset.floor();
                    (
                        [floor.x as i64, floor.y as i64, floor.z as i64],
                        (world_offset - floor).to_array(),
                    )
                }
                // Voxel / Block snap: quantize the corner to the lattice (round to nearest voxel,
                // then `snap_offset` coarsens to a block multiple for Block) — no sub-voxel part.
                position => (
                    snap_offset(
                        [
                            world_offset.x.round() as i64,
                            world_offset.y.round() as i64,
                            world_offset.z.round() as i64,
                        ],
                        position,
                        frame.density,
                    ),
                    [0.0, 0.0, 0.0],
                ),
            };
            (offset_voxels, offset_local, rotation)
        };

        // Tier 1 — geometry. A picked surface is unambiguous, and a node dropped on a geometry
        // surface is ALWAYS **seated**: it contacts the surface with the surface's own normal (ADR
        // 0027 — "Seated placement"). There is no upright mode here; "upright" is the degenerate
        // world-vertical case, and that belongs to the world-plane tier below (world planes are
        // never seated). So the picked voxel face — only an axis-aligned staircase of the true
        // surface (a cylinder's curved side reads as +X/+Y steps) — is refined to the CONTINUOUS
        // surface: build the composed SDF, project the pick onto it, and turn the node's local +Z
        // to the exact gradient normal, so a tube on a curved side lies along the radial normal.
        //
        // `snap.angle` selects the ANGLE-snap granularity (ADR 0027 §2): `Continuous` seats to the
        // exact gradient normal, `Deg15` runs the joint solve that trades the contact against the
        // 15° angle lattice. Neither ever means "upright on geometry" — that is the world-plane
        // tier's job.
        if let Some(pick) = self.pick_voxel(cursor, viewport, frame) {
            // The composed field over the scene's op-stack — the SAME fold the classifier resolves
            // (`evaluation::composed_field_at`), so the surface the node seats on is the surface it
            // will occupy once dropped. Built lazily here (a geometry hit only), never per
            // empty-space hover frame.
            let leaves = scene.leaf_producers(frame.density);
            let leaf_refs: Vec<&LeafProducer> = leaves.iter().collect();
            let field = |probe: Vec3| evaluation::composed_field_at(&leaf_refs, probe, frame.density);

            // `pick_voxel`'s DDA answers WHICH surface is under the cursor — a definite face even at
            // a box edge/corner. It contributes only a graze-safe FALLBACK contact (the picked
            // face CENTRE projected onto the composed surface) for the rare tangent where the cursor
            // march slips past the surface; the seat itself reads from the SDF at the real contact.
            let face_centre = Vec3::new(
                pick.absolute_voxel[0] as f32 + 0.5 + pick.face_normal[0] as f32 * 0.5,
                pick.absolute_voxel[1] as f32 + 0.5 + pick.face_normal[1] as f32 * 0.5,
                pick.absolute_voxel[2] as f32 + 0.5 + pick.face_normal[2] as f32 * 0.5,
            );
            let stable_surface = raycast::project_to_surface(face_centre, field);

            // The cursor ray answers WHERE on that surface, continuously: cast it at the composed
            // field — the SAME sphere-trace the GPU ghost runs (`raycast::raymarch` ↔
            // `placement_ghost.wgsl`) — so a `NoSnap` drop keeps the exact sub-voxel point under the
            // cursor instead of snapping to the picked voxel's centre (the geometry-vs-ground
            // asymmetry the owner hit). Fall back to the stable surface point if the march grazes
            // past (a rare tangent).
            let continuous_contact = self
                .cursor_pick_ray(cursor, viewport, frame)
                .and_then(|(render_ray, recentre_vec, unit_direction)| {
                    let origin = self.cursor_ray_origin_absolute(&render_ray, recentre_vec);
                    raycast::raymarch(origin, unit_direction, field, &raycast::MarchParams::default())
                })
                .map(|surface_hit| surface_hit.point)
                .unwrap_or(stable_surface);

            // Seat and snap on the SDF, never the rendered voxel geometry (owner ruling: quantizing
            // the continuous field simplifies the assumptions). The tilt is the composed field's
            // gradient, but sampled at the CORNER-SAFE `stable_surface` (the entered face's interior)
            // — reading it at the raw contact would pick up the diagonal at a box corner, which the
            // cursor sits on under an orbit view. The DDA face only chooses WHERE to sample; the
            // normal itself is the SDF's.
            let normal = raycast::gradient_normal(stable_surface, field);
            let (seat_contact, seat_normal) = match snap.angle {
                // `Continuous` places the sub-voxel cursor point with that corner-safe tilt.
                AngleSnap::Continuous => (continuous_contact, normal),
                AngleSnap::Deg15 => {
                    let step = position_lattice_step(snap.position, frame.density);
                    if step > 0.0 {
                        // Position is snapped: the joint solve trades position against angle error
                        // over the lattice, seeded from the corner-safe surface point.
                        solve_seated_15deg(stable_surface, full_size, snap, step, &field)
                    } else {
                        // NoSnap position: keep the exact sub-voxel cursor point and quantize the
                        // corner-safe SDF normal to the 15° lattice — on a flat face that is the face
                        // axis (already a 15° multiple), so the drop is under the cursor, not snapped
                        // to the picked voxel's centre. This is the geometry side of the same
                        // continuous-position fix the ground plane already had.
                        (continuous_contact, raycast::quantize_normal_to_15deg(normal))
                    }
                }
            };
            let (offset, offset_local, rotation) = seat_at(seat_contact, seat_normal);
            return PlacementOutcome {
                // `point` is the continuous surface contact (the cursor location) for the affordance.
                target: PlacementTarget::OnSurface {
                    point: continuous_contact,
                    face_normal: pick.face_normal,
                },
                intent: Some(place_node(offset, offset_local, Some(rotation.to_array()))),
            };
        }

        // Tier 3 — the built-in world planes. Cast the SAME cursor ray the geometry tier used
        // ([`cursor_pick_ray`], the one shared ray construction), already rebased into the ABSOLUTE
        // voxel frame. `resolve_placement` then returns a point already in absolute voxels.
        //
        // `unproject_screen_point_to_ray` returns the NEAR-PLANE point as the origin. Whether a
        // world plane is "in front" of that point is the crux, and it is NOT the same question for
        // the two projections — so the ray's reachability is resolved per projection.
        //
        // Precision caveat (ADR 0008): `recentre_voxels as f32` loses integer precision past ~16M
        // voxels. Correct for the small scenes this placement slice targets; the eventual fix is the
        // i64 origin-rebase, not a fudge here.
        let Some((render_ray, recentre_vec, unit_direction)) =
            self.cursor_pick_ray(cursor, viewport, frame)
        else {
            return PlacementOutcome { target: PlacementTarget::NoSurface, intent: None };
        };
        // A block spans `density` voxels in this frame, so the authorability limit is asked
        // in voxel units with the density as the block size.
        let block_size = frame.density.max(1) as f32;

        let target = match self.camera.projection_mode {
            // Perspective — cast from the EYE (the centre of projection). The near-plane
            // point is wrong here: it grows with `orbit_distance` and at a far zoom its
            // lower half dips BELOW the ground, so a downward cursor ray whose near-plane
            // origin already sits under the ground reports the ground as *behind* it
            // (`NoSurface`) even while the eye is far above it — placement silently died
            // across the foreground half of the screen. The eye lies on the same ray line,
            // sits where the camera actually is, and gives the true eye-distance the
            // authorability check wants. A perspective view genuinely straddles ground and
            // sky, so reachability is legitimately per-pixel — exactly `resolve_placement`'s
            // `t > 0` test from the eye.
            ProjectionMode::Perspective => {
                let eye_ray = Ray::new(self.camera.eye() + recentre_vec, unit_direction);
                resolve_placement(None, eye_ray, MIN_GROUND_FACING, |depth| {
                    self.camera.depth_is_authorable(depth, block_size)
                })
            }
            // Orthographic — rays are PARALLEL, so there is no eye on the ray and no
            // per-pixel near/far truth: reachability is a property of the whole VIEW,
            // uniform across the screen. Any origin-based `t`-sign test would split the
            // screen along the plane's intersection line (the bug that made the foreground
            // half report `NoSurface`). Instead, select the plane on the shared direction
            // and ask the directional question: the plane is reachable iff the eye sits on
            // its FRONT side while the view looks TOWARD it — `sign(eye·n)` opposes
            // `sign(dir·n)`. The hit POINT still comes from the pixel's own parallel line
            // (each strikes a different spot). Depth does not enter ortho authorability (it
            // keys off `orbit_distance`), so the limit is asked once.
            ProjectionMode::Orthographic => {
                let plane = select_world_plane(unit_direction, MIN_GROUND_FACING);
                let normal = plane.normal();
                let eye_abs = self.camera.eye() + recentre_vec;
                let reachable = eye_abs.dot(normal) * unit_direction.dot(normal) < 0.0;
                if !reachable {
                    PlacementTarget::NoSurface
                } else if !self
                    .camera
                    .depth_is_authorable(self.camera.orbit_distance, block_size)
                {
                    PlacementTarget::TooFar
                } else {
                    let line = Ray::new(render_ray.origin + recentre_vec, unit_direction);
                    let (point, _t) = world_plane_hit(line, plane);
                    PlacementTarget::OnWorldPlane { point, plane }
                }
            }
        };

        // Only place on a world plane the user can SEE (owner ruling 2026-07-21): the two
        // vertical planes are never visualized, so they are never a placement target — a
        // grazing ray that would fall back to one reports NoSurface ("point at a surface")
        // instead of dropping a node, vertical and centred, on an invisible plane far away.
        // The ground plane is a target only when its floor grid is shown.
        let target = match target {
            PlacementTarget::OnWorldPlane { plane, .. } => {
                let plane_visible = match plane {
                    raycast::WorldPlane::Ground => ground_plane_visible,
                    // The x=0 / y=0 verticals have no visualization, so they are always hidden.
                    raycast::WorldPlane::VerticalFacingX | raycast::WorldPlane::VerticalFacingY => {
                        false
                    }
                };
                if plane_visible {
                    target
                } else {
                    PlacementTarget::NoSurface
                }
            }
            other => other,
        };

        let intent = match target {
            // A world plane seats exactly like a geometry surface (owner ruling 2026-07-21): the
            // node orients to the plane normal facing the side it is placed FROM, so a drop on the
            // ground's UNDERSIDE hangs it upside down. `-sign(dir·n)` selects the approach-facing
            // normal — identity rotation for the ground seen from above (upright), a 180° flip from
            // below. The old face-anchored "world-vertical" rule is retired.
            PlacementTarget::OnWorldPlane { point, plane } => {
                let plane_normal = plane.normal();
                let facing_normal = if unit_direction.dot(plane_normal) > 0.0 {
                    -plane_normal
                } else {
                    plane_normal
                };
                let (offset, offset_local, rotation) = seat_at(point, facing_normal);
                Some(place_node(offset, offset_local, Some(rotation.to_array())))
            }
            // A geometry face cannot come back from a `None` surface, and the two
            // negative answers place nothing.
            PlacementTarget::OnSurface { .. }
            | PlacementTarget::NoSurface
            | PlacementTarget::TooFar => None,
        };
        PlacementOutcome { target, intent }
    }
}

#[cfg(test)]
mod tests;
