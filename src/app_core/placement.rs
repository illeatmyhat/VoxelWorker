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

            // Seed the solve at the entered face (the boundary between the solid voxel and its
            // empty neighbour) and let damped Newton settle onto the composed surface under the
            // cursor (`raycast::project_to_surface`, ADR 0027 §5).
            let seed = Vec3::new(
                pick.absolute_voxel[0] as f32 + 0.5 + pick.face_normal[0] as f32 * 0.5,
                pick.absolute_voxel[1] as f32 + 0.5 + pick.face_normal[1] as f32 * 0.5,
                pick.absolute_voxel[2] as f32 + 0.5 + pick.face_normal[2] as f32 * 0.5,
            );
            let hit = raycast::project_to_surface(seed, field);
            let normal = raycast::gradient_normal(hit, field);
            // Seat flush to the gradient normal (ADR 0027) — `Continuous` uses it directly; `Deg15`
            // slides the contact to minimize the combined position + angle error, seating with that
            // contact's quantized normal. The seat map itself is the ONE definition shared with the
            // world-plane tier below.
            let (seat_contact, seat_normal) = match snap.angle {
                AngleSnap::Continuous => (hit, normal),
                AngleSnap::Deg15 => solve_seated_15deg(
                    hit,
                    full_size,
                    snap,
                    position_lattice_step(snap.position, frame.density),
                    &field,
                ),
            };
            let (offset, offset_local, rotation) = seat_at(seat_contact, seat_normal);
            return PlacementOutcome {
                // `point` is the surface hit (the cursor location) for the affordance.
                target: PlacementTarget::OnSurface { point: hit, face_normal: pick.face_normal },
                intent: Some(place_node(offset, offset_local, Some(rotation.to_array()))),
            };
        }

        // Tier 3 — the built-in world planes. Rebuild the render-frame cursor ray
        // exactly as `pick_voxel` does (NDC in the viewport rect, unprojected through
        // the same view-projection), then rebase it into the ABSOLUTE voxel frame:
        // `absolute = render + recentre_voxels`. `resolve_placement` then returns a
        // point already in absolute voxels.
        if viewport[2] <= 0.0 || viewport[3] <= 0.0 {
            return PlacementOutcome { target: PlacementTarget::NoSurface, intent: None };
        }
        let aspect_ratio = viewport[2] / viewport[3];
        let view_projection = self.view_projection(aspect_ratio, frame.region_dimensions);
        let normalized_x = (cursor[0] - viewport[0]) / viewport[2] * 2.0 - 1.0;
        let normalized_y = 1.0 - (cursor[1] - viewport[1]) / viewport[3] * 2.0;
        let Some(render_ray) =
            unproject_screen_point_to_ray(view_projection, normalized_x, normalized_y)
        else {
            return PlacementOutcome { target: PlacementTarget::NoSurface, intent: None };
        };

        // `unproject_screen_point_to_ray` returns the NEAR-PLANE point as the origin.
        // Whether a world plane is "in front" of that point is the crux, and it is NOT the
        // same question for the two projections — so the ray's reachability is resolved per
        // projection.
        //
        // Precision caveat (ADR 0008): `recentre_voxels as f32` loses integer precision
        // past ~16M voxels. Correct for the small scenes this placement slice targets; the
        // eventual fix is the i64 origin-rebase, not a fudge here.
        let recentre = frame.recentre_voxels;
        let recentre_vec = Vec3::new(recentre[0] as f32, recentre[1] as f32, recentre[2] as f32);
        let unit_direction = render_ray.direction.normalize();
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
mod tests {
    use std::sync::Arc;

    use camera::{OrbitCamera, ProjectionMode};
    use display::renderer::LayerBand;
    use document::scene::Scene;
    use document::voxel::{GeometryParams, SdfShape};
    use evaluation::two_layer_store::TwoLayerChunk;
    use voxel_core::core_geom::{MaterialChoice, CHUNK_BLOCKS};
    use voxel_core::voxel::ShapeKind;

    use super::*;
    use crate::{AppCore, RebuildOutcome};

    /// **place_primitive seats CONTINUOUSLY to the surface normal (ADR 0027).** On a flat box
    /// face the composed-field gradient normal equals the entered face normal, so the emitted
    /// intent carries a continuous `rotation_quaternion` that tilts the node's local +Z to that
    /// normal. The whole tilt lives in the quaternion. (The occupancy correctness of an
    /// arbitrary rotation is proven by the classifier's own tests; this pins the placement wiring
    /// that writes the quaternion.)
    #[test]
    fn a_placed_primitive_tilts_to_the_entered_surface_normal() {
        let fixture = placement_fixture(OrbitCamera::default());
        let cursor = [640.0, 360.0];
        // A tall (asymmetric) armed tool, so a tilt is observable.
        let shape = SdfShape::from_blocks(ShapeKind::Cylinder, [1, 1, 3], 1, DENSITY);

        // The entered face normal — the direction the continuous seat must tilt local +Z toward.
        let pick = fixture
            .app_core
            .pick_voxel(cursor, VIEWPORT, &fixture.frame())
            .expect("the centre cursor hits the Box");
        let face_normal = Vec3::new(
            pick.face_normal[0] as f32,
            pick.face_normal[1] as f32,
            pick.face_normal[2] as f32,
        );

        let outcome = fixture.app_core.place_primitive(
            cursor,
            VIEWPORT,
            &fixture.frame(),
            &fixture.scene,
            shape.clone(),
            MaterialChoice::Stone,
            true,
            PlacementSnap::default(),
        );
        let Some(Intent::PlaceNode { rotation_quaternion, .. }) = outcome.intent else {
            panic!("a geometry hit produces a PlaceNode, got {:?}", outcome.intent);
        };
        // The whole tilt lives in the continuous quaternion.
        let quaternion = rotation_quaternion.expect("surface snap carries a continuous rotation");
        let axis = Quat::from_array(quaternion) * Vec3::Z;
        // On a flat box face the gradient normal IS the face normal, so the node's axis tilts to it.
        assert!(
            axis.dot(face_normal) > 0.99,
            "the node's +Z axis {axis:?} must tilt to the entered face normal {face_normal:?}"
        );
    }

    const DENSITY: u32 = 8;
    const VIEWPORT: [f32; 4] = [0.0, 0.0, 1280.0, 720.0];

    /// A rebuilt scene the placement flow runs against — keeps the SCENE too (unlike
    /// the picking fixture) because a placement test must apply the returned intent
    /// and rebuild to check where the node landed.
    struct Fixture {
        app_core: AppCore,
        scene: Scene,
        region_dimensions: [u32; 3],
        recentre_voxels: [i64; 3],
        chunks: Vec<([i32; 3], Arc<TwoLayerChunk>)>,
    }

    impl Fixture {
        fn frame(&self) -> PickFrame<'_> {
            PickFrame {
                region_dimensions: self.region_dimensions,
                recentre_voxels: self.recentre_voxels,
                density: DENSITY,
                chunks: &self.chunks,
                band: LayerBand::FULL,
            }
        }

        /// Apply an intent to the scene and rebuild, refreshing the resident chunks —
        /// the pipeline a live placement drives (`apply_intent` → `rebuild`).
        fn apply_and_rebuild(&mut self, intent: Intent) {
            self.app_core.apply_intent(&mut self.scene, intent);
            let RebuildOutcome::Built(output) = self.app_core.rebuild(&self.scene, DENSITY) else {
                panic!("the rebuilt density is in bounds");
            };
            self.region_dimensions = output.region_dimensions;
            self.recentre_voxels = output.recentre_voxels.voxels();
            self.chunks = output.two_layer_chunks.clone();
        }
    }

    /// A one-Box scene, rebuilt, with a chosen camera. The Box FILLS its grid (its SDF
    /// is negative across the whole `[off, off + grid)` span), so every voxel of a
    /// placed Box is solid — which lets a placement test assert occupancy at the exact
    /// dropped voxel rather than hunting for a curved shape's surface.
    fn placement_fixture(camera: OrbitCamera) -> Fixture {
        let blocks = [4u32, 4, 4];
        let shape = SdfShape::from_blocks(ShapeKind::Box, blocks, 1, DENSITY);
        let scene = Scene::from_geometry(
            GeometryParams {
                shape: ShapeKind::Box,
                size_voxels: shape.size_voxels,
                size_measurements: None,
                voxels_per_block: DENSITY,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        );
        let mut app_core = AppCore::new(camera);
        let RebuildOutcome::Built(output) = app_core.rebuild(&scene, DENSITY) else {
            panic!("the fixture's density is in bounds");
        };
        Fixture {
            app_core,
            scene,
            region_dimensions: output.region_dimensions,
            recentre_voxels: output.recentre_voxels.voxels(),
            chunks: output.two_layer_chunks.clone(),
        }
    }

    /// Is the ABSOLUTE voxel `v` occupied in the resident two-layer chunks? The
    /// chunks are keyed by absolute chunk coordinate (ADR 0008), so this decodes `v`
    /// the same way `pick_voxel`'s occupancy closure does — the independent solidity
    /// oracle the placement tests check the dropped node against.
    fn absolute_voxel_is_solid(chunks: &[([i32; 3], Arc<TwoLayerChunk>)], v: [i64; 3]) -> bool {
        let chunk_extent = (CHUNK_BLOCKS * DENSITY) as i64;
        let coord = v.map(|c| c.div_euclid(chunk_extent) as i32);
        let local = v.map(|c| c.rem_euclid(chunk_extent) as u32);
        chunks
            .iter()
            .find(|(c, _)| *c == coord)
            .is_some_and(|(_, chunk)| chunk.voxel_occupied(local))
    }

    /// The armed tool for these tests: a 2-block Box (small so it is a distinct new
    /// body, not a rescale of the fixture's Box), stone.
    fn tool_shape() -> SdfShape {
        SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, DENSITY)
    }

    /// **Geometry tier + bottom-centre drop.** A cursor over the existing solid resolves to
    /// the OUTER side of the entered face — `absolute_voxel + face_normal` — and the node is
    /// seated FLUSH against that face: anchored along the face's normal axis (its facing
    /// side touches the surface) and centred on the other two. The surface voxel (the cursor
    /// point) always lies inside the placed span `[offset, offset + grid)`, so it is solid
    /// once dropped.
    #[test]
    fn a_cursor_on_geometry_places_a_node_seated_on_the_entered_face() {
        let mut fixture = placement_fixture(OrbitCamera::default());
        // The default iso view centres the Box under the screen centre, so a centre
        // cursor is a guaranteed geometry hit (the picking net proves this framing).
        let cursor = [640.0, 360.0];

        // The first empty voxel just outside the entered face — where the seated node's base
        // must land (continuous seat: base on the surface `hit`, extending outward along the
        // normal, ADR 0027).
        let pick = fixture
            .app_core
            .pick_voxel(cursor, VIEWPORT, &fixture.frame())
            .expect("the centre cursor hits the Box");
        let surface_voxel: [i64; 3] =
            std::array::from_fn(|axis| pick.absolute_voxel[axis] + pick.face_normal[axis] as i64);

        let outcome =
            fixture
                .app_core
                .place_primitive(cursor, VIEWPORT, &fixture.frame(), &fixture.scene, tool_shape(), MaterialChoice::Stone, true, PlacementSnap::default());

        assert!(
            matches!(outcome.target, PlacementTarget::OnSurface { .. }),
            "a geometry hit is OnSurface, got {:?}",
            outcome.target
        );
        let intent = outcome.intent.expect("a geometry hit produces a PlaceNode");

        // Apply the ACTUAL returned intent (rotation and all) and rebuild — the end-to-end
        // frame check: the seated node must occupy the empty neighbour just outside the entered
        // face, proving both the corner-anchored seat and that its `offset_voxels` lines up with
        // the resident chunks' frame (a wrong seat or a lost recentre term misses this voxel).
        fixture.apply_and_rebuild(intent);
        assert!(
            absolute_voxel_is_solid(&fixture.chunks, surface_voxel),
            "the dropped node must occupy the neighbour voxel just outside the face {surface_voxel:?}"
        );
    }

    /// **World-plane tier + the frame guard.** A top-down cursor aimed OFF the object
    /// misses geometry, resolves `OnWorldPlane { Ground }`, and drops a node that sits
    /// on the ground straddling the clicked point.
    ///
    /// This is the frame-conversion guard: the expected placement is derived
    /// INDEPENDENTLY here (unproject the same cursor, rebase by `recentre_voxels`,
    /// intersect `z = 0`, floor), so a wrong rebase term inside `place_primitive`
    /// (e.g. the shading `recentre − half` instead of the absolute `+ recentre`) shifts
    /// the ground point by half the region and this assertion fails loudly.
    #[test]
    fn a_cursor_over_the_ground_places_a_node_on_it() {
        // Straight-down orthographic, framed loosely so the Box occupies the middle of
        // the viewport and its sides are empty screen. Orthographic keeps the depth
        // authorable at any reach and maps the screen linearly to world XY.
        let camera = OrbitCamera {
            target: Vec3::ZERO,
            orbit_theta: -std::f32::consts::FRAC_PI_2,
            orbit_phi: 0.0, // top pole: looking straight down −Z
            orbit_distance: 60.0,
            roll: 0.0,
            projection_mode: ProjectionMode::Orthographic,
        };
        let fixture = placement_fixture(camera);
        // Aim near the right edge: well outside the Box's centred silhouette, so the
        // ray misses geometry and falls through to the ground plane.
        let cursor = [1200.0, 360.0];

        // Independent expected placement (the documented absolute-frame formula).
        let aspect_ratio = VIEWPORT[2] / VIEWPORT[3];
        let view_projection = fixture.app_core.view_projection(aspect_ratio, fixture.region_dimensions);
        let ndc_x = (cursor[0] - VIEWPORT[0]) / VIEWPORT[2] * 2.0 - 1.0;
        let ndc_y = 1.0 - (cursor[1] - VIEWPORT[1]) / VIEWPORT[3] * 2.0;
        let render_ray = unproject_screen_point_to_ray(view_projection, ndc_x, ndc_y)
            .expect("the ortho matrix inverts");
        let recentre = fixture.recentre_voxels;
        let absolute_origin =
            render_ray.origin + Vec3::new(recentre[0] as f32, recentre[1] as f32, recentre[2] as f32);
        let direction = render_ray.direction; // already unit
        // Intersect the ground plane z = 0 through the origin.
        let t = -absolute_origin.z / direction.z;
        assert!(t > 0.0, "the ground must be in front of the ray (t = {t})");
        let ground_point = absolute_origin + direction * t;
        // The node drops BOTTOM-CENTRED on the ground point (ADR 0027 continuous seat): the
        // authoring pivot (base centre) lands on the ground point, so the corner offset is the
        // pivot minus half the footprint in X/Y, base-aligned in Z — Voxel-snapped to the NEAREST
        // lattice corner (round, not the old floor). The ground point stays inside
        // `[offset, offset + grid)`, so it is solid once dropped.
        let size = tool_shape().size_voxels;
        let expected_offset = [
            (ground_point.x - size[0] as f32 * 0.5).round() as i64,
            (ground_point.y - size[1] as f32 * 0.5).round() as i64,
            ground_point.z.round() as i64,
        ];

        let mut fixture = fixture;
        let outcome = fixture.app_core.place_primitive(
            cursor,
            VIEWPORT,
            &fixture.frame(),
            &fixture.scene,
            tool_shape(),
            MaterialChoice::Stone,
            true,
            PlacementSnap::default(),
        );

        assert!(
            matches!(
                outcome.target,
                PlacementTarget::OnWorldPlane { plane: raycast::WorldPlane::Ground, .. }
            ),
            "a top-down empty-space cursor lands on the ground, got {:?}",
            outcome.target
        );
        let Some(Intent::PlaceNode { offset_voxels, .. }) = outcome.intent.clone() else {
            panic!("a world-plane hit produces a PlaceNode, got {:?}", outcome.intent);
        };
        assert_eq!(
            offset_voxels, expected_offset,
            "the ground placement must be bottom-centred on the independently-derived ground \
             voxel — a wrong recentre term (or a lost centre offset) fails here"
        );

        // Applying + rebuilding leaves BOTH bodies present: the original Box at the origin
        // and the new Box standing bottom-centred on the ground point. The voxel CONTAINING the
        // clicked point (its floor) must be solid — the pivot sits inside the placed footprint.
        let clicked_voxel = [
            ground_point.x.floor() as i64,
            ground_point.y.floor() as i64,
            ground_point.z.floor() as i64,
        ];
        fixture.apply_and_rebuild(outcome.intent.unwrap());
        assert!(
            absolute_voxel_is_solid(&fixture.chunks, clicked_voxel),
            "the dropped ground node's bottom-centre occupies the clicked point {clicked_voxel:?}"
        );
        assert!(
            absolute_voxel_is_solid(&fixture.chunks, [16, 16, 16]),
            "the original Box (absolute [0,32)^3) is still present after the placement"
        );
    }

    /// **The empty scene still places on the ground.** With NO resident chunks, tier 1
    /// (`pick_voxel`) misses and the cursor falls through to the world-plane tier — so an
    /// armed tool previews and drops on the ground before anything is built. This guards
    /// the regression the shell had (a `!resident_chunks.is_empty()` gate suppressed the
    /// ghost on an empty scene); the logic core never needed chunks for the ground tier,
    /// and this pins that so no future gate can re-hide it.
    #[test]
    fn a_cursor_over_the_ground_of_an_empty_scene_places_a_node() {
        // Same top-down ortho framing as the populated ground test, so the ray meets the
        // ground plane in front of it — the only difference is an EMPTY resident set.
        let camera = OrbitCamera {
            target: Vec3::ZERO,
            orbit_theta: -std::f32::consts::FRAC_PI_2,
            orbit_phi: 0.0,
            orbit_distance: 60.0,
            roll: 0.0,
            projection_mode: ProjectionMode::Orthographic,
        };
        let fixture = placement_fixture(camera);
        // Reuse the fixture's frame geometry (region dims + recentre) but strip the
        // chunks: the tool is armed on a scene with nothing resident.
        let empty_frame = PickFrame { chunks: &[], ..fixture.frame() };

        let outcome = fixture.app_core.place_primitive(
            [1200.0, 360.0],
            VIEWPORT,
            &empty_frame,
            &fixture.scene,
            tool_shape(),
            MaterialChoice::Stone,
            true,
            PlacementSnap::default(),
        );

        assert!(
            matches!(
                outcome.target,
                PlacementTarget::OnWorldPlane { plane: raycast::WorldPlane::Ground, .. }
            ),
            "an empty-scene cursor over the ground lands on it, got {:?}",
            outcome.target
        );
        assert!(
            matches!(outcome.intent, Some(Intent::PlaceNode { .. })),
            "the empty-scene ground placement still produces a PlaceNode, got {:?}",
            outcome.intent
        );
    }

    /// **A downward cursor over visible ground never spuriously misses (bug 6),
    /// on BOTH projections at every angle.** The unprojected cursor ray's near-plane
    /// point sweeps a wide z-range at a far zoom and its lower half can sit past the
    /// ground plane; judging "in front" from that point made a downward foreground ray
    /// report `NoSurface` ("point toward the ground") even while the camera looked
    /// straight at the ground — placement silently died across the lower half of the
    /// screen. `place_primitive` now resolves reachability correctly per projection
    /// (perspective casts from the eye; orthographic asks the uniform directional
    /// question), so a view that faces the ground places across the WHOLE viewport.
    ///
    /// This sweeps the viewport for both projections across a spread of downward pitches,
    /// at a distance far enough that the pre-fix near-plane point sank below the ground
    /// (the exact condition of the bug), and asserts not one cursor over a ground-facing
    /// view returns `NoSurface`, and that real placements happen.
    ///
    /// **STEEP views only (updated 2026-07-21).** Since the invisible vertical planes are no
    /// longer a placement target, a SHALLOW view's corner rays — which graze the ground and
    /// used to fall back to a vertical — now correctly report `NoSurface`. The bug-6 guard is
    /// about the ground being reachable across the foreground where the view *faces it*, so
    /// this now sweeps steep pitches (the ground well-faced everywhere on screen); the shallow
    /// grazing case is covered by `a_grazing_ray_no_longer_places_on_an_invisible_vertical_plane`.
    #[test]
    fn a_downward_cursor_over_ground_never_misses_in_either_projection() {
        for projection_mode in [ProjectionMode::Perspective, ProjectionMode::Orthographic] {
            // STEEP downward views: phi is the polar angle from the top pole, so 0.15..0.45 rad
            // is a near-top-down look where the ground is well-faced across the WHOLE screen —
            // none may report NoSurface (the near-plane-origin regression).
            for &orbit_phi in &[0.15_f32, 0.30, 0.45] {
                let camera = OrbitCamera {
                    target: Vec3::ZERO,
                    orbit_theta: 0.6,
                    orbit_phi,
                    // Far enough that the near plane (`orbit_distance - scene_radius -
                    // margin`) dips below the ground for the foreground — the pre-fix bug's
                    // trigger — while the near foreground stays inside the authorable limit
                    // (density 8 ⇒ ~772) so the fixed path yields real placements.
                    orbit_distance: 500.0,
                    roll: 0.0,
                    projection_mode,
                };
                let fixture = placement_fixture(camera);
                let mut placements = 0;
                for row in 0..20 {
                    for col in 0..20 {
                        let cursor = [
                            VIEWPORT[0] + VIEWPORT[2] * (col as f32 + 0.5) / 20.0,
                            VIEWPORT[1] + VIEWPORT[3] * (row as f32 + 0.5) / 20.0,
                        ];
                        let target = fixture
                            .app_core
                            .place_primitive(cursor, VIEWPORT, &fixture.frame(), &fixture.scene, tool_shape(), MaterialChoice::Stone, true, PlacementSnap::default())
                            .target;
                        assert_ne!(
                            target,
                            PlacementTarget::NoSurface,
                            "{projection_mode:?} phi={orbit_phi}: cursor {cursor:?} faces the \
                             ground but reported NoSurface"
                        );
                        if matches!(
                            target,
                            PlacementTarget::OnWorldPlane { .. } | PlacementTarget::OnSurface { .. }
                        ) {
                            placements += 1;
                        }
                    }
                }
                assert!(
                    placements > 0,
                    "{projection_mode:?} phi={orbit_phi}: a ground-facing view must actually place"
                );
            }
        }
    }

    /// **A geometry drop ALWAYS seats to the surface, whatever the orientation-snap setting (ADR
    /// 0027, owner ruling 2026-07-21).** "Seated placement": a node on a geometry surface contacts
    /// it with the surface's own normal — there is NO upright mode on geometry (upright is the
    /// world-plane tier's job). So both snap settings tilt the node's local +Z to the entered face
    /// normal; the setting will only pick the ANGLE-snap granularity (continuous vs 15°) once slice
    /// 6 wires it. (Guards the 2026-07-21 regression where `NoSnap` wrongly stood the node upright
    /// on geometry — burying it in a vertical wall / dropping it off the curved surface.)
    #[test]
    fn a_geometry_drop_always_seats_to_the_surface() {
        let fixture = placement_fixture(OrbitCamera::default());
        let cursor = [640.0, 360.0];
        let shape = SdfShape::from_blocks(ShapeKind::Cylinder, [1, 1, 3], 1, DENSITY);

        for angle in [AngleSnap::Continuous, AngleSnap::Deg15] {
            let outcome = fixture.app_core.place_primitive(
                cursor, VIEWPORT, &fixture.frame(), &fixture.scene, shape.clone(), MaterialChoice::Stone, true,
                PlacementSnap { position: PositionSnap::Voxel, angle, ..PlacementSnap::default() },
            );
            let (PlacementTarget::OnSurface { face_normal, .. }, Some(Intent::PlaceNode { rotation_quaternion, .. })) =
                (outcome.target, outcome.intent)
            else {
                panic!("{angle:?}: a geometry hit seats OnSurface with a PlaceNode");
            };
            let normal = Vec3::new(face_normal[0] as f32, face_normal[1] as f32, face_normal[2] as f32);
            let axis = Quat::from_array(rotation_quaternion.expect("a seated drop carries the tilt")) * Vec3::Z;
            assert!(
                axis.dot(normal) > 0.99,
                "{angle:?}: a geometry drop must seat — its +Z axis {axis:?} tilts to the surface normal {normal:?}"
            );
        }
    }

    /// **The 15° joint solve lands a quantized normal on a curved surface (ADR 0027 §2).** On a
    /// cylinder the constant-normal contour is a vertical line, so a 15° target is reachable: a drop
    /// seeded at an off-lattice angle (37° around) must slide to a contact whose seated normal is ON
    /// the 15° lattice while staying seated on the surface. Continuous placement would keep the raw
    /// 37° normal; this pins that `Deg15` actually quantizes. (Exercises the free-fn solver directly,
    /// off a synthetic field, so it needs no camera framing — the render loop verifies the wired
    /// path.)
    #[test]
    fn the_15deg_joint_solve_lands_a_quantized_normal_on_a_curved_surface() {
        // A true distance field for a cylinder about world Z.
        let radius = 6.0_f32;
        let field = |p: Vec3| (p.x * p.x + p.y * p.y).sqrt() - radius;
        let seed_angle = 37.0_f32.to_radians();
        let hit = Vec3::new(radius * seed_angle.cos(), radius * seed_angle.sin(), 3.0);
        let full = Vec3::new(2.0, 2.0, 6.0);
        let snap = PlacementSnap {
            position: PositionSnap::NoSnap,
            angle: AngleSnap::Deg15,
            pivot: PlacementPivot::Base,
        };

        let (contact, seat_normal) = solve_seated_15deg(hit, full, snap, 0.0, &field);
        // The seated normal is a fixed point of the 15° quantization — it is ON the lattice.
        let requantized = raycast::quantize_normal_to_15deg(seat_normal);
        assert!(
            seat_normal.dot(requantized) > 0.9999,
            "the seated normal {seat_normal:?} must lie on the 15° lattice"
        );
        // The solved contact is still on the surface.
        assert!(
            field(contact).abs() < 1.0e-2,
            "the solved contact must stay seated on the surface, field = {}",
            field(contact)
        );
        // The raw seed normal (37° azimuth) was NOT on the lattice, so the solve genuinely moved.
        let continuous = raycast::gradient_normal(hit, field);
        assert!(
            continuous.dot(raycast::quantize_normal_to_15deg(continuous)) < 0.9999,
            "the seed's continuous normal must be off-lattice, else the test proves nothing"
        );
    }

    /// **Block position snap rounds the drop to block boundaries; voxel / no-snap keep the
    /// finest offset.** The offset math directly (owner ruling 2026-07-21).
    #[test]
    fn block_snap_rounds_the_offset_to_block_boundaries() {
        // Density 8: each axis rounds to the nearest multiple of 8 (round-half via +d/2).
        assert_eq!(snap_offset([3, 12, -5], PositionSnap::Block, 8), [0, 16, -8]);
        assert_eq!(snap_offset([3, 12, -5], PositionSnap::Voxel, 8), [3, 12, -5]);
        assert_eq!(snap_offset([3, 12, -5], PositionSnap::NoSnap, 8), [3, 12, -5]);
        // Already block-aligned stays put.
        assert_eq!(snap_offset([16, -8, 0], PositionSnap::Block, 8), [16, -8, 0]);
    }

    /// **An invisible world plane is not a placement target (owner ruling 2026-07-21).** The
    /// same top-down cursor that lands on the ground when its floor grid is shown reports
    /// `NoSurface` (no intent) when it is hidden — a hidden plane can't be placed on.
    #[test]
    fn a_hidden_ground_plane_places_nothing() {
        let camera = OrbitCamera {
            target: Vec3::ZERO,
            orbit_theta: -std::f32::consts::FRAC_PI_2,
            orbit_phi: 0.0, // straight down
            orbit_distance: 60.0,
            roll: 0.0,
            projection_mode: ProjectionMode::Orthographic,
        };
        let fixture = placement_fixture(camera);
        let cursor = [1200.0, 360.0]; // off the box, onto the ground
        // Visible → lands on the ground.
        let visible = fixture.app_core.place_primitive(
            cursor, VIEWPORT, &fixture.frame(), &fixture.scene, tool_shape(), MaterialChoice::Stone, true, PlacementSnap::default(),
        );
        assert!(
            matches!(visible.target, PlacementTarget::OnWorldPlane { plane: raycast::WorldPlane::Ground, .. }),
            "ground visible ⇒ places on it, got {:?}", visible.target
        );
        // Hidden → nothing to place on.
        let hidden = fixture.app_core.place_primitive(
            cursor, VIEWPORT, &fixture.frame(), &fixture.scene, tool_shape(), MaterialChoice::Stone, false, PlacementSnap::default(),
        );
        assert_eq!(hidden.target, PlacementTarget::NoSurface, "hidden ground ⇒ NoSurface");
        assert_eq!(hidden.intent, None, "hidden ground drops no node");
    }

    /// **A grazing ray no longer drops a node on an invisible vertical plane.** The two
    /// vertical world planes are never visualized, so a near-horizontal view that used to
    /// fall back to one (dropping a node vertical and centred on a far invisible plane —
    /// the 2026-07-21 bug) now reports `NoSurface`, even with the ground's floor grid on.
    #[test]
    fn a_grazing_ray_no_longer_places_on_an_invisible_vertical_plane() {
        // Near-horizontal orthographic view: the ground grazes, so `select_world_plane`
        // would choose a vertical — which is invisible, hence not a target.
        let camera = OrbitCamera {
            target: Vec3::ZERO,
            orbit_theta: 0.3,
            orbit_phi: 1.49, // ~85° — nearly horizontal, the dump's pose
            orbit_distance: 120.0,
            roll: 0.0,
            projection_mode: ProjectionMode::Orthographic,
        };
        let fixture = placement_fixture(camera);
        let empty = PickFrame { chunks: &[], ..fixture.frame() };
        // Sweep the viewport; with the verticals suppressed, NONE may drop on a world plane.
        for row in 0..8 {
            for col in 0..8 {
                let cursor = [VIEWPORT[2] * (col as f32 + 0.5) / 8.0, VIEWPORT[3] * (row as f32 + 0.5) / 8.0];
                let outcome = fixture.app_core.place_primitive(
                    cursor, VIEWPORT, &empty, &fixture.scene, tool_shape(), MaterialChoice::Stone, true, PlacementSnap::default(),
                );
                assert!(
                    !matches!(
                        outcome.target,
                        PlacementTarget::OnWorldPlane { plane: raycast::WorldPlane::VerticalFacingX | raycast::WorldPlane::VerticalFacingY, .. }
                    ),
                    "a grazing ray placed on an invisible vertical plane: {:?}", outcome.target
                );
            }
        }
    }

    /// **Looking at the sky is `NoSurface`, no intent.** A camera aimed straight up has
    /// the ground plane behind the ray, so there is nothing in front to place on — the
    /// honest answer, with no node dropped.
    #[test]
    fn a_cursor_at_the_sky_places_nothing() {
        // Eye ABOVE the object looking straight UP (+Z): phi = π gives direction
        // (0,0,−1), so forward = −direction = +Z. Casting from the eye, the ground plane
        // is behind the ray, and the object sits below the eye so a centre cursor clears
        // it. (The orthographic counterpart is `an_orthographic_skyward_cursor_places_nothing`,
        // which the directional reachability test now answers correctly too.)
        let camera = OrbitCamera {
            target: Vec3::new(0.0, 0.0, 50.0),
            orbit_theta: 0.0,
            orbit_phi: std::f32::consts::PI,
            orbit_distance: 20.0,
            roll: 0.0,
            projection_mode: ProjectionMode::Perspective,
        };
        let fixture = placement_fixture(camera);
        let outcome = fixture.app_core.place_primitive(
            [640.0, 360.0],
            VIEWPORT,
            &fixture.frame(),
            &fixture.scene,
            tool_shape(),
            MaterialChoice::Stone,
            true,
            PlacementSnap::default(),
        );
        assert_eq!(
            outcome.target,
            PlacementTarget::NoSurface,
            "a skyward cursor has no surface to place on"
        );
        assert_eq!(outcome.intent, None, "NoSurface drops no node");
    }

    /// **Orthographic looking up is `NoSurface` too — the directional reachability test.**
    /// The old ortho path judged "in front" from the near-plane point and got this wrong
    /// (it would place on the ground behind the view); the fix asks the uniform directional
    /// question — the eye must sit on the plane's front side while looking toward it — so an
    /// upward orthographic view correctly finds nothing to place on. Uses an EMPTY scene so
    /// no geometry can mask the world-plane answer.
    #[test]
    fn an_orthographic_skyward_cursor_places_nothing() {
        // Eye above the ground looking straight UP (+Z), orthographic. The ground is behind
        // the view, so every parallel ray faces away from it.
        let camera = OrbitCamera {
            target: Vec3::new(0.0, 0.0, 50.0),
            orbit_theta: 0.0,
            orbit_phi: std::f32::consts::PI,
            orbit_distance: 20.0,
            roll: 0.0,
            projection_mode: ProjectionMode::Orthographic,
        };
        let fixture = placement_fixture(camera);
        // Empty resident set: tier 1 (geometry) misses, isolating the world-plane tier.
        let empty_frame = PickFrame { chunks: &[], ..fixture.frame() };
        // Sweep the whole viewport — the directional answer is uniform, so NONE may place.
        for row in 0..8 {
            for col in 0..8 {
                let cursor = [
                    VIEWPORT[2] * (col as f32 + 0.5) / 8.0,
                    VIEWPORT[3] * (row as f32 + 0.5) / 8.0,
                ];
                let outcome = fixture.app_core.place_primitive(
                    cursor, VIEWPORT, &empty_frame, &fixture.scene, tool_shape(), MaterialChoice::Stone, true, PlacementSnap::default(),
                );
                assert_eq!(
                    outcome.target,
                    PlacementTarget::NoSurface,
                    "an orthographic skyward cursor {cursor:?} must find no surface"
                );
            }
        }
    }
}
