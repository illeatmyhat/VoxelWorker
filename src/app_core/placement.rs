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
use glam::Vec3;
use raycast::{resolve_placement, select_world_plane, world_plane_hit, PlacementTarget};
use substrate::spatial::Ray;

use document::intent::{Intent, NodeSpec};
use document::voxel::SdfShape;
use ui::panel::{OrientationSnap, PlacementSnap, PositionSnap};
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

/// The grazing threshold [`resolve_placement`] selects a world plane by:
/// `sin(20°) ≈ 0.342`, the smallest `|ray·plane_normal|` at which a plane is still
/// worth placing on. Must stay `≤ 1/√3` — the bound under which the three fixed
/// normals are guaranteed to keep one plane well-faced for any view direction (see
/// `raycast::select_world_plane`).
const MIN_GROUND_FACING: f32 = 0.342_f32;

/// Turn a placement POINT (the absolute neighbour voxel the cursor resolved to) and the
/// FACE it was entered through into the corner-anchored `offset_voxels` that seats the
/// object flush against that face, centred on it. The object is anchored along the face's
/// NORMAL axis (its facing side touches the surface) and centred on the other two — so a
/// box on a wall sits flush against the wall centred on the click, not half-buried, and a
/// box on a top face (or the ground, normal `+Z`) stands on it centred under the cursor.
///
/// Producers are corner-anchored (`[offset, offset + grid)`, [`placed_extent_voxels`]), so
/// on the anchored axis a `+` face seats the object's LOW face at the neighbour
/// (`offset[axis] = point[axis]`) and a `-` face seats its HIGH face there
/// (`offset[axis] = point[axis] - (size[axis] - 1)`); the other two axes centre
/// (`offset = point - size/2`). This replaces the old Z-only bottom-centre (which is
/// exactly the `+Z` case here) so a side face no longer mis-centres the object into the wall.
fn face_anchored_offset(point: [i64; 3], size_voxels: [u32; 3], face_normal: [i32; 3]) -> [i64; 3] {
    std::array::from_fn(|axis| {
        let size = size_voxels[axis] as i64;
        match face_normal[axis] {
            // The anchored axis: seat the object's facing side flush at the neighbour.
            n if n > 0 => point[axis],
            n if n < 0 => point[axis] - (size - 1),
            // A perpendicular axis: centre the object on the click.
            _ => point[axis] - size / 2,
        }
    })
}

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
        shape: SdfShape,
        material: MaterialChoice,
        ground_plane_visible: bool,
        snap: PlacementSnap,
    ) -> PlacementOutcome {
        // ADR 0026: orientation is derived from the entered face in the geometry tier below;
        // a world-plane drop stays identity (world-vertical). Defaulted here, set per-tier.
        let place_node = |offset_voxels: [i64; 3], orientation: substrate::spatial::LatticeOrientation| {
            Intent::PlaceNode {
                content: NodeSpec::Tool {
                    shape: shape.clone(),
                    material,
                },
                offset_voxels,
                orientation,
            }
        };

        // Tier 1 — geometry. A picked surface is unambiguous; the node lands on the
        // OUTER side of the entered face (the empty neighbour), so the placement voxel
        // is the hit voxel stepped one unit along its outward face normal. Both are in
        // the absolute lattice (ADR 0008), so this is exact integer arithmetic. The node
        // is then dropped BOTTOM-CENTRED on that voxel (centre X/Y, bottom-align Z), so it
        // stands on the face under the cursor rather than hanging by its low corner.
        if let Some(pick) = self.pick_voxel(cursor, viewport, frame) {
            let placement_voxel = std::array::from_fn(|axis| {
                pick.absolute_voxel[axis] + pick.face_normal[axis] as i64
            });
            // `target.point` stays the surface hit (the cursor location) for the
            // affordance; only the emitted `offset_voxels` is face-anchored.
            let point = Vec3::new(
                placement_voxel[0] as f32,
                placement_voxel[1] as f32,
                placement_voxel[2] as f32,
            );
            // ADR 0026 + owner ruling 2026-07-21: the entered face sets orientation ONLY when
            // the armed tool asks to snap to the surface — then the node's local +Z turns to the
            // face normal (a cylinder on a wall lies on its side, seated flush by its TURNED
            // extent). With orientation snap off, the node stays upright (identity) and seats by
            // its un-turned extent.
            let orientation = match snap.orientation {
                OrientationSnap::Surface => {
                    substrate::spatial::LatticeOrientation::from_face_normal(pick.face_normal)
                }
                OrientationSnap::NoSnap => substrate::spatial::LatticeOrientation::IDENTITY,
            };
            let turned_size = orientation.turn_extent(shape.size_voxels);
            let offset = snap_offset(
                face_anchored_offset(placement_voxel, turned_size, pick.face_normal),
                snap.position,
                frame.density,
            );
            return PlacementOutcome {
                target: PlacementTarget::OnSurface {
                    point,
                    face_normal: pick.face_normal,
                },
                intent: Some(place_node(offset, orientation)),
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
            // The world planes never orient (they only ever position, standing the object
            // upright — the design's "world-vertical" rule). The ground is a `+Z` face, so
            // the object stands bottom-centred on the hit under the cursor.
            PlacementTarget::OnWorldPlane { point, plane } => Some(place_node(
                snap_offset(
                    face_anchored_offset(
                        [point.x.floor() as i64, point.y.floor() as i64, point.z.floor() as i64],
                        shape.size_voxels,
                        plane.normal().to_array().map(|n| n as i32),
                    ),
                    snap.position,
                    frame.density,
                ),
                // The world planes never orient (ADR 0026): the node stays world-vertical.
                substrate::spatial::LatticeOrientation::IDENTITY,
            )),
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

    /// **An oriented leaf's occupancy is the TURNED occupancy of the un-oriented one**
    /// (ADR 0026) — the definitive classifier proof. A tall Cylinder (axis-locked to local Z)
    /// turned onto +X must occupy exactly what the upright cylinder occupies after each voxel
    /// is turned by the same rotation. This exercises the whole evaluation path: the turned
    /// world extent, the INVERSE-permuted SDF sample (so the curved *field* turns, not merely
    /// its bounding box), and the forward-permuted voxel emission.
    #[test]
    fn an_oriented_leaf_occupies_the_turned_cells_of_the_upright_one() {
        use std::collections::HashSet;
        use substrate::spatial::LatticeOrientation;

        // A tall cylinder: radius in XY, axis (height) along local Z — asymmetric, so a turn is
        // observable. Small so the occupancy scan below is cheap.
        let shape = SdfShape::from_blocks(ShapeKind::Cylinder, [1, 1, 3], 1, DENSITY);
        let size = shape.size_voxels; // the producer-local extent
        let offset = [8i64, 8, 8]; // away from the origin, to catch offset bugs

        // Place one cylinder with `orientation` on a fresh scene; collect its solid ABSOLUTE
        // voxels within a box that covers the upright AND any turned extent.
        let occupied = |orientation: LatticeOrientation| -> HashSet<[i64; 3]> {
            let mut scene = Scene::default();
            let mut app_core = AppCore::new(OrbitCamera::default());
            app_core.apply_intent(
                &mut scene,
                Intent::PlaceNode {
                    content: NodeSpec::Tool {
                        shape: shape.clone(),
                        material: MaterialChoice::Stone,
                    },
                    offset_voxels: offset,
                    orientation,
                },
            );
            let RebuildOutcome::Built(output) = app_core.rebuild(&scene, DENSITY) else {
                panic!("the density is in bounds");
            };
            let chunks = output.two_layer_chunks;
            let span = *size.iter().max().unwrap() as i64;
            let mut solids = HashSet::new();
            for x in -1..=span {
                for y in -1..=span {
                    for z in -1..=span {
                        let v = [offset[0] + x, offset[1] + y, offset[2] + z];
                        if absolute_voxel_is_solid(&chunks, v) {
                            solids.insert(v);
                        }
                    }
                }
            }
            solids
        };

        let upright = occupied(LatticeOrientation::IDENTITY);
        assert!(!upright.is_empty(), "the upright cylinder must occupy something");

        // +Z -> +X: the cylinder lies on its side, poking out along +X.
        let turn = LatticeOrientation::from_face_normal([1, 0, 0]);
        let on_its_side = occupied(turn);

        // Expected: each upright cell turned about the leaf's corner-anchored box.
        let expected: HashSet<[i64; 3]> = upright
            .iter()
            .map(|v| {
                let local = [v[0] - offset[0], v[1] - offset[1], v[2] - offset[2]];
                let turned = turn.turn_point_in_box(local, size);
                [offset[0] + turned[0], offset[1] + turned[1], offset[2] + turned[2]]
            })
            .collect();

        assert_eq!(
            on_its_side, expected,
            "the oriented cylinder must occupy exactly the turned cells of the upright one"
        );
        // The turn is REAL, not a no-op: the shape is asymmetric in Z, so its cells move.
        assert_ne!(on_its_side, upright, "a tall cylinder turned onto +X must move its cells");
    }

    /// **place_primitive wires the face turn (ADR 0026).** Whatever face the cursor enters,
    /// the emitted intent's orientation is `from_face_normal(face)` and its offset anchors the
    /// TURNED extent — so a tall cylinder on a side wall lies along the normal, flush and
    /// centred, rather than staying upright and half-buried. (The turn's occupancy correctness
    /// is proven by `an_oriented_leaf_occupies_the_turned_cells_of_the_upright_one`; this pins
    /// the placement wiring that feeds it.)
    #[test]
    fn a_placed_primitive_is_oriented_and_anchored_to_the_entered_face() {
        use substrate::spatial::LatticeOrientation;
        let fixture = placement_fixture(OrbitCamera::default());
        let cursor = [640.0, 360.0];
        // A tall (asymmetric) armed tool, so the turned extent differs from the upright one.
        let shape = SdfShape::from_blocks(ShapeKind::Cylinder, [1, 1, 3], 1, DENSITY);

        // Independent expectation from the pick: turn by the entered face, anchor the turned size.
        let pick = fixture
            .app_core
            .pick_voxel(cursor, VIEWPORT, &fixture.frame())
            .expect("the centre cursor hits the Box");
        let placement_voxel: [i64; 3] =
            std::array::from_fn(|axis| pick.absolute_voxel[axis] + pick.face_normal[axis] as i64);
        let expected_orientation = LatticeOrientation::from_face_normal(pick.face_normal);
        let expected_offset = face_anchored_offset(
            placement_voxel,
            expected_orientation.turn_extent(shape.size_voxels),
            pick.face_normal,
        );

        let outcome = fixture.app_core.place_primitive(
            cursor,
            VIEWPORT,
            &fixture.frame(),
            shape.clone(),
            MaterialChoice::Stone,
            true,
            PlacementSnap::default(),
        );
        let Some(Intent::PlaceNode { offset_voxels, orientation, .. }) = outcome.intent else {
            panic!("a geometry hit produces a PlaceNode, got {:?}", outcome.intent);
        };
        assert_eq!(orientation, expected_orientation, "oriented by the entered face");
        assert_eq!(offset_voxels, expected_offset, "seated flush by the turned extent");
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
    /// **The face-anchor rule, per normal.** Anchor along the normal axis (the object's
    /// facing side flush at the neighbour), centre on the other two — and the neighbour
    /// voxel always lands inside the placed span `[offset, offset + size)`.
    #[test]
    fn face_anchored_offset_seats_flush_per_normal() {
        let point = [10i64, 20, 30];
        let size = [4u32, 6, 8];
        // +Z (a top face / the ground): stand on it, centred — the old bottom-centre.
        assert_eq!(face_anchored_offset(point, size, [0, 0, 1]), [8, 17, 30]);
        // -Z (a bottom face): hang under it, the object's TOP flush at the neighbour.
        assert_eq!(face_anchored_offset(point, size, [0, 0, -1]), [8, 17, 30 - (8 - 1)]);
        // +X (a side wall): the object's -X face flush against it, centred in Y and Z.
        assert_eq!(face_anchored_offset(point, size, [1, 0, 0]), [10, 17, 26]);
        // -X (the opposite wall): the object's +X face flush.
        assert_eq!(face_anchored_offset(point, size, [-1, 0, 0]), [10 - (4 - 1), 17, 26]);

        // Invariant: the neighbour voxel is inside the placed span for every axis face.
        for normal in [[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]] {
            let off = face_anchored_offset(point, size, normal);
            for axis in 0..3 {
                let inside = point[axis] >= off[axis] && point[axis] < off[axis] + size[axis] as i64;
                assert!(inside, "neighbour {point:?} outside span for normal {normal:?} on axis {axis}");
            }
        }
    }

    /// the OUTER side of the entered face — `absolute_voxel + face_normal` — and the node is
    /// seated FLUSH against that face: anchored along the face's normal axis (its facing
    /// side touches the surface) and centred on the other two. The surface voxel (the cursor
    /// point) always lies inside the placed span `[offset, offset + grid)`, so it is solid
    /// once dropped.
    #[test]
    fn a_cursor_on_geometry_places_a_node_anchored_flush_to_the_entered_face() {
        let mut fixture = placement_fixture(OrbitCamera::default());
        // The default iso view centres the Box under the screen centre, so a centre
        // cursor is a guaranteed geometry hit (the picking net proves this framing).
        let cursor = [640.0, 360.0];

        // Independent expectation: pick the same cursor, step along the face, then apply the
        // face-anchor rule (anchor along the normal axis, centre on the rest).
        let pick = fixture
            .app_core
            .pick_voxel(cursor, VIEWPORT, &fixture.frame())
            .expect("the centre cursor hits the Box");
        let surface_voxel: [i64; 3] =
            std::array::from_fn(|axis| pick.absolute_voxel[axis] + pick.face_normal[axis] as i64);
        let size = tool_shape().size_voxels;
        let expected_offset: [i64; 3] = std::array::from_fn(|axis| {
            let s = size[axis] as i64;
            match pick.face_normal[axis] {
                n if n > 0 => surface_voxel[axis],
                n if n < 0 => surface_voxel[axis] - (s - 1),
                _ => surface_voxel[axis] - s / 2,
            }
        });

        let outcome =
            fixture
                .app_core
                .place_primitive(cursor, VIEWPORT, &fixture.frame(), tool_shape(), MaterialChoice::Stone, true, PlacementSnap::default());

        assert!(
            matches!(outcome.target, PlacementTarget::OnSurface { .. }),
            "a geometry hit is OnSurface, got {:?}",
            outcome.target
        );
        let Some(Intent::PlaceNode { offset_voxels, .. }) = outcome.intent else {
            panic!("a geometry hit produces a PlaceNode, got {:?}", outcome.intent);
        };
        assert_eq!(
            offset_voxels, expected_offset,
            "the node seats flush against the entered face"
        );

        // Applying + rebuilding must make the CURSOR point (inside the seated span) occupied
        // — proving both the anchor transform and that `offset_voxels` lines up with the
        // resident chunks' frame.
        fixture.apply_and_rebuild(Intent::PlaceNode {
            content: NodeSpec::Tool { shape: tool_shape(), material: MaterialChoice::Stone },
            offset_voxels,
            orientation: substrate::spatial::LatticeOrientation::IDENTITY,
        });
        assert!(
            absolute_voxel_is_solid(&fixture.chunks, surface_voxel),
            "the dropped node must occupy the cursor voxel {surface_voxel:?}"
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
        let ground_voxel = [
            ground_point.x.floor() as i64,
            ground_point.y.floor() as i64,
            ground_point.z.floor() as i64,
        ];
        // The node drops BOTTOM-CENTRED on the ground point: centre X/Y, bottom-align Z. So
        // its corner offset is `ground_voxel − [sx/2, sy/2, 0]`, and the ground point itself
        // is the bottom-centre (still solid, inside `[offset, offset + grid)`).
        let size = tool_shape().size_voxels;
        let expected_offset = [
            ground_voxel[0] - (size[0] / 2) as i64,
            ground_voxel[1] - (size[1] / 2) as i64,
            ground_voxel[2],
        ];

        let mut fixture = fixture;
        let outcome = fixture.app_core.place_primitive(
            cursor,
            VIEWPORT,
            &fixture.frame(),
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
        // and the new Box standing bottom-centred on the ground point.
        fixture.apply_and_rebuild(outcome.intent.unwrap());
        assert!(
            absolute_voxel_is_solid(&fixture.chunks, ground_voxel),
            "the dropped ground node's bottom-centre occupies the clicked point {ground_voxel:?}"
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
                            .place_primitive(cursor, VIEWPORT, &fixture.frame(), tool_shape(), MaterialChoice::Stone, true, PlacementSnap::default())
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

    /// **Orientation snap governs the face turn (owner ruling 2026-07-21).** With orientation
    /// snap OFF the placed node is upright (identity) whatever face it entered; with it on it
    /// orients to that face (ADR 0026). Position snap is Voxel (finest) for both.
    #[test]
    fn orientation_snap_toggles_the_face_turn() {
        use substrate::spatial::LatticeOrientation;
        let fixture = placement_fixture(OrbitCamera::default());
        let cursor = [640.0, 360.0];
        let shape = SdfShape::from_blocks(ShapeKind::Cylinder, [1, 1, 3], 1, DENSITY);

        let upright = fixture.app_core.place_primitive(
            cursor, VIEWPORT, &fixture.frame(), shape.clone(), MaterialChoice::Stone, true,
            PlacementSnap { position: PositionSnap::Voxel, orientation: OrientationSnap::NoSnap },
        );
        let Some(Intent::PlaceNode { orientation, .. }) = upright.intent else {
            panic!("a geometry hit places, got {:?}", upright.intent);
        };
        assert!(orientation.is_identity(), "orientation snap off ⇒ node stays upright");

        let oriented = fixture.app_core.place_primitive(
            cursor, VIEWPORT, &fixture.frame(), shape.clone(), MaterialChoice::Stone, true,
            PlacementSnap { position: PositionSnap::Voxel, orientation: OrientationSnap::Surface },
        );
        let (PlacementTarget::OnSurface { face_normal, .. }, Some(Intent::PlaceNode { orientation, .. })) =
            (oriented.target, oriented.intent)
        else {
            panic!("expected an oriented surface placement");
        };
        assert_eq!(
            orientation,
            LatticeOrientation::from_face_normal(face_normal),
            "orientation snap on ⇒ node turns to the entered face"
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
            cursor, VIEWPORT, &fixture.frame(), tool_shape(), MaterialChoice::Stone, true, PlacementSnap::default(),
        );
        assert!(
            matches!(visible.target, PlacementTarget::OnWorldPlane { plane: raycast::WorldPlane::Ground, .. }),
            "ground visible ⇒ places on it, got {:?}", visible.target
        );
        // Hidden → nothing to place on.
        let hidden = fixture.app_core.place_primitive(
            cursor, VIEWPORT, &fixture.frame(), tool_shape(), MaterialChoice::Stone, false, PlacementSnap::default(),
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
                    cursor, VIEWPORT, &empty, tool_shape(), MaterialChoice::Stone, true, PlacementSnap::default(),
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
                    cursor, VIEWPORT, &empty_frame, tool_shape(), MaterialChoice::Stone, true, PlacementSnap::default(),
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
