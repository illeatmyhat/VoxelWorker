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

use camera::unproject_screen_point_to_ray;
use glam::Vec3;
use raycast::{resolve_placement, PlacementTarget};
use substrate::spatial::Ray;

use document::intent::{Intent, NodeSpec};
use document::voxel::SdfShape;
use voxel_core::core_geom::MaterialChoice;

use super::picking::PickFrame;
use super::AppCore;

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
    pub fn place_primitive(
        &self,
        cursor: [f32; 2],
        viewport: [f32; 4],
        frame: &PickFrame<'_>,
        shape: SdfShape,
        material: MaterialChoice,
    ) -> PlacementOutcome {
        let place_node = |offset_voxels: [i64; 3]| Intent::PlaceNode {
            content: NodeSpec::Tool {
                shape: shape.clone(),
                material,
            },
            offset_voxels,
        };

        // Tier 1 — geometry. A picked surface is unambiguous; the node lands on the
        // OUTER side of the entered face (the empty neighbour), so the placement voxel
        // is the hit voxel stepped one unit along its outward face normal. Both are in
        // the absolute lattice (ADR 0008), so this is exact integer arithmetic.
        if let Some(pick) = self.pick_voxel(cursor, viewport, frame) {
            let placement_voxel = std::array::from_fn(|axis| {
                pick.absolute_voxel[axis] + pick.face_normal[axis] as i64
            });
            let point = Vec3::new(
                placement_voxel[0] as f32,
                placement_voxel[1] as f32,
                placement_voxel[2] as f32,
            );
            return PlacementOutcome {
                target: PlacementTarget::OnSurface {
                    point,
                    face_normal: pick.face_normal,
                },
                intent: Some(place_node(placement_voxel)),
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

        // Precision caveat (ADR 0008): `recentre_voxels as f32` loses integer
        // precision past ~16M voxels. Correct for the small scenes this placement
        // slice targets; the eventual fix is the i64 origin-rebase, not a fudge here.
        let recentre = frame.recentre_voxels;
        let absolute_ray = Ray::new(
            render_ray.origin
                + Vec3::new(recentre[0] as f32, recentre[1] as f32, recentre[2] as f32),
            render_ray.direction,
        );

        // A block spans `density` voxels in this frame, so the authorability limit is
        // asked in voxel units with the density as the block size.
        let block_size = frame.density.max(1) as f32;
        let target = resolve_placement(None, absolute_ray, MIN_GROUND_FACING, |depth| {
            self.camera.depth_is_authorable(depth, block_size)
        });

        let intent = match target {
            PlacementTarget::OnWorldPlane { point, .. } => Some(place_node([
                point.x.floor() as i64,
                point.y.floor() as i64,
                point.z.floor() as i64,
            ])),
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

    /// **Geometry tier.** A cursor over the existing solid places a node on the OUTER
    /// side of the entered face — `absolute_voxel + face_normal` — and the dropped Box
    /// occupies that voxel once applied and rebuilt.
    ///
    /// **Empirical centre-vs-corner (the briefing's open question).** The producer is
    /// CORNER-anchored: a node with `offset_voxels = V` occupies absolute
    /// `[V, V + grid)`, so `V` itself is solid for a grid-filling Box and NO
    /// `size/2` subtraction is applied. This test would fail (the placed voxel would
    /// be air, off by half the node's size) if the producer centre-emitted; it passes,
    /// confirming corner emission — consistent with `placed_extent_voxels`' documented
    /// `[off, off + grid)` span.
    #[test]
    fn a_cursor_on_geometry_places_a_node_against_the_entered_face() {
        let mut fixture = placement_fixture(OrbitCamera::default());
        // The default iso view centres the Box under the screen centre, so a centre
        // cursor is a guaranteed geometry hit (the picking net proves this framing).
        let cursor = [640.0, 360.0];

        // Independent expectation: pick the same cursor and step along the face.
        let pick = fixture
            .app_core
            .pick_voxel(cursor, VIEWPORT, &fixture.frame())
            .expect("the centre cursor hits the Box");
        let expected_voxel: [i64; 3] =
            std::array::from_fn(|axis| pick.absolute_voxel[axis] + pick.face_normal[axis] as i64);

        let outcome =
            fixture
                .app_core
                .place_primitive(cursor, VIEWPORT, &fixture.frame(), tool_shape(), MaterialChoice::Stone);

        assert!(
            matches!(outcome.target, PlacementTarget::OnSurface { .. }),
            "a geometry hit is OnSurface, got {:?}",
            outcome.target
        );
        let Some(Intent::PlaceNode { offset_voxels, .. }) = outcome.intent else {
            panic!("a geometry hit produces a PlaceNode, got {:?}", outcome.intent);
        };
        assert_eq!(
            offset_voxels, expected_voxel,
            "the node lands on the outer side of the entered face"
        );

        // The placed Box is solid at its low corner (grid-filling), so applying the
        // intent and rebuilding must make the dropped voxel occupied — proving the
        // absolute `offset_voxels` frame lines up with the resident chunks' frame.
        fixture.apply_and_rebuild(Intent::PlaceNode {
            content: NodeSpec::Tool { shape: tool_shape(), material: MaterialChoice::Stone },
            offset_voxels,
        });
        assert!(
            absolute_voxel_is_solid(&fixture.chunks, expected_voxel),
            "the dropped node must occupy the placed voxel {expected_voxel:?}"
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
        let expected_voxel = [
            ground_point.x.floor() as i64,
            ground_point.y.floor() as i64,
            ground_point.z.floor() as i64,
        ];

        let mut fixture = fixture;
        let outcome = fixture.app_core.place_primitive(
            cursor,
            VIEWPORT,
            &fixture.frame(),
            tool_shape(),
            MaterialChoice::Stone,
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
            offset_voxels, expected_voxel,
            "the ground placement must match the independently-derived absolute voxel — \
             a wrong recentre term fails here"
        );

        // Applying + rebuilding leaves BOTH bodies present: the original Box at the
        // origin and the new Box straddling the ground point.
        fixture.apply_and_rebuild(outcome.intent.unwrap());
        assert!(
            absolute_voxel_is_solid(&fixture.chunks, expected_voxel),
            "the dropped ground node occupies the clicked point {expected_voxel:?}"
        );
        assert!(
            absolute_voxel_is_solid(&fixture.chunks, [16, 16, 16]),
            "the original Box (absolute [0,32)^3) is still present after the placement"
        );
    }

    /// **Looking at the sky is `NoSurface`, no intent.** A camera aimed straight up has
    /// the ground plane behind the ray, so there is nothing in front to place on — the
    /// honest answer, with no node dropped.
    #[test]
    fn a_cursor_at_the_sky_places_nothing() {
        // Eye ABOVE the object looking straight UP (+Z): phi = π gives direction
        // (0,0,−1), so forward = −direction = +Z. With target above the eye and a
        // perspective near plane at the eye, the cursor ray starts above the ground and
        // travels up — the ground plane is behind it, and the object sits below the eye
        // so a centre cursor also clears it. (Orthographic fails here: its near plane is
        // on the −Z side, putting the ray origin below the ground.)
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
        );
        assert_eq!(
            outcome.target,
            PlacementTarget::NoSurface,
            "a skyward cursor has no surface to place on"
        );
        assert_eq!(outcome.intent, None, "NoSurface drops no node");
    }
}
