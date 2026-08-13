//! The sketch plane's projected frame, measured on the REAL strike at a camera that broke.
//!
//! Every number in this file came out of the owner's own F9 repro dump, and the frame under test
//! is [`voxel_worker::windowed::a_sketch_planes_frame`] — the one the shell draws with. The two
//! measurement rounds that diagnosed the eighth report were run against a Python replica of that
//! arithmetic, which is a second authority over the same question; this file is what retires it.
//!
//! The camera: a `PlaneAxis::Z` sketch (profile on the XY ground plane), `orbit_phi` at pi/2 —
//! the polar angle from +Z — so the eye sits exactly in the plane's own plane and the drawing
//! images as a LINE. Orthographic, `orbit_distance` 1997.97 against a home of 10.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use camera::{OrbitCamera, ProjectionMode};
use ui::gizmos::dimension::PlaneFrame;

/// The dump's camera, field for field. `%TEMP%/voxelworker-repro.json`, written 2026-08-12.
const THETA: f32 = -7.853_982;
const PHI: f32 = 1.570_796_4;
const DISTANCE: f32 = 1997.966_3;
const TARGET: [f32; 3] = [-22.553_219, -30.701_385, -5.110_014];

/// The 3D viewport in physical pixels, and the display's scale. The window was 3840x2054; the
/// viewport is the window less the panels, and the exact split does not matter to what this file
/// asserts — every claim below is either a RATIO between the two projected axes or a length in
/// plane units, and both are free of the viewport's own scale.
const VIEWPORT_PX: [f32; 4] = [0.0, 0.0, 3840.0, 2054.0];
const PIXELS_PER_POINT: f32 = 2.0;

/// The dump's camera as the shell builds it.
fn the_dumps_camera(distance: f32) -> OrbitCamera {
    OrbitCamera {
        target: glam::Vec3::from_array(TARGET),
        orbit_center: glam::Vec3::ZERO,
        orbit_theta: THETA,
        orbit_phi: PHI,
        orbit_distance: distance,
        projection_mode: ProjectionMode::Orthographic,
        ..OrbitCamera::default()
    }
}

/// The frame the shell would strike for this sketch at this camera.
///
/// `profile_to_render` for a `PlaneAxis::Z` sketch carries `(u, v)` to `(u, v, 0)` plus the
/// render frame's recentre offset. The offset is a CONSTANT, so it moves the frame's translation
/// column and leaves its Jacobian — the two projected axes, which is all this file reads —
/// untouched. Supplied directly rather than through a `Scene` so the fixture is the camera and
/// the plane, with no scene graph standing between the assertion and the thing it is about.
fn the_frame_at(distance: f32, scene_radius: f32) -> PlaneFrame {
    let camera = the_dumps_camera(distance);
    let aspect = VIEWPORT_PX[2] / VIEWPORT_PX[3];
    let view_projection = camera.view_projection(aspect, glam::Vec3::ZERO, scene_radius);
    let clip_of = |coord: [f64; 2]| {
        #[allow(clippy::cast_possible_truncation)]
        let vertex = [coord[0] as f32, coord[1] as f32, 0.0];
        view_projection * glam::Vec4::new(vertex[0], vertex[1], vertex[2], 1.0)
    };
    voxel_worker::windowed::a_sketch_planes_frame(&clip_of, VIEWPORT_PX, PIXELS_PER_POINT)
        .expect("the dump's camera strikes a frame")
}

/// How far each of the plane's two unit steps reaches on screen, in points per plane unit.
fn axis_reach(frame: PlaneFrame, at: egui::Pos2) -> [f32; 2] {
    let [across, down] = frame
        .axes_at(at)
        .expect("the frame answers at its own center");
    [across.length(), down.length()]
}

/// Roughly the middle of the viewport in points, which is where the drawing sits.
fn the_middle() -> egui::Pos2 {
    egui::Pos2::new(
        VIEWPORT_PX[2] / PIXELS_PER_POINT / 2.0,
        VIEWPORT_PX[3] / PIXELS_PER_POINT / 2.0,
    )
}

/// **The reported camera really does draw the plane as a line, and the frame says so.**
///
/// The collapse is three million to one. A frame that reported anything else — in particular one
/// that refused to answer and let its caller reach for the screen's own square — would be
/// describing a plane the author is not looking at.
#[test]
fn the_dumps_camera_draws_the_sketch_plane_as_a_line() {
    let [across, down] = axis_reach(the_frame_at(DISTANCE, 64.0), the_middle());
    assert!(
        across > 0.1,
        "the surviving axis must still reach: {across} points per plane unit"
    );
    assert!(
        down / across < 1e-5,
        "the collapsed axis must be gone: {down} against {across}, ratio {}",
        down / across
    );
}

/// **The frame keeps answering however far the author zooms out.**
///
/// This is the eighth report as a gate. `axes_at` used to require both projected axes to exceed
/// `f32::EPSILON` in POINTS PER PLANE UNIT — an absolute threshold on a quantity that shrinks
/// with zoom — so it stopped answering somewhere past `orbit_distance` 3400 and every caller fell
/// back to the screen's own reading at full length. The figure lying in the plane stood up out of
/// it, at a zoom rather than at a geometry.
///
/// **Seen red** at distance 5000 and beyond, where `axes_at` returned `None`.
#[test]
fn the_frame_answers_at_every_zoom_rather_than_declining_at_one() {
    for distance in [10.0, 1000.0, DISTANCE, 5000.0, 20000.0, 100_000.0] {
        let frame = the_frame_at(distance, 64.0);
        assert!(
            frame.axes_at(the_middle()).is_some(),
            "the frame stopped answering at orbit_distance {distance}"
        );
        // And what it answers stays SHORT rather than snapping back to a screen-length square.
        let square = frame.square_to(egui::Vec2::X, the_middle());
        assert!(
            square.length() < 0.01,
            "at orbit_distance {distance} the collapsed square measured {} — a full-length \
             screen perpendicular is 1.0",
            square.length()
        );
    }
}

/// **The projected axes shrink with zoom in proportion, so the collapse is the geometry's and
/// never the zoom's.**
///
/// The distinction the old threshold could not draw. Both axes scale together as the author
/// zooms; what makes this plane degenerate is the RATIO between them, which does not move.
#[test]
fn zoom_scales_both_axes_together_and_leaves_the_collapse_where_it_was() {
    let near = axis_reach(the_frame_at(1000.0, 64.0), the_middle());
    let far = axis_reach(the_frame_at(10000.0, 64.0), the_middle());
    let shrink = near[0] / far[0];
    assert!(
        (shrink - 10.0).abs() < 0.1,
        "ten times the distance must be a tenth of the reach, got {shrink}"
    );
    assert!(
        (near[1] / near[0]).max(far[1] / far[0]) < 1e-5,
        "the ratio is the geometry and stays put: {} then {}",
        near[1] / near[0],
        far[1] / far[0]
    );
}

/// **An orthographic frame does not depend on the scene's bounding sphere.**
///
/// The sphere sets near and far, which are the z row; this file reads the other two. Asserted
/// rather than assumed, because it is the one input the fixture supplies without provenance from
/// the dump — if it ever starts to matter, this fails rather than quietly changing the numbers
/// every other test here is measured against.
#[test]
fn the_orthographic_frame_ignores_the_scene_radius() {
    let small = axis_reach(the_frame_at(DISTANCE, 1.0), the_middle());
    let large = axis_reach(the_frame_at(DISTANCE, 100_000.0), the_middle());
    assert!(
        (small[0] - large[0]).abs() < 1e-6,
        "the scene radius moved the frame: {} against {}",
        small[0],
        large[0]
    );
}
