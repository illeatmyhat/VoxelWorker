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
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use camera::{OrbitCamera, ProjectionMode};
use ui::chrome::ConstraintBadge;
use ui::gizmos::dimension::{PlaneFrame, PlaneMap};

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

/// The same, at a chosen polar angle — the badge tests need a camera that is NOT edge-on to say
/// anything about size, because at edge-on the honest size is a sliver.
fn the_frame_at_angle(distance: f32, phi: f32) -> PlaneFrame {
    let camera = OrbitCamera {
        orbit_phi: phi,
        ..the_dumps_camera(distance)
    };
    let aspect = VIEWPORT_PX[2] / VIEWPORT_PX[3];
    let view_projection = camera.view_projection(aspect, glam::Vec3::ZERO, 64.0);
    let clip_of = |coord: [f64; 2]| {
        view_projection * glam::Vec4::new(coord[0] as f32, coord[1] as f32, 0.0, 1.0)
    };
    voxel_worker::windowed::a_sketch_planes_frame(&clip_of, VIEWPORT_PX, PIXELS_PER_POINT)
        .expect("the camera strikes a frame")
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

/// **How open the plane stands is a property of the camera's ANGLE and not of its distance.**
///
/// The reading every guard in this drawing should have been taking. `opening_at` is the Jacobian's
/// inverse condition number, which is dimensionless — so pulling the camera back a factor of ten
/// thousand leaves it where it was, while every quantity the old thresholds measured moved by
/// four orders. Turning the camera off the equator moves it, because that is a real change to how
/// the plane presents itself.
#[test]
fn how_open_the_plane_stands_does_not_move_when_the_camera_pulls_back() {
    let near = the_frame_at(10.0, 64.0).opening_at(the_middle());
    let far = the_frame_at(100_000.0, 64.0).opening_at(the_middle());
    assert!(
        near < 1e-5 && far < 1e-5,
        "edge-on at both ends: {near} near, {far} far"
    );

    // The same camera lifted to the isometric angle the app calls home. The plane opens up, and
    // the reading says so — which is the discriminability this file would otherwise lack, since a
    // function that simply returned zero would satisfy the assertion above.
    let isometric = OrbitCamera {
        orbit_phi: 0.955_316_6,
        ..the_dumps_camera(DISTANCE)
    };
    let aspect = VIEWPORT_PX[2] / VIEWPORT_PX[3];
    let view_projection = isometric.view_projection(aspect, glam::Vec3::ZERO, 64.0);
    let clip_of = |coord: [f64; 2]| {
        view_projection * glam::Vec4::new(coord[0] as f32, coord[1] as f32, 0.0, 1.0)
    };
    let open =
        voxel_worker::windowed::a_sketch_planes_frame(&clip_of, VIEWPORT_PX, PIXELS_PER_POINT)
            .expect("the isometric camera strikes a frame")
            .opening_at(the_middle());
    assert!(
        open > 0.5,
        "a plane seen from the home angle stands open, got {open}"
    );
}

/// **What the inverse costs at the evaluated point, measured rather than argued.**
///
/// `axes_at` recovers the plane coordinate under a screen point through the inverse before it
/// evaluates the Jacobian there, and on a near-singular frame that inverse is ill-conditioned.
/// The residue is named and left in `axes_at`'s own doc; this is the number behind that note, so
/// the day it matters it arrives measured. Under an orthographic projection the Jacobian is
/// constant, so a wrong place costs nothing and the round trip is exact anyway.
#[test]
fn the_inverse_round_trip_at_the_evaluated_point_stays_small() {
    let frame = the_frame_at(DISTANCE, 64.0);
    for plane in [[0.0, 0.0], [83.0, 0.0], [-60.0, 40.0], [120.0, -75.0]] {
        let Some(screen) = frame.at(plane) else {
            panic!("the dump's sketch is in front of the camera at {plane:?}");
        };
        // Back out through the surviving axis: the collapsed one carries no information, so the
        // honest thing to check is that the screen point is reproduced, not that both plane
        // coordinates are.
        let Some(again) = frame
            .axes_at(screen)
            .and_then(|_| frame.at(plane))
            .map(|back| (back - screen).length())
        else {
            panic!("the frame declined at its own image of {plane:?}");
        };
        assert!(
            again < 0.01,
            "the round trip at {plane:?} moved {again} points"
        );
    }
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

/// A badge seated where the sketch's own drawing is, on a given camera.
fn a_badge_at(distance: f32, phi: f32) -> ConstraintBadge {
    let frame = the_frame_at_angle(distance, phi);
    let seat = frame
        .plane_of(the_middle())
        .expect("the frame answers under the middle of the viewport");
    ConstraintBadge::seated(
        frame.forward(),
        seat,
        ui::icons::Icon::ConstraintParallel,
        1,
        false,
    )
    .expect("a badge seats where the drawing is")
}

/// The glyph's four corners on screen: the images of the four plane points its sides span.
///
/// Built from the badge's own public geometry rather than by calling the paint, so the assertion is
/// about the MARK's shape and needs no egui context. These are the same four plane points the
/// painter walks its grid over.
fn corners_of(badge: ConstraintBadge) -> Vec<egui::Pos2> {
    [(0.5, 0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, -0.5)]
        .into_iter()
        .filter_map(|(across, down)| {
            badge.plane.at([
                badge.across[0].mul_add(across, badge.down[0].mul_add(down, badge.seat[0])),
                badge.across[1].mul_add(across, badge.down[1].mul_add(down, badge.seat[1])),
            ])
        })
        .collect()
}

/// **A constraint badge's ink lies on the LINE the sketch plane images to, at the camera the
/// eighth report was filed from.**
///
/// The ninth report, as a gate. A badge used to be built on the glass from two screen directions
/// sampled off the projection at one point, and at this camera one of those directions is the
/// image of an axis three million to one shorter than the other — a vector with no length left to
/// have a direction, whose reading is f32 noise. The mark stood up out of a plane that draws as a
/// line. Every point of the glyph is now a point of the plane, so it can only land where the plane
/// lands.
///
/// **Seen red** by making the glyph's second side the screen square of its first, pulled back into
/// the plane — which is what every version of this before the rewrite amounted to. The corners then
/// stood 16 points off the line the plane draws, which is half a badge.
#[test]
fn a_badges_ink_lies_on_the_line_its_plane_images_to() {
    let badge = a_badge_at(DISTANCE, PHI);
    // The line the plane images to, taken from the plane itself and NOT from the badge: two plane
    // points far apart along the surviving axis. A guard that read the badge would be asking the
    // mark to agree with itself.
    let seat = badge.seat;
    let (near, far) = (
        badge
            .plane
            .at([seat[0] - 500.0, seat[1]])
            .expect("the plane images in front of the camera"),
        badge
            .plane
            .at([seat[0] + 500.0, seat[1]])
            .expect("the plane images in front of the camera"),
    );
    let along = (far - near).normalized();
    let off = |point: egui::Pos2| {
        let away = point - near;
        (away - along * away.dot(along)).length()
    };

    // Discriminability: the plane really does draw as a line here, so a mark that left it would
    // have somewhere to go. Without this the assertion below passes on any camera at all.
    let spread = corners_of(badge)
        .into_iter()
        .map(|corner| (corner - badge.center).length())
        .fold(0.0_f32, f32::max);
    assert!(
        spread > 10.0,
        "the badge has no extent to be wrong about: {spread} points"
    );
    assert!(
        badge.plane.opening_at_plane(badge.seat) < 1e-5,
        "this camera is not edge-on, so the test cannot fail: opening {}",
        badge.plane.opening_at_plane(badge.seat)
    );

    for corner in corners_of(badge) {
        assert!(
            off(corner) < 0.05,
            "a corner of the glyph stands {} points off the line its own plane draws",
            off(corner)
        );
    }
}

/// **A badge keeps its size on screen however far the author zooms out, and its seat converges with
/// the geometry it names.**
///
/// The owner's own prediction of what the fix should look like, run as an assertion: *"I should
/// expect that if I zoom out far enough all of the badges should overlap each other at the
/// origin"*, and separately *"It doesn't become a point when I zoom out. The size is still
/// stable."* Both fall out of the same mechanism and neither is coded for. The glyph is sized in
/// PLANE units against how far the plane reaches on screen, so the number of points it covers is
/// the same at both ends while the plane coordinate it occupies grows by the zoom factor — which is
/// exactly how the drawing under it behaves, so the marks pile up as their anchors do.
#[test]
fn a_badge_holds_its_screen_size_while_its_seat_spreads_with_the_zoom() {
    let isometric = 0.955_316_6;
    let near = a_badge_at(1000.0, isometric);
    let far = a_badge_at(100_000.0, isometric);

    let extent = |badge: ConstraintBadge| {
        corners_of(badge)
            .into_iter()
            .map(|corner| (corner - badge.center).length())
            .fold(0.0_f32, f32::max)
    };
    let (near_px, far_px) = (extent(near), extent(far));
    assert!(
        (near_px - far_px).abs() < 0.01 * near_px.max(1.0),
        "the badge changed size across a hundredfold zoom: {near_px} then {far_px} points"
    );

    let spans = |badge: ConstraintBadge| badge.across[0].hypot(badge.across[1]);
    let grew = spans(far) / spans(near);
    assert!(
        (grew - 100.0).abs() < 1.0,
        "a hundred times the distance must be a hundred times the plane units: {grew}"
    );
}

/// A plane whose second axis images to NOTHING: the projection is exactly rank one and the plane
/// draws as the horizontal line `y = 240`.
///
/// The tenth report's camera, posed as a matrix rather than found with an orbit angle, because
/// exactly singular is a different branch from very nearly singular and an angle can only be asked
/// to land near it. Every fixture in this file before this one was invertible, including the ones
/// named for being edge-on — which is why eight rounds of work never reached the branch the report
/// is about.
const fn a_plane_imaged_to_a_line() -> PlaneMap {
    PlaneMap::new([[0.6, 0.0, 320.0], [0.0, 0.0, 240.0], [0.0, 0.0, 1.0]])
}

/// **A badge on a plane that images to a line is a sliver ON that line — the mark does not exist
/// at full size on the glass.**
///
/// The tenth report: *"It works up until I'm exactly edge-on, then the badges pop out of the
/// plane."* The frame the pass used to build carries the projection's INVERSE, which asks which
/// plane coordinate lies under a pixel — a question with no answer here, because every pixel of the
/// line has a whole line of plane coordinates under it. So the frame declined, the pass substituted
/// the flat page, and every badge drew as an upright 32-point square standing off a drawing that had
/// gone to a line. The forward map has no such trouble: projecting a plane coordinate is well
/// defined at every camera, which is exactly why the sketch CURVES were still drawing correctly
/// while the marks beside them were not.
///
/// **Seen red** by the substitution the pass actually made, asserted below: the flat page puts the
/// same glyph's corners 16 points off the line, which is half a badge.
#[test]
fn a_badge_on_a_plane_imaged_to_a_line_lies_along_that_line() {
    let plane = a_plane_imaged_to_a_line();
    let seat = [10.0, -4.0];
    let badge = ConstraintBadge::seated(plane, seat, ui::icons::Icon::ConstraintParallel, 1, false)
        .expect("a plane drawn as a line still seats a badge");

    // The line, from the plane and not from the badge.
    let (near, far) = (
        plane.at([-500.0, 0.0]).expect("in front of the camera"),
        plane.at([500.0, 0.0]).expect("in front of the camera"),
    );
    let along = (far - near).normalized();
    let off = |point: egui::Pos2| {
        let away = point - near;
        (away - along * away.dot(along)).length()
    };

    for corner in corners_of(badge) {
        assert!(
            off(corner) < 1e-3,
            "a corner of the glyph stands {} points off the line its own plane draws",
            off(corner)
        );
    }
    // And it is still a badge: 32 points of it, along the one direction there is.
    let extent = corners_of(badge)
        .into_iter()
        .map(|corner| (corner - badge.center).length())
        .fold(0.0_f32, f32::max);
    assert!(
        (extent - 16.0).abs() < 0.01,
        "the sliver is not a badge long: {extent} points from its center"
    );

    // The red: what the pass substituted when the frame declined. Same glyph, same screen place,
    // laid out on the flat page — a full-size upright square standing clear of the drawing.
    let substituted = ConstraintBadge::seated(
        PlaneMap::flat(),
        [f64::from(badge.center.x), f64::from(badge.center.y)],
        ui::icons::Icon::ConstraintParallel,
        1,
        false,
    )
    .expect("the flat page seats a badge");
    let stood_off = corners_of(substituted)
        .into_iter()
        .map(off)
        .fold(0.0_f32, f32::max);
    assert!(
        stood_off > 15.0,
        "the fallback this test exists to condemn is not measurably wrong here: {stood_off} points"
    );
}

/// **No camera makes the badge pass decline.** The forward map is total.
///
/// The frame beside it is not, and that asymmetry is the whole fix: a mark that only ever projects
/// plane coordinates forward has nothing to fail at, while one that asks the inverse a question
/// loses its answer at exactly the camera that makes the question ill-posed.
#[test]
fn the_planes_forward_map_answers_at_every_orbit_angle() {
    let mut declined = 0;
    for step in 0_u8..=36 {
        let phi = f32::from(step) * std::f32::consts::PI / 36.0 - std::f32::consts::FRAC_PI_2;
        let camera = OrbitCamera {
            orbit_phi: phi,
            ..the_dumps_camera(DISTANCE)
        };
        let aspect = VIEWPORT_PX[2] / VIEWPORT_PX[3];
        let view_projection = camera.view_projection(aspect, glam::Vec3::ZERO, 64.0);
        let clip_of = |coord: [f64; 2]| {
            view_projection * glam::Vec4::new(coord[0] as f32, coord[1] as f32, 0.0, 1.0)
        };
        let map =
            voxel_worker::windowed::a_sketch_planes_map(&clip_of, VIEWPORT_PX, PIXELS_PER_POINT);
        assert!(
            ConstraintBadge::seated(
                map,
                [0.0, 0.0],
                ui::icons::Icon::ConstraintParallel,
                1,
                false
            )
            .is_some(),
            "no badge at orbit angle {phi}"
        );
        if voxel_worker::windowed::a_sketch_planes_frame(&clip_of, VIEWPORT_PX, PIXELS_PER_POINT)
            .is_none()
        {
            declined += 1;
        }
    }
    // Reported and not asserted on: whether an ORBIT ANGLE lands exactly on the singular matrix is
    // a question about float arithmetic, not about the drawing, and the branch itself is gated by
    // `a_badge_on_a_plane_imaged_to_a_line`, which poses the matrix directly.
    println!("the frame declined at {declined} of 37 orbit angles");
}

/// A drawing with something of every badge-placing species on it: a segment relation, a point
/// relation, and a corner relation.
///
/// On `PlaneAxis::Z`, so the profile lies on the ground plane and the dump's camera — polar angle
/// pi/2 — sees it exactly edge-on.
fn a_scene_with_marked_up_geometry() -> (document::scene::Scene, document::scene::NodeId) {
    use document::sketch::{ConstraintKind, Sketch, SketchPoint};

    let mut sketch = Sketch::empty(document::sketch::PlaneAxis::Z);
    let corner = sketch.add_free_point(SketchPoint::new(0, 0));
    let right = sketch.add_free_point(SketchPoint::new(40, 0));
    let up = sketch.add_free_point(SketchPoint::new(0, 30));
    let along = sketch.connect(corner, right).expect("a run");
    let rising = sketch.connect(corner, up).expect("a rise");

    let context =
        parametric::EvaluationContext::new(std::num::NonZeroU32::new(16).expect("a density"));
    let producer = document::sketch::SketchSolid::extrude(sketch, 4);
    let producer = producer
        .with_constraint(ConstraintKind::Horizontal { segment: along }, context)
        .expect("horizontal")
        .0;
    let producer = producer
        .with_constraint(
            ConstraintKind::Perpendicular {
                first: along,
                second: rising,
            },
            context,
        )
        .expect("perpendicular")
        .0;
    let producer = producer
        .with_constraint(
            ConstraintKind::Fix {
                point: up,
                at: SketchPoint::new(0, 30),
            },
            context,
        )
        .expect("fix")
        .0;

    let node = document::scene::Node::new(
        "Sketch",
        document::scene::NodeContent::SketchTool {
            producer,
            material: voxel_core::core_geom::MaterialChoice::Stone,
        },
    );
    let scene = document::scene::Scene::single_node(node);
    let id = scene.roots[0];
    (scene, id)
}

/// **The badges the app would actually draw, at the camera the reports were filed from, all lie on
/// the line the sketch plane images to.**
///
/// The other tests in this file pose a badge; this one runs the SHELL's own layout — the anchor
/// rules, the standoff, the stacking, the constructor — over a real scene, and measures where every
/// mark of every species lands. Nine rounds of these reports were closed against a posed badge or a
/// replica of the arithmetic, and the tenth was a defect in the layout rather than in the mark.
///
/// **Seen red at 16 points off the line**, measured in the test itself: the flat page posed at each
/// badge's own screen place, which is half a badge of standoff on a drawing that has no width.
#[test]
fn every_badge_the_shell_lays_out_lies_on_the_line_at_the_reporters_camera() {
    let (scene, id) = a_scene_with_marked_up_geometry();
    let handles = scene
        .sketch_handles(id, 16, scene.recenter_voxels_for_resolve(16))
        .expect("handles");
    let document::scene::NodeContent::SketchTool { producer, .. } =
        &scene.node_by_id(id).expect("the node").content
    else {
        panic!("a sketch node");
    };

    let camera = the_dumps_camera(DISTANCE);
    let aspect = VIEWPORT_PX[2] / VIEWPORT_PX[3];
    let view_projection = camera.view_projection(aspect, glam::Vec3::ZERO, 64.0);
    let clip_of = |coord: [f64; 2]| {
        let vertex = handles.profile_to_render(coord);
        view_projection * glam::Vec4::new(vertex[0], vertex[1], vertex[2], 1.0)
    };
    let plane =
        voxel_worker::windowed::a_sketch_planes_map(&clip_of, VIEWPORT_PX, PIXELS_PER_POINT);

    let context = Some(parametric::EvaluationContext::new(
        std::num::NonZeroU32::new(16).expect("a density"),
    ));
    let badges = voxel_worker::windowed::a_sketchs_constraint_badges(
        &producer.sketch,
        &handles,
        plane,
        context,
        &|_| false,
    );
    assert!(
        badges.len() >= 3,
        "the drawing carries no marks to be wrong about: {}",
        badges.len()
    );

    // The line, struck from the plane and not from any badge.
    let (near, far) = (
        plane.at([-5000.0, 0.0]).expect("in front of the camera"),
        plane.at([5000.0, 0.0]).expect("in front of the camera"),
    );
    let along = (far - near).normalized();
    let off = |point: egui::Pos2| {
        let away = point - near;
        (away - along * away.dot(along)).length()
    };

    let mut worst = 0.0_f32;
    for badge in &badges {
        for corner in corners_of(*badge) {
            worst = worst.max(off(corner));
        }
        // Including the selection plate, which is drawn and therefore has to lie in the plane too.
        for corner in badge.plate() {
            worst = worst.max(off(corner));
        }
        worst = worst.max(off(badge.center));
    }
    assert!(
        worst < 0.05,
        "a badge stands {worst} points off the line its own sketch plane draws"
    );

    // The red, measured rather than asserted from memory: the flat page this pass fell back to
    // when the frame declined, posed at each badge's own screen place, so the only difference is
    // the plane the mark is built in.
    let substituted = badges
        .iter()
        .filter_map(|badge| {
            ConstraintBadge::seated(
                PlaneMap::flat(),
                [f64::from(badge.center.x), f64::from(badge.center.y)],
                badge.icon,
                badge.constraint,
                false,
            )
        })
        .flat_map(corners_of)
        .map(off)
        .fold(0.0_f32, f32::max);
    println!("the flat page stood {substituted} points off the line");
    assert!(
        substituted > 10.0,
        "the fallback this test exists to condemn is not measurably wrong here: {substituted}"
    );
}

/// The same plane drawn as a line, but seen from BEHIND: its +x images leftward.
const fn a_plane_imaged_to_a_line_from_behind() -> PlaneMap {
    PlaneMap::new([[-0.6, 0.0, 320.0], [0.0, 0.0, 240.0], [0.0, 0.0, 1.0]])
}

/// **A sliver badge still reads left to right, and the coin it tosses when there is no second
/// direction left falls the way the code says it does.**
///
/// Collinearity cannot see orientation: a sliver turned 180 degrees is still on the line, so
/// `a_badge_on_a_plane_imaged_to_a_line` passes whichever way the glyph faces. That leaves the one
/// DECISION this cut names — "a plane whose axis images to nothing keeps its authored axis" — with
/// no falsifier at all, and a later tidy of `>=` to `>` would flip it at exactly zero with every
/// other test still green. The flicker would then arrive as an unattributable golden diff.
///
/// Two claims, and only the first is visible on screen:
///
/// * The **reading** direction is observable and is asserted on both a plane imaged forward and one
///   imaged from behind. The glyph's own +x side runs left to right either way, which is the point
///   of folding the basis upright rather than pinning it to the plane's authored axes.
/// * The **coin** is not observable, by definition — it orients the side that images to nothing.
///   It is asserted in PLANE coordinates because that is the only place it exists, and because an
///   unobservable decision left unasserted is one nobody can change on purpose.
#[test]
fn a_sliver_badge_reads_left_to_right_and_lands_its_named_coin() {
    for (facing_the_camera, plane) in [
        (true, a_plane_imaged_to_a_line()),
        (false, a_plane_imaged_to_a_line_from_behind()),
    ] {
        let seat = [10.0, -4.0];
        let badge =
            ConstraintBadge::seated(plane, seat, ui::icons::Icon::ConstraintParallel, 1, false)
                .expect("a plane drawn as a line still seats a badge");

        // Where the glyph's own +x side points ON SCREEN — the same arithmetic the painter walks
        // its grid over, so this is the drawing and not a stored claim.
        let origin = plane.at(seat).expect("in front of the camera");
        let reading = plane
            .at([
                badge.across[0].mul_add(0.5, seat[0]),
                badge.across[1].mul_add(0.5, seat[1]),
            ])
            .expect("in front of the camera")
            - origin;
        assert!(
            reading.x > 0.0,
            "the glyph reads right to left on a plane {}: {reading:?}",
            if facing_the_camera {
                "imaged forward"
            } else {
                "imaged from behind"
            }
        );
        // And the plane's authored +x is the one that got flipped to achieve it, which is what
        // makes the mark unmirrored rather than merely pointed the right way.
        assert_eq!(
            badge.across[0] > 0.0,
            facing_the_camera,
            "the basis folded the wrong axis"
        );

        // The coin. The plane's second axis images to nothing here, so there is no side for the
        // handedness comparison to land on and it falls to the plane's authored up — which the
        // glyph's grid, running y downward, reverses. Nothing on screen can tell; that is why it
        // is pinned here.
        assert!(
            badge.down[1] < 0.0,
            "the unobservable side flipped: down {:?}",
            badge.down
        );
        assert!(
            badge.down[0].abs() < 1e-12,
            "the second side left the plane's own axis: down {:?}",
            badge.down
        );
    }
}
