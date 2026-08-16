//! Where a sketch's own marks land on this frame's glass.
//!
//! The drawing is not geometry the renderer resolves — a profile has no volume until it is
//! extruded, and an open one never gets that far — so the dots, edges and curves an author works
//! with are painted by egui over the viewport. That makes the SKETCH PLANE's map onto pixels the
//! one thing every mark is built from, and this module is that map: strike it once per frame, ask
//! it where a plane coordinate lands, and let the chord count for a curve follow from how far the
//! plane reaches on screen rather than from a fixed number.
//!
//! Free of the shell on purpose. The windowed viewer decorates what is here with hover, selection
//! and the gesture in flight; a headless capture has none of those and needs the same geometry, and
//! ten reports about marks leaving their plane were closed against replicas of this arithmetic
//! rather than against it.

use document::scene::SketchHandles;
use document::sketch::EntityId;
use egui::Pos2;

use super::render::{break_piece_points, circle_ring, SketchEdgeHit, ARC_SCREEN_SAGITTA_PX};

/// The sketch plane's map onto one frame's PHYSICAL pixels, and the chord tolerance that follows
/// from it.
///
/// Physical and not points, because this is also what the pointer is hit-tested against and a
/// pointer arrives in pixels. Divide on the way out to paint.
pub struct SketchPlaneProjection<'a> {
    handles: &'a SketchHandles,
    view_projection: glam::Mat4,
    /// `[x, y, width, height]` of the 3D viewport within the frame, in physical pixels.
    viewport: [f32; 4],
}

impl<'a> SketchPlaneProjection<'a> {
    /// Strike the map for this frame's camera and viewport.
    pub fn new(
        handles: &'a SketchHandles,
        view_projection: glam::Mat4,
        viewport_px: [u32; 4],
    ) -> Self {
        Self {
            handles,
            view_projection,
            #[allow(clippy::cast_precision_loss)]
            viewport: viewport_px.map(|component| component as f32),
        }
    }

    /// Where a plane coordinate lands, or `None` when it is behind the camera.
    ///
    /// Through the handles' own `profile_to_render`, so a mark and the vertex it is anchored to
    /// reach the screen by ONE path. Two projections of the same point agree until they do not.
    pub fn at(&self, coord: [f64; 2]) -> Option<Pos2> {
        let [x, y, width, height] = self.viewport;
        let vertex = self.handles.profile_to_render(coord);
        let clip = self.view_projection * glam::Vec4::new(vertex[0], vertex[1], vertex[2], 1.0_f32);
        (clip.w > 0.0).then(|| {
            Pos2::new(
                x + (clip.x / clip.w).mul_add(0.5, 0.5) * width,
                y + (1.0 - (clip.y / clip.w).mul_add(0.5, 0.5)) * height,
            )
        })
    }

    /// How finely to flatten a turn of `radius` centered at `center`, in the PLANE's own units.
    ///
    /// The one rule every curve on this page is flattened by. The projected radius says what a
    /// plane unit is currently worth in pixels — one number already carrying the zoom, the
    /// foreshortening and the plane's tilt — and the tolerance follows from it, so the same handful
    /// of chords is drawn at every zoom and a magnified curve never reads as a visible polygon.
    /// Never coarser than the resolve tolerance, which is the profile's own meaning: this is the
    /// same curve, drawn smoothly.
    pub fn chord_tolerance(&self, center: [f64; 2], on_rim: [f64; 2], radius: f64) -> f64 {
        self.at(center)
            .zip(self.at(on_rim))
            .map(|(center_px, rim_px)| f64::from(center_px.distance(rim_px)))
            .filter(|radius_px| *radius_px > 1.0)
            .map_or(document::sketch::ARC_SAGITTA_TOLERANCE, |radius_px| {
                radius * ARC_SCREEN_SAGITTA_PX / radius_px
            })
            .min(document::sketch::ARC_SAGITTA_TOLERANCE)
    }

    /// Every handle vertex, in handle order, `None` where one is behind the camera.
    ///
    /// The hole is kept rather than skipped: segment adjacency is stated as indices into this
    /// list, so dropping a vertex would silently re-point every edge after it.
    pub fn vertex_px(&self) -> Vec<Option<Pos2>> {
        self.handles
            .profile
            .iter()
            .map(|coord| self.at(*coord))
            .collect()
    }

    /// Each arc as the chord polyline it flattens to. A behind-camera chord culls the whole arc,
    /// matching the segment rule: a half-projected curve would fold across the viewport.
    pub fn arc_chords(&self) -> Vec<(EntityId, Vec<Pos2>)> {
        self.handles
            .arcs
            .iter()
            .filter_map(|arc| {
                let (from, to, sweep) = (arc.from, arc.to, arc.sweep_degrees);
                let tolerance = document::sketch::arc_center_radius(from, to, sweep).map_or(
                    document::sketch::ARC_SAGITTA_TOLERANCE,
                    |(center, radius)| self.chord_tolerance(center, from, radius),
                );
                let mut profile = vec![from];
                profile.extend(
                    document::sketch::arc_interior_points_within(from, to, sweep, tolerance)
                        .iter()
                        .map(document::sketch::SketchPoint::in_plane),
                );
                profile.push(to);
                let chords: Option<Vec<Pos2>> =
                    profile.into_iter().map(|coord| self.at(coord)).collect();
                chords.map(|chords| (arc.entity, chords))
            })
            .collect()
    }

    /// Each whole circle as its closed ring of chords, culled the same way an arc is.
    pub fn circle_chords(&self) -> Vec<(EntityId, Vec<Pos2>)> {
        self.handles
            .circles
            .iter()
            .filter_map(|circle| {
                let (center, radius) = (circle.center, circle.radius);
                let tolerance =
                    self.chord_tolerance(center, [center[0] + radius, center[1]], radius);
                let chords: Option<Vec<Pos2>> = circle_ring(center, radius, tolerance)
                    .into_iter()
                    .map(|coord| self.at(coord))
                    .collect();
                chords.map(|chords| (circle.entity, chords))
            })
            .collect()
    }

    /// Each span of each higher-order curve, every span answering to the AGGREGATE's id — so
    /// selecting an ellipse lights all four quarters instead of one.
    pub fn higher_curve_chords(&self) -> Vec<(document::sketch::SketchCurve, Vec<Pos2>)> {
        self.handles
            .higher_curves
            .iter()
            .flat_map(|curve| {
                curve.pieces.iter().filter_map(move |piece| {
                    let chords: Option<Vec<Pos2>> = break_piece_points(piece)
                        .into_iter()
                        .map(|coord| self.at(coord))
                        .collect();
                    chords.map(|chords| (curve.entity, chords))
                })
            })
            .collect()
    }
}

/// What the author's hand is doing to the drawing this frame.
///
/// Every field is a RESOLVED reading and not a piece of shell machinery: where the pointer is,
/// which curve the tool in hand would take, what a drag holds, what is picked. The windowed viewer
/// works those out from live gesture state; a headless capture has no hand and says so with
/// [`resting`](Self::resting). Both then draw through [`a_sketchs_marks`], which is the point — the
/// alternative is a second implementation of the drawing for the only observer that can photograph
/// it.
///
/// The resting reading is not a synthetic state invented for captures. A pointer that has left the
/// window is already nothing here in the running app, so a capture pins a picture the author can
/// actually see.
pub struct SketchHand<'a> {
    /// Which sketch these readings are about — selection is keyed by it.
    pub sketch: document::scene::NodeId,
    /// What is picked, anywhere in the workspace.
    pub selection: &'a ui::panel::Selection,
    /// Where the pointer is in PHYSICAL pixels, or nothing when it has left the window.
    pub cursor_px: Option<Pos2>,
    /// The point a drag currently holds.
    pub dragging_point: Option<EntityId>,
    /// The curve under the pointer and the state it lights in — already resolved, because WHICH
    /// curve a tool would take is the tool's question and not the mark's.
    pub hovered_edge: Option<(SketchEdgeHit, ui::gizmos::HandleState)>,
    /// The arms of every tangent lever currently drawn. An arm shows with its lever and never on
    /// its own: a green dot with no stick under it is a manipulator the author cannot read.
    pub arms_out: std::collections::BTreeSet<EntityId>,
}

impl<'a> SketchHand<'a> {
    /// No pointer, no drag, no hover, no lever out — the drawing as it stands when nobody is
    /// touching it.
    pub fn resting(sketch: document::scene::NodeId, selection: &'a ui::panel::Selection) -> Self {
        Self {
            sketch,
            selection,
            cursor_px: None,
            dragging_point: None,
            hovered_edge: None,
            arms_out: std::collections::BTreeSet::new(),
        }
    }
}

/// The drawing, ready to paint: its dots, its straight edges and its curves, in egui POINTS.
pub struct SketchMarks {
    /// The vertex dots that draw — which is not all of them; see
    /// [`the_dots_the_drawing_reveals`].
    pub points: Vec<ui::chrome::SketchVertexHandle>,
    /// One line per committed segment whose both ends are in front of the camera.
    pub segment_lines: Vec<ui::chrome::SketchEdgeLine>,
    /// Arcs, whole circles and higher-order curves, already flattened to chords.
    pub curve_lines: Vec<ui::chrome::SketchCurveLine>,
}

/// The marks a sketch shows, given where its plane lands and what the hand is doing.
///
/// The one implementation of the drawing itself. The windowed viewer appends what belongs to the
/// gesture in flight — previews, snap marks, the marquee, a spline's control frame — and a capture
/// takes what comes back as it is.
pub fn a_sketchs_marks(
    sketch: &document::sketch::Sketch,
    handles: &SketchHandles,
    plane: &SketchPlaneProjection,
    hand: &SketchHand,
    pixels_per_point: f32,
) -> SketchMarks {
    let vertex_px = plane.vertex_px();
    SketchMarks {
        points: the_dots_that_draw(sketch, handles, plane, hand, pixels_per_point, &vertex_px),
        segment_lines: the_segment_lines(handles, hand, pixels_per_point, &vertex_px),
        curve_lines: the_curve_lines(handles, plane, hand, pixels_per_point),
    }
}

/// A forgiving grab radius in physical pixels, so a hover reads as "draggable" near the thumb.
fn hover_radius_px(pixels_per_point: f32) -> f32 {
    (ui::chrome::SKETCH_HANDLE_HALF + ui::chrome::SKETCH_HANDLE_GRAB_PAD) * pixels_per_point
}

/// Whether the pointer is within grabbing distance of `at`.
fn under_the_pointer(hand: &SketchHand, at: Pos2, pixels_per_point: f32) -> bool {
    hand.cursor_px
        .is_some_and(|cursor| cursor.distance(at) <= hover_radius_px(pixels_per_point))
}

/// Every curve of this sketch the selection holds.
fn selected_curves(hand: &SketchHand) -> Vec<document::sketch::SketchCurve> {
    hand.selection
        .targets()
        .filter_map(|picked| match picked {
            ui::panel::SelectionTarget::SketchSegment { sketch, entity }
                if sketch == hand.sketch =>
            {
                Some(document::sketch::SketchCurve::Segment(entity))
            }
            ui::panel::SelectionTarget::SketchArc { sketch, entity } if sketch == hand.sketch => {
                Some(document::sketch::SketchCurve::Arc(entity))
            }
            ui::panel::SelectionTarget::SketchCircle { sketch, entity }
                if sketch == hand.sketch =>
            {
                Some(document::sketch::SketchCurve::Circle(entity))
            }
            ui::panel::SelectionTarget::SketchHigherCurve { sketch, curve }
                if sketch == hand.sketch =>
            {
                Some(curve)
            }
            _ => None,
        })
        .collect()
}

/// The curve an edge hit names.
fn curve_of_edge_hit(hit: SketchEdgeHit) -> document::sketch::SketchCurve {
    match hit {
        SketchEdgeHit::Segment(id) => document::sketch::SketchCurve::Segment(id),
        SketchEdgeHit::Arc(id) => document::sketch::SketchCurve::Arc(id),
        SketchEdgeHit::Circle(id) => document::sketch::SketchCurve::Circle(id),
        SketchEdgeHit::HigherCurve(curve) => curve,
    }
}

/// Every point that is one end of a tangent lever.
fn tangent_arm_points(sketch: &document::sketch::Sketch) -> std::collections::BTreeSet<EntityId> {
    sketch
        .splines()
        .iter()
        .flat_map(|spline| spline.tangents.values().flat_map(|handle| handle.arms()))
        .collect()
}

/// Which dots this drawing shows, with the hand's own reveals folded in.
///
/// A dot draws where the drawing is OPEN — where the ink has not already said where the point is
/// ([`point_draws_at_rest`](document::sketch::Sketch::point_draws_at_rest) is that rule). The hand
/// adds to it: selecting a line is a way of saying "this one", so the corners it runs between are
/// part of the answer; a point being touched answers for itself; and hovering a line has to bring
/// up its corners, or the author cannot tell a joined corner from a seam without clicking.
fn the_dots_the_drawing_reveals(
    sketch: &document::sketch::Sketch,
    hand: &SketchHand,
    touched: &std::collections::BTreeSet<EntityId>,
) -> std::collections::BTreeSet<EntityId> {
    let mut revealed: std::collections::BTreeSet<EntityId> = sketch
        .points()
        .iter()
        .filter(|point| sketch.point_draws_at_rest(point.id))
        .map(|point| point.id)
        .collect();
    for curve in selected_curves(hand) {
        revealed.extend(sketch.points_of(curve));
    }
    revealed.extend(touched.iter().copied());
    if let Some((hit, _)) = hand.hovered_edge {
        revealed.extend(sketch.points_of(curve_of_edge_hit(hit)));
    }
    // An arm shows with its lever and never on its own, whatever else revealed it.
    let arms = tangent_arm_points(sketch);
    revealed.retain(|id| !arms.contains(id) || hand.arms_out.contains(id));
    revealed.extend(hand.arms_out.iter().copied());
    revealed
}

/// The vertex dots that draw, in egui points.
fn the_dots_that_draw(
    sketch: &document::sketch::Sketch,
    handles: &SketchHandles,
    plane: &SketchPlaneProjection,
    hand: &SketchHand,
    pixels_per_point: f32,
    vertex_px: &[Option<Pos2>],
) -> Vec<ui::chrome::SketchVertexHandle> {
    let arms = tangent_arm_points(sketch);
    let on_ink: std::collections::BTreeSet<EntityId> = sketch
        .points()
        .iter()
        .filter(|point| sketch.point_stands_on_ink(point.id))
        .map(|point| point.id)
        .collect();
    // A dot standing under another dot never draws, whatever revealed it. Hovering an arc brings
    // up the points it stands on and one of those is the center it derives, so without this the
    // stack the rest-rule collapsed comes straight back the moment the author looks at the shape.
    let stacked: std::collections::BTreeSet<EntityId> = sketch
        .points()
        .iter()
        .filter(|point| sketch.a_better_dot_stands_here(point.id))
        .map(|point| point.id)
        .collect();

    let mut touched: std::collections::BTreeSet<EntityId> = std::collections::BTreeSet::new();
    let mut pending: Vec<(EntityId, ui::chrome::SketchVertexHandle)> =
        Vec::with_capacity(vertex_px.len());
    // Zipped rather than indexed: the handles build both lists from the same `points()` walk, so a
    // vertex ALWAYS has an id, and zipping is how this function says so. Indexing would hand back
    // an `Option` that no drawing can produce, and an `Option<EntityId>` compared against the
    // dragged point reads `None == None` as "this one is being dragged" — a rung of the ladder
    // firing on the absence of a point rather than on a point.
    for (point_id, center_px) in handles.point_ids.iter().copied().zip(vertex_px) {
        let Some(center_px) = *center_px else {
            continue;
        };
        let hovered = under_the_pointer(hand, center_px, pixels_per_point);
        let selected = hand
            .selection
            .contains(ui::panel::SelectionTarget::SketchPoint {
                sketch: hand.sketch,
                entity: point_id,
            });
        // Precedence: dragged > selected > hover > idle. A selected vertex stays filled-accent even
        // under the cursor, matching the segment rule so a point and an edge read alike.
        let dragged = hand.dragging_point == Some(point_id);
        let state = if dragged {
            ui::gizmos::HandleState::Snapped
        } else if selected {
            ui::gizmos::HandleState::Selected
        } else if hovered {
            ui::gizmos::HandleState::Hover
        } else {
            ui::gizmos::HandleState::Idle
        };
        // A dot the author is already touching answers for itself.
        if hovered || selected || dragged {
            touched.insert(point_id);
        }
        pending.push((
            point_id,
            ui::chrome::SketchVertexHandle {
                at: Pos2::new(
                    center_px.x / pixels_per_point,
                    center_px.y / pixels_per_point,
                ),
                state,
                ink: if arms.contains(&point_id) {
                    ui::chrome::SketchVertexInk::TangentArm
                } else if on_ink.contains(&point_id) {
                    ui::chrome::SketchVertexInk::OnInk
                } else {
                    ui::chrome::SketchVertexInk::OffInk
                },
            },
        ));
    }

    let revealed = the_dots_the_drawing_reveals(sketch, hand, &touched);
    let mut dots: Vec<ui::chrome::SketchVertexHandle> = pending
        .into_iter()
        .filter(|(point_id, _)| revealed.contains(point_id) && !stacked.contains(point_id))
        .map(|(_, handle)| handle)
        .collect();

    // A conic's shoulder, which is a reading rather than a point and so has no id to be revealed
    // by. It draws unconditionally: rho is the conic's one authored freedom and this is the only
    // mark that shows it, where every other dot here is answering whether the ink has already said
    // the same thing. It reads as ON the ink because it is, and it needs no grab of its own for the
    // same reason — the press under it lands on the conic, whose body drag is already the rho drag.
    for (_, at) in sketch.conic_shoulders() {
        let Some(px) = plane.at(at) else {
            continue;
        };
        dots.push(ui::chrome::SketchVertexHandle {
            at: Pos2::new(px.x / pixels_per_point, px.y / pixels_per_point),
            state: if under_the_pointer(hand, px, pixels_per_point) {
                ui::gizmos::HandleState::Hover
            } else {
                ui::gizmos::HandleState::Idle
            },
            ink: ui::chrome::SketchVertexInk::OnInk,
        });
    }
    dots
}

/// Precedence for an edge: Selected > plain Hover > Idle. A selected edge stays bold even under
/// the cursor, so Select's hover never shrinks it.
fn edge_state(
    hand: &SketchHand,
    selected: bool,
    is_the_hovered_one: impl Fn(SketchEdgeHit) -> bool,
) -> ui::gizmos::HandleState {
    if selected {
        return ui::gizmos::HandleState::Selected;
    }
    match hand.hovered_edge {
        Some((hit, state)) if is_the_hovered_one(hit) => state,
        _ => ui::gizmos::HandleState::Idle,
    }
}

/// Each committed edge between its two projected endpoints, in egui points.
///
/// An open sketch resolves to nothing, so the edges are the only thing that shows the profile is
/// connected. A behind-camera endpoint culls its line, matching the vertex-dot cull.
fn the_segment_lines(
    handles: &SketchHandles,
    hand: &SketchHand,
    pixels_per_point: f32,
    vertex_px: &[Option<Pos2>],
) -> Vec<ui::chrome::SketchEdgeLine> {
    handles
        .segments
        .iter()
        .filter_map(|segment| {
            let (a, b) = (
                (*vertex_px.get(segment.from)?)?,
                (*vertex_px.get(segment.to)?)?,
            );
            let selected = hand
                .selection
                .contains(ui::panel::SelectionTarget::SketchSegment {
                    sketch: hand.sketch,
                    entity: segment.entity,
                });
            Some(ui::chrome::SketchEdgeLine {
                a: Pos2::new(a.x / pixels_per_point, a.y / pixels_per_point),
                b: Pos2::new(b.x / pixels_per_point, b.y / pixels_per_point),
                state: edge_state(hand, selected, |hit| {
                    hit == SketchEdgeHit::Segment(segment.entity)
                }),
                construction: segment.role == document::sketch::EntityRole::Construction,
            })
        })
        .collect()
}

/// Arcs, whole circles and higher-order curves, in egui points and in that order.
///
/// The same precedence the segments use, so a picked arc and a picked segment read identically.
/// Every span of an aggregate reads the SAME state, resolved from the aggregate identity — so
/// selecting an ellipse lights all four quarters, and nothing here can draw one object in two
/// states at once.
fn the_curve_lines(
    handles: &SketchHandles,
    plane: &SketchPlaneProjection,
    hand: &SketchHand,
    pixels_per_point: f32,
) -> Vec<ui::chrome::SketchCurveLine> {
    let is_construction =
        |role: document::sketch::EntityRole| role == document::sketch::EntityRole::Construction;
    let construction_arcs: std::collections::BTreeSet<EntityId> = handles
        .arcs
        .iter()
        .filter(|arc| is_construction(arc.role))
        .map(|arc| arc.entity)
        .collect();
    let construction_circles: std::collections::BTreeSet<EntityId> = handles
        .circles
        .iter()
        .filter(|circle| is_construction(circle.role))
        .map(|circle| circle.entity)
        .collect();
    let construction_higher: std::collections::BTreeSet<document::sketch::SketchCurve> = handles
        .higher_curves
        .iter()
        .filter(|curve| is_construction(curve.role))
        .map(|curve| curve.entity)
        .collect();
    let in_points = |chords: Vec<Pos2>| -> Vec<Pos2> {
        chords
            .into_iter()
            .map(|px| Pos2::new(px.x / pixels_per_point, px.y / pixels_per_point))
            .collect()
    };

    let arcs = plane.arc_chords().into_iter().map(|(entity, chords)| {
        let selected = hand
            .selection
            .contains(ui::panel::SelectionTarget::SketchArc {
                sketch: hand.sketch,
                entity,
            });
        ui::chrome::SketchCurveLine {
            chords: in_points(chords),
            state: edge_state(hand, selected, |hit| hit == SketchEdgeHit::Arc(entity)),
            ink: super::render::curve_ink(construction_arcs.contains(&entity)),
        }
    });
    let circles = plane.circle_chords().into_iter().map(|(entity, chords)| {
        let selected = hand
            .selection
            .contains(ui::panel::SelectionTarget::SketchCircle {
                sketch: hand.sketch,
                entity,
            });
        ui::chrome::SketchCurveLine {
            chords: in_points(chords),
            state: edge_state(hand, selected, |hit| hit == SketchEdgeHit::Circle(entity)),
            ink: super::render::curve_ink(construction_circles.contains(&entity)),
        }
    });
    let higher = plane
        .higher_curve_chords()
        .into_iter()
        .map(|(curve, chords)| {
            let selected = hand
                .selection
                .contains(ui::panel::SelectionTarget::SketchHigherCurve {
                    sketch: hand.sketch,
                    curve,
                });
            ui::chrome::SketchCurveLine {
                chords: in_points(chords),
                state: edge_state(hand, selected, |hit| {
                    hit == SketchEdgeHit::HigherCurve(curve)
                }),
                ink: super::render::curve_ink(construction_higher.contains(&curve)),
            }
        });
    arcs.chain(circles).chain(higher).collect()
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unreachable)]
mod tests {
    use super::*;

    /// One free point on a Z sketch, and the node it belongs to.
    fn a_sketch_with_one_free_point() -> (
        document::scene::Scene,
        document::scene::NodeId,
        document::sketch::EntityId,
    ) {
        let mut sketch = document::sketch::Sketch::empty(document::sketch::PlaneAxis::Z);
        let point = sketch.add_free_point(document::sketch::SketchPoint::new(32, 48));
        let scene = document::scene::Scene::from_nodes(vec![document::scene::Node::new(
            "Ladder",
            document::scene::NodeContent::SketchTool {
                producer: document::sketch::SketchSolid::extrude(sketch, 16),
                material: voxel_core::core_geom::MaterialChoice::Wood,
            },
        )]);
        let node = scene.roots[0];
        (scene, node, point)
    }

    /// The state that one free point draws in, with `picked` saying whether the selection holds it
    /// and `shape_the_hand` adding whatever else the hand is doing.
    fn the_dots_state(
        picked: bool,
        shape_the_hand: impl FnOnce(&mut SketchHand, Pos2, document::sketch::EntityId),
    ) -> ui::gizmos::HandleState {
        let (scene, node, point) = a_sketch_with_one_free_point();
        let handles = scene
            .sketch_handles(node, 16, scene.recenter_voxels_for_resolve(16))
            .expect("a sketch node");
        let document::scene::NodeContent::SketchTool { producer, .. } =
            &scene.node_by_id(node).expect("just built").content
        else {
            unreachable!("built as a sketch tool")
        };
        // Identity: every plane coordinate lands in front of the camera at a knowable pixel, so
        // this is about the ladder and not about a projection.
        let plane = SketchPlaneProjection::new(&handles, glam::Mat4::IDENTITY, [0, 0, 1000, 1000]);
        let at = plane
            .at(handles.profile[0])
            .expect("identity puts every point in front of the camera");
        let selection = if picked {
            ui::panel::Selection::from_targets([ui::panel::SelectionTarget::SketchPoint {
                sketch: node,
                entity: point,
            }])
        } else {
            ui::panel::Selection::default()
        };
        let mut hand = SketchHand::resting(node, &selection);
        shape_the_hand(&mut hand, at, point);
        let marks = a_sketchs_marks(&producer.sketch, &handles, &plane, &hand, 1.0);
        assert_eq!(
            marks.points.len(),
            1,
            "one free point, and it draws at rest"
        );
        marks.points[0].state
    }

    /// Dragged beats selected beats hovered beats idle — and each rung is checked while the ones
    /// under it are also true, because a ladder only says anything where the rungs overlap.
    #[test]
    fn a_dot_reads_the_topmost_thing_the_hand_is_doing_to_it() {
        assert_eq!(
            the_dots_state(false, |_, _, _| {}),
            ui::gizmos::HandleState::Idle,
            "nothing is touching it"
        );
        assert_eq!(
            the_dots_state(false, |hand, at, _| hand.cursor_px = Some(at)),
            ui::gizmos::HandleState::Hover,
            "the pointer is on it"
        );
        assert_eq!(
            the_dots_state(true, |_, _, _| {}),
            ui::gizmos::HandleState::Selected,
            "picked, with nothing else true"
        );
        assert_eq!(
            the_dots_state(true, |hand, at, _| hand.cursor_px = Some(at)),
            ui::gizmos::HandleState::Selected,
            "picked outranks hovered: a selected dot must not shrink under the cursor"
        );
        assert_eq!(
            the_dots_state(true, |hand, at, point| {
                hand.cursor_px = Some(at);
                hand.dragging_point = Some(point);
            }),
            ui::gizmos::HandleState::Snapped,
            "a drag outranks both: the hand is holding this one"
        );
    }

    /// The top rung fires on a POINT, never on the absence of one.
    ///
    /// A vertex's id used to arrive as an `Option`, indexed out of a list the handles build in
    /// lockstep with the vertices — so it could never actually be missing, but `None == None`
    /// against an empty drag still read as "the hand is holding this one". Zipping the two lists
    /// removed the shape; this pins the reading, with a real drag in flight so the rung is live.
    #[test]
    fn a_dot_the_drag_is_not_holding_reads_untouched() {
        let mut sketch = document::sketch::Sketch::empty(document::sketch::PlaneAxis::Z);
        let held = sketch.add_free_point(document::sketch::SketchPoint::new(0, 0));
        let other = sketch.add_free_point(document::sketch::SketchPoint::new(64, 40));
        let scene = document::scene::Scene::from_nodes(vec![document::scene::Node::new(
            "Two dots",
            document::scene::NodeContent::SketchTool {
                producer: document::sketch::SketchSolid::extrude(sketch, 16),
                material: voxel_core::core_geom::MaterialChoice::Wood,
            },
        )]);
        let node = scene.roots[0];
        let handles = scene
            .sketch_handles(node, 16, scene.recenter_voxels_for_resolve(16))
            .expect("a sketch node");
        let document::scene::NodeContent::SketchTool { producer, .. } =
            &scene.node_by_id(node).expect("just built").content
        else {
            unreachable!("built as a sketch tool")
        };
        let plane = SketchPlaneProjection::new(&handles, glam::Mat4::IDENTITY, [0, 0, 1000, 1000]);
        let nothing = ui::panel::Selection::default();
        let mut hand = SketchHand::resting(node, &nothing);
        hand.dragging_point = Some(held);
        let marks = a_sketchs_marks(&producer.sketch, &handles, &plane, &hand, 1.0);
        assert_eq!(marks.points.len(), 2, "two free points, both drawing");
        let index_of = |wanted: document::sketch::EntityId| {
            handles
                .point_ids
                .iter()
                .position(|id| *id == wanted)
                .expect("both points are in the handles")
        };
        assert_eq!(
            marks.points[index_of(held)].state,
            ui::gizmos::HandleState::Snapped,
            "the dot the drag holds"
        );
        assert_eq!(
            marks.points[index_of(other)].state,
            ui::gizmos::HandleState::Idle,
            "the other dot: a drag elsewhere is not a hand on this one"
        );
    }

    /// A resting hand is what a headless capture brings, so what it draws has to be the drawing —
    /// the picture that answers "are these marks in the plane?" is worth nothing if the curves are
    /// missing from it.
    #[test]
    fn a_resting_hand_still_draws_the_curves() {
        let mut sketch = document::sketch::Sketch::empty(document::sketch::PlaneAxis::Z);
        let corners = [[0, 0], [64, 0], [64, 40], [0, 40]].map(|[across, up]| {
            sketch.add_free_point(document::sketch::SketchPoint::new(across, up))
        });
        for index in 0..4 {
            sketch
                .connect(corners[index], corners[(index + 1) % 4])
                .expect("four distinct corners");
        }
        let scene = document::scene::Scene::from_nodes(vec![document::scene::Node::new(
            "Rectangle",
            document::scene::NodeContent::SketchTool {
                producer: document::sketch::SketchSolid::extrude(sketch, 16),
                material: voxel_core::core_geom::MaterialChoice::Wood,
            },
        )]);
        let node = scene.roots[0];
        let handles = scene
            .sketch_handles(node, 16, scene.recenter_voxels_for_resolve(16))
            .expect("a sketch node");
        let document::scene::NodeContent::SketchTool { producer, .. } =
            &scene.node_by_id(node).expect("just built").content
        else {
            unreachable!("built as a sketch tool")
        };
        let plane = SketchPlaneProjection::new(&handles, glam::Mat4::IDENTITY, [0, 0, 1000, 1000]);
        let nothing = ui::panel::Selection::default();
        let marks = a_sketchs_marks(
            &producer.sketch,
            &handles,
            &plane,
            &SketchHand::resting(node, &nothing),
            1.0,
        );
        assert_eq!(marks.segment_lines.len(), 4, "the rectangle's four edges");
        assert!(
            marks
                .segment_lines
                .iter()
                .all(|line| line.state == ui::gizmos::HandleState::Idle),
            "a resting hand lights nothing"
        );
    }
}
