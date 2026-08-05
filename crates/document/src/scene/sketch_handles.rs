//! Sketch-mode vertex handles: each profile vertex's position in the display's
//! recentered **render frame**, plus the inverse map (a cursor hit on the sketch plane
//! back to a profile `(c0, c1)` voxel coordinate). This is the geometry the interactive
//! vertex drag draws and hit-tests.
//!
//! **The frame is carried, never re-derived.** Every position is routed through the SAME
//! [`substrate::spatial::LeafPlacement`] the resolver folds occupancy through — the profile
//! vertex is a producer-LOCAL voxel point, `world_of` places it, and the composite recenter
//! rebases it into the render frame — so a handle coincides with the resolved geometry's
//! profile corner BY CONSTRUCTION rather than by a kept-in-sync mirror (mirroring the
//! placement ghost's `center_world`).
//!
//! **The convention is corner-anchored, like extrude.** A profile point `(c0, c1)` maps
//! to producer-local `(c0 − min0, c1 − min1)` on the plane's two in-plane axes, at `0`
//! along the plane normal — the profile's bounding-box minimum sits on the node's world
//! anchor. That is exactly where the extrude resolve seats the profile; the handles
//! therefore represent the 2D PROFILE on its plane (the authoring surface), independent
//! of which operation later lifts it into a volume.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::must_use_candidate,
    clippy::similar_names,
    clippy::doc_link_code,
    clippy::missing_const_for_fn,
    clippy::too_many_lines,
    clippy::wildcard_imports
)]

use super::*;
use crate::sketch::{EntityId, EntityRole, Operation, PointLifetime, SketchCurve};
use glam::Vec3;
use parametric::EvaluationContext;
use std::num::NonZeroU32;
use substrate::curve_intersection::PlanarCurve;
use substrate::spatial::{LeafPlacement, ProducerLocalVoxelPoint, TrueWorldVoxelPoint};

/// One straight profile edge ready for display.
///
/// The role rides along because it is a LINETYPE, not a hit-test property: construction geometry
/// locates the shape without being part of it, and the viewer cannot say so unless the handle
/// says which it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SketchSegmentHandle {
    /// The segment entity this display handle names.
    pub entity: EntityId,
    /// Index into [`SketchHandles::vertices`] of the edge's tail.
    pub from: usize,
    /// Index into [`SketchHandles::vertices`] of the edge's head.
    pub to: usize,
    /// Whether this edge is part of the shape or merely locates it.
    pub role: EntityRole,
}

/// One arc ready for display, in the canonical endpoint form the viewer tessellates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SketchArcHandle {
    /// The arc entity this display handle names.
    pub entity: EntityId,
    /// The tail endpoint in profile coordinates.
    pub from: [f64; 2],
    /// The head endpoint in profile coordinates.
    pub to: [f64; 2],
    /// The signed sweep in degrees; its sign picks the arc's direction.
    pub sweep_degrees: f64,
    /// Whether this arc is part of the shape or merely locates it.
    pub role: EntityRole,
}

/// A circle ready for display: its identity stays paired with its profile-space geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SketchCircleHandle {
    /// The circle entity this display handle names.
    pub entity: EntityId,
    /// The circle's center in profile coordinates.
    pub center: [f64; 2],
    /// The circle's radius in voxels.
    pub radius: f64,
    /// Whether this circle is part of the shape or merely locates it.
    pub role: EntityRole,
}

/// One higher-order authored curve resolved into the planar pieces the viewer can sample.
#[derive(Debug, Clone, PartialEq)]
pub struct SketchCurveHandle {
    /// Stable aggregate identity used by selection and modifiers.
    pub entity: SketchCurve,
    /// Exact substrate pieces; tessellation remains a screen-space viewer decision.
    pub pieces: Vec<PlanarCurve>,
    /// Whether this curve is part of the shape or merely locates it. One role for the whole
    /// aggregate, like every other property of an aggregate identity.
    pub role: EntityRole,
}

/// The sketch's profile vertices in the recentered render frame, with everything the UI
/// needs to draw draggable handles and turn a cursor ray back into a profile coordinate.
///
/// Positions are in the SAME render frame the resolved voxels and the transform gizmo
/// live in (voxel units, composite-recentered), so the UI projects them through the same
/// `view_projection` it uses for everything else.
#[derive(Debug, Clone)]
pub struct SketchHandles {
    /// EVERY point entity's position in the render frame, in `points()` order (index `i`
    /// corresponds to point id [`point_ids`](Self::point_ids)`[i]`). All points are shown —
    /// including free points and the vertices of an open graph — so any entity is selectable,
    /// not just the ones on a closed loop.
    pub vertices: Vec<[f32; 3]>,
    /// The point id of each vertex, in the SAME order as [`vertices`](Self::vertices), so a
    /// drag / delete can map a hit index back to the stable entity it must mutate (the entity
    /// store has no positional index).
    pub point_ids: Vec<EntityId>,
    /// Whether each vertex is one the drawing DERIVES, in the same order.
    ///
    /// A derived point is a readout, not a freedom: dragging it authors the quantity behind it
    /// rather than a position. Slots stack an authored handle exactly on top of one, so a hit-test
    /// that only knows distance is picking between them by accident — see
    /// [`SketchHandles::point_ids`] for the identity that hit resolves to.
    pub derived: Vec<bool>,
    /// Each segment entity, its two endpoint indices into
    /// [`vertices`](Self::vertices)/[`point_ids`](Self::point_ids), and its role. The UI draws a
    /// line per entry and hit-tests add-point against them (splitting the named segment by id).
    /// A segment with a dangling endpoint is omitted.
    pub segments: Vec<SketchSegmentHandle>,
    /// Each arc entity in canonical form, with the endpoint ids resolved to PROFILE coordinates.
    /// The curve is deliberately NOT tessellated here: chord count belongs to whoever knows how
    /// many pixels a voxel is currently worth, and
    /// [`profile_to_render`](Self::profile_to_render) maps each sample it produces into this
    /// frame. An arc with a dangling endpoint is omitted.
    pub arcs: Vec<SketchArcHandle>,
    /// Each circle ready for display. Tessellation stays in the viewer, which alone knows the
    /// screen-space tolerance.
    pub circles: Vec<SketchCircleHandle>,
    /// Rational Bézier, ellipse, conic, and spline aggregates ready for display.
    pub higher_curves: Vec<SketchCurveHandle>,
    /// A point ON the sketch plane in the render frame (the first vertex) — the ray
    /// intersection anchor.
    pub plane_point: [f32; 3],
    /// The sketch plane's unit normal in the render frame (`rotation · e_normal`).
    pub plane_normal: [f32; 3],
    /// The placement affine (carried so the inverse map rotates through the exact same
    /// transform the forward map placed vertices with).
    placement: LeafPlacement,
    /// The composite recenter (render frame = true world − recenter), in voxels.
    recenter: [i64; 3],
    /// The profile's in-plane bounding-box minimum, in voxels — added back so a local
    /// coordinate returns to absolute profile space.
    profile_min: [i64; 2],
    /// The plane's two in-plane world axes (`PlaneAxis::in_plane_axes`).
    in_plane_axes: [usize; 2],
}

impl SketchHandles {
    /// Map a CONTINUOUS profile coordinate `(c0, c1)` back into the render frame — the
    /// forward twin of [`render_hit_to_profile`](Self::render_hit_to_profile), through the
    /// same placement, so a drawing tool's preview (the snapped polyline endpoint, the
    /// rectangle ghost corners) lands exactly where the committed vertex does.
    pub fn profile_to_render(&self, coord: [f64; 2]) -> [f32; 3] {
        let [in0, in1] = self.in_plane_axes;
        let mut local = [0.0f32; 3];
        local[in0] = (coord[0] - self.profile_min[0] as f64) as f32;
        local[in1] = (coord[1] - self.profile_min[1] as f64) as f32;
        let world = self
            .placement
            .world_of(ProducerLocalVoxelPoint::from_voxels(Vec3::from_array(
                local,
            )))
            .voxels();
        [
            world.x - self.recenter[0] as f32,
            world.y - self.recenter[1] as f32,
            world.z - self.recenter[2] as f32,
        ]
    }

    /// Map a hit point on the sketch plane (in the RENDER frame — e.g. a cursor ray's
    /// intersection with [`plane_point`](Self::plane_point) / [`plane_normal`](Self::plane_normal))
    /// back to a CONTINUOUS profile coordinate `(c0, c1)` in voxels. The caller snaps it
    /// (round for grid-snap, floor+fraction for sub-voxel) and writes it into the dragged
    /// `SketchPoint.offset_voxels`.
    ///
    /// The inverse of the forward placement: rebase the render hit into true world
    /// (`+ recenter`), invert the placement to producer-local, read the two in-plane
    /// components and add the profile minimum back. `render_hit` need not lie exactly on
    /// the plane — the normal component is simply discarded by reading only the in-plane
    /// axes — but a ray/plane intersection keeps it on-plane so the drag tracks the cursor.
    pub fn render_hit_to_profile(&self, render_hit: [f32; 3]) -> [f64; 2] {
        let world = Vec3::new(
            render_hit[0] + self.recenter[0] as f32,
            render_hit[1] + self.recenter[1] as f32,
            render_hit[2] + self.recenter[2] as f32,
        );
        let local = self
            .placement
            .local_of(TrueWorldVoxelPoint::from_voxels(world))
            .voxels();
        let [in0, in1] = self.in_plane_axes;
        [
            local[in0] as f64 + self.profile_min[0] as f64,
            local[in1] as f64 + self.profile_min[1] as f64,
        ]
    }
}

impl Scene {
    /// The [`SketchHandles`] for the sketch node `node_id` — EVERY point entity placed into
    /// the render frame with its stable id, the segment connectivity, and the inverse
    /// cursor-to-profile map. `None` only when the id is not an enabled `SketchTool` node.
    ///
    /// Independent of the operation's degeneracy AND of whether a closed loop exists: an open
    /// or un-extruded sketch STILL returns handles, so every vertex stays draggable and
    /// deletable while the sketch is authored — entities, not a loop, are the truth.
    /// A totally EMPTY sketch returns handles with no vertices: the plane frame and inverse
    /// map still stand, which is what lets a drawing tool place the FIRST point.
    pub fn sketch_handles(&self, node_id: NodeId, voxels_per_block: u32) -> Option<SketchHandles> {
        let context = EvaluationContext::new(NonZeroU32::new(voxels_per_block)?);
        let node = self.node_by_id(node_id)?;
        if !node.enabled {
            return None;
        }
        let NodeContent::SketchTool { producer, .. } = &node.content else {
            return None;
        };
        let points = producer.sketch.points();
        let point_ids: Vec<EntityId> = points.iter().map(|point| point.id).collect();
        let derived: Vec<bool> = point_ids
            .iter()
            .map(|id| producer.sketch.is_arc_center(*id))
            .collect();

        // The overlay frame anchors on the RESOLVE's anchor — the filled region's bbox-min, the
        // same `profile_bbox_min` the producer re-seats to the node origin. One anchor, so a
        // handle is on the solid by construction rather than by the two definitions agreeing.
        //
        // Anchoring instead on the bbox over the real POINTS is equal only while every point
        // is on the filled boundary. Draw a line reaching past the fill — a free polyline, a
        // vertex outside it — and the points-min moves while the resolve's does not, sliding
        // the whole drawing off the solid it belongs to. Worse, the anchor compensation on
        // every edit (`SketchSolid::anchor_preserving_offset`) corrects for a change in the
        // RESOLVE's anchor, so a points-min move is a shift nothing cancels.
        //
        // A sketch with nothing filled anchors on `[0, 0]`: it resolves to nothing, so there is
        // no solid to sit on and every point draws at its own offset from the node origin — where
        // the author put it, and where it stays as further points are placed around it.
        let anchor = producer.profile_bbox_min(context);

        // The extent of the box the HANDLES occupy, which is theirs and not the resolve's — it
        // covers free points and open chains that no face contains. Points a curve anchors are
        // excluded: an arc's center can sit well outside the profile.
        let mut real = points
            .iter()
            .filter(|point| point.lifetime == PointLifetime::Freestanding)
            .map(|point| point.at.offset_voxels);
        let mut min = real.next().unwrap_or([0, 0]);
        let mut max = min;
        for coords in real {
            for axis in 0..2 {
                min[axis] = min[axis].min(coords[axis]);
                max[axis] = max[axis].max(coords[axis]);
            }
        }

        let [in0, in1] = producer.sketch.plane.in_plane_axes();
        let normal = producer.sketch.plane.normal_axis();

        // The producer-local box extent `full`. The two in-plane axes span the profile
        // bbox; the normal axis carries the operation's extrude thickness (0 for revolve /
        // an un-extruded profile). `full` only re-anchors the box under a genuine rotation
        // (`min_rotated_corner`); with an axis-aligned plane the rotation is identity and it
        // drops out — but it is routed through `LeafPlacement` so a free-angle plane stays
        // exact.
        let mut full = [0.0f32; 3];
        full[in0] = (max[0] - min[0]) as f32;
        full[in1] = (max[1] - min[1]) as f32;
        full[normal] = match producer.operation {
            Operation::Extrude { height_voxels } => height_voxels as f32,
            Operation::Revolve { .. } => 0.0,
        };

        // The node's world placement: accumulated parent offset + its own integer offset,
        // its sub-voxel slide, and its continuous rotation.
        let path = self.path_of(node_id)?;
        let (_target, parent_offset) = self.subtree_walk_target(&path)?;
        let world_offset: [i64; 3] =
            std::array::from_fn(|axis| parent_offset[axis] + node.transform.offset_voxels[axis]);
        let placement = LeafPlacement::from_origin_and_local(
            node.transform.rotation(),
            Vec3::from_array(full),
            world_offset,
            node.transform.offset_local_voxels,
        );

        let recenter = self.recenter_voxels_for_resolve(voxels_per_block).voxels();
        let recenter_vec = Vec3::new(recenter[0] as f32, recenter[1] as f32, recenter[2] as f32);

        // One continuous profile coordinate into the render frame — the map every handle and
        // arc chord goes through, so a drawn curve and a dragged vertex share one frame.
        let to_render = |coord: [f64; 2]| {
            let mut local = [0.0f32; 3];
            local[in0] = (coord[0] - anchor[0] as f64) as f32;
            local[in1] = (coord[1] - anchor[1] as f64) as f32;
            // local[normal] stays 0.0 — the profile lives on the plane.
            let world = placement
                .world_of(ProducerLocalVoxelPoint::from_voxels(Vec3::from_array(
                    local,
                )))
                .voxels();
            (world - recenter_vec).to_array()
        };

        let vertices: Vec<[f32; 3]> = points
            .iter()
            .map(|point| to_render(point.at.in_plane()))
            .collect();

        // Segment connectivity, mapped to vertex indices; a dangling endpoint drops the segment.
        let index_of = |id: EntityId| point_ids.iter().position(|&pid| pid == id);
        let segments: Vec<SketchSegmentHandle> = producer
            .sketch
            .segments()
            .iter()
            .filter_map(|seg| {
                Some(SketchSegmentHandle {
                    entity: seg.id,
                    from: index_of(seg.from)?,
                    to: index_of(seg.to)?,
                    role: seg.role,
                })
            })
            .collect();

        // Each arc's canonical form with its endpoints resolved — the viewer picks the chord
        // count, so nothing is tessellated here.
        let position_of = |id: EntityId| {
            points
                .iter()
                .find(|point| point.id == id)
                .map(|point| point.at.in_plane())
        };
        let arcs: Vec<SketchArcHandle> = producer
            .sketch
            .arcs()
            .iter()
            .filter_map(|arc| {
                let form = producer.sketch.arc_form(arc)?;
                Some(SketchArcHandle {
                    entity: arc.id,
                    from: form.from,
                    to: form.to,
                    sweep_degrees: form.sweep_degrees,
                    role: arc.role,
                })
            })
            .collect();
        let circles: Vec<SketchCircleHandle> = producer
            .sketch
            .circles()
            .iter()
            .filter_map(|circle| {
                Some(SketchCircleHandle {
                    entity: circle.id,
                    center: position_of(circle.center)?,
                    radius: circle.resolved_radius(context),
                    role: circle.role,
                })
            })
            .collect();
        let higher_sources = producer
            .sketch
            .beziers()
            .iter()
            .map(|curve| SketchCurve::Bezier(curve.id))
            .chain(
                producer
                    .sketch
                    .ellipses()
                    .iter()
                    .map(|curve| SketchCurve::Ellipse(curve.id)),
            )
            .chain(
                producer
                    .sketch
                    .conics()
                    .iter()
                    .map(|curve| SketchCurve::Conic(curve.id)),
            )
            .chain(
                producer
                    .sketch
                    .splines()
                    .iter()
                    .map(|curve| SketchCurve::Spline(curve.id)),
            );
        let higher_curves = higher_sources
            .filter_map(|entity| {
                let (pieces, role) = producer.sketch.source_planar_curves(entity, context)?;
                Some(SketchCurveHandle {
                    entity,
                    pieces,
                    role,
                })
            })
            .collect();

        let plane_normal = (node.transform.rotation() * unit_axis(normal)).to_array();
        // ANY on-plane point anchors the ray intersection; the producer-local origin (the
        // profile bbox-min corner, normal component 0) works for every sketch including an
        // empty one, where there is no vertex to borrow.
        let plane_point = (placement
            .world_of(ProducerLocalVoxelPoint::from_voxels(Vec3::ZERO))
            .voxels()
            - recenter_vec)
            .to_array();

        Some(SketchHandles {
            vertices,
            point_ids,
            derived,
            segments,
            arcs,
            circles,
            higher_curves,
            plane_point,
            plane_normal,
            placement,
            recenter,
            profile_min: anchor,
            in_plane_axes: [in0, in1],
        })
    }
}

/// The unit vector along world `axis` (0 = X, 1 = Y, 2 = Z).
fn unit_axis(axis: usize) -> Vec3 {
    let mut v = [0.0f32; 3];
    v[axis] = 1.0;
    Vec3::from_array(v)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::arithmetic_side_effects,
        clippy::as_conversions,
        clippy::cast_lossless,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::expect_used,
        clippy::float_cmp,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used
    )]

    use super::Scene;
    use crate::scene::{Node, NodeContent, NodeId, NodeTransform};
    use crate::sketch::{PlaneAxis, Sketch, SketchPoint, SketchSolid};
    use parametric::EvaluationContext;
    use std::num::NonZeroU32;
    use voxel_core::core_geom::MaterialChoice;

    const DENSITY: u32 = 8;

    /// Build a single-node scene holding one extruded sketch and return the node id.
    fn scene_with_sketch(
        sketch: Sketch,
        height_voxels: u32,
        offset_voxels: [i64; 3],
    ) -> (Scene, NodeId) {
        let mut node = Node::new(
            "Sketch",
            NodeContent::SketchTool {
                producer: SketchSolid::extrude(sketch, height_voxels),
                material: MaterialChoice::Stone,
            },
        );
        node.transform = NodeTransform::from_offset_voxels(offset_voxels);
        let scene = Scene::single_node(node);
        let id = scene.roots[0];
        (scene, id)
    }

    #[test]
    fn handles_land_on_profile_corners_of_a_lone_axis_aligned_rectangle() {
        // A 4×6 rectangle on the ground plane (XY), extruded up along Z. A lone node
        // recenters onto the origin, so its handles are symmetric about it.
        let sketch = Sketch::rectangle(PlaneAxis::Z, 4, 6);
        let (scene, id) = scene_with_sketch(sketch, 3, [0, 0, 0]);

        let handles = scene.sketch_handles(id, DENSITY).expect("sketch handles");
        assert_eq!(handles.vertices.len(), 4, "one handle per rectangle corner");

        // Every handle must invert back to the profile coordinate it came from.
        let profile = &[
            SketchPoint::new(0, 0),
            SketchPoint::new(4, 0),
            SketchPoint::new(4, 6),
            SketchPoint::new(0, 6),
        ];
        for (vertex, expected) in handles.vertices.iter().zip(profile) {
            let round_trip = handles.render_hit_to_profile(*vertex);
            assert!(
                (round_trip[0] - expected.offset_voxels[0] as f64).abs() < 1e-3
                    && (round_trip[1] - expected.offset_voxels[1] as f64).abs() < 1e-3,
                "render_hit_to_profile({vertex:?}) = {round_trip:?}, expected {:?}",
                expected.offset_voxels,
            );
        }
    }

    /// The overlay has to be able to tell a handle from the point it stands on.
    ///
    /// A slot pins an authored center onto the center its rails turn about, so the two project to
    /// the same pixel and a hit-test that knows only distance picks between them by accident. What
    /// this pins is that the flags are aligned with the ids AND that the stacking is real — if a
    /// later change stopped stacking them the tie-break would go quietly untested.
    #[test]
    fn stacked_slot_handles_report_which_point_the_drawing_derives() {
        let made = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 4)
            .with_center_arc_slot(
                SketchPoint::new(0, 0),
                SketchPoint::new(8, 0),
                SketchPoint::new(0, 8),
                parametric::sketch::ArcTurn::CounterClockwise,
                SketchPoint::new(10, 0),
                EvaluationContext::new(NonZeroU32::new(DENSITY).unwrap()),
            )
            .expect("a quarter-turn arc slot");
        let (scene, id) = scene_with_sketch((*made.sketch).clone(), 3, [0, 0, 0]);
        let handles = scene.sketch_handles(id, DENSITY).expect("sketch handles");

        assert_eq!(handles.derived.len(), handles.point_ids.len());
        for (index, point) in handles.point_ids.iter().enumerate() {
            assert_eq!(handles.derived[index], made.sketch.is_arc_center(*point));
        }
        let stacked = handles.vertices.iter().enumerate().any(|(index, vertex)| {
            handles.vertices.iter().enumerate().any(|(other, twin)| {
                other != index && twin == vertex && handles.derived[other] != handles.derived[index]
            })
        });
        assert!(stacked, "a slot stands an authored center on a derived one");
    }

    #[test]
    fn handle_extent_matches_the_profile_span_in_render_units() {
        // The rectangle spans 4 voxels along in-plane axis 0 (world X) and 6 along
        // in-plane axis 1 (world Y). The handle bounding box must span exactly that,
        // regardless of where the composite recenter puts the origin.
        let sketch = Sketch::rectangle(PlaneAxis::Z, 4, 6);
        let (scene, id) = scene_with_sketch(sketch, 3, [0, 0, 0]);
        let handles = scene.sketch_handles(id, DENSITY).expect("sketch handles");

        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for v in &handles.vertices {
            for a in 0..3 {
                lo[a] = lo[a].min(v[a]);
                hi[a] = hi[a].max(v[a]);
            }
        }
        assert!(
            (hi[0] - lo[0] - 4.0).abs() < 1e-3,
            "X span 4 voxels, got {}",
            hi[0] - lo[0]
        );
        assert!(
            (hi[1] - lo[1] - 6.0).abs() < 1e-3,
            "Y span 6 voxels, got {}",
            hi[1] - lo[1]
        );
        assert!(
            (hi[2] - lo[2]).abs() < 1e-3,
            "profile is flat on the plane (no Z span)"
        );
    }

    #[test]
    fn inverse_of_an_arbitrary_plane_hit_snaps_to_the_expected_voxel() {
        // A hit a little past a corner inverts to a fractional profile coord the caller
        // would round to the nearest voxel (grid density = voxel density).
        let sketch = Sketch::rectangle(PlaneAxis::Z, 4, 6);
        let (scene, id) = scene_with_sketch(sketch, 3, [0, 0, 0]);
        let handles = scene.sketch_handles(id, DENSITY).expect("sketch handles");

        // Nudge the first handle by (+0.4, −0.3) in the plane's in-plane world axes.
        let [in0, in1] = handles.in_plane_axes;
        let mut hit = handles.vertices[0];
        hit[in0] += 0.4;
        hit[in1] -= 0.3;
        let profile = handles.render_hit_to_profile(hit);
        assert!(
            (profile[0].round() - 0.0).abs() < 1e-6,
            "rounds back to c0 = 0"
        );
        assert!(
            (profile[1].round() - 0.0).abs() < 1e-6,
            "rounds back to c1 = 0"
        );
        // And the fractional part is carried (sub-voxel NoSnap would keep it).
        assert!((profile[0] - 0.4).abs() < 1e-3, "carries the +0.4 fraction");
    }

    #[test]
    fn empty_sketch_has_no_handles_but_a_two_point_sketch_does() {
        // No points ⇒ no vertices, but the plane frame stands so a drawing tool can place
        // the first point: the inverse map answers at the node origin.
        let empty = Sketch::empty(PlaneAxis::Z);
        let (scene, id) = scene_with_sketch(empty, 3, [0, 0, 0]);
        let handles = scene
            .sketch_handles(id, DENSITY)
            .expect("an empty sketch still carries its plane frame");
        assert!(handles.vertices.is_empty(), "no vertices to handle");
        let profile = handles.render_hit_to_profile(handles.plane_point);
        assert!(
            profile[0].abs() < 1e-3 && profile[1].abs() < 1e-3,
            "the plane anchor inverts to the profile origin, got {profile:?}"
        );

        // Two points do not form a closed loop, but every point is still a draggable /
        // deletable handle — entities, not a loop, drive the overlay.
        let open = Sketch::new(
            PlaneAxis::Z,
            vec![SketchPoint::new(0, 0), SketchPoint::new(4, 0)],
        );
        let (scene, id) = scene_with_sketch(open, 3, [0, 0, 0]);
        let handles = scene
            .sketch_handles(id, DENSITY)
            .expect("two-point sketch shows handles");
        assert_eq!(handles.vertices.len(), 2, "one handle per point entity");
    }

    #[test]
    fn handles_carry_a_circle_without_minting_a_perimeter_vertex() {
        let sketch = Sketch::circle(PlaneAxis::Z, SketchPoint::new(2, 3), 5);
        let (scene, id) = scene_with_sketch(sketch, 3, [0, 0, 0]);
        let handles = scene.sketch_handles(id, DENSITY).expect("sketch handles");

        assert_eq!(
            handles.vertices.len(),
            1,
            "only the circle center is a point"
        );
        assert_eq!(handles.circles.len(), 1);
        assert_eq!(handles.circles[0].center, [2.0, 3.0]);
        assert_eq!(handles.circles[0].radius, 5.0);
    }

    #[test]
    fn handle_positions_match_the_resolver_frame_independently() {
        // A frame bug (wrong anchor, a dropped recenter, a half-voxel offset) survives the
        // round-trip tests because forward AND inverse share the bias. This pins the ABSOLUTE
        // render-frame positions against values derived by hand from the resolver's centering
        // rule, NOT from `sketch_handles` itself.
        //
        // Rectangle 4x6 on PlaneAxis::Z (in-plane X,Y; normal Z) extruded 2, single node at the
        // origin. The composite recenter is the AABB center `(min+max).div_euclid(2)` =
        // `[4,6,2]/2 = [2,3,1]`. The profile lives at the producer's local origin corner
        // (bbox-min → local 0) on the base face (normal = 0), so each vertex's render position
        // is `vertex_in_plane − recenter` on X/Y and `0 − recenter_z = −1` on Z.
        let sketch = Sketch::rectangle(PlaneAxis::Z, 4, 6);
        let (scene, id) = scene_with_sketch(sketch, 2, [0, 0, 0]);
        let handles = scene.sketch_handles(id, DENSITY).expect("sketch handles");

        // profile order: (0,0), (4,0), (4,6), (0,6) → render X/Y = coord − [2,3], Z = −1.
        let expected = [
            [-2.0, -3.0, -1.0],
            [2.0, -3.0, -1.0],
            [2.0, 3.0, -1.0],
            [-2.0, 3.0, -1.0],
        ];
        for (vertex, want) in handles.vertices.iter().zip(expected) {
            for axis in 0..3 {
                assert!(
                    (vertex[axis] - want[axis]).abs() < 1e-4,
                    "handle {vertex:?} != expected {want:?} (axis {axis})",
                );
            }
        }

        // Cross-check against a DIFFERENT query: the handles' in-plane centroid must coincide
        // with the transform gizmo's pivot (the node AABB center in the same render frame),
        // which is the origin for a lone centered node.
        let (pivot, _extent) = scene
            .gizmo_placement_for_id(id, DENSITY)
            .expect("gizmo placement");
        let mut centroid = [0.0f32; 3];
        for vertex in &handles.vertices {
            for axis in 0..3 {
                centroid[axis] += vertex[axis] / handles.vertices.len() as f32;
            }
        }
        assert!(
            (centroid[0] - pivot[0]).abs() < 1e-4,
            "in-plane X centroid == gizmo pivot X"
        );
        assert!(
            (centroid[1] - pivot[1]).abs() < 1e-4,
            "in-plane Y centroid == gizmo pivot Y"
        );
        assert!(
            pivot[0].abs() < 1e-4 && pivot[1].abs() < 1e-4,
            "lone node pivots on the origin"
        );
    }

    /// Placing an entity that reaches past the filled region must not move what is already
    /// drawn. The overlay anchors on the RESOLVE's anchor, so extending the drawing extends it —
    /// it does not drag the drawing.
    ///
    /// Anchoring on the bbox over the real points instead lets a point outside the fill move
    /// the anchor, and every handle, segment, arc and the region wash with it. Nothing cancels
    /// that, because the anchor compensation applied on every edit corrects for a change in
    /// the resolve's anchor, which has not moved.
    #[test]
    fn a_point_reaching_past_the_fill_does_not_move_the_drawing() {
        let sketch = Sketch::rectangle(PlaneAxis::Z, 4, 6);
        let (scene, id) = scene_with_sketch(sketch.clone(), 3, [0, 0, 0]);
        let before = scene.sketch_handles(id, DENSITY).expect("sketch handles");

        let mut grown = sketch;
        grown.add_free_point(SketchPoint::new(-20, -20));
        let (scene, id) = scene_with_sketch(grown, 3, [0, 0, 0]);
        let after = scene.sketch_handles(id, DENSITY).expect("sketch handles");

        assert_eq!(after.vertices.len(), 5, "the placed point is drawn too");
        for (index, corner) in before.vertices.iter().enumerate() {
            assert_eq!(
                &after.vertices[index], corner,
                "corner {index} moved when a point was placed outside the fill"
            );
        }
    }

    /// The same, for a line drawn from a corner out past the profile.
    /// The chain dangles, so it encloses nothing and the resolved solid is unchanged — which is
    /// exactly why the drawing must not move either.
    #[test]
    fn a_line_drawn_past_the_fill_does_not_move_the_drawing() {
        let sketch = Sketch::rectangle(PlaneAxis::Z, 4, 6);
        let (scene, id) = scene_with_sketch(sketch.clone(), 3, [0, 0, 0]);
        let before = scene.sketch_handles(id, DENSITY).expect("sketch handles");

        let mut grown = sketch;
        let corner = grown.points()[0].id;
        let reached = grown.add_free_point(SketchPoint::new(-30, 12));
        grown.connect(corner, reached);
        let (scene, id) = scene_with_sketch(grown, 3, [0, 0, 0]);
        let after = scene.sketch_handles(id, DENSITY).expect("sketch handles");

        for (index, corner) in before.vertices.iter().enumerate() {
            assert_eq!(
                &after.vertices[index], corner,
                "corner {index} moved when a line was drawn past the fill"
            );
        }
    }

    /// The overlay and the resolve share ONE anchor. Stated directly, because the two agreeing
    /// is the property every test above depends on.
    #[test]
    fn the_overlay_anchors_where_the_resolve_does() {
        let mut sketch = Sketch::rectangle(PlaneAxis::Z, 4, 6);
        sketch.add_free_point(SketchPoint::new(-20, -20));
        let producer = SketchSolid::extrude(sketch, 3);
        let (scene, id) = scene_with_sketch((*producer.sketch).clone(), 3, [0, 0, 0]);
        let handles = scene.sketch_handles(id, DENSITY).expect("sketch handles");
        // The profile origin maps to producer-local zero exactly when the two anchors agree.
        let anchor = producer.profile_bbox_min(
            crate::sketch::evaluation_context_from_density(DENSITY)
                .expect("test density is non-zero"),
        );
        assert_eq!(
            handles.render_hit_to_profile(
                handles.profile_to_render([anchor[0] as f64, anchor[1] as f64])
            ),
            [anchor[0] as f64, anchor[1] as f64],
        );
    }

    #[test]
    fn zero_height_profile_still_yields_handles() {
        // Nothing resolves at height 0, but the profile is still authorable, so its
        // vertices must remain draggable.
        let sketch = Sketch::rectangle(PlaneAxis::Z, 4, 6);
        let (scene, id) = scene_with_sketch(sketch, 0, [0, 0, 0]);
        let handles = scene.sketch_handles(id, DENSITY);
        assert!(
            handles.is_some(),
            "a zero-height sketch still shows draggable handles"
        );
        assert_eq!(handles.unwrap().vertices.len(), 4);
    }
}
