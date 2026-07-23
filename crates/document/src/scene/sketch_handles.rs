//! Sketch-mode vertex handles: each profile vertex's position in the display's
//! recentred **render frame**, plus the inverse map (a cursor hit on the sketch plane
//! back to a profile `(c0, c1)` voxel coordinate). This is the geometry #94's
//! interactive vertex drag draws and hit-tests.
//!
//! **The frame is carried, never re-derived (ADR 0008).** Every position is routed
//! through the SAME [`substrate::spatial::LeafPlacement`] the resolver folds occupancy
//! through — the profile vertex is a producer-LOCAL voxel point, `world_of` places it,
//! and the composite recentre rebases it into the render frame — so a handle coincides
//! with the resolved geometry's profile corner BY CONSTRUCTION rather than by a
//! kept-in-sync mirror (mirroring the placement ghost's `center_world`).
//!
//! **The convention is corner-anchored, like extrude.** A profile point `(c0, c1)` maps
//! to producer-local `(c0 − min0, c1 − min1)` on the plane's two in-plane axes, at `0`
//! along the plane normal — the profile's bounding-box minimum sits on the node's world
//! anchor. That is exactly where the extrude resolve seats the profile; the handles
//! therefore represent the 2D PROFILE on its plane (the authoring surface), independent
//! of which operation later lifts it into a volume.

use super::*;
use crate::sketch::Operation;
use glam::Vec3;
use substrate::spatial::{LeafPlacement, ProducerLocalVoxelPoint, TrueWorldVoxelPoint};

/// The sketch's profile vertices in the recentred render frame, with everything the UI
/// needs to draw draggable handles and turn a cursor ray back into a profile coordinate.
///
/// Positions are in the SAME render frame the resolved voxels and the transform gizmo
/// live in (voxel units, composite-recentred), so the UI projects them through the same
/// `view_projection` it uses for everything else.
#[derive(Debug, Clone)]
pub struct SketchHandles {
    /// Each profile vertex's position in the render frame, in profile order (index `i`
    /// is `producer.sketch.profile[i]`).
    pub vertices: Vec<[f32; 3]>,
    /// A point ON the sketch plane in the render frame (the first vertex) — the ray
    /// intersection anchor.
    pub plane_point: [f32; 3],
    /// The sketch plane's unit normal in the render frame (`rotation · e_normal`).
    pub plane_normal: [f32; 3],
    /// The placement affine (carried so the inverse map rotates through the exact same
    /// transform the forward map placed vertices with).
    placement: LeafPlacement,
    /// The composite recentre (render frame = true world − recentre), in voxels.
    recentre: [i64; 3],
    /// The profile's in-plane bounding-box minimum, in voxels — added back so a local
    /// coordinate returns to absolute profile space.
    profile_min: [i64; 2],
    /// The plane's two in-plane world axes (`PlaneAxis::in_plane_axes`).
    in_plane_axes: [usize; 2],
}

impl SketchHandles {
    /// Map a hit point on the sketch plane (in the RENDER frame — e.g. a cursor ray's
    /// intersection with [`plane_point`](Self::plane_point) / [`plane_normal`](Self::plane_normal))
    /// back to a CONTINUOUS profile coordinate `(c0, c1)` in voxels. The caller snaps it
    /// (round for grid-snap, floor+fraction for sub-voxel) and writes it into the dragged
    /// `SketchPoint.offset_voxels`.
    ///
    /// The inverse of the forward placement: rebase the render hit into true world
    /// (`+ recentre`), invert the placement to producer-local, read the two in-plane
    /// components and add the profile minimum back. `render_hit` need not lie exactly on
    /// the plane — the normal component is simply discarded by reading only the in-plane
    /// axes — but a ray/plane intersection keeps it on-plane so the drag tracks the cursor.
    pub fn render_hit_to_profile(&self, render_hit: [f32; 3]) -> [f64; 2] {
        let world = Vec3::new(
            render_hit[0] + self.recentre[0] as f32,
            render_hit[1] + self.recentre[1] as f32,
            render_hit[2] + self.recentre[2] as f32,
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
    /// The [`SketchHandles`] for the sketch node `node_id` — each profile vertex placed
    /// into the render frame, plus the inverse cursor-to-profile map (#94 vertex drag).
    /// `None` when the id is not an enabled `SketchTool` node, or its profile has fewer
    /// than three vertices (no polygon to handle).
    ///
    /// Independent of the operation's degeneracy: a profile whose extrude height is still
    /// `0` (nothing resolves yet) STILL returns handles, so the vertices are draggable
    /// while the sketch is being authored.
    pub fn sketch_handles(
        &self,
        node_id: NodeId,
        voxels_per_block: u32,
    ) -> Option<SketchHandles> {
        let node = self.node_by_id(node_id)?;
        if !node.enabled {
            return None;
        }
        let NodeContent::SketchTool { producer, .. } = &node.content else {
            return None;
        };
        let profile = &producer.sketch.profile;
        if profile.len() < 3 {
            return None;
        }

        // The profile's in-plane bounding box (min anchors the corner-anchored frame).
        let mut min = profile[0].offset_voxels;
        let mut max = min;
        for point in profile {
            for axis in 0..2 {
                min[axis] = min[axis].min(point.offset_voxels[axis]);
                max[axis] = max[axis].max(point.offset_voxels[axis]);
            }
        }

        let [in0, in1] = producer.sketch.plane.in_plane_axes();
        let normal = producer.sketch.plane.normal_axis();

        // The producer-local box extent `full`. The two in-plane axes span the profile
        // bbox; the normal axis carries the operation's extrude thickness (0 for revolve /
        // a not-yet-extruded profile). `full` only re-anchors the box under a genuine
        // rotation (`min_rotated_corner`); with an axis-aligned plane (every plane today)
        // the rotation is identity and it drops out — but it is routed through
        // `LeafPlacement` so a future free-angle plane stays exact (ADR 0027).
        let mut full = [0.0f32; 3];
        full[in0] = (max[0] - min[0]) as f32;
        full[in1] = (max[1] - min[1]) as f32;
        full[normal] = match producer.operation {
            Operation::Extrude { height_voxels } => height_voxels as f32,
            Operation::Revolve { .. } => 0.0,
        };

        // The node's world placement: accumulated parent offset + its own integer offset,
        // its sub-voxel slide, and its continuous rotation (ADR 0027).
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

        let recentre = self.recentre_voxels_for_resolve(voxels_per_block).voxels();
        let recentre_vec = Vec3::new(
            recentre[0] as f32,
            recentre[1] as f32,
            recentre[2] as f32,
        );

        let vertices: Vec<[f32; 3]> = profile
            .iter()
            .map(|point| {
                let mut local = [0.0f32; 3];
                local[in0] = (point.offset_voxels[0] - min[0]) as f32;
                local[in1] = (point.offset_voxels[1] - min[1]) as f32;
                // local[normal] stays 0.0 — the profile lives on the plane.
                let world = placement
                    .world_of(ProducerLocalVoxelPoint::from_voxels(Vec3::from_array(local)))
                    .voxels();
                (world - recentre_vec).to_array()
            })
            .collect();

        let plane_normal = (node.transform.rotation() * unit_axis(normal)).to_array();
        let plane_point = vertices[0];

        Some(SketchHandles {
            vertices,
            plane_point,
            plane_normal,
            placement,
            recentre,
            profile_min: min,
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
    use super::Scene;
    use crate::scene::{Node, NodeContent, NodeId, NodeTransform};
    use crate::sketch::{PlaneAxis, Sketch, SketchPoint, SketchSolid};
    use voxel_core::core_geom::MaterialChoice;

    const DENSITY: u32 = 8;

    /// Build a single-node scene holding one extruded sketch and return the node id.
    fn scene_with_sketch(sketch: Sketch, height_voxels: u32, offset_voxels: [i64; 3]) -> (Scene, NodeId) {
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
        // recentres onto the origin, so its handles are symmetric about it: the profile
        // spans 4 voxels in X and 6 in Z... no — plane Z ⇒ in-plane axes are X, Y.
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

    #[test]
    fn handle_extent_matches_the_profile_span_in_render_units() {
        // The rectangle spans 4 voxels along in-plane axis 0 (world X) and 6 along
        // in-plane axis 1 (world Y). The handle bounding box must span exactly that,
        // regardless of where the composite recentre puts the origin.
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
        assert!((hi[0] - lo[0] - 4.0).abs() < 1e-3, "X span 4 voxels, got {}", hi[0] - lo[0]);
        assert!((hi[1] - lo[1] - 6.0).abs() < 1e-3, "Y span 6 voxels, got {}", hi[1] - lo[1]);
        assert!((hi[2] - lo[2]).abs() < 1e-3, "profile is flat on the plane (no Z span)");
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
        assert!((profile[0].round() - 0.0).abs() < 1e-6, "rounds back to c0 = 0");
        assert!((profile[1].round() - 0.0).abs() < 1e-6, "rounds back to c1 = 0");
        // And the fractional part is carried (sub-voxel NoSnap would keep it).
        assert!((profile[0] - 0.4).abs() < 1e-3, "carries the +0.4 fraction");
    }

    #[test]
    fn degenerate_profile_returns_none() {
        // A two-point "profile" is not a polygon (fewer than three vertices).
        let sketch = Sketch::new(PlaneAxis::Z, vec![SketchPoint::new(0, 0), SketchPoint::new(4, 0)]);
        let (scene, id) = scene_with_sketch(sketch, 3, [0, 0, 0]);
        assert!(scene.sketch_handles(id, DENSITY).is_none(), "degenerate profile ⇒ None");
    }

    #[test]
    fn handle_positions_match_the_resolver_frame_independently() {
        // A frame bug (wrong anchor, a dropped recentre, a half-voxel offset) survives the
        // round-trip tests because forward AND inverse share the bias. This pins the ABSOLUTE
        // render-frame positions against values derived by hand from the resolver's centring
        // rule, NOT from `sketch_handles` itself.
        //
        // Rectangle 4x6 on PlaneAxis::Z (in-plane X,Y; normal Z) extruded 2, single node at the
        // origin. The composite recentre is the AABB centre `(min+max).div_euclid(2)` =
        // `[4,6,2]/2 = [2,3,1]`. The profile lives at the producer's local origin corner
        // (bbox-min → local 0) on the base face (normal = 0), so each vertex's render position
        // is `vertex_in_plane − recentre` on X/Y and `0 − recentre_z = −1` on Z.
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
        // with the transform gizmo's pivot (the node AABB centre in the same render frame),
        // which is the origin for a lone centred node.
        let (pivot, _extent) = scene
            .gizmo_placement_for_id(id, DENSITY)
            .expect("gizmo placement");
        let mut centroid = [0.0f32; 3];
        for vertex in &handles.vertices {
            for axis in 0..3 {
                centroid[axis] += vertex[axis] / handles.vertices.len() as f32;
            }
        }
        assert!((centroid[0] - pivot[0]).abs() < 1e-4, "in-plane X centroid == gizmo pivot X");
        assert!((centroid[1] - pivot[1]).abs() < 1e-4, "in-plane Y centroid == gizmo pivot Y");
        assert!(pivot[0].abs() < 1e-4 && pivot[1].abs() < 1e-4, "lone node pivots on the origin");
    }

    #[test]
    fn zero_height_profile_still_yields_handles() {
        // Nothing resolves at height 0, but the profile is still authorable, so its
        // vertices must remain draggable.
        let sketch = Sketch::rectangle(PlaneAxis::Z, 4, 6);
        let (scene, id) = scene_with_sketch(sketch, 0, [0, 0, 0]);
        let handles = scene.sketch_handles(id, DENSITY);
        assert!(handles.is_some(), "a zero-height sketch still shows draggable handles");
        assert_eq!(handles.unwrap().vertices.len(), 4);
    }
}
