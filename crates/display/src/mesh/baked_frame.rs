//! The one door from a camera into the frame an artifact's buffers were baked in.

use voxel_core::voxel::RecenterVoxels;

/// A GPU artifact whose vertices stand in a floating origin recorded when they were built.
///
/// The scene's origin is the midpoint of the composite extent, so it moves the moment any shape
/// grows. Buffers do not move with it: an async mesh goes on standing in the frame it was emitted
/// in until the next build lands, and the selection outline and the operand ghost rebuild on a
/// SELECTION change while the origin moves on a GEOMETRY change. So between events there is no
/// single frame that serves every pass, and each one has to be asked which frame it is in.
///
/// Implementing this is how a pass gets a camera. The walk is a provided method that reads the
/// baked side out of `self`, which is the point: the two frames are the same type, so a free
/// function taking both would let a caller transpose them, and a transposition is a bit-exact
/// no-op whenever the frames agree — every golden, every parity test, and the whole steady state.
/// It would go green and reintroduce the drift in the async window alone. Reading one side from
/// the artifact means that call cannot be written.
pub(crate) trait BakedInAFrame {
    /// The frame this artifact's resident buffers are expressed in.
    fn baked_frame(&self) -> RecenterVoxels;

    /// The camera to draw this artifact with, given the frame the CALLER is standing in.
    ///
    /// Nothing here moves a vertex — the two frames differ by a pure translation, so this is one
    /// concat against however many vertices, and it leaves a buffer somebody else may still be
    /// filling untouched.
    fn camera_visiting(
        &self,
        view_projection: glam::Mat4,
        current_frame: RecenterVoxels,
    ) -> CameraInBakedFrame {
        CameraInBakedFrame(walk_camera(
            view_projection,
            self.baked_frame(),
            current_frame,
        ))
    }
}

/// A camera that has been walked into some artifact's baked frame.
///
/// Only [`BakedInAFrame::camera_visiting`] can build one — the field is private to this module —
/// and the uniform builders for baked passes accept nothing else. So the remaining way to get
/// this wrong is closed too: a pass cannot pack the caller's raw camera by simply not walking it,
/// because that camera has the wrong type. Skipping the door is a compile error rather than a
/// delta-zero green that looks right in every steady-state artifact and drifts only in the window
/// where a rebuild is in flight.
///
/// It also makes the distinction legible in signatures. A pass drawing LIVE content — points,
/// grids, sketch billboards, anything rebuilt from the scene this frame — keeps taking a plain
/// `Mat4`, because it has no bake to stand in. The type says which kind of pass you are reading.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CameraInBakedFrame(glam::Mat4);

impl CameraInBakedFrame {
    /// The matrix, for packing into a uniform. The guarantee is on where it came from, not on
    /// what is done with it here.
    pub(crate) fn matrix(self) -> glam::Mat4 {
        self.0
    }
}

/// A camera built for `current_frame`, re-expressed to draw vertices baked in `baked_frame`.
///
/// Private on purpose: [`BakedInAFrame::camera_visiting`] is the only way in, so no caller ever
/// names both frames. Written out separately because the DIRECTION is the whole content and a
/// sign error here draws a scene that looks plausible and stands in the wrong place.
///
/// When the frames agree — the steady state — the matrix is returned untouched rather than
/// multiplied by an identity, so the parity is bit-level rather than close enough.
fn walk_camera(
    view_projection: glam::Mat4,
    baked_frame: RecenterVoxels,
    current_frame: RecenterVoxels,
) -> glam::Mat4 {
    let walk = baked_frame.a_point_of_this_frame_seen_from(current_frame);
    if walk == glam::Vec3::ZERO {
        return view_projection;
    }
    view_projection * glam::Mat4::from_translation(walk)
}

#[cfg(test)]
mod tests {
    // The inert-walk assert below is a BIT-identity check: `close enough` would pass on the
    // very multiply it exists to forbid.
    #![allow(clippy::float_cmp)]

    use super::*;

    /// **A stale mesh draws where it always stood.**
    ///
    /// The reported symptom: grow a shape and the voxels drift away from the sketch beside them
    /// until they are rebuilt. Nothing rescaled — the render origin is the midpoint of the
    /// composite extent, so growing a shape moves it, the camera is compensated at dispatch, and
    /// the mesh goes on standing in the frame it was emitted in until the next build lands. For
    /// that window the camera is somewhere the vertices are not, and the whole model slides by
    /// the difference.
    ///
    /// The far corner of a shape that did NOT change must image at exactly the same clip position
    /// before and after a growth it had no part in. It is checked at a wide baseline, because a
    /// floating origin exists for wide baselines and a walk that is merely close is a walk that
    /// still drifts out there.
    ///
    /// This pins the arithmetic and its direction, and nothing else: the frames and the vertex
    /// are hand-built here. Whether the pass hands the walk the right two frames is a separate
    /// question, asked of the uniform the pass actually uploads — see
    /// `a_grown_neighbour_does_not_move_the_uniform_a_stale_mesh_uploads`.
    #[test]
    fn a_mesh_baked_in_an_older_frame_still_images_where_it_belongs() {
        let camera_of = |frame: RecenterVoxels| {
            // A camera whose target is pinned to the same WORLD point in each frame — which is
            // exactly what the shell's recenter-shift compensation maintains.
            let world_target = glam::Vec3::new(4000.0, 250.0, 60.0);
            let offset = glam::Vec3::from_array(frame.voxels().map(|axis| axis as f32));
            glam::Mat4::perspective_rh(0.9, 1.6, 1.0, 100_000.0)
                * glam::Mat4::look_at_rh(
                    world_target - offset + glam::Vec3::new(900.0, -1300.0, 700.0),
                    world_target - offset,
                    glam::Vec3::Z,
                )
        };
        let clip_of = |world: glam::Vec3, baked: RecenterVoxels, current: RecenterVoxels| {
            let vertex = world - glam::Vec3::from_array(baked.voxels().map(|axis| axis as f32));
            walk_camera(camera_of(current), baked, current) * vertex.extend(1.0)
        };

        // The far corner of the untouched shape, in true world voxels.
        let corner = glam::Vec3::new(3200.0, -180.0, 44.0);
        let opening = RecenterVoxels::new([640, -20, 8]);
        let settled = clip_of(corner, opening, opening);

        // The neighbour grows; the origin moves by half of it on one axis, and by a little on the
        // others because a composite AABB is not axis-independent.
        for grown in [
            RecenterVoxels::new([1280, -20, 8]),
            RecenterVoxels::new([12_800, -20, 8]),
            RecenterVoxels::new([1280, 340, -96]),
        ] {
            let stale = clip_of(corner, opening, grown);
            for axis in 0..4 {
                assert!(
                    (stale[axis] - settled[axis]).abs() < 1e-2,
                    "the untouched corner moved on clip axis {axis} when the origin went to \
                     {:?}: {} vs {}",
                    grown.voxels(),
                    stale[axis],
                    settled[axis],
                );
            }
        }

        // And the walk is inert when there is nothing to walk: same matrix, bit for bit.
        let camera = camera_of(opening);
        assert_eq!(
            walk_camera(camera, opening, opening).to_cols_array(),
            camera.to_cols_array(),
            "an agreeing frame must leave the matrix untouched, not multiply it by an identity",
        );
    }
}
