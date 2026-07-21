//! **Cursor picking** — turning a screen position into the voxel under it
//! (`docs/design/direct-manipulation.md`, the picked point).
//!
//! Every tool in the grammar needs the same primitive: *what is under the cursor, and which way
//! is its surface facing.* The armed preview lands there, the sketch plane aligns to the normal
//! it returns, and the manipulator hit-tests against it. Written once here rather than per tool,
//! because tools that each answer "what did I click" their own way would disagree.
//!
//! **This queries CPU truth, never a display artifact** (ADR 0006). The occupancy comes from the
//! resident two-layer chunks — the same set the mesher and the brick sink read — through
//! [`TwoLayerChunk::voxel_occupied`]. It deliberately does not march the brick field: the shell
//! uploads its bricks and drops the CPU build, so there is nothing on this side to march, and
//! picking a *display cache* would make selection depend on what the renderer happened to have
//! resident.
//!
//! The kernel is [`raycast::march_exact_occupancy`], the same flat voxel DDA the parity net uses
//! as its independent content oracle. Picking therefore agrees with what is drawn by
//! construction rather than by a second implementation kept in step.

use std::collections::HashMap;
use std::sync::Arc;

use camera::unproject_screen_point_to_ray;
use display::renderer::LayerBand;
use evaluation::two_layer_store::TwoLayerChunk;
use voxel_core::core_geom::CHUNK_BLOCKS;

use super::AppCore;

/// What the cursor is over: a voxel, and the face the ray entered it through.
///
/// `face_normal` is an exact `±1` axis vector — a consequence of a voxel lattice rather than a
/// tolerance, which is what makes "align this to the surface I clicked" a finite question with
/// no fallback case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoxelPick {
    /// The hit voxel's ABSOLUTE lattice coordinate (ADR 0008 — absolute, not render-frame, so
    /// the caller never has to know which frame it came back in).
    pub absolute_voxel: [i64; 3],
    /// The outward normal of the face the ray entered through.
    pub face_normal: [i32; 3],
}

/// Everything a pick needs about the frame it is picking in — all of it produced by the last
/// [`rebuild`](AppCore::rebuild), which is why it travels as one value rather than as five
/// parameters that must be kept mutually consistent by every caller.
pub struct PickFrame<'a> {
    /// The region's voxel dimensions, whose floored half corner-anchors the render frame.
    pub region_dimensions: [u32; 3],
    /// The floating origin the render frame is expressed against (ADR 0008).
    pub recentre_voxels: [i64; 3],
    /// Voxels per block.
    pub density: u32,
    /// The resident two-layer chunks — CPU truth, borrowed from the last rebuild.
    pub chunks: &'a [([i32; 3], Arc<TwoLayerChunk>)],
    /// The band the caller DREW with, so a pick cannot select what is not on screen.
    pub band: LayerBand,
}

impl AppCore {
    /// The voxel under a cursor position, or `None` if the ray misses all geometry.
    ///
    /// `cursor` and `viewport` are in the same physical-pixel space (`[x, y, width, height]`).
    /// The frame's chunks come from the last [`rebuild`](AppCore::rebuild) — borrowed rather
    /// than re-resolved, because the shell already holds them and re-resolving to answer a hover
    /// would be absurd.
    ///
    /// **The band is honoured.** A pick clipped out of the visible band must miss: picking
    /// something you cannot see is a worse failure than picking nothing, and the onion viewer
    /// modes make invisible-but-present geometry an ordinary state rather than an edge case.
    pub fn pick_voxel(
        &self,
        cursor: [f32; 2],
        viewport: [f32; 4],
        frame: &PickFrame<'_>,
    ) -> Option<VoxelPick> {
        let PickFrame { region_dimensions, recentre_voxels, density, chunks, band } = *frame;
        if chunks.is_empty() || viewport[2] <= 0.0 || viewport[3] <= 0.0 {
            return None;
        }
        let density = density.max(1);
        let chunk_extent = (CHUNK_BLOCKS * density) as i64;

        // The march runs in the SHADING-ABSOLUTE frame the display path uses: the render frame
        // corner-anchored by the floored half. `absolute = shading_absolute + (recentre − half)`.
        // The brick path additionally splits that bias into a lattice shift so block boundaries
        // land on brick-edge multiples; a pick has no bricks to align to, so it uses the whole
        // bias directly and no shift.
        let half = region_dimensions.map(|dimension| (dimension / 2) as i64);
        let shading_to_absolute: [i64; 3] =
            std::array::from_fn(|axis| recentre_voxels[axis] - half[axis]);

        let aspect_ratio = viewport[2] / viewport[3];
        let view_projection = self.view_projection(aspect_ratio, region_dimensions);
        let normalized_x = (cursor[0] - viewport[0]) / viewport[2] * 2.0 - 1.0;
        let normalized_y = 1.0 - (cursor[1] - viewport[1]) / viewport[3] * 2.0;
        let world_ray = unproject_screen_point_to_ray(view_projection, normalized_x, normalized_y)?;
        // Into the shading-absolute frame: the same `+ grid_half_extent` the display's camera ray
        // applies, so a pick and a pixel agree about where the ray is.
        let half_extent = glam::Vec3::new(half[0] as f32, half[1] as f32, half[2] as f32);
        let march_ray =
            substrate::spatial::Ray::new(world_ray.origin + half_extent, world_ray.direction);

        // Index the resident set by chunk coord so the occupancy closure is a hash lookup per
        // step rather than a scan of the covering set — a ray crosses many chunks, and the huge
        // scenes hold thousands.
        let resident: HashMap<[i32; 3], &TwoLayerChunk> = chunks
            .iter()
            .map(|(coord, chunk)| (*coord, chunk.as_ref()))
            .collect();

        // The traversal box: the resident chunks' absolute bounds, expressed in the march frame.
        let mut lo = [i64::MAX; 3];
        let mut hi = [i64::MIN; 3];
        for (coord, _) in chunks {
            for axis in 0..3 {
                let chunk_lo = coord[axis] as i64 * chunk_extent;
                lo[axis] = lo[axis].min(chunk_lo);
                hi[axis] = hi[axis].max(chunk_lo + chunk_extent);
            }
        }
        let to_march_frame = |bound: [i64; 3]| {
            glam::Vec3::new(
                (bound[0] - shading_to_absolute[0]) as f32,
                (bound[1] - shading_to_absolute[1]) as f32,
                (bound[2] - shading_to_absolute[2]) as f32,
            )
        };

        // The band is REGION-LOCAL Z layers (`queries.rs` clamps `band_min`/`band_max` to
        // `[0, scene_grid_z]`), and the march voxel frame is ALSO region-local — `march_voxel =
        // absolute − shading_to_absolute`, and `shading_to_absolute` IS the region's absolute base
        // (`recentre − half`), so the two frames coincide. The band therefore needs NO conversion:
        // `band_low = band_min`, half-open `[band_min, band_max + 1)`. (The brick raymarch, the
        // working reference, likewise takes `band_min` directly, only adding its block-align
        // `lattice_shift`, which a pick does not have.) An earlier `− shading_to_absolute[2] +
        // half[2]` here added a spurious `half[2]` floor — for a centred scene it clipped every
        // voxel below the region mid-plane, making the LOWER HALF of any object unpickable.
        let clamp_to_i32 = |value: i64| value.clamp(i32::MIN as i64 + 1, i32::MAX as i64 - 1) as i32;
        let band_low = clamp_to_i32(band.band_min as i64);
        let band_high = clamp_to_i32(band.band_max as i64 + 1);

        let params = raycast::ExactMarchParams {
            traversal_lo: to_march_frame(lo),
            traversal_hi: to_march_frame(hi),
            band_voxel_sv: [band_low, band_high],
            voxel_bias: [
                clamp_to_i32(shading_to_absolute[0]),
                clamp_to_i32(shading_to_absolute[1]),
                clamp_to_i32(shading_to_absolute[2]),
            ],
        };

        let hit = raycast::march_exact_occupancy(march_ray, &params, |absolute| {
            let chunk_coord = [
                (absolute[0].div_euclid(chunk_extent)) as i32,
                (absolute[1].div_euclid(chunk_extent)) as i32,
                (absolute[2].div_euclid(chunk_extent)) as i32,
            ];
            match resident.get(&chunk_coord) {
                // A chunk outside the resident set is air, not a miss to resolve — the covering
                // set is every chunk the scene's AABB touches, so absence IS emptiness.
                None => false,
                Some(chunk) => chunk.voxel_occupied([
                    absolute[0].rem_euclid(chunk_extent) as u32,
                    absolute[1].rem_euclid(chunk_extent) as u32,
                    absolute[2].rem_euclid(chunk_extent) as u32,
                ]),
            }
        })?;

        Some(VoxelPick {
            absolute_voxel: [
                hit.absolute_voxel[0] as i64,
                hit.absolute_voxel[1] as i64,
                hit.absolute_voxel[2] as i64,
            ],
            face_normal: hit.face_normal,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use camera::OrbitCamera;
    use document::scene::Scene;
    use document::voxel::{GeometryParams, SdfShape};
    use voxel_core::core_geom::MaterialChoice;
    use voxel_core::voxel::ShapeKind;

    use crate::{AppCore, RebuildOutcome};

    const DENSITY: u32 = 8;
    const VIEWPORT: [f32; 4] = [0.0, 0.0, 1280.0, 720.0];

    /// A rebuilt one-node scene plus the frame a pick runs against.
    struct Fixture {
        app_core: AppCore,
        region_dimensions: [u32; 3],
        recentre_voxels: [i64; 3],
        chunks: Vec<([i32; 3], Arc<TwoLayerChunk>)>,
    }

    impl Fixture {
        /// The frame at `band`, borrowing this fixture's chunks.
        fn frame(&self, band: LayerBand) -> PickFrame<'_> {
            PickFrame {
                region_dimensions: self.region_dimensions,
                recentre_voxels: self.recentre_voxels,
                density: DENSITY,
                chunks: &self.chunks,
                band,
            }
        }
    }

    /// A one-node scene, rebuilt, with everything a pick needs alongside it.
    fn picking_fixture(kind: ShapeKind, blocks: [u32; 3]) -> Fixture {
        let shape = SdfShape::from_blocks(kind, blocks, 1, DENSITY);
        let scene = Scene::from_geometry(
            GeometryParams {
                shape: kind,
                size_voxels: shape.size_voxels,
                size_measurements: None,
                voxels_per_block: DENSITY,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        );
        let mut app_core = AppCore::new(OrbitCamera::default());
        let RebuildOutcome::Built(output) = app_core.rebuild(&scene, DENSITY) else {
            panic!("the fixture's density is in bounds");
        };
        let region_dimensions = output.region_dimensions;
        let recentre_voxels = output.recentre_voxels.voxels();
        let chunks = output.two_layer_chunks.clone();
        Fixture { app_core, region_dimensions, recentre_voxels, chunks }
    }

    /// **The invariant that catches a frame error.** Whatever the pick returns must be a
    /// genuinely SOLID voxel — checked against the same chunks, independently of the march.
    ///
    /// This is the assertion worth making because the failure mode it guards is silent: the
    /// march frame carries a bias (`absolute = shading_absolute + recentre − half`), and getting
    /// that bias wrong by one voxel still returns a plausible-looking hit near the object. Only
    /// asking "is the thing you named actually solid" catches it, and it catches it on every ray
    /// rather than at a hand-computed coordinate.
    #[test]
    fn a_pick_names_a_voxel_that_is_actually_solid() {
        // The two fixtures do different jobs, so they earn different bars. The sphere is a
        // solid presenting a large silhouette: it should catch nearly every ray, and a low
        // count there would mean the frame is wrong. The tube is thin-walled and hollow — its
        // value is that almost every block is BOUNDARY, so the cuboid path carries the whole
        // answer with no coarse layer to mask a mistake — but a flat annulus presents little
        // area, so demanding ray density from it would only pin the camera's default framing.
        for (label, kind, blocks, minimum_hits) in [
            ("sphere", ShapeKind::Sphere, [6u32, 6, 6], 25),
            ("tube", ShapeKind::Tube, [8, 4, 8], 1),
        ] {
            let fixture = picking_fixture(kind, blocks);
            let chunk_extent = (CHUNK_BLOCKS * DENSITY) as i64;
            let resident: HashMap<[i32; 3], &TwoLayerChunk> = fixture
                .chunks
                .iter()
                .map(|(coord, chunk)| (*coord, chunk.as_ref()))
                .collect();

            // Sweep a grid of cursor positions so the assertion covers rays entering through
            // every face, not just the one the centre ray happens to hit.
            let mut hits = 0;
            for row in 1..8 {
                for column in 1..8 {
                    let cursor = [
                        VIEWPORT[2] * column as f32 / 8.0,
                        VIEWPORT[3] * row as f32 / 8.0,
                    ];
                    let Some(pick) = fixture.app_core.pick_voxel(
                        cursor,
                        VIEWPORT,
                        &fixture.frame(LayerBand::FULL),
                    ) else {
                        continue;
                    };
                    hits += 1;

                    let coord = pick.absolute_voxel.map(|v| v.div_euclid(chunk_extent) as i32);
                    let local = pick
                        .absolute_voxel
                        .map(|v| v.rem_euclid(chunk_extent) as u32);
                    let solid = resident
                        .get(&coord)
                        .is_some_and(|chunk| chunk.voxel_occupied(local));
                    assert!(
                        solid,
                        "[{label}] pick at {cursor:?} named {:?}, which is not solid",
                        pick.absolute_voxel
                    );

                    // An entered face's normal is one unit axis vector, never a diagonal.
                    let magnitude: i32 = pick.face_normal.iter().map(|c| c.abs()).sum();
                    assert_eq!(
                        magnitude, 1,
                        "[{label}] face normal {:?} must be a unit axis vector",
                        pick.face_normal
                    );
                }
            }
            eprintln!("[{label}] {hits}/49 rays hit");
            assert!(
                hits >= minimum_hits,
                "[{label}] only {hits}/49 rays hit, expected at least {minimum_hits} — too few \
                 for the solidity check above to mean much"
            );
        }
    }

    /// An empty resident set misses rather than panicking or reporting a phantom voxel. The
    /// shell can ask for a pick on any frame, including before anything is built.
    #[test]
    fn a_pick_against_nothing_misses() {
        let fixture = picking_fixture(ShapeKind::Sphere, [4, 4, 4]);
        let empty = PickFrame { chunks: &[], ..fixture.frame(LayerBand::FULL) };
        let pick = fixture.app_core.pick_voxel([640.0, 360.0], VIEWPORT, &empty);
        assert_eq!(pick, None, "an empty resident set has nothing to hit");
    }

    /// A band that excludes everything makes every pick miss — you cannot select what the
    /// viewer mode is not drawing.
    #[test]
    fn a_pick_outside_the_band_misses() {
        let fixture = picking_fixture(ShapeKind::Sphere, [6, 6, 6]);
        let empty_band = LayerBand { band_min: 0, band_max: 0, onion_depth: 0 };
        // Layer 0 alone: the sphere's equator is nowhere near it, so a centre pick must miss
        // even though the same ray hits under a full band.
        let full =
            fixture.app_core.pick_voxel([640.0, 360.0], VIEWPORT, &fixture.frame(LayerBand::FULL));
        assert!(full.is_some(), "the control ray must hit under a full band");
        let clipped =
            fixture.app_core.pick_voxel([640.0, 360.0], VIEWPORT, &fixture.frame(empty_band));
        assert_ne!(
            clipped, full,
            "a one-layer band must not return the same hit as an unclipped march"
        );
    }

    /// **The LOWER half of a centred object is pickable under a FULL band (2026-07-21
    /// regression).** The band→march conversion once added a spurious `half_z` floor
    /// (`band_min − shading_to_absolute + half`), clipping every voxel below the region
    /// mid-plane — so a pick could never name a voxel in an object's lower half even under a
    /// FULL (mask-nothing) band. The owner hit this trying to place a tube on the bottom half of
    /// a tall cylinder. This sweeps a tall box and asserts a hit lands BELOW the region mid.
    #[test]
    fn the_lower_half_of_a_centred_object_is_pickable() {
        // A tall box viewed SIDE-ON (horizontal orthographic), so its whole near face — top to
        // bottom — is on screen and reachable; the default iso view looks down and cannot see a
        // tall object's lower half at all, which would mask the bug rather than expose it.
        let shape = SdfShape::from_blocks(ShapeKind::Box, [2, 2, 8], 1, DENSITY); // [16,16,64]
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
        let camera = OrbitCamera {
            target: glam::Vec3::new(8.0, 8.0, 32.0), // the box centre (it fills [0,16)²×[0,64))
            orbit_theta: 0.4,
            orbit_phi: std::f32::consts::FRAC_PI_2, // horizontal — the tall face side-on
            orbit_distance: 160.0,
            roll: 0.0,
            projection_mode: camera::ProjectionMode::Orthographic,
        };
        let mut app_core = AppCore::new(camera);
        let RebuildOutcome::Built(output) = app_core.rebuild(&scene, DENSITY) else {
            panic!("the fixture's density is in bounds");
        };
        let recentre_voxels = output.recentre_voxels.voxels();
        let chunks = output.two_layer_chunks.clone();
        let frame = PickFrame {
            region_dimensions: output.region_dimensions,
            recentre_voxels,
            density: DENSITY,
            chunks: &chunks,
            band: LayerBand::FULL,
        };

        let mid_layer = recentre_voxels[2]; // region mid-plane in absolute Z
        let mut lowest_hit_z = i64::MAX;
        for row in 0..40 {
            let cursor = [VIEWPORT[2] * 0.5, VIEWPORT[3] * (row as f32 + 0.5) / 40.0];
            if let Some(pick) = app_core.pick_voxel(cursor, VIEWPORT, &frame) {
                lowest_hit_z = lowest_hit_z.min(pick.absolute_voxel[2]);
            }
        }
        assert!(
            lowest_hit_z < mid_layer,
            "a FULL-band pick must reach BELOW the region mid-plane {mid_layer} (lowest hit was \
             {lowest_hit_z}) — the spurious half-Z band floor made the lower half unpickable"
        );
    }
}
