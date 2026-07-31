//! Analytic feature-edge catalog of a sketch solid (ADR 0032 selection feedback):
//! the authored profile's own creases, lifted by the operation. An extrude creases
//! along its two cap outlines and at every non-tangent profile vertex; a revolve
//! creases on a latitude circle per non-tangent off-axis vertex, plus the profile
//! outline at both sweep ends of a partial turn. A tangent vertex (collinear,
//! same-direction neighbors — e.g. a `split_segment` midpoint) creases nothing, and neither
//! does a vertex the author never placed: an arc reaches the boundary as a run of tessellation
//! samples, and those are steps around a smooth curve, not corners.

use super::solid::revolve_axes;
use super::*;

/// The catalog's fixed-point resolution: profile coords quantize to 1/256 voxel so the
/// tangency test stays EXACT integer arithmetic for sub-voxel vertices (#101). Display
/// only — the resolve never quantizes.
const EDGE_FIXED_SCALE: f64 = 256.0;

/// A profile coordinate on the 1/256-voxel lattice the tangency test works over.
fn to_fixed(coords: [f64; 2]) -> [i64; 2] {
    [
        (coords[0] * EDGE_FIXED_SCALE).round() as i64,
        (coords[1] * EDGE_FIXED_SCALE).round() as i64,
    ]
}

/// Exact tangency at a loop vertex: collinear AND same-direction (i128 over fixed-point
/// profile coords — no epsilon).
fn vertex_is_tangent(previous: [i64; 2], vertex: [i64; 2], next: [i64; 2]) -> bool {
    let edge_in = [vertex[0] - previous[0], vertex[1] - previous[1]];
    let edge_out = [next[0] - vertex[0], next[1] - vertex[1]];
    let cross = edge_in[0] as i128 * edge_out[1] as i128 - edge_in[1] as i128 * edge_out[0] as i128;
    let dot = edge_in[0] as i128 * edge_out[0] as i128 + edge_in[1] as i128 * edge_out[1] as i128;
    cross == 0 && dot > 0
}

impl SketchSolid {
    /// The catalog as polylines in the producer-local `[0, grid_dimensions()]` voxel
    /// frame — the SAME frame the resolve samples (ADR 0008: extrude fully
    /// corner-anchored on the profile bbox min; revolve corner-anchored axially,
    /// centered on the two radial axes). Empty for a degenerate producer.
    /// `circle_segments` tessellates one full latitude turn; a partial arc keeps the
    /// same angular density.
    pub(crate) fn profile_edge_polylines_local(&self, circle_segments: u32) -> Vec<Vec<[f32; 3]>> {
        let Some((profile_min, _)) = self.profile_bounds() else {
            return Vec::new();
        };
        // EVERY loop of the region creases, holes included (#100): the wall of a pocket is as
        // much a feature edge as the outside of the shape.
        let mut polylines = Vec::new();
        // A crease line IS a polyline, so this is a terminal adapter: it flattens at the default
        // tolerance rather than passing one back up to the region.
        for profile_loop in self.sketch.region() {
            self.ring_edge_polylines(
                &profile_loop.flatten(super::ARC_SAGITTA_TOLERANCE_VOXELS),
                profile_min,
                circle_segments,
                &mut polylines,
            );
        }
        polylines
    }

    /// The catalog contribution of ONE closed boundary, appended to `polylines`. Split out so
    /// a multi-loop region reuses one definition per loop rather than duplicating the crease
    /// rules (#100).
    fn ring_edge_polylines(
        &self,
        boundary: &[SketchPoint],
        profile_min: [i64; 2],
        circle_segments: u32,
        polylines: &mut Vec<Vec<[f32; 3]>>,
    ) {
        // The ring in 1/256-voxel FIXED POINT: sub-voxel coords (#101) quantize onto an
        // integer lattice so dedup/tangency stay exact; every emitted coordinate divides
        // back out through `fixed_to_voxels`.
        let mut ring: Vec<[i64; 2]> = boundary
            .iter()
            .map(|point| to_fixed(point.in_plane()))
            .collect();
        ring.dedup();
        while ring.len() > 1 && ring.first() == ring.last() {
            ring.pop();
        }
        if ring.len() < 3 {
            return;
        }
        // Where the author actually put a point. An arc's interior samples are DERIVED — a chord
        // joint is a step around a smooth curve, not a corner — so creasing at each one draws a
        // facet fan where the author drew one arc, and the fan gets denser as the tessellation
        // refines. A vertex creases only when it stands on an authored point: exact equality on
        // the same lattice, no tolerance.
        let authored: std::collections::BTreeSet<[i64; 2]> = self
            .sketch
            .points()
            .iter()
            .map(|point| to_fixed(point.at.in_plane()))
            .collect();
        let vertex_count = ring.len();
        let neighbors = |index: usize| {
            (
                ring[(index + vertex_count - 1) % vertex_count],
                ring[index],
                ring[(index + 1) % vertex_count],
            )
        };
        let [in_plane_0, in_plane_1] = self.sketch.plane.in_plane_axes();
        let normal = self.sketch.plane.normal_axis();
        match self.operation {
            Operation::Extrude { height_voxels } => {
                let height = height_voxels as f32;
                let local_point = |vertex: [i64; 2], along_normal: f32| -> [f32; 3] {
                    let mut point = [0.0f32; 3];
                    point[in_plane_0] =
                        (vertex[0] as f64 / EDGE_FIXED_SCALE - profile_min[0] as f64) as f32;
                    point[in_plane_1] =
                        (vertex[1] as f64 / EDGE_FIXED_SCALE - profile_min[1] as f64) as f32;
                    point[normal] = along_normal;
                    point
                };
                for cap in [0.0, height] {
                    let mut outline: Vec<[f32; 3]> = ring
                        .iter()
                        .map(|&vertex| local_point(vertex, cap))
                        .collect();
                    outline.push(outline[0]);
                    polylines.push(outline);
                }
                for index in 0..vertex_count {
                    let (previous, vertex, next) = neighbors(index);
                    if authored.contains(&vertex) && !vertex_is_tangent(previous, vertex, next) {
                        polylines.push(vec![local_point(vertex, 0.0), local_point(vertex, height)]);
                    }
                }
            }
            Operation::Revolve { axis, sweep } => {
                let dimensions = self.grid_dimensions();
                let (axial_world_axis, axial_min, radial_a, radial_b) =
                    revolve_axes(axis, in_plane_0, in_plane_1, normal, profile_min);
                let half_a = dimensions[radial_a] as f32 / 2.0;
                let half_b = dimensions[radial_b] as f32 / 2.0;
                let (axial_coord, radial_coord) = match axis {
                    RevolveAxis::InPlane0 => (0usize, 1usize),
                    RevolveAxis::InPlane1 => (1, 0),
                };
                let turn_degrees = sweep.turn_degrees.min(360);
                let turn_radians = (turn_degrees as f32).to_radians();
                let fixed_to_voxels = |fixed: i64| (fixed as f64 / EDGE_FIXED_SCALE) as f32;
                let place = |axial: f32, radius: f32, angle: f32| -> [f32; 3] {
                    let mut point = [0.0f32; 3];
                    point[axial_world_axis] = axial - axial_min as f32;
                    point[radial_a] = half_a + radius * angle.cos();
                    point[radial_b] = half_b + radius * angle.sin();
                    point
                };
                // A latitude circle (arc, for a partial turn) per non-tangent off-axis
                // vertex; a vertex ON the axis is a pole and creases nothing.
                let steps = (circle_segments * turn_degrees).div_ceil(360).max(1);
                for index in 0..vertex_count {
                    let (previous, vertex, next) = neighbors(index);
                    if !authored.contains(&vertex)
                        || vertex_is_tangent(previous, vertex, next)
                        || vertex[radial_coord] == 0
                    {
                        continue;
                    }
                    let radius = fixed_to_voxels(vertex[radial_coord].abs());
                    let axial = fixed_to_voxels(vertex[axial_coord]);
                    let mut arc: Vec<[f32; 3]> = (0..=steps)
                        .map(|step| place(axial, radius, turn_radians * step as f32 / steps as f32))
                        .collect();
                    if turn_degrees == 360 {
                        // Close the loop bit-exactly (sin(TAU) is not exactly 0.0).
                        *arc.last_mut().expect("steps >= 1") = arc[0];
                    }
                    polylines.push(arc);
                }
                // A partial sweep exposes the profile at both ends. The revolve folds
                // radius by |·|, so a straddling edge bends AT the axis: insert the
                // crossing point so the outline follows the folded silhouette.
                if turn_degrees < 360 {
                    for angle in [0.0, turn_radians] {
                        let mut outline = Vec::with_capacity(vertex_count + 1);
                        for index in 0..vertex_count {
                            let vertex = ring[index];
                            let next = ring[(index + 1) % vertex_count];
                            outline.push(place(
                                fixed_to_voxels(vertex[axial_coord]),
                                fixed_to_voxels(vertex[radial_coord].abs()),
                                angle,
                            ));
                            let radial_here = vertex[radial_coord];
                            let radial_next = next[radial_coord];
                            if radial_here != 0
                                && radial_next != 0
                                && (radial_here < 0) != (radial_next < 0)
                            {
                                let toward_crossing = radial_here.unsigned_abs() as f32
                                    / (radial_here.unsigned_abs() + radial_next.unsigned_abs())
                                        as f32;
                                let axial_at_crossing = fixed_to_voxels(vertex[axial_coord])
                                    + toward_crossing
                                        * fixed_to_voxels(next[axial_coord] - vertex[axial_coord]);
                                outline.push(place(axial_at_crossing, 0.0, angle));
                            }
                        }
                        outline.push(outline[0]);
                        polylines.push(outline);
                    }
                }
            }
        }
    }
}
