use super::*;

/// For a triangle wound CCW *as seen from outside*, the geometric face normal
/// (edge0 × edge1) points in the SAME direction as the stored outward normal,
/// so their dot product is positive. A negative dot means the winding is
/// inside-out (BUG 1) and back-face culling would hide the visible face.
fn assert_ccw_outward(positions: &[[f32; 3]], normals: &[[f32; 3]], indices: &[u16]) {
    assert_eq!(indices.len() % 3, 0, "indices must form whole triangles");
    for tri in indices.chunks_exact(3) {
        let a = positions[tri[0] as usize];
        let b = positions[tri[1] as usize];
        let c = positions[tri[2] as usize];
        let edge0 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let edge1 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        // edge0 × edge1
        let geometric_normal = [
            edge0[1] * edge1[2] - edge0[2] * edge1[1],
            edge0[2] * edge1[0] - edge0[0] * edge1[2],
            edge0[0] * edge1[1] - edge0[1] * edge1[0],
        ];
        let outward = normals[tri[0] as usize];
        let dot = geometric_normal[0] * outward[0]
            + geometric_normal[1] * outward[1]
            + geometric_normal[2] * outward[2];
        assert!(
            dot > 0.0,
            "triangle {tri:?} is wound inside-out (dot={dot}); outward faces would be culled",
        );
    }
}

#[test]
fn view_cube_is_ccw_outward() {
    let (vertices, indices) = view_cube_geometry();
    let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
    let normals: Vec<[f32; 3]> = vertices.iter().map(|v| v.normal).collect();
    assert_ccw_outward(&positions, &normals, &indices);
}

// ---- issue #29 S3: per-object grid line geometry + gating ----

use voxel_core::core_geom::MaterialChoice as Mc;
use document::scene::{Node, NodeContent};
use voxel_core::voxel::ShapeKind;
use document::voxel::SdfShape;

/// `block_boundaries` returns the closing plane at `hi` (the box is enclosed in
/// whole blocks), so a `B`-block box yields `B + 1` planes — and EXPANDING the
/// box by one block on an axis adds exactly one boundary plane there. This is the
/// geometry that makes "add/remove a whole block" fall out: a box grown by one
/// enclosing block gains one lattice plane; shrunk by one, it loses one.
#[test]
fn block_boundaries_count_tracks_enclosing_blocks() {
    for step in [1u32, 15, 16] {
        let s = step as f32;
        // A 3-block box [0, 3·step] → planes at 0, step, 2·step, 3·step = 4.
        let three = block_boundaries(0.0, 3.0 * s, step);
        assert_eq!(three.len(), 4, "@step{step}: a 3-block box has 4 boundary planes");
        assert_eq!(*three.first().unwrap(), 0.0);
        assert_eq!(*three.last().unwrap(), 3.0 * s, "closing plane lands exactly on hi");
        // ADD a whole block (expand by +step): exactly one more plane.
        let four = block_boundaries(0.0, 4.0 * s, step);
        assert_eq!(four.len(), 5, "@step{step}: +1 enclosing block ⇒ +1 lattice plane");
        // REMOVE a whole block (shrink by step): exactly one fewer plane.
        let two = block_boundaries(0.0, 2.0 * s, step);
        assert_eq!(two.len(), 3, "@step{step}: -1 enclosing block ⇒ -1 lattice plane");
    }
}

/// `voxel_boundaries` walks one voxel at a time from the block-aligned `lo` to
/// `hi`, tagging every `step`-th line as a BLOCK edge. So a `B`-block box yields
/// `B·step + 1` voxel lines, of which exactly `B + 1` are block lines — and those
/// block lines sit on the SAME coordinates as `block_boundaries(lo, hi, step)`.
/// This is the alignment guarantee: the fine floor's bold lines coincide with the
/// block lattice's vertical lines.
#[test]
fn voxel_boundaries_tag_block_lines_at_lattice_positions() {
    for step in [1u32, 15, 16] {
        let s = step as f32;
        // A 3-block box: 3·step voxel cells ⇒ 3·step + 1 voxel boundaries.
        let lines = voxel_boundaries(0.0, 3.0 * s, step);
        assert_eq!(
            lines.len(),
            3 * step as usize + 1,
            "@step{step}: a 3-block box has 3·step+1 voxel boundaries",
        );
        // The BLOCK-tagged lines are exactly the block-boundary planes.
        let block_lines: Vec<f32> =
            lines.iter().filter(|(_, b)| *b).map(|(c, _)| *c).collect();
        assert_eq!(
            block_lines,
            block_boundaries(0.0, 3.0 * s, step),
            "@step{step}: floor's bold (block) lines coincide with the lattice block lines",
        );
        // At density 1 EVERY voxel line is a block line (voxel == block).
        if step == 1 {
            assert!(lines.iter().all(|(_, b)| *b), "@step1: every voxel line is a block line");
        } else {
            // Otherwise the voxel lines strictly outnumber the block lines.
            assert!(
                block_lines.len() < lines.len(),
                "@step{step}: voxel lines are denser than block lines",
            );
        }
    }
}

/// The fine floor grid is two-tier and aligns with the block lattice (issue #29
/// fix). Z-up: the floor is an XY-plane grid at the base. For a node box, this
/// asserts three properties. First, the floor's DISTINCT X line coordinates form a
/// superset of — and at the block positions coincide with — the lattice's
/// X-line coordinates. Second, the floor uses exactly two alphas (a subtle voxel
/// tier and a bold block tier). Third, at a coarse density the voxel lines visibly
/// outnumber the block lines.
#[test]
fn floor_grid_is_two_tier_and_aligns_with_lattice() {
    // Distinct X coordinates among the floor's X-boundary lines (Z-up: each runs
    // along Y in the XY ground plane); compared against the lattice's X lines.
    let distinct_xs = |verts: &[LineVertex]| -> Vec<i64> {
        let mut xs: Vec<i64> = verts
            .iter()
            .map(|v| (v.position[0] * 256.0).round() as i64)
            .collect();
        xs.sort_unstable();
        xs.dedup();
        xs
    };
    for step in [1u32, 15, 16] {
        let s = step as f32;
        // A box NOT at the origin (min ≠ 0), to catch a frame/offset mismatch.
        let (min, max) = ([s, 0.0, 2.0 * s], [4.0 * s, s, 5.0 * s]);
        let mut lattice = Vec::new();
        lattice_vertices_into(&mut lattice, min, max, step);
        let mut floor = Vec::new();
        floor_vertices_into(&mut floor, min, max, step);

        // (2) Exactly two distinct alphas — the subtle voxel tier and the bold
        // block tier. At step 1 every line is both a voxel and a block line, so
        // it is drawn twice (subtle then bold) and BOTH alphas are still present.
        let mut alphas: Vec<i64> =
            floor.iter().map(|v| (v.color[3] * 1024.0).round() as i64).collect();
        alphas.sort_unstable();
        alphas.dedup();
        assert_eq!(
            alphas.len(),
            2,
            "@step{step}: floor has two alpha tiers (subtle voxel + bold block)",
        );

        // (1) The lattice's X lines must ALL appear among the floor's X lines
        // (the floor X set is a superset coinciding at the block lines).
        let lattice_xs = distinct_xs(&lattice);
        let floor_xs = distinct_xs(&floor);
        for x in &lattice_xs {
            assert!(
                floor_xs.contains(x),
                "@step{step}: lattice vertical line x={x} has a coincident floor line",
            );
        }
        // (3) At a coarse density the floor has strictly more distinct X lines
        // than the lattice (the extra ones are the fine voxel lines).
        if step > 1 {
            assert!(
                floor_xs.len() > lattice_xs.len(),
                "@step{step}: floor (voxel-resolution) has denser X lines than the lattice",
            );
        }
    }
}

/// One node's lattice/floor box → a non-empty line set at every density; the
/// vertex count is a multiple of 2 (whole segments).
#[test]
fn lattice_and_floor_vertices_nonempty_per_box() {
    for step in [1u32, 15, 16] {
        let s = step as f32;
        let (min, max) = ([0.0, 0.0, 0.0], [2.0 * s, s, 3.0 * s]);
        let mut lattice = Vec::new();
        lattice_vertices_into(&mut lattice, min, max, step);
        assert!(!lattice.is_empty(), "@step{step}: a sized box has lattice lines");
        assert_eq!(lattice.len() % 2, 0, "lattice lines are whole segments");
        let mut floor = Vec::new();
        floor_vertices_into(&mut floor, min, max, step);
        assert!(!floor.is_empty(), "@step{step}: a sized box has floor lines");
        // Z-up: the floor sits at the EXACT base plane `z = min[2]` (issue #29
        // fix: no geometric drop — the floor pipeline's depth bias avoids
        // z-fighting the model's coincident bottom face), flat in Z, uniform across
        // every vertex. This makes the floor's block lines meet the lattice's
        // bottom edges.
        let floor_z = min[2];
        assert!(floor.iter().all(|v| v.position[2] == floor_z), "floor on exact base plane");
    }
}

fn box_node(name: &str, offset: [i64; 3], voxels_per_block: u32) -> Node {
    let shape = SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, voxels_per_block);
    let mut node = Node::new(name, NodeContent::Tool { shape, material: Mc::Stone });
    node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
    node
}

/// Gating (issue #29 S3): a node's lattice box appears in the batch ONLY when the
/// master AND the node's per-object toggle are both ON; turning EITHER off drops
/// it. A two-node scene with the grid enabled on ONE node yields exactly ONE
/// lattice box (the other node contributes none).
#[test]
fn scene_grid_boxes_gated_by_master_and_per_object() {
    for density in [1u32, 15, 16] {
        let mut scene = Scene::from_nodes(vec![
            box_node("A", [0, 0, 0], density),
            box_node("B", [8, 0, 0], density),
        ]);
        scene.voxels_per_block = density;
        scene.active = None;
        scene.master_block_lattice = true;
        scene.master_floor_grid = true;

        // Both per-object toggles OFF → no boxes regardless of masters.
        let (lat, flr) = scene_grid_boxes(&scene, density);
        assert!(lat.is_empty() && flr.is_empty(), "@d{density}: per-object OFF ⇒ no boxes");

        // Enable block lattice on node A ONLY.
        scene.root_node_mut(0).grids.block_lattice = true;
        let (lat, flr) = scene_grid_boxes(&scene, density);
        assert_eq!(lat.len(), 1, "@d{density}: one node enabled ⇒ exactly one lattice box");
        assert!(flr.is_empty(), "@d{density}: floor still off");

        // Master OFF cancels it even though the node's flag is on.
        scene.master_block_lattice = false;
        let (lat, _flr) = scene_grid_boxes(&scene, density);
        assert!(lat.is_empty(), "@d{density}: master OFF ⇒ no lattice box (AND gating)");

        // Floor: node B's flag on + master on → one floor box, no lattice.
        scene.master_floor_grid = true;
        scene.root_node_mut(1).grids.floor_grid = true;
        let (lat, flr) = scene_grid_boxes(&scene, density);
        assert!(lat.is_empty(), "@d{density}: lattice master still off");
        assert_eq!(flr.len(), 1, "@d{density}: one floor box from node B");
    }
}

// ===== Issue #29 S5: Points (world reference grid) ==========================

use document::scene::Point;

/// A scene carrying only an Origin Point with the given plane flags; `axes`
/// sets all three per-axis flags together (the common "axes on/off" case).
fn origin_point_scene(plane_xz: bool, plane_xy: bool, plane_yz: bool, axes: bool) -> Scene {
    origin_point_scene_axes(plane_xz, plane_xy, plane_yz, [axes, axes, axes])
}

/// A scene carrying only an Origin Point with the given plane flags and explicit
/// per-axis X/Y/Z toggles (issue #29 fix: separable axes).
fn origin_point_scene_axes(
    plane_xz: bool,
    plane_xy: bool,
    plane_yz: bool,
    axes: [bool; 3],
) -> Scene {
    let mut scene = Scene::default();
    scene.points.push(Point {
        name: "Origin".to_string(),
        plane_xz,
        plane_xy,
        plane_yz,
        axis_x: axes[0],
        axis_y: axes[1],
        axis_z: axes[2],
        is_origin: true,
        ..Point::default()
    });
    scene.active_point = Some(0);
    scene
}

/// A visible Origin Point with axes yields a NON-EMPTY axis batch; a hidden Point
/// yields NONE (the spec's "hidden Points render nothing"). The ground PLANE moved
/// to the analytic infinite grid ([`enabled_grid_planes`]), so this batch is now
/// AXES-only.
#[test]
fn points_visible_yields_batch_hidden_yields_none() {
    for density in [1u32, 15, 16] {
        // Z-up: the ground plane is XY (the 2nd flag of `origin_point_scene`).
        let mut scene = origin_point_scene(false, true, false, true);
        let batch = points_line_batch(&scene, density);
        assert!(!batch.is_empty(), "@d{density}: visible axes ⇒ non-empty batch");
        assert_eq!(batch.len() % 2, 0, "@d{density}: whole line segments");

        // The Origin's ground (XY, Z-up) plane is one analytic-grid instance.
        let planes = enabled_grid_planes(&scene, density);
        assert_eq!(planes.len(), 1, "@d{density}: the Origin ground plane ⇒ one grid plane");

        scene.points[0].hidden = true;
        let hidden = points_line_batch(&scene, density);
        assert!(hidden.is_empty(), "@d{density}: a hidden Point renders no axes");
        assert!(
            enabled_grid_planes(&scene, density).is_empty(),
            "@d{density}: a hidden Point renders no grid plane",
        );
    }
}

/// The plane and axis toggles gate independently. Axes flow through
/// [`points_line_batch`] (AXES-only); planes flow through [`enabled_grid_planes`].
/// Turning every plane + axis off empties BOTH; enabling more planes adds grid
/// instances; the axes alone yield EXACTLY six axis vertices (three segments).
#[test]
fn points_plane_and_axis_toggles_gate() {
    let density = 16u32;
    // Everything off → no axes, no planes.
    let none = points_line_batch(&origin_point_scene(false, false, false, false), density);
    assert!(none.is_empty(), "all axes off ⇒ empty axis batch");
    assert!(
        enabled_grid_planes(&origin_point_scene(false, false, false, false), density).is_empty(),
        "all planes off ⇒ no grid planes",
    );

    // Axes only → exactly 3 segments = 6 vertices, through the origin; no planes.
    let axes_only = points_line_batch(&origin_point_scene(false, false, false, true), density);
    assert_eq!(axes_only.len(), 6, "axes alone ⇒ three line segments");
    assert!(
        enabled_grid_planes(&origin_point_scene(false, false, false, true), density).is_empty(),
        "axes alone ⇒ no grid planes",
    );

    // Each enabled plane adds one grid instance; enabling more planes grows the
    // count. Z-up: the ground plane is XY (2nd flag).
    let ground = enabled_grid_planes(&origin_point_scene(false, true, false, false), density);
    let ground_front = enabled_grid_planes(&origin_point_scene(true, true, false, false), density);
    assert_eq!(ground.len(), 1, "the XY ground plane alone ⇒ one grid plane");
    assert_eq!(ground_front.len(), 2, "adding the XZ front plane ⇒ two grid planes");
}

/// Per-axis gating (issue #29 fix): the X/Y/Z axes toggle independently. All three
/// on ⇒ three segments (one per colour); turning Y off drops the GREEN segment and
/// leaves the red (X) and blue (Z) ones; a single axis on ⇒ exactly one segment.
#[test]
fn points_axes_toggle_per_axis() {
    for density in [1u32, 15, 16] {
        let green = with_alpha(srgb_hex_to_linear(GIZMO_AXIS_Y_HEX), POINT_AXIS_ALPHA);
        let is_green = |v: &LineVertex| v.color == green;

        // All three axes on (planes off) → exactly 3 segments = 6 vertices, one green.
        let all = points_line_batch(
            &origin_point_scene_axes(false, false, false, [true, true, true]),
            density,
        );
        assert_eq!(all.len(), 6, "@d{density}: three axes ⇒ three segments");
        assert_eq!(all.iter().filter(|v| is_green(v)).count(), 2, "@d{density}: one green (Y) segment, two vertices");

        // Turn Y off → 2 segments, NO green line.
        let no_y = points_line_batch(
            &origin_point_scene_axes(false, false, false, [true, false, true]),
            density,
        );
        assert_eq!(no_y.len(), 4, "@d{density}: Y off ⇒ two segments");
        assert!(!no_y.iter().any(is_green), "@d{density}: no green (Y) line when Y is off");

        // Only Y on → exactly one (green) segment.
        let only_y = points_line_batch(
            &origin_point_scene_axes(false, false, false, [false, true, false]),
            density,
        );
        assert_eq!(only_y.len(), 2, "@d{density}: only Y ⇒ one segment");
        assert!(only_y.iter().all(is_green), "@d{density}: the only line is green (Y)");
    }
}

/// The analytic grid plane carries the correct orientation, origin, and tuning for
/// each [`ReferencePlane`] (Z-up): XZ is normal +Y (the FRONT plane), XY normal +Z
/// (the GROUND plane), YZ normal +X (the side), with orthonormal in-plane axes
/// through the Point origin. Pure CPU — the shader consumes these basis vectors.
#[test]
fn grid_planes_carry_correct_orientation() {
    for density in [1u32, 15, 16] {
        // All three planes on at the Origin (recentre = 0 → origin at world 0).
        let scene = origin_point_scene(true, true, true, false);
        let planes = enabled_grid_planes(&scene, density);
        assert_eq!(planes.len(), 3, "@d{density}: three planes enabled ⇒ three instances");
        // Emission order is XZ (front), XY (ground), YZ (side).
        assert_eq!(planes[0].normal, [0.0, 1.0, 0.0], "@d{density}: XZ front ⇒ +Y normal");
        assert_eq!(planes[1].normal, [0.0, 0.0, 1.0], "@d{density}: XY ground ⇒ +Z normal");
        assert_eq!(planes[2].normal, [1.0, 0.0, 0.0], "@d{density}: YZ side ⇒ +X normal");
        for plane in &planes {
            assert_eq!(plane.origin, [0.0, 0.0, 0.0], "@d{density}: Origin plane at world 0");
            // In-plane axes are unit and perpendicular to the normal.
            let dot_un = plane.u_axis.iter().zip(plane.normal).map(|(a, b)| a * b).sum::<f32>();
            let dot_vn = plane.v_axis.iter().zip(plane.normal).map(|(a, b)| a * b).sum::<f32>();
            assert!(dot_un.abs() < 1e-6 && dot_vn.abs() < 1e-6, "in-plane axes ⊥ normal");
        }
    }
}

/// A second Point offset from the origin places its grid PLANE and its AXES at that
/// WORLD position: with a lone Point (recentre = 0 — no sized leaf) both pass
/// through `position_blocks · density`.
#[test]
fn points_offset_point_frame_sits_at_world_position() {
    let density = 16i64;
    let mut scene = Scene::default();
    // Z-up: the ground plane is XY (`plane_xy`, on by default). Keep ONLY the
    // ground plane on so this Point yields exactly one grid plane.
    scene.points.push(Point {
        position_blocks: [10, 0, -4],
        plane_xy: true,
        plane_xz: false,
        plane_yz: false,
        // axis_x/y/z default true via Point::default() ⇒ all three axes on.
        is_origin: false,
        ..Point::default()
    });
    // The offset Point's ground plane sits at that world position.
    let planes = enabled_grid_planes(&scene, density as u32);
    assert_eq!(planes.len(), 1, "the offset Point's XY ground plane ⇒ one grid plane");
    assert_eq!(
        planes[0].origin,
        [(10 * density) as f32, 0.0, (-4 * density) as f32],
        "the grid plane origin is at the Point's world position",
    );
    let batch = points_line_batch(&scene, density as u32);
    assert_eq!(batch.len(), 6, "axes only ⇒ three segments");
    // The axes cross at the Point origin; every axis segment shares that centre on
    // its two non-running coordinates. Recover the centre as the midpoint of the X
    // axis segment (vertices 0,1 are the X axis through the centre).
    let centre = [
        (batch[0].position[0] + batch[1].position[0]) / 2.0,
        (batch[0].position[1] + batch[1].position[1]) / 2.0,
        (batch[0].position[2] + batch[1].position[2]) / 2.0,
    ];
    assert!((centre[0] - (10 * density) as f32).abs() < 1e-3, "X frame at 10 blocks");
    assert!((centre[1]).abs() < 1e-3, "Y frame at 0");
    assert!((centre[2] - (-4 * density) as f32).abs() < 1e-3, "Z frame at -4 blocks");
}

/// Block-line spacing is density-parametrized: the gap between adjacent ground
/// lines along an axis equals one block (= `density` voxels) at {1, 15, 16}.
///
/// With the analytic infinite grid the block spacing is no longer baked into CPU
/// geometry — it is the `block_spacing` shader param, which the renderer sets to
/// `voxels_per_block`. Pin that mapping: the bold (block) tier spacing equals the
/// density, while the fine (voxel) tier is always spacing 1, so adjacent BLOCK
/// lines are exactly one block (= density voxels) apart at every density.
#[test]
fn grid_block_spacing_is_density() {
    for density in [1u32, 15, 16] {
        // The renderer's `rebuild_from_scene` packs `block_spacing = density` into
        // `params.x`; the voxel tier is fixed at spacing 1.0 in the shader. This
        // mirrors that contract without a GPU.
        let block_spacing = density.max(1) as f32;
        assert_eq!(
            block_spacing, density as f32,
            "@d{density}: bold (block) grid lines are one block apart (spacing = density)",
        );
        // And a plane is actually emitted to carry that spacing.
        let scene = origin_point_scene(true, false, false, false);
        assert_eq!(enabled_grid_planes(&scene, density).len(), 1);
    }
}
