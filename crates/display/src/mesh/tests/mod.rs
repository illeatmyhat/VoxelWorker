use super::*;
use voxel_core::voxel::Voxel;

mod cuboid;
mod apron;
mod two_layer;

/// Build a tiny grid from a set of occupied voxel indices, all one material, with
/// the given dimensions, in the RECENTRED render frame the live cuboid path sees.
///
/// The stored `local_index` reproduces the retired f32 fixture's
/// `world_position = index + 0.5 − dim/2` EXACTLY for an EVEN dim (where the centre
/// is a half-integer): `local_index = floor(index + 0.5 − dim/2)`, so
/// `world_position()` (= `local_index + 0.5`) equals the old value bit-for-bit and the
/// band-clip's `half = floor(dim/2)` frame assumption still holds. (An ODD dim's old
/// centre fell on an INTEGER, which the integer payload — whose centres are always
/// half-integers — cannot represent; the one odd-dim test below corner-anchors and
/// reads the world planes directly, since the mesher is anchor-shift-invariant.)
pub(super) fn grid_from_indices(dimensions: [u32; 3], cells: &[[u32; 3]], material: u16) -> VoxelGrid {
    let half = [
        dimensions[0] as f32 / 2.0,
        dimensions[1] as f32 / 2.0,
        dimensions[2] as f32 / 2.0,
    ];
    let mut grid = VoxelGrid::new(dimensions);
    for &[i, j, k] in cells {
        grid.occupied.push(Voxel {
            local_index: [
                (i as f32 + 0.5 - half[0]).floor() as i32,
                (j as f32 + 0.5 - half[1]).floor() as i32,
                (k as f32 + 0.5 - half[2]).floor() as i32,
            ],
            block_local_coord: [0, 0, 0],
            block_id: voxel_core::core_geom::BlockId(material),
            attrs: voxel_core::core_geom::BlockAttrs::DEFAULT,
            grid_overlay: false,
        });
    }
    grid
}

/// A single UNIT exposed face: the absolute integer plane coordinate on the
/// face's axis, the two in-plane unit-cell lower coords, and the face axis +
/// sign. Canonical regardless of how a co-planar face is split into abutting
/// quads, so it is the granularity at which whole-region meshing and per-chunk
/// apron meshing must produce the IDENTICAL set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) struct UnitFace {
    /// 0 = X face, 1 = Y face, 2 = Z face.
    axis: u8,
    /// `+1` / `-1` outward direction along `axis`.
    sign: i8,
    /// Integer world plane on `axis` (the quad's constant coordinate).
    plane: i64,
    /// The two in-plane unit-cell lower coords (the axes other than `axis`).
    cell: [i64; 2],
}

/// Round a world coordinate that must land on an integer plane (box corners are
/// integer planes in world space once the shift-invariant offset is folded in).
pub(super) fn round_plane(value: f32) -> i64 {
    let rounded = value.round();
    assert!(
        (value - rounded).abs() < 1e-3,
        "vertex coord {value} is not on an integer world plane"
    );
    rounded as i64
}

/// Explode a vertex/index mesh into its SET of unit exposed faces (the canonical
/// granularity), in the GLOBAL INDEX frame. Mesh vertices live in world space at
/// `global_index + world_offset`, so subtracting `world_offset` recovers the
/// integer global-index planes the ground-truth `genuine_exposed_faces` keys off.
/// Each quad (6 indices) lies on a plane perpendicular to its normal; it is split
/// into the unit cells it covers in the two in-plane axes.
pub(super) fn unit_faces_in_index_frame(
    vertices: &[CuboidVertex],
    indices: &[u32],
    world_offset: [f32; 3],
) -> std::collections::HashSet<UnitFace> {
    let mut faces = std::collections::HashSet::new();
    let to_index = |pos: [f32; 3]| -> [f32; 3] {
        [
            pos[0] - world_offset[0],
            pos[1] - world_offset[1],
            pos[2] - world_offset[2],
        ]
    };
    // Each quad is two triangles emitted as [b, b+1, b+2, b, b+2, b+3], so the
    // four distinct corner vertices are indices[i], [i+1], [i+2], [i+5].
    let mut i = 0;
    while i < indices.len() {
        let corners = [
            vertices[indices[i] as usize],
            vertices[indices[i + 1] as usize],
            vertices[indices[i + 2] as usize],
            vertices[indices[i + 5] as usize],
        ];
        let normal = corners[0].normal;
        let axis = if normal[0].abs() > 0.5 {
            0usize
        } else if normal[1].abs() > 0.5 {
            1
        } else {
            2
        };
        let sign: i8 = if normal[axis] > 0.0 { 1 } else { -1 };
        let (a, b) = match axis {
            0 => (1usize, 2usize),
            1 => (0usize, 2usize),
            _ => (0usize, 1usize),
        };
        let plane = round_plane(to_index(corners[0].position)[axis]);
        // The quad's span in the two in-plane axes (integer index planes).
        let mut a_lo = i64::MAX;
        let mut a_hi = i64::MIN;
        let mut b_lo = i64::MAX;
        let mut b_hi = i64::MIN;
        for corner in &corners {
            let idx = to_index(corner.position);
            let av = round_plane(idx[a]);
            let bv = round_plane(idx[b]);
            a_lo = a_lo.min(av);
            a_hi = a_hi.max(av);
            b_lo = b_lo.min(bv);
            b_hi = b_hi.max(bv);
        }
        for ca in a_lo..a_hi {
            for cb in b_lo..b_hi {
                faces.insert(UnitFace {
                    axis: axis as u8,
                    sign,
                    plane,
                    cell: [ca, cb],
                });
            }
        }
        i += 6;
    }
    faces
}

/// The world offset (`min_world - 0.5` per axis) the mesher anchors a grid's
/// vertices on — subtract it from a mesh vertex to get the integer global-index
/// frame `genuine_exposed_faces` uses.
pub(super) fn grid_world_offset(grid: &VoxelGrid) -> [f32; 3] {
    let mut min_world = [f32::INFINITY; 3];
    for v in &grid.occupied {
        let position = v.world_position();
        for (axis, m) in min_world.iter_mut().enumerate() {
            *m = m.min(position[axis]);
        }
    }
    [min_world[0] - 0.5, min_world[1] - 0.5, min_world[2] - 0.5]
}
/// The set of GENUINELY-exposed unit faces of an occupancy set: a `(voxel,
/// direction)` whose neighbour cell is air. This is the VISIBLE silhouette — the
/// surface that survives back-face culling + depth testing. The cuboid mesher's
/// `face_is_exposed` emits a whole MERGED box face when ANY cell behind it is
/// air, so it OVER-DRAWS the sub-faces backed by solid; those over-draw quads are
/// always either back-face-culled or depth-occluded by the solid they are buried
/// in, so they never reach a pixel. The genuinely-exposed set is therefore the
/// invariant that determines the rendered image — and the structural parity claim:
/// it must be IDENTICAL for whole-region and per-chunk meshing. We derive it
/// straight from the occupancy (the ground truth) and also use it to filter an
/// emitted mesh's unit faces down to its visible subset.
pub(super) fn genuine_exposed_faces(
    occupied: &std::collections::HashSet<[i64; 3]>,
) -> std::collections::HashSet<UnitFace> {
    let dirs: [(usize, i8, [i64; 3]); 6] = [
        (0, 1, [1, 0, 0]),
        (0, -1, [-1, 0, 0]),
        (1, 1, [0, 1, 0]),
        (1, -1, [0, -1, 0]),
        (2, 1, [0, 0, 1]),
        (2, -1, [0, 0, -1]),
    ];
    let mut faces = std::collections::HashSet::new();
    for &v in occupied {
        for (axis, sign, delta) in dirs {
            let neighbor = [v[0] + delta[0], v[1] + delta[1], v[2] + delta[2]];
            if occupied.contains(&neighbor) {
                continue; // backed by solid → interior, not visible
            }
            // The face plane on `axis`: for +sign it's the voxel's far plane
            // (v[axis] + 1), for -sign the near plane (v[axis]).
            let plane = if sign > 0 { v[axis] + 1 } else { v[axis] };
            let (a, b) = match axis {
                0 => (1usize, 2usize),
                1 => (0usize, 2usize),
                _ => (0usize, 1usize),
            };
            faces.insert(UnitFace {
                axis: axis as u8,
                sign,
                plane,
                cell: [v[a], v[b]],
            });
        }
    }
    faces
}

/// Filter an emitted mesh's unit faces down to the VISIBLE subset (those whose
/// `(plane, cell, axis, sign)` is a genuinely-exposed face), discarding the
/// over-draw quads `face_is_exposed` emits for partially-exposed merged boxes.
pub(super) fn visible_unit_faces(
    vertices: &[CuboidVertex],
    indices: &[u32],
    world_offset: [f32; 3],
    genuine: &std::collections::HashSet<UnitFace>,
) -> std::collections::HashSet<UnitFace> {
    unit_faces_in_index_frame(vertices, indices, world_offset)
        .into_iter()
        .filter(|f| genuine.contains(f))
        .collect()
}

/// Absolute integer occupancy (global indices `round(world - min_world)`) of a
/// grid — the same frame the cuboid mesher's vertices live in, so a `UnitFace`
/// derived from occupancy and one derived from a mesh vertex compare directly.
pub(super) fn occupancy_indices(grid: &VoxelGrid) -> std::collections::HashSet<[i64; 3]> {
    let mut min_world = [f32::INFINITY; 3];
    for v in &grid.occupied {
        let position = v.world_position();
        for (axis, m) in min_world.iter_mut().enumerate() {
            *m = m.min(position[axis]);
        }
    }
    let mut set = std::collections::HashSet::new();
    for v in &grid.occupied {
        let position = v.world_position();
        set.insert([
            (position[0] - min_world[0]).round() as i64,
            (position[1] - min_world[1]).round() as i64,
            (position[2] - min_world[2]).round() as i64,
        ]);
    }
    set
}

/// Map a wholesale/filtered mesh build to `coord -> (vertex bytes, indices)` — the
/// per-chunk GPU buffer set proxy (the renderer uploads exactly these bytes), so a
/// byte-equal map == a byte-equal buffer set.
pub(super) fn mesh_map(
    meshes: &[CuboidChunkMesh],
) -> std::collections::HashMap<[i32; 3], (Vec<u8>, Vec<u32>)> {
    meshes
        .iter()
        .map(|m| {
            (
                m.coord,
                (bytemuck::cast_slice::<_, u8>(&m.vertices).to_vec(), m.indices.clone()),
            )
        })
        .collect()
}
