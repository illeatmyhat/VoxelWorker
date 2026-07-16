use super::*;

/// Emit a COARSE-SOLID block as ONE box (ADR 0010 Decision 4): the whole `density³` block
/// at `block_id`, culling each of its 6 block faces when the neighbour block's matching
/// face is fully solid (seam-flag culling — no densified apron, no per-voxel decompose of
/// the solid interior). `block_low_recentred` is the block's low voxel corner in the
/// recentred frame; `abs_block` its absolute block coord (to look up neighbours).
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_coarse_block_box(
    block_id: voxel_core::core_geom::BlockId,
    overlay: bool,
    density: u32,
    block_low_recentred: [i64; 3],
    abs_block: [i64; 3],
    face_solidity_at: &dyn Fn([i64; 3]) -> BlockFaceSolidity,
    vertices: &mut Vec<CuboidVertex>,
    indices: &mut Vec<u32>,
    indices_overlay: &mut Vec<u32>,
    aabb: &mut Aabb,
) {
    let material_id = CellKey::compose(block_id.0, overlay).raw();
    // The box spans the block: world min corner = block_low_recentred, far plane = + density.
    let lo = [
        block_low_recentred[0] as f32,
        block_low_recentred[1] as f32,
        block_low_recentred[2] as f32,
    ];
    let hi = [
        (block_low_recentred[0] + density as i64) as f32,
        (block_low_recentred[1] + density as i64) as f32,
        (block_low_recentred[2] + density as i64) as f32,
    ];
    aabb.expand(glam::Vec3::new(lo[0], lo[1], lo[2]));
    aabb.expand(glam::Vec3::new(hi[0], hi[1], hi[2]));

    let sink = if overlay { indices_overlay } else { indices };
    let clean_material = CellKey::from_raw(material_id).block_id() as u32;
    for face in &FACE_TEMPLATES {
        // The face's axis + side, and the neighbour block across it.
        let (axis, side) = face_axis_side(face.neighbor_delta);
        let neighbour = [
            abs_block[0] + face.neighbor_delta[0] as i64,
            abs_block[1] + face.neighbor_delta[1] as i64,
            abs_block[2] + face.neighbor_delta[2] as i64,
        ];
        // The neighbour's MATCHING face is on the same axis, OPPOSITE side. If it is fully
        // solid, every cell behind this face is backed ⇒ cull. Otherwise emit the whole
        // block face (the merged-box over-draw rule — any partly-exposed face is emitted,
        // and a fully-occluded over-draw is back-face-culled / depth-buried).
        let neighbour_face_solid = face_solidity_at(neighbour).face_is_solid(axis, 1 - side);
        if neighbour_face_solid {
            continue;
        }
        let base = vertices.len() as u32;
        for corner in &face.corners {
            let world = [
                if corner[0] == 0 { lo[0] } else { hi[0] },
                if corner[1] == 0 { lo[1] } else { hi[1] },
                if corner[2] == 0 { lo[2] } else { hi[2] },
            ];
            vertices.push(CuboidVertex {
                position: world,
                normal: face.normal,
                material_id: clean_material,
            });
        }
        sink.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// Emit a BOUNDARY block's stored microblock cuboids (ADR 0010 Decision 4), exposure tested
/// against a `(density+2)³` apron region whose interior is the block's own voxels (re-expanded
/// from the cuboids) and whose 1-voxel border is filled PER CELL from each NEIGHBOUR block's
/// face occupancy (coarse → whole-face solid via the seam flag, NO densification of the coarse
/// interior; boundary → its own cuboids' face layer; air → empty). This reproduces the dense
/// apron EXACTLY at the block seam, so it reuses [`emit_box_faces`] / [`face_is_exposed`]
/// unchanged and culls every boundary face the dense mesher culls — no over-draw at a partial
/// boundary-to-boundary seam (which would otherwise render as a spurious surface).
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_boundary_block_cuboids(
    geometry: &MicroblockGeometry,
    density: u32,
    block_low_recentred: [i64; 3],
    abs_block: [i64; 3],
    face_cells_at: &dyn Fn([i64; 3], usize, usize) -> NeighbourFace,
    vertices: &mut Vec<CuboidVertex>,
    indices: &mut Vec<u32>,
    indices_overlay: &mut Vec<u32>,
    aabb: &mut Aabb,
) {
    // Apron frame: a (density+2)³ region with the block's voxels at local index +1, so the
    // 1-voxel border is the apron. `face_is_exposed` then tests the neighbour cell exactly.
    let apron_extent = [density + 2, density + 2, density + 2];
    let mut apron = VoxelRegion::new_empty(apron_extent);

    // Interior: the block's own voxels (the cuboids' render keys), shifted +1.
    for cuboid in &geometry.cuboids {
        for vz in cuboid.min[2]..=cuboid.max[2] {
            for vy in cuboid.min[1]..=cuboid.max[1] {
                for vx in cuboid.min[0]..=cuboid.max[0] {
                    apron.set(vx + 1, vy + 1, vz + 1, Some(cuboid.material_id()));
                }
            }
        }
    }

    // Apron border: each of the 6 outer planes is filled PER CELL from the neighbour block's
    // matching (opposite-side) face. A constant non-zero key marks "solid" (the apron is only
    // read for occupancy by `face_is_exposed`).
    const APRON_SOLID: u16 = 1;
    let d = density;
    for (axis, side, delta) in [
        (0usize, 0usize, [-1i64, 0, 0]),
        (0, 1, [1, 0, 0]),
        (1, 0, [0, -1, 0]),
        (1, 1, [0, 1, 0]),
        (2, 0, [0, 0, -1]),
        (2, 1, [0, 0, 1]),
    ] {
        let neighbour = [
            abs_block[0] + delta[0],
            abs_block[1] + delta[1],
            abs_block[2] + delta[2],
        ];
        // The neighbour's MATCHING face is on the same axis, OPPOSITE side.
        let neighbour_face = face_cells_at(neighbour, axis, 1 - side);
        if matches!(neighbour_face, NeighbourFace::Air) {
            continue; // fully air ⇒ nothing to cull against on this plane
        }
        let plane = if side == 0 { 0u32 } else { d + 1 };
        let (axis_a, axis_b) = in_plane_axes(axis);
        for ai in 0..d {
            for bi in 0..d {
                let solid = match &neighbour_face {
                    NeighbourFace::Solid => true,
                    NeighbourFace::Cells(cells) => cells[(bi * d + ai) as usize],
                    NeighbourFace::Air => false,
                };
                if !solid {
                    continue;
                }
                // Apron-local cell: the block's in-plane index `ai/bi` sits at apron +1; the
                // out-of-plane coord is the border `plane`.
                let mut coord = [0u32; 3];
                coord[axis] = plane;
                coord[axis_a] = ai + 1;
                coord[axis_b] = bi + 1;
                apron.set(coord[0], coord[1], coord[2], Some(APRON_SOLID));
            }
        }
    }

    // Region offset maps apron-local index 0 to the recentred frame: the block's low voxel
    // is apron-local +1, so apron-local 0 sits at `block_low_recentred - 1`.
    let region_offset = [
        (block_low_recentred[0] - 1) as f32,
        (block_low_recentred[1] - 1) as f32,
        (block_low_recentred[2] - 1) as f32,
    ];
    for cuboid in &geometry.cuboids {
        // The cuboid in apron-local frame (+1 shift).
        let shifted = VoxelBox {
            min: [cuboid.min[0] + 1, cuboid.min[1] + 1, cuboid.min[2] + 1],
            max: [cuboid.max[0] + 1, cuboid.max[1] + 1, cuboid.max[2] + 1],
            label: cuboid.material_id(),
        };
        let sink = if box_has_overlay(&shifted) {
            &mut *indices_overlay
        } else {
            &mut *indices
        };
        emit_box_faces(&shifted, &apron, region_offset, vertices, sink, aabb);
    }
}

/// Whether a recentred-frame voxel `v` is meshed under the layer band + optional region
/// clip (ADR 0018 Decision 5). `z_in_band` is the band's Z-slice test (or the ghost slab's,
/// for a ghost build). With no region this is the plain scene-wide band (pre-ADR-0018). With
/// a region the band is either CONFINED to it (solid: outside renders finished) or the region
/// HARD-CLIPS (ghost: only inside is meshed) — see [`RegionRole`].
#[inline]
pub(crate) fn voxel_meshed(v: [i64; 3], z_in_band: &dyn Fn(i64) -> bool, region: Option<RegionClip>) -> bool {
    match region {
        None => z_in_band(v[2]),
        Some(clip) => {
            let inside = clip.contains(v);
            match clip.role {
                // Solid: inside the region clip to the band; outside render finished.
                RegionRole::ConfineBand => {
                    if inside {
                        z_in_band(v[2])
                    } else {
                        true
                    }
                }
                // Ghost: only voxels inside the region AND in the ghost slab are meshed.
                RegionRole::ClipToRegion => inside && z_in_band(v[2]),
            }
        }
    }
}

/// Stamp the block at chunk-local-or-neighbour block index `abs_block`'s per-voxel occupancy
/// into `region` at the apron-local offset `dst_lo` (so a neighbour block lands at the apron
/// border), CLIPPED to the band + optional region via [`voxel_meshed`] (ADR 0010 #53 / ADR
/// 0018 Decision 5). A coarse-solid block fills
/// every `density³` cell at its render key; a boundary block stamps each cuboid; an air /
/// missing block stamps nothing. `block_low_recentred_z` is the block's low voxel-Z in the
/// recentred frame, so a block-local voxel-Z `vz` maps to recentred Z
/// `block_low_recentred_z + vz` for the band test — masking out-of-band voxels to air on BOTH
/// the meshed interior and the neighbour apron, exactly as the dense banded path masks apron.
///
/// Writes only cells whose apron-local index lands inside `region.extent` (a neighbour block
/// contributes only its 1-voxel abutting border layer). Returns nothing; the caller sizes the
/// apron and supplies `dst_lo`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn stamp_block_into_region_banded(
    chunk_by_coord: &std::collections::HashMap<[i32; 3], &TwoLayerChunk>,
    abs_block: [i64; 3],
    density: u32,
    block_low_recentred: [i64; 3],
    dst_lo: [i64; 3],
    z_in_band: &dyn Fn(i64) -> bool,
    clip: Option<RegionClip>,
    out_region: &mut VoxelRegion,
) {
    let chunk_blocks = CHUNK_BLOCKS as i64;
    let chunk_coord = [
        abs_block[0].div_euclid(chunk_blocks) as i32,
        abs_block[1].div_euclid(chunk_blocks) as i32,
        abs_block[2].div_euclid(chunk_blocks) as i32,
    ];
    let Some(chunk) = chunk_by_coord.get(&chunk_coord) else {
        return; // no covering chunk → air
    };
    let local = [
        abs_block[0].rem_euclid(chunk_blocks) as u32,
        abs_block[1].rem_euclid(chunk_blocks) as u32,
        abs_block[2].rem_euclid(chunk_blocks) as u32,
    ];
    let [ex, ey, ez] = out_region.extent;

    // Stamp one block-local voxel `(vx, vy, vz)` of render key `key` into the region,
    // masked to the band + optional region clip (ADR 0018 Decision 5) and bounds-checked
    // against the apron extent.
    let stamp = |vx: u32, vy: u32, vz: u32, key: u16, out_region: &mut VoxelRegion| {
        let recentred = [
            block_low_recentred[0] + vx as i64,
            block_low_recentred[1] + vy as i64,
            block_low_recentred[2] + vz as i64,
        ];
        if !voxel_meshed(recentred, z_in_band, clip) {
            return;
        }
        let lx = dst_lo[0] + vx as i64;
        let ly = dst_lo[1] + vy as i64;
        let lz = dst_lo[2] + vz as i64;
        if lx < 0 || ly < 0 || lz < 0 || lx >= ex as i64 || ly >= ey as i64 || lz >= ez as i64 {
            return;
        }
        out_region.set(lx as u32, ly as u32, lz as u32, Some(key));
    };

    if let Some(block_id) = chunk.coarse_block(local) {
        let key = CellKey::compose(block_id.0, chunk.coarse_block_overlay(local)).raw();
        for vz in 0..density {
            for vy in 0..density {
                for vx in 0..density {
                    stamp(vx, vy, vz, key, out_region);
                }
            }
        }
    } else if let Some(geometry) = chunk.microblocks.get(&local) {
        for cuboid in &geometry.cuboids {
            for vz in cuboid.min[2]..=cuboid.max[2] {
                for vy in cuboid.min[1]..=cuboid.max[1] {
                    for vx in cuboid.min[0]..=cuboid.max[0] {
                        stamp(vx, vy, vz, cuboid.material_id(), out_region);
                    }
                }
            }
        }
    }
    // else: air block, nothing to stamp.
}

/// Mesh ONE block (coarse OR boundary) under an ACTIVE layer band (ADR 0010 #53). Builds a
/// `(density+2)³` apron region whose INTERIOR is the block's own band-clipped voxels and whose
/// 1-voxel border is each neighbour block's abutting band-clipped face — then decomposes the
/// interior and emits via [`emit_box_faces`]/[`face_is_exposed`], so a band-edge cut (the
/// out-of-band neighbour cell reads as AIR) synthesises a real cap face, and a non-cut seam
/// against a solid neighbour is still culled. This is the dense banded apron restricted to one
/// block: it densifies ONLY the band-cut block (never the whole solid interior). Returns the
/// number of boxes the interior decomposed into (the diagnostic box count).
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_block_banded(
    density: u32,
    block_low_recentred: [i64; 3],
    abs_block: [i64; 3],
    chunk_by_coord: &std::collections::HashMap<[i32; 3], &TwoLayerChunk>,
    z_in_band: &dyn Fn(i64) -> bool,
    clip: Option<RegionClip>,
    vertices: &mut Vec<CuboidVertex>,
    indices: &mut Vec<u32>,
    indices_overlay: &mut Vec<u32>,
    aabb: &mut Aabb,
) -> u32 {
    // Apron frame: a (density+2)³ region with the block's voxels at local index +1, so the
    // 1-voxel border is the apron (identical to `emit_boundary_block_cuboids`).
    let apron_extent = [density + 2, density + 2, density + 2];
    let mut interior = VoxelRegion::new_empty(apron_extent);

    // Interior = THIS block's own voxels at local +1, band- + region-clipped.
    stamp_block_into_region_banded(
        chunk_by_coord,
        abs_block,
        density,
        block_low_recentred,
        [1, 1, 1],
        z_in_band,
        clip,
        &mut interior,
    );

    // The interior decomposition + the apron border share one region: decompose reads only the
    // interior (+1 shift keeps the border air for the decompose), and `face_is_exposed` reads
    // the SAME region's border. We therefore clone the interior-only region for decomposition
    // BEFORE filling the apron border (so a box never grows into the border), then fill the
    // border into the exposure region.
    let decompose_region = interior.clone();
    let mut apron = interior; // reuse as the exposure region; add the neighbour border below.

    // Apron border: each of the 6 neighbour blocks' abutting face, band-clipped. A neighbour
    // block landed at the apron border via `dst_lo` = its block offset relative to this block
    // (−1 block on the low side, +1 on the high side, scaled to the apron's +1 interior
    // origin). Only the single border layer of each neighbour falls inside the apron extent.
    for (delta, dst_lo) in [
        ([-1i64, 0, 0], [1 - density as i64, 1, 1]),
        ([1, 0, 0], [1 + density as i64, 1, 1]),
        ([0, -1, 0], [1, 1 - density as i64, 1]),
        ([0, 1, 0], [1, 1 + density as i64, 1]),
        ([0, 0, -1], [1, 1, 1 - density as i64]),
        ([0, 0, 1], [1, 1, 1 + density as i64]),
    ] {
        let neighbour = [
            abs_block[0] + delta[0],
            abs_block[1] + delta[1],
            abs_block[2] + delta[2],
        ];
        let neighbour_low_recentred = [
            block_low_recentred[0] + delta[0] * density as i64,
            block_low_recentred[1] + delta[1] * density as i64,
            block_low_recentred[2] + delta[2] * density as i64,
        ];
        stamp_block_into_region_banded(
            chunk_by_coord,
            neighbour,
            density,
            neighbour_low_recentred,
            dst_lo,
            z_in_band,
            clip,
            &mut apron,
        );
    }

    // Region offset maps apron-local index 0 to the recentred frame: the block's low voxel is
    // apron-local +1, so apron-local 0 sits at `block_low_recentred - 1`.
    let region_offset = [
        (block_low_recentred[0] - 1) as f32,
        (block_low_recentred[1] - 1) as f32,
        (block_low_recentred[2] - 1) as f32,
    ];
    let boxes = decompose_into_boxes(&decompose_region);
    for voxel_box in &boxes {
        let sink = if box_has_overlay(voxel_box) {
            &mut *indices_overlay
        } else {
            &mut *indices
        };
        emit_box_faces(voxel_box, &apron, region_offset, vertices, sink, aabb);
    }
    boxes.len() as u32
}

/// The `(axis, side)` a face-template's `neighbor_delta` points along: axis 0/1/2 = X/Y/Z,
/// side 0 = low (delta −1), side 1 = high (delta +1).
#[inline]
pub(crate) fn face_axis_side(delta: [i32; 3]) -> (usize, usize) {
    for (axis, &d) in delta.iter().enumerate() {
        if d > 0 {
            return (axis, 1);
        }
        if d < 0 {
            return (axis, 0);
        }
    }
    (0, 0)
}

/// Emit the exposed faces of one box into the shared vertex/index buffers,
/// expanding `aabb` to contain the box. A face is exposed when the voxel cell
/// immediately beyond it (per axis, across the box's full extent on the other two
/// axes) is air — at minimum this culls box-internal faces; here it also culls
/// faces fully covered by adjacent solid voxels.
pub(crate) fn emit_box_faces(
    voxel_box: &VoxelBox,
    region: &VoxelRegion,
    world_offset: [f32; 3],
    vertices: &mut Vec<CuboidVertex>,
    indices: &mut Vec<u32>,
    aabb: &mut Aabb,
) {
    let [min_x, min_y, min_z] = voxel_box.min;
    let [max_x, max_y, max_z] = voxel_box.max;
    // Inclusive box → the far plane is at max + 1.
    let lo = [min_x as f32, min_y as f32, min_z as f32];
    let hi = [
        (max_x + 1) as f32,
        (max_y + 1) as f32,
        (max_z + 1) as f32,
    ];

    // Expand the chunk AABB to this box's world extent (local index + offset).
    aabb.expand(glam::Vec3::new(lo[0] + world_offset[0], lo[1] + world_offset[1], lo[2] + world_offset[2]));
    aabb.expand(glam::Vec3::new(hi[0] + world_offset[0], hi[1] + world_offset[1], hi[2] + world_offset[2]));

    // The clean colour index (ADR 0003 §3c): the box's on-face-grid flag is NOT a vertex
    // attribute — the caller routed this box to the overlay-on or overlay-off index run by
    // its key bit, and the draw sets the per-draw overlay-active uniform per run. So strip
    // the overlay bit here and write only the categorical id into the vertex.
    let material_id = CellKey::from_raw(voxel_box.material_id()).block_id() as u32;

    for face in &FACE_TEMPLATES {
        if !face_is_exposed(voxel_box, region, face.neighbor_delta) {
            continue;
        }
        let base = vertices.len() as u32;
        for corner in &face.corners {
            // 0 → min plane (lo), 1 → max+1 plane (hi); shift into world space.
            let world = [
                (if corner[0] == 0 { lo[0] } else { hi[0] }) + world_offset[0],
                (if corner[1] == 0 { lo[1] } else { hi[1] }) + world_offset[1],
                (if corner[2] == 0 { lo[2] } else { hi[2] }) + world_offset[2],
            ];
            vertices.push(CuboidVertex {
                position: world,
                normal: face.normal,
                material_id,
            });
        }
        // Two CCW triangles per quad (matching the instanced winding scheme).
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// Whether a decomposed box carries the on-face-grid overlay marker in its region-cell
/// key (ADR 0003 §3c). Routes the box to the overlay-on index run.
#[inline]
pub(crate) fn box_has_overlay(voxel_box: &VoxelBox) -> bool {
    CellKey::from_raw(voxel_box.material_id()).has_overlay()
}

/// Is the given face of the box exposed against the dense apron `region`? Thin domain
/// adapter over the substrate [`CulledBoxMeshing`] culling kernel (slice S10): it supplies
/// the neighbour-solidity oracle by reading this mesher's [`VoxelRegion`] occupancy — a cell
/// is solid iff it is in bounds and carries a render key. Negative or out-of-extent cells
/// answer air (exposed), reproducing the dense apron's border-is-air convention exactly.
///
/// The kernel keeps ONE quad per box face (not per voxel): if a merged face is partially
/// exposed, the whole quad is emitted (over-draw of at most the box's own face, never a
/// hole). See [`CulledBoxMeshing::face_is_exposed`] and `docs/architecture/03-display.md`.
pub(crate) fn face_is_exposed(voxel_box: &VoxelBox, region: &VoxelRegion, delta: [i32; 3]) -> bool {
    CulledBoxMeshing::face_is_exposed(voxel_box, delta, |[nx, ny, nz]| {
        nx >= 0
            && ny >= 0
            && nz >= 0
            && region.cell_at(nx as u32, ny as u32, nz as u32).is_some()
    })
}
