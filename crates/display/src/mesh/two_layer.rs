use super::*;

// ===========================================================================
// ADR 0010 E3 — the TWO-LAYER mesher (one-box coarse + cuboid microblock +
// seam-flag culling). Builds a chunk's mesh from its [`TwoLayerChunk`] instead of
// a dense `VoxelGrid`, and PROVES (the E3 parity gate) the exposed-face set is
// identical to the dense [`build_chunk_meshes_with_apron`].
// ===========================================================================

/// Whether the WHOLE shared face of one block is solid, for seam-flag culling (ADR 0010
/// Decision 4). A face that is fully solid backs every cell on the neighbour's matching
/// face, so the neighbour's face there is occluded and culled. A coarse-solid block is
/// solid on all 6 faces; an air block on none; a boundary block per its [`SeamSolidity`].
/// `None` = the block is air / does not exist (no covering chunk) ⇒ never solid.
#[derive(Debug, Clone, Copy)]
pub(crate) enum BlockFaceSolidity {
    /// Every face fully solid (a coarse-solid block, or a fully-interior boundary block).
    AllSolid,
    /// Per-face solidity (a boundary block's stored seam flags).
    PerFace(SeamSolidity),
    /// Air / outside any covering chunk — no face is solid.
    None,
}

impl BlockFaceSolidity {
    /// Whether this block's face on `axis` (0/1/2), `side` (0 low / 1 high) is fully solid.
    pub(crate) fn face_is_solid(&self, axis: usize, side: usize) -> bool {
        match self {
            BlockFaceSolidity::AllSolid => true,
            BlockFaceSolidity::PerFace(seam) => seam.face_is_solid(axis, side),
            BlockFaceSolidity::None => false,
        }
    }
}

/// The PER-CELL occupancy of a neighbour block's face abutting a boundary block (ADR 0010 E3).
/// A coarse-solid neighbour is `Solid` (the seam-flag fast path — no densification); an air /
/// missing neighbour is `Air`; a boundary neighbour carries its face layer's `density²`
/// occupancy bitmap. This is the exact neighbour info the dense apron carried, restricted to
/// the SURFACE blocks so coarse interiors are never densified.
pub(crate) enum NeighbourFace {
    /// The whole face is solid (a coarse-solid neighbour).
    Solid,
    /// The whole face is air (no covering chunk, or an air block).
    Air,
    /// Per-cell occupancy, indexed `cells[in_plane_b * density + in_plane_a]` over the two
    /// axes other than the face axis (ascending order — see [`in_plane_axes`]).
    Cells(Vec<bool>),
}

/// The two axes IN the plane of a face whose normal is along `axis` (0/1/2 = X/Y/Z), in
/// ascending order: axis 0 → (1, 2), axis 1 → (0, 2), axis 2 → (0, 1). The canonical
/// in-plane (a, b) ordering both the neighbour-face bitmap and the apron fill index by.
#[inline]
pub(crate) fn in_plane_axes(axis: usize) -> (usize, usize) {
    match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

/// How the two-layer mesher meshes one block under a band + optional region clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockRoute {
    /// Nothing to emit (fully carved away by band / region).
    Skip,
    /// Densify + apron-mesh with the per-voxel band + region mask (`emit_block_banded`).
    Banded,
    /// Emit the E3 FAST path (coarse one-box / boundary cuboids), unclipped/finished.
    Fast,
}

/// Decide how block `[block_lo, block_hi]` (INCLUSIVE recentred corners) is meshed under
/// `band_active`/`region` (ADR 0018 Decision 5). See the routing analysis in the mesh
/// module: a SOLID pass renders a block wholly outside the region FAST (finished), routes
/// the region's 1-block shell + interior band-cut blocks through the banded mesher (so a
/// fast block never abuts a band-clipped one), and skips a fully-inside block that the band
/// wholly excludes; a GHOST pass skips everything outside the region or the slab.
#[inline]
fn decide_block_route(
    band_active: bool,
    region: Option<RegionClip>,
    block_lo: [i64; 3],
    block_hi: [i64; 3],
    block_extent: i64,
    fully_out_of_band_z: bool,
) -> BlockRoute {
    match region {
        Some(clip) => match clip.role {
            RegionRole::ConfineBand => {
                if clip.block_fully_outside(block_lo, block_hi)
                    && !clip.block_intersects_dilated(block_lo, block_hi, block_extent)
                {
                    // Wholly outside the region and not adjacent to it → finished, fast.
                    BlockRoute::Fast
                } else if clip.block_fully_inside(block_lo, block_hi) && fully_out_of_band_z {
                    // Wholly inside the region but entirely out of band → all air.
                    BlockRoute::Skip
                } else {
                    BlockRoute::Banded
                }
            }
            RegionRole::ClipToRegion => {
                if clip.block_fully_outside(block_lo, block_hi) || fully_out_of_band_z {
                    BlockRoute::Skip
                } else {
                    BlockRoute::Banded
                }
            }
        },
        None => {
            if !band_active {
                BlockRoute::Fast
            } else if fully_out_of_band_z {
                BlockRoute::Skip
            } else {
                BlockRoute::Banded
            }
        }
    }
}

/// Build the per-chunk exposed-face meshes from the two-layer chunks (ADR 0010 E3). A
/// coarse-solid block emits ONE box (no per-voxel decompose of the solid interior); a
/// boundary block emits its stored microblock cuboids; inter-block / inter-chunk seam
/// faces are culled via the per-face seam-solidity flags (the coarse-vs-microblock apron
/// analogue) rather than a densified neighbour apron.
///
/// `chunks` is `(absolute_chunk_coord, TwoLayerChunk)` per covering chunk;
/// `grid_dimensions` is the whole composite voxel dims — only the Z half is read, to map a
/// recentred-frame voxel index to its ABSOLUTE layer for the band clip (Z-up: layers are
/// Z-slices); `recentre_voxels` is the resolve's carried recentre (ADR 0008) so the emitted
/// vertices land in the SAME world frame the dense path assembles (its global cloud-min
/// anchor cancels to exactly this recentred index — proven in the E3 parity test).
/// `voxels_per_block` is the chunk density.
///
/// `band` (ADR 0010 #53): a layer-range (Z-slice) clip. `LayerBand::FULL` (the default) keeps
/// the E3-proven FAST paths byte-for-byte — a coarse-solid block is ONE box, a boundary block
/// its stored cuboids. An ACTIVE band (the layer scrubber) clips each block to the band's
/// recentred voxel-Z range: a coarse block the band CUTS through emits the clipped one-box (the
/// block ∩ band), a boundary block clips each cuboid; blocks fully outside the band are skipped.
/// Cut-plane faces are VISIBLE — a band edge reads the out-of-band neighbour cell as AIR, so the
/// clip synthesises a real cap face there, mirroring the dense [`build_chunk_meshes_with_apron`]
/// banded behaviour exactly (it masks the apron + interior so a merged column caps at the edge).
pub(crate) fn build_two_layer_chunk_meshes(
    chunks: &[([i32; 3], Arc<TwoLayerChunk>)],
    grid_dimensions: [u32; 3],
    recentre: RecentreVoxels,
    voxels_per_block: u32,
    band: LayerBand,
    region: Option<RegionClip>,
) -> Vec<CuboidChunkMesh> {
    build_two_layer_chunk_meshes_filtered(
        chunks,
        None,
        grid_dimensions,
        recentre,
        voxels_per_block,
        band,
        region,
    )
}

/// Like [`build_two_layer_chunk_meshes`] but meshes ONLY the chunks in `only` (when
/// `Some`), the two-layer analogue of [`build_chunk_meshes_with_apron_filtered`] (issue
/// #55). Seam-flag culling reads every chunk's neighbours from the FULL `chunks` set (the
/// `chunk_by_coord` lookup below is over ALL chunks), so a subset build is byte-identical to
/// the same chunks within a wholesale build — a skipped neighbour's coarse / microblock face
/// solidity still culls the meshed chunk's seam faces. `None` meshes every chunk (the
/// wholesale path). This is the seam the two-layer INCREMENTAL rebuild
/// ([`CuboidMeshRenderer::incremental_rebuild_from_two_layer_chunks`]) uses: it passes the
/// full resident set for correct seam culling but re-meshes only the dirty-dilated subset.
pub(crate) fn build_two_layer_chunk_meshes_filtered(
    chunks: &[([i32; 3], Arc<TwoLayerChunk>)],
    only: Option<&std::collections::HashSet<[i32; 3]>>,
    grid_dimensions: [u32; 3],
    recentre: RecentreVoxels,
    voxels_per_block: u32,
    band: LayerBand,
    region: Option<RegionClip>,
) -> Vec<CuboidChunkMesh> {
    // Unwrap the carried frame at the per-chunk rebase arithmetic below (`chunk_min_recentred`).
    let recentre_voxels = recentre.voxels();
    let density = voxels_per_block.max(1);
    let block_extent = density as i64;

    // Z-up band clip (ADR 0010 #53): the band is in ABSOLUTE layer indices. A voxel at
    // recentred-frame min-corner `v` (the frame this mesher emits in) sits at world.z = v +
    // 0.5, so its absolute layer = floor(world.z + half_z) = v + half_z (integer-valued for
    // an integer `v`, `half_z`). Inverting the band into the recentred frame: a recentred
    // voxel-Z `v` is in-band iff `band_min - half_z <= v <= band_max - half_z`. FLOORED half
    // (matches the dense path's `floor(world.z + floor(dim/2))` for any dim parity).
    let band_active = band.band_min > 0 || band.band_max != u32::MAX;
    let half_z = (grid_dimensions[2] / 2) as i64;
    let band_lo_recentred = band.band_min as i64 - half_z;
    let band_hi_recentred = (band.band_max as i64).saturating_sub(half_z);
    // Whether a recentred-frame voxel-Z index is inside the band.
    let z_in_band = |recentred_z: i64| -> bool {
        if !band_active {
            return true;
        }
        recentred_z >= band_lo_recentred && recentred_z <= band_hi_recentred
    };
    let chunk_extent_voxels = (CHUNK_BLOCKS * density) as i64;

    // A lookup of every covering chunk by coord so a block can consult its neighbour's
    // coarse / microblock face solidity across a block OR chunk seam.
    let chunk_by_coord: std::collections::HashMap<[i32; 3], &TwoLayerChunk> =
        chunks.iter().map(|(coord, chunk)| (*coord, chunk.as_ref())).collect();

    // The block-face solidity of the block at ABSOLUTE block coord `abs_block` (across all
    // chunks): resolve which chunk + chunk-local block it is, then read its layer.
    let face_solidity_at = |abs_block: [i64; 3]| -> BlockFaceSolidity {
        let chunk_blocks = CHUNK_BLOCKS as i64;
        let chunk_coord = [
            abs_block[0].div_euclid(chunk_blocks) as i32,
            abs_block[1].div_euclid(chunk_blocks) as i32,
            abs_block[2].div_euclid(chunk_blocks) as i32,
        ];
        let Some(chunk) = chunk_by_coord.get(&chunk_coord) else {
            return BlockFaceSolidity::None;
        };
        let local = [
            abs_block[0].rem_euclid(chunk_blocks) as u32,
            abs_block[1].rem_euclid(chunk_blocks) as u32,
            abs_block[2].rem_euclid(chunk_blocks) as u32,
        ];
        if chunk.coarse_block(local).is_some() {
            BlockFaceSolidity::AllSolid
        } else if let Some(geometry) = chunk.microblocks.get(&local) {
            BlockFaceSolidity::PerFace(geometry.seam_solidity)
        } else {
            BlockFaceSolidity::None
        }
    };

    // The PER-CELL occupancy of the block at `abs_block`'s face on `(axis, side)` — the
    // 1-voxel layer that abuts a neighbouring block across that face. A coarse-solid block
    // is fully solid (the seam-flag fast path — no densification); an air block fully air; a
    // boundary block expands ITS cuboids' face layer per cell. This is the exact neighbour
    // info the dense apron carried — but only for the SURFACE (boundary) blocks, so coarse
    // interiors are still never densified. The returned bitmap is indexed
    // `cell[in_plane_b * density + in_plane_a]`, with `(in_plane_a, in_plane_b)` = the two
    // axes other than `axis` in ascending order — the SAME order the apron fill walks.
    let face_cells_at = |abs_block: [i64; 3], axis: usize, side: usize| -> NeighbourFace {
        let chunk_blocks = CHUNK_BLOCKS as i64;
        let chunk_coord = [
            abs_block[0].div_euclid(chunk_blocks) as i32,
            abs_block[1].div_euclid(chunk_blocks) as i32,
            abs_block[2].div_euclid(chunk_blocks) as i32,
        ];
        let Some(chunk) = chunk_by_coord.get(&chunk_coord) else {
            return NeighbourFace::Air;
        };
        let local = [
            abs_block[0].rem_euclid(chunk_blocks) as u32,
            abs_block[1].rem_euclid(chunk_blocks) as u32,
            abs_block[2].rem_euclid(chunk_blocks) as u32,
        ];
        if chunk.coarse_block(local).is_some() {
            return NeighbourFace::Solid;
        }
        let Some(geometry) = chunk.microblocks.get(&local) else {
            return NeighbourFace::Air;
        };
        // Expand the boundary block's cuboids' face layer (the plane `coord == 0` for the low
        // face, `coord == density-1` for the high face on `axis`) into a density² bitmap.
        let (axis_a, axis_b) = in_plane_axes(axis);
        let plane = if side == 0 { 0u32 } else { density - 1 };
        let mut cells = vec![false; (density * density) as usize];
        for cuboid in &geometry.cuboids {
            // Does this cuboid touch the requested plane on `axis`?
            if (cuboid.min[axis]..=cuboid.max[axis]).contains(&plane) {
                for a in cuboid.min[axis_a]..=cuboid.max[axis_a] {
                    for b in cuboid.min[axis_b]..=cuboid.max[axis_b] {
                        cells[(b * density + a) as usize] = true;
                    }
                }
            }
        }
        NeighbourFace::Cells(cells)
    };

    // Each chunk meshes INDEPENDENTLY: the per-chunk body below writes only its own local
    // `vertices` / `indices` / `indices_overlay` / `aabb` / `box_count`, reading only shared-
    // IMMUTABLE state (the `chunk_by_coord` map, the `face_solidity_at` / `face_cells_at` /
    // `z_in_band` closures, the `only` filter, the band bounds). So the chunk list is meshed in
    // parallel with rayon. A parallel `.collect()` PRESERVES the source order (issue #57
    // convention), so the output Vec — hence GPU buffer order and the goldens — is byte-identical
    // to the former serial loop.
    let meshes: Vec<CuboidChunkMesh> = chunks
        .par_iter()
        .filter_map(|(chunk_coord, chunk)| {
        // Incremental subset (issue #55): skip chunks not in the rebuild set. Seam culling
        // still consults every chunk (the `chunk_by_coord` lookup above is over the FULL set),
        // so a skipped neighbour's face solidity correctly culls the meshed chunk's seam faces.
        if let Some(only) = only {
            if !only.contains(chunk_coord) {
                return None;
            }
        }
        // The chunk's low voxel corner in the RECENTRED frame (ADR 0008): a chunk-local
        // voxel index `lv` lands at world min-corner `chunk_min - recentre + lv`. Emitting
        // box corners there matches the dense path's `global_index + (min_world - 0.5)`
        // exactly (its cloud-min anchor cancels — see the parity test).
        let chunk_min_recentred = [
            chunk_coord[0] as i64 * chunk_extent_voxels - recentre_voxels[0],
            chunk_coord[1] as i64 * chunk_extent_voxels - recentre_voxels[1],
            chunk_coord[2] as i64 * chunk_extent_voxels - recentre_voxels[2],
        ];
        // Each block's absolute block coord low = chunk_coord * CHUNK_BLOCKS + local block.
        let chunk_block_base = [
            chunk_coord[0] as i64 * CHUNK_BLOCKS as i64,
            chunk_coord[1] as i64 * CHUNK_BLOCKS as i64,
            chunk_coord[2] as i64 * CHUNK_BLOCKS as i64,
        ];

        let mut vertices: Vec<CuboidVertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut indices_overlay: Vec<u32> = Vec::new();
        let mut aabb = Aabb::empty();
        let mut box_count = 0u32;

        for block_z in 0..CHUNK_BLOCKS {
            for block_y in 0..CHUNK_BLOCKS {
                for block_x in 0..CHUNK_BLOCKS {
                    let block = [block_x, block_y, block_z];
                    let abs_block = [
                        chunk_block_base[0] + block_x as i64,
                        chunk_block_base[1] + block_y as i64,
                        chunk_block_base[2] + block_z as i64,
                    ];
                    // The block's low voxel corner in the recentred frame.
                    let block_low_recentred = [
                        chunk_min_recentred[0] + block_x as i64 * block_extent,
                        chunk_min_recentred[1] + block_y as i64 * block_extent,
                        chunk_min_recentred[2] + block_z as i64 * block_extent,
                    ];

                    // ADR 0010 #53 / ADR 0018 Decision 5: decide this block's route. A
                    // band-cut block (or a block the region clip straddles) goes through the
                    // band-aware apron mesher `emit_block_banded` — it densifies only the
                    // block (never the whole solid interior), masks out-of-band /
                    // out-of-region voxels to air on BOTH interior and apron (so a cut
                    // synthesises a real cap face), and skips blocks fully carved away. A
                    // block wholly outside the region (SOLID pass) keeps the E3-proven FAST
                    // paths below (rendered finished). FULL-band + no region ⇒ every block
                    // takes the fast path byte-for-byte.
                    let block_lo_z = block_low_recentred[2];
                    let block_hi_z = block_lo_z + block_extent - 1;
                    let fully_out_of_band_z =
                        block_hi_z < band_lo_recentred || block_lo_z > band_hi_recentred;
                    let block_hi = [
                        block_low_recentred[0] + block_extent - 1,
                        block_low_recentred[1] + block_extent - 1,
                        block_hi_z,
                    ];
                    let route = decide_block_route(
                        band_active,
                        region,
                        block_low_recentred,
                        block_hi,
                        block_extent,
                        fully_out_of_band_z,
                    );
                    match route {
                        BlockRoute::Skip => continue,
                        BlockRoute::Banded => {
                            box_count += emit_block_banded(
                                density,
                                block_low_recentred,
                                abs_block,
                                &chunk_by_coord,
                                &z_in_band,
                                region,
                                &mut vertices,
                                &mut indices,
                                &mut indices_overlay,
                                &mut aabb,
                            );
                        }
                        BlockRoute::Fast => {
                            if let Some(block_id) = chunk.coarse_block(block) {
                                // COARSE-SOLID → ONE box spanning the block (no per-voxel decompose).
                                let overlay = chunk.coarse_block_overlay(block);
                                emit_coarse_block_box(
                                    block_id,
                                    overlay,
                                    density,
                                    block_low_recentred,
                                    abs_block,
                                    &face_solidity_at,
                                    &mut vertices,
                                    &mut indices,
                                    &mut indices_overlay,
                                    &mut aabb,
                                );
                                box_count += 1;
                            } else if let Some(geometry) = chunk.microblocks.get(&block) {
                                // BOUNDARY → its stored microblock cuboids, exposure tested against a
                                // block-local apron filled PER CELL from the NEIGHBOUR blocks' face
                                // occupancy (coarse → whole-face solid via the seam flag; boundary →
                                // its own cuboids' face layer) — matching the dense apron exactly.
                                emit_boundary_block_cuboids(
                                    geometry,
                                    density,
                                    block_low_recentred,
                                    abs_block,
                                    &face_cells_at,
                                    &mut vertices,
                                    &mut indices,
                                    &mut indices_overlay,
                                    &mut aabb,
                                );
                                box_count += geometry.cuboids.len() as u32;
                            }
                            // else: air block, nothing to emit.
                        }
                    }
                }
            }
        }

        if indices.is_empty() && indices_overlay.is_empty() {
            return None;
        }
        Some(CuboidChunkMesh {
            coord: *chunk_coord,
            vertices,
            indices,
            indices_overlay,
            aabb,
            box_count,
        })
        })
        .collect();
    meshes
}
