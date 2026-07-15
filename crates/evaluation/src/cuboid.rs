//! Domain seam for the cuboid mesher's box decomposition.
//!
//! The greedy 3D box-growing algorithm itself is domain-free and lives in
//! [`substrate::solids::greedy_cuboid_decomposition`] as [`GreedyCuboidDecomposition`] over a
//! generic labeled [`substrate::solids::CellGrid`]. This module is the thin adapter that reads
//! that algorithm in the project's own vocabulary: a [`VoxelRegion`] is a dense grid of
//! per-voxel render-cell keys (`u16`), a [`VoxelBox`] is one single-material cuboid, and
//! [`decompose_into_boxes`] runs the substrate decomposition over them.
//!
//! Vintage Story renders a chiseled block not as one instanced cube per voxel but by
//! merging its solid voxels into a small set of axis-aligned, single-material cuboids
//! (`BlockEntityMicroBlock.GenShape`); see the cuboid mesher in
//! `docs/architecture/03-display.md`. The decomposition step is domain-free — its only
//! project-facing pieces are the `u16` render-cell key payload and the
//! [`region_from_voxel_grid`] densifier below, which builds a region from a
//! [`VoxelGrid`] sub-box.

use voxel_core::voxel::VoxelGrid;

/// One axis-aligned, single-material cuboid — the domain reading of a substrate
/// [`Cuboid`] whose label is a `u16` render-cell key.
///
/// Coordinates are **region-local voxel indices** (the lattice the [`VoxelRegion`] is
/// indexed by), `min`/`max` both **inclusive**; the box's material is `label`. See
/// [`Cuboid`] for the coordinate/count semantics.
pub type VoxelBox = Cuboid<u16>;

/// A dense, bounded region of solid/air voxels with a per-voxel `u16` material key —
/// the domain reading of a substrate [`CellGrid`]. `Some(material)` is a solid voxel,
/// `None` is air; see [`CellGrid`] for the row-major layout (X fastest).
///
/// Built by hand (tests), from a [`VoxelGrid`] sub-box ([`region_from_voxel_grid`]), or
/// from a chunk's resolved voxels.
pub type VoxelRegion = CellGrid<u16>;

pub use substrate::solids::{CellGrid, Cuboid, GreedyCuboidDecomposition};

/// Reads a [`VoxelBox`]'s label back in the domain's word. Substrate names the generic
/// payload `label` (it assigns cuboid labels no meaning); at this seam a `VoxelBox`'s label
/// IS the render-cell material key — the `u16` categorical block id with its overlay bit,
/// consumed by the cuboid mesher and the two-layer store. The extension trait restores that
/// vocabulary at every READ site without renaming the substrate field (impossible on a
/// generic). Construction still sets `label` directly (a struct-literal field name cannot be
/// aliased).
pub trait VoxelBoxMaterial {
    /// This box's render-cell material key (substrate's `label`).
    fn material_id(&self) -> u16;
}

impl VoxelBoxMaterial for VoxelBox {
    #[inline]
    fn material_id(&self) -> u16 {
        self.label
    }
}

/// Greedy box decomposition of a region into single-material [`VoxelBox`]es — the
/// domain entry point onto substrate's [`GreedyCuboidDecomposition`].
///
/// The emitted boxes exactly cover the solid voxels, are pairwise disjoint, each carry a
/// single material, and the output is deterministic (see the substrate module for the
/// algorithm and its NP-hardness rationale). Not guaranteed minimal in box count.
pub fn decompose_into_boxes(region: &VoxelRegion) -> Vec<VoxelBox> {
    GreedyCuboidDecomposition::decompose(region)
}

/// Build a dense [`VoxelRegion`] from an axis-aligned sub-box of a [`VoxelGrid`]
/// (the per-chunk adapter the cuboid mesher calls).
///
/// `origin` is the region's minimum voxel index in grid space; `extent` is its
/// size in voxels. A grid voxel maps to its index with the project-wide
/// CORNER-ANCHORED rule `i = round(world_position + floor(dimensions/2) - 0.5)`
/// (see `VoxelGrid::widest_run_in_band` / `renderer::upload_grid`) — the producers
/// emit corner-anchored half-integer centres, so this is exact for any dim parity.
/// Voxels whose index falls inside `[origin, origin + extent)` are copied into the
/// region with their material key; everything else stays air. Out-of-grid origins
/// simply yield air for the uncovered cells.
///
/// Passing `origin = [0, 0, 0]` and `extent = grid.dimensions` decomposes the
/// whole grid in one call.
pub fn region_from_voxel_grid(grid: &VoxelGrid, origin: [u32; 3], extent: [u32; 3]) -> VoxelRegion {
    let mut region = VoxelRegion::new_empty(extent);
    let [grid_x, grid_y, grid_z] = grid.dimensions;
    if grid_x == 0 || grid_y == 0 || grid_z == 0 {
        return region;
    }
    // Corner-anchoring: the grid's index space is `[0, dim)` with voxel centres at
    // `idx + 0.5`, so `idx = floor(world)` — exact for any parity (centres are
    // half-integers). (Was `round(world + dim/2 − 0.5)` for the retired origin-centred
    // grid, which broke for odd dim.)
    for voxel in &grid.occupied {
        let i = voxel.local_index[0] as i64;
        let j = voxel.local_index[1] as i64;
        let k = voxel.local_index[2] as i64;
        if i < 0 || j < 0 || k < 0 {
            continue;
        }
        // Shift into region-local coordinates.
        let lx = i - origin[0] as i64;
        let ly = j - origin[1] as i64;
        let lz = k - origin[2] as i64;
        if lx < 0
            || ly < 0
            || lz < 0
            || lx >= extent[0] as i64
            || ly >= extent[1] as i64
            || lz >= extent[2] as i64
        {
            continue;
        }
        region.set(lx as u32, ly as u32, lz as u32, Some(voxel.cell_key().raw()));
    }
    region
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::voxel::Voxel;

    #[test]
    fn adapter_from_voxel_grid_whole_grid() {
        // Build a tiny VoxelGrid by hand and decompose the whole thing via the
        // adapter, confirming the world_position → index mapping round-trips.
        let dimensions = [2u32, 2, 2];
        let mut grid = VoxelGrid::new(dimensions);
        // Corner-anchored grid: voxel (i,j,k) centre at `idx + 0.5`. Fill a 2×2×1 slab
        // (z=0) with material 5; leave z=1 air.
        for k in 0..1u32 {
            for j in 0..2u32 {
                for i in 0..2u32 {
                    grid.occupied.push(Voxel {
                        local_index: [i as i32, j as i32, k as i32],
                        block_local_coord: [i as u8, j as u8, k as u8],
                        block_id: voxel_core::core_geom::BlockId(5),
                        attrs: voxel_core::core_geom::BlockAttrs::DEFAULT,
                        grid_overlay: false,
                    });
                }
            }
        }
        let region = region_from_voxel_grid(&grid, [0, 0, 0], dimensions);
        // The z=0 slab is solid material 5, z=1 is air.
        for j in 0..2 {
            for i in 0..2 {
                assert_eq!(region.cell_at(i, j, 0), Some(5));
                assert_eq!(region.cell_at(i, j, 1), None);
            }
        }
        let boxes = decompose_into_boxes(&region);
        assert_eq!(boxes.len(), 1, "2×2×1 same-material slab → one box");
        assert_eq!(boxes[0].min, [0, 0, 0]);
        assert_eq!(boxes[0].max, [1, 1, 0]);
        assert_eq!(boxes[0].label, 5);
    }

    #[test]
    fn grid_overlay_bit_blocks_box_merge() {
        // The on-face-grid flag is NOT in the per-voxel `block_id`. The CUBOID MESHER
        // composes a transient region-cell key (`block_id | overlay<<15`, via
        // `Voxel::cell_key`) from each voxel's clean `block_id` + its
        // `grid_overlay` marker, so this opaque `u16` (which `decompose_into_boxes`
        // treats representation-agnostically) splits a box across differing overlay
        // flags — without a render flag ever entering the categorical id.
        let make_voxel = |block: u16, overlay: bool| voxel_core::voxel::Voxel {
            local_index: [0, 0, 0],
            block_local_coord: [0, 0, 0],
            block_id: voxel_core::core_geom::BlockId(block),
            attrs: voxel_core::core_geom::BlockAttrs::DEFAULT,
            grid_overlay: overlay,
        };
        let base = 1u16; // Wood
        let flagged = make_voxel(base, true).cell_key().raw();
        let plain = make_voxel(base, false).cell_key().raw();
        assert_ne!(flagged, plain, "the overlay marker must change the mesher's cell key");
        // A 4×1×1 row: x<2 flagged, x>=2 plain — same base block, differing overlay.
        let extent = [4, 1, 1];
        let mut region = VoxelRegion::new_empty(extent);
        for x in 0..4u32 {
            region.set(x, 0, 0, Some(if x < 2 { flagged } else { plain }));
        }
        let boxes = decompose_into_boxes(&region);
        assert_eq!(
            boxes.len(),
            2,
            "differing overlay flag must split the row into two boxes (no merge)"
        );
        // Each box keeps its exact (overlay-bearing) cell key, so `emit_box_faces` later
        // splits it back into the clean `block_id` + the per-box overlay attribute.
        let flagged_box = boxes.iter().find(|b| b.label == flagged).unwrap();
        let plain_box = boxes.iter().find(|b| b.label == plain).unwrap();
        assert_eq!((flagged_box.min, flagged_box.max), ([0, 0, 0], [1, 0, 0]));
        assert_eq!((plain_box.min, plain_box.max), ([2, 0, 0], [3, 0, 0]));
        // A row that is UNIFORMLY flagged still merges to one box (the marker is a
        // per-box splitter, not a per-voxel one).
        let mut uniform = VoxelRegion::new_empty(extent);
        for x in 0..4u32 {
            uniform.set(x, 0, 0, Some(flagged));
        }
        assert_eq!(
            decompose_into_boxes(&uniform).len(),
            1,
            "a uniformly-flagged row merges to one box"
        );
    }

    #[test]
    fn round_trip_sdf_shapes_via_adapter() {
        // Resolve real SDF primitives (sphere, cylinder, box, torus, tube) into a
        // VoxelGrid, densify with the adapter, and verify the adapter drops no voxel
        // and the decomposition covers exactly the resolved solid set.
        use voxel_core::voxel::{ShapeKind};
        use document::voxel::{SdfShape, VoxelProducer};

        for &kind in &[
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Box,
            ShapeKind::Torus,
            ShapeKind::Tube,
        ] {
            for &size in &[[3u32, 3, 3], [5, 1, 5], [4, 2, 3]] {
                let voxels_per_block = 4;
                let shape = SdfShape::from_blocks(kind, size, 1, voxels_per_block);
                let dimensions = shape.grid_dimensions(voxels_per_block);
                let mut grid = VoxelGrid::new(dimensions);
                shape.resolve(&mut grid, voxels_per_block);
                if grid.occupied.is_empty() {
                    continue;
                }
                let region = region_from_voxel_grid(&grid, [0, 0, 0], dimensions);
                // The adapter must not drop any voxel for an origin-centred grid.
                let region_solid = region.cells.iter().filter(|c| c.is_some()).count();
                assert_eq!(
                    region_solid,
                    grid.occupied.len(),
                    "{kind:?} {size:?}: adapter dropped voxels ({region_solid} of {})",
                    grid.occupied.len()
                );
                // Decomposition covers exactly the resolved solid cells (as a set).
                let boxes = decompose_into_boxes(&region);
                let mut covered = std::collections::HashSet::new();
                for b in &boxes {
                    for z in b.min[2]..=b.max[2] {
                        for y in b.min[1]..=b.max[1] {
                            for x in b.min[0]..=b.max[0] {
                                assert!(covered.insert([x, y, z]), "overlap at ({x},{y},{z})");
                                assert_eq!(region.cell_at(x, y, z), Some(b.label));
                            }
                        }
                    }
                }
                assert_eq!(covered.len(), region_solid, "{kind:?} {size:?}: cover count mismatch");
                assert!(boxes.len() as u64 <= region_solid as u64);
            }
        }
    }

    #[test]
    fn adapter_subregion_offset() {
        // A 4×4×4 grid, fill the high-X half (i>=2) with material 8; extract just
        // the sub-box [2,0,0]..[2+2,4,4) and confirm the offset shift is correct.
        let dimensions = [4u32, 4, 4];
        let mut grid = VoxelGrid::new(dimensions);
        for k in 0..4u32 {
            for j in 0..4u32 {
                for i in 2..4u32 {
                    grid.occupied.push(Voxel {
                        local_index: [i as i32, j as i32, k as i32],
                        block_local_coord: [0, 0, 0],
                        block_id: voxel_core::core_geom::BlockId(8),
                        attrs: voxel_core::core_geom::BlockAttrs::DEFAULT,
                        grid_overlay: false,
                    });
                }
            }
        }
        let region = region_from_voxel_grid(&grid, [2, 0, 0], [2, 4, 4]);
        // Whole sub-region is solid material 8.
        let boxes = decompose_into_boxes(&region);
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].cell_count(), 2 * 4 * 4);
        assert_eq!(boxes[0].label, 8);
    }
}
