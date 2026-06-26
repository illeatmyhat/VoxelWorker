//! Greedy box (cuboid) decomposition — the pure CPU algorithm behind the
//! cuboid mesher (ADR 0002 Decision 1, part of #18 / checkpoint E3).
//!
//! Vintage Story renders a chiseled block not as one instanced cube per voxel
//! but by merging its solid voxels into a small set of **axis-aligned cuboids**,
//! each a single material (`BlockEntityMicroBlock.GenShape`). This module is the
//! decomposition step **only**: it turns a bounded region of solid, materialled
//! voxels into a minimal-ish set of [`VoxelBox`]es. No rendering, no GPU — the
//! rendering task (E3) consumes the boxes this produces.
//!
//! The core ([`decompose_into_boxes`]) is representation-agnostic: it takes a
//! dense `[w, h, d]` region of `Option<u16>` materials (`None` = air). A thin
//! adapter ([`region_from_voxel_grid`]) builds that region from a [`VoxelGrid`]
//! sub-box so the next task can call it per render-chunk.

use crate::voxel::VoxelGrid;

/// One axis-aligned, single-material cuboid covering a contiguous block of solid
/// voxels.
///
/// Coordinates are **region-local voxel indices** (the same `(x, y, z)` lattice
/// the [`VoxelRegion`] is indexed by). `min` is **inclusive** and `max` is
/// **inclusive** — a single voxel at `(2, 3, 4)` is `min == max == [2, 3, 4]`,
/// and the box spans the voxel cells `min.x..=max.x`, `min.y..=max.y`,
/// `min.z..=max.z`. The voxel **count** of a box is therefore
/// `(max.x - min.x + 1) * (max.y - min.y + 1) * (max.z - min.z + 1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoxelBox {
    /// Inclusive minimum voxel index per axis.
    pub min: [u32; 3],
    /// Inclusive maximum voxel index per axis.
    pub max: [u32; 3],
    /// The single material shared by every voxel in the box.
    pub material_id: u16,
}

impl VoxelBox {
    /// Number of voxel cells this (inclusive-inclusive) box covers.
    pub fn voxel_count(&self) -> u64 {
        let dx = (self.max[0] - self.min[0] + 1) as u64;
        let dy = (self.max[1] - self.min[1] + 1) as u64;
        let dz = (self.max[2] - self.min[2] + 1) as u64;
        dx * dy * dz
    }
}

/// A dense, bounded region of solid/air voxels with per-voxel material.
///
/// `extent` is `[w, h, d]` in voxels; `cells` is row-major with the **X axis
/// fastest**, then Y, then Z: `index(x, y, z) = (z * h + y) * w + x` — the same
/// densification order [`crate::renderer::VoxelRenderer::upload_grid`] uses for
/// its occupancy volume. `Some(material_id)` is a solid voxel, `None` is air.
///
/// This is the representation-agnostic input to [`decompose_into_boxes`]; it can
/// be built by hand (tests), from a [`VoxelGrid`] sub-box
/// ([`region_from_voxel_grid`]), or later from a chunk's resolved voxels.
#[derive(Debug, Clone)]
pub struct VoxelRegion {
    /// Region size in voxels `[w, h, d]`.
    pub extent: [u32; 3],
    /// Row-major material cells, X fastest: `(z * h + y) * w + x`.
    pub cells: Vec<Option<u16>>,
}

impl VoxelRegion {
    /// Create an all-air region of the given voxel extent.
    pub fn new_empty(extent: [u32; 3]) -> Self {
        let count = extent[0] as usize * extent[1] as usize * extent[2] as usize;
        Self {
            extent,
            cells: vec![None; count],
        }
    }

    /// Flat row-major index for `(x, y, z)` (X fastest). Caller guarantees the
    /// coordinate is in bounds.
    #[inline]
    fn flat_index(&self, x: u32, y: u32, z: u32) -> usize {
        let [w, h, _d] = self.extent;
        (z as usize * h as usize + y as usize) * w as usize + x as usize
    }

    /// Material at `(x, y, z)`, or `None` for air / out-of-bounds.
    #[inline]
    pub fn material_at(&self, x: u32, y: u32, z: u32) -> Option<u16> {
        let [w, h, d] = self.extent;
        if x >= w || y >= h || z >= d {
            return None;
        }
        self.cells[self.flat_index(x, y, z)]
    }

    /// Set the material at `(x, y, z)`. Panics if out of bounds (test helper).
    pub fn set(&mut self, x: u32, y: u32, z: u32, material_id: Option<u16>) {
        let index = self.flat_index(x, y, z);
        self.cells[index] = material_id;
    }
}

/// Greedy box decomposition: merge the solid voxels of `region` into a
/// minimal-ish set of single-material [`VoxelBox`]es (ADR 0002 Decision 1).
///
/// Algorithm — the classic 3D greedy box grow, in a fixed Z→Y→X scan order so
/// the output is **deterministic**:
///
/// 1. Scan cells in `(z, y, x)` order. For the first not-yet-consumed solid
///    cell, take its material as the seed.
/// 2. **Grow +X:** extend the run along +X while the next cell is the same
///    material and unconsumed.
/// 3. **Grow +Y:** extend the whole X-run along +Y while *every* cell of the
///    candidate row matches the material and is unconsumed.
/// 4. **Grow +Z:** extend the whole XY-slab along +Z while *every* cell of the
///    candidate slab matches the material and is unconsumed.
/// 5. Mark every covered cell consumed and emit the box. Repeat until no solid
///    cell is unconsumed.
///
/// Invariants (exercised exhaustively by the unit tests): the emitted boxes
/// **exactly cover** the solid set (no air, nothing missed), are **pairwise
/// disjoint**, are each **single-material**, and the output is **deterministic**
/// for a given input. The result is *not* guaranteed minimal in box count —
/// greedy is sufficient (and what VS uses in spirit).
pub fn decompose_into_boxes(region: &VoxelRegion) -> Vec<VoxelBox> {
    let [w, h, d] = region.extent;
    if w == 0 || h == 0 || d == 0 {
        return Vec::new();
    }

    let mut consumed = vec![false; region.cells.len()];
    let mut boxes = Vec::new();

    // Local helpers over the consumed bitmap, sharing the region's flat layout.
    let idx = |x: u32, y: u32, z: u32| (z as usize * h as usize + y as usize) * w as usize + x as usize;

    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let seed_index = idx(x, y, z);
                if consumed[seed_index] {
                    continue;
                }
                let material = match region.cells[seed_index] {
                    Some(material) => material,
                    None => continue, // air
                };

                // --- Grow +X: longest same-material unconsumed run from x. ---
                let mut max_x = x;
                while max_x + 1 < w {
                    let next = idx(max_x + 1, y, z);
                    if consumed[next] || region.cells[next] != Some(material) {
                        break;
                    }
                    max_x += 1;
                }

                // --- Grow +Y: extend the whole [x..=max_x] row along +Y. ---
                let mut max_y = y;
                'grow_y: while max_y + 1 < h {
                    let candidate_y = max_y + 1;
                    for cx in x..=max_x {
                        let cell = idx(cx, candidate_y, z);
                        if consumed[cell] || region.cells[cell] != Some(material) {
                            break 'grow_y;
                        }
                    }
                    max_y = candidate_y;
                }

                // --- Grow +Z: extend the whole [x..=max_x]×[y..=max_y] slab. ---
                let mut max_z = z;
                'grow_z: while max_z + 1 < d {
                    let candidate_z = max_z + 1;
                    for cy in y..=max_y {
                        for cx in x..=max_x {
                            let cell = idx(cx, cy, candidate_z);
                            if consumed[cell] || region.cells[cell] != Some(material) {
                                break 'grow_z;
                            }
                        }
                    }
                    max_z = candidate_z;
                }

                // --- Consume the whole grown box and emit it. ---
                for cz in z..=max_z {
                    for cy in y..=max_y {
                        for cx in x..=max_x {
                            consumed[idx(cx, cy, cz)] = true;
                        }
                    }
                }
                boxes.push(VoxelBox {
                    min: [x, y, z],
                    max: [max_x, max_y, max_z],
                    material_id: material,
                });
            }
        }
    }

    boxes
}

/// Build a dense [`VoxelRegion`] from an axis-aligned sub-box of a [`VoxelGrid`]
/// (the per-chunk adapter the rendering task — E3 — will call).
///
/// `origin` is the region's minimum voxel index in grid space; `extent` is its
/// size in voxels. A grid voxel maps to its index with the project-wide rule
/// `i = round(world_position + dimensions/2 - 0.5)` (see
/// `VoxelGrid::widest_run_in_band` / `renderer::upload_grid`). Voxels whose index
/// falls inside `[origin, origin + extent)` are copied into the region with their
/// `material_id`; everything else stays air. Out-of-grid origins simply yield air
/// for the uncovered cells.
///
/// Passing `origin = [0, 0, 0]` and `extent = grid.dimensions` decomposes the
/// whole grid in one call.
pub fn region_from_voxel_grid(
    grid: &VoxelGrid,
    origin: [u32; 3],
    extent: [u32; 3],
) -> VoxelRegion {
    let mut region = VoxelRegion::new_empty(extent);
    let [grid_x, grid_y, grid_z] = grid.dimensions;
    if grid_x == 0 || grid_y == 0 || grid_z == 0 {
        return region;
    }
    let half_x = grid_x as f32 / 2.0;
    let half_y = grid_y as f32 / 2.0;
    let half_z = grid_z as f32 / 2.0;

    for voxel in &grid.occupied {
        let i = (voxel.world_position[0] + half_x - 0.5).round() as i64;
        let j = (voxel.world_position[1] + half_y - 0.5).round() as i64;
        let k = (voxel.world_position[2] + half_z - 0.5).round() as i64;
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
        region.set(lx as u32, ly as u32, lz as u32, Some(voxel.material_id));
    }
    region
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::Voxel;

    /// Build a region from a closure `(x, y, z) -> Option<u16>` over `extent`.
    fn region_from_fn<F: Fn(u32, u32, u32) -> Option<u16>>(
        extent: [u32; 3],
        f: F,
    ) -> VoxelRegion {
        let mut region = VoxelRegion::new_empty(extent);
        for z in 0..extent[2] {
            for y in 0..extent[1] {
                for x in 0..extent[0] {
                    region.set(x, y, z, f(x, y, z));
                }
            }
        }
        region
    }

    /// Assert the three core invariants of a decomposition against its region:
    /// exact cover, no overlap, single material. Returns the box count.
    fn assert_invariants(region: &VoxelRegion, boxes: &[VoxelBox]) -> usize {
        let [w, h, d] = region.extent;
        // Per-cell coverage count + which material covered it.
        let mut cover_count = vec![0u32; region.cells.len()];
        let idx = |x: u32, y: u32, z: u32| {
            (z as usize * h as usize + y as usize) * w as usize + x as usize
        };

        for b in boxes {
            // Box stays inside the region.
            assert!(
                b.max[0] < w && b.max[1] < h && b.max[2] < d,
                "box {b:?} out of region extent {:?}",
                region.extent
            );
            assert!(
                b.min[0] <= b.max[0] && b.min[1] <= b.max[1] && b.min[2] <= b.max[2],
                "box {b:?} has min > max"
            );
            for z in b.min[2]..=b.max[2] {
                for y in b.min[1]..=b.max[1] {
                    for x in b.min[0]..=b.max[0] {
                        // Single material: every covered cell IS this material.
                        assert_eq!(
                            region.material_at(x, y, z),
                            Some(b.material_id),
                            "box {b:?} covers cell ({x},{y},{z}) with wrong/absent material"
                        );
                        cover_count[idx(x, y, z)] += 1;
                    }
                }
            }
        }

        // Exact cover + no overlap: every solid cell covered exactly once, every
        // air cell covered zero times.
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let solid = region.material_at(x, y, z).is_some();
                    let covered = cover_count[idx(x, y, z)];
                    if solid {
                        assert_eq!(
                            covered, 1,
                            "solid cell ({x},{y},{z}) covered {covered} times (want exactly 1)"
                        );
                    } else {
                        assert_eq!(
                            covered, 0,
                            "air cell ({x},{y},{z}) covered {covered} times (want 0)"
                        );
                    }
                }
            }
        }
        boxes.len()
    }

    #[test]
    fn single_voxel_one_box() {
        let region = region_from_fn([3, 3, 3], |x, y, z| {
            if [x, y, z] == [1, 1, 1] {
                Some(7)
            } else {
                None
            }
        });
        let boxes = decompose_into_boxes(&region);
        assert_eq!(boxes.len(), 1);
        assert_eq!(
            boxes[0],
            VoxelBox {
                min: [1, 1, 1],
                max: [1, 1, 1],
                material_id: 7,
            }
        );
        assert_eq!(boxes[0].voxel_count(), 1);
        assert_invariants(&region, &boxes);
    }

    #[test]
    fn full_block_single_box() {
        // A solid 4×3×5 block of one material collapses to ONE box, not 60.
        let extent = [4, 3, 5];
        let region = region_from_fn(extent, |_x, _y, _z| Some(2));
        let boxes = decompose_into_boxes(&region);
        assert_eq!(boxes.len(), 1, "solid block must be a single box");
        assert_eq!(boxes[0].min, [0, 0, 0]);
        assert_eq!(boxes[0].max, [3, 2, 4]);
        assert_eq!(boxes[0].material_id, 2);
        assert_eq!(
            boxes[0].voxel_count(),
            (extent[0] * extent[1] * extent[2]) as u64
        );
        assert_invariants(&region, &boxes);
    }

    #[test]
    fn two_material_split() {
        // 4×2×2 block split along X: x<2 material 1, x>=2 material 9.
        let extent = [4, 2, 2];
        let region = region_from_fn(extent, |x, _y, _z| {
            if x < 2 {
                Some(1)
            } else {
                Some(9)
            }
        });
        let boxes = decompose_into_boxes(&region);
        assert_eq!(boxes.len(), 2, "two materials → two boxes");
        assert_invariants(&region, &boxes);

        // Find each material's box and check its extent.
        let box1 = boxes.iter().find(|b| b.material_id == 1).unwrap();
        let box9 = boxes.iter().find(|b| b.material_id == 9).unwrap();
        assert_eq!((box1.min, box1.max), ([0, 0, 0], [1, 1, 1]));
        assert_eq!((box9.min, box9.max), ([2, 0, 0], [3, 1, 1]));
    }

    #[test]
    fn l_shape_concavity() {
        // 3×3 L in the z=0 plane (and z=1), exercising a concave outline:
        //   y=2: X . .
        //   y=1: X . .
        //   y=0: X X X
        let extent = [3, 3, 2];
        let region = region_from_fn(extent, |x, y, _z| {
            if y == 0 || x == 0 {
                Some(4)
            } else {
                None
            }
        });
        let boxes = decompose_into_boxes(&region);
        // No box may cover the concave air cells; invariants enforce it.
        let count = assert_invariants(&region, &boxes);
        // Sanity: fewer boxes than solid cells (some merging happened).
        let solid: usize = region.cells.iter().filter(|c| c.is_some()).count();
        assert!(count < solid, "L-shape should merge SOME cells");
    }

    #[test]
    fn ring_hole() {
        // A 5×5 ring (border solid, centre hollow) over depth 1 — a hole that no
        // box may cover.
        let extent = [5, 5, 1];
        let region = region_from_fn(extent, |x, y, _z| {
            if x == 0 || y == 0 || x == 4 || y == 4 {
                Some(3)
            } else {
                None
            }
        });
        let boxes = decompose_into_boxes(&region);
        assert_invariants(&region, &boxes);
        // The centre 3×3 must be air, hence uncovered (invariants already check,
        // this is an explicit guard on the hole).
        for b in &boxes {
            for x in 1..=3 {
                for y in 1..=3 {
                    let inside_x = b.min[0] <= x && x <= b.max[0];
                    let inside_y = b.min[1] <= y && y <= b.max[1];
                    assert!(
                        !(inside_x && inside_y),
                        "box {b:?} covers hole cell ({x},{y})"
                    );
                }
            }
        }
    }

    #[test]
    fn empty_region_no_boxes() {
        let region = VoxelRegion::new_empty([4, 4, 4]);
        assert!(decompose_into_boxes(&region).is_empty());
        // Zero-extent region too.
        let zero = VoxelRegion::new_empty([0, 5, 5]);
        assert!(decompose_into_boxes(&zero).is_empty());
    }

    #[test]
    fn determinism_same_input_same_output() {
        let extent = [6, 6, 6];
        let region = region_from_fn(extent, |x, y, z| {
            if (x + y + z) % 2 == 0 {
                Some((x % 3) as u16)
            } else {
                None
            }
        });
        let first = decompose_into_boxes(&region);
        let second = decompose_into_boxes(&region);
        assert_eq!(first, second, "decomposition must be deterministic");
    }

    #[test]
    fn greedy_beats_naive_box_count() {
        // The headline sanity: a solid 4×4×4 single-material block is 1 box, not
        // 64 (the per-voxel-cube count it replaces).
        let region = region_from_fn([4, 4, 4], |_x, _y, _z| Some(0));
        let boxes = decompose_into_boxes(&region);
        assert_eq!(boxes.len(), 1);
        assert!(boxes.len() < 4 * 4 * 4);
    }

    /// Deterministic LCG so the randomized test needs no `rand` crate.
    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            // Numerical Recipes LCG constants.
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
    }

    #[test]
    fn randomized_invariants_safety_net() {
        // Several pseudo-random material patterns over varied extents and
        // material counts; assert the three invariants every time. This is the
        // real safety net for the greedy growth logic.
        let mut lcg = Lcg(0x1234_5678_9abc_def0);
        let extents = [
            [1, 1, 1],
            [5, 5, 5],
            [8, 3, 6],
            [4, 9, 2],
            [10, 10, 1],
            [1, 12, 7],
            [7, 7, 7],
        ];
        for &extent in &extents {
            for materials in [1u32, 2, 3, 5] {
                for fill_percent in [10u32, 35, 65, 90, 100] {
                    let mut region = VoxelRegion::new_empty(extent);
                    for cell in region.cells.iter_mut() {
                        let solid = (lcg.next_u32() % 100) < fill_percent;
                        if solid {
                            *cell = Some((lcg.next_u32() % materials) as u16);
                        }
                    }
                    let boxes = decompose_into_boxes(&region);
                    assert_invariants(&region, &boxes);
                    // Box count never exceeds solid-cell count.
                    let solid = region.cells.iter().filter(|c| c.is_some()).count();
                    assert!(boxes.len() <= solid.max(0));
                }
            }
        }
    }

    #[test]
    fn adapter_from_voxel_grid_whole_grid() {
        // Build a tiny VoxelGrid by hand and decompose the whole thing via the
        // adapter, confirming the world_position → index mapping round-trips.
        let dimensions = [2u32, 2, 2];
        let half = [1.0f32, 1.0, 1.0]; // dims/2
        let mut grid = VoxelGrid::new(dimensions);
        // Fill a 2×2×1 slab (z=0) with material 5; leave z=1 air.
        for k in 0..1u32 {
            for j in 0..2u32 {
                for i in 0..2u32 {
                    grid.occupied.push(Voxel {
                        world_position: [
                            i as f32 + 0.5 - half[0],
                            j as f32 + 0.5 - half[1],
                            k as f32 + 0.5 - half[2],
                        ],
                        block_local_coord: [i as u8, j as u8, k as u8],
                        material_id: 5,
                    });
                }
            }
        }
        let region = region_from_voxel_grid(&grid, [0, 0, 0], dimensions);
        // The z=0 slab is solid material 5, z=1 is air.
        for j in 0..2 {
            for i in 0..2 {
                assert_eq!(region.material_at(i, j, 0), Some(5));
                assert_eq!(region.material_at(i, j, 1), None);
            }
        }
        let boxes = decompose_into_boxes(&region);
        assert_eq!(boxes.len(), 1, "2×2×1 same-material slab → one box");
        assert_eq!(boxes[0].min, [0, 0, 0]);
        assert_eq!(boxes[0].max, [1, 1, 0]);
        assert_eq!(boxes[0].material_id, 5);
        assert_invariants(&region, &boxes);
    }

    /// Expand a decomposition back into the SET of `(x, y, z)` cells it covers,
    /// paired with the covering box's `material_id`. The structural round-trip
    /// tests compare this against the region's own solid cells.
    fn expand_boxes_to_cells(boxes: &[VoxelBox]) -> std::collections::HashMap<[u32; 3], u16> {
        let mut cells = std::collections::HashMap::new();
        for voxel_box in boxes {
            for z in voxel_box.min[2]..=voxel_box.max[2] {
                for y in voxel_box.min[1]..=voxel_box.max[1] {
                    for x in voxel_box.min[0]..=voxel_box.max[0] {
                        let previous = cells.insert([x, y, z], voxel_box.material_id);
                        assert!(
                            previous.is_none(),
                            "cell ({x},{y},{z}) covered by more than one box (overlap)"
                        );
                    }
                }
            }
        }
        cells
    }

    /// Collect a region's solid cells as a `(x, y, z) -> material_id` map.
    fn region_solid_cells(region: &VoxelRegion) -> std::collections::HashMap<[u32; 3], u16> {
        let [w, h, d] = region.extent;
        let mut cells = std::collections::HashMap::new();
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    if let Some(material) = region.material_at(x, y, z) {
                        cells.insert([x, y, z], material);
                    }
                }
            }
        }
        cells
    }

    /// The full structural round-trip: decompose, expand the boxes back to cells,
    /// and assert the expanded cell+material map EXACTLY equals the region's solid
    /// cells. This subsumes exact-cover, no-overlap (checked in `expand_boxes_to_cells`),
    /// and per-cell material correctness (no merging across differing materials) in
    /// one set-equality assertion. Returns the box count for sanity follow-ups.
    fn assert_round_trip_exact(region: &VoxelRegion) -> usize {
        let boxes = decompose_into_boxes(region);
        let expanded = expand_boxes_to_cells(&boxes);
        let original = region_solid_cells(region);
        assert_eq!(
            expanded, original,
            "expanded box cells (with materials) must exactly equal the region's solid cells"
        );
        // Belt-and-braces against the existing per-axis invariant checker.
        assert_invariants(region, &boxes);
        boxes.len()
    }

    #[test]
    fn round_trip_single_voxel() {
        let region = region_from_fn([3, 3, 3], |x, y, z| {
            if [x, y, z] == [1, 1, 1] {
                Some(7)
            } else {
                None
            }
        });
        assert_eq!(assert_round_trip_exact(&region), 1);
    }

    #[test]
    fn round_trip_empty_and_full() {
        let empty = VoxelRegion::new_empty([4, 4, 4]);
        assert_eq!(assert_round_trip_exact(&empty), 0);

        let full = region_from_fn([4, 3, 5], |_x, _y, _z| Some(2));
        assert_eq!(assert_round_trip_exact(&full), 1, "solid block → one box");
    }

    #[test]
    fn grid_overlay_bit_blocks_box_merge() {
        // Issue #29 S4: the on-face-grid flag bit is folded into `material_id` BEFORE
        // decomposition, so two otherwise-identical-material voxels that differ only
        // in the flag bit are DIFFERENT materials to the greedy mesher and must NOT
        // merge — the flagged voxels carry the grid onto their own faces, the
        // unflagged ones don't.
        use crate::voxel::GRID_OVERLAY_BIT;
        let base = 1u16; // Wood
        let flagged = base | GRID_OVERLAY_BIT;
        // A 4×1×1 row: x<2 flagged, x>=2 plain — same base material, differing bit.
        let extent = [4, 1, 1];
        let region = region_from_fn(extent, |x, _y, _z| {
            Some(if x < 2 { flagged } else { base })
        });
        let boxes = decompose_into_boxes(&region);
        assert_eq!(
            boxes.len(),
            2,
            "differing grid bit must split the row into two boxes (no merge)"
        );
        assert_invariants(&region, &boxes);
        // Each box keeps its exact (bit-bearing) material id, so the bit travels onto
        // the box's faces — masked off only at colour-lookup time in the shader.
        let flagged_box = boxes.iter().find(|b| b.material_id == flagged).unwrap();
        let plain_box = boxes.iter().find(|b| b.material_id == base).unwrap();
        assert_eq!((flagged_box.min, flagged_box.max), ([0, 0, 0], [1, 0, 0]));
        assert_eq!((plain_box.min, plain_box.max), ([2, 0, 0], [3, 0, 0]));
        // A row that is UNIFORMLY flagged still merges to one box (the bit is not a
        // per-voxel splitter, only a per-material one).
        let uniform = region_from_fn(extent, |_x, _y, _z| Some(flagged));
        assert_eq!(
            decompose_into_boxes(&uniform).len(),
            1,
            "a uniformly-flagged row merges to one box"
        );
    }

    #[test]
    fn round_trip_multiple_materials() {
        // A 4×4×2 region quartered into four distinct materials along X and Y, so a
        // correct decomposition can NEVER merge two materials into one box. The
        // round-trip's per-cell material equality is the guard.
        let extent = [4, 4, 2];
        let region = region_from_fn(extent, |x, y, _z| {
            let material = match (x < 2, y < 2) {
                (true, true) => 11,
                (false, true) => 22,
                (true, false) => 33,
                (false, false) => 44,
            };
            Some(material)
        });
        let count = assert_round_trip_exact(&region);
        // Four material quadrants → at least four boxes (greedy may not exceed the
        // solid-cell count, but must distinguish all four materials).
        assert!(count >= 4, "four materials must yield at least four boxes, got {count}");
        let materials: std::collections::HashSet<u16> =
            decompose_into_boxes(&region).iter().map(|b| b.material_id).collect();
        assert_eq!(
            materials,
            [11, 22, 33, 44].into_iter().collect(),
            "every material must survive into the boxes"
        );
    }

    #[test]
    fn round_trip_handmade_concave_shapes() {
        // A handful of irregular/concave outlines whose holes and notches no box may
        // cover. The round-trip set-equality enforces that exactly.

        // (1) L-shape across depth 2.
        let l_shape = region_from_fn([3, 3, 2], |x, y, _z| {
            if y == 0 || x == 0 {
                Some(4)
            } else {
                None
            }
        });
        assert_round_trip_exact(&l_shape);

        // (2) 5×5 ring (hollow centre) over depth 1.
        let ring = region_from_fn([5, 5, 1], |x, y, _z| {
            if x == 0 || y == 0 || x == 4 || y == 4 {
                Some(3)
            } else {
                None
            }
        });
        assert_round_trip_exact(&ring);

        // (3) A "plus"/cross silhouette with a concave notch at each corner, two
        //     materials in the arms, extruded over depth 3.
        let cross = region_from_fn([5, 5, 3], |x, y, _z| {
            let on_vertical_arm = x == 2;
            let on_horizontal_arm = y == 2;
            if on_vertical_arm || on_horizontal_arm {
                Some(if on_vertical_arm { 8 } else { 9 })
            } else {
                None
            }
        });
        assert_round_trip_exact(&cross);

        // (4) A diagonal staircase: a concave shape with single-cell steps.
        let staircase = region_from_fn([4, 4, 1], |x, y, _z| {
            if y <= x {
                Some(1)
            } else {
                None
            }
        });
        assert_round_trip_exact(&staircase);
    }

    #[test]
    fn round_trip_sdf_shapes_via_adapter() {
        // Resolve real SDF primitives (sphere, cylinder, box, torus, tube) into a
        // VoxelGrid, densify with the adapter, and round-trip-verify. These grids
        // are origin-centred (SdfShape::resolve), so `region_from_voxel_grid`'s
        // anchor-at-0 convention maps every voxel into bounds losslessly; the test
        // first asserts no voxel was dropped, THEN the structural round-trip.
        use crate::voxel::{SdfShape, ShapeKind, VoxelProducer};

        for &kind in &[
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Box,
            ShapeKind::Torus,
            ShapeKind::Tube,
        ] {
            for &size in &[[3u32, 3, 3], [5, 1, 5], [4, 2, 3]] {
                let shape = SdfShape {
                    kind,
                    size_blocks: size,
                    voxels_per_block: 4,
                    wall_blocks: 1,
                };
                let dimensions = shape.grid_dimensions();
                let mut grid = VoxelGrid::new(dimensions);
                shape.resolve(&mut grid);
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
                // Full structural round-trip: boxes expand back to exactly the
                // region's solid cells, with materials, no overlap, no holes.
                let count = assert_round_trip_exact(&region);
                assert!(
                    count as u64 <= region_solid as u64,
                    "{kind:?} {size:?}: box count {count} exceeds solid cells {region_solid}"
                );
            }
        }
    }

    #[test]
    fn round_trip_randomized_multi_material() {
        // Pseudo-random multi-material fills over varied extents: the structural
        // round-trip (set + material equality, no overlap) on every sample, the
        // real safety net for the greedy growth + material split logic.
        let mut lcg = Lcg(0xfeed_face_dead_beef);
        let extents = [[1, 1, 1], [6, 4, 5], [9, 2, 7], [3, 8, 4], [7, 7, 7]];
        for &extent in &extents {
            for materials in [1u32, 2, 4] {
                for fill_percent in [25u32, 55, 100] {
                    let mut region = VoxelRegion::new_empty(extent);
                    for cell in region.cells.iter_mut() {
                        if (lcg.next_u32() % 100) < fill_percent {
                            *cell = Some((lcg.next_u32() % materials) as u16);
                        }
                    }
                    assert_round_trip_exact(&region);
                }
            }
        }
    }

    #[test]
    fn adapter_subregion_offset() {
        // A 4×4×4 grid, fill the high-X half (i>=2) with material 8; extract just
        // the sub-box [2,0,0]..[2+2,4,4) and confirm the offset shift is correct.
        let dimensions = [4u32, 4, 4];
        let half = 2.0f32;
        let mut grid = VoxelGrid::new(dimensions);
        for k in 0..4u32 {
            for j in 0..4u32 {
                for i in 2..4u32 {
                    grid.occupied.push(Voxel {
                        world_position: [
                            i as f32 + 0.5 - half,
                            j as f32 + 0.5 - half,
                            k as f32 + 0.5 - half,
                        ],
                        block_local_coord: [0, 0, 0],
                        material_id: 8,
                    });
                }
            }
        }
        let region = region_from_voxel_grid(&grid, [2, 0, 0], [2, 4, 4]);
        // Whole sub-region is solid material 8.
        let boxes = decompose_into_boxes(&region);
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].voxel_count(), 2 * 4 * 4);
        assert_eq!(boxes[0].material_id, 8);
        assert_invariants(&region, &boxes);
    }
}
