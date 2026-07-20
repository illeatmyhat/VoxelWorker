//! Greedy decomposition of a labeled dense grid into axis-aligned cuboids.
//!
//! [`GreedyCuboidDecomposition`] covers the labeled cells of a dense 3D
//! [`CellGrid`] with a small set of axis-aligned [`Cuboid`]s, each holding a
//! **single label**. It is the 3D lift of **greedy meshing**: sweep the grid in a
//! fixed order and, at each not-yet-consumed labeled cell, grow a box as far as it
//! will go along one axis, then the whole run along the next, then the whole slab
//! along the third — consuming every cell it swallows so nothing is emitted twice.
//!
//! Concretely the box grows in three nested stages from a seed cell:
//!
//! 1. **Grow +X:** extend a run along +X while the next cell carries the same label
//!    and is unconsumed.
//! 2. **Grow +Y:** extend the whole X-run along +Y while *every* cell of the
//!    candidate row matches the label and is unconsumed.
//! 3. **Grow +Z:** extend the whole XY-slab along +Z while *every* cell of the
//!    candidate slab matches the label and is unconsumed.
//!
//! The seed sweep is a fixed `Z→Y→X` scan, so the output is a deterministic
//! function of the input.
//!
//! ## Invariants (exercised exhaustively by the in-file tests)
//!
//! The emitted cuboids **exactly cover** the labeled set (no unlabeled cell
//! covered, no labeled cell missed), are **pairwise disjoint** (each cell covered
//! at most once), and are each **single-label**. For a fixed input the output is
//! **deterministic**.
//!
//! ## Why greedy, not minimal
//!
//! The result is *not* guaranteed minimal in cuboid count. That is a deliberate,
//! informed trade: partitioning a rectilinear region into the **fewest** axis-aligned
//! boxes is NP-hard (Soltan & Gorpinevich 1993, *Minimum dissection of a rectilinear
//! polygon with arbitrary holes into rectangles*), so an exact optimum is not
//! affordable per invocation. Greedy growth is `O(cells)` amortized, deterministic,
//! and empirically collapses a solid `n³` block to a single box — enough of the win
//! for a fraction of the cost.
//!
//! Cite: the greedy-meshing lineage — Lysenko, *Meshing in a Minecraft Game* (0fps,
//! 2012) — merges same-attribute voxels into maximal quads/boxes by exactly this
//! grow-and-consume sweep; the 3D box-cover reading is its natural extension.
//! Hardness of the minimal cover: Soltan & Gorpinevich 1993 (rectilinear box cover
//! is NP-hard). Deviation: a fixed `Z→Y→X` seed order (for determinism, not a cost
//! heuristic) and a plain `Vec<bool>` consumed mask over the grid's flat layout
//! (not a bitset) — the grids are per-boundary-block-sized, so the simple mask wins.

/// One axis-aligned integer cuboid carrying a single label, over the cell lattice
/// of a [`CellGrid`].
///
/// `min` is **inclusive** and `max` is **inclusive**: a single cell at `(2, 3, 4)`
/// is `min == max == [2, 3, 4]`, and the cuboid spans the cells `min.x..=max.x`,
/// `min.y..=max.y`, `min.z..=max.z`. Its cell **count** is therefore
/// `(max.x − min.x + 1) · (max.y − min.y + 1) · (max.z − min.z + 1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cuboid<T> {
    /// Inclusive minimum cell index per axis.
    pub min: [u32; 3],
    /// Inclusive maximum cell index per axis.
    pub max: [u32; 3],
    /// The single label shared by every cell in the cuboid.
    pub label: T,
}

impl<T> Cuboid<T> {
    /// Number of cells this (inclusive–inclusive) cuboid covers.
    pub fn cell_count(&self) -> u64 {
        let dx = (self.max[0] - self.min[0] + 1) as u64;
        let dy = (self.max[1] - self.min[1] + 1) as u64;
        let dz = (self.max[2] - self.min[2] + 1) as u64;
        dx * dy * dz
    }

    /// Whether `cell` lies inside this cuboid — the POINT question, as against
    /// [`cell_count`](Self::cell_count)'s bulk one.
    ///
    /// Both bounds are inclusive, so this is `min[axis] <= cell[axis] <= max[axis]` on every
    /// axis and nothing more. It exists because a decomposition is a perfectly good spatial
    /// index for asking "is this one cell solid" — answering that by expanding the cuboids
    /// back to cells would cost the volume to learn one bit, which is exactly what the
    /// decomposition was chosen to avoid.
    pub fn contains(&self, cell: [u32; 3]) -> bool {
        (0..3).all(|axis| cell[axis] >= self.min[axis] && cell[axis] <= self.max[axis])
    }
}

/// A dense, bounded 3D grid of optionally-labeled cells — `Some(label)` is a
/// labeled cell, `None` is empty.
///
/// `extent` is `[w, h, d]` in cells; `cells` is row-major with the **X axis
/// fastest**, then Y, then Z: `index(x, y, z) = (z · h + y) · w + x`. This is the
/// input to [`GreedyCuboidDecomposition::decompose`]; build it by hand or from any
/// domain adapter at the crate seam.
#[derive(Debug, Clone)]
pub struct CellGrid<T> {
    /// Grid size in cells `[w, h, d]`.
    pub extent: [u32; 3],
    /// Row-major label cells, X fastest: `(z · h + y) · w + x`.
    pub cells: Vec<Option<T>>,
}

impl<T: Copy> CellGrid<T> {
    /// Create an all-empty grid of the given extent.
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

    /// Label at `(x, y, z)`, or `None` for empty / out-of-bounds.
    #[inline]
    pub fn cell_at(&self, x: u32, y: u32, z: u32) -> Option<T> {
        let [w, h, d] = self.extent;
        if x >= w || y >= h || z >= d {
            return None;
        }
        self.cells[self.flat_index(x, y, z)]
    }

    /// Set the label at `(x, y, z)`. Panics if out of bounds.
    pub fn set(&mut self, x: u32, y: u32, z: u32, label: Option<T>) {
        let index = self.flat_index(x, y, z);
        self.cells[index] = label;
    }
}

/// The greedy 3D box-growing decomposition — see the module documentation for the
/// algorithm, invariants, and the NP-hardness argument for choosing greedy over
/// minimal.
pub struct GreedyCuboidDecomposition;

impl GreedyCuboidDecomposition {
    /// Cover the labeled cells of `grid` with a set of single-label [`Cuboid`]s by
    /// greedy X-run → Y-slab → Z-slab growth in a deterministic `Z→Y→X` seed scan.
    ///
    /// Guarantees (see the module doc): exact cover of the labeled set, pairwise
    /// disjoint cuboids, each single-label, deterministic output. Not guaranteed
    /// minimal in cuboid count.
    pub fn decompose<T: Copy + Eq>(grid: &CellGrid<T>) -> Vec<Cuboid<T>> {
        let [w, h, d] = grid.extent;
        if w == 0 || h == 0 || d == 0 {
            return Vec::new();
        }

        let mut consumed = vec![false; grid.cells.len()];
        let mut cuboids = Vec::new();

        // Local index helper over the consumed bitmap, sharing the grid's flat layout.
        let idx =
            |x: u32, y: u32, z: u32| (z as usize * h as usize + y as usize) * w as usize + x as usize;

        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let seed_index = idx(x, y, z);
                    if consumed[seed_index] {
                        continue;
                    }
                    let label = match grid.cells[seed_index] {
                        Some(label) => label,
                        None => continue, // empty
                    };

                    // --- Grow +X: longest same-label unconsumed run from x. ---
                    let mut max_x = x;
                    while max_x + 1 < w {
                        let next = idx(max_x + 1, y, z);
                        if consumed[next] || grid.cells[next] != Some(label) {
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
                            if consumed[cell] || grid.cells[cell] != Some(label) {
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
                                if consumed[cell] || grid.cells[cell] != Some(label) {
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
                    cuboids.push(Cuboid {
                        min: [x, y, z],
                        max: [max_x, max_y, max_z],
                        label,
                    });
                }
            }
        }

        cuboids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a grid from a closure `(x, y, z) -> Option<u16>` over `extent`.
    fn grid_from_fn<F: Fn(u32, u32, u32) -> Option<u16>>(extent: [u32; 3], f: F) -> CellGrid<u16> {
        let mut grid = CellGrid::new_empty(extent);
        for z in 0..extent[2] {
            for y in 0..extent[1] {
                for x in 0..extent[0] {
                    grid.set(x, y, z, f(x, y, z));
                }
            }
        }
        grid
    }

    /// Assert the three core invariants of a decomposition against its grid:
    /// exact cover, no overlap, single label. Returns the cuboid count.
    fn assert_invariants(grid: &CellGrid<u16>, cuboids: &[Cuboid<u16>]) -> usize {
        let [w, h, d] = grid.extent;
        // Per-cell coverage count.
        let mut cover_count = vec![0u32; grid.cells.len()];
        let idx = |x: u32, y: u32, z: u32| (z as usize * h as usize + y as usize) * w as usize + x as usize;

        for c in cuboids {
            // Cuboid stays inside the grid.
            assert!(
                c.max[0] < w && c.max[1] < h && c.max[2] < d,
                "cuboid {c:?} out of grid extent {:?}",
                grid.extent
            );
            assert!(
                c.min[0] <= c.max[0] && c.min[1] <= c.max[1] && c.min[2] <= c.max[2],
                "cuboid {c:?} has min > max"
            );
            for z in c.min[2]..=c.max[2] {
                for y in c.min[1]..=c.max[1] {
                    for x in c.min[0]..=c.max[0] {
                        // Single label: every covered cell IS this label.
                        assert_eq!(
                            grid.cell_at(x, y, z),
                            Some(c.label),
                            "cuboid {c:?} covers cell ({x},{y},{z}) with wrong/absent label"
                        );
                        cover_count[idx(x, y, z)] += 1;
                    }
                }
            }
        }

        // Exact cover + no overlap: every labeled cell covered exactly once, every
        // empty cell covered zero times.
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let labeled = grid.cell_at(x, y, z).is_some();
                    let covered = cover_count[idx(x, y, z)];
                    if labeled {
                        assert_eq!(
                            covered, 1,
                            "labeled cell ({x},{y},{z}) covered {covered} times (want exactly 1)"
                        );
                    } else {
                        assert_eq!(
                            covered, 0,
                            "empty cell ({x},{y},{z}) covered {covered} times (want 0)"
                        );
                    }
                }
            }
        }
        cuboids.len()
    }

    #[test]
    fn single_cell_one_cuboid() {
        let grid = grid_from_fn([3, 3, 3], |x, y, z| if [x, y, z] == [1, 1, 1] { Some(7) } else { None });
        let cuboids = GreedyCuboidDecomposition::decompose(&grid);
        assert_eq!(cuboids.len(), 1);
        assert_eq!(
            cuboids[0],
            Cuboid {
                min: [1, 1, 1],
                max: [1, 1, 1],
                label: 7,
            }
        );
        assert_eq!(cuboids[0].cell_count(), 1);
        assert_invariants(&grid, &cuboids);
    }

    #[test]
    fn full_block_single_cuboid() {
        // A solid 4×3×5 block of one label collapses to ONE cuboid, not 60.
        let extent = [4, 3, 5];
        let grid = grid_from_fn(extent, |_x, _y, _z| Some(2));
        let cuboids = GreedyCuboidDecomposition::decompose(&grid);
        assert_eq!(cuboids.len(), 1, "solid block must be a single cuboid");
        assert_eq!(cuboids[0].min, [0, 0, 0]);
        assert_eq!(cuboids[0].max, [3, 2, 4]);
        assert_eq!(cuboids[0].label, 2);
        assert_eq!(cuboids[0].cell_count(), (extent[0] * extent[1] * extent[2]) as u64);
        assert_invariants(&grid, &cuboids);
    }

    #[test]
    fn two_label_split() {
        // 4×2×2 block split along X: x<2 label 1, x>=2 label 9.
        let extent = [4, 2, 2];
        let grid = grid_from_fn(extent, |x, _y, _z| if x < 2 { Some(1) } else { Some(9) });
        let cuboids = GreedyCuboidDecomposition::decompose(&grid);
        assert_eq!(cuboids.len(), 2, "two labels → two cuboids");
        assert_invariants(&grid, &cuboids);

        let c1 = cuboids.iter().find(|c| c.label == 1).unwrap();
        let c9 = cuboids.iter().find(|c| c.label == 9).unwrap();
        assert_eq!((c1.min, c1.max), ([0, 0, 0], [1, 1, 1]));
        assert_eq!((c9.min, c9.max), ([2, 0, 0], [3, 1, 1]));
    }

    #[test]
    fn l_shape_concavity() {
        // 3×3 L in the z=0 plane (and z=1), exercising a concave outline:
        //   y=2: X . .
        //   y=1: X . .
        //   y=0: X X X
        let extent = [3, 3, 2];
        let grid = grid_from_fn(extent, |x, y, _z| if y == 0 || x == 0 { Some(4) } else { None });
        let cuboids = GreedyCuboidDecomposition::decompose(&grid);
        // No cuboid may cover the concave empty cells; invariants enforce it.
        let count = assert_invariants(&grid, &cuboids);
        let labeled: usize = grid.cells.iter().filter(|c| c.is_some()).count();
        assert!(count < labeled, "L-shape should merge SOME cells");
    }

    #[test]
    fn ring_hole() {
        // A 5×5 ring (border labeled, centre hollow) over depth 1 — a hole that no
        // cuboid may cover.
        let extent = [5, 5, 1];
        let grid = grid_from_fn(extent, |x, y, _z| {
            if x == 0 || y == 0 || x == 4 || y == 4 {
                Some(3)
            } else {
                None
            }
        });
        let cuboids = GreedyCuboidDecomposition::decompose(&grid);
        assert_invariants(&grid, &cuboids);
        // The centre 3×3 must be empty, hence uncovered.
        for c in &cuboids {
            for x in 1..=3 {
                for y in 1..=3 {
                    let inside_x = c.min[0] <= x && x <= c.max[0];
                    let inside_y = c.min[1] <= y && y <= c.max[1];
                    assert!(!(inside_x && inside_y), "cuboid {c:?} covers hole cell ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn empty_grid_no_cuboids() {
        let grid: CellGrid<u16> = CellGrid::new_empty([4, 4, 4]);
        assert!(GreedyCuboidDecomposition::decompose(&grid).is_empty());
        // Zero-extent grid too.
        let zero: CellGrid<u16> = CellGrid::new_empty([0, 5, 5]);
        assert!(GreedyCuboidDecomposition::decompose(&zero).is_empty());
    }

    #[test]
    fn determinism_same_input_same_output() {
        let extent = [6, 6, 6];
        let grid = grid_from_fn(extent, |x, y, z| {
            if (x + y + z) % 2 == 0 {
                Some((x % 3) as u16)
            } else {
                None
            }
        });
        let first = GreedyCuboidDecomposition::decompose(&grid);
        let second = GreedyCuboidDecomposition::decompose(&grid);
        assert_eq!(first, second, "decomposition must be deterministic");
    }

    #[test]
    fn greedy_beats_naive_cuboid_count() {
        // A solid 4×4×4 single-label block is 1 cuboid, not 64.
        let grid = grid_from_fn([4, 4, 4], |_x, _y, _z| Some(0));
        let cuboids = GreedyCuboidDecomposition::decompose(&grid);
        assert_eq!(cuboids.len(), 1);
        assert!(cuboids.len() < 4 * 4 * 4);
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
        // Several pseudo-random label patterns over varied extents and label counts;
        // assert the three invariants every time. The real safety net for the growth
        // logic.
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
            for labels in [1u32, 2, 3, 5] {
                for fill_percent in [10u32, 35, 65, 90, 100] {
                    let mut grid: CellGrid<u16> = CellGrid::new_empty(extent);
                    for cell in grid.cells.iter_mut() {
                        let labeled = (lcg.next_u32() % 100) < fill_percent;
                        if labeled {
                            *cell = Some((lcg.next_u32() % labels) as u16);
                        }
                    }
                    let cuboids = GreedyCuboidDecomposition::decompose(&grid);
                    assert_invariants(&grid, &cuboids);
                    // Cuboid count never exceeds labeled-cell count.
                    let labeled = grid.cells.iter().filter(|c| c.is_some()).count();
                    assert!(cuboids.len() <= labeled);
                }
            }
        }
    }

    /// Expand a decomposition back into the SET of `(x, y, z)` cells it covers,
    /// paired with the covering cuboid's label. The structural round-trip tests
    /// compare this against the grid's own labeled cells.
    fn expand_cuboids_to_cells(cuboids: &[Cuboid<u16>]) -> std::collections::HashMap<[u32; 3], u16> {
        let mut cells = std::collections::HashMap::new();
        for cuboid in cuboids {
            for z in cuboid.min[2]..=cuboid.max[2] {
                for y in cuboid.min[1]..=cuboid.max[1] {
                    for x in cuboid.min[0]..=cuboid.max[0] {
                        let previous = cells.insert([x, y, z], cuboid.label);
                        assert!(
                            previous.is_none(),
                            "cell ({x},{y},{z}) covered by more than one cuboid (overlap)"
                        );
                    }
                }
            }
        }
        cells
    }

    /// Collect a grid's labeled cells as a `(x, y, z) -> label` map.
    fn grid_labeled_cells(grid: &CellGrid<u16>) -> std::collections::HashMap<[u32; 3], u16> {
        let [w, h, d] = grid.extent;
        let mut cells = std::collections::HashMap::new();
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    if let Some(label) = grid.cell_at(x, y, z) {
                        cells.insert([x, y, z], label);
                    }
                }
            }
        }
        cells
    }

    /// The full structural round-trip: decompose, expand the cuboids back to cells,
    /// and assert the expanded cell+label map EXACTLY equals the grid's labeled
    /// cells. Subsumes exact-cover, no-overlap, and per-cell label correctness (no
    /// merging across differing labels) in one set-equality assertion. Returns the
    /// cuboid count.
    fn assert_round_trip_exact(grid: &CellGrid<u16>) -> usize {
        let cuboids = GreedyCuboidDecomposition::decompose(grid);
        let expanded = expand_cuboids_to_cells(&cuboids);
        let original = grid_labeled_cells(grid);
        assert_eq!(
            expanded, original,
            "expanded cuboid cells (with labels) must exactly equal the grid's labeled cells"
        );
        // Belt-and-braces against the per-axis invariant checker.
        assert_invariants(grid, &cuboids);
        cuboids.len()
    }

    #[test]
    fn round_trip_single_cell() {
        let grid = grid_from_fn([3, 3, 3], |x, y, z| if [x, y, z] == [1, 1, 1] { Some(7) } else { None });
        assert_eq!(assert_round_trip_exact(&grid), 1);
    }

    #[test]
    fn round_trip_empty_and_full() {
        let empty: CellGrid<u16> = CellGrid::new_empty([4, 4, 4]);
        assert_eq!(assert_round_trip_exact(&empty), 0);

        let full = grid_from_fn([4, 3, 5], |_x, _y, _z| Some(2));
        assert_eq!(assert_round_trip_exact(&full), 1, "solid block → one cuboid");
    }

    #[test]
    fn round_trip_multiple_labels() {
        // A 4×4×2 grid quartered into four distinct labels along X and Y, so a correct
        // decomposition can NEVER merge two labels into one cuboid. The round-trip's
        // per-cell label equality is the guard.
        let extent = [4, 4, 2];
        let grid = grid_from_fn(extent, |x, y, _z| {
            let label = match (x < 2, y < 2) {
                (true, true) => 11,
                (false, true) => 22,
                (true, false) => 33,
                (false, false) => 44,
            };
            Some(label)
        });
        let count = assert_round_trip_exact(&grid);
        assert!(count >= 4, "four labels must yield at least four cuboids, got {count}");
        let labels: std::collections::HashSet<u16> =
            GreedyCuboidDecomposition::decompose(&grid).iter().map(|c| c.label).collect();
        assert_eq!(
            labels,
            [11, 22, 33, 44].into_iter().collect(),
            "every label must survive into the cuboids"
        );
    }

    #[test]
    fn round_trip_handmade_concave_shapes() {
        // Irregular/concave outlines whose holes and notches no cuboid may cover.

        // (1) L-shape across depth 2.
        let l_shape = grid_from_fn([3, 3, 2], |x, y, _z| if y == 0 || x == 0 { Some(4) } else { None });
        assert_round_trip_exact(&l_shape);

        // (2) 5×5 ring (hollow centre) over depth 1.
        let ring = grid_from_fn([5, 5, 1], |x, y, _z| {
            if x == 0 || y == 0 || x == 4 || y == 4 {
                Some(3)
            } else {
                None
            }
        });
        assert_round_trip_exact(&ring);

        // (3) A "plus"/cross silhouette with a concave notch at each corner, two
        //     labels in the arms, extruded over depth 3.
        let cross = grid_from_fn([5, 5, 3], |x, y, _z| {
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
        let staircase = grid_from_fn([4, 4, 1], |x, y, _z| if y <= x { Some(1) } else { None });
        assert_round_trip_exact(&staircase);
    }

    #[test]
    fn round_trip_randomized_multi_label() {
        // Pseudo-random multi-label fills over varied extents: the structural
        // round-trip on every sample, the safety net for growth + label split logic.
        let mut lcg = Lcg(0xfeed_face_dead_beef);
        let extents = [[1, 1, 1], [6, 4, 5], [9, 2, 7], [3, 8, 4], [7, 7, 7]];
        for &extent in &extents {
            for labels in [1u32, 2, 4] {
                for fill_percent in [25u32, 55, 100] {
                    let mut grid: CellGrid<u16> = CellGrid::new_empty(extent);
                    for cell in grid.cells.iter_mut() {
                        if (lcg.next_u32() % 100) < fill_percent {
                            *cell = Some((lcg.next_u32() % labels) as u16);
                        }
                    }
                    assert_round_trip_exact(&grid);
                }
            }
        }
    }
}
