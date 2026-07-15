//! A sparse, world-fixed **min-mip occupancy pyramid** over a set of packed lattice keys.
//!
//! Given a sparse set of occupied lattice cells (each a [`crate::spatial::lattice_key`]-packed signed
//! 3-vector), a [`MinMipLevel`] folds every key to the coarser cell that contains it — cell
//! coordinate `floor_div(lattice_coordinate, cell_edge)` per axis — then keeps the folded cell
//! keys **sorted ascending and deduplicated**. Because a cell is present whenever *any* of its
//! finer keys is, the level is a **conservative superset** of the true occupancy: a consumer that
//! finds a cell absent may soundly skip the whole cell; a present cell may be over-reported but
//! never under-reported. A [`SparseMinMipPyramid`] stacks several such levels over one key set at a
//! geometrically growing `cell_edge` (e.g. 8, 64, 512), so a hierarchical traverser can leap the
//! coarsest empty cell covering its position in one stride and descend to finer work only where a
//! level reports occupancy.
//!
//! The sorted key layout is the same one a binary search (on CPU, and — split to `(hi, lo)` u32
//! pairs — on a GPU) relies on; see [`crate::spatial::lattice_key`]. Folding, sorting, and lookup are pure
//! functions of the key set and the edge list — no traversal of any source structure lives here;
//! a producer walks its own domain data, emits keys, and hands them to this fold.
//!
//! ## Literature
//!
//! This is the **clip-map / mip-pyramid over a sparse voxel occupancy set** of the volumetric-
//! rendering literature. Cite: Tanner, Migdal & Jones 1998, *The Clipmap: a virtual mipmap* (the
//! clip-map); Losasso & Hoppe 2004, *Geometry clipmaps* (the terrain-LOD pyramid); Crassin,
//! Neyret, Lefebvre & Eisemann 2009, *GigaVoxels* (a brick/occupancy pyramid traversed by a
//! hierarchical DDA); Amanatides & Woo 1987, *A fast voxel traversal algorithm* (the per-cell DDA a
//! consumer runs against a level); Museth 2013, *VDB: high-resolution sparse volumes* (the sparse-
//! hierarchy prior art). **Deviation:** the levels are **world-fixed** min-mip occupancy sets over
//! a *sparse* key set — not the camera-centred, dense, toroidally-updated rings of the original
//! clipmap. There is no clip region and no re-centring; each level is simply the deduplicated fold
//! of the key set at its edge, so its footprint is proportional to the *occupied* cell count, not
//! to a window size.

use rayon::prelude::*;

use crate::spatial::lattice_key::{pack_lattice_key, unpack_lattice_key};

/// Fold a signed lattice coordinate to the coordinate of the cell of edge `cell_edge` that
/// contains it: `floor_div` per axis (Euclidean, so negatives round toward −∞ and cells tile the
/// lattice without a gap at the origin). `cell_edge` is clamped to at least 1 (a 0-edge cell is
/// ill-defined; edge 1 is the identity fold).
pub fn fold_coordinate_to_cell(coordinate: [i64; 3], cell_edge: u32) -> [i64; 3] {
    let edge = cell_edge.max(1) as i64;
    [
        coordinate[0].div_euclid(edge),
        coordinate[1].div_euclid(edge),
        coordinate[2].div_euclid(edge),
    ]
}

/// Binary-search a **sorted, deduplicated** cell-key set for the cell (of edge `cell_edge`) that
/// contains `coordinate`. `false` for the empty set — the pure predicate carries no "empty means
/// everything" policy; a consumer that reads an empty level as "no skip information" applies that
/// itself. The set MUST be sorted ascending (the invariant [`MinMipLevel`] maintains), or the
/// search result is meaningless.
pub fn sorted_cell_keys_contain(cell_keys: &[u64], coordinate: [i64; 3], cell_edge: u32) -> bool {
    let cell = fold_coordinate_to_cell(coordinate, cell_edge);
    cell_keys.binary_search(&pack_lattice_key(cell)).is_ok()
}

/// One min-mip occupancy level: the sparse set of occupied cells of edge [`cell_edge`](Self::cell_edge)
/// lattice units, each a [`crate::spatial::lattice_key`]-packed cell key. [`cell_keys`](Self::cell_keys) is
/// kept **sorted strictly ascending and unique** — the order a binary search (and a GPU's split-key
/// search) depends on. The set is a conservative superset of the true occupancy: every occupied
/// finer key's cell is present, and no cell without an occupied key is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinMipLevel {
    /// The cell edge in lattice units (the fold divisor). At least 1.
    pub cell_edge: u32,
    /// The occupied cells' packed keys, sorted strictly ascending + deduplicated.
    pub cell_keys: Vec<u64>,
}

impl MinMipLevel {
    /// An empty level (no occupied cells) at the given edge.
    pub fn empty(cell_edge: u32) -> Self {
        MinMipLevel {
            cell_edge: cell_edge.max(1),
            cell_keys: Vec::new(),
        }
    }

    /// Fold packed lattice keys to their cells of edge `cell_edge`, then sort + deduplicate — the
    /// min-mip of the key set. Each input key maps to exactly one cell; distinct keys collapsing to
    /// the same cell dedup to one entry. Pure function of the keys and the edge; the input need not
    /// be sorted.
    pub fn from_keys(keys: &[u64], cell_edge: u32) -> Self {
        Self::from_key_iter(keys.iter().copied(), cell_edge)
    }

    /// Fold packed lattice keys drawn from any iterator to their cells of edge `cell_edge`, then
    /// sort + deduplicate — the single-pass form of [`from_keys`](Self::from_keys). A producer that
    /// already streams keys (e.g. a domain that maps over its records) folds straight into the level
    /// without first materialising an intermediate `Vec<u64>`; only the folded cell-key output is
    /// allocated. Byte-identical to [`from_keys`](Self::from_keys) over the same key sequence.
    pub fn from_key_iter(keys: impl IntoIterator<Item = u64>, cell_edge: u32) -> Self {
        let cell_edge = cell_edge.max(1);
        let cell_keys = keys
            .into_iter()
            .map(|key| pack_lattice_key(fold_coordinate_to_cell(unpack_lattice_key(key), cell_edge)))
            .collect();
        Self::from_folded_cell_keys(cell_keys, cell_edge)
    }

    /// Assemble a level from cell keys a producer has **already folded** to this edge (e.g. one
    /// that emits cell keys directly during its own traversal, including a bulk range emission) —
    /// sort + deduplicate only, no re-fold. The keys must already be at cell granularity for
    /// `cell_edge`; this is the sink of a producer that did its own folding.
    pub fn from_folded_cell_keys(mut cell_keys: Vec<u64>, cell_edge: u32) -> Self {
        cell_keys.par_sort_unstable();
        cell_keys.dedup();
        MinMipLevel {
            cell_edge: cell_edge.max(1),
            cell_keys,
        }
    }

    /// Whether this level holds the given already-packed cell key (a binary search).
    pub fn contains_cell(&self, cell_key: u64) -> bool {
        self.cell_keys.binary_search(&cell_key).is_ok()
    }

    /// Whether the cell of this level's edge containing `coordinate` is occupied (fold then binary
    /// search). `false` for an empty level — the pure predicate carries no policy; see
    /// [`sorted_cell_keys_contain`].
    pub fn contains_coordinate(&self, coordinate: [i64; 3]) -> bool {
        sorted_cell_keys_contain(&self.cell_keys, coordinate, self.cell_edge)
    }
}

/// A stack of [`MinMipLevel`]s over ONE key set at geometrically growing edges — the sparse min-mip
/// pyramid. [`levels`](Self::levels) is in the order the caller supplied the edges (a hierarchical
/// traverser typically supplies them fine→coarse and descends coarsest-first). Each level is folded
/// independently from the SAME keys, so a coarser edge yields no more cells than a finer one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseMinMipPyramid {
    /// One occupancy level per supplied edge, in the supplied order.
    pub levels: Vec<MinMipLevel>,
}

impl SparseMinMipPyramid {
    /// Build one [`MinMipLevel`] per edge from the shared key set (each level folds the same keys
    /// at its own edge). The edge list is domain configuration — this kernel names no particular
    /// level count or edge progression.
    pub fn from_keys(keys: &[u64], cell_edges: &[u32]) -> Self {
        SparseMinMipPyramid {
            levels: cell_edges
                .iter()
                .map(|&edge| MinMipLevel::from_keys(keys, edge))
                .collect(),
        }
    }

    /// Build the pyramid from packed lattice keys drawn from any iterator — the single-pass form
    /// of [`from_keys`](Self::from_keys) for a producer that streams keys (e.g. maps over its
    /// records). Every level folds the SAME key set, so the keys are collected into ONE buffer
    /// here (a multi-pass fold cannot replay a single-pass iterator) and each edge folds that
    /// buffer; the caller is spared building its own intermediate `Vec<u64>`. Byte-identical to
    /// [`from_keys`](Self::from_keys) over the same key sequence.
    pub fn from_key_iter(keys: impl IntoIterator<Item = u64>, cell_edges: &[u32]) -> Self {
        let keys: Vec<u64> = keys.into_iter().collect();
        Self::from_keys(&keys, cell_edges)
    }

    /// An all-empty pyramid — one empty level per edge.
    pub fn empty(cell_edges: &[u32]) -> Self {
        SparseMinMipPyramid {
            levels: cell_edges.iter().map(|&edge| MinMipLevel::empty(edge)).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(coordinate: [i64; 3]) -> u64 {
        pack_lattice_key(coordinate)
    }

    /// The fold produces a SORTED, STRICTLY-ASCENDING, DEDUPLICATED set that is a conservative
    /// SUPERSET: every input key's cell is present, and the level holds no cell that no input key
    /// folds to. Uses an edge that collapses several distinct keys into shared cells.
    #[test]
    fn fold_is_a_sorted_deduplicated_conservative_superset() {
        let edge = 8u32;
        // Distinct lattice points, several inside the same 8-cell, some negative (Euclidean fold).
        let coordinates = [
            [0i64, 0, 0],
            [7, 7, 7],   // same cell as origin
            [8, 0, 0],   // neighbouring cell on x
            [-1, -1, -1],// cell (-1,-1,-1)
            [-8, 0, 0],  // cell (-1,0,0)
            [100, 3, 50],
            [103, 1, 49],// same cell as the previous
        ];
        let keys: Vec<u64> = coordinates.iter().map(|&c| key(c)).collect();
        let level = MinMipLevel::from_keys(&keys, edge);

        // Sorted strictly ascending + unique.
        assert!(
            level.cell_keys.windows(2).all(|pair| pair[0] < pair[1]),
            "cell keys must be strictly ascending and unique"
        );

        // Conservative superset: every input's cell is present.
        for &c in &coordinates {
            assert!(
                level.contains_coordinate(c),
                "the cell of an occupied key must be present"
            );
        }

        // Exactness: the level holds exactly the set of input cells — no stray cell.
        use std::collections::BTreeSet;
        let expected: BTreeSet<u64> = coordinates
            .iter()
            .map(|&c| pack_lattice_key(fold_coordinate_to_cell(c, edge)))
            .collect();
        let actual: BTreeSet<u64> = level.cell_keys.iter().copied().collect();
        assert_eq!(actual, expected);
        // The seven inputs collapse to five distinct cells.
        assert_eq!(level.cell_keys.len(), 5);
    }

    /// Edge 1 is the identity fold (each key is its own cell); a coarser edge yields no more cells
    /// than a finer one over the same keys (min-mip monotonicity).
    #[test]
    fn coarser_edge_never_grows_the_cell_count() {
        let coordinates: Vec<[i64; 3]> = (0..64)
            .map(|i| [i % 10, (i / 10) % 7, i % 5])
            .collect();
        let keys: Vec<u64> = coordinates.iter().map(|&c| key(c)).collect();
        let pyramid = SparseMinMipPyramid::from_keys(&keys, &[1, 8, 64, 512]);

        // Edge 1 is the identity fold: its cell count equals the DISTINCT key count.
        use std::collections::BTreeSet;
        let distinct: BTreeSet<u64> = keys.iter().copied().collect();
        assert_eq!(pyramid.levels[0].cell_keys.len(), distinct.len());
        assert_eq!(pyramid.levels[0].cell_edge, 1);

        // Monotone non-increasing cell count as the edge grows.
        for pair in pyramid.levels.windows(2) {
            assert!(
                pair[1].cell_keys.len() <= pair[0].cell_keys.len(),
                "a coarser level cannot hold more cells than a finer one"
            );
        }
    }

    /// The pre-folded-key sink sorts + dedups exactly like the folding constructor: feeding a
    /// producer's raw (unsorted, duplicated) cell keys yields the same level as folding the finer
    /// keys, and equals the folding path when the producer folded the same points.
    #[test]
    fn folded_cell_key_sink_sorts_and_dedups() {
        // Raw cell keys straight from a producer: out of order, with duplicates.
        let raw = vec![
            key([2, 0, 0]),
            key([0, 0, 0]),
            key([2, 0, 0]), // duplicate
            key([-1, 5, 3]),
            key([0, 0, 0]), // duplicate
        ];
        let level = MinMipLevel::from_folded_cell_keys(raw, 8);
        assert_eq!(
            level.cell_keys,
            {
                let mut expected = vec![key([2, 0, 0]), key([0, 0, 0]), key([-1, 5, 3])];
                expected.sort_unstable();
                expected
            }
        );
        assert!(level.contains_cell(key([0, 0, 0])));
        assert!(!level.contains_cell(key([9, 9, 9])));
    }

    /// The empty set: no cell is reported occupied, at any edge, and the pyramid form matches.
    #[test]
    fn empty_set_reports_nothing_occupied() {
        let level = MinMipLevel::from_keys(&[], 8);
        assert!(level.cell_keys.is_empty());
        assert!(!level.contains_coordinate([0, 0, 0]));
        assert!(!level.contains_cell(key([0, 0, 0])));

        let empty = SparseMinMipPyramid::empty(&[8, 64, 512]);
        assert_eq!(empty.levels.len(), 3);
        assert!(empty.levels.iter().all(|l| l.cell_keys.is_empty()));
        // `empty(edges)` equals `from_keys(&[], edges)`.
        assert_eq!(empty, SparseMinMipPyramid::from_keys(&[], &[8, 64, 512]));
    }
}
