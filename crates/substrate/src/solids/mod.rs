//! Constructive-solid-geometry cell kernels: the black/white/grey cell
//! classification (linear and scoped), the greedy cuboid decomposition of a cell
//! grid into boxes, and the culled box meshing that determines exposed faces.
//! Each module carries its own literature citations.

pub mod cell_classification;
pub mod culled_box_meshing;
pub mod greedy_cuboid_decomposition;
pub mod scoped_cell_classification;

pub use cell_classification::{CellClassification, CellCombineOp, CellContribution};
pub use culled_box_meshing::CulledBoxMeshing;
pub use greedy_cuboid_decomposition::{CellGrid, Cuboid, GreedyCuboidDecomposition};
pub use scoped_cell_classification::{ScopedCellClassification, ScopedCellEvent};
