//! The bit / atlas kit: the bit-packed occupancy cube and its payload twin, the
//! slot free-list allocator, the linear→3D cube tile packing, the shelf
//! rectangle packer, and the sorted-key bitmask map. Each module carries its own
//! literature citations.

pub mod bit_cube;
pub mod bitmask_map;
pub mod cube_packing;
pub mod free_list;
pub mod shelf_bin_pack;
pub mod value_cube;

pub use bit_cube::BitCube;
pub use bitmask_map::{mask_bit_is_set, set_mask_bit, SortedKeyBitmaskMap};
pub use cube_packing::CubeTilePacking;
pub use free_list::SlotFreeList;
pub use shelf_bin_pack::{
    NormalizedTileRect, PackedTilePlacement, ShelfBinPack, TileImage, TileSize,
};
pub use value_cube::ValueCube;
