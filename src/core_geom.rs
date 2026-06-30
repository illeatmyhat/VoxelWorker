//! ADR 0003 bottom layer: dependency-free geometry primitives + the streaming
//! quantum; depends on nothing in the crate.

/// Edge length of a render chunk, in BLOCKS (ADR 0002 Decision 3, part of #19).
/// A chunk therefore spans `CHUNK_BLOCKS * voxels_per_block` voxels per axis
/// (e.g. 4 blocks × density 16 = 64 voxels/axis). Chosen as a small whole-block
/// multiple so a chunk stays a phase-aligned, frustum-cullable unit while the
/// draw-call count stays sane. The resolved grid's occupied voxels are bucketed
/// into these chunks at rebuild time; each frame only the chunks whose world
/// AABB intersects the camera frustum are drawn.
pub const CHUNK_BLOCKS: u32 = 4;

/// Procedural material choice. Selects which procedural texture (Stone/Wood/
/// Plain) binds in the M4 texture-slice shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum MaterialChoice {
    #[default]
    Stone,
    Wood,
    Plain,
}

impl MaterialChoice {
    /// The number of distinct procedural materials (Stone/Wood/Plain). The
    /// renderer's per-voxel base-colour uniform array is sized to this, and a
    /// `material_id` is always `< MATERIAL_COUNT`.
    pub const MATERIAL_COUNT: usize = 3;

    /// The per-voxel `material_id` this choice stamps onto its voxels (ADR 0001
    /// step 3 "Materials"). Stable, dense (`0..MATERIAL_COUNT`), so it indexes both
    /// the renderer's base-colour uniform array and the procedural-texture table.
    /// Stone = 0, Wood = 1, Plain = 2.
    pub fn material_id(self) -> u16 {
        match self {
            MaterialChoice::Stone => 0,
            MaterialChoice::Wood => 1,
            MaterialChoice::Plain => 2,
        }
    }

    /// The inverse of [`material_id`](Self::material_id): the choice for a stamped
    /// id. Ids outside the known set fall back to [`Stone`](Self::Stone).
    pub fn from_material_id(id: u16) -> Self {
        match id {
            0 => MaterialChoice::Stone,
            1 => MaterialChoice::Wood,
            2 => MaterialChoice::Plain,
            _ => MaterialChoice::Stone,
        }
    }

    /// The categorical [`BlockId`] this material maps to in the block palette (ADR 0003
    /// §3a). Today the three procedural materials ARE the palette, so the id is the same
    /// dense `0..MATERIAL_COUNT` value [`material_id`](Self::material_id) returns — the
    /// categorical capability over the existing three materials, no rich content yet.
    pub fn block_id(self) -> BlockId {
        BlockId(self.material_id())
    }
}

/// A categorical block-palette id (ADR 0003 §3a — the per-voxel cell's block handle).
///
/// This replaces the old 3-value `material_id` enum jammed into a `u16` (with a render
/// flag in its high bit). It is an OPAQUE palette index that rides through the store,
/// the chunk-storage codec and meshing; the active block palette
/// ([`crate::block_palette`]) maps it to a colour / texture, and `.vox` export maps it
/// through that same palette. The three procedural materials occupy ids `0..3`
/// (Stone/Wood/Plain), so existing scenes resolve byte-identically; the rich VS palette
/// CONTENT (hundreds of named blocks) is the deferred part — this is only the
/// categorical CAPABILITY.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct BlockId(pub u16);

impl BlockId {
    /// The default block id a bare producer emits before a Tool overrides it (the old
    /// `material_id: 0` default — Stone in the procedural palette).
    pub const DEFAULT: BlockId = BlockId(0);

    /// The colour / atlas index the renderer + `.vox` export use for this id. Today the
    /// palette is the three procedural materials, so the index IS the id; a clamp keeps
    /// it inside the shader's `[0, MATERIAL_COUNT)` colour range for any stray id.
    pub fn color_index(self) -> u16 {
        self.0
    }
}

/// Typed per-`block_id` attributes (ADR 0003 §3a-bis).
///
/// **Minimal forward-compat placeholder.** ADR 0003 §3a-bis pins `BlockAttrs` as a typed
/// schema (orientation in the order-48 group + variant flags + neighbour-connection
/// bits) so a rotated stateful block re-composes its facing and VS schematic export is
/// not lossy. That whole schema — and the connection-resolve pass and block-entity
/// side-table — is **explicitly out of scope** for this slice (it is ADR 0003 §3a-bis /
/// ADR 0005). This zero-sized placeholder reserves the per-voxel field so the payload's
/// shape is forward-compatible: the schema is filled in later without touching the
/// payload's call sites again.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Default,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct BlockAttrs;

impl BlockAttrs {
    /// The default (empty) attributes — the only inhabitant of the placeholder schema.
    pub const DEFAULT: BlockAttrs = BlockAttrs;
}
