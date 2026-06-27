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
}
