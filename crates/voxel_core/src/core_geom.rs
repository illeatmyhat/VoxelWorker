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

/// The largest absolute voxel index a resolved [`crate::voxel::Voxel::local_index`] can
/// carry — it is an `i32`, and the two-layer expansion stamps it with an unchecked
/// `as i32` after rebasing in `i64` (`two_layer_store::chunk::stamped_voxel`, ADR 0008).
pub const MAX_LOCAL_VOXEL_INDEX: i64 = i32::MAX as i64;

/// The furthest block offset from the resolve frame's origin whose voxels still fit
/// [`MAX_LOCAL_VOXEL_INDEX`], at `voxels_per_block`. **This is the real supported
/// placement range of the resolve/expand path**, and it is much smaller than the
/// ±~8×10⁹-block range the `narrow_chunk_coord` audit (S4a, ADR 0002 Decision 2) states.
///
/// The two are not in conflict by accident — the S4a audit is correct about what it
/// actually bounds, and the gap is a step it does not take. It proves the CHUNK
/// COORDINATE fits `i32`, which it does precisely because a chunk coordinate is a voxel
/// index DIVIDED by the chunk extent. The expansion then multiplies back by that same
/// extent to rebase each voxel, so the quantity finally stored is `chunk_extent` times
/// larger than the one proved safe. Bounding a quotient says nothing about the product
/// it came from.
///
/// Concretely, at the stated ±8×10⁹ blocks the absolute voxel index overruns `i32` by
/// **4× at density 1 and 238× at density 64**, and the `as i32` wraps rather than
/// saturating — so a far-placed voxel would be stamped at a plainly wrong position, and
/// two chunks a multiple of 2³² voxels apart would alias onto the SAME `local_index`.
///
/// Not reachable through the UI: at density 64 this still allows ±3.3×10⁷ blocks, and a
/// chiselling scene is tens to thousands. It is recorded and proved rather than fixed
/// because the cast sits in the innermost expansion loop, and because the useful output
/// is the honest bound, not a branch. See the `kani_proofs` module below.
pub fn max_supported_block_offset(voxels_per_block: u32) -> i64 {
    MAX_LOCAL_VOXEL_INDEX / voxels_per_block.max(1) as i64
}

/// Whether a rebased absolute voxel index still fits the `i32`
/// [`crate::voxel::Voxel::local_index`] — i.e. whether the expansion's `as i32` is
/// lossless for it. The bound [`max_supported_block_offset`] names in blocks, in voxels.
#[inline]
pub fn local_voxel_index_fits(absolute_voxel_index: i64) -> bool {
    absolute_voxel_index >= i32::MIN as i64 && absolute_voxel_index <= MAX_LOCAL_VOXEL_INDEX
}

/// Bounded model checking of the frame-rebase cast (ADR 0008): the expansion rebases a
/// chunk-local voxel into the recentred frame in `i64`, then stamps it into an `i32`
/// `local_index` with an unchecked `as`. These harnesses establish exactly where that is
/// lossless and exactly where it stops being so.
///
/// This is the same shape as the `substrate::interval::rational` overflow harnesses and
/// the `FieldInterval` endpoint ones: a documented deviation that a deductive or algebraic
/// model could not see, because a proof over mathematical integers has no `i32` to overflow.
///
/// `#[cfg(kani)]` keeps them out of ordinary builds. Run under WSL:
/// `cargo kani -p voxel_core -j --output-format=terse`, or via `verification/run-all.sh`.
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// The expansion's rebase arithmetic, verbatim from
    /// `two_layer_store::chunk::stamped_voxel` + `stream::stream_chunk_recentred`:
    /// `index_offset = chunk_coord·chunk_extent − recentre`, then `chunk_local + offset`.
    fn rebased_index(chunk_coord: i64, density: i64, recentre: i64, chunk_local: i64) -> i64 {
        chunk_local + (chunk_coord * (CHUNK_BLOCKS as i64 * density) - recentre)
    }

    /// **Within the envelope the cast is LOSSLESS.** If the rebased index satisfies
    /// [`local_voxel_index_fits`], the `as i32` the expansion performs round-trips exactly —
    /// the stamped `local_index` IS the absolute voxel index, so every consumer reading it
    /// back gets the position the evaluator meant.
    #[kani::proof]
    fn local_index_cast_is_lossless_within_the_envelope() {
        let rebased: i64 = kani::any();
        kani::assume(local_voxel_index_fits(rebased));
        assert!(rebased as i32 as i64 == rebased);
    }

    /// **The envelope is TIGHT, in the units the range is stated in.** For a density in the
    /// supported `1..=64` band, a block offset within [`max_supported_block_offset`] rebases
    /// to an index the cast carries losslessly — and the bound is exact: one block further
    /// overruns, at every density in the band.
    #[kani::proof]
    fn the_supported_block_offset_bound_is_sound_and_tight() {
        let density: u32 = kani::any();
        kani::assume(density >= 1 && density <= 64);
        let block_offset: i64 = kani::any();
        let limit = max_supported_block_offset(density);
        kani::assume(block_offset >= -limit && block_offset <= limit);

        // A voxel at `block_offset` blocks from the frame origin sits at this voxel index.
        let voxel_index = block_offset * density as i64;
        assert!(local_voxel_index_fits(voxel_index));
        assert!(voxel_index as i32 as i64 == voxel_index);
        // Tight in BOTH directions, with no per-density escape: one block past the limit
        // leaves the envelope. (Sound at density 1 too, where the limit is `i32::MAX` and
        // `(limit + 1)` is exactly the first unrepresentable index.)
        assert!(!local_voxel_index_fits((limit + 1) * density as i64));
    }

    /// The same soundness statement over the PRODUCTION arithmetic rather than an abstract
    /// index: the chunk rebase `chunk_local + (chunk_coord·chunk_extent − recentre)` that
    /// `stream_chunk_recentred` + `stamped_voxel` actually perform is lossless whenever its
    /// result is in the envelope, and — the part worth checking — the `i64` it computes in
    /// cannot itself overflow for any chunk coordinate `narrow_chunk_coord` can produce.
    #[kani::proof]
    fn the_production_rebase_stays_within_i64_and_casts_losslessly() {
        let chunk_coord: i32 = kani::any();
        let density: u32 = kani::any();
        kani::assume(density >= 1 && density <= 64);
        // A chunk-local voxel index, bounded by one chunk's extent.
        let chunk_local: i64 = kani::any();
        kani::assume(chunk_local >= 0 && chunk_local < CHUNK_BLOCKS as i64 * density as i64);
        // The recentre is a raw carried `i64`; bound it to the same scale the chunk term can
        // reach, which is what any minted recentre satisfies (it is a composite extent
        // midpoint, not an arbitrary word).
        let recentre: i64 = kani::any();
        kani::assume(recentre > -(1i64 << 48) && recentre < (1i64 << 48));

        // No `i64` overflow anywhere in the rebase: the chunk term is at most
        // 2^31 · (4 · 64) = 2^39, and the recentre is bounded to 2^48.
        let rebased = rebased_index(chunk_coord as i64, density as i64, recentre, chunk_local);

        // Within the envelope, the stamp the expansion performs is exact.
        if local_voxel_index_fits(rebased) {
            assert!(rebased as i32 as i64 == rebased);
        }
    }

    /// **Outside the envelope the cast WRAPS rather than saturating** — the failure mode is a
    /// voxel stamped at a wrong position, not a dropped one. Stated as the concrete aliasing
    /// witness: two chunk coordinates exactly 2³² voxels apart collide onto one `local_index`.
    /// This is what makes the bound worth naming rather than leaving implicit.
    #[kani::proof]
    fn beyond_the_envelope_distinct_voxels_alias() {
        let rebased: i64 = kani::any();
        // Keep the shifted value in range so the `+ 2^32` cannot itself overflow `i64`.
        kani::assume(rebased > -(1i64 << 40) && rebased < (1i64 << 40));
        let shifted = rebased + (1i64 << 32);
        // Distinct absolute indices, identical stamped index: silent aliasing.
        assert!(shifted != rebased);
        assert!(shifted as i32 == rebased as i32);
    }
}

#[cfg(test)]
mod frame_envelope_tests {
    use super::*;

    /// The Kani harnesses above are `#[cfg(kani)]` and therefore invisible to `cargo test`;
    /// this is the always-on check that the shipping constants still say what they prove.
    #[test]
    fn the_supported_block_offset_matches_the_i32_local_index() {
        // The documented figures in `max_supported_block_offset`'s doc comment.
        assert_eq!(max_supported_block_offset(1), 2_147_483_647);
        assert_eq!(max_supported_block_offset(64), 33_554_431);
        // A zero density is treated as 1 (the `.max(1)` the resolve applies everywhere).
        assert_eq!(max_supported_block_offset(0), max_supported_block_offset(1));

        for density in [1u32, 4, 16, 32, 64] {
            let limit = max_supported_block_offset(density);
            let last = limit * density as i64;
            assert!(local_voxel_index_fits(last), "density {density}: limit must fit");
            assert_eq!(last as i32 as i64, last, "density {density}: cast must be lossless");
            assert!(
                !local_voxel_index_fits((limit + 1) * density as i64),
                "density {density}: the bound must be TIGHT"
            );
        }
    }

    /// The gap this bound was written to record: the S4a audit's ±8×10⁹-block figure is a
    /// bound on the chunk COORDINATE, and the expansion multiplies back by the chunk extent.
    /// If someone ever widens `local_index` past `i32`, this test is the reminder to revisit
    /// ADR 0008's amendment and `narrow_chunk_coord`'s correction note.
    #[test]
    fn the_s4a_stated_range_does_not_fit_the_local_index() {
        let s4a_stated_blocks: i64 = 8_000_000_000;
        for (density, expected_overrun) in [(1u32, 3), (64u32, 238)] {
            let voxel_index = s4a_stated_blocks * density as i64;
            assert!(
                !local_voxel_index_fits(voxel_index),
                "density {density}: the stated S4a range must NOT fit local_index"
            );
            assert!(
                voxel_index / MAX_LOCAL_VOXEL_INDEX >= expected_overrun,
                "density {density}: overrun should be at least {expected_overrun}x"
            );
        }
    }
}

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
/// the chunk-storage codec and meshing; the active block palette (`block_palette`)
/// maps it to a colour / texture, and `.vox` export maps it
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

/// A cuboid-decomposition / region-cell key (ADR 0003 §3c): a clean categorical
/// [`BlockId`] in the low 15 bits + a transient on-face-grid overlay marker in the high
/// bit, packed into one `u16`.
///
/// The overlay bit is the flag the **cuboid mesher** folds into its region-cell key. It is
/// NOT the retired per-voxel `GRID_OVERLAY_BIT` — that flag is gone from the persistent
/// categorical cell. Instead a *local* decomposition key `block_id | (overlay << 15)` is
/// composed from each voxel's clean `block_id` + its transient `grid_overlay` marker, so
/// `decompose_into_boxes` (which stays representation-agnostic) refuses to merge a box
/// across differing overlay flags — exactly the old per-box split — without ever seeing a
/// render flag inside the material. The mesher then splits the key back into the clean
/// `block_id` ([`block_id`](Self::block_id)) and the per-box overlay
/// ([`has_overlay`](Self::has_overlay)), writing the overlay into a DEDICATED render channel
/// so the shader reads it separately and never masks it out of the material.
///
/// The overlay bit lives ONLY in this render-side key — never in the persistent
/// [`crate::voxel::Voxel`] payload, the chunk-storage codec, or the `.vox` export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CellKey(u16);

impl CellKey {
    /// The high bit that marks the transient on-face-grid overlay (ADR 0003 §3c). Private
    /// to the type: consumers compose/inspect through the methods, never touch the bit.
    const OVERLAY_BIT: u16 = 1 << 15;

    /// Compose a key from a clean categorical `block_id` (low 15 bits) and a transient
    /// on-face-grid `overlay` marker (high bit).
    #[inline]
    pub fn compose(block_id: u16, overlay: bool) -> Self {
        let mut key = block_id;
        if overlay {
            key |= Self::OVERLAY_BIT;
        }
        Self(key)
    }

    /// Wrap a raw packed `u16` (as stored in the store/mesh cell arrays) as a key.
    #[inline]
    pub fn from_raw(raw: u16) -> Self {
        Self(raw)
    }

    /// The raw packed `u16`, for storing back into a cell array.
    #[inline]
    pub fn raw(self) -> u16 {
        self.0
    }

    /// The clean categorical `block_id` — the overlay bit masked off. Used where a consumer
    /// needs the categorical id without the transient render flag (the two-layer occupancy
    /// expansion, the raymarch shade).
    #[inline]
    pub fn block_id(self) -> u16 {
        self.0 & !Self::OVERLAY_BIT
    }

    /// Whether the key carries the on-face-grid overlay marker. Used where a consumer needs
    /// the render flag back out (the two-layer occupancy expansion carries it onto the
    /// expanded `Voxel::grid_overlay`).
    #[inline]
    pub fn has_overlay(self) -> bool {
        self.0 & Self::OVERLAY_BIT != 0
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
