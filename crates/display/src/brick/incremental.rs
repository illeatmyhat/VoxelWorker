use super::*;

/// What an [`IncrementalBrickField::apply_dirty_update`] touched — the per-edit "dirty
/// region" made observable so the GPU sink patches ONLY these atlas slots (never the
/// untouched ones) and the parity net can assert the cost is proportional to the edit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BrickFieldUpdate {
    /// Atlas slots (re)written this edit — newly allocated or overwritten sculpted
    /// bricks. When `atlas_grew` is false these are the ONLY slots the GPU patch writes.
    pub written_slots: Vec<u32>,
    /// Slots FREED this edit (their block became air/coarse or its chunk was removed);
    /// their tiles are now dead until reallocated. Free bytes are never uploaded.
    pub freed_slots: Vec<u32>,
    /// Whether the atlas tile geometry GREW (`bricks_per_axis` increased) — then every
    /// slot's 3D position moved, so the sink MUST re-pack + re-upload the whole atlas
    /// (the one legitimate wholesale re-pack, ADR 0011 pitfalls / ADR 0007 resize
    /// precedent). False ⇒ untouched slots keep their texels.
    pub atlas_grew: bool,
    /// MATERIAL SIDE ATLAS slots (re)written this edit — the cell-key tiles of the MIXED
    /// bricks the edit (re)emitted, in the second pool's OWN numbering. Empty for an edit that
    /// touched no mixed block (the common case: the pool is sparse). The GPU patch's cell-key
    /// work-list, exactly as `written_slots` is the occupancy atlas's.
    pub written_cell_key_slots: Vec<u32>,
    /// Cell-key slots FREED this edit: a block that stopped being mixed (or stopped existing)
    /// releases its material tile. Dead until reallocated; never uploaded.
    pub freed_cell_key_slots: Vec<u32>,
    /// Whether the MATERIAL SIDE ATLAS's tile geometry grew — its OWN signal, independent of
    /// [`atlas_grew`](Self::atlas_grew) (the pools size from their own slot counts, so either
    /// may grow without the other). True ⇒ the sink re-packs + re-uploads the whole side atlas.
    pub cell_key_atlas_grew: bool,
}

/// The PERSISTENT incremental brick field (ADR 0011 slice G3). Maintains the sorted
/// [`BrickRecord`] array + a slot-allocated atlas ACROSS edits so a per-edit update
/// re-evaluates only the DIRTY chunks' blocks and patches only their slots — the
/// "per-edit cost proportional to the dirty region, not the scene" win ADR 0009 promised.
///
/// Slots are managed by a **free-list** (allocate on a new sculpted brick, free when a
/// brick becomes air/coarse or its chunk is dirtied away), so slot numbers are STABLE
/// across edits and differ from the wholesale build's dense `0..count`. The invariant the
/// parity gate proves: after any edit, every LIVE record's slot bytes equal a from-scratch
/// [`build_brick_field`] of the same scene (free slots may hold garbage — they are
/// unreachable). The pyramid is REBUILT (not patched) from the merged record keys per
/// edit (a cheap pure function; incremental pyramid patching is deferred to G4).
#[derive(Debug, Clone)]
pub struct IncrementalBrickField {
    /// The brick edge in voxels (`voxels_per_block`, the ONE-BLOCK granule) — fixed for
    /// the field's life (a density change resets the field via a wholesale rebuild).
    brick_edge_voxels: u32,
    /// Records sorted strictly ascending by packed world-block key — the same order and
    /// content [`build_brick_field`] emits, only the sculpted records' slot NUMBERS differ.
    records: Vec<BrickRecord>,
    /// Per-slot occupancy tiles (bit-packed `edge²` X-row words each — see
    /// [`BrickOccupancyTile`]) indexed by atlas slot, WITH their free-list, delegated to
    /// substrate's [`SlotFreeList`]: a FREED slot's tile is retained (kept `edge²` words so
    /// the atlas packer never trips) but unreferenced — dead bits until the slot is
    /// reallocated. A new sculpted brick pops a freed slot (deterministic reuse — largest free
    /// index first) before growing; the reuse order is a test-readability nicety, not a
    /// correctness contract (parity tolerates slot renumbering — see the `records` doc above).
    slot_tiles: SlotFreeList<BrickOccupancyTile>,
    /// Per-cell-key-slot tiles of the MIXED bricks, in a **separate pool with its own
    /// free-list** — a mixed brick's `cell_key_slot` is not its `atlas_slot` and the two
    /// numberings never coincide (the material side atlas is sparse: only mixed bricks hold a
    /// slot, so tying it to the occupancy numbering would waste a tile per uniform brick).
    /// Same single-owner tile law as `slot_tiles`: tiles are MOVED in at emission, a freed
    /// slot's tile is dead until reallocated. A block flipping uniform↔mixed under an edit
    /// frees or allocates here exactly like any slot churn — the occupancy pool is untouched
    /// by that flip (its slot stays a sculpted slot either way).
    cell_key_tiles: SlotFreeList<BrickCellKeyTile>,
}

impl IncrementalBrickField {
    /// Seed the incremental field from a wholesale [`build_brick_field`] BY MOVE (the reset
    /// a scene load / density change / gate re-engagement performs), returning the mirror
    /// AND the [`SculptedAtlasPayload`] the install seam uploads. Consuming the build is the
    /// single-owner win (`docs/architecture/`, the brick-field display chapter): the record
    /// Vec moves straight into the mirror (no clone) and the flat atlas byte blob moves into
    /// the payload — the wholesale channel/inline reset now ships ONE copy of the field, not
    /// a build plus a mirror seeded from it. Slots are the build's dense `0..sculpted_count`;
    /// the free-list starts empty.
    pub fn from_wholesale(build: BrickFieldBuild) -> (Self, SculptedAtlasPayload) {
        // Re-derive the bit tiles from the build's flat atlas bytes (the one O(sculpted)
        // seeding cost) — the entry for callers that hold ONLY a `BrickFieldBuild` (the
        // golden `shot` tool, the parity/perf tests). The live worker/orchestrator wholesale
        // path instead calls [`from_wholesale_with_tiles`], MOVING the tiles the build just
        // rasterised straight in (no re-gather, no re-bit-pack).
        let sculpted_count = build.sculpted_brick_count();
        let slot_tiles: Vec<BrickOccupancyTile> = (0..sculpted_count as u32)
            .map(|slot| {
                BrickOccupancyTile::from_bytes(
                    build.brick_edge_voxels,
                    &build.sculpted_brick_occupancy(slot),
                )
            })
            .collect();
        Self::from_wholesale_with_tiles(build, slot_tiles)
    }

    /// Seed the incremental field from a wholesale build AND its already-rasterised per-slot
    /// occupancy tiles (dense slot order), MOVING both in — the zero-re-derive path for the
    /// live worker/orchestrator wholesale build. [`build_brick_field_with_tiles`] returns the
    /// build alongside the very tiles it rasterised; handing them here skips the
    /// `from_wholesale` re-gather (`sculpted_brick_occupancy` per slot) + re-bit-pack of
    /// bytes the packer just produced. The tiles MUST be the build's own sculpted tiles (one
    /// per sculpted record, slot order); a debug assert pins the count.
    pub fn from_wholesale_with_tiles(
        build: BrickFieldBuild,
        slot_tiles: Vec<BrickOccupancyTile>,
    ) -> (Self, SculptedAtlasPayload) {
        let sculpted_count = build.sculpted_brick_count();
        // Finding #5: a misordered / wrong-length tile vec would silently desync the mirror
        // from the build. The slot count + a representative brick-edge match are O(1) and
        // catch the structural mistakes, so they earn a RELEASE-mode assert. The exhaustive
        // per-slot byte-equality (O(sculpted·brick)) stays a debug-only check.
        assert_eq!(
            slot_tiles.len(),
            sculpted_count,
            "the carried tiles must be exactly the build's sculpted slots (dense 0..count)"
        );
        assert!(
            slot_tiles
                .first()
                .is_none_or(|tile| tile.edge() == build.brick_edge_voxels),
            "the carried tiles must share the build's brick edge"
        );
        debug_assert!(
            slot_tiles.iter().enumerate().all(|(slot, tile)| {
                tile.edge() == build.brick_edge_voxels
                    && tile.expand_to_bytes(SCULPTED_BRICK_OCCUPIED)
                        == build.sculpted_brick_occupancy(slot as u32)
            }),
            "each carried tile must byte-match the build's own sculpted slot in dense order"
        );
        let BrickFieldBuild {
            brick_records,
            sculpted_atlas_bytes,
            cell_key_tiles,
            brick_edge_voxels,
            bricks_per_axis,
            atlas_dim_voxels,
        } = build;
        // The cell-key tiles ride in the build (the material side atlas has no byte-packed
        // GPU form yet), so they MOVE straight into the mirror's pool — one owner, no clone.
        debug_assert_eq!(
            cell_key_tiles.len(),
            brick_records
                .iter()
                .filter(|record| record.payload.cell_key_slot().is_some())
                .count(),
            "a wholesale build carries exactly one cell-key tile per MIXED record"
        );
        let payload = SculptedAtlasPayload {
            bytes: sculpted_atlas_bytes,
            geometry: SculptedAtlasGeometry {
                bricks_per_axis,
                atlas_dim_voxels,
                brick_edge_voxels,
            },
            sculpted_slot_count: sculpted_count as u32,
        };
        let mirror = Self {
            brick_edge_voxels,
            records: brick_records,
            // Dense-seed the free-lists: every carried tile is a live slot `0..count`, no holes.
            slot_tiles: SlotFreeList::from_slots(slot_tiles),
            cell_key_tiles: SlotFreeList::from_slots(cell_key_tiles),
        };
        (mirror, payload)
    }

    /// One MIXED brick's per-voxel cell-key tile by its record's `cell_key_slot` — the CPU
    /// read of the material side atlas (the sink that samples it on the GPU is a later slice).
    /// A freed/dead slot yields its stale tile (unreachable from any live record).
    pub fn cell_key_tile(&self, cell_key_slot: u32) -> &BrickCellKeyTile {
        &self.cell_key_tiles[cell_key_slot]
    }

    /// The material side atlas's slot high-water mark (live + freed cell-key slots) — the
    /// pool's own growth signal, independent of the occupancy atlas's.
    pub fn cell_key_slot_high_water(&self) -> usize {
        self.cell_key_tiles.len()
    }

    /// The MATERIAL SIDE ATLAS's tile geometry, derived from ITS OWN slot high-water mark
    /// exactly as [`pack_cell_key_atlas`] would — the twin of
    /// [`atlas_geometry`](Self::atlas_geometry) for the second pool (the patch seam's
    /// slot-origin inputs, without materialising a build).
    pub fn cell_key_atlas_geometry(&self) -> SculptedCellKeyAtlasGeometry {
        let bricks_per_axis = CubeTilePacking::tiles_per_axis(self.cell_key_tiles.len());
        SculptedCellKeyAtlasGeometry {
            bricks_per_axis,
            atlas_dim_voxels: bricks_per_axis * self.brick_edge_voxels,
            brick_edge_voxels: self.brick_edge_voxels,
        }
    }

    /// One cell-key slot's `2 · edge³` little-endian u16 texel bytes — the DIRTY-SLOT upload
    /// the incremental patch writes into the side atlas, straight from the owning tile (no
    /// whole-atlas re-pack). A freed/dead slot yields its stale bytes (unreachable, never
    /// uploaded).
    pub fn cell_key_slot_bytes(&self, cell_key_slot: u32) -> Vec<u8> {
        self.cell_key_tiles[cell_key_slot].to_le_bytes()
    }

    /// Materialise the full MATERIAL SIDE ATLAS as a [`SculptedCellKeyAtlasPayload`] — the
    /// second pool's wholesale re-pack, done only on a side-atlas GROW
    /// ([`BrickFieldUpdate::cell_key_atlas_grew`]) where every cell-key slot's 3D position
    /// moved. Reuses [`pack_cell_key_atlas`], so it stays byte-identical to
    /// [`to_build`](Self::to_build)'s + [`BrickFieldBuild::cell_key_atlas_payload`]'s.
    pub fn pack_cell_key_atlas_payload(&self) -> SculptedCellKeyAtlasPayload {
        let (bricks_per_axis, atlas_dim_voxels, bytes) =
            pack_cell_key_atlas(self.cell_key_tiles.as_slice(), self.brick_edge_voxels);
        SculptedCellKeyAtlasPayload {
            bytes,
            geometry: SculptedCellKeyAtlasGeometry {
                bricks_per_axis,
                atlas_dim_voxels,
                brick_edge_voxels: self.brick_edge_voxels,
            },
            cell_key_slot_count: self.mixed_brick_count() as u32,
        }
    }

    /// How many LIVE records are MIXED sculpted bricks (== live cell-key tiles).
    pub fn mixed_brick_count(&self) -> usize {
        self.records
            .iter()
            .filter(|record| record.payload.cell_key_slot().is_some())
            .count()
    }

    /// The live records — the sorted [`BrickRecord`] array the GPU record pack + the
    /// pyramid derive from. The mirror is the single CPU owner (item 9): the renderer's
    /// install/patch seams read records straight from here, never via [`to_build`](Self::to_build).
    pub fn records(&self) -> &[BrickRecord] {
        &self.records
    }

    /// How many live records are sculpted bricks — uniform AND mixed (mirror of
    /// [`BrickFieldBuild::sculpted_brick_count`]) — the wholesale install's slot count.
    pub fn sculpted_brick_count(&self) -> usize {
        self.records
            .iter()
            .filter(|record| record.payload.occupancy_atlas_slot().is_some())
            .count()
    }

    /// The sculpted atlas's tile geometry, derived from the slot high-water mark exactly as
    /// [`pack_sculpted_atlas`] would — the frame scalars + slot-origin inputs the patch seam
    /// needs without materialising a build.
    pub fn atlas_geometry(&self) -> SculptedAtlasGeometry {
        let bricks_per_axis = sculpted_atlas_bricks_per_axis(self.slot_tiles.len());
        SculptedAtlasGeometry {
            bricks_per_axis,
            atlas_dim_voxels: bricks_per_axis * self.brick_edge_voxels,
            brick_edge_voxels: self.brick_edge_voxels,
        }
    }

    /// One slot's `edge³` occupancy bytes (bit tile → R8 bytes, O(brick)) — the DIRTY-SLOT
    /// upload the incremental patch writes, straight from the owning tile (no whole-atlas
    /// re-pack). A freed/dead slot yields its stale bytes (unreachable, never uploaded).
    pub fn sculpted_slot_bytes(&self, slot: u32) -> Vec<u8> {
        self.slot_tiles[slot].expand_to_bytes(SCULPTED_BRICK_OCCUPIED)
    }

    /// Materialise the full atlas as a [`SculptedAtlasPayload`] — the ONE legitimate
    /// wholesale re-pack, done only on an atlas GROW (`BrickFieldUpdate::atlas_grew`) where
    /// every slot's 3D position moved. Reuses [`pack_sculpted_atlas`] so it stays
    /// byte-identical to [`to_build`](Self::to_build)'s atlas.
    pub fn pack_atlas_payload(&self) -> SculptedAtlasPayload {
        let (bricks_per_axis, atlas_dim_voxels, bytes) =
            pack_sculpted_atlas(self.slot_tiles.as_slice(), self.brick_edge_voxels);
        SculptedAtlasPayload {
            bytes,
            geometry: SculptedAtlasGeometry {
                bricks_per_axis,
                atlas_dim_voxels,
                brick_edge_voxels: self.brick_edge_voxels,
            },
            sculpted_slot_count: self.sculpted_brick_count() as u32,
        }
    }

    /// The brick edge (voxels_per_block) the field is bound to.
    pub fn brick_edge_voxels(&self) -> u32 {
        self.brick_edge_voxels
    }

    /// The live record count (coarse + sculpted).
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// The atlas slot high-water mark (live + freed slots) — the tile count the atlas is
    /// sized to address. `>= ` the live sculpted count (holes from freed slots).
    pub fn slot_high_water(&self) -> usize {
        self.slot_tiles.len()
    }

    /// Re-evaluate ONLY the blocks of the dirty chunks (plus, for occlusion verdicts, their
    /// 26-neighbourhood ring) and merge them into the field.
    ///
    /// * `fresh_chunks` — the FULL current covering set (dirty chunks freshly resolved,
    ///   clean chunks reused verbatim). Only the dirty chunks + their ring are read.
    /// * `dirty_chunks` — the chunk coords the edit invalidated
    ///   ([`TwoLayerResidentCache::invalidate_aabb`](evaluation::two_layer_store::TwoLayerResidentCache::invalidate_aabb)
    ///   evicted). Every OCCUPANCY change lives in one of these; a block's record content
    ///   (key, material, seam flags, occupancy) is intrinsic to its own chunk.
    ///
    /// **The occlusion dilation (ADR 0011 interior elision — the tricky seam).** Under the
    /// surface-only record contract, whether a coarse block emits a record at all depends on
    /// its six FACE-NEIGHBOURS — which may live in an adjacent, NON-dirty chunk. An edit can
    /// therefore flip records in the 1-chunk dilation of the dirty set: carving a hole
    /// exposes previously-interior blocks of the neighbour chunk (their records must appear),
    /// and filling can occlude previously-surface blocks (their records must vanish). So the
    /// re-mask covers the dirty set DILATED by the 26-neighbourhood (the same dilation the
    /// mesh's cross-chunk seam culling uses; face-dilation would suffice for the 6-neighbour
    /// test, the 26-ring is the conservative shared convention):
    ///
    /// * **dirty chunks** — all records dropped (sculpted slots freed) and rebuilt from the
    ///   fresh data, exactly as before, with occlusion fused in.
    /// * **ring chunks** (dilated \ dirty) — their DATA is unchanged, only occlusion verdicts
    ///   of their COARSE blocks can flip: coarse records are dropped and re-derived against
    ///   the fresh oracle. Sculpted records (and their atlas slots) are KEPT untouched —
    ///   occupancy didn't change, so the per-edit atlas write-set stays ∝ the dirty region
    ///   (the `one_chunk_edit_writes_only_that_chunks_slots` guarantee). Coarse records carry
    ///   no slot, so the ring contributes zero atlas traffic.
    /// * **outside the dilation** — a block's verdict reads only its own chunk + face
    ///   neighbours, all unchanged, so its record is provably identical; kept verbatim.
    ///
    /// Byte-equality vs a from-scratch surface-only [`build_brick_field`] after every edit is
    /// the acceptance bar (`incremental_dirty_update_equals_wholesale_after_every_step`, the
    /// cross-chunk carve case, and the gpu_parity render gate). Returns the
    /// [`BrickFieldUpdate`] describing exactly which slots were touched (the GPU patch's
    /// work-list).
    pub fn apply_dirty_update(
        &mut self,
        fresh_chunks: &[([i32; 3], Arc<TwoLayerChunk>)],
        dirty_chunks: &[[i32; 3]],
    ) -> BrickFieldUpdate {
        let edge = self.brick_edge_voxels;
        let dirty: std::collections::BTreeSet<[i32; 3]> = dirty_chunks.iter().copied().collect();
        // The occlusion ring: the dirty set's 26-neighbourhood minus the dirty set itself.
        let mut ring: std::collections::BTreeSet<[i32; 3]> = std::collections::BTreeSet::new();
        for coord in &dirty {
            for offset_z in -1i32..=1 {
                for offset_y in -1i32..=1 {
                    for offset_x in -1i32..=1 {
                        let neighbour =
                            [coord[0] + offset_x, coord[1] + offset_y, coord[2] + offset_z];
                        if !dirty.contains(&neighbour) {
                            ring.insert(neighbour);
                        }
                    }
                }
            }
        }
        let previous_bricks_per_axis = sculpted_atlas_bricks_per_axis(self.slot_tiles.len());
        let previous_cell_key_bricks_per_axis =
            CubeTilePacking::tiles_per_axis(self.cell_key_tiles.len());

        // 1. Drop every previous record whose block is in a dirty chunk (freeing its slot),
        //    and every COARSE record of a ring chunk (its occlusion verdict may have flipped;
        //    ring SCULPTED records are kept — their chunk's data is unchanged, so record and
        //    slot are still exact, and the atlas is never touched for the ring).
        let mut freed_slots = Vec::new();
        // The MIXED bricks' cell-key slots freed alongside — the second, independent pool
        // (a block that stops being mixed — or stops existing — releases its material tile;
        // a block that BECOMES mixed allocates one below).
        let mut freed_cell_key_slots = Vec::new();
        self.records.retain(|record| {
            let chunk =
                chunk_coord_of_world_block(unpack_world_block_key(record.packed_world_block_key));
            if dirty.contains(&chunk) {
                if let Some(atlas_slot) = record.payload.occupancy_atlas_slot() {
                    freed_slots.push(atlas_slot);
                }
                if let Some(cell_key_slot) = record.payload.cell_key_slot() {
                    freed_cell_key_slots.push(cell_key_slot);
                }
                false
            } else if ring.contains(&chunk) {
                // Ring SCULPTED records (uniform or mixed) are kept verbatim — their chunk's
                // data is unchanged, so both their slots stay exact and neither pool is touched.
                record.payload.occupancy_atlas_slot().is_some()
            } else {
                true
            }
        });
        // Freed slots return to the pool; the free-list keeps them sorted/deduped so reuse is
        // deterministic (largest free index first). This is a nicety for test readability, not
        // correctness: incremental and wholesale agree only up to slot RENUMBERING (the parity
        // oracle compares atlas BYTES, not slot numbers — see `IncrementalBrickField`'s records
        // doc and `incremental_matches_wholesale`), so the reuse order never affects byte parity.
        self.slot_tiles.free(freed_slots.iter().copied());
        self.cell_key_tiles.free(freed_cell_key_slots.iter().copied());

        // 2. Rebuild the dirty chunks' records fully — and the ring chunks' COARSE records —
        //    from the FRESH data, with occlusion verdicts from the fresh oracle (the same
        //    fused elision `build_brick_field` performs, so incremental == wholesale stays
        //    structural).
        let oracle = BrickOcclusionOracle::new(fresh_chunks);
        let mut written_slots = Vec::new();
        // The side atlas's own write-list: the cell-key slots the (re)emitted MIXED bricks took.
        let mut written_cell_key_slots = Vec::new();
        for (chunk_coord, chunk) in fresh_chunks {
            let chunk_is_dirty = dirty.contains(chunk_coord);
            if !chunk_is_dirty && !ring.contains(chunk_coord) {
                continue;
            }
            // Interior-chunk fast path, exactly as the wholesale build: an all-interior
            // chunk emits nothing (it has no microblocks, so no sculpted record is skipped).
            if oracle.chunk_is_all_interior(*chunk_coord) {
                continue;
            }
            let occlusion = oracle.context_for_chunk(*chunk_coord, chunk.as_ref());
            for block_z in 0..CHUNK_BLOCKS {
                for block_y in 0..CHUNK_BLOCKS {
                    for block_x in 0..CHUNK_BLOCKS {
                        let block = [block_x, block_y, block_z];
                        let world_block = [
                            chunk_coord[0] as i64 * CHUNK_BLOCKS as i64 + block_x as i64,
                            chunk_coord[1] as i64 * CHUNK_BLOCKS as i64 + block_y as i64,
                            chunk_coord[2] as i64 * CHUNK_BLOCKS as i64 + block_z as i64,
                        ];
                        match classify_block_brick(chunk, block, world_block, edge) {
                            BlockBrick::Air => {}
                            BlockBrick::Coarse(record) => {
                                if !occlusion.coarse_block_occluded(block) {
                                    self.records.push(record);
                                }
                            }
                            BlockBrick::Sculpted {
                                material_id,
                                overlay,
                                seam_solidity,
                                tile,
                                cell_keys,
                            } => {
                                // Ring chunks keep their existing sculpted records (data
                                // unchanged); only a DIRTY chunk re-allocates and rewrites.
                                if !chunk_is_dirty {
                                    continue;
                                }
                                let slot = self.slot_tiles.allocate(tile);
                                written_slots.push(slot);
                                // A MIXED block allocates from the SEPARATE cell-key pool
                                // (its own free-list, its own high-water mark); a uniform
                                // block takes no material slot at all.
                                let payload = match cell_keys {
                                    None => BrickPayload::Sculpted { atlas_slot: slot },
                                    Some(cell_key_tile) => {
                                        let cell_key_slot =
                                            self.cell_key_tiles.allocate(cell_key_tile);
                                        written_cell_key_slots.push(cell_key_slot);
                                        BrickPayload::SculptedMixed {
                                            atlas_slot: slot,
                                            cell_key_slot,
                                        }
                                    }
                                };
                                self.records.push(BrickRecord {
                                    packed_world_block_key: pack_world_block_key(world_block),
                                    material_id,
                                    overlay,
                                    payload,
                                    seam_solidity,
                                });
                            }
                        }
                    }
                }
            }
        }

        // 3. Re-sort (O(n log n) over records — trivially small next to atlas work).
        self.records
            .sort_unstable_by_key(|record| record.packed_world_block_key);
        debug_assert!(
            self.records
                .windows(2)
                .all(|pair| pair[0].packed_world_block_key < pair[1].packed_world_block_key),
            "brick keys must stay unique + sorted after an incremental merge"
        );

        let atlas_grew =
            sculpted_atlas_bricks_per_axis(self.slot_tiles.len()) != previous_bricks_per_axis;
        // The side atlas grows on ITS OWN slot count — a mixed brick appearing can move every
        // cell-key tile without the occupancy grid moving at all (and vice versa).
        let cell_key_atlas_grew = CubeTilePacking::tiles_per_axis(self.cell_key_tiles.len())
            != previous_cell_key_bricks_per_axis;
        BrickFieldUpdate {
            written_slots,
            freed_slots,
            atlas_grew,
            written_cell_key_slots,
            freed_cell_key_slots,
            cell_key_atlas_grew,
        }
    }

    /// Materialise the current field as a [`BrickFieldBuild`] (records + packed atlas).
    ///
    /// **Parity-oracle materialisation ONLY (item 9).** No production / per-frame path may
    /// call this: it clones ALL records and re-packs the ENTIRE flat atlas blob, the exact
    /// cost the single-owner rework removed from the per-edit patch path. The renderer's
    /// install/patch seams now read records / atlas geometry / dirty-slot bytes straight
    /// from the mirror ([`records`](Self::records), [`atlas_geometry`](Self::atlas_geometry),
    /// [`sculpted_slot_bytes`](Self::sculpted_slot_bytes), [`pack_atlas_payload`](Self::pack_atlas_payload)).
    /// This survives as the parity gate's witness — `to_build() == build_brick_field(...)`
    /// after every edit is the G3 acceptance bar. The atlas is sized to the slot high-water
    /// mark (live + freed holes), so a live record's slot bytes are always in range.
    pub fn to_build(&self) -> BrickFieldBuild {
        let (bricks_per_axis, atlas_dim_voxels, sculpted_atlas_bytes) =
            pack_sculpted_atlas(self.slot_tiles.as_slice(), self.brick_edge_voxels);
        BrickFieldBuild {
            brick_records: self.records.clone(),
            sculpted_atlas_bytes,
            // Cell-key tiles in SLOT order (freed holes included, exactly as the occupancy
            // atlas is packed over its high-water mark): a live record's `cell_key_slot`
            // indexes this vec, and a dead slot's tile is unreachable garbage.
            cell_key_tiles: self.cell_key_tiles.as_slice().to_vec(),
            brick_edge_voxels: self.brick_edge_voxels,
            bricks_per_axis,
            atlas_dim_voxels,
        }
    }
}
