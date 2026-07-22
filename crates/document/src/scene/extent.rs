//! Units, extent and measurement: block-extent regions, the per-node
//! [`NodeTransform`] placement (with retained parametric measurements, ADR 0003
//! §3f(0)), and the block/voxel bounding-box derivations that drive extent,
//! recentring and region sizing.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use substrate::spatial::LatticeOrientation;

use voxel_core::units::{ExactRational, Measurement};

use super::producers::{outset_voxels_at, LeafBody};
use super::*;
use voxel_core::voxel::RecentreVoxels;

/// The working volume the scene resolves into, expressed in **whole blocks**
/// (ADR 0001 "Scale": the canvas is the user-set stock / build volume). The whole
/// extent always resolves as a single region — for a multi-node scene this is the
/// union of every placed leaf's block extent (`Scene::full_extent_blocks`), not
/// just a lone node's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionBlocks {
    /// Size of the region in whole blocks (X, Y, Z).
    pub size_blocks: [u32; 3],
}

impl RegionBlocks {
    /// A region of the given whole-block size.
    pub fn new(size_blocks: [u32; 3]) -> Self {
        Self { size_blocks }
    }
}
/// A node's LOCAL placement. v1 exposes translation only, but the type targets a
/// full affine (translation + rotation + scale) so rotation / scale (with voxel
/// resampling) slot in later without a rewrite (ADR 0001 decision 3). The offset
/// is the live placement field every node authors through (`SetOffset`,
/// `NodeTransform::from_blocks` / `from_measurements`) — nothing pins it to zero.
///
/// NOT `Copy`: it owns an optional boxed retained-measurement expression (the
/// parametric units layer, ADR 0003 §3f(0)), so it is `Clone` only. The canonical
/// `offset_voxels` is read by-field everywhere; the few sites that moved a whole
/// transform out of a `&Node` now `.clone()` it.
// ADR 0027: the continuous rotation (a `Quat`) and the float local offset make this
// type float-bearing, so it can no longer derive `Eq` (only `PartialEq`). Every type
// that contains a `NodeTransform` and derived `Eq` loses it too — none are used as hash
// keys, so this is a marker-trait removal with no behavioural effect.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct NodeTransform {
    /// Translation in **voxels** at the document's density `d`
    /// ([`Scene::voxels_per_block`]) — the single canonical placement field
    /// (ADR 0003 §3f(0)). The planning unit is the voxel; "blocks" are a DERIVED
    /// overlay (a grid line every `d` voxels), exposed via the [`blocks`] /
    /// [`block_aligned`] accessors below — **not** a stored field. Sub-block
    /// placement (an offset not divisible by `d`) is the kit-authoring primitive;
    /// inter-part mating stays block-aligned via `offset_voxels % d == 0`.
    ///
    /// **64-bit world addressing (S4a, ADR 0002 Decision 2):** the offset is `i64`
    /// so far-apart nodes compose down the tree without overflow (a node placed at
    /// ±10⁹ blocks, or a deep nest summing past the i32 range, is exact). It enters
    /// the i64 placement sum at resolve as-is, with no rounding (the resolved grid
    /// *is* `d`).
    ///
    /// [`blocks`]: NodeTransform::blocks
    /// [`block_aligned`]: NodeTransform::block_aligned
    #[serde(default)]
    pub offset_voxels: [i64; 3],

    /// The node's **lattice orientation** (ADR 0026) — one of the 24 axis-aligned cube
    /// rotations, how the node is turned in its parent's frame. Identity for every node until
    /// placement stands one against a geometry face (a cylinder on a wall lies on its side).
    /// The discrete, lattice-exact first half of the affine ADR 0001 reserved; a continuous
    /// *rotation* (with resampling) is the deferred second half.
    ///
    /// Reaches the document, so it is versioned: an old scene without the field loads as
    /// identity (`serde(default)`). Substrate is serde-free (its boundary law), so the on-disk
    /// form travels through the domain [`orientation_serde`](crate::orientation_serde) adapter over the type's stable
    /// gather codec.
    #[serde(default, with = "crate::orientation_serde")]
    pub orientation: LatticeOrientation,

    /// The **continuous local position**, in voxels, relative to the integer
    /// [`offset_voxels`](Self::offset_voxels) wandering origin (ADR 0027). The field's
    /// world position is `offset_voxels + offset_local_voxels` per axis — the integer
    /// part is the far-world-safe anchor (ADR 0008's carried frame), the float part is
    /// the sub-voxel / continuous slide a NoSnap placement authors. Zero for every
    /// voxel-snapped placement (the default), so a snapped scene resolves exactly as an
    /// integer offset does.
    ///
    /// **Wandering origin (deferred):** when this grows past a rebase threshold it folds
    /// into `offset_voxels` so the float never holds a large magnitude far from origin;
    /// v1 scenes are small enough that the fold is not yet wired.
    ///
    /// Reaches the document, so it is versioned: an old scene without the field loads as
    /// `[0.0, 0.0, 0.0]` (`serde(default)`), byte-identical to a pure integer placement.
    #[serde(default)]
    pub offset_local_voxels: [f32; 3],

    /// The node's **continuous rotation** (ADR 0027), stored as a quaternion `[x, y, z, w]`.
    /// `None` = identity (the node stands unturned), which keeps the common case
    /// pointer-small and loads an old document — predating the field — as upright.
    ///
    /// This is the general affine rotation ADR 0001 decision 3 reserved and ADR 0026
    /// deferred behind the word *rotation*; it **subsumes** the discrete
    /// [`orientation`](Self::orientation) (a lattice turn is just a rotation that lands on
    /// the exact classifier path). During the ADR 0027 migration both fields coexist —
    /// `orientation` is retired once every consumer reads the quaternion. Because a
    /// rotation is an isometry it preserves a field's Lipschitz bound, so per-voxel
    /// occupancy stays exact under it; only a non-axis turn loosens the block interval
    /// bound.
    ///
    /// glam's `serde` feature is off in this crate (its math stays serde-free, the
    /// boundary law), so the quaternion travels as a plain `[f32; 4]`; read it as a
    /// [`Quat`] via [`rotation`](Self::rotation).
    #[serde(default)]
    pub rotation_quaternion: Option<[f32; 4]>,

    /// The RETAINED authored unit expression per axis (ADR 0003 §3f(0)).
    ///
    /// `offset_voxels` stays the canonical source of truth for ALL geometry /
    /// resolve; this is the parametric expression the user typed (e.g. `"3.5
    /// blocks"`), kept ALONGSIDE the voxels so a later density re-target is
    /// lossless (the same measurement re-evaluates at the new `d`). It is NOT read
    /// by resolve — only by the inspector (seed/undo) and a future density change.
    ///
    /// **Versioning:** `#[serde(default)]` makes this `None` on an OLD scene that
    /// predates the field, so old documents still load. The accessor
    /// [`offset_measurements`](NodeTransform::offset_measurements) SYNTHESISES a
    /// pure-voxel measurement from `offset_voxels` when this is `None`, so the
    /// retained expression is always correct (just non-parametric — a whole-voxel
    /// count — for a placement authored before the field existed or via a path
    /// that has no expression, e.g. a drag gizmo).
    ///
    /// **Boxed** so the common (`None`) case keeps [`NodeTransform`] pointer-small:
    /// three `Measurement`s are ~120 bytes, which would otherwise bloat every
    /// `Node` (and the arena's `Leaf(Node)` variant). The box is allocated only
    /// when a real authored block expression is retained. `serde` treats
    /// `Option<Box<T>>` transparently, so the on-disk shape is unchanged (`null`
    /// or the three-measurement array).
    #[serde(default)]
    pub(super) offset_measurements: Option<Box<[Measurement; 3]>>,
    // future: rotation, scale → a general affine.
}

impl NodeTransform {
    /// The identity transform (zero offset) — the only transform step 1 uses.
    pub fn identity() -> Self {
        Self::default()
    }

    /// Build a transform from a whole-**block** translation at density
    /// `voxels_per_block` (`offset_voxels = blocks · d`). The block-valued
    /// convenience constructor used by demos, tests and `GroupSpec` placement
    /// (ADR 0003 §3f(0)). The inspector's Offset path now authors through
    /// [`from_measurements`](NodeTransform::from_measurements) (blocks + voxels);
    /// this remains the terse whole-block entry point. It retains each axis as a
    /// whole-block measurement, so a later density re-target scales it losslessly.
    pub fn from_blocks(blocks: [i64; 3], voxels_per_block: u32) -> Self {
        // Clamp density to ≥1 like every resolve site, so a 0-density doc can't
        // multiply placement to zero / mis-scale.
        let density = voxels_per_block.max(1) as i64;
        let offset_voxels = [blocks[0] * density, blocks[1] * density, blocks[2] * density];
        // Retain a whole-BLOCK measurement per axis (no voxel remainder), so a later
        // density re-target scales the block count losslessly — but normalise the
        // all-zero case to `None` so a zero placement matches a fresh identity.
        let measurements = [
            Measurement::new(ExactRational::from_integer(blocks[0] as i128), 0),
            Measurement::new(ExactRational::from_integer(blocks[1] as i128), 0),
            Measurement::new(ExactRational::from_integer(blocks[2] as i128), 0),
        ];
        Self {
            offset_voxels,
            offset_measurements: Self::retained_or_none(measurements, offset_voxels),
            ..Default::default()
        }
    }

    /// Build a transform from a raw canonical **voxel** offset (ADR 0008), retaining
    /// NO parametric block expression — the placement door for a picked cursor drop
    /// (`Intent::PlaceNode`), a drag gizmo, or any path whose offset is a whole-voxel
    /// count with no authored `blocks` term.
    ///
    /// The retained measurement is left `None`: a pure-voxel offset synthesises its
    /// own measurement back from `offset_voxels` (see
    /// [`offset_measurements`](Self::offset_measurements)), so storing one would be a
    /// redundant husk. This normalises the all-zero case to a fresh identity exactly
    /// as [`from_blocks`](Self::from_blocks) does — a placement at `[0, 0, 0]` is
    /// byte-identical to [`identity`](Self::identity).
    pub fn from_offset_voxels(offset_voxels: [i64; 3]) -> Self {
        Self {
            offset_voxels,
            offset_measurements: None,
            ..Default::default()
        }
    }

    /// This transform with its **orientation** replaced (ADR 0026). The placement door
    /// (`Intent::PlaceNode`) calls this to stand a node against the face it was dropped on,
    /// leaving the offset it was built with. Chainable after
    /// [`from_offset_voxels`](Self::from_offset_voxels).
    pub fn with_orientation(mut self, orientation: LatticeOrientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// This node's continuous rotation as a [`Quat`] (ADR 0027) — identity when the
    /// stored quaternion is `None` (an unturned node, or an old document predating the
    /// field). The document-side read of the serde-free `[f32; 4]` storage.
    pub fn rotation(&self) -> Quat {
        self.rotation_quaternion
            .map(Quat::from_array)
            .unwrap_or(Quat::IDENTITY)
    }

    /// This transform with its continuous [`rotation`](Self::rotation) replaced
    /// (ADR 0027). The quaternion is normalised before storage; a rotation within `f32`
    /// epsilon of identity is stored as `None`, keeping an unturned placement in the
    /// canonical (old-document-identical) form so apply→undo→apply is byte-stable.
    pub fn with_rotation(mut self, rotation: Quat) -> Self {
        let rotation = rotation.normalize();
        self.rotation_quaternion =
            (!is_identity_rotation(rotation)).then(|| rotation.to_array());
        self
    }

    /// The field's **world position in voxels** as a continuous value (ADR 0027): the
    /// integer [`offset_voxels`](Self::offset_voxels) wandering origin plus the float
    /// [`offset_local_voxels`](Self::offset_local_voxels) local slide, per axis. The
    /// integer part is added first (far-world-safe), then the small float — so precision
    /// is spent near the origin, never on a large magnitude.
    pub fn world_field_position_voxels(&self) -> [f32; 3] {
        std::array::from_fn(|axis| self.offset_voxels[axis] as f32 + self.offset_local_voxels[axis])
    }

    /// Build a transform from a per-axis authored [`Measurement`] at density
    /// `voxels_per_block` (ADR 0003 §3f(0)). The canonical voxel offset is DERIVED
    /// via [`Measurement::to_voxels`]; the measurements are RETAINED for lossless
    /// density re-targeting and for the inspector to seed/undo the exact authored
    /// expression.
    ///
    /// **Self-consistency invariant:** the result NEVER carries a retained
    /// measurement that disagrees with `offset_voxels`. On the UI path every axis
    /// lands on a whole voxel (the inspector validates before emitting), so the
    /// authored measurement is kept verbatim. On the LOSSY density-retarget path
    /// (`SetDensity` re-evaluating an expression at a `d` it no longer divides
    /// cleanly, e.g. `3.5 blocks` at `d = 15`), the offending axis is floored to a
    /// whole voxel AND its retained measurement is RESYNTHESISED to the pure-voxel
    /// form of that floored value — so the canonical voxels and the retained
    /// expression always agree (the block-term parametricity is lost for that axis,
    /// which is the honest outcome of a non-dividing re-target). Landing axes keep
    /// their authored (block-parametric) measurement.
    pub fn from_measurements(measurements: [Measurement; 3], voxels_per_block: u32) -> Self {
        // Per axis, derive the voxel count AND the measurement to retain. A landing
        // axis keeps its authored measurement; a non-landing axis floors and
        // resynthesises to the pure-voxel form of the floored value so the two can
        // never disagree.
        let resolve_axis = |measurement: Measurement| -> (i64, Measurement) {
            match measurement.to_voxels(voxels_per_block) {
                Ok(voxels) => (voxels, measurement),
                Err(voxel_core::units::MeasurementError::BlockTermNotWholeVoxels {
                    nearest_floor_voxels,
                    ..
                }) => (nearest_floor_voxels, Measurement::from_voxels(nearest_floor_voxels)),
                Err(voxel_core::units::MeasurementError::ZeroDensity) => {
                    let voxels = measurement.voxel_term();
                    (voxels, Measurement::from_voxels(voxels))
                }
            }
        };
        let (voxels_x, retained_x) = resolve_axis(measurements[0]);
        let (voxels_y, retained_y) = resolve_axis(measurements[1]);
        let (voxels_z, retained_z) = resolve_axis(measurements[2]);
        let offset_voxels = [voxels_x, voxels_y, voxels_z];
        let retained = [retained_x, retained_y, retained_z];
        Self {
            offset_voxels,
            offset_measurements: Self::retained_or_none(retained, offset_voxels),
            ..Default::default()
        }
    }

    /// Normalise the retained measurements to `None` when they carry NO parametric
    /// content beyond the derived voxel count — i.e. every axis is exactly the
    /// pure-voxel measurement [`Measurement::from_voxels`] of its derived voxels.
    /// This keeps a placement with no real authored block expression (a zero
    /// offset, a drag, a `from_voxels` round-trip) in the SAME canonical form as a
    /// freshly-built / freshly-loaded transform (`None`), so apply→undo→apply is
    /// byte-identical and serde does not gain a redundant `Some([...])` husk. A
    /// real block expression (e.g. `3 blocks`, `3.5 blocks`) does NOT synthesise
    /// from its voxel count, so it is retained as `Some` for lossless re-targeting.
    fn retained_or_none(
        measurements: [Measurement; 3],
        offset_voxels: [i64; 3],
    ) -> Option<Box<[Measurement; 3]>> {
        let is_synthesisable = (0..3).all(|axis| {
            measurements[axis] == Measurement::from_voxels(offset_voxels[axis])
        });
        if is_synthesisable {
            None
        } else {
            Some(Box::new(measurements))
        }
    }

    /// The RETAINED per-axis authored measurement (ADR 0003 §3f(0)).
    ///
    /// When the placement carries no stored expression (an OLD scene predating the
    /// field, or a transform built without one), this SYNTHESISES a pure-voxel
    /// measurement equal to `offset_voxels` per axis — correct (it re-evaluates
    /// back to the same voxels at any density), just non-parametric for a block
    /// re-target. The canonical `offset_voxels` always wins for geometry; this is
    /// retention/display only.
    pub fn offset_measurements(&self) -> [Measurement; 3] {
        match &self.offset_measurements {
            Some(measurements) => **measurements,
            None => [
                Measurement::from_voxels(self.offset_voxels[0]),
                Measurement::from_voxels(self.offset_voxels[1]),
                Measurement::from_voxels(self.offset_voxels[2]),
            ],
        }
    }

    /// Whether this transform carries a GENUINELY retained authored expression
    /// (the stored field is `Some`) versus a placement whose measurement is only
    /// SYNTHESISED from `offset_voxels` (the field is `None` — an old document, a
    /// drag, a pure-voxel offset). The density re-target (`SetDensity`) uses this
    /// to decide between RE-EVALUATING the authored block expression at the new
    /// density (lossless block scaling, exact voxel terms) and the legacy integer
    /// rescale that preserves a non-parametric offset's physical position.
    pub fn has_retained_measurements(&self) -> bool {
        self.offset_measurements.is_some()
    }

    /// The whole-**block** view of this placement (the derived block overlay,
    /// ADR 0003 §3f(0)): the floor of `offset_voxels / d` componentwise (the same
    /// single floor rule the extent derivations use, see
    /// [`world_block_corner_floor`]). EXACT while placement is block-aligned — which
    /// it is today; for future negative sub-voxel offsets the floor is the correct
    /// (block-containing) view.
    pub fn blocks(&self, voxels_per_block: u32) -> [i64; 3] {
        world_block_corner_floor(self.offset_voxels, voxels_per_block)
    }

    /// Whether this placement sits on the whole-block lattice — the connector /
    /// joint mating predicate `offset_voxels % d == 0` per axis (ADR 0003 §3f(0)
    /// / §3i "block-aligned where you mate").
    pub fn block_aligned(&self, voxels_per_block: u32) -> bool {
        // Clamp density to ≥1 so a 0-density doc can't panic on `% 0`.
        let density = voxels_per_block.max(1) as i64;
        self.offset_voxels.iter().all(|&v| v.rem_euclid(density) == 0)
    }
}

/// Whether a unit quaternion is (within `f32` epsilon) the identity rotation
/// (ADR 0027). A quaternion and its negation denote the *same* rotation, so the test
/// compares `|dot(rotation, identity)|` to 1 rather than the raw components — `−IDENTITY`
/// is identity too. Used to keep an unturned placement's stored quaternion `None`.
fn is_identity_rotation(rotation: Quat) -> bool {
    rotation.dot(Quat::IDENTITY).abs() >= 1.0 - 1e-6
}

/// The whole-**block** corner of a world VOXEL offset: `floor(offset_voxels / d)`
/// per axis via `div_euclid` (the codebase convention, e.g. `main.rs`'s
/// `point_add_position_blocks`). The single owner of the voxel→block-corner rule —
/// [`NodeTransform::blocks`] and both extent derivations
/// ([`Scene::placed_extent_blocks`], [`Scene::node_subtree_extent_blocks`]) route
/// through it.
///
/// This is EXACT while placement is block-aligned — which it is today (every offset
/// is a block multiple); Slice 2's sub-voxel placement makes it a truncating
/// (floor) view, correct for the LOW corner of a leaf box but requiring outward
/// (ceil) rounding for the HIGH corner at the call sites (see those).
fn world_block_corner_floor(world_offset_voxels: [i64; 3], voxels_per_block: u32) -> [i64; 3] {
    let density = voxels_per_block.max(1) as i64;
    [
        world_offset_voxels[0].div_euclid(density),
        world_offset_voxels[1].div_euclid(density),
        world_offset_voxels[2].div_euclid(density),
    ]
}

/// The world-axis **voxel** extent of a leaf's corner-anchored local grid `[0, grid_voxels)` after
/// the continuous rotation `rotation` (ADR 0027) — the span the coverage/broadphase walks anchor at
/// each leaf's world offset (`[world_offset, world_offset + this)`). Delegates to
/// [`substrate::spatial::LeafPlacement`] built at world-offset zero (so its AABB is `[0, span)`) —
/// the ONE placement definition the classifier and the dense oracle fold through, so the reserved
/// extent can never diverge from the resampled body (the tube-truncation bug). Axis-aligned rounds
/// (bit-exact with the pre-0027 `turn_extent`); a genuine rotation ceils to conservatively enclose
/// (ADR 0027 §4).
pub(super) fn rotated_grid_extent_voxels(rotation: Quat, grid_voxels: [i64; 3]) -> [i64; 3] {
    let full = Vec3::new(grid_voxels[0] as f32, grid_voxels[1] as f32, grid_voxels[2] as f32);
    substrate::spatial::LeafPlacement::new(rotation, full, Vec3::ZERO).world_aabb().1
}

/// The world-axis **block** extent, the block-frame twin of [`rotated_grid_extent_voxels`] for the
/// whole-block size readouts — the same substrate placement, cast to block units.
pub(super) fn rotated_grid_extent_blocks(rotation: Quat, size_blocks: [u32; 3]) -> [u32; 3] {
    let full = Vec3::new(size_blocks[0] as f32, size_blocks[1] as f32, size_blocks[2] as f32);
    let max = substrate::spatial::LeafPlacement::new(rotation, full, Vec3::ZERO).world_aabb().1;
    [max[0] as u32, max[1] as u32, max[2] as u32]
}

impl Scene {
    /// The per-object **block lattice box** for the node at `path`, in the SAME
    /// recentred render frame the resolved voxels live in (issue #29 S3). Returns
    /// `(min_corner, max_corner)` in voxels.
    ///
    /// The box is the node's voxel AABB **expanded out to enclosing whole blocks** —
    /// i.e. the union of every enabled leaf under the node, each leaf's corner-anchored
    /// voxel span `[off, off + size·density)` grown to whole blocks by FLOORING the low
    /// corner and CEILING the high corner (the split `node_subtree_extent_blocks`
    /// forms), then scaled by `density` and shifted by `− recentre_voxels_for_resolve`.
    /// Because the low corner floors and the high corner ceils INDEPENDENTLY, a
    /// sub-block (1-voxel) translate that crosses a block boundary grows the
    /// enclosing-block box by exactly one whole block — the spec's "a 1-voxel translate
    /// adds/removes a whole block" requirement — and the box always fully contains the
    /// geometry (a non-block-aligned leaf never pokes out of its own cage).
    ///
    /// For a Group / Instance node the box is the union of all leaves under it.
    /// A size-less node (a VoxelBody-only / empty subtree, or a path that descends
    /// through a non-Group) returns `None` — there is no block lattice to draw.
    pub fn node_block_lattice_box_recentred(
        &self,
        path: &NodePath,
        voxels_per_block: u32,
    ) -> Option<([f32; 3], [f32; 3])> {
        let (min_corner, max_corner) = self.node_subtree_extent_blocks(path, voxels_per_block)?;
        let density = voxels_per_block.max(1) as i64;
        let mut min_box = [0.0f32; 3];
        let mut max_box = [0.0f32; 3];
        // Unwrap the carried frame at the recentred block-corner arithmetic.
        let recentre = self.recentre_voxels_for_resolve(voxels_per_block).voxels();
        for axis in 0..3 {
            // Whole-block corners → voxels (exact), then into the recentred frame.
            min_box[axis] = (min_corner[axis] * density - recentre[axis]) as f32;
            max_box[axis] = (max_corner[axis] * density - recentre[axis]) as f32;
        }
        Some((min_box, max_box))
    }

    /// The block-aligned AABB (`min_corner, max_corner`, whole blocks) of the
    /// subtree rooted at `path` — the union of every enabled leaf under that node,
    /// each leaf spanning `[off − floor(size/2), off − floor(size/2) + size)` (the
    /// same split [`placed_extent_blocks`] uses scene-wide). The accumulated world
    /// offset down to `path` seeds the walk so a Group/Instance child is measured at
    /// its world location. `None` when the subtree has no intrinsic-size leaf.
    pub(super) fn node_subtree_extent_blocks(
        &self,
        path: &NodePath,
        voxels_per_block: u32,
    ) -> Option<([i64; 3], [i64; 3])> {
        // Accumulate the world VOXEL offset of every node ABOVE the target (the
        // parent offset), and grab the target node itself. `walk_nodes` below
        // re-adds the target's own offset (also voxels), so we must stop
        // accumulating at its parent. Walk the id-spine for ORDER, fetch content
        // from the arena (ADR 0003 B5).
        let mut siblings: &[NodeId] = &self.roots;
        let mut parent_offset_voxels = [0i64; 3];
        let mut target: Option<&Node> = None;
        for (depth, &index) in path.indices.iter().enumerate() {
            let &child_id = siblings.get(index)?;
            let node = self.arena.get(&child_id)?;
            let is_last = depth + 1 == path.indices.len();
            if is_last {
                target = Some(node);
            } else if let NodeContent::Group(children) = &node.content {
                parent_offset_voxels = [
                    parent_offset_voxels[0] + node.transform.offset_voxels[0],
                    parent_offset_voxels[1] + node.transform.offset_voxels[1],
                    parent_offset_voxels[2] + node.transform.offset_voxels[2],
                ];
                siblings = children;
            } else {
                return None;
            }
        }
        let target = target?;
        if !target.enabled {
            return None;
        }
        let target_id = target.id;

        // Union the leaf boxes under the target. `walk_nodes` adds the target's own
        // voxel offset to `parent_offset_voxels`, giving the leaf its true world
        // location. The single-element id spine carries the target itself (ADR 0003
        // B5).
        let mut min_corner = [i64::MAX; 3];
        let mut max_corner = [i64::MIN; 3];
        let mut any = false;
        let mut def_path: Vec<DefId> = Vec::new();
        let mut scope_path: Vec<ScopeFrame> = Vec::new();
        self.walk_nodes(
            &[target_id],
            parent_offset_voxels,
            [0.0, 0.0, 0.0],
            &mut def_path,
            &mut scope_path,
            &mut |world_offset_voxels, _offset_local_voxels, _orientation, rotation, body, _grid_on_faces, _operation, outset, _scope_path| {
                let outset_voxels = outset_voxels_at(outset, voxels_per_block);
                // The dilated body's low corner sits `N` BELOW the producer's, so the extent
                // must start there — growing the size alone would put a right-sized box in
                // the wrong place (ADR 0008 — the frame is carried).
                let world_offset_voxels: [i64; 3] =
                    std::array::from_fn(|axis| world_offset_voxels[axis] - outset_voxels);
                let Some(size_blocks) = leaf_size_blocks(&body, voxels_per_block, outset_voxels) else {
                    return;
                };
                // ADR 0026: turn the block extent into world axes for an oriented leaf.
                let size_blocks = rotated_grid_extent_blocks(rotation, size_blocks);
                any = true;
                let density = voxels_per_block.max(1) as i64;
                // The leaf's whole-block low corner, via the single floor rule.
                let world_blocks = world_block_corner_floor(world_offset_voxels, voxels_per_block);
                for axis in 0..3 {
                    // Corner-anchored ENCLOSING-block box: the low corner FLOORS the
                    // leaf's low voxel to its block, and the high corner independently
                    // CEILS the leaf's high voxel (`offset + size·density`) to its block.
                    // A leaf that is NOT block-aligned touches one more block than its
                    // block size — flooring low and adding `size_blocks` would instead
                    // slide the whole box toward the low corner and clip the geometry off
                    // the high side (it pokes out of its own grid cage). Ceiling the high
                    // corner is what realises the doc's "a 1-voxel translate that crosses a
                    // block boundary adds a whole block" contract. A block-aligned leaf has
                    // no remainder, so `high == low + size_blocks` exactly (goldens hold).
                    let low = world_blocks[axis];
                    let high_voxel = world_offset_voxels[axis] + size_blocks[axis] as i64 * density;
                    // Ceil to a whole block (signed `div_ceil` is still unstable): for a
                    // positive divisor `ceil(a/d) == −floor(−a/d) == −((−a).div_euclid(d))`.
                    let high = -((-high_voxel).div_euclid(density));
                    min_corner[axis] = min_corner[axis].min(low);
                    max_corner[axis] = max_corner[axis].max(high);
                }
            },
        );
        any.then_some((min_corner, max_corner))
    }

    /// The PRODUCER-TRUE voxel AABB (`min_corner, max_corner`, in voxels) of the
    /// subtree rooted at `path` — the union of every enabled leaf under that node,
    /// each leaf spanning the center-emitted `[off·d − grid/2, off·d + grid/2)` (the
    /// exact frame [`placed_extent_voxels`] forms scene-wide). This is the frame the
    /// composite recentre and the resolved voxels live in, so the gizmo pivot derived
    /// from it lands exactly on the object. `None` when the subtree has no
    /// intrinsic-size leaf. Mirrors [`node_subtree_extent_blocks`] but in voxels with
    /// no block-floor split (so odd sizes are centred, not snapped).
    pub(super) fn node_subtree_extent_voxels(
        &self,
        path: &NodePath,
        voxels_per_block: u32,
    ) -> Option<([i64; 3], [i64; 3])> {
        let mut siblings: &[NodeId] = &self.roots;
        let mut parent_offset_voxels = [0i64; 3];
        let mut target: Option<&Node> = None;
        for (depth, &index) in path.indices.iter().enumerate() {
            let &child_id = siblings.get(index)?;
            let node = self.arena.get(&child_id)?;
            let is_last = depth + 1 == path.indices.len();
            if is_last {
                target = Some(node);
            } else if let NodeContent::Group(children) = &node.content {
                parent_offset_voxels = [
                    parent_offset_voxels[0] + node.transform.offset_voxels[0],
                    parent_offset_voxels[1] + node.transform.offset_voxels[1],
                    parent_offset_voxels[2] + node.transform.offset_voxels[2],
                ];
                siblings = children;
            } else {
                return None;
            }
        }
        let target = target?;
        if !target.enabled {
            return None;
        }
        let target_id = target.id;

        let mut min_corner = [i64::MAX; 3];
        let mut max_corner = [i64::MIN; 3];
        let mut any = false;
        let mut def_path: Vec<DefId> = Vec::new();
        let mut scope_path: Vec<ScopeFrame> = Vec::new();
        self.walk_nodes(
            &[target_id],
            parent_offset_voxels,
            [0.0, 0.0, 0.0],
            &mut def_path,
            &mut scope_path,
            &mut |world_offset_voxels, _offset_local_voxels, _orientation, rotation, body, _grid_on_faces, _operation, outset, _scope_path| {
                let outset_voxels = outset_voxels_at(outset, voxels_per_block);
                // The dilated body's low corner sits `N` BELOW the producer's, so the extent
                // must start there — growing the size alone would put a right-sized box in
                // the wrong place (ADR 0008 — the frame is carried).
                let world_offset_voxels: [i64; 3] =
                    std::array::from_fn(|axis| world_offset_voxels[axis] - outset_voxels);
                let Some(grid_voxels) = body.grid_voxels(voxels_per_block, outset_voxels) else {
                    return;
                };
                // ADR 0026: turn the grid into world axes for an oriented leaf.
                let grid_voxels = rotated_grid_extent_voxels(rotation, grid_voxels);
                any = true;
                for axis in 0..3 {
                    // Corner-anchored span `[off, off + grid)` (offset is the low corner).
                    let grid = grid_voxels[axis];
                    let low = world_offset_voxels[axis];
                    let high = low + grid;
                    min_corner[axis] = min_corner[axis].min(low);
                    max_corner[axis] = max_corner[axis].max(high);
                }
            },
        );
        any.then_some((min_corner, max_corner))
    }

    /// The placed AABB of the ACTIVE selection's subtree in the **recentred voxel
    /// frame** — the frame the display mesher emits vertices in and the layer band
    /// clips within (a voxel at absolute producer coord `a` lands at `a −
    /// recentre_voxels`, ADR 0008). Half-open `[min, max)` per axis.
    ///
    /// This is the region ADR 0018 Decision 5 confines the onion-fog band clip to:
    /// inside it the selected object clips to the band (ghost outside the band),
    /// everything outside renders finished. Selecting the **root part**
    /// ([`ROOT_NODE_ID`]) returns the WHOLE scene's extent (the scene-wide clip,
    /// i.e. the pre-ADR-0018 behaviour). `None` when nothing is selected, the
    /// selection is hidden, or its subtree has no intrinsic extent (a lone
    /// VoxelBody) — the caller then applies no region clip.
    pub fn selected_region_extent_recentred_voxels(
        &self,
        voxels_per_block: u32,
    ) -> Option<([i64; 3], [i64; 3])> {
        let active = self.active?;
        let (min_abs, max_abs) = if active == ROOT_NODE_ID {
            // The root part IS the whole scene (its subtree is every top-level node),
            // and it is not addressable in the `roots` spine, so use the scene-wide
            // producer-true extent directly.
            self.placed_extent_voxels(voxels_per_block)?
        } else {
            let path = self.path_of(active)?;
            self.node_subtree_extent_voxels(&path, voxels_per_block)?
        };
        // Rebase the absolute producer-true corners into the recentred frame (ADR
        // 0008: subtract the composite recentre the resolve applies).
        let recentre = self.recentre_voxels_for_resolve(voxels_per_block).voxels();
        Some((
            [
                min_abs[0] - recentre[0],
                min_abs[1] - recentre[1],
                min_abs[2] - recentre[2],
            ],
            [
                max_abs[0] - recentre[0],
                max_abs[1] - recentre[1],
                max_abs[2] - recentre[2],
            ],
        ))
    }

    /// The whole-block extent of the scene: the per-axis size of the bounding box
    /// that encompasses every placed leaf node (ADR 0001 step 3). Each leaf
    /// occupies `block-offset ± size/2` (its placement's derived block view,
    /// ADR 0003 §3f(0)); the composite extent is the union of
    /// those boxes (`max_corner - min_corner` per axis). With every node at a zero
    /// offset this reduces to the per-axis MAX of the node sizes (the step-2
    /// behaviour). A VoxelBody-only node (the cloud field, which has no intrinsic size)
    /// contributes no box and adopts whatever extent the Tools establish.
    ///
    /// Returns a zero-sized region when no leaf has an intrinsic size.
    pub fn full_extent_blocks(&self, voxels_per_block: u32) -> RegionBlocks {
        match self.placed_extent_blocks(voxels_per_block) {
            Some((min_corner, max_corner)) => RegionBlocks::new([
                (max_corner[0] - min_corner[0]) as u32,
                (max_corner[1] - min_corner[1]) as u32,
                (max_corner[2] - min_corner[2]) as u32,
            ]),
            // NOTE: the corners are `i64` (S4a 64-bit block addressing); the
            // DIFFERENCE (the region size) is bounded by the placed geometry's own
            // extent, never by how far from the origin it sits, so narrowing to u32
            // is safe — a scene whose *span* exceeds 4G blocks is not representable
            // as a single monolithic grid regardless of addressing width.
            None => RegionBlocks::new([0, 0, 0]),
        }
    }

    /// The composite bounding box of all placed leaf nodes, in **whole-block**
    /// coordinates: `(min_corner, max_corner)` where each leaf with intrinsic
    /// `size_blocks` is CORNER-ANCHORED at its block-offset (the derived block view of
    /// its voxel placement, ADR 0003 §3f(0)) and so spans `[offset, offset + size]`.
    /// `None` when no leaf has an intrinsic size (a VoxelBody-only scene). Drives
    /// [`full_extent_blocks`] (the whole-block size readout) and the block-lattice
    /// overlay extent.
    ///
    /// CORNER-ANCHORING: the offset block is the LOW corner (no `± size/2` split), so
    /// the block frame matches the corner-anchored producer voxel frame exactly.
    fn placed_extent_blocks(&self, voxels_per_block: u32) -> Option<([i64; 3], [i64; 3])> {
        let mut min_corner = [i64::MAX; 3];
        let mut max_corner = [i64::MIN; 3];
        let mut any = false;
        self.for_each_leaf(&mut |world_offset_voxels, _offset_local_voxels, _orientation, rotation, body, _grid_on_faces, _operation, outset, _scope_path| {
            let outset_voxels = outset_voxels_at(outset, voxels_per_block);
            let world_offset_voxels: [i64; 3] =
                std::array::from_fn(|axis| world_offset_voxels[axis] - outset_voxels);
            let Some(size_blocks) = leaf_size_blocks(&body, voxels_per_block, outset_voxels) else {
                return;
            };
            // ADR 0026: turn the block extent into world axes, so an oriented leaf's block
            // readout spans the axes it actually occupies.
            let size_blocks = rotated_grid_extent_blocks(rotation, size_blocks);
            any = true;
            let density = voxels_per_block.max(1) as i64;
            // The leaf's whole-block low corner, via the single floor rule.
            let world_blocks = world_block_corner_floor(world_offset_voxels, voxels_per_block);
            for axis in 0..3 {
                // Corner-anchored ENCLOSING-block box: floor the low voxel to its block
                // and CEIL the high voxel to its block INDEPENDENTLY. An off-block leaf
                // touches one more block than its block size — `low + size_blocks` would
                // slide the box toward the low corner and under-report the high side (the
                // same clip fixed in `node_subtree_extent_blocks`). Block-aligned leaves
                // have no remainder, so `high == low + size_blocks` exactly.
                let low = world_blocks[axis];
                let high_voxel = world_offset_voxels[axis] + size_blocks[axis] as i64 * density;
                // Ceil to a whole block (signed `div_ceil` is unstable): `ceil(a/d) ==
                // −((−a).div_euclid(d))` for a positive divisor.
                let high = -((-high_voxel).div_euclid(density));
                min_corner[axis] = min_corner[axis].min(low);
                max_corner[axis] = max_corner[axis].max(high);
            }
        });
        any.then_some((min_corner, max_corner))
    }

    /// The recentre offset (in voxels) that [`resolve_region`] subtracts from every
    /// voxel to centre the composite on the origin. The chunk path does NOT apply
    /// this, so it is the exact translation between the two frames:
    /// `resolve_region.world_position == chunk_path.world_position − recentre_voxels`.
    /// Exposed (crate-internal) so the S0 equivalence tests can normalise one frame
    /// to the other. `[0, 0, 0]` for a scene with no intrinsic-size leaf.
    ///
    /// Returns the RAW triple by rule: its only callers feed it straight into
    /// `occupied_multiset`'s per-voxel rebase arithmetic (a comparison oracle), so the
    /// unwrap belongs here at the arithmetic's edge rather than being pushed into the test.
    #[cfg(test)]
    pub(crate) fn recentre_voxels(&self, voxels_per_block: u32) -> [i64; 3] {
        self.recentre_voxels_for_resolve(voxels_per_block).voxels()
    }

    /// The recentre offset (in voxels) that `resolve_region` subtracts from every
    /// voxel to centre the composite on the origin (issue #27 S2). This is the
    /// SAME computation `resolve_region` inlines; the chunk cache
    /// (`ChunkResolveCache::resolve_region`) calls it to apply
    /// the identical offset when reassembling the recentred monolithic grid from
    /// absolute per-chunk pieces, so the assembled output is bit-identical. `[0, 0,
    /// 0]` for a scene with no intrinsic-size leaf.
    ///
    /// Derived from the **producer-true voxel frame** ([`placed_extent_voxels`]) —
    /// the exact span the producers center-emit — NOT the block-floored frame. This
    /// makes the composite centre coincide with the producers' own centres for ALL
    /// `size·d` parities (including odd size at density 1), so no per-leaf lattice
    /// shift is needed.
    ///
    /// [`placed_extent_voxels`]: Self::placed_extent_voxels
    ///
    /// **The one mint point (ADR 0008 / the frame law).** Returns the recentre already
    /// wrapped as [`RecentreVoxels`] — a build's frame value is born here carrying its
    /// frame, so downstream never re-wraps a raw triple. Consumers that still speak
    /// `[i64; 3]` unwrap with [`RecentreVoxels::voxels`] at their boundary.
    pub fn recentre_voxels_for_resolve(&self, voxels_per_block: u32) -> RecentreVoxels {
        let voxels = match self.placed_extent_voxels(voxels_per_block) {
            // FLOOR division (`div_euclid`), NOT truncation: for an odd composite span
            // `(min + max)` is odd, and `/` rounds toward zero — which biases a
            // negative-X composite the OPPOSITE way from a positive-X one, breaking
            // +X/−X symmetry. `div_euclid(2)` always rounds toward −∞, so the recentre
            // direction is consistent regardless of where the composite sits.
            Some((min_corner, max_corner)) => [
                (min_corner[0] + max_corner[0]).div_euclid(2),
                (min_corner[1] + max_corner[1]).div_euclid(2),
                (min_corner[2] + max_corner[2]).div_euclid(2),
            ],
            None => [0i64; 3],
        };
        RecentreVoxels::new(voxels)
    }

    /// The full composite extent in voxels — the size the whole-region grids
    /// (`resolve_region`, `resolve_region_via_chunks`) are seeded with. The chunk
    /// cache (issue #20 S2) seeds its reassembled grid to the same dimensions.
    ///
    /// **Producer voxel frame (center-anchoring retirement).** This is the EXACT
    /// occupied span `max_v − min_v` from [`placed_extent_voxels`] — NOT
    /// `size_blocks·d`. The region MUST share the placement frame: producers
    /// center-emit and are recentred by `(min_v + max_v)/2` (see
    /// [`recentre_voxels_for_resolve`]); the recentred composite occupies exactly
    /// `[−D/2, D/2)` with `D = max_v − min_v`, so a block-framed region (`size·d`)
    /// would be too SMALL for a parity-mismatched multi-leaf composite and silently
    /// clip voxels off each end. (The whole-block SIZE readout / block-lattice
    /// overlay still read [`full_extent_blocks`] — that is the only legitimate
    /// block-frame consumer.)
    ///
    /// **This IS the size the assembled render grid takes** for a chunkable scene:
    /// both `resolve_region` and the chunk-cache reassembly size their output to
    /// exactly this value (asserted in `placed_region_dimensions_equals_assembled_grid`).
    /// `pub` so the `shot` binary can do the same substitution.
    ///
    /// **Caveat — a VoxelBody-only scene** (no intrinsic-size leaf, e.g. a lone
    /// debug-cloud field) returns `[0, 0, 0]` here because it has no composite
    /// extent; such a scene is resolved through the *explicit-region* monolithic
    /// path (sized to the caller's chosen region, not this), so a consumer of a
    /// VoxelBody-only scene must use that explicit region — not this — as its dimensions.
    ///
    /// [`placed_extent_voxels`]: Self::placed_extent_voxels
    /// [`recentre_voxels_for_resolve`]: Self::recentre_voxels_for_resolve
    /// [`full_extent_blocks`]: Self::full_extent_blocks
    pub fn placed_region_dimensions(&self, voxels_per_block: u32) -> [u32; 3] {
        match self.placed_extent_voxels(voxels_per_block) {
            // The EXACT voxel span (`max − min`). Corner-anchored producers emit
            // half-integer centres, so the region-relative decode
            // (`floor(world − region_low)`, see `resolve_region`) is exact for any
            // span parity — no even-padding is needed.
            Some((min_corner, max_corner)) => [
                (max_corner[0] - min_corner[0]) as u32,
                (max_corner[1] - min_corner[1]) as u32,
                (max_corner[2] - min_corner[2]) as u32,
            ],
            None => [0, 0, 0],
        }
    }

}

/// The whole-block extent of a leaf node's producer, or `None` for a non-leaf /
/// not-yet-implemented content kind.
/// The leaf's whole-block extent, GROWN by its outset (ADR 0019 Decision 7).
///
/// The region sizing must see the DILATED extent: an outset body reaches beyond its
/// producer's own bounds, and a region sized to the undilated extent would clip the
/// dilation away at the composite edge.
fn leaf_size_blocks(
    body: &LeafBody<'_>,
    voxels_per_block: u32,
    outset_voxels: i64,
) -> Option<[u32; 3]> {
    let density = voxels_per_block.max(1);
    // A pre-composed scope reports the extent of its composed body (ADR 0019 Decision 7).
    let content = match body {
        LeafBody::Content(content) => *content,
        LeafBody::Composed { .. } => {
            let grid_voxels = body.grid_voxels(voxels_per_block, outset_voxels)?;
            return Some(std::array::from_fn(|axis| {
                (grid_voxels[axis].max(0) as u32).div_ceil(density)
            }));
        }
    };
    // Grown by `N` on both sides of every axis, in VOXELS, before the round up to whole
    // blocks — rounding first would lose a sub-block outset entirely.
    let grow = |voxels: u32| {
        (voxels as i64 + 2 * outset_voxels).max(0) as u32
    };
    match content {
        // A Tool's size is now voxel-granular (ADR 0003 §3f(0)). The composite region
        // SIZING reports whole blocks, so round the exact voxel span UP to whole
        // blocks (a sub-block remainder still claims its block, exactly like a
        // SketchTool prism) — a whole-block size divides cleanly and is unchanged.
        NodeContent::Tool { shape, .. } => {
            let ceil_blocks = |voxels: u32| grow(voxels).div_ceil(density);
            Some([
                ceil_blocks(shape.size_voxels[0]),
                ceil_blocks(shape.size_voxels[1]),
                ceil_blocks(shape.size_voxels[2]),
            ])
        }
        // A sketch→extrude prism reports its AABB rounded UP to whole blocks so the
        // composite region SIZING (`full_extent_blocks`) sees its extent — exactly
        // like a Tool. The recentre / chunk-coverage / spatial-index use the exact
        // producer voxel frame (`leaf_producer_grid_voxels`) instead.
        NodeContent::SketchTool { producer, .. } => {
            let [grid_x, grid_y, grid_z] = producer.grid_dimensions();
            let ceil_blocks = |voxels: u32| grow(voxels).div_ceil(density);
            Some([
                ceil_blocks(grid_x),
                ceil_blocks(grid_y),
                ceil_blocks(grid_z),
            ])
        }
        // The cloud field has no intrinsic size; today it adopts the shape's grid
        // dimensions, so a step-1 VoxelBody-only scene has no extent of its own. The
        // call sites that resolve a VoxelBody always pass the region explicitly, so
        // this path is unused by them; report whole blocks for completeness.
        NodeContent::VoxelBody(VoxelBody::DebugClouds { .. }) => {
            // A VoxelBody stamped at the app density occupies `dimensions / density`
            // blocks; with no stored body in step 1 it has no size. Returning
            // `None` keeps `full_extent_blocks` deferring to the next leaf.
            let _ = density;
            None
        }
        NodeContent::Group(_) | NodeContent::Instance(_) => None,
    }
}

#[cfg(test)]
mod continuity_schema_tests {
    //! ADR 0027 schema migration: the continuous rotation + local offset must default
    //! to identity so an old (pre-0027) document loads byte-identical to a pure integer
    //! placement, and a rotated placement must survive a JSON round-trip.
    use super::*;

    #[test]
    fn old_document_without_continuity_fields_loads_upright_and_unslid() {
        // A NodeTransform serialised before ADR 0027 carries neither the local offset
        // nor the quaternion. It must deserialise to identity rotation + zero slide.
        let old_json = r#"{ "offset_voxels": [4, -2, 7] }"#;
        let transform: NodeTransform = serde_json::from_str(old_json).unwrap();
        assert_eq!(transform.offset_voxels, [4, -2, 7]);
        assert_eq!(transform.offset_local_voxels, [0.0, 0.0, 0.0]);
        assert_eq!(transform.rotation_quaternion, None);
        assert_eq!(transform.rotation(), Quat::IDENTITY);
        // The pure-integer placement reads back through the continuous accessor exactly.
        assert_eq!(transform.world_field_position_voxels(), [4.0, -2.0, 7.0]);
    }

    #[test]
    fn identity_rotation_is_stored_as_none() {
        // Setting the rotation to identity (or its negation) keeps the canonical `None`
        // form, so an unturned placement never grows a redundant quaternion husk.
        let upright = NodeTransform::from_offset_voxels([0, 0, 0]).with_rotation(Quat::IDENTITY);
        assert_eq!(upright.rotation_quaternion, None);
        let also_upright =
            NodeTransform::from_offset_voxels([0, 0, 0]).with_rotation(-Quat::IDENTITY);
        assert_eq!(also_upright.rotation_quaternion, None);
    }

    #[test]
    fn a_rotated_placement_survives_json_round_trip() {
        let turn = Quat::from_rotation_z(std::f32::consts::FRAC_PI_3); // 60° about +Z
        let transform = NodeTransform::from_offset_voxels([1, 0, 0])
            .with_rotation(turn);
        assert!(transform.rotation_quaternion.is_some());
        let json = serde_json::to_string(&transform).unwrap();
        let restored: NodeTransform = serde_json::from_str(&json).unwrap();
        // Quaternions compare up to sign; the restored rotation is the same turn.
        assert!(restored.rotation().abs_diff_eq(turn, 1e-5) || restored.rotation().abs_diff_eq(-turn, 1e-5));
    }

    #[test]
    fn a_sub_voxel_slide_reads_back_continuously() {
        let mut transform = NodeTransform::from_offset_voxels([10, 0, 0]);
        transform.offset_local_voxels = [0.25, -0.5, 0.0];
        assert_eq!(transform.world_field_position_voxels(), [10.25, -0.5, 0.0]);
    }
}
