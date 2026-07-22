//! A sealed scope evaluated as a single producer, so a whole Part can be outset
//! (ADR 0019 Decision 7, ADR 0017 Decision 3).

use super::{Field, FieldInterval, VoxelProducer};
use crate::scene::CombineOp;
use voxel_core::core_geom::BlockId;
use voxel_core::spatial_index::VoxelAabb;
use voxel_core::voxel::{BlockAttrs, Voxel, VoxelGrid, SURFACE_ISOLEVEL};

/// One member of a [`CompositeProducer`], in the composite's own `[0, full_dim)` frame.
pub struct CompositeMember {
    /// The member's low corner relative to the composite's frame origin.
    pub offset_voxels: [i64; 3],
    /// The member's role in the ordered fold (ADR 0017).
    pub operation: CombineOp,
    /// The single material a `Union` member stamps, or `None` for a member that brings its
    /// own per-voxel materials (a nested composite, a VoxelBody).
    pub material: Option<BlockId>,
    pub producer: Box<dyn VoxelProducer>,
}

/// A sealed composition scope — a **Part** (ADR 0018 Decision 1) or a sealed definition
/// body — evaluated as ONE producer, so it can be dilated as a whole.
///
/// # Why this exists
///
/// ADR 0019 Decision 7 requires that a Group or Instance may carry an outset "so a composed
/// cutter dilates as a whole", and explicitly REJECTS leaf-only outset. The two are not
/// interchangeable: dilation distributes over union, so a pure-union Part would agree either
/// way, but a Part with an internal `Subtract` diverges sharply — dilating members
/// individually makes the inner cutter carve MORE, while dilating the composed Part grows the
/// finished body and partly closes that cut.
///
/// A scope is already defined as "pre-compose the children into one body" (ADR 0017 Decision
/// 3). This type makes that composition an explicit producer, which
/// [`OutsetProducer`](super::OutsetProducer) then wraps like any other. Nothing downstream
/// changes: the scope arrives at both folds as a single leaf.
///
/// # The fold is sign-exact
///
/// The field composes through the ordered fold as `min` / `max`, starting from `+INFINITY`
/// (the empty accumulator — which is exactly why intersecting or subtracting from the fold
/// start yields empty, per ADR 0017's ordering law, with no special case):
///
/// ```text
/// Union      d = min(d, member)
/// Subtract   d = max(d, −member)
/// Intersect  d = max(d, member)
/// ```
///
/// All three are **exact in SIGN**: `min` is negative iff either is, `max(a, −b)` iff inside
/// `a` and outside `b`, `max` iff inside both. So at outset zero this composite's occupancy
/// equals the voxel fold's exactly. Only MAGNITUDES go approximate, and only near concave
/// seams, where `max` under-estimates distance while staying 1-Lipschitz — the posture ADR
/// 0017 Decision 6 and ADR 0019 Decision 5 already take. The practical consequence is that a
/// dilated Part is very slightly under-grown in an interior corner, never over-grown.
pub struct CompositeProducer {
    members: Vec<CompositeMember>,
}

impl CompositeProducer {
    pub fn new(members: Vec<CompositeMember>) -> Self {
        Self { members }
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// The member's own field distance at a composite-frame point, or `None` if its geometry
    /// is not a distance — which makes the WHOLE composite fieldless (see [`Self::as_field`]).
    fn member_distance(
        &self,
        member: &CompositeMember,
        point_local_voxels: [f32; 3],
        voxels_per_block: u32,
    ) -> Option<f32> {
        let field = member.producer.as_field()?;
        let local = std::array::from_fn(|axis| {
            point_local_voxels[axis] - member.offset_voxels[axis] as f32
        });
        Some(field.signed_distance(local, voxels_per_block))
    }

    /// The composed distance AND the material at a point, in one walk of the fold.
    ///
    /// Material follows two different rules by design, and they meet exactly at the surface:
    ///
    /// * **Inside** the body, the LAST `Union` member containing the point wins — ADR 0017's
    ///   "on overlap the later node wins the material". Keeping this means an outset Part's
    ///   interior is coloured identically to the same Part at outset zero.
    /// * **Outside** it (the shell the dilation ADDS), the NEAREST `Union` member wins. There
    ///   is no "later" to appeal to out there — no member contains the point — and the shell
    ///   is continuous with the surface it grew from, so it takes that surface's material.
    ///
    /// `Subtract` and `Intersect` members never contribute material: they are occupancy-only
    /// masks and surviving cells keep what they had (ADR 0017 Decision 1).
    fn sample(&self, point_local_voxels: [f32; 3], voxels_per_block: u32) -> (f32, Option<BlockId>) {
        // The empty accumulator is "infinitely far outside", which makes the fold-start rules
        // fall out: union takes the member, subtract and intersect stay empty.
        let mut distance = f32::INFINITY;
        let mut last_inside_material: Option<BlockId> = None;
        let mut nearest_material: Option<BlockId> = None;
        let mut nearest_distance = f32::INFINITY;

        for member in &self.members {
            let Some(member_distance) = self.member_distance(member, point_local_voxels, voxels_per_block)
            else {
                continue;
            };
            match member.operation {
                CombineOp::Union => {
                    distance = distance.min(member_distance);
                    // A member that brings its own per-voxel materials answers for itself.
                    let material = member.material.or_else(|| {
                        let local = std::array::from_fn(|axis| {
                            point_local_voxels[axis] - member.offset_voxels[axis] as f32
                        });
                        member.producer.material_at(local, voxels_per_block)
                    });
                    // `is_sign_negative`, not `< 0.0`: a sample can land exactly on the
                    // surface, where the distance is zero and only its sign bit carries the
                    // inside/outside verdict (ADR 0019 amendment).
                    if member_distance.is_sign_negative() {
                        last_inside_material = material;
                    }
                    if member_distance < nearest_distance {
                        nearest_distance = member_distance;
                        nearest_material = material;
                    }
                }
                CombineOp::Subtract => distance = distance.max(-member_distance),
                CombineOp::Intersect => distance = distance.max(member_distance),
                // ADR 0020 Decision 4. `A` is the accumulator, `C` this member, `N` the
                // signed amount; the accumulator appears TWICE, which is precisely why
                // emboss cannot decompose into existing fold steps.
                //
                //   outward (N > 0)   A' = min(A, max(A − N, C))
                //   inward  (N < 0)   A' = max(A, min(A − N, −C))
                //
                // Verified in the ADR against a set-theoretic ground truth over 64,000
                // samples, and exactly 1-Lipschitz, so the cell classifier's bound survives.
                CombineOp::Emboss { amount } => {
                    let raise = amount.to_voxels(voxels_per_block).unwrap_or(0) as f32;
                    distance = if raise >= 0.0 {
                        distance.min((distance - raise).max(member_distance))
                    } else {
                        distance.max((distance - raise).min(-member_distance))
                    };
                }
            }
        }
        (distance, last_inside_material.or(nearest_material))
    }

    /// Members that can GROW the composite's extent: `Union` and `Emboss` ones.
    ///
    /// A `Subtract` or `Intersect` member's effect is contained in the accumulator (ADR 0020
    /// Decision 3), so it can never push the bounds outward. An OUTWARD `Emboss` can — it
    /// raises the surface — but only within its own footprint, since
    /// `A' = A ∪ (dilate(A, N) ∩ C) ⊆ A ∪ C`. So the member's own extent bounds it exactly
    /// and no `N`-sized margin is needed. (An inward emboss only removes, so including it is
    /// merely conservative.)
    pub(super) fn extent_members(members: &[CompositeMember]) -> impl Iterator<Item = &CompositeMember> {
        members.iter().filter(|member| {
            matches!(member.operation, CombineOp::Union | CombineOp::Emboss { .. })
        })
    }
}

impl VoxelProducer for CompositeProducer {
    fn resolve(&self, grid: &mut VoxelGrid, voxels_per_block: u32) {
        let [x, y, z] = self.full_dimensions(voxels_per_block);
        self.resolve_into(
            grid,
            voxels_per_block,
            VoxelAabb::new([0, 0, 0], [x as i64, y as i64, z as i64]),
        );
    }

    /// Fill from the composed field's sign, carrying each voxel's material from the same
    /// fold walk. Sign-exactness (see the type docs) is what lets this agree with the
    /// voxel-set fold rather than approximate it.
    fn resolve_into(
        &self,
        grid: &mut VoxelGrid,
        voxels_per_block: u32,
        window_local_voxels: VoxelAabb,
    ) {
        let dimensions = self.full_dimensions(voxels_per_block);
        grid.dimensions = dimensions;
        let low: [i64; 3] = std::array::from_fn(|axis| {
            window_local_voxels.min[axis].clamp(0, dimensions[axis] as i64)
        });
        let high: [i64; 3] = std::array::from_fn(|axis| {
            window_local_voxels.max[axis].clamp(low[axis], dimensions[axis] as i64)
        });

        let mut occupied = Vec::new();
        for k in low[2]..high[2] {
            for j in low[1]..high[1] {
                for i in low[0]..high[0] {
                    let centre = [i as f32 + 0.5, j as f32 + 0.5, k as f32 + 0.5];
                    let (distance, material) = self.sample(centre, voxels_per_block);
                    if distance <= SURFACE_ISOLEVEL {
                        occupied.push(Voxel {
                            local_index: [i as i32, j as i32, k as i32],
                            block_local_coord: [
                                (i % voxels_per_block as i64) as u8,
                                (j % voxels_per_block as i64) as u8,
                                (k % voxels_per_block as i64) as u8,
                            ],
                            block_id: material.unwrap_or(BlockId::DEFAULT),
                            attrs: BlockAttrs::DEFAULT,
                            grid_overlay: false,
                        });
                    }
                }
            }
        }
        grid.occupied = occupied;
    }

    fn material_at(&self, point_local_voxels: [f32; 3], voxels_per_block: u32) -> Option<BlockId> {
        self.sample(point_local_voxels, voxels_per_block).1
    }

    /// The Lipschitz bracket of the composed field, in the composite's own metric.
    fn cell_field_interval(
        &self,
        cell_local_voxels: VoxelAabb,
        voxels_per_block: u32,
    ) -> Option<FieldInterval> {
        if cell_local_voxels.is_empty() || self.as_field().is_none() {
            return None;
        }
        Some(super::metric_cell_bracket(cell_local_voxels, self.metric(), |centre| {
            self.sample(centre, voxels_per_block).0
        }))
    }

    /// The composite has a field only if EVERY member does — one fieldless member leaves the
    /// fold with nothing to compose, and ADR 0020 Decision 1 says answer honestly rather than
    /// fabricate a distance. Such a Part simply cannot be outset.
    ///
    /// **Not because of the cloud.** ADR 0021 withdrew that justification: the cloud is
    /// boundable (`cell_field_interval` classifies a cell from puff geometry alone). It still
    /// answers `None` here, but on the narrower ground that its geometry is not a *distance* —
    /// `radial + BILLOW·fbm` has the right zero set and the wrong magnitude away from it. The
    /// `Option` itself rests on freehand sculpt, which is occupancy-native (ADR 0021 §5).
    fn as_field(&self) -> Option<&dyn Field> {
        if self.members.iter().all(|member| member.producer.as_field().is_some()) {
            Some(self)
        } else {
            None
        }
    }

    /// The union of the `Union` members' placed extents.
    fn full_dimensions(&self, voxels_per_block: u32) -> [u32; 3] {
        let mut high = [0i64; 3];
        for member in Self::extent_members(&self.members) {
            let dimensions = member.producer.full_dimensions(voxels_per_block);
            for axis in 0..3 {
                high[axis] = high[axis].max(member.offset_voxels[axis] + dimensions[axis] as i64);
            }
        }
        std::array::from_fn(|axis| high[axis].max(0) as u32)
    }
}

impl Field for CompositeProducer {
    fn signed_distance(&self, point_local_voxels: [f32; 3], voxels_per_block: u32) -> f32 {
        self.sample(point_local_voxels, voxels_per_block).0
    }

    /// **The weakest of the members' metrics** (ADR 0019 Decision 7: "a group mixing a box
    /// and a sphere outsets round"), which is sound rather than merely conventional: since
    /// `‖·‖∞ <= ‖·‖₂`, a field 1-Lipschitz under Chebyshev is automatically 1-Lipschitz
    /// under Euclidean, so widening to Euclidean can never overstate the bound. Chebyshev is
    /// claimed only when EVERY member measures square.
    fn metric(&self) -> substrate::geom2d::Metric {
        let all_square = self.members.iter().all(|member| {
            member
                .producer
                .as_field()
                .map(|field| field.metric() == substrate::geom2d::Metric::Chebyshev)
                .unwrap_or(false)
        });
        if all_square {
            substrate::geom2d::Metric::Chebyshev
        } else {
            substrate::geom2d::Metric::Euclidean
        }
    }
}
