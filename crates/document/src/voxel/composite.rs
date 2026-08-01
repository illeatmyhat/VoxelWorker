//! A sealed scope evaluated as a single producer, so a whole Part can be outset.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::missing_const_for_fn,
    clippy::map_unwrap_or,
    clippy::must_use_candidate,
    clippy::unused_self,
    clippy::doc_markdown,
    clippy::similar_names
)]

use super::{Field, FieldInterval, PreparedField, VoxelProducer};
use crate::scene::{CombineOp, LeafOrigin};
use voxel_core::core_geom::BlockId;
use voxel_core::spatial_index::VoxelAabb;
use voxel_core::voxel::{BlockAttrs, Voxel, VoxelGrid, SURFACE_ISOLEVEL};

/// One member of a [`CompositeProducer`], in the composite's own `[0, full_dim)` frame.
pub struct CompositeMember {
    /// The member's low corner relative to the composite's frame origin.
    pub offset_voxels: [i64; 3],
    /// The member's role in the ordered fold.
    pub operation: CombineOp,
    /// Which node this member came from. A pre-composed scope is ONE leaf to the walk, so
    /// without this a viewport pick anywhere inside it could only name the scope — and a
    /// single top-level Emboss pre-composes the whole scene.
    pub source: LeafOrigin,
    /// The single material a `Union` member stamps, or `None` for a member that brings its
    /// own per-voxel materials (a nested composite, a VoxelBody).
    pub material: Option<BlockId>,
    pub producer: Box<dyn VoxelProducer>,
}

/// The field half of a [`CompositeProducer`] resolved once for one dense sampling operation.
///
/// Each child evaluator owns its density-dependent state (a sketch's resolved region included),
/// so the composite fold never re-enters a child `Field` from its per-voxel loop.
struct PreparedCompositeField<'a> {
    members: Vec<PreparedCompositeMember<'a>>,
    voxels_per_block: u32,
}

struct PreparedCompositeMember<'a> {
    member: &'a CompositeMember,
    field: Box<dyn PreparedField + 'a>,
}

impl PreparedCompositeField<'_> {
    /// The composed distance AND the material at a point, in one prepared field fold.
    fn sample(&self, point_local_voxels: [f32; 3]) -> (f32, Option<BlockId>) {
        let mut distance = f32::INFINITY;
        let mut last_inside_material: Option<BlockId> = None;
        let mut nearest_material: Option<BlockId> = None;
        let mut nearest_distance = f32::INFINITY;

        for prepared in &self.members {
            let member = prepared.member;
            let local = std::array::from_fn(|axis| {
                point_local_voxels[axis] - member.offset_voxels[axis] as f32
            });
            let member_distance = prepared.field.signed_distance(local);
            match member.operation {
                CombineOp::Union => {
                    distance = distance.min(member_distance);
                    let material = member
                        .material
                        .or_else(|| member.producer.material_at(local, self.voxels_per_block));
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
                CombineOp::Emboss { amount } => {
                    let raise = amount.to_voxels(self.voxels_per_block).unwrap_or(0) as f32;
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
}

impl PreparedField for PreparedCompositeField<'_> {
    fn signed_distance(&self, point_local_voxels: [f32; 3]) -> f32 {
        self.sample(point_local_voxels).0
    }

    fn metric(&self) -> substrate::geom2d::Metric {
        let all_square = self
            .members
            .iter()
            .all(|prepared| prepared.field.metric() == substrate::geom2d::Metric::Chebyshev);
        if all_square {
            substrate::geom2d::Metric::Chebyshev
        } else {
            substrate::geom2d::Metric::Euclidean
        }
    }

    fn preserves_native_interval(&self) -> bool {
        true
    }

    /// Preserve the composite producer's native interval proof while evaluating child geometry
    /// through the evaluators prepared for this sweep.
    fn native_cell_field_interval(&self, cell_local_voxels: VoxelAabb) -> FieldInterval {
        super::metric_cell_bracket(cell_local_voxels, self.metric(), |center| {
            self.sample(center).0
        })
    }
}

/// A sealed composition scope — a **Part** or a sealed definition body — evaluated as ONE
/// producer, so it can be dilated as a whole.
///
/// # Why this exists
///
/// A Group or Instance carries the outset so a composed cutter dilates as a whole; outsetting
/// its leaves instead is NOT the same operation. Dilation distributes over union, so a
/// pure-union Part would agree either way, but a Part with an internal `Subtract` diverges
/// sharply — dilating members individually makes the inner cutter carve MORE, while dilating
/// the composed Part grows the finished body and partly closes that cut.
///
/// A scope already means "pre-compose the children into one body". This type makes that
/// composition an explicit producer, which [`OutsetProducer`](super::OutsetProducer) then
/// wraps like any other, and the scope arrives at both folds as a single leaf.
///
/// # The fold is sign-exact
///
/// The field composes through the ordered fold as `min` / `max`, starting from `+INFINITY`
/// (the empty accumulator — which is exactly why intersecting or subtracting from the fold
/// start yields empty, with no special case):
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
/// seams, where `max` under-estimates distance while staying 1-Lipschitz. The practical
/// consequence is that a dilated Part is very slightly under-grown in an interior corner,
/// never over-grown.
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

    /// Prepare every field-bearing member once for one sampling operation. A fieldless member is
    /// skipped exactly as the legacy field fold skips it; callers that need an honest composite
    /// field still go through [`VoxelProducer::as_field`], which rejects the composite unless all
    /// members have one.
    fn prepare_field(&self, voxels_per_block: u32) -> PreparedCompositeField<'_> {
        PreparedCompositeField {
            members: self
                .members
                .iter()
                .filter_map(|member| {
                    member
                        .producer
                        .as_field()
                        .map(|field| PreparedCompositeMember {
                            member,
                            field: field.prepare(voxels_per_block),
                        })
                })
                .collect(),
            voxels_per_block,
        }
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
    /// * **Inside** the body, the LAST `Union` member containing the point wins — on overlap
    ///   the later node wins the material. That keeps an outset Part's interior colored
    ///   identically to the same Part at outset zero.
    /// * **Outside** it (the shell the dilation ADDS), the NEAREST `Union` member wins. There
    ///   is no "later" to appeal to out there — no member contains the point — and the shell
    ///   is continuous with the surface it grew from, so it takes that surface's material.
    ///
    /// `Subtract` and `Intersect` members never contribute material: they are occupancy-only
    /// masks and surviving cells keep what they had.
    fn sample(
        &self,
        point_local_voxels: [f32; 3],
        voxels_per_block: u32,
    ) -> (f32, Option<BlockId>) {
        // The empty accumulator is "infinitely far outside", which makes the fold-start rules
        // fall out: union takes the member, subtract and intersect stay empty.
        let mut distance = f32::INFINITY;
        let mut last_inside_material: Option<BlockId> = None;
        let mut nearest_material: Option<BlockId> = None;
        let mut nearest_distance = f32::INFINITY;

        for member in &self.members {
            let Some(member_distance) =
                self.member_distance(member, point_local_voxels, voxels_per_block)
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
                    // inside/outside verdict.
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
                // `A` is the accumulator, `C` this member, `N` the signed amount; the
                // accumulator appears TWICE, which is precisely why emboss cannot decompose
                // into existing fold steps.
                //
                //   outward (N > 0)   A' = min(A, max(A − N, C))
                //   inward  (N < 0)   A' = max(A, min(A − N, −C))
                //
                // Exactly 1-Lipschitz, so the cell classifier's bound survives.
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

    /// Which node authored the geometry at a composite-frame point — the same walk
    /// [`sample`](Self::sample) does for material, answering with origins instead.
    ///
    /// The two rules are deliberately identical (last containing `Union` member inside the
    /// body, nearest one out in an outset shell), because the pick follows the material: the
    /// node you select is the node that colored the voxel you clicked. A nested composite
    /// answers for itself, so a pick names the innermost authored leaf at any depth rather
    /// than the Group enclosing it; only the instance boundary redirects, and that redirect
    /// is already baked into each member's origin by the walk.
    ///
    /// Masks are skipped rather than folded: the caller runs the scoped fold and only asks
    /// about a point the composite already resolved as solid, so a `Subtract` that would
    /// have removed it did not.
    ///
    /// **The outset shell.** A point out in the dilation reaches no member's body, and the
    /// nearest member answers — the same member whose material the shell takes
    /// ([`OutsetProducer::material_at`](super::OutsetProducer)), because the shell is
    /// continuous with the surface it grew from. So a click on a Part's dilation selects the
    /// member under it, not the scope node carrying the outset; reaching that property is a
    /// navigation step (parent), not a different pick.
    fn origin_at_point(
        &self,
        point_local_voxels: [f32; 3],
        voxels_per_block: u32,
    ) -> Option<LeafOrigin> {
        let mut last_inside: Option<LeafOrigin> = None;
        let mut nearest: Option<LeafOrigin> = None;
        let mut nearest_distance = f32::INFINITY;

        for member in &self.members {
            if member.operation != CombineOp::Union {
                continue;
            }
            let Some(member_distance) =
                self.member_distance(member, point_local_voxels, voxels_per_block)
            else {
                continue;
            };
            let local = std::array::from_fn(|axis| {
                point_local_voxels[axis] - member.offset_voxels[axis] as f32
            });
            // A nested scope answers for its own innermost member; a plain body has no
            // opinion and IS the member, so it answers as itself.
            let origin = member
                .producer
                .origin_at(local, voxels_per_block)
                .unwrap_or(member.source);
            // `is_sign_negative`, not `< 0.0`: a sample can land exactly on the surface,
            // where only the sign bit carries the inside/outside verdict (as `sample` does).
            if member_distance.is_sign_negative() {
                last_inside = Some(origin);
            }
            if member_distance < nearest_distance {
                nearest_distance = member_distance;
                nearest = Some(origin);
            }
        }
        last_inside.or(nearest)
    }

    /// Members that can GROW the composite's extent: `Union` and `Emboss` ones.
    ///
    /// A `Subtract` or `Intersect` member's effect is contained in the accumulator, so it can
    /// never push the bounds outward. An OUTWARD `Emboss` can — it raises the surface — but
    /// only within its own footprint, since `A' = A ∪ (dilate(A, N) ∩ C) ⊆ A ∪ C`. So the
    /// member's own extent bounds it exactly and no `N`-sized margin is needed. (An inward
    /// emboss only removes, so including it is merely conservative.)
    pub(super) fn extent_members(
        members: &[CompositeMember],
    ) -> impl Iterator<Item = &CompositeMember> {
        members.iter().filter(|member| {
            matches!(
                member.operation,
                CombineOp::Union | CombineOp::Emboss { .. }
            )
        })
    }
}

impl VoxelProducer for CompositeProducer {
    fn origin_at(&self, point_local_voxels: [f32; 3], voxels_per_block: u32) -> Option<LeafOrigin> {
        self.origin_at_point(point_local_voxels, voxels_per_block)
    }

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

        let prepared = self.prepare_field(voxels_per_block);
        let mut occupied = Vec::new();
        for k in low[2]..high[2] {
            for j in low[1]..high[1] {
                for i in low[0]..high[0] {
                    let center = [i as f32 + 0.5, j as f32 + 0.5, k as f32 + 0.5];
                    let (distance, material) = prepared.sample(center);
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
        Some(super::metric_cell_bracket(
            cell_local_voxels,
            self.metric(),
            |center| self.sample(center, voxels_per_block).0,
        ))
    }

    /// The composite has a field only if EVERY member does — one fieldless member leaves the
    /// fold with nothing to compose, and the honest answer is `None` rather than a fabricated
    /// distance. Such a Part simply cannot be outset.
    ///
    /// Being fieldless is not the same as being unboundable: the debug cloud brackets cells
    /// fine (`cell_field_interval` classifies one from puff geometry alone) yet answers `None`
    /// here, because `radial + BILLOW·fbm` has the right zero set and the wrong magnitude away
    /// from it. Occupancy-native geometry — freehand sculpt — is the general case the
    /// `Option` exists for.
    fn as_field(&self) -> Option<&dyn Field> {
        if self
            .members
            .iter()
            .all(|member| member.producer.as_field().is_some())
        {
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

    fn has_native_interval(&self) -> bool {
        true
    }

    /// **The weakest of the members' metrics**, so a group mixing a box and a sphere outsets
    /// round. That is sound rather than merely conventional: since `‖·‖∞ <= ‖·‖₂`, a field
    /// 1-Lipschitz under Chebyshev is automatically 1-Lipschitz under Euclidean, so widening
    /// to Euclidean can never overstate the bound. Chebyshev is claimed only when EVERY
    /// member measures square.
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

    fn prepare(&self, voxels_per_block: u32) -> Box<dyn PreparedField + '_> {
        Box::new(self.prepare_field(voxels_per_block))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{LeafOrigin, NodeId};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct CountingField {
        preparations: Arc<AtomicUsize>,
    }

    struct PreparedCountingField;

    impl PreparedField for PreparedCountingField {
        fn signed_distance(&self, _point_local_voxels: [f32; 3]) -> f32 {
            -1.0
        }

        fn metric(&self) -> substrate::geom2d::Metric {
            substrate::geom2d::Metric::Chebyshev
        }
    }

    impl VoxelProducer for CountingField {
        fn resolve(&self, grid: &mut VoxelGrid, _voxels_per_block: u32) {
            grid.dimensions = [4, 4, 4];
            grid.occupied.clear();
        }

        fn resolve_into(
            &self,
            grid: &mut VoxelGrid,
            voxels_per_block: u32,
            _window_local_voxels: VoxelAabb,
        ) {
            self.resolve(grid, voxels_per_block);
        }

        fn as_field(&self) -> Option<&dyn Field> {
            Some(self)
        }

        fn full_dimensions(&self, _voxels_per_block: u32) -> [u32; 3] {
            [4, 4, 4]
        }
    }

    impl Field for CountingField {
        fn signed_distance(&self, _point_local_voxels: [f32; 3], _voxels_per_block: u32) -> f32 {
            -1.0
        }

        fn metric(&self) -> substrate::geom2d::Metric {
            substrate::geom2d::Metric::Chebyshev
        }

        fn prepare(&self, _voxels_per_block: u32) -> Box<dyn PreparedField + '_> {
            self.preparations.fetch_add(1, Ordering::Relaxed);
            Box::new(PreparedCountingField)
        }
    }

    #[test]
    fn dense_composite_resolution_prepares_each_child_once_not_per_voxel() {
        let preparations = Arc::new(AtomicUsize::new(0));
        let member = |node| CompositeMember {
            offset_voxels: [0, 0, 0],
            operation: CombineOp::Union,
            source: LeafOrigin::authored(NodeId(node)),
            material: Some(BlockId::DEFAULT),
            producer: Box::new(CountingField {
                preparations: Arc::clone(&preparations),
            }),
        };
        let composite = CompositeProducer::new(vec![member(1), member(2)]);
        let mut grid = VoxelGrid::new([0, 0, 0]);

        composite.resolve(&mut grid, 16);

        assert_eq!(grid.occupied_count(), 64, "the 4³ sample grid was walked");
        assert_eq!(
            preparations.load(Ordering::Relaxed),
            2,
            "one prepared evaluator per child, not once for each of 64 voxels"
        );
    }
}
