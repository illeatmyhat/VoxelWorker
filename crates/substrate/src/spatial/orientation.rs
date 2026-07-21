//! [`LatticeOrientation`] — an axis-aligned orientation: one of the 24 rotations that
//! carry the integer lattice onto itself.
//!
//! ## What it is
//!
//! A **signed axis permutation** whose matrix is orthogonal with entries in `{-1, 0, 1}`
//! and determinant `+1` — exactly one non-zero per row and column. These are the **proper
//! rotations of the cube** (the chiral octahedral group *O*, order 24): the rigid turns
//! that map `{±x, ±y, ±z}` axes to axes. They are the rotation subgroup of the
//! *hyperoctahedral group* of signed permutation matrices (order 48); the 24 orientation-
//! reversing reflections are excluded — a placed body is turned, never mirrored.
//!
//! ## Why it earns a type
//!
//! A voxel face normal is an exact `±1` axis vector, so the turn that stands a body against
//! a face is always one of these — no float rotation, no resampling. Represented and applied
//! in integers it is **exact**: it relabels and negates coordinate axes, so it preserves an
//! axis-aligned box's shape, a field's Lipschitz bound (it is an isometry), and integer
//! occupancy. That exactness is what lets the domain apply it inside the classifier without
//! disturbing the SDF parity surface (see `docs/adr/0026`).
//!
//! ## Representation — the gather form
//!
//! Stored as, per **output** axis `o`, the **source** input axis it reads and the sign
//! applied: `apply(v)[o] = sign[o] · v[source[o]]`. The gather form is chosen because the hot
//! caller (the classifier mapping absolute voxels into a producer's local frame) applies the
//! *inverse* to a query point, and a gather reads cleanly there. `source` is a permutation of
//! `{0,1,2}` and the signs are constrained so the determinant is `+1`.

/// One of the 24 axis-aligned rotations of the integer lattice (the proper cube rotations,
/// group *O*). Lattice-exact: applying it relabels and negates axes and never resamples.
///
/// See the module docs for the group-theory framing and `docs/adr/0026` for why placement
/// orientation is one of these and not a general float rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatticeOrientation {
    /// `source[o]` — the input axis output axis `o` reads from (a permutation of `0,1,2`).
    source: [u8; 3],
    /// `sign[o]` — the `±1` applied to that source component. Constrained so `det = +1`.
    sign: [i8; 3],
}

impl Default for LatticeOrientation {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl LatticeOrientation {
    /// The identity turn — every axis maps to itself, unturned. A node with this orientation
    /// is world-aligned; the built-in world planes always place at identity (ADR 0026).
    pub const IDENTITY: Self = Self { source: [0, 1, 2], sign: [1, 1, 1] };

    /// The gather form — `(source, sign)` where `apply(v)[o] = sign[o] · v[source[o]]`. This
    /// is the type's stable, self-describing codec: a caller that must persist an orientation
    /// (the document serializes it through here) stores these six small integers and rebuilds
    /// via [`from_gather`](Self::from_gather), which re-validates. Substrate itself stays
    /// serde-free (the crate's boundary law), so the serde adapter lives at the domain seam.
    pub fn to_gather(&self) -> ([u8; 3], [i8; 3]) {
        (self.source, self.sign)
    }

    /// Construct from the gather form, if it is a valid proper rotation (`source` a
    /// permutation and `det = +1`); `None` otherwise. The invariant every other constructor
    /// upholds, and the re-validation a deserialized orientation passes through.
    pub fn from_gather(source: [u8; 3], sign: [i8; 3]) -> Option<Self> {
        let is_permutation = {
            let [a, b, c] = source;
            a < 3 && b < 3 && c < 3 && a != b && b != c && a != c
        };
        let signs_unit = sign.iter().all(|&s| s == 1 || s == -1);
        if !is_permutation || !signs_unit {
            return None;
        }
        let candidate = Self { source, sign };
        (candidate.determinant() == 1).then_some(candidate)
    }

    /// The determinant of the rotation matrix (`+1` for a proper rotation, `-1` for a
    /// reflection). The sign of the permutation times the product of the axis signs.
    fn determinant(&self) -> i32 {
        // Parity of the permutation `source`: +1 if even, -1 if odd. With three elements a
        // permutation is odd iff it is a single transposition (exactly one fixed point) or
        // the reverse; counting inversions is simplest and unambiguous.
        let [a, b, c] = self.source.map(|s| s as i32);
        let inversions = (a > b) as i32 + (a > c) as i32 + (b > c) as i32;
        let permutation_sign = if inversions % 2 == 0 { 1 } else { -1 };
        permutation_sign * self.sign.iter().map(|&s| s as i32).product::<i32>()
    }

    /// Apply the turn to an integer vector: `out[o] = sign[o] · v[source[o]]`. Exact.
    pub fn apply(&self, v: [i64; 3]) -> [i64; 3] {
        std::array::from_fn(|o| self.sign[o] as i64 * v[self.source[o] as usize])
    }

    /// Apply the turn to a real vector — the same relabel-and-negate, for the ghost preview's
    /// analytic sample and any float-frame caller.
    pub fn apply_f32(&self, v: [f32; 3]) -> [f32; 3] {
        std::array::from_fn(|o| self.sign[o] as f32 * v[self.source[o] as usize])
    }

    /// Permute a non-negative **extent** (grid dimensions) by the turn, ignoring sign — a
    /// box's side lengths are orientation-invariant in magnitude, only relabelled. `out[o]`
    /// is the input extent along the axis that lands on output `o`. This is the world span of
    /// a turned producer grid, still corner-anchored at its world offset.
    pub fn turn_extent(&self, extent: [u32; 3]) -> [u32; 3] {
        std::array::from_fn(|o| extent[self.source[o] as usize])
    }

    /// [`turn_extent`](Self::turn_extent) for an `i64` span — the domain carries grid extents
    /// as `i64` for its offset arithmetic (ADR 0008); the values are magnitudes, so this is
    /// the same sign-ignoring permutation.
    pub fn turn_extent_i64(&self, extent: [i64; 3]) -> [i64; 3] {
        std::array::from_fn(|o| extent[self.source[o] as usize])
    }

    /// Turn a **local cell index** in the corner-anchored box `[0, extent)` to its cell index
    /// in the turned box `[0, turn_extent(extent))`, re-anchored so the low corner stays at the
    /// origin. A positive axis passes through; a negated axis is **reversed in place**
    /// (`ext−1−l`) rather than sent negative — which is what keeps a turned producer grid
    /// corner-anchored at its world offset, exactly like an un-turned one (ADR 0008 frame).
    pub fn turn_point_in_box(&self, local: [i64; 3], extent: [u32; 3]) -> [i64; 3] {
        std::array::from_fn(|o| {
            let s = self.source[o] as usize;
            if self.sign[o] > 0 { local[s] } else { extent[s] as i64 - 1 - local[s] }
        })
    }

    /// The inverse of [`turn_point_in_box`](Self::turn_point_in_box) on a **half-open AABB**:
    /// map a world-frame voxel box `[world_min, world_max)` (lying within the turned box) back
    /// into the producer's local half-open box, given the producer's LOCAL `extent`. A negated
    /// axis reverses the interval — `[lo, hi)` becomes `[ext−hi, ext−lo)` — so the result is a
    /// proper half-open box the producer can bound. This is the map the classifier applies to
    /// send a world block/voxel cell into the producer's unturned frame.
    pub fn unturn_box(
        &self,
        world_min: [i64; 3],
        world_max: [i64; 3],
        extent: [u32; 3],
    ) -> ([i64; 3], [i64; 3]) {
        let mut local_min = [0i64; 3];
        let mut local_max = [0i64; 3];
        for o in 0..3 {
            let s = self.source[o] as usize;
            let (wlo, whi) = (world_min[o], world_max[o]);
            let (lo, hi) = if self.sign[o] > 0 {
                (wlo, whi)
            } else {
                (extent[s] as i64 - whi, extent[s] as i64 - wlo)
            };
            local_min[s] = lo;
            local_max[s] = hi;
        }
        (local_min, local_max)
    }

    /// The composition `self ∘ other` — the turn that applies `other` first, then `self`
    /// (matrix product `self · other`). Closed over the 24: the result is always another
    /// proper rotation.
    pub fn compose(&self, other: &LatticeOrientation) -> LatticeOrientation {
        // (A·B) as a gather: out o reads A's source o, whose input is B's output `k = A.source[o]`,
        // which itself reads B's source k with B's sign. So source = B.source[A.source[o]] and
        // sign = A.sign[o] · B.sign[A.source[o]].
        let source = std::array::from_fn(|o| other.source[self.source[o] as usize]);
        let sign = std::array::from_fn(|o| self.sign[o] * other.sign[self.source[o] as usize]);
        LatticeOrientation { source, sign }
    }

    /// The inverse turn — `self.inverse().apply(self.apply(v)) == v`. A rotation matrix's
    /// inverse is its transpose; for the gather form the transpose sends output axis `o` back
    /// to input `source[o]` carrying the same sign.
    pub fn inverse(&self) -> LatticeOrientation {
        let mut source = [0u8; 3];
        let mut sign = [1i8; 3];
        for o in 0..3 {
            let s = self.source[o] as usize;
            source[s] = o as u8;
            sign[s] = self.sign[o];
        }
        LatticeOrientation { source, sign }
    }

    /// The 3×3 rotation matrix, row-major, entries in `{-1, 0, 1}` — for a GPU uniform or any
    /// caller that wants the dense form. Row `o` has `sign[o]` in column `source[o]`.
    pub fn to_matrix(&self) -> [[i32; 3]; 3] {
        let mut m = [[0i32; 3]; 3];
        for o in 0..3 {
            m[o][self.source[o] as usize] = self.sign[o] as i32;
        }
        m
    }

    /// The orientation that stands a body's local **+Z** axis against a geometry face whose
    /// outward `normal` is an exact `±1` axis vector, by the shortest-arc swing (zero twist).
    ///
    /// `+Z` is the identity; the other five are the single 90°/180° turns that carry `+Z` onto
    /// the given axis. A cylinder (axis-locked to local Z) placed on a `+X` wall therefore lies
    /// on its side poking out along `+X`. The twist about the normal is left at zero — the
    /// curved primitives are symmetric about their axis, so it is unobservable for them; a box
    /// is symmetric under 90° twists, so it is unobservable for it too at authoring sizes.
    ///
    /// Panics on a non-axis `normal` (not exactly one `±1` component) — placement only ever
    /// derives this from a voxel face normal, which is always axis-aligned.
    pub fn from_face_normal(normal: [i32; 3]) -> LatticeOrientation {
        let parts = match normal {
            [0, 0, 1] => ([0, 1, 2], [1, 1, 1]),    // +Z: identity
            [0, 0, -1] => ([0, 1, 2], [1, -1, -1]), // -Z: 180° about X
            [1, 0, 0] => ([2, 1, 0], [1, 1, -1]),   // +X: +90° about Y
            [-1, 0, 0] => ([2, 1, 0], [-1, 1, 1]),  // -X: -90° about Y
            [0, 1, 0] => ([0, 2, 1], [1, 1, -1]),   // +Y: -90° about X
            [0, -1, 0] => ([0, 2, 1], [1, -1, 1]),  // -Y: +90° about X
            other => panic!("face normal must be an axis vector, got {other:?}"),
        };
        Self::from_gather(parts.0, parts.1).expect("the six face-normal turns are proper rotations")
    }

    /// Whether this is the identity turn (world-aligned). The world-plane placement path
    /// asserts this — those planes position only, never orient (ADR 0026).
    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All 24 proper cube rotations, built by filtering the 48 signed permutations down to
    /// determinant `+1`. The oracle the closure/inverse tests enumerate over.
    fn all_proper() -> Vec<LatticeOrientation> {
        let permutations: [[u8; 3]; 6] =
            [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
        let mut all = Vec::new();
        for source in permutations {
            for bits in 0..8u8 {
                let sign = std::array::from_fn(|i| if bits & (1 << i) == 0 { 1 } else { -1 });
                if let Some(orientation) = LatticeOrientation::from_gather(source, sign) {
                    all.push(orientation);
                }
            }
        }
        all
    }

    /// There are exactly 24 proper rotations — the order of the chiral octahedral group.
    #[test]
    fn there_are_twenty_four_proper_rotations() {
        assert_eq!(all_proper().len(), 24);
    }

    /// Every proper rotation is an **isometry on the lattice**: it maps the six unit axis
    /// vectors to the six unit axis vectors, bijectively (a permutation with signs).
    #[test]
    fn each_rotation_permutes_the_signed_axes() {
        let axes: [[i64; 3]; 6] =
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]];
        for orientation in all_proper() {
            let images: Vec<[i64; 3]> = axes.iter().map(|&a| orientation.apply(a)).collect();
            for axis in axes {
                assert!(images.contains(&axis), "{orientation:?} lost {axis:?}");
            }
        }
    }

    /// **Inverse round-trips.** `inverse().apply(apply(v)) == v` for every rotation and an
    /// arbitrary vector — the property the classifier's abs→local→abs remap relies on.
    #[test]
    fn inverse_undoes_apply() {
        let v = [3i64, -7, 11];
        for orientation in all_proper() {
            assert_eq!(orientation.inverse().apply(orientation.apply(v)), v, "{orientation:?}");
        }
    }

    /// **Compose matches applying in sequence**, and is closed over the 24.
    #[test]
    fn compose_matches_sequential_application() {
        let v = [5i64, -2, 9];
        let all = all_proper();
        for a in &all {
            for b in &all {
                let composed = a.compose(b);
                assert!(all.contains(&composed), "{a:?} ∘ {b:?} escaped the group");
                assert_eq!(composed.apply(v), a.apply(b.apply(v)), "{a:?} ∘ {b:?}");
            }
        }
    }

    /// **Identity is the composition unit** on both sides.
    #[test]
    fn identity_is_the_unit() {
        for orientation in all_proper() {
            assert_eq!(orientation.compose(&LatticeOrientation::IDENTITY), orientation);
            assert_eq!(LatticeOrientation::IDENTITY.compose(&orientation), orientation);
        }
    }

    /// **The face-normal derivation stands +Z against the face.** For each of the six axis
    /// normals, the turn carries local `+Z` onto that normal — the placement contract.
    #[test]
    fn from_face_normal_carries_z_onto_the_normal() {
        for normal in [[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]] {
            let orientation = LatticeOrientation::from_face_normal(normal);
            assert_eq!(
                orientation.apply([0, 0, 1]),
                normal.map(|c| c as i64),
                "normal {normal:?}: +Z did not land on the face normal"
            );
            assert_eq!(orientation.determinant(), 1, "normal {normal:?} produced a reflection");
        }
        // +Z is the identity (an on-top / ground placement stays upright).
        assert!(LatticeOrientation::from_face_normal([0, 0, 1]).is_identity());
    }

    /// **The corner-anchored cell turn stays inside the turned box, bijectively.** Every
    /// local cell of `[0, extent)` maps to a distinct cell of `[0, turn_extent(extent))` — a
    /// turned producer grid tiles its world box with no gap or overlap.
    #[test]
    fn turn_point_in_box_is_a_bijection_onto_the_turned_box() {
        let extent = [3u32, 4, 2];
        for orientation in all_proper() {
            let turned = orientation.turn_extent(extent);
            let mut seen = std::collections::HashSet::new();
            for x in 0..extent[0] as i64 {
                for y in 0..extent[1] as i64 {
                    for z in 0..extent[2] as i64 {
                        let w = orientation.turn_point_in_box([x, y, z], extent);
                        for axis in 0..3 {
                            assert!(
                                w[axis] >= 0 && w[axis] < turned[axis] as i64,
                                "{orientation:?}: cell {:?} left the turned box at axis {axis}",
                                [x, y, z]
                            );
                        }
                        assert!(seen.insert(w), "{orientation:?}: collision at {w:?}");
                    }
                }
            }
            assert_eq!(seen.len(), (extent[0] * extent[1] * extent[2]) as usize);
        }
    }

    /// **`unturn_box` inverts the cell turn on a half-open sub-box.** A world sub-box maps back
    /// to exactly the local cells that turn into it — the classifier's world→local remap.
    #[test]
    fn unturn_box_recovers_the_local_cells() {
        let extent = [4u32, 3, 5];
        for orientation in all_proper() {
            // A world sub-box: one turned cell (a 1³ block) at a chosen world position.
            for x in 0..extent[0] as i64 {
                for y in 0..extent[1] as i64 {
                    for z in 0..extent[2] as i64 {
                        let local = [x, y, z];
                        let w = orientation.turn_point_in_box(local, extent);
                        let world_min = w;
                        let world_max = [w[0] + 1, w[1] + 1, w[2] + 1];
                        let (lmin, lmax) = orientation.unturn_box(world_min, world_max, extent);
                        assert_eq!(lmin, local, "{orientation:?}: unturn min");
                        assert_eq!(lmax, [x + 1, y + 1, z + 1], "{orientation:?}: unturn max");
                    }
                }
            }
        }
    }

    /// `to_matrix` agrees with `apply`: `M · v == apply(v)`.
    #[test]
    fn matrix_agrees_with_apply() {
        let v = [4i64, -6, 8];
        for orientation in all_proper() {
            let m = orientation.to_matrix();
            let by_matrix: [i64; 3] = std::array::from_fn(|o| {
                (0..3).map(|i| m[o][i] as i64 * v[i]).sum()
            });
            assert_eq!(by_matrix, orientation.apply(v), "{orientation:?}");
        }
    }
}

/// Kani harnesses — the group laws over the *whole* set of signed permutations, proved rather
/// than sampled. Bounded-model-checked in WSL on demand (`[[kani-wsl-toolchain]]`), never in the
/// cargo gate. Each calls the production functions, never a re-model.
#[cfg(kani)]
mod proofs {
    use super::*;

    /// An arbitrary proper rotation: a signed permutation Kani constructs and we constrain to
    /// the valid set via the production `from_gather`. `assume` prunes the reflections and
    /// non-permutations, leaving exactly the 24.
    fn any_orientation() -> LatticeOrientation {
        let source: [u8; 3] = kani::any();
        let sign: [i8; 3] = kani::any();
        match LatticeOrientation::from_gather(source, sign) {
            Some(orientation) => orientation,
            None => kani::reject(),
        }
    }

    /// Inverse is a two-sided inverse under composition: `a⁻¹ ∘ a = identity` for every `a`.
    #[kani::proof]
    fn inverse_is_a_group_inverse() {
        let a = any_orientation();
        assert!(a.inverse().compose(&a) == LatticeOrientation::IDENTITY);
        assert!(a.compose(&a.inverse()) == LatticeOrientation::IDENTITY);
    }

    /// Compose is associative — `(a ∘ b) ∘ c = a ∘ (b ∘ c)`.
    #[kani::proof]
    fn compose_is_associative() {
        let a = any_orientation();
        let b = any_orientation();
        let c = any_orientation();
        assert!(a.compose(&b).compose(&c) == a.compose(&b.compose(&c)));
    }

    /// Compose is realized by sequential application, for a bounded vector.
    #[kani::proof]
    fn compose_realizes_sequential_apply() {
        let a = any_orientation();
        let b = any_orientation();
        let v: [i64; 3] = kani::any();
        for c in v {
            kani::assume(c > -1000 && c < 1000);
        }
        assert!(a.compose(&b).apply(v) == a.apply(b.apply(v)));
    }
}
