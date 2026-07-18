//! Interval arithmetic for a signed scalar field under CSG lattice operations.
//!
//! [`FieldInterval`] is a conservative bound `[minimum, maximum]` on the value a
//! signed scalar field takes over some region (a cell). It is the classic tool of
//! **interval analysis**: replace a value by an enclosing interval and lift each
//! operation to its interval extension, so a computation over intervals brackets the
//! same computation over any point sample the intervals contain. Here the operations
//! are the Boolean set operations of **constructive solid geometry (CSG)** expressed
//! on the field: with the sign convention "inside where `field <= isolevel`", union of
//! two solids is the pointwise `min` of their fields (the nearer surface wins),
//! intersection is `max`, complement is negation, and difference `A − B` is
//! `max(field_A, −field_B)`. Lifting each to intervals gives:
//!
//! * union       `[min(aMin,bMin), min(aMax,bMax)]`   (a bound on `min(a, b)`)
//! * intersect   `[max(aMin,bMin), max(aMax,bMax)]`   (a bound on `max(a, b)`)
//! * negate      `[−aMax, −aMin]`
//! * subtract    `intersect(a, negate(b))`
//!
//! Given a bound, [`FieldInterval::classify`] answers the three-way membership query
//! of a whole region against a threshold `isolevel`: wholly outside, wholly inside, or
//! straddling the surface (the classic **black / white / grey** cell trichotomy). The
//! bound is **conservative** — it may be WIDER than the true field range but never
//! narrower — so a coarse "wholly inside / wholly outside" verdict can never disagree
//! with a per-sample evaluation; only the always-safe "straddling" verdict can be
//! reported where per-sample evaluation would have decided. That one-sided soundness
//! is the whole value of the structure.
//!
//! [`FieldInterval::from_lipschitz_center`] builds the bound for a **1-Lipschitz**
//! field (a true signed-distance field, whose value changes by at most the travelled
//! distance): from the centre sample and the region's circumradius `r`, the field over
//! the region lies within `[centre − r, centre + r]`. Those two endpoints are the only
//! place `f32` rounding could narrow an interval, so both are rounded **outward** — the
//! standard directed-rounding discipline of interval arithmetic. Every other operation
//! here (`min`/`max`/negation/compare) is exact in IEEE-754, so containment is rigorous
//! throughout, not merely display-grade.
//!
//! Cite: Moore 1966 / Moore, Kearfott & Cloud, *Introduction to Interval Analysis*
//! (2009) — interval arithmetic and the containment (inclusion) property. Duff 1992,
//! *Interval arithmetic and recursive subdivision for implicit functions and
//! constructive solid geometry* (SIGGRAPH) — exactly this classify-a-cell-under-CSG
//! use, the source of the black/white/grey subdivision. Hart 1996, *Sphere tracing* —
//! the Lipschitz bound that makes a distance field's centre-plus-radius interval sound.
//! Deviation: bounds are `f32` rather than a wider type, and the classify threshold is a
//! plain parameter rather than a fixed constant.

/// A conservative interval `[minimum, maximum]` bounding a signed scalar field over a
/// region. Conservative means the true field range over the region is CONTAINED in
/// `[minimum, maximum]` (the bound is never narrower than the truth).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldInterval {
    /// Conservative LOWER bound on the field over the region (`<=` its true minimum).
    pub minimum: f32,
    /// Conservative UPPER bound on the field over the region (`>=` its true maximum).
    pub maximum: f32,
}

impl FieldInterval {
    /// An interval `[minimum, maximum]` (callers pass already-conservative bounds).
    pub fn new(minimum: f32, maximum: f32) -> Self {
        Self { minimum, maximum }
    }

    /// The interval of a 1-Lipschitz signed field over a region whose centre sample is
    /// `field_at_center` and whose circumradius (half the space-diagonal) is
    /// `cell_circumradius`: `[field_at_center − r, field_at_center + r]`. A 1-Lipschitz
    /// field changes by at most `r` between the centre and any point within radius `r`,
    /// so this brackets every in-region sample. If a given field is only *approximately*
    /// 1-Lipschitz the caller must WIDEN `cell_circumradius`, never narrow it.
    ///
    /// Both endpoints are rounded OUTWARD by one ULP. `fl(c − r)` rounds to nearest, so
    /// it may land ABOVE the true difference (and `fl(c + r)` below the true sum) by up
    /// to half an ULP — which would make the interval marginally too NARROW, breaking the
    /// never-narrower contract the coarse verdicts rest on. Widening each endpoint one ULP
    /// outward restores rigorous containment; too-wide is always safe here, since it can
    /// only yield `Boundary` where a sharper bound would have decided.
    ///
    /// The widening is a contract repair, not a bug fix for any observed misclassification.
    /// At the isolevel `0`, the narrowing is UNREACHABLE: an endpoint can only round across
    /// zero when `|c| ≈ r`, and by **Sterbenz's lemma** (`fl(a − b)` is exact when `a` and
    /// `b` lie within a factor of two) that regime is exactly where the arithmetic does not
    /// round at all. A search over 5.6e8 `(c, r)` pairs — exhaustive across the cancellation
    /// band — found zero flipped verdicts at isolevel `0`, and flips within seconds once the
    /// threshold moved off zero. Since `classify` takes the isolevel as a PARAMETER, the
    /// guarantee should not rest on every caller happening to pass `0`.
    pub fn from_lipschitz_center(field_at_center: f32, cell_circumradius: f32) -> Self {
        let radius = cell_circumradius.abs();
        Self {
            minimum: (field_at_center - radius).next_down(),
            maximum: (field_at_center + radius).next_up(),
        }
    }

    /// Classify the region against an occupancy threshold `isolevel` (inside where
    /// `field <= isolevel`). Because the interval is conservative, an "all outside" or
    /// "all inside" verdict can never disagree with a per-sample evaluation.
    pub fn classify(&self, isolevel: f32) -> FieldClassification {
        if self.minimum > isolevel {
            // Every sample is strictly outside ⇒ the region is entirely empty.
            FieldClassification::Air
        } else if self.maximum <= isolevel {
            // Every sample is at-or-below the isolevel ⇒ the region is entirely full.
            FieldClassification::CoarseSolid
        } else {
            // The interval straddles the isolevel ⇒ the region cannot be decided coarsely.
            FieldClassification::Boundary
        }
    }

    /// CSG UNION of two field intervals (`min(field_a, field_b)`, the nearer surface
    /// wins): the bound on `min(a, b)` is `[min(aMin,bMin), min(aMax,bMax)]`.
    pub fn union(self, other: FieldInterval) -> FieldInterval {
        FieldInterval {
            minimum: self.minimum.min(other.minimum),
            maximum: self.maximum.min(other.maximum),
        }
    }

    /// CSG INTERSECTION of two field intervals (`max(field_a, field_b)`): the bound on
    /// `max(a, b)` is `[max(aMin,bMin), max(aMax,bMax)]`.
    pub fn intersect(self, other: FieldInterval) -> FieldInterval {
        FieldInterval {
            minimum: self.minimum.max(other.minimum),
            maximum: self.maximum.max(other.maximum),
        }
    }

    /// The NEGATED field interval (`−field`): `[−maximum, −minimum]`. Used to compose
    /// a subtraction, whose field is `max(field_a, −field_b)`.
    pub fn negate(self) -> FieldInterval {
        FieldInterval {
            minimum: -self.maximum,
            maximum: -self.minimum,
        }
    }

    /// CSG SUBTRACTION `A − B` of two field intervals (`max(field_a, −field_b)`): the
    /// region inside `A` and outside `B`. Composed as `self.intersect(other.negate())`.
    pub fn subtract(self, other: FieldInterval) -> FieldInterval {
        self.intersect(other.negate())
    }
}

/// The three-way verdict a [`FieldInterval`] yields against a threshold: the classic
/// black / white / grey (empty / full / partial) cell classification of the octree-CSG
/// literature (Duff 1992; Samet).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldClassification {
    /// Every sample in the region is EMPTY (the conservative interval is all-above the
    /// isolevel) — the "white" node.
    Air,
    /// Every sample in the region is FULL (the interval is all-at-or-below the isolevel)
    /// — the "black" node; no per-sample data needed.
    CoarseSolid,
    /// The interval straddles the isolevel (or the field could not be bounded): the
    /// region must be resolved per-sample. The "grey" node — always the SAFE verdict.
    Boundary,
}

/// The conservative field interval of a CSG UNION over a list of operands — `None`
/// (unboundable ⇒ straddling) the moment any operand is `None`, since a union with an
/// unbounded field cannot be coarsely decided. An empty list yields `None`.
pub fn union_field_intervals(
    intervals: impl IntoIterator<Item = Option<FieldInterval>>,
) -> Option<FieldInterval> {
    let mut accumulated: Option<FieldInterval> = None;
    let mut any = false;
    for interval in intervals {
        any = true;
        let interval = interval?;
        accumulated = Some(match accumulated {
            Some(existing) => existing.union(interval),
            None => interval,
        });
    }
    if any {
        accumulated
    } else {
        None
    }
}

/// Bounded model checking of the INCLUSION property — the one-sided soundness every coarse
/// verdict rests on — over symbolic IEEE-754 floats rather than sampled ones. The unit tests
/// below fuzz this with an LCG; Kani quantifies over every non-NaN bit pattern, which is what
/// it takes to claim the property rather than observe it. This is the float counterpart to the
/// machine-integer harnesses in `interval::rational`: same question (does the abstraction leak
/// through the machine representation?), different representation.
///
/// The CSG operations are `min`/`max`/negation only, all exact in IEEE-754, so inclusion there
/// is expected to hold outright. The interesting harness is the Lipschitz one, where `−`/`+`
/// genuinely round — it checks the `f32` bound against the same computation in `f64`, which is
/// exact for these operands (any `f32 ± f32` is representable in `f64`).
///
/// `#[cfg(kani)]` keeps these out of ordinary builds. Run under WSL with
/// `cargo kani -p substrate -j --output-format=terse`, or all three tiers via
/// `verification/run-all.sh`. No `unwind` bound is needed: these harnesses are loop-free.
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// A finite (non-NaN, non-infinite) symbolic float. Infinities are excluded because they
    /// make the containment question degenerate — `(+inf).next_down()` is `f32::MAX`, which
    /// reads as a narrowing but bounds nothing meaningful. Callers pass sampled field values
    /// and a geometric radius; neither is ever infinite.
    fn any_finite_f32() -> f32 {
        let value: f32 = kani::any();
        kani::assume(value.is_finite());
        value
    }

    /// A symbolic interval together with a symbolic sample it contains.
    fn any_interval_with_contained_sample() -> (FieldInterval, f32) {
        let minimum = any_finite_f32();
        let maximum = any_finite_f32();
        let sample = any_finite_f32();
        kani::assume(minimum <= maximum);
        kani::assume(minimum <= sample && sample <= maximum);
        (FieldInterval { minimum, maximum }, sample)
    }

    /// The inclusion property under every CSG operation: if `sample_a` is bracketed by `a` and
    /// `sample_b` by `b`, then the composed field value is bracketed by the composed interval.
    /// This is the theorem `csg_composition_brackets_every_sample` samples 5000 points of.
    #[kani::proof]
    fn csg_composition_brackets_every_contained_sample() {
        let (a, sample_a) = any_interval_with_contained_sample();
        let (b, sample_b) = any_interval_with_contained_sample();

        let union = a.union(b);
        let union_value = sample_a.min(sample_b);
        assert!(union.minimum <= union_value && union_value <= union.maximum);

        let intersect = a.intersect(b);
        let intersect_value = sample_a.max(sample_b);
        assert!(intersect.minimum <= intersect_value && intersect_value <= intersect.maximum);

        let subtract = a.subtract(b);
        let subtract_value = sample_a.max(-sample_b);
        assert!(subtract.minimum <= subtract_value && subtract_value <= subtract.maximum);
    }

    /// The verdicts are ONE-SIDED SOUND: a coarse `Air` or `CoarseSolid` can never disagree with
    /// a per-sample evaluation of any point the interval contains. `Boundary` asserts nothing —
    /// it is the always-safe answer, which is exactly why widening a bound is harmless.
    #[kani::proof]
    fn coarse_verdicts_never_disagree_with_a_contained_sample() {
        let (interval, sample) = any_interval_with_contained_sample();
        let isolevel = any_finite_f32();
        match interval.classify(isolevel) {
            // Claimed entirely empty ⇒ the sample must indeed be outside.
            FieldClassification::Air => assert!(sample > isolevel),
            // Claimed entirely full ⇒ the sample must indeed be at-or-below the isolevel.
            FieldClassification::CoarseSolid => assert!(sample <= isolevel),
            FieldClassification::Boundary => {}
        }
    }

    /// The Lipschitz bound never rounds INWARD. `f64` arithmetic on `f32` operands is exact, so
    /// it stands in for real arithmetic: the `f32` endpoints must enclose the exact ones. Without
    /// the outward `next_down`/`next_up` this fails — round-to-nearest moves each endpoint up to
    /// half an ULP the wrong way, producing an interval narrower than the truth. (Verified: with
    /// the widening removed, BOTH assertions below are refuted in 0.1 s, so neither is vacuous.)
    ///
    /// The two endpoints are SEPARATE harnesses purely for cost. Mixed-precision float reasoning
    /// is what makes this expensive — CBMC bit-blasts the `f32`→`f64` conversions and the `f64`
    /// arithmetic — and splitting lets the battery's `-j` verify them on parallel threads, which
    /// is the only lever that moved the number. Restricting the operands to the scene coordinate
    /// domain, the obvious first idea, bought almost nothing (199 s → 174 s): unlike the integer
    /// harnesses in `interval::rational`, where narrowing the domain was decisive, a float range
    /// assumption does not shrink the search — the bit-width does, and that is fixed.
    fn lipschitz_endpoints() -> (FieldInterval, f64, f64) {
        let field_at_center = any_finite_f32();
        let cell_circumradius = any_finite_f32();
        let interval = FieldInterval::from_lipschitz_center(field_at_center, cell_circumradius);
        kani::assume(interval.minimum.is_finite() && interval.maximum.is_finite());
        let radius = cell_circumradius.abs() as f64;
        (
            interval,
            field_at_center as f64 - radius,
            field_at_center as f64 + radius,
        )
    }

    /// The LOWER endpoint encloses the exact difference (never rounds up past it).
    #[kani::proof]
    fn lipschitz_lower_bound_encloses_exact_arithmetic() {
        let (interval, exact_minimum, _) = lipschitz_endpoints();
        assert!((interval.minimum as f64) <= exact_minimum);
    }

    /// The UPPER endpoint encloses the exact sum (never rounds down past it).
    #[kani::proof]
    fn lipschitz_upper_bound_encloses_exact_arithmetic() {
        let (interval, _, exact_maximum) = lipschitz_endpoints();
        assert!((interval.maximum as f64) >= exact_maximum);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny deterministic LCG so the composition fuzz is reproducible with no
    /// dev-dependency (Numerical Recipes constants).
    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn next_u64(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.state
        }
    }

    #[test]
    fn csg_interval_union_is_min_of_fields() {
        let a = FieldInterval::new(-2.0, 3.0);
        let b = FieldInterval::new(-5.0, 1.0);
        // union = min(a, b) ⇒ [min(min), min(max)].
        assert_eq!(a.union(b), FieldInterval::new(-5.0, 1.0));
        assert_eq!(a.union(b), b.union(a), "union is commutative");
    }

    #[test]
    fn csg_interval_intersect_is_max_of_fields() {
        let a = FieldInterval::new(-2.0, 3.0);
        let b = FieldInterval::new(-5.0, 1.0);
        // intersect = max(a, b) ⇒ [max(min), max(max)].
        assert_eq!(a.intersect(b), FieldInterval::new(-2.0, 3.0));
        assert_eq!(a.intersect(b), b.intersect(a), "intersect is commutative");
    }

    #[test]
    fn csg_interval_subtract_is_intersect_with_negated() {
        let a = FieldInterval::new(-2.0, 3.0);
        let b = FieldInterval::new(-5.0, 1.0);
        // A − B = max(dA, −dB); −b = [−1, 5]; max ⇒ [max(-2,-1), max(3,5)] = [-1, 5].
        assert_eq!(b.negate(), FieldInterval::new(-1.0, 5.0));
        assert_eq!(a.subtract(b), FieldInterval::new(-1.0, 5.0));
    }

    /// The interval-arithmetic composition must AGREE with the brute-force field over a
    /// sampled field range: for any pair of sample fields drawn from `[aMin,aMax]` and
    /// `[bMin,bMax]`, the composed value lies inside the composed interval. This is the
    /// soundness (inclusion) property the classifier relies on.
    #[test]
    fn csg_composition_brackets_every_sample() {
        let a = FieldInterval::new(-2.0, 3.0);
        let b = FieldInterval::new(-5.0, 1.0);
        let union = a.union(b);
        let intersect = a.intersect(b);
        let subtract = a.subtract(b);
        let mut rng = Lcg::new(0xC56_u64);
        for _ in 0..5000 {
            let sample_a =
                a.minimum + (a.maximum - a.minimum) * (rng.next_u64() as f32 / u64::MAX as f32);
            let sample_b =
                b.minimum + (b.maximum - b.minimum) * (rng.next_u64() as f32 / u64::MAX as f32);
            let union_value = sample_a.min(sample_b);
            let intersect_value = sample_a.max(sample_b);
            let subtract_value = sample_a.max(-sample_b);
            assert!(
                (union.minimum..=union.maximum).contains(&union_value),
                "union {union:?} fails to bracket {union_value}"
            );
            assert!(
                (intersect.minimum..=intersect.maximum).contains(&intersect_value),
                "intersect {intersect:?} fails to bracket {intersect_value}"
            );
            assert!(
                (subtract.minimum..=subtract.maximum).contains(&subtract_value),
                "subtract {subtract:?} fails to bracket {subtract_value}"
            );
        }
    }

    #[test]
    fn union_field_intervals_is_none_when_any_operand_unboundable() {
        let bounded = FieldInterval::new(-1.0, 1.0);
        assert_eq!(
            union_field_intervals([Some(bounded), Some(FieldInterval::new(-3.0, 0.5))]),
            Some(FieldInterval::new(-3.0, 0.5))
        );
        // Any None operand collapses the union to None (the unbounded operand could be
        // occupied anywhere ⇒ the whole union must be treated as straddling).
        assert_eq!(
            union_field_intervals([Some(bounded), None, Some(bounded)]),
            None
        );
        // Empty list ⇒ None.
        assert_eq!(union_field_intervals(std::iter::empty()), None);
    }

    #[test]
    fn classify_is_three_way_against_the_threshold() {
        let isolevel = 0.0;
        // All-above ⇒ empty.
        assert_eq!(
            FieldInterval::new(0.5, 2.0).classify(isolevel),
            FieldClassification::Air
        );
        // All-at-or-below ⇒ full (note the closed upper bound: max == isolevel is full).
        assert_eq!(
            FieldInterval::new(-2.0, 0.0).classify(isolevel),
            FieldClassification::CoarseSolid
        );
        // Straddling ⇒ boundary.
        assert_eq!(
            FieldInterval::new(-1.0, 1.0).classify(isolevel),
            FieldClassification::Boundary
        );
    }

    #[test]
    fn lipschitz_center_brackets_by_the_circumradius() {
        // The endpoints round one ULP OUTWARD, so the bound encloses [2.5, 5.5] rather
        // than equalling it — never narrower, which is the contract.
        let interval = FieldInterval::from_lipschitz_center(4.0, 1.5);
        assert!(interval.minimum < 2.5 && interval.minimum == 2.5f32.next_down());
        assert!(interval.maximum > 5.5 && interval.maximum == 5.5f32.next_up());
        // A negative circumradius is treated by magnitude (never narrows the bound).
        assert_eq!(
            FieldInterval::from_lipschitz_center(4.0, -1.5),
            interval,
            "circumradius sign must not affect the bound"
        );
    }

    /// The Lipschitz bound must enclose the true real-arithmetic endpoints even where
    /// `f32` subtraction/addition rounds. Computing the endpoints in `f64` (exact for
    /// these operands, since `f32 ± f32` always fits `f64`) gives the truth to compare
    /// against; the `f32` bound must never fall inside it.
    #[test]
    fn lipschitz_bound_is_never_narrower_than_exact_arithmetic() {
        let mut rng = Lcg::new(0x11F5_u64);
        for _ in 0..20_000 {
            // Spread magnitudes across the large-coordinate range where ULPs get coarse.
            let center = (rng.next_u64() as f32 / u64::MAX as f32 - 0.5) * 40_000.0;
            let radius = (rng.next_u64() as f32 / u64::MAX as f32) * 100.0;
            let interval = FieldInterval::from_lipschitz_center(center, radius);
            let exact_minimum = center as f64 - radius as f64;
            let exact_maximum = center as f64 + radius as f64;
            assert!(
                (interval.minimum as f64) <= exact_minimum,
                "lower bound {} rounded INWARD past {exact_minimum}",
                interval.minimum
            );
            assert!(
                (interval.maximum as f64) >= exact_maximum,
                "upper bound {} rounded INWARD past {exact_maximum}",
                interval.maximum
            );
        }
    }
}
