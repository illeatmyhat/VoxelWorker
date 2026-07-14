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
//! the region lies within `[centre − r, centre + r]`.
//!
//! Cite: Moore 1966 / Moore, Kearfott & Cloud, *Introduction to Interval Analysis*
//! (2009) — interval arithmetic and the containment (inclusion) property. Duff 1992,
//! *Interval arithmetic and recursive subdivision for implicit functions and
//! constructive solid geometry* (SIGGRAPH) — exactly this classify-a-cell-under-CSG
//! use, the source of the black/white/grey subdivision. Hart 1996, *Sphere tracing* —
//! the Lipschitz bound that makes a distance field's centre-plus-radius interval sound.
//! Deviation: `f32` bounds (display-grade precision, not a rigorous rounded-interval
//! containment); the classify threshold is a plain parameter, not a fixed constant.

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
    pub fn from_lipschitz_center(field_at_center: f32, cell_circumradius: f32) -> Self {
        let radius = cell_circumradius.abs();
        Self {
            minimum: field_at_center - radius,
            maximum: field_at_center + radius,
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
        let interval = FieldInterval::from_lipschitz_center(4.0, 1.5);
        assert_eq!(interval, FieldInterval::new(2.5, 5.5));
        // A negative circumradius is treated by magnitude (never narrows the bound).
        assert_eq!(
            FieldInterval::from_lipschitz_center(4.0, -1.5),
            FieldInterval::new(2.5, 5.5)
        );
    }
}
