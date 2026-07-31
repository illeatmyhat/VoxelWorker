//! Intersection-curve tracing between two implicit surfaces — the CSG junction
//! curves selection feedback creases along: where one body's surface crosses
//! another's (a cutter's wall meeting the face it carved), the composed surface
//! creases along the space curve `{ F = 0 } ∩ { G = 0 }`. No per-shape catalog
//! contains that curve, and most shape pairs have no closed form (two turned
//! cylinders intersect in a degree-4 space curve), so it is traced numerically on
//! the two LEAF fields — never the folded CSG field, whose `min`/`max` is
//! non-differentiable exactly on the curve being traced.
//!
//! Structure: a deterministic octree descent over the pair's overlap box seeds the
//! curve (a cell is pruned only when a caller-supplied conservative BRACKET of
//! either field excludes zero over it), then a predictor–corrector marches along
//! the curve: predictor along the tangent `∇F × ∇G`, corrector a damped Newton
//! step on the pair `(F, G)` through the 2×3 Jacobian's pseudo-inverse. Tangential
//! (near-parallel-gradient) contact — a flush face — is not a crease and is
//! discarded by the same guard that keeps the pseudo-inverse well-conditioned.
//!
//! ## Verification note
//!
//! The SOUNDNESS half — "a pruned cell contains no surface point" — reduces to the
//! bracket contract plus the prune's [`PRUNE_SLOP`]. The `#[cfg(kani)]` lemma
//! below proves exactly the slopped Lipschitz form on the ±4096 value domain (the
//! slop-free form is REFUTED — its witness harness is kept beside the proof), and
//! it covers ONLY [`lipschitz_cell_bracket`]: production callers supplying the
//! evaluator's `cell_field_interval` brackets rest on that primitive's own
//! conservative-never-narrow contract, not on this lemma. The predictor–corrector
//! ITERATION is
//! numeric refinement with no finite-domain invariant a model checker can close
//! over; its failure mode is a missed or truncated polyline in a display-only
//! overlay, never an unsound claim, and it is pinned by analytic-oracle unit tests
//! (sphere∩sphere = circle, plane∩cylinder = ellipse, box∩box = the notch frame)
//! instead.

use glam::Vec3;

/// Tuning of one [`trace_intersection_curves`] run. Every length is in the
/// caller's frame units (voxels, for the placement frames this crate carries).
#[derive(Debug, Clone, Copy)]
pub struct SurfaceIntersectionConfig {
    /// Predictor arc-length step along the curve.
    pub step: f32,
    /// Central-difference half-width for gradients.
    pub gradient_half_width: f32,
    /// Newton corrector iterations per step.
    pub corrector_iterations: u32,
    /// Both fields must land within this of zero for a point to be ON the curve.
    pub on_surface_tolerance: f32,
    /// `|∇F × ∇G| / (|∇F| |∇G|)` — the sine of the dihedral opening — below this
    /// is tangential contact (a flush face, not a crease): seeds are discarded and
    /// a marching curve ends. Doubles as the pseudo-inverse conditioning guard.
    pub tangency_sine_floor: f32,
    /// Octree leaf cell size — surviving leaves become seed candidates.
    pub seed_cell: f32,
    /// Hard cap on marched steps per curve (runaway guard, not a tuning knob).
    pub max_steps_per_curve: u32,
    /// Hard cap on octree seed candidates (runaway guard).
    pub max_seeds: u32,
}

impl Default for SurfaceIntersectionConfig {
    fn default() -> Self {
        Self {
            step: 0.5,
            gradient_half_width: 0.05,
            corrector_iterations: 4,
            on_surface_tolerance: 1e-2,
            // Kills the on-tolerance phantom ring around a tangential (kissing)
            // contact: at tolerance τ the phantom sits at sine ≈ 2·sqrt(2τ/R)·…,
            // measured ≈ 0.13 for kissing spheres at τ = 1e-2 — a flush face is
            // exactly 0. True creases this shallow (< 9°) read as flush anyway.
            tangency_sine_floor: 0.15,
            seed_cell: 1.0,
            max_steps_per_curve: 4096,
            max_seeds: 4096,
        }
    }
}

/// The two implicit surfaces of one pair, with the conservative cell brackets the
/// seeding descent prunes by.
///
/// **The bracket is the soundness obligation**: `bracket(min, size)` must return an
/// interval containing EVERY value the field takes on the cube `[min, min + size]³`
/// (wider is safe, narrower silently loses curves). [`lipschitz_cell_bracket`]
/// discharges it for an L-Lipschitz field; the evaluator's `cell_field_interval`
/// discharges it per producer.
pub struct ImplicitSurfacePair<'a> {
    pub field_f: &'a dyn Fn(Vec3) -> f32,
    pub field_g: &'a dyn Fn(Vec3) -> f32,
    pub bracket_f: &'a dyn Fn(Vec3, f32) -> (f32, f32),
    pub bracket_g: &'a dyn Fn(Vec3, f32) -> (f32, f32),
}

/// Everything one [`trace_intersection_curves`] run produced, with its truncation
/// flags — no silent caps: a missing- or short-curve report starts by reading
/// these, not by hunting.
#[derive(Debug, Clone)]
pub struct SurfaceIntersectionOutcome {
    pub curves: Vec<TracedCurve>,
    /// Seed enumeration stopped at `max_seeds` with cells still unvisited —
    /// whole curves may be missing.
    pub seed_budget_exhausted: bool,
    /// At least one march stopped at `max_steps_per_curve` — that polyline is
    /// truncated mid-curve.
    pub step_budget_exhausted: bool,
}

/// One traced junction curve: an ordered polyline on `F = 0 ∩ G = 0`.
#[derive(Debug, Clone)]
pub struct TracedCurve {
    /// Curve points in the caller's frame. A closed curve does NOT repeat its
    /// first point — read [`closed`](Self::closed) to close it.
    pub points: Vec<Vec3>,
    /// Whether the march returned to its start (a loop) rather than running off
    /// the overlap box or into tangential contact.
    pub closed: bool,
    /// The smallest dihedral sine met along the curve — the caller's fade/skip
    /// weight (1 = perpendicular crease, → 0 = flush).
    pub min_dihedral_sine: f32,
}

/// The rounding slop the seeding prune grants both brackets: a cell is discarded
/// only when a bracket clears zero by MORE than this. f32 evaluates the brackets,
/// and one rounding step at an interval edge can otherwise exclude a zero sitting
/// exactly on the frontier — Kani REFUTED the slop-free claim (the witness
/// harness below keeps the counterexample class on record) and PROVES the slopped
/// claim on the ±4096 value domain the selection overlap boxes inhabit.
pub const PRUNE_SLOP: f32 = 1e-2;

/// The bracket of an L-Lipschitz field over a cell, from its center sample: every
/// value on the cell lies within `L · circumradius` of the center value — up to
/// f32 rounding at the interval edges, which the prune's [`PRUNE_SLOP`] absorbs
/// (see the `#[cfg(kani)]` lemma below for exactly what is proven).
pub fn lipschitz_cell_bracket(center_value: f32, lipschitz: f32, circumradius: f32) -> (f32, f32) {
    let spread = lipschitz * circumradius;
    (center_value - spread, center_value + spread)
}

fn gradient(field: &dyn Fn(Vec3) -> f32, point: Vec3, half_width: f32) -> Vec3 {
    let h = half_width;
    Vec3::new(
        field(point + Vec3::X * h) - field(point - Vec3::X * h),
        field(point + Vec3::Y * h) - field(point - Vec3::Y * h),
        field(point + Vec3::Z * h) - field(point - Vec3::Z * h),
    ) / (2.0 * h)
}

/// The curve tangent `∇F × ∇G` at `point`, with the dihedral sine it subtends.
/// `None` when either gradient vanishes or the surfaces are tangential there.
fn curve_tangent(
    pair: &ImplicitSurfacePair<'_>,
    point: Vec3,
    config: &SurfaceIntersectionConfig,
) -> Option<(Vec3, f32)> {
    let grad_f = gradient(pair.field_f, point, config.gradient_half_width);
    let grad_g = gradient(pair.field_g, point, config.gradient_half_width);
    let lengths = grad_f.length() * grad_g.length();
    if lengths < 1e-9 {
        return None;
    }
    let cross = grad_f.cross(grad_g);
    let sine = cross.length() / lengths;
    if sine < config.tangency_sine_floor {
        return None;
    }
    Some((cross / cross.length(), sine))
}

/// One damped Newton step toward `F = 0 ∩ G = 0` through the 2×3 Jacobian's
/// pseudo-inverse, iterated to convergence. `None` when the system is degenerate
/// (tangential) or fails to land on the curve.
fn correct_onto_curve(
    pair: &ImplicitSurfacePair<'_>,
    start: Vec3,
    config: &SurfaceIntersectionConfig,
) -> Option<Vec3> {
    let mut point = start;
    for _ in 0..config.corrector_iterations {
        let value_f = (pair.field_f)(point);
        let value_g = (pair.field_g)(point);
        if value_f.abs() <= config.on_surface_tolerance
            && value_g.abs() <= config.on_surface_tolerance
        {
            return Some(point);
        }
        let grad_f = gradient(pair.field_f, point, config.gradient_half_width);
        let grad_g = gradient(pair.field_g, point, config.gradient_half_width);
        // Solve J Δ = −(F, G) with the minimum-norm Δ = Jᵀ (J Jᵀ)⁻¹ · −(F, G).
        let ff = grad_f.dot(grad_f);
        let fg = grad_f.dot(grad_g);
        let gg = grad_g.dot(grad_g);
        let determinant = ff * gg - fg * fg;
        if determinant < 1e-9 {
            return None;
        }
        let alpha = (-value_f * gg + value_g * fg) / determinant;
        let beta = (value_f * fg - value_g * ff) / determinant;
        let delta = grad_f * alpha + grad_g * beta;
        // Damp: a corrector jump far beyond the march step means the local model
        // broke (a Chebyshev face seam) — clamp rather than fly off.
        let max_jump = 2.0 * config.step;
        point += if delta.length() > max_jump {
            delta * (max_jump / delta.length())
        } else {
            delta
        };
    }
    let value_f = (pair.field_f)(point);
    let value_g = (pair.field_g)(point);
    (value_f.abs() <= config.on_surface_tolerance && value_g.abs() <= config.on_surface_tolerance)
        .then_some(point)
}

/// March from `seed` (already ON the curve) in the direction of `first_tangent`,
/// collecting points until the curve closes, leaves `bounds`, or degenerates.
/// Returns the marched points (excluding the seed) and whether it closed;
/// `step_exhausted` is raised (never cleared) when the step cap cut the march.
fn march(
    pair: &ImplicitSurfacePair<'_>,
    seed: Vec3,
    first_tangent: Vec3,
    bounds: (Vec3, Vec3),
    config: &SurfaceIntersectionConfig,
    min_sine: &mut f32,
    step_exhausted: &mut bool,
) -> (Vec<Vec3>, bool) {
    let mut points = Vec::new();
    let mut current = seed;
    let mut heading = first_tangent;
    for step_index in 0..config.max_steps_per_curve {
        let Some((tangent, sine)) = curve_tangent(pair, current, config) else {
            return (points, false);
        };
        *min_sine = min_sine.min(sine);
        let tangent = if tangent.dot(heading) < 0.0 {
            -tangent
        } else {
            tangent
        };
        let Some(next) = correct_onto_curve(pair, current + tangent * config.step, config) else {
            return (points, false);
        };
        if step_index > 2 && next.distance(seed) < config.step * 0.75 {
            return (points, true);
        }
        let outside = (0..3).any(|axis| next[axis] < bounds.0[axis] || next[axis] > bounds.1[axis]);
        if outside {
            return (points, false);
        }
        points.push(next);
        heading = tangent;
        current = next;
    }
    *step_exhausted = true;
    (points, false)
}

/// Trace every junction curve of one surface pair inside `overlap_min ..=
/// overlap_max` (the pair's inflated placement overlap). Deterministic for given
/// inputs: the octree descends in fixed child order, seeds trace in descent
/// order, and a seed within a step of an already-traced curve is consumed.
pub fn trace_intersection_curves(
    pair: &ImplicitSurfacePair<'_>,
    overlap_min: Vec3,
    overlap_max: Vec3,
    config: &SurfaceIntersectionConfig,
) -> SurfaceIntersectionOutcome {
    let mut seed_budget_exhausted = false;
    let mut step_budget_exhausted = false;
    // Seed candidates: octree descent, pruning by the conservative brackets.
    let mut seeds: Vec<Vec3> = Vec::new();
    let extent = overlap_max - overlap_min;
    let root_size = extent.max_element();
    if root_size <= 0.0 || root_size.is_nan() {
        return SurfaceIntersectionOutcome {
            curves: Vec::new(),
            seed_budget_exhausted,
            step_budget_exhausted,
        };
    }
    let mut stack = vec![(overlap_min, root_size)];
    while let Some((cell_min, size)) = stack.pop() {
        if seeds.len() >= config.max_seeds as usize {
            seed_budget_exhausted = true;
            break;
        }
        // Clip to the overlap box (the root cube squares it off).
        if (0..3).any(|axis| cell_min[axis] > overlap_max[axis]) {
            continue;
        }
        let (low_f, high_f) = (pair.bracket_f)(cell_min, size);
        if low_f > PRUNE_SLOP || high_f < -PRUNE_SLOP {
            continue;
        }
        let (low_g, high_g) = (pair.bracket_g)(cell_min, size);
        if low_g > PRUNE_SLOP || high_g < -PRUNE_SLOP {
            continue;
        }
        if size <= config.seed_cell {
            let center = cell_min + Vec3::splat(size * 0.5);
            // A FLUSH contact floods this budget: both fields are ≈0 over the whole
            // 2D patch, every cell there passes the brackets, and the cap can then
            // exhaust before genuinely transversal cells enumerate. Screen the
            // tangency-doomed cells out here — at half the trace floor, so a
            // borderline cell whose center sits off the curve still seeds.
            let grad_f = gradient(pair.field_f, center, config.gradient_half_width);
            let grad_g = gradient(pair.field_g, center, config.gradient_half_width);
            let lengths = grad_f.length() * grad_g.length();
            if lengths >= 1e-9
                && grad_f.cross(grad_g).length() / lengths >= config.tangency_sine_floor * 0.5
            {
                seeds.push(center);
            }
            continue;
        }
        let half = size * 0.5;
        // Reverse child order: the stack pops low corners first — deterministic.
        for child in (0..8u8).rev() {
            let offset = Vec3::new(
                if child & 1 != 0 { half } else { 0.0 },
                if child & 2 != 0 { half } else { 0.0 },
                if child & 4 != 0 { half } else { 0.0 },
            );
            stack.push((cell_min + offset, half));
        }
    }

    // Trace, consuming seeds the growing curves cover.
    let bounds = (overlap_min, overlap_max);
    // A seed cell's center sits up to half its space diagonal from the curve that
    // crosses the cell — the consume radius must cover that plus a march step.
    let consume_radius = config.seed_cell * 0.87 + config.step;
    let consume_radius_squared = consume_radius * consume_radius;
    let mut curves: Vec<TracedCurve> = Vec::new();
    let mut consumed = vec![false; seeds.len()];
    for seed_index in 0..seeds.len() {
        if consumed[seed_index] {
            continue;
        }
        let Some(on_curve) = correct_onto_curve(pair, seeds[seed_index], config) else {
            continue;
        };
        // Seed consumption alone cannot dedup: a bracket admits cells FARTHER from
        // the curve than the consume radius (its slack is the caller's, unbounded
        // for an unknown interval), and the corrector then pulls such a seed onto
        // an already-traced curve. Dedup at the corrected point — two genuinely
        // distinct curves closer than the consume radius are under-resolved at
        // this step size anyway.
        if curves.iter().any(|curve| {
            curve
                .points
                .iter()
                .any(|point| point.distance_squared(on_curve) <= consume_radius_squared)
        }) {
            continue;
        }
        let mut min_sine = f32::MAX;
        let Some((tangent, sine)) = curve_tangent(pair, on_curve, config) else {
            continue;
        };
        min_sine = min_sine.min(sine);
        let (forward, closed) = march(
            pair,
            on_curve,
            tangent,
            bounds,
            config,
            &mut min_sine,
            &mut step_budget_exhausted,
        );
        let mut points = vec![on_curve];
        points.extend(forward);
        if !closed {
            let (backward, _) = march(
                pair,
                on_curve,
                -tangent,
                bounds,
                config,
                &mut min_sine,
                &mut step_budget_exhausted,
            );
            points.reverse();
            points.extend(backward);
            points.reverse();
            // `points` is now backward-reversed ++ [seed] ++ forward, seed-ordered.
        }
        if points.len() < 2 {
            continue;
        }
        for (other_index, other_seed) in seeds.iter().enumerate().skip(seed_index) {
            if consumed[other_index] {
                continue;
            }
            if points
                .iter()
                .any(|point| point.distance_squared(*other_seed) <= consume_radius_squared)
            {
                consumed[other_index] = true;
            }
        }
        curves.push(TracedCurve {
            points,
            closed,
            min_dihedral_sine: min_sine,
        });
    }
    SurfaceIntersectionOutcome {
        curves,
        seed_budget_exhausted,
        step_budget_exhausted,
    }
}

/// The seed-prune soundness lemmas, machine-checked over nondet f32. Bounded
/// model check — `cargo kani` in WSL (memory: kani-wsl-toolchain).
///
/// What is and is NOT proven: the slop-free claim "any sample consistent with the
/// Lipschitz hypothesis lies inside [`lipschitz_cell_bracket`]" is FALSE in f32 —
/// Kani refuted it (rounding of `sample - center` versus `center ± spread` under
/// catastrophic cancellation), and the witness harness pins that refutation so the
/// slop never reads as optional. The proven form adds [`PRUNE_SLOP`], on the
/// ±4096 value domain selection overlap boxes inhabit (half-ulp there is ~5e-4,
/// 20× inside the slop). Production callers supplying `cell_field_interval`
/// brackets are NOT covered by this lemma — they rest on that primitive's own
/// conservative-never-narrow contract.
#[cfg(kani)]
mod kani_proofs {
    use super::{lipschitz_cell_bracket, PRUNE_SLOP};

    const DOMAIN: f32 = 4096.0;

    /// A sample consistent with the Lipschitz hypothesis is never pruned: it lies
    /// within [`PRUNE_SLOP`] of the bracket, so a bracket clearing zero by more
    /// than the slop excludes every zero.
    ///
    /// Quantifies over `spread` directly (as `lipschitz` with circumradius 1.0,
    /// whose product is exact) rather than the `L · r` factor pair: the SAME
    /// float feeds the hypothesis and the bracket, so the product's own rounding
    /// cancels out of the theorem — any `fl(L · r)` production computes IS some
    /// such spread — while sparing the solver a nondet multiplication it cannot
    /// close within the harness timeout.
    #[kani::proof]
    fn lipschitz_bracket_with_slop_contains_every_consistent_sample() {
        let center_value: f32 = kani::any();
        let spread: f32 = kani::any();
        let sample: f32 = kani::any();
        kani::assume(center_value.abs() <= DOMAIN && sample.abs() <= DOMAIN);
        kani::assume(spread >= 0.0 && spread <= DOMAIN);
        kani::assume((sample - center_value).abs() <= spread);
        let (low, high) = lipschitz_cell_bracket(center_value, spread, 1.0);
        assert!(sample >= low - PRUNE_SLOP && sample <= high + PRUNE_SLOP);
    }

    /// The refutation witness: WITHOUT the slop the same claim fails — this
    /// harness must keep failing, or the slop has silently become dead weight.
    #[kani::proof]
    #[kani::should_panic]
    fn lipschitz_bracket_without_slop_is_refuted() {
        let center_value: f32 = kani::any();
        let lipschitz: f32 = kani::any();
        let circumradius: f32 = kani::any();
        let sample: f32 = kani::any();
        kani::assume(center_value.is_finite() && sample.is_finite());
        kani::assume(lipschitz.is_finite() && lipschitz >= 0.0);
        kani::assume(circumradius.is_finite() && circumradius >= 0.0);
        let spread = lipschitz * circumradius;
        kani::assume(spread.is_finite());
        kani::assume((sample - center_value).abs() <= spread);
        let (low, high) = lipschitz_cell_bracket(center_value, lipschitz, circumradius);
        assert!(sample >= low && sample <= high);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lipschitz_pair_config() -> SurfaceIntersectionConfig {
        SurfaceIntersectionConfig::default()
    }

    /// Brackets for a 1-Lipschitz field from its center sample.
    fn bracket_of<'a>(field: &'a dyn Fn(Vec3) -> f32) -> impl Fn(Vec3, f32) -> (f32, f32) + 'a {
        move |cell_min, size| {
            let center = cell_min + Vec3::splat(size * 0.5);
            let circumradius = (size * 0.5) * 3f32.sqrt();
            lipschitz_cell_bracket(field(center), 1.0, circumradius)
        }
    }

    /// Two offset spheres intersect in a circle: every traced point is on both
    /// surfaces, the loop closes, and the radius matches the closed form.
    #[test]
    fn sphere_pair_traces_the_analytic_circle() {
        let radius = 8.0f32;
        let center_a = Vec3::new(10.0, 10.0, 10.0);
        let center_b = Vec3::new(16.0, 10.0, 10.0);
        let field_f = move |p: Vec3| p.distance(center_a) - radius;
        let field_g = move |p: Vec3| p.distance(center_b) - radius;
        let config = lipschitz_pair_config();
        let f: &dyn Fn(Vec3) -> f32 = &field_f;
        let g: &dyn Fn(Vec3) -> f32 = &field_g;
        let bracket_f = bracket_of(f);
        let bracket_g = bracket_of(g);
        let pair = ImplicitSurfacePair {
            field_f: f,
            field_g: g,
            bracket_f: &bracket_f,
            bracket_g: &bracket_g,
        };
        let curves =
            trace_intersection_curves(&pair, Vec3::splat(0.0), Vec3::splat(26.0), &config).curves;
        assert_eq!(curves.len(), 1, "one intersection circle");
        let curve = &curves[0];
        assert!(curve.closed, "the circle closes");
        assert!(curve.points.len() > 20);
        // Closed form: the circle sits on the bisector plane x = 13 with radius
        // sqrt(r² − d²) for half-separation d = 3.
        let expected_radius = (radius * radius - 9.0f32).sqrt();
        for point in &curve.points {
            assert!(field_f(*point).abs() < 0.05, "on sphere A");
            assert!(field_g(*point).abs() < 0.05, "on sphere B");
            assert!((point.x - 13.0).abs() < 0.05, "on the bisector plane");
            let ring_radius = ((point.y - 10.0).powi(2) + (point.z - 10.0).powi(2)).sqrt();
            assert!((ring_radius - expected_radius).abs() < 0.1);
        }
        assert!(curve.min_dihedral_sine > 0.3, "a transversal crossing");
    }

    /// A plane crossing a cylinder traces the circular rim — the cavity-mouth case.
    #[test]
    fn plane_through_cylinder_traces_the_rim() {
        let field_f = |p: Vec3| p.z - 12.0; // plane z = 12
        let field_g = |p: Vec3| ((p.x - 10.0).powi(2) + (p.y - 10.0).powi(2)).sqrt() - 5.0;
        let config = lipschitz_pair_config();
        let f: &dyn Fn(Vec3) -> f32 = &field_f;
        let g: &dyn Fn(Vec3) -> f32 = &field_g;
        let bracket_f = bracket_of(f);
        let bracket_g = bracket_of(g);
        let pair = ImplicitSurfacePair {
            field_f: f,
            field_g: g,
            bracket_f: &bracket_f,
            bracket_g: &bracket_g,
        };
        let curves =
            trace_intersection_curves(&pair, Vec3::splat(0.0), Vec3::splat(20.0), &config).curves;
        assert_eq!(curves.len(), 1);
        let curve = &curves[0];
        assert!(curve.closed);
        for point in &curve.points {
            assert!((point.z - 12.0).abs() < 0.05);
            let ring_radius = ((point.x - 10.0).powi(2) + (point.y - 10.0).powi(2)).sqrt();
            assert!((ring_radius - 5.0).abs() < 0.05);
        }
        // Plane ⊥ cylinder wall: dihedral sine ≈ 1 all the way round.
        assert!(curve.min_dihedral_sine > 0.9);
    }

    /// Two Chebyshev boxes overlapping corner-in-corner: the junction survives the
    /// C0 face seams — every traced point is on both box surfaces.
    #[test]
    fn chebyshev_box_pair_traces_the_notch_frame() {
        let box_field = |center: Vec3, half: Vec3| {
            move |p: Vec3| {
                let d = (p - center).abs() - half;
                d.x.max(d.y).max(d.z)
            }
        };
        let field_f = box_field(Vec3::splat(8.0), Vec3::splat(8.0));
        let field_g = box_field(Vec3::splat(18.0), Vec3::splat(6.0));
        let config = lipschitz_pair_config();
        let f: &dyn Fn(Vec3) -> f32 = &field_f;
        let g: &dyn Fn(Vec3) -> f32 = &field_g;
        let bracket_f = bracket_of(f);
        let bracket_g = bracket_of(g);
        let pair = ImplicitSurfacePair {
            field_f: f,
            field_g: g,
            bracket_f: &bracket_f,
            bracket_g: &bracket_g,
        };
        let curves =
            trace_intersection_curves(&pair, Vec3::splat(8.0), Vec3::splat(20.0), &config).curves;
        assert!(!curves.is_empty(), "the notch junction exists");
        let mut on_surface_points = 0usize;
        for curve in &curves {
            for point in &curve.points {
                assert!(field_f(*point).abs() < 0.05, "on box A at {point}");
                assert!(field_g(*point).abs() < 0.05, "on box B at {point}");
                on_surface_points += 1;
            }
        }
        assert!(on_surface_points > 10);
    }

    /// Tangential contact — two spheres kissing at one point — creases nothing.
    #[test]
    fn tangential_contact_traces_nothing() {
        let field_f = move |p: Vec3| p.distance(Vec3::new(10.0, 10.0, 10.0)) - 5.0;
        let field_g = move |p: Vec3| p.distance(Vec3::new(20.0, 10.0, 10.0)) - 5.0;
        let config = lipschitz_pair_config();
        let f: &dyn Fn(Vec3) -> f32 = &field_f;
        let g: &dyn Fn(Vec3) -> f32 = &field_g;
        let bracket_f = bracket_of(f);
        let bracket_g = bracket_of(g);
        let pair = ImplicitSurfacePair {
            field_f: f,
            field_g: g,
            bracket_f: &bracket_f,
            bracket_g: &bracket_g,
        };
        let curves =
            trace_intersection_curves(&pair, Vec3::splat(0.0), Vec3::splat(30.0), &config).curves;
        assert!(
            curves.iter().all(|curve| curve.points.len() < 4),
            "kissing spheres have no transversal junction"
        );
    }

    /// Disjoint surfaces: the bracket prune leaves no seeds at all.
    #[test]
    fn disjoint_surfaces_seed_nothing() {
        let field_f = move |p: Vec3| p.distance(Vec3::new(5.0, 5.0, 5.0)) - 2.0;
        let field_g = move |p: Vec3| p.distance(Vec3::new(25.0, 25.0, 25.0)) - 2.0;
        let config = lipschitz_pair_config();
        let f: &dyn Fn(Vec3) -> f32 = &field_f;
        let g: &dyn Fn(Vec3) -> f32 = &field_g;
        let bracket_f = bracket_of(f);
        let bracket_g = bracket_of(g);
        let pair = ImplicitSurfacePair {
            field_f: f,
            field_g: g,
            bracket_f: &bracket_f,
            bracket_g: &bracket_g,
        };
        let curves =
            trace_intersection_curves(&pair, Vec3::splat(0.0), Vec3::splat(30.0), &config).curves;
        assert!(curves.is_empty());
    }

    /// Determinism: the same inputs trace bit-identical curves.
    #[test]
    fn tracing_is_deterministic() {
        let field_f = move |p: Vec3| p.distance(Vec3::new(10.0, 10.0, 10.0)) - 8.0;
        let field_g = move |p: Vec3| p.distance(Vec3::new(16.0, 10.0, 10.0)) - 8.0;
        let config = lipschitz_pair_config();
        let f: &dyn Fn(Vec3) -> f32 = &field_f;
        let g: &dyn Fn(Vec3) -> f32 = &field_g;
        let bracket_f = bracket_of(f);
        let bracket_g = bracket_of(g);
        let pair = ImplicitSurfacePair {
            field_f: f,
            field_g: g,
            bracket_f: &bracket_f,
            bracket_g: &bracket_g,
        };
        let run = || {
            trace_intersection_curves(&pair, Vec3::splat(0.0), Vec3::splat(26.0), &config)
                .curves
                .into_iter()
                .flat_map(|curve| curve.points)
                .flat_map(|point| [point.x.to_bits(), point.y.to_bits(), point.z.to_bits()])
                .collect::<Vec<u32>>()
        };
        assert_eq!(run(), run());
    }

    /// The tangency floor's two sides, on a GRAZING but transversal crease: a
    /// cylinder entering a plane at 10° off-horizontal (dihedral sine ≈ 0.17 at
    /// the ellipse's shallow tips, just above the 0.15 floor) traces its full
    /// closed ellipse; the same cylinder at 5° (sine ≈ 0.09) loses its tips to
    /// the floor and degenerates to open side arcs — never a phantom closure.
    #[test]
    fn grazing_transversal_crease_survives_the_floor_until_it_reads_flush() {
        let config = lipschitz_pair_config();
        let field_f = |p: Vec3| p.z - 10.0;
        let trace_at_tilt = |tilt_degrees: f32| {
            let tilt = tilt_degrees.to_radians();
            let axis_direction = Vec3::new(tilt.cos(), 0.0, tilt.sin());
            let axis_point = Vec3::new(30.0, 30.0, 10.0);
            let field_g = move |p: Vec3| {
                let offset = p - axis_point;
                (offset - axis_direction * offset.dot(axis_direction)).length() - 4.0
            };
            let f: &dyn Fn(Vec3) -> f32 = &field_f;
            let g: &dyn Fn(Vec3) -> f32 = &field_g;
            let bracket_f = bracket_of(f);
            let bracket_g = bracket_of(g);
            let pair = ImplicitSurfacePair {
                field_f: f,
                field_g: g,
                bracket_f: &bracket_f,
                bracket_g: &bracket_g,
            };
            trace_intersection_curves(&pair, Vec3::splat(0.0), Vec3::splat(60.0), &config)
        };

        let above = trace_at_tilt(10.0);
        assert!(!above.seed_budget_exhausted && !above.step_budget_exhausted);
        assert_eq!(above.curves.len(), 1, "one ellipse, traced once");
        let ellipse = &above.curves[0];
        assert!(ellipse.closed, "shallow but legitimate: the ellipse closes");
        let min_x = ellipse.points.iter().map(|p| p.x).fold(f32::MAX, f32::min);
        let max_x = ellipse.points.iter().map(|p| p.x).fold(f32::MIN, f32::max);
        // Semi-major axis r / sin(tilt) ≈ 23: the march crossed the shallow tips.
        assert!(max_x - min_x > 40.0, "full elongation, tips included");
        assert!(
            ellipse.min_dihedral_sine > 0.15 && ellipse.min_dihedral_sine < 0.25,
            "the recorded minimum is the tip dihedral, got {}",
            ellipse.min_dihedral_sine
        );

        let below = trace_at_tilt(5.0);
        assert!(
            below.curves.iter().all(|curve| !curve.closed),
            "below the floor the tips read as flush: open side arcs only"
        );
    }
}
