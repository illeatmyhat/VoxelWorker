//! Continuous value geometry shared by planar curve relations.

pub(super) const SATISFACTION_TOLERANCE: f64 = 1.0e-6;
pub(super) const COLLAPSE_TOLERANCE: f64 = 1.0e-6;

/// The minimum continuous geometry needed to evaluate a sketch curve relation. It is a value
/// descriptor, not a persistent entity reference: document ids and solver slots remain outside.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CurveGeometry {
    Segment { from: [f64; 2], to: [f64; 2] },
    Circular(CircularCurve),
}

/// A supporting circle, optionally restricted to an authored arc domain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CircularCurve {
    pub center: [f64; 2],
    pub radius: f64,
    pub arc: Option<ArcDomain>,
}

/// The finite authored sweep of a circular arc, measured from `from` with its sign.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArcDomain {
    pub from: [f64; 2],
    pub to: [f64; 2],
    pub sweep_radians: f64,
}

/// The place on a segment's DRAWN extent nearest `at` — the foot of the perpendicular, pulled
/// back to whichever end it overshot.
///
/// This is the companion to [`within_drawn_extent`]: that one asks whether a point is already on
/// the piece, this one says where the piece would have it stand. A tool that inserts a point ONTO
/// a curve has to ask, because the place the author pointed at is a cursor reading and may have
/// been snapped to a grid the curve does not run along — committing it verbatim puts a bend in a
/// straight edge and calls it a split.
///
/// `None` for a collapsed span, which has no direction to project along.
#[must_use]
pub fn foot_on_span(from: [f64; 2], to: [f64; 2], at: [f64; 2]) -> Option<[f64; 2]> {
    let span = [to[0] - from[0], to[1] - from[1]];
    let length_squared = span[0].mul_add(span[0], span[1] * span[1]);
    if length_squared <= COLLAPSE_TOLERANCE.powi(2) {
        return None;
    }
    let delta = [at[0] - from[0], at[1] - from[1]];
    let parameter =
        (delta[0].mul_add(span[0], delta[1] * span[1]) / length_squared).clamp(0.0, 1.0);
    Some([
        parameter.mul_add(span[0], from[0]),
        parameter.mul_add(span[1], from[1]),
    ])
}

/// Whether `at` lies within a curve's DRAWN extent — the finite piece between its own ends —
/// rather than merely somewhere on the support it was cut from.
///
/// The support answers WHERE a curve lies; this answers HOW FAR it runs, and the two are different
/// questions that the same value can be asked. Keeping them apart is the point: a residual wants
/// the support, because a test that had to report "off the end" would be discontinuous where the
/// piece stops and the optimizer would be walking a cliff. An authoring GATE wants this, because
/// a discontinuity at the endpoint is exactly the answer — either the author drew the two things
/// touching or they did not.
///
/// `slack` is a distance, converted internally to whatever the curve measures its extent in: a
/// fraction of the span for a segment, an angle for an arc. Callers set it from their own
/// tolerance rather than inheriting one, because a solver contact and an author's assertion are
/// held to deliberately different standards.
///
/// A whole circle has no ends, so everything standing on one is within it.
#[must_use]
pub fn within_drawn_extent(curve: CurveGeometry, at: [f64; 2], slack: f64) -> bool {
    match curve {
        CurveGeometry::Segment { from, to } => {
            let span = [to[0] - from[0], to[1] - from[1]];
            let length_squared = span[0].mul_add(span[0], span[1] * span[1]);
            if length_squared <= COLLAPSE_TOLERANCE.powi(2) {
                return false;
            }
            let length = length_squared.sqrt();
            let delta = [at[0] - from[0], at[1] - from[1]];
            let parameter = delta[0].mul_add(span[0], delta[1] * span[1]) / length_squared;
            parameter >= -slack / length && parameter <= 1.0 + slack / length
        }
        CurveGeometry::Circular(circular) => {
            let Some(arc) = circular.arc else {
                return true;
            };
            if !arc.sweep_radians.is_finite() || arc.sweep_radians.abs() <= f64::EPSILON {
                return false;
            }
            let start_angle =
                (arc.from[1] - circular.center[1]).atan2(arc.from[0] - circular.center[0]);
            let here_angle = (at[1] - circular.center[1]).atan2(at[0] - circular.center[0]);
            let travel = if arc.sweep_radians.is_sign_positive() {
                (here_angle - start_angle).rem_euclid(std::f64::consts::TAU)
            } else {
                (start_angle - here_angle).rem_euclid(std::f64::consts::TAU)
            };
            let angular_slack = (slack / circular.radius).clamp(f64::EPSILON * 64.0, 1.0e-3);
            // `travel` alone wraps a point a hair before the start to almost one full turn.
            // Endpoints are authored domain boundaries, so the same bounded tolerance applies on
            // either side of both ends and works for positive and negative sweeps.
            let near_start = (here_angle - start_angle)
                .abs()
                .min(std::f64::consts::TAU - (here_angle - start_angle).abs())
                <= angular_slack;
            travel <= arc.sweep_radians.abs() + angular_slack || near_start
        }
    }
}
