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
