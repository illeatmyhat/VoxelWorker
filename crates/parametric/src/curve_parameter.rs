//! Source-or-free authority for curve-intrinsic scalar values.
//!
//! The document owns persistence and evaluation policy; this module only states which value is
//! solver-writable and which is source-owned. A fixed source never carries a resolved-value cache.

use crate::units::{AngleMeasurement, ExactRational, Measurement, RationalFromF64Error};
use serde::de::Error as _;
use serde::ser::SerializeStruct;

/// One curve-intrinsic parameter with exactly one authority.
///
/// A free value is persisted solver state. A fixed source is resolved by a caller-supplied
/// evaluation context at the document-to-parametric seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct CurveParameter<FreeValue, FixedSource>(CurveParameterState<FreeValue, FixedSource>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum CurveParameterState<FreeValue, FixedSource> {
    Free(FreeValue),
    Fixed(FixedSource),
}

impl<FreeValue, FixedSource> CurveParameter<FreeValue, FixedSource> {
    /// A solver-writable intrinsic value.
    pub const fn free(value: FreeValue) -> Self {
        Self(CurveParameterState::Free(value))
    }

    /// A source-owned intrinsic value, fixed for one solve.
    pub const fn fixed(source: FixedSource) -> Self {
        Self(CurveParameterState::Fixed(source))
    }

    /// The solver-writable value, when this parameter is free.
    pub const fn free_value(&self) -> Option<&FreeValue> {
        match &self.0 {
            CurveParameterState::Free(value) => Some(value),
            CurveParameterState::Fixed(_) => None,
        }
    }

    /// The source-owned value, when this parameter is fixed.
    pub const fn fixed_source(&self) -> Option<&FixedSource> {
        match &self.0 {
            CurveParameterState::Free(_) => None,
            CurveParameterState::Fixed(source) => Some(source),
        }
    }
}

/// A solved length in exact voxel units.
///
/// This generic resolved value may be signed for future offset-like parameters. A radius's
/// positive-domain invariant belongs to its construction/solver door, not this representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedLength {
    numerator: i128,
    denominator: i128,
}

impl serde::Serialize for ResolvedLength {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut stored = serializer.serialize_struct("ResolvedLength", 2)?;
        stored.serialize_field("numerator", &self.numerator)?;
        stored.serialize_field("denominator", &self.denominator)?;
        stored.end()
    }
}

impl<'de> serde::Deserialize<'de> for ResolvedLength {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Stored {
            numerator: i128,
            denominator: i128,
        }
        let stored = Stored::deserialize(deserializer)?;
        ExactRational::new(stored.numerator, stored.denominator)
            .map(Self::from_rational)
            .ok_or_else(|| D::Error::custom("resolved length has a zero denominator"))
    }
}

impl ResolvedLength {
    /// Preserve an exact rational solved value.
    #[must_use]
    pub const fn from_rational(value: ExactRational) -> Self {
        Self {
            numerator: value.numerator(),
            denominator: value.denominator(),
        }
    }

    /// A whole-voxel value, possibly signed.
    #[must_use]
    #[allow(clippy::as_conversions)]
    pub const fn from_voxels(voxels: i64) -> Self {
        Self::from_rational(ExactRational::from_integer(voxels as i128))
    }

    /// Preserve a finite solved floating-point value as its exact rational ratio.
    ///
    /// # Errors
    ///
    /// Returns the exact-rational conversion error when `voxels` is non-finite or cannot fit in
    /// the durable `i128` ratio representation.
    pub fn try_from_f64(voxels: f64) -> Result<Self, RationalFromF64Error> {
        ExactRational::try_from_f64_exact(voxels).map(Self::from_rational)
    }

    /// The exact value.
    ///
    /// # Panics
    ///
    /// Panics only if a value was fabricated outside this type's constructors or deserializer;
    /// both preserve the non-zero canonical denominator invariant.
    #[allow(clippy::expect_used)]
    pub fn rational(self) -> ExactRational {
        ExactRational::new(self.numerator, self.denominator)
            .expect("a resolved length stores a canonical rational")
    }

    /// The evaluation width consumed by the continuous solver.
    #[must_use]
    pub fn value(self) -> f64 {
        self.rational().to_f64()
    }

    /// Re-target an authored free length by an exact integer ratio. This is the density migration
    /// door: it never round-trips through `f64`, and reports overflow rather than minting a
    /// nearby value.
    pub fn scaled_by_ratio(self, numerator: u32, denominator: u32) -> Option<Self> {
        let numerator = self.numerator.checked_mul(i128::from(numerator))?;
        let denominator = self.denominator.checked_mul(i128::from(denominator))?;
        ExactRational::new(numerator, denominator).map(Self::from_rational)
    }
}

/// The scalar state of an arc's signed included angle.
pub type ArcSweep = CurveParameter<AngleMeasurement, AngleMeasurement>;
/// The scalar state of a circle radius: a free exact value or a fixed measurement source.
pub type CircleRadius = CurveParameter<ResolvedLength, Measurement>;

impl CurveParameter<AngleMeasurement, AngleMeasurement> {
    /// Resolve the density-free angular source or free value for geometry evaluation.
    #[must_use]
    pub fn to_degrees_f64(&self) -> f64 {
        match &self.0 {
            CurveParameterState::Free(value) | CurveParameterState::Fixed(value) => {
                value.to_degrees_f64()
            }
        }
    }
}
