//! [`Quantity`] — an exact value that carries its own [`Dimension`], and the arithmetic the
//! expression evaluator performs on it (ADR 0035 Decision 12).
//!
//! This is the **dynamically typed** half of the quantity model. The statically typed half
//! lives in [`units`](crate::units): a [`Measurement`] is always a
//! length and an [`AngleMeasurement`] is always an angle, so
//! a radius field simply cannot hold a sweep. That works everywhere the kind is known when
//! the code is written, which is everywhere except one place — an expression. `wall / gap`
//! has a dimension only once it has been evaluated, so the evaluator needs a value that
//! carries its dimension at runtime, and this is it.
//!
//! ## The canonical unit per dimension
//!
//! A quantity's `value` is exact and unit-canonical: **voxels** for each power of length,
//! **degrees** for each power of angle. So a length quantity of `3/2` is one and a half
//! voxels, and an angle quantity of `45` is forty-five degrees.
//!
//! Voxels, not blocks, because that is what the document stores and what a
//! [`Measurement`] evaluates to. It is also why converting a
//! measurement into a quantity takes a **density** — `3 blocks` is only a number of voxels
//! once `d` is known — exactly as
//! [`Measurement::to_voxels`](crate::units::Measurement::to_voxels) does.
//!
//! The value is a [`Rational`](crate::ExactRational) rather than an integer because
//! intermediate results are routinely fractional: `wall / 3` is a perfectly good half-way
//! step even where the final store must land on a whole voxel. The whole-voxel check belongs
//! at the door into a `Measurement`, not at every step of the arithmetic.

use crate::dimension::Dimension;
use crate::units::{AngleMeasurement, ExactRational, Measurement};

/// An exact value tagged with its dimension — the evaluator's working value.
///
/// See the [module docs](self) for why the value is a rational and what unit it is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Quantity {
    /// The exact value, in voxels per power of length and degrees per power of angle.
    pub value: ExactRational,
    /// What kind of thing the value is.
    pub dimension: Dimension,
}

/// Why an expression could not be evaluated. Every variant is an ordinary authoring
/// mistake the UI reports, never a bug.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantityError {
    /// A sum or difference whose operands are different kinds of thing — the error the whole
    /// dimension algebra exists to produce.
    MismatchedDimensions {
        /// The dimension of the left operand.
        left: Dimension,
        /// The dimension of the right operand.
        right: Dimension,
    },
    /// Division by an expression that evaluated to zero.
    DividedByZero,
    /// The exact result has no `i128` form — a rational chain long enough to overflow.
    /// Documented as reachable rather than dismissed: `Rational` uses `i128` limbs, not a
    /// bignum.
    Overflowed,
    /// The value is the right dimension but the wrong shape for its destination — a length
    /// that is not a whole number of voxels, or an angle finer than the exact store keeps.
    NotRepresentable {
        /// What the destination needed, for the message.
        needed: &'static str,
    },
}

impl core::fmt::Display for QuantityError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            QuantityError::MismatchedDimensions { left, right } => write!(
                formatter,
                "cannot combine {} with {}",
                left.describe(),
                right.describe()
            ),
            QuantityError::DividedByZero => write!(formatter, "division by zero"),
            QuantityError::Overflowed => {
                write!(formatter, "the exact value grew too large to represent")
            }
            QuantityError::NotRepresentable { needed } => {
                write!(formatter, "not representable: {needed}")
            }
        }
    }
}

impl std::error::Error for QuantityError {}

impl Quantity {
    /// A quantity from a raw exact value and a dimension.
    pub fn new(value: ExactRational, dimension: Dimension) -> Self {
        Self { value, dimension }
    }

    /// A pure number — what a bare literal in an expression evaluates to.
    pub fn dimensionless(value: ExactRational) -> Self {
        Self::new(value, Dimension::DIMENSIONLESS)
    }

    /// A length, in voxels.
    pub fn length_voxels(value: ExactRational) -> Self {
        Self::new(value, Dimension::LENGTH)
    }

    /// An angle, in degrees.
    pub fn angle_degrees(value: ExactRational) -> Self {
        Self::new(value, Dimension::ANGLE)
    }

    /// A length quantity from an authored [`Measurement`], at the document density.
    ///
    /// Takes `density` for the same reason
    /// [`Measurement::to_voxels`](crate::units::Measurement::to_voxels) does: a block term is
    /// only a count of voxels once `d` is known.
    pub fn from_measurement(measurement: Measurement, density: u32) -> Self {
        Self::length_voxels(measurement.to_voxels_exact(density))
    }

    /// An angle quantity from an authored [`AngleMeasurement`]. No density: an angle has none.
    pub fn from_angle(angle: AngleMeasurement) -> Self {
        Self::angle_degrees(angle.degrees())
    }

    /// `self + other`, or [`MismatchedDimensions`](QuantityError::MismatchedDimensions).
    pub fn plus(self, other: Self) -> Result<Self, QuantityError> {
        let dimension =
            self.dimension
                .added(other.dimension)
                .ok_or(QuantityError::MismatchedDimensions {
                    left: self.dimension,
                    right: other.dimension,
                })?;
        Ok(Self::new(self.value.plus(other.value), dimension))
    }

    /// `self - other`, or [`MismatchedDimensions`](QuantityError::MismatchedDimensions).
    pub fn minus(self, other: Self) -> Result<Self, QuantityError> {
        let dimension =
            self.dimension
                .added(other.dimension)
                .ok_or(QuantityError::MismatchedDimensions {
                    left: self.dimension,
                    right: other.dimension,
                })?;
        Ok(Self::new(self.value.minus(other.value), dimension))
    }

    /// `self * other`. Never fails on dimensions — exponents simply add.
    pub fn times(self, other: Self) -> Self {
        Self::new(
            self.value.times(other.value),
            self.dimension.multiplied(other.dimension),
        )
    }

    /// `self / other`. Fails only on a zero divisor; the dimension always exists, because
    /// exponents subtract.
    pub fn divided_by(self, other: Self) -> Result<Self, QuantityError> {
        let value = self
            .value
            .divided_by(other.value)
            .ok_or(QuantityError::DividedByZero)?;
        Ok(Self::new(value, self.dimension.divided(other.dimension)))
    }

    /// `-self`.
    pub fn negated(self) -> Result<Self, QuantityError> {
        let value = self.value.negated().ok_or(QuantityError::Overflowed)?;
        Ok(Self::new(value, self.dimension))
    }

    /// The whole voxel count this length lands on, or an error.
    ///
    /// **The door back into the statically typed world.** It enforces both halves at once:
    /// the dimension must be a length, and the value must be a whole number of voxels —
    /// nothing is finer than a voxel, which is the same rule
    /// [`units::parse`](crate::units::parse) applies to an authored voxel term.
    pub fn to_whole_voxels(self) -> Result<i64, QuantityError> {
        if self.dimension != Dimension::LENGTH {
            return Err(QuantityError::MismatchedDimensions {
                left: self.dimension,
                right: Dimension::LENGTH,
            });
        }
        let whole = self
            .value
            .to_integer()
            .ok_or(QuantityError::NotRepresentable {
                needed: "a whole number of voxels",
            })?;
        i64::try_from(whole).map_err(|_| QuantityError::Overflowed)
    }

    /// The [`AngleMeasurement`] this angle lands on, or an error. The angle door, mirroring
    /// [`to_whole_voxels`](Self::to_whole_voxels) — an angle has no whole-unit floor, so only
    /// the dimension is checked.
    pub fn to_angle(self) -> Result<AngleMeasurement, QuantityError> {
        if self.dimension != Dimension::ANGLE {
            return Err(QuantityError::MismatchedDimensions {
                left: self.dimension,
                right: Dimension::ANGLE,
            });
        }
        Ok(AngleMeasurement::new(self.value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn whole(value: i128) -> ExactRational {
        ExactRational::from_integer(value)
    }

    fn ratio(numerator: i128, denominator: i128) -> ExactRational {
        ExactRational::new(numerator, denominator).expect("non-zero denominator")
    }

    #[test]
    fn a_length_plus_an_angle_is_refused() {
        // The headline of the whole dimension layer: this is the mistake it exists to catch,
        // and it must fail before the value can reach a document field.
        let wall = Quantity::length_voxels(whole(32));
        let sweep = Quantity::angle_degrees(whole(90));
        assert_eq!(
            wall.plus(sweep),
            Err(QuantityError::MismatchedDimensions {
                left: Dimension::LENGTH,
                right: Dimension::ANGLE,
            })
        );
    }

    #[test]
    fn a_ratio_of_lengths_scales_a_length() {
        // `wall / gap` is a pure number, so multiplying a length by it leaves a length.
        // This is the chain `voxel_density` rides on.
        let wall = Quantity::length_voxels(whole(32));
        let gap = Quantity::length_voxels(whole(8));
        let ratio = wall.divided_by(gap).expect("non-zero divisor");
        assert_eq!(ratio.dimension, Dimension::DIMENSIONLESS);
        assert_eq!(ratio.value, whole(4));

        let scaled = Quantity::length_voxels(whole(5)).times(ratio);
        assert_eq!(scaled.dimension, Dimension::LENGTH);
        assert_eq!(scaled.value, whole(20));
    }

    #[test]
    fn arithmetic_is_exact_where_a_float_would_drift() {
        // The invariant the whole crate is built on: a third of a voxel, tripled, is exactly
        // one voxel. In f64 this is 0.9999999999999998.
        let third = Quantity::length_voxels(ratio(1, 3));
        let whole_again = third
            .plus(third)
            .and_then(|sum| sum.plus(third))
            .expect("same dimension");
        assert_eq!(whole_again.value, whole(1));
        assert_eq!(whole_again.to_whole_voxels(), Ok(1));
    }

    #[test]
    fn a_fractional_length_cannot_become_a_voxel_count() {
        // Sub-voxel input is rejected rather than rounded — the same policy the authored
        // voxel term has. The intermediate value was legal; only the door refuses it.
        let half = Quantity::length_voxels(ratio(1, 2));
        assert_eq!(
            half.to_whole_voxels(),
            Err(QuantityError::NotRepresentable {
                needed: "a whole number of voxels"
            })
        );
    }

    #[test]
    fn the_voxel_door_checks_the_dimension_not_just_the_value() {
        // A whole number that happens to be an angle must not slip into a length field.
        let ninety_degrees = Quantity::angle_degrees(whole(90));
        assert!(matches!(
            ninety_degrees.to_whole_voxels(),
            Err(QuantityError::MismatchedDimensions { .. })
        ));
    }

    #[test]
    fn dividing_by_zero_reports_rather_than_panics() {
        let wall = Quantity::length_voxels(whole(32));
        let nothing = Quantity::dimensionless(whole(0));
        assert_eq!(wall.divided_by(nothing), Err(QuantityError::DividedByZero));
    }

    #[test]
    fn an_angle_round_trips_through_its_door() {
        let sweep = Quantity::angle_degrees(ratio(45, 2));
        let measurement = sweep.to_angle().expect("an angle");
        assert_eq!(measurement.degrees(), ratio(45, 2));
        assert_eq!(Quantity::from_angle(measurement), sweep);
    }

    #[test]
    fn a_product_of_lengths_cannot_reach_a_length_field() {
        // `wall * wall` is an area. Without the AREA dimension it would have to be called a
        // length and would pass this door.
        let wall = Quantity::length_voxels(whole(4));
        let area = wall.times(wall);
        assert_eq!(area.dimension, Dimension::AREA);
        assert!(matches!(
            area.to_whole_voxels(),
            Err(QuantityError::MismatchedDimensions { .. })
        ));
    }
}
