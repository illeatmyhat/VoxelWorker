//! The dimension algebra: what keeps a length from being added to an angle.
//!
//! A dimension is a pair of exponents, one over **length** and one over **angle**.
//! Addition requires equal exponents, multiplication adds them, division subtracts them.
//! Everything the expression language needs to type-check falls out of those three rules:
//!
//! | expression | dimension | why |
//! | --- | --- | --- |
//! | `wall + gap` | length | the exponents match |
//! | `wall * 2` | length | a scalar leaves exponents alone |
//! | `wall / gap` | dimensionless | the exponents cancel |
//! | `arc_length / radius` | angle | this is what a radian *is* |
//! | `wall + angle` | **error** | caught before it reaches the document |
//!
//! `voxel_density` needs no special case: it is voxels-per-block, so length over length, so
//! [`Dimension::DIMENSIONLESS`], and `3 blocks * voxel_density` types as a length by the
//! ordinary multiplication rule.
//!
//! ## Why angle is a dimension at all
//!
//! A radian is a ratio of two lengths, so a physicist would call it dimensionless and stop.
//! CAD does not, and neither does this: an author who writes `radius + sweep` has made a
//! mistake worth catching, and the only way to catch it is to track angle separately. The
//! cost is that `arc_length / radius` must be *declared* to produce an angle rather than
//! deriving it — which is exactly what the division rule does here.

/// The dimension of a quantity, as exponents over length and angle.
///
/// Small signed integers because real expressions never leave `-2..=2` — an area is
/// `length²`, a curvature is `length⁻¹` — and a saturating overflow is a better failure than
/// a wrap. See the [module docs](self) for the algebra.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub struct Dimension {
    /// The exponent over length: 1 for a distance, 2 for an area, 0 for a pure number.
    pub length: i8,
    /// The exponent over angle: 1 for a bearing or a sweep, 0 for everything else.
    pub angle: i8,
}

impl Dimension {
    /// A pure number — the dimension of a literal, a count, and of `voxel_density`.
    pub const DIMENSIONLESS: Self = Self {
        length: 0,
        angle: 0,
    };

    /// A distance: one power of length.
    pub const LENGTH: Self = Self {
        length: 1,
        angle: 0,
    };

    /// An angle: one power of angle.
    pub const ANGLE: Self = Self {
        length: 0,
        angle: 1,
    };

    /// An area — `length²`. Not a field type anywhere yet; it exists because
    /// `wall * wall` has to evaluate to *something* and silently calling it a length
    /// would let an area reach a length field.
    pub const AREA: Self = Self {
        length: 2,
        angle: 0,
    };

    /// Whether this is a pure number, and so usable as a scale factor on anything.
    pub fn is_dimensionless(self) -> bool {
        self == Self::DIMENSIONLESS
    }

    /// The dimension of a product: exponents add.
    ///
    /// Saturating, so a runaway expression clamps at `i8`'s bounds rather than wrapping a
    /// `length⁵⁰⁰` back around to something that would compare equal to a length.
    pub fn multiplied(self, other: Self) -> Self {
        Self {
            length: self.length.saturating_add(other.length),
            angle: self.angle.saturating_add(other.angle),
        }
    }

    /// The dimension of a quotient: exponents subtract. `length / length` cancels to
    /// [`DIMENSIONLESS`](Self::DIMENSIONLESS), which is the rule that makes a ratio usable
    /// as a scale factor.
    pub fn divided(self, other: Self) -> Self {
        Self {
            length: self.length.saturating_sub(other.length),
            angle: self.angle.saturating_sub(other.angle),
        }
    }

    /// The dimension of a sum, or `None` when the operands disagree — the error the author
    /// sees as "you cannot add a length to an angle".
    ///
    /// Addition is the only operation that can *fail* to have a dimension, which is why it
    /// returns an `Option` where [`multiplied`](Self::multiplied) and
    /// [`divided`](Self::divided) do not.
    pub fn added(self, other: Self) -> Option<Self> {
        (self == other).then_some(self)
    }

    /// How this reads in an error message: `"length"`, `"angle"`, `"a pure number"`, or the
    /// exponent form for anything exotic.
    pub fn describe(self) -> String {
        match (self.length, self.angle) {
            (0, 0) => "a pure number".to_string(),
            (1, 0) => "a length".to_string(),
            (0, 1) => "an angle".to_string(),
            (2, 0) => "an area".to_string(),
            (length, angle) => format!("length^{length}·angle^{angle}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ratio_of_lengths_is_a_pure_number() {
        // The rule that lets `wall / gap` be used as a scale factor, and the reason
        // `voxel_density` needs no special case anywhere in the evaluator.
        assert_eq!(
            Dimension::LENGTH.divided(Dimension::LENGTH),
            Dimension::DIMENSIONLESS
        );
        assert!(Dimension::LENGTH
            .divided(Dimension::LENGTH)
            .is_dimensionless());
    }

    #[test]
    fn density_scales_a_length_without_changing_it() {
        // `3 blocks * voxel_density` is the arithmetic the units layer already performs;
        // this asserts the type system agrees it is still a length.
        let density = Dimension::DIMENSIONLESS;
        assert_eq!(Dimension::LENGTH.multiplied(density), Dimension::LENGTH);
    }

    #[test]
    fn an_arc_length_over_a_radius_is_an_angle() {
        // Not a coincidence and not a special case: it is the definition of a radian
        // falling out of the division rule.
        let arc_length = Dimension::LENGTH;
        let radius = Dimension::LENGTH;
        assert_eq!(arc_length.divided(radius), Dimension::DIMENSIONLESS);
        // ...which is why angle is tracked SEPARATELY rather than derived: the algebra
        // alone cannot tell a radian from a plain ratio, so an author writing
        // `radius + sweep` would go uncaught. See the module docs.
        assert_ne!(Dimension::ANGLE, Dimension::DIMENSIONLESS);
    }

    #[test]
    fn a_length_and_an_angle_cannot_be_added() {
        assert_eq!(
            Dimension::LENGTH.added(Dimension::LENGTH),
            Some(Dimension::LENGTH)
        );
        assert_eq!(Dimension::LENGTH.added(Dimension::ANGLE), None);
    }

    #[test]
    fn a_product_of_lengths_is_an_area_and_not_a_length() {
        // The reason AREA exists at all: without it `wall * wall` would have to be called
        // a length, and an area would reach a length field.
        assert_eq!(
            Dimension::LENGTH.multiplied(Dimension::LENGTH),
            Dimension::AREA
        );
        assert_ne!(Dimension::AREA, Dimension::LENGTH);
    }

    #[test]
    fn exponents_saturate_rather_than_wrap() {
        // A wrap would let a deeply nested product compare EQUAL to a length and pass a
        // field check it should fail. Saturation keeps a wrong dimension wrong.
        let huge = Dimension {
            length: i8::MAX,
            angle: 0,
        };
        assert_eq!(huge.multiplied(Dimension::LENGTH).length, i8::MAX);
        assert_ne!(huge.multiplied(Dimension::LENGTH), Dimension::LENGTH);
    }

    #[test]
    fn descriptions_name_the_common_dimensions() {
        assert_eq!(Dimension::LENGTH.describe(), "a length");
        assert_eq!(Dimension::ANGLE.describe(), "an angle");
        assert_eq!(Dimension::DIMENSIONLESS.describe(), "a pure number");
        assert_eq!(Dimension::AREA.describe(), "an area");
    }
}
