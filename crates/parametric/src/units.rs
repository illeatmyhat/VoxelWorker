//! The units layer: the parametric blocks/voxels measurement core.
//!
//! Placement, sizes and radii are stored as **canonical voxels** at the
//! document's density `d` (`blocks · d = voxels`). A user-facing measurement is
//! a *unit expression* parsed onto that canonical store and formatted back. The
//! two grid-native units are **blocks** and **voxels**.
//!
//! A [`Measurement`] RETAINS its authored expression (parametric): it is a sum of
//! a BLOCK term (an exact rational — blocks may be integer, decimal, fraction or
//! the VS sixteenths idiom) plus a VOXEL term (an integer count of voxels). The
//! canonical voxel count is the DERIVED value, recomputed at a supplied density
//! `d` via [`Measurement::to_voxels`]. Because `d` is supplied at eval time, the
//! SAME measurement re-evaluates at a new `d` — the lossless refine for
//! integer-ratio re-targets (`"3.5 blocks"` → 56 at d16, 112 at d32).
//!
//! Parser policy is STRICT: measurements evaluate
//! as EXACT RATIONALS (no floats), fractions/decimals are allowed on block-terms
//! only, a voxel-term must be an integer, and a block-fraction that does not land
//! on a whole voxel at the current `d` is rejected with the nearest representable
//! values reported — never silently rounded.
//!
//! This module is pure logic: it has no UI wiring and depends on nothing else in
//! the crate. It feeds `NodeTransform::from_measurements` and sketch profile points
//! (`SketchPoint.offset_measurements`), both re-evaluated on `SetDensity`.
//!
//! [`AngleMeasurement`] is the family's second kind: an authored angle in exact
//! degrees, density-free, consumed by the sketch arc bulge.

use crate::EvaluationContext;
use std::fmt;

/// The units layer's exact rational — substrate's [`substrate::interval::Rational`], the
/// sign-normalized, gcd-reduced `i128` ratio.
///
/// The units layer must not touch `f64`: parsing `"3.5"` as `7/2` and never an `f64`
/// is what makes `"3.5 blocks"` land on exactly 56 voxels at d16 with no float drift.
/// The rational arithmetic itself (reduction, floor/ceil, the Euclidean gcd) is a pure
/// CS primitive and lives in substrate; the domain keeps the name `ExactRational` at
/// this seam because it is the public measurement vocabulary used across the scene and
/// intent layers. See `docs/architecture/01-document.md` (the units/measurement core).
pub use substrate::interval::{Rational as ExactRational, RationalFromF64Error};

/// A parametric blocks + voxels measurement.
///
/// This is the STORED, authored expression — `block_term · d + voxel_term`
/// voxels at a density `d` supplied at eval time. It is serde-serializable
/// because it is persisted alongside the document: the placement Intents carry the
/// expression, not just the derived voxel count, so replay and undo preserve
/// authored intent.
///
/// The block term is an exact rational so `"3.5 blocks"`, `"8/16 blocks"` and
/// `"3 8/16 blocks"` are all retained losslessly; the voxel term is a plain
/// integer because nothing is finer than a voxel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct Measurement {
    /// Block count as an exact rational (numerator, denominator). Serialized as
    /// the reduced pair so a persisted document is float-free end to end.
    block_term_numerator: i128,
    block_term_denominator: i128,
    /// Whole voxels added on top of the block term.
    voxel_term: i64,
}

impl<'de> serde::Deserialize<'de> for Measurement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Stored {
            block_term_numerator: i128,
            block_term_denominator: i128,
            voxel_term: i64,
        }

        let stored = Stored::deserialize(deserializer)?;
        let block_term =
            ExactRational::new(stored.block_term_numerator, stored.block_term_denominator)
                .ok_or_else(|| {
                    serde::de::Error::custom("measurement has an invalid block denominator")
                })?;
        Ok(Self::new(block_term, stored.voxel_term))
    }
}

/// Convert an exact value to the storage type without wrapping at either boundary.
fn saturating_i64(value: i128) -> i64 {
    match i64::try_from(value) {
        Ok(value) => value,
        Err(_) if value.is_negative() => i64::MIN,
        Err(_) => i64::MAX,
    }
}

impl Default for Measurement {
    /// The zero measurement: a `0/1` block term and `0` voxels (NOT the derived
    /// `i128::default()` denominator of 0, which would be an invalid rational).
    /// This is what an `[Measurement; 3]` field defaults to (e.g. a fresh
    /// identity transform), and it evaluates to 0 voxels at any density.
    fn default() -> Self {
        Self::from_voxels(0)
    }
}

impl Measurement {
    /// Build a measurement from an exact block rational and a whole voxel count.
    #[must_use]
    pub const fn new(block_term: ExactRational, voxel_term: i64) -> Self {
        Self {
            block_term_numerator: block_term.numerator(),
            block_term_denominator: block_term.denominator(),
            voxel_term,
        }
    }

    /// The block term as an exact rational.
    #[allow(clippy::expect_used)]
    ///
    /// # Panics
    ///
    /// This cannot panic for a value made by [`Measurement::new`], which stores the
    /// denominator from an already-valid [`ExactRational`].
    pub fn block_term(self) -> ExactRational {
        // The stored pair came from a reduced `ExactRational`, so reconstruction
        // is exact; `expect` cannot fire (the denominator is never zero).
        ExactRational::new(self.block_term_numerator, self.block_term_denominator)
            .expect("stored block-term denominator is non-zero")
    }

    /// The whole-voxel term.
    #[must_use]
    pub const fn voxel_term(self) -> i64 {
        self.voxel_term
    }

    /// A pure-voxel measurement equal to `voxels` (zero block term).
    ///
    /// The synthesis path for documents/offsets that only have a canonical voxel
    /// count and no authored expression to retain (e.g. an OLD scene loaded
    /// without an `offset_measurements` field, or a placement produced by a drag
    /// gizmo): the retained measurement is just the voxel count, which
    /// re-evaluates back to exactly `voxels` at any density (the block term is 0,
    /// so density does not scale it).
    #[must_use]
    pub const fn from_voxels(voxels: i64) -> Self {
        Self::new(ExactRational::from_integer(0), voxels)
    }

    /// Evaluate to an exact voxel count at the given density `d`.
    ///
    /// `voxels = block_term · d + voxel_term`. The block contribution MUST land
    /// on a whole voxel: if `block_term · d` is not an integer (e.g. `"3.5
    /// blocks"` at `d = 15` = 52.5), this returns
    /// [`MeasurementError::BlockTermNotWholeVoxels`] reporting the nearest
    /// representable floor/ceil voxel counts — it never silently rounds.
    ///
    /// # Errors
    ///
    /// Returns [`MeasurementError::ZeroDensity`] for a zero density, or
    /// [`MeasurementError::BlockTermNotWholeVoxels`] when the block term does not
    /// evaluate to an integer number of voxels.
    pub fn to_voxels(self, density: u32) -> Result<i64, MeasurementError> {
        if density == 0 {
            return Err(MeasurementError::ZeroDensity);
        }
        let block_voxels = self
            .block_term()
            .times(ExactRational::from_integer(i128::from(density)));
        let Some(whole_block_voxels) = block_voxels.to_integer() else {
            // Report the nearest representable voxel counts for the WHOLE
            // measurement (block contribution rounded each way, plus the
            // exact voxel term) so the caller can show "did you mean 52 or
            // 53 voxels?".
            let floor_voxels = block_voxels
                .floor()
                .saturating_add(i128::from(self.voxel_term));
            let ceil_voxels = block_voxels
                .ceil()
                .saturating_add(i128::from(self.voxel_term));
            return Err(MeasurementError::BlockTermNotWholeVoxels {
                density,
                nearest_floor_voxels: saturating_i64(floor_voxels),
                nearest_ceil_voxels: saturating_i64(ceil_voxels),
            });
        };
        Ok(saturating_i64(whole_block_voxels).saturating_add(self.voxel_term))
    }

    /// Evaluate to an EXACT voxel value at density `d`, without the whole-voxel policy.
    ///
    /// `block_term · d + voxel_term` as a rational, so `"3.5 blocks"` at `d = 15` is exactly
    /// `105/2` rather than an error. This is the door the **expression evaluator** takes
    /// ([`Quantity::from_measurement`](crate::quantity::Quantity::from_measurement)):
    /// intermediate results are routinely fractional, and refusing them mid-expression would
    /// reject `wall / 3 * 3`.
    ///
    /// [`to_voxels`](Self::to_voxels) remains the door into *storage*, where the whole-voxel
    /// rule applies and the nearest representable values are reported. Nothing is finer than
    /// a voxel in the document; plenty is finer than a voxel on the way there.
    ///
    /// Total: a density of 0 yields just the voxel term, which is what `block_term · 0`
    /// literally is.
    pub fn to_voxels_exact(self, density: u32) -> ExactRational {
        self.block_term()
            .times(ExactRational::from_integer(i128::from(density)))
            .plus(ExactRational::from_integer(i128::from(self.voxel_term)))
    }

    /// Resolve this authored source for one document evaluation.
    ///
    /// The context carries the document-owned density; the measurement stores no resolved cache,
    /// so a fixed curve parameter cannot accidentally grow a second density authority.
    pub fn to_voxel_rational(self, context: EvaluationContext) -> ExactRational {
        self.to_voxels_exact(context.voxels_per_block().get())
    }
}

/// A parametric ANGLE measurement in degrees.
///
/// The authored-quantity family's second kind, realized as its own type rather than a
/// runtime tag on [`Measurement`]: an angle and a length share retention semantics (the
/// stored expression is the truth, exact rationals, float-free persistence) but none of the
/// arithmetic — an angle has no block term, no density, and no voxel evaluation, so a
/// shared representation would force every length call-site through a kind check it can
/// never fail. The degree value is a reduced exact rational, mirroring the block term.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct AngleMeasurement {
    /// Degrees as an exact rational (numerator, denominator), serialized reduced.
    degrees_numerator: i128,
    degrees_denominator: i128,
}

impl<'de> serde::Deserialize<'de> for AngleMeasurement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Stored {
            degrees_numerator: i128,
            degrees_denominator: i128,
        }

        let stored = Stored::deserialize(deserializer)?;
        let degrees = ExactRational::new(stored.degrees_numerator, stored.degrees_denominator)
            .ok_or_else(|| serde::de::Error::custom("angle has an invalid degree denominator"))?;
        Ok(Self::new(degrees))
    }
}

impl Default for AngleMeasurement {
    /// The zero angle, as a valid `0/1` rational (not the all-zero derive, whose
    /// denominator would be an invalid rational).
    fn default() -> Self {
        Self::from_degrees(0)
    }
}

impl AngleMeasurement {
    /// Build an angle from an exact degree rational.
    #[must_use]
    pub const fn new(degrees: ExactRational) -> Self {
        Self {
            degrees_numerator: degrees.numerator(),
            degrees_denominator: degrees.denominator(),
        }
    }

    /// A whole-degree angle.
    #[must_use]
    pub fn from_degrees(degrees: i64) -> Self {
        Self::new(ExactRational::from_integer(i128::from(degrees)))
    }

    /// Store a solved continuous degree value as its exact IEEE-754 ratio.
    ///
    /// A solved value is already a binary float; preserving that exact value avoids inventing an
    /// arc-second grid the solver never used. The bounded exact-rational store still rejects a
    /// finite value whose numerator or denominator has no `i128` representation.
    ///
    /// # Errors
    ///
    /// Propagates [`RationalFromF64Error::NonFinite`] and
    /// [`RationalFromF64Error::OutOfRange`] from the exact-rational conversion.
    pub fn try_from_degrees_f64(degrees: f64) -> Result<Self, RationalFromF64Error> {
        ExactRational::try_from_f64_exact(degrees).map(Self::new)
    }

    /// The exact degree value.
    #[allow(clippy::expect_used)]
    ///
    /// # Panics
    ///
    /// This cannot panic for a value made by [`AngleMeasurement::new`], which stores the
    /// denominator from an already-valid [`ExactRational`].
    pub fn degrees(self) -> ExactRational {
        ExactRational::new(self.degrees_numerator, self.degrees_denominator)
            .expect("stored degree denominator is non-zero")
    }

    /// The degree value evaluated to `f64` — the tessellation/display evaluation, the
    /// analog of [`Measurement::to_voxels`]. Angles carry no density, so unlike a length
    /// the evaluation cannot fail; the float is derived, never stored.
    #[must_use]
    #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
    pub fn to_degrees_f64(self) -> f64 {
        self.degrees_numerator as f64 / self.degrees_denominator as f64
    }
}

/// A unit a [`Measurement`] / voxel count can be FORMATTED into — the display side of
/// the units layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayUnit {
    /// Decimal blocks: `"3.5 blocks"`.
    DecimalBlocks,
    /// Whole blocks plus the remainder voxels: `"3 blocks 8 voxels"`.
    BlocksAndVoxels,
    /// Whole blocks plus a `remainder/density` block-fraction: `"3 8/16 blocks"`
    /// (denominator = density — the VS "sixteenths" idiom generalised to any
    /// `d`).
    BlockFraction,
    /// A raw voxel count: `"56 voxels"`.
    Voxels,
}

/// Format a canonical voxel count back into a unit string at the given density.
///
/// The inverse of [`parse`] + [`Measurement::to_voxels`] for the round-trip
/// display path. `density` of 0 is treated as 1 (degenerate, but never panics);
/// callers always pass the document's real `d`.
#[must_use]
pub fn format(voxels: i64, density: u32, style: DisplayUnit) -> String {
    let density = i64::from(density.max(1));
    match style {
        DisplayUnit::Voxels => format!("{voxels} {}", pluralise(voxels, "voxel")),
        DisplayUnit::BlocksAndVoxels => {
            let whole_blocks = voxels.div_euclid(density);
            let remainder_voxels = voxels.rem_euclid(density);
            format!(
                "{whole_blocks} {} {remainder_voxels} {}",
                pluralise(whole_blocks, "block"),
                pluralise(remainder_voxels, "voxel"),
            )
        }
        DisplayUnit::BlockFraction => {
            let whole_blocks = voxels.div_euclid(density);
            let remainder_voxels = voxels.rem_euclid(density);
            if remainder_voxels == 0 {
                format!("{whole_blocks} {}", pluralise(whole_blocks, "block"))
            } else {
                // Sixteenths idiom: keep the denominator AT the density (VS shows
                // `8/16`, not the reduced `1/2`), so the display reads as
                // "8 of 16 sub-positions".
                format!(
                    "{whole_blocks} {remainder_voxels}/{density} {}",
                    pluralise(whole_blocks.max(1), "block"),
                )
            }
        }
        DisplayUnit::DecimalBlocks => {
            // Exact-rational blocks: voxels / density, reduced, rendered as a
            // terminating decimal when the reduced denominator is 2/5-smooth, else
            // fall back to whole blocks + voxels so we never emit a rounded float.
            let Some(blocks) = ExactRational::new(i128::from(voxels), i128::from(density)) else {
                return format!("{voxels} {}", pluralise(voxels, "voxel"));
            };
            blocks.to_terminating_decimal().map_or_else(
                || {
                    // Non-terminating in base 10 (e.g. 1/3 of a block): present the
                    // honest mixed form rather than a truncated decimal.
                    let whole_blocks = voxels.div_euclid(density);
                    let remainder_voxels = voxels.rem_euclid(density);
                    format!(
                        "{whole_blocks} {} {remainder_voxels} {}",
                        pluralise(whole_blocks, "block"),
                        pluralise(remainder_voxels, "voxel"),
                    )
                },
                |text| format!("{text} {}", pluralise_rational(blocks, "block")),
            )
        }
    }
}

/// `"block"`/`"blocks"` etc. — singular only for an exact `1`.
fn pluralise(count: i64, singular: &str) -> String {
    if count == 1 {
        singular.to_string()
    } else {
        format!("{singular}s")
    }
}

/// Plural agreement for a rational block count (only `1/1` is singular).
fn pluralise_rational(blocks: ExactRational, singular: &str) -> String {
    if blocks == ExactRational::from_integer(1) {
        singular.to_string()
    } else {
        format!("{singular}s")
    }
}

/// An error from [`parse`] — descriptive variants for each malformed-input class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeasurementParseError {
    /// The input was empty or only whitespace.
    Empty,
    /// A number was found without a following unit (e.g. `"3"` alone, or
    /// `"3.5"`).
    MissingUnit { number_text: String },
    /// A unit word was found without a preceding number (e.g. `"blocks"`).
    MissingNumber { unit_text: String },
    /// An unrecognized unit word (not blocks/voxels and not a number).
    UnknownUnit { unit_text: String },
    /// A token could not be parsed as any known number form.
    InvalidNumber { number_text: String },
    /// A voxel term carried a fraction or decimal — sub-voxel input is rejected
    /// (nothing is finer than a voxel; nudge to a block-fraction or a denser
    /// document).
    SubVoxel { number_text: String },
    /// The same unit appeared more than once (e.g. `"3 blocks 2 blocks"`).
    DuplicateUnit { unit_text: String },
    /// A fraction with a zero denominator (e.g. `"8/0 blocks"`).
    ZeroDenominator { number_text: String },
    /// A unit word from the OTHER dimension: a degree closing a length term, or a block closing
    /// an angle.
    ///
    /// Named rather than folded into [`UnknownUnit`](Self::UnknownUnit), because the word IS a
    /// unit and is spelled correctly — it just does not measure the thing being read. "unknown
    /// unit `deg`" would send the author hunting for a typo that is not there.
    WrongDimension {
        /// The unit word as written.
        unit_text: String,
        /// What the grammar that refused it was reading: "length" or "angle".
        reading: &'static str,
    },
}

impl fmt::Display for MeasurementParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(formatter, "empty measurement"),
            Self::MissingUnit { number_text } => write!(
                formatter,
                "number `{number_text}` has no unit (expected blocks or voxels)"
            ),
            Self::MissingNumber { unit_text } => {
                write!(formatter, "unit `{unit_text}` has no preceding number")
            }
            Self::UnknownUnit { unit_text } => {
                write!(formatter, "unknown unit `{unit_text}` (expected blocks or voxels)")
            }
            Self::InvalidNumber { number_text } => {
                write!(formatter, "`{number_text}` is not a valid number")
            }
            Self::SubVoxel { number_text } => write!(
                formatter,
                "`{number_text}` voxels is sub-voxel; voxels must be whole (use a block-fraction or a denser document)"
            ),
            Self::DuplicateUnit { unit_text } => {
                write!(formatter, "unit `{unit_text}` appears more than once")
            }
            Self::ZeroDenominator { number_text } => {
                write!(formatter, "`{number_text}` has a zero denominator")
            }
            Self::WrongDimension { unit_text, reading } => {
                write!(formatter, "`{unit_text}` is not a unit of {reading}")
            }
        }
    }
}

impl std::error::Error for MeasurementParseError {}

/// An error from [`Measurement::to_voxels`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeasurementError {
    /// The block term does not land on a whole voxel at this density. Carries the
    /// nearest representable voxel counts (floor and ceil of the FULL
    /// measurement) so the UI can offer them instead of silently rounding.
    BlockTermNotWholeVoxels {
        density: u32,
        nearest_floor_voxels: i64,
        nearest_ceil_voxels: i64,
    },
    /// A density of zero was supplied (no game uses `d = 0`).
    ZeroDensity,
}

impl fmt::Display for MeasurementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BlockTermNotWholeVoxels {
                density,
                nearest_floor_voxels,
                nearest_ceil_voxels,
            } => write!(
                formatter,
                "block term does not land on a whole voxel at density {density}; nearest are {nearest_floor_voxels} or {nearest_ceil_voxels} voxels"
            ),
            Self::ZeroDensity => write!(formatter, "density must be at least 1"),
        }
    }
}

impl std::error::Error for MeasurementError {}

/// Which unit a token names.
///
/// **The whole unit VOCABULARY is this one table.** Degrees sit in it beside blocks and voxels
/// even though no length is ever measured in them, because the alternative is a second table
/// somewhere else and then `45deg` has two implementations that agree until one of them moves.
/// What a given grammar ACCEPTS is a separate question, asked by [`UnitKind::dimension`] — see
/// the note on [`UnitDimension`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnitKind {
    Blocks,
    Voxels,
    Degrees,
}

/// What a unit measures.
///
/// **The vocabulary is shared and the GRAMMARS are not.** One lexer knows every unit word, so the
/// compact spelling splits the same way whatever follows it; then the length grammar refuses a
/// degree BY NAME and the angle grammar refuses a block BY NAME. That is what makes `3 blocks +
/// 45 deg` say "`deg` is not a unit of length" instead of "unknown unit `deg`", which would send
/// the author hunting for a typo in a word they spelled correctly.
///
/// Which dimension a FIELD accepts is neither grammar's question — it is the binding's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnitDimension {
    /// Blocks and voxels: a [`Measurement`].
    Length,
    /// Degrees: an [`AngleMeasurement`].
    Angle,
}

impl UnitDimension {
    /// The word this dimension goes by in a complaint, as it reads after "is not a unit of".
    const fn name(self) -> &'static str {
        match self {
            Self::Length => "length",
            Self::Angle => "angle",
        }
    }
}

impl UnitKind {
    /// What this unit measures.
    const fn dimension(self) -> UnitDimension {
        match self {
            Self::Blocks | Self::Voxels => UnitDimension::Length,
            Self::Degrees => UnitDimension::Angle,
        }
    }
}

/// Classify a unit word (case-insensitive). `None` for anything that is not a
/// recognized unit. Accepts the long, short and single-letter spellings.
///
/// Degrees have no single-letter spelling: `d` next to `b` and `v` reads as another grid unit,
/// and the author who means degrees can write three letters.
fn classify_unit(word: &str) -> Option<UnitKind> {
    match word.to_ascii_lowercase().as_str() {
        "blocks" | "block" | "b" => Some(UnitKind::Blocks),
        "voxels" | "voxel" | "v" => Some(UnitKind::Voxels),
        "degrees" | "degree" | "deg" | "\u{b0}" => Some(UnitKind::Degrees),
        _ => None,
    }
}

/// What dimension the unit word `word` measures, if it is one at all.
///
/// The expression grammar asks this to decide which literal reader a munched group belongs to.
pub(crate) fn unit_dimension(word: &str) -> Option<UnitDimension> {
    classify_unit(word).map(UnitKind::dimension)
}

/// Parse a units expression into a [`Measurement`] (STRICT).
///
/// Grammar: a sum of terms. A BLOCK term is a block-number + a block unit; a
/// VOXEL term is an integer + a voxel unit. Block-number forms: integer (`"3"`),
/// decimal (`"3.5"`), fraction (`"8/16"`) and mixed integer+fraction (`"3 8/16"`,
/// the VS sixteenths idiom = `3 + 8/16`). Units are case-insensitive
/// `blocks`/`block`/`b` and `voxels`/`voxel`/`v`. Examples that parse:
/// `"3 blocks 8 voxels"`, `"3b 8v"`, `"3.5 blocks"`, `"8/16 blocks"`,
/// `"56 voxels"`, `"3 8/16 blocks"`.
///
/// Tokenisation splits on whitespace AND on the unit letters glued to a number
/// (`"3b"` → `3`, `b`), so the spaced and compact spellings parse identically.
/// Each term must end in a unit; each unit may appear once.
///
/// # Errors
///
/// Returns a [`MeasurementParseError`] describing the first malformed term.
pub fn parse(input: &str) -> Result<Measurement, MeasurementParseError> {
    let tokens = tokenise(input);
    if tokens.is_empty() {
        return Err(MeasurementParseError::Empty);
    }
    measurement_from_tokens(&tokens)
}

/// The exact value of one number token, for a grammar that wants a bare count rather than a
/// measurement term.
///
/// Same number forms as a measurement's — integer, decimal, fraction — read by the same code, so
/// `3.5` means seven halves whether it is a scale factor or half a block.
///
/// # Errors
///
/// Returns the literal grammar's own complaint for a malformed number.
pub(crate) fn rational_from_number_token(
    text: &str,
) -> Result<ExactRational, MeasurementParseError> {
    parse_number(text)?
        .as_ref()
        .map(NumberLiteral::to_rational)
        .ok_or_else(|| MeasurementParseError::InvalidNumber {
            number_text: text.to_owned(),
        })
}

/// Whether a word names one of the grid-native units.
///
/// The expression grammar asks this to find where a measurement literal ENDS: a number followed
/// by a unit word is part of the literal, a number followed by anything else is a bare count.
/// It asks here rather than keeping a list of its own, which is the whole point of
/// [`Token`] being shared.
pub(crate) fn is_unit_word(word: &str) -> bool {
    classify_unit(word).is_some()
}

/// The voxel term a run of accumulated numbers adds up to.
///
/// SUB-VOXEL REJECTION lives here: a voxel is the grid's atom, so every number closed by `voxels`
/// must be a whole one. `3.5 voxels` is not a rounding question, it is a quantity the grid cannot
/// hold, and it is refused by name.
fn whole_voxels_of(numbers: &[NumberLiteral]) -> Result<i64, MeasurementParseError> {
    let mut term: i64 = 0;
    for number in numbers {
        let Some(whole) = number.to_rational().to_integer() else {
            return Err(MeasurementParseError::SubVoxel {
                number_text: number.source_text.clone(),
            });
        };
        let Ok(whole) = i64::try_from(whole) else {
            return Err(MeasurementParseError::InvalidNumber {
                number_text: number.source_text.clone(),
            });
        };
        let Some(sum) = term.checked_add(whole) else {
            return Err(MeasurementParseError::InvalidNumber {
                number_text: number.source_text.clone(),
            });
        };
        term = sum;
    }
    Ok(term)
}

/// The one-literal grammar, over a token slice a caller has already carved out.
///
/// [`parse`] hands it a whole input; [`crate::expression::parse`] hands it the greedy munch it
/// identified as one measurement operand. Both then get the sixteenths idiom, the duplicate-unit
/// rule and the sub-voxel rejection from the same place — which is the only way those three stay
/// one answer rather than two that agree until they do not.
pub(crate) fn measurement_from_tokens(
    tokens: &[Token],
) -> Result<Measurement, MeasurementParseError> {
    let mut block_total = ExactRational::from_integer(0);
    let mut voxel_total: i64 = 0;
    let mut seen_blocks = false;
    let mut seen_voxels = false;

    // Number parts accumulate until a unit closes the term. The VS sixteenths
    // idiom `"3 8/16"` is two number tokens (an integer then a fraction) before
    // the unit, so we collect a small buffer and sum it when the unit arrives.
    let mut pending_numbers: Vec<NumberLiteral> = Vec::new();
    let mut pending_text = String::new();

    for token in tokens.iter().cloned() {
        // The ONE-LITERAL grammar over the shared token stream: numbers accumulate, a unit word
        // closes a term, and anything an expression would need — an operator, a bracket — is not
        // part of a single measurement and says so. `expression::parse` is the grammar that reads
        // those; this one deliberately does not grow to meet it.
        let token = match token {
            Token::Word(word) => word,
            Token::Number(text) => {
                match parse_number(&text)? {
                    Some(number) => {
                        if !pending_text.is_empty() {
                            pending_text.push(' ');
                        }
                        pending_text.push_str(&text);
                        pending_numbers.push(number);
                    }
                    None => return Err(MeasurementParseError::InvalidNumber { number_text: text }),
                }
                continue;
            }
            Token::Operator(sign) => {
                return Err(MeasurementParseError::InvalidNumber {
                    number_text: sign.to_string(),
                })
            }
            Token::OpenParen => {
                return Err(MeasurementParseError::InvalidNumber {
                    number_text: "(".to_owned(),
                })
            }
            Token::CloseParen => {
                return Err(MeasurementParseError::InvalidNumber {
                    number_text: ")".to_owned(),
                })
            }
            Token::Unexpected(text) => {
                return Err(MeasurementParseError::InvalidNumber { number_text: text })
            }
        };
        match classify_unit(&token) {
            Some(unit) => {
                if pending_numbers.is_empty() {
                    return Err(MeasurementParseError::MissingNumber { unit_text: token });
                }
                match unit {
                    UnitKind::Blocks => {
                        if seen_blocks {
                            return Err(MeasurementParseError::DuplicateUnit { unit_text: token });
                        }
                        seen_blocks = true;
                        let mut term = ExactRational::from_integer(0);
                        for number in &pending_numbers {
                            term = term.plus(number.to_rational());
                        }
                        block_total = block_total.plus(term);
                    }
                    // The refusal the vocabulary/grammar split exists for. Degrees are in the
                    // one unit table so the lexer splits `45deg` the way it splits `3b`, and the
                    // LENGTH grammar turns them away by name right here.
                    UnitKind::Degrees => {
                        return Err(MeasurementParseError::WrongDimension {
                            unit_text: token,
                            reading: UnitDimension::Length.name(),
                        })
                    }
                    UnitKind::Voxels => {
                        if seen_voxels {
                            return Err(MeasurementParseError::DuplicateUnit { unit_text: token });
                        }
                        seen_voxels = true;
                        let term = whole_voxels_of(&pending_numbers)?;
                        let Some(sum) = voxel_total.checked_add(term) else {
                            return Err(MeasurementParseError::InvalidNumber {
                                number_text: pending_text.clone(),
                            });
                        };
                        voxel_total = sum;
                    }
                }
                pending_numbers.clear();
                pending_text.clear();
            }
            // A word the unit table does not know. It could be a parameter name, but a single
            // measurement literal has no table to look it up in.
            None => return Err(MeasurementParseError::UnknownUnit { unit_text: token }),
        }
    }

    // A trailing number with no closing unit is incomplete (`"3"`, `"3.5"`).
    if !pending_numbers.is_empty() {
        return Err(MeasurementParseError::MissingUnit {
            number_text: pending_text,
        });
    }

    Ok(Measurement::new(block_total, voxel_total))
}

/// The angle-literal grammar, over a token slice a caller has already carved out.
///
/// The length grammar's sibling, and deliberately not a branch inside it. It reads
/// `number+ degree_word` — a run of numbers closed by a degree, so the mixed-fraction idiom
/// carries over (`22 1/2 deg` is 45/2 degrees, exactly) — and refuses a block or a voxel BY NAME,
/// which is the same refusal the length grammar makes in the other direction.
///
/// **An angle keeps no split, and needs none.** A length is retained as blocks-and-voxels because
/// that split re-targets when the density changes; degrees have no density and no second unit, so
/// the exact rational IS the retention.
pub(crate) fn angle_from_tokens(
    tokens: &[Token],
) -> Result<AngleMeasurement, MeasurementParseError> {
    let mut total = ExactRational::from_integer(0);
    let mut closed = false;
    let mut pending_numbers: Vec<NumberLiteral> = Vec::new();
    let mut pending_text = String::new();

    for token in tokens.iter().cloned() {
        let word = match token {
            Token::Word(word) => word,
            Token::Number(text) => {
                let Some(number) = parse_number(&text)? else {
                    return Err(MeasurementParseError::InvalidNumber { number_text: text });
                };
                if !pending_text.is_empty() {
                    pending_text.push(' ');
                }
                pending_text.push_str(&text);
                pending_numbers.push(number);
                continue;
            }
            other => {
                return Err(MeasurementParseError::InvalidNumber {
                    number_text: describe_token(&other),
                })
            }
        };
        match classify_unit(&word) {
            Some(UnitKind::Degrees) => {
                if pending_numbers.is_empty() {
                    return Err(MeasurementParseError::MissingNumber { unit_text: word });
                }
                if closed {
                    return Err(MeasurementParseError::DuplicateUnit { unit_text: word });
                }
                closed = true;
                for number in &pending_numbers {
                    total = total.plus(number.to_rational());
                }
                pending_numbers.clear();
                pending_text.clear();
            }
            Some(UnitKind::Blocks | UnitKind::Voxels) => {
                return Err(MeasurementParseError::WrongDimension {
                    unit_text: word,
                    reading: UnitDimension::Angle.name(),
                })
            }
            None => return Err(MeasurementParseError::UnknownUnit { unit_text: word }),
        }
    }

    if !pending_numbers.is_empty() {
        return Err(MeasurementParseError::MissingUnit {
            number_text: pending_text,
        });
    }
    Ok(AngleMeasurement::new(total))
}

/// A non-number, non-word token as it reads back in a literal grammar's complaint.
fn describe_token(token: &Token) -> String {
    match token {
        Token::Number(text) | Token::Word(text) | Token::Unexpected(text) => text.clone(),
        Token::Operator(sign) => sign.to_string(),
        Token::OpenParen => "(".to_owned(),
        Token::CloseParen => ")".to_owned(),
    }
}

/// One parsed number literal plus its original text (kept for error messages).
#[derive(Debug, Clone)]
struct NumberLiteral {
    value: ExactRational,
    source_text: String,
}

impl NumberLiteral {
    const fn to_rational(&self) -> ExactRational {
        self.value
    }
}

/// One lexical token of the authored-quantity language.
///
/// **The TOKEN layer is shared; the GRAMMAR over it is not.** [`parse`] reads a single
/// measurement literal from this stream and [`crate::expression::parse`] reads a whole
/// expression from the same one. Two tokenizers, each with its own idea of what a unit word is,
/// is the bug class where both sides have to move together and one of them does not — the day
/// someone adds `mm` to one list is the day the other stops agreeing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Token {
    /// A number-shaped run: digits, dots, the closed-up `a/b` fraction idiom, and a leading `-`
    /// folded in where the minus stood in prefix position.
    Number(String),
    /// An alphabetic word — a unit name to one grammar, a parameter name to the other. The
    /// lexer does not know which, and that is the point of the split.
    Word(String),
    /// `+`, `-`, `*` or `/` standing as an operator.
    Operator(char),
    /// `(`
    OpenParen,
    /// `)`
    CloseParen,
    /// A character neither grammar can read. Carried rather than dropped so the error can name
    /// it — a silently skipped character makes `3 blocks @ 4` parse as something.
    Unexpected(String),
}

/// Break the input into tokens.
///
/// Whitespace separates but is not required: `"3b"` lexes exactly as `"3 b"`, because the number
/// scan stops at the first letter and the word scan takes over. Two rules are worth stating
/// because both were decided rather than fallen into:
///
/// **A minus in PREFIX position belongs to the number after it.** `-3b` is a negative offset and
/// has always tokenised as one signed number; `a - 3` is a subtraction. The discriminator is what
/// precedes the minus, which is the ordinary unary-versus-binary rule, settled here so neither
/// grammar has to.
///
/// **A closed-up `a/b` is a fraction; a spaced `a / b` is a division.** Whitespace is the whole
/// discriminator, with no exception, because the sixteenths idiom `8/16 blocks` has to keep
/// meaning half a block and the only other way to know that is to look ahead for a unit — which
/// would make the lexer read the grammar it is supposed to be beneath. The cost is that a
/// closed-up fraction after a division sign binds as one operand: `24v / 2/3` is thirty-six
/// voxels, `24v / 2 / 3` is four. An earlier draft carved out an exception for exactly that case
/// and it went in untested; when a falsification pass turned the exception off, nothing reddened,
/// which is the whole argument against keeping it.
pub(crate) fn tokenise(input: &str) -> Vec<Token> {
    let characters: Vec<char> = input.chars().collect();
    let mut tokens: Vec<Token> = Vec::new();
    let mut index = 0usize;
    while let Some(&character) = characters.get(index) {
        if character.is_whitespace() {
            index = index.saturating_add(1);
            continue;
        }
        match character {
            '(' => {
                tokens.push(Token::OpenParen);
                index = index.saturating_add(1);
            }
            ')' => {
                tokens.push(Token::CloseParen);
                index = index.saturating_add(1);
            }
            '+' | '*' | '/' => {
                tokens.push(Token::Operator(character));
                index = index.saturating_add(1);
            }
            '-' => {
                let follows_a_value = matches!(
                    tokens.last(),
                    Some(Token::Number(_) | Token::Word(_) | Token::CloseParen)
                );
                let opens_a_number = characters
                    .get(index.saturating_add(1))
                    .is_some_and(|next| next.is_ascii_digit() || *next == '.');
                if follows_a_value || !opens_a_number {
                    tokens.push(Token::Operator('-'));
                    index = index.saturating_add(1);
                } else {
                    let (text, next) = scan_number(&characters, index.saturating_add(1));
                    tokens.push(Token::Number(format!("-{text}")));
                    index = next;
                }
            }
            _ if character.is_ascii_digit() || character == '.' => {
                let (text, next) = scan_number(&characters, index);
                tokens.push(Token::Number(text));
                index = next;
            }
            // The degree sign is a unit WORD that happens not to be a letter. It is here
            // rather than left to fall through as an unexpected character because it is what
            // the drawing paints: an angle reads `45\u{b0}`, so the box that opens over it seeds
            // with `45\u{b0}`, and a seed the lexer cannot read would make an untouched commit a
            // parse failure.
            '\u{b0}' => {
                tokens.push(Token::Word(character.to_string()));
                index = index.saturating_add(1);
            }
            _ if character.is_ascii_alphabetic() || character == '_' => {
                let mut text = String::new();
                while let Some(&letter) = characters.get(index) {
                    if letter.is_ascii_alphanumeric() || letter == '_' {
                        text.push(letter);
                        index = index.saturating_add(1);
                    } else {
                        break;
                    }
                }
                tokens.push(Token::Word(text));
            }
            _ => {
                tokens.push(Token::Unexpected(character.to_string()));
                index = index.saturating_add(1);
            }
        }
    }
    tokens
}

/// Read one number run starting at `start`, returning its text and the index after it.
///
/// The fraction tail closes up only when a slash has a digit on both sides and no space anywhere
/// — `8/16` is one operand and `8 / 16` is two.
fn scan_number(characters: &[char], start: usize) -> (String, usize) {
    let mut text = String::new();
    let mut index = start;
    while let Some(&character) = characters.get(index) {
        if character.is_ascii_digit() || character == '.' {
            text.push(character);
            index = index.saturating_add(1);
        } else {
            break;
        }
    }
    if characters.get(index) == Some(&'/')
        && characters
            .get(index.saturating_add(1))
            .is_some_and(char::is_ascii_digit)
    {
        text.push('/');
        index = index.saturating_add(1);
        while let Some(&digit) = characters.get(index) {
            if digit.is_ascii_digit() {
                text.push(digit);
                index = index.saturating_add(1);
            } else {
                break;
            }
        }
    }
    (text, index)
}

/// Parse a single number token into an exact rational, or `None` when the token
/// is not a number at all (the caller decides whether that is an unknown unit or
/// garbage). Recognizes integer, decimal and fraction forms. A malformed number
/// (e.g. `"3.5.6"`, `"8/"`) is a hard error.
#[allow(clippy::too_many_lines)]
fn parse_number(token: &str) -> Result<Option<NumberLiteral>, MeasurementParseError> {
    // A token is "number-shaped" if it is only digits, a single dot, a single
    // slash and an optional leading minus. A purely alphabetic token is not a
    // number (→ `None`).
    if token.is_empty() {
        return Ok(None);
    }
    if token
        .chars()
        .any(|character| character.is_ascii_alphabetic())
    {
        return Ok(None);
    }

    if let Some((numerator_text, denominator_text)) = token.split_once('/') {
        let numerator: i128 =
            numerator_text
                .parse()
                .map_err(|_| MeasurementParseError::InvalidNumber {
                    number_text: token.to_string(),
                })?;
        let denominator: i128 =
            denominator_text
                .parse()
                .map_err(|_| MeasurementParseError::InvalidNumber {
                    number_text: token.to_string(),
                })?;
        let value = ExactRational::new(numerator, denominator).ok_or_else(|| {
            MeasurementParseError::ZeroDenominator {
                number_text: token.to_string(),
            }
        })?;
        return Ok(Some(NumberLiteral {
            value,
            source_text: token.to_string(),
        }));
    }

    if let Some((whole_text, fraction_text)) = token.split_once('.') {
        // Decimal: parse as scaled integer over a power of ten — exact, no f64.
        // `"3.5"` → 35/10 → 7/2.
        let negative = whole_text.starts_with('-');
        let whole_digits = whole_text.trim_start_matches('-');
        // Allow an empty whole part (`".5"`) but require numeric digits otherwise.
        if !whole_digits
            .chars()
            .all(|character| character.is_ascii_digit())
            || !fraction_text
                .chars()
                .all(|character| character.is_ascii_digit())
            || fraction_text.is_empty()
        {
            return Err(MeasurementParseError::InvalidNumber {
                number_text: token.to_string(),
            });
        }
        let whole_value: i128 = if whole_digits.is_empty() {
            0
        } else {
            whole_digits
                .parse()
                .map_err(|_| MeasurementParseError::InvalidNumber {
                    number_text: token.to_string(),
                })?
        };
        let fraction_value: i128 =
            fraction_text
                .parse()
                .map_err(|_| MeasurementParseError::InvalidNumber {
                    number_text: token.to_string(),
                })?;
        let Ok(scale_power) = u32::try_from(fraction_text.len()) else {
            return Err(MeasurementParseError::InvalidNumber {
                number_text: token.to_string(),
            });
        };
        let Some(scale) = 10i128.checked_pow(scale_power) else {
            return Err(MeasurementParseError::InvalidNumber {
                number_text: token.to_string(),
            });
        };
        let Some(scaled_whole) = whole_value.checked_mul(scale) else {
            return Err(MeasurementParseError::InvalidNumber {
                number_text: token.to_string(),
            });
        };
        let Some(unsigned_numerator) = scaled_whole.checked_add(fraction_value) else {
            return Err(MeasurementParseError::InvalidNumber {
                number_text: token.to_string(),
            });
        };
        let numerator = if negative {
            unsigned_numerator.checked_neg().ok_or_else(|| {
                MeasurementParseError::InvalidNumber {
                    number_text: token.to_string(),
                }
            })?
        } else {
            unsigned_numerator
        };
        let value = ExactRational::new(numerator, scale).ok_or_else(|| {
            MeasurementParseError::InvalidNumber {
                number_text: token.to_string(),
            }
        })?;
        return Ok(Some(NumberLiteral {
            value,
            source_text: token.to_string(),
        }));
    }

    // Plain integer.
    let integer: i128 = token
        .parse()
        .map_err(|_| MeasurementParseError::InvalidNumber {
            number_text: token.to_string(),
        })?;
    Ok(Some(NumberLiteral {
        value: ExactRational::from_integer(integer),
        source_text: token.to_string(),
    }))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

    use super::*;
    use std::num::NonZeroU32;

    /// Helper: parse then evaluate at one density, asserting the parse succeeds.
    fn parse_and_evaluate(input: &str, density: u32) -> Result<i64, MeasurementError> {
        let measurement = parse(input).expect("input should parse");
        measurement.to_voxels(density)
    }

    /// The refusal the shared vocabulary buys. `deg` is a unit the LEXER knows and the LENGTH
    /// grammar declines, by name.
    ///
    /// Before degrees joined the table this said "unknown unit `deg`" — accidentally close to
    /// right, and wrong in the way that matters: it sends an author who spelled the word
    /// correctly looking for a typo.
    #[test]
    fn a_degree_is_not_a_length() {
        for spelling in [
            "45 deg",
            "45deg",
            "45 degrees",
            "1 degree",
            "3 blocks 45 deg",
        ] {
            assert_eq!(
                parse(spelling),
                Err(MeasurementParseError::WrongDimension {
                    unit_text: spelling
                        .rsplit(|c: char| !c.is_ascii_alphabetic())
                        .next()
                        .unwrap()
                        .to_owned(),
                    reading: "length",
                }),
                "`{spelling}` is not a length"
            );
        }
    }

    /// And the same refusal in the other direction, so neither grammar is the privileged one.
    #[test]
    fn a_block_is_not_an_angle() {
        assert_eq!(
            angle_from_tokens(&tokenise("3 blocks")),
            Err(MeasurementParseError::WrongDimension {
                unit_text: "blocks".to_owned(),
                reading: "angle",
            })
        );
    }

    /// An angle reads the same number forms a length does — including the mixed fraction — and
    /// keeps them EXACTLY. `22 1/2` degrees is 45/2, not 22.5 rounded to whatever a float holds.
    #[test]
    fn an_angle_reads_every_number_form_exactly() {
        let cases: [(&str, i128, i128); 6] = [
            ("45 deg", 45, 1),
            ("45deg", 45, 1),
            ("22 1/2 degrees", 45, 2),
            ("1/3 deg", 1, 3),
            ("45\u{b0}", 45, 1),
            ("-30 deg", -30, 1),
        ];
        for (text, numerator, denominator) in cases {
            let angle = angle_from_tokens(&tokenise(text)).expect("an angle literal");
            let expected = ExactRational::new(numerator, denominator).expect("a valid rational");
            assert_eq!(angle.degrees(), expected, "`{text}`");
        }
    }

    /// The same shape of complaint a length makes, so an angle is not a grammar with its own
    /// manners.
    #[test]
    fn an_angle_complains_like_a_length_does() {
        assert_eq!(
            angle_from_tokens(&tokenise("45")),
            Err(MeasurementParseError::MissingUnit {
                number_text: "45".to_owned()
            })
        );
        assert_eq!(
            angle_from_tokens(&tokenise("deg")),
            Err(MeasurementParseError::MissingNumber {
                unit_text: "deg".to_owned()
            })
        );
        assert_eq!(
            angle_from_tokens(&tokenise("45 deg 10 deg")),
            Err(MeasurementParseError::DuplicateUnit {
                unit_text: "deg".to_owned()
            })
        );
    }

    /// `d` is deliberately NOT a degree: a single letter beside `b` and `v` reads as a third grid
    /// unit, and the author who means degrees can write three letters.
    #[test]
    fn a_lone_d_is_not_a_degree() {
        assert!(unit_dimension("d").is_none());
    }

    #[test]
    fn parse_and_evaluate_canonical_forms_at_density_sixteen() {
        assert_eq!(parse_and_evaluate("3.5 blocks", 16).unwrap(), 56);
        assert_eq!(parse_and_evaluate("8/16 blocks", 16).unwrap(), 8);
        assert_eq!(parse_and_evaluate("3 blocks 8 voxels", 16).unwrap(), 56);
        assert_eq!(parse_and_evaluate("3b 8v", 16).unwrap(), 56);
        assert_eq!(parse_and_evaluate("3 8/16 blocks", 16).unwrap(), 56);
        assert_eq!(parse_and_evaluate("56 voxels", 16).unwrap(), 56);
        assert_eq!(parse_and_evaluate("0.25 blocks", 16).unwrap(), 4);
    }

    #[test]
    fn parse_is_case_insensitive_and_accepts_spellings() {
        assert_eq!(parse_and_evaluate("3 BLOCKS 8 Voxels", 16).unwrap(), 56);
        assert_eq!(parse_and_evaluate("3 Block 8 Voxel", 16).unwrap(), 56);
        assert_eq!(parse_and_evaluate("56 V", 16).unwrap(), 56);
    }

    #[test]
    fn measurement_re_evaluates_parametrically_at_a_new_density() {
        // The SAME object evaluated at two densities — the lossless integer-ratio
        // refine: "3.5 blocks" → 56 voxels at d16 AND 112 voxels at d32.
        let measurement = parse("3.5 blocks").expect("parses");
        assert_eq!(measurement.to_voxels(16).unwrap(), 56);
        assert_eq!(measurement.to_voxels(32).unwrap(), 112);
    }

    #[test]
    fn evaluation_context_preserves_the_exact_voxel_rational() {
        let measurement = parse("3.5 blocks 1 voxels").expect("parses");
        let context = EvaluationContext::new(NonZeroU32::new(16).expect("non-zero"));
        assert_eq!(
            measurement.to_voxel_rational(context),
            ExactRational::from_integer(57)
        );
    }

    #[test]
    fn strict_non_landing_block_fraction_reports_nearest_voxels() {
        // "3.5 blocks" at an odd d=15 = 52.5 voxels: rejected with 52 and 53.
        let measurement = parse("3.5 blocks").expect("parses");
        match measurement.to_voxels(15) {
            Err(MeasurementError::BlockTermNotWholeVoxels {
                density,
                nearest_floor_voxels,
                nearest_ceil_voxels,
            }) => {
                assert_eq!(density, 15);
                assert_eq!(nearest_floor_voxels, 52);
                assert_eq!(nearest_ceil_voxels, 53);
            }
            other => panic!("expected non-landing error, got {other:?}"),
        }
    }

    #[test]
    fn non_landing_carries_voxel_term_in_nearest_values() {
        // "3.5 blocks 2 voxels" at d15 = 52.5 + 2 = 54.5 → nearest 54 and 55.
        let measurement = parse("3.5 blocks 2 voxels").expect("parses");
        match measurement.to_voxels(15) {
            Err(MeasurementError::BlockTermNotWholeVoxels {
                nearest_floor_voxels,
                nearest_ceil_voxels,
                ..
            }) => {
                assert_eq!(nearest_floor_voxels, 54);
                assert_eq!(nearest_ceil_voxels, 55);
            }
            other => panic!("expected non-landing error, got {other:?}"),
        }
    }

    #[test]
    fn reject_sub_voxel_voxel_terms() {
        assert_eq!(
            parse("8.5 voxels"),
            Err(MeasurementParseError::SubVoxel {
                number_text: "8.5".to_string()
            })
        );
        assert_eq!(
            parse("8/16 voxels"),
            Err(MeasurementParseError::SubVoxel {
                number_text: "8/16".to_string()
            })
        );
    }

    #[test]
    fn reject_malformed_input() {
        assert_eq!(parse(""), Err(MeasurementParseError::Empty));
        assert_eq!(parse("   "), Err(MeasurementParseError::Empty));
        assert_eq!(
            parse("5 furlongs"),
            Err(MeasurementParseError::UnknownUnit {
                unit_text: "furlongs".to_string()
            })
        );
        // A bare unit with no number.
        assert_eq!(
            parse("blocks"),
            Err(MeasurementParseError::MissingNumber {
                unit_text: "blocks".to_string()
            })
        );
        // A bare number with no unit.
        assert_eq!(
            parse("3"),
            Err(MeasurementParseError::MissingUnit {
                number_text: "3".to_string()
            })
        );
        // Garbage number.
        assert!(matches!(
            parse("3.5.6 blocks"),
            Err(MeasurementParseError::InvalidNumber { .. })
        ));
        // Zero denominator.
        assert_eq!(
            parse("8/0 blocks"),
            Err(MeasurementParseError::ZeroDenominator {
                number_text: "8/0".to_string()
            })
        );
        // Duplicate unit.
        assert_eq!(
            parse("3 blocks 2 blocks"),
            Err(MeasurementParseError::DuplicateUnit {
                unit_text: "blocks".to_string()
            })
        );
    }

    #[test]
    fn zero_density_evaluation_is_rejected() {
        let measurement = parse("3 blocks").expect("parses");
        assert_eq!(measurement.to_voxels(0), Err(MeasurementError::ZeroDensity));
    }

    #[test]
    fn formatter_canonical_forms() {
        assert_eq!(format(56, 16, DisplayUnit::DecimalBlocks), "3.5 blocks");
        assert_eq!(
            format(56, 16, DisplayUnit::BlocksAndVoxels),
            "3 blocks 8 voxels"
        );
        assert_eq!(format(56, 16, DisplayUnit::Voxels), "56 voxels");
        assert_eq!(format(56, 16, DisplayUnit::BlockFraction), "3 8/16 blocks");
    }

    #[test]
    fn formatter_whole_and_singular_agreement() {
        // Exactly one block: singular, no remainder.
        assert_eq!(format(16, 16, DisplayUnit::DecimalBlocks), "1 block");
        assert_eq!(format(16, 16, DisplayUnit::BlockFraction), "1 block");
        assert_eq!(
            format(16, 16, DisplayUnit::BlocksAndVoxels),
            "1 block 0 voxels"
        );
        assert_eq!(format(1, 16, DisplayUnit::Voxels), "1 voxel");
        // A whole multiple of blocks renders as whole blocks in DecimalBlocks.
        assert_eq!(format(48, 16, DisplayUnit::DecimalBlocks), "3 blocks");
    }

    #[test]
    fn formatter_round_trips_through_parser() {
        // parse(format(x)) re-evaluates back to x for several values and styles.
        for voxels in [0_i64, 4, 8, 16, 32, 56, 100, 257] {
            for style in [
                DisplayUnit::DecimalBlocks,
                DisplayUnit::BlocksAndVoxels,
                DisplayUnit::BlockFraction,
                DisplayUnit::Voxels,
            ] {
                let text = format(voxels, 16, style);
                let reparsed = parse(&text)
                    .unwrap_or_else(|error| panic!("`{text}` should re-parse: {error}"));
                assert_eq!(
                    reparsed.to_voxels(16).unwrap(),
                    voxels,
                    "round-trip failed for {voxels} via `{text}` ({style:?})"
                );
            }
        }
    }

    #[test]
    fn decimal_formatter_falls_back_when_not_terminating() {
        // 1/3 of a block (voxels not a 2/5-smooth fraction of d) cannot be a clean
        // decimal, so DecimalBlocks falls back to the honest mixed form rather than
        // a rounded float. At d=3, 1 voxel = 1/3 block.
        assert_eq!(format(1, 3, DisplayUnit::DecimalBlocks), "0 blocks 1 voxel");
    }

    #[test]
    fn exact_decimal_parses_without_float_loss() {
        // "3.5" is 7/2, never an f64. Verify the stored rational is exactly 7/2.
        let block_term = parse("3.5 blocks").expect("parses").block_term();
        assert_eq!(block_term, ExactRational::new(7, 2).unwrap());
    }

    #[test]
    fn mixed_fraction_idiom_sums_integer_and_fraction() {
        // "3 8/16 blocks" must mean 3 + 8/16 = 3.5 blocks, NOT 3 then a separate
        // 8/16 term. Verify the retained block term is exactly 7/2.
        let block_term = parse("3 8/16 blocks").expect("parses").block_term();
        assert_eq!(block_term, ExactRational::new(7, 2).unwrap());
    }

    #[test]
    fn parse_accepts_signed_offsets() {
        // Offsets are signed: a leading minus on each term parses through the
        // tokeniser (the minus stays glued to the number, the unit letter peels
        // off after it) and the term parses negative.
        assert_eq!(parse_and_evaluate("-3b", 16).unwrap(), -48);
        assert_eq!(parse_and_evaluate("-1b 4v", 16).unwrap(), -12);
        assert_eq!(parse_and_evaluate("-3.5 blocks", 16).unwrap(), -56);
        assert_eq!(parse_and_evaluate("-8/16 blocks", 16).unwrap(), -8);
        assert_eq!(parse_and_evaluate("-12 voxels", 16).unwrap(), -12);
        // A negative block term with a positive voxel term sums signed.
        assert_eq!(parse_and_evaluate("-1 blocks 4 voxels", 16).unwrap(), -12);
    }

    #[test]
    fn signed_offsets_round_trip_through_formatter() {
        // parse(format(x)) re-evaluates back to x for negative voxel counts and
        // every style, so a negative offset displays and re-parses losslessly.
        for voxels in [-1_i64, -8, -16, -56, -100, -257] {
            for style in [
                DisplayUnit::DecimalBlocks,
                DisplayUnit::BlocksAndVoxels,
                DisplayUnit::BlockFraction,
                DisplayUnit::Voxels,
            ] {
                let text = format(voxels, 16, style);
                let reparsed = parse(&text)
                    .unwrap_or_else(|error| panic!("`{text}` should re-parse: {error}"));
                assert_eq!(
                    reparsed.to_voxels(16).unwrap(),
                    voxels,
                    "round-trip failed for {voxels} via `{text}` ({style:?})"
                );
            }
        }
    }

    #[test]
    fn from_voxels_is_a_pure_voxel_measurement() {
        // A synthesized measurement re-evaluates to exactly its voxel count at any
        // density (the block term is zero, so density never scales it).
        for voxels in [-48_i64, -1, 0, 7, 56] {
            let measurement = Measurement::from_voxels(voxels);
            assert_eq!(measurement.block_term(), ExactRational::from_integer(0));
            assert_eq!(measurement.voxel_term(), voxels);
            assert_eq!(measurement.to_voxels(16).unwrap(), voxels);
            assert_eq!(measurement.to_voxels(32).unwrap(), voxels);
        }
    }

    #[test]
    fn measurement_is_serde_round_trippable() {
        let measurement = parse("3 8/16 blocks 5 voxels").expect("parses");
        let json = serde_json::to_string(&measurement).expect("serializes");
        let restored: Measurement = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(restored, measurement);
        assert_eq!(
            restored.to_voxels(16).unwrap(),
            measurement.to_voxels(16).unwrap()
        );
    }

    #[test]
    fn solved_angle_keeps_the_exact_f64_value_or_reports_the_rational_boundary() {
        let value = 123.4567;
        let solved = AngleMeasurement::try_from_degrees_f64(value).expect("ordinary angle fits");
        assert_eq!(solved.to_degrees_f64().to_bits(), value.to_bits());
        assert_eq!(
            AngleMeasurement::try_from_degrees_f64(f64::NAN),
            Err(RationalFromF64Error::NonFinite)
        );
        assert_eq!(
            AngleMeasurement::try_from_degrees_f64(f64::from_bits(1)),
            Err(RationalFromF64Error::OutOfRange)
        );
    }
}
