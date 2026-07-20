//! ADR 0003 §3f(0) units layer: the parametric blocks/voxels measurement core.
//!
//! Placement, sizes and radii are stored as **canonical voxels** at the
//! document's density `d` (`blocks · d = voxels`). A user-facing measurement is
//! a *unit expression* parsed onto that canonical store and formatted back —
//! exactly Fusion's model (it stores one canonical unit internally and lets you
//! type measurements in any unit). The two grid-native units are **blocks** and
//! **voxels**.
//!
//! A [`Measurement`] RETAINS its authored expression (parametric): it is a sum of
//! a BLOCK term (an exact rational — blocks may be integer, decimal, fraction or
//! the VS sixteenths idiom) plus a VOXEL term (an integer count of voxels). The
//! canonical voxel count is the DERIVED value, recomputed at a supplied density
//! `d` via [`Measurement::to_voxels`]. Because `d` is supplied at eval time, the
//! SAME measurement re-evaluates at a new `d` — the lossless refine for
//! integer-ratio re-targets (`"3.5 blocks"` → 56 at d16, 112 at d32).
//!
//! Parser policy is STRICT (ADR 0003 §3f(0), 2026-06-28): measurements evaluate
//! as EXACT RATIONALS (no floats), fractions/decimals are allowed on block-terms
//! only, a voxel-term must be an integer, and a block-fraction that does not land
//! on a whole voxel at the current `d` is rejected with the nearest representable
//! values reported — never silently rounded.
//!
//! This module is pure logic: it has no UI wiring and depends on nothing else in
//! the crate. It feeds `SetOffset` (the placement Intent, now landed as
//! `NodeTransform::from_measurements`); sketch dimensions do not retain a
//! `Measurement` yet and stay plain voxel-granular integers.

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
pub use substrate::interval::Rational as ExactRational;

/// A parametric blocks + voxels measurement (ADR 0003 §3f(0)).
///
/// This is the STORED, authored expression — `block_term · d + voxel_term`
/// voxels at a density `d` supplied at eval time. It is serde-serializable
/// because it is persisted alongside the document later (the placement Intents
/// carry the expression, not just the derived voxel count, so replay/undo
/// preserve authored intent).
///
/// The block term is an exact rational so `"3.5 blocks"`, `"8/16 blocks"` and
/// `"3 8/16 blocks"` are all retained losslessly; the voxel term is a plain
/// integer because nothing is finer than a voxel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Measurement {
    /// Block count as an exact rational (numerator, denominator). Serialised as
    /// the reduced pair so a persisted document is float-free end to end.
    block_term_numerator: i128,
    block_term_denominator: i128,
    /// Whole voxels added on top of the block term.
    voxel_term: i64,
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
    pub fn new(block_term: ExactRational, voxel_term: i64) -> Self {
        Self {
            block_term_numerator: block_term.numerator(),
            block_term_denominator: block_term.denominator(),
            voxel_term,
        }
    }

    /// The block term as an exact rational.
    pub fn block_term(self) -> ExactRational {
        // The stored pair came from a reduced `ExactRational`, so reconstruction
        // is exact; `expect` cannot fire (the denominator is never zero).
        ExactRational::new(self.block_term_numerator, self.block_term_denominator)
            .expect("stored block-term denominator is non-zero")
    }

    /// The whole-voxel term.
    pub fn voxel_term(self) -> i64 {
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
    pub fn from_voxels(voxels: i64) -> Self {
        Self::new(ExactRational::from_integer(0), voxels)
    }

    /// Evaluate to an exact voxel count at the given density `d`.
    ///
    /// `voxels = block_term · d + voxel_term`. The block contribution MUST land
    /// on a whole voxel: if `block_term · d` is not an integer (e.g. `"3.5
    /// blocks"` at `d = 15` = 52.5), this returns
    /// [`MeasurementError::BlockTermNotWholeVoxels`] reporting the nearest
    /// representable floor/ceil voxel counts — it never silently rounds.
    pub fn to_voxels(self, density: u32) -> Result<i64, MeasurementError> {
        if density == 0 {
            return Err(MeasurementError::ZeroDensity);
        }
        let block_voxels = self
            .block_term()
            .times(ExactRational::from_integer(density as i128));
        let whole_block_voxels = match block_voxels.to_integer() {
            Some(value) => value,
            None => {
                // Report the nearest representable voxel counts for the WHOLE
                // measurement (block contribution rounded each way, plus the
                // exact voxel term) so the caller can show "did you mean 52 or
                // 53 voxels?".
                let floor_voxels = block_voxels.floor() + self.voxel_term as i128;
                let ceil_voxels = block_voxels.ceil() + self.voxel_term as i128;
                return Err(MeasurementError::BlockTermNotWholeVoxels {
                    density,
                    nearest_floor_voxels: floor_voxels as i64,
                    nearest_ceil_voxels: ceil_voxels as i64,
                });
            }
        };
        Ok(whole_block_voxels as i64 + self.voxel_term)
    }
}

/// A unit a [`Measurement`] / voxel count can be FORMATTED into (the display
/// side of the units layer; ADR 0003 §3f(0)).
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
pub fn format(voxels: i64, density: u32, style: DisplayUnit) -> String {
    let density = density.max(1) as i64;
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
            let blocks =
                ExactRational::new(voxels as i128, density as i128).expect("density >= 1");
            match blocks.to_terminating_decimal() {
                Some(text) => format!("{text} {}", pluralise_rational(blocks, "block")),
                None => {
                    // Non-terminating in base 10 (e.g. 1/3 of a block): present the
                    // honest mixed form rather than a truncated decimal.
                    let whole_blocks = voxels.div_euclid(density);
                    let remainder_voxels = voxels.rem_euclid(density);
                    format!(
                        "{whole_blocks} {} {remainder_voxels} {}",
                        pluralise(whole_blocks, "block"),
                        pluralise(remainder_voxels, "voxel"),
                    )
                }
            }
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
    /// An unrecognised unit word (not blocks/voxels and not a number).
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
}

impl fmt::Display for MeasurementParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MeasurementParseError::Empty => write!(formatter, "empty measurement"),
            MeasurementParseError::MissingUnit { number_text } => write!(
                formatter,
                "number `{number_text}` has no unit (expected blocks or voxels)"
            ),
            MeasurementParseError::MissingNumber { unit_text } => {
                write!(formatter, "unit `{unit_text}` has no preceding number")
            }
            MeasurementParseError::UnknownUnit { unit_text } => {
                write!(formatter, "unknown unit `{unit_text}` (expected blocks or voxels)")
            }
            MeasurementParseError::InvalidNumber { number_text } => {
                write!(formatter, "`{number_text}` is not a valid number")
            }
            MeasurementParseError::SubVoxel { number_text } => write!(
                formatter,
                "`{number_text}` voxels is sub-voxel; voxels must be whole (use a block-fraction or a denser document)"
            ),
            MeasurementParseError::DuplicateUnit { unit_text } => {
                write!(formatter, "unit `{unit_text}` appears more than once")
            }
            MeasurementParseError::ZeroDenominator { number_text } => {
                write!(formatter, "`{number_text}` has a zero denominator")
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
            MeasurementError::BlockTermNotWholeVoxels {
                density,
                nearest_floor_voxels,
                nearest_ceil_voxels,
            } => write!(
                formatter,
                "block term does not land on a whole voxel at density {density}; nearest are {nearest_floor_voxels} or {nearest_ceil_voxels} voxels"
            ),
            MeasurementError::ZeroDensity => write!(formatter, "density must be at least 1"),
        }
    }
}

impl std::error::Error for MeasurementError {}

/// Which grid-native unit a token names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnitKind {
    Blocks,
    Voxels,
}

/// Classify a unit word (case-insensitive). `None` for anything that is not a
/// recognised unit. Accepts the long, short and single-letter spellings.
fn classify_unit(word: &str) -> Option<UnitKind> {
    match word.to_ascii_lowercase().as_str() {
        "blocks" | "block" | "b" => Some(UnitKind::Blocks),
        "voxels" | "voxel" | "v" => Some(UnitKind::Voxels),
        _ => None,
    }
}

/// Parse a units expression into a [`Measurement`] (STRICT; ADR 0003 §3f(0)).
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
pub fn parse(input: &str) -> Result<Measurement, MeasurementParseError> {
    let tokens = tokenise(input);
    if tokens.is_empty() {
        return Err(MeasurementParseError::Empty);
    }

    let mut block_total = ExactRational::from_integer(0);
    let mut voxel_total: i64 = 0;
    let mut seen_blocks = false;
    let mut seen_voxels = false;

    // Number parts accumulate until a unit closes the term. The VS sixteenths
    // idiom `"3 8/16"` is two number tokens (an integer then a fraction) before
    // the unit, so we collect a small buffer and sum it when the unit arrives.
    let mut pending_numbers: Vec<NumberLiteral> = Vec::new();
    let mut pending_text = String::new();

    for token in tokens {
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
                    UnitKind::Voxels => {
                        if seen_voxels {
                            return Err(MeasurementParseError::DuplicateUnit { unit_text: token });
                        }
                        seen_voxels = true;
                        // Sub-voxel rejection: every accumulated number for a
                        // voxel term must be a whole integer.
                        let mut term: i64 = 0;
                        for number in &pending_numbers {
                            match number.to_rational().to_integer() {
                                Some(whole) => term += whole as i64,
                                None => {
                                    return Err(MeasurementParseError::SubVoxel {
                                        number_text: number.source_text.clone(),
                                    })
                                }
                            }
                        }
                        voxel_total += term;
                    }
                }
                pending_numbers.clear();
                pending_text.clear();
            }
            None => {
                // Not a unit, so it must be a number literal; a non-number here is
                // either an unknown unit word or garbage.
                match parse_number(&token)? {
                    Some(number) => {
                        if !pending_text.is_empty() {
                            pending_text.push(' ');
                        }
                        pending_text.push_str(&token);
                        pending_numbers.push(number);
                    }
                    None => {
                        // Alphabetic but not a known unit → unknown unit; otherwise
                        // an unparseable number token.
                        if token.chars().any(|character| character.is_ascii_alphabetic()) {
                            return Err(MeasurementParseError::UnknownUnit { unit_text: token });
                        }
                        return Err(MeasurementParseError::InvalidNumber { number_text: token });
                    }
                }
            }
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

/// One parsed number literal plus its original text (kept for error messages).
#[derive(Debug, Clone)]
struct NumberLiteral {
    value: ExactRational,
    source_text: String,
}

impl NumberLiteral {
    fn to_rational(&self) -> ExactRational {
        self.value
    }
}

/// Break the input into number / unit tokens.
///
/// Splits on whitespace, then peels a trailing or leading unit letter that is
/// glued to digits (`"3b"`, `"8v"`) so the compact spelling tokenises the same as
/// the spaced one. A `/` stays inside its token so a fraction is one token.
fn tokenise(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for raw in input.split_whitespace() {
        // Peel a glued alphabetic suffix off a digit run: `"3b"` → `["3", "b"]`,
        // `"3.5blocks"` → `["3.5", "blocks"]`. Only split when the head is numeric
        // and the tail is alphabetic, so a bare `"blocks"` stays whole.
        let split_at = raw
            .char_indices()
            .find(|(_, character)| character.is_ascii_alphabetic())
            .map(|(index, _)| index);
        match split_at {
            Some(index) if index > 0 => {
                tokens.push(raw[..index].to_string());
                tokens.push(raw[index..].to_string());
            }
            _ => tokens.push(raw.to_string()),
        }
    }
    tokens
}

/// Parse a single number token into an exact rational, or `None` when the token
/// is not a number at all (the caller decides whether that is an unknown unit or
/// garbage). Recognises integer, decimal and fraction forms. A malformed number
/// (e.g. `"3.5.6"`, `"8/"`) is a hard error.
fn parse_number(token: &str) -> Result<Option<NumberLiteral>, MeasurementParseError> {
    // A token is "number-shaped" if it is only digits, a single dot, a single
    // slash and an optional leading minus. A purely alphabetic token is not a
    // number (→ `None`).
    if token.is_empty() {
        return Ok(None);
    }
    if token.chars().any(|character| character.is_ascii_alphabetic()) {
        return Ok(None);
    }

    if let Some((numerator_text, denominator_text)) = token.split_once('/') {
        let numerator: i128 = numerator_text
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
        let value = ExactRational::new(numerator, denominator).ok_or(
            MeasurementParseError::ZeroDenominator {
                number_text: token.to_string(),
            },
        )?;
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
        if !whole_digits.chars().all(|character| character.is_ascii_digit())
            || !fraction_text.chars().all(|character| character.is_ascii_digit())
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
        let scale = 10i128.pow(fraction_text.len() as u32);
        let mut numerator = whole_value * scale + fraction_value;
        if negative {
            numerator = -numerator;
        }
        let value = ExactRational::new(numerator, scale).expect("power of ten is non-zero");
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
    use super::*;

    /// Helper: parse then evaluate at one density, asserting the parse succeeds.
    fn parse_and_evaluate(input: &str, density: u32) -> Result<i64, MeasurementError> {
        let measurement = parse(input).expect("input should parse");
        measurement.to_voxels(density)
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
        // A synthesised measurement re-evaluates to exactly its voxel count at any
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
        let json = serde_json::to_string(&measurement).expect("serialises");
        let restored: Measurement = serde_json::from_str(&json).expect("deserialises");
        assert_eq!(restored, measurement);
        assert_eq!(restored.to_voxels(16).unwrap(), measurement.to_voxels(16).unwrap());
    }
}
