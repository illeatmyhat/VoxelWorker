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
//! the crate. It feeds sketch dimensions and `SetOffset` (the placement Intents)
//! in a later slice.

use std::fmt;

/// An exact, always-reduced rational number backed by `i128`.
///
/// The units layer must not touch `f64` — parsing `"3.5"` as `7/2` and never an
/// `f64` is what makes `"3.5 blocks"` land on exactly 56 voxels at d16 with no
/// float drift. `num-rational` is not a project dependency and the policy is "no
/// new external dependency", so this is a minimal in-house rational: numerator +
/// denominator reduced by their gcd, denominator kept positive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExactRational {
    numerator: i128,
    denominator: i128,
}

impl ExactRational {
    /// A reduced rational from a raw numerator/denominator. Returns `None` on a
    /// zero denominator (the only un-representable case). The sign is normalised
    /// onto the numerator so the denominator is always positive, and both are
    /// divided through by their greatest common divisor.
    pub fn new(numerator: i128, denominator: i128) -> Option<Self> {
        if denominator == 0 {
            return None;
        }
        let sign = if denominator < 0 { -1 } else { 1 };
        let mut numerator = numerator * sign;
        let mut denominator = denominator * sign;
        let divisor = greatest_common_divisor(numerator.unsigned_abs(), denominator.unsigned_abs())
            as i128;
        if divisor > 1 {
            numerator /= divisor;
            denominator /= divisor;
        }
        Some(Self {
            numerator,
            denominator,
        })
    }

    /// A whole-number rational (`value / 1`).
    pub fn from_integer(value: i128) -> Self {
        Self {
            numerator: value,
            denominator: 1,
        }
    }

    /// The reduced numerator (sign lives here; denominator is always positive).
    pub fn numerator(self) -> i128 {
        self.numerator
    }

    /// The reduced denominator (always `>= 1`).
    pub fn denominator(self) -> i128 {
        self.denominator
    }

    /// `self * other`, reduced.
    pub fn times(self, other: ExactRational) -> ExactRational {
        // Operands are already reduced; reducing again after the cross-multiply
        // keeps the magnitudes small and the result canonical.
        ExactRational::new(
            self.numerator * other.numerator,
            self.denominator * other.denominator,
        )
        .expect("non-zero denominators multiply to a non-zero denominator")
    }

    /// `self + other`, reduced.
    pub fn plus(self, other: ExactRational) -> ExactRational {
        ExactRational::new(
            self.numerator * other.denominator + other.numerator * self.denominator,
            self.denominator * other.denominator,
        )
        .expect("non-zero denominators add to a non-zero denominator")
    }

    /// `true` when this rational is a whole number (denominator reduced to 1).
    pub fn is_integer(self) -> bool {
        self.denominator == 1
    }

    /// The whole-number value when [`is_integer`](Self::is_integer); otherwise
    /// `None`. Used by the evaluator to demand a clean voxel landing.
    pub fn to_integer(self) -> Option<i128> {
        if self.is_integer() {
            Some(self.numerator)
        } else {
            None
        }
    }

    /// The largest integer `<= self` (toward negative infinity).
    pub fn floor(self) -> i128 {
        // Truncating division rounds toward zero; for a negative non-integer that
        // is one too large, so step down.
        let truncated = self.numerator / self.denominator;
        if self.numerator % self.denominator != 0 && self.numerator < 0 {
            truncated - 1
        } else {
            truncated
        }
    }

    /// The smallest integer `>= self` (toward positive infinity).
    pub fn ceil(self) -> i128 {
        let truncated = self.numerator / self.denominator;
        if self.numerator % self.denominator != 0 && self.numerator > 0 {
            truncated + 1
        } else {
            truncated
        }
    }
}

/// Euclid's algorithm on unsigned magnitudes. `gcd(x, 0) == x`, so a `0`
/// numerator reduces against any denominator to leave the denominator as the
/// divisor (giving the canonical `0/1`).
fn greatest_common_divisor(mut first: u128, mut second: u128) -> u128 {
    while second != 0 {
        let remainder = first % second;
        first = second;
        second = remainder;
    }
    first.max(1)
}

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
            match decimal_string(blocks) {
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

/// Render a reduced rational as a terminating decimal string, or `None` when it
/// does not terminate in base 10 (denominator has a prime factor other than 2 or
/// 5). Pure integer arithmetic — no `f64` anywhere.
fn decimal_string(value: ExactRational) -> Option<String> {
    if value.is_integer() {
        return Some(value.numerator().to_string());
    }
    // Strip factors of 2 and 5 from the denominator; whatever remains must be 1
    // for the decimal to terminate.
    let mut denominator = value.denominator();
    let mut factor_twos = 0;
    let mut factor_fives = 0;
    while denominator % 2 == 0 {
        denominator /= 2;
        factor_twos += 1;
    }
    while denominator % 5 == 0 {
        denominator /= 5;
        factor_fives += 1;
    }
    if denominator != 1 {
        return None;
    }
    // Scale numerator/denominator up to a power of ten, then split off the
    // fractional digits.
    let fractional_digits = factor_twos.max(factor_fives);
    let mut scaled_numerator = value.numerator();
    for _ in 0..(fractional_digits - factor_twos) {
        scaled_numerator *= 2;
    }
    for _ in 0..(fractional_digits - factor_fives) {
        scaled_numerator *= 5;
    }
    let scale = 10i128.pow(fractional_digits as u32);
    let negative = scaled_numerator < 0;
    let magnitude = scaled_numerator.unsigned_abs();
    let whole_part = (magnitude / scale as u128) as i128;
    let fraction_part = (magnitude % scale as u128) as i128;
    let mut fraction_text = format!("{fraction_part:0width$}", width = fractional_digits as usize);
    while fraction_text.ends_with('0') {
        fraction_text.pop();
    }
    let sign = if negative { "-" } else { "" };
    if fraction_text.is_empty() {
        Some(format!("{sign}{whole_part}"))
    } else {
        Some(format!("{sign}{whole_part}.{fraction_text}"))
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
    fn exact_rationals_do_not_drift_like_floats() {
        // 0.1 + 0.2 is the canonical f64 trap (== 0.30000000000000004). As exact
        // rationals it is precisely 3/10.
        let tenth = parse("0.1 blocks").expect("parses").block_term();
        let fifth = parse("0.2 blocks").expect("parses").block_term();
        let sum = tenth.plus(fifth);
        assert_eq!(sum, ExactRational::new(3, 10).unwrap());
        // And "0.1 blocks" + "0.2 blocks" at d=10 lands on exactly 3 voxels.
        let combined = Measurement::new(sum, 0);
        assert_eq!(combined.to_voxels(10).unwrap(), 3);
    }

    #[test]
    fn exact_decimal_parses_without_float_loss() {
        // "3.5" is 7/2, never an f64. Verify the stored rational is exactly 7/2.
        let block_term = parse("3.5 blocks").expect("parses").block_term();
        assert_eq!(block_term, ExactRational::new(7, 2).unwrap());
    }

    #[test]
    fn rational_floor_and_ceil_handle_signs() {
        let half = ExactRational::new(1, 2).unwrap();
        assert_eq!(half.floor(), 0);
        assert_eq!(half.ceil(), 1);
        let negative_half = ExactRational::new(-1, 2).unwrap();
        assert_eq!(negative_half.floor(), -1);
        assert_eq!(negative_half.ceil(), 0);
        let whole = ExactRational::from_integer(5);
        assert_eq!(whole.floor(), 5);
        assert_eq!(whole.ceil(), 5);
    }

    #[test]
    fn mixed_fraction_idiom_sums_integer_and_fraction() {
        // "3 8/16 blocks" must mean 3 + 8/16 = 3.5 blocks, NOT 3 then a separate
        // 8/16 term. Verify the retained block term is exactly 7/2.
        let block_term = parse("3 8/16 blocks").expect("parses").block_term();
        assert_eq!(block_term, ExactRational::new(7, 2).unwrap());
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
