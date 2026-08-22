//! What the text in a quantity box MEANS: one binding per dimension.
//!
//! [`QuantityEntry`](super::quantity_entry::QuantityEntry) owns the text and the protocol and is
//! dimension-free. A binding owns the other half — which grammar reads the text, what bounds it
//! must satisfy, how the accepted value renders back, and what sentence explains a refusal.
//!
//! **Two bindings, side by side, and no enum over them.** A commit shaped as
//! `Committed(Length | Angle)` would teach every consumer the dimension list, so a field that
//! only ever holds one would still have to name the other. Here a caller picks the binding it
//! wants, gets that binding's own value type back, and the day a third dimension arrives it
//! touches neither of these.
//!
//! ## Every binding must be able to read its own seed
//!
//! A box the author opens and leaves alone must commit nothing. The protocol arranges that by
//! comparing against the seed — but the seed is the BINDING's rendering, so a binding whose
//! formatter emits something its parser cannot read would turn an untouched box into a parse
//! failure the moment focus moved. The round-trip is asserted per binding below, and it is why
//! the degree sign is in the lexer.

use super::quantity_entry::Accepted;
use parametric::units::{self, AngleMeasurement, DisplayUnit, Measurement, MeasurementError};

/// A successful length commit: the authored expression AND what it landed on.
///
/// Both halves matter and neither is derivable from the other at the call site — the
/// `measurement` is RETAINED on the document (lossless density re-targeting and exact-expression
/// undo), while `voxels` is the canonical value the resolve actually uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeasurementCommit {
    /// The authored expression, to retain on the document.
    pub measurement: Measurement,
    /// The canonical voxel value it lands on at the current density.
    pub voxels: i64,
}

/// The LENGTH binding: blocks and voxels, at a density, optionally floored.
///
/// ## Signed by default
///
/// A length is signed unless [`floor`](Self::floor) says otherwise. An offset moves either way,
/// and an outset that insets is a NEGATIVE outset — so the bound is opt-in, and the sites that
/// need one carry their own message.
#[derive(Debug, Clone, Copy)]
pub struct LengthBinding<'a> {
    density: u32,
    floor: Option<(i64, &'a str)>,
}

impl<'a> LengthBinding<'a> {
    /// A signed length binding reading at `density`.
    #[must_use]
    pub const fn new(density: u32) -> Self {
        Self {
            density,
            floor: None,
        }
    }

    /// Reject anything below `minimum` voxels, reporting `message`.
    ///
    /// For quantities that are not signed — a size of zero is not a size. Omit this for anything
    /// that may legitimately go negative.
    #[must_use]
    pub const fn floor(mut self, minimum: i64, message: &'a str) -> Self {
        self.floor = Some((minimum, message));
        self
    }

    /// How a voxel count renders as the text a box opens on.
    #[must_use]
    pub fn seed(voxels: i64, density: u32) -> String {
        units::format(voxels, density, DisplayUnit::BlocksAndVoxels)
    }

    /// Read `text` as a length, or say why it is not one.
    ///
    /// # Errors
    ///
    /// Returns the sentence to show under the box: a grammar complaint, a value that does not
    /// land on a whole voxel, or this binding's own floor message.
    pub fn read(&self, text: &str) -> Result<Accepted<MeasurementCommit>, String> {
        let measurement = self.authored_measurement(text)?;
        let voxels = measurement
            .to_voxels(self.density)
            .map_err(|error| measurement_error_text(&error))?;
        match self.floor {
            Some((minimum, message)) if voxels < minimum => Err(message.to_owned()),
            _ => Ok(Accepted {
                value: MeasurementCommit {
                    measurement,
                    voxels,
                },
                settled_text: Self::seed(voxels, self.density),
            }),
        }
    }

    /// Read `text` as an expression and reduce it to the [`Measurement`] to retain.
    ///
    /// The grammar read here is the same one a named parameter will be typed into. Today the
    /// symbol table is empty, so a name has no definition and is refused as one; nothing here
    /// changes on the day it has some.
    ///
    /// **A lone literal keeps its authored split, and only a lone literal can.** `3 blocks` is
    /// retained as a block term, which re-targets when the density changes. Anything compound is
    /// retained as the voxel count it evaluated to, because the evaluated value is a count and a
    /// count has no split to keep. That closes when the document retains the author's TEXT beside
    /// the measurement and re-reads it.
    fn authored_measurement(&self, text: &str) -> Result<Measurement, String> {
        let expression = parametric::expression::parse(text).map_err(|error| error.to_string())?;
        if let Some(measurement) = expression.as_authored_length() {
            return Ok(measurement);
        }
        let value = parametric::expression::SymbolTable::new()
            .evaluate(&expression, self.density)
            .map_err(|error| error.to_string())?;
        // Text that named no unit takes this binding's: blocks. The grammar keeps a bare number
        // dimensionless — `2 * 3 blocks` needs its scale factor — so the default is applied here,
        // by the party that knows which quantity the field holds, and keyed on the TREE rather
        // than the answer so a cancelled ratio (`3 blocks / 1 block`) is not mistaken for a count.
        // It is minted as a BLOCK term, so `3` re-targets with the density exactly as `3 blocks`
        // does. (`2 blocks + 4` stays a mismatch: the default is not a rescue for one term of a sum.)
        if expression.names_no_unit() {
            return Ok(Measurement::new(value.value, 0));
        }
        // The dimension check, by name, before the arithmetic gets a chance to complain in its
        // own words. `45 deg` now PARSES — the grammar has angle literals — so what stops it here
        // is this binding saying which quantity it holds, and the alternative is the algebra's
        // "cannot combine an angle with a length", which describes a sum nobody wrote.
        if value.dimension != parametric::Dimension::LENGTH {
            return Err(format!(
                "`{text}` is not a length; type blocks and voxels, like `2 blocks 4 voxels`"
            ));
        }
        let voxels = value.to_whole_voxels().map_err(|error| error.to_string())?;
        Ok(Measurement::from_voxels(voxels))
    }
}

/// The ANGLE binding: degrees, within the half turn a drawing can tell apart.
///
/// ## Why the range is real
///
/// The solver's angle row is `sin(turn - radians)`, and a sine repeats every half turn. That is
/// not a shortcut: a segment has two ends and no preferred one, so which way it points is not
/// something the drawing knows, and `200` and `20` are literally the same claim about it. An
/// author who types 200 and watches the drawing settle at 20 has been told nothing; refusing by
/// name and offering the equivalent is the only honest answer.
///
/// There is no `density` here and no floor, which is the whole reason this is not the length
/// binding with a flag: an angle has neither.
#[derive(Debug, Clone, Copy, Default)]
pub struct AngleBinding;

/// The largest angle a drawing can state, in degrees. See [`AngleBinding`].
pub const LARGEST_STATABLE_DEGREES: f64 = 180.0;

impl AngleBinding {
    /// How a stored angle renders — on the drawing AND as the text a box opens on.
    ///
    /// **One renderer, called from both places.** The seed has to be character-for-character what
    /// the drawing paints, or a box the author opened and left alone would differ from its seed
    /// and commit on the way out. Two formatters that agree today is how that stops being true.
    ///
    /// Two decimal places with the trailing zeros taken off, so a whole angle reads `45\u{b0}` and a
    /// solved one reads `31.24\u{b0}`. The degree SIGN rather than the word, because that is what the
    /// author is looking at; the lexer reads the sign, which is what keeps the seed parseable.
    #[must_use]
    pub fn seed(angle: AngleMeasurement) -> String {
        let text = format!("{:.2}", angle.to_degrees_f64());
        let trimmed = match text.trim_end_matches('0').trim_end_matches('.') {
            // A value that rounds away to nothing IS zero, and nobody writes that as "-0".
            "" | "-" | "-0" => "0",
            trimmed => trimmed,
        };
        format!("{trimmed}\u{b0}")
    }

    /// Read `text` as an angle, or say why it is not one.
    ///
    /// # Errors
    ///
    /// Returns the sentence to show under the box: a grammar complaint, a value the exact-rational
    /// store cannot hold, or the half-turn refusal.
    pub fn read(&self, text: &str) -> Result<Accepted<AngleMeasurement>, String> {
        let expression = parametric::expression::parse(text).map_err(|error| error.to_string())?;
        // Density is irrelevant to an angle but the evaluator takes one for lengths; any nonzero
        // value gives the same answer, and 1 says "not used" loudest.
        let value = parametric::expression::SymbolTable::new()
            .evaluate(&expression, 1)
            .map_err(|error| error.to_string())?;
        // Text that named no unit is degrees, this binding's default; the length binding's mirror.
        if value.dimension != parametric::Dimension::ANGLE && !expression.names_no_unit() {
            return Err(format!(
                "`{text}` is not an angle; type degrees or radians, like `45\u{b0}` or `1 rad`"
            ));
        }
        let angle = AngleMeasurement::new(value.value);
        let degrees = angle.to_degrees_f64();
        if !(0.0..=LARGEST_STATABLE_DEGREES).contains(&degrees) {
            let equivalent = degrees.rem_euclid(LARGEST_STATABLE_DEGREES);
            return Err(format!(
                "a segment has two ends, so {degrees:.2}\u{b0} and {equivalent:.2}\u{b0} are the \
                 same angle; state it between 0\u{b0} and {LARGEST_STATABLE_DEGREES:.0}\u{b0}"
            ));
        }
        Ok(Accepted {
            settled_text: Self::seed(angle),
            value: angle,
        })
    }
}

/// A [`MeasurementError`] as the sentence shown under the box.
///
/// The non-landing case names BOTH neighboring whole-voxel values, because the useful next action
/// is picking one of them.
pub fn measurement_error_text(error: &MeasurementError) -> String {
    match error {
        MeasurementError::BlockTermNotWholeVoxels {
            density,
            nearest_floor_voxels,
            nearest_ceil_voxels,
        } => format!(
            "doesn't land on a whole voxel at density {density}; nearest are {nearest_floor_voxels} or {nearest_ceil_voxels} voxels"
        ),
        MeasurementError::ZeroDensity => "density must be at least 1".to_string(),
    }
}

#[cfg(test)]
mod tests;
