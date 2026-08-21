//! The measurement commit protocol, with no opinion about what the box looks like.
//!
//! One authored spatial quantity, edited as text. Everything about HOW that text gets on screen
//! belongs to the caller; everything about when it becomes a value belongs here.
//!
//! ## Why the protocol is a component and the chrome is not
//!
//! The protocol is subtle enough that a second hand-rolled copy drifts from the first — it did,
//! twice, before [`MeasurementField`](super::MeasurementField) was extracted. But the CHROME
//! differs genuinely between customers: a rail row is a label beside a box, and an inline editor
//! is a bare box standing at a place on the drawing. Binding the two together would have forced
//! the second customer to reimplement the first's rules to get its own appearance.
//!
//! So the caller draws, and hands back the [`egui::Response`]. Everything below is driven off
//! that response and off nothing else.
//!
//! ## The rules
//!
//! 1. **The buffer is local, not bound to the document.** In-progress text lives in egui temp
//!    memory so a partial edit survives across frames without writing anything.
//! 2. **`lost_focus()` is the single commit trigger.** It fires on Enter, on click-away, and on
//!    Tab — which is why moving to the next box commits this one, at no cost in machinery.
//! 3. **Escape abandons.** Escape also surrenders focus, so without this the abandon would arrive
//!    as a commit. [`egui::DragValue`] guards the same way for the same reason.
//! 4. **An UNFOCUSED entry with no error re-syncs to the canonical seed**, so undo, external edits
//!    and density changes reflect. An entry showing an error keeps the rejected text instead — a
//!    silent revert would discard what the author can still fix.
//! 5. **A failed commit writes nothing.** The complaint shows inline and the document is untouched.
//!
//! Rules 4 and 5 depend on each other, which is why the complaint is drawn HERE and not by the
//! caller: an entry that kept rejected text with no visible reason would read as broken.

use crate::theme;
use parametric::expression::{self, Expression, SymbolTable};
use parametric::units::{self, DisplayUnit, Measurement, MeasurementError};

/// A successful commit: the authored expression AND what it landed on.
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

/// What one frame of an entry did.
///
/// Four outcomes and not `Option<MeasurementCommit>`, because an inline editor has to tell a
/// refusal (keep the box open, the author is fixing it) from an abandon (close it) from an
/// ordinary idle frame, and a bare `None` says all three at once. A rail row that cares about
/// only one of them still has to name the rest, which is the point: a fifth outcome would break
/// every caller on the day it lands rather than being silently absorbed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeasurementEntryOutcome {
    /// No commit was attempted this frame. Mid-edit, or nothing happening at all.
    Idle,
    /// A value parsed, validated, and is ready to be written.
    Committed(MeasurementCommit),
    /// A commit was attempted and refused. The complaint is on screen and the text is still there.
    Refused,
    /// The author pressed Escape. Nothing was written and the in-progress text is discarded.
    Cancelled,
}

/// The commit protocol around a caller-drawn text box.
pub struct MeasurementEntry<'a> {
    id_base: egui::Id,
    seed: String,
    density: u32,
    min_voxels: Option<i64>,
    min_error: &'a str,
    focus_when_new: bool,
}

impl<'a> MeasurementEntry<'a> {
    /// A signed entry showing `seed`, parsed at `density`.
    ///
    /// **The seed is TEXT the caller supplies, never a number this type formats.** Today every
    /// caller hands over the canonical blocks+voxels rendering of the value the document holds;
    /// when the document retains the author's own typed text, the caller hands THAT over and
    /// nothing here changes. Deriving it internally would have made that a rewrite.
    ///
    /// `id_base` must be stable per edited value AND distinct across values — the in-progress
    /// buffer and the last error hang off it. Key it on whatever identifies the value being
    /// edited, so moving to another value re-seeds rather than inheriting half-typed text.
    #[must_use]
    pub fn new(id_base: egui::Id, seed: impl Into<String>, density: u32) -> Self {
        Self {
            id_base,
            seed: seed.into(),
            density,
            min_voxels: None,
            min_error: "",
            focus_when_new: false,
        }
    }

    /// Reject anything below `minimum` voxels, reporting `message`.
    ///
    /// For quantities that are not signed — a size of zero is not a size. Omit this for anything
    /// that may legitimately go negative (offsets, insetting outsets).
    #[must_use]
    pub fn min_voxels(mut self, minimum: i64, message: &'a str) -> Self {
        self.min_voxels = Some(minimum);
        self.min_error = message;
        self
    }

    /// Take the keyboard on the first frame this entry exists.
    ///
    /// For an entry the author OPENED — an inline editor answers a gesture that already said
    /// "I mean to change this", so making them click the box they just summoned is a step that
    /// asks nothing. A permanently-present rail row must not do this: it would steal the keyboard
    /// from whatever the author was actually typing in the moment a panel appeared.
    #[must_use]
    pub const fn focus_when_new(mut self) -> Self {
        self.focus_when_new = true;
        self
    }

    /// Run one frame. `draw` puts a box on screen bound to the live buffer and returns its
    /// response; everything else is this function's.
    pub fn run(
        self,
        ui: &mut egui::Ui,
        draw: impl FnOnce(&mut egui::Ui, &mut String) -> egui::Response,
    ) -> MeasurementEntryOutcome {
        let text_id = self.id_base.with("text");
        let error_id = self.id_base.with("error");
        let seed = self.seed.clone();

        let held = ui.memory(|memory| memory.data.get_temp::<String>(text_id));
        let opening = held.is_none();
        let mut buffer = held.unwrap_or_else(|| seed.clone());

        let response = draw(ui, &mut buffer);
        if opening && self.focus_when_new {
            response.request_focus();
        }

        // Editing again clears any stale error, so the complaint tracks the LAST committed
        // attempt rather than in-progress typing.
        if response.changed() {
            ui.memory_mut(|memory| memory.data.remove::<String>(error_id));
        }

        // Rule 3. Escape surrenders focus, so this must be read BEFORE the commit trigger or the
        // abandon arrives as a commit of whatever was half-typed.
        if response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Escape)) {
            ui.memory_mut(|memory| {
                memory.data.remove::<String>(text_id);
                memory.data.remove::<String>(error_id);
            });
            return MeasurementEntryOutcome::Cancelled;
        }

        // Rule 2. `lost_focus()` fires on Enter, on click-away AND on Tab. The typed `buffer` is
        // still live here — the unfocused re-sync below happens only on NON-commit frames, so a
        // commit always reads the author's text, never the seed. A focus loss with no actual edit
        // is a no-op.
        let mut outcome = MeasurementEntryOutcome::Idle;
        if response.lost_focus() && buffer.trim() != seed {
            match self.parse_and_validate(&buffer) {
                Ok(committed) => {
                    ui.memory_mut(|memory| memory.data.remove::<String>(error_id));
                    // Settle on the canonical form of the applied value.
                    buffer =
                        units::format(committed.voxels, self.density, DisplayUnit::BlocksAndVoxels);
                    outcome = MeasurementEntryOutcome::Committed(committed);
                }
                Err(message) => {
                    ui.memory_mut(|memory| memory.data.insert_temp(error_id, message));
                    outcome = MeasurementEntryOutcome::Refused;
                }
            }
        } else if !response.has_focus() {
            // Rule 4: mirror the canonical value, UNLESS a prior commit failed — then keep the
            // rejected text on screen beside its complaint rather than silently reverting work
            // the author can still fix.
            let has_error = ui.memory(|memory| memory.data.get_temp::<String>(error_id).is_some());
            if !has_error {
                buffer = seed.clone();
            }
        }

        // Rule 1: persist the in-progress text for the next frame.
        ui.memory_mut(|memory| memory.data.insert_temp(text_id, buffer));

        if let Some(message) = ui.memory(|memory| memory.data.get_temp::<String>(error_id)) {
            ui.colored_label(theme::WARN, message);
        }

        outcome
    }

    /// Parse `text` and check it lands on a whole voxel within the bound, or say why not.
    pub(super) fn parse_and_validate(&self, text: &str) -> Result<MeasurementCommit, String> {
        let measurement = self.authored_measurement(text)?;
        let voxels = measurement
            .to_voxels(self.density)
            .map_err(|error| measurement_error_text(&error))?;
        match self.min_voxels {
            Some(minimum) if voxels < minimum => Err(self.min_error.to_string()),
            _ => Ok(MeasurementCommit {
                measurement,
                voxels,
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
        let expression = expression::parse(text).map_err(|error| error.to_string())?;
        if let Some(measurement) = expression.as_authored_length() {
            return Ok(measurement);
        }
        let voxels = self.evaluated_voxels(&expression)?;
        Ok(Measurement::from_voxels(voxels))
    }

    /// What a compound expression is worth, at this entry's density.
    fn evaluated_voxels(&self, expression: &Expression) -> Result<i64, String> {
        SymbolTable::new()
            .evaluate(expression, self.density)
            .map_err(|error| error.to_string())?
            .to_whole_voxels()
            .map_err(|error| error.to_string())
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
