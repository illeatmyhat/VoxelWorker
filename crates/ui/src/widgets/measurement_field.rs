//! The blocks+voxels [`Measurement`] text field.
//!
//! One authored spatial quantity, edited as text. This is the single owner of the
//! commit protocol every measurement editor in the app shares — see [`MeasurementField`].

use crate::theme;
use voxel_core::units::{self, DisplayUnit, Measurement, MeasurementError};

/// The width of the text box, in points. Every measurement field is this wide so the
/// columns line up down a panel regardless of which section drew them.
const FIELD_WIDTH_POINTS: f32 = 142.0;

/// A successful commit: the authored expression AND what it landed on.
///
/// Both halves matter and neither is derivable from the other at the call site — the
/// `measurement` is RETAINED on the document (lossless density re-targeting and
/// exact-expression undo, ADR 0003 §3f(0)), while `voxels` is the canonical value the
/// resolve actually uses. See [`crate::widgets::measurement_field`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeasurementCommit {
    /// The authored expression, to retain on the document.
    pub measurement: Measurement,
    /// The canonical voxel value it lands on at the current density.
    pub voxels: i64,
}

/// A labelled blocks+voxels text field that commits on Enter or click-away.
///
/// ## Why this is a component and not a `fn(&mut Ui, …)`
///
/// The commit protocol is subtle enough that a second hand-rolled copy drifts from the
/// first. It has four rules that all have to hold together:
///
/// 1. **The buffer is local, not bound to the document.** In-progress text lives in egui
///    temp memory so a partial edit survives across frames without writing anything.
/// 2. **`lost_focus()` is the single commit trigger** — it fires on Enter AND on
///    click-away, so there is exactly one path a value can take into the document.
/// 3. **An UNFOCUSED field with no error re-syncs to the canonical seed**, so undo,
///    external edits and density changes reflect in the field. A field showing an error
///    keeps the user's rejected text instead, so they can see and fix it — a silent
///    revert would discard what they typed.
/// 4. **A failed commit writes nothing.** Parse and validation errors are shown inline
///    and the document is untouched.
///
/// ## Signed by default
///
/// A measurement is signed unless [`min_voxels`](Self::min_voxels) says otherwise. An
/// offset moves either way, and an outset that insets is a NEGATIVE outset — so the
/// bound is opt-in, and the sites that need one carry their own message.
pub struct MeasurementField<'a> {
    id_base: egui::Id,
    label: &'a str,
    seed_voxels: i64,
    density: u32,
    min_voxels: Option<i64>,
    min_error: &'a str,
}

impl<'a> MeasurementField<'a> {
    /// A signed field seeded from `seed_voxels`, displayed at `density`.
    ///
    /// `id_base` must be stable per edited value AND distinct across values — the
    /// in-progress buffer and the last error hang off it. Key it on whatever identifies
    /// the value being edited (typically the node and the axis), so re-selecting a node
    /// re-seeds rather than inheriting the previous node's half-typed text.
    pub fn new(id_base: egui::Id, label: &'a str, seed_voxels: i64, density: u32) -> Self {
        Self { id_base, label, seed_voxels, density, min_voxels: None, min_error: "" }
    }

    /// Reject anything below `minimum` voxels, reporting `message`.
    ///
    /// For quantities that are not signed — a size of zero is not a size. Omit this for
    /// anything that may legitimately go negative (offsets, insetting outsets).
    pub fn min_voxels(mut self, minimum: i64, message: &'a str) -> Self {
        self.min_voxels = Some(minimum);
        self.min_error = message;
        self
    }

    /// Draw the field, returning the commit when this frame produced one.
    ///
    /// Returns `None` on every frame that is not a successful commit — including frames
    /// where the user is mid-edit and frames where a commit was REJECTED. The caller
    /// therefore only ever sees values that parsed and validated, and writing the
    /// document on `Some` is always correct.
    pub fn show(self, ui: &mut egui::Ui) -> Option<MeasurementCommit> {
        let text_id = self.id_base.with("text");
        let error_id = self.id_base.with("error");
        // The canonical seed: what the document currently says, as a blocks+voxels string.
        let seed = units::format(self.seed_voxels, self.density, DisplayUnit::BlocksAndVoxels);

        let mut buffer = ui
            .memory(|memory| memory.data.get_temp::<String>(text_id))
            .unwrap_or_else(|| seed.clone());

        let widget = egui::TextEdit::singleline(&mut buffer)
            .desired_width(FIELD_WIDTH_POINTS)
            .hint_text("blocks + voxels");
        let edit_response = ui
            .horizontal(|ui| {
                ui.label(format!("{} ", self.label));
                ui.add(widget)
            })
            .inner;

        // Editing again clears any stale error, so the red message tracks the LAST
        // committed attempt rather than in-progress typing.
        if edit_response.changed() {
            ui.memory_mut(|memory| memory.data.remove::<String>(error_id));
        }

        // Rule 2: `lost_focus()` fires on Enter AND click-away. The typed `buffer` is
        // still live here — the unfocused re-sync below happens only on NON-commit
        // frames, so a commit always reads the user's text, never the seed. A focus loss
        // with no actual edit is a no-op.
        let mut commit = None;
        if edit_response.lost_focus() && buffer.trim() != seed {
            match self.parse_and_validate(&buffer) {
                Ok(committed) => {
                    ui.memory_mut(|memory| memory.data.remove::<String>(error_id));
                    // Settle the field on the canonical form of the applied value.
                    buffer = units::format(
                        committed.voxels,
                        self.density,
                        DisplayUnit::BlocksAndVoxels,
                    );
                    commit = Some(committed);
                }
                Err(message) => {
                    ui.memory_mut(|memory| memory.data.insert_temp(error_id, message));
                }
            }
        } else if !edit_response.has_focus() {
            // Rule 3: mirror the canonical value, UNLESS a prior commit failed — then
            // keep the rejected text on screen beside its error rather than silently
            // reverting work the user can still fix.
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

        commit
    }

    /// Parse `text` and check it lands on a whole voxel within the bound, or say why not.
    fn parse_and_validate(&self, text: &str) -> Result<MeasurementCommit, String> {
        let measurement = units::parse(text).map_err(|error| error.to_string())?;
        let voxels = measurement
            .to_voxels(self.density)
            .map_err(|error| measurement_error_text(&error))?;
        match self.min_voxels {
            Some(minimum) if voxels < minimum => Err(self.min_error.to_string()),
            _ => Ok(MeasurementCommit { measurement, voxels }),
        }
    }
}

/// A [`MeasurementError`] as the sentence shown under the field.
///
/// The non-landing case names BOTH neighbouring whole-voxel values, because the useful
/// next action is picking one of them.
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
