//! The blocks+voxels [`Measurement`](parametric::units::Measurement) text field.
//!
//! One authored spatial quantity as a labeled row in a panel. The commit protocol is
//! [`QuantityEntry`]'s and the meaning is [`LengthBinding`]'s; this is the chrome around the two
//! — a label, a fixed-width box, and a hint.

use super::quantity_binding::{LengthBinding, MeasurementCommit};
use super::quantity_entry::{QuantityEntry, QuantityEntryOutcome};

/// The width of the text box, in points. Every measurement field is this wide so the
/// columns line up down a panel regardless of which section drew them.
const FIELD_WIDTH_POINTS: f32 = 142.0;

/// A labeled blocks+voxels text field that commits on Enter, Tab or click-away.
///
/// ## Signed by default
///
/// A measurement is signed unless [`min_voxels`](Self::min_voxels) says otherwise. An
/// offset moves either way, and an outset that insets is a NEGATIVE outset — so the
/// bound is opt-in, and the sites that need one carry their own message.
///
/// ## It does not take the keyboard
///
/// A rail row is present because a panel is open, not because the author asked to type. Focusing
/// it on appearance would take the keyboard away from whatever they were actually doing. An
/// inline editor, which answers a gesture that already said "I mean to change this", opts in with
/// [`QuantityEntry::focus_when_new`].
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
        Self {
            id_base,
            label,
            seed_voxels,
            density,
            min_voxels: None,
            min_error: "",
        }
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
        let (label, box_id) = (self.label, Self::box_id(self.id_base));
        let binding = self.binding();
        let seed = LengthBinding::seed(self.seed_voxels, self.density);
        let drawn = QuantityEntry::new(self.id_base, seed).run(
            ui,
            |text| binding.read(text),
            |ui, buffer| {
                ui.horizontal(|ui| {
                    ui.label(format!("{label} "));
                    ui.add(
                        egui::TextEdit::singleline(buffer)
                            .id(box_id)
                            .desired_width(FIELD_WIDTH_POINTS)
                            .hint_text("blocks + voxels"),
                    )
                })
                .inner
            },
        );
        match drawn {
            QuantityEntryOutcome::Committed(commit) => Some(commit),
            // A rail row has nowhere to go on any of these. It stays where it is, showing
            // whatever the protocol left in it.
            QuantityEntryOutcome::Idle
            | QuantityEntryOutcome::Refused
            | QuantityEntryOutcome::Cancelled => None,
        }
    }

    /// The text box's own egui id, derived from the field's.
    ///
    /// Stable rather than auto-generated, so the box can be addressed — focused, or driven by a
    /// test — without depending on where in a layout it happened to be built.
    #[must_use]
    pub fn box_id(id_base: egui::Id) -> egui::Id {
        id_base.with("box")
    }

    /// This field's binding, with the bound applied when it has one.
    ///
    /// The canonical SEED — what the document currently says, as a blocks+voxels string — is
    /// struck at the call site above, because the FIELD is what knows the value as a voxel count.
    fn binding(&self) -> LengthBinding<'a> {
        let binding = LengthBinding::new(self.density);
        match self.min_voxels {
            Some(minimum) => binding.floor(minimum, self.min_error),
            None => binding,
        }
    }
}

#[cfg(test)]
mod tests;
