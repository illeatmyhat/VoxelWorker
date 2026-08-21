//! The commit protocol for one authored quantity edited as text.
//!
//! **This half is dimension-free.** It owns the text, the focus, and the moment a commit is
//! attempted; it does not know whether the thing being typed is a length, an angle, or something
//! not invented yet. What the text MEANS is a binding's question — see
//! [`quantity_binding`](super::quantity_binding).
//!
//! That fence is load-bearing rather than tidy. The protocol was extracted with a density and a
//! voxel floor sitting inside it, and those are length words: the day an angle needed the same
//! protocol, every consumer of the shared type would have had to learn the dimension list to get
//! at it. A validator handed IN keeps the list at one place per dimension, and adding a third
//! dimension touches no existing binding.
//!
//! ## Why the protocol is a component and the chrome is not
//!
//! The protocol is subtle enough that a second hand-rolled copy drifts from the first — it did,
//! twice, before it was extracted. But the CHROME differs genuinely between customers: a rail row
//! is a label beside a box, and an inline editor is a bare box standing at a place on the drawing.
//! Binding the two together would have forced the second customer to reimplement the first's
//! rules to get its own appearance.
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
//! 6. **An untouched seed commits nothing.** A box the author opened and left alone is a no-op,
//!    which is also why every binding must be able to READ the seed it hands out — see the
//!    round-trip tests in [`quantity_binding`](super::quantity_binding).
//!
//! Rules 4 and 5 depend on each other, which is why the complaint is drawn HERE and not by the
//! caller: an entry that kept rejected text with no visible reason would read as broken.

use crate::theme;

/// Text a binding accepted, and what the box should settle to showing.
///
/// The settled text is the binding's because canonicalisation is dimension knowledge: a length
/// settles to blocks-and-voxels, an angle settles to degrees, and the protocol knows neither
/// spelling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Accepted<T> {
    /// The value to write.
    pub value: T,
    /// The canonical rendering of that value, which the box adopts.
    pub settled_text: String,
}

/// What one frame of an entry did.
///
/// Four outcomes and not `Option<T>`, because an inline editor has to tell a refusal (keep the
/// box open, the author is fixing it) from an abandon (close it) from an ordinary idle frame, and
/// a bare `None` says all three at once. A rail row that cares about only one of them still has
/// to name the rest, which is the point: a fifth outcome would break every caller on the day it
/// lands rather than being silently absorbed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantityEntryOutcome<T> {
    /// No commit was attempted this frame. Mid-edit, or nothing happening at all.
    Idle,
    /// A value parsed, validated, and is ready to be written.
    Committed(T),
    /// A commit was attempted and refused. The complaint is on screen and the text is still there.
    ///
    /// The REFUSAL CHANNEL: a binding says no, and the protocol leaves the field exactly as rules
    /// 4 and 5 require without ever learning why the answer was no.
    Refused,
    /// The author pressed Escape. Nothing was written and the in-progress text is discarded.
    Cancelled,
}

/// The commit protocol around a caller-drawn text box.
pub struct QuantityEntry {
    id_base: egui::Id,
    seed: String,
    focus_when_new: bool,
}

impl QuantityEntry {
    /// An entry showing `seed`.
    ///
    /// **The seed is TEXT the caller supplies, never a number this type formats.** Today a caller
    /// hands over the canonical rendering of the value the document holds; when the document
    /// retains the author's own typed text, the caller hands THAT over and nothing here changes.
    /// Deriving it internally would have made that a rewrite.
    ///
    /// `id_base` must be stable per edited value AND distinct across values — the in-progress
    /// buffer and the last error hang off it. Key it on whatever identifies the value being
    /// edited, so moving to another value re-seeds rather than inheriting half-typed text.
    #[must_use]
    pub fn new(id_base: egui::Id, seed: impl Into<String>) -> Self {
        Self {
            id_base,
            seed: seed.into(),
            focus_when_new: false,
        }
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

    /// Run one frame.
    ///
    /// `read` is the BINDING: it turns the author's text into a value, or into the sentence
    /// explaining why not. `draw` puts a box on screen bound to the live buffer and returns its
    /// response. Everything between the two is this function's.
    pub fn run<T>(
        self,
        ui: &mut egui::Ui,
        read: impl FnOnce(&str) -> Result<Accepted<T>, String>,
        draw: impl FnOnce(&mut egui::Ui, &mut String) -> egui::Response,
    ) -> QuantityEntryOutcome<T> {
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
            return QuantityEntryOutcome::Cancelled;
        }

        // Rule 2, and rule 6 in the `!=`. `lost_focus()` fires on Enter, on click-away AND on Tab,
        // and the seed comparison is what keeps a box the author opened and left alone from
        // writing the value it was already showing. The typed `buffer` is still live here: the
        // unfocused re-sync below happens only on NON-commit frames, so a commit always reads the
        // author's text, never the seed.
        let mut outcome = QuantityEntryOutcome::Idle;
        if response.lost_focus() && buffer.trim() != seed {
            match read(&buffer) {
                Ok(accepted) => {
                    ui.memory_mut(|memory| memory.data.remove::<String>(error_id));
                    buffer = accepted.settled_text;
                    outcome = QuantityEntryOutcome::Committed(accepted.value);
                }
                Err(message) => {
                    ui.memory_mut(|memory| memory.data.insert_temp(error_id, message));
                    outcome = QuantityEntryOutcome::Refused;
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
}

#[cfg(test)]
mod tests;
