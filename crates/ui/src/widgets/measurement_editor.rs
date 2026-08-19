//! The inline measurement editor: one box, opened at a place, seeded with text.
//!
//! ## What this is for
//!
//! An authored quantity should be edited where it is READ. A number painted on the drawing and a
//! field in a side rail are two places to look for one value, and the second is the one nobody
//! finds — the rail field for a sketch dimension existed for months behind an authoring step and
//! a selection, and went unused because the number the author was looking at was not the number
//! they could type into.
//!
//! ## Two customers, and only one of them is built
//!
//! The box is deliberately ignorant. It is told WHERE to sit and WHAT text to open on, and it
//! reports back the text the author left in it. It does not know what that text becomes. That is
//! not fastidiousness — it is the fence that keeps the first customer's shape out of the widget,
//! because there is a second one already named:
//!
//! 1. **Editing an existing dimension.** The anchor is the number's own screen box; committing
//!    restates the dimension through the door every other dimension edit uses.
//! 2. **Typing during placement** — NOT BUILT. The anchor tracks the cursor through a live
//!    drawing gesture, and committing LOCKS that value so the tool commits a driving dimension
//!    instead of the size the cursor happened to reach.
//!
//! The second is why [`MeasurementEdit`] carries no entity id, no constraint, and no document
//! reference: a placement box has nothing to name yet, and a widget that required one would have
//! to be rebuilt rather than reused.
//!
//! ## Measured by default, typed by choice, never demanded
//!
//! The law the placement customer must not break. A sketch tool answers the question "how big is
//! this" from the drawing itself and always has; an author who ignores the box entirely gets the
//! size they drew. Typing is the author volunteering a value, never the tool asking for one. The
//! day a box makes Enter REQUIRED to finish a gesture, this law is gone and the drawing has
//! stopped moving at the speed of thought.

/// An OPEN measurement editor: where the box sits, and what it opened on.
///
/// Absence is the closed state. Two states, one of them empty, is an [`Option`] — spelling it as
/// an enum would add a name without adding a case.
///
/// The `seed` is a STRING the opener supplies, never a number this type formats. Today a
/// dimension's opener formats the value it holds; when the document retains the author's own
/// TEXT, the opener hands that over instead and nothing here changes. That is the whole reason
/// the seed is not a `Measurement`.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementEdit {
    /// The screen box the editor sits against, in egui POINTS.
    ///
    /// For a dimension this is the number's own hit box — the same rect the click landed in, so
    /// the box opens exactly where the author was already looking. It is axis-aligned even where
    /// the number it replaces is not: a dimension's value is sheared into the sketch plane, and
    /// no text input can be, so the editor stands square to the screen and the anchor is the
    /// sheared text's bounding box.
    pub anchor: egui::Rect,
    /// The text the box opens containing, selected, ready to be replaced.
    pub seed: String,
}

impl MeasurementEdit {
    /// Open an editor at `anchor`, showing `seed`.
    #[must_use]
    pub fn new(anchor: egui::Rect, seed: impl Into<String>) -> Self {
        Self {
            anchor,
            seed: seed.into(),
        }
    }
}
