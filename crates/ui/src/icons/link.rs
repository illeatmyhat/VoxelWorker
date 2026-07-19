//! `link` — a definition and an instance of it, tied.
//!
//! The asset drawer places LINKED instances, not copies: edit the definition and every
//! instance follows. That is the whole reason make-unique has to be a deliberate act, so
//! the mark has to show a reference relationship rather than a duplicate.
//!
//! The definition is solid and the instance dashed — the set's established convention
//! that dashed means "the referenced thing", the same sense it carries on `array` and on
//! every operand mark. The tie between them is what separates this from `array`: two
//! bodies in a reference relationship, not three repeats of one.

use super::IconPainter;

pub(super) fn draw(g: &IconPainter) {
    // The definition: the body that is actually authored.
    g.rect((1.5, 6.5), (7.0, 11.5));
    // An instance of it — a reference, carrying no body of its own.
    g.dashed_rect((11.0, 6.5), (16.5, 11.5));
    // The link itself.
    g.line(&[(7.0, 9.0), (11.0, 9.0)]);
}
