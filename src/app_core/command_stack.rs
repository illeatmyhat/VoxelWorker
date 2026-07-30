//! The undo/redo stacks — workspace state, not document state (ADR 0032).
//!
//! ADR 0003 Phase C settles undo/redo as a **linear stack of inverse commands**, and
//! [`document::command`] owns the pieces that need `Scene` internals: the forward
//! [`Intent`](document::intent::Intent) and its [`Inverse`](document::command::Inverse).
//! Which stack a command sits on and what order transactions reverse in is runtime state
//! belonging to the shell's `AppCore`, so the stacks live here.
//!
//! The stacks hold [`Command`]s bare. They used to ride a `RecordedCommand` wrapper
//! carrying the pre-edit selection, but ADR 0033 removes selection from undo entirely
//! (the Fusion rule) — undo touches only the document, and a validity prune after every
//! mutation is what keeps the selection honest.

use document::command::Command;

/// One atomic undo step on the main stack — a **transaction** of one or more
/// [`Command`]s applied together and reversed together (ADR 0028 §4). A normal
/// edit is a singleton transaction; a finished sketch session is the whole batch of its
/// edits as ONE step, so a single undo past the sketch reverses all of it. `undo`
/// reverses the commands in REVERSE order (each restores its own captured counter, so
/// the batch lands on the pre-transaction state); `redo` replays them in forward order.
pub type Transaction = Vec<Command>;

/// The linear undo/redo command stack (ADR 0003 Phase C C2): two Vecs, no branching.
/// A new apply pushes to `undo` and CLEARS `redo`; `undo` moves the top transaction from
/// `undo` to `redo` (after applying its inverses); `redo` moves it back (after
/// re-dispatching its intents).
#[derive(Default)]
pub struct CommandStack {
    /// Applied transactions, newest last — the next `undo` pops the back.
    pub undo: Vec<Transaction>,
    /// Undone transactions, newest last — the next `redo` pops the back. CLEARED on a new
    /// apply (the linear-stack rule: a fresh edit invalidates the redo future).
    pub redo: Vec<Transaction>,
    /// The OPEN sketch-editing group (ADR 0028 §4), or `None` outside sketch mode.
    ///
    /// While a group is open, EVERY undoable edit routes into its own [`session_undo`] /
    /// [`session_redo`] instead of the main `undo`/`redo` — giving fine-grained IN-MODE
    /// undo/redo (reverse the last vertex move without leaving the mode) with the SAME apply
    /// door, so apply and undo can never disagree about which stack an in-mode edit lives on.
    /// **Finish** moves the whole session onto `undo` as ONE [`Transaction`]; **Cancel**
    /// reverses the session (each command by its own inverse, restoring the enter producer
    /// AND counter) and discards it. This is the ADR's "one concept, one stack, no
    /// parallel history": the session is a scoped detour on the same machinery.
    ///
    /// [`session_undo`]: SketchGroup::session_undo
    /// [`session_redo`]: SketchGroup::session_redo
    pub open_group: Option<SketchGroup>,
}

/// The transient history of ONE sketch-editing session (ADR 0028 §4) — non-document, like all
/// undo history. Opened on enter, closed by Finish (commit) / Cancel (discard). A general
/// batch of full [`Command`]s (not a producer-only collapse), so a material edit, an
/// operation switch and a vertex move mid-session are all captured and reversed uniformly. See
/// [`CommandStack::open_group`].
#[derive(Default)]
pub struct SketchGroup {
    /// The session's applied edits, oldest first, each a [`Transaction`] — in-mode `undo` pops
    /// the back; on Finish the whole session flattens into one main-stack transaction.
    ///
    /// Transactions and not bare [`Command`]s because ONE authoring act is one in-mode undo step
    /// (owner 2026-07-29): a click that both places a vertex and re-anchors the node emits two
    /// intents, and reversing half of it leaves the profile somewhere the author never put it.
    pub session_undo: Vec<Transaction>,
    /// The session's in-mode-undone edits — in-mode `redo` pops the back; cleared on a fresh
    /// edit, and discarded on Finish/Cancel.
    pub session_redo: Vec<Transaction>,
}

impl CommandStack {
    /// An empty stack.
    pub fn new() -> Self {
        Self::default()
    }
}
