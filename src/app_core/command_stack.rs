//! The undo/redo stacks — workspace state, not document state (ADR 0032).
//!
//! ADR 0003 Phase C settles undo/redo as a **linear stack of inverse commands**, and
//! [`document::command`] still owns the two pieces that need `Scene` internals: the
//! forward [`Intent`](document::intent::Intent) and its [`Inverse`](document::command::Inverse).
//! Everything around them — which stack a command sits on, what order transactions
//! reverse in, what the selection was before the edit — is runtime state belonging to
//! the shell's `AppCore`, so it lives here.
//!
//! The selection capture is the reason this module exists. ADR 0032 moves selection out
//! of the document, but undo must still restore it, so the capture rides the SHELL's
//! [`RecordedCommand`] wrapper rather than the document's `Command`.

use document::command::Command;
use document::scene::NodeId;

/// One applied [`Command`] plus the workspace state to restore when it is reversed
/// (ADR 0032). The document half (`intent` + `inverse` + `counter_before`) reverses the
/// scene; the fields here reverse the SELECTION, which no longer lives in the document.
pub struct RecordedCommand {
    /// The document mutation and its captured reverse.
    pub command: Command,
    /// The node selection BEFORE the forward op (restored on undo).
    pub selection_before: Option<NodeId>,
    /// The point selection BEFORE the forward op (restored on undo).
    pub point_selection_before: Option<usize>,
}

/// One atomic undo step on the main stack — a **transaction** of one or more
/// [`RecordedCommand`]s applied together and reversed together (ADR 0028 §4). A normal
/// edit is a singleton transaction; a finished sketch session is the whole batch of its
/// edits as ONE step, so a single undo past the sketch reverses all of it. `undo`
/// reverses the commands in REVERSE order (each restores its own captured
/// selection/counter, so the batch lands on the pre-transaction state); `redo` replays
/// them in forward order.
pub type Transaction = Vec<RecordedCommand>;

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
    /// reverses the session (each command by its own inverse, restoring the enter producer,
    /// selection AND counter) and discards it. This is the ADR's "one concept, one stack, no
    /// parallel history": the session is a scoped detour on the same machinery.
    ///
    /// [`session_undo`]: SketchGroup::session_undo
    /// [`session_redo`]: SketchGroup::session_redo
    pub open_group: Option<SketchGroup>,
}

/// The transient history of ONE sketch-editing session (ADR 0028 §4) — non-document, like all
/// undo history. Opened on enter, closed by Finish (commit) / Cancel (discard). A general
/// batch of full [`RecordedCommand`]s (not a producer-only collapse), so a material edit, an
/// operation switch and a vertex move mid-session are all captured and reversed uniformly. See
/// [`CommandStack::open_group`].
#[derive(Default)]
pub struct SketchGroup {
    /// The session's applied edits, oldest first — in-mode `undo` pops the back; on Finish the
    /// whole `Vec` becomes one main-stack [`Transaction`].
    pub session_undo: Vec<RecordedCommand>,
    /// The session's in-mode-undone edits — in-mode `redo` pops the back; cleared on a fresh
    /// edit, and discarded on Finish/Cancel.
    pub session_redo: Vec<RecordedCommand>,
}

impl CommandStack {
    /// An empty stack.
    pub fn new() -> Self {
        Self::default()
    }
}
