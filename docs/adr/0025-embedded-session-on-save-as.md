# ADR 0025 — Embedded session: the author's view, opt in on Save As

- **Status:** Accepted (2026-07-20 — owner ruling; **implementation not started**).
  **Supersedes [ADR 0022](0022-document-dump-and-state-classification.md) decision 2** — the
  rollback cursor may now travel inside the document, when the author asks for it. Extends
  [ADR 0024](0024-session-state.md)'s **session** category with a second, opt-in destination,
  and closes ADR 0024's Open item on the rollback cursor's category. Rests on the evidence in
  [`docs/design/session-storage-prior-art.md`](../design/session-storage-prior-art.md), whose
  findings are cited here rather than restated.
- **Date:** 2026-07-20
- **Layer:** document/shell boundary — what persists, into which artifact, and who chooses.

## Context

Two things settled since ADR 0024, and together they reopen a decision that ADR 0024 had
just finished defending.

**A project is a single file, not a project directory.** The owner ruled this on reading the
prior-art report, and the report records the ruling and its consequences in full: no sidecar
is available, so session state that stays outside the document lives in a keyed app-data
store; the LRU is therefore mandatory rather than optional (cap 50, JetBrains-style
insertion-order eviction that skips open documents); the key is a generated id with the
path→id map held in the app-data index and never in the shared file; copies collide and
renames survive, deliberately. **All of that is settled context for this record, not a
decision it makes.**

**The owner then ruled that session state may be embedded in the document after all** — as an
explicit, author-chosen opt-in on Save As, sticky thereafter. That is the decision recorded
here, and it reverses ADR 0022 decision 2, which said the rollback cursor "goes in the dump
and stays out of the document" and called that a deliberate divergence from Fusion 360. The
owner restated that position as recently as ADR 0024 decision 3, which cited the cursor as
the companion case to the viewer mode. **That history is part of the record: this is a
reversal of a decision made twice, not a gap being filled.**

What changed is not the argument against embedding — the prior-art report strengthens it, and
the strongest single artifact in it is Vintage Story's `collapsedPaths`, one author's outliner
fold state committed to shared mod assets across 1,420 public files, in the very format this
project reads. What changed is that the report also found the case *for* embedding, made by
someone with no stake here: Sybren Stüvel, refusing to strip viewport state from `.blend`
files, because "when reporting a bug it helps that the viewport is showing the buggy area of
the model." Handover has a real requirement. A file that arrives saying *look at it like this*
is doing something a file plus a verbal instruction does not.

The reason ADR 0022 could reject that requirement outright was that the dump served it. The
dump still does — for faults, between builds of the same version, on this machine. It does not
serve the case where the recipient is a person, the artifact is the project itself, and the
message is not a bug report but "here is where I got to." Nothing in the pipeline addressed
that, because until the document shape was settled there was no document to put it in.

## Decisions

### 1. Session may be embedded in the document, chosen by the author on Save As

The Save As dialog carries a checkbox. Off by default; whatever the author chooses is written
into the document alongside the state, and **subsequent ordinary saves keep that choice
without asking again**. A project that was saved with the view embedded keeps embedding it on
every `Ctrl+S` until the author says otherwise on a later Save As.

Stickiness is the whole point and is not a convenience detail — see decision 4. A choice that
must be re-made on every save is a choice that gets made wrongly, which is Blender's shipped
outcome from the other side of the same fence.

### 2. The checkbox covers `view_mode` and the rollback cursor, and nothing else

ADR 0024 named four session fields. Two are eligible for embedding and two are not, and the
line between them is not "which are small":

| Field | Keyed store | Embeddable |
| --- | --- | --- |
| `view_mode` | yes | **yes** |
| rollback cursor (per scope) | yes | **yes** |
| `stack` — the panel fold state | yes | no |
| `debug_face_orientation` | yes | no |
| `debug_brick_faces` | yes | no |

**The test is whether the field describes the model or the workspace.** `view_mode` and the
rollback cursor both answer *what state was this model in when you looked at it* — rolled back
to the third operation, seen in Onion. That is a statement about the design, and it is exactly
the statement a handover wants to carry. `stack` and the two debug flags answer *how was my
tooling arranged* — which panels I had folded, which renderer diagnostic I had on. A reader
did not ask for either.

The two debug flags are the clearer half. Embedding them ships a debugging state to somebody
who did not ask to debug anything, and the dump already carries them for the one reader who
does (ADR 0024 decision 4). There is no case for a second route.

**`stack` is the judgement call, and it stays out.** It is the field the prior-art report found
embedded *badly*: Vintage Story's `editor.collapsedPaths` is `SignalStackState` under another
name, sitting in shared, version-controlled assets in 1,420 public repositories. That is not an
analogy — it is the same field, in the same ecosystem, with the cost visible in the open. It
also fails the model-versus-workspace test on its own terms: which panels a reader has folded
is a property of their screen, and inheriting a stranger's fold state is the "you inherit a
stranger's layout" complaint (Blender [#100155](https://projects.blender.org/blender/blender/issues/100155))
in miniature. The keyed store already restores it per document, which is where it belongs.

The residual argument for `stack` — that a handover might want to point at a specific node by
leaving its ancestors expanded — is real but weak: pointing at a node is a *selection*, and if
that need becomes pressing the answer is to carry a selection, not to carry a hundred fold
bits and hope one of them lands.

### 3. Law A — embedded session is ADVISORY, never load-bearing

**This is the condition that makes the whole feature safe, and it outranks every other
statement in this record.**

If the embedded block is absent, fails to parse, is of an unrecognised shape, or names a viewer
mode, composition scope or node that no longer exists, **it is dropped silently and the document
opens with defaults.** No error, no warning, no repair prompt, no migration. A document with an
unreadable session block is not a damaged document; it is a document whose advice was declined.

The reasoning is the one cost the prior-art report found that genuinely transfers from Blender,
and it is a schema cost, not a merge cost:

> Embedded UI state is a *versioned schema* that must survive every UI refactor. When it does
> not, the document appears broken rather than the UI appearing stale.

This repository has a standing law that old configs may break. That law is affordable precisely
because a broken config resets a preference. The same law applied to a *document* would convert
a tolerated reset into a corrupted-project report from a user, about a file containing weeks of
work, caused by a UI refactor that touched no geometry. Advisory-only is what keeps that from
being possible: **no UI refactor can ever make a document fail to open**, because there is no UI
state the loader is entitled to require.

**The rollback cursor has more ways to dangle than `view_mode` does**, and this must be said
sharply because it is the sole substantive risk the reversal introduces. A viewer mode is a
small closed enumeration; an unrecognised value has one obvious floor. The cursor names a
*position in the fold*, per composition scope — a scope that may have been deleted, and a
position that may be past the end of a fold that has since been shortened, or that no longer
means what it meant when a node was reordered. **When a cursor entry cannot be resolved, it
drops to the end of its scope's fold — that is, no rollback for that scope.** Per scope, not
per document: a document whose part scopes resolve and whose root scope does not opens with the
parts rolled as advised and the root complete.

Dropping to the end is chosen because it is the *safe* failure in the exact sense ADR 0022
decision 2 argued for: the reader sees the complete model. The failure mode of this feature is
therefore the behaviour ADR 0022 mandated unconditionally. That is worth stating plainly,
because it is what makes the reversal small rather than a change of philosophy: **the old
decision is now this feature's fallback.**

A consequence follows for the writer: the embedded cursor must be *validated on load rather than
trusted*. An implementation that resolves the cursor optimistically and panics on a stale
reference has violated this law even if the file parses.

**And that constrains how the cursor names its position.** A first draft of this ADR worried
about "stable node identity"; the code does not have that problem. `NodeId` is a plain `u64`
minted from a monotonic counter with **no slotmap generations** — the counter alone prevents
stale-id aliasing (`crates/document/src/scene/graph.rs:37`) — the arena is a `BTreeMap` keyed by
id that serializes with the document, and `ensure_node_ids` preserves loaded ids verbatim,
minting only for the `NodeId(0)` sentinel (`graph.rs:586`). An id written into the same document
it refers to survives the round trip intact.

The real exposure is **referential integrity, not identity**: the referent can be deleted while
the id stays perfectly valid. And because decision 2 keeps the cursor **per scope**, both halves
can dangle — the scope node can be removed, and so can the node the cursor sits at inside it.

That decides the representation, and it decides it against the cheaper-looking option:

* **An index into the spine stays plausible when it is wrong.** Delete a node above the cursor
  and index 5 still resolves — to a different node. The failure is silent and undetectable.
* **A `NodeId` fails detectably.** It is either present in the arena, under the named scope, or
  it is not.

**Law A is only enforceable if a stale reference is detectable, so the embedded cursor names the
`NodeId` it is rolled back to** — resolved by locating that id in the scope's spine, and dropping
to the end of the fold when the lookup misses. An index would satisfy the letter of "advisory"
while making the advisory check impossible to write.

### 4. Law B — the choice belongs on the writer, not the reader

Blender puts it on the reader. `Load UI` is a checkbox in the file browser, it defaults to on
(`USER_FILENOUI` is absent from the shipped flag set), and the prior-art report documents what
that costs:

- **It is not sticky.** Users report disabling it on every single open; it is ignored entirely
  on drag-and-drop.
- **The escape hatch is itself a crash source** —
  [#126392](https://projects.blender.org/blender/blender/issues/126392), double-clicking a file
  crashes when Load UI is off, and
  [#128012](https://projects.blender.org/blender/blender/issues/128012) belongs to the same
  family.
- **It does not even save the read.** There is no parse-time skip; the UI datablocks are read
  and then discarded.

The structural error underneath all three is that the reader is asked a question they cannot
answer. They have not opened the file yet. They do not know whether its author left anything
worth adopting, so the checkbox is answered by habit, and a checkbox answered by habit is an
option that only ever costs.

**The author can answer it.** They know whether they are saving a working file or handing
something over. So the flag is written once, by the writer, and recorded in the file; the reader
makes no decision at all, and there is no reader-side toggle to be sticky, to be forgotten, or
to crash. Sybren Stüvel's defence of in-file viewport state — the one primary studio voice the
report found on the subject, and it argues *for* embedding — is fully satisfied by a writer-side
flag, because his case is a handover: an author deliberately sending a view along with a model.

### 5. The local session wins after the first open; the embedded block seeds it

Both destinations can hold a session for the same project, so precedence must be stated rather
than discovered.

> **The embedded block is used if and only if there is no local session entry for this
> document. It seeds that entry. From then on, the local session wins, and the embedded block
> is inert until the document is saved again with the checkbox on — which writes the current
> local session into it.**

The reasoning is that the two destinations answer different questions and the answer that
should win changes exactly once. On a document you have never opened on this machine, there is
no *your* view to restore, and the author's is both the best available answer and the one they
meant you to see. On every open after that, restoring the author's view instead of yours would
throw away what you were last doing — the same loss ADR 0024 called "losing work, in the small,"
now with a stranger's preferences as the cause.

Stating the rule as *absence of a local entry* rather than as *the first open ever* is
deliberate, and it is the one place this record tightens the derivation it was given. It is
clock-free and needs no "have I seen this file before" flag anywhere: the store's own state is
the whole condition. It also makes LRU eviction (settled context, cap 50) benign — a document
evicted after fifty others and then reopened simply re-seeds from the author's view, which is a
better answer than defaults and is the only answer still available.

**The known cost, named rather than left to be found:** a *revised* copy of a document you have
already opened does not re-seed. If a colleague sends v2 with the cursor moved to make a point,
you keep your own view. The alternatives all fail worse — a modification-time comparison makes
the rule clock-dependent and lets an unrelated re-save silently move a reader's cursor, and a
prompt puts the decision back on the reader, which is exactly what decision 4 forbids. Since
the embedded state is advisory in both directions, the recovery is that the reader moves the
cursor themselves, which costs one gesture.

## Alternatives considered

- **Keep ADR 0022 decision 2 unchanged and serve handover with the dump.** This is the standing
  position and it was reversed on evidence, not preference. A dump is not versioned, is read by
  the build that wrote it, and carries every setting and every diagnostic — it is a bug report,
  and sending one to a collaborator who wanted a design is the wrong artifact with the wrong
  contents.
- **Embed unconditionally, as Blender and Vintage Story do.** Rejected on the report's own
  evidence: `collapsedPaths` in 1,420 shared files is what unconditional embedding looks like at
  rest. It also converts every UI refactor into a document-compatibility event, which Law A
  exists to prevent.
- **Put the choice on the reader (`Load UI`).** Rejected in decision 4, with three shipped
  failure modes cited.
- **A per-field checklist in the Save As dialog** — let the author pick which session fields
  travel. Rejected as a dialog that asks a question nobody has an opinion about. Decision 2's
  cut is a property of the fields, not of the occasion; an author choosing per-save would be
  guessing at an answer this record already has.
- **Make the embedded cursor load-bearing, with a repair prompt when it dangles.** Rejected as
  the whole hazard in one feature. It makes a UI-shaped schema a precondition for opening a
  document, and it puts a modal in front of a user whose only crime was reordering two nodes.
- **Embed `stack` as well.** Rejected in decision 2, on the strongest artifact in the report.

## Consequences

- **ADR 0022 decision 2's Status is amended to point here.** Its per-scope requirement and its
  suppression-versus-disabling separation (decision 3 there) are untouched; only its "stays out
  of the document" half is superseded. The side-map storage it prescribes stays right for the
  local session, and the embedded block is written *from* that map rather than replacing it.

- **The document format grows a versioned optional block, and the version is allowed to be
  ignored.** This is the first thing in the document whose schema may be broken by a UI change,
  and Law A is what makes that acceptable: the block declares its shape, and a loader that does
  not recognise the shape drops it. The document's own versioning promise (ADR 0022 decision 1)
  is unaffected, because nothing in the block is document data.

- **The sticky flag is itself document state, not session state.** "This project embeds its
  session" is a property of the file that the author chose and that must survive being sent to
  somebody else and saved by them. It is not "how the workspace was left." Classifying it
  `session` would make it the one session field that cannot be dropped, which contradicts Law A.

- **ADR 0024's Open item on the rollback cursor's category is closed.** The cursor is
  **session**, as that item suspected, and it is one of the two fields the checkbox covers.
  Being session state is what makes it eligible for the local store *and* for the embedded
  block; the two destinations are a property of this decision, not of the category.

- **`stack` and the two debug flags now have exactly one destination each, and that is a
  narrowing worth review.** ADR 0024 routed all four fields identically. This record splits
  them, so a future field classified `session` must answer a second question that did not exist
  before: model or workspace. That question should be answered at the field, in the same place
  and the same review-visible way the category is.

- **The keyed store's design is unchanged and remains mandatory.** Nothing here reduces the need
  for it — it is still the only home for `stack` and the diagnostics, still the only home for
  any session at all in a document saved with the checkbox off, and still the destination the
  embedded block seeds *into*. The LRU is not made optional by this decision; if anything the
  seeding rule depends on eviction being survivable, which it is.

## Open

- ~~Whether a handover wants to carry a selection.~~ **Settled by the owner, 2026-07-20: no.**
  And the question turned out to be already answered by the code — `Scene.active` is an
  `Option<NodeId>` carried with `#[serde(default)]` and no skip, so **selection is serialized
  into the document today**, unconditionally and outside this checkbox.

  That is not an oversight to correct. It is the field's boundary: the prior-art report found
  Krita storing `selected="true"` per layer in `.kra`, Photoshop's image resource **1024** being
  "the index of target layer", and GIMP's `PROP_ACTIVE_LAYER` in the XCF — while none of the
  three stores zoom, pan or rotation in the document at all. **Three independent raster editors
  put selection inside the document and the camera outside it.** Selection is therefore
  *document* state, not session state, which is exactly why it needs neither the checkbox nor a
  handover mechanism.

  Worth noting for the implementation: `active`'s own contract is that a stale id "simply
  resolves to `None`". That is Law A, already shipped, for the same class of dangling `NodeId`
  reference the embedded cursor will carry — so the cursor's fallback has a working precedent in
  the codebase rather than being novel.
- ~~What the Save As checkbox is called.~~ **Settled by the owner, 2026-07-20: "Save viewer
  state".** It names what is saved in the vocabulary the product already uses for viewer modes,
  rather than naming the mechanism. Note that it deliberately does not say *whose* view or
  promise the reader anything — Law A means the advice may be dropped, so a name that promised
  the reader a view would overclaim.
- ~~Whether the sticky flag is visible anywhere outside Save As.~~ **Settled by the owner,
  2026-07-20: it appears in a Document Properties window.** That is the right home precisely
  because the flag is document state and not session state (see Consequences) — it is a property
  of the file, so it belongs with the file's other properties rather than in viewport chrome.
  The window does not exist yet; this records where the control lands when it does.
- **The static / thread-local / GPU audit** owed since ADR 0022 remains open. Nothing here
  narrows it.
