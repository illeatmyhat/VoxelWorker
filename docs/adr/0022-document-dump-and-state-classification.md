# ADR 0022 — The document, the dump, and classified state

- **Status:** Accepted (2026-07-20 — design grill; **implementation not started**).
  **Decision 2 is superseded in part by [ADR 0025](0025-embedded-session-on-save-as.md)**: the
  rollback cursor is **session** state (ADR 0024's category) and *may* travel inside the
  document, as an author-chosen opt-in on Save As. Decision 2's per-scope requirement and its
  side-map storage stand; only "stays out of the document" is reversed, and it survives as ADR
  0025's fallback — an embedded cursor that cannot be resolved drops to the end of its fold, so
  a document that cannot honour the advice opens complete, exactly as decision 2 required.
  Introduces the rollback cursor as classified view state, and the classification scheme that
  places it.
  Relates to ADR 0017 (the ordered fold this rolls back), ADR 0016 (crate structure — the
  derive would be a new crate kind), and ADR 0018 (viewer modes, the existing precedent for
  display state that never enters the document).
- **Date:** 2026-07-20
- **Layer:** document/shell boundary — what persists, into which artifact, and how that is
  enforced.

## Context

Three things collided in one grill and turn out to be the same decision.

**The insert cursor needed a home.** The direct-manipulation grammar
(`docs/design/direct-manipulation.md`) says a drop "commits one `AddNode` at the insert
cursor", but `Scene::add_node` appends to `roots`. The cursor is the Fusion 360 rollback bar:
a per-scope position in the ordered fold, where new nodes land, and past which nodes do not
evaluate. `crates/ui/src/workspace/fold_strip.rs` deliberately refused to implement it,
recording in its own module doc that adding it silently would be deciding an architecture
question inside a widget. That refusal was correct.

**The flag it would have been built on was lying.** `Node.visible` decided whether a node
contributes to the *composed geometry* — `producers.rs` pruned it from the op-stack walk
before evaluation, so disabling a `Subtract` filled the hole back in. It was renamed to
`enabled` (2026-07-20). The audit that accompanied the rename found no site treating it as
display visibility, which establishes a fact this ADR depends on: **the document has no
display-only hide at all.** "Show me this cutter without its cut" is not a question the fold
can presently answer.

**And there is exactly one persistence artifact today.** `AppConfig::capture` is called by
both `save_config` (on exit) and `export_repro` (F9), and the whole `Scene` lives inside it.
So the config file, the project, and the debug repro are one thing wearing three hats. The
camera pan target was once missing from the repro — not because anyone decided it did not
matter, but because nothing forced the question to be asked.

## Decisions

### 1. The document and the dump are different artifacts with different jobs

- **The document** is the project: the thing a user saves, shares, and reopens. It carries what
  the model *is*. It is not geared toward reproducing faults, and it needs versioning because
  it travels between people and across releases.
- **The dump** is the debugging artifact: *a scene must be completely reproducible from it.*
  Every setting, every input, every piece of view state is in it. It needs no versioning — it
  is read by the version that wrote it.

The dump is therefore a superset of the document, not a variant of it. These are different
questions and conflating them is what let the pan target go missing.

### 2. The rollback cursor is view state, and per scope

The cursor position is where *you* are working, not what the model *is*. It goes in the dump
and stays out of the document. Reopening a project therefore always shows the complete model,
and a rollback left set is not what a collaborator sees.

**This is a deliberate divergence from Fusion 360**, which stores its timeline marker in the
document. The reasoning: a shared file should mean the same thing to everyone who opens it,
and a fold that silently evaluates only its first three nodes because of where someone else's
cursor was parked is a trap, not a feature.

There is one cursor **per composition scope**, not one globally. Descending into a part,
rolling back inside it, and coming back out must leave the outer scope's cursor where it was —
a rollback bar that does not stay put is not doing its job. Because it is view state, it is
stored as a side map keyed by scope, never as a field on the scope node: a field on the node
would ride along inside the `Scene` and land in the document, on the wrong side of decision 1
by construction.

### 3. Suppression and disabling stay separate concepts

A node is absent from the evaluation for either of two independent reasons, and they must not
be implemented in terms of each other:

```
skip the node when   !enabled  ||  past_the_scope's_rollback
```

`enabled` is **per-node, sticky, authored** — "I turned this off." Rollback is **positional and
derived** — "everything after here, because of where I am working." Implementing the second by
setting the first destroys the distinction: moving the cursor back would have to restore each
node's prior state from somewhere, which is the separate state again, built badly; and any node
the user disabled by hand while the cursor was parked would be silently re-enabled when it
moved.

The consequence is a UI obligation: "why is this node not in my model?" now has two answers, so
a disabled card and a rolled-past card must not look the same.

### 4. Every piece of state is classified, and the compiler enforces it

Each field of application state carries a category — **settings**, **document**, **view**, and
whatever further categories prove necessary. Each persistence artifact handles the categories
it is defined to carry. **A field that is classified but reaches neither the document nor the
dump is a compile error.**

The enforcement mechanism is **exhaustive destructuring**: an artifact's capture function
destructures the state with no `..` rest pattern, so adding a field fails to compile until
someone handles it. This is the part that delivers the guarantee. A trait or attribute alone
would not: it classifies a *type*, and says nothing about whether every *field* made the trip —
which is precisely how the pan target was lost, inside a camera that was already "captured".

Classification and completeness are two mechanisms, not one. The category records the decision
at the field, where a reader will look; the destructuring forces the decision to exist.

## Consequences

- **`AppConfig` splits.** One structure currently serves as config, project, and repro. Under
  decision 1 it becomes at least a document and a dump, with settings distinguished from both.
  This is the largest implementation cost here and it touches persistence for every feature.

- **A category that means "neither" must be explicit and rare.** Some state is genuinely
  transient (whether the mouse is currently held mid-drag). That must be a stated, justified
  category subject to the same compile-time pressure — never the default. The moment
  "unclassified" silently means "transient", the guarantee is gone. This is the most likely
  way for the scheme to rot: people route around friction, and marking a field transient will
  always be the cheapest way to make the compiler stop complaining. Whether it stays honest
  depends on review, not on the type system.

- **A derive macro means a new crate kind.** The workspace has no proc-macro crate
  (ADR 0016 cut it into layer crates, all ordinary libraries). A proc-macro crate is build-time
  and orthogonal to the layer stack, so it does not violate the downward-only flow law, but it
  is a new kind of member and should be recognised as such rather than appearing by accident.
  **Exhaustive destructuring alone needs no macro and delivers the completeness guarantee** —
  the macro buys the *classification*, which is reviewability rather than safety. Whether that
  is worth a proc-macro crate is not settled here.

- **State the derive cannot see is still invisible.** Anything living in a `static`, a
  thread-local, or on the GPU is outside any struct the scheme covers. The guarantee is
  "no field of a classified struct is forgotten", not "nothing is forgotten". An explicit audit
  for such state is owed, and the scheme's coverage should not be assumed total.

- **Deleted scopes leave dangling cursor entries.** A per-scope side map outlives the scopes it
  keys. Harmless for lookup (a missing entry reads as "no rollback"), but it grows without a
  sweep, and it grows inside the dump.

- **Rollback interacts with cost, favourably.** A rollback changes which nodes evaluate, so
  moving it is a document-shaped edit in everything but persistence. Measured per-edit costs
  (`tests/edit_cost_probe.rs`, `tests/remesh_cost_probe.rs`) show the resolve is flat in scene
  size while a wholesale re-mesh is linear in resident chunks — so a cursor drag across many
  nodes is bounded by re-meshing, not by re-resolving, and should be treated like any other
  gesture that dirties a large volume.

## Amendment 2026-07-20 — the derive macro, and classification does not recurse

Two owner rulings that close the two questions this ADR left open about the mechanism.

**The classification is a derive macro.** Not because it is safer — hand-written exhaustive
destructuring catches an unclassified field just as well — but because `#[snapshot(transient)]`
sitting on the field is **visible in review**, where a `skipped:` line inside a `classify`
function is not. This ADR named review as the only thing keeping `transient` honest; the macro
is what makes that review possible. The cost is the workspace's first proc-macro crate.

**Classification applies to whole objects and does not recurse into their fields.** The
destructuring says which category an object belongs to; the object is then saved **entire, and
recursively**, by serialization. Nested fields are not annotated.

This corrects a concern raised during the grill — that a classified `camera: OrbitCamera` says
nothing about whether every field *inside* `OrbitCamera` made the trip, so the pan-target bug
could recur one level down. It cannot, and the reasoning was wrong: serialization already
carries every field of a saved object. The historical bug was never "a field was added and not
serialized" — it was that the *state itself was not reached* by the capture. That is exactly
what top-level exhaustive destructuring prevents.

Annotating nested fields would be enormous and would buy nothing. The guarantee is: **every
object is classified, and a classified object is saved whole.**

## Amendment 2026-07-20 — what the derive actually guarantees

**Status: partially implemented.** `crates/snapshot` + `crates/snapshot_derive` exist, and
`PanelState` and `AppConfig` are classified (31 fields). The artifact split is not done.

Landing it exposed an **overstatement in Decision 4 above**, which is corrected here rather
than quietly left standing. That decision says:

> A field that is classified but reaches neither the document nor the dump is a compile error.

**A derive cannot deliver that, and this one does not.** The derive proves a field is
*classified*. Whether a classified field *reaches an artifact* is a property of the capture
function, and that guarantee comes from exhaustive destructuring against the split artifacts —
which do not exist yet. What exists now is the classification table those artifacts will be
built against (`document_fields()` / `dump_fields()`), plus the fact that producing it forced
every field to be decided.

The two halves are still the two halves this ADR always described — classification records the
decision at the field, destructuring forces the decision to exist. The error was in implying a
single mechanism delivers both. **Only the first half is built.**

Two consequences of the shape it took, neither surprising but both worth recording:

* **It is two crates, and that is forced.** A proc-macro crate can export nothing but macros,
  so the trait and category enum the generated code refers to cannot live inside it — the same
  split `serde`/`serde_derive` uses. `crates/snapshot_derive` is build-time and orthogonal to
  the layer stack (no data flows through it at all), so ADR 0016's downward-only law has
  nothing to say about it; `crates/snapshot` is a true leaf beside `substrate`, naming no
  domain type, so any layer may import it.
* **Every unclassified field reports in one build**, not one per recompile. Retrofitting onto a
  twenty-field struct one error at a time would be miserable enough to discourage classifying
  anything, and a mechanism people avoid guarantees nothing. The message is pinned by
  `trybuild` fixtures, so a regression in it fails a test — the error text is the feature's real
  interface, not incidental output.

First classification pass: **zero `transient`**, two `derived`, both meeting ADR 0023's
admission test literally. That the unfalsifiable hatch went unused on the first real struct is
the healthiest available early signal, and it is the number to watch over time.


## Amendment 2026-07-20 — the artifact split, and where the guarantee stops

**Status: the split is implemented.** `src/artifacts.rs` holds `DocumentArtifact`,
`SettingsArtifact` and `Dump`, and every capture in it destructures `AppConfig` with no
`..` rest pattern. Adding a field to that struct now fails the build with
`error[E0027]: pattern does not mention field` until somebody routes it, which is the
mechanism decision 4 named. `crates/snapshot/tests/compile_fail/capture_misses_a_classified_field.rs`
pins that error, so a refactor that reaches for `..` to get things building deletes a test
rather than deleting the guarantee silently.

Two shapes the implementation took, and one thing it revealed.

**The dump is the only file, and that is not a compromise.** F9 writes one, and exit writes
one, because restoring a session needs the scene *and* the preferences *and* the camera
pose — which is the dump's field set exactly, not the document's. The document and the
settings are real types that a dump is composed of; giving either its own path is a
save/open workflow, which is a product decision this ADR does not make. The on-disk JSON
stays flat (the three parts are merged into one object) so that every repro file already
written still replays, verified by rendering the same pre-split repro through
`shot --from-config` before and after: byte-identical PNGs.

**`#[serde(flatten)]` cannot be used for that merge.** It buffers the whole object through
serde's internal `Content` type, and the scene's id-keyed node arena does not survive the
trip. Recorded because the failure is a runtime `i128 is not supported` on load, not a
compile error, and the tests that caught it are the scene round-trips rather than anything
about persistence shape.

### What is still overstated: the guarantee holds at one seam only

Decision 4 reads as a property of classified state generally. It is now a property of
**`AppConfig`** specifically, because that is the struct the artifacts destructure.
`PanelState` is equally classified and **nothing captures it exhaustively** —
`AppConfig::capture` reads it field by field, by hand, in precisely the shape of the
capture that lost the pan target. So on that seam the compiler still checks only that every
field is *decided*, never that the decision is *honoured*.

That is not hypothetical. Four `PanelState` fields are classified `view` — which the
derive's own error text defines as reaching the dump — and reach nothing:
`debug_face_orientation`, `debug_brick_faces`, `view_mode`, `stack`. The last two were
deliberately excluded from persistence by ADR 0018 decision 3 and issue #88, so this is not
a bug to fix quietly; it is **two decisions contradicting each other**. Either those fields
are misclassified, or the dump is not capturing what its category promises.

A second, subtler version of the same gap: the amendment above says "a classified object is
saved whole". That holds where the object is what gets serialized, and `PanelState`'s are
not — `geometry` and `layer_range` are each classified as one view object and each reaches
the dump as a hand-picked subset of its fields. The band bounds are a defensible omission
(they re-derive against the live grid); the problem is that nothing distinguishes a
defensible omission from the pan-target kind. Both gaps are pinned by tests in
`tests/state_classification.rs` so they cannot widen unnoticed.

The honest statement of what is now true: **a field of `AppConfig` cannot fail to reach an
artifact. A field of `PanelState` still can, and four currently do.**

## Open

- ~~**Extending the guarantee to the `PanelState` → `AppConfig` seam**, which is where the
  reachability promise currently stops (see the amendment above). Blocked on an owner ruling:
  `view_mode` and `stack` are classified `view` but were deliberately excluded from
  persistence, and those two positions cannot both stand.~~ **Closed by
  [ADR 0024](0024-session-state.md) (2026-07-20).** The owner ruled: the four fields are
  **session** state — a third top-level category — and they now route through
  `SessionArtifact`. The seam gets a test rather than a compiler (`PanelState` is read by hand
  on purpose), and the subset carriers the amendment above called subtler are now declared by
  name in `tests/state_classification.rs` instead of being merely true.
- The **static / thread-local / GPU audit** this ADR says is owed. Nothing built so far narrows
  it.
- Whether a display-only **hidden** — distinct from `enabled` — is worth adding. In a fold
  model, "show me this cutter without its cut" may not be a meaningful request; it is currently
  not expressible at all, which is a fact rather than a decision.
- ~~The category list beyond settings / document / view, and which artifacts each reaches.~~
  Two answers so far: **derived** (ADR 0023) and **session** (ADR 0024), which also records
  the general shape — only `document` is a routing decision, every other category reaches the
  dump and exists to make the *reason* legible at the field.
- Whether the **document** ever earns a file of its own, which is the point at which it needs
  the versioning decision 1 promises it. Nothing in the split forecloses it; nothing yet
  requires it.
