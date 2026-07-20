# ADR 0024 — Session state: the workspace comes back

- **Status:** Accepted & implemented (2026-07-20). **Supersedes [ADR 0018](0018-viewer-modes-and-the-root-part.md)
  decision 3's persistence half** — the viewer mode stays out of the document, as decision 3
  required, and is restored across relaunch, which decision 3 never required. Adds a **session**
  category to [ADR 0022](0022-document-dump-and-state-classification.md)'s classification, the
  way [ADR 0023](0023-rollback-cache-deltas-and-keyframes.md) added **derived**, and closes the
  `PanelState` reachability gap that ADR 0022's third amendment recorded as open.
- **Date:** 2026-07-20
- **Layer:** document/shell boundary — what persists, into which artifact.

## Context

ADR 0022's third amendment ends with a sentence that is unusual for this repository, because
it names a live defect rather than a decision:

> **a field of `AppConfig` cannot fail to reach an artifact. A field of `PanelState` still
> can, and four currently do.**

The four are `view_mode`, `stack`, `debug_face_orientation` and `debug_brick_faces`. Each was
classified `view` — which by the derive's own error text means it reaches the dump — and each
was hard-coded to a default in `AppConfig::to_panel_state`, captured by nobody. The
classification said one thing and the code did another, for a release.

That happened for a structural reason worth restating: the compiler's guarantee stops at
`AppConfig`, whose captures destructure exhaustively. `PanelState` is read field by field, by
hand, in exactly the shape of the capture that lost the camera's pan target. Classification
without a capture that honours it is a comment.

But the amendment also declined to fix it, and was right to, because two decisions
contradicted each other. ADR 0018 decision 3 says the viewer mode is:

> **viewer state, never document state**: it follows the active selection, is not saved with
> the scene, and never enters undo history.

Issue #88 said the same of the display stack's fold state. Both were implemented as *reset on
launch*. So either those fields are misclassified, or the dump is not carrying what its
category promises — and no amount of care inside the persistence layer settles which.

## Decisions

### 1. Session is a third destination, beside settings and view

State splits three ways at the top, not two:

| Category | The question it answers | Document | Dump |
| --- | --- | --- | --- |
| **document** | What the model *is* | yes | yes |
| **settings** | What the user *prefers* | no | yes |
| **session** | How the workspace was *left* | no | yes |
| **view** | Where the author was *looking from* | no | yes |

The owner's framing is the web browser's: **close it, open it, and your tabs come back.**
Nobody files restored tabs under preferences, and nobody expects them inside a document they
share with a colleague.

The two membership tests, stated so they can be applied rather than admired:

* **Against settings — chosen versus left.** A setting is something the user picked and would
  want honoured in every project: the Home view they pressed a button to keep, the window
  size, the projection. Session state is merely where they stopped. Nobody *chooses* to be in
  Onion fog the way they choose a projection; they were in the middle of something.
* **Against view — looking versus arranging.** View state answers "where was the camera" — the
  orbit pose, the layer band, the density mirror. Session state answers "what was the
  workspace doing" — which mode, which panels, which diagnostics.

### 2. The category is meaning; only the document boundary is routing

`session` and `view` reach exactly the same artifacts. So do `settings` and `view`. That is not
an oversight in any of the three, and it is not an argument for collapsing them.

Of the four categories, **only `document` makes a routing decision** — everything else reaches
the dump, because the dump is a superset by construction (ADR 0022 decision 1). What the other
categories buy is the thing ADR 0022's first amendment says the derive exists for at all:
`#[snapshot(session)]` on the line above a field is *visible in review*, and it tells a reader
which of the membership tests above was applied. A scheme whose categories were merged whenever
their routing coincided would have exactly two words in it and would answer no question anyone
actually asks at a field.

`SessionArtifact` is therefore a real type beside `SettingsArtifact` and `ViewArtifact`, and
`Dump` is composed of all four.

### 3. ADR 0018 decision 3's persistence half is superseded; its document half stands

Decision 3 conflated two claims that read alike:

* **"never document state"** — a viewer mode must not travel inside a shared project file.
  **This stands, unchanged and unweakened.** It is the same argument decision 3 made and the
  same one ADR 0022 decision 2 makes about the rollback cursor: a shared file should mean the
  same thing to everyone who opens it.
* **"not saved with the scene"** — implemented as *not saved at all*. **This is superseded.**
  It is the wider claim, and decision 3 never argued for it; the phrase describes the document
  and was read as describing persistence.

The user-visible change: leaving the app in Onion fog with the display stack folded and finding
it in Normal with everything expanded on relaunch is losing work, in the small. It is the
pan-target complaint at a smaller scale — nobody decided the mode did not matter.

### 4. The debug diagnostics are session state, not transient

`debug_face_orientation` and `debug_brick_faces` were the easy ones to wave through as
`transient`, and that would have been the scheme's first dishonest hatch. The dump's defining
law is that *a scene must be completely reproducible from it*. Both flags change what the
renderer draws. A dump taken while chasing a rendering fault — which is when the brick-faces
diagnostic is *ever* on, since it exists for exactly that — and replayed without it reproduces
a different picture than the one the bug was seen in.

`PanelState`'s own doc comment had already worked this out and left it as an observation: "a
debug mode a fault was observed under is precisely the sort of thing a dump must carry." This
decision is that observation acquiring a route.

### 5. The `PanelState` seam gets a test, because it cannot get a compiler

The reachability gap is not closed by making `PanelState` destructure exhaustively somewhere.
It is read by hand for a reason — `AppConfig` is a flat mirror precisely so that internal panel
churn stays away from anything durable — and inverting that would trade a small, testable gap
for a large coupling.

Instead `tests/state_classification.rs` asserts the property the compiler cannot: **every
`PanelState` field whose category reaches the dump has a route there** — a same-named
`AppConfig` field, or membership in a named `CARRIED_AS_A_SUBSET` list. That list is the second
half of the same gap, the one ADR 0022's amendment called subtler: `geometry` and `layer_range`
are classified as one object each and travel as a hand-picked subset. Those subsets are
defensible; what was missing is that nothing *declared* them, so nothing distinguished a
defensible omission from the pan-target kind. Now a subset costs a deliberate edit to a named
constant.

## Alternatives considered

- **Classify the four as `transient` and delete the contradiction that way.** Cheapest, and it
  is the rot ADR 0022 predicted in as many words: "marking a field transient will always be the
  cheapest way to make the compiler stop complaining". It also states something false about the
  debug flags (decision 4) and something the owner rejects about the viewer mode.
- **Leave them `view` and simply route them.** This works mechanically — `view` already reaches
  the dump — and was rejected on meaning rather than behaviour. The dump is not the only reader
  of these categories; a human deciding where a *new* field goes is, and "view" would have told
  them the panel's fold state is a camera pose. The four fields also share a distinct question
  ("how was the workspace left") that neither `view` nor `settings` asks.
- **Make them `settings`.** Fails the chosen-versus-left test, and would be actively wrong for
  the diagnostics: a debug overlay is not a preference, and presenting it as one in a future
  settings UI would be a bug with a straight face.
- **A separate session *file*.** The dump is still the only artifact written (ADR 0022's third
  amendment), and giving any category its own path is a save/open workflow — a product decision,
  not a structural one. Nothing here forecloses it.
- **Make `PanelState` capture exhaustively too.** Rejected in decision 5: it couples the durable
  record to the panel's internal shape, which is the coupling `AppConfig` exists to prevent.

## Consequences

- **`shot --from-config` adopts the session.** The repro flow is the dump's reader, and a
  replay that reset the viewer mode would defeat the point of carrying it. An explicitly passed
  `--view-mode` still wins — hence a new `view_mode_explicit` marker, since a defaulted `Normal`
  and a chosen `Normal` were previously indistinguishable and the default would have silently
  overridden every repro. The set-only bool flags OR on top of the dump's values, having no way
  to express `false`. The goldens pass their modes as flags and never use `--from-config`, so
  none of them move.

- **Old dumps still load.** Every session key carries a serde default, so a repro written before
  this decision replays as the finished look rather than failing. That is the dump's standing
  tolerance (ADR 0022), not back-compat for its own sake: the config law still says old configs
  may break.

- **`ui` still cannot name serde**, so `ViewMode` and `SignalStackState` persist through remote
  derive shims in `src/artifacts.rs`, exactly as `ProjectionMode` does. The crate boundary law
  (ADR 0016) is unchanged; the cost is one shim per persisted type from a domain crate, which is
  the price already being paid.

- **The classification's healthiest signal still holds.** Two `derived`, zero `transient`, after
  a decision whose whole subject matter was four fields that had every excuse to become
  `transient`. That is the number ADR 0022 said to watch.

- **One gap named in ADR 0022 is now closed and one remains open.** The `PanelState`
  reachability promise is enforced (decision 5). The static / thread-local / GPU audit is
  untouched — nothing here narrows it, and the scheme's coverage should still not be assumed
  total.

## Open

- **Whether `session` needs its own file** if the document ever earns one. The two questions
  arrive together and neither is forced yet.
- **The rollback cursor's category**, when ADR 0022 decision 2 is implemented. It was called
  view state before this category existed, and "where you are working in the fold" reads more
  like session than like a camera pose. Worth re-asking at implementation rather than settling
  here on a feature that does not exist.
