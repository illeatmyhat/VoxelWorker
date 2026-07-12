# 04 — Work

This document is about *time*: how the system spends it, and the discipline that keeps
the interface responsive while arbitrarily large work happens somewhere else. The
governing law is Law 7 — **the interface never waits** — and it decomposes into three
tempos and one iron rule about staleness.

## Three tempos

Every computation in the system belongs to exactly one tempo, and knowing which one is
the first question asked of any new feature:

1. **Per-frame** — runs every frame, so it must be constant-time in the scene: camera
   math, uniform uploads, display clipping (the onion skin), polling workers for
   finished results. Nothing per-frame may touch work proportional to the document.
2. **Per-edit, incremental** — runs inline when an edit is *localised*: re-evaluating
   the dirty chunks, patching the mesh's touched chunk buffers, patching the brick
   field's touched records and atlas slots. Inline is correct here because the cost is
   proportional to the edit, and inline work has no swap latency — the user sees their
   chisel stroke this frame.
3. **Wholesale** — a rebuild of an entire derived artifact: the first build of a large
   scene, a density change, an edit spanning the region, an undo of one of those. Small
   wholesale rebuilds run inline (cheaper than any coordination); large ones are
   dispatched to a worker and the shell keeps drawing what it has —
   **stale-while-rebuilding** — until the replacement lands.

The threshold between "small" and "large" is a single named constant, expressed in
covering chunks, shared by every artifact that routes work.

## The worker pattern

All background work follows one shape:

- A dedicated thread owns a build function. Requests go in on a channel; results come
  back on another. The shell polls for results once per frame and never blocks.
- **Drain-to-latest.** Before building, the worker drains its queue and builds only the
  newest request. A burst of edits collapses into one build; a worker never develops a
  backlog.
- **Generations.** Every request carries a monotonically increasing generation number.
  The shell records the newest generation dispatched, and a result is installed only if
  its generation is still the newest — a result overtaken by a later edit is discarded
  on arrival. Anything that makes the resident artifact current *bumps the generation*,
  so a superseded in-flight result can never install over fresher state.
- **Panic containment.** A build that panics is caught on the worker; the worker
  survives, the shell keeps its current artifact, and the next edit re-dispatches.
  A background thread is never allowed to die quietly and wedge the pipeline.

Workers carry data, not authority: a request is an immutable snapshot (shared chunk
handles and plain values), and a result is inert until the shell — the only writer —
installs it on the main thread.

## The staleness law

The one rule that keeps concurrent display sane:

> **Never patch a stale artifact. Look at it if you must; replace it when you can.**

An artifact is *stale* the moment it no longer reflects the latest evaluation: a
display kept as a placeholder during a handover, a mesh skipped because the brick field
was on stage, any artifact whose replacement is still building on a worker. Applying an
incremental patch to such an artifact produces a chimera — the patch's chunks at the
new state, every other chunk at some older one — which is worse than either state
because it looks plausible.

The law is enforced structurally, not by vigilance:

- **Routing is a pure function.** For each derived artifact, one side-effect-free
  function decides patch-inline / rebuild-inline / rebuild-async from three facts: is a
  rebuild outstanding, is the edit localised, does the resident artifact actually
  reflect the latest evaluation. Pure functions are exhaustively unit-tested; the shell
  merely obeys them.
- **While a rebuild is outstanding, every edit routes wholesale.** No exception for
  "small" edits — the interlock is what prevents the chimera.
- **Install seams are single.** Each artifact has exactly one function through which a
  fresh build becomes current; that seam bumps the generation, clears the outstanding
  flag, and completes any pending handover. Nothing installs around the seam.

## What the user experiences

The sum of the discipline, stated as promises:

- A chisel stroke appears the frame it is made.
- A wholesale rebuild of any size leaves the model on screen and the camera live; the
  new state pops in when ready, and only the *newest* state ever pops in.
- Opening a huge document shows the window immediately; the model arrives when built.
- No sequence of rapid edits, cancellations, and mode toggles can produce a display
  that mixes two states of the document.
