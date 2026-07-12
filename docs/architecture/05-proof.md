# 05 — Proof

A system whose every hot path is an optimization needs a doctrine for staying honest.
This is that doctrine. It has one axiom:

> **Exactness is a property you keep by construction and prove by comparison — never a
> property you argue for.**

## Oracles

An **oracle** is the simplest possible implementation of an answer — dense, brute-force,
obviously correct, and as slow as it likes. Oracles are not production code; they are
the measuring stick production code is held against, and the only place in the system
where volume-proportional memory is legal (Law 2 stops at the test boundary, on
purpose: an oracle's honesty *is* its density).

Every load-bearing fast path names its oracle:

| Fast path | Oracle it must match |
| --- | --- |
| Interval-bound block classification | Per-voxel evaluation of the composed field |
| Two-layer chunk contents | Dense resolution of the same region |
| Incremental chunk invalidation | A from-scratch rebuild of the whole covering set |
| Surface-only display records | The all-blocks record build |
| Incremental brick patch | A wholesale field build of the same document state |
| Occupancy pyramid traversal | The same march with the pyramid disabled |
| Mesh built by a worker | The same mesh built synchronously |

The required relation is **byte-equality** wherever representations coincide, and
occupancy-equality (same voxels shown, same hits) where only meaning coincides. "Close
enough" is not a relation this system uses for occupancy.

## Parity gates

A **parity gate** is the oracle comparison made permanent: a test that constructs a
scene, runs both the fast path and the oracle, and asserts the relation. Gates ride the
ordinary test suite so they run on every change forever — an optimization is guarded
not by the memory of the person who wrote it but by a machine that never forgets.

The discipline for new work follows from this mechanically. A proposed fast path must
state, before it is built: *what is the oracle, and what is the relation?* If no oracle
can be stated, what is being proposed is not an optimization but a second
implementation with its own opinions — and the system does not accept second opinions
about occupancy.

## Construction, types, and machine-checked proof

Comparison is not the only instrument. The doctrine ranks three ways of holding a
property, and uses the strongest one each property admits:

1. **Make it unrepresentable.** The strongest proof is a program in which the defect
   has no syntax. The type system already carries several laws this way: handed-out
   chunk sets are immutable and reference-counted, so "the data changed under a
   worker" is not a bug that testing must find — it is a program Rust will not
   compile. Ownership makes torn state unwritable; exhaustive matching makes an
   unhandled routing case a compile error. Wherever a law can be phrased as a type —
   a coordinate that knows its frame, a handle that cannot outlive its arena — the
   type is preferred over both the proof and the test, because it is checked on every
   keystroke and can never be skipped.

2. **Prove the kernel.** Beneath the system's exactness claims sits a small pure
   kernel of genuinely mathematical statements: the conservatism of an interval bound
   ("*all-solid* is asserted only when every point is solid"), the soundness of
   composing bounds through boolean operators (an abstract-interpretation argument),
   the algebra of incremental patches ("patched state equals rebuilt state"), the
   supersede protocol ("no interleaving installs anything but the newest
   generation"). These fit within machine-checkable fragments — a model in a proof
   assistant, or a verifier that discharges annotations on the code itself — and the
   kernel is small and stable enough that formalizing it is a bounded investment, not
   a research program. A theorem, where obtainable, subsumes any number of test
   cases.

3. **Compare end-to-end.** What no source-level proof reaches is everything between
   the source and the pixels: the shader compiler, the driver, the floating-point
   units, the windowing system, the scheduler. A proof about the raymarch's algorithm
   says nothing about the code a graphics driver actually compiled it into — and
   drivers are, empirically, where the strangest defects live. The oracle relation is
   the only instrument that checks the *delivered* system, because it compares
   observable outputs of the whole stack.

So construction, proof, and comparison are layered, not rivals: types for what types
can say, theorems for the pure kernel, oracles for the world. The parity gates remain
even where a theorem exists — a theorem verifies the mathematics; the gate verifies
that the shipping binary still implements it.

## Golden images

Above occupancy sits rendering, and rendering is proven visually: **golden images** —
offscreen renders of fixture scenes compared pixel-wise against blessed references.
Goldens catch what unit relations cannot: a correct occupancy drawn with the wrong
face, seam, or clip. The two display paths' agreement (raymarch vs mesh over identical
chunks) is itself a golden-class property, held by construction (shared texture rule,
shared chunk source) and checked by comparison.

Headless rendering is a first-class capability, not a test trick: the renderer can
produce a frame with no window, which is what makes visual proof automatable.

## Probes

Correctness gates do not catch performance regressions, and a system whose contract is
a *cost envelope* must test the envelope. **Scaling probes** are ignored-by-default
tests that build scenes along a size axis and measure both time **and** memory —
always both, because history shows a fix for one axis silently spends the other. A
probe's output is a table a human reads against the envelope's promise ("per-edit ∝
edit", "resident ∝ boundary"); its value is that the measurement exists, is
reproducible, and can be extended rather than re-derived when a new question arrives.

## The live binary is part of the suite

Headless tests share the fate of all tests: they exercise what they were written to
exercise. The final gate for user-facing behaviour is the release binary, launched and
observed — startup output, resident memory, the first rendered frame, an edit made by
hand. A change to interactive behaviour is not done when its tests pass; it is done
when the running application has been watched doing the right thing.
