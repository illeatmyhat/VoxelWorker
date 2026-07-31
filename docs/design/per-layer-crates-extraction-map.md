# The layer split — what was left uncarved

Every layer is now its own library and the flow is compiler-enforced; that law and the layer
chain live in `docs/architecture/README.md`. The migration itself is finished, so what follows
is only the residue: the things deliberately not done, and the one boundary that came out
different from how it was drawn.

## The boundary that moved: the work layer links graphics

The split was drawn expecting exactly one library to link the graphics API. The survey found
otherwise — the geometry worker builds meshes on its own thread and the orchestrator owns the
device, so both hold it directly. The work layer therefore links graphics too, deliberately.

Three ways out existed and the cheapest was taken: cut the layer anyway and revise the law.
Folding work into the shell was rejected (the shell would swallow the tempo discipline), and
decoupling the workers from the device by passing it per call remains available as a real
refactor if the boundary is ever wanted clean. Nothing needs it yet.

## Deliberately not separate libraries

A register of restraint, so each is not re-proposed:

- **Mesh and brick as siblings of display.** They interoperate through one orchestrator;
  folders, not libraries.
- **Proof and oracles.** Parity tests travel with the code they check; the capture bin stays a
  bin.
- **Measurement queries.** They fold into the evaluator's query surface.
- **Device creation.** It is born from the window surface, so it belongs to the shell.

## Files left whole

No production library module remains above roughly thirteen hundred lines that is not a single
cohesive type. What is still large is large on purpose:

- Per-module test suites (the scene, brick and mesh suites), where test code travels as one
  file per owner.
- The two binaries — the window event loop and the capture oracle.
- The brick raymarcher, which is one renderer type, like the mesh pipeline beside it.

Splitting the test suites and carving the two binaries are the optional follow-ups. Neither
blocks anything.

## The standard each library is held to

Every library states its law in its root and cites the chapter it implements, and its module
documentation carries the rationale-and-citation voice rather than a restatement of the
signatures. As a file moves, its documentation comes up to that bar — the bar being the
"readable spec" the pure computer-science libraries set.
