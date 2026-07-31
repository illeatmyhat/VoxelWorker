# What a drag costs — measured, and the one number still missing

The tool grammar itself is folded into `docs/architecture/06-authoring.md`: arm → drop →
selected, the picked point, manipulators belonging to the selection, and the four laws a
manipulator obeys. The field-versus-array rule that decides which rotation a node offers is in
`docs/architecture/01-document.md`. All of that is shipped and described there as it stands.

This file holds the measurements that settled how a drag behaves, and the one measurement that
has not been taken.

## The question

A preview is free to move, but a *committed* node's move changes composed geometry, so
something recomposes. The choice was between moving **live per snap step** — truest to the
premise, but it recomposes on every step — and **ghosting until release**, which is one clean
intent but leaves the object standing still during the gesture. It was left to a measurement
rather than to a preference.

## The answer: cost tracks the dirtied volume, not the scene

| backdrop | voxels | drag a small node inside it | drag the backdrop itself |
| --- | --- | --- | --- |
| 5×1×5 | 102 K | 3.5 ms · 2 chunks | 3.0 ms |
| 20×8×20 | 13 M | 4.0 ms · 2 chunks | 23.7 ms |
| 50×10×50 | 102 M | 2.5 ms · 1 chunk | 121.9 ms |
| 100×20×100 | 819 M | **1.7 ms** · 2 chunks | 501.9 ms |

Dragging a small node is **flat in scene size** — 2–4 ms whether the scene holds a hundred
thousand voxels or eight hundred million. Targeted invalidation evicts one or two chunks and
every other resident chunk survives as a refcount bump, so the largest scene costs *less* than
a small one whose single node is proportionally larger.

The right-hand column is the case that grows, and it grows because there the moved node **is**
the scene, so its dirty region covers everything. That is real — a first node in an empty
document — but it is not what a manipulator usually drags.

So the rule is **adaptive** rather than one behavior for every scene: move live while the last
rebuild was cheap, fall back to a ghost when it was not. Either way the gesture collapses to
one intent on release.

## Leaving the extent is not the second path it looked like

A node dragged past the composite's bound grows the region, which moves the floating origin,
which withholds the incremental rebuild hint — so every baked vertex buffer must be re-meshed.
That looked like a costlier path worth measuring separately.

Holding region growth constant and splitting a single outward drag on whether the extent
midpoint actually moved, the two regimes are **indistinguishable**: 2.4/2.4, 1.9/1.8, 4.8/5.1,
2.0/2.0 ms. Withholding the hint is a branch, not work — invalidation has already run and
already localized, the resident cache being frame-independent, so a reframing rebuild
re-classifies the same handful of chunks and then sets a flag.

A smaller surprise from the same table: outward steps are often *cheaper* than inside ones
(1.9 ms against 5.3 ms on the medium scene). A node dragged into empty space touches fewer
occupied chunks than the same node nudged through dense geometry. At this layer cost tracks
**locality**, not extent.

## Open: the number this does not contain

Every figure above is the rebuild only, and the rebuild returns before the mesher runs. The
wholesale re-mesh a reframing drag forces is real and lands entirely downstream. So these are
**lower bounds** on what a drag step costs the user, and the extent-growing figures are the
loosest of them.

Measuring the re-mesh requires a probe where the mesher runs, not where these ran. That is the
open number, and it is the one that decides whether the adaptive rule needs a second trigger
for "this drag left the extent".

## Noted for later: an SDF viewer mode

If the preview renders the parametric field directly, the machinery for *seeing the field
instead of the voxels* exists as a side effect — a way to look at what the document means
before voxelization, with the lattice out of the way. It would join the exclusive viewer modes
rather than being a toggle bolted onto one tool. Not scoped, not scheduled; recorded so the
preview work does not foreclose it.
