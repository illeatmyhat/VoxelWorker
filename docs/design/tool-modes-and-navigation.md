# Tool modes & navigation — what is left to build

The interaction model itself is folded into `docs/architecture/06-authoring.md`: the left
button's verb, the mode set, the two pivots, the two orbit types and their seam, the menu, and
the binding registry. All of that is shipped and described there as it stands.

This file holds only the remainder.

## The transform-gizmo subsystem (W / E / R)

The mode keys are bound and act on the selection, but **translate, rotate and scale gizmos do
not exist**. The only on-canvas manipulators today are the sketch vertex handles and the camera
axis guide. Every mode past selection is therefore currently a mode with nothing in it, and
building the manipulators is the bulk of the work — a keybinding is the easy part.

Deferred deliberately until selection is correct app-wide, on the grounds that a manipulator
acting on a shaky selection is two problems at once.

## Rotate and scale inside a sketch — undecided

Rotate and scale are not meaningful on a lattice profile as the sketch stands, so those two
modes are either disabled in sketch mode or remapped to something that is meaningful. **What,
if anything, they map to is open.** Disabling is the safe default and is what happens now;
nobody has argued for a remapping yet.

## Move-the-whole-selection in a sketch

The sketch's move mode is the constraint-mediated request described in the architecture
chapter — propose a delta, solve, apply. The single-vertex drag is that path's degenerate case
and is shipped. **Moving an entire multi-entity selection is not**, and it is the slice that
makes the solve step carry its weight.

## Scene-node marquee

The marquee's semantics are settled and the sketch implementation is shipped; extending it to
scene nodes is a separate slice, to be built once against both rather than twice.

## Multi-select inspector editing

With more than one thing selected the inspector shows a count summary only. Editing a shared
field across a heterogeneous selection is unspecified.
