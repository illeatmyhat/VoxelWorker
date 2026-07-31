# Sketch selection — what is left to build

The selection model is folded into `docs/architecture/06-authoring.md`: the four gestures, the
single mixed set, hit priority, the directional marquee, delete as an action on the selection,
and the mode-dispatched menu. All of that is shipped and described there as it stands.

Two things remain, plus one departure worth keeping written down.

## Move the whole selection

Shipped: the single-vertex drag, which is the degenerate case of *propose a delta → solve →
apply*. Not shipped: proposing a delta for a multi-entity selection. The gesture and the apply
are unchanged; the work is in the solve step, which now has a real solver behind it to correct
against.

## Scene-node marquee

The sketch marquee is shipped with both window and crossing semantics. Scene nodes do not have
one yet, and it should be built once against both rather than twice.

## The departure: a region is pickable but not a selection member

Worth recording because the shipped code deliberately differs from what was first specified.
A region was originally to be a selection-set member like a point or a segment. It is not.

A region's identity is a **set** of boundary-edge ids, while a selection target is a small
value copied by hand throughout the shell. Admitting a variable-length payload would change
that representation everywhere it is passed. Since a region's only verb is pick / unpick, and
that verb was always going to live on the context menu, it never needed to enter the set to be
operated on.

The cost is real and is accepted: any future verb that wants to act on "the selection,
including regions" has to reopen this. Nothing wants that yet.
