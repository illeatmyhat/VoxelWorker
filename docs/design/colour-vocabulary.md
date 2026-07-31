# The channel emboss does not have

The vocabulary itself — what the accent, warn red, the x-ray reds, the axis colors and
material color are each spent on, why texture is the channel to reach for before a second hue,
and the rule that no state is carried by color alone — is folded into
`docs/architecture/06-authoring.md`. That is the language as it stands.

One case remains unanswered, and one procedure is worth keeping written down.

## Emboss shows nothing, because red would lie

Emboss neither adds nor removes: within a footprint it *moves* an accumulated surface. The
operand overlay is built around add and remove, so emboss has no channel in it. Drawing its
footprint in the subtractive red would tell a viewer who has learned that red the opposite of
what is happening — the band is being raised, not cut away.

The current answer is to draw **nothing**: emboss leaves are filtered out of the operand ghost
entirely. That is honest and it is also a hole — the one operation whose footprint is hardest
to picture is the one with no feedback at all.

Two candidates, neither built:

- **The accent.** Cheap, and it does not lie. It spends the accent's "this is the thing you
  are working on" meaning on an operation rather than a selection, which is a real cost.
- **A texture on the footprint.** Consistent with the standing preference for texture over a
  new hue, and it stacks with whatever hue is already there. More work, and the footprint is a
  volume rather than an outline, so what texture even means on it is unresolved.

The icon set already made the same ruling in its own medium — the emboss mark is the
*footprint*, not a raised ridge — so whatever is drawn should agree with it.

## Before spending a color

A prompt rather than a gate, kept because the failure mode is silent and cumulative:

1. What does this hue already mean, here and in whatever the person came from?
2. Could a **texture** carry it instead, leaving hue free?
3. Could a **word or mark** carry it? Labels are cheap and unambiguous.
4. If it is genuinely new and genuinely color-worthy, what is it *taking* the color from, and
   is that trade written down?
5. Does it survive the red–green test, and does it survive being seen once rather than learned?

The recurring answer so far has been (2).
