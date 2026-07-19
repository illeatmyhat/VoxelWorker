# Colour and channel vocabulary — recommendations

Companion to [the Signal chrome spec](viewport-chrome-signal.md), which owns the token
*values*. This document is about what those values may be asked to *mean*, and it is
advisory: it records what the surrounding software world has already spent its colours on,
what we have spent ours on, and where a departure is likely to cost more than it looks.
Depart when you have a reason — but say the reason, because the failure mode here is
silent and cumulative.

## The premise

**Colour is a scarce channel, and every meaning assigned to a hue reduces what any hue can
mean.** A viewer learns "red means removed" from ten encounters and applies it on the
eleventh whether or not that was intended. This is why a new state cannot be given a
colour in isolation: the question is never "what colour should *this* be", it is "what is
left, and what will this teach the viewer about everything else".

Two rounds of design exploration produced the same collision three times over. Five of six
concepts wanted a second accent, all reached for amber, and each meant something different
by it — emboss identity, uncommitted state, the fold cursor. Any one of those in isolation
is reasonable. All three at once is a language where amber means nothing.

## What prior art has already spent

Treat these as occupied. Fighting a convention this broad costs the user's transfer from
every other tool they use, and buys very little.

| Convention | Where it is established | What it therefore means to a new user |
| --- | --- | --- |
| **X red · Y green · Z blue** | Blender, Maya, Fusion, Unity, Unreal — essentially universal, mapping RGB onto XYZ | Spatial axis. The strongest convention in 3D software; a red arrow in a viewport is an axis until proven otherwise |
| **Red = removal / destructive / error** | Boolean cutters in hard-surface workflows (HardOps, BoxCutter, MagicaCSG), plus the near-universal UI meaning of red | "This is being taken away", or "this is wrong" |
| **Amber / yellow = pending, stale, needs attention** | CAD timelines flagging out-of-date or suppressed features; unsaved-asset markers in game editors | "Not settled yet" — attention without alarm |
| **One reserved selection hue** | Blender orange (with a lighter variant for the active object), Unreal orange, Maya green-and-white | "You picked this" — never anything else |
| **Translucency / ghosting = not really there** | Onion skinning in animation tools; section analysis in CAD | "Present in the document, absent from what you are looking at" |
| **Hatching = a cut surface** | Orthographic drafting convention, centuries old | "You are seeing the inside of something that was sectioned" |

The hatch row is the one worth pausing on, because we use hatch differently (see below) and
the drafting meaning is close enough to our onion clip to be a genuine near-miss.

## What VoxelWorker has spent

Values live in the chrome spec; this is the meaning side.

| Channel | Meaning here | Recommended not to mean |
| --- | --- | --- |
| **The accent** (onion-haze blue) | Active, selected, current, live. It is also the onion ghost's own hue, deliberately, so chrome and ghost read as one system | Anything with valence — it is not "good", not "confirmed", not "safe" |
| **Warn red** | Subtraction and removal, plus genuine warnings | Emphasis, or "important". A red that sometimes just means loud stops meaning removed |
| **X-ray reds** (quiet / loud) | Boolean operand bodies under Show booleans; the quiet/loud split is depth, not severity | Any non-boolean overlay. This is the collision this document was written for |
| **Axis red / green / blue** | Axes, in the view cube, gizmos and triads | Anything at all elsewhere in a spatial context |
| **Material colour** | The material. Pigment is the model's own, not the interface's | Interface state. If UI state tints geometry, a brick stops looking like brick |

**Emboss is the specific case that prompted this.** It neither adds nor removes — it moves
an accumulated surface within a footprint — so it has no home in a vocabulary built around
add and remove. Drawing its footprint in x-ray red tells a viewer who has learned the red
vocabulary that the band is being cut away, when it is being raised. Recommended: give
emboss the accent or a texture, and reserve red for material that genuinely leaves.

## Prefer a second channel over a second hue

The strongest recommendation here, and the one the exploration converged on independently
in five of six concepts: **when you need a new meaning, reach for texture before hue.**

Hue does not scale. Linked-ness, staleness, uncommitted-ness, and dropped-by-the-cursor are
four states that can co-occur on one element, and there is one accent to spend. Texture is
orthogonal — it stacks with colour instead of competing with it, so a linked instance can
be selected *and* drifted and still read.

A vocabulary that has worked in practice:

| Texture | Suggested meaning |
| --- | --- |
| **Hatch** | "Touches something not shown here" — a linked instance, the other instances of a definition being edited, a definition banner |
| **Warn-hatch** | Hatch plus warn: the binding is stale (a sculpt delta whose surface moved) |
| **Dashed outline** | Uncommitted — a live session, a preview, something not yet in the fold |
| **Dimmed + hatched** | Present in the document, excluded from what is currently evaluated |

Against the drafting convention that hatch means a section cut: our clip is expressed by
ghosting and haze rather than hatch, so the two do not collide on screen today. If a future
surface wants hatch for section-cut faces, that is the moment to revisit — the drafting
meaning is older and better established than ours.

## Accessibility

Roughly one man in twelve has some red–green colour deficiency. Two recommendations follow,
and the first is close to non-negotiable:

- **Never encode a distinction in red-versus-green alone.** Our axis colours already use
  that pair, which is tolerable because axes are also distinguished by position, direction
  and a letter. State should not rely on it.
- **Every colour-carried state should have a non-colour carrier too** — a texture, a glyph,
  a word, an inset bar. This pays off beyond accessibility: it is also what makes a state
  legible in a screenshot, in a code review, and to anyone reading a design mock rather
  than driving it.

## Before giving something a colour

A short checklist, offered as a prompt rather than a gate:

1. What does this hue already mean, here and in the tools the user came from?
2. Could a **texture** carry it instead, leaving hue free?
3. Could a **word or glyph** carry it? Labels are cheap and unambiguous.
4. If it is genuinely new and genuinely colour-worthy, what is it *taking* the colour from,
   and is that trade written down?
5. Does it survive the red–green test, and does it survive being seen once rather than
   learned?

The recurring answer, in this project so far, has been (2).
