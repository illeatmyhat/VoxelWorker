# Load-Bearing Data Structures

This is a tour of the data structures the system's promises actually rest on. Each
entry says, in plain terms: what the structure is, what shape it takes, and which
quality of the whole — speed, memory, stability, exactness — it is personally
responsible for. The tour runs in the order data flows: from the user's design, through
evaluation, to the screen, with the machinery of safe concurrency at the end.

A note on reading: where a term of art is unavoidable, it is defined at first use.
Nothing here requires prior familiarity with the codebase.

---

## 1. The operation stack — the design itself

**What it is.** The user's design is stored as a small tree of *operations*: "a box of
this size, here", "this drawn outline, spun around an axis", "these two shapes
combined, that one subtracted". It is a recipe, not a photograph — the voxels a design
implies are never what is saved; the *instructions for producing them* are.

**Shape.** A tree of nodes. Leaves are geometric operations with their parameters;
inner nodes group, name, and reuse. Each node carries a position on the block grid and
a material.

**What it buys.** Permanence and smallness. A file's size tracks the number of design
decisions, not the number of voxels — a fortress wall a kilometre long is a dozen
numbers. Editing means changing a parameter and re-deriving, so nothing is ever
destroyed by an edit; and any cache anywhere in the system can be thrown away at will,
because the recipe can always cook it again. Every other structure in this document is,
formally, disposable. This one is not.

## 2. The command stack — undo that cannot drift

**What it is.** Every edit, before it is applied, records the exact instruction that
would reverse it. Undo means applying the reversal; redo means applying the original
again.

**Shape.** A list of (edit, inverse-edit) pairs with a cursor.

**What it buys.** Stability of the editing experience. Because *every* edit passes
through one door and records its inverse there, undo is one mechanism with no special
cases — there is no widget that "doesn't undo right", because no widget has its own
editing machinery to get wrong.

## 3. Measurements that remember their units

**What it is.** When the user types a size — "3 blocks" or "40 voxels" — the system
stores *what they typed*, alongside the concrete voxel count it currently works out to.

**Shape.** A value plus its unit, kept next to the derived canonical number.

**What it buys.** Preservation of intent. The document has a global *density* setting —
how many voxels each block is divided into (bounded between 1 and 64; see entry 9 for
why the bound is a gift). When density changes, a wall authored as "3 blocks" is
re-derived and stays three blocks wide, because the system still knows that "3 blocks"
is what was meant. Storing only the voxel count would silently freeze old intent at the
old fineness.

## 4. The conservative interval bound — how big solids stay cheap

**What it is.** Before evaluating a region voxel-by-voxel, the system asks each
operation a cheaper question about a whole box of space: "over this box, are you
*certainly* all solid, *certainly* all empty, or can't you say?" The answer is allowed
to be unhelpful ("can't say") but never allowed to be wrong.

**Shape.** A function each geometric operation implements, returning one of three
answers for an axis-aligned box; the answers compose through the boolean operators
(the union of "certainly empty" and "certainly empty" is certainly empty, and so on).

**What it buys.** Nearly all of the system's speed, at zero cost to exactness. Whole
regions inside a large solid, or in open air, are settled with one question instead of
thousands; expensive per-voxel work happens only in the thin shell where a surface
actually passes. And because a wrong "certainly" is forbidden — uncertainty falls back
to honest per-voxel work — the fast answer is *identical* to the slow one everywhere,
which is what lets the proof gates hold the whole edifice to byte-equality.

## 5. The two-layer chunk — memory that follows the surface

**What it is.** Space is divided into fixed-size cubes called *chunks*. Within a
chunk, every block is classified: **air** (stores nothing), **coarse-solid** (stores
only *which block it is* — one small ID, no voxels), or **boundary** (a surface passes
through it — only here is per-voxel shape stored, and even then as a short list of
solid boxes rather than a filled 3D array).

**Shape.** Per chunk: a flat array of block IDs (the coarse layer), plus a sparse map
from the few boundary blocks to their box-lists (the microblock layer), plus per-face
flags described in entry 11.

**What it buys.** The memory law: cost proportional to a design's *skin*, never its
volume. A solid mountain is almost entirely coarse-solid IDs; only its surface becomes
voxel data. This mirrors how the target game itself stores chiseled builds, which makes
export natural, and it is the single shared form every consumer — display, exporter,
measurement — reads, so no consumer can disagree with another about what exists.

## 6. The resident cache and the shared handout

**What it is.** Chunks, once classified, are kept in a cache. When an edit happens, only
the chunks the edit touches are discarded and re-derived; everything else is reused
as-is. Consumers receive the current chunk set as *shared references* — a
reference-counted pointer per chunk (a pointer that knows how many holders it has and
frees the data when the last one lets go), not a copy.

**Shape.** A map from chunk coordinate to a shared, immutable chunk; handouts are lists
of (coordinate, shared pointer) pairs.

**What it buys.** Two things. *Per-edit speed*: an edit's cost tracks the edit, because
untouched chunks are handed out again for the price of bumping a counter. *Concurrency
safety*: because a handed-out chunk is immutable and stays alive as long as anyone
holds it, a background worker can keep computing over the set it was given even while
the main thread moves on — no locks, no torn reads, no "the data changed under me."

## 7. The producer-bounds tree — finding what an edit touches

**What it is.** To know which chunks an edit dirties, the system needs to answer "which
operations' geometry overlaps this region?" quickly. It builds a *bounding-volume
hierarchy* — a tree of nested boxes, where each tree node's box encloses all the
geometry beneath it — over the operations' world-space bounds, and answers overlap
questions by descending only into boxes that intersect the query.

**Shape.** A binary tree of axis-aligned boxes with operations at the leaves. Notably,
it is rebuilt from scratch on every edit rather than kept up to date.

**What it buys.** Fast dirtying without a whole category of bugs. Descending a box tree
turns "check every operation" into "check a logarithmic handful". Rebuilding per edit
sounds wasteful and is not: at authoring scales the rebuild is microseconds, and a
structure that is *always freshly built* can never be stale — the entire class of
"index out of sync with reality" defects is not fixed but made unexpressible.

*The next three entries are, taken together, the renderer.* The primary display is a
*raymarcher*: for every pixel, a ray is walked cell-by-cell across the block grid (a
"DDA" — the digital equivalent of drawing a straight line through graph-paper squares)
until it meets something solid. Everything the ray consults along the way is one of
these three structures — the records say *whether and what*, the atlas says *what
shape inside*, and the pyramid says *how far ahead is certainly empty*. The ray never
consults the document's geometry operations themselves; it walks a cache. That is the
renderer's founding bargain (see [Display](03-display.md) for its lineage): frame cost
tracks what is on screen, not how the design was authored.

## 8. Sorted display records — a frame's cost decoupled from the scene

**What it is.** The primary display draws by casting rays into a *cache* of the scene.
The cache's spine is a flat list of *records*, one per visible surface block, each
tagged with a single integer key encoding its world position, and the list is kept
sorted by that key. A ray asking "is there a block here?" finds out by binary search —
repeatedly halving the sorted list — in a few dozen steps even for millions of records.

Records exist *only for the surface*: a block completely buried behind solid
neighbours can never be the first thing a ray meets, so it never becomes a record at
all.

**Shape.** A sorted array of small fixed-size records (position key, kind, material,
and — for carved blocks — where their voxel data lives; see entry 9).

**What it buys.** The per-frame law. Drawing cost depends on what is on screen and the
list's logarithm — not on how many operations the document composes, and not on the
volume of the solids. And because the record set tracks the *skin*, the data uploaded
to the graphics card after a full rebuild also tracks the skin: enormous solid
interiors cost the display nothing.

## 9. The carved-block atlas and its free list

**What it is.** A block the chisel has touched needs its actual voxel shape available
to the ray-caster. All such shapes live together in one big reusable 3D texture — an
*atlas* — divided into equal slots, one carved block per slot. A *free list* (a simple
list of vacated slot numbers) recycles slots as carved blocks come and go, so the atlas
does not grow just because editing happened.

**Shape.** One pooled 3D texture of `edge³` slots plus a list of free slot indices;
each record from entry 8 that represents a carved block stores its slot number.

**What it buys.** Per-edit display cost proportional to the edit: re-carving one block
re-uploads one slot, never the atlas. The free list keeps long editing sessions from
leaking texture memory.

**Where the density bound pays off.** Because density — voxels per block edge — is
bounded at 64, one *row* of voxels through a block always fits in a single 64-bit
machine word (and at density 32 or below, in the 32-bit words graphics hardware is
happiest with). A row stored as a word makes the fundamental occupancy questions
one-instruction cheap: "is this row empty?" is a comparison with zero, "how many voxels
here?" is a popcount (a hardware instruction that counts set bits), "does this face
touch that face?" is a bitwise AND of two rows. The bound is what entitles occupancy
storage to be *bit-per-voxel with word-aligned rows* rather than byte-per-voxel — an
eight-fold saving and a faster one — and the same entitlement extends to measurement
queries (a widest-run scan over rows becomes shifts and masks) wherever occupancy is
touched.

## 10. The occupancy pyramid — how rays skip emptiness

**What it is.** Above the fine record set sits a small pyramid of coarser summaries:
"does this 8-block cell contain *any* record?", then the same question for cells of 64
blocks, and so on. A ray marching through space consults the coarsest level first and,
finding a cell empty, strides across the whole cell in one step instead of testing
every block within it.

**Shape.** A few levels of sparse occupancy sets, coarsest to finest, keyed the same
way as the records.

**What it buys.** Traversal speed across sparse scenes. Most of a large scene is empty
air or open sky; the pyramid makes crossing it cost the logarithm of the emptiness
rather than the width of it. Crucially, the pyramid answers only *"might there be
something here?"* — never "what is here?" — so a wrong pyramid can only make a ray
slower, never make it wrong; exactness stays with the records.

## 11. Seam flags — cooperation without inspection

**What it is.** Every boundary block records, for each of its six faces, whether that
face is effectively solid. When a neighbouring chunk decides which of its own faces are
hidden and need not be drawn, it reads these flags instead of opening the neighbour's
voxel data.

**Shape.** Six booleans per boundary block, stored with the chunk that owns the block.

**What it buys.** Independence between chunks. Chunks can be built, cached, and meshed
in parallel without ever expanding a neighbour's contents; the flags are a tiny,
pre-digested summary of exactly what a neighbour is entitled to know. This is the same
courtesy the pyramid extends to rays, applied at chunk seams.

## 12. The generation tracker — how late answers are refused

**What it is.** Long rebuilds happen on background threads while the interface keeps
drawing the previous state. Each dispatched rebuild is stamped with a number from a
counter that only counts upward; the interface remembers the newest number it issued
and, when a finished rebuild arrives, installs it only if its stamp is still the newest.
Anything that makes the current state fresh by other means — an inline rebuild, a
superseding edit — advances the counter, which silently condemns every rebuild still in
flight.

**Shape.** One monotonically increasing integer per background pipeline, plus the
stamp carried on every request and result.

**What it buys.** The stability half of responsiveness. Without it, a slow rebuild of
an *old* state could land after a fast rebuild of a *new* one and overwrite it — the
display would flicker backward in time. With it, no interleaving of edits, rebuilds,
cancellations, and completions can install anything but the newest state. It is a
two-integer solution to a problem usually solved with locks, and it composes with one
rule of conduct: while any rebuild is in flight, the artifact it will replace is
looked at but never patched.

## 13. Wide-integer frames — correctness at the edges of huge scenes

**What it is.** Positions in the world are kept as 64-bit integers on the voxel
lattice, and every derived artifact records the *reference point* (the "recentre") it
was expressed relative to. Rendering happens near the origin in floating point; the
recentre travels with the data so that nothing ever has to guess which reference point
a value meant.

**Shape.** 64-bit integer coordinates in the document and evaluator; an explicit
integer offset carried on every artifact that baked one in.

**What it buys.** Exactness at scale. Floating-point positions degrade quietly as
coordinates grow — geometry a hundred thousand voxels from the origin starts to swim
by fractions of a voxel, which in a chiseling tool is corruption, not noise. Integer
positions cannot drift, and carrying the reference point explicitly means two parts of
the system can never disagree about where "here" is: the value says so itself.

---

## Substrate — the structures as pure computer science

Several of the structures above have two identities. To the system they are a
producer-bounds tree, a carved-block atlas, an occupancy pyramid; to computer
science they are a bounding-volume hierarchy, a slot allocator over a packed
cube, a sparse min-mip pyramid. The second identity is the durable one — it can
be read, tested, and reasoned about (including its performance) without knowing a
single thing about voxels — and it lives in its own library: the **substrate**
crate.

**What it is.** A separate library crate holding the load-bearing structures
whose identity is purely algorithmic. The application depends on it; it depends
on no application code, and that direction is enforced by the compiler rather
than by convention — the reason it is a crate and not a folder.

**The boundary law.** A structure belongs in substrate exactly when it is
describable entirely in textbook computer-science / mathematics vocabulary and is
parameterized only by plain numbers and generics — never by a scene, a producer,
a chunk, or any other domain type. The moment a structure must name something
from the domain it is an *adapter*, and adapters stay in the application at their
own seams. This keeps the library free of the vocabulary that dates fastest.

**The naming rule.** Inside substrate the well-known name from the literature *is*
the type's name — a median-split bounding-volume hierarchy is `MedianSplitBvh`, a
bit-packed occupancy cube is `BitCube`, an exact rational is `ExactRational`. Each
component's own definition carries the explanation of the structure and the
citations to the canonical literature, noting where the local variant deviates.
The domain's own words survive only at the adapter seams.

**Where the inventory lives.** Which structures move, in what order, and with
which oracles is a dated engineering input, not a timeless fact, so it lives apart
from this chapter in `docs/design/substrate-extraction-map.md`. This section
states the shape; that map tracks the migration.

## The shape of the whole

Read together, the thirteen structures implement four promises:

- **Permanence** — the operation stack and command stack make the design and its
  history indestructible; measurements keep intent alive across re-derivation.
- **Memory ∝ surface** — the interval bound decides cheaply, the two-layer chunk
  stores only skins, surface-only records extend the same law to the GPU, and the
  density bound lets the skin itself be stored bit-per-voxel.
- **Time ∝ what changed / what's visible** — the bounds tree scopes edits, the
  resident cache reuses the untouched, the atlas re-uploads only touched slots, the
  sorted records and pyramid decouple frames from documents.
- **Stability under concurrency and scale** — shared immutable handouts, generation
  stamps, seam flags, and explicit frames remove, by construction, the bug classes
  (torn state, time-travel installs, neighbour peeking, frame drift) that vigilance
  alone never removes.
