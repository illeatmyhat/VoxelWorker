# 02 — Evaluation

Evaluation is the act of turning the document (a program) into occupancy (an answer).
It is the hinge of the whole system: everything above it is intent, everything below it
is presentation. The evaluator's design goal is stated as a cost envelope:

> **Per-edit cost proportional to the edit. Resident memory proportional to the
> boundary. Exactness everywhere, always.**

## The one evaluator

There is exactly one evaluator. The mesh, the brick field, the exporter, the
measurement queries — none of them evaluate producers themselves; they consume the
evaluator's output. This is Law 6 ("classified once, consumed everywhere") and it is
what makes exactness *composable*: if the evaluator is right, every sink is right,
and two sinks can never drift apart because neither has an opinion of its own.

## Block classification by interval bound

Space is sliced into fixed-size cubic **chunks** of blocks. For each block in a covering
chunk, the evaluator classifies:

- **air** — the composed field is certainly empty over the block's cell;
- **coarse-solid** — certainly full; the block exists as a block ID only, with no voxel
  data at all;
- **boundary** — the surface passes through (or no bound is available); only here does
  per-voxel evaluation run.

Classification is driven by each producer's **conservative interval bound** composed
through the boolean operators. Conservatism is the load-bearing property: the bound may
say "cannot tell" when it could have known, but it may never say "all solid" or "all
empty" wrongly. Therefore classification is *occupancy-identical* to brute-force
per-voxel evaluation — the fast path and the honest path agree everywhere, and the
proof gates in [Proof](05-proof.md) hold them to it.

A bound may come from an exact predicate or from a field, and the evaluator prefers the
predicate wherever one exists. The reason is the geometry of a cell: a distance bound
must cover a cube's corners, so it can only decide a cell whose centre is further from
the boundary than the cell's half-diagonal, while an exact containment predicate decides
any cell inside the profile however close to an edge. Levelling the two would cost a
several-voxel shell of interior that currently elides. Where an exact square-metric field
exists the covering ball *is* the cell, which tightens the bound further — one more
reason the metric is tracked rather than assumed.

**The fold is evaluated twice, over two different domains, and they are one semantics.**
Once over voxel sets, once over intervals — the second is what makes classification
cheap, and the two diverge silently if either learns an operation the other has not.
Every combine operation therefore lands in both, and a parity fuzzer holds them to
agreement; an operation added to only one is unlanded work, not a partial feature.

This is where Law 2 ("memory follows the surface") is enforced. A mountain of solid
stone is a lattice of coarse-solid block IDs; only its skin — the boundary blocks —
ever becomes voxels.

## The two-layer chunk

The evaluator's output for one chunk mirrors the host game's own storage split:

- a **coarse layer**: one block ID per block (palette-friendly, trivially compressible),
  holding every un-chiseled block;
- a **microblock layer**: a sparse map from boundary block to its sub-block geometry,
  stored **already decomposed into cuboids** — not a dense per-voxel grid — because
  boundary geometry is overwhelmingly boxy and cuboids are what both the mesher and the
  exporter want;
- **seam solidity flags** per boundary block face, so a neighbouring chunk can cull the
  faces it shares with this one without ever expanding this chunk's voxels.

The two-layer chunk is the lingua franca of the system: the single classified form that
every derivation reads.

## Residency and targeted invalidation

Chunks live in a **resident cache**. An edit does not rebuild the world:

1. The scene's leaf producers are gathered and a bounding-volume hierarchy is built
   over their world bounds — the **edit broadphase**. It is stateless, rebuilt per
   edit; at authoring scales (thousands of producers) rebuilding costs less than any
   scheme for keeping it consistent would, and statelessness eliminates the entire
   class of stale-index bugs.
2. The edit's dirty region — the union of what changed — is intersected with the chunk
   lattice, and only those chunks are evicted. The region is computed from a node's
   *effective* reach, not its producer's: a node carrying an outset dirties its dilated
   bounds, and an unbounded producer contributes the bounds of the accumulator its
   operation is confined to. A dirty region derived from producer bounds alone would be
   too small in exactly the cases that are hardest to notice.
3. The next resident handout rebuilds exactly the missing chunks (in parallel; each
   chunk builds independently from the broadphase's candidate producers) and reuses
   every untouched resident verbatim.

The handout itself is an owned list of `(chunk coordinate, shared chunk)` pairs, where
"shared" means reference-counted: handing the covering set to the mesher, the brick
builder, and an async worker costs one refcount bump per chunk, never a copy. Whoever
still holds a set holds immutable, internally consistent data, however long ago it was
handed out — which is what makes the concurrency in [Work](04-work.md) safe.

## Frames and the floating origin

Scenes are large enough that world coordinates must be wide integers, and rendering
wants coordinates near zero. The resolution is a **recentre**: an integer voxel offset,
computed at placement, that every derived artifact is expressed relative to.

The law (Law 5) is that the recentre — like every spatial value — **travels with the
data it frames**. A chunk is chunk-local and integer; a mesh bakes the recentre into
its vertices at emit time; a display cache records the recentre it was built at. No
consumer ever re-derives a frame from the scene "because it knows how" — the moment two
code paths derive the same frame independently is the moment they can disagree, and a
half-voxel disagreement in a chiseling tool is not a cosmetic bug.

A floating-origin shift is therefore cheap and safe: resident chunks stay valid (they
are chunk-local), and only artifacts that *baked* the old frame are rebuilt.
