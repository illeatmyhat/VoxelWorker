# Substrate extraction map (2026-07-13)

**Provenance:** owner direction — "break the most complex data structures out into a library as
discrete named components… logical separation between objects of computer science and pure math,
and objects of domain." A very-thorough survey of `src/` produced the inventory below. Dated
analysis input; the decision record is `docs/adr/0014`; the boundary law is in `CONTEXT.md`
(**substrate**).

**Status (2026-07-13, end of day): EXECUTED IN FULL — S0–S10 all landed** (`72bf07b` workspace →
`248f171` S1 → `c8af77d` S2 → `f0a0099` S3 → `84fac30` S4 → `251f414` S5 → `2f91a92` S6 benches →
`c4541c0` S7 → `d76cb27` S8 → `7f2d6e1` S9 → `d9dbc2c` S10). All 15 components live in
`crates/substrate` under their literature names with citations; every slice's parity oracles and
goldens passed unmodified; app lib 478/6 + substrate 72 tests (total grew from 510 to 550+ across
the extraction). Remaining from this document: the deferred cold items (palette codec kernel,
LRU spill cache — trigger: disk path goes live), the future crates section below, and the
machine-checked construction plan (10b), which starts now that tiers 1–2 exist.

**The law (short form):** a component belongs in `crates/substrate` iff it is describable
entirely in textbook CS/math vocabulary, parameterized by plain numbers/generics — never by
domain types. Domain adapters stay in the app crate at their own seams. Not for release;
for reading, naming, and isolated performance reasoning (criterion benches for hot components).

Severance legend: **Clean** = mechanical move. **Mild** = one or two injected
parameters/constants. **Kernel-only** = extract the pure core, leave the domain traversal.

## Tier 1 — clean movers, well-oracled (extract first)

| # | Component (substrate name) | Today | CS identity | Hot? | Oracles that move |
|---|---|---|---|---|---|
| 1 | `Aabb` (`aabb.rs`) | `VoxelAabb`, spatial_index.rs ~43 | half-open integer AABB: intersect/union/contains (the well-known name; integer half-open semantics explained in the definition) | foundation | (small; #2's property test exercises it) |
| 2 | `Bvh` (`bvh.rs`) | `EditBroadphaseBvh`, spatial_index.rs ~298 | BVH over AABBs (the well-known name; median-split build, flattened nodes, leaf cap 8 explained in the definition) | per edit, ~1ms @10k | matches-naive-filter property test + empty-population test |
| 3 | `LatticeKey` pack/unpack + hi/lo split (`lattice_key.rs`) | `pack_world_block_key` etc., brick_field.rs ~60–93, ~332 | biased z-major signed-3-vector → sortable u64; u64→[u32;2] | every record, per edit | round-trip + ordering asserts |
| 4 | `FieldInterval` + classification (`field_interval.rs`) | voxel.rs ~421–535 | interval arithmetic under CSG lattice ops (min/max/negate), 1-Lipschitz centre bound, 3-way classify vs threshold | per block classify | algebra tests; E1 classifier parity stays in domain |
| 5 | `DisjointIntervalSet` (`disjoint_interval_set.rs`) | `insert_run`, two_layer_store.rs ~1482 | sorted disjoint non-touching interval set (the well-known interval-set structure, cf. Boost ICL); O(1) ascending append; overlap/abut splice merge | widest-run inner loop | dense-oracle parity (cell_interval_parity_tests) |
| 6 | `GreedyCuboidDecomposition` (`greedy_cuboid_decomposition.rs`) | `decompose_into_boxes`, cuboid.rs ~28–200 | greedy 3D box growing (X-run→Y-slab→Z-slab, consumed mask, deterministic scan); exact cover | per edit per boundary block | exhaustive in-file invariant tests |
| 7 | `Rational` + gcd (`rational.rs`) | `ExactRational`, units.rs ~38–153 | sign-normalized reduced i128 rational | cold (correctness reuse) | drift/floor/ceil/round-trip tests |
| 8 | `GenerationTracker` + latest-wins `Worker<Req,Resp>` (`supersede.rs`) | workers/mod.rs ~50–144 + display/routing.rs ~328 | drain-to-latest monotonic-generation supersede, panic-contained worker loop | concurrency correctness | panic-survival + newest-wins tests |

## Tier 2 — mild severance (one injected parameter)

| # | Component | Today | Injection needed |
|---|---|---|---|
| 9 | `BitCube` (`bit_cube.rs`) — edge ≤ 64, u64-per-X-row 3D bitset: popcount, point test, run set, dense↔packed expand | `BrickOccupancyTile`, brick_field.rs ~1109 | the "occupied byte" constant becomes a parameter of expand; `rasterize_brick_occupancy` stays behind as the domain caller |
| 10 | `SlotFreeList` (`free_list.rs`) — pop-or-append stable-index slot allocator | brick_field.rs ~1387, ~1716 | generic over slot payload `T` |
| 11 | `CubeTilePacking` (`cube_packing.rs`) — linear slot → 3D tile grid, edge = ceil(cbrt(count)) | `pack_sculpted_atlas` / `sculpted_atlas_bricks_per_axis`, brick_field.rs ~1295, ~1756 | byte-write becomes a callback (shares #9's expand seam) |

## Tier 3 — kernel-only extraction (structural; do after tiers 1–2 have settled the crate)

| # | Component | Today | Split |
|---|---|---|---|
| 12 | `SparseMinMipPyramid` (`min_mip_pyramid.rs`) — fold sorted key set to coarser cells (8→64→512), sort+dedup, conservative superset | `ClipmapPyramid`, brick_field.rs ~117–327 | fold/min-mip core moves; `from_chunks` chunk traversal + solid-chunk fast path stay in domain; `from_records` becomes a thin adapter |
| 13 | `SortedKeyBitmaskMap` (`bitmask_map.rs`) — sorted keys ∥ fixed 512-bit masks, binary-searchable, per-cell fallback scalar | `BlockOccupancyMasks`, brick_field.rs ~343–430 | storage shape moves; `from_chunks` builder stays |
| 14 | `CellClassification` — the **black/white/grey cell classification** of the octree-CSG literature (Duff 1992; Samet): fold per-op field intervals under CSG combine (later-wins union), classify cell empty/full/partial against a threshold | `classify_chunk_block`, two_layer_store.rs ~381–457 | the interval fold + 3-way verdict moves (generic over an iterator of per-op interval evaluators); leaf iteration, world offsets, and per-voxel fallback stay in domain (owner ruling 2026-07-13: the well-known name applies, extract the kernel) |
| 15 | `CulledBoxMeshing` — the well-known **culled/greedy box meshing** of the voxel-meshing literature (Lysenko 2012): exposed-face determination over disjoint boxes (neighbour-solidity culling incl. seam flags) | face-culling core of cuboid_mesh.rs | the face-culling kernel moves; wgpu vertex/UV/atlas-layer/overlay assembly stays in domain (same owner ruling — previously on the restraint list, now kernel-extracted under its literature name) |

## Deferred — genuinely substrate, but cold and not wired into the live path yet

- Palette min-bits codec kernel (`chunk_storage.rs` compress/decompress heuristic) — extract when
  the disk path goes live.
- `LruSpillCache<K,V>` (`disk_chunk_store.rs`) — standard LRU; same trigger.

## Deliberately NOT substrate (restraint list)

- `LeafSpatialIndex` — its identity is "must equal the `for_each_leaf` walk"; domain-shaped by design.
- The sweep body of `streamed_widest_run_in_band` — the band partition/reduce is domain-bound;
  its pure core is #5.
- One-line linearization/floor-div helpers, `SeamSolidity` face indexing, `core_geom.rs`
  constants/newtypes — too small or purely domain vocabulary.

(Naming ruling, owner 2026-07-13: **if a structure has a well-known literature name, that IS its
substrate name** — `GreedyCuboidDecomposition`, `DisjointIntervalSet`, black/white/grey
`CellClassification`, `CulledBoxMeshing` — with the explanation living in the data-structure
definition. This ruling moved `classify_chunk_block` and `cuboid_mesh`'s kernels OFF this
restraint list into tier 3.)

## Execution shape

- Workspace: root `Cargo.toml` becomes a workspace; app crate stays at `src/`; library at
  `crates/substrate` (no gpu/oracle/tracy features; std only; rayon only if a moved component
  already uses it).
- One commit-sized slice per extraction group, each moving the component + its tests, leaving a
  domain adapter/`pub use` where call-site churn would otherwise dominate the diff. Full gate
  baseline after every slice.
- Benches (`crates/substrate/benches/`, criterion, on-demand — never in the commit gates) for the
  hot components: `BitCube` ops, BVH build+query, run-list insert/merge, box decomposition,
  pyramid fold.
- Naming: substrate components carry textbook names (table above); the domain keeps its
  vocabulary at the adapter seams. Doc comments in substrate speak CS; doc comments at adapters
  speak domain and reference architecture chapters.

Slice order: S0 workspace plumbing → S1 (#1–#3 spatial) → S2 (#4–#5 intervals + #7 rational) →
S3 (#6 greedy cuboid decomposition) → S4 (#8 supersede) → S5 (#9–#11 bit/atlas kit) →
S6 benches → S7+ (tier 3, one slice per kernel, #12–#15). Tiers 1–2 are prerequisites for
reusing the same primitives from the ADR 0013 material-atlas epic (its cell-key tiles want
`BitCube`'s row layout and the same free-list/cube-packing kit).

## Mathematical construction (map item 10b, unlocked by this split)

Owner ruling 2026-07-13: backing the substrate components with machine-checked construction
makes sense — sequenced AFTER tiers 1–2 land (verify components where they now live, don't
block the extraction on proofs). Tool fit per component, matched to what each tool proves:

- **Kani** (bounded model checking on the real Rust; harnesses in-file under `#[cfg(kani)]`;
  runs in WSL/CI, not native Windows): `BitCube` run-set/popcount/expand-pack inverses,
  `LatticeKey` round-trip + order preservation, `CubeTilePacking` index bijection. The density
  bound 1..=64 doubles as the verification bound — these checks are effectively exhaustive.
- **Creusot or Verus** (deductive proofs on the real Rust) for stateful invariants:
  `DisjointRunList` (sorted ∧ disjoint ∧ non-touching after any insert; widest-run correctness),
  `SlotFreeList` (no double-allocation, stable indices), generation-supersede (newest-wins,
  stale never accepted).
- **Lean model** (proves the mathematics, linked to code by the existing parity oracles) for the
  two genuinely mathematical statements: `FieldInterval` conservatism (the interval algebra
  bounds the CSG field — the exact-classification theorem) and `SparseMinMipPyramid`'s
  conservative-superset property.

Standing limit (reaffirmed by the FXC X3500 episode): the GPU side is not a proof target — no
source-level theorem catches a shader-compiler bug. Verification hardens substrate kernels; the
oracle gates remain permanent regardless.

## Future crates (owner-approved 2026-07-13 — trigger-gated, do NOT split early)

The crate test is the one that justified substrate: **a dependency law worth compile-enforcing**.
Two more pass it; each waits for its trigger:

- **`document`** — the authoring truth: operation stack, producers, sketch, intents, command,
  units/`Measurement`, document schema+serialization. Law enforced: the authoring-truth boundary
  (`docs/architecture/01-document.md` / ADR 0006) — truth physically cannot import display/wgpu.
  Also the natural home for shared-document versioning and the headless crate the agent-authoring
  stack links. **Trigger:** the versioned-shared-documents work, or sculpt's new Intents —
  whichever comes first.
- **`evaluation`** — the two-layer store + the interval-bound evaluator producing boundary sets.
  Law enforced: sinks consume boundary sets, never reach into evaluator internals
  (`docs/architecture/02-evaluation.md`). **Trigger:** heavy sink-side churn (the material-atlas
  epic or sculpt).

End-state dependency chain: `substrate` ← `document` ← `evaluation` ← `voxel_worker`
(display + shell + workers), each boundary carrying a named law.

Deliberately NOT crates (taxonomy ahead of need, per ADR 0014's rejection): display/shell split
(the GPU-never-truth law is already proven once document+evaluation compile without wgpu),
.vox/interchange codecs (one file today; graduates near `document` when export goes plural),
shot/oracles (already isolated by `--features oracle`), UI/workers/orchestrator (connective
tissue). **Amended same day by ADR 0015:** the "display/shell" line stands for wgpu *plumbing*
only — graphics *mathematics* got its own crates (`camera`, `raycast`); see
docs/design/graphics-crates-extraction-map.md.

## Literature anchors (owner ruling: cite the science)

Part of each component's definition of done: the substrate module doc names the textbook
identity AND cites the canonical literature, noting our variant's deviation. Anchors for the
slices (extend as verification work surfaces more):

- `MedianSplitBvh` — Kay & Kajiya 1986 (bounding-volume hierarchies); Ericson, *Real-Time
  Collision Detection* 2005 ch. 6; note: spatial-median split, leaf cap 8, no SAH — rebuilt per
  edit, so build speed beats query optimality.
- `IntegerAabb` — Ericson 2005 (half-open integer boxes).
- `LatticeKey` — Morton 1966 (space-filling linearization); Samet, *Foundations of
  Multidimensional and Metric Data Structures* 2006; note: z-major lexicographic (not bit
  interleave) so integer order == lexicographic cell order.
- `FieldInterval` — Moore 1966 / Moore–Kearfott–Cloud 2009 (interval analysis); **Duff 1992**,
  *Interval arithmetic and recursive subdivision for implicit functions and constructive solid
  geometry* (exactly our classify-under-CSG use); Hart 1996 (Lipschitz/sphere-tracing bound).
- `DisjointIntervalSet` — union-of-intervals folklore; CLRS interval material; interval-set
  containers (cf. Boost ICL); kept simple deliberately (sorted vec, splice merge).
- `GreedyCuboidDecomposition` — greedy meshing lineage (Lysenko, "Meshing in a Minecraft
  Game", 0fps 2012); optimal rectilinear box cover is NP-hard (Soltan & Gorpinevich 1993) —
  greedy non-minimal is the informed choice, cite the hardness result to justify it.
- `CellClassification` — Duff 1992 (interval CSG classification); Samet 2006 (black/white/grey
  octree node classification — our air/coarse-solid/boundary IS this trichotomy).
- `CulledBoxMeshing` — Lysenko 2012 (culled vs greedy voxel meshing); hidden-surface face
  culling folklore.
- `Rational` — Knuth TAOCP vol. 2 §4.5 (rational arithmetic, Euclid's gcd).
- generation-supersede (`CoalescingWorker` + `GenerationTracker`) — no single canonical
  name; the confluence of work coalescing / conflation, stale-while-revalidate, and a
  monotonic version counter as a lost-update guard. Cite Herlihy & Shavit, *The Art of
  Multiprocessor Programming* (2nd ed. 2021) for the monotonic-counter reasoning; note our
  variant is std mpsc + `catch_unwind` panic containment, no external primitive.
- `BitCube` — Warren, *Hacker's Delight* 2003; Knuth TAOCP vol. 4A (bitwise techniques).
- `SlotFreeList` — Wilson et al. 1995 (dynamic storage allocation survey); Knuth vol. 1.
- `SparseMinMipPyramid` — Tanner et al. 1998 (clipmap); Losasso & Hoppe 2004 (geometry
  clipmaps); Crassin et al. 2009 GigaVoxels (brick pyramid + hierarchical DDA); Amanatides &
  Woo 1987 (voxel DDA); Museth 2013 (VDB, the sparse-hierarchy prior art).
- Umbrella prior art for the whole Dreams-style machinery — Evans, *Learning from Failure*,
  SIGGRAPH 2015 (already the basis of the engine's brick-field lineage).
- Verification citations ride with the proofs: Kani (Kani/CBMC docs), Creusot (Denis et al.
  2022), Verus (Lattuada et al. 2023), and the specific theorem sources above (Duff 1992 for
  the conservatism statement).

## Addendum — category modules + later-surveyed candidates (2026-07-14)

Two follow-on surveys after the S0–S10 extraction, plus a reorganization, all landed this day.

**Category-module reorg** (`ae63d65`). Substrate's 19 flat modules were grouped into category
submodules so the taxonomy is visible at the call site (`substrate::spatial::LatticeAabb`,
`substrate::occupancy::BitCube`): `spatial/` (aabb, bvh, lattice_key, ray, min_mip_pyramid),
`interval/` (field_interval, disjoint_interval_set, rational), `occupancy/` (bit_cube,
value_cube, free_list, cube_packing, shelf_bin_pack, bitmask_map), `solids/`
(cell_classification, greedy_cuboid_decomposition, culled_box_meshing); `supersede` + `srgb`
belong to no family and stay at the crate root. Public paths carry the category (no flat
facade) — the decluttering is load-bearing, not cosmetic. Chosen over a new crate deliberately:
per ADR 0014, a crate must enforce a *dependency law*, and a pure-math decluttering introduces
none. `document`/`evaluation` remain the only future crates (they DO enforce a boundary law).

**Two new well-known-structure clusters extracted** (owner-directed second survey — "identify
well-known data structures as candidates", same lens as ADR 0014/0015):

- **`geom2d`** (`f3ae779`) — the planar computational-geometry predicates that were private
  `fn`s in `sketch.rs`: `orient2d` (signed-area/CCW; Shewchuk 1997, O'Rourke 1998),
  `segments_intersect` (CLRS §33.1), `segment_intersects_rect` (Ericson 2005), `point_in_polygon`
  (crossing-number / Franklin PNPOLY), `rectangle_inside_polygon` (connectedness). Generic over
  `[f64; 2]`; sketch keeps the `SketchPoint → [f64; 2]` adapter, converted once per resolve so
  the per-voxel loops don't re-allocate. `revolve_box_within_sweep_arc` stayed domain (its EPS
  and seam handling are tuned to the resolve's f32 `atan2`).
- **`noise`** (`4318c88`) — the procedural-generation kit from `debug_clouds.rs`: `noise::rng`
  (`SmallRng`, an LCG + Fisher–Yates shuffle; Numerical Recipes constants, Knuth) and
  `noise::perlin` (`PerlinNoise` improved gradient noise, Perlin 2002, + fBm). Kept as substrate
  (not a new crate) on the `srgb`/`ray` precedent: pure math with a WGSL mirror lives in
  substrate, and `perlin` is documented as the shader's readable CPU spec. debug_clouds keeps the
  metaball-union field + jittered octant scatter that consume the kit. Byte-preserving (identical
  RNG call sequence).

Both are behavior-preserving (substrate 92 → 106 tests; all sketch + cloud parity tests
unchanged). **Deliberately left domain** (surveyed, rejected): the relaxed-JSON normalizer
(`assets/faces.rs` — single consumer, not CS/math structure), the `BlockTypeIndex` inverted index
(`assets/vintage_story.rs` — domain-baked scoring, one consumer), `resize_rgba_nearest` (small,
fold in opportunistically), and vox_export's TLV framer / atomic-write idiom (marginal). None
introduces a dependency law; none warrants a crate.

**Third scan (2026-07-14, `fcbf278`).** A follow-on survey of the un-surveyed display/shell/UI
files (renderer, main, gpu, panel, app_core, orchestrator, routing) and a residual sweep of the
already-mined big files (two_layer_store, brick_field, brick_raymarch, voxel, cuboid_mesh, units,
chunk_storage). One extraction taken: **`Rational::to_terminating_decimal`** (was
`units.rs::decimal_string`) — the base-10 terminating-decimal criterion (2/5-smooth denominator)
+ exact power-of-ten expansion; pure number theory, distinct from Rational's gcd/reduction, landed
as a method on the existing type. Surveyed and **deliberately left**: the Pineda edge-function
rasterizer + Porter–Duff compositing in `renderer.rs` (real textbook kernels — Pineda 1988, Porter
& Duff 1984 — but the guts of the restraint-listed chrome-glyph rasterizer; owner declined to
reconsider the restraint); `voxel.rs::signed_distance_box` (exact sdBox — clean, but extracting
only the box splits the SDF family, whose siblings are approximate and stay domain); the
3×3×3 Moore-neighbourhood dilation in `cuboid_mesh.rs` (real name, but a 12-line loop whose
function identity is the domain rebuild-plan). The display/shell files are otherwise wgpu
plumbing + UI glue with no severable math. **The component hunt is considered complete** — three
scans have driven the yield down to formulas and restraint-listed kernels.

**Tooling note (2026-07-14):** structural rewrites here used **ast-grep** (installed) + the
compiler as oracle; module *moves* have no end-to-end tool (`rust-analyzer ssr` CLI is broken
upstream — verified across two versions). See the [[refactor-tooling-astgrep]] /
[[rust-analyzer-ssr-cli-broken]] memories. A **CI doc-link gate** was added the same day
(`56b35b6`): `cargo doc --workspace` under `-D warnings`, broken/redundant links fail,
public→private permitted by a crate allow.
