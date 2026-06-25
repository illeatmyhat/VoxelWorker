# REPRESENTATION & MODES — addendum for the in-flight port (VoxelWorker)

Status note: this is **additive guidance for work already underway** in `illeatmyhat/VoxelWorker`,
not a from-scratch instruction. It's layout-agnostic — adapt names to whatever module/struct
structure already exists. The single ask: introduce one architectural seam now, while it's cheap,
so the tool can serve both CAD-style users and direct-sculpt users later without a rewrite.

## The core decision: the voxel grid is the one consumed truth

Today voxels come straight from the SDF. That quietly hard-wires an assumption that will fight a
whole class of future users. Two authoring styles want fundamentally different "sources of truth":

- **Procedural / CAD-like** (SDF tree, booleans, lathe, arrays): voxels are *derived*, recomputed
  every frame from parameters. Nothing per-voxel is stored — that's why density can rescale freely
  and booleans are cheap.
- **Direct sculpting** (paint/erase voxels on surfaces): the per-voxel grid *is* the truth; edits
  are authored data with no generating formula.

They don't blend, and the reason is concrete: **regeneration destroys authored data.** The instant
someone paints a voxel the SDF wouldn't produce, the shape is no longer `sdf(p) <= iso`, so the next
parameter change re-evaluates the field and erases their work. This is the same reason parametric
CAD and direct mesh editing are separate modes in Fusion. Don't try to make them silently coexist.

The fix is an indirection, not a feature: **the renderer must read a resolved grid, never the SDF
directly.** Both authoring styles become *producers* into that grid.

```
final_occupancy(p) = apply(overlay, evaluate(tree, p))
                     └── sculpt edits ──┘  └── procedural SDF ──┘
the renderer/slice/export only ever see `final_occupancy` (the resolved grid)
```

### What to introduce in the code now (small, even mid-flight)
- A `VoxelSource`/producer concept that yields occupancy + per-voxel data (`iLocal`, eventually
  material id), instead of the renderer calling `sdf()` itself.
- v1 has exactly one producer (the SDF tree). The point is the *seam*: renderer, 2D slice, and
  `.vox` export all consume the resolved grid, so adding a second producer later touches nothing
  downstream.

Rough shape (illustrative, adapt to the repo):
```rust
struct Voxel { local: [u8;3], material: u16 }          // iLocal etc.
trait VoxelProducer { fn resolve(&self, grid: &mut VoxelGrid); }   // writes into shared grid
// v1: SdfTree implements VoxelProducer. Renderer reads VoxelGrid, not the SDF.
```

## The three modes (ship the simplest; design for the others)

1. **Bake-then-sculpt (ship this as v1 boundary).** Procedural stays live until the user hits
   "Bake to voxels" — that snapshots the current field into an explicit grid; from then on it's a
   sculpt document and the SDF tree is frozen. Simple, predictable, and it matches the actual
   chiseling workflow: design the curve parametrically, bake, then hand-tweak individual rim
   voxels. The trapdoor (no param edits after bake) is acceptable for v1.

2. **Sparse override layer (growth path).** Keep the SDF as a base and store sculpting as a sparse
   set of force-on / force-off voxels applied on top. CAD params can still move underneath and
   hand edits persist *while they still make sense*. This is the "both at once" model and the
   reason for the producer-stack seam. Catch: when the base moves out from under an override you
   get orphaned/stale voxels — needs a rule (clamp overrides to the current bounding box; flag or
   drop stale ones). Most of the complexity lives here; don't build it until sculptor users are
   proven real.

3. **Two explicit modes with a visible bake boundary (the honest UX).** A "Design" mode
   (procedural) and a "Sculpt" mode (explicit), with a deliberate, visible bake step to cross from
   the first to the second; crossing back starts a fresh procedural base. This is option 2's
   semantics without hiding the seam, and it's the version to actually ship around once both
   audiences matter.

## The one thing to avoid

A UI that *implies* free intermixing — letting someone paint while parameters are still live
without surfacing the bake/override semantics. That's where users silently lose work and trust the
tool less. Whatever the mode, make "is this voxel procedural or authored, and what happens if I
change a parameter now" legible.

## Sequencing recommendation

1. **Now (cheap):** put in the resolved-grid-as-truth seam — renderer/slice/export consume a
   `VoxelGrid`; the SDF tree is the first `VoxelProducer`. No user-visible change.
2. **v1:** ship **bake-then-sculpt**. Bake = run the producer once into an editable grid; sculpt =
   paint/erase on that grid. Simplest correct boundary.
3. **Later, only if sculptor users materialize:** promote bake into the **sparse override layer**
   (option 2) with stale-voxel handling, optionally surfaced as explicit Design/Sculpt modes
   (option 3).

This keeps the door open for the painter/sculptor audience without forcing the painful
"everything live-parametric *and* hand-editable" problem before it's earned. It also composes with
the phase-2 SDF construction tree (booleans/lathe/array): that tree is just the procedural
producer; the sculpt overlay is a second producer into the same grid.
