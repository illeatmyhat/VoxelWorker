# Placement: what other tools do, and where ours diverges

Researched 2026-07-20, after `crates/raycast/src/placement.rs` was already written. The headline
is uncomfortable and is stated first because it challenges a premise this project has been
carrying, not a detail.

## The finding: the mainstream does not place against a fixed ground plane

The grammar in `direct-manipulation.md` says the picked point is *the nearer of the ray's hit on
geometry and its hit on the ground plane*. **That is not the standard primitive. It is the thing
mature tools spend their complexity budget escaping.**

| tool | what a placement lands against |
| --- | --- |
| Fusion 360 | "place on the ground" is **not a primitive** — you pick a plane, and the ground is one of three equal origin planes |
| Rhino 8 | **CPlane**, a first-class movable object, plus Auto CPlane aligning to selected geometry (with a lock, because auto-inference is volatile) |
| SketchUp | no plane object at all — the inference engine resolves it per stroke |
| Unreal Modeling Mode | Ctrl+Click **repositions the drawing grid** to the surface under the cursor |
| Blender Add Object | Depth = Surface, and *"if there is no surface, it does the same as Cursor Plane"* — a plane through the user-movable **3D cursor**, not world Z=0 |

Blender could trivially have defaulted to the world ground plane. It deliberately did not.

### And the grazing case is structural, not incidental

Blender's `ED_view3d_win_to_3d` uses `rv3d->viewinv[2]` — **the view axis** — as the picking
plane's normal. The plane is therefore always exactly perpendicular to the rays, so the
ray-plane denominator can never vanish. **There is no grazing case because there is no arbitrary
plane.** Clicking empty space in Front Ortho leaves the cursor's depth coordinate unchanged.

If placement used a view-aligned plane anchored to a movable depth (the orbit pivot, the last
placement, a 3D-cursor equivalent), with geometry hits overriding it:

* the horizon-flight problem does not arise,
* the distance clamp largely evaporates,
* and the "too far to author" state becomes much rarer.

**This is the decision to take before wiring the current placement code into the viewport.**

## Where our design was validated

### The angular bound is right, and standardised

"Perceptual, not pixel-based, and the display's angular size cancels" is exactly the **W3C CSS
reference pixel**: 1 px is *defined* as a visual angle of 0.0213° (the angle subtended by 1/96
inch at 28 inches), specifically so it is invariant across displays. Unreal's LOD **Screen Size**
uses the same normalized family. The rejection of "a block must cover N pixels" has normative
precedent.

**But 0.3° is a chosen number, not a derived one**, and the literature brackets it awkwardly:

| source | threshold | ≈ CSS px |
| --- | --- | --- |
| Hourcade & Bullock-Rest, CHI 2012 | pointing degrades steeply below **3 arcmin** (0.05°) | ~2 |
| WCAG 2.5.8 Target Size (Minimum) | **0.51°** | 24 |
| WCAG 2.5.5 Target Size (Enhanced) | **0.94°** | 44 |
| visionOS (WWDC25 §303) | **2.5°** | — |

Ours is ~14 CSS px — **below every accessibility minimum**, six times above the raw pointing
cliff. Defensible as a "you can still author" floor rather than a comfortable one, but it should
be described that way rather than as a derivation. "Typical display ≈ 25°" also ranges 20–30° in
practice, so 1/80 is really 1/65–1/100.

**No 3D tool gates authoring by apparent size at all.** SketchUp's authorability limit is
*model-space* (faces fail below ~1/1000 inch) and is disliked enough that the community invented
the "Dave Method" to work around it. Our rule is **novel** — arguably better than SketchUp's, but
unprecedented, and the constant is arbitrary.

### The four cursor states are novel and well-founded

**No shipped tool distinguishes "nothing there" from "too far."** In Minecraft the entire
vocabulary is the selection wireframe appearing or not — one bit, and out-of-range is
byte-identical to empty air. In Luanti the raycast is built as eye + direction × range, so a miss
yields `pointed_thing.type == "nothing"`: the two states are indistinguishable **in the data
model**, not merely in the presentation. SketchUp has ~25 named inference types on three
simultaneous channels — a very rich *positive* vocabulary and **zero** documented negative
states.

The literature disagrees with all of them. Vermeulen et al. (CHI 2013) frame a placement cursor
as answering *"what happens if I click here"*; **"nothing" is strictly weaker than "nothing,
because it is too far,"** because only the second names the corrective action. NN/g's guidance
says an unavailable affordance must explain why. Keeping the two states separate is ahead of the
field, not behind it.

### Deriving orthographic scale from orbit distance is standard

Verified in three independent codebases:

* Blender `camera.cc`: `params->ortho_scale = rv3d->dist * sensor_size / v3d->lens;`
* Godot `node_3d_editor_plugin.cpp`: `float height = 2.0 * cursor.distance * Math::tan(half_fov);`
* Rhino/openNURBS `ChangeToParallelProjection` scales by `target_distance/m_frus_near`

**SketchUp is the counterexample and it is a bug factory** — `Camera` stores `height` and `fov`
as independent fields, and users report a camera jump on projection toggle. Our derivation is the
majority practice for a good reason.

Worth stealing: Blender's `ED_view3d_pixel_size` is one branchless expression for
world-units-per-pixel, and in orthographic the perspective divide is identically 1 — so ortho
pixel size *equals* perspective pixel size at the orbit pivot. That identity is what makes the
"one rule covers both projections" claim rigorous rather than coincidental.

## Where our design was wrong

### The clamp introduces a dead zone; Blender's substitution does not

Blender's interactive Add Object tool (`view3d_placement.cc`) carries this verbatim:

```c
/* Dot products below this will be considered view aligned.
 * In this case we can't usefully project the mouse cursor onto the plane,
 * so use a fall-back plane instead. */
static const float eps_view_align = 1e-2f;
```

When it trips, Blender **substitutes a camera-facing plane** through the drag origin, then
rotates the result back onto the target plane about the two planes' intersection line and calls
`closest_to_plane_v3` to land it exactly on. So Blender confirms our law — *the point must stay
on the plane* — but its fallback **preserves continuity of mouse motion**.

Our clamp slides the point toward the orbit target until it reaches the limit. Past that limit,
large mouse movement produces no preview movement: a **dead zone**. Blender's has none.

### The threshold should be an angle, evaluated before the intersection

`eps_view_align = 1e-2` is a **~0.57° angular threshold on |view · plane normal|**, tested
*before* projecting. Ours is a distance tested *after*. The angular form is cheaper, degenerates
gracefully, and is the only source-verified constant anyone has published for this problem.

There appears to be **no published treatment of the horizon-flight problem** — searched for
directly, nothing found. Textbook ray-plane code guards only `|denominator| < 1e-6`, which is a
numerical guard and does nothing about placement sanity.

### The real CAD answer is prevention, not clamping

Fusion 360 ships **Auto Look At Sketch** enabled by default, rotating the camera to face the
plane on selection. AutoCAD documents flatly that you *cannot* draw or edit in the XY plane from
a side view. Nobody resolves the edge-on pick: they reorient the view or decline the operation.

**A "look at the plane" nudge may be worth more than any clamp.**

## Three consequences that only appear once you take the finding seriously

### "Too far to author" may be an artifact of the design we are leaving

If the picking plane is **view-anchored**, depth is bounded by the pivot rather than by the
horizon — so the state may simply not arise. **Two of the four cursor states could be artifacts
of the fixed-ground-plane world.** Do not build a four-state vocabulary for a premise that is
under review; settle the plane question first, then see which states survive.

### In orthographic the failure is the ray ORIGIN, not the denominator

Blender's ortho branch performs no intersection at all — `lambda = ray_point_factor_v3(...)`, a
dot product. The known bug there (Blender #101347, curve-draw picking a random Z in ortho) has a
root cause worth internalising: **edge-on in ortho, every ray grazes simultaneously, and what is
ill-defined is where the ray starts.** A distance clamp cannot fix that, because it is not a
distance problem.

Blender's actual behaviour: clicking empty space in Front Ortho leaves the 3D cursor's Y
*unchanged* — it reuses the cursor's own prior position as the depth reference, falling back to
`rv3d->ofs`, the orbit pivot. The manual's prescription is not to solve the invisible axis at
all, but to use two perpendicular views.

### `ED_view3d_pixel_size` is one branchless expression, and that is the point

```c
return mul_project_m4_v3_zfac(rv3d->persmat, co) * rv3d->pixsize * U.pixelsize;
```

**No `is_persp` branch.** In orthographic the perspective-divide factor is identically 1 and it
collapses to the constant `dist * sensor / lens` — so **ortho pixel size equals perspective pixel
size evaluated at the orbit pivot.** That identity is what makes our "one rule covers both
projections" claim rigorous rather than coincidental, and it means every screen-space threshold
(the 1/80 rule, gizmo hit radii, snap tolerances) can come from a single function.

Two guards to copy with it: clamp `zfac` to 1.0 when the reference point is within ±1e-6 of the
view plane, and flip sign for points behind the camera.

## A citable alternative to 1/80

WCAG 2.5.8's 24 CSS px is **0.51°**, which works out to **1/50 of viewport height** — roughly
1.6× our figure, and defensible by citation rather than by assertion. If the constant is going to
be arbitrary either way, it may as well be arbitrary in a direction someone else has already
argued for.

## The decision (2026-07-20): view-aligned plane at a movable anchor

Taken by the owner after the finding above. **The fixed infinite ground plane is abandoned.**

The question that made the choice easy was "how does that mesh with placing objects on top of
surfaces?" — and the answer is that it does not have to, because the two are not the same
question:

* *Surface placement* answers **the ray hit something; where and how does the object sit against
  it?**
* *The picking plane* answers **the ray hit nothing; what depth do I invent?**

Blender ships exactly this separation and names it in the interface — `Depth: Surface | Cursor
Plane` and `Orientation: Surface | Default` are two independent dropdowns. Conflating them is
what makes the ground plane look load-bearing when it is not.

### Resolution order

1. **The ray hits geometry.** Place against the hit face, offset along its normal,
   lattice-snapped. This is the dominant case and it is *entirely unaffected* by the plane
   question.
2. **The ray hits nothing.** Intersect a plane perpendicular to the view axis, through a movable
   depth anchor.
3. **The ground plane is not special.** It is a surface if something is there, and nothing if not.

### Why this is cheaper for us than for Blender

Blender's surfaces are arbitrary triangles, so "on top of" needs a normal, a tangent frame, and a
tie-break for the remaining degree of freedom. **Ours are axis-aligned block faces.** The hit
normal is one of six axes and the result snaps to the lattice, so the orientation problem largely
evaporates. What remains is Minecraft's question — *which side of the hit face* — and the face
normal already answers it.

### The anchor is what makes this one system rather than two modes

**A surface hit updates the depth anchor.** Place against a face, and the fallback plane now sits
at that depth; drag off the edge into empty space and the next placement continues at the depth
you were just working at, rather than snapping back to the orbit pivot. Blender's 3D cursor
behaves this way when snapped to a surface, and it is the reason the fallback does not feel like
a separate mode.

Anchor precedence, most recent wins: last placement → orbit pivot. There is no world-origin term.

### What this costs the four cursor states

`TooFar` is **not** eliminated, contrary to the first reading of the finding above.

* On the **empty-space** path it dies. Depth is bounded by the anchor, so the ray cannot fly to a
  horizon, and the state cannot arise.
* On the **geometry** path it survives. A block face 500 blocks out is still sub-pixel and still
  not worth authoring at, so the angular rule keeps a job — a smaller one, on one path instead of
  two.

`NoSurface` narrows correspondingly: it now means *nothing was hit and the anchor plane is also
unusable*, which is close to unreachable. Confirm that before building an affordance for it.

**Confirmed unreachable, and deleted.** With the view axis as the plane's normal the ray-plane
denominator is `dot(ray_direction, view_direction)`, which a perspective frustum bounds well away
from zero and which orthographic pins at exactly 1. The only input with no intersection is a ray
exactly perpendicular to the view axis, which no camera can produce. `anchor_plane_hit` is
therefore **total** and `PlacementTarget` has three variants, not four.

One precondition falls out of it and belongs to the anchor policy rather than to placement: a
last-placement anchor can end up *behind* the eye after an orbit, and a plane behind the eye puts
the preview behind the viewer. Placement cannot detect this — under orthographic the ray origin
sits on an arbitrary near plane, so the sign of the depth says nothing about the viewer — so
whatever selects the anchor owes the check, falling back to the pivot (which satisfies it by
construction).

`TooFar` moved rather than shrank: it is now asked **per hit**, at whichever depth the answer
landed at, through an injected `depth_is_authorable` predicate. At the orbit pivot's depth that
predicate is identically `can_author_at_all`, so the old camera-level question is the anchor-path
special case of the new one rather than a separate guard asked before the ray.

## Superseded 2026-07-20 (same day): three world planes, not one view-aligned plane

The view-aligned plane above was reconsidered the same day and **replaced**, after two further
prior-art passes the CAD study had not covered:

* `voxel-placement-prior-art.md` — the block-*game* shelf. Mass-market builders are
  face-or-nothing; only the tools that removed the ground (Space Engineers, Dreams) reinvented a
  movable view-ray depth.
* `orbit-editor-placement-prior-art.md` — the block-*editor* shelf. The flagship (Blockbench)
  does not raycast to create at all; the one genuine invention (Crocotile3D) **snaps the draw
  plane to the axis-aligned plane the camera faces most squarely**, so it can never graze.

The owner's decisive reframing: this app authors architectural assets **for block worlds that
have a ground**, so a ground plane is authoring *in context*, not a CAD relic — and the view-
aligned plane's wandering normal was the thing that felt unpredictable. The synthesis keeps the
ground and kills grazing without a wandering normal.

### Position — the picked point, resolved in order

1. **Geometry hit** → place against the face (unchanged).
2. **A custom (user-created) plane the ray faces** → place on it. **If a custom plane is coplanar
   with a world plane, the custom plane wins** — it carries orientation that the world plane does
   not (see below).
3. **The three built-in world-origin planes**, ground-privileged: use the **ground** (`+Z`, we are
   Z-up) unless the ray grazes it past a threshold; then defer to whichever **vertical** plane
   (`+X` or `+Y`) the ray faces more squarely. Not "best-faced always wins" — the ground has
   priority and the verticals are only a grazing fallback.

### Why this is provably total, and never grazes

A unit ray direction cannot have all three components small: `max(|d.x|, |d.y|, |d.z|) >= 1/√3
≈ 0.577`. So the best-faced of three orthogonal planes always has a healthy denominator (worst
case ~54.7°, the body diagonal) — **you can never graze all three at once.** The view-aligned
plane dodged grazing with a wandering normal; the three fixed normals plus a selection kill it
outright, which is the predictability the ground plane was wanted for. The totality proof moves
up from the single plane to the *selection*, and gets stronger: it becomes the max-component
bound, not a frustum-half-angle argument.

### Orientation — separate from position, and world-fixed for the built-ins

**Verticality is a product value, so the built-in planes are a *positioning* device only — they
never rotate the placed object.** A block located via the vertical grazing-fallback still stands
upright. This is Blender's `Orientation` axis set to **Default** for the built-ins:

* Geometry hit → orient to the **face normal** (`Orientation: Surface`).
* Custom plane → orient **normal to that plane**, camera picks the side (`Orientation: Surface`).
* Built-in world plane → **world-vertical**, never the plane's own normal (`Orientation: Default`).

### Still open (feel knobs, not blockers)

* **The ground's grazing threshold** — how oblique before it hands to a vertical. Start ~20°, tune.
* **The vertical fallback's locus** — the three built-ins are at the **world origin** (owner's
  words), the provisional default; the wrinkle is that a grazing fallback then lands on `x=0`/`y=0`,
  possibly far from a structure built off-origin. Accepted for now because once the first course is
  down the geometry path dominates and the fallback is a bootstrap edge case; revisit (verticals
  through the cursor's ground point) only if detached far-from-origin placement becomes common.

### What this costs the code already written

`crates/raycast/src/placement.rs` is not wired to anything, so revising it is cheap. The
`anchor_plane_hit` ray-plane primitive survives; what changes is the caller — `AnchorPlane`'s
`view_direction` normal becomes one of three fixed axis normals chosen by the selection, the
movable `anchor_point`/behind-the-eye concern evaporates (planes are at the origin), and the Kani
totality harness is restated as the max-component bound. The `OnAnchorPlane` variant should be
renamed to name a world plane, and orientation must be carried out of `resolve_placement` (it now
depends on which plane won, not just where the point is).

## What to do with this

1. ~~Decide the picking-plane question first.~~ **Decided, then superseded — see above.**
2. ~~Keep the four cursor states, but re-derive their reachability against the resolution
   order.~~ **Done.** Three survive: `OnSurface`, `OnAnchorPlane`, `TooFar`. `NoSurface` was the
   dead one.
3. Keep the orthographic derivation; it is the majority practice.
4. Keep the angular framing, but **restate 1/80 as a chosen floor**, note it is below WCAG
   minimums, and expect to tune it by feel. The owner has ruled the exact fraction unimportant.
5. ~~The distance clamp in `crates/raycast/src/placement.rs` should be **deleted rather than
   repaired**.~~ **Deleted.** No guard survived it, not even the angular one: the angle it would
   have tested is the frustum half-angle, which is bounded by construction.

The decision is recorded here rather than as an ADR because nothing has landed in code yet. It
warrants an ADR when the viewport wiring lands, since it supersedes the picked-point definition
in `direct-manipulation.md`.
