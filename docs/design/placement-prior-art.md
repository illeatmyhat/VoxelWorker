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

## What to do with this

1. **Decide the picking-plane question first** — fixed ground plane versus view-aligned plane at
   a movable depth. Everything else here is downstream of it.
2. Keep the four cursor states; they are the one place this design leads.
3. Keep the orthographic derivation; it is the majority practice.
4. Keep the angular framing, but **restate 1/80 as a chosen floor**, note it is below WCAG
   minimums, and expect to tune it by feel.
5. If a plane fallback survives the first decision, switch its guard to an **angle before
   projection**, and substitute a plane rather than clamping, so there is no dead zone.

Not settled here; recorded so the decision is taken deliberately rather than by default.
