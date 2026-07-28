# Tool modes & navigation — the app-wide interaction model

How the left mouse button, the tool modes, and the camera work across the *whole* app (not just
sketch). Decided with the owner 2026-07-23. This is a foundational pivot: it changes the default
left-mouse verb and the camera bindings every mode depends on, so it is captured before any code.
It **supersedes** "left-drag orbits" in `docs/design/direct-manipulation.md` and subsumes the
ADR 0028 sketch rail into a global mode set. Still a living spec; graduates to an ADR (likely two —
one for tool modes, one for navigation) once the model is complete and sliced.

The reference is **Fusion 360 / Maya-family** conventions, in the owner's words.

## The pivot: left mouse selects, it does not orbit

Today left-drag orbits the camera and scene nodes are selected in the browser. That inverts: **the
left mouse button's default verb is Select**, in the viewport, for scene nodes *and* sketch entities.
Orbit moves off the left button entirely (see Navigation). This is the single change everything else
hangs off.

## Tool modes (Q / W / E / R)

A **global** mode set over the current selection, the Maya/3ds-Max industry shortcuts. Every mode is
"select **plus** a manipulator"; Q is select alone.

| Key | Mode | Manipulator on the selection | Sketch mode |
| --- | --- | --- | --- |
| **Q** | **Select** | none — pick / marquee / shift-accumulate only | as shipped (slice 1) |
| **W** | **Select + Move** | translate gizmo (position handles) | the constraint-mediated move |
| **E** | **Select + Rotate** | rotate gizmo | invalid → disabled or remapped (TBD) |
| **R** | **Select + Scale** | scale gizmo | invalid → disabled or remapped (TBD) |

- The modes are **global** — they act on whatever is selected (scene nodes in normal mode, sketch
  entities in sketch mode), so selection (Q) is the shared substrate.
- **W/E/R are defined by their manipulators.** Translate / rotate / scale gizmos on the selection do
  **not exist today** (the only "gizmos" are the sketch vertex handles and the camera axis-guide).
  This is a new **transform-gizmo subsystem**, the bulk of the W/E/R work — a keybinding is the easy
  part.
- **Sketch remaps:** W is the constraint-mediated move (`docs/design/sketch-selection.md` — a *request*
  the solver corrects). E (rotate) and R (scale) are not meaningful on a 2D lattice profile yet;
  they are disabled or remapped to something sensible. **Open — what, if anything, E/R map to in
  a sketch.**

## Navigation — the camera (Fusion model)

Orbit is **not** on the left button. There are two **orbit sub-modes** and two ways to enter orbit,
around two different pivots.

### Orbit types (owner-resolved 2026-07-26, Fusion naming)

- **Constrained Orbit** — keeps the world-up fixed, so the camera never rolls (the turntable).
  The **default**.
- **Free Orbit** — full trackball, roll allowed.
- The active type is a **most-recently-used session variable** (never Settings, never the
  document), shared by every orbit entry path — Shift+MMB and explicit orbit mode both perform
  whichever type was last used. The two types share the orbit logic; the entry paths differ
  only by pivot (see below).
- **UI (Fusion's split button):** the display icon rail holds an orbit button whose face is the
  MRU type, with a dropdown offering the other (Free Orbit lives only there); the context menu
  offers Constrained Orbit.
- **Camera representation:** whichever parameterization is most durable to degenerate cases
  (gimbal lock at the poles) — i.e. orientation-first (quaternion), with theta/phi as a derived
  readout for the view cube / Home persistence, not the storage.

### Entering orbit — two paths, two pivots (owner-resolved 2026-07-23, restated 2026-07-27)

There really are **two pivots**, one per entry path — they do not share a point. What separates
them is **which mechanisms may move each one**, and nothing else: the orbit math is identical.

1. **Shift + Middle-mouse → the orbit center.** Hold Shift+MMB to orbit about the **orbit
   center**: a point put down by a deliberate act (the general context menu's **place / reset
   orbit center**, which raycasts a surface — geometry or a visible picking plane) and moved by
   *nothing else*. Panning does not move it. Zooming does not move it. That is the whole
   feature: slide the view across the model and the thing you are inspecting stays the thing you
   turn around. Until a center has ever been placed it reads as `camera.target`, so a fresh
   document turns about what it is looking at. (Plain MMB stays **pan**.)
2. **Explicit orbit mode → `camera.target`.** Entered by a button in the **display-settings icon
   rail** or the **context menu**. A **targeting reticle** overlays the viewport; **LMB-drag
   orbits about `camera.target`**, and an **LMB-click raycasts a surface and sets `camera.target`
   to the hit — a pan** that re-centers the view on it. Every non-Shift+MMB mechanism (this mode,
   the view cube, zoom) orbits/operates about `camera.target`. Leaving the mode restores
   LMB = select. This mode is **independent of the orbit center** and never writes it.

> **Read this before editing the section above.** The two pivots are easy to collapse into one,
> and doing so has already cost a shipped-then-reverted binding. The write-up briefly said
> Shift+MMB orbits "the surface point under the cursor at press, raycast per gesture, never
> stored" — a misreading of *"Shift+MMB is always for the clicked surface point"*, where
> **clicked** means the click that **placed** the center, not the press that starts the drag.
> There is no transient pivot anywhere in this design.

### The rest

- **Pan:** middle-drag (unchanged).
- **Zoom:** scroll wheel (unchanged).
- **View cube:** the existing corner cube stays a separate orbit-to-face affordance.
- The **orbit center** is a new concept the context menu must let you place and reset.

## What is new vs what exists

- **New:** viewport click-to-select for scene nodes (today browser-only); the transform-gizmo
  subsystem (W/E/R); the orbit-center concept + its context-menu place/reset; the explicit orbit
  mode + its rail button + raycast-recenter; the Q/W/E/R global mode state and keybinds; rebinding
  orbit off LMB onto Shift+MMB / orbit mode.
- **Exists / reused:** middle-drag pan; scroll zoom; the view cube; the sketch selection (Q) from
  slice 1; the sketch vertex-drag (becomes the sketch W move); the general context menu (slice 2 of
  the sketch epic — now also hosts orbit-center place/reset and the orbit-mode toggle).

## Open questions

1. ~~Orbit type toggle + default~~ — RESOLVED 2026-07-26 (see "Orbit types": Constrained
   default, MRU split button, session variable).
2. Sketch E/R: disabled, or remapped — and to what. (Still open; deferred with the W/E/R epic.)
3. ~~Orbit-mode legibility~~ — RESOLVED 2026-07-26: the targeting reticle overlay IS the
   flipped-verb affordance.
4. ~~Q/W/E/R × armed placement~~ — RESOLVED 2026-07-26: arming is a **transient overlay on the
   current mode**, never a mode (CONTEXT.md "Armed placement"); a mode key disarms first.
5. Scope / sequencing (below).

## Q-slice scope (owner-resolved 2026-07-26)

**In:** unified `Selection` (one mixed-kind set; `Scene::active` + `active_point` die; session
state, undo steers it as an effect — CONTEXT.md "Selection"); the picked-node resolver
(CONTEXT.md "Picked node"); LMB = select + shift-click accumulate; orbit rebind (Shift+MMB about
the orbit center, plus the context menu's place/reset that puts it down); projection mode
reclassified Settings→session.

This scope box used to also list the explicit orbit mode + reticle, the orbit-type split button
and the quaternion camera, which contradicted sequencing step 5 below. Step 5 wins: those three
need surfaces (the rail) and a camera refactor that the Q slice does not.

**Deferred:** scene-node marquee (built once with the sketch marquee, sequencing step 3);
viewport Point picking (Points enter `Selection` as targets now, gizmo hit-testing later);
multi-select inspector editing (N>1 shows a count summary only).

**Selection feedback (owner-resolved 2026-07-26):** a selected node renders with a **cel
shader over its geometry** (outline-emphasis, depth-tested against the composed model) — not a
translucent x-ray tint. The node's own body is derived per selection change via the same
bounded per-node derivation the operand ghost uses; only the shading differs. Applies in all
three view modes.

## Sequencing (owner-ordered 2026-07-23)

1. **General context menu** — ✅ SHIPPED (`9b074c8`). Right-click in the viewport opens a
   mode-dispatched menu; Delete (warn-red ✕) removes the sketch selection (sketch mode) or the
   active node (normal mode). The shared surface the orbit-center place/reset and the orbit-mode
   toggle will later hang off.
2. **Reorganize the "Q" subsystem** — make selection work correctly app-wide: left = select for scene
   nodes (viewport picking) + the Q mode + rebind orbit off LMB (Shift+MMB about the orbit center).
   Getting selection *right* here is the point, before layering more on it.
3. **Back to sketch selection** — the remaining `sketch-selection.md` slices (delete-as-action,
   marquee) once Q is solid.
4. **Follow-up epics (deferred):** the **W/E/R transform-gizmo subsystem** (translate/rotate/scale for
   scene nodes) and the **sketch E/R remapping** — both explicitly punted by the owner to a later
   pass, after the Q selection system is correct.
5. Explicit orbit mode + orbit center, and the sketch W move — fold in around 2–4 as their surfaces
   (the context menu, the rail) land.
