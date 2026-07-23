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

### Orbit sub-modes

- **Axis-constrained** — keeps the world-up fixed, so the camera never rolls (the turntable).
- **Free orbit** — full trackball, roll allowed.
- **Open:** how the two are toggled, and which is the default.

### Entering orbit — two paths, two pivots

1. **Shift + Middle-mouse (transient).** Hold Shift+MMB to orbit about the **orbit center** — a
   persistent point the camera pivots around, **placed / reset via the general context menu**. (Plain
   MMB stays **pan**, which the app already has.)
2. **Explicit orbit mode (persistent).** Entered by a button in the **display-settings icon rail** or
   the **context menu**. In this mode the left button *is* orbit, and it is **independent of the orbit
   center**: **left-clicking geometry raycasts and pans the camera to that hit point**, making it the
   pivot, so an **LMB-drag then orbits around whatever you clicked**. Leaving the mode restores
   LMB = select.

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

1. Orbit sub-mode toggle + default (axis-constrained vs free).
2. Sketch E/R: disabled, or remapped — and to what.
3. Does entering explicit orbit mode change the cursor / chrome so the flipped LMB verb is legible?
4. Interaction of Q/W/E/R with the existing **armed placement** flow (the "+ Add" ghost-follows-cursor
   tool) — is arming a placement a transient state on top of a mode, or its own mode?
5. Scope / sequencing (below).

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
