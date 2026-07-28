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
- The **default type** is a **most-recently-used session variable** (never Settings, never the
  document). The two types share the orbit logic; the entry paths differ only by pivot (see
  below).

- **Which entry paths write the default (owner-resolved 2026-07-27, from Fusion's behaviour).**
  The MRU is the *default* type, not "the type", and the separator is whether the entry path
  **names a type**:

  | Entry | Names a type? | Orbits as | Writes the default |
  | --- | --- | --- | --- |
  | Shift + MMB | no | the default | no |
  | Rail split-button face | no | the default | no |
  | Rail dropdown pick | yes | the picked type | **yes** |
  | Context menu → Constrained Orbit | yes | Constrained | **no** |

  The rail dropdown writes because that is what a split button *is*: picking re-faces the button,
  which is the same act as setting the default. The context-menu item is not a settings control
  at all — it is a **command that invokes a tool**, and invoking a tool has never meant "make this
  the default". Fusion behaves exactly this way, and the earlier line here ("shared by every orbit
  entry path") was true only of the unqualified paths.

  This is the same shape as the two pivots below: things identical in effect, separated by *which
  mechanisms may write the persistent value*. Name it; never merge it.

- **Override lifetime (owner-resolved 2026-07-27): the orbit-mode session.** A type-naming command
  runs its type until you leave the mode, then the default reasserts. Not one drag — an override
  the user cannot see the boundaries of is one they cannot reason about, and "mode" already means
  a state with an exit.

  Consequence for the UI: while an override is active the rail button's face is *not* what the
  next orbit will do. It must show the ACTIVE type as distinct from the default, or the face lies
  for the duration of the mode.

- **UI (Fusion's split button):** the display icon rail holds an orbit button whose face is the
  default type, with a dropdown offering the other (Free Orbit lives only there); the context menu
  offers Constrained Orbit.

  *Shipped 2026-07-28 as a real split button.* The rail's fourth button is divided by a hairline
  into a glyph face and a narrower caret half, each hovering on its own; the face carries a
  distinct glyph per type (`orbit-constrained` / `orbit-free`) and lights on the accent while the
  default is Free — the same non-default signal the viewport-mode button uses.

  *Completed 2026-07-28 (slice D).* The face now **toggles the explicit orbit mode** and the caret
  half alone opens the type menu, so the two halves finally do different things. The face draws the
  ACTIVE type — the override's, while one is running — per the rule above, and its tooltip names
  both when they differ ("Leave free orbit — this mode only; the default stays constrained").
- **Camera representation (owner-resolved 2026-07-27, REVERSING the 2026-07-26 line below):**
  **two representations, Constrained is primary.** The spherical chart (`theta`/`phi`/`roll`)
  stays the stored truth with its own math; the quaternion/trackball is a **secondary**
  representation with its own dedicated operations, authoritative only while Free Orbit is the
  active type. Exactly **one is authoritative at a time**, keyed off the MRU type — never both
  live and synced, which would be the two-truths bug (and an F9 dump would capture whichever was
  stale).

  *Superseded:* "orientation-first (quaternion), with theta/phi as a derived readout" —
  owner-resolved 2026-07-26, reversed the next day. Why the reversal: quaternion-primary makes
  every `theta`/`phi` consumer convert on day one — `HomeView`, `SnapTween` (which *writes*
  angles over time), the view-cube snap tables, `is_face_constrained`, config persistence — and
  that migration is the expensive part of the work, not the integrator. Chart-primary pays none
  of it until Free Orbit ships, and Free's cost is then purely additive. The competing claim,
  that quaternion storage is more durable at the poles, does not survive contact: the pole
  problem it solves is a *trajectory* continuity problem, and the seam below only ever converts
  at discrete events.

- **The seam.** `theta`/`phi`/`roll` is a proper chart of SO(3), so **both conversion directions
  are exact**. Constrained → Free evaluates the chart forward: no choice, no loss. Free →
  Constrained inverts it — unique away from the poles, and *at* a pole `theta` and `roll` are not
  individually determined (only their combination is), which is gauge freedom, not information
  loss: every consistent choice reproduces a bit-identical view. Resolve it the way
  `nearest_equivalent_theta` already resolves its equivalents — take the `theta` nearest the last
  chart `theta` and let `roll` absorb the remainder.

  Gimbal lock does not enter. Lock is a failure of continuity *along a trajectory*; a type switch
  is a single point. **Guard:** the active type may never change mid-gesture, so the conversion
  stays a point and never becomes a path.

- **Re-levelling on Free → Constrained (owner-resolved 2026-07-27): animate it.** A free orbit
  leaves accumulated `roll`, and Constrained Orbit's promise is that world-up stays up, so the
  switch drops the roll to zero. That is the one deliberately lossy step in the seam, and a hard
  cut reads as a glitch — so it runs as an eased `SnapTween`, the same machinery Home already
  uses to re-upright roll. Animated, it reads as intent.

- **What shipped 2026-07-27 (slices A–C).** `crates/camera/src/free_orbit.rs` holds the trackball
  integrator and the seam: `free_orientation: Option<Quat>` on `OrbitCamera` IS the authority bit
  (no second flag to disagree with it), `orbit_by_drag_as` / `orbit_about_point_as` are the one
  door both types go through, and `ensure_free` / `ensure_constrained` are idempotent so the
  shell's type variable can be policy and the `Option` mechanism without the two ever being proven
  in sync. `direction()` and `up_vector()` dispatch; `is_face_constrained` was rewritten against
  those two vectors instead of the `roll` field so it answers under both types.

  Chart-native surfaces close the seam before they act — the view cube (drag, click, and its
  Home/Fit/Set-home menu) and the rail's Home/Fit, because all of them read or write `theta`/`phi`
  and a live trackball leaves those stale. So do the two persistence paths, through the
  non-mutating `OrbitCamera::as_chart`: a config or an F9 dump written mid-Free would otherwise
  record a pose the user left some time ago.

  The animated re-level plays where it is the *only* thing happening — picking Constrained from
  the rail menu. It deliberately does not fire on a chart-native op that is about to snap anyway
  (a face snap re-uprights on its own) nor mid-drag (the tween would fight the gesture).

- **Integrator before UI.** Within this slice: the orbit-type split button is vacuous until Free
  Orbit exists, and Free Orbit *is* the trackball representation. Building explicit orbit mode
  first would ship a dead dropdown and add a second gesture call-site.

- Every per-frame camera consumer (`direction()`, the up vector, `eye()`, the view cube's
  matrices, `is_face_constrained`) reads through **representation-generic accessors** — the view
  cube still renders during a Free drag and must not read a frozen `theta`/`phi`.

### Entering orbit — two paths, two pivots (owner-resolved 2026-07-23, restated 2026-07-27)

There really are **two pivots**, one per entry path — they do not share a point. What separates
them is **which mechanisms may move each one**, and nothing else: the orbit math is identical.

1. **Shift + Middle-mouse → the orbit center.** Hold Shift+MMB to orbit about the **orbit
   center**: a point put down by a deliberate act (the general context menu's **place / reset
   orbit center**, which raycasts a surface — geometry or a visible picking plane) and moved by
   *nothing else*. Panning does not move it. Zooming does not move it. That is the whole
   feature: slide the view across the model and the thing you are inspecting stays the thing you
   turn around. Until a center has ever been placed it sits at the **world origin**, and reset
   sends it back there. (Plain MMB stays **pan**.)

   While a placement is armed and while Shift+MMB is turning about it, the center draws as a
   ringed-crosshair marker — the pivot is the one camera quantity with no on-screen geometry of
   its own, so an unplaced or forgotten one is otherwise invisible. It is **continuous**, not
   voxel-snapped: it is a camera quantity with no lattice meaning, and a snapped one visibly
   jumps a cell at a time under the cursor.

   **While armed, the marker sits under the cursor — not on the surface.** It first tracked the
   raycast hit, which meant every mouse move paid for a full CPU raycast and the marker visibly
   lagged the pointer. The ray now runs **once, at the click**. The armed marker is a cursor, and a
   cursor that trails the mouse is broken however accurate it is; the surface point it will land on
   is the one under it either way.
2. **Explicit orbit mode → `camera.target`.** Entered by a button in the **display-settings icon
   rail** or the **context menu**. A **targeting reticle** overlays the viewport; **LMB-drag
   orbits about `camera.target`**, and an **LMB-click raycasts a surface and sets `camera.target`
   to the hit — a pan** that re-centers the view on it. Every non-Shift+MMB mechanism (this mode,
   the view cube, zoom) orbits/operates about `camera.target`. Leaving the mode restores
   LMB = select. This mode is **independent of the orbit center** and never writes it.

   *Shipped 2026-07-28 (slice D).* `PanelState::orbit_mode` is the state — `Off` / `UsingDefault` /
   `Named(type)`, the three-way that lets a type override live and die with the mode session
   without the default ever moving. Entered from the rail face or the viewport menu's "Constrained
   Orbit" (the naming entry), left by the rail face or by the **OK / Cancel** rows below. While it
   runs the left button is taken from selection, placement and the sketch tools alike; a press
   latches the active type (the mid-gesture guard), a drag past the click threshold turns, and a
   stationary release raycasts through the same `surface_point_at` the orbit-center placement uses.
   A miss is a **refusal**, not a fallback: the target keeps its old value rather than flying to a
   guessed one.

   The re-centre **animates**. It is the same `SnapTween` the view cube and re-level already use,
   which now lerps `camera.target` alongside the angles — one tween type, so a snap and a re-centre
   can never race to write the camera, and every angle-only constructor simply holds the target.
   A jump-cut here reads as a teleport; the ease is what makes it read as the pan it is.

   The **reticle** is a large ring — 72% of viewport height — with four cardinal ticks outside it
   and a small centre cross, scaled off height alone so it keeps its proportions at any aspect. It
   needs no projection: the camera looks *at* `camera.target`, so the target is the viewport centre
   by construction. It draws in neutral **gray at half alpha** (`color_palette::RETICLE`) — the one
   mark deliberately outside the accent, because it spans the whole frame and is a place-marker,
   not a live value; the tone is a theme token so it tracks the theme's contrast. It **hides while
   a turn is in progress** and returns on release — not on mere click, which would make it blink
   at every re-centre.

> **Read this before editing the section above.** The two pivots are easy to collapse into one,
> and doing so has already cost a shipped-then-reverted binding. The write-up briefly said
> Shift+MMB orbits "the surface point under the cursor at press, raycast per gesture, never
> stored" — a misreading of *"Shift+MMB is always for the clicked surface point"*, where
> **clicked** means the click that **placed** the center, not the press that starts the drag.
> There is no transient pivot anywhere in this design.

### OK / Cancel — the general modal-command menu (owner-resolved 2026-07-28)

While any viewport command is running, the context menu is replaced *entirely* by two rows: **OK**
(`Return`) and **Cancel** (`Esc`). This is the general variant for all viewports, not an orbit-mode
special case — while a command is up there is no third choice, because a menu offering unrelated
verbs mid-command would be offering to start a second one. What each row *means* is the command's
business; for the explicit orbit mode both simply end it, since navigating IS the result and it has
already happened, so there is nothing left to discard. `ModeCommand::{Accept, Cancel}` on
`PanelResponse` is the wire.

There is deliberately **no per-command "leave" row**: that would be a second exit for one command
and no exit for the rest. The unrelated verbs a running command still wants to reach are planned to
live in a Fusion/Maya-style **pie menu above** this list — which is what keeps the list exactly two
rows.

Every menu row shows its **keybind flushed right**, in the weak tone. A row the keyboard cannot
reach leaves the column empty rather than inventing a binding, so the menu doubles as the honest
list of what is bound.

Nothing here spells a key, though. `ui::shortcuts` is the one place a binding is written down: a
menu row is handed a **command** and looks the binding up, and the shell asks which commands the
frame's presses meant. A hardcoded shortcut in a row is a type error, and a `KeyCode` named
anywhere in the shell is a clippy failure. The structure follows Blender and Krita — keyed by
command, the default declared beside the command's own label, the user's changes stored as a sparse
override, and a whole alternative set treated as a first-class swappable thing. That last one is
how the **per-platform** sets work: macOS and Windows/Linux each get their own binding per command,
decided on its own merits rather than by substituting ⌘ for Ctrl. Return and Escape land the same
on both because they genuinely are the same; the repro dump is `Ctrl+Shift+P` and `⌘⇧P`.

**Delete** is the binding the per-platform law exists for: `Delete` on Windows/Linux, `Backspace`
(⌫) on macOS, where no laptop has ever had a forward-delete key. The *key* differs, not the
modifier — no ⌘-for-Ctrl rule produces that. Both reach one implementation, and **what "delete"
means is the shell's call, not the menu's**: inside a sketch it is the picked entities, outside one
the picked node. The keyboard path arrives with no menu to have been built in a mode, so the branch
cannot live where the row is drawn.

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
   default, MRU split button, session variable). SHIPPED 2026-07-27 (slices A–C): the Free Orbit
   integrator, the seam, and the rail's type menu. CLOSED 2026-07-28 (slice D): the **type
   override** and the rail face showing active-vs-default both landed with the explicit orbit mode
   that carries the override's lifetime.
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
   (the context menu, the rail) land. The orbit center ✅ SHIPPED with step 2; the **explicit orbit
   mode** ✅ SHIPPED 2026-07-28 (slice D of the camera epic). The sketch W move remains.
