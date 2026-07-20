# ADR 0018 — Exclusive viewer modes (Normal / Onion fog / Show booleans) and the reified root part

- **Status:** **Accepted & shipped (2026-07-16)** — epic #80, slices #81–#88 (`8653ded..444ba16`): Part
  vocabulary + reified root part, the three exclusive viewer modes, the region-scoped onion clip on both
  display paths, the #78/#79 overlay-surface retirement, and the Signal chrome (view cube / rail / status
  line / folding display stack per `docs/design/viewport-chrome-signal.md`). Supersedes the display
  surfaces of issues #78/#79 and amends ADR 0012's scene-wide onion clip to a per-object, mode-gated one.
  **Decision 3 partly superseded by [ADR 0024](0024-session-state.md) (2026-07-20):** the viewer mode
  remains out of the document and out of undo, exactly as decision 3 requires, but "is not saved with the
  scene" had been implemented as "is not saved at all" — a wider claim decision 3 never made. The mode is
  now **session** state and is restored across relaunch. The rest of this record stands.
- **Date:** 2026-07-16
- **Layer:** DISPLAY + UI + document vocabulary. Governed by [ADR 0006](0006-authoring-truth-and-gpu-boundary.md)
  (display is never truth) and [ADR 0017](0017-composition-beyond-union.md) (sealed scopes, ordered fold —
  unchanged by this ADR).

## Context

Issue #78 shipped an always-on selection x-ray (the selected node's body ghosts red/amber/union-blue over the
composed scene) and #79 a per-node persisted "Show child booleans" checkbox. Both are ghost overlays independent
of the ADR 0012 onion band clip — and they compose badly with it: the #78 ghost is meshed at the FULL band, so
scrubbing layers draws the selected object's entire unclipped body over the revealed slice (the "fog over the
object" bug; the regenerated `onion-ghost.png` golden had baked the wrong image in as reference). The owner's
direction: rather than patching the overlay/band interaction, make the display treatments **mutually exclusive
viewer modes**, and give them a well-defined object unit to apply to.

The object unit exposed a vocabulary misalignment: the code spent the name `Part` on a static voxel-body
producer (`NodeContent::Part`, e.g. DebugClouds) while the actual assembly container went by "Group" — yet the
glossary already used "part" for the composable object ("a cutter is a part placed under subtract"). The
owner's ontology (Fusion 360 parlance): **a part is the fundamental assembly container**; primitives, tools and
operators are its ingredients, never assembly citizens on their own; **the scene root is itself a part**.

## Decision

1. **Part is the container.** The sealed composition scope (today's Group node) is the user-facing **part** —
   the unit assemblies are made of and the unit selection-scoped display applies to. A one-off part is authored
   in place; a reusable part is a definition placed by instances (ADR 0017 semantics unchanged). The static
   voxel-body producer variant is renamed (`NodeContent::Part` → `NodeContent::VoxelBody`) to free the word.

2. **The scene root is a concrete part node** in the document and the tree (the Fusion root-component model).
   Root-level primitives are its ingredients. Whole-scene display treatment is expressed by **selecting the
   root part explicitly** — never by an implicit empty-selection rule, so a background misclick can't silently
   retarget a mode from an object to the world.

3. **The viewer is always in exactly one of three modes** — *Normal*, *Onion fog*, *Show booleans*. The mode is
   **viewer state, never document state**: it follows the active selection, is not saved with the scene, and
   never enters undo history (the PanelState display-param precedent). Sticky across selection changes; with no
   selection a mode has no target and the scene renders finished.

4. **Normal** is the finished look: no ghosts, no band clip, anywhere. The layer scrubber does not apply in
   Normal mode (it is Onion fog's tool, not a scene-wide constant).

5. **Onion fog is a region-scoped clip.** The selected object clips to the layer band with the ADR 0012 ghost
   haze outside it, **inside the selected object's placed AABB only**; all geometry outside that region renders
   finished. The clip test on both display paths (cuboid mesh + brick raymarch) gains the AABB intersect as
   uniforms — band scrubs stay O(1), no second display set, no re-evaluation on mode or selection change. What
   is shown inside the region is the **composed** geometry (carves by later cutters included) — the layers the
   user would actually build. Accepted softness: another object interpenetrating the selected object's AABB
   clips inside the overlap too (the Fusion section-analysis behaviour). The layer track spans the **selected
   object's Z extent**, not the scene's.

6. **Show booleans is the #79 walk, selection-scoped.** Every Subtract/Intersect operand body in the selected
   subtree x-rays over the finished scene in the shipped style vocabulary (red/amber, quiet `LessEqual` / loud
   `Greater` depth split, buried cutters wholly loud). Selecting the root part x-rays every boolean in the
   scene — the scene-wide master #79 deferred. The selected node need not be a part (an ingredient selection
   scopes the walk to that ingredient — degenerate but consistent).

7. **Retired:** the #78 always-on selection x-ray (including the union "which voxels are mine" tint — it can
   return later as a deliberate feature, not a side effect), the #79 per-node checkbox with its persisted
   `Node::show_child_booleans` field and `SetShowChildBooleans` intent (deleted, no back-compat per the config
   law), and the scene-wide always-on band clip.

8. **UI: a viewport icon strip + a collapsible display panel** (Blender × Fusion). An icon strip next to the
   View Cube holds Home, Fit, and the **viewport-mode cycle button** (Normal → Onion fog → Show booleans →
   Normal, current mode iconified). Detailed display settings that need real controls — the layer scrubber and
   onion depth among them — live in a highly collapsible stack panel, surfaced when Onion fog is active.
   **Visual language decided 2026-07-16** after a four-way prototype round: the "Signal" instrument language —
   view cube top-right with Fusion's 26 selectable zones, icon-only rail beneath it, panel folding to vertical
   edge tabs, one accent (the onion-haze hue). Spec + tokens: `docs/design/viewport-chrome-signal.md`; the
   interactive reference is the owner's claude.ai/design project "VoxelWorker — Viewport Chrome".

## Alternatives considered

- **Per-node persisted view-mode enum with nearest-ancestor inheritance** (the first design iterated in the
  grill): dissolved by the owner as over-complicated — persisted display state, an Inherit resolution rule,
  intent/undo surface, and per-node UI, all replaced by one viewer mode plus selection.
- **True per-object body separation for onion fog** (evaluate the scene-minus-selection and the selection's own
  body as separate display sets): voxel-exact under overlap, but a mode flip or selection change re-evaluates
  displays wholesale, doubles display invalidation forever, and shows the selected object's *standalone* body —
  which lacks carves later root-scope cutters made on it, i.e. layers that don't match the finished build.
- **Keeping the scene-wide band clip** and merely gating the ghosts: fails the per-object requirement.
- **A transparent (non-sealing) grouping folder** as the "part": two nesting concepts with invisibly different
  fold behaviour; ADR 0017 already rejected composition semantics that depend on container flavour.

## Consequences

- The fog-over-object bug class is dissolved structurally: onion clip and boolean x-ray can never co-render.
- Reifying the root part touches every `Scene::roots` consumer (tree UI, selection, fold entry, add flow) —
  the epic's foundation slice. Existing saves are not migrated (owner's no-back-compat law).
- The display goldens re-baseline broadly: union tints leave every `*-selected` reference, the onion golden
  returns to a correct image (selection-scoped clip), and `shot` grows a view-mode arg replacing the #79
  flag plumbing.
- The band-scoped diameter stat ("widest run in band") becomes an Onion-fog-mode readout.
- The always-visible layer scrubber leaves the permanent panel; panel space is reclaimed by the collapsible
  display stack.
