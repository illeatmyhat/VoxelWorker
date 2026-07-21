# Voxel placement: what block builders do, and what the CAD study missed

Researched 2026-07-20 to fill the gap in `placement-prior-art.md`, whose evidence was almost
entirely CAD/DCC (Blender, Fusion, Rhino, SketchUp). That doc concluded the field abandoned the
fixed ground plane for a *view-aligned plane through a movable depth anchor*, and the owner now
suspects that is a CAD artifact that a voxel builder answers more simply. This doc looked at the
shelf that study skipped: block and voxel builders. Every claim is marked **VERIFIED** (saw the
value in a wiki, source, or documented behaviour) or **INFERRED**.

## The finding: the block-builder idiom is neither plane — it is *adjacent-face-only*, and it rejects the empty-space click

The owner's instinct was "ground plane, else fixed/arbitrary depth." The evidence says the
mass-market block builders are **simpler than that**, and it is the sharper result the brief asked
for: **there is no ground plane and no fallback depth, because there is no free placement.** You
can only place a block *against an existing face*. The ground is not a special plane — it is just
the first face you happen to hit. Point at sky and the answer is **nothing happens.**

| tool | ground plane? | reach | ray misses geometry → | free floating placement? |
| --- | --- | --- | --- | --- |
| Minecraft (Java) | **none** — terrain is blocks, "ground" is a face | 4.5 blocks survival / **5 creative** | **rejected** (no selection box, no place) | **no** — adjacent face only |
| Minecraft (Bedrock) | none | 5 blocks (12 on touch) | rejected | no |
| Luanti / Minetest | none | `range`, **default 4** nodes | `pointed_thing.type == "nothing"` → **rejected** | no — places at `above` (the hit face's outer neighbour) |
| Vintage Story | none | ~survival-tier (community-reported ~4–?) | rejected | no — face-adjacent |
| Valheim | terrain is the surface | ~short build reach | ghost needs "something to float on" → **rejected** | no — snap points + structural support |
| **Space Engineers** (in space) | **none at all** | CTRL+scroll, **movable** | places ghost at chosen depth in empty space | **YES — the adversarial case** |
| **Dreams** | none | grab held at camera focus distance | grabs/holds at depth, push-pull along ray | **YES — grab-at-depth** |
| Astroneer | planet surface only | tool range | no placement — deform of existing surface only | n/a (no blocks) |
| Roblox Studio | **Baseplate** (a real, finite object) | camera raycast | drag snaps down to nearest surface | via drag-to-surface, not free air |
| MagicaVoxel / VoxEdit | **bounded editor volume** (a box grid) | n/a (editor) | box-drag inside the volume | inside a finite grid only |

The two rows that matter are the last two of the "free floating" column that say **YES**: Space
Engineers and Dreams. **They are the tools that removed the ground** — and the moment they did,
they independently reinvented the *view-aligned movable-depth grab*. That is the same primitive
the CAD shelf converged on. So the shelves do not disagree; they agree the instant free placement
is on the table.

## Answering the owner's question 2, per key tool

This is the exact question — *ray misses geometry, misses ground, now what?*

* **Minecraft — rejected.** The client raycasts out to the interaction range; if it hits no block
  the selection wireframe simply does not draw and right-click places nothing. Out-of-range is
  byte-identical to empty air (this is the "one bit" observation already in the CAD doc). There is
  **no** fallback depth and **no** floating placement in vanilla creative — the long-standing
  "Free Placement button to outline the block in front of you" is a *feature request*, not a
  feature. **VERIFIED** (Minecraft Wiki *Interaction range*; MinecraftForum suggestion threads).

* **Luanti / Minetest — rejected, in the data model.** The engine builds the ray as `eye +
  direction × range` and returns a `pointed_thing`; a miss yields `type == "nothing"`, which the
  place handler treats as no-op. On a hit, the placed position is the `above` field — the outer
  neighbour of the pointed face. Free-air placement does not exist because there is no code path
  that invents a depth. **VERIFIED** (Luanti API docs, `Raycast` class + `pointed_thing`).

* **Space Engineers — movable depth along the view ray. This is the owner's fallback, shipped in
  a voxel builder.** In creative there is no ground plane in space at all; the block ghost floats
  in front of the camera, and **CTRL + mouse-wheel moves it nearer/farther along the view
  direction.** You place the first block of a brand-new grid at that chosen depth in genuinely
  empty space. Survival locks the distance to a fixed max; creative lets you scrub it. This is
  *exactly* the view-aligned movable anchor — depth is a scrubbed scalar along the ray, not a
  ground intersection. **VERIFIED** (Space Engineers Wiki *Block Placement Mode*; Keen support
  threads on CTRL-scroll distance).

* **Dreams — grab-at-depth, held at the camera's focus distance.** The imp grabs a point with R1
  and the grabbed object is held *at the camera's focus distance*; you push/pull it along that
  axis (left-stick toward/away; "Reel" in VR). Beyond a max grab distance you can only pull
  inward. This is freeform 3D placement whose depth is a movable anchor tied to the camera — the
  same structure as the view-aligned plane, driven by an explicit depth control instead of a plane
  intersection. **VERIFIED** (Indreams forums: grabbed-object position "dictated by the camera's
  focus distance"; controls guides).

* **Astroneer — no empty-space placement exists to answer the question.** The terrain tool only
  *deforms an existing surface* the cursor is aimed at (dig/raise/flatten); the Alignment Mod
  locks digging perpendicular to the planet core. There is no "place object in empty air"
  operation, so the miss case never has to invent a depth — a miss just does nothing. **VERIFIED**
  (Astroneer Wiki *Terrain Tool*, *Alignment Mod*).

## The two families, stated plainly

**Family A — face-only, fixed reach, no fallback (the mass market).** Minecraft, Luanti, Vintage
Story, Valheim. Placement is a raycast to `range`; a hit places against the face, a miss places
nothing. There is no ground plane and no depth-invention *because the design never lets you
author in empty space at all*. This is **simpler than the owner's proposal** — it is not "ground
plane else fixed depth," it is "**face or nothing**," and the ground is just the first face.

**Family B — free 3D placement with a movable depth anchor (the tools that removed the ground).**
Space Engineers (in space), Dreams. The instant a tool must let you author in genuine 3D void, it
grows a depth control that runs *along the view/camera axis* and is scrubbed by the user (wheel,
stick, reel). None of them resurrected a fixed world-Z ground plane to do it, and none used a
"fixed reach snap." They use a **movable depth along the view ray** — the CAD shelf's answer,
reached independently.

**Family C — bounded workspace (the modelling editors).** MagicaVoxel, VoxEdit, and Roblox's
Baseplate sidestep the question by making the world *finite*: a box grid, or a literal Baseplate
object. There is always a surface (the grid floor / the plate) under the cursor, so "miss
geometry" degrades to "hit the workspace bounds," never "fly to the horizon." **VERIFIED**
(MagicaVoxel Wiki Face/Box modes; Roblox drag-to-surface raycast snapping down in 1.2-stud
increments to the nearest surface, Baseplate being a standard finite part).

## Verdict: is the view-aligned movable anchor over-engineered for this domain?

**No — but only because this app is in Family B, not Family A.**

The owner's suspicion is *half right*. If VoxelWorker were a survival/creative block game, the
native idiom is Family A: face-or-nothing with a fixed reach (Minecraft's 5 creative blocks), and
*both* the ground plane and the view-aligned plane would be over-engineered — you would never
invent a depth because you would never place in air. That is the genuinely simpler idiom the brief
asked me to name, and it is real.

But VoxelWorker is a **modelling tool** that deliberately supports free placement in empty space
(that is why the placement question exists at all). Every voxel builder that *also* supports free
placement — Space Engineers, a voxel builder with no ground in space — converged on the
**movable depth along the view ray.** So the view-aligned anchor is not a CAD import that fails to
fit; it is the answer the voxel shelf *also* reaches once it has the same requirement. The CAD and
voxel shelves agree.

Two refinements the voxel shelf adds to the CAD conclusion:

1. **Family C's lesson is cheaper than the anchor for the common case.** MagicaVoxel/Roblox never
   hit the horizon problem because there is always a surface under the cursor. Our resolution
   order already has this: "ground is a surface if something is there." The anchor only earns its
   keep for genuine void placement — keep it, but expect it to be the rare path, not the hot one.

2. **The scrub control is explicit in the voxel shelf, implicit in CAD.** Space Engineers
   (CTRL-wheel) and Dreams (stick/Reel) make the depth anchor a *thing the user actively moves*,
   not a value silently inherited from the orbit pivot. The current design's "last placement
   updates the anchor" is the implicit half of this; a Space-Engineers-style **explicit
   depth-scrub** (a modifier + wheel to push the anchor plane in/out) is the piece the voxel shelf
   has that the CAD-derived design does not, and it is worth stealing. It also gives the user the
   thing Minecraft players keep requesting (place-in-front-of-me) without a mode.

## What to keep from the current placement decision

* The three-way resolution order (hit face → anchor plane → ground-is-just-a-face) is **confirmed
  by the voxel shelf**, not contradicted. Family A is the degenerate case where only the first arm
  ever fires.
* The movable view-aligned anchor is **the correct primitive for free placement** and is what a
  voxel builder (Space Engineers) built with no ground. Not over-engineered *for a modelling
  tool*.
* **Add an explicit depth scrub** (modifier + wheel) as the user-facing handle on the anchor. This
  is the one concrete mechanic the voxel shelf has that the CAD study did not surface.
* Do **not** adopt Family A's "face-or-nothing." It is simpler, but it forbids the free placement
  that is this app's reason to have a placement question at all.

## Sources

* Minecraft reach: [Interaction range — Minecraft Wiki](https://minecraft.wiki/w/Interaction_range)
  (4.5 survival / 5 creative Java; 5 / 12-touch Bedrock). **VERIFIED**
* Minecraft floating request: [Creative Mode only: Place Blocks in Midair — Minecraft Forum](https://www.minecraftforum.net/forums/minecraft-java-edition/suggestions/53062-creative-mode-only-place-blocks-in-midair)
  (a *request*, i.e. it does not exist). **VERIFIED**
* Luanti raycast + range + `above`: [Raycast — Luanti Documentation](https://docs.luanti.org/for-creators/api/classes/raycast/)
  (`eye + dir × range`, default 4; miss → `type=="nothing"`; place at `above`). **VERIFIED**
* Space Engineers movable depth: [Block Placement Mode — Space Engineers Wiki](https://spaceengineers.fandom.com/wiki/Block_Placement_Mode)
  and [Keen support: CTRL-scroll placement distance](https://support.keenswh.com/spaceengineers/pc/topic/block-placement-distance-cannot-be-adjusted).
  **VERIFIED**
* Dreams grab-at-focus-distance: [How to move Imp-grabbed object on Z axis — Indreams forums](https://forums.indreams.me/hc/en-gb/community/posts/4408717139729-How-to-move-Imp-grabbed-object-on-Z-axis-with-camera)
  ("position away from the camera is dictated by the camera's focus distance"). **VERIFIED**
* Astroneer surface-only deform: [Terrain Tool — Astroneer Wiki](https://astroneer.fandom.com/wiki/Terrain_Tool),
  [Alignment Mod](https://astroneer.fandom.com/wiki/Alignment_Mod). **VERIFIED**
* Roblox drag-to-surface + Baseplate: [Preview of Studio's Dragger Tool](https://blog.haydz6.com/2013/08/preview-roblox-studios-better-faster-dragger-tool)
  (snap down in 1.2-stud increments to nearest surface). **VERIFIED**
* MagicaVoxel bounded volume + face/box attach: [Face mode — MagicaVoxel Wiki](https://magicavoxel.fandom.com/wiki/Face_mode_(interface)). **VERIFIED**
* Valheim snap points + support: [Building — Valheim Wiki](https://valheim.fandom.com/wiki/Building);
  ghost needs "something to float on" — [Steam building guide threads](https://steamah.com/valheim-indepth-building-system-guide/). **VERIFIED**
* Vintage Story reach ~survival-tier: [Limited block reach in creative — Vintage Story forums](https://www.vintagestory.at/forums/topic/10239-limited-block-reach-in-creative-mode/).
  **INFERRED** (forum reports, no single published reach constant found).
