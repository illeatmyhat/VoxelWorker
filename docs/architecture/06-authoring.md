# 06 — Authoring

The shell is the layer a person actually touches: the viewport, the pointer, the camera, the
menus, the chrome. It owns no truth. Every change it makes to the design leaves through the
one intent door, and everything it draws is a derived artifact it was handed. What it *does*
own is the grammar — what a click means, what a mode is, what moves and what refuses — and
that grammar has to be uniform, because a tool whose left button means four different things
in four places is four tools.

## Selection is the substrate

**The left button selects.** That is its verb everywhere: in the viewport, on scene nodes and
on sketch entities alike. Navigation is not on it. A pointing device has one unmodified
primary action, and spending it on camera motion means every act of *choosing something* has
to be reached some other way — which is how a browser tree becomes the only place a node can
be picked.

There is **one selection**, a single set that holds mixed kinds. Which kinds may enter it is
a property of the current editing mode, not of the set: normal mode admits scene nodes and
points, sketch mode admits sketch entities. Everything that acts on "the selection" therefore
has exactly one thing to read, and there is no second notion of *active* to fall out of step
with it.

Building it is four gestures and no mode switch: a click replaces the set with what was hit;
a shift-click toggles one entity in or out; a click on empty space clears it; a drag from
empty space is a marquee. Selection resolves on a **stationary release**, so it never fights a
drag that turned out to be a manipulation. Where two entities overlap, the smaller wins —
a point beats a segment through it — because the larger one is always still reachable
elsewhere along its length and the smaller one is not.

**The marquee's direction chooses its semantics.** Dragged one way it is a *window* and takes
what it fully encloses; dragged the other it is a *crossing* box and takes anything it
touches at all. These are genuinely different questions — a segment passing through the box
with both ends outside is crossing-only — and the difference is worth a gesture because the
alternative is a modifier the user must remember while already mid-drag. The two draw
differently, solid against dashed, because the user has to tell them apart *during* the drag,
when the result has not happened yet and only the box can say what is about to.

Not everything pickable is a member. A sketch region is picked and unpicked but never joins
the set: its identity is a *set* of boundary edges rather than a single id, and the selection
carries small copyable targets by value throughout the shell. A thing whose only verb is a
menu row does not need to enter the set to be operated on, and admitting it would cost the
representation everywhere.

## A mode is selection plus a manipulator

Modes are global and act on whatever is selected. The base mode is selection alone; the
others add a translate, rotate or scale manipulator over the same set. Because selection is
the shared substrate, a mode change never invalidates what is chosen — it changes only what
can be done to it, which is what makes the modes cheap to move between.

A mode is a state with an exit. That distinguishes it from **arming**, which is a transient
overlay on the current mode rather than a mode of its own: pressing a mode key disarms first,
and an armed placement never survives as a hidden fifth state.

## Creation is arm, drop, selected

Every creation tool has the same three-part shape and only the middle part differs. A tool is
**armed** and a preview follows the cursor, landing at the picked point. A click **drops** the
node, which comes into existence already **selected**. From there its manipulators are live
and *stay* live.

There is no adjust phase to enter or leave. A freshly dropped solid and one selected days
later show the same handles, because **manipulators belong to the selection, not to
placement**. Placement is not a mode with its own editing rules; it is a way of bringing a
node into existence already selected. The alternative — an editing state that expires — makes
the author's first seconds with an object different from every second after, and there is no
principle that says which rules apply when.

A preview is free to move. It participates in no composition, so nothing recomposes when it
follows the cursor. The preview is also necessarily **its own render pass**: no display path
carries a per-object transform — meshes bake world position into their vertices and the
raymarch walks one world-fixed lattice — so a preview can never be "the committed geometry,
moved". There is no seam to move it by, which is exactly why a dedicated pass that owns
nothing and reuses nothing is the cheap answer rather than the wasteful one.

## What a manipulator may do

- **Snap to the lattice, always.** The gesture is continuous and the result is an integer
  voxel count, so there is no float left for a caller to commit by accident. Block granularity
  is the coarse default where it reads better; voxel granularity is the fine step.
- **Emit exactly one intent per gesture.** A drag is one undoable act. Undoing a move one
  voxel at a time would be a worse tool than no undo at all.
- **Carry the frame.** Manipulators work in the recentred render frame and the document takes
  blocks and voxels; the reference point travels with the value rather than being re-derived
  at the far end.
- **Never offer what the document cannot say.** No operand picking, and no transform on a
  baked body that would silently resample it.

## A move is a request, not a write

Pressing and moving on a selection **proposes** a translation. The path is always *propose a
delta → solve → apply*, never "write the new positions." The solve is what clamps, projects,
or refuses the request, so an over-constrained selection may not move at all, and a partly
constrained one moves only along what is left. Building the gesture any other way makes the
constraint system something the interaction layer has to remember to consult, and it will
eventually forget.

## Delete is an action, not a mode

Delete lives on the selection, reached from the keyboard or from the menu. It is not a tool
to be armed. A delete *mode* costs a permanent slot on the rail to hold a verb that already
has a key, and it fights the habit every other application has trained.

One verb, one glyph, one place in the menu — the *target* is whatever the selection means in
the current mode. In a sketch it is the picked entities; outside one it is the picked node.
That branch belongs to the shell rather than to the menu, because the keyboard reaches the
same action with no menu ever having been built.

## One menu, and one place bindings are written

The viewport has a single context menu whose contents are dispatched by mode. Shared actions
draw identically in every mode so they can be found without reading; mode-specific rows join
them.

**While a command is running the menu is replaced entirely by accept and cancel.** A menu
offering unrelated verbs mid-command is offering to start a second one. What accepting and
cancelling *mean* is the running command's business — for a navigation command both simply
end it, because the navigation already happened and there is nothing to discard. There is
deliberately no per-command "leave" row: that would be a second exit for one command and no
exit for any other.

Every row shows its binding flushed right, and a row the keyboard cannot reach leaves the
column empty rather than inventing one — so the menu doubles as an honest list of what is
bound. **No row spells a key.** A row is handed a *command* and looks the binding up; the
shell asks which commands a frame's presses meant. Bindings live in exactly one registry,
keyed by command, with the default declared beside the command's own label and a user's
changes stored as a sparse override. A whole alternative set is a first-class swappable
thing, which is how per-platform sets work: each platform gets its own binding per command,
decided on its merits rather than by substituting one modifier for another. Delete is the
case that law exists for — the *key* differs between platforms, not the modifier, and no
substitution rule produces that.

## Navigation

Panning is the middle drag and zooming is the wheel. Orbiting is reached two ways, and they
turn about **two different pivots** that do not share a point. The orbit arithmetic is
identical; what separates them is *which mechanisms may move each pivot*, and that is the
whole of the distinction. Collapsing them into one has already cost a shipped-then-reverted
binding, so they are named separately and kept that way.

- **The orbit center** is put down by a deliberate act and moved by *nothing else*. Panning
  does not move it; zooming does not move it. That is the entire feature: slide the view
  across the model and the thing being inspected stays the thing you turn around. Before one
  has ever been placed it sits at the world origin. It is **continuous, not snapped** — it is
  a camera quantity with no lattice meaning, and a snapped pivot visibly jumps a cell at a
  time under the cursor. While being placed it draws as a marker, because the pivot is the
  one camera quantity with no geometry of its own and a forgotten one is otherwise invisible.
  That marker sits **under the cursor**, not on the surface: an armed marker is a cursor, and
  a cursor that trails the pointer is broken however accurate it is.
- **The view target** is what every other mechanism turns about, and the explicit orbit mode
  is where it is set. That mode overlays a reticle, takes the left button for its duration,
  turns on a drag, and on a stationary click raycasts a surface and re-centers on the hit. A
  miss is a **refusal**, not a fallback: the target keeps its old value rather than flying to
  a guessed one. The re-center animates, because a jump-cut reads as a teleport where an ease
  reads as the pan it is.

### Two orbit types, one authority

Constrained orbit keeps world-up fixed so the camera never rolls; free orbit is a full
trackball. The default is a most-recently-used session value — never a setting, never part of
the document.

Which entry paths may *write* that default is decided by one question: does the path **name a
type**? A split button's face and an unmodified drag do not name one, so they run the default
and leave it alone; picking from that button's menu does name one, and picking re-faces the
button, which is the same act as setting the default. A menu command that names a type is not
a settings control at all — invoking a tool has never meant "make this the default." Things
identical in effect and separated only by which mechanisms may write a persistent value get
named, never merged.

A type-naming command runs its type **until the mode is left**, then the default reasserts.
An override whose boundaries the user cannot see is one they cannot reason about, and "mode"
already means a state with an exit. The consequence is that while an override runs, the
button's face is not what the *next* orbit will do — so the face must show the active type as
distinct from the default, or it lies for the duration.

**Exactly one representation is authoritative at a time.** The spherical chart is the stored
truth and the primary; the trackball orientation is a secondary representation with its own
operations, authoritative only while free orbit is the active type. Never both live and
synced — that is the two-truths bug, and a diagnostic dump would capture whichever had gone
stale. One value carries the authority, so there is no second flag able to disagree with it,
and the idempotent conversions in each direction let the shell's own notion be policy without
the two ever needing to be proven in step.

The seam between them is exact in both directions, because the chart is a proper chart of the
rotation group. Inverting it is unique away from the poles, and *at* a pole the two angles are
not individually determined — only their combination is, which is gauge freedom rather than
lost information: every consistent choice reproduces a bit-identical view, and the ambiguity
is resolved by taking the value nearest the last one. Gimbal lock does not enter, because
lock is a failure of continuity *along a trajectory* and a type switch is a single point.
**The active type may never change mid-gesture**, which is what keeps that conversion a point
and stops it becoming a path.

Dropping accumulated roll on the way back to constrained orbit is the one deliberately lossy
step, and it **animates**. A hard cut reads as a glitch; eased, it reads as intent. It plays
only where it is the sole thing happening — not on an operation that was about to snap
anyway, and never mid-drag, where the animation would fight the gesture.

Surfaces that read or write the chart close the seam before acting, and so do both
persistence paths — a session or a diagnostic dump written mid-trackball would otherwise
record a pose the user left some time ago. Every per-frame consumer reads through accessors
that work under either representation, since the view cube still renders during a trackball
drag and must not read frozen angles.

## Picking a point in space

A click names a pixel; placing geometry needs a point. Where there is geometry under the
cursor, the hit is the answer. Where there is not, the ray meets one of **three fixed world
planes through the origin, with the ground privileged** — and those planes never move.

A plane that follows the view or hangs off a movable depth anchor is the mainstream answer
and it was rejected here. In a lattice tool the author is placing something *on* a structure
they can see, and a plane that moved because the camera moved makes the same click mean two
different points a second apart. Fixed planes are predictable, and predictability is what a
placement gesture is for.

Pointing at the sky is a **refusal**, not an invented depth. Eye above the ground looking up,
there is no plane in front to place on, and answering with a point is answering a question
that was not asked.

The grazing case is bounded **by angle, before the intersection is attempted** — not by a
distance clamped after it. Testing after means a ray nearly parallel to the plane has already
produced a wildly distant point that then has to be pulled back, which introduces a dead zone
where large pointer motion produces no visible movement. Testing the angle first is cheaper
and degrades gracefully. Three mutually perpendicular planes make the bound total: a unit
vector cannot be near-parallel to all three at once, so some plane always faces the ray well
enough and there is no grazing case left to clamp.

The point that comes back is **continuous**. What the lattice quantizes is what gets stored,
not what the hand is doing — an authoring anchor that snapped under the cursor would make
sub-voxel intent unexpressible before it was ever recorded.

## Feedback

**A gizmo is a manipulator, so it keeps its size on screen.** A handle that shrank with
distance would become unusable exactly when the camera pulled back to show the whole model,
which is when it is most needed. Three-dimensional gizmos are therefore built at a scale
derived from their distance to the eye, so they occupy a fixed fraction of the frame at any
zoom. Two-dimensional marks — a sketch's points and edges — are screen-space billboards for
the same reason and a stronger one: they must be visible and hittable at any camera angle,
including edge-on to their own plane, which no three-dimensional representation of them
survives.

**A selection announces itself over the geometry**, not through it — an outline and a wash
depth-tested against the composed model, rather than a translucent tint that would make a
selected body look like a different material. The distinction matters because this tool
already spends translucency on something else: showing what a boolean is about to remove.

**An assertion that has been satisfied is invisible** unless it is marked. That principle is
the sketch's (a satisfied constraint leaves geometry that merely *looks* the way it was
asked to), but the shell is where it is paid for — every such relation draws a mark anchored
through the entities it names.

## The chrome

The chrome is an **instrument panel**: near-black opaque surfaces, hairline strokes, zero
corner radius, flat fills, uppercase monospace micro-labels, and **exactly one accent**.
Nothing is decorated. State is shown by an accent inset bar or an accent-filled cell — never
by a glow, a shadow or a gradient. Surfaces are fully opaque so they read solid over a
textured voxel scene, and the whole application wears the same language, so every panel reads
as part of one instrument rather than as a separate window.

The single accent is the same hue the ghost pass uses, so chrome and geometry-preview share
one identity. That is also the constraint the language is most likely to be asked to break:
a second hue is the obvious way to express a second state, and the answer is usually a
different *texture* — dashed against solid, filled against hollow — rather than a new colour.
Semantic exceptions are few and each is earned: a destructive verb, a subtractive operation,
the three axis colours, and one deliberately neutral place-marker tone for a mark that spans
the whole frame and stands for a location rather than a value.

### What each channel is spent on

Color is a scarce channel: every meaning assigned to a hue reduces what any hue can mean. A
viewer learns a hue from ten encounters and applies it on the eleventh whether or not that was
intended, so a new state is never given a color in isolation — the question is what is left,
and what this will teach about everything else.

| Channel | What it means | What it may not mean |
| --- | --- | --- |
| The accent | Active, selected, current, live. Also the ghost pass's own hue, so chrome and preview read as one system | Anything with valence — not "good", not "confirmed", not "safe" |
| Warn red | Subtraction, removal, and genuine warnings | Emphasis. A red that sometimes just means *loud* stops meaning *removed* |
| X-ray reds, quiet and loud | Operand bodies shown under a boolean; the quiet/loud split is depth, not severity | Any non-boolean overlay |
| Axis red, green, blue | Axes — in the orientation cube, in gizmos, in triads | Anything at all in a spatial context |
| Material color | The material. Pigment belongs to the model, not to the interface | Interface state. If state tints geometry, a brick stops looking like brick |

Texture is the orthogonal channel and the one to reach for first, because states co-occur and
there is only one accent to spend: hatch for *touches something not shown here*, a dashed
outline for *uncommitted*, dimming for *present in the document, absent from what is
evaluated*. These stack with color instead of competing with it.

**Every color-carried state has a non-color carrier as well** — a texture, a mark, a word, an
inset bar — and no distinction is ever carried by red against green alone. The axis colors use
that pair, which is tolerable only because an axis is also distinguished by position,
direction, and a letter. Beyond the roughly one man in twelve with a red–green deficiency,
this is what keeps a state legible in a still image and to anyone reading the interface rather
than driving it.

Layout is anchored to the viewport, not to the window. The orientation cube sits top-right at
a fixed on-screen size, generated from the real projection and partitioned into selectable
face, edge and corner zones; hovering a zone lights every facet of it **across the fold**,
since edge and corner zones span more than one face. Its strokes, silhouette and labels
render at a constant screen-space width so glancing angles never thin them to nothing. An
icon-only rail sits beneath it with its words in tooltips. A collapsible panel occupies one
edge and folds to vertical tabs; the cube and rail **track the viewport's usable corner** and
slide when it folds. A faint status line names the mode and the selection.

Sections that only make sense in one viewer mode exist only in that mode, rather than
appearing disabled. A control that cannot do anything is noise that the reader has to
re-evaluate every time they scan the panel.

Icons are drawn, never typed. A glyph is a painted path on its own authoring grid, and its
stroke width is that grid divided by a fixed ratio — stated once as a single number both icon
families derive from, because a hand-set width per family drifts invisibly at small sizes and
only reveals itself when something is scaled up.

## What the shell may not do

It may not write the document except through an intent. It may not hold a second copy of
anything the document owns. It may not decide occupancy, exactness, or units. And it may not
allocate its floating chrome inside the region that receives camera input — a floating panel
that reserves layout space carves a dead band out of the viewport where drags silently stop
working, so chrome is placed over the viewport and hit-tested explicitly.
