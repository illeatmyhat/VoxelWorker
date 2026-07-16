# Viewport chrome — the "Signal" design language

The visual language for the ADR 0018 viewport chrome (view cube, icon strip, collapsible
display panel), decided 2026-07-16 after a four-language prototype round (Blender-derived,
Fusion-derived, a Vintage-Story-flavoured direction, and this one). The interactive
reference lives in the owner's claude.ai/design project **"VoxelWorker — Viewport Chrome"**
(`chrome/d-signal/chrome.html` + `icons.html`) — an HTML/SVG mock whose behaviours
(mode cycling, panel folding, cube-zone hover) are the spec for the egui implementation.

## Character

Near-black instrument panel: hairline strokes, sharp corners, flat fills, monospace
micro-labels, and exactly one accent colour — the onion-haze blue, so the chrome and the
ghost pass share an identity. Nothing is decorated; state is shown by an accent inset bar
or an accent-filled cell, never by glows or shadows.

## Tokens

| Role | Value |
| --- | --- |
| Panel background | `#0b0d0f` at ~85% over the viewport |
| Hairline border (outer) | `#2b3238` |
| Hairline rule (inner) | `#1c2126` |
| Hover fill | `#12161b` (panel rows), `#161a1e` (rail buttons) |
| Text — primary | `#dfe7ef` |
| Text — secondary | `#aeb9c4` |
| Text — muted | `#78828c` |
| Text — faint | `#4d565f` (readouts), `#3c444c` (hints) |
| **Accent (the only one)** | `#9cb4d8` — the ADR 0012 onion-haze hue |
| Text on accent fill | `#0b0d0f` |
| Warn / subtract red | `#d9603f` (already the app's warn colour) |
| Boolean x-ray reds | quiet `#d94f4f`, loud `#e5533a` (ADR 0017/#78 vocabulary) |
| Axis colours | X `#d9603f` · Y `#7dba6a` · Z `#9cb4d8` |

Typography: monospace (Cascadia/Consolas class), 10–11 px; section headers and labels
UPPERCASE with ~2 px letter-spacing; counts and hints 9–9.5 px faint.

Shape rules: **zero corner radius**, 1 px hairlines everywhere, no drop shadows, no
gradients. Active/selected = 2 px accent inset bar on the leading edge, or an
accent-filled segmented cell with dark text. Sliders: 2 px track, hollow square thumbs
with accent border. Steppers/segments: bordered cells separated by hairlines.

## Chrome layout (all anchored to the viewport, Z-up world)

- **View cube — top-right** (industry norm), ~116 px. Generated from real projection
  (yaw ≈ 31°, pitch ≈ 22°, front face dominant). Each visible face partitioned 3×3 with a
  68% centre patch → the **26 selectable zones** (6 faces, 12 edges, 8 corners; 19
  visible). Hover highlights every facet of the zone **across the fold** (edge/corner
  zones span faces) at accent ~45% fill, with a faint readout line under the cube naming
  the zone (`TOP·FRONT`, `TOP·FRONT·RIGHT`). Cube faces are translucent panel-fills with
  hairline slice lines; face labels (FRONT/RIGHT/TOP) are projected onto the faces, never
  skewed by hand. **Axis-coloured edges** all emanate from the front-bottom-right corner —
  the one fully visible corner: X red along the front-bottom edge, Y green up the receding
  right-bottom edge, Z blue up the front-right vertical — with letter labels at the far
  ends. egui: raycast picking against the 26 zone quads; minimum on-screen size so the
  edge strips stay hittable.
- **Icon rail — directly under the cube**, 34 px wide, vertical, **icon-only** (tooltips
  carry the words): Home, Fit, and the viewport-mode cycle button (Normal → Onion fog →
  Show booleans). The mode button in a non-Normal mode is "lit": accent glyph + 2 px
  accent inset bar.
- **Display panel — right edge**, ~226 px: a `DISPLAY »` header bar, then collapsible
  sections (chevron + UPPERCASE name + faint control-count), each body a grid of
  label/value rows. The **Onion fog section exists only while the viewer is in Onion
  mode** (layer range slider, onion depth stepper, widest-run readout). `»` folds the
  whole panel to **vertical edge tabs** (Blender N-panel style): rotated text boxes, one
  per section plus a `«` expander; clicking a tab expands with that section opened; the
  onion tab obeys the same mode rule. **The cube and rail slide toward the edge when the
  panel folds** and back when it expands — they track the viewport's usable corner.
- **Status line — bottom-left**, faint mono: `VIEWPORT <MODE> · SEL <node> · <dims> ·
  <density>`; mode name and selection in accent.

## Icon set

18-unit grid, 1.25 px stroke, square joints, no rounding. Glyphs: home, fit (corner
brackets + square), mode-normal (solid cube), mode-onion (lifted layer slices), 
mode-booleans (solid ∩ dashed squares), part (container cube), root part (container over
a ground line), fold chevron. Idle `#78828c`, hover `#c7d3e0`, active accent.

## Rejected directions (for the record)

Blender-derived flat-grey density and Fusion-derived light floating chrome both read fine
but generic; the Vintage-Story-flavoured walnut/brass direction was cut first as too
costumed. Signal won for making the display chrome share the ghost pass's own colour and
for the instrument-panel restraint suiting a planning tool.
