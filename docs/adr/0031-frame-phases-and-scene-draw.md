# ADR 0031 — The viewport render is ordered frame phases of `SceneDraw`

- **Status:** Accepted
- **Date:** 2026-07-23
- **Relates to:** [ADR 0011](0011-gpu-brick-field-display-sink.md) (the model draw — brick raymarch
  or cuboid mesh — is the one phase this refactor keeps special), [ADR 0018](0018-viewer-modes.md)
  (the over-model ghosts are the operand x-ray; the onion ghost rides the model phase),
  [ADR 0022](0022-document-dump-and-state-classification.md) (display toggles like the point-axes
  on-top flag are viewer state, not document).

## Context

`render_frame` in `src/lib.rs` recorded the whole viewport into one MSAA pass by hand: ~12 draws in
a fixed sequence, each reached through a named field of a ~15-field `FrameOverlays` god-struct
(`background_gradient`, `gizmo`, `scene_grid`, `points`, `infinite_grid`, the ghosts, …). Every new
thing drawn in the viewport cost three coordinated edits — a field on `FrameOverlays`, a
`if let Some(x) = overlays.x { x.draw(pass) }` block in `render_frame`, and wiring at *both* call
sites (`windowed/render.rs` and `shot/capture.rs`). The coupling grew O(n) with the number of draws,
and `src/lib.rs` (1043 lines) bundled this GPU orchestration with the ~470-line egui frame builder.

The word "overlay" was also overloaded: the glossary already spends it on the on-face-grid flag half
of a [cell key](../../CONTEXT.md), not on "a thing drawn in the pass."

## Decision

### 1. The viewport render is a fixed sequence of frame phases

The single viewport MSAA pass is recorded as ordered **frame phases**, grouped by depth semantics:

1. **background** — fullscreen, pre-solid, depth off.
2. **model** — the solid voxels (brick raymarch or cuboid mesh) plus its onion ghost. **Special**,
   not a `SceneDraw` list: it needs the material bind group and the brick/mesh choice (ADR 0011).
3. **over-model** — translucent ghosts that blend over the solid: the operand x-ray, the placement
   ghost.
4. **scaffold** — depth-tested reference lines the model occludes: block/floor grids, point axes.
5. **on-top** — depth off, drawn through the model: the manipulator gizmos.

The **view cube** is a separate scissored corner pass, not a phase. The phase *order* lives in one
place (`render_frame`); each phase's *contents* are a caller-filled slice, so shot can gate draws off
(goldens) while windowed shows them, without either touching `render_frame`.

### 2. `SceneDraw` — one uniform draw call per phase entry

A **scene draw** implements `trait SceneDraw { fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>); }`.
Every phase entry is a `&dyn SceneDraw`; `render_frame` iterates each phase slice in order. The trait
lives in `display` beside the renderers, and each renderer impls it in its own file (a phase's draw
stays legible where the renderer is). The model and view cube do **not** implement it — they are
genuinely different kinds of thing, and forcing them under the trait would only distort its signature.

`FrameOverlays` is replaced by `FramePhases { background, over_model, scaffold, on_top:
&[&dyn SceneDraw], .. }` plus the model/cube specials. The name "overlay" is retired from the render
path and left to its glossary meaning.

### 3. `src/lib.rs` splits into `src/frame/`

`egui` frame building (`run_egui_frame`, `EguiPaintBridge`, `PreparedEguiFrame`) and GPU pass
recording (`render_frame`, `FramePhases`) move to `src/frame/egui.rs` and `src/frame/render.rs`;
`lib.rs` re-exports. The two responsibilities stop sharing a file.

## Consequences

- Adding a viewport draw is now: impl `SceneDraw` (usually one line — the method exists) and push it
  into the right phase slice at the call site. `render_frame` and `FramePhases` do not change.
- Phase *order* is safe (one definition). Within-phase order is the slice order, chosen per call
  site; the two sites legitimately differ (shot gates for goldens), so no shared builder is imposed.
- The model + onion ghost + view cube stay hand-written; the refactor does not chase full uniformity
  it would have to fake.
- Goldens (`cargo test -p shot`) verify the shot path is a pure relocation; the windowed path is
  feel-tested (no goldens there).
