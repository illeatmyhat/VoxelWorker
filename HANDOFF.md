# Chisel Bench — Rust (wgpu + egui) port: handoff

## What this is

**Chisel Bench** is a planning tool for **Vintage Story** chiseling. VS lets you carve a
placed block into a 16×16×16 grid of microblocks ("voxels"). Planning curved/round shapes
by hand is painful, so this tool: defines a parametric shape in a block-sized box, samples it
onto the voxel grid, and renders the result as **hard voxels** (one textured cube per filled
voxel) so you can see the exact stair-stepped quantization *before* you chisel in-game. It
also shows a top-down pixel "slice" map to chisel against, and (eventually) loads the real VS
block textures so the preview looks like the actual material.

There is a **working browser prototype** in this folder (`chisel-bench-reference.html`, a single
HTML file using three.js). It is the source of truth for behavior and math. The job is to
**re-implement it as a native Rust app** using **wgpu + egui + winit**. Re-implement; do not
try to embed the HTML.

## Why Rust / wgpu / egui (decisions already made — don't relitigate)

- **Native, not browser.** The prototype is blocked by the browser File System Access sandbox:
  it cannot read `%APPDATA%\Vintagestory\...` (AppData is on Chrome's blocklist). Native Rust has
  unrestricted file access, which is the whole reason for the port.
- **Rust specifically** for the compiler-guaranteed tight loop (good for agentic iteration; no
  segfaults to chase).
- **wgpu + egui, NOT a game engine.** Critical insight that drove this choice: **we are not
  SDF-rendering.** The signed distance functions are used *only on the CPU* as an inside/outside
  test to decide which voxels exist. The actual render is the most conventional thing possible:
  a pile of axis-aligned **instanced textured cubes**. No raymarching, no marching cubes, no GPU
  SDF. So we need a GPU API (instancing + one custom shader + 2 cameras), not an engine. Godot
  was ruled out (not Rust). Bevy was ruled out (ECS ceremony + API churn for no benefit here).
  three-d + egui is an acceptable lower-effort fallback if raw wgpu plumbing is too much, but
  prefer wgpu for control and the tight `cargo check` loop.

## The data model (this is the core; get it right first)

- Dimensions are **whole blocks**: `X, Y, Z` (the bounding box). Default `5 × 1 × 5`.
- **`density` = voxels per block** (chisel fineness). Default **16** (Vintage Story's grid).
  Other voxel games could differ; keep it a parameter.
- Therefore the voxel grid is `Nx = X*density, Ny = Y*density, Nz = Z*density`.
- The shape is **inscribed in the X×Y×Z box** (NOT parameterized by radius). Cylinder fills
  X/Z as cross-section + Y as height; Sphere = ellipsoid in the box; Box fills it; Torus uses
  X/Z for outer diameter and Y for tube thickness; Tube = hollow cylinder needing one extra
  `wall` param (in blocks) — the only shape that needs more than X/Y/Z.
- **Critical decoupling (a bug we already fixed once):** density must NOT change the object's
  block size or the texture scale. Texture is **one tile per block** always; raising density
  only subdivides each block into more voxels (smoother curve, same size, same texture). If you
  ever see the texture rescale when density changes, dimensions are wrongly expressed in voxels
  instead of blocks. See ARCHITECTURE.md "Units & the density bug".

## Rendering model (what the GPU actually does)

1. CPU builds a voxel list: triple-loop over `i,j,k`; world-centered position
   `p = (i+0.5 - Nx/2, j+0.5 - Ny/2, k+0.5 - Nz/2)`; keep the voxel if `sdf(p) <= isolevel`.
   For each kept voxel store: world position **and** block-local coords `iLocal = (i%density,
   j%density, k%density)`.
2. Upload as an **instance buffer**. One unit-cube vertex buffer, N instances.
3. **Custom shader** does two things the stock material can't:
   - **Per-voxel texture slice:** each voxel shows only its `1/density` slice of the parent
     block texture, so the texture spans a whole block and chisel cuts reveal a cross-section
     (NOT the whole texture repeated per cube — that was bug #1).
   - **Grid overlay (toggle):** thin lines on every voxel boundary, bolder/darker lines on block
     boundaries. **Compute this from world position, not face UVs** — deriving it from UVs caused
     an off-by-one on vertical faces because cube faces flip UV direction. See ARCHITECTURE.md
     "The two shader bugs".
4. Cameras: **perspective and orthographic**, sharing an orbit rig (spherical `theta, phi, dist`
   around a target). Orthographic frustum tracks `dist` so zoom works and switching keeps framing.
5. A **view cube** (Autodesk-style) in a corner: click a face to snap to front/back/left/right/
   top/bottom with an eased tween. An **origin gizmo** (X/Y/Z arrows + perpendicularity squares,
   depth-test off so it shows through the model), toggle-able.
6. A 2D **top-down slice** of the mid-height layer, drawn as a pixel map with block-boundary
   lines — the literal chisel reference. In Rust this can be an egui image/texture you fill on CPU.

## UI (egui panel)

Mirror the prototype's right-hand panel: Shape chips (Cylinder/Tube/Sphere/Torus/Box); Size
sliders X/Y/Z (whole blocks) + conditional `wall` for Tube; `density`; `isolevel`; Material
(Stone/Wood/Plain procedural defaults + load real block); Display toggles (voxel grid on faces,
block lattice, fine floor grid, view cube, origin gizmo); Camera projection (Perspective/Ortho);
a **palette dock** along the bottom of detected blocks.

## The thing that motivated the port: loading real VS textures

Native file access is the payoff. Implement:
- A "Connect VS folder" action → OS folder picker (`rfd` crate) → user picks
  `…\Vintagestory\assets\survival` (or the `block` dir directly).
- Recursively scan `textures/block/**.png`. Keep files matching the **known chiselable** name
  list (all vanilla rock types + plank/ashlar/polished/cobblestone/drystone/brick); skip
  ore/gravel/soil/overlay/normal-maps. Group variants by stripping trailing digits
  (`granite1, granite2 → "Granite", 2 variants`). See `chiselable_blocks` in DATA.md.
- Palette dock: one tile per block, a **45° orthographic thumbnail** of a textured cube, click to
  apply a **pseudo-random variant** to the current shape.
- **Next step the browser version couldn't do:** resolve the block's JSON in
  `assets/survival/blocktypes/**` to get true **per-face** textures (top vs side) and exact
  face orientation, instead of one texture on all faces. This is the main reason to go native.

## Build order (suggested milestones, each independently testable)

1. **Window + clear.** winit + wgpu + egui-wgpu compositing in one window. Empty panel + a
   colored clear. Confirms the stack compiles and runs.
2. **Voxel core (no texture).** Data model + SDFs + instance buffer + flat-shaded instanced
   cubes + orbit camera (perspective). Hard-code a 5×1×5 cylinder. This is the heart — get the
   quantization visually correct vs the prototype.
3. **Panel + live params.** egui sliders/chips driving rebuild. All shapes. Orthographic toggle.
4. **Shaders.** Per-voxel texture slice (procedural Stone/Wood first), then the position-based
   grid overlay. Verify block boundaries align on ALL faces (the off-by-one regression test).
5. **View cube + gizmo + 2D slice map.**
6. **VS folder scan + palette + thumbnails** (`rfd` + `walkdir` + `image`).
7. **Block-JSON resolution for per-face textures** (the native-only win).
8. Polish: `.vox` export (feeds the Automatic Chiselling REBORN mod), config persistence.

## Files in this handoff

- `HANDOFF.md` — this file.
- `ARCHITECTURE.md` — the math, the shader logic, the two bugs we fixed, camera rig, render layering.
- `DATA.md` — units model, chiselable block list, VS install paths, texture/JSON layout.
- `chisel-bench-reference.html` — the working three.js prototype. **Behavioral source of truth.**
  Read its `<script>` for exact SDF formulas, the WGSL-portable shader injection, camera math,
  scan/group logic, and palette/thumbnail rendering.

## Suggested crates

`winit` (window/input), `wgpu` (render), `egui` + `egui-wgpu` + `egui-winit` (UI), `glam`
(math), `bytemuck` (instance/uniform structs → bytes), `rfd` (native file dialog), `walkdir`
(recursive scan), `image` (PNG decode). Optional later: `serde`/`serde_json` (block JSON),
`pollster` or async runtime for wgpu init.

## Don'ts (learned the hard way)

- Don't repeat texture per-cube. One texture per block, sliced per voxel by `iLocal/density`.
- Don't derive the grid overlay from UVs. Use world/voxel position (face UVs flip → off-by-one).
- Don't express dimensions in voxels. Blocks for size; density is fineness only.
- Don't let shape selection mutate the X/Y/Z values or jump the camera.
- Don't fork an engine. Don't reach for Bevy/Godot — wgpu+egui matches the actual (small) need.
